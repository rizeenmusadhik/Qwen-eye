"""Map D (Chefer et al., ICCV 2021, "Generic attention-model explainability") for Qwen3-VL's
ViT, and from the same pass C (rollout, Abnar & Zuidema 2020) and W (the readout alone).

Map E/H at block L says which features of block L mattered; by L16 every token has mixed with
its whole frame pair, so that is not which *input* patches mattered. D carries the relevance
back to the input through every block's attention, each edge weighted by its gradient.

Per frame pair (the ViT's attention is block-diagonal over pairs), n patches, block l = 0..26:
  A_l    the softmax [heads, n, n], rows = queries (capture.py recomputes it in fp32)
  dA_l   d(score)/d(A_l). The block computes O_h = A_h V_h, so dA_h = dO_h V_h^T, with dO the
         gradient at the attention output (grads.OutputGradTaps): never formed by autograd.
  M_l    mean over heads of ReLU(A_l * dA_l): Chefer's avg_heads, ReLU before the mean. The
         "against" side uses ReLU(-A_l * dA_l).
  R      relevance of every input patch (columns) for every token after block l (rows);
         R = I before block 0, then Chefer's self-attention rule R <- R + M_l R (the identity
         term is the residual connection).
C uses Abnar's rollout instead, R <- (R + mean_h(A_l) R) / 2: no gradient, so D vs C isolates
the gradient weighting of the attention edges.

R is kept as diag(s) + Q: s is what each token still holds of itself through the residual
stream (1 for D; it halves every block for C, Abnar's identity weight), Q what attention moved
between tokens. So R <- R + M R is s' = s, Q' = Q + M (diag(s) + Q), and the rollout
s' = s / 2, Q' = (Q + M (diag(s) + Q)) / 2.

Readout. Qwen3-VL has no CLS token: the score reaches the LLM through the DeepStack mergers
after blocks 8, 16, 24 and the final merger after block 26 ("routes", grads.RouteTaps). Each
merger turns a 2x2 block into one LLM token y_b. Route t's weight per token is
w_t = ReLU(+-sum_d y_b * g_b) / 4 for the 4 patches of block b, g_b = d(score)/d(y_b): the
LLM token's gradient x input, shared equally by its patches. (Not x . g at the merger's input:
the merger starts with a LayerNorm, which ignores its input's scale, so that is 0.) w_t weighs
the rows of R after block t:
  D       sum_t w_t^T Q_t: what attention carried to the readout tokens. Chefer zeroes his CLS
          token's self-relevance R[0, 0]; here every readout token's own identity term is left
          out, which is W.
  D norm  sum_t w_t^T rownorm(Q_t): every token's Q row scaled to sum 1 (Chefer's rule 10
          normalisation), so a token passes on exactly its readout weight.
  W       sum_t w_t * s_t: that identity part alone, the readout through no attention.
          D + W = sum_t w_t^T R_t, the whole matrix readout.
  C       sum_t w_t^T Q_t for the rollout state.
Per map: [sides (for, against), parts, pairs, n] in token order. D's parts: "all" (the sum),
"r<t>" per route, "norm"; C and W: "all".
"""

from __future__ import annotations

import torch

HEAD_CHUNK = 4          # heads per matmul in update: 4 x n^2 fp32 (130 MB at n = 2880)


def rownorm(Q: torch.Tensor) -> torch.Tensor:
    """Rows scaled to sum 1; a row summing to 0 stays 0."""
    s = Q.sum(-1, keepdim=True)
    return torch.where(s > 0, Q / s.clamp_min(1e-30), torch.zeros_like(Q))


class State:
    """R = diag(s) + Q for one frame pair, fp32."""

    def __init__(self, n: int, device):
        self.s = torch.ones(n, device=device)
        self.Q = torch.zeros(n, n, device=device)

    def gae(self, M: torch.Tensor) -> None:
        """R <- R + M R (Chefer)."""
        self.Q = self.Q + M * self.s + M @ self.Q

    def rollout(self, M: torch.Tensor) -> None:
        """R <- (R + M R) / 2 (Abnar)."""
        self.Q = 0.5 * (self.Q + M * self.s + M @ self.Q)
        self.s = 0.5 * self.s


def edge_weights(probs: torch.Tensor, v: torch.Tensor, dO: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """probs [heads, n, n], v and dO [heads, n, d] -> mean_h ReLU(+-A * dA), dA = dO V^T."""
    heads, n, _ = probs.shape
    plus = torch.zeros(n, n, device=probs.device)
    minus = torch.zeros(n, n, device=probs.device)
    for h in range(0, heads, HEAD_CHUNK):
        x = probs[h:h + HEAD_CHUNK] * (dO[h:h + HEAD_CHUNK] @ v[h:h + HEAD_CHUNK].transpose(1, 2))
        plus += x.clamp_min(0).sum(0)
        minus += (-x).clamp_min(0).sum(0)
    return plus / heads, minus / heads


class ViTRelevance:
    """Reducer (capture.AttentionCapture, which hands it V) for maps D, C, W of one score.

    dO: block -> d(score)/d(attention output) [tokens, heads * d] (grads.OutputGradTaps, CPU);
    w: route block -> signed readout weight per token [tokens] (its LLM token's y . g / 4,
    gradcam._relevance from grads.RouteTaps)."""

    uses_values = True

    def __init__(self, dO: dict[int, torch.Tensor], w: dict[int, torch.Tensor], pairs: int, n: int):
        self.dO, self.w = dO, {L: x.float() for L, x in w.items()}
        self.routes = sorted(w)
        self.pairs, self.n = pairs, n
        self.parts = ["all", *(f"r{L}" for L in self.routes), "norm"]
        self.states: dict[int, dict[str, State]] = {}
        self.out: dict[str, torch.Tensor] = {}
        self.seen: set[tuple[int, int]] = set()

    def update(self, layer, pair, probs, v):
        heads, n, d = v.shape
        dO = self.dO[layer][pair * n:(pair + 1) * n].to(probs.device, torch.float32)
        plus, minus = edge_weights(probs, v, dO.view(n, heads, d).transpose(0, 1))
        st = self.states.get(pair)
        if st is None:
            st = self.states[pair] = {k: State(n, probs.device) for k in ("D+", "D-", "C")}
        st["D+"].gae(plus)
        st["D-"].gae(minus)
        st["C"].rollout(probs.mean(0))
        self.seen.add((layer, pair))
        if layer in self.w:
            self._read(layer, pair, st)

    def _acc(self, key: str, shape) -> torch.Tensor:
        if key not in self.out:
            self.out[key] = torch.zeros(shape)
        return self.out[key]

    def _read(self, layer, pair, st):
        n = self.n
        w = self.w[layer][pair * n:(pair + 1) * n].to(st["C"].Q.device)
        D, C, W = (self._acc(k, (2, parts, self.pairs, n)) for k, parts in
                   (("D", len(self.parts)), ("C", 1), ("W", 1)))
        r = self.parts.index(f"r{layer}")
        for side, (ws, s) in enumerate(((w.clamp_min(0), st["D+"]), ((-w).clamp_min(0), st["D-"]))):
            d = (ws @ s.Q).cpu()
            D[side, 0, pair] += d
            D[side, r, pair] = d
            D[side, -1, pair] += (ws @ rownorm(s.Q)).cpu()
            W[side, 0, pair] += (ws * s.s).cpu()
            C[side, 0, pair] += (ws @ st["C"].Q).cpu()

    def result(self) -> dict:
        missing = [(L, p) for L in range(1 + max(L for L, _ in self.seen)) for p in range(self.pairs)
                   if (L, p) not in self.seen]
        if missing or max(L for L, _ in self.seen) < self.routes[-1]:
            raise RuntimeError(f"relevance missing blocks/pairs {missing[:5]}")
        out = {k: v.numpy() for k, v in self.out.items()}
        D, W = out["D"][:, 0], out["W"][:, 0]                       # [sides, pairs, n]
        stats = {"mixing_share": [float(D[s].sum() / max(D[s].sum() + W[s].sum(), 1e-30)) for s in range(2)],
                 "D_mass_per_pair": D.sum(-1).tolist(), "W_mass_per_pair": W.sum(-1).tolist()}
        return out | {"parts": {"D": self.parts, "C": ["all"], "W": ["all"]}, "stats": stats}
