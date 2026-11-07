"""Gradients of a score with respect to inside the vision tower ("taps"), for gradient maps.

FeatureTaps (map E, Grad-CAM): d(score)/d(block output) at chosen ViT blocks. By default the
lowest tapped block's output is handed on as a fresh leaf that requires grad, so the blocks below
it build no graph (weights are frozen, pixels need no grad); every route out of the tapped
blocks (later blocks, the DeepStack mergers, the final merger) still reaches the score.
Checkpointed blocks run their forward again during backward; a tap keeps only the first call.

Map D (relevance.py) needs the graph down to block 0 and two more taps:
  EmbedLeaf       the patch embedding's output as the leaf instead (FeatureTaps(cut=False))
  OutputGradTaps  d(score)/d(attention output) per block, the input of attn.proj: with it,
                  d(score)/d(attention) = dO @ V^T per head, no attention table in the graph
  RouteTaps       per route out of the ViT (each DeepStack merger, the final merger): its output,
                  the LLM tokens, and their gradient
"""

from __future__ import annotations

import functools

import torch


class FeatureTaps:
    def __init__(self, tower, layers, cut: bool = True):
        self.layers = sorted(set(layers))
        self.lowest = self.layers[0]
        self.cut = cut
        self.handles = [tower.blocks[L].register_forward_hook(functools.partial(self._forward, L))
                        for L in self.layers]
        self.reset()

    def reset(self) -> None:
        self.feats: dict[int, torch.Tensor] = {}
        self.grads: dict[int, torch.Tensor] = {}

    def _forward(self, layer, module, args, output):
        if layer in self.feats or not torch.is_grad_enabled():      # recompute, or a no-grad pass
            return None
        if layer == self.lowest and self.cut:
            output = output.detach().requires_grad_(True)
        self.feats[layer] = output
        output.register_hook(functools.partial(self._backward, layer))
        return output

    def _backward(self, layer, grad):
        self.grads[layer] = grad.detach()

    def result(self) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
        """{layer: (features, gradient)}, both [tokens, hidden] in the tower's token order."""
        missing = [L for L in self.layers if L not in self.grads]
        if missing:
            raise RuntimeError(f"no gradient reached layers {missing}")
        return {L: (self.feats[L].detach(), self.grads[L]) for L in self.layers}

    def remove(self) -> None:
        """Unhook and drop the stored tensors. Each stored output's gradient hook refers back to
        this object, a cycle that otherwise kept every score's features and gradients on the GPU
        (+0.9 GB per score on a 40x72-patch clip)."""
        for h in self.handles:
            h.remove()
        self.reset()


class EmbedLeaf:
    """The patch embedding's output becomes a fresh leaf that requires grad, so the graph starts
    at block 0's input (the position embedding is added after it). The embedding runs outside
    every checkpoint, so its forward is called once."""

    def __init__(self, tower):
        self.handle = tower.patch_embed.register_forward_hook(self._forward)

    @staticmethod
    def _forward(module, args, output):
        return output.detach().requires_grad_(True) if torch.is_grad_enabled() else None

    def remove(self) -> None:
        self.handle.remove()


class OutputGradTaps:
    """d(score)/d(attention output) of every ViT block: the input of `attn.proj`,
    [tokens, heads * head_dim], pairs in row blocks and heads in column blocks (transformers
    concatenates the per-pair outputs [1, n, heads, d] and reshapes). Copied to the CPU as it
    arrives (27 x 93 MB on a 40x72-patch clip). A forward pre-hook marks the tensor on a block's
    first call only: under checkpointing the forward runs again in backward, and the first
    tensor's hook still fires with the gradient."""

    def __init__(self, tower):
        self.handles = [b.attn.proj.register_forward_pre_hook(functools.partial(self._forward, i))
                        for i, b in enumerate(tower.blocks)]
        self.depth = len(tower.blocks)
        self.calls = [0] * self.depth
        self.reset()

    def reset(self) -> None:
        self.marked: set[int] = set()
        self.grads: dict[int, torch.Tensor] = {}

    def _forward(self, layer, module, args):
        if not torch.is_grad_enabled():
            return None
        self.calls[layer] += 1
        if layer not in self.marked:
            self.marked.add(layer)
            args[0].register_hook(functools.partial(self._backward, layer))
        return None

    def _backward(self, layer, grad):
        self.grads[layer] = grad.detach().to("cpu", copy=True)

    def result(self) -> dict[int, torch.Tensor]:
        missing = [L for L in range(self.depth) if L not in self.grads]
        if missing:
            raise RuntimeError(f"no attention-output gradient reached blocks {missing}")
        return dict(self.grads)

    def remove(self) -> None:
        for h in self.handles:
            h.remove()
        self.reset()


class RouteTaps:
    """The routes by which the ViT reaches the LLM: a DeepStack merger after each of
    `deepstack_visual_indexes`, and the final merger after the last block. Each merger's output,
    one LLM token per 2x2 merge block ([tokens / 4, LLM hidden], blocks in token order), and its
    gradient. That output feeds only its route (the final merger's the input embeddings, a
    DeepStack merger's the residual stream after LLM layer 0, 1 or 2). Keyed by the block whose
    output the merger reads. The mergers run outside every checkpoint.

    The output, not the merger's input: every merger starts with a LayerNorm, which ignores the
    scale of its input, so input . gradient is 0 there (per token for the final merger; per 2x2
    block for a DeepStack merger, whose norm spans the block's 4 tokens)."""

    def __init__(self, tower):
        mergers = list(zip(tower.deepstack_visual_indexes, tower.deepstack_merger_list))
        mergers.append((len(tower.blocks) - 1, tower.merger))
        self.layers = [L for L, _ in mergers]
        self.handles = [m.register_forward_hook(functools.partial(self._forward, L)) for L, m in mergers]
        self.reset()

    def reset(self) -> None:
        self.feats: dict[int, torch.Tensor] = {}
        self.grads: dict[int, torch.Tensor] = {}

    def _forward(self, layer, module, args, output):
        if layer in self.feats or not torch.is_grad_enabled():
            return None
        self.feats[layer] = output
        output.register_hook(functools.partial(self._backward, layer))
        return None

    def _backward(self, layer, grad):
        self.grads[layer] = grad.detach()

    def result(self) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
        """{block: (merger output, its gradient)}, [tokens / 4, LLM hidden]."""
        missing = [L for L in self.layers if L not in self.grads]
        if missing:
            raise RuntimeError(f"no gradient reached the mergers after blocks {missing}")
        return {L: (self.feats[L].detach(), self.grads[L]) for L in self.layers}

    def remove(self) -> None:
        """As FeatureTaps.remove: the outputs' hooks refer back to this object."""
        for h in self.handles:
            h.remove()
        self.reset()


def backprop(model, batch: dict, target, taps: FeatureTaps) -> tuple[float, torch.Tensor]:
    """One forward + backward of `target`'s score (call inside fullmodel.grad_mode); the taps
    then hold features and gradients. Returns the score and the kept logits."""
    taps.reset()
    out = model(**batch, use_cache=False, logits_to_keep=target.keep)
    score = target.score(out.logits)
    score.backward()
    return float(score.detach()), out.logits.detach()
