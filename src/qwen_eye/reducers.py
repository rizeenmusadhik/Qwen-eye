"""One reducer per map: each keeps what it needs from every (layer, pair) softmax.

`update(layer, pair, probs)` gets fp32 probs [heads, n, n] (rows = queries, both axes in
token order, see layout.py); `result()` returns numpy arrays, still in token order. A new map
(rollout, per-head B, ...) is a new class here plus a line in run.py.
"""

from __future__ import annotations

import numpy as np
import torch


class _PerLayerPair:
    def __init__(self):
        self.store: dict[tuple[int, int], torch.Tensor] = {}

    def _stack(self, layers: list[int]) -> np.ndarray:
        pairs = 1 + max(p for _, p in self.store)
        return torch.stack([torch.stack([self.store[(L, p)] for p in range(pairs)]) for L in layers]).numpy()


class Received(_PerLayerPair):
    """Map A: attention each patch receives, averaged over all query patches; per head, every
    layer. Every row sums to 1, so the mean over patches is 1/n; result() is in units of that
    (x average). -> [layers, pairs, heads, n]"""

    def update(self, layer, pair, probs):
        self.store[(layer, pair)] = probs.mean(1).cpu()

    def result(self) -> dict[str, np.ndarray]:
        n = next(iter(self.store.values())).shape[-1]
        return {"A": self._stack(sorted({L for L, _ in self.store})) * n}


class FromBlocks(_PerLayerPair):
    """Map B: where each 2x2 merge block (one LLM token) looks: the mean of its m*m patches'
    attention rows, heads averaged, chosen layers only. x average. -> [layers, pairs, blocks, n]"""

    def __init__(self, layers, merge: int = 2):
        super().__init__()
        self.layers = sorted(layers)
        self.group = merge * merge

    def update(self, layer, pair, probs):
        if layer in self.layers:
            n = probs.shape[-1]
            self.store[(layer, pair)] = probs.mean(0).view(n // self.group, self.group, n).mean(1).cpu()

    def result(self) -> dict[str, np.ndarray]:
        n = next(iter(self.store.values())).shape[-1]
        return {"B": self._stack(self.layers) * n}


class MeanDistance(_PerLayerPair):
    """Per head, the attention-weighted distance in patches from a query to what it attends,
    averaged over queries. ViTs attend nearby early and far late (Dosovitskiy et al. 2021,
    fig. 7): a sanity check on the capture. -> [layers, pairs, heads]"""

    def __init__(self, coords: torch.Tensor):
        super().__init__()
        self.dist = torch.cdist(coords.float(), coords.float())

    def update(self, layer, pair, probs):
        self.dist = self.dist.to(probs.device)
        self.store[(layer, pair)] = (probs * self.dist).sum(-1).mean(-1).cpu()

    def result(self) -> dict[str, np.ndarray]:
        return {"mean_dist": self._stack(sorted({L for L, _ in self.store}))}


class GradCAM(_PerLayerPair):
    """Map E: Grad-CAM (Selvaraju et al. 2017) per frame pair, each pair being its own image
    to the ViT. `update(layer, pair, feats, grads)` gets one pair's block output and
    d(score)/d(it), [n, hidden] in token order (grads.FeatureTaps). alpha = the gradient
    averaged over the pair's patches; s = feats @ alpha, kept signed: ReLU(s) is the usual
    map, ReLU(-s) the counterfactual one. -> {"E": [layers, pairs, n]}"""

    def update(self, layer, pair, feats, grads):
        alpha = grads.float().mean(0)
        self.store[(layer, pair)] = (feats.float() @ alpha).cpu()

    def result(self) -> dict[str, np.ndarray]:
        return {"E": self._stack(sorted({L for L, _ in self.store}))}


class HiResCAM(_PerLayerPair):
    """Map E, HiResCAM variant (Draelos & Carin 2020): each patch's own gradient times its own
    features, summed over features, so where the gradient came from is kept. Grad-CAM first
    averages the gradient over the whole pair (alpha), which spreads it. Signed as GradCAM.
    Summed over the pair it is the first-order change of the score if the pair's features were
    zeroed (gradient x input). -> {"H": [layers, pairs, n]}"""

    def update(self, layer, pair, feats, grads):
        self.store[(layer, pair)] = (feats.float() * grads.float()).sum(-1).cpu()

    def result(self) -> dict[str, np.ndarray]:
        return {"H": self._stack(sorted({L for L, _ in self.store}))}
