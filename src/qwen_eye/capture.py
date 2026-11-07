"""Capture the Qwen3-VL vision tower's attention without changing its output.

transformers 4.57.6 `Qwen3VLVisionAttention` runs attention once per frame pair (one
cu_seqlens chunk; every implementation but flash_attention_2 splits them) through
ALL_ATTENTION_FUNCTIONS[config._attn_implementation], and keeps only the output. An
AttentionCapture registered there returns the stock sdpa result, so the tower computes
exactly what it computes in eval. On the side it recomputes the softmax in fp32 and hands it
to reducers, which keep only what their map needs (one full [heads, n, n] table is ~50 MB
per layer and pair). A reducer with `uses_values = True` also gets V (fp32 [heads, n, d]):
map D needs it for d(score)/d(attention) = dO @ V^T.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterable

import torch
from transformers import AttentionInterface
from transformers.integrations.sdpa_attention import sdpa_attention_forward

IMPL = "qwen_eye_capture"


class AttentionCapture:
    """Called once per block per frame pair, pairs in order. `recon_gap` is the largest
    max|softmax @ V - sdpa output| / max|sdpa output| seen: ~0 means the captured softmax is
    the one the tower used. Relative, like run.py's other check: late blocks' outputs are large
    and GPU fp32 kernels sum in another order (absolute 1.6e-4 at 40x72 patches on an A100)."""

    def __init__(self, tower, reducers: Iterable):
        self.layer_of = {id(b.attn): i for i, b in enumerate(tower.blocks)}
        self.reducers = list(reducers)
        self.calls = [0] * len(tower.blocks)
        self.recon_gap = 0.0

    def __call__(self, module, query, key, value, attention_mask, **kwargs):
        out, _ = sdpa_attention_forward(module, query, key, value, attention_mask, **kwargs)
        layer = self.layer_of[id(module)]
        pair = self.calls[layer]
        self.calls[layer] += 1
        q, k, v = (x[0].float() for x in (query, key, value))     # [heads, n, d]
        probs = torch.softmax(q @ k.transpose(1, 2) * kwargs["scaling"], dim=-1)
        ref = out[0].float()
        gap = (((probs @ v).transpose(0, 1) - ref).abs().max() / ref.abs().max().clamp_min(1e-30)).item()
        self.recon_gap = max(self.recon_gap, gap)
        for r in self.reducers:
            if getattr(r, "uses_values", False):
                r.update(layer, pair, probs, v)
            else:
                r.update(layer, pair, probs)
        return out, None


def attach(tower, capture: AttentionCapture | None) -> None:
    """Route every block's attention through `capture`, or back to stock sdpa (None)."""
    if capture is not None:
        AttentionInterface.register(IMPL, capture)
    tower.config._attn_implementation = IMPL if capture is not None else "sdpa"
    if any(b.attn.config is not tower.config for b in tower.blocks):
        raise RuntimeError("vision blocks do not share the tower's config; the switch would not reach them")


@contextmanager
def attached(tower, capture: AttentionCapture):
    """attach for one pass inside the full model: back to stock sdpa on exit, also on error."""
    attach(tower, capture)
    try:
        yield capture
    finally:
        attach(tower, None)
