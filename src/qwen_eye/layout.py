"""Qwen3-VL video token layout: which patch each vision token is, and the frames it came from.

The video processor emits a frame pair's patches by 2x2 merge block, not row by row:
video_processing_qwen3_vl.py permutes (t, T, C, h/m, m, P, w/m, m, P) to
(t, h/m, w/m, m, m, C, T, P, P) before `flatten_patches`, and the vision tower keeps that
order. So the m*m tokens of one block are contiguous and block b is merged into LLM token b.
Anything drawn on a frame goes through `to_grid` first.
"""

from __future__ import annotations

import numpy as np
import torch


def block_order(h: int, w: int, merge: int = 2) -> torch.Tensor:
    """Row-major patch index of every token of one frame pair, in the processor's order."""
    return torch.arange(h * w).view(h // merge, merge, w // merge, merge).permute(0, 2, 1, 3).reshape(-1)


def to_grid(x: torch.Tensor, order: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """[..., h*w] in token order -> [..., h, w] row-major."""
    g = x.new_empty(x.shape)
    g[..., order] = x
    return g.view(*x.shape[:-1], h, w)


def token_coords(order: torch.Tensor, w: int) -> torch.Tensor:
    """(row, col) in patches of every token, [n, 2]."""
    return torch.stack([order // w, order % w], dim=1)


def block_of(row: int, col: int, w: int, merge: int = 2) -> int:
    """Index of the merge block (= LLM token) holding patch (row, col)."""
    return (row // merge) * (w // merge) + col // merge


def frames_from_pixels(pixels: torch.Tensor, grid: tuple[int, int, int], order: torch.Tensor, *,
                       patch: int, tpatch: int, mean, std) -> np.ndarray:
    """Undo the processor's patchify: [t*h*w, C*T*P*P] -> uint8 [t*T, H, W, 3], exactly the
    frames the vision tower saw (resized, paired, last frame repeated for an odd count).
    A pixel row is (channel, frame in pair, y, x)."""
    t, h, w = grid
    x = pixels.float().view(t, h * w, 3, tpatch, patch, patch).permute(0, 2, 3, 4, 5, 1)
    x = to_grid(x, order, h, w)                                    # [t, C, T, Py, Px, h, w]
    x = x.permute(0, 2, 5, 3, 6, 4, 1).reshape(t * tpatch, h * patch, w * patch, 3)
    x = x * torch.tensor(std, dtype=x.dtype) + torch.tensor(mean, dtype=x.dtype)
    return (x * 255).round().clamp(0, 255).to(torch.uint8).numpy()
