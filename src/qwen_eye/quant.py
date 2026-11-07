"""Maps (>= 0) to uint8 for the viewer, on a log scale by default.

q = round(255 * log1p(x) / log1p(xmax)) for attention maps in x average: a few sink patches
at 100x average then cost the rest little resolution (the step is log1p(xmax)/255 in
log1p(x), under 2% relative for xmax = 100). Grad-CAM maps have no natural unit and are
drawn linearly: q = round(255 * x / xmax). viewer_template.html inverts both per layer.
"""

from __future__ import annotations

import numpy as np


def encode(x: np.ndarray, xmax: float, log: bool = True) -> np.ndarray:
    x = np.maximum(x, 0).astype(np.float32)
    if not log:
        return np.clip(np.rint(x * (255.0 / max(float(xmax), 1e-30))), 0, 255).astype(np.uint8)
    k = 255.0 / np.log1p(max(float(xmax), 1e-6))
    return np.clip(np.rint(np.log1p(x) * k), 0, 255).astype(np.uint8)


def decode(q: np.ndarray, xmax: float, log: bool = True) -> np.ndarray:
    if not log:
        return q.astype(np.float32) * (max(float(xmax), 1e-30) / 255.0)
    return np.expm1(q.astype(np.float32) * (np.log1p(max(float(xmax), 1e-6)) / 255.0))
