"""The gradient maps a gradcam dir holds (gradcam.npz, described by meta.json["maps"]).

Each map is stored per score in grid order, in one of two layouts:
  signed  [scores, parts, pairs, h, w]     E, H: one signed value; side "for" = ReLU(+x),
                                           "against" = ReLU(-x)
  2       [scores, 2, parts, pairs, h, w]  D, C, W: the two sides computed separately
                                           (relevance.py), both >= 0
"parts" are the layers (E, H: "2", "8", ...) or relevance parts ("all", "r8", ..., "norm").
A score's sides are its target's maps, (label, sign): sign +1 is side 0, -1 side 1.
Dirs from before map D have no meta["maps"]: E and H, parts = meta["layers"].
"""

from __future__ import annotations

import numpy as np

NAMES = {"H": "HiResCAM", "E": "Grad-CAM", "D": "Chefer relevance", "C": "rollout", "W": "readout only"}
UNITS = {"H": "gradient units", "E": "gradient units", "D": "relevance", "C": "relevance", "W": "relevance"}


def layouts(meta: dict, keys) -> dict[str, dict]:
    """{key: {"name", "sides", "parts"}} for the maps in `keys` (gradcam.npz's files), in NAMES order."""
    saved = meta.get("maps") or {k: {"name": NAMES[k], "sides": "signed", "parts": [str(L) for L in meta["layers"]]}
                                 for k in ("H", "E")}
    return {k: saved[k] for k in NAMES if k in saved and k in keys}


def side_maps(x: np.ndarray, info: dict, scores: list[dict]) -> list[dict]:
    """x [scores, ...] in its layout -> one map >= 0 [parts, pairs, h, w] per side of every score."""
    out = []
    for s, rec in enumerate(scores):
        for m in rec["maps"]:
            side = x[s, 0 if m["sign"] > 0 else 1] if info["sides"] == 2 else np.maximum(m["sign"] * x[s], 0)
            out.append({"label": m["label"], "group": s, "sign": m["sign"], "x": side})
    return out


def side_pair(x: np.ndarray, info: dict, part: str) -> tuple[np.ndarray, np.ndarray]:
    """One score's map x [...] -> its (for, against) maps [pairs, h, w] at `part`."""
    i = info["parts"].index(part)
    if info["sides"] == 2:
        return x[0, i], x[1, i]
    return x[i], -x[i]
