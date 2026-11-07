"""Draw a saved clip (run.py's maps.npz + meta.json + stats.json): viewer.html and, where
OpenCV and PyAV are installed, attention.mp4 + preview.png. No model needed,
so `qwen-eye render` can redraw with other points or layers in seconds.
render_gradcam draws a gradient-map dir (gradcam.py) the same way, over its clip's A/B maps.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from qwen_eye import viewer
from qwen_eye.maps import layouts, side_maps


def load(out_dir: Path) -> tuple[dict, dict, dict]:
    with np.load(out_dir / "maps.npz") as z:
        maps = {k: z[k] for k in z.files}
    meta = json.loads((out_dir / "meta.json").read_text())
    stats = json.loads((out_dir / "stats.json").read_text())
    return maps, meta, stats


def render(out_dir: Path, *, points=None, layers=None) -> list[Path]:
    out_dir = Path(out_dir)
    maps, meta, stats = load(out_dir)
    written = [viewer.write(out_dir / "viewer.html", maps, meta, stats)]
    try:
        from qwen_eye import video
    except ImportError as e:            # cv2 / av missing
        print(f"mp4 skipped ({e}); viewer.html written")
    else:
        written += video.write(out_dir, maps, meta, points=points or [(0.5, 0.5)], layers=layers)
    for p in written:
        print(f"wrote {p} ({p.stat().st_size / 1e6:.1f} MB)")
    return written


def _cell_label(p: str) -> str:
    return {"all": "all routes"}.get(p) or (f"route {p[1:]}" if p.startswith("r") else f"L{p}")


def mp4_rows(gmaps: dict, gmeta: dict) -> list[dict]:
    """gradcam.mp4's rows after A: the answer contrast's both sides, then each phrase's "yes"
    side. With D saved, the answer in H and D and the phrases in D (E, C, W stay in the viewer);
    without, the answer in every map and the phrases in the first. Columns: a signed map's
    layers, D's "all" then its routes. vmax per cell: the clip's 99th percentile over both sides
    of that score in that map and part, as in the viewer."""
    lay = layouts(gmeta, gmaps)
    scores = gmeta["scores"]
    answer_keys = [k for k in ("H", "D") if k in lay] if "D" in lay else list(lay)
    phrase_key = "D" if "D" in lay else next(iter(lay))
    rows = []
    for k in dict.fromkeys(answer_keys + [phrase_key]):
        info = lay[k]
        parts = [(i, p) for i, p in enumerate(info["parts"]) if p != "norm"]
        per = side_maps(gmaps[k], info, scores)
        for g in per:
            answer = scores[g["group"]]["name"] == "answer"
            if not (k in answer_keys if answer else k == phrase_key and g["sign"] > 0):
                continue
            both = np.stack([o["x"] for o in per if o["group"] == g["group"]])         # [sides, parts, pairs, h, w]
            vmax = np.percentile(np.moveaxis(both, 1, 0).reshape(both.shape[1], -1), 99, axis=1)
            rows.append({"label": f"{info['name']} {g['label']}", "answer": answer,
                         "cells": [(g["x"][i], float(vmax[i]), _cell_label(p)) for i, p in parts]})
    return sorted(rows, key=lambda r: not r["answer"])


def render_gradcam(out_dir: Path) -> list[Path]:
    """<clip>/gradcam/<tag>/: viewer.html with A, B and E, and gradcam.mp4 (A row + E rows)."""
    out_dir = Path(out_dir)
    gmeta = json.loads((out_dir / "meta.json").read_text())
    with np.load(out_dir / "gradcam.npz") as z:
        gmaps = {k: z[k] for k in z.files}
    maps, meta, stats = load(Path(gmeta["clip_dir"]))
    if list(meta["grid"]) != list(gmeta["grid"]):
        raise SystemExit(f"{out_dir}: Grad-CAM grid {gmeta['grid']} != its clip's {meta['grid']}")
    written = [viewer.write(out_dir / "viewer.html", maps, meta, stats, gradcam={"maps": gmaps, "meta": gmeta})]
    try:
        from qwen_eye import video
    except ImportError as e:
        print(f"mp4 skipped ({e}); viewer.html written")
    else:
        written += video.write_gradcam(out_dir, maps, meta, mp4_rows(gmaps, gmeta), gmeta["layers"])
    for p in written:
        print(f"wrote {p} ({p.stat().st_size / 1e6:.1f} MB)")
    return written
