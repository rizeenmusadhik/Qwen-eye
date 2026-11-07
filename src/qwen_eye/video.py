"""attention.mp4 + preview.png from a saved clip. Columns: the frame, then one per layer;
rows: A (heads averaged), then B for each point's 2x2 block. Played at the sampling fps, so
every frame shown is one the model got. Colours as in the viewer: colormap and the clip's
99th percentile per layer and row. gradcam.mp4 (write_gradcam) has rows A and then gradient
maps (render.mp4_rows), each with its own columns (a signed map's layers, D's routes); a short
row is padded with the bare frame. Needs OpenCV and PyAV.
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import av
import cv2
import numpy as np

from qwen_eye import layout
from qwen_eye.viewer import colormap


def _label(img: np.ndarray, text: str) -> None:
    for colour, width in (((0, 0, 0), 3), ((255, 255, 255), 1)):
        cv2.putText(img, text, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, colour, width, cv2.LINE_AA)


def _write_mp4(path: Path, frames: list[np.ndarray], fps: float) -> None:
    with av.open(str(path), "w") as c:
        s = c.add_stream("libx264", rate=Fraction(fps).limit_denominator(1000))
        s.height, s.width = frames[0].shape[:2]
        s.pix_fmt = "yuv420p"
        for f in frames:
            for pkt in s.encode(av.VideoFrame.from_ndarray(f, format="rgb24")):
                c.mux(pkt)
        for pkt in s.encode():
            c.mux(pkt)


def _compose(out_dir: Path, name: str, frames: np.ndarray, meta: dict, rows: list[dict], *,
             panel_width: int, alpha: float) -> list[Path]:
    """rows: {"label": str, "box": (row, col) of a 2x2 block or None, "cells": [(maps
    [pairs, h, w], vmax, label)]}; the frame, then one column per cell."""
    t, h, w = meta["grid"]
    mg = meta["merge"]
    lut = colormap()
    pw = panel_width
    ph = round(frames.shape[1] * pw / frames.shape[2])
    times = list(meta["frame_times"]) + [meta["frame_times"][-1]] * (len(frames) - len(meta["frame_times"]))
    columns = max(len(r["cells"]) for r in rows)

    def over(img, grid, vmax):
        m = cv2.resize(grid, (pw, ph), interpolation=cv2.INTER_LINEAR)
        heat = lut[(np.clip(m / max(vmax, 1e-30), 0, 1) * 255).astype(np.uint8)]
        return (alpha * heat + (1 - alpha) * img).astype(np.uint8)

    out = []
    for f, frame in enumerate(frames):
        pair = min(f // meta["tpatch"], t - 1)
        img = cv2.resize(frame, (pw, ph), interpolation=cv2.INTER_AREA)
        stacked = []
        for i, row in enumerate(rows):
            box = None
            if row["box"] is not None:
                br, bc = (row["box"][0] // mg) * mg, (row["box"][1] // mg) * mg
                box = ((int(bc * pw / w), int(br * ph / h)), (int((bc + mg) * pw / w), int((br + mg) * ph / h)))
            first = img.copy()
            if box:
                cv2.rectangle(first, *box, (74, 222, 128), 2)
            _label(first, row["label"] if i else f"t={times[f]:.2f}s pair {pair + 1}/{t}"
                   + (" pad" if f >= len(meta["frame_times"]) else ""))
            cells = [first]
            for maps, vmax, text in row["cells"]:
                o = over(img, maps[pair], vmax)
                if box:
                    cv2.rectangle(o, *box, (74, 222, 128), 1)
                _label(o, text)
                cells.append(o)
            cells += [img] * (columns - len(row["cells"]))
            stacked.append(np.concatenate(cells, axis=1))
        canvas = np.concatenate(stacked, axis=0)
        out.append(np.pad(canvas, ((0, canvas.shape[0] % 2), (0, canvas.shape[1] % 2), (0, 0))))

    # one frame per pair, stacked; JPEG past the two-row A/B sheet (a PNG of 4+ rows is ~16 MB)
    mp4 = out_dir / f"{name}.mp4"
    preview = out_dir / ("preview.png" if name == "attention" else f"{name}_preview.jpg")
    _write_mp4(mp4, out, meta["fps"])
    cv2.imwrite(str(preview), np.concatenate(out[:: meta["tpatch"]], axis=0)[..., ::-1],
                [cv2.IMWRITE_JPEG_QUALITY, 85] if preview.suffix == ".jpg" else [])
    return [mp4, preview]


def _a_row(maps: dict, layers) -> dict:
    A = maps["A"].astype(np.float32).mean(2)                    # [27, pairs, h, w], heads averaged
    cells = []
    for L in layers:
        vmax = float(np.percentile(A[L], 99))
        cells.append((A[L], vmax, f"A received L{L} 0-{vmax:.1f}x"))
    return {"label": "A received", "box": None, "cells": cells}


def write(out_dir: Path, maps: dict, meta: dict, *, points, layers=None, panel_width: int = 272,
          alpha: float = 0.55) -> list[Path]:
    t, h, w = meta["grid"]
    layers = list(layers or meta["layers_B"])
    missing = [L for L in layers if L not in meta["layers_B"]]
    if missing:
        raise SystemExit(f"no B maps saved for layers {missing}; saved: {meta['layers_B']} (rerun qwen-eye attn --layers)")
    B = maps["B"].astype(np.float32)                            # [len(layers_B), pairs, blocks, h, w]
    rows = [_a_row(maps, layers)]
    for x, y in points:
        r, c = min(int(y * h), h - 1), min(int(x * w), w - 1)
        b = layout.block_of(r, c, w, meta["merge"])
        cells = []
        for L in layers:
            grids = B[meta["layers_B"].index(L), :, b]
            vmax = float(np.percentile(grids, 99))
            cells.append((grids, vmax, f"B from block L{L} 0-{vmax:.1f}x"))
        rows.append({"label": f"B block at ({x:.2f},{y:.2f})", "box": (r, c), "cells": cells})
    return _compose(out_dir, "attention", maps["frames"], meta, rows, panel_width=panel_width, alpha=alpha)


def write_gradcam(out_dir: Path, maps: dict, meta: dict, erows: list[dict], layers, *, panel_width: int = 272,
                  alpha: float = 0.55) -> list[Path]:
    """gradcam.mp4: row A at `layers` (where the ViT attends, for reference), then one row per
    gradient map: {"label", "cells": [(map [pairs, h, w] >= 0, vmax, label)]} (render.mp4_rows)."""
    rows = [_a_row(maps, layers)]
    for g in erows:
        cells = [(x.astype(np.float32), vmax, text) for x, vmax, text in g["cells"]]
        rows.append({"label": g["label"], "box": None, "cells": cells})
    return _compose(out_dir, "gradcam", maps["frames"], meta, rows, panel_width=panel_width, alpha=alpha)
