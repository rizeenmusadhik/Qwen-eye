"""viewer.html: one self-contained page (no network) to scrub through a clip's maps.

viewer_template.html (vanilla JS + canvas) gets a single JSON payload: the frames as JPEG,
A and B as log-quantised uint8 (quant.py) in base64, per-layer colour-scale percentiles, the
colormap, meta and stats. A carries the head average at head index 0, then the 16 heads.
A gradient page (gradcam.py) adds E: per saved map (maps.py: HiResCAM, Grad-CAM, D, C, W) a
block with its own parts (layers, or relevance parts) and one linear-quantised map per (score,
side), the two sides of one score sharing a colour scale per map and part; plus the item and
the scores' records as G.
"""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import numpy as np

from qwen_eye import quant
from qwen_eye.maps import UNITS, layouts, side_maps

TEMPLATE = Path(__file__).with_name("viewer_template.html")
MARK = "/*__PAYLOAD__*/null"
# Map B costs blocks x patches bytes per (layer, pair): n^2 / 4. An HD MotionBench clip (40x72
# patches, 14 pairs, 5 layers) would put 145 MB of B in the page; above this it is left out.
MAX_B_MB = 48
# The gradient maps cost scores x 2 sides x parts x pairs x patches bytes each (~13 MB in all
# on that clip); above this, D's per-route parts are left out first.
MAX_E_MB = 48


def colormap() -> np.ndarray:
    """[256, 3] uint8 RGB: inferno from OpenCV (eval env), else a black-red-yellow-white ramp."""
    try:
        import cv2
    except ImportError:
        x = np.linspace(0, 1, 256)
        ramp = np.stack([np.clip(1.5 * x, 0, 1), np.clip(1.5 * x - 0.5, 0, 1), np.clip(3 * x - 2, 0, 1)], 1)
        return (ramp * 255).astype(np.uint8)
    return cv2.applyColorMap(np.arange(256, dtype=np.uint8)[:, None], cv2.COLORMAP_INFERNO)[:, 0, ::-1].copy()


def _jpeg(frame: np.ndarray) -> str:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(frame).save(buf, "JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _quantised(x: np.ndarray, per_layer_axes: int) -> dict:
    """x [L, ...]: uint8 per layer, its xmax, and the clip-wide 99th percentile per layer (and
    per index of axis 2 when per_layer_axes == 2: the head choice of map A)."""
    xmax = x.reshape(len(x), -1).max(1)
    if per_layer_axes == 2:
        p99 = np.percentile(np.moveaxis(x, 2, 1).reshape(x.shape[0], x.shape[2], -1), 99, axis=-1)
    else:
        p99 = np.percentile(x.reshape(len(x), -1), 99, axis=-1)
    q = np.stack([quant.encode(x[i], xmax[i]) for i in range(len(x))])
    return {"shape": list(q.shape), "xmax": xmax.tolist(), "p99": p99.tolist(),
            "b64": base64.b64encode(np.ascontiguousarray(q).tobytes()).decode()}


def _quantised_gradcam(x: np.ndarray, group: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """x [maps, parts, pairs, h, w] >= 0 -> linear uint8 of that shape; xmax and the clip-wide
    99th percentile per (map, part), shared by the maps (sides) of one score so they compare."""
    x = x.astype(np.float32)
    xmax, p99 = np.zeros(x.shape[:2]), np.zeros(x.shape[:2])
    for gi in set(group.tolist()):
        sel = np.moveaxis(x[group == gi], 1, 0).reshape(x.shape[1], -1)  # [parts, everything of this score]
        xmax[group == gi] = sel.max(1)
        p99[group == gi] = np.percentile(sel, 99, axis=1)
    q = np.stack([[quant.encode(x[m, i], xmax[m, i], log=False) for i in range(x.shape[1])] for m in range(len(x))])
    return q, xmax, p99


def payload(maps: dict, meta: dict, stats: dict, gradcam: dict | None = None) -> dict:
    A = maps["A"].astype(np.float32)                                   # [L, pairs, heads, h, w]
    A = np.concatenate([A.mean(2, keepdims=True), A], axis=2)          # head 0 = average
    frames = maps["frames"]
    times = list(meta["frame_times"])
    padded = [i >= len(times) for i in range(len(frames))]
    times += [times[-1]] * (len(frames) - len(times))
    return {
        "meta": {**meta, "frame_times": times, "padded": padded},
        "stats": stats,
        "frames": [_jpeg(f) for f in frames],
        "A": _quantised(A, 2),
        "B": _quantised(maps["B"].astype(np.float32), 1) if maps["B"].size <= MAX_B_MB * 2**20 else None,
        "lut": colormap().tolist(),
    } | (_gradcam_payload(gradcam) if gradcam else {})


def _gradcam_payload(gradcam: dict) -> dict:
    """gradcam: {"maps": gradcam.npz's arrays, "meta": its meta.json}. E in the payload: the
    sides' labels and score groups, and per map {key, name, unit, parts, q [maps, parts, pairs,
    h, w] in b64, xmax/p99 [maps, parts]}."""
    gm = gradcam["meta"]
    scores = [{k: v for k, v in r.items() if k not in ("grad_norms", "attn_grad_norms")} for r in gm["scores"]]
    lay = layouts(gm, gradcam["maps"])
    per = {k: side_maps(gradcam["maps"][k], info, gm["scores"]) for k, info in lay.items()}
    first = next(iter(per.values()))
    group = np.array([g["group"] for g in first])
    blocks = {k: (list(info["parts"]), np.stack([g["x"] for g in per[k]])) for k, info in lay.items()}
    dropped = []
    if sum(x.size for _, x in blocks.values()) > MAX_E_MB * 2**20:
        for k, (parts, x) in blocks.items():
            keep = [i for i, p in enumerate(parts) if not p.startswith("r")]
            if len(keep) < len(parts):
                dropped += [f"{k}@{p}" for i, p in enumerate(parts) if i not in keep]
                blocks[k] = ([parts[i] for i in keep], x[:, keep])
    methods = []
    for k, (parts, x) in blocks.items():
        q, xmax, p99 = _quantised_gradcam(x, group)
        methods.append({"key": k, "name": lay[k]["name"], "unit": UNITS[k], "parts": parts, "shape": list(q.shape),
                        "xmax": xmax.tolist(), "p99": p99.tolist(),
                        "b64": base64.b64encode(np.ascontiguousarray(q).tobytes()).decode()})
    E = {"labels": [g["label"] for g in first], "group": group.tolist(), "methods": methods, "dropped": dropped}
    return {"E": E,
            "G": {"item": gm["item"], "phrase_prompt": gm["phrase_prompt"], "scores": scores,
                  "checks_failed": gm["checks_failed"]}}


def write(path: Path, maps: dict, meta: dict, stats: dict, gradcam: dict | None = None) -> Path:
    html = TEMPLATE.read_text()
    if html.count(MARK) != 1:
        raise RuntimeError(f"{TEMPLATE} must contain {MARK} exactly once")
    data = json.dumps(payload(maps, meta, stats, gradcam), separators=(",", ":")).replace("</", "<\\/")
    path.write_text(html.replace(MARK, data))
    return path
