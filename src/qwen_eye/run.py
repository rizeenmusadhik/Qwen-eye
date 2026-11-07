"""One clip end to end: eval-identical input -> vision tower with capture -> checks -> saved
maps -> viewer.html + attention.mp4.

Output dir (one per clip):
  maps.npz    A [27, pairs, heads, h, w], B [len(layers), pairs, blocks, h, w] (x average,
              float16, grid order), mean_dist [27, pairs, heads], frames [2*pairs, H, W, 3]
  meta.json   grid, frame times, the LLM's per-pair "<x seconds>" labels, settings, checks
  stats.json  per-layer mean attention distance, top-1% share, max x average
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from qwen_eye import layout
from qwen_eye.capture import AttentionCapture, attach
from qwen_eye.reducers import FromBlocks, MeanDistance, Received
from qwen_eye.tower import load_vision_tower, preprocess_video

DEFAULT_LAYERS = (2, 8, 16, 24, 26)
# max |softmax @ V - sdpa| and max |pair 0 alone - pair 0 in clip| / max |output|
TOLERANCE = {torch.float32: {"recon_gap": 1e-4, "pair0_alone_vs_in_clip": 1e-4},
             torch.bfloat16: {"recon_gap": 5e-2, "pair0_alone_vs_in_clip": 5e-2}}


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def layer_stats(A: np.ndarray, mean_dist: np.ndarray, coords: torch.Tensor) -> dict:
    """A [L, pairs, heads, n] x average, token order; mean_dist [L, pairs, heads]."""
    n = A.shape[-1]
    top = max(1, n // 100)
    received = A.mean(2) / n                                            # [L, pairs, n], sums to 1
    share = np.sort(received, axis=-1)[..., -top:].sum(-1).mean(-1)
    return {
        "uniform_mean_dist": float(torch.cdist(coords.float(), coords.float()).mean()),
        "uniform_top1_share": top / n,
        "layers": [{"layer": L, "mean_dist": float(mean_dist[L].mean()), "top1_share": float(share[L]),
                    "max_x_average": float(A[L].mean(1).max())} for L in range(A.shape[0])],
    }


def analyse(video: str, model_dir: str, out_dir: Path, *, variant: dict, layers=DEFAULT_LAYERS,
            device: str = "auto", dtype: torch.dtype | None = None, render_outputs: bool = True) -> dict:
    """dtype: default float32 on CPU, bfloat16 (the eval's) on GPU."""
    device = resolve_device(device)
    dtype = dtype or (torch.float32 if device == "cpu" else torch.bfloat16)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    vi = preprocess_video(model_dir, video, variant)
    t, h, w = vi.grid
    n = h * w
    print(f"{video}: {len(vi.frame_times)} frames -> {t} pairs of {h}x{w} patches "
          f"({h * vi.patch}x{w * vi.patch} px); {vi.video_args}; {device} {dtype}", flush=True)

    tower = load_vision_tower(model_dir, device, dtype)
    bad = [L for L in layers if not 0 <= L < len(tower.blocks)]
    if bad:
        raise SystemExit(f"layers {bad} outside 0-{len(tower.blocks) - 1}")
    order = layout.block_order(h, w, vi.merge)
    coords = layout.token_coords(order, w)
    reducers = [Received(), FromBlocks(layers, vi.merge), MeanDistance(coords)]
    cap = AttentionCapture(tower, reducers)
    grid = torch.tensor([[t, h, w]], device=device)
    attach(tower, cap)
    try:
        with torch.inference_mode():
            merged, _ = tower(vi.pixels.to(device, dtype), grid_thw=grid)
    finally:
        attach(tower, None)
    # the first pair alone through stock sdpa: the capture changed nothing, pairs don't interact
    with torch.inference_mode():
        alone, _ = tower(vi.pixels[:n].to(device, dtype), grid_thw=torch.tensor([[1, h, w]], device=device))
    checks = {
        "recon_gap": cap.recon_gap,
        "pair0_alone_vs_in_clip": (alone.float() - merged[: alone.shape[0]].float()).abs().max().item()
                                  / merged.float().abs().max().item(),
    }
    failed = [k for k, v in checks.items() if v > TOLERANCE[dtype][k]]
    if cap.calls != [t] * len(tower.blocks):
        failed.append(f"calls per layer {sorted(set(cap.calls))} != {t}")
    print(f"forward + capture + check {time.time() - t0:.0f}s; checks "
          + ", ".join(f"{k} {v:.1e}" for k, v in checks.items()), flush=True)

    maps = {}
    for r in reducers:
        maps.update(r.result())
    stats = layer_stats(maps["A"], maps["mean_dist"], coords)
    to_grid = lambda x: layout.to_grid(torch.from_numpy(x), order, h, w).numpy().astype(np.float16)  # noqa: E731
    frames = layout.frames_from_pixels(vi.pixels, vi.grid, order, patch=vi.patch, tpatch=vi.tpatch,
                                       mean=vi.mean, std=vi.std)
    np.savez_compressed(out_dir / "maps.npz", A=to_grid(maps["A"]), B=to_grid(maps["B"]),
                        mean_dist=maps["mean_dist"], frames=frames)
    meta = {
        "video": str(video), "model": str(model_dir), "created": datetime.now().isoformat(timespec="seconds"),
        "grid": [t, h, w], "patch": vi.patch, "tpatch": vi.tpatch, "merge": vi.merge,
        "frame_times": vi.frame_times, "pair_labels": vi.pair_labels, "video_args": vi.video_args,
        "fps": vi.video_args["fps"], "layers_B": sorted(layers), "heads": int(maps["A"].shape[2]),
        "device": device, "dtype": str(dtype).replace("torch.", ""), "checks": checks,
        "tolerance": TOLERANCE[dtype], "checks_failed": failed,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=1))
    print(format_stats(stats), flush=True)
    if failed:
        raise SystemExit(f"{out_dir}: checks failed: {failed} (tolerance {TOLERANCE[dtype]})")
    if render_outputs:
        from qwen_eye.render import render
        render(out_dir)
    return meta


def format_stats(stats: dict) -> str:
    lines = [f"{'layer':>5} {'mean dist':>9} {'top1% share':>11} {'max x avg':>9}   uniform: dist "
             f"{stats['uniform_mean_dist']:.1f} patches, top-1% share {stats['uniform_top1_share']:.3f}"]
    lines += [f"{s['layer']:>5} {s['mean_dist']:>9.2f} {s['top1_share']:>11.3f} {s['max_x_average']:>9.1f}"
              for s in stats["layers"]]
    return "\n".join(lines)


def run_plan(plan: dict) -> int:
    """Every clip of an attn plan (`qwen-eye attn`, `python -m qwen_eye run`)."""
    for video in plan["videos"]:
        analyse(video, plan["model"], Path(plan["out_root"]) / Path(video).stem, variant=plan["variant_args"],
                layers=plan["layers"], device=plan["device"])
    return 0
