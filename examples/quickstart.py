"""Qwen-eye from Python: maps A/B for a clip, then gradient maps for a question about it.

    python examples/quickstart.py clip.mp4 [--model DIR_OR_HF_ID] [--out out]

Maps A/B run anywhere (CPU in fp32 is fine); the question needs a GPU and is skipped without
one. The `qwen-eye` command builds the same plans (`qwen-eye attn`, `qwen-eye gradcam`).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from qwen_eye import items, plans

VIDEO_ARGS = {"fps": 2.0, "max_num_frames": 32}        # lmms-eval qwen3_vl backend args


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--out", default="out")
    args = ap.parse_args()
    model = args.model
    if not Path(model).is_dir():
        from huggingface_hub import snapshot_download

        model = snapshot_download(model)
    out = Path(args.out).resolve()
    clip = out / Path(args.video).stem

    # 1. ViT attention: viewer.html, attention.mp4, maps.npz
    plans.run(plans.attn_plan(model=model, videos=[args.video], out_root=out, variant_args=VIDEO_ARGS))
    with np.load(clip / "maps.npz") as z:
        A = z["A"].astype(np.float32)                     # [27 blocks, pairs, heads, h, w], x average
    meta = json.loads((clip / "meta.json").read_text())
    print(f"grid {meta['grid']} (pairs, patch rows, cols); pairs at {meta['pair_labels']} s")
    for L in (2, 16, 26):
        top = A[L].mean(1).max()                          # heads averaged, the strongest patch
        print(f"block {L}: strongest patch gets {top:.1f}x the average attention")

    if not torch.cuda.is_available():
        print("no GPU: skipping the gradient maps")
        return

    # 2. Your own question: answer contrast (correct vs. strongest other letter) + one phrase
    q = items.question(args.video, "What happens in the video?\nA. Something moves.\nB. Nothing moves.\n"
                       "Answer with the option's letter.", correct="A")
    plans.run(plans.gradcam_plan(model=model, items=[q.to_dict()], videos=[], phrases=["a person"],
                                 prompt="Is there {phrase} in the video? Answer yes or no.", out_root=out,
                                 variant_args=VIDEO_ARGS))
    gdir = clip / "gradcam" / q.tag
    gmeta = json.loads((gdir / "meta.json").read_text())
    with np.load(gdir / "gradcam.npz") as z:
        H = z["H"]                                        # [scores, layers, pairs, h, w], signed
    for i, s in enumerate(gmeta["scores"]):
        per_pair = np.maximum(H[i, gmeta["layers"].index(8)], 0).sum((1, 2))
        print(f"{s['name']}: score {s['value']:+.2f}; HiResCAM@8 'for' mass per pair {np.round(per_pair, 2)}")
    print(f"open {gdir / 'viewer.html'}")


if __name__ == "__main__":
    main()
