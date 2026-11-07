"""Shared test models, targets and fake outputs: tiny random Qwen3-VL models in
float64, so every test runs on CPU without downloads."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel


T, H, W = 3, 4, 6          # frame pairs, patch rows, patch cols
N = H * W


class FullProbs:
    """Test reducer: keeps every softmax."""

    def __init__(self):
        self.probs = {}

    def update(self, layer, pair, probs):
        self.probs[(layer, pair)] = probs.clone()


def tiny_tower() -> Qwen3VLVisionModel:
    torch.manual_seed(0)
    cfg = Qwen3VLVisionConfig(depth=3, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=16,
                              num_position_embeddings=64, deepstack_visual_indexes=[1])
    cfg._attn_implementation = "sdpa"
    return Qwen3VLVisionModel._from_config(cfg).eval()


def tiny_full_model():
    """A random Qwen3VLForConditionalGeneration small enough for CPU float64: ViT depth 3
    (DeepStack tap after block 1), 2 LLM layers."""
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLForConditionalGeneration

    torch.manual_seed(0)
    cfg = Qwen3VLConfig(
        text_config=dict(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                         num_attention_heads=2, num_key_value_heads=1, head_dim=16, max_position_embeddings=256,
                         rope_scaling={"rope_type": "default", "mrope_section": [2, 3, 3], "mrope_interleaved": True}),
        vision_config=dict(depth=3, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=32,
                           num_position_embeddings=64, deepstack_visual_indexes=[1]),
        image_token_id=60, video_token_id=61, vision_start_token_id=62, vision_end_token_id=63)
    cfg._attn_implementation = "sdpa"
    model = Qwen3VLForConditionalGeneration._from_config(cfg).double().eval().requires_grad_(False)
    ids = [5, 6]
    for _ in range(T):                                  # "<t seconds>" stand-in, then the pair's tokens
        ids += [7, 62] + [61] * (N // 4) + [63]
    ids += [8, 9, 10, 11]
    batch = dict(input_ids=torch.tensor([ids]), attention_mask=torch.ones(1, len(ids), dtype=torch.long),
                 pixel_values_videos=torch.randn(T * N, 3 * 2 * 16 * 16, dtype=torch.float64),
                 video_grid_thw=torch.tensor([[T, H, W]]))
    return model, batch


class LogitGap:
    """Test target: logit 3 - logit 4 at the last position (LetterContrast's shape)."""
    keep = 1

    def score(self, logits):
        return logits[0, -1, 3] - logits[0, -1, 4]


TAP_LAYERS = [1, 2]                                     # lowest = the DeepStack index


class FakePre:
    """Stands in for tower.EvalPreprocessor: every prompt gives the tiny model's batch."""

    def __init__(self, batch):
        self.batch = batch


class GapTarget(LogitGap):
    name = "gap"
    maps = [("up", 1), ("down", -1)]

    def build(self, pre):
        from transformers import BatchFeature

        return BatchFeature(pre.batch)

    def report(self, logits):
        return {}


class BigGap(GapTarget):
    """The gap x 1e4. The tiny model's attention gradients are small (mean_h ReLU(A * dA) rows sum
    to ~3e-5), so Chefer's M R term would hide under float32 rounding; scaled, rows sum to ~0.3."""
    K = 1e4

    def score(self, logits):
        return self.K * super().score(logits)


class ExplicitAttention:
    """Test attention for the ViT: the softmax as its own tensor in the graph (retain_grad), so
    plain autograd gives d(score)/d(attention) to hold dO @ V^T against."""

    def __init__(self, tower):
        self.layer_of = {id(b.attn): i for i, b in enumerate(tower.blocks)}
        self.calls = [0] * len(tower.blocks)
        self.probs, self.v = {}, {}

    def __call__(self, module, query, key, value, attention_mask, **kwargs):
        layer = self.layer_of[id(module)]
        pair = self.calls[layer]
        self.calls[layer] += 1
        probs = torch.softmax(query @ key.transpose(-1, -2) * kwargs["scaling"], dim=-1)    # [1, heads, n, n]
        probs.retain_grad()
        self.probs[(layer, pair)], self.v[(layer, pair)] = probs, value.detach()
        return (probs @ value).transpose(1, 2).contiguous(), None


def mergers(tower):
    return [*zip(tower.deepstack_visual_indexes, tower.deepstack_merger_list), (len(tower.blocks) - 1, tower.merger)]


def rel_close(got, want) -> bool:
    """got from relevance.py's fp32 state, want float64: relative to want's largest entry."""
    want = want.numpy() if isinstance(want, torch.Tensor) else want
    got = got.numpy() if isinstance(got, torch.Tensor) else got
    return np.abs(want).max() > 0 and np.abs(got - want).max() <= 1e-4 * np.abs(want).max()


def brute_force(probs, grads, w):
    """One pair, float64, as the papers write it: Chefer's R = I, R += mean_h ReLU(+-A * dA) R;
    Abnar's product of (I + mean_h A) / 2; readouts at the route blocks (relevance.py). probs,
    grads: per block [heads, n, n]; w: route block -> signed weight [n]. Returns per side [2, n]:
    D, norm, W, C, and D per route."""
    n = probs[0].shape[-1]
    eye = torch.eye(n, dtype=torch.float64)
    R, roll = [eye.clone(), eye.clone()], eye.clone()
    out = {k: torch.zeros(2, n, dtype=torch.float64) for k in ("D", "norm", "W", "C")}
    routes = {}
    for layer, (A, dA) in enumerate(zip(probs, grads)):
        A, cam = A.double(), (A * dA).double()
        R = [R[0] + cam.clamp(min=0).mean(0) @ R[0], R[1] + (-cam).clamp(min=0).mean(0) @ R[1]]
        roll = 0.5 * (eye + A.mean(0)) @ roll
        if layer not in w:
            continue
        routes[layer] = torch.zeros(2, n, dtype=torch.float64)
        for side, ws in enumerate((w[layer].double().clamp(min=0), (-w[layer].double()).clamp(min=0))):
            Q = R[side] - eye
            rows = Q.sum(1, keepdim=True)
            routes[layer][side] = ws @ Q
            out["D"][side] += ws @ Q
            out["norm"][side] += ws @ torch.where(rows > 0, Q / rows, torch.zeros_like(Q))
            out["W"][side] += ws
            out["C"][side] += ws @ (roll - 0.5 ** (layer + 1) * eye)
    return out, routes


def fake_clip(pairs=2, layers_b=(0, 2)):
    rng = np.random.default_rng(1)
    maps = {"A": rng.gamma(1.0, 1.0, (3, pairs, 2, H, W)).astype(np.float16),
            "B": rng.gamma(1.0, 1.0, (len(layers_b), pairs, N // 4, H, W)).astype(np.float16),
            "mean_dist": np.ones((3, pairs, 2), np.float32),
            "frames": rng.integers(0, 256, (2 * pairs, H * 16, W * 16, 3), dtype=np.uint8)}
    meta = {"video": "/x/clip.mp4", "model": "/m/qwen3", "grid": [pairs, H, W], "patch": 16, "tpatch": 2,
            "merge": 2, "frame_times": [0.0, 0.5, 1.0], "pair_labels": ["0.3", "1.0"], "fps": 2.0,
            "layers_B": list(layers_b), "heads": 2, "device": "cpu", "dtype": "float32",
            "checks": {"recon_gap": 1e-6}}
    stats = {"uniform_mean_dist": 2.0, "uniform_top1_share": 1 / N,
             "layers": [{"layer": L, "mean_dist": 1.0, "top1_share": 0.1, "max_x_average": 3.0} for L in range(3)]}
    return maps, meta, stats


def fake_gradcam(relevance=False):
    """A gradcam dir's arrays + meta: E, H as pre-0.1.0 versions saved them (no meta["maps"]), or
    with D [scores, sides, parts, pairs, h, w] and its layouts."""
    rng = np.random.default_rng(2)
    E, Hm = (rng.normal(size=(2, 2, 2, H, W)).astype(np.float32) for _ in range(2))   # scores, layers, pairs
    scores = [{"name": "answer", "maps": [{"label": "for correct (B)", "sign": 1}, {"label": "for wrong (A)", "sign": -1}],
               "value": 1.0, "plain_value": 1.0, "report": {}, "grad_norms": {"0": [1.0]}},
              {"name": "phrase: a car", "maps": [{"label": '"a car": for yes', "sign": 1},
                                                 {"label": '"a car": for no', "sign": -1}],
               "value": -2.0, "plain_value": -2.0, "report": {}, "grad_norms": {}}]
    gm = {"item": None, "phrase_prompt": "Is there {phrase} in the video? Answer yes or no.", "scores": scores,
          "layers": [0, 2], "checks_failed": []}
    if not relevance:
        return {"E": E, "H": Hm}, gm
    D = np.abs(rng.normal(size=(2, 2, len(D_PARTS), 2, H, W))).astype(np.float32)
    gm["maps"] = {k: {"name": n, "sides": "signed", "parts": ["0", "2"]} for k, n in (("H", "HiResCAM"), ("E", "Grad-CAM"))}
    gm["maps"]["D"] = {"name": "Chefer relevance", "sides": 2, "parts": D_PARTS}
    return {"E": E, "H": Hm, "D": D}, gm


D_PARTS = ["all", "r1", "norm"]


def deletion_settings(**kw):
    return {"fractions": [0.25, 1.0], "random": 2, "fill": "grey"} | kw


SIGNED = {"H": {"name": "HiResCAM", "sides": "signed", "parts": ["0", "1"]}}


def fake_gradcam_dir(root, prompt="Is there {phrase} in the video? Answer yes or no.", layers=(2, 8)):
    import json

    from qwen_eye.items import Item

    d = root / "clip" / "gradcam" / "type_5949"
    d.mkdir(parents=True)
    item = Item(key="type:5949", benchmark="vitatecs", video="v.mp4", prompt="q", correct="B", wrong="A",
                eval_answer="B", eval_letter="B", options={"A": "a", "B": "b"})
    (d / "meta.json").write_text(json.dumps({
        "model": "/m/qwen3", "video": "v.mp4", "clip_dir": str(root / "clip"), "layers": list(layers),
        "phrase_prompt": prompt, "item": item.to_dict(), "checks_failed": [],
        "scores": [{"name": "answer"}, {"name": "phrase: a dog"}]}))
    np.savez(d / "gradcam.npz", E=np.zeros((2, len(layers), T, H, W)), H=np.zeros((2, len(layers), T, H, W)))
    return d


def snapshot(needs: str = "preprocessor_config.json") -> Path:
    """The Qwen3-VL snapshot dir in $QWEN_EYE_MODEL (processor and tokenizer files only are read);
    the test is skipped without it."""
    name = os.environ.get("QWEN_EYE_MODEL")
    path = Path(name).expanduser() if name else None
    if path is None or not (path / needs).is_file():
        pytest.skip(f"set QWEN_EYE_MODEL to a local Qwen3-VL snapshot dir (with {needs}) to run this test")
    return path

