"""Gradient maps for multiple-choice questions and phrases: the full model, one forward +
backward per score (targets.py), features and gradients at chosen ViT blocks (grads.py).
  E, H     Grad-CAM and HiResCAM at each tapped block (reducers.GradCAM, reducers.HiResCAM)
  D, C, W  Chefer relevance, rollout and readout-only back to the input patches
           (relevance.py): the same backward taken down to block 0, plus one no-grad forward
           that captures every block's attention
E and H come with every backward; `maps` says whether D, C, W are computed too.

A job (plan kind "gradcam", from `qwen-eye gradcam`) loads the model once and goes clip by
clip. Each clip first gets its A/B maps if it has none (run.analyse, fp32: they are the
viewer's other maps and its frames). Then:
  every item    -> <out_root>/<clip>/gradcam/<item tag>/       answer contrast + every phrase
  every --video -> <out_root>/<clip>/gradcam/phrases/           every phrase
Phrase maps are computed once per clip and shared by its items.

Per dir: gradcam.npz  one array per map, float32, grid order, layouts in maps.py
         meta.json    item, layers, maps (name, sides, parts), per score: label, maps (its
                      sides), value, report, checks
then viewer.html + gradcam.mp4 (render.render_gradcam). A rerun into an existing dir with
the same scores adds the maps it does not have yet. A saved map is replaced only if the plan
names it (`gradcam --maps`); then every deletion*.json that ranked it is renamed
*.stale.json. Saved maps the rerun computes along the way (E, H come with every backward) are
kept bit for bit and only compared: meta.json "reproduced" (the bf16 backward is not
deterministic: ~1e-2 relative, correlation ~0.9998).

Checks per score (a failure stops the job; all are recorded):
  taps_change_nothing  |score with taps + checkpointing - plain eval-mode no-grad score|
  grads_reach_all      gradient finite at every tapped layer and nonzero in every pair
  eval_answer (items)  our greedy first token is the eval's recorded letter (a flip only
                       passes as a bf16 near-tie: |logit ours - logit eval's| < NEAR_TIE);
                       an item with no recorded answer (your own question) records its
                       first_token instead
  with D, C, W:
  attn_grads_reach_all   dO finite and nonzero for every block and pair
  routes_reach_all       every route's readout weight finite and nonzero in every pair
  relevance_recon_gap    the captured softmax @ V vs sdpa's output (run.py's tolerance)
  relevance_pass_score   |score of the capture pass - plain score|
Recorded, not enforced: phrases' first_token (the greedy answer, expected yes or no in either
case; if not, the yes/no contrast is about an answer the model would not give); for D its
mixing share (D's mass / (D + W)) and mass per pair.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from qwen_eye import layout
from qwen_eye.capture import AttentionCapture, attached
from qwen_eye.fullmodel import grad_mode, load_model
from qwen_eye.grads import EmbedLeaf, FeatureTaps, OutputGradTaps, RouteTaps, backprop
from qwen_eye.items import Item
from qwen_eye.maps import NAMES, layouts
from qwen_eye.reducers import GradCAM, HiResCAM
from qwen_eye.relevance import ViTRelevance
from qwen_eye.run import TOLERANCE, resolve_device
from qwen_eye.targets import DEFAULT_PROMPT, LetterContrast, Phrase
from qwen_eye.tower import EvalPreprocessor

TAPS_TOLERANCE = 0.25      # nats/logits, bf16 forward with vs without grad + checkpointing
NEAR_TIE = 0.5
RELEVANCE = ("D", "C", "W")
OWNED = {"grad_norms": ("E", "H"), "relevance": RELEVANCE, "attn_grad_norms": RELEVANCE}   # record field: its maps


def explain(model, pre: EvalPreprocessor, target, layers, device, maps=tuple(NAMES)) -> dict:
    """One score: its signed maps E (Grad-CAM) and H (HiResCAM) [layers, pairs, n], and with any
    of D, C, W in `maps` all three [sides, parts, pairs, n] (token order); "layout" per map
    (maps.py) and the score's record."""
    t0 = time.time()
    batch = target.build(pre).to(device)
    t, h, w = (int(x) for x in batch["video_grid_thw"][0])
    n = h * w
    with torch.no_grad():
        plain = model(**batch, use_cache=False, logits_to_keep=target.keep).logits
    if hasattr(target, "resolve"):
        target.resolve(plain)
    plain_score = float(target.score(plain))
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    relevance = bool(set(maps) & set(RELEVANCE))
    visual = model.model.visual
    taps = FeatureTaps(visual, layers, cut=not relevance)
    extra = (EmbedLeaf(visual), OutputGradTaps(visual), RouteTaps(visual)) if relevance else ()
    try:
        with grad_mode(model, -1 if relevance else taps.lowest):
            score, logits = backprop(model, batch, target, taps)
        tapped = taps.result()
        if relevance:
            dO, routes = extra[1].result(), extra[2].result()
    finally:
        taps.remove()
        for x in extra:
            x.remove()

    reds = [GradCAM(), HiResCAM()]
    grad_norms = {}
    for L, (feats, grads) in tapped.items():
        for p in range(t):
            for red in reds:
                red.update(L, p, feats[p * n:(p + 1) * n], grads[p * n:(p + 1) * n])
        grad_norms[L] = grads.float().view(t, n, -1).norm(dim=(1, 2)).tolist()
    out = reds[0].result() | reds[1].result()
    lay = {k: {"name": NAMES[k], "sides": "signed", "parts": [str(L) for L in sorted(layers)]} for k in ("H", "E")}

    checks = {"taps_change_nothing": abs(score - plain_score)}
    failed = []
    if checks["taps_change_nothing"] > TAPS_TOLERANCE:
        failed.append(f"taps_change_nothing {checks['taps_change_nothing']:.3f} > {TAPS_TOLERANCE}")
    bad = [L for L, g in grad_norms.items() if not all(np.isfinite(g)) or min(g) == 0]
    checks["grads_reach_all"] = not bad
    if bad:
        failed.append(f"gradient zero or not finite at layers {bad}")
    extra_record = {}
    if relevance:
        rel, rchecks, rfailed, attn_norms = _relevance(model, batch, target, dO, routes, (t, h, w), plain_score)
        del dO, routes
        out |= {k: rel[k] for k in RELEVANCE}
        lay |= {k: {"name": NAMES[k], "sides": 2, "parts": rel["parts"][k]} for k in RELEVANCE}
        checks |= rchecks
        failed += rfailed
        extra_record = {"relevance": rel["stats"], "attn_grad_norms": {str(L): v for L, v in attn_norms.items()}}
    report = target.report(logits)
    if isinstance(target, LetterContrast) and target.item.eval_answer is None:
        checks["first_token"] = report["argmax_token"].strip()
    elif isinstance(target, LetterContrast):
        got, item, lg = report["argmax_token"].strip(), target.item, report["logits"]
        want = item.eval_letter
        margin = lg[want] - lg[got] if want in lg and got in lg else None
        checks["eval_answer"] = {"ours": got, "eval": want, "eval_reply": item.eval_answer, "logit_margin": margin}
        same = got == want if want else bool(got) and item.eval_answer.strip().startswith(got)
        if not same and (margin is None or abs(margin) >= NEAR_TIE):
            failed.append(f"eval answered {item.eval_answer!r}, our forward {got!r}")
    if isinstance(target, Phrase):
        checks["first_token"] = report["argmax_token"].strip()
    record = {
        "name": target.name, "maps": [{"label": lab, "sign": s} for lab, s in target.maps],
        "value": score, "plain_value": plain_score, "report": report, "checks": checks, "failed": failed,
        "grad_norms": {str(L): v for L, v in grad_norms.items()}, **extra_record,
        "seconds": round(time.time() - t0, 1),
        "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2) if device.startswith("cuda") else None,
        "seq_len": int(batch["input_ids"].shape[1]),
    }
    mix = extra_record.get("relevance", {}).get("mixing_share")
    print(f"  {target.name}: score {score:+.3f} (plain {plain_score:+.3f}); "
          + ", ".join(f"p({k}) {v:.3f}" for k, v in report.get("p", {}).items())
          + f", first token {report.get('argmax_token')!r}"
          + (f"; D/(D+W) {mix[0]:.2f}/{mix[1]:.2f}" if mix else "")
          + f"; {record['seconds']}s, peak {record['peak_gpu_gb']} GB" + (f"; FAILED {failed}" if failed else ""),
          flush=True)
    return out | {"layout": lay, "grid": (t, h, w), "record": record}


def _relevance(model, batch, target, dO, routes, grid, plain_score):
    """Maps D, C, W of one score from pass 1's taps: one no-grad forward with every block's
    attention captured (relevance.ViTRelevance). Returns them, checks, failures, dO norms."""
    t, h, w = grid
    n = h * w
    visual = model.model.visual
    checks, failed = {}, []
    norms = {L: g.float().view(t, n, -1).norm(dim=(1, 2)).tolist() for L, g in dO.items()}
    bad = [L for L, g in norms.items() if not all(np.isfinite(g)) or min(g) == 0]
    checks["attn_grads_reach_all"] = not bad
    if bad:
        failed.append(f"attention-output gradient zero or not finite at blocks {bad}")
    unit = visual.spatial_merge_unit                # patches per LLM token
    # each LLM token's output . gradient, shared equally by its 2x2 block's patches
    weights = {L: (x.float() * g.float()).sum(-1).repeat_interleave(unit) / unit for L, (x, g) in routes.items()}
    per_pair = {L: w.view(t, n).abs().sum(1) for L, w in weights.items()}
    bad = [L for L, w in per_pair.items() if not torch.isfinite(w).all() or (w == 0).any()]
    checks["routes_reach_all"] = not bad
    if bad:
        failed.append(f"readout weight zero or not finite in a pair of the routes after blocks {bad}")
    red = ViTRelevance(dO, weights, t, n)
    cap = AttentionCapture(visual, [red])
    with attached(visual, cap), torch.no_grad():
        again = float(target.score(model(**batch, use_cache=False, logits_to_keep=target.keep).logits))
    dtype = next(visual.parameters()).dtype
    tol = TOLERANCE[torch.bfloat16 if dtype in (torch.bfloat16, torch.float16) else torch.float32]["recon_gap"]
    checks["relevance_recon_gap"] = cap.recon_gap
    if cap.recon_gap > tol:
        failed.append(f"relevance_recon_gap {cap.recon_gap:.2e} > {tol}")
    checks["relevance_pass_score"] = abs(again - plain_score)
    if checks["relevance_pass_score"] > TAPS_TOLERANCE:
        failed.append(f"capture pass score {again:+.3f} != plain {plain_score:+.3f}")
    if cap.calls != [t] * len(visual.blocks):
        failed.append(f"capture saw {cap.calls} calls per block, expected {t} each")
    rel = red.result()
    if not all(np.isfinite(rel[k]).all() for k in RELEVANCE):
        failed.append("relevance maps not finite")
    return rel, checks, failed, norms


def compare(new: np.ndarray, saved: np.ndarray) -> dict:
    """How far a recomputed map is from the saved one: max |diff| / max |saved|, and the lowest
    correlation over its (score, side, part) maps."""
    if new.shape != saved.shape:
        return {"max_rel_diff": float("inf"), "min_corr": None, "shape": list(saved.shape)}
    a, b = new.reshape(-1, int(np.prod(new.shape[-3:]))), saved.reshape(-1, int(np.prod(saved.shape[-3:])))
    corr = [float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else float(np.array_equal(x, y))
            for x, y in zip(a, b)]
    return {"max_rel_diff": float(np.abs(new - saved).max() / max(float(np.abs(saved).max()), 1e-30)),
            "min_corr": min(corr)}


def save(out_dir: Path, results: list[dict], *, item: Item | None, video: str, model_dir: str, layers,
         clip_dir: Path, prompt: str, maps=tuple(NAMES), replace=()) -> Path:
    """Write (or add to) one dir's gradcam.npz + meta.json: E, H and the asked `maps` that the
    results have. An existing dir must hold the same scores, grid and layers; of its saved maps
    only those in `replace` are rewritten."""
    from qwen_eye.deletion import used_maps

    out_dir.mkdir(parents=True, exist_ok=True)
    t, h, w = results[0]["grid"]
    order = layout.block_order(h, w)
    keys = [k for k in NAMES if k in results[0] and (k in maps or k in ("E", "H"))]
    new = {k: np.stack([layout.to_grid(torch.from_numpy(r[k]), order, h, w).numpy() for r in results]).astype(np.float32)
           for k in keys}
    records = [r["record"] for r in results]
    arrays, saved_lay, reproduced, stale, old = {}, {}, {}, [], None
    if (out_dir / "meta.json").exists() and (out_dir / "gradcam.npz").exists():
        old = json.loads((out_dir / "meta.json").read_text())
        problem = _mismatch(old, [r["name"] for r in records], sorted(layers), prompt, [t, h, w])
        if problem:
            raise SystemExit(f"{out_dir}: {problem}; move it away to start over")
        with np.load(out_dir / "gradcam.npz") as z:
            arrays = {k: z[k] for k in z.files}
        saved_lay = layouts(old, arrays)
    written = set()
    for k, x in new.items():
        if k in arrays and k not in replace:
            reproduced[k] = compare(x, arrays[k])
            continue
        if k in arrays:
            stale.append(k)
        arrays[k] = x
        saved_lay[k] = results[0]["layout"][k]
        written.add(k)
    if old:                 # a kept map keeps the record fields of the pass that made it
        records = [o | {f: v for f, v in r.items() if f not in OWNED or set(OWNED[f]) & written}
                   | {"checks": o.get("checks", {}) | r["checks"]} for o, r in zip(old["scores"], records)]
    np.savez_compressed(out_dir / "gradcam.npz", **arrays)
    meta = {
        "video": video, "model": model_dir, "created": datetime.now().isoformat(timespec="seconds"),
        "clip_dir": str(clip_dir), "grid": [t, h, w], "layers": sorted(layers), "phrase_prompt": prompt,
        "item": item.to_dict() if item else None, "maps": {k: saved_lay[k] for k in NAMES if k in saved_lay},
        "reproduced": reproduced, "scores": records,
        "checks_failed": [f for r in results for f in r["record"]["failed"]],
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))
    for f in sorted(out_dir.glob("deletion*.json")):
        if stale and not f.name.endswith(".stale.json") and used_maps(json.loads(f.read_text())) & set(stale):
            f.rename(f.with_name(f.name[:-len(".json")] + ".stale.json"))
            print(f"  {f.name}: ranked {sorted(stale)}, which this run changed -> renamed .stale.json", flush=True)
    return out_dir


def _mismatch(meta: dict, names: list[str], layers: list[int], prompt: str, grid=None) -> str | None:
    """Why a saved dir cannot take this run's maps, or None."""
    if [s["name"] for s in meta["scores"]] != names:
        return f"it holds scores {[s['name'] for s in meta['scores']]}, this run {names}"
    if meta["layers"] != layers:
        return f"it holds layers {meta['layers']}, this run {layers}"
    if meta["phrase_prompt"] != prompt and any(n.startswith("phrase: ") for n in names):
        return f"its phrase prompt is {meta['phrase_prompt']!r}, this run's {prompt!r}"
    if grid is not None and meta["grid"] != grid:
        return f"its grid is {meta['grid']}, this run's {grid}"
    return None


def run_plan(plan: dict) -> int:
    """Every item and clip of a gradcam plan (`qwen-eye gradcam`, `python -m qwen_eye run`)."""
    from qwen_eye.run import analyse

    device = resolve_device(plan["device"])
    if not device.startswith("cuda"):
        raise SystemExit("gradcam runs the full model with a backward pass: it needs a GPU")
    layers, prompt, out_root = sorted(plan["layers"]), plan.get("prompt", DEFAULT_PROMPT), Path(plan["out_root"])
    asked = list(plan.get("maps") or [])               # named maps are computed and replace saved ones
    keys = tuple(asked or NAMES)
    jobs = [(it["video"], Item(**it)) for it in plan["items"]] + [(v, None) for v in plan["videos"]]
    phrase_names = [f"phrase: {p}" for p in plan["phrases"]]
    problems = []                                       # before the model loads: dirs this run cannot add to
    for video, item in jobs:
        d = out_root / Path(video).stem / "gradcam" / (item.tag if item else "phrases")
        if (d / "meta.json").exists():
            p = _mismatch(json.loads((d / "meta.json").read_text()), (["answer"] if item else []) + phrase_names,
                          layers, prompt)
            if p:
                problems.append(f"{d}: {p}")
    if problems:
        raise SystemExit("cannot add maps to existing dirs (move them away to start over):\n  " + "\n  ".join(problems))
    t0 = time.time()
    model = load_model(plan["model"], device)
    print(f"model loaded in {time.time() - t0:.0f}s: {plan['model']}; maps {list(keys)}"
          + (f", replacing saved {asked}" if asked else ", adding only missing ones"), flush=True)
    failed = []
    for video in dict.fromkeys(v for v, _ in jobs):
        clip_dir = out_root / Path(video).stem
        done = (clip_dir / "meta.json").exists() and not json.loads((clip_dir / "meta.json").read_text())["checks_failed"]
        if not (done and (clip_dir / "maps.npz").exists()):
            analyse(video, plan["model"], clip_dir, variant=plan["variant_args"], layers=layers, device=device,
                    dtype=torch.float32)
        pre = EvalPreprocessor(plan["model"], video, plan["variant_args"])
        print(f"{video}:", flush=True)
        phrases = [explain(model, pre, Phrase(p, prompt), layers, device, keys) for p in plan["phrases"]]
        for v, item in jobs:
            if v != video:
                continue
            results = ([explain(model, pre, LetterContrast(item), layers, device, keys)] if item else []) + phrases
            if not results:
                continue
            out_dir = save(clip_dir / "gradcam" / (item.tag if item else "phrases"), results, item=item, video=video,
                           model_dir=plan["model"], layers=layers, clip_dir=clip_dir, prompt=prompt, maps=keys,
                           replace=asked)
            failed += [f"{out_dir.name}: {f}" for r in results for f in r["record"]["failed"]]
            from qwen_eye.render import render_gradcam
            render_gradcam(out_dir)
    if failed:
        raise SystemExit(f"checks failed: {failed}")
    return 0
