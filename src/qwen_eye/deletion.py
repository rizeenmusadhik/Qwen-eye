"""Deletion test of the gradient maps: does blanking what a map ranks highest move its score the most?

For a saved gradcam dir (gradcam.py: gradcam.npz + meta.json), score by score: the clip's
2x2 merge blocks (one LLM token each, every pair's blocks together) are ranked by a map's
block sums. The first k are blanked in the model's input pixels (all four patches, both frames
of the pair; mid grey, i.e. FILL = 0 after Qwen3-VL's mean = std = 0.5 normalisation), and the
full model runs forward again on the same prompt. If a map points at what the score uses,
blanking along it moves the score more than blanking random blocks:

  for_first      the "for" side's biggest blocks first: the score should fall fastest
  against_first  the "against" side's biggest first: the score should rise
  random         seeded random orders, the same ones for every map: the baseline

A map is named by a spec "<key>@<part>" (maps.py): H@8 (HiResCAM at block 8), D@all, D@r16,
D@norm, C@all, W@all. Signed maps (E, H) rank one array both ways (for = most positive first,
against = most negative first); two-sided maps (D, C, W) rank each side's own array. Fractions
of the clip's blocks are the curve's points. Blanking pixels changes every feature after it, so
this measures the model's real response; HiResCAM summed over a block is only a first-order
guess, and only for zeroing that layer's features.

deletion.json, next to gradcam.npz:
  settings  fractions, random, fill (a rerun with other settings refuses)
  maps      every spec with curves in the file
  blocks    the clip's block count; ks = blocks blanked at each fraction
  scores    per score name: clean (recomputed) and recorded score, curves
            [{map, method, layer, order} or {order: random, seed}, score per fraction],
            summary (per map: both orders' areas, random's mean and sd; see `area`), seconds,
            failed
It is written after every score. A rerun skips what is done: finished scores, and in a score
the maps and random orders it already has, so maps can be added to a file later and a clip
that does not fit one job's time limit goes on in the next job (a failed score is redone).
Files from before 0.1.0 (settings with methods + layers, fill 0.0) are read as maps = methods
x layers, fill grey; the first time one is rewritten, the original is kept as deletion.v1.json.

Check: the clean forward reproduces the score gradcam.py recorded (plain_value) within
CLEAN_TOLERANCE. That covers the same prompt, the same pixels and the same wrong letter.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from qwen_eye.items import Item
from qwen_eye.maps import layouts, side_pair
from qwen_eye.targets import LetterContrast, Phrase

FILL = 0.0                 # normalised pixel value; (x - 0.5) / 0.5 = 0 is mid grey
FILLS = ("grey",)
CLEAN_TOLERANCE = 0.25     # = gradcam.TAPS_TOLERANCE: bf16, same input
DEFAULT_FRACTIONS = (0.01, 0.02, 0.05, 0.1, 0.2)
DEFAULT_RANDOM = 3


def result_path(gdir: Path, fill: str) -> Path:
    return Path(gdir) / ("deletion.json" if fill == "grey" else f"deletion_{fill}.json")


def parse_spec(spec: str) -> tuple[str, str]:
    key, at, part = spec.partition("@")
    if not at or not key or not part:
        raise ValueError(f"map spec {spec!r} is not <key>@<part>, e.g. H@8 or D@all")
    return key, part


def upgrade(result: dict) -> dict:
    """A first-build file (settings methods + layers, fill 0.0, curves {method, layer}) in the
    current form; a current one unchanged."""
    s = result["settings"]
    if "methods" not in s:
        return result
    out = {**result, "settings": {"fractions": s["fractions"], "random": s["random"], "fill": "grey"},
           "maps": [f"{m}@{L}" for m in s["methods"] for L in s["layers"]], "scores": {}}
    for name, res in result["scores"].items():
        curves = [c if c["order"] == "random" else
                  {"map": f"{c['method']}@{c['layer']}", **c, "layer": str(c["layer"])} for c in res["curves"]]
        out["scores"][name] = {**res, "curves": curves, "summary": summary({"curves": curves})}
    return out


def used_maps(result: dict) -> set[str]:
    """The map keys (E, H, D, ...) a deletion file ranked."""
    if "methods" in result["settings"]:
        return set(result["settings"]["methods"])
    return {parse_spec(s)[0] for s in result.get("maps", [])}


def load_result(path: Path) -> dict | None:
    return upgrade(json.loads(Path(path).read_text())) if Path(path).exists() else None


def pending(previous: dict | None, specs, random: int) -> tuple[list[str], int]:
    """What one score still needs: the specs without curves, and how many random orders."""
    if not previous or previous["failed"]:
        return list(specs), random
    done = {c.get("map") for c in previous["curves"]}
    have = sum(c["order"] == "random" for c in previous["curves"])
    return [s for s in specs if s not in done], max(0, random - have)


def block_sums(x: np.ndarray, merge: int = 2) -> np.ndarray:
    """Grid-order maps [..., pairs, h, w] -> [..., pairs * blocks], the sum over each merge
    block. Blocks are row-major within a pair (layout.block_of), pairs in order, so global
    block g = pair * blocks + block is the g-th group of merge^2 rows of pixel_values_videos."""
    *lead, t, h, w = x.shape
    s = x.reshape(*lead, t, h // merge, merge, w // merge, merge).sum((-3, -1))
    return s.reshape(*lead, t * (h // merge) * (w // merge))


def blank(pixels: torch.Tensor, blocks, merge: int = 2, fill: float = FILL) -> torch.Tensor:
    """pixel_values_videos [t*h*w, C*T*P*P] (token order) with the given global blocks set to
    fill: all merge^2 patches, both frames of the pair."""
    out = pixels.clone()
    out.view(-1, merge * merge, pixels.shape[-1])[torch.as_tensor(blocks, dtype=torch.long)] = fill
    return out


def orders(for_sums: np.ndarray, against_sums: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """A map's block sums [blocks] -> its two deletion orders. A signed map gives one array:
    against = its negation, so against_first is the most negative block first."""
    against = -for_sums if against_sums is None else against_sums
    return {"for_first": np.argsort(-for_sums, kind="stable"), "against_first": np.argsort(-against, kind="stable")}


def random_orders(n_blocks: int, count: int, seed: int = 0) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [rng.permutation(n_blocks) for _ in range(count)]


def counts(n_blocks: int, fractions) -> list[int]:
    return [max(1, round(f * n_blocks)) for f in fractions]


def area(fractions, deltas) -> float:
    """The mean change of the score over fractions [0, max] (trapezoid, from 0 change at 0):
    the area under the deletion curve, in logits."""
    f, d = np.concatenate([[0.0], fractions]), np.concatenate([[0.0], deltas])
    return float(((d[1:] + d[:-1]) / 2 * np.diff(f)).sum() / f[-1])


def target_for(record: dict, meta: dict):
    """The score gradcam.py recorded, rebuilt: the same letters, or the same phrase question."""
    if record["name"] == "answer":
        item = Item(**meta["item"])
        item.wrong = record["report"].get("wrong") or item.wrong
        return LetterContrast(item)
    if record["name"].startswith("phrase: "):
        return Phrase(record["name"][len("phrase: "):], meta["phrase_prompt"])
    raise ValueError(f"unknown score {record['name']!r}")


def scorer(model, batch, target):
    """pixels -> the target's score on the full model, the rest of the batch fixed."""
    def run(pixels: torch.Tensor) -> float:
        with torch.inference_mode():
            logits = model(**{**batch, "pixel_values_videos": pixels}, use_cache=False,
                           logits_to_keep=target.keep).logits
        return float(target.score(logits))
    return run


def delete(run, pixels: torch.Tensor, order: np.ndarray, ks: list[int], merge: int = 2) -> list[float]:
    return [run(blank(pixels, order[:k], merge)) for k in ks]


def curves_for(model, pre, target, ranked: dict, settings: dict, device: str, recorded: float, merge: int = 2,
               previous: dict | None = None) -> dict:
    """One score. ranked: spec -> its (for, against) block sums [blocks]; recorded: gradcam.py's
    plain_value. previous: this score's earlier result, whose curves are kept (a failed one is
    redone). No curves if the clean score does not reproduce the recorded one (the maps would
    not belong to this input)."""
    t0 = time.time()
    batch = target.build(pre).to(device)
    pixels = batch["pixel_values_videos"]
    run = scorer(model, batch, target)
    n_blocks = pixels.shape[0] // merge ** 2
    ks = counts(n_blocks, settings["fractions"])
    clean = run(pixels)
    res = {"clean": clean, "recorded": recorded, "blocks": n_blocks, "ks": ks, "curves": [], "failed": []}
    kept = 0.0
    if abs(clean - recorded) > CLEAN_TOLERANCE:
        res["failed"].append(f"clean score {clean:+.3f} != recorded {recorded:+.3f}")
    else:
        todo, n_random = pending(previous, ranked, settings["random"])
        if previous and not previous["failed"]:
            res["curves"], kept = list(previous["curves"]), previous.get("seconds", 0.0)
        for spec in todo:
            for_sums, against_sums = ranked[spec]
            if for_sums.shape != (n_blocks,):
                raise ValueError(f"map {spec} has {for_sums.shape[0]} blocks, the input {n_blocks}")
            key, part = parse_spec(spec)
            for name, o in orders(for_sums, against_sums).items():
                res["curves"].append({"map": spec, "method": key, "layer": part, "order": name,
                                      "score": delete(run, pixels, o, ks, merge)})
        seeds = random_orders(n_blocks, settings["random"])
        for seed in range(settings["random"] - n_random, settings["random"]):
            res["curves"].append({"order": "random", "seed": seed, "score": delete(run, pixels, seeds[seed], ks, merge)})
    for c in res["curves"]:
        c["area"] = area(settings["fractions"], np.array(c["score"]) - clean)
    res["summary"] = summary(res)
    res["seconds"] = round(time.time() - t0 + kept, 1)              # kept: the earlier runs' share
    return res


def summary(res: dict) -> list[dict]:
    """Per map: area of both orders, and random's mean and sd (logits)."""
    rand = np.array([c["area"] for c in res["curves"] if c["order"] == "random"])
    rows = {}
    for c in res["curves"]:
        if c["order"] != "random":
            row = rows.setdefault(c["map"], {"map": c["map"], "method": c["method"], "layer": c["layer"]})
            row[c["order"]] = c["area"]
    return [r | {"random": float(rand.mean()) if rand.size else None,
                 "random_sd": float(rand.std(ddof=1)) if rand.size > 1 else None} for r in rows.values()]


def _print(name: str, res: dict) -> None:
    print(f"  {name}: clean {res['clean']:+.3f} (recorded {res['recorded']:+.3f}); {res['blocks']} blocks, "
          f"k {res['ks']}; {res['seconds']}s" + (f"; FAILED {res['failed']}" if res["failed"] else ""), flush=True)
    for r in res["summary"]:
        rnd = "" if r["random"] is None else f"   random {r['random']:+.3f}" + (
            "" if r["random_sd"] is None else f" ± {r['random_sd']:.3f}")
        print(f"    {r['map']:<8} for_first {r['for_first']:+.3f}   against_first {r['against_first']:+.3f}{rnd}",
              flush=True)


def ranked_maps(arrays: dict, lay: dict, score: int, specs, merge: int = 2) -> dict:
    """spec -> (for, against) block sums of one score's map."""
    out = {}
    for spec in specs:
        key, part = parse_spec(spec)
        f, a = side_pair(arrays[key][score], lay[key], part)
        out[spec] = (block_sums(f, merge), block_sums(a, merge))
    return out


def run_dir(model, gdir: Path, settings: dict, specs, device: str, only=(), pre=None) -> list[str]:
    """Every (or every `only`) score of one gradcam dir, for every spec; returns the failed checks."""
    from qwen_eye.tower import EvalPreprocessor

    gdir = Path(gdir)
    meta = json.loads((gdir / "meta.json").read_text())
    clip = json.loads((Path(meta["clip_dir"]) / "meta.json").read_text())
    with np.load(gdir / "gradcam.npz") as z:
        lay = layouts(meta, z.files)
        missing = [s for s in specs if parse_spec(s)[0] not in lay
                   or parse_spec(s)[1] not in lay[parse_spec(s)[0]]["parts"]]
        if missing:
            raise SystemExit(f"{gdir}: no saved map {missing}")
        arrays = {k: z[k] for k in {parse_spec(s)[0] for s in specs}}
    out = result_path(gdir, settings["fill"])
    raw = json.loads(out.read_text()) if out.exists() else None
    result = upgrade(raw) if raw else {"gradcam_dir": str(gdir), "settings": settings, "maps": [], "scores": {}}
    if result["settings"] != settings:
        raise SystemExit(f"{out} was made with {result['settings']}, not {settings}: move it away to start over")
    todo = [(i, r) for i, r in enumerate(meta["scores"])
            if (not only or r["name"] in only) and pending(result["scores"].get(r["name"]), specs, settings["random"])
            != ([], 0)]
    print(f"{gdir}: {len(todo)} score(s) to do for {list(specs)}", flush=True)
    if not todo:
        return []
    if raw and "methods" in raw["settings"] and not out.with_name(out.stem + ".v1.json").exists():
        shutil.copy2(out, out.with_name(out.stem + ".v1.json"))       # the pre-0.1.0 file, as written
    pre = pre or EvalPreprocessor(meta["model"], meta["video"], clip["video_args"])
    failed = []
    for i, rec in todo:
        res = curves_for(model, pre, target_for(rec, meta), ranked_maps(arrays, lay, i, specs, clip["merge"]),
                         settings, device, rec["plain_value"], merge=clip["merge"],
                         previous=result["scores"].get(rec["name"]))
        failed += [f"{gdir.name} {rec['name']}: {f}" for f in res["failed"]]
        result["scores"][rec["name"]] = res
        result["maps"] = sorted({c["map"] for r in result["scores"].values() for c in r["curves"] if "map" in c})
        tmp = out.with_suffix(".json.tmp")          # a job killed at walltime leaves the last good file
        tmp.write_text(json.dumps(result, indent=1))
        tmp.replace(out)
        _print(rec["name"], res)
    return failed


def run_plan(plan: dict) -> int:
    """Every dir of a deletion plan (`qwen-eye deletion`, `python -m qwen_eye run`)."""
    from qwen_eye.fullmodel import load_model
    from qwen_eye.run import resolve_device

    device = resolve_device(plan["device"])
    if not device.startswith("cuda"):
        raise SystemExit("deletion runs the full model hundreds of times: it needs a GPU")
    settings = {"fractions": plan["fractions"], "random": plan["random"], "fill": plan.get("fill", "grey")}
    t0 = time.time()
    model = load_model(plan["model"], device)
    print(f"model loaded in {time.time() - t0:.0f}s: {plan['model']}", flush=True)
    failed = []
    for d in plan["dirs"]:
        failed += run_dir(model, Path(d), settings, plan["maps"], device, only=plan.get("scores") or ())
    if failed:
        raise SystemExit(f"checks failed: {failed}")
    return 0
