"""Job plans: the plan.json every job runs from, and the one entry point that runs it.

`qwen-eye attn | gradcam | deletion` (cli.py) builds a plan with the functions below and runs
it here, or saves it (`--save-plan DIR`) for `python -m qwen_eye run DIR` elsewhere, e.g. in a
cluster job. A host tool that has its own configs (model paths, eval variants, job
submission) calls the same builders, so every plan is checked the same way.

Each builder checks its inputs (SystemExit with the reason), prints what the job will do and
returns the plan. Plan kinds: "attn" (maps A/B, run.py; the default when "kind" is absent),
"gradcam" (maps E, H, D, C, W; gradcam.py) and "deletion" (deletion.py). Extra keys a caller
adds (`extra`) are kept in plan.json and ignored by the job.
"""

from __future__ import annotations

import json
from pathlib import Path

ATTN_LAYERS = [2, 8, 16, 24, 26]           # = run.DEFAULT_LAYERS (not imported: it pulls in torch)


def _clips(videos) -> list[str]:
    out = [str(Path(v).expanduser().resolve()) for v in videos or []]
    missing = [v for v in out if not Path(v).is_file()]
    if missing:
        raise SystemExit(f"no such clip: {missing}")
    return out


def attn_plan(*, model: str, videos, out_root, variant_args: dict, layers=ATTN_LAYERS, device: str = "auto",
              name: str = "", where: str = "running here", extra: dict | None = None) -> dict:
    videos = _clips(videos)
    if not videos:
        raise SystemExit("give at least one video")
    plan = {"name": name, "model": str(model), "videos": videos, "out_root": str(out_root), **(extra or {}),
            "variant_args": dict(variant_args), "layers": sorted(set(layers)), "device": device}
    print(f"── attn{' ' + name if name else ''}: {len(videos)} clip(s), {model}\n"
          f"   video args {plan['variant_args']}; B layers {plan['layers']}; device {device}; {where}\n"
          f"   -> {out_root}/<clip>/viewer.html, attention.mp4", flush=True)
    return plan


def gradcam_plan(*, model: str, items: list[dict], videos, phrases, prompt: str, out_root, variant_args: dict,
                 layers=ATTN_LAYERS, maps=None, device: str = "auto", name: str = "", samples=(),
                 where: str = "running here", extra: dict | None = None) -> dict:
    """items: Item.to_dict() of each question; videos: clips for phrase maps only."""
    if not items and not videos:
        raise SystemExit("give a question or item, and/or a video with phrases")
    if videos and not phrases:
        raise SystemExit("a video without a question needs --phrase: it only has phrase scores")
    if "{phrase}" not in prompt:
        raise SystemExit(f"--prompt {prompt!r} needs a {{phrase}} placeholder")
    videos = _clips(videos)
    plan = {"kind": "gradcam", "name": name, "model": str(model), "items": list(items), "videos": videos,
            "phrases": list(dict.fromkeys(phrases or [])), "prompt": prompt, "samples": [str(s) for s in samples],
            "out_root": str(out_root), **(extra or {}), "variant_args": dict(variant_args),
            "layers": sorted(set(layers)), "maps": list(dict.fromkeys(maps)) if maps else None, "device": device}
    print(f"── gradcam{' ' + name if name else ''}: {len(items)} item(s), {len(videos)} clip(s), "
          f"{len(plan['phrases'])} phrase(s); {model}\n"
          + (f"   items from {', '.join(map(str, samples))}\n" if samples else "")
          + f"   video args {plan['variant_args']}; layers {plan['layers']}; "
          f"maps {plan['maps'] or 'all, adding only missing ones'}; {where}", flush=True)
    for it in items:
        said = "" if it["eval_answer"] is None else f", eval answered {it['eval_answer']!r}"
        print(f"   {it['key']}: correct {it['correct']}{said}, contrast with "
              f"{it['wrong'] or 'the strongest other option'}; {Path(it['video']).name}", flush=True)
    print(f"   -> {out_root}/<clip>/gradcam/<item or phrases>/viewer.html", flush=True)
    return plan


def deletion_plan(dirs, *, scores=(), fractions=(0.01, 0.02, 0.05, 0.1, 0.2), random: int = 3,
                  methods=("H", "E"), layers=None, maps=None, fill: str = "grey", device: str = "auto",
                  where: str = "running here") -> dict:
    """dirs: gradcam output dirs. maps: exact specs (KEY@PART); otherwise every saved part of each
    of `methods`, a signed map's layers cut to `layers`."""
    import numpy as np

    from qwen_eye import deletion
    from qwen_eye.items import Item
    from qwen_eye.maps import layouts

    if any(not 0 < f <= 1 for f in fractions) or random < 0:
        raise SystemExit(f"--fractions in (0, 1], --random >= 0; got {list(fractions)}, {random}")
    dirs = [Path(d).expanduser().resolve() for d in dirs]
    metas, lays, chosen = [], [], {}
    for d in dirs:
        if not (d / "meta.json").is_file() or not (d / "gradcam.npz").is_file():
            raise SystemExit(f"{d}: not a gradcam dir (needs meta.json and gradcam.npz)")
        m = json.loads((d / "meta.json").read_text())
        if m["checks_failed"]:
            raise SystemExit(f"{d}: its gradient-map checks failed: {m['checks_failed']}")
        names = [r["name"] for r in m["scores"]]
        chosen[d] = [n for n in names if not scores or n in scores]
        missing = [n for n in scores or [] if n not in names]
        if missing or not chosen[d]:
            raise SystemExit(f"{d}: no score {missing or list(scores)}; it has {names}")
        if any(n.startswith("phrase: ") for n in chosen[d]) and "{phrase}" not in m["phrase_prompt"]:
            raise SystemExit(f"{d}: its phrase scores are the retired log-prob score ({m['phrase_prompt']!r}); "
                             "rerun gradcam, or --score answer")
        if "answer" in chosen[d]:
            try:
                Item(**m["item"])
            except TypeError as e:
                raise SystemExit(f"{d}: item saved by an older version ({e}); rerun gradcam")
        with np.load(d / "gradcam.npz") as z:
            lays.append(layouts(m, z.files))
        metas.append(m)
    if len({m["model"] for m in metas}) != 1:
        raise SystemExit(f"the dirs come from different models: {sorted({m['model'] for m in metas})}")
    if maps:
        try:
            specs = list(dict.fromkeys("@".join(deletion.parse_spec(s)) for s in maps))
        except ValueError as e:
            raise SystemExit(str(e))
    else:
        specs = []
        for k in dict.fromkeys(methods):
            if any(k not in lay for lay in lays):
                raise SystemExit(f"map {k} is not saved in every dir (rerun gradcam --maps {k})")
            parts = [p for p in lays[0][k]["parts"] if all(p in lay[k]["parts"] for lay in lays)]
            if lays[0][k]["sides"] == "signed" and layers:
                parts = [str(L) for L in sorted(set(layers))]
            specs += [f"{k}@{p}" for p in parts]
    for d, lay in zip(dirs, lays):
        missing = [s for s in specs if deletion.parse_spec(s)[0] not in lay
                   or deletion.parse_spec(s)[1] not in lay[deletion.parse_spec(s)[0]]["parts"]]
        if missing:
            saved = {k: v["parts"] for k, v in lay.items()}
            raise SystemExit(f"{d}: maps saved {saved}, not all of {missing}")
    fractions = sorted(set(fractions))
    settings = {"fractions": fractions, "random": random, "fill": fill}
    per_score = 1 + len(fractions) * (2 * len(specs) + random)
    plan = {"kind": "deletion", "model": metas[0]["model"], "dirs": [str(d) for d in dirs],
            "scores": list(dict.fromkeys(scores or [])), "fractions": fractions, "random": random,
            "maps": specs, "fill": fill, "device": device}
    print(f"── deletion: {len(dirs)} dir(s); {plan['model']}\n"
          f"   maps {specs}, fill {fill}, fractions {fractions}, {random} random order(s): "
          f"{per_score} forwards per score from scratch; {where}", flush=True)
    for d in dirs:
        path = deletion.result_path(d, fill)
        prev = deletion.load_result(path)
        if prev and prev["settings"] != settings:
            raise SystemExit(f"{path} was made with {prev['settings']}, not {settings}: move it away to start over")
        todo = [deletion.pending(prev["scores"].get(n) if prev else None, specs, random) for n in chosen[d]]
        forwards = sum(1 + len(fractions) * (2 * len(t) + r) for t, r in todo if t or r)
        print(f"   {d}: {sum(1 for t, r in todo if t or r)}/{len(todo)} score(s) to do, {forwards} forwards "
              f"-> {path.name}", flush=True)
    return plan


def save(plan: dict, run_dir) -> Path:
    """plan.json into run_dir, for `python -m qwen_eye run run_dir`."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "plan.json").write_text(json.dumps(plan, indent=1))
    return run_dir / "plan.json"


def run(plan: dict) -> int:
    """Run a plan by its kind."""
    kind = plan.get("kind", "attn")
    if kind == "gradcam":
        from qwen_eye.gradcam import run_plan
    elif kind == "deletion":
        from qwen_eye.deletion import run_plan
    elif kind == "attn":
        from qwen_eye.run import run_plan
    else:
        raise SystemExit(f"unknown plan kind {kind!r}")
    return run_plan(plan)


def run_dir(run_dir) -> int:
    """`python -m qwen_eye run <run_dir>`: the plan.json in run_dir."""
    return run(json.loads((Path(run_dir) / "plan.json").read_text()))
