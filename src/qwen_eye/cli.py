"""The `qwen-eye` command.

    qwen-eye attn     --model M --video clip.mp4 --out OUT          maps A/B: viewer.html + attention.mp4
    qwen-eye gradcam  --model M --video clip.mp4 --phrase "a dog" --out OUT
                      --question "..." --correct B [--wrong A]     maps E, H, D, C, W (GPU)
                      --item location_motion:525 --samples DIR --data-root HF_HOME
    qwen-eye deletion OUT/<clip>/gradcam/<tag> ...                  deletion test of saved maps (GPU)
    qwen-eye render   OUT/<clip>[/gradcam/<tag>]                    redraw viewer + mp4, no model
    qwen-eye run      RUN_DIR                                       a plan.json saved with --save-plan

--model is a local Qwen3-VL snapshot dir or a Hugging Face id (e.g. Qwen/Qwen3-VL-8B-Instruct,
downloaded on first use); default $QWEN_EYE_MODEL. Video sampling follows lmms-eval's qwen3_vl
backend; the defaults (2 fps, at most 32 frames) are what we evaluate with.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# = targets.DEFAULT_PROMPT, maps.NAMES, deletion.FILLS / DEFAULT_FRACTIONS / DEFAULT_RANDOM, run.DEFAULT_LAYERS
# (not imported here: they pull in torch; tests/test_cli.py checks they agree)
DEFAULT_PROMPT = "Is there {phrase} in the video? Answer yes or no."
MAP_KEYS = ["E", "H", "D", "C", "W"]
FILLS = ["grey"]
DEFAULT_FRACTIONS = [0.01, 0.02, 0.05, 0.1, 0.2]
DEFAULT_RANDOM = 3
DEFAULT_LAYERS = [2, 8, 16, 24, 26]


def _model(arg: str | None) -> str:
    name = arg or os.environ.get("QWEN_EYE_MODEL")
    if not name:
        raise SystemExit("give --model (a Qwen3-VL snapshot dir or Hugging Face id) or set QWEN_EYE_MODEL")
    path = Path(name).expanduser()
    if path.is_dir():
        return str(path.resolve())
    if path.exists() or name.startswith((".", "/", "~")):
        raise SystemExit(f"--model {name}: not a directory")
    from huggingface_hub import snapshot_download

    return snapshot_download(name)


def _video_args(args) -> dict:
    """The lmms-eval qwen3_vl backend args that shape the video input; unset ones keep its defaults."""
    given = {"fps": args.fps, "max_num_frames": args.max_frames, "min_pixels": args.min_pixels,
             "max_pixels": args.max_pixels, "total_pixels": args.total_pixels}
    return {k: v for k, v in given.items() if v is not None}


def _add_video_args(p) -> None:
    g = p.add_argument_group("video sampling (as lmms-eval's qwen3_vl backend)")
    g.add_argument("--fps", type=float, default=2.0, help="frames per second sampled (default 2)")
    g.add_argument("--max-frames", type=int, default=32, help="at most this many frames (default 32)")
    g.add_argument("--min-pixels", type=int, help="per frame (backend default 200704)")
    g.add_argument("--max-pixels", type=int, help="per frame (backend default 1605632)")
    g.add_argument("--total-pixels", type=int, help="over the whole clip (backend default: none)")


def _add_run_args(p, gpu: bool) -> None:
    p.add_argument("--device", default="auto",
                   help="auto: cuda if available" + (" (a GPU is required)" if gpu else ", else cpu (fp32)"))
    p.add_argument("--dry-run", action="store_true", help="check the inputs and print the plan, run nothing")
    p.add_argument("--save-plan", metavar="DIR",
                   help="write DIR/plan.json instead of running (then: python -m qwen_eye run DIR)")


def _finish(plan: dict, args) -> int:
    from qwen_eye import plans

    if args.save_plan:
        path = plans.save(plan, args.save_plan)
        print(f"   plan -> {path}; run it with: python -m qwen_eye run {path.parent}")
        return 0
    if args.dry_run:
        return 0
    return plans.run(plan)


def cmd_attn(args) -> int:
    from qwen_eye import plans

    plan = plans.attn_plan(model=_model(args.model), videos=args.video, out_root=Path(args.out).expanduser().resolve(),
                           variant_args=_video_args(args), layers=args.layers, device=args.device, name=args.name)
    return _finish(plan, args)


def cmd_gradcam(args) -> int:
    from qwen_eye import items as itemsmod
    from qwen_eye import plans

    if args.question is not None:
        if len(args.video or []) != 1 or args.item:
            raise SystemExit("--question goes with exactly one --video (and no --item)")
        if not args.correct:
            raise SystemExit("--question needs --correct (the right option's letter)")
        items = [itemsmod.question(args.video[0], args.question, args.correct.upper(),
                                   args.wrong.upper() if args.wrong else None, args.letters, args.tag).to_dict()]
        videos = []
    else:
        if args.correct or args.wrong or args.tag != "question":
            raise SystemExit("--correct / --wrong / --tag go with --question")
        keys = list(dict.fromkeys(args.item or []))
        if keys and not (args.samples and args.data_root):
            raise SystemExit("--item reads an lmms-eval run's samples: give --samples DIR and --data-root HF_HOME")
        samples = [Path(s).expanduser() for s in args.samples or []]
        items = [itemsmod.load(k, samples, Path(args.data_root).expanduser()).to_dict() for k in keys]
        videos = args.video
    plan = plans.gradcam_plan(model=_model(args.model), items=items, videos=videos, phrases=args.phrase,
                              prompt=args.prompt, out_root=Path(args.out).expanduser().resolve(),
                              variant_args=_video_args(args), layers=args.layers, maps=args.maps,
                              device=args.device, name=args.name, samples=args.samples or ())
    return _finish(plan, args)


def cmd_deletion(args) -> int:
    from qwen_eye import plans

    plan = plans.deletion_plan(args.dir, scores=args.score or (), fractions=args.fractions, random=args.random,
                               methods=args.methods, layers=args.layers, maps=args.maps, fill=args.fill,
                               device=args.device)
    return _finish(plan, args)


def cmd_render(args) -> int:
    from qwen_eye.render import render, render_gradcam

    out_dir = Path(args.out_dir).expanduser()
    if (out_dir / "gradcam.npz").is_file():
        if args.point or args.layers:
            raise SystemExit("--point / --layers are for clip dirs; a gradient dir's mp4 rows are fixed")
        render_gradcam(out_dir)
        return 0
    points = None
    if args.point:
        points = [tuple(float(v) for v in p.split(",")) for p in args.point]
        if any(len(p) != 2 or not all(0 <= v <= 1 for v in p) for p in points):
            raise SystemExit(f"--point takes X,Y as fractions of width,height in [0, 1]; got {args.point}")
    render(out_dir, points=points, layers=args.layers)
    return 0


def cmd_run(args) -> int:
    from qwen_eye import plans

    return plans.run_dir(args.run_dir)


def build_parser() -> argparse.ArgumentParser:
    from qwen_eye import __version__

    parser = argparse.ArgumentParser(prog="qwen-eye", description="Where Qwen3-VL's vision tower looks in a video, "
                                     "and which patches its answers use.")
    parser.add_argument("--version", action="version", version=f"qwen-eye {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    a = sub.add_parser("attn", help="ViT attention maps A/B over clips: viewer.html + attention.mp4 (CPU or GPU)")
    a.add_argument("--video", action="append", required=True, help="clip (repeatable)")
    a.add_argument("--model", help="Qwen3-VL snapshot dir or Hugging Face id (default $QWEN_EYE_MODEL)")
    a.add_argument("--out", required=True, help="output root: OUT/<clip stem>/")
    a.add_argument("--name", default="", help="a label for the run, printed and kept in plan.json")
    a.add_argument("--layers", type=int, nargs="+", default=DEFAULT_LAYERS,
                   help="layers that get map B and an mp4 column (map A is saved for all 27)")
    _add_video_args(a)
    _add_run_args(a, gpu=False)
    a.set_defaults(func=cmd_attn)

    g = sub.add_parser("gradcam", help="gradient maps (E/H Grad-CAM + HiResCAM; D, C, W relevance) for questions "
                                       "and phrases (full model, GPU)")
    g.add_argument("--video", action="append", help="clip (repeatable): phrase maps, or the clip of --question")
    g.add_argument("--phrase", action="append",
                   help="thing to look for, e.g. 'a dog', scored by --prompt on every clip (repeatable)")
    g.add_argument("--prompt", default=DEFAULT_PROMPT, help="phrase question; score = log-odds of yes vs no")
    g.add_argument("--question", help="your own multiple-choice question on the one --video, sent verbatim "
                                      "(question, lettered options, answer instruction)")
    g.add_argument("--correct", help="with --question: the right option's letter")
    g.add_argument("--wrong", help="with --question: the letter to contrast with (default: the strongest other)")
    g.add_argument("--letters", default="AB", help="with --question: its option letters (default AB)")
    g.add_argument("--tag", default="question",
                   help="with --question: its name and output dir OUT/<clip>/gradcam/<tag> (default question; "
                        "give each question on a clip its own)")
    g.add_argument("--item", action="append", metavar="TASK:DOC",
                   help="VITATECS (direction:13) or MotionBench (location_motion:525) item from an lmms-eval run's "
                        "samples (repeatable): answer contrast on the exact eval prompt")
    g.add_argument("--samples", action="append", help="dir of that run's *_samples_*.jsonl (repeatable)")
    g.add_argument("--data-root", help="the run's HF_HOME (where its vitatecs/ and motionbench/ clips are)")
    g.add_argument("--model", help="Qwen3-VL snapshot dir or Hugging Face id (default $QWEN_EYE_MODEL)")
    g.add_argument("--out", required=True, help="output root: OUT/<clip stem>/gradcam/<item tag or phrases>/")
    g.add_argument("--name", default="", help="a label for the run, printed and kept in plan.json")
    g.add_argument("--layers", type=int, nargs="+", default=DEFAULT_LAYERS, help="ViT blocks to map")
    g.add_argument("--maps", nargs="+", choices=MAP_KEYS,
                   help="E Grad-CAM and H HiResCAM come with every backward; D (Chefer relevance), C (rollout), "
                        "W (readout only) add a pass to block 0 + one capture forward. Default: all, and into an "
                        "existing dir with the same scores only the missing ones are added; maps named here "
                        "replace saved ones (their deletion files become *.stale.json)")
    _add_video_args(g)
    _add_run_args(g, gpu=True)
    g.set_defaults(func=cmd_gradcam)

    d = sub.add_parser("deletion", help="deletion test of saved gradient maps: blank their top 2x2 blocks, rerun, "
                                        "compare with random blocks (full model, GPU)")
    d.add_argument("dir", nargs="+", help="gradcam output dir, OUT/<clip>/gradcam/<tag>")
    d.add_argument("--score", action="append",
                   help="only this score, as its meta.json names it: 'answer', 'phrase: a dog' (repeatable; "
                        "default: all)")
    d.add_argument("--fractions", type=float, nargs="+", default=DEFAULT_FRACTIONS,
                   help="fractions of the clip's 2x2 blocks to blank, one forward each per order")
    d.add_argument("--random", type=int, default=DEFAULT_RANDOM, help="random-order baselines (seeds 0..n-1)")
    d.add_argument("--methods", nargs="+", default=["H", "E"], choices=["H", "E", "D", "C", "W"],
                   help="H HiResCAM, E Grad-CAM (one map per layer); D Chefer relevance (all, per route, norm), "
                        "C rollout, W readout only: every saved part of each")
    d.add_argument("--layers", type=int, nargs="+", help="H/E layers (default: every layer all dirs have maps for)")
    d.add_argument("--maps", nargs="+", metavar="KEY@PART",
                   help="exact maps instead of --methods/--layers, e.g. H@8 D@all D@r16 C@all")
    d.add_argument("--fill", default="grey", choices=FILLS,
                   help="what a deleted block becomes: grey = mid-grey pixels (deletion.json)")
    _add_run_args(d, gpu=True)
    d.set_defaults(func=cmd_deletion)

    r = sub.add_parser("render", help="redraw viewer.html + attention.mp4 of a clip dir, or viewer.html + "
                                      "gradcam.mp4 of a gradcam dir (CPU, no model)")
    r.add_argument("out_dir", help="OUT/<clip stem>, or its gradcam/<tag> dir")
    r.add_argument("--point", action="append", metavar="X,Y",
                   help="mp4 B row for the 2x2 block under X,Y (fractions of width,height; repeatable; "
                        "default 0.5,0.5)")
    r.add_argument("--layers", type=int, nargs="+", help="mp4 columns (default: the layers with B maps)")
    r.set_defaults(func=cmd_render)

    j = sub.add_parser("run", help="run a saved plan: RUN_DIR/plan.json (from --save-plan, or a host tool)")
    j.add_argument("run_dir")
    j.set_defaults(func=cmd_run)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(sys.argv[1:] if argv is None else argv)
    return args.func(args)
