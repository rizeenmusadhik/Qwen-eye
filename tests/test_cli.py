"""The `qwen-eye` command: the plans it builds and saves, its checks, render and run (no model)."""

from __future__ import annotations

import argparse
import json

import numpy as np
import pytest

from qwen_eye import cli

from helpers import T, H, W, fake_gradcam_dir


def _plan(run_dir) -> dict:
    return json.loads((run_dir / "plan.json").read_text())


@pytest.fixture
def model_dir(tmp_path, monkeypatch):
    """--model must be a dir (or a Hugging Face id); no file in it is read when planning."""
    d = tmp_path / "Qwen3-VL-8B-Instruct"
    d.mkdir()
    monkeypatch.delenv("QWEN_EYE_MODEL", raising=False)
    return d


def test_cli_defaults_match_the_code():
    from qwen_eye import deletion, plans
    from qwen_eye.maps import NAMES
    from qwen_eye.run import DEFAULT_LAYERS
    from qwen_eye.targets import DEFAULT_PROMPT

    parser = cli.build_parser()
    sub = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction)).choices

    def action(command, dest):
        return next(a for a in sub[command]._actions if a.dest == dest)

    assert tuple(action("attn", "layers").default) == tuple(action("gradcam", "layers").default) == DEFAULT_LAYERS
    assert tuple(plans.ATTN_LAYERS) == DEFAULT_LAYERS
    assert action("gradcam", "prompt").default == DEFAULT_PROMPT
    assert set(action("gradcam", "maps").choices) == set(action("deletion", "methods").choices) == set(NAMES)
    assert tuple(action("deletion", "fill").choices) == deletion.FILLS
    assert tuple(action("deletion", "fractions").default) == deletion.DEFAULT_FRACTIONS
    assert action("deletion", "random").default == deletion.DEFAULT_RANDOM


def test_version(capsys):
    from qwen_eye import __version__

    with pytest.raises(SystemExit) as e:
        cli.main(["--version"])
    assert e.value.code == 0 and capsys.readouterr().out.strip() == f"qwen-eye {__version__}"


def test_attn_saves_the_plan_with_the_video_args(tmp_path, model_dir, capsys, monkeypatch):
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"")
    run = tmp_path / "run"
    assert cli.main(["attn", "--model", str(model_dir), "--video", str(clip), "--out", str(tmp_path / "out"),
                     "--save-plan", str(run)]) == 0
    assert f"python -m qwen_eye run {run}" in capsys.readouterr().out
    assert _plan(run) == {"name": "", "model": str(model_dir.resolve()), "videos": [str(clip.resolve())],
                          "out_root": str((tmp_path / "out").resolve()), "variant_args": {"fps": 2.0,
                          "max_num_frames": 32}, "layers": [2, 8, 16, 24, 26], "device": "auto"}
    monkeypatch.setenv("QWEN_EYE_MODEL", str(model_dir))                       # --model from the environment
    assert cli.main(["attn", "--video", str(clip), "--out", "o", "--fps", "1", "--max-frames", "8",
                     "--max-pixels", "1000", "--layers", "8", "2", "--name", "x", "--save-plan", str(run)]) == 0
    plan = _plan(run)
    assert plan["variant_args"] == {"fps": 1.0, "max_num_frames": 8, "max_pixels": 1000} and plan["layers"] == [2, 8]
    assert plan["model"] == str(model_dir.resolve()) and plan["name"] == "x"
    assert cli.main(["attn", "--video", str(clip), "--out", "o", "--dry-run"]) == 0
    with pytest.raises(SystemExit, match="no such clip"):
        cli.main(["attn", "--video", str(tmp_path / "nope.mp4"), "--out", "o", "--dry-run"])
    with pytest.raises(SystemExit, match="not a directory"):
        cli.main(["attn", "--model", str(clip), "--video", str(clip), "--out", "o", "--dry-run"])
    monkeypatch.delenv("QWEN_EYE_MODEL")
    with pytest.raises(SystemExit, match="QWEN_EYE_MODEL"):
        cli.main(["attn", "--video", str(clip), "--out", "o", "--dry-run"])


def test_gradcam_plans_items_questions_and_phrases(tmp_path, model_dir, capsys):
    q = "Which of the following best describes the content of the video: \n(A) x. \n(B) y."
    samples = tmp_path / "samples"
    samples.mkdir()
    (samples / "s_samples_vitatecs_intensity.jsonl").write_text(json.dumps(
        {"doc_id": 177, "target": "(B) y.", "filtered_resps": "A", "input": q + "\nPlease response with a single letter (A or B):",
         "accuracy": {"src_dataset": "MSRVTT", "video_id": "v.mp4", "question": q}}) + "\n")
    mb = "Which way?\nA. Left\nB. Up\nC. Right\nAnswer with the option's letter from the given choices directly (A, B or C)."
    (samples / "s_samples_motionbench_dev_location_motion.jsonl").write_text(json.dumps(
        {"doc_id": 525, "target": "C", "filtered_resps": "C", "input": mb,
         "motionbench_accuracy": {"answer": "C", "pred": "C", "n_options": 3, "video_path": "m.mp4"}}) + "\n")
    hf = tmp_path / "hf"
    (hf / "vitatecs" / "MSRVTT").mkdir(parents=True)
    (hf / "vitatecs" / "MSRVTT" / "v.mp4").write_bytes(b"")
    (hf / "motionbench" / "MotionBench" / "self-collected").mkdir(parents=True)
    (hf / "motionbench" / "MotionBench" / "self-collected" / "m.mp4").write_bytes(b"")
    common = ["--model", str(model_dir), "--out", str(tmp_path / "out")]
    run = tmp_path / "run"
    assert cli.main(["gradcam", *common, "--item", "intensity:177", "--item", "location_motion:525", "--phrase", "a dog",
                     "--samples", str(samples), "--data-root", str(hf), "--save-plan", str(run)]) == 0
    plan = _plan(run)
    assert plan["kind"] == "gradcam" and plan["phrases"] == ["a dog"] and plan["videos"] == []
    assert plan["samples"] == [str(samples)] and "{phrase}" in plan["prompt"]
    assert [(i["key"], i["correct"], i["wrong"], i["eval_answer"]) for i in plan["items"]] == \
        [("intensity:177", "B", "A", "A"), ("location_motion:525", "C", None, "C")]
    assert plan["variant_args"] == {"fps": 2.0, "max_num_frames": 32}
    assert plan["maps"] is None                                     # every map; into a saved dir only missing ones
    capsys.readouterr()
    assert cli.main(["gradcam", *common, "--item", "intensity:177", "--samples", str(samples), "--data-root", str(hf),
                     "--maps", "D", "C", "D", "--save-plan", str(run)]) == 0
    assert "maps ['D', 'C']" in capsys.readouterr().out
    assert _plan(run)["maps"] == ["D", "C"]                         # named: these replace saved ones
    clip = hf / "vitatecs" / "MSRVTT" / "v.mp4"
    prompt = "What happens?\nA. It opens.\nB. It closes.\nC. Nothing.\nAnswer with the option's letter."
    assert cli.main(["gradcam", *common, "--video", str(clip), "--question", prompt, "--correct", "b",
                     "--letters", "ABC", "--phrase", "a door", "--save-plan", str(run)]) == 0
    plan = _plan(run)
    (it,) = plan["items"]
    assert plan["videos"] == [] and plan["phrases"] == ["a door"]   # the question's clip gets the phrases too
    assert (it["key"], it["benchmark"], it["prompt"], it["correct"], it["wrong"], it["eval_answer"]) == \
        ("question", "custom", prompt, "B", None, None)
    assert it["video"] == str(clip.resolve()) and list(it["options"]) == ["A", "B", "C"]
    with pytest.raises(SystemExit, match="two different letters"):
        cli.main(["gradcam", *common, "--video", str(clip), "--question", prompt, "--correct", "B", "--wrong", "B",
                  "--dry-run"])
    with pytest.raises(SystemExit, match="needs --correct"):
        cli.main(["gradcam", *common, "--video", str(clip), "--question", prompt, "--dry-run"])
    with pytest.raises(SystemExit, match="exactly one --video"):
        cli.main(["gradcam", *common, "--video", str(clip), "--video", str(clip), "--question", prompt,
                  "--correct", "A", "--dry-run"])
    with pytest.raises(SystemExit, match="go with --question"):
        cli.main(["gradcam", *common, "--video", str(clip), "--phrase", "x", "--correct", "A", "--dry-run"])
    with pytest.raises(SystemExit, match="--data-root"):
        cli.main(["gradcam", *common, "--item", "intensity:177", "--samples", str(samples), "--dry-run"])
    with pytest.raises(SystemExit, match="needs --phrase"):
        cli.main(["gradcam", *common, "--video", str(clip), "--dry-run"])
    with pytest.raises(SystemExit, match="placeholder"):
        cli.main(["gradcam", *common, "--video", str(clip), "--phrase", "x", "--prompt", "Describe it.", "--dry-run"])


def test_gradcam_plan_runs_its_question(tmp_path, model_dir, monkeypatch):
    """What the job makes of the plan's question: the item gradcam.run_plan rebuilds."""
    from qwen_eye.items import Item

    clip = tmp_path / "c.mp4"
    clip.write_bytes(b"")
    run = tmp_path / "run"
    cli.main(["gradcam", "--model", str(model_dir), "--out", "o", "--video", str(clip), "--question", "Q?\nA. a\nB. b",
              "--correct", "A", "--save-plan", str(run)])
    it = Item(**_plan(run)["items"][0])
    assert it.tag == "question" and it.wrong is None and it.options == {"A": "", "B": ""}
    cli.main(["gradcam", "--model", str(model_dir), "--out", "o", "--video", str(clip), "--question", "Q?\nA. a\nB. b",
              "--correct", "A", "--wrong", "b", "--tag", "door-2", "--save-plan", str(run)])
    it = Item(**_plan(run)["items"][0])
    assert (it.key, it.tag, it.wrong) == ("door-2", "door-2", "B")
    for bad in ("phrases", "a/b", "x:1"):
        with pytest.raises(SystemExit, match="names a dir"):
            cli.main(["gradcam", "--model", str(model_dir), "--out", "o", "--video", str(clip), "--question", "Q?",
                      "--correct", "A", "--tag", bad, "--dry-run"])
    with pytest.raises(SystemExit, match="go with --question"):
        cli.main(["gradcam", "--model", str(model_dir), "--out", "o", "--video", str(clip), "--phrase", "x",
                  "--tag", "t", "--dry-run"])


def test_deletion_plans_check_the_saved_maps(tmp_path, capsys):
    from qwen_eye import deletion

    d = fake_gradcam_dir(tmp_path)
    run = tmp_path / "run"
    assert cli.main(["deletion", str(d), "--score", "answer", "--layers", "8", "--save-plan", str(run)]) == 0
    out = capsys.readouterr().out
    per_score = 1 + 5 * (2 * 2 + 3)                                  # clean + fractions x (2 maps x 2 orders + random)
    assert "maps ['H@8', 'E@8'], fill grey" in out and f"{per_score} forwards per score from scratch" in out
    assert f"1/1 score(s) to do, {per_score} forwards -> deletion.json" in out
    assert _plan(run) == {"kind": "deletion", "model": "/m/qwen3", "dirs": [str(d)], "scores": ["answer"],
                          "fractions": list(deletion.DEFAULT_FRACTIONS), "random": 3, "maps": ["H@8", "E@8"],
                          "fill": "grey", "device": "auto"}
    assert cli.main(["deletion", str(d), "--dry-run"]) == 0
    assert "maps ['H@2', 'H@8', 'E@2', 'E@8']" in capsys.readouterr().out    # default: every saved layer
    assert cli.main(["deletion", str(d), "--maps", "E@2", "H@8", "E@2", "--dry-run"]) == 0
    assert "maps ['E@2', 'H@8']" in capsys.readouterr().out
    (d / "deletion.json").write_text(json.dumps({                     # done: H@8 + 3 random for the answer
        "settings": {"fractions": list(deletion.DEFAULT_FRACTIONS), "random": 3, "fill": "grey"}, "maps": ["H@8"],
        "scores": {"answer": {"failed": [], "curves": [{"map": "H@8", "order": o} for o in ("for_first", "against_first")]
                              + [{"order": "random", "seed": s} for s in range(3)]}}}))
    assert cli.main(["deletion", str(d), "--maps", "H@8", "E@8", "--dry-run"]) == 0
    assert f"2/2 score(s) to do, {1 + 5 * 2 + per_score} forwards" in capsys.readouterr().out   # answer: E@8 only
    with pytest.raises(SystemExit, match="move it away"):
        cli.main(["deletion", str(d), "--random", "4", "--dry-run"])
    (d / "deletion.json").unlink()
    with pytest.raises(SystemExit, match="no score"):
        cli.main(["deletion", str(d), "--score", "phrase: a cat", "--dry-run"])
    with pytest.raises(SystemExit, match="not all of"):
        cli.main(["deletion", str(d), "--layers", "16", "--dry-run"])
    with pytest.raises(SystemExit, match="not all of"):
        cli.main(["deletion", str(d), "--maps", "D@all", "--dry-run"])
    with pytest.raises(SystemExit, match="rerun gradcam --maps D"):
        cli.main(["deletion", str(d), "--methods", "D", "--dry-run"])
    with pytest.raises(SystemExit, match="<key>@<part>"):
        cli.main(["deletion", str(d), "--maps", "H8", "--dry-run"])
    with pytest.raises(SystemExit, match="not a gradcam dir"):
        cli.main(["deletion", str(tmp_path), "--dry-run"])
    old = fake_gradcam_dir(tmp_path / "old", prompt="Describe the video.")
    with pytest.raises(SystemExit, match="retired"):
        cli.main(["deletion", str(old), "--dry-run"])
    assert cli.main(["deletion", str(old), "--score", "answer", "--dry-run"]) == 0
    np.savez(old / "gradcam.npz", E=np.zeros((2, 2, T, H, W)))        # H missing in one of two dirs
    with pytest.raises(SystemExit, match="rerun gradcam --maps H"):
        cli.main(["deletion", str(d), str(old), "--score", "answer", "--dry-run"])


def test_render_redraws_clip_and_gradient_dirs(monkeypatch, tmp_path):
    from qwen_eye import render

    seen = []
    monkeypatch.setattr(render, "render_gradcam", lambda d: seen.append(("gradcam", d)))
    monkeypatch.setattr(render, "render", lambda d, points=None, layers=None: seen.append(("clip", d, points, layers)))
    (tmp_path / "gradcam.npz").write_bytes(b"")
    assert cli.main(["render", str(tmp_path)]) == 0
    with pytest.raises(SystemExit, match="clip dirs"):
        cli.main(["render", str(tmp_path), "--layers", "8"])
    assert cli.main(["render", str(tmp_path / "clip"), "--point", "0.25,0.5", "--layers", "8"]) == 0
    with pytest.raises(SystemExit, match="fractions"):
        cli.main(["render", str(tmp_path / "clip"), "--point", "2,0.5"])
    assert seen == [("gradcam", tmp_path), ("clip", tmp_path / "clip", [(0.25, 0.5)], [8])]


def test_run_dispatches_on_plan_kind(tmp_path):
    from qwen_eye.__main__ import main

    for kind in ("gradcam", "deletion"):
        (tmp_path / "plan.json").write_text(json.dumps({"kind": kind, "device": "cpu"}))
        with pytest.raises(SystemExit, match="needs a GPU"):
            main(["run", str(tmp_path)])
    (tmp_path / "plan.json").write_text(json.dumps({"kind": "other"}))
    with pytest.raises(SystemExit, match="unknown plan kind"):
        main(["run", str(tmp_path)])
