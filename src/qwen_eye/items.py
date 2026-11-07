"""Multiple-choice items for the answer-contrast maps: your own question (`question`), or a
VITATECS / MotionBench item read from an lmms-eval run's saved samples (`load`).

lmms-eval writes one samples jsonl per task: `*_samples_vitatecs_<aspect>.jsonl` and
`*_samples_motionbench_dev_<category>.jsonl`. Each row carries the exact prompt text the
backend sent (`input` = doc_to_text: question, lettered options, post prompt), the clip, the
correct letter and the model's reply. So an item needs no dataset load, and our forward can
be checked against the answer the eval recorded.

The key names the task: `direction:13` (a VITATECS aspect) or `location_motion:525` (a
MotionBench category); the two sets of names do not overlap.

The contrast's wrong letter: VITATECS has two options, so it is the other one. A MotionBench
item has 3-4: a wrongly answered one is contrasted with the model's own pick; for a correctly
answered one `wrong` stays None and targets.LetterContrast takes the strongest other option
(highest logit) at run time.

Clip paths follow lmms-eval's task code under its HF_HOME (`hf_home`): VITATECS clips in
<hf_home>/vitatecs/<src_dataset>/<video_id>, MotionBench clips in
<hf_home>/motionbench/MotionBench/{public-dataset,self-collected}/<name>, where a re-encoded
copy in <hf_home>/motionbench/repaired/ listed in its manifest.json wins (for clips the
original files cannot be decoded from).
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

ASPECTS = ("direction", "intensity", "sequence", "localization", "compositionality", "type")
CATEGORIES = ("motion_recognition", "motion_objects", "location_motion", "action_order", "repetition_count",
              "camera_motion")
MOTIONBENCH_SUBDIRS = ("public-dataset", "self-collected")


@dataclass
class Item:
    key: str            # "direction:13", "location_motion:525", or your own question's name
    benchmark: str      # "vitatecs" | "motionbench" | "custom"
    video: str
    prompt: str         # the user text the eval sent, verbatim
    correct: str        # letter
    wrong: str | None   # letter, or None: the strongest other option, chosen at run time
    eval_answer: str | None   # the eval's reply, verbatim; None for your own question
    eval_letter: str    # the letter the eval parsed from it ("" if none)
    options: dict       # {"A": "...", "B": "...", ...}

    @property
    def tag(self) -> str:
        return self.key.replace(":", "_")

    def to_dict(self) -> dict:
        return asdict(self)


def question(video: str | Path, prompt: str, correct: str, wrong: str | None = None, letters: str = "AB",
             name: str = "question") -> Item:
    """Your own multiple-choice question on a clip. `prompt` is the user text sent verbatim
    (question, lettered options, answer instruction); `letters` the option letters it offers.
    With no `wrong`, the contrast is with the strongest other letter at run time. `name` is the
    item's key, and its output dir <clip>/gradcam/<name>: give each question on a clip its own."""
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", name) or name in ("phrases", ".", ".."):
        raise SystemExit(f"question name {name!r}: letters, digits, _ . - only, not 'phrases' (it names a dir)")
    letters = "".join(dict.fromkeys(letters.upper()))
    if correct not in letters or (wrong is not None and (wrong not in letters or wrong == correct)):
        raise SystemExit(f"--correct {correct!r} and --wrong {wrong!r} must be two different letters of {letters!r}")
    if not Path(video).is_file():
        raise SystemExit(f"no such clip: {video}")
    return Item(key=name, benchmark="custom", video=str(Path(video).resolve()), prompt=prompt, correct=correct,
                wrong=wrong, eval_answer=None, eval_letter="", options={k: "" for k in letters})


def parse_key(key: str) -> tuple[str, str, int]:
    """-> (benchmark, task, doc_id)"""
    task, sep, doc = key.partition(":")
    if sep and doc.isdigit() and task in ASPECTS:
        return "vitatecs", task, int(doc)
    if sep and doc.isdigit() and task in CATEGORIES:
        return "motionbench", task, int(doc)
    raise SystemExit(f"item {key!r}: expected <task>:<doc_id>, task a VITATECS aspect {ASPECTS} "
                     f"or a MotionBench category {CATEGORIES}; e.g. direction:13, location_motion:525")


def _pattern(benchmark: str, task: str) -> str:
    return f"*_samples_vitatecs_{task}.jsonl" if benchmark == "vitatecs" else f"*_samples_motionbench_dev_{task}.jsonl"


def _vitatecs_options(question: str) -> dict:
    """{'A': text, 'B': text} from '...video: \\n(A) x. \\n(B) y.'"""
    a = question.split("\n(A) ", 1)[1]
    a, b = a.split(" \n(B) ", 1)
    return {"A": a.strip(), "B": b.strip()}


def _motionbench_options(prompt: str) -> dict:
    """{'A': text, ...} from the lines 'A. text' of motionbench_doc_to_text's prompt."""
    return {m[1]: m[2].strip() for m in re.finditer(r"^([A-F])\. (.*)$", prompt, re.M)}


def motionbench_video(hf_home: str | Path, name: str) -> Path:
    """The clip the eval read: a re-encoded copy in ../repaired listed in its finished
    manifest wins over the original."""
    root = Path(hf_home) / "motionbench" / "MotionBench"
    manifest = root.parent / "repaired" / "manifest.json"
    if manifest.exists() and name in json.loads(manifest.read_text())["clips"] \
            and (root.parent / "repaired" / name).exists():
        return root.parent / "repaired" / name
    for sub in MOTIONBENCH_SUBDIRS:
        if (root / sub / name).exists():
            return root / sub / name
    return root / MOTIONBENCH_SUBDIRS[0] / name


def load(key: str, samples: str | Path | list, hf_home: str | Path) -> Item:
    """The item from the one samples dir (of `samples`, a dir or a list of them) that has its task."""
    benchmark, task, doc_id = parse_key(key)
    dirs = [Path(d) for d in (samples if isinstance(samples, (list, tuple)) else [samples])]
    found = [(d, sorted(d.glob(_pattern(benchmark, task)))) for d in dirs]
    found = [(d, f) for d, f in found if f]
    if len(found) != 1:
        raise SystemExit(f"{key}: need {_pattern(benchmark, task)} in exactly one of {[str(d) for d in dirs]}, "
                         f"found it in {[str(d) for d, _ in found]}")
    path = found[0][1][-1]
    with open(path) as f:
        row = next((r for r in map(json.loads, f) if r["doc_id"] == doc_id), None)
    if row is None:
        raise SystemExit(f"{path} has no doc_id {doc_id}")
    if benchmark == "vitatecs":
        acc = row["accuracy"]
        correct = row["target"][1]
        video = Path(hf_home) / "vitatecs" / acc["src_dataset"] / acc["video_id"]
        options, wrong = _vitatecs_options(acc["question"]), {"A": "B", "B": "A"}[correct]
        letter = acc.get("video-llm-prediction") or ""
    else:
        acc = row["motionbench_accuracy"]
        correct, letter = acc["answer"], acc.get("pred") or ""
        video = motionbench_video(hf_home, acc["video_path"])
        options = _motionbench_options(row["input"])
        wrong = letter if letter in options and letter != correct else None
        if len(options) != acc["n_options"] or correct not in options:
            raise SystemExit(f"{key}: parsed options {sorted(options)} disagree with the sample "
                             f"({acc['n_options']} options, answer {correct})")
    if not video.is_file():
        raise SystemExit(f"{key}: clip {video} not found")
    return Item(key=key, benchmark=benchmark, video=str(video), prompt=row["input"], correct=correct, wrong=wrong,
                eval_answer=row["filtered_resps"], eval_letter=letter if letter in options else "", options=options)
