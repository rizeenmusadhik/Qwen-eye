"""Benchmark items read from lmms-eval sample files."""

from __future__ import annotations

import pytest


def test_items_come_from_the_eval_samples(tmp_path):
    import json

    from qwen_eye import items

    q = ("Which of the following best describes the content of the video: \n(A) A car stops. \n"
         "(B) A car drives.")
    row = {"doc_id": 13, "target": "(B) A car drives.", "filtered_resps": "B",
           "accuracy": {"src_dataset": "MSRVTT", "video_id": "v1.mp4", "question": q, "video-llm-prediction": "B"},
           "input": q + "\nPlease response with a single letter (A or B):"}
    (tmp_path / "x_samples_vitatecs_direction.jsonl").write_text(json.dumps({**row, "doc_id": 12}) + "\n"
                                                                + json.dumps(row) + "\n")
    (tmp_path / "vitatecs" / "MSRVTT").mkdir(parents=True)
    (tmp_path / "vitatecs" / "MSRVTT" / "v1.mp4").write_bytes(b"")
    it = items.load("direction:13", tmp_path, tmp_path)
    assert (it.prompt, it.correct, it.wrong, it.eval_answer, it.eval_letter, it.tag, it.benchmark) == \
        (row["input"], "B", "A", "B", "B", "direction_13", "vitatecs")
    assert it.options == {"A": "A car stops.", "B": "A car drives."}
    assert it.video == str(tmp_path / "vitatecs" / "MSRVTT" / "v1.mp4")
    with pytest.raises(SystemExit):
        items.parse_key("speed:3")


def test_motionbench_items_contrast_the_models_pick_or_leave_it_to_run_time(tmp_path):
    import json

    from qwen_eye import items

    prompt = ("In which direction does the boy move?\nA. Left\nB. Towards the camera\nC. Away\nD. Right\n"
              "Answer with the option's letter from the given choices directly (A, B, C or D).")
    rows = [{"doc_id": d, "target": "D", "filtered_resps": pred + ".", "input": prompt,
             "motionbench_accuracy": {"answer": "D", "pred": pred, "n_options": 4, "video_path": name}}
            for d, pred, name in ((525, "C", "a.mp4"), (1, "D", "b.mp4"))]
    mb = tmp_path / "mb"
    mb.mkdir()
    (mb / "x_samples_motionbench_dev_location_motion.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    root = tmp_path / "hf" / "motionbench"
    for d in ("MotionBench/self-collected", "MotionBench/public-dataset", "repaired"):
        (root / d).mkdir(parents=True)
    for n in ("a.mp4", "b.mp4"):
        (root / "MotionBench" / "self-collected" / n).write_bytes(b"")
    (root / "repaired" / "b.mp4").write_bytes(b"")
    (root / "repaired" / "manifest.json").write_text(json.dumps({"clips": ["b.mp4"]}))
    vt = tmp_path / "vt"
    vt.mkdir()
    (vt / "x_samples_vitatecs_direction.jsonl").write_text("")
    wrong = items.load("location_motion:525", [vt, mb], tmp_path / "hf")
    assert (wrong.benchmark, wrong.correct, wrong.wrong, wrong.eval_letter) == ("motionbench", "D", "C", "C")
    assert wrong.options == {"A": "Left", "B": "Towards the camera", "C": "Away", "D": "Right"}
    assert wrong.video == str(root / "MotionBench" / "self-collected" / "a.mp4")
    right = items.load("location_motion:1", [vt, mb], tmp_path / "hf")
    assert right.wrong is None and right.eval_letter == "D"
    assert right.video == str(root / "repaired" / "b.mp4")          # the eval read the repaired copy
    with pytest.raises(SystemExit, match="exactly one"):
        items.load("location_motion:1", [mb, mb], tmp_path / "hf")

