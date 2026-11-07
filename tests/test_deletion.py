"""The deletion test: blanking, orders, curves, resuming, old files."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from qwen_eye import layout

from helpers import T, H, W, N, FakePre, GapTarget, deletion_settings, SIGNED


def test_ranked_block_is_the_blanked_block():
    """A block ranked in grid order (block_sums) is the 2x2 patch square blank() greys, in both
    frames of its pair and nowhere else."""
    from qwen_eye import deletion

    order = layout.block_order(H, W)
    pixels = torch.rand(T * N, 3 * 2 * 16 * 16) * 2 - 1
    kw = dict(patch=16, tpatch=2, mean=[0.5] * 3, std=[0.5] * 3)
    before = layout.frames_from_pixels(pixels, (T, H, W), order, **kw)
    for p, r, c in [(0, 0, 0), (1, 2, 5), (2, 3, 2)]:
        x = np.zeros((T, H, W))
        x[p, r, c] = 1.0
        sums = deletion.block_sums(x)
        (g,) = np.flatnonzero(sums)
        assert sums.shape == (T * N // 4,) and g == p * (N // 4) + layout.block_of(r, c, W)
        after = layout.frames_from_pixels(deletion.blank(pixels, [g]), (T, H, W), order, **kw)
        r0, c0 = 32 * (r // 2), 32 * (c // 2)
        changed = np.zeros(after.shape[:3], bool)
        changed[2 * p:2 * p + 2, r0:r0 + 32, c0:c0 + 32] = True
        assert (after[changed] == 128).all()                                   # mid grey: 0.5 * 255, rounded
        assert (after[~changed] == before[~changed]).all()


def test_deletion_orders_counts_and_area():
    from qwen_eye import deletion

    o = deletion.orders(np.array([1.0, -2.0, 3.0, 0.0]))
    assert o["for_first"].tolist() == [2, 0, 3, 1] and o["against_first"].tolist() == [1, 3, 0, 2]
    o = deletion.orders(np.array([1.0, 0.0, 3.0]), np.array([0.0, 5.0, 1.0]))     # two-sided: each its own
    assert o["for_first"].tolist() == [2, 0, 1] and o["against_first"].tolist() == [1, 2, 0]
    assert deletion.parse_spec("D@r16") == ("D", "r16")
    with pytest.raises(ValueError, match="<key>@<part>"):
        deletion.parse_spec("H8")
    a, b = deletion.random_orders(10, 2), deletion.random_orders(10, 2)
    assert all(sorted(x.tolist()) == list(range(10)) for x in a) and all((x == y).all() for x, y in zip(a, b))
    assert deletion.counts(300, [0.001, 0.1, 1.0]) == [1, 30, 300]
    assert deletion.area([0.1, 0.2], [-1.0, -1.0]) == pytest.approx(-0.75)    # (0.05 + 0.1) / 0.2


def test_deletion_curves_are_forwards_on_blanked_pixels(tapped):
    from qwen_eye import deletion

    model, batch = tapped["model"], tapped["batch"]
    clean = float(GapTarget().score(tapped["plain"]))
    rng = np.random.default_rng(0)
    maps = {"H": rng.normal(size=(1, 2, T, H, W))}                              # one score, layers 0 and 1
    ranked = deletion.ranked_maps(maps, SIGNED, 0, ["H@1"])
    res = deletion.curves_for(model, FakePre(batch), GapTarget(), ranked, deletion_settings(), "cpu", clean)
    assert res["failed"] == [] and res["clean"] == pytest.approx(clean, abs=1e-9)
    assert res["blocks"] == T * N // 4 and res["ks"] == [round(0.25 * T * N / 4), T * N // 4]
    assert [(c.get("map"), c.get("layer"), c["order"]) for c in res["curves"]] == \
        [("H@1", "1", "for_first"), ("H@1", "1", "against_first"), (None, None, "random"), (None, None, "random")]
    # the first point of for_first by hand: the top blocks of layer 1's map, their rows zeroed
    top = np.argsort(-deletion.block_sums(maps["H"][0, 1]))[:res["ks"][0]]
    px = batch["pixel_values_videos"].clone()
    for g in top:
        px[4 * g:4 * g + 4] = 0
    with torch.no_grad():
        want = float(GapTarget().score(model(**{**batch, "pixel_values_videos": px}, use_cache=False,
                                                logits_to_keep=1).logits))
    assert res["curves"][0]["score"][0] == pytest.approx(want, abs=1e-9)
    assert want != pytest.approx(clean, abs=1e-6)
    # every order ends with every block blanked: the same input
    assert len({round(c["score"][-1], 9) for c in res["curves"]}) == 1
    (row,) = res["summary"]
    assert row["map"] == "H@1" and row["for_first"] == res["curves"][0]["area"] and row["random_sd"] is not None
    bad = deletion.curves_for(model, FakePre(batch), GapTarget(), ranked, deletion_settings(), "cpu", clean + 1.0)
    assert bad["failed"] and bad["curves"] == []


def test_deletion_adds_maps_and_random_orders_to_a_done_score(tapped, monkeypatch):
    """Curves already done are kept as they are; only the new map and the missing seed run."""
    from qwen_eye import deletion

    model, batch = tapped["model"], tapped["batch"]
    clean = float(GapTarget().score(tapped["plain"]))
    rng = np.random.default_rng(0)
    maps = {"H": rng.normal(size=(1, 2, T, H, W)), "D": np.abs(rng.normal(size=(1, 2, 1, T, H, W)))}
    lay = SIGNED | {"D": {"name": "Chefer relevance", "sides": 2, "parts": ["all"]}}
    first = deletion.curves_for(model, FakePre(batch), GapTarget(), deletion.ranked_maps(maps, lay, 0, ["H@1"]),
                                deletion_settings(), "cpu", clean)
    runs = []
    real = deletion.delete
    monkeypatch.setattr(deletion, "delete", lambda run, px, o, ks, merge=2: runs.append(o) or real(run, px, o, ks, merge))
    ranked = deletion.ranked_maps(maps, lay, 0, ["H@1", "D@all"])
    res = deletion.curves_for(model, FakePre(batch), GapTarget(), ranked, deletion_settings(random=3), "cpu", clean,
                              previous=first)
    assert res["curves"][:4] == first["curves"]
    assert [(c.get("map"), c["order"], c.get("seed")) for c in res["curves"][4:]] == \
        [("D@all", "for_first", None), ("D@all", "against_first", None), (None, "random", 2)]
    for_sums, against_sums = ranked["D@all"]
    assert np.array_equal(runs[0], np.argsort(-for_sums, kind="stable"))        # each side ranked on its own
    assert np.array_equal(runs[1], np.argsort(-against_sums, kind="stable"))
    assert np.array_equal(runs[2], deletion.random_orders(T * N // 4, 3)[2]) and len(runs) == 3
    assert [r["map"] for r in res["summary"]] == ["H@1", "D@all"] and res["summary"][0]["random_sd"] is not None
    assert deletion.pending(res, ["H@1", "D@all"], 3) == ([], 0)
    assert deletion.pending(res, ["H@1", "W@all"], 4) == (["W@all"], 1)
    assert deletion.pending({**res, "failed": ["x"]}, ["H@1"], 3) == (["H@1"], 3)


def test_deletion_dir_resumes_and_refuses_other_settings(tapped, tmp_path, monkeypatch):
    import json

    from qwen_eye import deletion

    model, batch = tapped["model"], tapped["batch"]
    clean = float(GapTarget().score(tapped["plain"]))
    clip, gdir = tmp_path / "clip", tmp_path / "clip" / "gradcam" / "phrases"
    gdir.mkdir(parents=True)
    (clip / "meta.json").write_text(json.dumps({"video_args": {}, "merge": 2}))
    (gdir / "meta.json").write_text(json.dumps({
        "model": "m", "video": "v.mp4", "clip_dir": str(clip), "layers": [0, 1], "item": None, "phrase_prompt": "",
        "scores": [{"name": "gap", "plain_value": clean}, {"name": "gap2", "plain_value": clean + 1.0}]}))
    rng = np.random.default_rng(0)
    np.savez(gdir / "gradcam.npz", H=rng.normal(size=(2, 2, T, H, W)), E=rng.normal(size=(2, 2, T, H, W)))
    monkeypatch.setattr(deletion, "target_for", lambda rec, meta: GapTarget())
    pre, settings = FakePre(batch), deletion_settings(random=1)
    failed = deletion.run_dir(model, gdir, settings, ["H@1"], "cpu", pre=pre)
    out = json.loads((gdir / "deletion.json").read_text())
    assert out["settings"] == settings and out["maps"] == ["H@1"] and set(out["scores"]) == {"gap", "gap2"}
    assert failed == [f"phrases gap2: {out['scores']['gap2']['failed'][0]}"] and not out["scores"]["gap"]["failed"]
    with pytest.raises(SystemExit, match="no saved map"):
        deletion.run_dir(model, gdir, settings, ["D@all"], "cpu", pre=pre)
    calls = []
    real = deletion.curves_for
    monkeypatch.setattr(deletion, "curves_for", lambda *a, **k: calls.append((a[-1], sorted(a[3]))) or real(*a, **k))
    deletion.run_dir(model, gdir, settings, ["H@1"], "cpu", pre=pre)
    assert calls == [(clean + 1.0, ["H@1"])]                         # the failed score again, the done one not
    calls.clear()
    deletion.run_dir(model, gdir, settings, ["H@1", "E@0"], "cpu", pre=pre)      # a map added later
    assert [c[0] for c in calls] == [clean, clean + 1.0]
    out = json.loads((gdir / "deletion.json").read_text())
    assert out["maps"] == ["E@0", "H@1"] and [c.get("map") for c in out["scores"]["gap"]["curves"]] == \
        ["H@1", "H@1", None, "E@0", "E@0"]
    with pytest.raises(SystemExit, match="move it away"):
        deletion.run_dir(model, gdir, settings | {"random": 2}, ["H@1"], "cpu", pre=pre)


def test_deletion_reads_the_first_builds_file_and_keeps_it(tapped, tmp_path, monkeypatch):
    """A file from before map specs (methods x layers, fill 0.0): read as fill grey and specs,
    its curves kept; the original is copied to deletion.v1.json before the first rewrite."""
    import json

    from qwen_eye import deletion

    model, batch = tapped["model"], tapped["batch"]
    clean = float(GapTarget().score(tapped["plain"]))
    clip, gdir = tmp_path / "clip", tmp_path / "clip" / "gradcam" / "phrases"
    gdir.mkdir(parents=True)
    (clip / "meta.json").write_text(json.dumps({"video_args": {}, "merge": 2}))
    (gdir / "meta.json").write_text(json.dumps({
        "model": "m", "video": "v.mp4", "clip_dir": str(clip), "layers": [0, 1], "item": None, "phrase_prompt": "",
        "scores": [{"name": "gap", "plain_value": clean}]}))
    rng = np.random.default_rng(0)
    np.savez(gdir / "gradcam.npz", H=rng.normal(size=(1, 2, T, H, W)), E=rng.normal(size=(1, 2, T, H, W)))
    monkeypatch.setattr(deletion, "target_for", lambda rec, meta: GapTarget())
    settings = deletion_settings(random=1)
    deletion.run_dir(model, gdir, settings, ["H@1"], "cpu", pre=FakePre(batch))
    new = json.loads((gdir / "deletion.json").read_text())
    old = {"gradcam_dir": str(gdir), "scores": {"gap": {**new["scores"]["gap"], "curves": [
               {k: v for k, v in c.items() if k != "map"} | ({"layer": int(c["layer"])} if "layer" in c else {})
               for c in new["scores"]["gap"]["curves"]], "summary": [{"method": "H", "layer": 1}]}},
           "settings": {"fractions": settings["fractions"], "random": 1, "methods": ["H"], "layers": [1], "fill": 0.0}}
    (gdir / "deletion.json").write_text(json.dumps(old))
    up = deletion.upgrade(old)
    assert up["settings"] == settings and up["maps"] == ["H@1"] and up["scores"]["gap"]["curves"] == \
        new["scores"]["gap"]["curves"] and up["scores"]["gap"]["summary"] == new["scores"]["gap"]["summary"]
    assert deletion.used_maps(old) == {"H"} and deletion.used_maps(new) == {"H"}
    assert deletion.run_dir(model, gdir, settings, ["H@1"], "cpu", pre=FakePre(batch)) == []
    assert not (gdir / "deletion.v1.json").exists()                   # nothing to do: nothing rewritten
    deletion.run_dir(model, gdir, settings, ["H@1", "H@0"], "cpu", pre=FakePre(batch))
    assert json.loads((gdir / "deletion.v1.json").read_text()) == old
    out = json.loads((gdir / "deletion.json").read_text())
    assert out["maps"] == ["H@0", "H@1"] and out["scores"]["gap"]["curves"][:3] == new["scores"]["gap"]["curves"]


def test_deletion_rebuilds_the_recorded_scores():
    from qwen_eye import deletion
    from qwen_eye.items import Item

    item = Item(key="location_motion:525", benchmark="motionbench", video="m.mp4", prompt="q", correct="C",
                wrong=None, eval_answer="D", eval_letter="D", options={"A": "a", "B": "b", "C": "c", "D": "d"})
    meta = {"item": item.to_dict(), "phrase_prompt": "Is there {phrase} in the video? Answer yes or no."}
    t = deletion.target_for({"name": "answer", "report": {"wrong": "B"}}, meta)
    assert t.item.correct == "C" and t.wrong == "B"
    p = deletion.target_for({"name": "phrase: a dog", "report": {}}, meta)
    assert p.text == "a dog" and p.question == "Is there a dog in the video? Answer yes or no."

