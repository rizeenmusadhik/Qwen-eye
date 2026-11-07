"""Quantisation, viewer.html payloads and the mp4 rows (no model)."""

from __future__ import annotations

import numpy as np
import pytest

from helpers import H, W, N, fake_clip, fake_gradcam, D_PARTS


def test_quant_round_trip_is_within_one_log_step():
    from qwen_eye import quant

    x = np.concatenate([np.random.default_rng(0).gamma(1.0, 1.0, 5000), [0.0, 90.0, 120.0]]).astype(np.float32)
    xmax = float(x.max())
    back = quant.decode(quant.encode(x, xmax), xmax)
    step = np.log1p(xmax) / 255
    assert np.abs(np.log1p(back) - np.log1p(x)).max() <= step / 2 + 1e-6


def test_quant_linear_round_trip():
    from qwen_eye import quant

    x = np.random.default_rng(0).uniform(0, 3e-4, 1000).astype(np.float32)
    back = quant.decode(quant.encode(x, x.max(), log=False), x.max(), log=False)
    assert np.abs(back - x).max() <= x.max() / 255 / 2 + 1e-12


def test_viewer_embeds_a_decodable_payload(tmp_path):
    import base64
    import json
    import re

    from qwen_eye import quant, viewer

    maps, meta, stats = fake_clip()
    path = viewer.write(tmp_path / "viewer.html", maps, meta, stats)
    html = path.read_text()
    assert viewer.MARK not in html and "<script>" in html
    p = json.loads(re.search(r"const P = (.*?);\nconst M = P\.meta", html, re.S).group(1))
    assert len(p["frames"]) == 4 and p["meta"]["padded"] == [False, False, False, True]
    assert p["meta"]["frame_times"] == [0.0, 0.5, 1.0, 1.0]
    a = np.frombuffer(base64.b64decode(p["A"]["b64"]), np.uint8).reshape(p["A"]["shape"])
    assert a.shape == (3, 2, 3, H, W)                                  # head 0 = average of the 2 heads
    got = quant.decode(a[1, 0, 0], p["A"]["xmax"][1])
    want = maps["A"][1, 0].astype(np.float32).mean(0)
    assert np.allclose(got, want, rtol=0.02, atol=0.02)
    b = np.frombuffer(base64.b64decode(p["B"]["b64"]), np.uint8).reshape(p["B"]["shape"])
    assert b.shape == (2, 2, N // 4, H, W) and len(p["A"]["p99"][0]) == 3 and len(p["lut"]) == 256


def test_gradcam_viewer_payload_has_both_sides_of_a_score_on_one_scale(tmp_path):
    import base64
    import json
    import re

    from qwen_eye import quant, viewer

    maps, meta, stats = fake_clip()
    gmaps, gm = fake_gradcam()
    html = viewer.write(tmp_path / "v.html", maps, meta, stats, gradcam={"maps": gmaps, "meta": gm}).read_text()
    p = json.loads(re.search(r"const P = (.*?);\nconst M = P\.meta", html, re.S).group(1))
    pe = p["E"]
    assert [m["name"] for m in pe["methods"]] == ["HiResCAM", "Grad-CAM"]
    assert pe["labels"] == ["for correct (B)", "for wrong (A)", '"a car": for yes', '"a car": for no']
    assert pe["group"] == [0, 0, 1, 1] and pe["dropped"] == []
    for m, key in zip(pe["methods"], ("H", "E")):
        assert m["key"] == key and m["parts"] == ["0", "2"] and m["shape"] == [4, 2, 2, H, W]
        assert m["xmax"][0] == m["xmax"][1] and m["p99"][2] == m["p99"][3]     # one scale per score and part
        q = np.frombuffer(base64.b64decode(m["b64"]), np.uint8).reshape(m["shape"])
        wrong = quant.decode(q[1, 1], m["xmax"][1][1], log=False)
        assert np.allclose(wrong, np.maximum(-gmaps[key][0, 1], 0), atol=m["xmax"][1][1] / 255)
    assert "grad_norms" not in p["G"]["scores"][0]
    viewer.MAX_B_MB, keep = 0, viewer.MAX_B_MB                                  # B over budget: left out
    try:
        small = viewer.payload(maps, meta, stats, {"maps": gmaps, "meta": gm})
    finally:
        viewer.MAX_B_MB = keep
    assert small["B"] is None and small["E"]["methods"] == pe["methods"]
    old = json.loads(re.search(r"const P = (.*?);\nconst M = P\.meta", viewer.write(       # a first-build dir: E only
        tmp_path / "o.html", maps, meta, stats, gradcam={"maps": {"E": gmaps["E"]}, "meta": gm}).read_text(), re.S).group(1))
    assert [m["name"] for m in old["E"]["methods"]] == ["Grad-CAM"]
    plain = json.loads(re.search(r"const P = (.*?);\nconst M = P\.meta",
                                 viewer.write(tmp_path / "a.html", maps, meta, stats).read_text(), re.S).group(1))
    assert "E" not in plain and "G" not in plain


def test_gradcam_viewer_payload_carries_two_sided_maps_and_drops_routes_over_budget():
    import base64

    from qwen_eye import quant, viewer

    maps, meta, stats = fake_clip()
    gmaps, gm = fake_gradcam(relevance=True)
    pe = viewer.payload(maps, meta, stats, {"maps": gmaps, "meta": gm})["E"]
    assert [m["key"] for m in pe["methods"]] == ["H", "E", "D"]
    d = pe["methods"][2]
    assert d["name"] == "Chefer relevance" and d["parts"] == D_PARTS and d["shape"] == [4, 3, 2, H, W]
    assert d["xmax"][2] == d["xmax"][3]                                         # the phrase's yes and no
    q = np.frombuffer(base64.b64decode(d["b64"]), np.uint8).reshape(d["shape"])
    against = quant.decode(q[1, 0], d["xmax"][1][0], log=False)                 # answer, side 1, "all"
    assert np.allclose(against, gmaps["D"][0, 1, 0], atol=d["xmax"][1][0] / 255)
    viewer.MAX_E_MB, keep = 0, viewer.MAX_E_MB
    try:
        small = viewer.payload(maps, meta, stats, {"maps": gmaps, "meta": gm})["E"]
    finally:
        viewer.MAX_E_MB = keep
    assert small["dropped"] == ["D@r1"] and small["methods"][2]["parts"] == ["all", "norm"]
    assert small["methods"][0]["parts"] == ["0", "2"]                           # layers are never dropped


def test_mp4_rows_answer_in_both_methods_then_phrases_yes_side():
    from qwen_eye.render import mp4_rows

    gmaps, gm = fake_gradcam()
    rows = mp4_rows(gmaps, gm)
    assert [r["label"] for r in rows] == ["HiResCAM for correct (B)", "HiResCAM for wrong (A)",
                                          "Grad-CAM for correct (B)", "Grad-CAM for wrong (A)",
                                          'HiResCAM "a car": for yes']
    assert [c[2] for c in rows[4]["cells"]] == ["L0", "L2"]
    both = np.stack([np.maximum(gmaps["H"][1], 0), np.maximum(-gmaps["H"][1], 0)])   # phrase, both sides
    assert rows[4]["cells"][1][1] == pytest.approx(np.percentile(both[:, 1], 99))   # scale shared with "no"


def test_mp4_rows_with_relevance_show_the_answer_in_h_and_d_and_phrases_in_d():
    from qwen_eye.render import mp4_rows

    gmaps, gm = fake_gradcam(relevance=True)
    rows = mp4_rows(gmaps, gm)
    assert [r["label"] for r in rows] == ["HiResCAM for correct (B)", "HiResCAM for wrong (A)",
                                          "Chefer relevance for correct (B)", "Chefer relevance for wrong (A)",
                                          'Chefer relevance "a car": for yes']
    assert [c[2] for c in rows[4]["cells"]] == ["all routes", "route 1"]       # norm stays in the viewer
    assert np.array_equal(rows[4]["cells"][0][0], gmaps["D"][1, 0, 0])
    assert rows[4]["cells"][0][1] == pytest.approx(np.percentile(gmaps["D"][1, :, 0], 99))

