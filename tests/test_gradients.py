"""Feature taps, Grad-CAM / HiResCAM (maps E/H), the scores, explain() and save()."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from qwen_eye import layout

from helpers import T, H, W, N, snapshot, LogitGap, TAP_LAYERS, FakePre, GapTarget


def test_removed_taps_free_their_tensors(tapped):
    import weakref

    from qwen_eye.fullmodel import grad_mode
    from qwen_eye.grads import FeatureTaps, backprop

    model, batch = tapped["model"], tapped["batch"]
    taps = FeatureTaps(model.model.visual, TAP_LAYERS)
    with grad_mode(model, taps.lowest):
        backprop(model, batch, LogitGap(), taps)
    refs = [weakref.ref(t) for L in TAP_LAYERS for t in (taps.feats[L], taps.grads[L])]
    taps.remove()
    assert all(r() is None for r in refs)                # freed by refcount alone, no gc pass needed


def test_grad_mode_restores_the_model(tapped):
    model = tapped["model"]
    assert not model.training
    assert not any(b.gradient_checkpointing for b in [*model.model.visual.blocks, *model.model.language_model.layers])
    assert not any(p.requires_grad for p in model.parameters())


def test_taps_and_checkpointing_leave_the_forward_unchanged(tapped):
    assert torch.allclose(tapped["logits"], tapped["plain"], rtol=0, atol=1e-12)


def test_tapped_gradients_match_plain_autograd(tapped):
    """Independent path: pixels require grad, block outputs retain_grad, no checkpointing."""
    model, batch = tapped["model"], tapped["batch"]
    outs = {}
    hooks = [model.model.visual.blocks[L].register_forward_hook(
        lambda m, a, o, L=L: (o.retain_grad(), outs.__setitem__(L, o))[0]) for L in TAP_LAYERS]
    pix = batch["pixel_values_videos"].clone().requires_grad_(True)
    score = LogitGap().score(model(**{**batch, "pixel_values_videos": pix}, use_cache=False, logits_to_keep=1).logits)
    score.backward()
    for h in hooks:
        h.remove()
    assert float(score.detach()) == pytest.approx(tapped["score"], abs=1e-12)
    for L in TAP_LAYERS:
        feats, grads = tapped["result"][L]
        assert torch.allclose(feats, outs[L].detach(), atol=1e-12)
        assert torch.allclose(grads, outs[L].grad, rtol=1e-9, atol=1e-12) and grads.abs().max() > 0


def test_tapped_gradient_predicts_a_nudge(tapped):
    """Finite differences: nudging layer L's output by eps*d changes the score by eps*<g, d>.
    eps is large because RMSNorm and the rotary tables compute in float32 even in a float64
    model: below ~1e-2 their rounding dominates the difference (1e-4 is off by up to 200%)."""
    model, batch = tapped["model"], tapped["batch"]
    eps = 0.1
    for L in TAP_LAYERS:
        feats, grads = tapped["result"][L]
        d = torch.randn_like(feats)
        d /= d.norm()

        def scored(sign):
            h = model.model.visual.blocks[L].register_forward_hook(lambda m, a, o: o + sign * eps * d)
            with torch.no_grad():
                s = float(LogitGap().score(model(**batch, use_cache=False, logits_to_keep=1).logits))
            h.remove()
            return s

        fd = (scored(1) - scored(-1)) / (2 * eps)
        assert fd == pytest.approx(float((grads * d).sum()), rel=1e-2)


def test_gradcam_is_alpha_weighted_features_per_pair():
    from qwen_eye.reducers import GradCAM

    g = torch.Generator().manual_seed(0)
    feats, grads = torch.randn(2, N, 8, generator=g), torch.randn(2, N, 8, generator=g)
    red = GradCAM()
    for p in range(2):
        red.update(4, p, feats[p], grads[p])
    E = red.result()["E"]
    assert E.shape == (1, 2, N)
    for p in range(2):
        want = sum(feats[p][:, k] * grads[p][:, k].mean() for k in range(8))
        assert np.allclose(E[0, p], want.numpy(), atol=1e-5)


def test_hirescam_is_each_patchs_own_gradient_times_its_features():
    from qwen_eye.reducers import HiResCAM

    g = torch.Generator().manual_seed(0)
    feats, grads = torch.randn(2, N, 8, generator=g), torch.randn(2, N, 8, generator=g)
    red = HiResCAM()
    for p in range(2):
        red.update(4, p, feats[p], grads[p])
    Hm = red.result()["H"]
    assert Hm.shape == (1, 2, N)
    for p in range(2):
        for i in (0, N - 1):
            assert Hm[0, p, i] == pytest.approx(float(sum(feats[p][i, k] * grads[p][i, k] for k in range(8))), abs=1e-5)


def test_letter_contrast_and_phrase_scores():
    from types import SimpleNamespace

    from qwen_eye.targets import LetterContrast, Phrase

    logits = torch.zeros(1, 1, 50)
    logits[0, -1, [7, 8, 9, 10]] = torch.tensor([1.0, 3.0, 2.0, -1.0])        # A B C D
    ids = {"A": 7, "B": 8, "C": 9, "D": 10}
    lc = LetterContrast(SimpleNamespace(correct="B", wrong="A"))
    lc.ids = ids
    lc.resolve(logits)                                            # a given wrong letter is kept
    assert float(lc.score(logits)) == pytest.approx(3.0 - 1.0)
    assert lc.maps == [("for correct (B)", 1), ("for wrong (A)", -1)]
    lc = LetterContrast(SimpleNamespace(correct="B", wrong=None))  # answered right, 4 options
    lc.ids = ids
    lc.resolve(logits)
    assert lc.wrong == "C" and float(lc.score(logits)) == pytest.approx(1.0)
    ph = Phrase("a car")
    assert ph.question == "Is there a car in the video? Answer yes or no."
    ph.ids = {"Yes": 11, "No": 12, "yes": 13, "no": 14}
    logits[0, -1, [11, 12, 13, 14]] = torch.tensor([-3.0, -1.0, 4.0, 1.5])   # the model says lowercase
    p = torch.softmax(logits[0, -1], -1)
    want = float(torch.log(p[11] + p[13]) - torch.log(p[12] + p[14]))       # log-odds, case ignored
    assert float(ph.score(logits)) == pytest.approx(want) and [s for _, s in ph.maps] == [1, -1]
    with pytest.raises(ValueError):
        Phrase("a car", "Describe the video.")


def test_answer_tokens_are_single_tokens():
    from transformers import AutoTokenizer

    from qwen_eye.targets import _single_token

    tok = AutoTokenizer.from_pretrained(snapshot("tokenizer.json"))

    for t in ("A", "B", "C", "D", "E", "F", "Yes", "No", "yes", "no"):
        _single_token(tok, t)


def test_explain_splits_gradcam_by_pair_and_passes_its_checks(tapped):
    from qwen_eye.gradcam import explain

    res = explain(tapped["model"], FakePre(tapped["batch"]), GapTarget(), TAP_LAYERS, "cpu")
    rec = res["record"]
    assert res["E"].shape == res["H"].shape == (len(TAP_LAYERS), T, N) and res["grid"] == (T, H, W)
    assert rec["failed"] == [] and rec["checks"]["taps_change_nothing"] < 1e-9 and rec["checks"]["grads_reach_all"]
    for li, L in enumerate(TAP_LAYERS):
        feats, grads = tapped["result"][L]
        for p in range(T):
            f, g = feats[p * N:(p + 1) * N], grads[p * N:(p + 1) * N]
            assert np.allclose(res["E"][li, p], (f @ g.mean(0)).numpy(), atol=1e-9)
            assert np.allclose(res["H"][li, p], (f * g).sum(-1).numpy(), atol=1e-9)


def test_save_adds_missing_maps_and_replaces_only_named_ones(tapped, tmp_path):
    import json

    from qwen_eye import gradcam
    from qwen_eye.targets import DEFAULT_PROMPT

    model, pre = tapped["model"], FakePre(tapped["batch"])
    kw = dict(item=None, video="v.mp4", model_dir="m", layers=TAP_LAYERS, clip_dir=tmp_path, prompt=DEFAULT_PROMPT)
    d = tmp_path / "g"

    def saved():
        with np.load(d / "gradcam.npz") as z:
            return json.loads((d / "meta.json").read_text()), {k: z[k] for k in z.files}

    gradcam.save(d, [gradcam.explain(model, pre, GapTarget(), TAP_LAYERS, "cpu", maps=("E", "H"))], maps=("E", "H"), **kw)
    meta, first = saved()
    assert list(meta["maps"]) == ["H", "E"] and set(first) == {"E", "H"} and meta["reproduced"] == {}
    meta["scores"][0]["grad_norms"] = {"1": ["first pass"]}                   # to see whose record fields survive
    (d / "meta.json").write_text(json.dumps(meta))
    np.savez_compressed(d / "gradcam.npz", **(first | {"H": 2 * first["H"]}))  # a saved H this run would not make
    full = [gradcam.explain(model, pre, GapTarget(), TAP_LAYERS, "cpu")]
    gradcam.save(d, full, **kw)                                                 # nothing named: add D, C, W only
    meta, arrays = saved()
    assert list(meta["maps"]) == ["H", "E", "D", "C", "W"] and meta["maps"]["D"]["parts"] == ["all", "r1", "r2", "norm"]
    assert np.array_equal(arrays["H"], 2 * first["H"]) and np.array_equal(arrays["E"], first["E"])   # kept as saved
    assert meta["reproduced"]["H"]["max_rel_diff"] == pytest.approx(0.5, rel=1e-6)
    assert meta["reproduced"]["E"]["max_rel_diff"] < 1e-6 and meta["reproduced"]["E"]["min_corr"] > 1 - 1e-9
    assert meta["scores"][0]["grad_norms"] == {"1": ["first pass"]}           # E/H kept: so are their fields
    assert "relevance" in meta["scores"][0] and "routes_reach_all" in meta["scores"][0]["checks"]
    assert arrays["D"].shape == (1, 2, 4, T, H, W) and arrays["C"].shape == (1, 2, 1, T, H, W)
    order = layout.block_order(H, W)
    assert np.allclose(arrays["D"][0, 1, 0], layout.to_grid(torch.from_numpy(full[0]["D"][1, 0]), order, H, W).numpy())
    settings = {"fractions": [1.0], "random": 0}
    (d / "deletion.json").write_text(json.dumps({"settings": settings | {"fill": "grey"}, "maps": ["H@1"], "scores": {}}))
    (d / "deletion_blur.json").write_text(json.dumps({"settings": settings | {"fill": "blur"}, "maps": ["D@all"],
                                                      "scores": {}}))
    gradcam.save(d, full, replace=("D",), **kw)                                 # D named: rewritten, its file stale
    assert sorted(p.name for p in d.glob("deletion*")) == ["deletion.json", "deletion_blur.stale.json"]
    gradcam.save(d, full, replace=("H",), **kw)
    meta, again = saved()
    assert np.allclose(again["H"], first["H"], rtol=1e-6, atol=1e-12) and "H" not in meta["reproduced"]
    assert meta["scores"][0]["grad_norms"] != {"1": ["first pass"]}           # H rewritten: its fields are new
    assert sorted(p.name for p in d.glob("deletion*")) == ["deletion.stale.json", "deletion_blur.stale.json"]
    other = [{**full[0], "record": {**full[0]["record"], "name": "other"}}]
    with pytest.raises(SystemExit, match="it holds scores"):
        gradcam.save(d, other, **kw)

