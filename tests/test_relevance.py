"""Maps D, C, W: relevance taps, the reducer against the papers' loops, explain() on autograd."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from helpers import T, W, N, LogitGap, TAP_LAYERS, FakePre, BigGap, rel_close, brute_force


def test_relevance_taps_leave_the_forward_and_the_feature_gradients_unchanged(tapped, dpass):
    assert torch.allclose(dpass["logits"], tapped["plain"], rtol=0, atol=1e-12)
    for L in TAP_LAYERS:
        assert torch.allclose(dpass["feats"][L][1], tapped["result"][L][1], rtol=1e-9, atol=1e-12)


def test_attention_output_gradient_is_kept_once_per_block(dpass, reference):
    """Checkpointing runs each block's forward twice; the first call's tensor gets the gradient."""
    assert dpass["calls"] == [2, 2, 2]
    for L, g in reference["dO"].items():
        assert g.abs().max() > 0 and torch.allclose(dpass["dO"][L], g, rtol=1e-9, atol=1e-12)
    assert dpass["dO"][0].device.type == "cpu"


def test_attention_gradient_is_output_gradient_times_values(dpass, reference):
    """d(score)/dA = dO V^T per head (O = A V), equal to autograd's gradient of the explicit
    softmax in every block and pair. dO's columns are heads x head_dim, its rows pairs."""
    att = reference["att"]
    assert sorted(att.probs) == [(L, p) for L in range(3) for p in range(T)]
    for (L, p), probs in att.probs.items():
        v = att.v[(L, p)][0]                                                    # [heads, n, d]
        heads, _, d = v.shape
        dO = dpass["dO"][L][p * N:(p + 1) * N].view(N, heads, d).transpose(0, 1)
        want = probs.grad[0]
        assert want.abs().max() > 0 and (dO @ v.transpose(1, 2) - want).abs().max() <= 1e-9 * want.abs().max()


def test_route_taps_hold_each_llm_token_and_its_gradient(tapped, dpass, reference):
    assert sorted(dpass["routes"]) == [1, 2]                                    # DeepStack after 1, final after 2
    for L, (y, g) in reference["routes"].items():
        assert dpass["routes"][L][0].shape == (T * N // 4, 32)                  # one LLM token per 2x2 block
        assert torch.allclose(dpass["routes"][L][0], y, rtol=0, atol=1e-12)
        assert g.abs().max() > 0 and torch.allclose(dpass["routes"][L][1], g, rtol=1e-9, atol=1e-12)
        assert (y * g).sum(-1).abs().min() > 0                                  # every token has a weight
    model, batch = tapped["model"], tapped["batch"]
    y, g = dpass["routes"][1]
    d = torch.randn_like(y)
    d /= d.norm()
    eps = 0.01            # 18 tokens: a unit nudge is large per entry; the error falls as eps^2 (0.1: 7%)

    def scored(sign):
        h = model.model.visual.deepstack_merger_list[0].register_forward_hook(lambda m, a, o: o + sign * eps * d)
        with torch.no_grad():
            s = float(LogitGap().score(model(**batch, use_cache=False, logits_to_keep=1).logits))
        h.remove()
        return s

    assert (scored(1) - scored(-1)) / (2 * eps) == pytest.approx(float((g * d).sum()), rel=1e-2)


def test_input_times_gradient_vanishes_at_every_merger_input(tapped, reference):
    """Why the readout weight is taken at the merger's output: each merger starts with a
    LayerNorm, which ignores its input's scale, so x . d(score)/dx through it is 0. The final
    merger normalises per token, so HiResCAM at the last block (whose only consumer it is) is 0
    per token; a DeepStack merger normalises a 2x2 block's 4 tokens together, so its x . g is 0
    per block."""
    f, g = tapped["result"][2]                                                  # block 2 feeds only the final merger
    hires = (f * g).sum(-1)
    assert hires.abs().max() < 1e-6 * (f.norm(dim=-1) * g.norm(dim=-1)).max()
    x, g = reference["merger_in"][1]                                            # the DeepStack merger alone
    per_token = (x * g).sum(-1)
    scale = (x.norm(dim=-1) * g.norm(dim=-1)).max()
    assert per_token.abs().max() > 1e-2 * scale and per_token.view(-1, 4).sum(-1).abs().max() < 1e-6 * scale


def test_relevance_taps_free_their_tensors(tapped):
    import weakref

    from qwen_eye.fullmodel import grad_mode
    from qwen_eye.grads import EmbedLeaf, FeatureTaps, OutputGradTaps, RouteTaps, backprop

    model, batch = tapped["model"], tapped["batch"]
    visual = model.model.visual
    taps, leaf, out, routes = FeatureTaps(visual, TAP_LAYERS, cut=False), EmbedLeaf(visual), OutputGradTaps(visual), \
        RouteTaps(visual)
    with grad_mode(model, -1):
        backprop(model, batch, LogitGap(), taps)
    refs = [weakref.ref(t) for t in (*out.grads.values(), *routes.feats.values(), *routes.grads.values())]
    assert len(refs) == 3 + 2 + 2
    for x in (taps, leaf, out, routes):
        x.remove()
    assert all(r() is None for r in refs)
    assert not visual.patch_embed._forward_hooks and not visual.merger._forward_pre_hooks
    assert not any(b.attn.proj._forward_pre_hooks for b in visual.blocks)


def test_relevance_reducer_is_chefers_loop_and_abnars_rollout():
    from qwen_eye.relevance import ViTRelevance

    g = torch.Generator().manual_seed(0)
    depth, pairs, n, heads, d = 3, 2, 6, 2, 3
    f64 = dict(generator=g, dtype=torch.float64)
    probs = torch.softmax(3 * torch.randn(depth, pairs, heads, n, n, **f64), -1).float()
    v = torch.randn(depth, pairs, heads, n, d, **f64).float()
    dO = {L: torch.randn(pairs * n, heads * d, **f64) for L in range(depth)}
    w = {1: torch.randn(pairs * n, **f64), 2: torch.randn(pairs * n, **f64)}
    red = ViTRelevance(dO, w, pairs, n)
    for L in range(depth):                                          # the capture's order: block by block
        for p in range(pairs):
            red.update(L, p, probs[L, p], v[L, p])
    res = red.result()
    assert res["parts"] == {"D": ["all", "r1", "r2", "norm"], "C": ["all"], "W": ["all"]}
    assert res["D"].shape == (2, 4, pairs, n) and res["C"].shape == res["W"].shape == (2, 1, pairs, n)
    for p in range(pairs):
        rows = slice(p * n, (p + 1) * n)
        grads = [dO[L][rows].view(n, heads, d).transpose(0, 1).float() @ v[L, p].transpose(1, 2) for L in range(depth)]
        want, routes = brute_force(list(probs[:, p]), grads, {L: x[rows] for L, x in w.items()})
        assert rel_close(res["D"][:, 0, p], want["D"]) and rel_close(res["D"][:, 3, p], want["norm"])
        assert rel_close(res["D"][:, 1, p], routes[1]) and rel_close(res["D"][:, 2, p], routes[2])
        assert rel_close(res["W"][:, 0, p], want["W"]) and rel_close(res["C"][:, 0, p], want["C"])
    D, W = res["D"][:, 0].sum((1, 2)), res["W"][:, 0].sum((1, 2))
    assert res["stats"]["mixing_share"] == pytest.approx((D / (D + W)).tolist(), rel=1e-6)
    assert np.allclose(res["stats"]["D_mass_per_pair"], res["D"][:, 0].sum(-1), rtol=1e-6)
    half = ViTRelevance(dO, w, pairs, n)
    half.update(0, 0, probs[0, 0], v[0, 0])
    with pytest.raises(RuntimeError, match="missing"):
        half.result()


def test_rownorm_keeps_zero_rows_zero():
    from qwen_eye.relevance import rownorm

    assert torch.equal(rownorm(torch.tensor([[1.0, 3.0], [0.0, 0.0]])), torch.tensor([[0.25, 0.75], [0.0, 0.0]]))


def test_explain_with_relevance_is_the_papers_maps_on_plain_autograd(tapped, reference):
    """explain's D, C, W from the taps and the capture pass = the brute force on the reference's
    autograd dA and merger-only route gradients (both linear in the score: x K); E, H as without
    them."""
    from qwen_eye.gradcam import explain

    model, pre, K = tapped["model"], FakePre(tapped["batch"]), BigGap.K
    res = explain(model, pre, BigGap(), TAP_LAYERS, "cpu")
    plain = explain(model, pre, BigGap(), TAP_LAYERS, "cpu", maps=("E", "H"))
    rec, checks = res["record"], res["record"]["checks"]
    assert rec["failed"] == [] and not {"D", "C", "W"} & set(plain) and "relevance" not in plain["record"]
    assert checks["taps_change_nothing"] < 1e-9 and checks["attn_grads_reach_all"] and checks["routes_reach_all"]
    assert checks["relevance_recon_gap"] < 1e-5 and checks["relevance_pass_score"] < 1e-9
    assert res["layout"]["D"] == {"name": "Chefer relevance", "sides": 2, "parts": ["all", "r1", "r2", "norm"]}
    assert res["layout"]["H"] == plain["layout"]["H"] == {"name": "HiResCAM", "sides": "signed", "parts": ["1", "2"]}
    assert res["D"].shape == (2, 4, T, N) and res["C"].shape == res["W"].shape == (2, 1, T, N)
    for k in ("E", "H"):
        assert np.allclose(res[k], plain[k], rtol=1e-9, atol=1e-12)
    att = reference["att"]
    w = {L: K * (y * g).sum(-1).repeat_interleave(4) / 4 for L, (y, g) in reference["routes"].items()}
    for p in range(T):
        want, _ = brute_force([att.probs[(L, p)][0].detach() for L in range(3)],
                               [K * att.probs[(L, p)].grad[0] for L in range(3)],
                               {L: x[p * N:(p + 1) * N] for L, x in w.items()})
        for key, got in (("D", res["D"][:, 0, p]), ("norm", res["D"][:, 3, p]), ("W", res["W"][:, 0, p]),
                         ("C", res["C"][:, 0, p])):
            assert rel_close(got, want[key]), key
    assert all(0 < m < 1 for m in rec["relevance"]["mixing_share"]) and set(rec["attn_grad_norms"]) == {"0", "1", "2"}

