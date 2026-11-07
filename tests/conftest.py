"""Session fixtures: the tiny models' captured attention, tapped gradients and the
plain-autograd references, built once and shared by the test files."""

from __future__ import annotations

import pytest
import torch

from qwen_eye import layout
from qwen_eye.capture import AttentionCapture, attach
from qwen_eye.reducers import FromBlocks, MeanDistance, Received

from helpers import (
    T, H, W, N, FullProbs, tiny_tower, tiny_full_model, LogitGap, TAP_LAYERS, ExplicitAttention, mergers,
)


@pytest.fixture(scope="session")
def captured():
    tower = tiny_tower()
    pixels = torch.randn(T * N, 3 * 2 * 16 * 16)
    grid = torch.tensor([[T, H, W]])
    with torch.inference_mode():
        stock = tower(pixels, grid_thw=grid)
    order = layout.block_order(H, W)
    reducers = [Received(), FromBlocks([0, 2]), MeanDistance(layout.token_coords(order, W)), FullProbs()]
    cap = AttentionCapture(tower, reducers)
    attach(tower, cap)
    with torch.inference_mode():
        out = tower(pixels, grid_thw=grid)
    attach(tower, None)
    with torch.inference_mode():
        alone = tower(pixels[:N], grid_thw=torch.tensor([[1, H, W]]))
    return dict(tower=tower, stock=stock, out=out, alone=alone, cap=cap, reducers=reducers)


@pytest.fixture(scope="session")
def tapped():
    from qwen_eye.fullmodel import grad_mode
    from qwen_eye.grads import FeatureTaps, backprop

    model, batch = tiny_full_model()
    with torch.no_grad():
        plain = model(**batch, use_cache=False, logits_to_keep=1).logits
    taps = FeatureTaps(model.model.visual, TAP_LAYERS)
    with grad_mode(model, taps.lowest):
        assert all(b.gradient_checkpointing for b in model.model.visual.blocks[2:])
        score, logits = backprop(model, batch, LogitGap(), taps)
    result = taps.result()
    taps.remove()
    return dict(model=model, batch=batch, plain=plain, score=score, logits=logits, result=result)


@pytest.fixture(scope="session")
def reference(tapped):
    """What map D needs by plain autograd, without grads.py: no checkpointing, pixels require
    grad, the explicit softmax above, retain_grad on every attn.proj input and every merger's
    output (the LLM tokens of each route); and the gradient through each merger alone at its
    input, as the vjp of the merger."""
    import functools

    model, batch = tapped["model"], tapped["batch"]
    visual = model.model.visual
    att = ExplicitAttention(visual)
    proj, merged = {}, {}

    def keep_proj(layer, module, args):
        args[0].retain_grad()
        proj[layer] = args[0]

    def keep_merger(layer, module, args, output):
        output.retain_grad()
        merged[layer] = (module, args[0], output)

    hooks = [b.attn.proj.register_forward_pre_hook(functools.partial(keep_proj, i)) for i, b in enumerate(visual.blocks)]
    hooks += [m.register_forward_hook(functools.partial(keep_merger, L)) for L, m in mergers(visual)]
    pix = batch["pixel_values_videos"].clone().requires_grad_(True)
    attach(visual, att)
    try:
        score = LogitGap().score(model(**{**batch, "pixel_values_videos": pix}, use_cache=False, logits_to_keep=1).logits)
        score.backward()
    finally:
        attach(visual, None)
        for h in hooks:
            h.remove()
    routes = {L: (out.detach(), out.grad) for L, (m, x, out) in merged.items()}
    merger_in = {L: (x.detach(), torch.autograd.functional.vjp(m, x.detach(), out.grad)[1])
                 for L, (m, x, out) in merged.items()}
    return dict(att=att, dO={L: x.grad for L, x in proj.items()}, routes=routes, merger_in=merger_in,
                score=float(score.detach()))


@pytest.fixture(scope="session")
def dpass(tapped):
    """The backward as gradcam.explain takes it for map D: every block checkpointed, the leaf at
    the patch embedding, attention-output and route taps next to the feature taps."""
    from qwen_eye.fullmodel import grad_mode
    from qwen_eye.grads import EmbedLeaf, FeatureTaps, OutputGradTaps, RouteTaps, backprop

    model, batch = tapped["model"], tapped["batch"]
    visual = model.model.visual
    taps = FeatureTaps(visual, TAP_LAYERS, cut=False)
    extra = EmbedLeaf(visual), OutputGradTaps(visual), RouteTaps(visual)
    try:
        with grad_mode(model, -1):
            assert all(b.gradient_checkpointing for b in visual.blocks)
            score, logits = backprop(model, batch, LogitGap(), taps)
        return dict(score=score, logits=logits, feats=taps.result(), dO=extra[1].result(), calls=list(extra[1].calls),
                    routes=extra[2].result())
    finally:
        taps.remove()
        for x in extra:
            x.remove()

