"""The full Qwen3-VL model for gradient maps: loaded as lmms-eval's qwen3_vl backend loads it,
weights frozen.

`grad_mode` is the memory setup for one forward + backward of the 8B model on a 40 GB A100.
Only activations get gradients (every weight is frozen). Activation checkpointing
(non-reentrant) runs on every LLM decoder layer and on the ViT blocks above the lowest tapped
layer; blocks at or below it build no graph (grads.FeatureTaps). Map D needs the graph through
every block: lowest_layer = -1 checkpoints all of them (grads.EmbedLeaf). transformers
checkpoints only in training mode, and in 4.57.6 training mode changes nothing else in Qwen3-VL
but attention dropout, so `grad_mode` asserts every dropout is 0 first.
"""

from __future__ import annotations

import functools
import re
from contextlib import contextmanager
from pathlib import Path

import torch


def load_model(model_dir: str | Path, device: str = "cuda", dtype: torch.dtype = torch.bfloat16):
    """As lmms-eval's qwen3_vl backend loads it: the MoE class for an "...A3B"-style name (its
    rule), else the dense one; bf16, sdpa."""
    from transformers import Qwen3VLForConditionalGeneration, Qwen3VLMoeForConditionalGeneration

    model_cls = Qwen3VLMoeForConditionalGeneration if re.search(r"A\d+B", str(model_dir)) \
        else Qwen3VLForConditionalGeneration
    model = model_cls.from_pretrained(model_dir, dtype=dtype, attn_implementation="sdpa", device_map=device).eval()
    return model.requires_grad_(False)


def active_dropout(model) -> list[str]:
    bad = [n for n, m in model.named_modules() if isinstance(m, torch.nn.Dropout) and m.p > 0]
    return bad + [n for n, m in model.named_modules() if getattr(m, "attention_dropout", 0.0) != 0.0]


@contextmanager
def grad_mode(model, lowest_layer: int):
    bad = active_dropout(model)
    if bad:
        raise RuntimeError(f"dropout active in {bad[:3]}: train() for checkpointing would change the forward")
    visual, decoder = model.model.visual, model.model.language_model.layers
    checkpointed = [*visual.blocks[lowest_layer + 1:], *decoder]
    func = functools.partial(torch.utils.checkpoint.checkpoint, use_reentrant=False)
    was_training = model.training
    for m in checkpointed:
        m.gradient_checkpointing, m._gradient_checkpointing_func = True, func
    model.train()
    try:
        with torch.enable_grad():
            yield
    finally:
        for m in checkpointed:
            m.gradient_checkpointing = False
        model.train(was_training)
