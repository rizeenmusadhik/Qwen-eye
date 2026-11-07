"""tower.EvalPreprocessor = lmms-eval's chat qwen3_vl backend, step by step. Runs where lmms-eval
is installed (`pip install qwen-eye[lmms]`); the full comparison also needs $QWEN_EYE_MODEL."""

from __future__ import annotations

import inspect

import numpy as np
import pytest
import torch

from qwen_eye import tower

from helpers import snapshot

VARIANTS = [{}, {"fps": 2.0, "max_num_frames": 32}, {"fps": 1.0, "max_num_frames": 8, "max_pixels": 200704},
            {"total_pixels": 2_000_000, "max_num_frames": 16}, {"fps": 2.0, "total_pixels": 2_000_000}]


def _lmms():
    return pytest.importorskip("lmms_eval.models.chat.qwen3_vl", reason="lmms-eval not installed")


def _backend(lm, args, processor=None):
    """The lmms-eval backend object without its weights: preprocessing reads only these fields."""
    m = object.__new__(lm.Qwen3_VL)
    for k, v in args.items():
        setattr(m, k, v)
    m.processor = processor
    return m


def test_backend_defaults_are_lmms_evals():
    lm = _lmms()
    params = inspect.signature(lm.Qwen3_VL.__init__).parameters
    assert tower.backend_defaults() == {k: params[k].default for k in tower.BACKEND_DEFAULTS}


@pytest.mark.parametrize("variant", VARIANTS)
def test_video_kwargs_are_lmms_evals(variant):
    lm = _lmms()
    args = {**tower.backend_defaults(), **variant}
    assert tower.video_kwargs(args) == _backend(lm, args)._build_video_kwargs()


def _clip(path, frames=12, fps=4, size=(96, 160)):
    import av

    rng = np.random.default_rng(0)
    with av.open(str(path), "w") as out:
        s = out.add_stream("mpeg4", rate=fps)
        s.height, s.width, s.pix_fmt = *size, "yuv420p"
        for _ in range(frames):
            img = rng.integers(0, 256, (*size, 3), dtype=np.uint8)
            out.mux(s.encode(av.VideoFrame.from_ndarray(img, format="rgb24")))
        out.mux(s.encode())
    return path


@pytest.mark.parametrize("variant", VARIANTS[1:3])
def test_eval_preprocessor_gives_lmms_evals_tensors(tmp_path, variant):
    lm = _lmms()
    from lmms_eval.protocol import ChatMessages
    from transformers import AutoProcessor

    model = snapshot()
    clip = str(_clip(tmp_path / "c.mp4"))
    pre = tower.EvalPreprocessor(model, clip, variant)
    args = {**tower.backend_defaults(), **variant}
    m = _backend(lm, args, AutoProcessor.from_pretrained(model, max_pixels=args["max_pixels"],
                                                         min_pixels=args["min_pixels"]))
    vk = m._build_video_kwargs()
    text = "Is there a car in the video? Answer yes or no."
    msgs = ChatMessages(messages=[
        {"role": "system", "content": [{"type": "text", "text": m.system_prompt}]},
        {"role": "user", "content": [{"type": "video", "url": clip}, {"type": "text", "text": text}]},
    ]).to_hf_messages(video_kwargs=vk)
    assert pre.messages(text) == msgs
    prompt = m._apply_chat_template([msgs])[0]
    assert pre.prompt(text) == prompt
    _, videos, kw = lm.process_vision_info([msgs], return_video_kwargs=True, image_patch_size=tower.IMAGE_PATCH_SIZE,
                                           return_video_metadata=True)
    vids, metas = map(list, zip(*videos))
    want = m.processor(text=[prompt], videos=vids, video_metadata=metas, **vk, **kw, do_resize=False,
                       return_tensors="pt")
    got = pre.encode(text)
    assert set(got) == set(want)
    for k in want:
        assert torch.equal(got[k], want[k]), k
