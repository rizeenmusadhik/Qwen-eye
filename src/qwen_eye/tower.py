"""The Qwen3-VL vision tower on its own, and the model's input built exactly as lmms-eval's
chat `qwen3_vl` backend builds it for an eval.

Only the `model.visual.*` tensors are read (0.6B of the 8B checkpoint, one shard), so a clip
runs on CPU. EvalPreprocessor repeats the backend's steps without importing lmms-eval:
_build_video_kwargs, the ChatMessages -> HF messages conversion, the chat template with the
generation prompt, qwen_vl_utils.process_vision_info and the processor call (lmms-eval
0.7.2, lmms_eval/models/simple/qwen3_vl.py + chat/qwen3_vl.py; tests/test_tower.py checks the
two give the same tensors where lmms-eval is installed). fps and the frame cap come from the
caller's video args. It also builds the full model's input for any prompt (gradcam.py).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoConfig
from transformers.modeling_utils import no_init_weights
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

# Args that change what the vision tower gets (lmms-eval model args of the qwen3_vl backend).
VIDEO_ARGS = ("min_pixels", "max_pixels", "total_pixels", "max_num_frames", "fps")
# Hard-coded in the backend's process_vision_info call.
IMAGE_PATCH_SIZE = 16
# The backend's defaults for the args that shape its input (Qwen3_VL.__init__).
BACKEND_DEFAULTS = {"min_pixels": 256 * 28 * 28, "max_pixels": 1605632, "total_pixels": None, "max_num_frames": 32,
                    "fps": None, "system_prompt": "You are a helpful assistant.", "enable_thinking": None}


@dataclass
class VideoInput:
    pixels: torch.Tensor           # pixel_values_videos [t*h*w, C*T*P*P], token order
    grid: tuple[int, int, int]     # (frame pairs, patch rows, patch cols)
    frame_times: list[float]       # seconds, every sampled frame
    pair_labels: list[str]         # the "<x seconds>" text the LLM reads before each pair
    patch: int
    tpatch: int
    merge: int
    mean: list[float]
    std: list[float]
    video_args: dict               # backend settings used


def load_vision_tower(model_dir: str | Path, device: str = "cpu", dtype: torch.dtype = torch.float32):
    model_dir = Path(model_dir)
    cfg = AutoConfig.from_pretrained(model_dir).vision_config
    cfg._attn_implementation = "sdpa"
    with no_init_weights():
        tower = Qwen3VLVisionModel._from_config(cfg)
    index = model_dir / "model.safetensors.index.json"
    if index.exists():
        weight_map = json.loads(index.read_text())["weight_map"]
    else:
        with safe_open(model_dir / "model.safetensors", framework="pt") as f:
            weight_map = {k: "model.safetensors" for k in f.keys()}
    prefix = "model.visual." if any(k.startswith("model.visual.") for k in weight_map) else "visual."
    state = {}
    for shard in sorted({f for k, f in weight_map.items() if k.startswith(prefix)}):
        with safe_open(model_dir / shard, framework="pt") as f:
            state.update({k[len(prefix):]: f.get_tensor(k) for k in f.keys() if k.startswith(prefix)})
    tower.load_state_dict(state, strict=True)
    return tower.to(device=device, dtype=dtype).eval()


def backend_defaults() -> dict:
    """The chat qwen3_vl backend's own defaults for the args that shape its input."""
    return dict(BACKEND_DEFAULTS)


def _meta(md, key):
    return md[key] if isinstance(md, dict) else getattr(md, key)


def video_kwargs(args: dict) -> dict:
    """The backend's _build_video_kwargs: what process_vision_info gets per video."""
    out = {"min_pixels": args["min_pixels"]}
    if args["fps"] is not None:
        out |= {"fps": args["fps"], "max_frames": args["max_num_frames"]}
    elif args["total_pixels"] is not None:
        out["max_frames"] = args["max_num_frames"]
    else:
        out["nframes"] = args["max_num_frames"]
    if args["total_pixels"] is not None:
        out["total_pixels"] = args["total_pixels"]
    else:
        out["max_pixels"] = args["max_pixels"]
    return out


class Backend:
    """The backend's chat state, under lmms-eval's names (callers that build their own turns
    use them): the system prompt and the chat template call."""

    def __init__(self, processor, system_prompt: str, enable_thinking):
        self.processor, self.system_prompt, self.enable_thinking = processor, system_prompt, enable_thinking

    def _apply_chat_template(self, batched_messages, **kwargs):
        template_kwargs = {} if self.enable_thinking is None else {"enable_thinking": self.enable_thinking}
        return self.processor.apply_chat_template(batched_messages, tokenize=False, add_generation_prompt=True,
                                                  **template_kwargs, **kwargs)


class EvalPreprocessor:
    """One clip through the backend's steps; `encode(text)` gives the processor output for that
    prompt (system + [video, text] user turn + generation prompt), as generate_until builds it.
    The video is decoded once: its frames do not depend on the text."""

    def __init__(self, model_dir: str | Path, video: str, variant: dict):
        from qwen_vl_utils import process_vision_info
        from transformers import AutoProcessor

        self.args = {**backend_defaults(), **{k: v for k, v in variant.items() if k in VIDEO_ARGS}}
        self.processor = AutoProcessor.from_pretrained(model_dir, max_pixels=self.args["max_pixels"],
                                                       min_pixels=self.args["min_pixels"])
        self.backend = Backend(self.processor, self.args["system_prompt"], self.args["enable_thinking"])
        self.video = str(video)
        self.video_kwargs = video_kwargs(self.args)
        _, videos, kw_qwen = process_vision_info([self.messages("?")], return_video_kwargs=True,
                                                 image_patch_size=IMAGE_PATCH_SIZE, return_video_metadata=True)
        self.videos, self.metas = map(list, zip(*videos))
        self.processor_kwargs = {**self.video_kwargs, **kw_qwen}

    def messages(self, text: str) -> list:
        """The HF chat messages of lmms-eval's ChatMessages.to_hf_messages for this prompt."""
        return [{"role": "system", "content": [{"type": "text", "text": self.backend.system_prompt}]},
                {"role": "user", "content": [{"type": "video", "video": self.video, **self.video_kwargs},
                                             {"type": "text", "text": text}]}]

    def prompt(self, text: str) -> str:
        """The chat-templated prompt string, ending with the assistant generation prompt."""
        return self.backend._apply_chat_template([self.messages(text)])[0]

    def encode(self, text: str):
        """Processor output for the prompt `text`, as the backend's generate_until builds it."""
        return self.processor(text=[self.prompt(text)], videos=self.videos, video_metadata=self.metas,
                              **self.processor_kwargs, do_resize=False, return_tensors="pt")

    def video_input(self, out) -> VideoInput:
        idx, src_fps = list(_meta(self.metas[0], "frames_indices")), _meta(self.metas[0], "fps")
        vp = self.processor.video_processor
        return VideoInput(
            pixels=out["pixel_values_videos"],
            grid=tuple(int(x) for x in out["video_grid_thw"][0]),
            frame_times=[float(i) / src_fps for i in idx],
            pair_labels=re.findall(r"<([\d.]+) seconds>", self.processor.batch_decode(out["input_ids"])[0]),
            patch=vp.patch_size, tpatch=vp.temporal_patch_size, merge=vp.merge_size,
            mean=list(vp.image_mean), std=list(vp.image_std),
            video_args={k: self.args[k] for k in VIDEO_ARGS},
        )


def preprocess_video(model_dir: str | Path, video: str, variant: dict) -> VideoInput:
    """The vision tower's input for a clip (the prompt text does not change it)."""
    pre = EvalPreprocessor(model_dir, video, variant)
    return pre.video_input(pre.encode("?"))
