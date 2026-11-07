# Qwen-eye

Where Qwen3-VL's vision tower looks in a video, and which patches its answers use.

Qwen-eye draws maps over exactly the frames Qwen3-VL gets: the ViT's own attention, gradient
maps for a question (Grad-CAM, HiResCAM), relevance carried back to the input patches
(Chefer et al.), and a deletion test that checks whether a map points at what the model
relies on. Every clip gets a self-contained `viewer.html` (no network needed) and an mp4.

| map | what it shows | runs on |
|---|---|---|
| **A** received | attention each patch receives, per block and head | CPU or GPU (vision tower only) |
| **B** from a block | where one 2×2 block (one LLM token) looks | CPU or GPU |
| **H** HiResCAM, **E** Grad-CAM | which patches' features pushed a score up or down | GPU (full model + backward) |
| **D** Chefer relevance, **C** rollout, **W** readout | the score carried back to the input patches | GPU |

A *score* is either an answer contrast for a multiple-choice question (logit of the correct
letter minus a wrong one) or a yes/no phrase question ("Is there a dog in the video?"). The
input is built exactly as [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval)'s
`qwen3_vl` backend builds it for an eval, so the maps belong to the same input the benchmark
numbers came from.

## Install

```bash
git clone https://github.com/rizeenmusadhik/Qwen-eye.git
cd Qwen-eye
pip install -e .            # or: pip install git+https://github.com/rizeenmusadhik/Qwen-eye.git
```

Python ≥ 3.10. transformers is pinned to 4.57.6: the attention capture and the gradient
taps hook into Qwen3-VL's modules as written there. Install a PyTorch build for your GPU first
if pip would pick the wrong one.

## Quickstart

`--model` is a local Qwen3-VL snapshot dir or a Hugging Face id (downloaded on first use);
set `QWEN_EYE_MODEL` to leave it out.

```bash
export QWEN_EYE_MODEL=Qwen/Qwen3-VL-8B-Instruct

# ViT attention (maps A, B): CPU works, a few minutes per clip
qwen-eye attn --video clip.mp4 --out out
# -> out/clip/viewer.html, attention.mp4

# your own multiple-choice question (maps E, H, D, C, W; GPU)
qwen-eye gradcam --video clip.mp4 --out out \
    --question $'What does the person do?\nA. Opens the door.\nB. Closes the door.\nAnswer with the option\'s letter.' \
    --correct B --phrase "a door"
# -> out/clip/gradcam/question/viewer.html, gradcam.mp4

# phrase questions only
qwen-eye gradcam --video clip.mp4 --phrase "a dog" --phrase "a ball" --out out
# -> out/clip/gradcam/phrases/

# deletion test: blank each map's top blocks, rerun, compare with random blocks (GPU)
qwen-eye deletion out/clip/gradcam/question --maps H@8 H@16 D@all W@all C@all

# redraw a viewer / mp4 from saved maps, no model (e.g. other B points)
qwen-eye render out/clip --point 0.3,0.6 --point 0.7,0.4
```

A second question on the same clip needs its own `--tag NAME` (its dir becomes
`gradcam/NAME`). Video sampling follows the lmms-eval backend: `--fps` (default 2), `--max-frames` (default 32),
`--min-pixels`, `--max-pixels`, `--total-pixels`. `--dry-run` checks the inputs and prints what
would run. `qwen-eye <command> --help` lists everything.

**Benchmark items.** For a VITATECS or MotionBench item scored by lmms-eval, `--item
direction:13` (or `location_motion:525`) with `--samples <dir of the run's *_samples_*.jsonl>`
and `--data-root <the run's HF_HOME>` reads the exact prompt the eval sent and the model's
recorded answer from the run's sample files, and checks that our forward gives the same answer.

**Cluster jobs.** `--save-plan DIR` writes `DIR/plan.json` instead of running; a job then runs
`python -m qwen_eye run DIR`. Host tools can build the same plans with `qwen_eye.plans`.

### Python

```python
from huggingface_hub import snapshot_download
from qwen_eye import plans

model = snapshot_download("Qwen/Qwen3-VL-8B-Instruct")    # plans take a local snapshot dir
video_args = {"fps": 2.0, "max_num_frames": 32}
plans.run(plans.attn_plan(model=model, videos=["clip.mp4"], out_root="out", variant_args=video_args))
```

[examples/quickstart.py](examples/quickstart.py) runs maps A/B and a question, then reads the
saved arrays.

## What you get

```
out/<clip>/viewer.html      maps A and B: pick block, head, layer, frame; click a block for B
          attention.mp4     frame | one column per layer; rows A, then B
          maps.npz          A [27, pairs, heads, h, w], B, the frames the model saw
          gradcam/<tag>/    viewer.html (A, B + gradient maps), gradcam.mp4, gradcam.npz, meta.json,
                            deletion.json
```

Formats: [docs/outputs.md](docs/outputs.md). Methods, checks and tests:
[docs/methods.md](docs/methods.md). The deletion test: [docs/deletion.md](docs/deletion.md).

## How it stays faithful to the model

- The maps are drawn on frames rebuilt from the model's own `pixel_values_videos`, so
  resolution, crop and frame pairing are exact.
- The attention capture returns the stock sdpa output (the model's output is unchanged) and
  recomputes the softmax in fp32 for the maps; every run checks the reconstruction and stops
  if it is off.
- The gradient passes check that the taps and activation checkpointing leave the score
  unchanged, and that the gradient reaches every frame pair.
- The tests run on tiny random Qwen3-VL models in float64 and compare every map with plain
  autograd or the papers' formulas written out (no downloads; `pytest`).

## Hardware

- `attn` loads only the vision tower (the `model.visual.*` weights): CPU in fp32 takes a few
  minutes for a 32-frame clip; on a GPU it runs in bf16.
- `gradcam` and `deletion` run the full model in bf16 on a GPU. With activation
  checkpointing, a 32-frame HD clip (~10k tokens) of the 8B model fits an A100 40 GB (peak
  ~28 GB), at about 7 s per score for E/H and about 20 s with D, C, W. A deletion forward takes
  0.2–1.5 s.

## Limits

- Qwen3-VL only, checked on Qwen3-VL-8B-Instruct; other sizes and the MoE models load the same
  way but are untested.
- Videos only, one clip per prompt, batch 1.
- The ViT attends within each frame pair, so its maps never relate pairs; how the score
  depends on time comes from the readout (which pairs light up), not from attention between
  frames.
- Grey-block deletion is an off-distribution perturbation; compare against the random
  baseline, as the tool does.

## Tests

```bash
pip install -e ".[test]"
pytest
```

`QWEN_EYE_MODEL=<local snapshot dir>` also runs the checks against the real processor and
tokenizer; `pip install -e ".[lmms]"` adds the check that the preprocessing equals lmms-eval's.

## References

- Grad-CAM: R. R. Selvaraju et al., ICCV 2017.
- HiResCAM: R. L. Draelos, L. Carin, arXiv:2011.08891.
- Relevance: H. Chefer, S. Gur, L. Wolf, "Generic attention-model explainability for
  interpreting bi-modal and encoder-decoder transformers", ICCV 2021.
- Rollout: S. Abnar, W. Zuidema, "Quantifying attention flow in transformers", ACL 2020.
- Deletion: V. Petsiuk, A. Das, K. Saenko, "RISE", BMVC 2018.
- Qwen3-VL: Qwen Team, Qwen3-VL technical report, 2025.

## License

MIT, see [LICENSE](LICENSE).
