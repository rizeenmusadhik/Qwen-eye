"""Qwen3-VL vision-tower (ViT) spatial attention, drawn over the frames the model sees.

    qwen-eye attn --model DIR --video clip.mp4 --out OUT        # viewer.html + attention.mp4 per clip
    qwen-eye render OUT/clip --point 0.3,0.6                    # redraw from the saved maps, no model
    qwen-eye gradcam --model DIR --video clip.mp4 --phrase "a dog" --out OUT   # E, H, D, C, W; GPU
    qwen-eye deletion OUT/clip/gradcam/phrases --maps H@8 D@all  # do the maps point at what matters?
    python -m qwen_eye run RUN_DIR                              # a saved plan.json (e.g. on a cluster)

Maps (docs/methods.md):
  A  received:   attention each patch receives, averaged over all query patches
  B  from block: where one 2x2 patch block (one LLM token) looks
  E  gradients:  which patches' content pushed a score (answer contrast, or a yes/no phrase
                 question) up: HiResCAM (H) and Grad-CAM (E)
  D  relevance:  the score carried back to the input patches through every block's attention
                 (Chefer); C the same by rollout, W the readout alone

Modules: layout (token order, frames), capture (attention without changing the output),
reducers (one per map: the extension point), tower (weights + the model's input, built as
lmms-eval's qwen3_vl backend builds it), run (one clip end to end), quant/viewer/video/render
(drawing). Map E: items (questions: your own, or VITATECS / MotionBench items from lmms-eval
samples), fullmodel (the full model, memory setup), targets (the scores),
grads (gradient taps), relevance (maps D, C, W), maps (the saved maps' layouts), gradcam (the
gradient-map job), deletion (blank a map's top blocks in the input, rerun, compare with random
blocks). plans builds the plan.json every job runs from; cli is the `qwen-eye` command.
"""

__version__ = "0.1.0"
