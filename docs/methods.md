# Methods

Qwen-eye maps what Qwen3-VL's vision tower (the ViT, SigLIP-2 based) does with a video, drawn
on exactly the frames the model gets. Block numbers below are Qwen3-VL-8B's (27 blocks, 16
heads, 16 px patches, DeepStack mergers after blocks 8, 16 and 24), the model the tool was
built and checked on; other Qwen3-VL checkpoints load the same way with their own depth and
DeepStack blocks, but are untested. There are three families of maps:

| key | name | needs | depends on the question |
|---|---|---|---|
| A | attention received | the vision tower only (CPU works) | no |
| B | attention from one 2×2 block | the vision tower only | no |
| E | Grad-CAM | full model + backward (GPU) | yes |
| H | HiResCAM | full model + backward | yes |
| D | Chefer relevance | full model + backward + capture pass | yes |
| C | attention rollout (Abnar & Zuidema), same readout as D | as D | yes (readout) |
| W | readout only, no attention | as D | yes |

The [deletion test](deletion.md) then checks whether E/H/D/C/W point at what the model uses.
File formats are in [outputs.md](outputs.md).

## What the model sees

- **Input.** The input is built as lmms-eval's chat `qwen3_vl` backend (lmms-eval 0.7.2)
  builds it for an eval: its video kwargs, the system prompt "You are a helpful assistant.",
  a [video, text] user turn, the chat template with the generation prompt,
  `qwen_vl_utils.process_vision_info` (patch size 16) and the processor call. `tower.py`
  repeats these steps without importing lmms-eval; `tests/test_tower.py` checks that both
  give the same tensors. The video arguments are the backend's (`--fps`, `--max-frames`,
  `--min-pixels`, `--max-pixels`, `--total-pixels`); the CLI defaults are 2 fps and at most 32
  frames.
- **Frames shown.** The frames drawn are rebuilt from the model's own `pixel_values_videos`
  (inverse patchify, `layout.frames_from_pixels`), so their resolution, crop and pairing are
  exact. When the frame count is odd the processor repeats the last frame; the viewer marks
  it "padded".
- **Labels.** The viewer shows each pair's `<x seconds>` label as the LLM reads it.
- **Video reader.** qwen-vl-utils decodes with torchcodec if installed, else decord, else
  torchvision (`FORCE_QWENVL_VIDEO_READER` overrides). Different readers can give slightly
  different frames; use the reader your eval used.

## Maps A and B: ViT attention

- **One map per frame pair.** The video processor merges 2 consecutive frames into one 3-D
  patch. The ViT then attends within one pair at a time, never across pairs (one `cu_seqlens`
  chunk per pair). The map is drawn on both frames of its pair.
- **The ViT never sees the question.** These maps are the same for every question; they show
  what the vision tower does, not what the LLM uses.
- **A, received:** how much attention each patch receives, averaged over all query patches
  (and over heads unless you pick one in the viewer). Saved for all 27 blocks and every head.
- **B, from a block:** where one 2×2 patch block looks: the mean of its 4 patches' attention
  rows, averaged over heads. One block is exactly one token the LLM gets. In the viewer you
  click to choose the block (fixed in the frame, not tracked). Saved for `--layers`
  (default 2, 8, 16, 24, 26).
- **Units are "× average".** 1 means the patch gets 1/n of the attention. The colour scale is
  fixed per layer over the whole clip (its 99th percentile), so brightness is comparable over
  time. The viewer can switch to a per-frame scale.

### How the capture works, and how we know it is right

transformers does not return the ViT's attention weights. An attention function registered
in `AttentionInterface` (`capture.py`) returns the **stock sdpa output**, so the tower's output
is unchanged, and also recomputes the softmax in fp32 for the maps. Every run checks the
following and stops if one fails:

| check | meaning |
|---|---|
| `recon_gap` | max \|captured softmax × V − sdpa output\| / max \|sdpa output\| |
| `pair0_alone_vs_in_clip` | pair 1 encoded alone vs. inside the clip, relative |
| calls per layer | one per block per pair |

Tests: `frames_from_pixels` inverts the real `Qwen3VLVideoProcessor` exactly (this pins the
2×2 token order); on a tiny random tower the output is bit-identical with and without the
capture; B equals the hand-computed mean of the 4 patch rows.

## Maps E and H: Grad-CAM and HiResCAM

Maps A and B are the same for every question. E and H show which patches' content mattered
for one number (a *score*) computed by the full model, so they depend on the question.

### Scores

- **Answer contrast** (`--question` with `--correct`, or `--item` for a benchmark item). The
  score is logit(correct letter) − logit(wrong letter) at the answer position. These are raw
  logits, as in the Grad-CAM paper: a confident model's probability barely moves, so its
  gradient would vanish.
  - The wrong letter: `--wrong` if given (for a two-option item, the other one), else the
    strongest other option (highest logit) on the plain forward. Benchmark items read from an
    lmms-eval run use the model's own pick when it answered wrong.
  - Content that both options share raises both letters alike and cancels. What is left is
    the evidence that separates them.
  - One backward gives two maps on one colour scale: **for correct** = ReLU(+map) and **for
    wrong** = ReLU(−map), the Grad-CAM paper's "counterfactual explanation". On a wrongly
    answered item, this is what misled the model.
- **Phrase** (`--phrase TEXT`). The input asks "Is there TEXT in the video? Answer yes or no."
  (`--prompt`, with a `{phrase}` placeholder, changes the wording). The score is the yes/no
  log-odds at the answer position, log(p(Yes)+p(yes)) − log(p(No)+p(no)), computed with
  logsumexp of the raw logits: **for yes** = where the model finds the thing, **for no** =
  what argues against it. Both cases count because Qwen3-VL answers in lowercase.
  - Scoring the log-probability of the phrase as a caption does not work: its shared first
    token ("a") dominates, so different phrases give nearly the same map. The yes/no contrast
    cancels the wording.
  - Every phrase is scored on every clip of the run, so a phrase on a clip that lacks the
    thing works as a control.

### Two methods, per frame pair

Each frame pair is encoded as its own image, so both maps are computed per pair, from the
features h_i (1152 numbers per patch i) and gradients g_i = d(score)/d(h_i) at a block's
output.

- **Grad-CAM** (Selvaraju et al. 2017): average the gradient over the pair's patches, giving
  one weight per feature (α); the map is h_i · α. The averaging throws away which patch the
  gradient came from, so the map spreads over everything with similar features.
- **HiResCAM** (Draelos & Carin 2020): each patch's own gradient times its own features,
  h_i · g_i. Summed over a pair it is the first-order change of the score if that pair's
  features were zeroed. The viewer shows it by default.
  - **H at the last block (26) is 0 by construction.** Block 26's only consumer is the final
    merger, which starts with a per-token LayerNorm. A LayerNorm ignores its input's scale,
    so its gradient is orthogonal to its input: h_i · g_i = 0, and what H@26 shows is bf16
    rounding noise. Grad-CAM at 26 uses the gradient's pair mean, so it is not 0.

Both are kept signed (ReLU(±) gives the two sides). The colour scale is fixed per layer over
the whole clip, so brighter pairs mattered more. The gradient reaches a block through every
route: later blocks, the DeepStack mergers after blocks 8/16/24, and the final merger.

**Memory.** The backward runs with activation checkpointing on the ViT blocks above the
lowest tapped one and on every decoder layer (`fullmodel.grad_mode`), so one A100-40GB holds a
32-frame clip with the 8B model.

### Checks

| check | meaning |
|---|---|
| `taps_change_nothing` | score with taps + checkpointing vs. a plain no-grad forward (≤ 0.25) |
| `grads_reach_all` | gradient finite at every tapped layer, nonzero in every pair |
| `eval_answer` | for a benchmark item: our greedy first token = the eval's recorded letter (a flip passes only as a near-tie, \|Δlogit\| < 0.5) |

Recorded but not enforced: the model's greedy first token for your own question and for
each phrase (Yes or No expected).

Tests use a tiny random full Qwen3-VL in float64: the tapped gradients equal plain autograd
(no hooks, no checkpointing, DeepStack route included), and a finite-difference nudge agrees
with them.

## Maps D, C, W: relevance back to the input patches

E/H at block L say which features *of block L* mattered. By block 16 every token has mixed
with its whole frame pair, so that is not the same as which input patches mattered. Map **D**
(Chefer et al., ICCV 2021, "Generic attention-model explainability") carries the score back
to the input through every block's attention, each attention edge weighted by its gradient.
**C** (attention rollout, Abnar & Zuidema 2020) does the same without gradients. **W** is the
readout alone, through no attention: the baseline that shows what following the attention
adds.

### How it is computed

Per frame pair (the ViT's attention never crosses pairs), n patches, block l = 0..26:

- A_l is the attention [heads, n, n], recomputed in fp32 by the capture (as for map A).
- ∇A_l = d(score)/dA_l. The block computes O = A V per head, so ∇A = dO Vᵀ, with dO the
  gradient at the attention output (the input of `attn.proj`). No attention table enters the
  backward graph.
- Ā_l = mean over heads of ReLU(A ⊙ ∇A): Chefer's `avg_heads`, ReLU before the mean. The
  "against" side uses ReLU(−A ⊙ ∇A).
- R holds the relevance of every input patch (columns) for every token after block l (rows).
  R = I before block 0, then R ← R + Ā R; the identity is the residual connection.
- C uses R ← (R + mean_h(A) R) / 2 (Abnar's rollout) and the same readout. It has no
  gradient, so D vs. C isolates the gradient weighting of the edges.

**Readout.** Qwen3-VL has no CLS token. The score reaches the LLM by four routes: the
DeepStack mergers after blocks 8, 16 and 24, and the final merger after 26. Each merger turns
one 2×2 block into one LLM token y_b. Route t weighs the 4 patches of block b each with
w_t = ReLU(±Σ y_b ⊙ g_b) / 4, where g_b = d(score)/d(y_b): the LLM token's gradient × input,
shared equally by its patches. R is kept as diag(s) + Q, where s is what a token still holds
of itself and Q what attention moved between tokens. Then:

- **D** = Σ_t w_tᵀ Q_t, what attention carried to the readout tokens. Chefer zeroes his CLS
  token's self-relevance; here every readout token's own identity term is left out, and that
  is W. `D@all` is the sum over the routes, and `D@r8` … `D@r26` are one route each.
- **D@norm** scales each row of Q to sum 1 (Chefer's rule 10), so a token passes on exactly
  its readout weight; a row of zeros stays zero.
- **W** = Σ_t w_t ⊙ s_t is the identity part alone. D + W is the whole readout of R.
- **C** = Σ_t w_tᵀ Q_t of the rollout state.

Why the merger's output and not its input: every merger starts with a LayerNorm, which
ignores the scale of its input, so the gradient through it is orthogonal to the input and
x · g = 0. The final merger normalises each token, so x · g is 0 per token; a DeepStack merger
normalises a block's 4 tokens together, so it is 0 per block.

**Passes per score.** The E/H backward is taken down to block 0: the leaf moves to the patch
embedding, every block is checkpointed, and taps keep dO (per block, on the CPU) and each
merger's output and gradient. Then one no-grad forward captures every block's attention and
updates R pair by pair (`relevance.ViTRelevance`).

**Recorded per score:** D's share of D + W per side (`relevance.mixing_share`), and D's and
W's mass per pair. The share is not scale-free: Ā = ReLU(A ⊙ ∇A) carries the score's units and
the residual identity carries none, so scaling a score by k scales D's first-order part by k²
and W by k. Compare shares only between scores in the same units (logits here), and do not
read the share as "how much the ViT mixes".

| check | meaning |
|---|---|
| `attn_grads_reach_all` | dO finite and nonzero in every block and pair |
| `routes_reach_all` | every route's readout weight finite and nonzero in every pair |
| `relevance_recon_gap` | the captured softmax @ V vs. sdpa's output (map A's tolerance) |
| `relevance_pass_score` | the capture pass's score vs. the plain score (≤ 0.25) |

**Tests** (tiny random full model in float64, ViT depth 3, DeepStack after block 1):

- dO equals `retain_grad` on the `attn.proj` input; under checkpointing the forward runs
  twice per block and the taps keep the first call.
- dO Vᵀ equals autograd's gradient of an explicit softmax, in every block and pair.
- The route taps hold each merger's output and its gradient, and a finite-difference nudge
  of the output agrees. x · g at every merger's input is 0 (per token for the final merger,
  per block for a DeepStack merger): the reason for the readout.
- The reducer equals Chefer's loop and Abnar's product written out.
- `explain`'s D, C, W equal that brute force on plain autograd. The score is scaled by 1e4
  there: the tiny model's Ā rows are small enough that the Ā R term would otherwise hide
  under float32 rounding.

These tests check the definitions above, not whether a definition is a good one; the
[deletion test](deletion.md) is the check of that.

## Determinism

Forward-only results (A, B, plain scores, deletion curves) are deterministic for a given
GPU, dtype and library versions. The bf16 backward is not: two runs of the same scores differ
by up to a few percent in gradient norm and at single pixels, with correlation ≥ 0.999 at
blocks 2–24. When maps are added to an existing dir, saved maps that are recomputed on the way
are kept bit for bit and only compared (`reproduced` in `meta.json`).

## Extending

- A new attention map is a reducer in `reducers.py`: `update(layer, pair, probs[heads, n, n])`,
  then `result()`. Add it in `run.analyse` and draw it in the viewer. A reducer that also
  needs V sets `uses_values = True` (as `ViTRelevance`).
- A new score is a class in `targets.py` (`build`, `keep`, `score`, `report`, `maps`,
  optionally `resolve`).
- A new gradient map is a reducer next to `GradCAM`/`HiResCAM`
  (`update(layer, pair, feats, grads)`), saved under its own key in `gradcam.npz` with its
  layout in `meta.json["maps"]`, and named in `maps.NAMES` and `maps.UNITS`; add its key to the
  choices of `qwen-eye gradcam --maps` and `qwen-eye deletion --methods` (`cli.py`).

## References

- R. R. Selvaraju et al. Grad-CAM: Visual explanations from deep networks via gradient-based
  localization. ICCV 2017.
- R. L. Draelos, L. Carin. Use HiResCAM instead of Grad-CAM for faithful explanations of
  convolutional neural networks. arXiv:2011.08891, 2020.
- H. Chefer, S. Gur, L. Wolf. Generic attention-model explainability for interpreting
  bi-modal and encoder-decoder transformers. ICCV 2021.
- S. Abnar, W. Zuidema. Quantifying attention flow in transformers. ACL 2020.
- V. Petsiuk, A. Das, K. Saenko. RISE: Randomized input sampling for explanation of black-box
  models. BMVC 2018.
- T. Darcet et al. Vision transformers need registers. ICLR 2024.
