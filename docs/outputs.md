# Output files

Everything goes under the `--out` root, one dir per clip (the clip's file stem):

```
OUT/<clip>/                       qwen-eye attn (also made by gradcam, which needs it)
  viewer.html  attention.mp4  preview.png  maps.npz  meta.json  stats.json
  gradcam/<tag>/                  qwen-eye gradcam: <tag> = the item's (direction_13,
                                  location_motion_525, question) or "phrases"
    viewer.html  gradcam.mp4  gradcam.npz  meta.json
    deletion.json                 qwen-eye deletion
```

All arrays are on the patch grid: `pairs` frame pairs × `h` × `w` patches (16 px each), as in
`meta.json["grid"]`. A 2×2 merge block (one LLM token) holds patches (r, c) with block index
(r // 2) · (w / 2) + c // 2, numbered within its pair.

## Clip dir (`qwen-eye attn`)

| file | content |
|---|---|
| `viewer.html` | Self-contained page (typically 20–30 MB), no network needed. Map B grows with the square of the patch count, so above 48 MB (`viewer.MAX_B_MB`) the page leaves B out and says so; `maps.npz` still has it. |
| `attention.mp4` | Columns: frame \| one column per B layer. Rows: A, then B for each `--point` of `qwen-eye render` (default the centre block). |
| `preview.png` | The mp4's first frame of every pair, stacked. |
| `maps.npz` | `A` float16 [27 blocks, pairs, heads, h, w]; `B` float16 [B layers, pairs, blocks, h, w]; `mean_dist` float32 [27, pairs, heads] (mean attention distance in patches); `frames` uint8 [2·pairs, 16h, 16w, 3], the frames the model saw. |
| `meta.json` | `video`, `model`, `grid`, `patch`, `tpatch`, `merge`, `frame_times` (s, every frame), `pair_labels` (the `<x seconds>` text), `video_args`, `layers_B`, `heads`, `device`, `dtype`, `checks`, `tolerance`, `checks_failed`. |
| `stats.json` | Per block: mean attention distance, share of attention in the top 1% of patches, max × average; and the values for uniform attention. |

`qwen-eye render OUT/<clip> --point X,Y --layers ...` redraws the viewer and mp4 from these
files, no model needed.

## Gradient dir (`qwen-eye gradcam`)

| file | content |
|---|---|
| `gradcam.npz` | float32. `E`, `H` signed [scores, layers, pairs, h, w]. `D` [scores, 2 sides, parts, pairs, h, w] (parts `all`, one `r<L>` per route, `norm`); `C`, `W` [scores, 2, 1, pairs, h, w]. Sides: 0 = for (correct / yes), 1 = against. |
| `meta.json` | `video`, `model`, `clip_dir`, `grid`, `layers`, `phrase_prompt`, `item` (the question: prompt, letters, correct, wrong, the eval's answer for benchmark items), `maps` (per saved key: `name`, `sides` = `signed` or 2, `parts`), `reproduced`, `scores`, `checks_failed`. |
| `viewer.html` | Maps A, B and the gradient maps with Score and Method selectors; the question with the correct answer, the model's answer and the contrasted option. Above 48 MB of gradient maps (`viewer.MAX_E_MB`) D's per-route parts are left out first. |
| `gradcam.mp4` | An A row, the answer contrast's two sides in HiResCAM and Grad-CAM, then each phrase's "for yes" side in HiResCAM. With D saved: the answer in HiResCAM and D, the phrases in D. |

Each entry of `scores` (one per score, in array order): `name` (`answer` or `phrase: TEXT`),
`maps` (the two sides' labels and signs), `value` (score on the tapped backward pass),
`plain_value` (score on a plain no-grad forward), `report` (the letters' or yes/no logits and
probabilities, the greedy first token and its probability, the contrasted letter or the
phrase question), `checks`, `failed`, `grad_norms`, `relevance` and `attn_grad_norms`
(with D), `seconds`, `peak_gpu_gb`, `seq_len`.

A rerun of `qwen-eye gradcam` into an existing dir must have the same scores, grid and layers.
It adds the maps the dir lacks; maps named in `--maps` replace saved ones (and their deletion
files become `*.stale.json`). Saved maps recomputed on the way are kept bit for bit and only
compared: `reproduced[key]` = {`max_rel_diff`, `min_corr`}.

## Deletion results (`qwen-eye deletion`)

`deletion.json` next to `gradcam.npz`:

- `settings`: `fractions`, `random`, `fill` (`grey`); a rerun must match them.
- `maps`: the map specs tested (`H@8`, `D@all`, …).
- `scores[name]`: `clean` (the clean forward's score), `recorded` (gradcam's `plain_value`),
  `blocks`, `ks` (blocks blanked at each fraction), `curves` (each with `map`, `order` =
  for_first / against_first / random, `seed` for random, `score` per fraction, `area`),
  `summary` (per map: `for_first`, `against_first`, `random`, `random_sd` areas), `failed`,
  `seconds`.

## Plans

Every command builds a `plan.json` first and runs it (`--save-plan DIR` writes it instead;
`python -m qwen_eye run DIR` runs it later, e.g. inside a cluster job). Its `kind` is `attn`
(the default when absent), `gradcam` or `deletion`; the other keys are the command's
arguments after checking (absolute paths, `variant_args` = the video arguments). Keys a host
tool adds are kept and ignored. Builders: `qwen_eye.plans.attn_plan`, `gradcam_plan`,
`deletion_plan`; `qwen_eye.plans.run(plan)` runs one.

## Older files

Files written by the tool's development versions before 0.1.0 are still read: gradient dirs
without `meta.json["maps"]` (E and H only), and deletion files that list methods × layers
with a numeric fill (read as `fill: grey` and map specs; the first rewrite keeps the original
as `deletion.v1.json`). Phrase scores saved with the retired caption log-prob score can only be
deletion-tested with `--score answer`.
