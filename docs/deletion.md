# Deletion test: does a map point at what the score uses?

A map can look right and still not be what the model relies on. `qwen-eye deletion` checks
that by perturbing the input along the map (Petsiuk et al. 2018, "deletion"; Chefer et al.
2021, "positive/negative perturbation" for ViTs) and running the full model again:

```bash
qwen-eye deletion OUT/<clip>/gradcam/<tag> [more dirs ...] --maps H@8 D@all W@all
```

## What it does

- The unit is a 2×2 merge block, i.e. one LLM token. A block's value is the map's signed sum
  over its 4 patches. All pairs' blocks are ranked together, because the map's colour scale is
  shared across the clip as well.
- Blanking a block sets its 4 patches, in both frames of the pair, to mid grey in the model's
  input pixels (0 after Qwen3-VL's mean = std = 0.5 normalisation). The full model then runs
  forward on the same prompt. The ViT is not reached into: everything after the pixels
  responds, as for a real input.
- A map is named `KEY@PART`: `H@8` (HiResCAM at block 8), `E@24`, `D@all`, `D@r16`, `D@norm`,
  `C@all`, `W@all`. `--maps` lists them exactly; without it, every saved part of each
  `--methods` map is tested (default H and E; `--layers` cuts H/E to those layers). Each map
  gives two orders:
  - **for_first** blanks the most positive blocks first. If the map is faithful, the score
    falls faster than with random blocks.
  - **against_first** blanks the most negative blocks first. The score should rise.
  - H and E are one signed array (for = most positive, against = most negative). D, C and W
    have one array per side, and each order ranks its own.
- **random** orders (`--random`, seeds 0..n−1) are the baseline. Every map uses the same ones.
- Blanking runs at every fraction of the clip's blocks (`--fractions`, default
  1/2/5/10/20%). Each point is one forward pass.
- Per curve, `area` is the score change averaged over [0, the largest fraction]: a trapezoid
  that starts at 0 change at fraction 0, in logits. A faithful map has for_first well below
  random and against_first above it.

## Cost

Forwards per score = 1 (clean) + fractions × (2 orders × maps + random). With the defaults
(H and E at 5 layers, 3 random orders, 5 fractions) that is 116. On an A100 a forward of the
8B model takes about 0.2 s for a clip of ~1.4k tokens and about 1.5 s at ~10k tokens (an HD
clip at 32 frames). The dry run (`--dry-run`) prints the count per dir.

## Resuming and adding maps

The result is `deletion.json` next to `gradcam.npz` ([outputs.md](outputs.md)). It is written
after every score. A rerun with the same settings skips what is done: finished scores, and
inside a score the maps and random orders it already has. So a long run can stop and continue
in another job, and maps can be added to a file later (the random curves are reused). Other
settings (fractions, random count, fill) are refused: move the file away to start over.

When `qwen-eye gradcam --maps K` later replaces a saved map K, every `deletion*.json` that
ranked K is renamed `*.stale.json`.

## Check

The clean forward must reproduce the score `qwen-eye gradcam` recorded (`plain_value`,
≤ 0.25): same prompt, same pixels, same contrasted letter, so the maps belong to this input.
A score that fails is recorded as failed and retried on the next run.

## Caveats

- Grey blocks are off the training distribution. A random grey block moves the score too,
  which is why the comparison is against random blocks, not against zero.
- Ranking globally lets a map that concentrates on one pair blank most of that pair. That is
  part of what the map claims (which pair mattered), but it is a different perturbation from
  blocks spread out.
- Effects in bf16 move in steps of about 0.125–0.25 logits; use several scores before reading
  a difference between maps.
