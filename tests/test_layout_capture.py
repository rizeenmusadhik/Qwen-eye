"""Token layout and the attention capture (maps A/B) on a tiny random ViT."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from qwen_eye import layout

from helpers import T, H, W, N, snapshot


def test_block_order_groups_each_merge_block_contiguously():
    order = layout.block_order(H, W)
    assert sorted(order.tolist()) == list(range(N))
    # tokens 0-3 are the top-left 2x2 block, row by row inside it
    assert order[:4].tolist() == [0, 1, W, W + 1]
    rc = layout.token_coords(order, W)
    for b in range(N // 4):
        rows, cols = rc[4 * b:4 * b + 4].T
        assert {layout.block_of(int(r), int(c), W) for r, c in zip(rows, cols)} == {b}
    grid = layout.to_grid(order.float(), order, H, W)          # token i holds its own position
    assert torch.equal(grid.flatten(), torch.arange(N).float())


@pytest.mark.parametrize("frames", [4, 3])
def test_frames_from_pixels_inverts_the_real_video_processor(frames):
    """Exact, including the processor repeating the last frame of an odd count."""
    from transformers import AutoVideoProcessor

    vp = AutoVideoProcessor.from_pretrained(snapshot())
    video = np.random.default_rng(0).integers(0, 256, (frames, 64, 96, 3), dtype=np.uint8)
    out = vp(videos=[video], do_resize=False, do_sample_frames=False, return_tensors="pt")
    t, h, w = (int(x) for x in out["video_grid_thw"][0])
    got = layout.frames_from_pixels(out["pixel_values_videos"], (t, h, w), layout.block_order(h, w, vp.merge_size),
                                    patch=vp.patch_size, tpatch=vp.temporal_patch_size,
                                    mean=vp.image_mean, std=vp.image_std)
    want = np.concatenate([video, video[-1:]]) if frames % 2 else video
    assert got.shape == want.shape and np.array_equal(got, want)


def test_capture_leaves_the_tower_output_bit_identical(captured):
    (merged, deep), (merged0, deep0) = captured["out"], captured["stock"]
    assert torch.equal(merged, merged0)
    assert all(torch.equal(a, b) for a, b in zip(deep, deep0))
    assert captured["tower"].config._attn_implementation == "sdpa"     # attach(None) restores it


def test_captured_softmax_is_the_one_the_tower_used(captured):
    cap = captured["cap"]
    assert cap.recon_gap < 1e-5
    assert cap.calls == [T] * 3                                       # one call per block per pair


def test_pairs_are_encoded_independently(captured):
    merged, alone = captured["out"][0], captured["alone"][0]
    assert torch.allclose(alone, merged[: N // 4], atol=1e-5)


def test_map_a_and_b_are_in_units_of_average(captured):
    received, blocks, dist, full = captured["reducers"]
    A, B = received.result()["A"], blocks.result()["B"]
    assert A.shape == (3, T, 2, N) and B.shape == (2, T, N // 4, N)
    assert np.allclose(A.mean(-1), 1, atol=1e-5) and np.allclose(B.mean(-1), 1, atol=1e-5)
    # B = the 4 patch rows of a block, heads averaged
    for li, L in enumerate([0, 2]):
        for p in range(T):
            want = full.probs[(L, p)].mean(0).view(N // 4, 4, N).mean(1).numpy() * N
            assert np.allclose(B[li, p], want, atol=1e-5)
    md = dist.result()["mean_dist"]
    assert md.shape == (3, T, 2) and (md > 0).all()

