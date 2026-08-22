# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import numpy as np
import onnx_ir as ir
import onnxruntime as ort

from mobius.generation import (
    build_autoregressive_audio_initializer,
    build_candidate_token_map,
    build_chunk_carry_update,
    build_chunk_overlap_prepare,
    build_chunk_plan,
    build_chunk_slice,
    build_drop_first_frame,
    build_flow_guidance,
    build_overlap_blend,
    build_waveform_stitch,
)


def _run(component, tmp_path, feeds):
    path = tmp_path / f"{component.model.graph.name}.onnx"
    ir.save(component.model, path)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return session.run(None, feeds)


def test_autoregressive_warmup_frame_is_removed(tmp_path):
    history, counter, active, done = _run(
        build_autoregressive_audio_initializer(ir.DataType.FLOAT, fused_hidden_size=4),
        tmp_path,
        {},
    )
    assert history.shape == (1, 0, 4)
    np.testing.assert_array_equal(counter, [0])
    np.testing.assert_array_equal(active, [True])
    np.testing.assert_array_equal(done, [False])

    frames = np.arange(12, dtype=np.float32).reshape(1, 3, 4)
    (emitted,) = _run(
        build_drop_first_frame(ir.DataType.FLOAT, fused_hidden_size=4),
        tmp_path,
        {"history": frames, "stopped": np.array([False])},
    )
    np.testing.assert_array_equal(emitted, frames[:, 1:])
    (emitted_after_stop,) = _run(
        build_drop_first_frame(ir.DataType.FLOAT, fused_hidden_size=4),
        tmp_path,
        {"history": frames, "stopped": np.array([True])},
    )
    np.testing.assert_array_equal(emitted_after_stop, frames[:, 1:-1])


def test_candidate_map_preserves_semantic_rows_and_stop_token(tmp_path):
    component = build_candidate_token_map(
        vocabulary_start=100,
        vocabulary_size=4,
        stop_token_id=999,
    )
    token, semantic, semantic_token, is_stop = _run(
        component, tmp_path, {"candidate": np.array([2])}
    )
    np.testing.assert_array_equal(token, [102])
    np.testing.assert_array_equal(semantic, [[2], [2]])
    np.testing.assert_array_equal(semantic_token, [[102], [102]])
    np.testing.assert_array_equal(is_stop, [False])

    token, _, _, is_stop = _run(component, tmp_path, {"candidate": np.array([4])})
    np.testing.assert_array_equal(token, [999])
    np.testing.assert_array_equal(is_stop, [True])


def test_chunk_plan_and_slice_follow_200_100_contract(tmp_path):
    frames = np.arange(250 * 4, dtype=np.float32).reshape(1, 250, 4)
    count, waveform, previous_latent, previous_condition = _run(
        build_chunk_plan(
            ir.DataType.FLOAT,
            fused_hidden_size=4,
            chunk_frames=200,
            chunk_hop=100,
            latent_channels=3,
            condition_size=5,
        ),
        tmp_path,
        {"frame_hiddens": frames},
    )
    assert count == 2
    assert waveform.shape == (1, 2, 0)
    assert previous_latent.shape == (1, 3, 0)
    assert previous_condition.shape == (1, 0, 5)

    (chunk,) = _run(
        build_chunk_slice(
            ir.DataType.FLOAT,
            fused_hidden_size=4,
            chunk_frames=200,
            chunk_hop=100,
        ),
        tmp_path,
        {"frame_hiddens": frames, "chunk_index": np.array(1, dtype=np.int64)},
    )
    np.testing.assert_array_equal(chunk, frames[:, 100:250])


def test_flow_cfg_and_overlap_carry_are_exact(tmp_path):
    sample = np.stack(
        [
            np.full((2, 5), 3.0, dtype=np.float32),
            np.full((2, 5), 1.0, dtype=np.float32),
        ]
    )
    (velocity,) = _run(
        build_flow_guidance(ir.DataType.FLOAT, latent_channels=2, guidance_scale=1.7),
        tmp_path,
        {"sample": sample},
    )
    np.testing.assert_allclose(velocity, 4.4)

    condition = np.arange(30, dtype=np.float32).reshape(1, 6, 5)
    previous_condition = np.full((1, 2, 5), -1.0, dtype=np.float32)
    previous_latent = np.full((1, 2, 2), 10.0, dtype=np.float32)
    guided, spliced, overlap, row_shape = _run(
        build_chunk_overlap_prepare(
            ir.DataType.FLOAT,
            latent_channels=2,
            condition_size=5,
        ),
        tmp_path,
        {
            "condition": condition,
            "previous_condition": previous_condition,
            "previous_latent": previous_latent,
        },
    )
    assert overlap == 2
    np.testing.assert_array_equal(row_shape, [2, 6])
    np.testing.assert_array_equal(guided[0, :2], previous_condition[0])
    np.testing.assert_array_equal(spliced, guided[:1])
    np.testing.assert_array_equal(guided[1], np.zeros_like(condition[0]))

    latents = np.arange(12, dtype=np.float32).reshape(1, 2, 6)
    (blended,) = _run(
        build_overlap_blend(ir.DataType.FLOAT, latent_channels=2),
        tmp_path,
        {
            "latents": latents,
            "initial_noise": latents,
            "previous_latent": previous_latent,
            "overlap": np.array(2, dtype=np.int64),
            "timestep": np.array([1.0], dtype=np.float32),
        },
    )
    np.testing.assert_allclose(blended[..., :2], previous_latent, atol=1e-5)

    restored, next_latent, next_condition = _run(
        build_chunk_carry_update(
            ir.DataType.FLOAT,
            latent_channels=2,
            condition_size=5,
            carry_length=2,
        ),
        tmp_path,
        {
            "latents": latents,
            "previous_latent": previous_latent,
            "condition": condition,
            "overlap": np.array(2, dtype=np.int64),
        },
    )
    np.testing.assert_array_equal(restored[..., :2], previous_latent)
    np.testing.assert_array_equal(next_latent, restored[..., 2:4])
    np.testing.assert_array_equal(next_condition, condition[:, 2:4])


def test_waveform_stitch_crops_latent_overlap_in_samples(tmp_path):
    component = build_waveform_stitch(
        dtype=ir.DataType.FLOAT,
        latent_hop_length=2,
        crop_left_latents=1,
        crop_right_latents=3,
    )
    waveform = np.arange(24, dtype=np.float32).reshape(1, 2, 12)
    history = np.full((1, 2, 2), -1.0, dtype=np.float32)
    (stitched,) = _run(
        component,
        tmp_path,
        {
            "waveform": waveform,
            "history": history,
            "chunk_index": np.array(1, dtype=np.int64),
            "chunk_count": np.array(3, dtype=np.int64),
        },
    )
    expected = np.concatenate([history, np.clip(waveform[..., 2:6], -1, 1)], axis=2)
    np.testing.assert_array_equal(stitched, expected)
