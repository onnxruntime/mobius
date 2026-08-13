# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Synthetic end-to-end parity for the Mage-VL video vision path."""

from __future__ import annotations

import dataclasses

import numpy as np
import onnx_ir as ir
import onnxruntime as ort
import pytest
import torch
from _test_configs import VL_CONFIGS, _base_config
from torch.nn import functional

from mobius import build_from_module
from mobius._registry import registry
from mobius._testing import count_op_type
from mobius._weight_loading import apply_weights
from mobius.tasks import get_task


def _linear(x: torch.Tensor, state: dict[str, torch.Tensor], prefix: str) -> torch.Tensor:
    return functional.linear(x, state[f"{prefix}.weight"], state.get(f"{prefix}.bias"))


def _layer_norm(
    x: torch.Tensor,
    state: dict[str, torch.Tensor],
    prefix: str,
) -> torch.Tensor:
    return functional.layer_norm(
        x,
        (x.shape[-1],),
        state[f"{prefix}.weight"],
        state[f"{prefix}.bias"],
        1e-6,
    )


def _reference_vision(
    pixel_values: torch.Tensor,
    grid_thw: torch.Tensor,
    patch_positions: torch.Tensor,
    state: dict[str, torch.Tensor],
) -> torch.Tensor:
    prefix = "vision_encoder.visual"
    patch_weight = state[f"{prefix}.embeddings.patch_embedding.weight"]
    hidden = functional.conv2d(
        pixel_values.to(patch_weight.dtype).reshape(-1, 3, 4, 4),
        patch_weight,
        stride=4,
    )
    hidden = hidden.reshape(1, -1, 64)

    # Mage-VL uses independent frequency scales for its 4:6:6 T/H/W split.
    axis_freqs = []
    for axis, size in enumerate((4, 6, 6)):
        inv_freq = 1.0 / (10_000.0 ** (torch.arange(size, dtype=torch.float32) / size))
        axis_freqs.append(patch_positions[:, axis].float().unsqueeze(1) * inv_freq)
    half_freqs = torch.cat(axis_freqs, dim=-1)
    freqs = torch.cat((half_freqs, half_freqs), dim=-1).unsqueeze(0).unsqueeze(2)
    cos, sin = freqs.cos(), freqs.sin()

    segment_ids = []
    for sample_id, (t, h, w) in enumerate(grid_thw.tolist()):
        patches_per_window = 4 * h * w
        for local_id in range(t * h * w):
            segment_ids.append((sample_id, local_id // patches_per_window))
    mask = torch.tensor(
        [[left == right for right in segment_ids] for left in segment_ids],
        dtype=torch.bool,
    )

    hidden = _layer_norm(hidden, state, f"{prefix}.layernorm_pre")
    for layer_idx in range(2):
        layer = f"{prefix}.encoder.layers.{layer_idx}"
        residual = hidden
        normed = _layer_norm(hidden, state, f"{layer}.layer_norm1")
        qkv = _linear(normed, state, f"{layer}.self_attn.qkv")
        q, k, v = qkv.chunk(3, dim=-1)

        def _rope(x: torch.Tensor, target_dtype: torch.dtype = qkv.dtype) -> torch.Tensor:
            x = x.reshape(1, x.shape[1], 2, 32).float()
            even, odd = x[..., ::2], x[..., 1::2]
            rotated = torch.stack((-odd, even), dim=-1).flatten(-2)
            return (x * cos + rotated * sin).to(target_dtype)

        q = _rope(q).transpose(1, 2)
        k = _rope(k).transpose(1, 2)
        v = v.reshape(1, v.shape[1], 2, 32).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-1, -2)) * (32**-0.5)
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        attended = torch.matmul(torch.softmax(scores, dim=-1), v)
        attended = attended.transpose(1, 2).reshape(1, -1, 64)
        hidden = residual + _linear(attended, state, f"{layer}.self_attn.proj")

        residual = hidden
        hidden = _layer_norm(hidden, state, f"{layer}.layer_norm2")
        hidden = _linear(hidden, state, f"{layer}.mlp.fc1")
        hidden = functional.gelu(hidden)
        hidden = residual + _linear(hidden, state, f"{layer}.mlp.fc2")

    hidden = _layer_norm(hidden, state, f"{prefix}.merger.ln_q").reshape(-1, 256)
    hidden = functional.gelu(_linear(hidden, state, f"{prefix}.merger.mlp.0"))
    return _linear(hidden, state, f"{prefix}.merger.mlp.2")


@pytest.mark.parametrize(
    ("dtype", "torch_dtype", "atol"),
    [
        (ir.DataType.FLOAT, torch.float32, 1e-4),
        (ir.DataType.FLOAT16, torch.float16, 1e-2),
    ],
)
@pytest.mark.parametrize(
    ("grid_thw", "frame_positions"),
    [
        (torch.tensor([[5, 2, 2]], dtype=torch.int64), (0, 3, 7, 12, 18)),
        (
            torch.tensor([[1, 2, 2]] * 6, dtype=torch.int64),
            (0, 0, 180, 360, 539, 719),
        ),
    ],
    ids=["packed-five-frame", "processor-image-plus-five-frames"],
)
def test_mage_vl_synthetic_video_parity(
    tmp_path,
    dtype,
    torch_dtype,
    atol,
    grid_thw,
    frame_positions,
):
    """Nonzero packed and processor-shaped media match PyTorch window boundaries."""
    overrides = next(overrides for mt, overrides, _ in VL_CONFIGS if mt == "mage_vl")
    config = dataclasses.replace(_base_config(**overrides), dtype=dtype)
    package = build_from_module(
        registry.get("mage_vl")(config),
        config,
        task="mage-vl",
    )
    vision = package["vision_encoder"]

    generator = torch.Generator().manual_seed(0)
    state = {
        name: (torch.randn(tuple(value.shape), generator=generator) * 0.02).to(torch_dtype)
        for name, value in vision.graph.initializers.items()
        if value.const_value is None
    }
    pixel_values = torch.randn((len(frame_positions) * 4, 48), generator=generator)
    patch_positions = torch.tensor(
        [
            (frame, height, width)
            for frame in frame_positions
            for height in range(2)
            for width in range(2)
        ],
        dtype=torch.int64,
    )
    expected = _reference_vision(pixel_values, grid_thw, patch_positions, state)

    apply_weights(vision, state)
    model_path = tmp_path / "mage_vl_vision.onnx"
    ir.save(vision, model_path)
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    (actual,) = session.run(
        None,
        {
            "pixel_values": pixel_values.numpy(),
            "image_grid_thw": grid_thw.numpy(),
            "patch_positions": patch_positions.numpy(),
        },
    )
    np.testing.assert_allclose(actual, expected.detach().numpy(), atol=atol, rtol=atol)


def test_mage_vl_decode_embedding_accepts_no_new_media(tmp_path):
    """Single-token decode can run with an empty packed feature tensor."""
    overrides = next(overrides for mt, overrides, _ in VL_CONFIGS if mt == "mage_vl")
    config = _base_config(**overrides)
    embedding = get_task("mage-vl").build(registry.get("mage_vl")(config), config)["embedding"]
    state = {
        name: torch.randn(tuple(value.shape), generator=torch.Generator().manual_seed(1))
        for name, value in embedding.graph.initializers.items()
        if value.const_value is None
    }
    apply_weights(embedding, state)
    model_path = tmp_path / "mage_vl_embedding.onnx"
    ir.save(embedding, model_path)
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    (actual,) = session.run(
        None,
        {
            "input_ids": np.array([[7]], dtype=np.int64),
            "image_features": np.empty((0, config.hidden_size), dtype=np.float32),
        },
    )
    np.testing.assert_allclose(
        actual,
        state["embedding.embed_tokens.weight"][[7]].numpy().reshape(1, 1, -1),
    )


def test_mage_vl_cuda_graph_is_fused_and_post_weight_optimized():
    """FP16 CUDA builds fuse hot paths and fold weight transposes after loading."""
    overrides = next(overrides for mt, overrides, _ in VL_CONFIGS if mt == "mage_vl")
    config = dataclasses.replace(_base_config(**overrides), dtype=ir.DataType.FLOAT16)
    package = build_from_module(
        registry.get("mage_vl")(config),
        config,
        task="mage-vl",
        execution_provider="cuda",
    )
    decoder = package["decoder"]
    vision = package["vision_encoder"]

    assert count_op_type(decoder.graph, "GroupQueryAttention") == 2
    assert count_op_type(decoder.graph, "Attention") == 0
    assert count_op_type(decoder.graph, "Swish") == 2
    assert count_op_type(decoder.graph, "SkipSimplifiedLayerNormalization") == 4
    assert count_op_type(vision.graph, "PackedMultiHeadAttention") == 2
    assert count_op_type(vision.graph, "Attention") == 0
    assert count_op_type(vision.graph, "SkipLayerNormalization") == 4
    assert count_op_type(vision.graph, "GreaterOrEqual") == 0

    for model in (decoder, vision):
        state = {
            name: torch.randn(tuple(value.shape), dtype=torch.float32)
            for name, value in model.graph.initializers.items()
            if value.const_value is None
        }
        apply_weights(model, state)
        assert count_op_type(model.graph, "Transpose") == 0


def test_mage_vl_embedding_uses_global_media_order_across_batch(tmp_path):
    """Packed visual features continue across prompt rows instead of restarting."""
    overrides = next(overrides for mt, overrides, _ in VL_CONFIGS if mt == "mage_vl")
    config = _base_config(**overrides)
    embedding = get_task("mage-vl").build(registry.get("mage_vl")(config), config)["embedding"]
    state = {
        name: torch.randn(tuple(value.shape), generator=torch.Generator().manual_seed(2))
        for name, value in embedding.graph.initializers.items()
        if value.const_value is None
    }
    apply_weights(embedding, state)
    model_path = tmp_path / "mage_vl_batched_embedding.onnx"
    ir.save(embedding, model_path)
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_ids = np.array(
        [
            [config.image_token_id, 7],
            [8, config.video_token_id],
        ],
        dtype=np.int64,
    )
    image_features = np.stack(
        [
            np.full(config.hidden_size, 11.0, dtype=np.float32),
            np.full(config.hidden_size, 22.0, dtype=np.float32),
        ]
    )
    (actual,) = session.run(
        None,
        {
            "input_ids": input_ids,
            "image_features": image_features,
        },
    )

    np.testing.assert_allclose(actual[0, 0], image_features[0])
    np.testing.assert_allclose(actual[1, 1], image_features[1])
    np.testing.assert_allclose(actual[0, 1], state["embedding.embed_tokens.weight"][7].numpy())
    np.testing.assert_allclose(actual[1, 0], state["embedding.embed_tokens.weight"][8].numpy())
