# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import math

import numpy as np
import onnx_ir as ir
import onnxruntime as ort
import torch
import torch.nn.functional as torch_functional
from onnxscript import nn

from mobius._builder import build_from_module
from mobius._configs import ArchitectureConfig
from mobius._configs._sub_configs import VisionConfig
from mobius._testing import create_test_builder, create_test_input
from mobius.components._ocr_encoders import (
    DeepSeekOCR2FullImageEncoder,
    DeepSeekOCR2QueryEncoder,
    DeepSeekOCRCLIPEncoder,
    DeepSeekOCRFullImageEncoder,
    Dots3NoteAudioEncoder,
    DotsVisionEncoder,
    Granite4VisionEncoder,
    Granite4WindowQFormerProjector,
    LightOnOCRVisionEncoder,
    PaddleOCRVisionEncoder,
    SigmoidTopKVisionMoE,
    YouTuVLVisionEncoder,
)
from mobius.components._sam_vision import SAMVisionEncoder
from mobius.tasks import (
    GGUFAudioProjectorModel,
    GGUFAudioProjectorTask,
    GGUFVisionProjectorModel,
    GGUFVisionProjectorTask,
)


def _run(module, inputs: dict[str, np.ndarray], *, seed: int = 0):
    builder, op, graph = create_test_builder()
    values = [
        create_test_input(
            builder,
            name,
            list(value.shape),
            dtype=ir.DataType.INT64 if value.dtype == np.int64 else ir.DataType.FLOAT,
        )
        for name, value in inputs.items()
    ]
    output = module(op, *values)
    graph.outputs.append(output)

    rng = np.random.default_rng(seed)
    state: dict[str, torch.Tensor] = {}
    for name, parameter in module.named_parameters():
        if parameter.const_value is not None:
            continue
        shape = tuple(int(dim) for dim in parameter.shape)
        if name.endswith(".weight") and "norm" in name:
            array = (1.0 + rng.normal(0.0, 0.03, shape)).astype(np.float32)
        else:
            array = rng.normal(0.0, 0.04, shape).astype(np.float32)
        parameter.const_value = ir.tensor(array)
        state[name] = torch.from_numpy(array)

    session = ort.InferenceSession(
        ir.serde.serialize_model(ir.Model(graph, ir_version=11)).SerializeToString(),
        providers=["CPUExecutionProvider"],
    )
    return session.run(None, inputs)[0], state, graph


def _linear(x: torch.Tensor, state: dict[str, torch.Tensor], stem: str) -> torch.Tensor:
    return torch_functional.linear(x, state[f"{stem}.weight"], state.get(f"{stem}.bias"))


def _layer_norm(
    x: torch.Tensor,
    state: dict[str, torch.Tensor],
    stem: str,
    eps: float,
) -> torch.Tensor:
    return torch_functional.layer_norm(
        x,
        (x.shape[-1],),
        state[f"{stem}.weight"],
        state[f"{stem}.bias"],
        eps,
    )


def _rms_norm(
    x: torch.Tensor,
    state: dict[str, torch.Tensor],
    stem: str,
    eps: float,
) -> torch.Tensor:
    x = x * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + eps).to(x.dtype)
    return x * state[f"{stem}.weight"]


def _merge_positions(height: int, width: int, merge: int) -> torch.Tensor:
    positions = []
    for block_h in range(height // merge):
        for block_w in range(width // merge):
            for inner_h in range(merge):
                for inner_w in range(merge):
                    positions.append((block_h * merge + inner_h, block_w * merge + inner_w))
    return torch.tensor(positions, dtype=torch.float32)


def _raster_positions(height: int, width: int) -> torch.Tensor:
    return torch.tensor(
        [(row, col) for row in range(height) for col in range(width)],
        dtype=torch.float32,
    )


def _apply_vision_rope(
    states: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    # Matches Qwen25VLVisionRotaryEmbedding + split-half application.
    head_dim = states.shape[-1]
    rotary_dim = head_dim // 2
    inv_freq = 1.0 / (
        10000.0 ** (torch.arange(rotary_dim // 2, dtype=torch.float32) / (rotary_dim // 2))
    )
    h_freq = positions[:, :1] * inv_freq
    w_freq = positions[:, 1:] * inv_freq
    frequency = torch.cat((h_freq, w_freq), dim=-1)
    frequency = torch.cat((frequency, frequency), dim=-1)
    cos = frequency.cos()[:, None, :]
    sin = frequency.sin()[:, None, :]
    first, second = states[..., : head_dim // 2], states[..., head_dim // 2 :]
    return torch.cat(
        (
            first * cos[..., : head_dim // 2] - second * sin[..., : head_dim // 2],
            first * sin[..., head_dim // 2 :] + second * cos[..., head_dim // 2 :],
        ),
        dim=-1,
    )


def _attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    # Inputs are (N, heads, dim); return (N, hidden).
    query = query.transpose(0, 1)
    key = key.transpose(0, 1)
    value = value.transpose(0, 1)
    scores = query @ key.transpose(-1, -2) / math.sqrt(query.shape[-1])
    if bias is not None:
        scores = scores + bias
    output = scores.softmax(dim=-1) @ value
    return output.transpose(0, 1).reshape(output.shape[1], -1)


def _gated_mlp(
    x: torch.Tensor,
    state: dict[str, torch.Tensor],
    stem: str,
) -> torch.Tensor:
    gate = _linear(x, state, f"{stem}.gate_proj")
    up = _linear(x, state, f"{stem}.up_proj")
    return _linear(torch_functional.silu(gate) * up, state, f"{stem}.down_proj")


def _dots_block(
    x: torch.Tensor,
    state: dict[str, torch.Tensor],
    positions: torch.Tensor,
    *,
    eps: float,
    qk_norm: bool = False,
) -> torch.Tensor:
    residual = x
    normalized = _rms_norm(x, state, "blocks.0.norm1", eps)
    qkv = _linear(normalized, state, "blocks.0.attn.qkv")
    query, key, value = qkv.chunk(3, dim=-1)
    query = query.reshape(-1, 2, 4)
    key = key.reshape(-1, 2, 4)
    if qk_norm:
        query = _rms_norm(query, state, "blocks.0.attn.q_norm", eps)
        key = _rms_norm(key, state, "blocks.0.attn.k_norm", eps)
    query = _apply_vision_rope(query, positions)
    key = _apply_vision_rope(key, positions)
    value = value.reshape(-1, 2, 4)
    x = residual + _linear(
        _attention(query, key, value),
        state,
        "blocks.0.attn.proj",
    )
    return x + _gated_mlp(_rms_norm(x, state, "blocks.0.norm2", eps), state, "blocks.0.mlp")


def test_dots_vision_dense_route_matches_independent_reference():
    module = DotsVisionEncoder(
        depth=1,
        hidden_size=8,
        intermediate_size=12,
        num_heads=2,
        patch_size=2,
        output_size=6,
        projector_intermediate_size=32,
        spatial_merge_size=2,
        norm_eps=1e-5,
        qk_norm=False,
    )
    pixels = np.random.default_rng(1).normal(size=(16, 12)).astype(np.float32)
    grid = np.array([[1, 4, 4]], dtype=np.int64)

    actual, state, _ = _run(
        module,
        {"pixel_values": pixels, "image_grid_thw": grid},
        seed=2,
    )
    x = torch_functional.linear(
        torch.from_numpy(pixels),
        state["patch_embed.weight"].reshape(8, -1),
        state["patch_embed.bias"],
    )
    x = _rms_norm(x, state, "pre_layernorm", 1e-5)
    x = _dots_block(x, state, _merge_positions(4, 4, 2), eps=1e-5)
    x = _rms_norm(x, state, "post_layernorm", 1e-5)
    x = _layer_norm(x, state, "projector.input_norm", 1e-6).reshape(4, 32)
    x = torch_functional.gelu(_linear(x, state, "projector.linear_0"))
    expected = _linear(x, state, "projector.linear_2")

    assert actual.shape == (4, 6)
    np.testing.assert_allclose(actual, expected.numpy(), rtol=3e-5, atol=3e-5)


def test_dots3note_vision_qk_norm_route_matches_reference():
    module = DotsVisionEncoder(
        depth=1,
        hidden_size=8,
        intermediate_size=12,
        num_heads=2,
        patch_size=2,
        output_size=6,
        projector_intermediate_size=32,
        spatial_merge_size=2,
        norm_eps=1e-5,
        qk_norm=True,
        expert_counts=[0],
        top_k=2,
    )
    pixels = np.random.default_rng(19).normal(size=(16, 12)).astype(np.float32)
    grid = np.array([[1, 4, 4]], dtype=np.int64)

    actual, state, _ = _run(
        module,
        {"pixel_values": pixels, "image_grid_thw": grid},
        seed=20,
    )
    x = torch_functional.linear(
        torch.from_numpy(pixels),
        state["patch_embed.weight"].reshape(8, -1),
        state["patch_embed.bias"],
    )
    x = _rms_norm(x, state, "pre_layernorm", 1e-5)
    x = _dots_block(
        x,
        state,
        _merge_positions(4, 4, 2),
        eps=1e-5,
        qk_norm=True,
    )
    x = _rms_norm(x, state, "post_layernorm", 1e-5)
    x = _layer_norm(x, state, "projector.input_norm", 1e-6).reshape(4, 32)
    x = torch_functional.gelu(_linear(x, state, "projector.linear_0"))
    expected = _linear(x, state, "projector.linear_2")

    np.testing.assert_allclose(actual, expected.numpy(), rtol=3e-5, atol=3e-5)


def test_dots_sigmoid_moe_uses_bias_only_for_selection():
    module = SigmoidTopKVisionMoE(4, 6, num_experts=3, top_k=2)
    features = np.random.default_rng(3).normal(size=(5, 4)).astype(np.float32)

    actual, state, _ = _run(module, {"features": features}, seed=4)
    x = torch.from_numpy(features)
    probabilities = torch.sigmoid(x @ state["ffn_gate_inp"].T)
    selected = torch.topk(probabilities + state["exp_probs_b"], k=2, dim=-1).indices
    weights = probabilities.gather(1, selected)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    expected = torch.zeros_like(x)
    for token in range(x.shape[0]):
        for slot, expert in enumerate(selected[token]):
            gate = torch_functional.linear(x[token], state["ffn_gate_exps"][expert])
            up = torch_functional.linear(x[token], state["ffn_up_exps"][expert])
            out = torch_functional.linear(
                torch_functional.silu(gate) * up,
                state["ffn_down_exps"][expert],
            )
            expected[token] += weights[token, slot] * out

    np.testing.assert_allclose(actual, expected.numpy(), rtol=3e-5, atol=3e-5)


def _split_block(
    x: torch.Tensor,
    state: dict[str, torch.Tensor],
    positions: torch.Tensor,
    *,
    eps: float,
    block_index: int = 0,
    attention_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    stem = f"blocks.{block_index}"
    residual = x
    normalized = _layer_norm(x, state, f"{stem}.norm1", eps)
    query = _apply_vision_rope(
        _linear(normalized, state, f"{stem}.attn.q_proj").reshape(-1, 2, 4),
        positions,
    )
    key = _apply_vision_rope(
        _linear(normalized, state, f"{stem}.attn.k_proj").reshape(-1, 2, 4),
        positions,
    )
    value = _linear(normalized, state, f"{stem}.attn.v_proj").reshape(-1, 2, 4)
    x = residual + _linear(
        _attention(query, key, value, attention_bias),
        state,
        f"{stem}.attn.out_proj",
    )
    normalized = _layer_norm(x, state, f"{stem}.norm2", eps)
    return x + _linear(
        torch_functional.gelu(
            _linear(normalized, state, f"{stem}.mlp.up_proj"),
            approximate="tanh",
        ),
        state,
        f"{stem}.mlp.down_proj",
    )


def test_paddleocr_route_matches_raster_position_and_merger_reference():
    module = PaddleOCRVisionEncoder(
        depth=1,
        hidden_size=8,
        intermediate_size=12,
        num_heads=2,
        patch_size=2,
        position_size=16,
        output_size=6,
        projector_intermediate_size=32,
    )
    pixels = np.random.default_rng(5).normal(size=(16, 3, 2, 2)).astype(np.float32)
    grid = np.array([[1, 4, 4]], dtype=np.int64)

    actual, state, _ = _run(
        module,
        {"pixel_values": pixels, "image_grid_thw": grid},
        seed=6,
    )
    x = torch_functional.linear(
        torch.from_numpy(pixels).reshape(16, -1),
        state["patch_embed.weight"].reshape(8, -1),
        state["patch_embed.bias"],
    )
    x = x + state["position_embedding"]
    x = _split_block(x, state, _raster_positions(4, 4), eps=1e-6)
    x = _layer_norm(x, state, "post_layernorm", 1e-6)
    x = _layer_norm(x, state, "projector.input_norm", 1e-5)
    x = x.reshape(2, 2, 2, 2, 8).permute(0, 2, 1, 3, 4).reshape(4, 32)
    x = torch_functional.gelu(
        _linear(x, state, "projector.linear_1"),
        approximate="tanh",
    )
    expected = _linear(x, state, "projector.linear_2")

    np.testing.assert_allclose(actual, expected.numpy(), rtol=4e-5, atol=4e-5)


def test_paddleocr_learned_positions_resize_bilinear_with_antialias():
    module = PaddleOCRVisionEncoder(
        depth=0,
        hidden_size=8,
        intermediate_size=12,
        num_heads=2,
        patch_size=2,
        position_size=16,
        output_size=6,
        projector_intermediate_size=32,
    )
    pixels = np.random.default_rng(21).normal(size=(8, 3, 2, 2)).astype(np.float32)
    grid = np.array([[1, 2, 4]], dtype=np.int64)

    actual, state, _ = _run(
        module,
        {"pixel_values": pixels, "image_grid_thw": grid},
        seed=22,
    )
    x = torch_functional.linear(
        torch.from_numpy(pixels).reshape(8, -1),
        state["patch_embed.weight"].reshape(8, -1),
        state["patch_embed.bias"],
    )
    positions = state["position_embedding"].T.reshape(1, 8, 4, 4)
    positions = (
        torch_functional.interpolate(
            positions,
            size=(2, 4),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        .reshape(8, -1)
        .T
    )
    x = _layer_norm(x + positions, state, "post_layernorm", 1e-6)
    x = _layer_norm(x, state, "projector.input_norm", 1e-5)
    x = x.reshape(1, 2, 2, 2, 8).permute(0, 2, 1, 3, 4).reshape(2, 32)
    x = torch_functional.gelu(
        _linear(x, state, "projector.linear_1"),
        approximate="tanh",
    )
    expected = _linear(x, state, "projector.linear_2")

    np.testing.assert_allclose(actual, expected.numpy(), rtol=1e-4, atol=1e-4)


def test_youtuvl_route_matches_merge_ordered_reference():
    module = YouTuVLVisionEncoder(
        depth=1,
        hidden_size=8,
        intermediate_size=12,
        num_heads=2,
        pixel_size=12,
        patch_size=2,
        output_size=6,
        projector_intermediate_size=32,
        spatial_merge_size=2,
        window_size=8,
        full_attention_layers=[0],
        norm_eps=1e-6,
    )
    pixels = np.random.default_rng(7).normal(size=(1, 16, 12)).astype(np.float32)
    grid = np.array([[1, 4, 4]], dtype=np.int64)

    actual, state, graph = _run(
        module,
        {"pixel_values": pixels, "spatial_shapes": grid[:, 1:]},
        seed=8,
    )
    x = _linear(torch.from_numpy(pixels).reshape(16, 12), state, "patch_embed")
    x = _split_block(x, state, _merge_positions(4, 4, 2), eps=1e-6)
    x = _layer_norm(x, state, "post_layernorm", 1e-6)
    x = _rms_norm(x, state, "projector.input_norm", 1e-6).reshape(4, 32)
    x = torch_functional.gelu(
        _linear(x, state, "projector.linear_0"),
        approximate="tanh",
    )
    expected = _linear(x, state, "projector.linear_2")

    assert any(node.op_type == "Scan" for node in graph)
    np.testing.assert_allclose(actual, expected.numpy(), rtol=4e-5, atol=4e-5)


def test_youtuvl_window_schedule_matches_block_diagonal_reference():
    module = YouTuVLVisionEncoder(
        depth=2,
        hidden_size=8,
        intermediate_size=12,
        num_heads=2,
        pixel_size=12,
        patch_size=2,
        output_size=6,
        projector_intermediate_size=32,
        spatial_merge_size=2,
        window_size=8,
        full_attention_layers=[1],
        norm_eps=1e-6,
    )
    pixels = np.random.default_rng(23).normal(size=(1, 64, 12)).astype(np.float32)
    spatial_shapes = np.array([[8, 8]], dtype=np.int64)

    actual, state, _ = _run(
        module,
        {"pixel_values": pixels, "spatial_shapes": spatial_shapes},
        seed=24,
    )
    x = _linear(torch.from_numpy(pixels).reshape(64, 12), state, "patch_embed")
    positions = _merge_positions(8, 8, 2)
    group_index = np.arange(16).reshape(2, 2, 2, 2).transpose(0, 2, 1, 3).reshape(-1)
    patch_index = (group_index[:, None] * 4 + np.arange(4)[None]).reshape(-1)
    x = x[patch_index]
    positions = positions[patch_index]
    window_bias = torch.full((64, 64), -1e9)
    for start in range(0, 64, 16):
        window_bias[start : start + 16, start : start + 16] = 0
    x = _split_block(
        x,
        state,
        positions,
        eps=1e-6,
        block_index=0,
        attention_bias=window_bias,
    )
    x = _split_block(
        x,
        state,
        positions,
        eps=1e-6,
        block_index=1,
    )
    x = _layer_norm(x, state, "post_layernorm", 1e-6)
    x = _rms_norm(x, state, "projector.input_norm", 1e-6).reshape(16, 32)
    x = torch_functional.gelu(
        _linear(x, state, "projector.linear_0"),
        approximate="tanh",
    )
    x = _linear(x, state, "projector.linear_2")
    expected = x[np.argsort(group_index)]

    np.testing.assert_allclose(actual, expected.numpy(), rtol=5e-5, atol=5e-5)


def test_lighton_full_pixtral_sidecar_graph_executes():
    config = ArchitectureConfig(
        vocab_size=1,
        hidden_size=6,
        intermediate_size=12,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=6,
        max_position_embeddings=1,
        hidden_act="silu",
        vision=VisionConfig(
            hidden_size=8,
            intermediate_size=12,
            num_hidden_layers=1,
            num_attention_heads=2,
            head_dim=4,
            image_size=8,
            patch_size=2,
            spatial_merge_size=2,
            norm_eps=1e-5,
            rope_theta=10_000.0,
        ),
    )
    module = LightOnOCRVisionEncoder(config)
    pixels = np.random.default_rng(29).normal(size=(1, 3, 8, 8)).astype(np.float32)

    actual, _, _ = _run(module, {"pixel_values": pixels}, seed=30)

    assert actual.shape == (1, 4, 6)
    assert np.isfinite(actual).all()
    assert not np.allclose(actual, 0)


def _partial_rope(states: torch.Tensor) -> torch.Tensor:
    # states: (B, heads, sequence, head_dim), rotate the first head_dim/2.
    rotary_dim = states.shape[-1] // 2
    inv_freq = 1.0 / (
        10000.0 ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim)
    )
    positions = torch.arange(states.shape[2], dtype=torch.float32)
    frequencies = positions[:, None] * inv_freq[None]
    cos = frequencies.cos()[None, None]
    sin = frequencies.sin()[None, None]
    half = rotary_dim // 2
    first = states[..., :half]
    second = states[..., half:rotary_dim]
    return torch.cat(
        (first * cos - second * sin, first * sin + second * cos, states[..., rotary_dim:]),
        dim=-1,
    )


def test_dots3note_audio_route_matches_partial_rope_reference():
    module = Dots3NoteAudioEncoder(
        num_mel_bins=8,
        conv_channels=4,
        depth=1,
        hidden_size=8,
        intermediate_size=12,
        num_heads=2,
        output_size=6,
        norm_eps=1e-6,
    )
    features = np.random.default_rng(9).normal(size=(1, 8, 16)).astype(np.float32)

    actual, state, _ = _run(module, {"input_features": features}, seed=10)
    x = torch.from_numpy(features)[:, None]
    for index in range(3):
        x = torch_functional.gelu(
            torch_functional.conv2d(
                x,
                state[f"conv2d.{index}.weight"],
                state[f"conv2d.{index}.bias"],
                stride=2,
                padding=1,
            )
        )
    x = x.permute(0, 3, 1, 2).reshape(1, x.shape[3], -1)
    x = _linear(x, state, "conv_out")
    residual = x
    normalized = _rms_norm(x, state, "blocks.0.norm1", 1e-6)
    batch, seq, hidden = normalized.shape
    query = _linear(normalized, state, "blocks.0.attn.q_proj")
    key = _linear(normalized, state, "blocks.0.attn.k_proj")
    value = _linear(normalized, state, "blocks.0.attn.v_proj")
    query = _partial_rope(query.reshape(batch, seq, 2, 4).permute(0, 2, 1, 3))
    key = _partial_rope(key.reshape(batch, seq, 2, 4).permute(0, 2, 1, 3))
    value = value.reshape(batch, seq, 2, 4).permute(0, 2, 1, 3)
    attention = torch_functional.scaled_dot_product_attention(query, key, value)
    attention = attention.permute(0, 2, 1, 3).reshape(batch, seq, hidden)
    x = residual + _linear(attention, state, "blocks.0.attn.out_proj")
    x = x + _gated_mlp(_rms_norm(x, state, "blocks.0.norm2", 1e-6), state, "blocks.0.mlp")
    x = _rms_norm(x, state, "post_layernorm", 1e-6)
    x = _layer_norm(x, state, "projector.norm_pre", 1e-5)
    x = torch_functional.gelu(_linear(x, state, "projector.linear_1"))
    expected = _linear(x, state, "projector.linear_3")

    np.testing.assert_allclose(actual, expected.numpy(), rtol=5e-5, atol=5e-5)


def test_dots3note_audio_fp16_keeps_rotary_frequency_math_float32():
    module = Dots3NoteAudioEncoder(
        num_mel_bins=8,
        conv_channels=4,
        depth=1,
        hidden_size=8,
        intermediate_size=12,
        num_heads=2,
        output_size=6,
        norm_eps=1e-6,
    )
    config = ArchitectureConfig(
        vocab_size=1,
        hidden_size=6,
        intermediate_size=12,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=6,
        max_position_embeddings=1,
        hidden_act="silu",
        dtype=ir.DataType.FLOAT16,
    )
    model = build_from_module(
        GGUFAudioProjectorModel(module),
        config,
        task=GGUFAudioProjectorTask(),
    )["audio_encoder"]
    rng = np.random.default_rng(33)
    for initializer in model.graph.initializers.values():
        if initializer.const_value is not None:
            continue
        numpy_dtype = np.float32 if initializer.dtype == ir.DataType.FLOAT else np.float16
        initializer.const_value = ir.tensor(
            rng.normal(0.0, 0.02, tuple(initializer.shape)).astype(numpy_dtype)
        )
    session = ort.InferenceSession(
        ir.serde.serialize_model(model).SerializeToString(),
        providers=["CPUExecutionProvider"],
    )

    (actual,) = session.run(
        None,
        {"input_features": rng.normal(size=(1, 8, 16)).astype(np.float32)},
    )

    assert actual.shape == (1, 2, 6)
    assert actual.dtype == np.float16
    assert np.isfinite(actual).all()


def test_deepseek_clip_stage_matches_cls_first_quick_gelu_reference():
    module = DeepSeekOCRCLIPEncoder(
        depth=1,
        hidden_size=8,
        intermediate_size=12,
        num_heads=2,
        position_size=10,
        norm_eps=1e-5,
    )
    sam = np.random.default_rng(11).normal(size=(1, 8, 2, 2)).astype(np.float32)

    actual, state, _ = _run(module, {"sam_features": sam}, seed=12)
    x = torch.from_numpy(sam).permute(0, 2, 3, 1).reshape(1, 4, 8)
    x = torch.cat((state["class_embedding"][None, None], x), dim=1)
    patch_positions = torch_functional.interpolate(
        state["position_embedding"][1:].T.reshape(1, 8, 3, 3),
        size=(2, 2),
        mode="bicubic",
        antialias=True,
        align_corners=False,
    )
    positions = torch.cat(
        (state["position_embedding"][:1], patch_positions.reshape(8, 4).T),
        dim=0,
    )
    x = x + positions[None]
    x = _layer_norm(x, state, "pre_layernorm", 1e-5)
    residual = x
    normed = _layer_norm(x, state, "blocks.0.norm1", 1e-5)
    qkv = _linear(normed, state, "blocks.0.attn.qkv")
    query, key, value = qkv.reshape(1, 5, 3, 2, 4).unbind(dim=2)
    attention = torch_functional.scaled_dot_product_attention(
        query.transpose(1, 2),
        key.transpose(1, 2),
        value.transpose(1, 2),
    )
    attention = attention.transpose(1, 2).reshape(1, 5, 8)
    x = residual + _linear(attention, state, "blocks.0.attn.out_proj")
    normed = _layer_norm(x, state, "blocks.0.norm2", 1e-5)
    hidden = _linear(normed, state, "blocks.0.mlp.up_proj")
    hidden = hidden * torch.sigmoid(1.702 * hidden)
    expected = x + _linear(hidden, state, "blocks.0.mlp.down_proj")

    # ONNX Resize's antialias filter differs slightly from PyTorch at the image edge.
    np.testing.assert_allclose(actual, expected[:, 1:].numpy(), rtol=4e-3, atol=4e-3)


def test_deepseek_sam_stage_matches_decomposed_relative_position_reference():
    module = SAMVisionEncoder(
        img_size=32,
        patch_size=16,
        embed_dim=8,
        depth=1,
        num_heads=2,
        out_chans=4,
        window_size=2,
        global_attn_indexes=(),
        downsample_channels=(6, 8),
        mlp_activation="gelu",
    )
    pixels = np.random.default_rng(25).normal(size=(1, 3, 32, 32)).astype(np.float32)

    actual, state, _ = _run(module, {"pixel_values": pixels}, seed=26)
    x = torch_functional.conv2d(
        torch.from_numpy(pixels),
        state["patch_embed.weight"],
        state["patch_embed.bias"],
        stride=16,
    ).permute(0, 2, 3, 1)
    x = x + state["pos_embed"]
    residual = x
    normalized = _layer_norm(x, state, "blocks.0.norm1", 1e-6)
    qkv = _linear(normalized.reshape(1, 4, 8), state, "blocks.0.attn.qkv")
    query, key, value = qkv.reshape(1, 4, 3, 2, 4).unbind(dim=2)
    query = query.permute(0, 2, 1, 3)
    key = key.permute(0, 2, 1, 3)
    value = value.permute(0, 2, 1, 3)
    query_grid = query.reshape(2, 2, 2, 4)
    indices = torch.tensor([[1, 0], [2, 1]])
    rel_h = state["blocks.0.attn.rel_pos_h"][indices]
    rel_w = state["blocks.0.attn.rel_pos_w"][indices]
    height_bias = torch.einsum("bhwc,hkc->bhwk", query_grid, rel_h)
    width_bias = torch.einsum("bhwc,wkc->bhwk", query_grid, rel_w)
    relative_bias = (height_bias[..., :, None] + width_bias[..., None, :]).reshape(1, 2, 4, 4)
    scores = query @ key.transpose(-1, -2) / 2.0 + relative_bias
    attention = (scores.softmax(dim=-1) @ value).permute(0, 2, 1, 3).reshape(1, 4, 8)
    attention = _linear(attention, state, "blocks.0.attn.proj").reshape(1, 2, 2, 8)
    x = residual + attention
    normalized = _layer_norm(x, state, "blocks.0.norm2", 1e-6)
    hidden = _linear(normalized, state, "blocks.0.mlp.up_proj")
    hidden = torch_functional.gelu(hidden)
    x = x + _linear(hidden, state, "blocks.0.mlp.down_proj")
    x = x.permute(0, 3, 1, 2)
    x = torch_functional.conv2d(x, state["neck.0.weight"])
    x = _layer_norm(x.permute(0, 2, 3, 1), state, "neck.1", 1e-6).permute(0, 3, 1, 2)
    x = torch_functional.conv2d(x, state["neck.2.weight"], padding=1)
    x = _layer_norm(x.permute(0, 2, 3, 1), state, "neck.3", 1e-6).permute(0, 3, 1, 2)
    x = torch_functional.conv2d(x, state["net_2.weight"], stride=2, padding=1)
    expected = torch_functional.conv2d(
        x,
        state["net_3.weight"],
        stride=2,
        padding=1,
    )

    np.testing.assert_allclose(actual, expected.numpy(), rtol=5e-5, atol=5e-5)


def test_deepseek_sam_window_padding_keeps_channels_unchanged():
    module = SAMVisionEncoder(
        img_size=40,
        patch_size=8,
        embed_dim=8,
        depth=1,
        num_heads=2,
        out_chans=4,
        window_size=4,
        global_attn_indexes=(),
        downsample_channels=(6, 8),
    )
    pixels = np.random.default_rng(31).normal(size=(1, 3, 40, 40)).astype(np.float32)

    actual, _, _ = _run(module, {"pixel_values": pixels}, seed=32)

    assert actual.shape == (1, 8, 2, 2)
    assert np.isfinite(actual).all()


def _qformer_attention(
    query: torch.Tensor,
    source: torch.Tensor,
    state: dict[str, torch.Tensor],
    stem: str,
) -> torch.Tensor:
    q = _linear(query, state, f"{stem}.q_proj").reshape(-1, query.shape[1], 1, 64)
    k = _linear(source, state, f"{stem}.k_proj").reshape(-1, source.shape[1], 1, 64)
    v = _linear(source, state, f"{stem}.v_proj").reshape(-1, source.shape[1], 1, 64)
    output = torch_functional.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
    )
    output = output.transpose(1, 2).reshape(query.shape)
    return _linear(output, state, f"{stem}.out_proj")


def test_granite_window_qformer_matches_independent_reference():
    module = Granite4WindowQFormerProjector(
        64,
        80,
        12,
        image_side=4,
        window_side=2,
        query_side=1,
        spatial_offset=-1,
        norm_eps=1e-6,
    )
    features = np.random.default_rng(13).normal(size=(1, 16, 64)).astype(np.float32)

    actual, state, _ = _run(module, {"hidden_states": features}, seed=14)
    x = _layer_norm(torch.from_numpy(features), state, "norm", 1e-6)
    image = x.reshape(1, 4, 4, 64)
    encoder = image.reshape(1, 2, 2, 2, 2, 64).permute(0, 1, 3, 2, 4, 5)
    encoder = encoder.reshape(4, 4, 64) + state["image_positions"]
    downsampled = torch_functional.avg_pool2d(image.permute(0, 3, 1, 2), 2, 2)
    query = downsampled.permute(0, 2, 3, 1).reshape(4, 1, 64) + state["query"]
    query = _layer_norm(query, state, "post_norm", 1e-12)
    self_out = _qformer_attention(query, query, state, "qformer.self_attn")
    query = _layer_norm(
        query + self_out,
        state,
        "qformer.self_attn_norm",
        1e-12,
    )
    cross_out = _qformer_attention(query, encoder, state, "qformer.cross_attn")
    query = _layer_norm(
        query + cross_out,
        state,
        "qformer.cross_attn_norm",
        1e-12,
    )
    ffn = _linear(
        torch_functional.gelu(_linear(query, state, "qformer.ffn_up")),
        state,
        "qformer.ffn_down",
    )
    query = _layer_norm(query + ffn, state, "qformer.ffn_norm", 1e-12)
    query = query.reshape(1, 2, 2, 1, 1, 64).permute(0, 1, 3, 2, 4, 5)
    query = query.reshape(1, 4, 64)
    expected = _linear(query, state, "linear")

    np.testing.assert_allclose(actual, expected.numpy(), rtol=5e-5, atol=5e-5)


def test_deepseek_ocr2_query_mask_matches_independent_reference():
    module = DeepSeekOCR2QueryEncoder(
        depth=1,
        hidden_size=8,
        intermediate_size=12,
        num_heads=2,
        num_kv_heads=1,
        norm_eps=1e-6,
    )
    features = np.random.default_rng(15).normal(size=(1, 8, 12, 12)).astype(np.float32)

    actual, state, graph = _run(module, {"sam_features": features}, seed=16)

    visual = torch.from_numpy(features).permute(0, 2, 3, 1).reshape(1, 144, 8)
    hidden = torch.cat((visual, state["query_768"][None]), dim=1)
    residual = hidden
    normalized = _rms_norm(hidden, state, "blocks.0.norm1", 1e-6)
    query = _linear(normalized, state, "blocks.0.attn.q_proj")
    key = _linear(normalized, state, "blocks.0.attn.k_proj")
    value = _linear(normalized, state, "blocks.0.attn.v_proj")
    query = query.reshape(1, 288, 2, 4).permute(0, 2, 1, 3)
    key = key.reshape(1, 288, 1, 4).permute(0, 2, 1, 3)
    value = value.reshape(1, 288, 1, 4).permute(0, 2, 1, 3)

    positions = torch.arange(288, dtype=torch.float32)
    inv_freq = 1.0 / (1_000_000.0 ** (torch.arange(0, 4, 2) / 4))
    frequencies = positions[:, None] * inv_freq[None]
    cos = frequencies.cos()[None, None]
    sin = frequencies.sin()[None, None]

    def rope(states):
        first, second = states[..., :2], states[..., 2:]
        return torch.cat((first * cos - second * sin, first * sin + second * cos), dim=-1)

    query = rope(query)
    key = rope(key).repeat_interleave(2, dim=1)
    value = value.repeat_interleave(2, dim=1)
    bias = torch.full((288, 288), -1e9)
    bias[:144, :144] = 0
    bias[144:, :144] = 0
    bias[144:, 144:] = torch.triu(torch.full((144, 144), -1e9), diagonal=1)
    attention = torch_functional.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=bias,
    )
    attention = attention.permute(0, 2, 1, 3).reshape(1, 288, 8)
    hidden = residual + _linear(attention, state, "blocks.0.attn.out_proj")
    hidden = hidden + _gated_mlp(
        _rms_norm(hidden, state, "blocks.0.norm2", 1e-6),
        state,
        "blocks.0.mlp",
    )
    expected = _rms_norm(hidden, state, "norm", 1e-6)[:, 144:]

    assert actual.shape == (1, 144, 8)
    assert np.isfinite(actual).all()
    assert any(node.op_type == "Attention" for node in graph)
    np.testing.assert_allclose(actual, expected.numpy(), rtol=5e-5, atol=5e-5)


def test_deepseek_ocr2_fp16_keeps_rotary_frequency_math_float32():
    module = DeepSeekOCR2QueryEncoder(
        depth=1,
        hidden_size=8,
        intermediate_size=12,
        num_heads=2,
        num_kv_heads=1,
        norm_eps=1e-6,
    )
    module.input_schema = (("sam_features", ir.DataType.FLOAT16, (1, 8, 12, 12)),)
    config = ArchitectureConfig(
        vocab_size=1,
        hidden_size=8,
        intermediate_size=12,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=1,
        hidden_act="silu",
        dtype=ir.DataType.FLOAT16,
    )
    model = build_from_module(
        GGUFVisionProjectorModel(module),
        config,
        task=GGUFVisionProjectorTask(),
    )["vision_encoder"]
    rng = np.random.default_rng(34)
    for initializer in model.graph.initializers.values():
        if initializer.const_value is not None:
            continue
        numpy_dtype = np.float32 if initializer.dtype == ir.DataType.FLOAT else np.float16
        initializer.const_value = ir.tensor(
            rng.normal(0.0, 0.02, tuple(initializer.shape)).astype(numpy_dtype)
        )
    session = ort.InferenceSession(
        ir.serde.serialize_model(model).SerializeToString(),
        providers=["CPUExecutionProvider"],
    )

    (actual,) = session.run(
        None,
        {"sam_features": rng.normal(size=(1, 8, 12, 12)).astype(np.float16)},
    )

    assert actual.shape == (1, 144, 8)
    assert actual.dtype == np.float16
    assert np.isfinite(actual).all()


class _FeatureStub(nn.Module):
    def __init__(self, tokens: int, hidden_size: int, offset: int):
        super().__init__()
        self._tokens = tokens
        self._hidden_size = hidden_size
        self._offset = offset

    def forward(self, op, pixel_values):
        batch = op.Squeeze(op.Shape(pixel_values, start=0, end=1))
        total = op.Mul(
            batch,
            op.Constant(value_int=self._tokens * self._hidden_size),
        )
        values = op.Range(
            op.Constant(value_int=self._offset),
            op.Add(total, op.Constant(value_int=self._offset)),
            op.Constant(value_int=1),
        )
        return op.Cast(
            op.Reshape(
                values,
                op.Concat(
                    op.Reshape(batch, [1]),
                    [self._tokens, self._hidden_size],
                    axis=0,
                ),
            ),
            to=ir.DataType.FLOAT,
        )


def test_deepseek_v1_full_media_order_and_newlines():
    module = DeepSeekOCRFullImageEncoder(
        sam_hidden_size=8,
        sam_num_heads=2,
        sam_depth=1,
        sam_window_size=2,
        clip_hidden_size=8,
        clip_intermediate_size=12,
        clip_num_heads=2,
        clip_depth=1,
        output_size=4,
    )
    module.global_encoder = _FeatureStub(256, 4, 10_000)
    module.local_encoder = _FeatureStub(100, 4, 0)
    global_pixels = np.zeros((1, 1), dtype=np.float32)
    local_pixels = np.zeros((1, 3, 640, 1280), dtype=np.float32)

    actual, state, _ = _run(
        module,
        {
            "global_pixel_values": global_pixels,
            "local_pixel_values": local_pixels,
        },
        seed=17,
    )
    local = np.arange(800, dtype=np.float32).reshape(2, 10, 10, 4)
    local = local.transpose(0, 1, 2, 3).reshape(1, 2, 10, 10, 4)
    local = local.transpose(0, 2, 1, 3, 4).reshape(10, 20, 4)
    newline = state["image_newline"].numpy()
    local = np.concatenate(
        (local, np.broadcast_to(newline, (10, 1, 4))),
        axis=1,
    ).reshape(-1, 4)
    overview = np.arange(10_000, 11_024, dtype=np.float32).reshape(16, 16, 4)
    overview = np.concatenate(
        (overview, np.broadcast_to(newline, (16, 1, 4))),
        axis=1,
    ).reshape(-1, 4)
    overview = np.concatenate(
        (overview, state["view_separator"].numpy()[None]),
        axis=0,
    )

    np.testing.assert_array_equal(actual, np.concatenate((local, overview), axis=0))


def test_deepseek_v2_full_media_order_and_global_separator():
    module = DeepSeekOCR2FullImageEncoder(
        sam_hidden_size=8,
        sam_num_heads=2,
        sam_depth=1,
        sam_window_size=2,
        hidden_size=8,
        intermediate_size=12,
        depth=1,
        num_heads=2,
        num_kv_heads=1,
        output_size=4,
        norm_eps=1e-6,
    )
    module.global_encoder = _FeatureStub(256, 4, 10_000)
    module.local_encoder = _FeatureStub(144, 4, 0)

    actual, state, _ = _run(
        module,
        {
            "global_pixel_values": np.zeros((1, 1), dtype=np.float32),
            "local_pixel_values": np.zeros((2, 1), dtype=np.float32),
        },
        seed=18,
    )
    local = np.arange(2 * 144 * 4, dtype=np.float32).reshape(-1, 4)
    overview = np.arange(10_000, 10_000 + 256 * 4, dtype=np.float32).reshape(-1, 4)
    overview = np.concatenate(
        (overview, state["view_separator"].numpy()[None]),
        axis=0,
    )

    np.testing.assert_array_equal(actual, np.concatenate((local, overview), axis=0))


def test_granite_anyres_assembly_adds_one_newline_per_unpadded_row():
    module = Granite4VisionEncoder(
        depth=1,
        hidden_size=64,
        intermediate_size=80,
        num_heads=4,
        image_size=8,
        patch_size=2,
        feature_layers=[0],
        spatial_offsets=[-1],
        query_side=1,
        window_side=2,
        output_size=4,
        qformer_intermediate_size=80,
        norm_eps=1e-6,
    )
    builder, op, graph = create_test_builder()
    features = create_test_input(builder, "features", [2, 4, 4])
    image_sizes = create_test_input(
        builder,
        "image_sizes",
        [1, 2],
        dtype=ir.DataType.INT64,
    )
    tile_grid = create_test_input(builder, "tile_grid", [2], dtype=ir.DataType.INT64)
    output = module._assemble_tiles(op, features, image_sizes, tile_grid)
    graph.outputs.append(output)
    newline = np.array([100, 101, 102, 103], dtype=np.float32)
    module.image_newline.const_value = ir.tensor(newline)
    graph.initializers["image_newline"] = module.image_newline
    session = ort.InferenceSession(
        ir.serde.serialize_model(ir.Model(graph, ir_version=11)).SerializeToString(),
        providers=["CPUExecutionProvider"],
    )
    input_features = np.arange(32, dtype=np.float32).reshape(2, 4, 4)
    actual = session.run(
        None,
        {
            "features": input_features,
            "image_sizes": np.array([[8, 8]], dtype=np.int64),
            "tile_grid": np.array([1, 1], dtype=np.int64),
        },
    )[0]
    tile = input_features[1].reshape(2, 2, 4)
    expected = np.concatenate(
        (tile, np.broadcast_to(newline, (2, 1, 4))),
        axis=1,
    ).reshape(-1, 4)

    np.testing.assert_array_equal(actual, expected)


def test_granite_full_sidecar_graph_executes_nonzero_tiles():
    module = Granite4VisionEncoder(
        depth=1,
        hidden_size=64,
        intermediate_size=80,
        num_heads=4,
        image_size=8,
        patch_size=2,
        feature_layers=[0],
        spatial_offsets=[-1],
        query_side=1,
        window_side=2,
        output_size=4,
        qformer_intermediate_size=80,
        norm_eps=1e-6,
    )
    pixels = np.random.default_rng(27).normal(size=(1, 2, 3, 8, 8)).astype(np.float32)

    actual, _, _ = _run(
        module,
        {
            "pixel_values": pixels,
            "image_sizes": np.array([[8, 8]], dtype=np.int64),
            "tile_grid": np.array([1, 1], dtype=np.int64),
        },
        seed=28,
    )

    # Four overview rows + four unpadded tile rows + two learned newlines.
    assert actual.shape == (10, 4)
    assert np.isfinite(actual).all()
    assert not np.allclose(actual, 0)


def test_granite_anyres_odd_padding_retains_symmetric_extent():
    module = Granite4VisionEncoder(
        depth=1,
        hidden_size=64,
        intermediate_size=80,
        num_heads=4,
        image_size=12,
        patch_size=2,
        feature_layers=[0],
        spatial_offsets=[-1],
        query_side=1,
        window_side=2,
        output_size=4,
        qformer_intermediate_size=80,
        norm_eps=1e-6,
    )
    builder, op, graph = create_test_builder()
    features = create_test_input(builder, "features", [2, 9, 4])
    image_sizes = create_test_input(
        builder,
        "image_sizes",
        [1, 2],
        dtype=ir.DataType.INT64,
    )
    tile_grid = create_test_input(builder, "tile_grid", [2], dtype=ir.DataType.INT64)
    output = module._assemble_tiles(op, features, image_sizes, tile_grid)
    graph.outputs.append(output)
    newline = np.array([100, 101, 102, 103], dtype=np.float32)
    module.image_newline.const_value = ir.tensor(newline)
    graph.initializers["image_newline"] = module.image_newline
    session = ort.InferenceSession(
        ir.serde.serialize_model(ir.Model(graph, ir_version=11)).SerializeToString(),
        providers=["CPUExecutionProvider"],
    )
    input_features = np.arange(72, dtype=np.float32).reshape(2, 9, 4)
    actual = session.run(
        None,
        {
            "features": input_features,
            "image_sizes": np.array([[2, 3]], dtype=np.int64),
            "tile_grid": np.array([1, 1], dtype=np.int64),
        },
    )[0]
    tile = input_features[1].reshape(3, 3, 4)
    expected = np.concatenate(
        (tile, np.broadcast_to(newline, (3, 1, 4))),
        axis=1,
    ).reshape(-1, 4)

    # floor((3 - 2) / 2) == 0, so symmetric cropping retains all three rows.
    np.testing.assert_array_equal(actual, expected)
