# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Synthetic prefill and cached-decode parity for exact PLaMo GGUF graphs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as functional

from mobius._configs import ArchitectureConfig
from mobius._testing.ort_inference import OnnxModelSession
from mobius.models.gguf_plamo import PlamoGGUFCausalLMModel
from mobius.tasks import PlamoCausalLMTask


@dataclass
class _ReferenceResult:
    logits: torch.Tensor
    cache: list[tuple[torch.Tensor, torch.Tensor]]


def _config() -> ArchitectureConfig:
    config = ArchitectureConfig(
        model_type="plamo",
        vocab_size=19,
        hidden_size=8,
        intermediate_size=12,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=2,
        hidden_act="silu",
        pad_token_id=0,
        max_position_embeddings=16,
        rms_norm_eps=1e-6,
        rope_type="default",
        rope_theta=10_000.0,
        tie_word_embeddings=False,
    )
    config._gguf_arch = "plamo"
    return config


def _weights(package) -> dict[str, torch.Tensor]:
    rng = np.random.default_rng(314159)
    weights: dict[str, torch.Tensor] = {}
    for name, initializer in package["model"].graph.initializers.items():
        if name.startswith("const_") or ".rotary_emb." in name:
            continue
        shape = tuple(initializer.shape)
        if name.endswith("layernorm.weight") or name == "model.norm.weight":
            value = 0.9 + 0.2 * rng.random(shape, dtype=np.float32)
        else:
            value = rng.uniform(-0.15, 0.15, shape).astype(np.float32)
        weights[name] = torch.from_numpy(value)
    return weights


def _norm(x: torch.Tensor, weight: torch.Tensor, epsilon: float) -> torch.Tensor:
    return x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + epsilon) * weight


def _linear(x: torch.Tensor, weights: dict[str, torch.Tensor], name: str) -> torch.Tensor:
    return functional.linear(x, weights[f"{name}.weight"])


def _rope(x: torch.Tensor, positions: torch.Tensor, theta: float) -> torch.Tensor:
    head_dim = x.shape[-1]
    inv_freq = theta ** (-torch.arange(0, head_dim, 2, dtype=x.dtype) / head_dim)
    angles = positions.to(x.dtype)[..., None] * inv_freq
    cos = angles.cos()[:, :, None, :]
    sin = angles.sin()[:, :, None, :]
    first, second = x[..., : head_dim // 2], x[..., head_dim // 2 :]
    return torch.cat((first * cos - second * sin, second * cos + first * sin), dim=-1)


def _reference(
    config: ArchitectureConfig,
    weights: dict[str, torch.Tensor],
    input_ids: torch.Tensor,
    position_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    past: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
) -> _ReferenceResult:
    hidden = functional.embedding(input_ids, weights["model.embed_tokens.weight"])
    batch, sequence = input_ids.shape
    repeat = config.num_attention_heads // config.num_key_value_heads
    presents = []
    past = past or [None] * config.num_hidden_layers
    for layer, layer_past in enumerate(past):
        prefix = f"model.layers.{layer}"
        normalized = _norm(
            hidden,
            weights[f"{prefix}.input_layernorm.weight"],
            config.rms_norm_eps,
        )
        attn = f"{prefix}.self_attn"
        query = _linear(normalized, weights, f"{attn}.q_proj").reshape(
            batch, sequence, config.num_attention_heads, config.head_dim
        )
        key = _linear(normalized, weights, f"{attn}.k_proj").reshape(
            batch, sequence, config.num_key_value_heads, config.head_dim
        )
        value = _linear(normalized, weights, f"{attn}.v_proj").reshape(
            batch, sequence, config.num_key_value_heads, config.head_dim
        )
        # torch.repeat matches PLaMo's cyclic head expansion, unlike repeat_interleave.
        key = key.repeat(1, 1, repeat, 1)
        value = value.repeat(1, 1, repeat, 1)
        query = _rope(query, position_ids, config.rope_theta)
        key = _rope(key, position_ids, config.rope_theta)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        if layer_past is not None:
            key = torch.cat((layer_past[0], key), dim=2)
            value = torch.cat((layer_past[1], value), dim=2)
        presents.append((key, value))

        scores = torch.matmul(query.transpose(1, 2), key.transpose(-1, -2))
        scores = scores * config.head_dim**-0.5
        key_positions = torch.arange(key.shape[2], dtype=torch.int64)
        causal = key_positions[None, None, :] <= position_ids[..., None]
        valid = attention_mask.bool().reshape(batch, 1, 1, -1)
        scores = scores.masked_fill(~(causal[:, None, :, :] & valid), float("-inf"))
        context = torch.matmul(scores.softmax(dim=-1), value)
        attention_output = _linear(
            context.transpose(1, 2).reshape(batch, sequence, config.hidden_size),
            weights,
            f"{attn}.o_proj",
        )
        mlp = f"{prefix}.mlp"
        gate = functional.silu(_linear(normalized, weights, f"{mlp}.gate_proj"))
        up = _linear(normalized, weights, f"{mlp}.up_proj")
        mlp_output = _linear(gate * up, weights, f"{mlp}.down_proj")
        hidden = hidden + attention_output + mlp_output

    hidden = _norm(hidden, weights["model.norm.weight"], config.rms_norm_eps)
    return _ReferenceResult(_linear(hidden, weights, "lm_head"), presents)


def _empty_cache(config: ArchitectureConfig, *, batch_size: int = 1) -> dict[str, np.ndarray]:
    empty = np.empty(
        (batch_size, config.num_attention_heads, 0, config.head_dim),
        dtype=np.float32,
    )
    return {
        "past_key_values.0.key": empty,
        "past_key_values.0.value": empty.copy(),
    }


def test_plamo_prefill_and_expanded_cached_decode_match_reference() -> None:
    config = _config()
    module = PlamoGGUFCausalLMModel(config)
    assert module.kv_cache_specs() == [(4, 2)]
    package = PlamoCausalLMTask().build(module, config)
    graph = package["model"].graph
    assert "model.layers.0.post_attention_layernorm.weight" not in graph.initializers
    assert [node.op_type for node in graph if node.op_type == "RMSNormalization"] == [
        "RMSNormalization",
        "RMSNormalization",
    ]

    weights = _weights(package)
    package.apply_weights(weights)
    session = OnnxModelSession(package)

    input_ids = torch.tensor([[2, 7, 4]], dtype=torch.int64)
    positions = torch.arange(3, dtype=torch.int64).unsqueeze(0)
    mask = torch.ones((1, 3), dtype=torch.int64)
    reference_prefill = _reference(config, weights, input_ids, positions, mask)
    ort_prefill = session.run(
        {
            "input_ids": input_ids.numpy(),
            "position_ids": positions.numpy(),
            "attention_mask": mask.numpy(),
            **_empty_cache(config),
        }
    )
    np.testing.assert_allclose(
        ort_prefill["logits"],
        reference_prefill.logits.numpy(),
        rtol=2e-4,
        atol=2e-5,
        strict=True,
    )
    assert ort_prefill["present.0.key"].shape == (1, 4, 3, 2)
    assert ort_prefill["present.0.value"].shape == (1, 4, 3, 2)

    decode_ids = torch.tensor([[11]], dtype=torch.int64)
    decode_positions = torch.tensor([[3]], dtype=torch.int64)
    decode_mask = torch.ones((1, 4), dtype=torch.int64)
    reference_decode = _reference(
        config,
        weights,
        decode_ids,
        decode_positions,
        decode_mask,
        reference_prefill.cache,
    )
    ort_decode = session.run(
        {
            "input_ids": decode_ids.numpy(),
            "position_ids": decode_positions.numpy(),
            "attention_mask": decode_mask.numpy(),
            "past_key_values.0.key": ort_prefill["present.0.key"],
            "past_key_values.0.value": ort_prefill["present.0.value"],
        }
    )
    np.testing.assert_allclose(
        ort_decode["logits"],
        reference_decode.logits.numpy(),
        rtol=2e-4,
        atol=2e-5,
        strict=True,
    )


def test_plamo_prefill_matches_reference_for_multiple_batches() -> None:
    config = _config()
    package = PlamoCausalLMTask().build(PlamoGGUFCausalLMModel(config), config)
    weights = _weights(package)
    package.apply_weights(weights)
    session = OnnxModelSession(package)

    input_ids = torch.tensor([[2, 7, 4], [5, 1, 9]], dtype=torch.int64)
    positions = torch.arange(3, dtype=torch.int64).expand(2, -1)
    mask = torch.ones((2, 3), dtype=torch.int64)
    reference = _reference(config, weights, input_ids, positions, mask)
    actual = session.run(
        {
            "input_ids": input_ids.numpy(),
            "position_ids": positions.numpy(),
            "attention_mask": mask.numpy(),
            **_empty_cache(config, batch_size=2),
        }
    )

    np.testing.assert_allclose(
        actual["logits"],
        reference.logits.numpy(),
        rtol=2e-4,
        atol=2e-5,
        strict=True,
    )
