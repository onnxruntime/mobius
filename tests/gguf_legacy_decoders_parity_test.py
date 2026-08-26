# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Synthetic value parity for the exact conventional GGUF decoder families."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pytest
import torch
import torch.nn.functional as functional

from mobius._configs import ArchitectureConfig
from mobius._testing.ort_inference import OnnxModelSession
from mobius.models.gguf_legacy_decoders import ExactLegacyGGUFCausalLMModel
from mobius.tasks import CausalLMTask

_ARCHITECTURES = ("gptneox", "jais", "mpt", "refact", "ernie4_5", "openelm")
_LAYER_NORM_ARCHES = {"gptneox", "jais", "mpt"}
_GATED_MLP_ARCHES = {"jais", "refact", "ernie4_5", "openelm"}


@dataclass
class _ReferenceResult:
    logits: torch.Tensor
    cache: list[tuple[torch.Tensor, torch.Tensor]]


def _config(architecture: str) -> ArchitectureConfig:
    openelm = architecture == "openelm"
    config = ArchitectureConfig(
        model_type=architecture,
        vocab_size=23,
        hidden_size=8,
        intermediate_size=12,
        num_hidden_layers=2 if openelm else 1,
        num_attention_heads=2,
        num_key_value_heads=1 if architecture in {"refact", "openelm"} else 2,
        head_dim=4,
        hidden_act="gelu" if architecture in {"gptneox", "mpt"} else "silu",
        pad_token_id=0,
        max_position_embeddings=16,
        rms_norm_eps=1e-5,
        rope_type="default" if architecture in {"gptneox", "ernie4_5", "openelm"} else None,
        rope_theta=10_000.0,
        partial_rotary_factor=0.5 if architecture == "gptneox" else 1.0,
        alibi_max_bias=8.0 if architecture in {"jais", "mpt", "refact"} else None,
        attention_scale=0.25 if architecture == "jais" else None,
        use_parallel_residual=architecture == "gptneox",
        attn_qkv_bias=architecture in {"gptneox", "jais"},
        attn_o_bias=architecture in {"gptneox", "jais"},
        mlp_bias=architecture in {"gptneox", "jais"},
        tie_word_embeddings=architecture in {"mpt", "refact", "openelm"},
        layer_attention_head_counts=(2, 1) if openelm else (),
        layer_attention_kv_head_counts=(1, 1) if openelm else (),
        layer_intermediate_sizes=(12, 16) if openelm else (),
    )
    config._gguf_arch = architecture
    return config


def _deterministic_weights(
    package, module: ExactLegacyGGUFCausalLMModel, config: ArchitectureConfig
) -> dict[str, torch.Tensor]:
    rng = np.random.default_rng(1701 + sum(map(ord, config._gguf_arch)))
    weights: dict[str, torch.Tensor] = {}
    for name, initializer in package["model"].graph.initializers.items():
        if name.startswith("const_") or ".rotary_emb." in name:
            continue
        shape = tuple(initializer.shape)
        if name.endswith(".weight") and ("layernorm" in name or name == "model.norm.weight"):
            value = 0.85 + 0.2 * rng.random(shape, dtype=np.float32)
        elif name.endswith(".bias"):
            value = rng.uniform(-0.025, 0.025, shape).astype(np.float32)
        else:
            value = rng.uniform(-0.16, 0.16, shape).astype(np.float32)
        weights[name] = torch.from_numpy(value)

    # Exercise the GGUF fused-row contract rather than loading separate Q/K/V tensors.
    for layer in range(config.num_hidden_layers):
        prefix = f"model.layers.{layer}.self_attn"
        q_name, k_name, v_name = (
            f"{prefix}.q_proj.weight",
            f"{prefix}.k_proj.weight",
            f"{prefix}.v_proj.weight",
        )
        fused = torch.cat(
            [weights.pop(q_name), weights.pop(k_name), weights.pop(v_name)], dim=0
        )
        weights[f"{prefix}.qkv_proj.weight"] = fused
        if config.attn_qkv_bias:
            q_name, k_name, v_name = (
                f"{prefix}.q_proj.bias",
                f"{prefix}.k_proj.bias",
                f"{prefix}.v_proj.bias",
            )
            weights[f"{prefix}.qkv_proj.bias"] = torch.cat(
                [weights.pop(q_name), weights.pop(k_name), weights.pop(v_name)], dim=0
            )

    processed = module.preprocess_weights(weights)
    for layer in range(config.num_hidden_layers):
        prefix = f"model.layers.{layer}.self_attn"
        fused = weights[f"{prefix}.qkv_proj.weight"]
        q_rows = (
            config.layer_attention_head_counts[layer]
            if config.layer_attention_head_counts
            else config.num_attention_heads
        ) * config.head_dim
        kv_rows = (
            config.layer_attention_kv_head_counts[layer]
            if config.layer_attention_kv_head_counts
            else config.num_key_value_heads
        ) * config.head_dim
        torch.testing.assert_close(processed[f"{prefix}.q_proj.weight"], fused[:q_rows])
        torch.testing.assert_close(
            processed[f"{prefix}.k_proj.weight"], fused[q_rows : q_rows + kv_rows]
        )
        torch.testing.assert_close(
            processed[f"{prefix}.v_proj.weight"], fused[q_rows + kv_rows :]
        )
    return processed


def _norm(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None, eps: float
) -> torch.Tensor:
    if bias is not None:
        return functional.layer_norm(x, (x.shape[-1],), weight, bias, eps)
    return x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + eps) * weight


def _linear(x: torch.Tensor, weights: dict[str, torch.Tensor], prefix: str) -> torch.Tensor:
    return functional.linear(x, weights[f"{prefix}.weight"], weights.get(f"{prefix}.bias"))


def _rope(
    x: torch.Tensor,
    positions: torch.Tensor,
    rotary_dim: int,
    theta: float,
    *,
    interleaved: bool,
) -> torch.Tensor:
    if rotary_dim == 0:
        return x
    inv_freq = theta ** (-torch.arange(0, rotary_dim, 2, dtype=x.dtype) / float(rotary_dim))
    angles = positions.to(x.dtype)[..., None] * inv_freq
    cos, sin = angles.cos()[:, :, None, :], angles.sin()[:, :, None, :]
    rotated, tail = x[..., :rotary_dim], x[..., rotary_dim:]
    if interleaved:
        pairs = rotated.reshape(*rotated.shape[:-1], rotary_dim // 2, 2)
        first, second = pairs[..., 0], pairs[..., 1]
        rotated = torch.stack(
            (first * cos - second * sin, second * cos + first * sin), dim=-1
        ).flatten(-2)
    else:
        half = rotary_dim // 2
        first, second = rotated[..., :half], rotated[..., half:]
        rotated = torch.cat(
            (first * cos - second * sin, second * cos + first * sin), dim=-1
        )
    return torch.cat((rotated, tail), dim=-1)


def _alibi_slopes(num_heads: int, max_bias: float) -> torch.Tensor:
    power = 1 << math.floor(math.log2(num_heads))
    first = 2.0 ** (-max_bias / power)
    second = 2.0 ** (-(max_bias / 2.0) / power)
    return torch.tensor(
        [
            first ** (head + 1) if head < power else second ** (2 * (head - power) + 1)
            for head in range(num_heads)
        ],
        dtype=torch.float32,
    )


def _reference(
    config: ArchitectureConfig,
    weights: dict[str, torch.Tensor],
    input_ids: torch.Tensor,
    position_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    past: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
) -> _ReferenceResult:
    architecture = config._gguf_arch
    hidden = functional.embedding(input_ids, weights["model.embed_tokens.weight"])
    presents: list[tuple[torch.Tensor, torch.Tensor]] = []
    past = past or [None] * config.num_hidden_layers

    for layer, layer_past in enumerate(past):
        prefix = f"model.layers.{layer}"
        heads = (
            config.layer_attention_head_counts[layer]
            if config.layer_attention_head_counts
            else config.num_attention_heads
        )
        kv_heads = (
            config.layer_attention_kv_head_counts[layer]
            if config.layer_attention_kv_head_counts
            else config.num_key_value_heads
        )
        intermediate = (
            config.layer_intermediate_sizes[layer]
            if config.layer_intermediate_sizes
            else config.intermediate_size
        )
        del intermediate  # The loaded projection shapes pin this independently.
        norm_bias = weights.get(f"{prefix}.input_layernorm.bias")
        attention_input = _norm(
            hidden,
            weights[f"{prefix}.input_layernorm.weight"],
            norm_bias,
            config.rms_norm_eps,
        )
        attn_prefix = f"{prefix}.self_attn"
        query = _linear(attention_input, weights, f"{attn_prefix}.q_proj")
        key = _linear(attention_input, weights, f"{attn_prefix}.k_proj")
        value = _linear(attention_input, weights, f"{attn_prefix}.v_proj")
        batch, query_length = input_ids.shape
        query = query.reshape(batch, query_length, heads, config.head_dim)
        key = key.reshape(batch, query_length, kv_heads, config.head_dim)
        value = value.reshape(batch, query_length, kv_heads, config.head_dim)

        if architecture == "openelm":
            query = _norm(
                query, weights[f"{attn_prefix}.q_norm.weight"], None, config.rms_norm_eps
            )
            key = _norm(
                key, weights[f"{attn_prefix}.k_norm.weight"], None, config.rms_norm_eps
            )

        rotary_dim = (
            int(config.head_dim * config.partial_rotary_factor)
            if config.rope_type is not None
            else 0
        )
        query = _rope(
            query,
            position_ids,
            rotary_dim,
            config.rope_theta,
            interleaved=config.rope_interleave,
        )
        key = _rope(
            key,
            position_ids,
            rotary_dim,
            config.rope_theta,
            interleaved=config.rope_interleave,
        )
        if layer_past is not None:
            key = torch.cat((layer_past[0], key.transpose(1, 2)), dim=2)
            value = torch.cat((layer_past[1], value.transpose(1, 2)), dim=2)
        else:
            key, value = key.transpose(1, 2), value.transpose(1, 2)
        presents.append((key, value))

        query = query.transpose(1, 2)
        key_for_attention = key.repeat_interleave(heads // kv_heads, dim=1)
        value_for_attention = value.repeat_interleave(heads // kv_heads, dim=1)
        scale = config.attention_scale or config.head_dim**-0.5
        scores = torch.matmul(query, key_for_attention.transpose(-1, -2)) * scale
        key_positions = torch.arange(key.shape[2], dtype=torch.int64)
        causal = key_positions[None, :] <= position_ids.reshape(-1, 1)
        valid = attention_mask.bool().reshape(batch, 1, 1, -1)
        scores = scores.masked_fill(~(causal[None, None, :, :] & valid), float("-inf"))
        if config.alibi_max_bias is not None:
            distance = position_ids.to(torch.float32).reshape(
                batch, 1, query_length, 1
            ) - key_positions.to(torch.float32).reshape(1, 1, 1, -1)
            scores = (
                scores
                - _alibi_slopes(heads, config.alibi_max_bias).reshape(1, heads, 1, 1)
                * distance.abs()
            )
        context = torch.matmul(scores.softmax(dim=-1), value_for_attention)
        attention_output = _linear(
            context.transpose(1, 2).reshape(batch, query_length, heads * config.head_dim),
            weights,
            f"{attn_prefix}.o_proj",
        )

        residual = hidden
        if config.use_parallel_residual:
            mlp_input = _norm(
                residual,
                weights[f"{prefix}.post_attention_layernorm.weight"],
                weights.get(f"{prefix}.post_attention_layernorm.bias"),
                config.rms_norm_eps,
            )
        else:
            hidden = residual + attention_output
            residual = hidden
            mlp_input = _norm(
                hidden,
                weights[f"{prefix}.post_attention_layernorm.weight"],
                weights.get(f"{prefix}.post_attention_layernorm.bias"),
                config.rms_norm_eps,
            )
        mlp_prefix = f"{prefix}.mlp"
        up = _linear(mlp_input, weights, f"{mlp_prefix}.up_proj")
        if architecture in _GATED_MLP_ARCHES:
            gate = _linear(mlp_input, weights, f"{mlp_prefix}.gate_proj")
            mlp_output = _linear(
                functional.silu(gate) * up, weights, f"{mlp_prefix}.down_proj"
            )
        else:
            mlp_output = _linear(functional.gelu(up), weights, f"{mlp_prefix}.down_proj")
        hidden = (
            residual + attention_output + mlp_output
            if config.use_parallel_residual
            else residual + mlp_output
        )

    hidden = _norm(
        hidden,
        weights["model.norm.weight"],
        weights.get("model.norm.bias"),
        config.rms_norm_eps,
    )
    head = (
        weights["model.embed_tokens.weight"]
        if config.tie_word_embeddings
        else weights["lm_head.weight"]
    )
    return _ReferenceResult(functional.linear(hidden, head), presents)


def _empty_cache_feeds(
    session: OnnxModelSession, config: ArchitectureConfig
) -> dict[str, np.ndarray]:
    feeds: dict[str, np.ndarray] = {}
    for layer in range(config.num_hidden_layers):
        kv_heads = (
            config.layer_attention_kv_head_counts[layer]
            if config.layer_attention_kv_head_counts
            else config.num_key_value_heads
        )
        empty = np.empty((1, kv_heads, 0, config.head_dim), dtype=np.float32)
        feeds[f"past_key_values.{layer}.key"] = empty
        feeds[f"past_key_values.{layer}.value"] = empty.copy()
    assert set(feeds) < set(session.input_names)
    return feeds


@pytest.mark.parametrize("architecture", _ARCHITECTURES)
def test_legacy_gguf_decoder_prefill_and_cached_decode_logits(architecture: str) -> None:
    """Full logits match independent equations for prefill and a reused-cache step."""
    torch.manual_seed(0)
    config = _config(architecture)
    module = ExactLegacyGGUFCausalLMModel(config)
    package = CausalLMTask().build(module, config)
    weights = _deterministic_weights(package, module, config)
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
            **_empty_cache_feeds(session, config),
        }
    )
    np.testing.assert_allclose(
        ort_prefill["logits"],
        reference_prefill.logits.numpy(),
        rtol=2e-4,
        atol=2e-5,
        strict=True,
    )

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
    decode_feeds = {
        "input_ids": decode_ids.numpy(),
        "position_ids": decode_positions.numpy(),
        "attention_mask": decode_mask.numpy(),
    }
    for layer in range(config.num_hidden_layers):
        decode_feeds[f"past_key_values.{layer}.key"] = ort_prefill[f"present.{layer}.key"]
        decode_feeds[f"past_key_values.{layer}.value"] = ort_prefill[f"present.{layer}.value"]
    ort_decode = session.run(decode_feeds)
    np.testing.assert_allclose(
        ort_decode["logits"],
        reference_decode.logits.numpy(),
        rtol=2e-4,
        atol=2e-5,
        strict=True,
    )

    # Make the intended architecture matrix explicit instead of relying on parametrization names.
    assert (architecture in _LAYER_NORM_ARCHES) == ("model.norm.bias" in weights)
