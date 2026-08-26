# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Exact Qwen3.8 Flash-Next text core (HuggingFace ``Qwen4ExpForCausalLM``).

The decoder combines a 3:1 Gated-DeltaNet/Qwen Sparse Attention schedule,
four-stream gated residual hyper-connections, hashed n-gram Per-Layer
Embeddings (PLE), and a routed-plus-shared MoE in every layer. The exported
state ABI is defined by :class:`mobius.tasks.Qwen4ExpCausalLMTask`.
"""

from __future__ import annotations

import logging
import math
from collections import Counter, defaultdict
from collections.abc import Mapping
from typing import ClassVar

import numpy as np
import onnx_ir as ir
import torch
from onnxscript import OpBuilder, nn

from mobius._configs import Qwen4ExpConfig
from mobius._weight_utils import preprocess_quantized_weights
from mobius.components import (
    Embedding,
    GatedDeltaNet,
    Linear,
    Qwen35Attention,
    SoftmaxTopKGate,
    apply_rotary_pos_emb,
    create_attention_bias,
    get_activation,
    initialize_rope,
)
from mobius.models.base import effective_tie_word_embeddings
from mobius.models.moe import Qwen2MoELayer
from mobius.models.qwen_vl import Qwen3VLVisionEncoderModel, Qwen25VLEmbeddingModel

_INT64_MAX = 9223372036854775807
_MASK64 = (1 << 64) - 1
_SPLITMIX_GAMMA = 0x9E3779B97F4A7C15
_SPLITMIX_M1 = 0xBF58476D1CE4E5B9
_SPLITMIX_M2 = 0x94D049BB133111EB
_PRIME_1 = 10007
_PINNED_PLE_WEIGHT_SCALE = 0.00019931793212890625
logger = logging.getLogger(__name__)


def _source_name_for_qwen4_exp_target(target_name: str) -> str:
    if target_name.startswith("model."):
        return f"model.language_model.{target_name[len('model.') :]}"
    return target_name


def _qwen4_exp_ple_scale_name(weight_name: str) -> str:
    marker = ".ngram_embedding.shard_"
    if marker not in weight_name or not weight_name.endswith(".weight"):
        raise ValueError(f"Not a Qwen4-Exp PLE shard weight: {weight_name}")
    prefix, _suffix = weight_name.split(marker, 1)
    return f"{prefix}.ngram_embedding.weight_scale"


def validate_qwen4_exp_fp8_header_contract(
    key_index: Mapping[str, tuple[str, list[int], str]],
) -> dict[str, int]:
    """Validate every FP8 weight and inverse scale in a Qwen4-Exp checkpoint."""
    scaled = 0
    scalar_scaled_ple = 0
    consumed_scales: set[str] = set()
    for weight_name, (_path, weight_shape, weight_dtype) in key_index.items():
        if not weight_dtype.startswith("F8"):
            continue
        if weight_dtype != "F8_E4M3":
            raise ValueError(
                f"Qwen4-Exp FP8 source '{weight_name}' has dtype {weight_dtype}; "
                "the pinned checkpoint requires F8_E4M3"
            )
        if ".ple.ple_embedding.ngram_embedding.shard_" in weight_name:
            if len(weight_shape) != 2:
                raise ValueError(
                    f"Qwen4-Exp scalar-scaled PLE source '{weight_name}' must be 2-D"
                )
            per_shard_scale = weight_name[: -len(".weight")] + ".weight_scale_inv"
            if per_shard_scale in key_index:
                raise ValueError(
                    f"Qwen4-Exp PLE source '{weight_name}' has forbidden per-shard "
                    f"scale '{per_shard_scale}'; the pinned layout uses one shared scalar"
                )
            ple_scale_name = _qwen4_exp_ple_scale_name(weight_name)
            ple_scale = key_index.get(ple_scale_name)
            if ple_scale is None:
                raise ValueError(
                    f"Qwen4-Exp PLE source '{weight_name}' is missing shared scalar "
                    f"'{ple_scale_name}'"
                )
            _ple_scale_path, ple_scale_shape, ple_scale_dtype = ple_scale
            if ple_scale_shape != [1] or ple_scale_dtype != "BF16":
                raise ValueError(
                    f"Qwen4-Exp PLE scalar '{ple_scale_name}' has dtype/shape "
                    f"{ple_scale_dtype}/{ple_scale_shape}; expected BF16/[1]"
                )
            consumed_scales.add(ple_scale_name)
            scalar_scaled_ple += 1
            continue

        scale_name = (
            weight_name[: -len(".weight")] + ".weight_scale_inv"
            if weight_name.endswith(".weight")
            else weight_name + "_scale_inv"
        )
        scale_entry = key_index.get(scale_name)
        if scale_entry is None:
            raise ValueError(f"Qwen4-Exp FP8 source '{weight_name}' has no scale")
        _scale_path, scale_shape, scale_dtype = scale_entry
        if len(weight_shape) != 2:
            raise ValueError(f"Qwen4-Exp scaled FP8 source '{weight_name}' must be 2-D")
        expected_grid = [
            (weight_shape[0] + 127) // 128,
            (weight_shape[1] + 127) // 128,
        ]
        if scale_shape != expected_grid:
            raise ValueError(
                f"Qwen4-Exp FP8 source '{weight_name}' has scale grid {scale_shape}; "
                f"expected {expected_grid} for strict 128x128 blocks"
            )
        if scale_dtype != "BF16":
            raise ValueError(
                f"Qwen4-Exp FP8 source '{weight_name}' has scale dtype {scale_dtype}; "
                "the pinned checkpoint requires BF16 inverse scales"
            )
        consumed_scales.add(scale_name)
        scaled += 1

    all_scales = {
        name for name in key_index if name.endswith((".weight_scale_inv", ".weight_scale"))
    }
    orphan_scales = sorted(all_scales - consumed_scales)
    if orphan_scales:
        raise ValueError(
            f"Qwen4-Exp checkpoint has {len(orphan_scales)} orphan inverse scale(s), "
            f"e.g. {orphan_scales[:5]}"
        )
    return {
        "checkpoint_scaled_fp8_tensors": scaled,
        "checkpoint_scalar_scaled_fp8_tensors": scalar_scaled_ple,
    }


def _splitmix64(value: int) -> int:
    value = (value + _SPLITMIX_GAMMA) & _MASK64
    value = ((value ^ (value >> 30)) * _SPLITMIX_M1) & _MASK64
    value = ((value ^ (value >> 27)) * _SPLITMIX_M2) & _MASK64
    return (value ^ (value >> 31)) & _MASK64


def _build_layer_multipliers(
    unigram_vocab_size: int, ngram_size: int, ple_layer_index: int, seed: int
) -> np.ndarray:
    max_long = (1 << 63) - 1
    multiplier_max = max_long // max(unigram_vocab_size, 1)
    half_bound = max(1, multiplier_max // 2)
    base_seed = seed + _PRIME_1 * ple_layer_index
    multipliers = []
    for index in range(ngram_size):
        value = (base_seed + _SPLITMIX_GAMMA * (index + 1)) & _MASK64
        multipliers.append(2 * (_splitmix64(value) % half_bound) + 1)
    return np.asarray(multipliers, dtype=np.int64)


def _is_prime(value: int) -> bool:
    if value < 2:
        return False
    if value % 2 == 0:
        return value == 2
    return all(value % divisor for divisor in range(3, math.isqrt(value) + 1, 2))


def _find_nth_prime_after(start: int, count: int) -> int:
    prime = start
    for _ in range(count):
        prime += 1
        while not _is_prime(prime):
            prime += 1
    return prime


def _qwen4_exp_ple_buffer_values(
    config: Qwen4ExpConfig,
    ple_layer_index: int,
) -> tuple[dict[str, np.ndarray], int]:
    """Return deterministic PLE buffers and the padded embedding vocabulary size."""
    ngram_heads = (config.ngram_size - 1) * config.heads_per_ngram
    head_vocab_sizes: list[int] = []
    head_offsets: list[int] = []
    total_vocab_size = 0
    for head_idx in range(ngram_heads):
        global_head_idx = ple_layer_index * ngram_heads + head_idx
        size = _find_nth_prime_after(
            config.ngram_vocab_size_base - 1,
            global_head_idx + 1,
        )
        head_vocab_sizes.append(size)
        head_offsets.append(total_vocab_size)
        total_vocab_size += size
    divisor = config.make_ngram_vocab_size_divisible_by
    padded_vocab_size = math.ceil(total_vocab_size / divisor) * divisor
    return (
        {
            "layer_multipliers": _build_layer_multipliers(
                config.vocab_size,
                config.ngram_size,
                ple_layer_index,
                config.seed,
            ),
            "ngram_heads_vocab_sizes": np.asarray(
                head_vocab_sizes,
                dtype=np.int64,
            ),
            "ngram_heads_offsets": np.asarray(head_offsets, dtype=np.int64),
        },
        padded_vocab_size,
    )


class Qwen4ExpOffsetRMSNorm(nn.Module):
    """Qwen4 offset RMSNorm with upstream float32 normalization/scaling."""

    def __init__(self, hidden_size: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter([hidden_size])
        self._eps = eps

    def forward(self, op: OpBuilder, hidden_states: ir.Value) -> ir.Value:
        hidden_f32 = op.Cast(hidden_states, to=ir.DataType.FLOAT)
        variance = op.ReduceMean(op.Mul(hidden_f32, hidden_f32), [-1], keepdims=True)
        normalized = op.Mul(
            hidden_f32,
            op.Reciprocal(op.Sqrt(op.Add(variance, self._eps))),
        )
        scale = op.Add(
            op.Cast(self.weight, to=ir.DataType.FLOAT),
            op.Constant(value_float=1.0),
        )
        return op.CastLike(op.Mul(normalized, scale), hidden_states)


class Qwen4ExpGroupedRMSNorm(nn.Module):
    """Offset RMSNorm that normalizes each residual stream independently."""

    def __init__(self, hidden_size: int, group_size: int, eps: float):
        super().__init__()
        if hidden_size % group_size:
            raise ValueError("Qwen4-Exp grouped RMSNorm size must divide hidden_size")
        self.weight = nn.Parameter([hidden_size])
        self._groups = hidden_size // group_size
        self._group_size = group_size
        self._eps = eps

    def forward(self, op: OpBuilder, hidden_states: ir.Value) -> ir.Value:
        original_shape = op.Shape(hidden_states)
        grouped = op.Reshape(
            op.Cast(hidden_states, to=ir.DataType.FLOAT),
            [-1, self._groups, self._group_size],
        )
        variance = op.ReduceMean(op.Mul(grouped, grouped), [-1], keepdims=True)
        normalized = op.Mul(
            grouped,
            op.Reciprocal(op.Sqrt(op.Add(variance, self._eps))),
        )
        grouped_weight = op.Reshape(
            op.Add(
                op.Cast(self.weight, to=ir.DataType.FLOAT),
                op.Constant(value_float=1.0),
            ),
            [self._groups, self._group_size],
        )
        return op.CastLike(
            op.Reshape(op.Mul(normalized, grouped_weight), original_shape),
            hidden_states,
        )


class Qwen4ExpGatedResidual(nn.Module):
    """Low-rank read/inject mixer for Qwen4-Exp residual streams."""

    def __init__(self, config: Qwen4ExpConfig, *, use_combine: bool = True):
        super().__init__()
        self._hc_count = config.hc_count
        self._hidden_size = config.hidden_size
        hc_hidden_size = config.hc_count * config.hidden_size
        self.hc_norm = Qwen4ExpGroupedRMSNorm(
            hc_hidden_size, config.hidden_size, config.rms_norm_eps
        )
        self.input_mix_weight_down = Linear(hc_hidden_size, config.hc_lowrank, bias=False)
        self.input_mix_weight_up = Linear(config.hc_lowrank, hc_hidden_size, bias=False)
        self.block_inject_weight = (
            Linear(hc_hidden_size, config.hc_count, bias=False) if use_combine else None
        )

    def forward(self, op: OpBuilder, hyper_input: ir.Value):
        # hyper_input: (B, S, hc_count * hidden_size)
        normalized = self.hc_norm(op, hyper_input)
        read = self.input_mix_weight_down(op, normalized)
        read = op.Swish(op.Div(read, float(self._hc_count)))
        read = op.Sigmoid(self.input_mix_weight_up(op, read))
        read = op.Reshape(read, [0, 0, self._hc_count, self._hidden_size])
        streams = op.Reshape(normalized, [0, 0, self._hc_count, self._hidden_size])
        mixed = op.ReduceMean(op.Mul(read, streams), [-2], keepdims=False)
        if self.block_inject_weight is None:
            return mixed
        inject = self.block_inject_weight(op, normalized)
        inject = op.Mul(op.Sigmoid(op.Div(inject, float(self._hc_count))), 2.0)
        return mixed, hyper_input, inject

    @staticmethod
    def inject(
        op: OpBuilder,
        block_output: ir.Value,
        hyper_input: ir.Value,
        injection_weights: ir.Value,
    ) -> ir.Value:
        # Broadcast the block output into all residual streams, then flatten.
        injection = op.Mul(
            op.Unsqueeze(block_output, [-2]),
            op.Unsqueeze(injection_weights, [-1]),
        )
        return op.Add(hyper_input, op.Reshape(injection, [0, 0, -1]))


class _Qwen4ExpDepthwiseConv1d(nn.Module):
    """Causal depthwise convolution with the upstream full-kernel cache ABI."""

    def __init__(self, channels: int, kernel_size: int, activation: str):
        super().__init__()
        self.weight = nn.Parameter([channels, 1, kernel_size])
        self._channels = channels
        self._kernel_size = kernel_size
        self._activation = get_activation(activation)

    def forward(
        self, op: OpBuilder, input_val: ir.Value, conv_state: ir.Value
    ) -> tuple[ir.Value, ir.Value]:
        # Upstream caches K values. Convolution consumes the newest K-1 cached
        # values plus the current chunk, while the next state keeps the last K.
        history = op.Concat(conv_state, input_val, axis=2)
        conv_input = op.Slice(history, [1], [_INT64_MAX], [2])
        output = op.Conv(
            conv_input,
            self.weight,
            group=self._channels,
            kernel_shape=[self._kernel_size],
        )
        present_state = op.Slice(history, [-self._kernel_size], [_INT64_MAX], [2])
        return self._activation(op, output), present_state


class _Qwen4ExpPostGatedRMSNorm(nn.Module):
    """Post-normalization output gate with configurable sigmoid/SiLU activation."""

    def __init__(self, hidden_size: int, eps: float, activation: str):
        super().__init__()
        self.weight = nn.Parameter([hidden_size])
        self._eps = eps
        self._activation = activation

    def forward(self, op: OpBuilder, hidden_states: ir.Value, gate: ir.Value) -> ir.Value:
        hidden_f32 = op.Cast(hidden_states, to=ir.DataType.FLOAT)
        variance = op.ReduceMean(op.Mul(hidden_f32, hidden_f32), [-1], keepdims=True)
        normalized_f32 = op.Mul(
            hidden_f32,
            op.Reciprocal(op.Sqrt(op.Add(variance, self._eps))),
        )
        normalized = op.CastLike(normalized_f32, hidden_states)
        weighted = op.Mul(normalized, self.weight)
        gate_f32 = op.Cast(gate, to=ir.DataType.FLOAT)
        activated = (
            op.Sigmoid(gate_f32) if self._activation == "sigmoid" else op.Swish(gate_f32)
        )
        return op.CastLike(
            op.Mul(op.Cast(weighted, to=ir.DataType.FLOAT), activated),
            hidden_states,
        )


class Qwen4ExpGatedDeltaNet(GatedDeltaNet):
    """Qwen4-Exp DeltaNet with exact cache width and output-gate activation."""

    def __init__(self, config: Qwen4ExpConfig):
        super().__init__(config)
        self.conv1d = _Qwen4ExpDepthwiseConv1d(
            self.conv_dim, self.conv_kernel_size, config.hidden_act or "silu"
        )
        self.norm = _Qwen4ExpPostGatedRMSNorm(
            self.head_v_dim,
            config.rms_norm_eps,
            config.output_gate_type or config.hidden_act or "silu",
        )


class Qwen4ExpNGramEmbedding(nn.Module):
    """Deterministic hashed bigram/trigram embedding used by PLE."""

    def __init__(
        self,
        config: Qwen4ExpConfig,
        embedding_dim: int,
        ple_layer_index: int,
    ):
        super().__init__()
        self._ngram_size = config.ngram_size
        self._context_len = config.ngram_size - 1
        self._heads_per_ngram = config.heads_per_ngram
        self._ngram_heads = self._context_len * config.heads_per_ngram
        self._eos_token_id = (
            config.eos_token_id[0]
            if isinstance(config.eos_token_id, list)
            else config.eos_token_id
        )
        assert self._eos_token_id is not None

        buffers, padded_vocab_size = _qwen4_exp_ple_buffer_values(
            config,
            ple_layer_index,
        )

        self.layer_multipliers = nn.Parameter(
            [config.ngram_size],
            dtype=ir.DataType.INT64,
            data=ir.tensor(buffers["layer_multipliers"]),
        )
        self.ngram_heads_vocab_sizes = nn.Parameter(
            [self._ngram_heads],
            dtype=ir.DataType.INT64,
            data=ir.tensor(buffers["ngram_heads_vocab_sizes"]),
        )
        self.ngram_heads_offsets = nn.Parameter(
            [self._ngram_heads],
            dtype=ir.DataType.INT64,
            data=ir.tensor(buffers["ngram_heads_offsets"]),
        )
        self.ngram_embedding = Qwen4ExpShardedEmbedding(
            padded_vocab_size,
            embedding_dim // self._ngram_heads,
            config.split_ngram_parts,
        )

    def _shifted_tokens(self, op: OpBuilder, history: ir.Value, shift: int) -> ir.Value:
        history_length = op.Shape(history, start=1, end=2)
        start = op.Constant(value_ints=[self._context_len - shift])
        end = op.Sub(history_length, op.Constant(value_ints=[shift]))
        shifted = op.Slice(history, start, end, [1])
        if shift <= 1:
            return shifted

        crossed_eos = None
        for nearer_shift in range(1, shift):
            nearer = self._shifted_tokens(op, history, nearer_shift)
            is_eos = op.Equal(nearer, self._eos_token_id)
            crossed_eos = is_eos if crossed_eos is None else op.Or(crossed_eos, is_eos)
        assert crossed_eos is not None
        eos = op.Expand(op.Constant(value_int=self._eos_token_id), op.Shape(shifted))
        return op.Where(crossed_eos, eos, shifted)

    def forward(
        self,
        op: OpBuilder,
        input_ids: ir.Value,
        past_context: ir.Value,
    ) -> tuple[ir.Value, ir.Value]:
        history = op.Concat(past_context, input_ids, axis=1)
        present_context = op.Slice(history, [-self._context_len], [_INT64_MAX], [1])
        shifted = [
            self._shifted_tokens(op, history, shift) for shift in range(self._ngram_size)
        ]

        blocks = []
        for ngram in range(2, self._ngram_size + 1):
            start = (ngram - 2) * self._heads_per_ngram
            end = start + self._heads_per_ngram
            mixed = op.Mul(shifted[0], op.Gather(self.layer_multipliers, 0))
            for position in range(1, ngram):
                part = op.Mul(
                    shifted[position],
                    op.Gather(self.layer_multipliers, position),
                )
                mixed = op.BitwiseXor(mixed, part)
            vocab_sizes = op.Slice(self.ngram_heads_vocab_sizes, [start], [end], [0])
            offsets = op.Slice(self.ngram_heads_offsets, [start], [end], [0])
            ids = op.Mod(op.Unsqueeze(mixed, [-1]), vocab_sizes)
            blocks.append(op.Add(ids, offsets))

        ngram_ids = op.Concat(*blocks, axis=-1)
        embeddings = self.ngram_embedding(op, ngram_ids)
        return op.Reshape(embeddings, [0, 0, -1]), present_context


class Qwen4ExpShardedEmbedding(nn.Module):
    """PLE embedding kept in checkpoint-native row shards for bounded-memory export."""

    def __init__(self, num_embeddings: int, embedding_dim: int, num_shards: int):
        super().__init__()
        if num_shards <= 0 or num_embeddings % num_shards:
            raise ValueError(
                "Qwen4-Exp PLE embedding rows must divide split_ngram_parts exactly: "
                f"{num_embeddings} % {num_shards}"
            )
        self._rows_per_shard = num_embeddings // num_shards
        self._num_shards = num_shards
        for shard_index in range(num_shards):
            setattr(
                self,
                f"shard_{shard_index}",
                Embedding(self._rows_per_shard, embedding_dim),
            )

    def forward(self, op: OpBuilder, input_ids: ir.Value) -> ir.Value:
        result = None
        for shard_index in range(self._num_shards):
            start = shard_index * self._rows_per_shard
            end = start + self._rows_per_shard
            local_ids = op.Clip(op.Sub(input_ids, start), 0, self._rows_per_shard - 1)
            gathered = getattr(self, f"shard_{shard_index}")(op, local_ids)
            selected = op.And(op.GreaterOrEqual(input_ids, start), op.Less(input_ids, end))
            shard_result = op.Where(
                op.Unsqueeze(selected, [-1]),
                gathered,
                op.CastLike(0.0, gathered),
            )
            result = shard_result if result is None else op.Add(result, shard_result)
        assert result is not None
        return result


class _PLEDepthwiseConv1d(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int):
        super().__init__()
        self.weight = nn.Parameter([channels, 1, kernel_size])
        self._channels = channels
        self._kernel_size = kernel_size
        self._dilation = dilation
        self._state_len = (kernel_size - 1) * dilation

    def forward(
        self, op: OpBuilder, hidden_states: ir.Value, past_state: ir.Value
    ) -> tuple[ir.Value, ir.Value]:
        current = op.Transpose(hidden_states, perm=[0, 2, 1])
        history = op.Concat(past_state, current, axis=2)
        output = op.Conv(
            history,
            self.weight,
            dilations=[self._dilation],
            group=self._channels,
            kernel_shape=[self._kernel_size],
        )
        present = op.Slice(history, [-self._state_len], [_INT64_MAX], [2])
        return op.Transpose(op.Swish(output), perm=[0, 2, 1]), present


class Qwen4ExpPLELayer(nn.Module):
    """Hashed lexical features injected into all hyper-connection streams."""

    def __init__(self, config: Qwen4ExpConfig, layer_idx: int, ple_layer_index: int):
        super().__init__()
        assert config.ple_embed_dim is not None
        self._hidden_size = config.hidden_size
        self._hc_count = config.hc_count
        hc_hidden_size = config.hidden_size * config.hc_count
        self.ple_embedding = Qwen4ExpNGramEmbedding(
            config, config.ple_embed_dim, ple_layer_index
        )
        self.key_proj = Linear(config.ple_embed_dim, hc_hidden_size, bias=False)
        self.value_proj = Linear(config.ple_embed_dim, config.hidden_size, bias=False)
        self.norm_key = Qwen4ExpGroupedRMSNorm(
            hc_hidden_size, config.hidden_size, config.rms_norm_eps
        )
        self.norm_query = Qwen4ExpGroupedRMSNorm(
            hc_hidden_size, config.hidden_size, config.rms_norm_eps
        )
        self.norm_conv = Qwen4ExpGroupedRMSNorm(
            hc_hidden_size, config.hidden_size, config.rms_norm_eps
        )
        self.conv1d = _PLEDepthwiseConv1d(
            hc_hidden_size, config.ple_conv_kernel_size, config.ngram_size
        )
        self.state_len = (config.ple_conv_kernel_size - 1) * config.ngram_size
        self.context_len = config.ngram_size - 1

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        input_ids: ir.Value,
        past_conv_state: ir.Value,
        past_context: ir.Value,
        token_mask: ir.Value,
    ) -> tuple[ir.Value, ir.Value, ir.Value]:
        embeddings, present_context = self.ple_embedding(op, input_ids, past_context)
        key = self.norm_key(op, self.key_proj(op, embeddings))
        key = op.Reshape(key, [0, 0, self._hc_count, self._hidden_size])
        query = self.norm_query(op, hidden_states)
        query = op.Reshape(query, [0, 0, self._hc_count, self._hidden_size])
        value = self.value_proj(op, embeddings)
        gate = op.ReduceSum(op.Mul(key, query), [-1], keepdims=True)
        gate = op.Div(gate, math.sqrt(self._hidden_size))
        signed_sqrt = op.Mul(
            op.Sign(gate), op.Sqrt(op.Max(op.Abs(gate), op.CastLike(1e-6, gate)))
        )
        gated = op.Mul(op.Sigmoid(signed_sqrt), op.Unsqueeze(value, [-2]))
        gated = op.Reshape(gated, [0, 0, -1])
        normalized = self.norm_conv(op, gated)
        mask = op.Unsqueeze(op.CastLike(token_mask, gated), [-1])
        gated = op.Mul(gated, mask)
        normalized = op.Mul(normalized, mask)
        convolved, present_conv_state = self.conv1d(op, normalized, past_conv_state)
        return op.Add(gated, convolved), present_conv_state, present_context


class Qwen4ExpQSAIndexer(nn.Module):
    """Vectorized exact block-pooling/token-selection QSA indexer."""

    def __init__(self, config: Qwen4ExpConfig):
        super().__init__()
        assert config.indexer_n_heads is not None
        assert config.indexer_kv_heads == 1
        assert config.indexer_head_dim is not None
        assert config.indexer_budget is not None
        assert config.indexer_compress_ratio is not None
        self._n_heads = config.indexer_n_heads
        self._head_dim = config.indexer_head_dim
        self._budget = config.indexer_budget
        self._ratio = config.indexer_compress_ratio
        self._block_topk = self._budget // self._ratio
        self._rotary_dim = int(config.head_dim * (config.partial_rotary_factor or 1.0))
        self._frequency_dim = self._rotary_dim // 2
        self._interleaved = config.rope_interleave
        self.index_qk_proj = Linear(
            config.hidden_size,
            (self._n_heads + 1) * self._head_dim,
            bias=False,
        )
        self.q_layernorm = Qwen4ExpOffsetRMSNorm(self._head_dim, eps=config.rms_norm_eps)
        self.k_layernorm = Qwen4ExpOffsetRMSNorm(self._head_dim, eps=config.rms_norm_eps)

    def _rotate(
        self,
        op: OpBuilder,
        value: ir.Value,
        position_embeddings: tuple[ir.Value, ir.Value],
        num_heads: int,
    ) -> ir.Value:
        # The shared RoPE cache follows ONNX RotaryEmbedding and stores one
        # cos/sin value per rotation pair. The op expands those values across
        # split halves or adjacent pairs according to ``interleaved``.
        value = op.Reshape(value, [0, 0, num_heads * self._head_dim])
        value = apply_rotary_pos_emb(
            op,
            value,
            position_embeddings,
            num_heads,
            rotary_embedding_dim=(
                self._rotary_dim if self._rotary_dim < self._head_dim else 0
            ),
            interleaved=self._interleaved,
        )
        return op.Reshape(value, [0, 0, num_heads, self._head_dim])

    @staticmethod
    def _gather_per_query(
        op: OpBuilder,
        data: ir.Value,
        indices: ir.Value,
        width: int,
    ) -> ir.Value:
        """Gather ``data[B,T,D]`` at arbitrary ``indices[B,S,...]``."""
        batch = op.Shape(data, start=0, end=1)
        queries = op.Shape(indices, start=1, end=2)
        total_length = op.Shape(data, start=1, end=2)
        index_tail = op.Shape(indices, start=2)
        gathered_count = op.ReduceProd(index_tail, [0], keepdims=True)
        batch_queries = op.Mul(batch, queries)
        expanded = op.Expand(
            op.Unsqueeze(data, [1]),
            op.Concat(
                batch,
                queries,
                total_length,
                op.Constant(value_ints=[width]),
                axis=0,
            ),
        )
        flat_data = op.Reshape(
            expanded,
            op.Concat(
                batch_queries,
                total_length,
                op.Constant(value_ints=[width]),
                axis=0,
            ),
        )
        flat_indices = op.Reshape(indices, op.Concat(batch_queries, gathered_count, axis=0))
        gather_indices = op.Expand(
            op.Unsqueeze(flat_indices, [-1]),
            op.Concat(
                batch_queries,
                gathered_count,
                op.Constant(value_ints=[width]),
                axis=0,
            ),
        )
        gathered = op.GatherElements(flat_data, gather_indices, axis=1)
        return op.Reshape(
            gathered,
            op.Concat(
                op.Shape(indices),
                op.Constant(value_ints=[width]),
                axis=0,
            ),
        )

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        current_position_embeddings: tuple[ir.Value, ir.Value],
        full_position_embeddings: tuple[ir.Value, ir.Value],
        attention_bias: ir.Value,
        past_index_key: ir.Value,
    ) -> tuple[ir.Value, ir.Value]:
        qk = self.index_qk_proj(op, hidden_states)
        query, current_key = op.Split(
            qk,
            [self._n_heads * self._head_dim, self._head_dim],
            axis=-1,
            _outputs=2,
        )
        query = op.Reshape(query, [0, 0, self._n_heads, self._head_dim])
        query = self._rotate(
            op, self.q_layernorm(op, query), current_position_embeddings, self._n_heads
        )
        present_index_key = op.Concat(past_index_key, current_key, axis=1)

        all_visible = op.Equal(
            op.Squeeze(attention_bias, [1]),
            op.CastLike(0.0, attention_bias),
        )
        visible_count = op.ReduceSum(
            op.Cast(all_visible, to=ir.DataType.INT64), [2], keepdims=True
        )
        query_num_blocks = op.Div(visible_count, self._ratio)

        total_length = op.Shape(present_index_key, start=1, end=2)
        batch = op.Shape(present_index_key, start=0, end=1)
        query_length = op.Shape(hidden_states, start=1, end=2)
        max_blocks = op.Div(op.Add(total_length, self._ratio - 1), self._ratio)
        block_ids = op.Range(
            op.Constant(value_int=0),
            op.Squeeze(max_blocks, [0]),
            op.Constant(value_int=1),
        )
        block_offsets = op.Range(
            op.Constant(value_int=0),
            op.Constant(value_int=self._ratio),
            op.Constant(value_int=1),
        )
        token_ids = op.Range(
            op.Constant(value_int=0),
            op.Squeeze(total_length, [0]),
            op.Constant(value_int=1),
        )
        # Sort visible token positions into chronological order. This preserves
        # exact upstream block construction for left padding and arbitrary holes.
        position_scores = op.Where(
            all_visible,
            op.Cast(
                op.Sub(op.Squeeze(total_length, [0]), token_ids),
                to=ir.DataType.FLOAT,
            ),
            op.Expand(op.Constant(value_float=-1.0), op.Shape(all_visible)),
        )
        _, ordered_visible_indices = op.TopK(
            position_scores,
            total_length,
            axis=-1,
            largest=1,
            sorted=1,
            _outputs=2,
        )
        candidate_ordinals = op.Add(
            op.Unsqueeze(op.Mul(block_ids, self._ratio), [1]),
            op.Unsqueeze(block_offsets, [0]),
        )
        candidate_ordinals = op.Min(candidate_ordinals, op.Sub(total_length, 1))
        candidate_ordinals = op.Expand(
            op.Unsqueeze(candidate_ordinals, [0, 1]),
            op.Concat(
                batch,
                query_length,
                max_blocks,
                op.Constant(value_ints=[self._ratio]),
                axis=0,
            ),
        )
        flat_candidate_ordinals = op.Reshape(
            candidate_ordinals,
            op.Concat(batch, query_length, op.Constant(value_ints=[-1]), axis=0),
        )
        candidate_indices = op.Reshape(
            op.GatherElements(
                ordered_visible_indices,
                flat_candidate_ordinals,
                axis=2,
            ),
            op.Concat(
                batch,
                query_length,
                max_blocks,
                op.Constant(value_ints=[self._ratio]),
                axis=0,
            ),
        )
        block_valid = op.Less(op.Unsqueeze(block_ids, [0, 1]), query_num_blocks)
        key_blocks = self._gather_per_query(
            op, present_index_key, candidate_indices, self._head_dim
        )
        pooled = op.ReduceMean(op.Cast(key_blocks, to=ir.DataType.FLOAT), [3], keepdims=False)
        pooled = self.k_layernorm(op, op.CastLike(pooled, present_index_key))

        block_starts = op.Squeeze(op.Gather(candidate_indices, [0], axis=3), [3])
        block_positions = (
            self._gather_per_query(
                op,
                full_position_embeddings[0],
                block_starts,
                self._frequency_dim,
            ),
            self._gather_per_query(
                op,
                full_position_embeddings[1],
                block_starts,
                self._frequency_dim,
            ),
        )
        flat_block_shape = op.Concat(
            op.Mul(batch, query_length),
            max_blocks,
            op.Constant(value_ints=[self._head_dim]),
            axis=0,
        )
        flat_position_shape = op.Concat(
            op.Mul(batch, query_length),
            max_blocks,
            op.Constant(value_ints=[self._frequency_dim]),
            axis=0,
        )
        pooled = self._rotate(
            op,
            op.Reshape(pooled, flat_block_shape),
            (
                op.Reshape(block_positions[0], flat_position_shape),
                op.Reshape(block_positions[1], flat_position_shape),
            ),
            1,
        )
        pooled = op.Reshape(
            op.Squeeze(pooled, [2]),
            op.Concat(
                batch,
                query_length,
                max_blocks,
                op.Constant(value_ints=[self._head_dim]),
                axis=0,
            ),
        )

        q_f32 = op.Cast(query, to=ir.DataType.FLOAT)
        k_f32 = op.Cast(pooled, to=ir.DataType.FLOAT)
        scores = op.MatMul(q_f32, op.Transpose(k_f32, perm=[0, 1, 3, 2]))
        scores = op.ReduceSum(op.Relu(scores), [2], keepdims=False)
        scores = op.Div(scores, math.sqrt(self._head_dim))
        scores = op.Where(
            block_valid,
            scores,
            op.Expand(op.CastLike(-1e30, scores), op.Shape(scores)),
        )

        # Padding by block_topk makes TopK valid even before one full block
        # exists. Dummy selections never compare equal to a real block id.
        padded_scores = op.Pad(
            scores,
            op.Constant(value_ints=[0, 0, 0, 0, 0, self._block_topk]),
            op.CastLike(-1e30, scores),
        )
        _, selected_blocks = op.TopK(
            padded_scores,
            op.Constant(value_ints=[self._block_topk]),
            axis=-1,
            _outputs=2,
        )
        real_block_ids = op.Range(
            op.Constant(value_int=0),
            op.Squeeze(max_blocks, [0]),
            op.Constant(value_int=1),
        )
        selected = op.Equal(
            op.Unsqueeze(selected_blocks, [-1]),
            op.Unsqueeze(real_block_ids, [0, 1, 2]),
        )
        selected = op.ReduceMax(op.Cast(selected, to=ir.DataType.INT64), [2], keepdims=False)
        selected = op.Cast(selected, to=ir.DataType.BOOL)
        selected = op.And(selected, block_valid)

        visible_ordinal = op.Sub(
            op.CumSum(
                op.Cast(all_visible, to=ir.DataType.INT64),
                op.Constant(value_int=2),
            ),
            1,
        )
        token_block = op.Div(op.Max(visible_ordinal, 0), self._ratio)
        token_block = op.Min(token_block, op.Sub(max_blocks, 1))
        selected_for_token = op.GatherElements(selected, token_block, axis=2)
        query_complete_length = op.Mul(query_num_blocks, self._ratio)
        complete_token = op.Less(visible_ordinal, query_complete_length)
        selected_complete = op.And(selected_for_token, op.And(all_visible, complete_token))
        query_tail = op.And(
            all_visible,
            op.GreaterOrEqual(visible_ordinal, query_complete_length),
        )
        selected_tokens = op.Or(selected_complete, query_tail)
        sparse_bias = op.Where(
            selected_tokens,
            op.CastLike(0.0, attention_bias),
            op.CastLike(-1e30, attention_bias),
        )
        return op.Unsqueeze(sparse_bias, [1]), present_index_key


class Qwen4ExpSparseAttention(Qwen35Attention):
    """Qwen3.5 gated GQA restricted by the exact QSA selected-token mask."""

    def __init__(self, config: Qwen4ExpConfig):
        super().__init__(config)
        self.q_norm = Qwen4ExpOffsetRMSNorm(config.head_dim, config.rms_norm_eps)
        self.k_norm = Qwen4ExpOffsetRMSNorm(config.head_dim, config.rms_norm_eps)
        self.indexer = Qwen4ExpQSAIndexer(config)

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        attention_bias: ir.Value,
        position_embeddings: tuple[ir.Value, ir.Value],
        qsa_current_position_embeddings: tuple[ir.Value, ir.Value],
        qsa_full_position_embeddings: tuple[ir.Value, ir.Value],
        past_key_value: tuple[ir.Value, ir.Value, ir.Value],
    ):
        sparse_bias, present_index_key = self.indexer(
            op,
            hidden_states,
            qsa_current_position_embeddings,
            qsa_full_position_embeddings,
            attention_bias,
            past_key_value[2],
        )
        output, (present_key, present_value) = super().forward(
            op,
            hidden_states,
            sparse_bias,
            position_embeddings,
            (past_key_value[0], past_key_value[1]),
        )
        return output, (present_key, present_value, present_index_key)


class Qwen4ExpMoEBlock(Qwen2MoELayer):
    """Softmax-first top-10 routed experts plus sigmoid-gated shared expert."""

    def __init__(self, config: Qwen4ExpConfig):
        assert config.num_local_experts is not None
        assert config.num_experts_per_tok is not None
        gate = Qwen4ExpTopKGate(
            config.hidden_size,
            config.num_local_experts,
            config.num_experts_per_tok,
            norm_topk_prob=config.norm_topk_prob,
        )
        super().__init__(config, gate=gate)
        self.experts = Qwen4ExpExperts(config)

    def forward(self, op: OpBuilder, hidden_states: ir.Value):
        routing_weights, selected_experts = self.gate(op, hidden_states)
        expert_output = self.experts(
            op,
            hidden_states,
            selected_experts,
            routing_weights,
        )
        shared_output = self.shared_expert(op, hidden_states)
        shared_gate = op.Sigmoid(self.shared_expert_gate(op, hidden_states))
        return op.Add(expert_output, op.Mul(shared_output, shared_gate))


class Qwen4ExpExperts(nn.Module):
    """Packed expert weights evaluated only for each token's selected experts."""

    def __init__(self, config: Qwen4ExpConfig):
        super().__init__()
        assert config.num_local_experts is not None
        assert config.moe_intermediate_size is not None
        self._hidden_size = config.hidden_size
        self._num_experts = config.num_local_experts
        self._top_k = config.num_experts_per_tok
        self._intermediate_size = config.moe_intermediate_size
        self.gate_up_proj = nn.Parameter(
            [
                config.num_local_experts,
                2 * config.moe_intermediate_size,
                config.hidden_size,
            ]
        )
        self.down_proj = nn.Parameter(
            [
                config.num_local_experts,
                config.hidden_size,
                config.moe_intermediate_size,
            ]
        )

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        selected_experts: ir.Value,
        routing_weights: ir.Value,
    ) -> ir.Value:
        # Dispatch only the tokens routed to each expert. Dynamic Gather on the
        # expert bank would replicate full matrices into (B,S,K,...), which is
        # prohibitive at production prefill lengths.
        original_shape = op.Shape(hidden_states)
        flat_hidden = op.Reshape(hidden_states, [-1, self._hidden_size])
        assert self._top_k is not None
        flat_indices = op.Reshape(selected_experts, [-1, self._top_k])
        flat_weights = op.Reshape(routing_weights, [-1, self._top_k])
        output = op.CastLike(
            op.ConstantOfShape(op.Shape(flat_hidden)),
            flat_hidden,
        )

        for expert_index in range(self._num_experts):
            matches = op.Equal(flat_indices, expert_index)
            coordinates = op.Transpose(op.NonZero(matches), perm=[1, 0])
            token_indices = op.Squeeze(op.Gather(coordinates, [0], axis=1), [1])
            expert_hidden = op.Gather(flat_hidden, token_indices, axis=0)

            gate_up = op.Squeeze(
                op.Gather(self.gate_up_proj, [expert_index], axis=0),
                [0],
            )
            projected = op.MatMul(
                expert_hidden,
                op.Transpose(gate_up, perm=[1, 0]),
            )
            gate, up = op.Split(
                projected,
                [self._intermediate_size, self._intermediate_size],
                axis=-1,
                _outputs=2,
            )
            activated = op.Mul(op.Swish(gate), up)

            down = op.Squeeze(
                op.Gather(self.down_proj, [expert_index], axis=0),
                [0],
            )
            expert_output = op.MatMul(
                activated,
                op.Transpose(down, perm=[1, 0]),
            )
            route_weight = op.GatherND(flat_weights, coordinates)
            weighted = op.Mul(expert_output, op.Unsqueeze(route_weight, [-1]))
            scattered = op.ScatterND(
                op.CastLike(
                    op.ConstantOfShape(op.Shape(flat_hidden)),
                    flat_hidden,
                ),
                op.Unsqueeze(token_indices, [-1]),
                weighted,
            )
            output = op.Add(output, scattered)
        return op.Reshape(output, original_shape)


class Qwen4ExpTopKGate(SoftmaxTopKGate):
    """Qwen4 router with the upstream float32 softmax/renormalization path."""

    def forward(self, op: OpBuilder, hidden_states: ir.Value):
        logits = op.MatMul(hidden_states, op.Transpose(self.weight, perm=[1, 0]))
        probabilities = op.Softmax(
            op.Cast(logits, to=ir.DataType.FLOAT),
            axis=-1,
        )
        routing_weights, selected_experts = op.TopK(
            probabilities,
            op.Constant(value_ints=[self.top_k]),
            axis=-1,
            _outputs=2,
        )
        if self.norm_topk_prob:
            routing_weights = op.Div(
                routing_weights,
                op.ReduceSum(routing_weights, [-1], keepdims=True),
            )
        if self.routed_scaling_factor != 1.0:  # noqa: RUF069
            routing_weights = op.Mul(routing_weights, self.routed_scaling_factor)
        return op.CastLike(routing_weights, hidden_states), selected_experts


class Qwen4ExpDecoderLayer(nn.Module):
    """One Qwen4-Exp hyper-connected attention-plus-MoE layer."""

    def __init__(self, config: Qwen4ExpConfig, layer_idx: int):
        super().__init__()
        assert config.layer_types is not None
        self.layer_type = config.layer_types[layer_idx]
        self.linear_attn = (
            Qwen4ExpGatedDeltaNet(config) if self.layer_type == "linear_attention" else None
        )
        self.self_attn = (
            Qwen4ExpSparseAttention(config)
            if self.layer_type == "qwen_sparse_attention"
            else None
        )
        self.mlp = Qwen4ExpMoEBlock(config)
        self.ple = None
        if layer_idx + 1 in (config.ple_layer_ids or []):
            ple_layer_index = (config.ple_layer_ids or []).index(layer_idx + 1)
            self.ple = Qwen4ExpPLELayer(config, layer_idx, ple_layer_index)
        self.attn_hyper_connection = Qwen4ExpGatedResidual(config)
        self.mlp_hyper_connection = Qwen4ExpGatedResidual(config)

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        input_ids: ir.Value,
        attention_bias: ir.Value,
        position_embeddings: tuple[ir.Value, ir.Value],
        qsa_current_position_embeddings: tuple[ir.Value, ir.Value],
        qsa_full_position_embeddings: tuple[ir.Value, ir.Value],
        past_state: tuple[ir.Value, ...],
        token_mask: ir.Value,
    ):
        ple_present: tuple[ir.Value, ir.Value] | None = None
        if self.ple is not None:
            ple_output, ple_conv, ple_context = self.ple(
                op,
                hidden_states,
                input_ids,
                past_state[2],
                past_state[3],
                token_mask,
            )
            hidden_states = op.Add(hidden_states, ple_output)
            ple_present = (ple_conv, ple_context)

        mixed, residual, inject = self.attn_hyper_connection(op, hidden_states)
        if self.linear_attn is not None:
            mixed = op.Mul(mixed, op.Unsqueeze(op.CastLike(token_mask, mixed), [-1]))
            output, conv_state, recurrent_state = self.linear_attn(
                op, mixed, past_state[0], past_state[1]
            )
            present_state: tuple[ir.Value, ...] = (
                conv_state,
                recurrent_state,
            )
            if ple_present is not None:
                present_state += ple_present
        else:
            assert self.self_attn is not None
            output, present_state = self.self_attn(
                op,
                mixed,
                attention_bias,
                position_embeddings,
                qsa_current_position_embeddings,
                qsa_full_position_embeddings,
                past_state,
            )
        hidden_states = Qwen4ExpGatedResidual.inject(op, output, residual, inject)

        mixed, residual, inject = self.mlp_hyper_connection(op, hidden_states)
        output = self.mlp(op, mixed)
        hidden_states = Qwen4ExpGatedResidual.inject(op, output, residual, inject)
        return hidden_states, present_state


class Qwen4ExpTextModel(nn.Module):
    """Qwen4-Exp text backbone with heterogeneous recurrent/QSA state."""

    def __init__(self, config: Qwen4ExpConfig):
        super().__init__()
        self._dtype = config.dtype
        self._hc_count = config.hc_count
        self._eos_token_id = (
            config.eos_token_id[0]
            if isinstance(config.eos_token_id, list)
            else config.eos_token_id
        )
        self.embed_tokens = Embedding(
            config.vocab_size, config.hidden_size, config.pad_token_id
        )
        self.layers = nn.ModuleList(
            [
                Qwen4ExpDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.rotary_emb = initialize_rope(config)
        self.hyper_connection_mixer = Qwen4ExpGatedResidual(config, use_combine=False)

    def forward(
        self,
        op: OpBuilder,
        input_ids: ir.Value | None,
        attention_mask: ir.Value,
        position_ids: ir.Value,
        past_position_ids: ir.Value,
        past_key_values: list[tuple[ir.Value, ...]],
        *,
        inputs_embeds: ir.Value | None = None,
        ple_input_ids: ir.Value | None = None,
        qsa_position_ids: ir.Value | None = None,
        past_qsa_position_ids: ir.Value | None = None,
    ):
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("Qwen4-Exp requires input_ids or inputs_embeds")
            embeddings = self.embed_tokens(op, input_ids)
        else:
            embeddings = inputs_embeds
        if ple_input_ids is None:
            if input_ids is None:
                raise ValueError(
                    "Qwen4-Exp inputs_embeds execution requires explicit ple_input_ids"
                )
            ple_input_ids = input_ids
        hidden_states = op.Concat(*[embeddings for _ in range(self._hc_count)], axis=-1)
        all_position_ids = op.Concat(past_position_ids, position_ids, axis=-1)
        position_embeddings = self.rotary_emb(op, position_ids)
        if qsa_position_ids is None:
            qsa_position_ids = position_ids
        if past_qsa_position_ids is None:
            past_qsa_position_ids = past_position_ids
        all_qsa_position_ids = op.Concat(past_qsa_position_ids, qsa_position_ids, axis=-1)
        qsa_current_position_embeddings = self.rotary_emb(op, qsa_position_ids)
        qsa_full_position_embeddings = self.rotary_emb(op, all_qsa_position_ids)
        attention_bias = create_attention_bias(
            op,
            input_ids=ple_input_ids,
            attention_mask=attention_mask,
            dtype=self._dtype,
        )
        total_length = op.Shape(attention_mask, start=1, end=2)
        current_length = op.Shape(ple_input_ids, start=1, end=2)
        current_start = op.Sub(total_length, current_length)
        current_token_mask = op.Not(
            op.Equal(
                op.Slice(
                    attention_mask,
                    current_start,
                    total_length,
                    op.Constant(value_ints=[1]),
                ),
                op.Constant(value_int=0),
            )
        )
        if self._eos_token_id is not None:
            ple_input_ids = op.Where(
                current_token_mask,
                ple_input_ids,
                op.Expand(
                    op.Constant(value_int=self._eos_token_id),
                    op.Shape(ple_input_ids),
                ),
            )

        presents = []
        for layer, past_state in zip(self.layers, past_key_values):
            hidden_states, present = layer(
                op,
                hidden_states,
                ple_input_ids,
                attention_bias,
                position_embeddings,
                qsa_current_position_embeddings,
                qsa_full_position_embeddings,
                past_state,
                current_token_mask,
            )
            presents.append(present)
        return (
            self.hyper_connection_mixer(op, hidden_states),
            presents,
            all_position_ids,
        )


class Qwen4ExpCausalLMModel(nn.Module):
    """Qwen3.8 Flash-Next/Qwen4-Exp causal decoder with exact text-core semantics."""

    default_task: str = "qwen4-exp-text-generation"
    category: str = "Mixture of Experts"
    config_class: type = Qwen4ExpConfig

    def __init__(self, config: Qwen4ExpConfig):
        nn.Module.__init__(self)
        self.config = config
        self.model = Qwen4ExpTextModel(config)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        op: OpBuilder,
        input_ids: ir.Value,
        attention_mask: ir.Value,
        position_ids: ir.Value,
        past_position_ids: ir.Value,
        past_key_values: list[tuple[ir.Value, ...]],
    ):
        hidden_states, presents, present_position_ids = self.model(
            op,
            input_ids,
            attention_mask,
            position_ids,
            past_position_ids,
            past_key_values,
            ple_input_ids=input_ids,
        )
        return self.lm_head(op, hidden_states), presents, present_position_ids

    def build_fp8_streaming_plan(
        self,
        key_index: Mapping[str, tuple[str, list[int], str]],
        initializers: Mapping[str, ir.Value],
    ):
        """Classify every composite-checkpoint tensor for dense streaming export."""
        from mobius.integrations._weight_loading import (
            StreamingExpertBankSource,
            StreamingWeightPlan,
            StreamingWeightSource,
        )

        scheme = self.config.block_quant_scheme
        if scheme is None or tuple(scheme.weight_block_size) != (128, 128):
            raise ValueError(
                "Qwen4-Exp FP8 streaming requires the pinned 128x128 block scheme"
            )
        header_report = validate_qwen4_exp_fp8_header_contract(key_index)

        targets: dict[
            str,
            StreamingWeightSource | StreamingExpertBankSource,
        ] = {}
        constants: dict[str, torch.Tensor] = {}
        ignored: dict[str, str] = {}
        modes: Counter[str] = Counter()
        exclusion_counts: Counter[str] = Counter()

        for source_name in key_index:
            if source_name.startswith("model.visual."):
                ignored[source_name] = "multimodal component belongs to the dependent PR"
                exclusion_counts["visual"] += 1
            elif source_name.startswith("mtp."):
                ignored[source_name] = "MTP sidecar has no authoritative forward/cache ABI"
                exclusion_counts["mtp"] += 1
            elif source_name.endswith("rotary_emb.inv_freq"):
                ignored[source_name] = "RoPE frequencies are graph-derived"
                exclusion_counts["graph_derived"] += 1

        def classify_source(source_name: str) -> StreamingWeightSource:
            located = key_index.get(source_name)
            if located is None:
                raise ValueError(f"Qwen4-Exp checkpoint is missing source '{source_name}'")
            _path, _source_shape, source_dtype = located
            scale_name = (
                source_name[: -len(".weight")] + ".weight_scale_inv"
                if source_name.endswith(".weight")
                else source_name + "_scale_inv"
            )
            if source_dtype.startswith("F8"):
                if ".ple.ple_embedding.ngram_embedding.shard_" in source_name:
                    scale_name = _qwen4_exp_ple_scale_name(source_name)
                    mode = "fp8_scalar"
                    source = StreamingWeightSource(
                        source_name,
                        mode,
                        scale_name,
                        expected_scale=_PINNED_PLE_WEIGHT_SCALE,
                    )
                elif scale_name in key_index:
                    mode = "fp8_block_128"
                    source = StreamingWeightSource(source_name, mode, scale_name)
                else:
                    raise ValueError(f"Qwen4-Exp FP8 source '{source_name}' has no scale")
            elif source_dtype in {"BF16", "F16", "F32"}:
                mode = "direct"
                source = StreamingWeightSource(source_name, mode)
            else:
                raise ValueError(
                    f"Qwen4-Exp source '{source_name}' has unsupported dtype {source_dtype}"
                )
            modes[mode] += 1
            return source

        for target_name, initializer in initializers.items():
            expert_prefix = None
            projection_names: tuple[str, ...] | None = None
            if target_name.endswith(".mlp.experts.gate_up_proj"):
                expert_prefix = target_name[: -len("experts.gate_up_proj")]
                projection_names = ("gate_proj", "up_proj")
            elif target_name.endswith(".mlp.experts.down_proj"):
                expert_prefix = target_name[: -len("experts.down_proj")]
                projection_names = ("down_proj",)
            if expert_prefix is not None and projection_names is not None:
                assert self.config.num_local_experts is not None
                source_prefix = _source_name_for_qwen4_exp_target(expert_prefix)
                experts = tuple(
                    tuple(
                        classify_source(
                            f"{source_prefix}experts.{expert_index}.{projection_name}.weight"
                        )
                        for projection_name in projection_names
                    )
                    for expert_index in range(self.config.num_local_experts)
                )
                targets[target_name] = StreamingExpertBankSource(experts)
                continue

            if target_name.startswith("model."):
                suffix = target_name[len("model.") :]
                candidates = (
                    f"model.language_model.{suffix}",
                    f"language_model.{suffix}",
                    target_name,
                )
            else:
                candidates = (target_name,)
            present = [candidate for candidate in candidates if candidate in key_index]
            if not present:
                if initializer.const_value is None:
                    raise ValueError(
                        f"Qwen4-Exp target '{target_name}' has no checkpoint source"
                    )
                continue
            source_name = present[0]
            for lower_priority in present[1:]:
                ignored[lower_priority] = (
                    f"lower-priority alias of preferred source {source_name}"
                )
                if lower_priority.endswith(".weight"):
                    lower_scale = lower_priority[: -len(".weight")] + ".weight_scale_inv"
                    if lower_scale in key_index:
                        ignored[lower_scale] = (
                            f"scale for lower-priority alias {lower_priority}"
                        )

            if initializer.const_value is not None:
                constants[source_name] = torch.from_numpy(
                    initializer.const_value.numpy().copy()
                )
                continue

            targets[target_name] = classify_source(source_name)

        # Graph optimization can inline these tiny deterministic PLE buffers as
        # Constant nodes, so validate them from the module parameter contract even
        # when they no longer appear in graph.initializers.
        for target_name, parameter in self.named_parameters():
            if parameter._const_value is None:
                continue
            source_name = _source_name_for_qwen4_exp_target(target_name)
            if source_name in key_index and source_name not in constants:
                constants[source_name] = torch.from_numpy(
                    parameter._const_value.numpy().copy()
                )

        return StreamingWeightPlan(
            targets=targets,
            ignored=ignored,
            constants=constants,
            report={
                "checkpoint_family": "Qwen3.8-Flash-Next-FP8",
                "dense_dtype": "graph-declared (BF16 for the pinned checkpoint)",
                "native_fp8_reason": (
                    "No available ONNX Runtime MatMul/MoE ABI consumes F8_E4M3 "
                    "weights with BF16 128x128 inverse-scale grids exactly."
                ),
                "scaled_fp8_tensors": modes["fp8_block_128"],
                "scalar_scaled_fp8_tensors": modes["fp8_scalar"],
                "direct_dense_tensors": modes["direct"],
                "excluded_tensors": {
                    "mtp": {
                        "count": exclusion_counts["mtp"],
                        "reason": "no authoritative MTP forward equation or NextN cache ABI",
                    },
                    "visual": {
                        "count": exclusion_counts["visual"],
                        "reason": "multimodal component belongs to the dependent PR",
                    },
                    "graph_derived": {
                        "count": exclusion_counts["graph_derived"],
                        "reason": "RoPE frequencies are constructed in the graph",
                    },
                },
                "mtp_exported": False,
                "visual_exported": False,
                "multimodal_package_complete": False,
                "multimodal_dependency": "separate stacked Qwen4-Exp multimodal PR",
                "source_preference": ("model.language_model.* > language_model.* > model.*"),
                **header_report,
            },
        )

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Map official packed experts and sharded PLE tables to ONNX parameters."""
        cleaned: dict[str, torch.Tensor] = {}
        ple_shards: dict[str, set[int]] = {}
        indexer_projections: dict[str, dict[str, torch.Tensor]] = defaultdict(dict)
        parameter_map = dict(self.named_parameters())
        skipped_mtp = 0
        for original_key, value in state_dict.items():
            key = original_key
            if key.startswith("mtp."):
                skipped_mtp += 1
                continue
            if key.startswith("model.language_model."):
                key = f"model.{key[len('model.language_model.') :]}"
            elif key.startswith("language_model."):
                key = f"model.{key[len('language_model.') :]}"
            if key.startswith("model.visual."):
                continue
            if key.endswith("rotary_emb.inv_freq"):
                continue
            indexer_marker = ".self_attn.indexer.index_"
            if key.endswith((".index_q_proj.weight", ".index_k_proj.weight")):
                prefix, projection = key.rsplit(indexer_marker, 1)
                indexer_projections[f"{prefix}.self_attn.indexer.index_qk_proj.weight"][
                    projection[0]
                ] = value
                continue
            if key.endswith(
                (
                    ".ple_embedding.layer_multipliers",
                    ".ple_embedding.ngram_heads_vocab_sizes",
                    ".ple_embedding.ngram_heads_offsets",
                )
            ):
                parameter = parameter_map.get(key)
                if parameter is None or parameter._const_value is None:
                    raise ValueError(f"Unexpected Qwen4-Exp deterministic buffer: {key}")
                expected = torch.from_numpy(parameter._const_value.numpy())
                if not torch.equal(value.cpu(), expected):
                    raise ValueError(
                        f"Qwen4-Exp deterministic buffer {key} does not match "
                        "the pinned hash construction"
                    )
                continue

            combined_marker = ".ple.ple_embedding.ngram_embedding.weight"
            if key.endswith(combined_marker):
                base = key[: -len(".weight")]
                first_target = f"{base}.shard_0.weight"
                first_parameter = parameter_map.get(first_target)
                if first_parameter is None:
                    raise ValueError(f"Unexpected Qwen4-Exp PLE table: {key}")
                shard_rows, embedding_width = (
                    int(first_parameter.shape[0]),
                    int(first_parameter.shape[1]),
                )
                expected_shape = (
                    shard_rows * self.config.split_ngram_parts,
                    embedding_width,
                )
                if tuple(value.shape) != expected_shape:
                    raise ValueError(
                        f"Qwen4-Exp combined PLE table {key} has shape "
                        f"{tuple(value.shape)}, expected {expected_shape}"
                    )
                for shard_index in range(self.config.split_ngram_parts):
                    target = f"{base}.shard_{shard_index}.weight"
                    row_start = shard_index * shard_rows
                    cleaned[target] = value[row_start : row_start + shard_rows]
                ple_shards[base] = set(range(self.config.split_ngram_parts))
                continue

            marker = ".ple.ple_embedding.ngram_embedding.shard_"
            if marker in key and key.endswith(".weight"):
                prefix, suffix = key.split(marker, 1)
                shard_index = int(suffix[: -len(".weight")])
                target = (
                    f"{prefix}.ple.ple_embedding.ngram_embedding.shard_{shard_index}.weight"
                )
                if not 0 <= shard_index < self.config.split_ngram_parts:
                    raise ValueError(
                        f"Unexpected Qwen4-Exp PLE shard index {shard_index} for {target}"
                    )
                parameter = parameter_map.get(target)
                if parameter is None or tuple(value.shape) != tuple(parameter.shape):
                    expected_shape = None if parameter is None else tuple(parameter.shape)
                    raise ValueError(
                        f"Qwen4-Exp PLE shard {target} has shape {tuple(value.shape)}, "
                        f"expected {expected_shape}"
                    )
                base = f"{prefix}.ple.ple_embedding.ngram_embedding"
                ple_shards.setdefault(base, set()).add(shard_index)
                cleaned[target] = value
                continue

            if key.endswith(".mlp.experts.gate_up_proj"):
                expected_shape = (
                    self.config.num_local_experts,
                    2 * self.config.moe_intermediate_size,
                    self.config.hidden_size,
                )
                if tuple(value.shape) != expected_shape:
                    raise ValueError(
                        f"Qwen4-Exp packed gate_up_proj has shape {tuple(value.shape)}, "
                        f"expected {expected_shape}"
                    )
            if key.endswith(".mlp.experts.down_proj"):
                expected_shape = (
                    self.config.num_local_experts,
                    self.config.hidden_size,
                    self.config.moe_intermediate_size,
                )
                if tuple(value.shape) != expected_shape:
                    raise ValueError(
                        f"Qwen4-Exp packed down_proj has shape {tuple(value.shape)}, "
                        f"expected {expected_shape}"
                    )
            cleaned[key] = value

        for target, projections in indexer_projections.items():
            missing = sorted({"q", "k"} - set(projections))
            unexpected = sorted(set(projections) - {"q", "k"})
            if missing or unexpected:
                raise ValueError(
                    f"Qwen4-Exp split indexer projection {target} has missing parts "
                    f"{missing} and unexpected parts {unexpected}"
                )
            parameter = parameter_map.get(target)
            if parameter is None:
                raise ValueError(f"Unexpected Qwen4-Exp indexer projection: {target}")
            assert self.config.indexer_n_heads is not None
            assert self.config.indexer_kv_heads is not None
            assert self.config.indexer_head_dim is not None
            query = projections["q"]
            key = projections["k"]
            expected_query_rows = self.config.indexer_n_heads * self.config.indexer_head_dim
            expected_key_rows = self.config.indexer_kv_heads * self.config.indexer_head_dim
            expected_input = self.config.hidden_size
            if tuple(query.shape) != (expected_query_rows, expected_input):
                raise ValueError(
                    f"Qwen4-Exp indexer query projection {target} has shape "
                    f"{tuple(query.shape)}, expected {(expected_query_rows, expected_input)}"
                )
            if tuple(key.shape) != (expected_key_rows, expected_input):
                raise ValueError(
                    f"Qwen4-Exp indexer key projection {target} has shape "
                    f"{tuple(key.shape)}, expected {(expected_key_rows, expected_input)}"
                )
            combined = torch.cat((query, key), dim=0)
            expected_shape = tuple(int(dim) for dim in parameter.shape)
            if tuple(combined.shape) != expected_shape:
                raise ValueError(
                    f"Qwen4-Exp fused indexer projection {target} has shape "
                    f"{tuple(combined.shape)}, expected {expected_shape}"
                )
            cleaned[target] = combined

        for target, shards in ple_shards.items():
            expected = set(range(self.config.split_ngram_parts))
            if shards != expected:
                missing = sorted(expected - shards)
                unexpected = sorted(shards - expected)
                raise ValueError(
                    f"Qwen4-Exp PLE table {target} has missing shard indices {missing} "
                    f"and unexpected shard indices {unexpected}"
                )
        if skipped_mtp:
            logger.warning(
                "Skipped %d Qwen4-Exp MTP sidecar tensors. Ordinary upstream "
                "next-token inference does not execute the MTP module, and Mobius "
                "does not expose an unsupported NextN task or cache ABI.",
                skipped_mtp,
            )
        qc = getattr(self.config, "quantization", None)
        return preprocess_quantized_weights(
            cleaned,
            qc,
            tie_embeddings=effective_tie_word_embeddings(self.config),
            qmoe_target_path=None,
        )


class Qwen4ExpVLDecoderModel(Qwen4ExpCausalLMModel):
    """Qwen4-Exp decoder with independent multimodal embeddings and PLE token IDs."""

    def forward(
        self,
        op: OpBuilder,
        input_ids: ir.Value | None,
        attention_mask: ir.Value,
        position_ids: ir.Value,
        past_position_ids: ir.Value,
        past_key_values: list[tuple[ir.Value, ...]],
        *,
        inputs_embeds: ir.Value | None = None,
        ple_input_ids: ir.Value | None = None,
    ):
        if ple_input_ids is None:
            if input_ids is None:
                raise ValueError(
                    "Qwen4-Exp VL decoding requires explicit ple_input_ids "
                    "when input_ids is absent"
                )
            ple_input_ids = input_ids
        mrope_position_ids = op.Slice(position_ids, [1], [4], [0])
        past_mrope_position_ids = op.Slice(past_position_ids, [1], [4], [0])
        hidden_states, presents, present_position_ids = self.model(
            op,
            input_ids,
            attention_mask,
            mrope_position_ids,
            past_mrope_position_ids,
            past_key_values,
            inputs_embeds=inputs_embeds,
            ple_input_ids=ple_input_ids,
            # QSA rotates its query/index keys with the same multimodal RoPE
            # coordinates as sparse attention. Channel 0 is only the text/
            # causal sequence axis retained for generation bookkeeping.
            qsa_position_ids=mrope_position_ids,
            past_qsa_position_ids=past_mrope_position_ids,
        )
        present_text_position_ids = op.Concat(
            op.Slice(past_position_ids, [0], [1], [0]),
            op.Slice(position_ids, [0], [1], [0]),
            axis=-1,
        )
        present_position_ids = op.Concat(
            present_text_position_ids,
            present_position_ids,
            axis=0,
        )
        return self.lm_head(op, hidden_states), presents, present_position_ids


class Qwen4ExpVisionEncoderModel(Qwen3VLVisionEncoderModel):
    """Qwen4-Exp vision encoder with a float32 processor boundary."""

    def forward(
        self,
        op: OpBuilder,
        pixel_values: ir.Value,
        image_grid_thw: ir.Value,
    ):
        # The HF image processor emits float32 packed patches. Cast once to
        # the checkpoint dtype before the Conv3d patch projection.
        pixel_values = op.CastLike(pixel_values, self.visual.patch_embed.weight)
        return super().forward(op, pixel_values, image_grid_thw)


class Qwen4ExpImageOnlyEmbeddingModel(Qwen25VLEmbeddingModel):
    """Image-only mixer that fails execution when a video token is present."""

    def __init__(self, config: Qwen4ExpConfig):
        super().__init__(config)
        if config.unsupported_video_token_id is None:
            raise ValueError(
                "Qwen4-Exp image-only embedding requires the source video token "
                "for its execution guard"
            )
        self._unsupported_video_token_id = config.unsupported_video_token_id
        self._pad_token_id = config.pad_token_id

    def forward(
        self,
        op: OpBuilder,
        input_ids: ir.Value,
        image_features: ir.Value,
    ):
        video_mask = op.Equal(
            input_ids,
            op.Constant(value_int=self._unsupported_video_token_id),
        )
        safe_input_ids = op.Where(
            video_mask,
            op.Expand(
                op.Constant(value_int=self._pad_token_id),
                op.Shape(input_ids),
            ),
            input_ids,
        )
        inputs_embeds = super().forward(
            op,
            safe_input_ids,
            image_features,
        )

        # Reshape one zero element to [1] for image/text and [2] for video.
        # ONNX Reshape requires equal input/output element counts, so every ORT
        # EP rejects video before it can return sanitized embeddings. ORT CUDA
        # calls the shared ReshapeHelper for this check
        # (onnxruntime@b1f76d58, cuda/tensor/reshape.h); unlike out-of-range
        # Gather, this cannot fail open by zero-filling on CUDA.
        video_present = op.ReduceMax(
            op.Cast(video_mask, to=ir.DataType.INT64),
            keepdims=False,
        )
        guard_shape = op.Unsqueeze(
            op.Add(video_present, op.Constant(value_int=1)),
            [0],
        )
        guard = op.Reshape(
            op.CastLike(op.Constant(value_floats=[0.0]), inputs_embeds),
            guard_shape,
        )
        return op.Add(inputs_embeds, op.ReduceSum(guard, keepdims=False))


class Qwen4ExpForConditionalGeneration(nn.Module):
    """Qwen3.8 Flash-Next/Qwen4-Exp three-model vision-language pipeline.

    Reuses the source-identical Qwen3/Qwen3.5 packed vision encoder and
    embedding mixer, while preserving the Qwen4-Exp decoder's independent PLE
    token stream and heterogeneous recurrent, QSA, and position state.
    """

    default_task: str = "qwen4-exp-vision-language"
    category: str = "Multimodal"
    config_class: type = Qwen4ExpConfig
    HF_COMPONENT_SOURCES: ClassVar[dict[str, tuple[str, ...]]] = {
        "decoder": (
            "model.language_model.layers",
            "model.language_model.rotary_emb",
            "model.language_model.hyper_connection_mixer",
            "lm_head",
        ),
        "vision_encoder": ("model.visual",),
        "embedding": ("model.language_model.embed_tokens",),
    }

    def __init__(self, config: Qwen4ExpConfig):
        super().__init__()
        if config.vision is None:
            raise ValueError("Qwen4-Exp multimodal export requires a vision config")
        if config.video_token_id is not None:
            raise ValueError(
                "Qwen4-Exp video inputs are unsupported until the package "
                "defines a video producer and preprocessing route"
            )
        if config.deepstack_visual_indexes or config.vision.deepstack_visual_indexes:
            raise ValueError("Qwen4-Exp multimodal export does not support DeepStack")
        self.config = config
        self.decoder = Qwen4ExpVLDecoderModel(config)
        self.vision_encoder = Qwen4ExpVisionEncoderModel(config)
        self.embedding = Qwen4ExpImageOnlyEmbeddingModel(config)

    def forward(self, op: OpBuilder, **kwargs):
        raise NotImplementedError(
            "Qwen4ExpForConditionalGeneration uses Qwen4ExpVisionLanguageTask, "
            "which builds decoder, vision_encoder, and embedding independently."
        )

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Route official composite weights to the three ONNX components."""
        decoder = self.decoder.preprocess_weights(dict(state_dict))
        vision = self.vision_encoder.preprocess_weights(dict(state_dict))
        result = {f"decoder.{name}": value for name, value in decoder.items()}
        result.update((f"vision_encoder.{name}", value) for name, value in vision.items())
        embed = decoder.get("model.embed_tokens.weight")
        if embed is not None:
            result["embedding.embed_tokens.weight"] = embed
        return result
