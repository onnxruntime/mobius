# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""T5 encoder-decoder model."""

from __future__ import annotations

import math
from typing import ClassVar

import onnx_ir as ir
import torch
from onnxscript import OpBuilder, nn

from mobius._configs import ArchitectureConfig
from mobius._weight_utils import preprocess_quantized_weights
from mobius.components._activations import ACT2FN
from mobius.components._common import Embedding, Linear
from mobius.components._encoder_decoder_attention import (
    EncoderDecoderAttention,
)
from mobius.components._rms_norm import RMSNorm
from mobius.models.base import embedding_for_config, linear_class_for_config

# ---------------------------------------------------------------------------
# T5 Relative Position Bias
# ---------------------------------------------------------------------------


def _relative_position_bucket(
    op: OpBuilder,
    relative_position,
    *,
    bidirectional: bool,
    num_buckets: int,
    max_distance: int,
):
    """Map relative positions to bucket indices (T5-style log-linear).

    Implements the same bucketing as HuggingFace
    ``T5Attention._relative_position_bucket``.

    Args:
        op: ONNX op builder.
        relative_position: INT64 tensor [query_length, key_length].
        bidirectional: True for encoder, False for decoder.
        num_buckets: Number of relative position buckets (e.g. 32).
        max_distance: Maximum distance for bucketing (e.g. 128).

    Returns:
        INT64 tensor [query_length, key_length] of bucket indices
        in ``[0, num_buckets)``.
    """
    zero = op.Constant(value_int=0)

    if bidirectional:
        # Half buckets for positive, half for negative relative positions
        half_buckets = num_buckets // 2
        is_positive = op.Greater(relative_position, zero)
        is_positive_int = op.Cast(is_positive, to=7)  # INT64
        relative_buckets = op.Mul(is_positive_int, op.Constant(value_int=half_buckets))
        abs_position = op.Abs(relative_position)
        effective_buckets = half_buckets
    else:
        # Unidirectional: only negative relative positions (past tokens)
        neg_position = op.Neg(op.Min(relative_position, zero))
        abs_position = neg_position
        relative_buckets = op.Expand(zero, op.Shape(relative_position))
        effective_buckets = num_buckets

    max_exact = effective_buckets // 2

    # Small positions: use direct index
    is_small = op.Less(abs_position, op.Constant(value_int=max_exact))

    # Large positions: log-linear bucketing
    abs_float = op.Cast(abs_position, to=1)  # FLOAT32
    # Clamp to avoid log(0); doesn't affect result because Where
    # selects the is_small path for abs_position < max_exact
    abs_clamped = op.Max(abs_float, 1.0)
    log_ratio = op.Log(op.Div(abs_clamped, float(max_exact)))
    log_scale = math.log(max_distance / max_exact)
    bucket_float = op.Add(
        float(max_exact),
        op.Mul(
            log_ratio,
            float(effective_buckets - max_exact) / log_scale,
        ),
    )
    large_bucket = op.Cast(bucket_float, to=7)  # INT64
    large_bucket = op.Min(large_bucket, op.Constant(value_int=effective_buckets - 1))

    # Select small or large bucket
    final_offset = op.Where(is_small, abs_position, large_bucket)
    return op.Add(relative_buckets, final_offset)


def _compute_position_bias(
    op: OpBuilder,
    embedding: Embedding,
    query_length,
    key_length,
    *,
    bidirectional: bool,
    num_buckets: int,
    max_distance: int,
    num_heads: int,
    query_offset=None,
):
    """Compute T5-style relative position bias from learned embeddings.

    Uses log-linear bucketing of relative positions (HuggingFace
    ``T5Attention.compute_bias``). Bidirectional for encoder,
    unidirectional for decoder self-attention.

    Args:
        op: ONNX op builder.
        embedding: Learned relative attention bias embedding
            with shape ``[num_buckets, num_heads]``.
        query_length: Scalar INT64 — number of query positions.
        key_length: Scalar INT64 — number of key positions.
        bidirectional: True for encoder, False for decoder.
        num_buckets: Number of relative position buckets.
        max_distance: Maximum distance for bucketing.
        num_heads: Number of attention heads.
        query_offset: Scalar INT64 — offset for query positions
            (e.g. past_sequence_length for decode steps). If None,
            query positions start at 0.

    Returns:
        Position bias tensor of shape
        ``[1, num_heads, query_length, key_length]`` (FLOAT32).
    """
    # Query positions: [query_offset, query_offset + query_length)
    if query_offset is None:
        query_offset = op.Constant(value_int=0)
    query_end = op.Add(query_offset, query_length)
    # context_position: [query_length]
    context_position = op.Range(query_offset, query_end, op.Constant(value_int=1))
    # memory_position: [key_length]
    memory_position = op.Range(op.Constant(value_int=0), key_length, op.Constant(value_int=1))

    # Relative position: memory - context → [query_length, key_length]
    context_2d = op.Unsqueeze(context_position, [1])
    memory_2d = op.Unsqueeze(memory_position, [0])
    relative_position = op.Sub(memory_2d, context_2d)

    # Map relative positions to bucket indices
    bucket_indices = _relative_position_bucket(
        op,
        relative_position,
        bidirectional=bidirectional,
        num_buckets=num_buckets,
        max_distance=max_distance,
    )

    # Gather from learned embedding → [query_len, key_len, num_heads]
    values = embedding(op, bucket_indices)
    # Transpose to [num_heads, query_length, key_length]
    values = op.Transpose(values, perm=[2, 0, 1])
    # Unsqueeze to [1, num_heads, query_length, key_length]
    values = op.Unsqueeze(values, [0])
    return values


# ---------------------------------------------------------------------------
# T5 Components
# ---------------------------------------------------------------------------


class T5EncoderBlock(nn.Module):
    """T5 encoder block: pre-norm self-attention + pre-norm FFN."""

    def __init__(self, config: ArchitectureConfig, *, has_relative_attention_bias: bool):
        super().__init__()
        linear_class = linear_class_for_config(config) or Linear
        # T5 does not scale attention scores by 1/sqrt(d_k)
        self.self_attn = EncoderDecoderAttention(
            config, bias=False, scale=1.0, linear_class=linear_class
        )
        self.self_attn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.ffn = _T5FFN(config, linear_class=linear_class)
        self.ffn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.relative_attention_bias = (
            Embedding(config.relative_attention_num_buckets, config.num_attention_heads)
            if has_relative_attention_bias
            else None
        )
        self._num_buckets = config.relative_attention_num_buckets
        self._max_distance = config.relative_attention_max_distance
        self._num_heads = config.num_attention_heads

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        attention_bias: ir.Value | None = None,
        attention_mask: ir.Value | None = None,
    ):
        if self.relative_attention_bias is not None:
            seq_len = op.Shape(hidden_states, start=1, end=2)
            attention_bias = _compute_position_bias(
                op,
                self.relative_attention_bias,
                seq_len,
                seq_len,
                bidirectional=True,
                num_buckets=self._num_buckets,
                max_distance=self._max_distance,
                num_heads=self._num_heads,
            )
            if attention_mask is not None:
                attention_bias = _add_padding_bias(op, attention_bias, attention_mask)
        if attention_bias is None:
            raise ValueError("T5 encoder block requires a relative-attention bias")

        residual = hidden_states
        hidden_states = self.self_attn_norm(op, hidden_states)
        hidden_states, _ = self.self_attn(op, hidden_states, attention_bias=attention_bias)
        hidden_states = op.Add(residual, hidden_states)

        residual = hidden_states
        hidden_states = self.ffn_norm(op, hidden_states)
        hidden_states = self.ffn(op, hidden_states)
        hidden_states = op.Add(residual, hidden_states)

        return hidden_states, attention_bias


class T5DecoderBlock(nn.Module):
    """T5 decoder block: self-attn + cross-attn + FFN, all pre-norm."""

    def __init__(self, config: ArchitectureConfig, *, has_relative_attention_bias: bool):
        super().__init__()
        linear_class = linear_class_for_config(config) or Linear
        # T5 does not scale attention scores by 1/sqrt(d_k)
        self.self_attn = EncoderDecoderAttention(
            config,
            is_causal=True,
            bias=False,
            scale=1.0,
            linear_class=linear_class,
        )
        self.self_attn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.cross_attn = EncoderDecoderAttention(
            config,
            bias=False,
            scale=1.0,
            linear_class=linear_class,
            use_cross_attention_cache=True,
        )
        self.cross_attn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.ffn = _T5FFN(config, linear_class=linear_class)
        self.ffn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.relative_attention_bias = (
            Embedding(config.relative_attention_num_buckets, config.num_attention_heads)
            if has_relative_attention_bias
            else None
        )
        self._num_buckets = config.relative_attention_num_buckets
        self._max_distance = config.relative_attention_max_distance
        self._num_heads = config.num_attention_heads

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        encoder_hidden_states: ir.Value,
        attention_bias: ir.Value | None = None,
        cross_attention_bias: ir.Value | None = None,
        past_key_value: tuple | None = None,
        cross_past_key_value: ir.Value | None = None,
        query_length: ir.Value | None = None,
        key_length: ir.Value | None = None,
        query_offset: ir.Value | None = None,
        attention_mask: ir.Value | None = None,
    ):
        if self.relative_attention_bias is not None:
            if query_length is None or key_length is None:
                raise ValueError("T5 decoder bias owner requires query/key lengths")
            attention_bias = _compute_position_bias(
                op,
                self.relative_attention_bias,
                query_length,
                key_length,
                bidirectional=False,
                num_buckets=self._num_buckets,
                max_distance=self._max_distance,
                num_heads=self._num_heads,
                query_offset=query_offset,
            )
            if attention_mask is not None:
                attention_bias = _add_padding_bias(op, attention_bias, attention_mask)
        if attention_bias is None:
            raise ValueError("T5 decoder block requires a relative-attention bias")

        # Self-attention
        residual = hidden_states
        hidden_states = self.self_attn_norm(op, hidden_states)
        hidden_states, self_kv = self.self_attn(
            op, hidden_states, attention_bias=attention_bias, past_key_value=past_key_value
        )
        hidden_states = op.Add(residual, hidden_states)

        # Cross-attention
        residual = hidden_states
        hidden_states = self.cross_attn_norm(op, hidden_states)
        hidden_states, cross_kv = self.cross_attn(
            op,
            hidden_states,
            key_value_states=encoder_hidden_states,
            attention_bias=cross_attention_bias,
            past_key_value=cross_past_key_value,
        )
        hidden_states = op.Add(residual, hidden_states)

        # FFN
        residual = hidden_states
        hidden_states = self.ffn_norm(op, hidden_states)
        hidden_states = self.ffn(op, hidden_states)
        hidden_states = op.Add(residual, hidden_states)

        return hidden_states, self_kv, cross_kv, attention_bias


class _T5FFN(nn.Module):
    """T5 feed-forward network.

    Standard T5 uses ``wi → act → wo``.
    Gated variants (mT5, FLAN-T5, UL2) use ``(wi_0(x) * act(wi_1(x))) → wo``
    where wi_0 is the gate and wi_1 is the up-projection.
    """

    def __init__(self, config: ArchitectureConfig, *, linear_class: type = Linear):
        super().__init__()
        self._is_gated = config.is_gated_act
        if self._is_gated:
            # Gated FFN: gate (wi_0) and up-projection (wi_1)
            self.wi_0 = linear_class(config.hidden_size, config.intermediate_size, bias=False)
            self.wi_1 = linear_class(config.hidden_size, config.intermediate_size, bias=False)
        else:
            self.wi = linear_class(config.hidden_size, config.intermediate_size, bias=False)
        self.wo = linear_class(config.intermediate_size, config.hidden_size, bias=False)
        self._act_fn = ACT2FN[config.hidden_act]

    def forward(self, op: OpBuilder, hidden_states: ir.Value):
        if self._is_gated:
            gate = self._act_fn(op, self.wi_0(op, hidden_states))
            hidden_states = op.Mul(gate, self.wi_1(op, hidden_states))
        else:
            hidden_states = self.wi(op, hidden_states)
            hidden_states = self._act_fn(op, hidden_states)
        return self.wo(op, hidden_states)


# ---------------------------------------------------------------------------
# T5 Encoder and Decoder top-level models
# ---------------------------------------------------------------------------


class T5Encoder(nn.Module):
    """T5 encoder stack."""

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.embed_tokens = embedding_for_config(config)
        relative_bias_layers = set(config.encoder_relative_attention_bias_layers or [0])
        self.block = nn.ModuleList(
            [
                T5EncoderBlock(config, has_relative_attention_bias=i in relative_bias_layers)
                for i in range(config.num_hidden_layers)
            ]
        )
        if not self.block or self.block[0].relative_attention_bias is None:
            raise ValueError("T5 encoder layer 0 must own a relative-attention bias tensor")
        self.final_layer_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        op: OpBuilder,
        input_ids: ir.Value,
        attention_mask: ir.Value | None = None,
    ):
        hidden_states = self.embed_tokens(op, input_ids)
        # Compute T5 relative position bias (bidirectional for encoder).
        # Shape: [1, num_heads, seq_len, seq_len]
        fallback_position_bias = None
        for block in self.block:
            hidden_states, position_bias = block(
                op,
                hidden_states,
                attention_bias=(
                    None
                    if block.relative_attention_bias is not None
                    else fallback_position_bias
                ),
                attention_mask=attention_mask,
            )
            if fallback_position_bias is None:
                fallback_position_bias = position_bias
        hidden_states = self.final_layer_norm(op, hidden_states)
        return hidden_states


class T5Decoder(nn.Module):
    """T5 decoder stack with cross-attention and KV cache."""

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.embed_tokens = embedding_for_config(config)
        self._hidden_size = config.hidden_size
        # HF T5 recently introduced scale_decoder_outputs (decoupled from
        # tie_word_embeddings). Original T5 sets it True; FLAN-T5/UL2 set
        # it False. MT5 doesn't have this field and never scales.
        self._scale_decoder_outputs = bool(config.scale_decoder_outputs)
        num_decoder_layers = config.num_decoder_layers or config.num_hidden_layers
        relative_bias_layers = set(config.decoder_relative_attention_bias_layers or [0])
        self.block = nn.ModuleList(
            [
                T5DecoderBlock(config, has_relative_attention_bias=i in relative_bias_layers)
                for i in range(num_decoder_layers)
            ]
        )
        if not self.block or self.block[0].relative_attention_bias is None:
            raise ValueError("T5 decoder layer 0 must own a relative-attention bias tensor")
        self.final_layer_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        quantization = config.quantization
        quantize_lm_head = quantization is not None and quantization.quantize_lm_head
        linear_class = linear_class_for_config(config) if quantize_lm_head else Linear
        assert linear_class is not None
        self.lm_head = linear_class(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        op: OpBuilder,
        input_ids: ir.Value,
        encoder_hidden_states: ir.Value,
        attention_mask: ir.Value | None = None,
        encoder_attention_mask: ir.Value | None = None,
        past_key_values: list | None = None,
        cross_past_key_values: ir.Value | None = None,
    ):
        hidden_states = self.embed_tokens(op, input_ids)

        # Compute T5 relative position bias for decoder self-attention.
        # Unidirectional (bidirectional=False) since decoder is causal.
        # query_offset = past_sequence_length for decode steps.
        query_length = op.Shape(input_ids, start=1, end=2)
        if past_key_values is not None:
            # past_key_values[0][0]: [batch, heads, past_seq_len, head_dim]
            past_len = op.Shape(past_key_values[0][0], start=2, end=3)
            key_length = op.Add(past_len, query_length)
        else:
            past_len = None
            key_length = query_length
        past_kvs = past_key_values or [None] * len(self.block)
        cross_past_kvs = cross_past_key_values or [None] * len(self.block)
        present_self_kvs = []
        present_cross_kvs = []

        fallback_position_bias = None
        for block, past_kv, cross_kv in zip(self.block, past_kvs, cross_past_kvs):
            cross_attention_bias = None
            if encoder_attention_mask is not None:
                cross_attention_bias = _padding_bias(
                    op,
                    hidden_states,
                    encoder_attention_mask,
                    query_length=query_length,
                )
            hidden_states, self_kv, cross_kv_out, position_bias = block(
                op,
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_bias=(
                    None
                    if block.relative_attention_bias is not None
                    else fallback_position_bias
                ),
                cross_attention_bias=cross_attention_bias,
                past_key_value=past_kv,
                cross_past_key_value=cross_kv,
                query_length=query_length,
                key_length=key_length,
                query_offset=past_len,
                attention_mask=attention_mask,
            )
            if fallback_position_bias is None:
                fallback_position_bias = position_bias
            present_self_kvs.append(self_kv)
            present_cross_kvs.append(cross_kv_out)

        hidden_states = self.final_layer_norm(op, hidden_states)
        # T5 scales hidden states by 1/sqrt(d_model) before projecting to
        # vocab. Controlled by scale_decoder_outputs (newer HF) or
        # tie_word_embeddings (legacy). Original T5 and mT5 use True;
        # FLAN-T5/UL2 use False (separate lm_head weights).
        if self._scale_decoder_outputs:
            hidden_states = op.Mul(
                hidden_states,
                float(self._hidden_size**-0.5),
            )
        logits = self.lm_head(op, hidden_states)
        return logits, present_self_kvs, present_cross_kvs


# ---------------------------------------------------------------------------
# T5 Model (wraps encoder + decoder)
# ---------------------------------------------------------------------------


class T5ForConditionalGeneration(nn.Module):
    """T5 encoder-decoder model for conditional generation (seq2seq).

    This model produces a ModelPackage with separate encoder and decoder
    components for efficient inference.
    """

    default_task = "seq2seq"
    category = "encoder-decoder"
    uses_encoder_attention_mask = True

    # Runtime HF ``named_modules()`` sub-trees per ONNX component.
    HF_COMPONENT_SOURCES: ClassVar[dict[str, tuple[str, ...]]] = {
        "encoder": ("encoder",),
        "decoder": ("decoder", "lm_head"),
    }

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.config = config
        self.encoder = T5Encoder(config)
        self.decoder = T5Decoder(config)

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        new_state_dict = {}
        for name, tensor in state_dict.items():
            new_name = _rename_t5_weight(name, is_gated_act=self.config.is_gated_act)
            if new_name is not None:
                new_state_dict[new_name] = tensor
        # Shared embeddings: encoder and decoder use the same embedding
        if "encoder.embed_tokens.weight" not in new_state_dict:
            shared = new_state_dict.get("shared.weight")
            if shared is not None:
                new_state_dict["encoder.embed_tokens.weight"] = shared
                new_state_dict["decoder.embed_tokens.weight"] = shared
        # Tied lm_head
        if "decoder.lm_head.weight" not in new_state_dict:
            embed = new_state_dict.get("encoder.embed_tokens.weight")
            if embed is not None:
                new_state_dict["decoder.lm_head.weight"] = embed
        return preprocess_quantized_weights(
            new_state_dict,
            self.config.quantization,
            tie_embeddings=self.config.tie_word_embeddings,
            embed_key="encoder.embed_tokens.weight",
            head_key="decoder.lm_head.weight",
        )


class T5EncoderModel(nn.Module):
    """Encoder-only T5 model used as a diffusion prompt encoder."""

    default_task = "t5-text-encoding"
    category = "encoder"

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.config = config
        self.encoder = T5Encoder(config)

    def forward(
        self,
        op: OpBuilder,
        input_ids: ir.Value,
        attention_mask: ir.Value | None = None,
    ):
        return self.encoder(op, input_ids=input_ids, attention_mask=attention_mask)

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        new_state_dict = {}
        for name, tensor in state_dict.items():
            new_name = _rename_t5_weight(name, is_gated_act=self.config.is_gated_act)
            if new_name is not None and new_name.startswith("encoder."):
                new_state_dict[new_name] = tensor
        if "encoder.embed_tokens.weight" not in new_state_dict:
            shared = state_dict.get("shared.weight")
            if shared is not None:
                new_state_dict["encoder.embed_tokens.weight"] = shared
        return new_state_dict


# ---------------------------------------------------------------------------
# Weight name mapping
# ---------------------------------------------------------------------------

_T5_COMMON_RENAMES = {
    # Self-attention (same for encoder and decoder)
    "layer.0.SelfAttention.q.": "self_attn.q_proj.",
    "layer.0.SelfAttention.k.": "self_attn.k_proj.",
    "layer.0.SelfAttention.v.": "self_attn.v_proj.",
    "layer.0.SelfAttention.o.": "self_attn.out_proj.",
    "layer.0.layer_norm.": "self_attn_norm.",
}

# Encoder: layer.0 = self-attn, layer.1 = FFN
_T5_ENCODER_RENAMES = {
    "layer.1.DenseReluDense.wi.": "ffn.wi.",
    # Gated FFN variants (mT5, FLAN-T5, UL2)
    "layer.1.DenseReluDense.wi_0.": "ffn.wi_0.",
    "layer.1.DenseReluDense.wi_1.": "ffn.wi_1.",
    "layer.1.DenseReluDense.wo.": "ffn.wo.",
    "layer.1.layer_norm.": "ffn_norm.",
}

# Decoder: layer.0 = self-attn, layer.1 = cross-attn, layer.2 = FFN
_T5_DECODER_RENAMES = {
    "layer.1.EncDecAttention.q.": "cross_attn.q_proj.",
    "layer.1.EncDecAttention.k.": "cross_attn.k_proj.",
    "layer.1.EncDecAttention.v.": "cross_attn.v_proj.",
    "layer.1.EncDecAttention.o.": "cross_attn.out_proj.",
    "layer.1.layer_norm.": "cross_attn_norm.",
    "layer.2.DenseReluDense.wi.": "ffn.wi.",
    # Gated FFN variants (mT5, FLAN-T5, UL2)
    "layer.2.DenseReluDense.wi_0.": "ffn.wi_0.",
    "layer.2.DenseReluDense.wi_1.": "ffn.wi_1.",
    "layer.2.DenseReluDense.wo.": "ffn.wo.",
    "layer.2.layer_norm.": "ffn_norm.",
}


def _rename_t5_weight(name: str, *, is_gated_act: bool = False) -> str | None:
    """Rename a HF T5 weight to our naming convention.

    Encoder sublayers: layer.0=self-attn, layer.1=FFN.
    Decoder sublayers: layer.0=self-attn, layer.1=cross-attn, layer.2=FFN.
    """
    # Keep shared embedding as-is for now (handled by preprocess_weights)
    if name == "shared.weight":
        return "shared.weight"
    if name == "lm_head.weight":
        return "decoder.lm_head.weight"

    # encoder.block.{i}.layer.X.{...} or decoder.block.{i}.layer.X.{...}
    for prefix in ("encoder.", "decoder."):
        if not name.startswith(prefix):
            continue

        rest = name[len(prefix) :]

        # Final layer norm
        if rest.startswith("final_layer_norm."):
            return name  # Already correct naming

        # Block weights
        if rest.startswith("block."):
            parts = rest.split(".", 2)  # block, idx, remainder
            if len(parts) < 3:
                return None
            block_idx = parts[1]
            remainder = parts[2]
            if remainder.startswith(
                (
                    "relative_attention_bias.",
                    "self_attn.",
                    "self_attn_norm.",
                    "cross_attn.",
                    "cross_attn_norm.",
                    "ffn.",
                    "ffn_norm.",
                )
            ):
                return name

            # Keep relative bias on its source layer. llama.cpp permits later
            # layers to override layer 0, which is otherwise the stack fallback.
            rel_bias_key = "layer.0.SelfAttention.relative_attention_bias."
            if remainder.startswith(rel_bias_key):
                suffix = remainder[len(rel_bias_key) :]
                return f"{prefix}block.{block_idx}.relative_attention_bias.{suffix}"

            # Pick context-specific rename table
            extra = _T5_ENCODER_RENAMES if prefix == "encoder." else _T5_DECODER_RENAMES

            # Try common renames first, then context-specific
            for table in (_T5_COMMON_RENAMES, extra):
                for old, new in table.items():
                    if remainder.startswith(old):
                        suffix = remainder[len(old) :]
                        if not is_gated_act and new == "ffn.wi_1.":
                            new = "ffn.wi."
                        return f"{prefix}block.{block_idx}.{new}{suffix}"

    return None


def _padding_bias(
    op: OpBuilder,
    like: ir.Value,
    attention_mask: ir.Value,
    *,
    query_length: ir.Value | None = None,
) -> ir.Value:
    """Convert a 1/0 key mask to a broadcastable additive attention bias."""
    valid = op.CastLike(attention_mask, like)
    padding = op.Mul(
        op.Sub(op.CastLike(op.Constant(value_float=1.0), like), valid),
        op.CastLike(op.Constant(value_float=-10000.0), like),
    )
    padding = op.Unsqueeze(padding, op.Constant(value_ints=[1, 2]))
    if query_length is not None:
        target_shape = op.Concat(
            op.Shape(attention_mask, start=0, end=1),
            op.Constant(value_ints=[1]),
            query_length,
            op.Shape(attention_mask, start=1, end=2),
            axis=0,
        )
        padding = op.Expand(padding, target_shape)
    return padding


def _add_padding_bias(
    op: OpBuilder, position_bias: ir.Value, attention_mask: ir.Value
) -> ir.Value:
    return op.Add(position_bias, _padding_bias(op, position_bias, attention_mask))
