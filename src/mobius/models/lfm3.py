# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""LiquidAI LFM3 hybrid decoder with sandwich normalization and head-gated GQA."""

from __future__ import annotations

import dataclasses

import onnx_ir as ir
from onnxscript import OpBuilder, nn

from mobius._configs import ArchitectureConfig, Lfm3Config
from mobius.components import (
    Attention,
    Embedding,
    Float32SiLU,
    GatedShortConv,
    Linear,
    ProportionalRope,
    RMSNorm,
    create_padding_mask,
)
from mobius.models.base import CausalLMModel


class Lfm3RMSNorm(RMSNorm):
    """Normalize in fp32, cast activations back, then multiply by gamma."""

    def forward(self, op: OpBuilder, hidden_states: ir.Value) -> ir.Value:
        return op.RMSNormalization(
            hidden_states,
            self.weight,
            axis=-1,
            epsilon=self.variance_epsilon,
            stash_type=ir.DataType.FLOAT,
        )


class Lfm3Attention(Attention):
    """Per-head sigmoid output gating and full-head, fp32 positional rotation."""

    def __init__(self, config: Lfm3Config):
        # Zero-tail frequencies encode partial rotation; never split a compact prefix.
        super().__init__(
            dataclasses.replace(config, partial_rotary_factor=1.0),
            rms_norm_class=Lfm3RMSNorm,
        )
        self.gate_proj = Linear(config.hidden_size, config.num_attention_heads, bias=False)

    def _apply_rotary(
        self, op: OpBuilder, x: ir.Value, position_embeddings: tuple, num_heads: int
    ) -> ir.Value:
        del num_heads
        cos, sin = position_embeddings
        # Tables have D/2 frequencies. Duplicate across the FULL head halves,
        # including zero-frequency tails (identity in both corresponding halves).
        cos = op.Unsqueeze(op.Concat(cos, cos, axis=-1), [2])  # (B, T, 1, D)
        sin = op.Unsqueeze(op.Concat(sin, sin, axis=-1), [2])
        cos = op.Cast(op.CastLike(cos, x), to=ir.DataType.FLOAT)
        sin = op.Cast(op.CastLike(sin, x), to=ir.DataType.FLOAT)
        heads = op.Reshape(x, [0, 0, -1, self.head_dim])  # (B, T, heads, D)
        heads_f32 = op.Cast(heads, to=ir.DataType.FLOAT)
        first, second = op.Split(heads_f32, num_outputs=2, axis=-1, _outputs=2)
        rotated = op.Concat(op.Neg(second), first, axis=-1)
        result = op.Add(op.Mul(heads_f32, cos), op.Mul(rotated, sin))
        return op.Reshape(op.CastLike(result, x), [0, 0, -1])

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        attention_bias: ir.Value,
        position_embeddings: tuple,
        past_key_value: tuple,
    ):
        gate = op.Unsqueeze(op.Sigmoid(self.gate_proj(op, hidden_states)), [-1])
        gate_shape = op.Concat(
            op.Shape(hidden_states, start=0, end=2),
            [self.num_attention_heads, self.head_dim],
            axis=0,
        )
        gate = op.Reshape(op.Expand(gate, gate_shape), [0, 0, -1])  # (B, T, heads * D)
        return super().forward(
            op,
            hidden_states,
            attention_bias,
            position_embeddings,
            past_key_value,
            output_gate=gate,
        )


class Lfm3MLP(nn.Module):
    """Direct-width SwiGLU, aligned with upstream w1/w3/w2 parameter names."""

    def __init__(self, config: Lfm3Config):
        super().__init__()
        self.w1 = Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.w3 = Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.w2 = Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.activation = Float32SiLU()

    def forward(self, op: OpBuilder, x: ir.Value) -> ir.Value:
        return self.w2(op, op.Mul(self.activation(op, self.w1(op, x)), self.w3(op, x)))


class Lfm3DecoderLayer(nn.Module):
    """Conv/full/sliding operator and FF residual branches with optional post norms."""

    def __init__(self, config: Lfm3Config, layer_idx: int):
        super().__init__()
        self.layer_type = config.layer_types[layer_idx]
        if self.layer_type == "conv":
            self.conv = GatedShortConv(
                config.hidden_size, config.short_conv_kernel, bias=config.short_conv_bias
            )
        else:
            self.self_attn = Lfm3Attention(config)
        self.feed_forward = Lfm3MLP(config)
        self.operator_norm = Lfm3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.ffn_norm = Lfm3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_operator_norm = (
            Lfm3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            if config.use_sandwich_norm
            else None
        )
        self.post_ffn_norm = (
            Lfm3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            if config.use_sandwich_norm
            else None
        )

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        attention_mask: ir.Value,
        current_mask: ir.Value,
        attention_bias: ir.Value | None,
        position_embeddings: tuple | None,
        past_key_value: tuple,
    ):
        operator_input = self.operator_norm(op, hidden_states)
        if self.layer_type == "conv":
            output, state = self.conv(
                op, operator_input, past_key_value[0], attention_mask, current_mask
            )
            present = (state,)
        else:
            output, present = self.self_attn(
                op, operator_input, attention_bias, position_embeddings, past_key_value
            )
        if self.post_operator_norm is not None:
            output = self.post_operator_norm(op, output)
        hidden_states = op.Add(hidden_states, output)  # (B, T, H)
        ffn = self.feed_forward(op, self.ffn_norm(op, hidden_states))
        if self.post_ffn_norm is not None:
            ffn = self.post_ffn_norm(op, ffn)
        return op.Add(hidden_states, ffn), present


def _sliding_mask(
    op: OpBuilder, input_ids: ir.Value, attention_mask: ir.Value, window: int
) -> ir.Value:
    """Mask physical cache slots, not cumsum positions (padding also occupies slots)."""
    total = op.Shape(attention_mask, start=1, end=2)
    query_len = op.Shape(input_ids, start=1, end=2)
    key_slots = op.Range(0, op.Squeeze(total, [0]), 1)  # (past + T,)
    query_slots = op.Range(
        op.Squeeze(op.Sub(total, query_len), [0]), op.Squeeze(total, [0]), 1
    )
    distance = op.Sub(op.Unsqueeze(query_slots, [1]), op.Unsqueeze(key_slots, [0]))
    allowed = op.And(op.GreaterOrEqual(distance, 0), op.Less(distance, window))
    allowed = op.And(
        op.Unsqueeze(allowed, [0]),
        op.Cast(op.Unsqueeze(attention_mask, [1]), to=ir.DataType.BOOL),
    )
    return op.Unsqueeze(allowed, [1])  # (B, 1, T, past + T)


class Lfm3TextModel(nn.Module):
    """Lfm3Model: token IDs and growing hybrid cache -> normalized hidden states/cache.

    Position IDs must describe ordinary, unpacked sequences. Document restarts
    and upstream's segmented-convolution/packed attention are not supported.
    Sliding layers retain a growing cache, with a bounded attention window;
    this deliberately preserves global cache-slot indices across chunks.
    """

    def __init__(self, config: Lfm3Config):
        super().__init__()
        self.config = config
        self.embed_tokens = Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [Lfm3DecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        ropes = {}
        for kind in set(config.layer_types) - {"conv"}:
            params = config.rope_parameters[kind]
            ropes[kind] = ProportionalRope(
                dataclasses.replace(
                    config,
                    rope_theta=params["rope_theta"],
                    partial_rotary_factor=params.get("partial_rotary_factor", 1.0),
                )
            )
        self._rope_types = sorted(ropes)
        self.rotary_emb = nn.ModuleList([ropes[kind] for kind in self._rope_types])
        self.embedding_norm = Lfm3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        op: OpBuilder,
        input_ids: ir.Value,
        attention_mask: ir.Value,
        position_ids: ir.Value,
        past_key_values: list,
    ):
        hidden_states = self.embed_tokens(op, input_ids)
        embeddings = {
            kind: rope(op, position_ids)
            for kind, rope in zip(self._rope_types, self.rotary_emb, strict=True)
        }
        masks = {}
        if "full_attention" in embeddings:
            masks["full_attention"] = create_padding_mask(
                op, input_ids=input_ids, attention_mask=attention_mask
            )
        if "sliding_attention" in embeddings:
            masks["sliding_attention"] = _sliding_mask(
                op, input_ids, attention_mask, self.config.sliding_window
            )
        total = op.Shape(attention_mask, start=1, end=2)
        current_mask = op.Slice(
            attention_mask, op.Sub(total, op.Shape(input_ids, start=1, end=2)), total, [1]
        )
        present = []
        intermediates = []
        for i, (layer, past) in enumerate(zip(self.layers, past_key_values, strict=True)):
            kind = layer.layer_type
            hidden_states, states = layer(
                op,
                hidden_states,
                attention_mask,
                current_mask,
                masks.get(kind),
                embeddings.get(kind),
                past,
            )
            present.append(states)
            if self.config.output_layer_indices and i in self.config.output_layer_indices:
                intermediates.append(hidden_states)
        hidden_states = self.embedding_norm(op, hidden_states)
        if self.config.output_layer_indices:
            by_index = dict(zip(sorted(self.config.output_layer_indices), intermediates))
            return (
                hidden_states,
                present,
                [by_index[i] for i in self.config.output_layer_indices],
            )
        return hidden_states, present


class Lfm3CausalLMModel(CausalLMModel):
    """LiquidAI LFM3 causal LM with gated convolutions and full/sliding attention.

    Replicates HuggingFace Lfm3ForCausalLM with head gating and sandwich RMSNorm.

    Accepts unpacked input_ids/attention_mask/position_ids and reusable hybrid
    cache; emits full logits and updated global-index KV/conv cache slots.
    """

    default_task: str = "hybrid-text-generation"
    category: str = "Hybrid Convolution+Attention"
    config_class: type = Lfm3Config
    preserve_standard_attention: bool = True

    def __init__(self, config: ArchitectureConfig):
        if not isinstance(config, Lfm3Config):
            config = Lfm3Config(
                **{
                    field.name: getattr(config, field.name)
                    for field in dataclasses.fields(config)
                }
            )
        config = dataclasses.replace(config)
        super().__init__(config)
        self._replace_text_model(Lfm3TextModel(config))
