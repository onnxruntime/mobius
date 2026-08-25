# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""LiquidAI LFM2 hybrid short-convolution and GQA causal language model."""

from __future__ import annotations

import dataclasses
from typing import TypeVar

import onnx_ir as ir
import torch
from onnxscript import OpBuilder, nn

from mobius._configs import ArchitectureConfig, Lfm2Config, Lfm2MoeConfig
from mobius.components import (
    MLP,
    Attention,
    Embedding,
    GatedShortConv,
    MoELayer,
    RMSNorm,
    create_padding_mask,
    initialize_rope,
)
from mobius.models.base import CausalLMModel
from mobius.models.moe import _rename_moe_expert_weights

_ConfigT = TypeVar("_ConfigT", bound=ArchitectureConfig)


def apply_lfm2_config_defaults(config: _ConfigT) -> _ConfigT:
    """Materialise the config knobs that HuggingFace's ``Lfm2`` hardcodes.

    LFM2 always uses per-head Q/K RMSNorm and a SiLU-gated feed-forward
    block, and ``Lfm2MLP`` adjusts the configured ``intermediate_size`` at
    construction time. Resolving both here keeps the text-only and
    vision-language entry points on exactly the same decoder configuration.
    """
    updates: dict[str, object] = {
        "attn_qk_norm": True,
        "hidden_act": config.hidden_act or "silu",
    }
    if isinstance(config, Lfm2Config):
        updates["intermediate_size"] = config.effective_intermediate_size
        # The width is now materialized; prevent subsequent consumers from
        # applying the HuggingFace construction-time adjustment again.
        updates["block_auto_adjust_ff_dim"] = False
    return dataclasses.replace(config, **updates)


def rename_lfm2_weight_key(key: str) -> str:
    """Map one upstream LFM2 projection name onto the shared components."""
    return (
        key.replace(".self_attn.out_proj.", ".self_attn.o_proj.")
        .replace(".self_attn.q_layernorm.", ".self_attn.q_norm.")
        .replace(".self_attn.k_layernorm.", ".self_attn.k_norm.")
        .replace(".feed_forward.w1.", ".feed_forward.gate_proj.")
        .replace(".feed_forward.w3.", ".feed_forward.up_proj.")
        .replace(".feed_forward.w2.", ".feed_forward.down_proj.")
    )


def rename_lfm2_weights(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Map upstream LFM2 projection names to shared mobius components."""
    return {rename_lfm2_weight_key(key): value for key, value in state_dict.items()}


class Lfm2RMSNorm(RMSNorm):
    """LFM2 RMSNorm with fp32 variance accumulation, matching Transformers."""

    def forward(self, op: OpBuilder, hidden_states: ir.Value) -> ir.Value:
        # stash_type=FLOAT computes normalization in fp32, casts the normalized
        # activation back to the input dtype, then applies gamma. This is the
        # exact ordering used by Transformers' LlamaRMSNorm and also exposes the
        # standard op to the residual SkipSimplifiedLayerNorm fusion.
        return op.RMSNormalization(
            hidden_states,
            self.weight,
            axis=-1,
            epsilon=self.variance_epsilon,
            stash_type=ir.DataType.FLOAT,
        )


class Lfm2DecoderLayer(nn.Module):
    """LFM2 pre-norm decoder layer with either short convolution or full GQA."""

    def __init__(self, config: ArchitectureConfig, layer_idx: int):
        super().__init__()
        layer_types = config.layer_types or []
        self.layer_type = (
            layer_types[layer_idx] if layer_idx < len(layer_types) else "full_attention"
        )
        if self.layer_type == "conv":
            self.conv = GatedShortConv(
                config.hidden_size,
                config.short_conv_kernel,
                bias=config.short_conv_bias,
            )
        else:
            self.self_attn = Attention(config, rms_norm_class=Lfm2RMSNorm)

        self.feed_forward = MLP(config)
        self.operator_norm = Lfm2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.ffn_norm = Lfm2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        attention_mask: ir.Value,
        attention_bias: ir.Value,
        position_embeddings: tuple[ir.Value, ir.Value],
        past_key_value: tuple[ir.Value, ...],
    ) -> tuple[ir.Value, tuple[ir.Value, ...]]:
        residual = hidden_states
        operator_input = self.operator_norm(op, hidden_states)

        if self.layer_type == "conv":
            (conv_state,) = past_key_value
            operator_output, present_state = self.conv(
                op,
                operator_input,
                conv_state,
                attention_mask,
            )
            present_key_value = (present_state,)
        else:
            operator_output, present_key_value = self.self_attn(
                op,
                hidden_states=operator_input,
                attention_bias=attention_bias,
                position_embeddings=position_embeddings,
                past_key_value=past_key_value,
            )

        # Both operators are pre-normalized and feed separate residual branches.
        hidden_states = op.Add(residual, operator_output)  # (B, T, H)
        feed_forward = self.feed_forward(op, self.ffn_norm(op, hidden_states))
        hidden_states = op.Add(hidden_states, feed_forward)  # (B, T, H)
        return hidden_states, present_key_value


class Lfm2MoETopKGate(nn.Module):
    """LFM2MoE sigmoid router with selection-only correction bias."""

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int,
        *,
        norm_topk_prob: bool,
        routed_scaling_factor: float,
    ):
        super().__init__()
        self.weight = nn.Parameter([num_experts, hidden_size])
        self._top_k = top_k
        self._norm_topk_prob = norm_topk_prob
        self._routed_scaling_factor = routed_scaling_factor

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        expert_bias: ir.Value | None,
    ) -> tuple[ir.Value, ir.Value]:
        router_logits = op.MatMul(hidden_states, op.Transpose(self.weight, perm=[1, 0]))
        routing_probs = op.Sigmoid(router_logits)
        selection_scores = routing_probs
        if expert_bias is not None:
            # The checkpoint keeps the learned correction bias in fp32. It changes
            # expert selection but never the probability used to combine experts.
            selection_scores = op.Add(
                op.Cast(routing_probs, to=ir.DataType.FLOAT),
                expert_bias,
            )
        _, selected_experts = op.TopK(
            selection_scores,
            op.Constant(value_ints=[self._top_k]),
            axis=-1,
            _outputs=2,
        )
        routing_weights = op.GatherElements(routing_probs, selected_experts, axis=-1)
        if self._norm_topk_prob:
            weight_sum = op.ReduceSum(routing_weights, [-1], keepdims=True)
            routing_weights = op.Div(
                routing_weights,
                op.Add(weight_sum, op.CastLike(1e-6, weight_sum)),
            )
        if self._routed_scaling_factor != 1.0:  # noqa: RUF069
            routing_weights = op.Mul(
                routing_weights,
                op.CastLike(self._routed_scaling_factor, routing_weights),
            )
        return routing_weights, selected_experts


class Lfm2MoEFeedForward(MoELayer):
    """LFM2MoE routed SwiGLU experts with optional selection correction bias."""

    def __init__(self, config: Lfm2MoeConfig):
        assert config.num_local_experts is not None
        assert config.num_experts_per_tok is not None
        gate = Lfm2MoETopKGate(
            config.hidden_size,
            config.num_local_experts,
            config.num_experts_per_tok,
            norm_topk_prob=config.norm_topk_prob,
            routed_scaling_factor=config.routed_scaling_factor,
        )
        super().__init__(config, gate=gate)
        self.expert_bias = (
            nn.Parameter([config.num_local_experts], dtype=ir.DataType.FLOAT)
            if config.use_expert_bias
            else None
        )

    def forward(self, op: OpBuilder, hidden_states: ir.Value) -> ir.Value:
        return super().forward(op, hidden_states, self.expert_bias)


class Lfm2MoEDecoderLayer(Lfm2DecoderLayer):
    """LFM2MoE hybrid operator layer with dense-prefix or routed feed-forward."""

    def __init__(self, config: Lfm2MoeConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        if layer_idx >= config.num_dense_layers:
            self.feed_forward = Lfm2MoEFeedForward(config)


class Lfm2TextModel(nn.Module):
    """LFM2 decoder backbone with mixed convolution and full-attention layers."""

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.embed_tokens = Embedding(
            config.vocab_size,
            config.hidden_size,
            config.pad_token_id,
        )
        self.layers = nn.ModuleList(
            [Lfm2DecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.embedding_norm = Lfm2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = initialize_rope(config)

    def forward(
        self,
        op: OpBuilder,
        input_ids: ir.Value | None,
        attention_mask: ir.Value,
        position_ids: ir.Value,
        past_key_values: list[tuple[ir.Value, ...]] | None = None,
        inputs_embeds: ir.Value | None = None,
    ) -> tuple[ir.Value, list[tuple[ir.Value, ...]]]:
        # Multimodal callers pre-fuse image features and pass inputs_embeds.
        if inputs_embeds is None:
            assert input_ids is not None, "one of input_ids/inputs_embeds is required"
            inputs_embeds = self.embed_tokens(op, input_ids)  # (B, T) -> (B, T, H)
        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(op, position_ids)
        # ONNX Attention applies causality internally; this mask carries padding only.
        # Only dims 0 and 1 of the first argument are read, so passing the
        # rank-3 embeddings works for both the token and embeds entry points.
        attention_bias = create_padding_mask(
            op,
            input_ids=hidden_states,
            attention_mask=attention_mask,
        )

        present_key_values = []
        past_kvs = past_key_values or [None] * len(self.layers)
        for layer, past_kv in zip(self.layers, past_kvs):
            hidden_states, present_kv = layer(
                op,
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                attention_bias=attention_bias,
                position_embeddings=position_embeddings,
                past_key_value=past_kv,
            )
            present_key_values.append(present_kv)

        hidden_states = self.embedding_norm(op, hidden_states)  # (B, T, H)
        return hidden_states, present_key_values


class Lfm2MoETextModel(Lfm2TextModel):
    """LFM2MoE decoder with the serialized dense-to-expert layer transition."""

    def __init__(self, config: Lfm2MoeConfig):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [Lfm2MoEDecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )


class Lfm2CausalLMModel(CausalLMModel):
    """LiquidAI LFM2 causal LM with double-gated short convolutions and QK-norm GQA."""

    default_task: str = "hybrid-text-generation"
    category: str = "Hybrid Convolution+Attention"
    config_class: type = Lfm2Config

    def __init__(self, config: ArchitectureConfig):
        # LFM2 hardcodes per-head Q/K RMSNorm and SiLU-gated feed-forward blocks.
        config = apply_lfm2_config_defaults(config)
        super().__init__(config)
        self.model = Lfm2TextModel(config)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def preprocess_weights(
        self,
        state_dict: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Map upstream LFM2 projection names to shared mobius components."""
        return super().preprocess_weights(rename_lfm2_weights(state_dict))


class Lfm2MoECausalLMModel(CausalLMModel):
    """LiquidAI LFM2MoE hybrid causal LM with correction-biased sigmoid routing."""

    default_task: str = "hybrid-text-generation"
    category: str = "Hybrid Convolution+Attention MoE"
    config_class: type = Lfm2MoeConfig

    def __init__(self, config: Lfm2MoeConfig):
        config = apply_lfm2_config_defaults(config)
        super().__init__(config)
        self.model = Lfm2MoETextModel(config)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def preprocess_weights(
        self,
        state_dict: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Map dense and routed LFM2MoE weights onto the dedicated graph."""
        state_dict = rename_lfm2_weights(state_dict)
        state_dict = _rename_moe_expert_weights(state_dict)
        return super().preprocess_weights(state_dict)
