# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Hunyuan-V3 full-attention dense-prefix and routed/shared MoE language model."""

from __future__ import annotations

import dataclasses
import math

import onnx_ir as ir
import torch
from onnxscript import OpBuilder, nn

from mobius._configs import ArchitectureConfig, HyV3Config, HyV3MtpConfig
from mobius.components import (
    Attention,
    Embedding,
    Linear,
    MoELayer,
    RMSNorm,
    create_padding_mask,
    initialize_rope,
)
from mobius.components._attention import StaticCacheState
from mobius.components._mlp import MLP
from mobius.models.base import CausalLMModel, TextModel, linear_class_for_config
from mobius.models.moe import _rename_moe_expert_weights


class HyV3TopKGate(nn.Module):
    """Sigmoid top-k routing with selection-only expert bias."""

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int,
        *,
        normalize: bool,
        normalization_floor: float | None,
        normalization_epsilon: float | None,
        routed_scaling_factor: float,
    ):
        super().__init__()
        self.top_k = top_k
        self.normalize = normalize
        self.normalization_floor = normalization_floor
        self.normalization_epsilon = normalization_epsilon
        self.routed_scaling_factor = routed_scaling_factor
        if normalize and (normalization_floor is None) == (normalization_epsilon is None):
            raise ValueError(
                "HYV3 routing normalization requires exactly one floor or epsilon"
            )
        self.weight = nn.Parameter([num_experts, hidden_size])

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        e_score_correction_bias: ir.Value | None,
    ):
        # The reference router computes both the input and matrix product in fp32.
        router_input = op.Cast(hidden_states, to=ir.DataType.FLOAT)
        router_weight = op.Cast(self.weight, to=ir.DataType.FLOAT)
        router_logits = op.MatMul(router_input, op.Transpose(router_weight, perm=[1, 0]))
        routing_probs = op.Sigmoid(router_logits)  # (batch, sequence, experts)
        choice_scores = (
            routing_probs
            if e_score_correction_bias is None
            else op.Add(
                routing_probs,
                op.Cast(e_score_correction_bias, to=ir.DataType.FLOAT),
            )
        )
        _, selected_experts = op.TopK(
            choice_scores,
            op.Constant(value_ints=[self.top_k]),
            axis=-1,
            largest=1,
            sorted=0,
            _outputs=2,
        )
        routing_weights = op.GatherElements(routing_probs, selected_experts, axis=-1)
        if self.normalize:
            denominator = op.ReduceSum(routing_weights, [-1], keepdims=True)
            if self.normalization_epsilon is not None:
                denominator = op.Add(
                    denominator,
                    op.CastLike(self.normalization_epsilon, denominator),
                )
            else:
                assert self.normalization_floor is not None
                denominator = op.Max(
                    denominator,
                    op.CastLike(self.normalization_floor, denominator),
                )
            routing_weights = op.Div(routing_weights, denominator)
        if not math.isclose(self.routed_scaling_factor, 1.0):
            routing_weights = op.Mul(
                routing_weights,
                op.CastLike(self.routed_scaling_factor, routing_weights),
            )
        return routing_weights, selected_experts


class HyV3MoEBlock(MoELayer):
    """Gated SwiGLU experts plus an always-active, ungated shared SwiGLU expert."""

    def __init__(self, config: ArchitectureConfig, linear_class: type | None = None):
        assert config.num_local_experts is not None
        assert config.num_experts_per_tok is not None
        assert config.shared_expert_intermediate_size is not None
        floor = config.routing_weight_normalization_floor
        epsilon = getattr(config, "routing_weight_normalization_epsilon", None)
        gate = HyV3TopKGate(
            config.hidden_size,
            config.num_local_experts,
            config.num_experts_per_tok,
            normalize=config.norm_topk_prob,
            normalization_floor=floor,
            normalization_epsilon=epsilon,
            routed_scaling_factor=config.routed_scaling_factor,
        )
        super().__init__(config, gate=gate, linear_class=linear_class)
        self.e_score_correction_bias = (
            nn.Parameter([config.num_local_experts])
            if config.use_expert_bias
            else None
        )
        shared_config = dataclasses.replace(
            config, intermediate_size=config.shared_expert_intermediate_size
        )
        self.shared_experts = MLP(shared_config, linear_class=linear_class)
        self._fp32_combine = bool(getattr(config, "enable_moe_fp32_combine", True))

    def forward(self, op: OpBuilder, hidden_states: ir.Value):
        assert self.experts is not None
        routing_weights, selected_experts = self.gate(
            op, hidden_states, self.e_score_correction_bias
        )
        routed = None
        for expert_idx, expert in enumerate(self.experts):
            expert_output = expert(op, hidden_states)
            expert_id = op.Constant(value_int=expert_idx)
            match = op.Equal(selected_experts, expert_id)
            match_float = op.CastLike(match, routing_weights)
            weight = op.ReduceSum(
                op.Mul(routing_weights, match_float),
                [-1],
                keepdims=True,
            )
            # Official HYV3 multiplies by fp32 routing weights, then casts each
            # selected-expert contribution before accumulating in activation dtype.
            contribution = op.CastLike(
                op.Mul(op.Cast(expert_output, to=ir.DataType.FLOAT), weight),
                hidden_states,
            )
            routed = contribution if routed is None else op.Add(routed, contribution)
        assert routed is not None
        shared = self.shared_experts(op, hidden_states)
        if self._fp32_combine:
            combined = op.Add(
                op.Cast(routed, to=ir.DataType.FLOAT),
                op.Cast(shared, to=ir.DataType.FLOAT),
            )
            return op.CastLike(combined, hidden_states)
        return op.Add(routed, shared)


class HyV3DecoderLayer(nn.Module):
    """One pre-norm full-attention HYV3 block with dense or sparse feed-forward."""

    _supports_static_cache = True

    def __init__(
        self,
        config: ArchitectureConfig,
        layer_idx: int,
        *,
        force_moe: bool | None = None,
    ):
        super().__init__()
        linear_class = linear_class_for_config(config)
        self.self_attn = Attention(config, linear_class=linear_class)
        use_moe = (
            layer_idx >= config.first_k_dense_replace
            if force_moe is None
            else force_moe
        )
        self.mlp = (
            HyV3MoEBlock(config, linear_class=linear_class)
            if use_moe
            else MLP(config, linear_class=linear_class)
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        attention_bias: ir.Value | None,
        position_embeddings: tuple | None,
        past_key_value: tuple | StaticCacheState | None,
    ):
        if isinstance(past_key_value, StaticCacheState):
            static_cache = past_key_value
            past_key_value = None
        else:
            static_cache = None

        residual = hidden_states
        hidden_states = self.input_layernorm(op, hidden_states)
        hidden_states, present_key_value = self.self_attn(
            op,
            hidden_states,
            attention_bias,
            position_embeddings,
            past_key_value,
            static_cache=static_cache,
        )
        hidden_states = op.Add(residual, hidden_states)

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(op, hidden_states)
        hidden_states = self.mlp(op, hidden_states)
        return op.Add(residual, hidden_states), present_key_value


class HyV3TextModel(TextModel):
    """HYV3 text body with full attention in every dense and routed layer."""

    def __init__(self, config: ArchitectureConfig):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [HyV3DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )


def _preprocess_hy_v3_weights(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    renamed = _rename_moe_expert_weights(state_dict)
    unpacked: dict[str, torch.Tensor] = {}
    for name, tensor in renamed.items():
        projection = next(
            (
                projection
                for projection in ("gate_proj", "up_proj")
                if name.endswith(f".experts.{projection}.weight") and tensor.dim() == 3
            ),
            None,
        )
        if projection is None:
            unpacked[name] = tensor
            continue
        prefix = name[: -len(f"{projection}.weight")]
        for expert, expert_weight in enumerate(tensor):
            unpacked[f"{prefix}{expert}.{projection}.weight"] = expert_weight
    return unpacked


class HyV3CausalLMModel(CausalLMModel):
    """Tencent Hunyuan-V3 causal LM with Q/K norm and dense-prefix shared MoE."""

    category: str = "Mixture of Experts"
    config_class: type = HyV3Config

    def __init__(self, config: ArchitectureConfig):
        super().__init__(config)
        self._replace_text_model(HyV3TextModel(config))

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        return super().preprocess_weights(_preprocess_hy_v3_weights(state_dict))


class HyV3MtpModel(nn.Module):
    """One cross-conditioned Hunyuan-V3 NextN block with an independent KV cache."""

    config_class: type = HyV3MtpConfig
    default_task: str = "hy-v3-mtp"
    category: str = "Text Generation"

    def __init__(self, config: HyV3MtpConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = (
            Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
            if config.use_dedicated_embeddings
            else None
        )
        self.pre_fc_norm_embedding = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_fc_norm_hidden = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.fc = Linear(2 * config.hidden_size, config.hidden_size, bias=False)
        self.layers = nn.ModuleList(
            [
                HyV3DecoderLayer(
                    config,
                    0,
                    force_moe=config.first_k_dense_replace == 0,
                )
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = (
            Linear(config.hidden_size, config.vocab_size, bias=False)
            if config.use_dedicated_lm_head
            else None
        )
        rotary_emb = initialize_rope(config)
        if rotary_emb is None:
            raise ValueError("HYV3 MTP requires rotary position embeddings")
        self.rotary_emb = rotary_emb

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        return _preprocess_hy_v3_weights(state_dict)

    def forward(
        self,
        op: OpBuilder,
        inputs_embeds: ir.Value | None,
        input_ids: ir.Value | None,
        hidden_states: ir.Value,
        attention_mask: ir.Value,
        position_ids: ir.Value,
        past_key_values: list | None = None,
    ):
        if self.embed_tokens is not None:
            if input_ids is None:
                raise ValueError("Dedicated HYV3 MTP embeddings require input_ids")
            inputs_embeds = self.embed_tokens(op, input_ids)
        elif inputs_embeds is None:
            raise ValueError("Shared HYV3 MTP embeddings require inputs_embeds")

        # NextN cross-conditioning: concat normalized token embedding and target seed.
        embedding_state = self.pre_fc_norm_embedding(op, inputs_embeds)
        target_state = self.pre_fc_norm_hidden(op, hidden_states)
        hidden_states = self.fc(
            op, op.Concat(embedding_state, target_state, axis=-1)
        )  # (batch, sequence, hidden)

        position_embeddings = self.rotary_emb(op, position_ids)
        attention_bias = create_padding_mask(
            op,
            input_ids=inputs_embeds,
            attention_mask=attention_mask,
        )
        present_key_values = []
        past_kvs = past_key_values or [None]
        for layer, past_kv in zip(self.layers, past_kvs):
            hidden_states, present_kv = layer(
                op,
                hidden_states,
                attention_bias,
                position_embeddings,
                past_kv,
            )
            present_key_values.append(present_kv)
        hidden_states = self.norm(op, hidden_states)
        prediction = (
            self.lm_head(op, hidden_states)
            if self.lm_head is not None
            else hidden_states
        )
        return prediction, present_key_values
