# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Moonshot Kimi Linear causal language model."""

from __future__ import annotations

import onnx_ir as ir
import torch
from onnxscript import OpBuilder, nn

from mobius._configs import ArchitectureConfig, KimiLinearConfig
from mobius.components import (
    Embedding,
    KimiDeltaAttention,
    KimiMLAAttention,
    Linear,
    MoELayer,
    RMSNorm,
    create_attention_bias,
)
from mobius.models.base import CausalLMModel, linear_class_for_config
from mobius.models.deepseek import DeepSeekMoEGate, _SharedExpertMLP


class _KimiMoEGate(DeepSeekMoEGate):
    """Kimi routing keeps the raw sigmoid score when only one expert is selected."""

    def __init__(self, config: ArchitectureConfig):
        super().__init__(config)
        self.norm_topk_prob = config.norm_topk_prob and self.top_k > 1


class _KimiExpertMLP(nn.Module):
    """Kimi SwiGLU expert with graph-standard projection names."""

    def __init__(self, config: ArchitectureConfig, linear_class: type | None = None):
        super().__init__()
        linear_class = linear_class or Linear
        self.gate_proj = linear_class(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = linear_class(config.intermediate_size, config.hidden_size, bias=False)
        self.up_proj = linear_class(config.hidden_size, config.intermediate_size, bias=False)

    def forward(self, op: OpBuilder, hidden_states: ir.Value):
        return self.down_proj(
            op,
            op.Mul(
                op.Swish(self.gate_proj(op, hidden_states)),
                self.up_proj(op, hidden_states),
            ),
        )


class _KimiSparseMoeBlock(nn.Module):
    def __init__(self, config: ArchitectureConfig, linear_class: type | None):
        super().__init__()
        gate = _KimiMoEGate(config)
        self.moe = MoELayer(
            config,
            gate=gate,
            linear_class=linear_class,
            expert_factory=lambda cfg, lc: _KimiExpertMLP(cfg, lc),
        )
        shared_size = config.moe_intermediate_size * config.n_shared_experts
        self.shared_experts = _SharedExpertMLP(config, shared_size, linear_class=linear_class)

    def forward(self, op: OpBuilder, hidden_states: ir.Value):
        return op.Add(self.moe(op, hidden_states), self.shared_experts(op, hidden_states))


class KimiLinearDecoderLayer(nn.Module):
    def __init__(self, config: ArchitectureConfig, layer_idx: int):
        super().__init__()
        linear_class = linear_class_for_config(config)
        self._is_kda = config.layer_types[layer_idx] == "kimi_linear_attention"
        self.self_attn = (
            KimiDeltaAttention(config, linear_class)
            if self._is_kda
            else KimiMLAAttention(config, linear_class)
        )
        if layer_idx >= config.first_k_dense_replace:
            self.block_sparse_moe = _KimiSparseMoeBlock(config, linear_class)
            self.mlp = None
        else:
            self.block_sparse_moe = None
            self.mlp = _KimiExpertMLP(config, linear_class)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        attention_bias: ir.Value,
        attention_mask: ir.Value,
        past_key_value: tuple[ir.Value, ...],
    ):
        residual = hidden_states
        normed = self.input_layernorm(op, hidden_states)
        if self._is_kda:
            attention_output, present = self.self_attn(
                op, normed, attention_mask, *past_key_value
            )
        else:
            attention_output, present = self.self_attn(
                op, normed, attention_bias, past_key_value
            )
        hidden_states = op.Add(residual, attention_output)

        residual = hidden_states
        normed = self.post_attention_layernorm(op, hidden_states)
        if self.block_sparse_moe is not None:
            feed_forward = self.block_sparse_moe(op, normed)
        else:
            feed_forward = self.mlp(op, normed)
        return op.Add(residual, feed_forward), present


class KimiLinearTextModel(nn.Module):
    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.embed_tokens = Embedding(
            config.vocab_size, config.hidden_size, config.pad_token_id
        )
        self.layers = nn.ModuleList(
            [
                KimiLinearDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self._dtype = config.dtype

    def forward(
        self,
        op: OpBuilder,
        input_ids: ir.Value,
        attention_mask: ir.Value,
        position_ids: ir.Value,
        past_key_values: list[tuple[ir.Value, ...]],
    ):
        del position_ids
        hidden_states = self.embed_tokens(op, input_ids)
        attention_bias = create_attention_bias(
            op, input_ids, attention_mask, dtype=self._dtype
        )
        present = []
        for layer, past in zip(self.layers, past_key_values):
            hidden_states, layer_present = layer(
                op, hidden_states, attention_bias, attention_mask, past
            )
            present.append(layer_present)
        return self.norm(op, hidden_states), present


class KimiLinearCausalLMModel(CausalLMModel):
    """Dedicated Kimi Linear KDA/NoPE-MLA/MoE decoder."""

    default_task = "kimi-linear-text-generation"
    config_class = KimiLinearConfig

    def __init__(self, config: ArchitectureConfig):
        super().__init__(config)
        self._replace_text_model(KimiLinearTextModel(config))

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        renamed: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            if key.endswith(".self_attn.kv_b_proj.weight"):
                prefix = key.removesuffix("kv_b_proj.weight")
                reshaped = value.reshape(
                    self.config.num_attention_heads,
                    self.config.qk_nope_head_dim + self.config.v_head_dim,
                    self.config.kv_lora_rank,
                )
                key_weight, value_weight = torch.split(
                    reshaped,
                    [self.config.qk_nope_head_dim, self.config.v_head_dim],
                    dim=1,
                )
                renamed[prefix + "k_b_proj.weight"] = key_weight.reshape(
                    -1, self.config.kv_lora_rank
                )
                renamed[prefix + "v_b_proj.weight"] = value_weight.reshape(
                    -1, self.config.kv_lora_rank
                )
                continue
            new_key = key
            new_key = new_key.replace(".block_sparse_moe.gate.", ".block_sparse_moe.moe.gate.")
            new_key = new_key.replace(
                ".block_sparse_moe.experts.",
                ".block_sparse_moe.moe.experts.",
            )
            new_key = new_key.replace(".w1.", ".gate_proj.")
            new_key = new_key.replace(".w2.", ".down_proj.")
            new_key = new_key.replace(".w3.", ".up_proj.")
            renamed[new_key] = value
        return super().preprocess_weights(renamed)
