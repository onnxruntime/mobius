# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Hy3-preview (hy_v3) model — dense MoE causal LM.

HuggingFace model: tencent/Hy3-preview
Architecture: HYV3ForCausalLM

Architecture overview:
- Standard GQA decoder with qk_norm
- First ``first_k_dense_replace`` layers use a dense MLP
- Remaining layers use MoE (sigmoid-gated, with shared expert)
- 192 experts, 8 active per token, 1 shared expert
- Sigmoid routing with expert correction bias and scaling factor

Very similar to DeepSeek-V3 but uses standard GQA attention (not MLA).
Reuses DeepSeek V3's MoE gate (sigmoid mode), shared expert, and fused
expert weight splitting logic.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from onnxscript import nn

from mobius._configs import ArchitectureConfig
from mobius.components import (
    Attention,
    Embedding,
    Linear,
    RMSNorm,
    create_attention_bias,
    initialize_rope,
)
from mobius.components._moe import MLP
from mobius.models.base import CausalLMModel
from mobius.models.deepseek import DeepSeekMoEGate, _DeepSeekMoEFFN

if TYPE_CHECKING:
    import onnx_ir as ir
    from onnxscript import OpBuilder


class _Hy3DecoderLayer(nn.Module):
    """Decoder layer for Hy3 with optional MoE.

    Standard pre-norm decoder with GQA attention + qk_norm.
    Dense layers use standard MLP; MoE layers use sigmoid-gated
    routing with shared expert.
    """

    def __init__(self, config: ArchitectureConfig, is_moe: bool = False):
        super().__init__()
        self.self_attn = Attention(config)
        if is_moe:
            gate = DeepSeekMoEGate(config)
            self.mlp = _DeepSeekMoEFFN(config, gate)
        else:
            self.mlp = MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        attention_bias: ir.Value | None,
        position_embeddings: tuple,
        past_key_value: tuple | None = None,
    ):
        # Self attention with pre-norm
        residual = hidden_states
        hidden_states = self.input_layernorm(op, hidden_states)
        hidden_states, present_kv = self.self_attn(
            op,
            hidden_states=hidden_states,
            attention_bias=attention_bias,
            position_embeddings=position_embeddings,
            past_key_value=past_key_value,
        )
        hidden_states = op.Add(residual, hidden_states)

        # FFN with pre-norm
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(op, hidden_states)
        hidden_states = self.mlp(op, hidden_states)
        hidden_states = op.Add(residual, hidden_states)

        return hidden_states, present_kv


class Hy3TextModel(nn.Module):
    """Text model for Hy3-preview.

    First ``first_k_dense_replace`` layers use standard MLP;
    remaining layers use MoE with sigmoid routing and shared expert.
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self._dtype = config.dtype
        self.embed_tokens = Embedding(
            config.vocab_size, config.hidden_size, config.pad_token_id
        )

        first_k = config.first_k_dense_replace
        if not config.num_local_experts:
            first_k = config.num_hidden_layers
        self.layers = nn.ModuleList(
            [
                _Hy3DecoderLayer(config, is_moe=(i >= first_k))
                for i in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = initialize_rope(config)

    def forward(
        self,
        op: OpBuilder,
        input_ids: ir.Value,
        attention_mask: ir.Value,
        position_ids: ir.Value,
        past_key_values: list | None = None,
    ):
        hidden_states = self.embed_tokens(op, input_ids)
        position_embeddings = self.rotary_emb(op, position_ids)
        attention_bias = create_attention_bias(
            op,
            input_ids=input_ids,
            attention_mask=attention_mask,
            dtype=self._dtype,
        )

        present_key_values = []
        past_kvs = past_key_values or [None] * len(self.layers)
        for layer, past_kv in zip(self.layers, past_kvs):
            hidden_states, present_kv = layer(
                op,
                hidden_states=hidden_states,
                attention_bias=attention_bias,
                position_embeddings=position_embeddings,
                past_key_value=past_kv,
            )
            present_key_values.append(present_kv)

        hidden_states = self.norm(op, hidden_states)
        return hidden_states, present_key_values


class Hy3CausalLMModel(CausalLMModel):
    """Causal LM for Hy3-preview (tencent/Hy3-preview).

    Dense MoE architecture with 192 experts (8 active), 1 shared expert,
    sigmoid routing with expert correction bias, and GQA with qk_norm.
    """

    default_task: str = "text-generation"
    category: str = "Mixture of Experts"

    def __init__(self, config: ArchitectureConfig):
        nn.Module.__init__(self)
        self.config = config
        self.model = Hy3TextModel(config)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Remap HuggingFace weight names to ONNX parameter names.

        Key mappings:
        - MoE gate: mlp.gate.weight → mlp.moe.gate.weight
        - Expert bias: mlp.e_score_correction_bias → mlp.moe.gate.e_score_correction_bias
        - Shared expert: mlp.shared_experts.* already aligns
        - Fused expert weights: HF stores all experts as 3D tensors
            experts.gate_up_proj: (n_experts, 2*intermediate, hidden)
            experts.down_proj:    (n_experts, hidden, intermediate)
          These are split into per-expert ONNX weights:
            moe.experts.{i}.gate_proj.weight: (intermediate, hidden)
            moe.experts.{i}.up_proj.weight:   (intermediate, hidden)
            moe.experts.{i}.down_proj.weight: (hidden, intermediate)
        """
        renamed = {}
        for key, value in state_dict.items():
            new_key = key

            # Remap MoE gate: mlp.gate.* → mlp.moe.gate.*
            new_key = new_key.replace(".mlp.gate.", ".mlp.moe.gate.")

            # Remap expert bias buffer: mlp.e_score_correction_bias
            # → mlp.moe.gate.e_score_correction_bias
            new_key = new_key.replace(
                ".mlp.e_score_correction_bias",
                ".mlp.moe.gate.e_score_correction_bias",
            )

            # Split fused expert gate_up_proj into per-expert gate_proj + up_proj
            if new_key.endswith(".mlp.experts.gate_up_proj"):
                prefix = new_key[: -len(".mlp.experts.gate_up_proj")]
                mid = value.shape[1] // 2
                for i in range(value.shape[0]):
                    renamed[f"{prefix}.mlp.moe.experts.{i}.gate_proj.weight"] = value[i, :mid]
                    renamed[f"{prefix}.mlp.moe.experts.{i}.up_proj.weight"] = value[i, mid:]
                continue

            # Split fused expert down_proj into per-expert down_proj
            if new_key.endswith(".mlp.experts.down_proj"):
                prefix = new_key[: -len(".mlp.experts.down_proj")]
                for i in range(value.shape[0]):
                    renamed[f"{prefix}.mlp.moe.experts.{i}.down_proj.weight"] = value[i]
                continue

            renamed[new_key] = value

        return super().preprocess_weights(renamed)
