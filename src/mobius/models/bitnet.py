# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""BitNet causal language model with sub-layer RMS normalization.

Replicates Hugging Face ``BitNetForCausalLM`` and llama.cpp's pinned BitNet
decoder topology. Attention is normalized after head concatenation and before
the output projection; the gated FFN is normalized before its down projection.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from onnxscript import OpBuilder, nn

from mobius._configs import ArchitectureConfig
from mobius.components import MLP, Attention, RMSNorm
from mobius.models.base import CausalLMModel, TextModel, linear_class_for_config

if TYPE_CHECKING:
    import onnx_ir as ir


class BitNetAttention(Attention):
    """BitNet attention with post-attention sub-layer RMSNorm."""

    def __init__(self, config: ArchitectureConfig, linear_class: type | None = None):
        super().__init__(config, linear_class=linear_class)
        self.attn_sub_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def _project_output(self, op: OpBuilder, attn_output: ir.Value) -> ir.Value:
        # Attention heads are concatenated as [B, S, hidden] before sub-normalization.
        return self.o_proj(op, self.attn_sub_norm(op, attn_output))


class BitNetMLP(MLP):
    """BitNet gated FFN with intermediate-width sub-layer RMSNorm."""

    def __init__(self, config: ArchitectureConfig, linear_class: type | None = None):
        super().__init__(config, linear_class=linear_class)
        self.ffn_sub_norm = RMSNorm(config.intermediate_size, eps=config.rms_norm_eps)

    def forward(self, op: OpBuilder, x: ir.Value) -> ir.Value:
        gate = self.act_fn(op, self.gate_proj(op, x))
        up = self.up_proj(op, x)
        # The sub-norm is over the gated intermediate [B, S, intermediate].
        hidden_states = self.ffn_sub_norm(op, op.Mul(gate, up))
        return self.down_proj(op, hidden_states)


class BitNetDecoderLayer(nn.Module):
    """Pre-norm BitNet decoder layer with attention and FFN sub-norms."""

    def __init__(self, config: ArchitectureConfig, linear_class: type | None = None):
        super().__init__()
        self.self_attn = BitNetAttention(config, linear_class=linear_class)
        self.mlp = BitNetMLP(config, linear_class=linear_class)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        attention_bias: ir.Value | None,
        position_embeddings: tuple | None,
        past_key_value,
    ):
        from mobius.components import StaticCacheState

        if isinstance(past_key_value, StaticCacheState):
            static_cache = past_key_value
            past_key_value = None
        else:
            static_cache = None

        residual = hidden_states
        hidden_states = self.input_layernorm(op, hidden_states)
        hidden_states, present_key_value = self.self_attn(
            op,
            hidden_states=hidden_states,
            attention_bias=attention_bias,
            position_embeddings=position_embeddings,
            past_key_value=past_key_value,
            static_cache=static_cache,
        )
        hidden_states = op.Add(residual, hidden_states)

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(op, hidden_states)
        hidden_states = self.mlp(op, hidden_states)
        return op.Add(residual, hidden_states), present_key_value


class BitNetTextModel(TextModel):
    """BitNet text backbone with architecture-specific decoder layers."""

    def __init__(self, config: ArchitectureConfig):
        super().__init__(config)
        linear_class = linear_class_for_config(config)
        self.layers = nn.ModuleList(
            [
                BitNetDecoderLayer(config, linear_class=linear_class)
                for _ in range(config.num_hidden_layers)
            ]
        )


class BitNetCausalLMModel(CausalLMModel):
    """BitNet decoder-only language model with exact attention/FFN sub-norm topology."""

    def __init__(self, config: ArchitectureConfig):
        super().__init__(config)
        self._replace_text_model(BitNetTextModel(config))
