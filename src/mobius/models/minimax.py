# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""MiniMax causal language model with Lightning Attention.

MiniMax is a hybrid model alternating between:
- Full GQA attention layers (standard softmax attention)
- Lightning Attention layers (pure retention linear attention, no KV cache)

All layers use a Sparse MoE FFN.

The residual connection uses an unusual pre-norm structure:
  layernorm is applied FIRST, then residual is taken from the normalized
  value (not the original input), with optional alpha/beta scaling.

HuggingFace reference: MiniMaxModel, MiniMaxDecoderLayer
"""

from __future__ import annotations

import math

import onnx_ir as ir
import torch
from onnxscript import OpBuilder, nn

from mobius._configs import ArchitectureConfig, MiniMaxConfig
from mobius.components import (
    Attention,
    MoELayer,
    RMSNorm,
    create_attention_bias,
    initialize_rope,
)
from mobius.components._lightning_attention import LightningAttention
from mobius.models.base import CausalLMModel, embedding_for_config
from mobius.models.moe import _quantized_linear_class, _rename_moe_expert_weights


class MiniMaxTopKGate(nn.Module):
    """MiniMax router: FP32 softmax, top-k selection, then selected-weight normalization."""

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        assert config.num_local_experts is not None
        assert config.num_experts_per_tok is not None
        self.top_k = config.num_experts_per_tok
        self.weight = nn.Parameter([config.num_local_experts, config.hidden_size])

    def forward(self, op: OpBuilder, hidden_states: ir.Value):
        logits = op.MatMul(hidden_states, op.Transpose(self.weight, perm=[1, 0]))
        probs = op.Softmax(op.Cast(logits, to=ir.DataType.FLOAT), axis=-1)
        weights, experts = op.TopK(
            probs,
            op.Constant(value_ints=[self.top_k]),
            axis=-1,
            _outputs=2,
        )
        weights = op.Div(weights, op.ReduceSum(weights, [-1], keepdims=True))
        return op.CastLike(weights, hidden_states), experts


class MiniMaxDecoderLayer(nn.Module):
    """MiniMax decoder layer with hybrid Lightning / full attention.

    Each layer is either ``"full_attention"`` (standard GQA) or
    ``"lightning_attention"`` (LightningAttention), controlled by
    ``config.layer_types[layer_idx]``.

    MiniMax uses an unusual residual structure: the layernorm is applied
    BEFORE the residual branch point, and alpha/beta scaling factors multiply
    the residual and sub-layer outputs. With default alpha=beta=1 this gives:
        x_out = layernorm(x_in) + sub_layer(layernorm(x_in))

    All layers use a Sparse MoE FFN block.
    """

    def __init__(self, config: ArchitectureConfig, layer_idx: int):
        super().__init__()
        assert config.layer_types is not None
        self.layer_type = config.layer_types[layer_idx]

        linear_class = _quantized_linear_class(config)
        if self.layer_type == "lightning_attention":
            self.self_attn = LightningAttention(config, layer_idx, linear_class=linear_class)
            self._attn_alpha: float = getattr(config, "linear_attn_alpha_factor", 1.0)
            self._attn_beta: float = getattr(config, "linear_attn_beta_factor", 1.0)
        else:
            self.self_attn = Attention(config, linear_class=linear_class)
            self._attn_alpha = getattr(config, "full_attn_alpha_factor", 1.0)
            self._attn_beta = getattr(config, "full_attn_beta_factor", 1.0)

        self._mlp_alpha: float = getattr(config, "mlp_alpha_factor", 1.0)
        self._mlp_beta: float = getattr(config, "mlp_beta_factor", 1.0)

        self.mlp = MoELayer(
            config,
            gate=MiniMaxTopKGate(config),
            linear_class=linear_class,
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        attention_bias: ir.Value | None,
        position_embeddings: tuple[ir.Value, ir.Value],
        past_key_value,
        attention_mask: ir.Value | None = None,
    ):
        # MiniMax pre-norm: apply layernorm first, then take residual from
        # the normalized value (not from the original hidden_states).
        hidden_states = self.input_layernorm(op, hidden_states)
        residual = hidden_states  # residual taken AFTER norm

        if self.layer_type == "lightning_attention":
            # Lightning Attention: single recurrent state (no conv_state)
            (recurrent_state,) = past_key_value
            attn_out, new_state = self.self_attn(
                op, hidden_states, recurrent_state, attention_mask
            )
            present_key_value = (new_state,)
        else:
            attn_out, present_key_value = self.self_attn(
                op,
                hidden_states=hidden_states,
                attention_bias=attention_bias,
                position_embeddings=position_embeddings,
                past_key_value=past_key_value,
            )

        # Scaled residual: alpha * residual + beta * attn_out
        hidden_states = _scaled_add(op, residual, attn_out, self._attn_alpha, self._attn_beta)

        # MLP sub-layer with MiniMax post-attn pre-norm pattern
        hidden_states = self.post_attention_layernorm(op, hidden_states)
        residual = hidden_states
        hidden_states = self.mlp(op, hidden_states)
        hidden_states = _scaled_add(
            op, residual, hidden_states, self._mlp_alpha, self._mlp_beta
        )

        return hidden_states, present_key_value


class MiniMaxTextModel(nn.Module):
    """MiniMax text backbone with hybrid attention layers."""

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        if config.layer_types is None or len(config.layer_types) != config.num_hidden_layers:
            raise ValueError(
                "MiniMax layer_types must contain exactly num_hidden_layers entries"
            )
        unknown = set(config.layer_types) - {"full_attention", "lightning_attention"}
        if unknown:
            raise ValueError(f"Unsupported MiniMax layer type(s): {sorted(unknown)}")
        if config.head_dim <= 0:
            raise ValueError("MiniMax head_dim must be explicit and positive")
        self._dtype = config.dtype
        self.embed_tokens = embedding_for_config(config)
        self.layers = nn.ModuleList(
            [MiniMaxDecoderLayer(config, layer_idx=i) for i in range(config.num_hidden_layers)]
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

        present_key_values: list = []
        past_kvs = past_key_values or [None] * len(self.layers)
        for layer, past_kv in zip(self.layers, past_kvs):
            hidden_states, present_kv = layer(
                op,
                hidden_states=hidden_states,
                attention_bias=attention_bias,
                position_embeddings=position_embeddings,
                past_key_value=past_kv,
                attention_mask=attention_mask,
            )
            present_key_values.append(present_kv)

        hidden_states = self.norm(op, hidden_states)
        return hidden_states, present_key_values


class MiniMaxCausalLMModel(CausalLMModel):
    """MiniMax causal language model with hybrid Lightning + GQA + MoE.

    Architecture:
    - ``config.layer_types`` preserves the checkpoint's explicit per-layer schedule
    - Full-attention layers use partial-RoPE GQA
    - Lightning layers use a fixed-size recurrent state and no KV cache

    Lightning Attention layers carry a single recurrent_state tensor of
    shape (B, num_heads, head_dim, head_dim) per layer. Full attention layers
    use the standard KV cache.

    Task: ``hybrid-text-generation`` (HybridCausalLMTask).

    HuggingFace model types: ``"MiniMaxText01"`` and ``"minimax"``
    """

    default_task: str = "hybrid-text-generation"
    config_class = MiniMaxConfig

    def __init__(self, config: ArchitectureConfig):
        super().__init__(config)
        self._replace_text_model(MiniMaxTextModel(config))

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        # Split fused MoE expert weights (gate_up_proj/down_proj → per-expert).
        renamed: dict[str, torch.Tensor] = {}
        for name, tensor in state_dict.items():
            name = name.replace(".self_attn.out_proj.", ".self_attn.o_proj.")
            if ".experts." in name and tensor.dim() == 3:
                prefix, projection = name.rsplit(".experts.", 1)
                if projection in {
                    "gate_proj.weight",
                    "up_proj.weight",
                    "down_proj.weight",
                }:
                    for expert, expert_tensor in enumerate(tensor):
                        renamed[f"{prefix}.experts.{expert}.{projection}"] = expert_tensor
                    continue
            renamed[name] = tensor
        state_dict = _rename_moe_expert_weights(renamed)
        return super().preprocess_weights(state_dict)


def _scaled_add(
    op: OpBuilder,
    residual: ir.Value,
    sub_layer_out: ir.Value,
    alpha: float,
    beta: float,
) -> ir.Value:
    """Compute alpha * residual + beta * sub_layer_out.

    When both alpha and beta are 1.0 (the default), emits a simple Add.
    Otherwise uses Mul + Add with scalar constants.
    """
    if math.isclose(alpha, 1.0) and math.isclose(beta, 1.0):
        return op.Add(residual, sub_layer_out)
    scaled_res = op.Mul(
        op.CastLike(op.Constant(value_float=alpha), residual),
        residual,
    )
    scaled_out = op.Mul(
        op.CastLike(op.Constant(value_float=beta), sub_layer_out),
        sub_layer_out,
    )
    return op.Add(scaled_res, scaled_out)
