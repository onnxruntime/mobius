"""BitNet b1.58 — ternary-weight language model.

BitNet uses {-1, 0, +1} weights (1.58 bits) with per-tensor ``weight_scale``.
The architecture is Llama-like with two key differences:

1. **Sub-layer norms**: ``attn_sub_norm`` applied to attention output before
   ``o_proj``, and ``ffn_sub_norm`` applied to gated activation before
   ``down_proj``.
2. **Activation**: squared ReLU (``relu2``) instead of SiLU.

HuggingFace stores weights as uint8-packed 2-bit ternary values.  This module
unpacks them to float in ``preprocess_weights`` so the ONNX graph uses
standard ``Linear`` ops.  A future optimisation could use ``MatMulNBits``
with ``bits=2`` or a custom ternary kernel.

Replicates HuggingFace's ``BitNetForCausalLM``.
"""

from __future__ import annotations

import onnx_ir as ir
import torch
from onnxscript._internal import builder

from mobius._configs import ArchitectureConfig
from mobius.components._attention import (
    Attention,
    StaticCacheState,
    _apply_attention,
    apply_rotary_pos_emb,
)
from mobius.components._mlp import MLP
from mobius.components._rms_norm import RMSNorm
from mobius.models.base import CausalLMModel


class BitNetAttention(Attention):
    """Attention with a sub-layer RMSNorm before the output projection.

    HuggingFace name: ``BitNetAttention`` — identical to standard
    multi-head attention except ``attn_sub_norm`` is applied to the
    concatenated attention heads *before* ``o_proj``.
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__(config)
        # Sub-layer norm applied to attention output before o_proj
        self.attn_sub_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        attention_bias: ir.Value | None,
        position_embeddings: tuple | None = None,
        past_key_value: tuple | None = None,
        static_cache: StaticCacheState | None = None,
    ):
        # Q/K/V projections — (batch, seq_len, num_heads * head_dim)
        query_states = self.q_proj(op, hidden_states)
        key_states = self.k_proj(op, hidden_states)
        value_states = self.v_proj(op, hidden_states)

        # Optional Q/K normalization (inherited from base Attention)
        if self.q_norm is not None and self.k_norm is not None:
            if self._qk_norm_full:
                query_states = self.q_norm(op, query_states)
                key_states = self.k_norm(op, key_states)
            else:
                query_states = op.Reshape(query_states, [0, 0, -1, self.head_dim])
                key_states = op.Reshape(key_states, [0, 0, -1, self.head_dim])
                query_states = self.q_norm(op, query_states)
                key_states = self.k_norm(op, key_states)
                query_states = op.Reshape(query_states, [0, 0, -1])
                key_states = op.Reshape(key_states, [0, 0, -1])

        # RoPE
        if position_embeddings is not None:
            query_states = apply_rotary_pos_emb(
                op,
                query_states,
                position_embeddings=position_embeddings,
                num_heads=self.num_attention_heads,
                rotary_embedding_dim=self.rotary_embedding_dim,
                interleaved=self._rope_interleave,
            )
            key_states = apply_rotary_pos_emb(
                op,
                key_states,
                position_embeddings=position_embeddings,
                num_heads=self.num_key_value_heads,
                rotary_embedding_dim=self.rotary_embedding_dim,
                interleaved=self._rope_interleave,
            )

        # Grouped-query attention — (batch, seq_len, num_heads * head_dim)
        attn_output, present_key, present_value = _apply_attention(
            op,
            query_states,
            key_states,
            value_states,
            attention_bias,
            past_key_value[0] if past_key_value is not None else None,
            past_key_value[1] if past_key_value is not None else None,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            scale=self.scaling,
            static_cache=static_cache,
        )

        # BitNet-specific: sub-layer norm before o_proj
        attn_output = self.attn_sub_norm(op, attn_output)
        attn_output = self.o_proj(op, attn_output)
        return attn_output, (present_key, present_value)


class BitNetMLP(MLP):
    """Gated MLP with a sub-layer RMSNorm before the down projection.

    HuggingFace name: ``BitNetMLP`` — same as standard gated MLP except
    ``ffn_sub_norm`` (over ``intermediate_size``) is applied between the
    gated activation and ``down_proj``.
    """

    def __init__(
        self,
        config: ArchitectureConfig,
        linear_class: type | None = None,
    ):
        super().__init__(config, linear_class=linear_class)
        # Sub-layer norm applied after gated activation, before down_proj
        self.ffn_sub_norm = RMSNorm(config.intermediate_size, eps=config.rms_norm_eps)

    def forward(self, op: builder.OpBuilder, x: ir.Value):
        # gate_proj → relu² → element-wise multiply with up_proj
        gate = self.act_fn(op, self.gate_proj(op, x))
        up = self.up_proj(op, x)
        hidden = op.Mul(gate, up)
        # BitNet-specific: sub-layer norm before down_proj
        hidden = self.ffn_sub_norm(op, hidden)
        return self.down_proj(op, hidden)


class BitNetCausalLMModel(CausalLMModel):
    """BitNet b1.58 causal language model.

    Builds on the standard CausalLMModel (Llama-like) and replaces each
    decoder layer's Attention and MLP with BitNet variants that include
    sub-layer RMSNorms.  Weight preprocessing unpacks HuggingFace's
    uint8-packed ternary weights to float32.

    Replicates HuggingFace's ``BitNetForCausalLM``.
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__(config)
        # Replace standard Attention/MLP with BitNet variants
        for layer in self.model.layers:
            layer.self_attn = BitNetAttention(config)
            layer.mlp = BitNetMLP(config)

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Unpack ternary weights and apply per-tensor weight_scale.

        HuggingFace packs ternary {-1, 0, +1} values as 2-bit unsigned
        integers, 4 values per uint8 byte.  This method:
        1. Collects ``*.weight_scale`` tensors (per-tensor scaling factors).
        2. For each packed ``*.weight`` with a matching scale, unpacks the
           uint8 → int8 ternary values and multiplies by the scale.
        3. Drops the consumed ``weight_scale`` keys from the state dict.
        """
        # Collect per-tensor weight scales
        weight_scales: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            if key.endswith(".weight_scale"):
                prefix = key[: -len(".weight_scale")]
                weight_scales[prefix] = value

        new_state_dict: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            if key.endswith(".weight_scale"):
                continue  # consumed by the matching .weight key
            if key.endswith(".weight") and key[: -len(".weight")] in weight_scales:
                scale = weight_scales[key[: -len(".weight")]]
                if value.dtype == torch.uint8:
                    value = _unpack_ternary_weights(value)
                value = value.float() * scale.float()
            new_state_dict[key] = value

        # Delegate remaining processing (tie_word_embeddings, etc.)
        return super().preprocess_weights(new_state_dict)


def _unpack_ternary_weights(packed: torch.Tensor) -> torch.Tensor:
    """Unpack uint8-packed ternary weights to float tensor.

    Packing format (HuggingFace BitNet):
    * Ternary values {-1, 0, +1} are mapped to unsigned {0, 1, 2} (add 1).
    * Four 2-bit values are packed per byte: ``v0 | (v1<<2) | (v2<<4) | (v3<<6)``.
    * Packed shape: ``[out_features // 4, in_features]``.
    * Unpacked shape: ``[out_features, in_features]``.

    Args:
        packed: uint8 tensor of shape ``[out_features // 4, in_features]``.

    Returns:
        Float tensor of shape ``[out_features, in_features]`` with values
        in {-1, 0, +1}.
    """
    if packed.ndim < 2:
        raise ValueError(f"Expected packed tensor with ≥2 dims, got shape {packed.shape}")
    # Extract 4 two-bit values per byte
    v0 = (packed >> 0) & 0x03  # bits 0-1
    v1 = (packed >> 2) & 0x03  # bits 2-3
    v2 = (packed >> 4) & 0x03  # bits 4-5
    v3 = (packed >> 6) & 0x03  # bits 6-7

    # Interleave: stack along new dim then reshape to expand out_features
    # packed shape: [out_features // 4, in_features]
    # stacked shape: [out_features // 4, 4, in_features]
    # result shape: [out_features, in_features]
    stacked = torch.stack([v0, v1, v2, v3], dim=-2)
    unpacked = stacked.reshape(-1, packed.shape[-1])

    # Map unsigned {0, 1, 2} back to signed {-1, 0, +1}
    return unpacked.float() - 1.0
