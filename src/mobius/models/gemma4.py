# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Gemma 4 model implementations.

Architecture variants:
- **Gemma4CausalLMModel**: Text-only causal LM (model_type ``gemma4_text``).
- **Gemma4Model**: Multimodal model — supports Image-Text-to-Text (26B-A4B, 31B)
  and Any-to-Any audio+vision+text (E2B, E4B) via ``Gemma4Task``.

Key architectural differences from Gemma3:
- Standard ``RMSNorm`` throughout (no ``OffsetRMSNorm``).
- Dual head_dim: local sliding-window layers use ``config.head_dim``; global
  full-attention layers use ``config.global_head_dim``.
- Dual RoPE: different ``rope_theta`` and ``partial_rotary_factor`` per layer type.
- Per-layer input gating (disabled when ``hidden_size_per_layer_input == 0``).
- Vision encoder: pre-patchified input ``[B, N, 3*P^2]`` with 2D position lookup,
  bidirectional attention, 4-norm structure, and scale-then-project pooling.
- Vision projector: scale-free RMSNorm -> Linear (matches ``embed_vision`` weights).
- KV sharing: last ``num_kv_shared_layers`` layers borrow K,V from earlier layers
  and have no k_proj/v_proj weights of their own.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import onnx_ir as ir
import torch
from onnxscript import nn
from onnxscript._internal import builder

from mobius._build_context import ep_capabilities
from mobius._configs import ArchitectureConfig, Gemma4Config
from mobius._weight_utils import vlm_decoder_weights, vlm_embedding_weights
from mobius.components import (
    MLP,
    Linear,
    RMSNorm,
    create_attention_bias,
    initialize_rope,
)
from mobius.components._activations import get_activation
from mobius.components._gemma4_audio import Gemma4AudioEncoder
from mobius.components._mlp import GatedMLP
from mobius.models.base import CausalLMModel
from mobius.models.gemma3_text import Gemma3TextScaledWordEmbedding

# ---------------------------------------------------------------------------
# Scale-free RMSNorm (Gemma4RMSNorm with with_scale=False)
# ---------------------------------------------------------------------------


class _Gemma4ScaleFreeRMSNorm(nn.Module):
    """RMSNorm with a constant all-ones scale (no learnable parameter).

    Matches ``Gemma4RMSNorm(with_scale=False)`` in HuggingFace.
    Used for V norms in the vision encoder and the projector pre-norm.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, op: builder.OpBuilder, hidden_states: ir.Value) -> ir.Value:
        # All-ones scale tensor — no learnable parameter to load.
        # Use value_floats (static constant) instead of ConstantOfShape so that
        # ONNX shape inference can resolve the output shape of RMSNormalization.
        # CastLike ensures the scale matches the input dtype (fp16/bf16/fp32).
        scale = op.CastLike(op.Constant(value_floats=[1.0] * self.dim), hidden_states)
        return op.RMSNormalization(hidden_states, scale, epsilon=self.eps, axis=-1)


# ---------------------------------------------------------------------------
# Gemma4 vision encoder
# ---------------------------------------------------------------------------


class Gemma4VisionSelfAttention(nn.Module):
    """Bidirectional multi-head self-attention for the Gemma4 vision encoder.

    Differences from standard text Attention:
    - No causal mask (bidirectional attention via ``op.Attention`` without ``is_causal``).
    - Per-head QK norms (``RMSNorm`` with learned scale).
    - Per-head V norm (scale-free RMSNorm, no learned parameter).
    - Scale = 1.0 matching HF ``Gemma4VisionAttention``.

    Weight names align with HF after stripping the ``.linear.`` infix from
    ``Gemma4ClippableLinear`` module attributes.
    """

    def __init__(self, hidden_size: int, num_heads: int, norm_eps: float = 1e-6):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.q_proj = Linear(hidden_size, num_heads * self.head_dim, bias=False)
        self.k_proj = Linear(hidden_size, num_heads * self.head_dim, bias=False)
        self.v_proj = Linear(hidden_size, num_heads * self.head_dim, bias=False)
        self.o_proj = Linear(num_heads * self.head_dim, hidden_size, bias=False)
        self.q_norm = RMSNorm(self.head_dim, eps=norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=norm_eps)
        self.v_norm = _Gemma4ScaleFreeRMSNorm(self.head_dim, eps=norm_eps)

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        attention_bias: ir.Value | None = None,
    ) -> ir.Value:
        # [B, N, hidden] -> project and reshape for per-head norms
        q = self.q_proj(op, hidden_states)
        k = self.k_proj(op, hidden_states)
        v = self.v_proj(op, hidden_states)

        # Reshape to [B, N, num_heads, head_dim] for per-head norms
        q = op.Reshape(q, [0, 0, -1, self.head_dim])
        k = op.Reshape(k, [0, 0, -1, self.head_dim])
        v = op.Reshape(v, [0, 0, -1, self.head_dim])
        q = self.q_norm(op, q)
        k = self.k_norm(op, k)
        v = self.v_norm(op, v)
        # Flatten back to [B, N, num_heads * head_dim]
        q = op.Reshape(q, [0, 0, -1])
        k = op.Reshape(k, [0, 0, -1])
        v = op.Reshape(v, [0, 0, -1])

        # Bidirectional attention (no is_causal, no KV cache).
        # attention_bias [B, 1, 1, N] masks out padding patches (value = -1e9).
        attn_output = op.Attention(
            q,
            k,
            v,
            attention_bias,
            q_num_heads=self.num_heads,
            kv_num_heads=self.num_heads,
            scale=1.0,
            _outputs=1,
        )
        return self.o_proj(op, attn_output)


class Gemma4VisionEncoderLayer(nn.Module):
    """Gemma4 vision transformer encoder layer.

    4-norm structure matching HF ``Gemma4VisionEncoderLayer``:
    pre-attn norm -> attention -> post-attn norm -> residual ->
    pre-MLP norm -> gated MLP -> post-MLP norm -> residual.

    Uses standard ``RMSNorm`` throughout (not ``OffsetRMSNorm``).
    """

    def __init__(
        self, hidden_size: int, intermediate_size: int, num_heads: int, norm_eps: float
    ):
        super().__init__()
        self.self_attn = Gemma4VisionSelfAttention(hidden_size, num_heads, norm_eps)
        self.input_layernorm = RMSNorm(hidden_size, eps=norm_eps)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=norm_eps)
        self.pre_feedforward_layernorm = RMSNorm(hidden_size, eps=norm_eps)
        self.post_feedforward_layernorm = RMSNorm(hidden_size, eps=norm_eps)
        # Gated MLP with SiLU activation (gate_proj * up_proj -> down_proj)
        self.mlp = MLP(
            ArchitectureConfig(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                hidden_act="silu",
                rms_norm_eps=norm_eps,
            )
        )

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        attention_bias: ir.Value | None = None,
    ) -> ir.Value:
        residual = hidden_states
        hidden_states = self.input_layernorm(op, hidden_states)
        hidden_states = self.self_attn(op, hidden_states, attention_bias)
        hidden_states = self.post_attention_layernorm(op, hidden_states)
        hidden_states = op.Add(residual, hidden_states)

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(op, hidden_states)
        hidden_states = self.mlp(op, hidden_states)
        hidden_states = self.post_feedforward_layernorm(op, hidden_states)
        hidden_states = op.Add(residual, hidden_states)
        return hidden_states


class _Gemma4VisionPatchEmbedder(nn.Module):
    """Gemma4 patch embedder: linear projection + 2D position lookup.

    Inputs:
    - ``pixel_values [B, N, 3*patch_size^2]``: pre-patchified, normalized to ``[-1, 1]``
    - ``pixel_position_ids [B, N, 2]``: (x, y) coordinates for each patch

    Output: ``[B, N, hidden_size]``

    Weight names match HF ``Gemma4VisionPatchEmbedder``:
    - ``input_proj.weight``
    - ``position_embedding_table`` (Parameter ``[2, pos_emb_size, hidden_size]``)
    """

    def __init__(self, patch_size: int, hidden_size: int, position_embedding_size: int):
        super().__init__()
        self.input_proj = Linear(3 * patch_size * patch_size, hidden_size, bias=False)
        # Position embedding table: [2, pos_emb_size, hidden] — x and y tables
        self.position_embedding_table = nn.Parameter([2, position_embedding_size, hidden_size])
        self.position_embedding_size = position_embedding_size

    def forward(
        self,
        op: builder.OpBuilder,
        pixel_values: ir.Value,
        pixel_position_ids: ir.Value,
    ) -> tuple[ir.Value, ir.Value]:
        """Return ``(hidden_states [B, N, hidden], is_padding [B, N bool])``.

        Padding patches are indicated by ``pixel_position_ids == -1``.  Their
        position embeddings are zeroed exactly as HF does in
        ``Gemma4VisionPatchEmbedder._position_embeddings``.
        """
        # pixel_values in [0,1] -> normalize to [-1, 1]: 2*(v - 0.5) = 2v - 1
        pixel_values = op.Sub(
            op.Mul(pixel_values, op.Constant(value_float=2.0)),
            op.Constant(value_float=1.0),
        )
        hidden_states = self.input_proj(op, pixel_values)  # [B, N, hidden]

        # Detect padding patches: x-coord == -1 means the patch is padding.
        # pixel_position_ids [B, N, 2]; gather x-coord [B, N].
        x_raw = op.Gather(pixel_position_ids, op.Constant(value_int=0), axis=2)  # [B, N]
        is_padding = op.Equal(x_raw, op.Constant(value_int=-1))  # [B, N] bool

        # Clamp to ≥0 before embedding lookup (padding → 0 temporarily)
        clamped = op.Clip(pixel_position_ids, op.Constant(value_int=0))

        # Extract x and y coordinates: each [B, N]
        x_coords = op.Gather(clamped, op.Constant(value_int=0), axis=2)  # [B, N]
        y_coords = op.Gather(clamped, op.Constant(value_int=1), axis=2)  # [B, N]

        # Look up position embeddings from table
        x_table = op.Gather(self.position_embedding_table, op.Constant(value_int=0), axis=0)
        y_table = op.Gather(self.position_embedding_table, op.Constant(value_int=1), axis=0)
        x_emb = op.Gather(x_table, x_coords, axis=0)  # [B, N, hidden]
        y_emb = op.Gather(y_table, y_coords, axis=0)  # [B, N, hidden]

        # Zero position embeddings for padding patches (HF: torch.where(padding.unsqueeze(-1), 0.0, pos_emb))
        pos_emb = op.Add(x_emb, y_emb)  # [B, N, hidden]
        is_pad_expanded = op.Unsqueeze(is_padding, [2])  # [B, N, 1] — broadcast over hidden
        zero = op.CastLike(op.Constant(value_float=0.0), pos_emb)
        pos_emb = op.Where(is_pad_expanded, zero, pos_emb)

        return op.Add(hidden_states, pos_emb), is_padding  # ([B, N, hidden], [B, N])


class _Gemma4VisionEncoderCore(nn.Module):
    """Gemma4 full vision encoder: patch embedding + transformer blocks.

    Accepts pre-patchified pixel values ``[B, N, 3*P^2]`` and position IDs
    ``[B, N, 2]``.  Returns patch features ``[B, N, vision_hidden]``.
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        vc = config.vision
        self.patch_embedder = _Gemma4VisionPatchEmbedder(
            patch_size=vc.patch_size or 16,
            hidden_size=vc.hidden_size,
            position_embedding_size=vc.position_embedding_size or 128,
        )
        self.layers = nn.ModuleList(
            [
                Gemma4VisionEncoderLayer(
                    hidden_size=vc.hidden_size,
                    intermediate_size=vc.intermediate_size,
                    num_heads=vc.num_attention_heads,
                    norm_eps=vc.norm_eps,
                )
                for _ in range(vc.num_hidden_layers)
            ]
        )
        # No post-encoder norm: HF Gemma4VisionEncoder has none.
        # The scale-free RMSNorm (embedding_pre_projection_norm) lives in
        # _Gemma4VisionEncoderModel.projector_norm, applied before the projector.

    def forward(
        self,
        op: builder.OpBuilder,
        pixel_values: ir.Value,
        pixel_position_ids: ir.Value,
    ) -> ir.Value:
        # Returns hidden_states [B, N, hidden] and is_padding [B, N bool]
        hidden_states, is_padding = self.patch_embedder(op, pixel_values, pixel_position_ids)

        # Build additive attention bias [B, 1, 1, N] masking out padding columns.
        # Valid positions get 0 (no effect), padding columns get -1e9 (suppressed).
        # This matches HF Gemma4VisionEncoder passing attention_mask=~padding_positions.
        neg_inf = op.CastLike(op.Constant(value_float=-1e9), hidden_states)
        zero_bias = op.CastLike(op.Constant(value_float=0.0), hidden_states)
        attn_bias = op.Where(is_padding, neg_inf, zero_bias)  # [B, N]
        attn_bias = op.Unsqueeze(attn_bias, [1, 2])  # [B, 1, 1, N]

        for layer in self.layers:
            hidden_states = layer(op, hidden_states, attn_bias)

        # Zero padding patches after all encoder blocks (matching HF pooler masked_fill).
        is_pad_expanded = op.Unsqueeze(is_padding, [2])  # [B, N, 1]
        zero = op.CastLike(op.Constant(value_float=0.0), hidden_states)
        hidden_states = op.Where(is_pad_expanded, zero, hidden_states)

        return hidden_states  # [B, N, vision_hidden]


# ---------------------------------------------------------------------------
# Gemma4 text decoder layers
# ---------------------------------------------------------------------------


class Gemma4TextAttention(nn.Module):
    """Gemma4 text multi-head attention with per-head QKV norms and KV sharing.

    Key differences from standard Attention:
    - Fixed scale=1.0 (HF hardcodes this)
    - Q and K normalized per-head with learnable RMSNorm
    - V normalized per-head with parameterless RMS (no learnable scale)
    - head_dim and rotary_embedding_dim differ between sliding/full layers
    - KV-shared layers borrow K,V from a source layer (no k/v projections)
    - Attention logit softcapping via the native ONNX Attention ``softcap``
      attribute (opset 24): ``tanh(qk / cap) * cap`` is applied after the
      QK dot-product and before softmax. The value is taken from
      ``config.attn_logit_softcapping`` (50.0 for Gemma4; 0.0 = disabled).

    Args:
        config: Gemma4Config.
        layer_idx: Index of this layer (0-based).
        layer_types: Full list of layer types for all layers.
        first_kv_shared_layer_idx: First layer index where KV sharing starts.
        head_dim: Head dimension (differs per layer type).
        rotary_embedding_dim: Dims to rotate (0 = full rotation).
    """

    def __init__(
        self,
        config: Gemma4Config,
        layer_idx: int,
        layer_types: list[str],
        first_kv_shared_layer_idx: int,
        head_dim: int,
        rotary_embedding_dim: int,
    ):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = head_dim
        self.scaling = 1.0
        # attn_logit_softcapping maps directly to the ONNX Attention op's
        # native ``softcap`` attribute (opset 24). No manual Tanh/scale ops needed.
        self.softcap = config.attn_logit_softcapping
        self._v_norm_eps = config.rms_norm_eps
        self.rotary_embedding_dim = rotary_embedding_dim
        self._rope_interleave = config.rope_interleave
        self.layer_idx = layer_idx

        # KV sharing: layers >= first_kv_shared_layer_idx borrow K,V from source
        self.is_kv_shared_layer = layer_idx >= first_kv_shared_layer_idx > 0
        prev_layers = layer_types[:first_kv_shared_layer_idx]
        if self.is_kv_shared_layer:
            # Reverse-scan prev_layers to find the last non-shared layer with the
            # same type — KV-shared layers borrow K,V from that source layer.
            self.kv_shared_layer_index = (
                len(prev_layers) - 1 - prev_layers[::-1].index(layer_types[layer_idx])
            )
            self.provides_shared_kv = False
        else:
            self.kv_shared_layer_index = None
            # True for the last non-shared layer of each type that has downstream
            # KV-shared layers depending on it — it must store its K,V for reuse.
            self.provides_shared_kv = first_kv_shared_layer_idx > 0 and (
                layer_idx
                == len(prev_layers) - 1 - prev_layers[::-1].index(layer_types[layer_idx])
            )

        # All layers have Q projection + Q norm + output projection
        self.q_proj = Linear(
            config.hidden_size, config.num_attention_heads * head_dim, bias=False
        )
        self.q_norm = RMSNorm(head_dim, eps=config.rms_norm_eps)
        self.o_proj = Linear(
            config.num_attention_heads * head_dim, config.hidden_size, bias=False
        )

        # KV-shared layers borrow K,V — no projections needed
        if not self.is_kv_shared_layer:
            self.k_proj = Linear(
                config.hidden_size, config.num_key_value_heads * head_dim, bias=False
            )
            self.v_proj = Linear(
                config.hidden_size, config.num_key_value_heads * head_dim, bias=False
            )
            self.k_norm = RMSNorm(head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        attention_bias: ir.Value,
        position_embeddings: tuple | None = None,
        shared_kv_states: dict | None = None,
        past_key_value: tuple | None = None,
    ):
        from mobius.components._attention import _apply_attention, apply_rotary_pos_emb

        # Q projection + per-head Q norm + optional RoPE
        query_states = self.q_proj(op, hidden_states)
        query_states = op.Reshape(query_states, [0, 0, -1, self.head_dim])
        query_states = self.q_norm(op, query_states)
        query_states = op.Reshape(query_states, [0, 0, -1])

        if position_embeddings is not None:
            query_states = apply_rotary_pos_emb(
                op,
                x=query_states,
                position_embeddings=position_embeddings,
                num_heads=self.num_attention_heads,
                rotary_embedding_dim=self.rotary_embedding_dim,
                interleaved=self._rope_interleave,
            )

        if self.is_kv_shared_layer:
            # Borrow full-history K,V from source layer.
            # present_key/value from the ONNX Attention op is 4D:
            #   [batch, kv_heads, total_seq, head_dim]
            # The Attention op expects key/value as 3D:
            #   [batch, total_seq, kv_heads * head_dim]
            # Transpose and reshape to match.
            src_key, src_value = shared_kv_states[self.kv_shared_layer_index]

            # [B, kv_heads, total_seq, head_dim] → [B, total_seq, kv_heads, head_dim]
            src_key = op.Transpose(src_key, perm=[0, 2, 1, 3])
            src_value = op.Transpose(src_value, perm=[0, 2, 1, 3])

            # [B, total_seq, kv_heads, head_dim] → [B, total_seq, kv_heads * head_dim]
            kv_hidden = self.num_key_value_heads * self.head_dim
            batch_d = op.Shape(src_key, start=0, end=1)
            seq_d = op.Shape(src_key, start=1, end=2)
            kv_h = op.Constant(value_ints=[kv_hidden])
            tgt_shape = op.Concat(batch_d, seq_d, kv_h, axis=0)
            src_key = op.Reshape(src_key, tgt_shape)
            src_value = op.Reshape(src_value, tgt_shape)

            attn_output, present_key, present_value = _apply_attention(
                op,
                query_states,
                src_key,
                src_value,
                attention_bias,
                past_key=None,
                past_value=None,
                num_attention_heads=self.num_attention_heads,
                num_key_value_heads=self.num_key_value_heads,
                scale=self.scaling,
                softcap=self.softcap,
            )
        else:
            # K projection + per-head K norm + optional RoPE
            key_states = self.k_proj(op, hidden_states)
            key_states = op.Reshape(key_states, [0, 0, -1, self.head_dim])
            key_states = self.k_norm(op, key_states)
            key_states = op.Reshape(key_states, [0, 0, -1])

            if position_embeddings is not None:
                key_states = apply_rotary_pos_emb(
                    op,
                    x=key_states,
                    position_embeddings=position_embeddings,
                    num_heads=self.num_key_value_heads,
                    rotary_embedding_dim=self.rotary_embedding_dim,
                    interleaved=self._rope_interleave,
                )

            # V projection + parameterless per-head V normalisation
            value_states = self.v_proj(op, hidden_states)
            value_states = op.Reshape(
                value_states,
                op.Constant(value_ints=[0, 0, self.num_key_value_heads, self.head_dim]),
            )
            sq = op.Mul(value_states, value_states)
            mean_sq = op.ReduceMean(sq, [-1], keepdims=1)
            # Use op.Constant to create a 1D tensor node (not a scalar initializer).
            # Scalar Python floats use a type-keyed cache that can fail when upstream
            # type information is missing (e.g., after custom ops like com.microsoft.MoE).
            eps = op.Constant(value_floats=[self._v_norm_eps])
            rms = op.Sqrt(op.Add(mean_sq, op.CastLike(eps, mean_sq)))
            value_states = op.Div(value_states, rms)
            value_states = op.Reshape(value_states, [0, 0, -1])

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
                softcap=self.softcap,
            )

            # Source layers store K,V for downstream KV-shared layers
            if self.provides_shared_kv and shared_kv_states is not None:
                shared_kv_states[self.layer_idx] = (present_key, present_value)

        attn_output = self.o_proj(op, attn_output)
        return attn_output, (present_key, present_value)


# ---------------------------------------------------------------------------
# Gemma4 MoE router
# ---------------------------------------------------------------------------


class _Gemma4MoeRouter(nn.Module):
    """Gemma4 MoE router: scale-free RMSNorm → learned scale → linear → softmax.

    Matches ``Gemma4TextRouter`` in HuggingFace.

    Parameters (aligned with HF state_dict, with one weight-fold):
    - ``scale`` [hidden_size]: ``router.scale * hidden_size^-0.5`` (folded during
      ``preprocess_weights`` — see ``Gemma4CausalLMModel.preprocess_weights``)
    - ``proj.weight`` [num_experts, hidden_size]: router logits (no bias)
    - ``per_expert_scale`` [num_experts]: per-expert output weight scaling

    ``router.norm`` is scale-free (no learnable parameter, omitted from state_dict).

    The ``hidden_size^-0.5`` scale factor is absorbed into ``self.scale`` at weight-load
    time to avoid graph-level float-constant collisions across multiple decoder layers.

    Args:
        hidden_size: Model hidden dimension.
        num_experts: Total number of experts.
        rms_norm_eps: Epsilon for the scale-free RMSNorm.
    """

    def __init__(self, hidden_size: int, num_experts: int, rms_norm_eps: float = 1e-6):
        super().__init__()
        self._eps = rms_norm_eps
        self._hidden_size = hidden_size
        # ``scale`` stores router.scale * hidden_size^-0.5 (folded in preprocess_weights)
        self.scale = nn.Parameter([hidden_size])
        self.proj = Linear(hidden_size, num_experts, bias=False)
        self.per_expert_scale = nn.Parameter([num_experts])

    def forward(self, op: builder.OpBuilder, hidden_states: ir.Value) -> ir.Value:
        """Compute router probabilities over all experts.

        Args:
            hidden_states: [num_tokens, hidden_size] (batch-sequence flattened)

        Returns:
            router_probs: [num_tokens, num_experts] full softmax probabilities
        """
        # Scale-free RMSNorm using RMSNormalization op (epsilon as attribute avoids
        # graph-level float constant name collisions across multiple decoder layers).
        h_shape = op.Shape(hidden_states, start=1, end=2)  # [1] containing hidden_size
        ones = op.CastLike(
            op.ConstantOfShape(h_shape, value=ir.tensor(np.ones(1, dtype=np.float32))),
            hidden_states,
        )
        x_normed = op.RMSNormalization(hidden_states, ones, epsilon=self._eps, axis=-1)
        # Scale: x_normed * self.scale  (hidden_size^-0.5 already folded into scale)
        x_scaled = op.Mul(x_normed, self.scale)
        # Linear projection then softmax -> router_probs [num_tokens, num_experts]
        expert_scores = self.proj(op, x_scaled)
        return op.Softmax(expert_scores, axis=-1)


class Gemma4DecoderLayer(nn.Module):
    """Gemma4 text decoder layer with 4 norms, layer_scalar, and optional per-layer input.

    Architecture (dense path):
        h = residual + post_attn_norm(attn(input_layernorm(h)))
        h = residual + post_ff_norm(mlp(pre_ff_norm(h)))
        if per_layer_input:
            h += post_per_layer_norm(project(act(gate(h)) * per_layer_input))
        h = h * layer_scalar   # LAST step, after per-layer input

    When ``enable_moe_block=True`` (Gemma4 26B-A4B, 31B), the MLP block uses a
    parallel architecture — a dense MLP and a MoE block run independently, then
    their outputs are summed before the final post-FF norm::

        h = pre_ff_norm(h)                                    # pre_feedforward_layernorm
        h_dense  = post_ff_norm_dense(mlp(h))                 # post_feedforward_layernorm_1
        h_moe    = post_ff_norm_moe(                          # post_feedforward_layernorm_2
            experts(pre_ff_norm_moe(residual), router(residual))  # pre_feedforward_layernorm_2
        )
        h = post_ff_norm(h_dense + h_moe)                    # post_feedforward_layernorm
        h = residual + h

    Uses standard RMSNorm (not OffsetRMSNorm). KV-shared layers use double-wide
    MLP when config.use_double_wide_mlp=True.
    """

    def __init__(self, config: Gemma4Config, layer_idx: int):
        super().__init__()
        layer_types = config.layer_types or ["sliding_attention"] * config.num_hidden_layers
        first_kv_shared = config.num_hidden_layers - config.num_kv_shared_layers
        layer_type = layer_types[layer_idx]
        is_full = layer_type == "full_attention"

        head_dim = (config.global_head_dim or config.head_dim) if is_full else config.head_dim
        # ProportionalRope handles partial rotation via zero-padded cos/sin (full head_dim
        # coverage). The ONNX RotaryEmbedding op must see rotary_embedding_dim=0 (full)
        # so it pairs dims using the split-half convention over the entire head_dim.
        # For sliding (DefaultRope, full rotation), rotary_embedding_dim=0 is also correct.
        rotary_dim = 0

        self.self_attn = Gemma4TextAttention(
            config,
            layer_idx=layer_idx,
            layer_types=layer_types,
            first_kv_shared_layer_idx=first_kv_shared,
            head_dim=head_dim,
            rotary_embedding_dim=rotary_dim,
        )

        is_kv_shared = layer_idx >= first_kv_shared > 0
        intermediate_size = config.intermediate_size * (
            2 if (config.use_double_wide_mlp and is_kv_shared) else 1
        )
        self.mlp = GatedMLP(
            hidden_size=config.hidden_size,
            intermediate_size=intermediate_size,
            activation=config.hidden_act,
            bias=config.mlp_bias,
        )

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # layer_scalar: learned or constant scalar applied at end of layer
        self.layer_scalar = nn.Parameter([1])

        self._per_layer_dim = config.hidden_size_per_layer_input
        if self._per_layer_dim > 0:
            self.per_layer_input_gate = Linear(
                config.hidden_size, self._per_layer_dim, bias=False
            )
            self.per_layer_projection = Linear(
                self._per_layer_dim, config.hidden_size, bias=False
            )
            self.post_per_layer_input_norm = RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )
            self.act_fn = get_activation(config.hidden_act)

        self._enable_moe_block = config.enable_moe_block
        if config.enable_moe_block:
            assert config.num_local_experts is not None, "num_local_experts required for MoE"
            assert config.num_experts_per_tok is not None, (
                "num_experts_per_tok required for MoE"
            )
            assert config.moe_intermediate_size is not None, (
                "moe_intermediate_size required for MoE"
            )

            self._top_k = config.num_experts_per_tok
            self._num_experts = config.num_local_experts
            moe_inter = config.moe_intermediate_size

            self.router = _Gemma4MoeRouter(
                config.hidden_size, config.num_local_experts, config.rms_norm_eps
            )
            # Expert weights stored as 3D tensors (all experts stacked).
            # fc1_experts_weights: gate+up combined [E, 2*moe_inter, hidden].
            # fc2_experts_weights: down projection [E, hidden, moe_inter].
            # These map to HF experts.gate_up_proj and experts.down_proj.
            self.fc1_experts_weights = nn.Parameter(
                [config.num_local_experts, 2 * moe_inter, config.hidden_size]
            )
            self.fc2_experts_weights = nn.Parameter(
                [config.num_local_experts, config.hidden_size, moe_inter]
            )
            # MoE-specific norms (in addition to the shared pre/post_feedforward norms)
            self.pre_feedforward_layernorm_2 = RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )
            self.post_feedforward_layernorm_1 = RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )
            self.post_feedforward_layernorm_2 = RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        attention_bias: ir.Value,
        position_embeddings: tuple,
        shared_kv_states: dict,
        per_layer_input: ir.Value | None,
        past_key_value: tuple | None,
    ):
        # Attention block: pre-norm -> attn -> post-norm -> residual
        residual = hidden_states
        hidden_states = self.input_layernorm(op, hidden_states)
        attn_output, present_key_value = self.self_attn(
            op,
            hidden_states=hidden_states,
            attention_bias=attention_bias,
            position_embeddings=position_embeddings,
            shared_kv_states=shared_kv_states,
            past_key_value=past_key_value,
        )
        hidden_states = self.post_attention_layernorm(op, attn_output)
        hidden_states = op.Add(residual, hidden_states)

        # MLP block: dense MLP path runs always; parallel MoE path added when enabled.
        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(op, hidden_states)
        hidden_states = self.mlp(op, hidden_states)

        if self._enable_moe_block:
            # Hybrid dense+MoE architecture:
            #   dense path output: post_ff_norm_1(mlp(pre_ff_norm(h)))
            #   moe   path output: post_ff_norm_2(experts(pre_ff_norm_2(residual)))
            #   combined: post_ff_norm(dense + moe) + residual
            dense_out = self.post_feedforward_layernorm_1(op, hidden_states)

            # MoE input is the pre-attention residual (flattened to 2D for routing).
            batch_size = op.Shape(residual, start=0, end=1)  # [1] scalar
            seq_len = op.Shape(residual, start=1, end=2)  # [1] scalar
            hidden_size = op.Shape(residual, start=2, end=3)  # [1] scalar
            num_tokens = op.Mul(batch_size, seq_len)  # [1]
            flat_shape = op.Concat(num_tokens, hidden_size, axis=0)  # [2]
            residual_flat = op.Reshape(residual, flat_shape)  # [B*S, H]

            # router_probs: [B*S, E] full softmax over all experts
            router_probs = self.router(op, residual_flat)  # [B*S, E]

            # Norm residual before experts
            normed_flat = self.pre_feedforward_layernorm_2(op, residual_flat)  # [B*S, H]

            caps = ep_capabilities()
            if caps.supports_fused_moe:
                # Fused MoE op: handles top-k selection + expert dispatch internally.
                # NOTE: per_expert_scale is NOT applied in fused path (ORT op limitation).
                # CastLike restores dtype after op.MoE (custom op, type=None on output)
                # so downstream ops can correctly infer types and share scalar initializers.
                # Using CastLike(target=normed_flat) preserves bf16/fp16/fp32 from the input.
                moe_out_flat = op.CastLike(
                    op.MoE(  # type: ignore[attr-defined]
                        normed_flat,
                        router_probs,
                        self.fc1_experts_weights,
                        self.fc2_experts_weights,
                        activation_type="silu",
                        k=self._top_k,
                        normalize_routing_weights=1,
                        _domain="com.microsoft",
                    ),
                    normed_flat,  # match input dtype (bf16/fp16/fp32)
                )  # [B*S, H]
            else:
                moe_out_flat = self._dispatch_moe_fallback(op, normed_flat, router_probs)

            moe_out = op.Reshape(moe_out_flat, op.Shape(residual))  # [B, S, H]
            moe_out = self.post_feedforward_layernorm_2(op, moe_out)
            ff_out = op.Add(dense_out, moe_out)
            hidden_states = self.post_feedforward_layernorm(op, ff_out)
            hidden_states = op.Add(residual, hidden_states)
        else:
            hidden_states = self.post_feedforward_layernorm(op, hidden_states)
            hidden_states = op.Add(residual, hidden_states)

        # Per-layer input gating (skip when disabled)
        if self._per_layer_dim > 0 and per_layer_input is not None:
            residual = hidden_states
            gated = self.per_layer_input_gate(op, hidden_states)
            gated = self.act_fn(op, gated)
            gated = op.Mul(gated, per_layer_input)
            projected = self.per_layer_projection(op, gated)
            projected = self.post_per_layer_input_norm(op, projected)
            hidden_states = op.Add(residual, projected)

        # Layer scalar LAST (after per-layer input contribution)
        hidden_states = op.Mul(hidden_states, self.layer_scalar)

        return hidden_states, present_key_value

    def _dispatch_moe_fallback(
        self,
        op: builder.OpBuilder,
        normed_flat: ir.Value,
        router_probs: ir.Value,
    ) -> ir.Value:
        """Fallback expert dispatch (static unroll) when fused MoE op is unavailable.

        Iterates over each expert (outer loop), accumulates the routing weight for
        all k slots that select that expert (inner loop), applies the SwiGLU MLP,
        and accumulates into the output tensor.

        This produces O(E x K) ONNX nodes (e.g. Gemma4 26B: 256 experts x 2 top-k =
        512 per layer).  The primary path uses a fused MoE op for supported EPs
        (CUDA/DML); this fallback is only invoked for CPU/other EPs where the fused
        op is absent.

        Args:
            normed_flat: [T, H] — pre-normed input for the experts (T = B*S).
            router_probs: [T, E] — full softmax router probabilities.

        Returns:
            [T, H] — weighted sum of expert outputs.
        """
        # Top-K selection: top_weights/top_indices both [T, K]
        top_weights_raw, top_indices = op.TopK(
            router_probs, op.Constant(value_ints=[self._top_k]), axis=-1
        )
        # Arithmetic normalisation: weights sum to 1 (matches HF: top_k_weights /= top_k_weights.sum(-1, keepdim=True))
        top_weights = op.Div(
            top_weights_raw, op.ReduceSum(top_weights_raw, [1], keepdims=1)
        )  # [T, K]

        # Scale routing weights by per_expert_scale for each selected expert
        # Gather from [E] using [T, K] indices → result [T, K]
        pes_topk = op.Gather(self.router.per_expert_scale, top_indices, axis=0)  # [T, K]
        top_weights = op.Mul(top_weights, op.CastLike(pes_topk, top_weights))  # [T, K]

        # Build output accumulator, matching dtype of the input
        out_shape = op.Shape(normed_flat)  # [T, H] shape vec
        output = op.CastLike(
            op.ConstantOfShape(out_shape, value=ir.tensor(np.zeros(1, dtype=np.float32))),
            normed_flat,
        )

        # T_shape: 1-D tensor [T] used for per-expert weight accumulation
        t_shape = op.Shape(normed_flat, start=0, end=1)  # shape vec of length 1

        for e_idx in range(self._num_experts):
            # Gather this expert's weight matrices
            fc1 = op.Squeeze(
                op.Gather(self.fc1_experts_weights, [e_idx], axis=0), [0]
            )  # [2*moe_inter, H]
            fc2 = op.Squeeze(
                op.Gather(self.fc2_experts_weights, [e_idx], axis=0), [0]
            )  # [H, moe_inter]

            # Gated SiLU MLP: fc1 produces gate+up concatenated → SwiGLU
            proj = op.MatMul(normed_flat, op.Transpose(fc1))  # [T, 2*moe_inter]
            half = op.Shape(fc2, start=1, end=2)  # [moe_inter]
            gate = op.Slice(proj, [0], half, [1])  # [T, moe_inter] first half
            up = op.Slice(proj, half, op.Shape(proj, start=1, end=2), [1])  # second half
            # SiLU(gate) = gate * sigmoid(gate)
            expert_out = op.MatMul(
                op.Mul(op.Mul(gate, op.Sigmoid(gate)), up),  # [T, moe_inter]
                op.Transpose(fc2),  # → [T, H]
            )

            # Accumulate routing weight for expert e_idx across all k slots
            e_weight = op.CastLike(
                op.ConstantOfShape(t_shape, value=ir.tensor(np.zeros(1, dtype=np.float32))),
                top_weights,
            )
            for k_idx in range(self._top_k):
                # idx_k: [T] — which expert was selected at slot k_idx
                idx_k = op.Squeeze(op.Slice(top_indices, [k_idx], [k_idx + 1], [1]), [1])
                # w_k: [T] — routing weight for slot k_idx
                w_k = op.Squeeze(op.Slice(top_weights, [k_idx], [k_idx + 1], [1]), [1])
                # Add w_k only for tokens routed to expert e_idx at this slot
                is_expert = op.CastLike(op.Equal(idx_k, op.Constant(value_int=e_idx)), w_k)
                e_weight = op.Add(e_weight, op.Mul(is_expert, w_k))

            # Weight expert output by aggregated routing weight and add to output
            e_weight_2d = op.Reshape(
                e_weight,
                op.Concat(t_shape, op.Constant(value_ints=[1]), axis=0),  # [T, 1]
            )
            output = op.Add(output, op.Mul(expert_out, op.CastLike(e_weight_2d, expert_out)))

        return output


# ---------------------------------------------------------------------------
# Gemma4 text model
# ---------------------------------------------------------------------------


class Gemma4TextModel(nn.Module):
    """Gemma4 text transformer with hybrid local/global attention.

    Key differences from Gemma3TextModel:
    - Standard ``RMSNorm`` (no ``OffsetRMSNorm``).
    - Dual head_dim: local layers use ``config.head_dim``, global layers use
      ``config.global_head_dim``.
    - Dual RoPE: separate ``rotary_emb_local`` and ``rotary_emb_global`` instances
      with different theta and partial_rotary_factor.
    - Optional per-layer input embeddings (disabled when
      ``hidden_size_per_layer_input == 0``).

    Inputs may be ``input_ids`` (text-only) or ``inputs_embeds`` (VL decoder path).
    """

    def __init__(self, config: Gemma4Config):
        super().__init__()
        self._dtype = config.dtype

        embed_scale = math.sqrt(config.hidden_size)
        self.embed_tokens = Gemma3TextScaledWordEmbedding(
            config.vocab_size,
            config.hidden_size,
            config.pad_token_id,
            embed_scale=embed_scale,
        )

        layer_types = config.layer_types or ["sliding_attention"] * config.num_hidden_layers
        if len(layer_types) != config.num_hidden_layers:
            raise ValueError(
                f"Gemma4Config.layer_types length ({len(layer_types)}) must match "
                f"num_hidden_layers ({config.num_hidden_layers})"
            )
        self.layer_types = layer_types
        self.sliding_window = config.sliding_window

        # Local (sliding window) config — full rotation, local rope_theta
        local_config = dataclasses.replace(
            config,
            rope_type="default",
            rope_scaling=None,
            partial_rotary_factor=1.0,
        )
        # Global (full attention) config — larger head_dim, proportional RoPE
        # (partial rotation via zero-padded inv_freq to cover full head_dim)
        global_head_dim = config.global_head_dim or config.head_dim
        global_config = dataclasses.replace(
            config,
            head_dim=global_head_dim,
            rope_theta=config.global_rope_theta,
            partial_rotary_factor=config.global_partial_rotary_factor,
            rope_type="proportional",
            rope_scaling=None,
            sliding_window=None,
        )

        self.layers = nn.ModuleList(
            [Gemma4DecoderLayer(config, layer_idx=i) for i in range(len(layer_types))]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.rotary_emb_local = initialize_rope(local_config)
        self.rotary_emb_global = initialize_rope(global_config)

        # Per-layer input embeddings (optional feature)
        self._per_layer_dim = getattr(config, "hidden_size_per_layer_input", 0)
        self._hidden_size = config.hidden_size
        if self._per_layer_dim:
            self._num_layers = config.num_hidden_layers
            vocab_per_layer = getattr(config, "vocab_size_per_layer_input", 0)
            self.embed_tokens_per_layer = Gemma3TextScaledWordEmbedding(
                vocab_per_layer,
                config.num_hidden_layers * self._per_layer_dim,
                config.pad_token_id,
                embed_scale=float(self._per_layer_dim**0.5),
            )
            self.per_layer_model_projection = Linear(
                config.hidden_size,
                config.num_hidden_layers * self._per_layer_dim,
                bias=False,
            )
            self.per_layer_projection_norm = RMSNorm(
                self._per_layer_dim, eps=config.rms_norm_eps
            )

    def _compute_per_layer_inputs(
        self,
        op: builder.OpBuilder,
        input_ids: ir.Value | None,
        inputs_embeds: ir.Value,
    ) -> ir.Value | None:
        """Compute per-layer input embeddings ``[B, S, num_layers, per_layer_dim]``."""
        if not self._per_layer_dim:
            return None

        # Project hidden states and scale by hidden_size**-0.5 (matches HF)
        proj = self.per_layer_model_projection(op, inputs_embeds)
        proj = op.Mul(proj, float(self._hidden_size**-0.5))
        proj = op.Reshape(
            proj, op.Constant(value_ints=[0, 0, self._num_layers, self._per_layer_dim])
        )
        proj = self.per_layer_projection_norm(op, proj)

        if input_ids is not None:
            token_emb = self.embed_tokens_per_layer(op, input_ids)
            token_emb = op.Reshape(
                token_emb,
                op.Constant(value_ints=[0, 0, self._num_layers, self._per_layer_dim]),
            )
            proj = op.Add(proj, token_emb)

        # Scale combined result by 2**-0.5 (matches HF project_per_layer_inputs)
        return op.Mul(proj, float(0.5**0.5))

    def forward(
        self,
        op: builder.OpBuilder,
        input_ids: ir.Value | None,
        attention_mask: ir.Value,
        position_ids: ir.Value,
        past_key_values: list | None = None,
        inputs_embeds: ir.Value | None = None,
    ) -> tuple[ir.Value, list]:
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.embed_tokens(op, input_ids)

        per_layer_inputs = self._compute_per_layer_inputs(op, input_ids, hidden_states)

        position_embeddings_dict = {
            "sliding_attention": self.rotary_emb_local(op, position_ids),
            "full_attention": self.rotary_emb_global(op, position_ids),
        }

        # Use hidden_states for query length when input_ids is None (VL decoder)
        query_input = input_ids if input_ids is not None else hidden_states
        attention_bias_dict = {
            "sliding_attention": create_attention_bias(
                op,
                input_ids=query_input,
                attention_mask=attention_mask,
                sliding_window=self.sliding_window,
                dtype=self._dtype,
            ),
            "full_attention": create_attention_bias(
                op,
                input_ids=query_input,
                attention_mask=attention_mask,
                dtype=self._dtype,
            ),
        }

        # shared_kv_states: source layers populate it, shared layers consume it
        shared_kv_states: dict = {}
        present_key_values = []

        # Build per-layer past_kv list for all num_hidden_layers layers.
        # past_key_values has only num_kv_layers entries (no entry for KV-shared
        # layers). Expand it to a full per-layer list so we can zip over all
        # layers without truncation.
        if past_key_values is not None:
            kv_iter = iter(past_key_values)
            past_kvs: list = [
                None if layer.self_attn.is_kv_shared_layer else next(kv_iter)
                for layer in self.layers
            ]
        else:
            past_kvs = [None] * len(self.layers)

        for i, (layer, layer_type, past_kv) in enumerate(
            zip(self.layers, self.layer_types, past_kvs)
        ):
            if per_layer_inputs is not None:
                idx = op.Constant(value_ints=[i])
                pli = op.Gather(per_layer_inputs, idx, axis=2)
                per_layer_input = op.Squeeze(pli, [2])
            else:
                per_layer_input = None
            hidden_states, present_kv = layer(
                op,
                hidden_states=hidden_states,
                attention_bias=attention_bias_dict[layer_type],
                position_embeddings=position_embeddings_dict[layer_type],
                shared_kv_states=shared_kv_states,
                per_layer_input=per_layer_input,
                past_key_value=past_kv,
            )
            # KV-shared layers borrow K,V from source layers — exclude from
            # present_key_values so the output has exactly num_kv_layers entries.
            if not layer.self_attn.is_kv_shared_layer:
                present_key_values.append(present_kv)

        hidden_states = self.norm(op, hidden_states)
        return hidden_states, present_key_values


# ---------------------------------------------------------------------------
# Gemma4CausalLMModel (text-only)
# ---------------------------------------------------------------------------


class Gemma4CausalLMModel(CausalLMModel):
    """Gemma 4 text-only causal language model.

    Wraps :class:`Gemma4TextModel` (transformer backbone) with an ``lm_head``
    linear projection to produce next-token logits.  Corresponds to HF
    ``Gemma4ForCausalLM``, which similarly wraps ``Gemma4Model`` (our
    :class:`Gemma4TextModel`) and adds the language-model head on top.

    Registered as ``gemma4_text`` in the model registry.  Uses hybrid
    local/global attention, standard ``RMSNorm``, and optional per-layer
    input embeddings.
    """

    config_class: type = Gemma4Config
    category: str = "Text"
    default_task: str = "gemma4-text-generation"

    def __init__(self, config: Gemma4Config):
        nn.Module.__init__(self)
        self.config = config
        self.model = Gemma4TextModel(config)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        op: builder.OpBuilder,
        input_ids: ir.Value,
        attention_mask: ir.Value,
        position_ids: ir.Value,
        past_key_values: list | None = None,
        inputs_embeds: ir.Value | None = None,
    ) -> tuple[ir.Value, list]:
        hidden_states, present_key_values = self.model(
            op,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
        )
        logits = self.lm_head(op, hidden_states)
        # Optional final logit soft-capping (tanh scaled): logit_cap * tanh(x / logit_cap)
        if self.config.final_logit_softcapping:
            cap = float(self.config.final_logit_softcapping)
            logits = op.Mul(op.Tanh(op.Div(logits, cap)), cap)
        return logits, present_key_values

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        # Strip optional 'language_model.' prefix from multimodal checkpoints
        for key in list(state_dict.keys()):
            if "language_model." in key:
                new_key = key.replace("language_model.", "")
                state_dict[new_key] = state_dict.pop(key)
            elif "vision_tower" in key or "embed_vision" in key:
                state_dict.pop(key, None)
        # Map HF expert weight names to our 3D stacked parameter names.
        # HF stores: layers.N.experts.gate_up_proj [E, 2*inter, H]
        #             layers.N.experts.down_proj     [E, H, inter]
        # We store:  layers.N.fc1_experts_weights   [E, 2*inter, H]
        #             layers.N.fc2_experts_weights   [E, H, inter]
        for key in list(state_dict.keys()):
            if ".experts.gate_up_proj" in key:
                new_key = key.replace(".experts.gate_up_proj", ".fc1_experts_weights")
                state_dict[new_key] = state_dict.pop(key)
            elif ".experts.down_proj" in key:
                new_key = key.replace(".experts.down_proj", ".fc2_experts_weights")
                state_dict[new_key] = state_dict.pop(key)
        # Fold hidden_size^-0.5 into router.scale.
        # The router computes: x_normed * scale * hidden_size^-0.5.
        # We pre-multiply scale by hidden_size^-0.5 here so the forward only needs
        # x_normed * self.scale, avoiding float-constant name collisions across layers.
        if self.config.enable_moe_block:
            scale_factor = float(self.config.hidden_size**-0.5)
            for key in list(state_dict.keys()):
                if ".router.scale" in key:
                    state_dict[key] = state_dict[key] * scale_factor
        return super().preprocess_weights(state_dict)


# ---------------------------------------------------------------------------
# Gemma4 multimodal sub-models
# ---------------------------------------------------------------------------


class _Gemma4DecoderModel(nn.Module):
    """Gemma4 text decoder sub-model accepting ``inputs_embeds``.

    When ``hidden_size_per_layer_input > 0`` (e.g. E2B), the text model also
    needs the original ``input_ids`` to compute per-layer token embeddings that
    condition each decoder layer.  HF's ``Gemma4ForConditionalGeneration``
    passes *both* ``inputs_embeds`` and ``input_ids`` to the language model for
    this reason.  We mirror that by accepting ``input_ids`` as an optional
    ONNX graph input and forwarding it to :class:`Gemma4TextModel`.
    """

    def __init__(self, config: Gemma4Config):
        super().__init__()
        self.config = config
        self.model = Gemma4TextModel(config)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        op: builder.OpBuilder,
        inputs_embeds: ir.Value,
        attention_mask: ir.Value,
        position_ids: ir.Value,
        input_ids: ir.Value | None = None,
        past_key_values: list | None = None,
    ) -> tuple[ir.Value, list]:
        hidden_states, present_key_values = self.model(
            op,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
        )
        logits = self.lm_head(op, hidden_states)
        # Gemma4 applies final logit soft-capping: logit_cap * tanh(x / logit_cap)
        if self.config.final_logit_softcapping:
            cap = float(self.config.final_logit_softcapping)
            logits = op.Mul(op.Tanh(op.Div(logits, cap)), cap)
        return logits, present_key_values

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        return vlm_decoder_weights(state_dict, tie=self.config.tie_word_embeddings)


class _Gemma4VisionEncoderModel(nn.Module):
    """Gemma4 vision encoder sub-model: pre-patchified input -> projected features.

    Pipeline:
    1. ``encoder``: patch embedding + N transformer blocks + final norm
    2. Scale by ``sqrt(vision_hidden)`` (HF ``VisionPooler`` scaling step)
    3. ``projector_norm``: scale-free RMSNorm (HF ``embedding_pre_projection_norm``)
    4. ``projector``: Linear to text hidden size (HF ``embedding_projection``)

    Weight name mapping strips:
    - ``vision_tower.`` prefix -> ``encoder.``
    - ``.linear.`` infix from ``Gemma4ClippableLinear`` wrapper
    - ``embed_vision.embedding_projection.*`` -> ``projector.*``
    - ``embed_vision.embedding_pre_projection_norm.*`` -> skip (scale-free, no weight)
    """

    def __init__(self, config: Gemma4Config):
        super().__init__()
        vc = config.vision
        self.encoder = _Gemma4VisionEncoderCore(config)
        self._pooler_scale = float(vc.hidden_size**0.5)
        self.projector_norm = _Gemma4ScaleFreeRMSNorm(vc.hidden_size, eps=vc.norm_eps)
        self.projector = Linear(vc.hidden_size, config.hidden_size, bias=False)

    def forward(
        self,
        op: builder.OpBuilder,
        pixel_values: ir.Value,
        pixel_position_ids: ir.Value,
    ) -> ir.Value:
        # [B, N, 3*P^2] -> [B, N, vision_hidden]
        vision_features = self.encoder(op, pixel_values, pixel_position_ids)

        # Scale by sqrt(hidden_size) as in HF VisionPooler
        vision_features = op.Mul(
            vision_features,
            op.Constant(value_float=self._pooler_scale),
        )

        # Scale-free norm + linear projection -> [B, N, text_hidden]
        vision_features = self.projector_norm(op, vision_features)
        vision_features = self.projector(op, vision_features)

        # Flatten batch and token dims: [B, N, text_hidden] -> [B*N, text_hidden]
        hidden_size = op.Shape(vision_features, start=2, end=3)
        vision_features = op.Reshape(
            vision_features, op.Concat(op.Constant(value_ints=[-1]), hidden_size, axis=0)
        )
        return vision_features  # [B*N, text_hidden]

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        renamed: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            if key.startswith("vision_tower."):
                new_key = "encoder." + key[len("vision_tower.") :]
                # HF Gemma4VisionModel wraps its layer list in a Gemma4VisionEncoder
                # submodule, adding an extra "encoder." level. Our _Gemma4VisionEncoderCore
                # exposes its layers as "layers" directly, so strip the extra prefix.
                new_key = new_key.replace("encoder.encoder.", "encoder.", 1)
                # Flatten Gemma4ClippableLinear's .linear. wrapper
                new_key = new_key.replace(".linear.weight", ".weight")
                new_key = new_key.replace(".linear.bias", ".bias")
                renamed[new_key] = value
            elif key.startswith("embed_vision.embedding_projection."):
                suffix = key[len("embed_vision.embedding_projection.") :]
                renamed["projector." + suffix] = value
            elif key.startswith("embed_vision.embedding_pre_projection_norm."):
                pass  # Scale-free RMSNorm: normalizes without a learnable scale, no parameter
        return renamed


class Gemma4EmbeddingModel(nn.Module):
    """Gemma4 embedding sub-model: scaled token lookup + multimodal feature fusion.

    Always scatters vision features at image-token positions.  When the model
    has audio support (``config.audio is not None``), ``forward()`` also accepts
    ``audio_features`` and scatters them at audio-token positions.

    Both image and audio use a one-row dummy guard so that ORT's eager
    evaluation of both ``Where`` branches never ``Gather`` on a zero-length
    tensor during text-only / decode steps.

    Inputs (image-only variant):
    - ``input_ids [B, S]`` INT64
    - ``image_features [num_img_tokens, hidden_size]``

    Inputs (image + audio variant):
    - ``input_ids [B, S]`` INT64
    - ``image_features [num_img_tokens, hidden_size]``
    - ``audio_features [num_aud_tokens, hidden_size]``

    Output: ``inputs_embeds [B, S, hidden_size]``
    """

    def __init__(self, config: Gemma4Config):
        super().__init__()
        self.config = config
        embed_scale = math.sqrt(config.hidden_size)
        self.embed_tokens = Gemma3TextScaledWordEmbedding(
            config.vocab_size,
            config.hidden_size,
            config.pad_token_id,
            embed_scale=embed_scale,
        )
        self.image_token_id = config.image_token_id or 0
        # Audio token ID is only set when the model has an audio encoder.
        self.audio_token_id: int | None = config.audio.audio_token_id if config.audio else None

    def _scatter_features(
        self,
        op: builder.OpBuilder,
        hidden: ir.Value,
        input_ids: ir.Value,
        token_id: int,
        features: ir.Value,
    ) -> ir.Value:
        """Scatter ``features`` into ``hidden`` at positions matching ``token_id``.

        Appends a dummy zero row to ``features`` before Gather so that ORT's
        eager evaluation of the Where branches never faults on an empty tensor
        during text-only / decode steps.
        """
        mask = op.Equal(input_ids, op.Constant(value_int=token_id))
        mask_3d = op.Unsqueeze(mask, [-1])

        # CumSum → sub-1 → clip gives 0-based index into features for each token
        mask_int = op.Cast(mask, to=7)  # INT64
        cumsum = op.CumSum(mask_int, op.Constant(value_int=1))
        indices = op.Clip(op.Sub(cumsum, op.Constant(value_int=1)), op.Constant(value_int=0))

        # One-row dummy prevents empty-tensor Gather faults during decode steps.
        # Use Constant (static) + Unsqueeze to avoid ConstantOfShape, whose
        # dynamic-shape input blocks ONNX shape inference.
        dummy_row = op.Unsqueeze(
            op.CastLike(
                op.Constant(value_floats=[0.0] * self.config.hidden_size),
                features,
            ),
            [0],
        )  # [1, hidden_size]
        features_safe = op.Concat(features, dummy_row, axis=0)
        gathered = op.Gather(features_safe, indices, axis=0)
        return op.Where(mask_3d, gathered, hidden)

    def forward(
        self,
        op: builder.OpBuilder,
        input_ids: ir.Value,
        image_features: ir.Value,
        audio_features: ir.Value | None = None,
    ) -> ir.Value:
        # [B, S] → [B, S, hidden]
        hidden = self.embed_tokens(op, input_ids)

        # Scatter image features at image-token positions
        hidden = self._scatter_features(
            op, hidden, input_ids, self.image_token_id, image_features
        )

        # Scatter audio features at audio-token positions (only for AnyToAny models)
        if audio_features is not None:
            assert self.audio_token_id is not None, (
                "Gemma4EmbeddingModel received audio_features but audio_token_id is not set. "
                "Ensure config.audio is provided."
            )
            hidden = self._scatter_features(
                op, hidden, input_ids, self.audio_token_id, audio_features
            )

        return hidden

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        return vlm_embedding_weights(state_dict)


# ---------------------------------------------------------------------------
# _Gemma4AudioEncoderModel — Conformer encoder + projector sub-model
# ---------------------------------------------------------------------------


class _Gemma4AudioEncoderModel(nn.Module):
    """Gemma4 Conformer audio encoder sub-model.

    Wraps :class:`Gemma4AudioEncoder` and applies a learned linear projector
    (``embed_audio.embedding_projection`` in HF) that maps from the encoder
    output dimension to the text model's hidden size.

    Inputs:
    - ``input_features [B, T, input_size]``: mel-spectrogram features

    Output:
    - ``audio_features [B, T//4, text_hidden_size]``: projected audio tokens
    """

    def __init__(self, config: Gemma4Config):
        super().__init__()
        self.config = config
        ac = config.audio  # Gemma4AudioConfig (guaranteed non-None when used)
        input_size = (ac.input_size if ac else None) or 128
        hidden_size = (ac.hidden_size if ac else None) or 1024
        num_layers = (ac.num_layers if ac else None) or 12
        # HF config field is output_proj_dims (not output_dim)
        output_proj_dims = (
            getattr(ac, "output_proj_dims", None) if ac else None
        ) or config.hidden_size
        conv_channels = ac.subsampling_conv_channels if ac else None
        rms_norm_eps = config.rms_norm_eps or 1e-6

        self.encoder = Gemma4AudioEncoder(
            input_size=input_size,
            hidden_size=hidden_size,
            num_heads=8,  # fixed per Gemma4 audio_config
            num_layers=num_layers,
            conv_kernel_size=5,  # fixed per Gemma4 audio_config
            conv_channels=conv_channels,
            attention_context_left=13,  # fixed per Gemma4 audio_config
            output_proj_dims=output_proj_dims,
            rms_norm_eps=rms_norm_eps,
        )
        # Scale-free RMSNorm applied before the projection (HF embed_audio.embedding_pre_projection_norm).
        # with_scale=False in HF → no learnable weight → no checkpoint key, no ONNX initializer.
        self.pre_projection_norm = _Gemma4ScaleFreeRMSNorm(output_proj_dims, eps=rms_norm_eps)
        # Learned projection from encoder output space → text hidden size.
        # Corresponds to HF's embed_audio.embedding_projection (no bias).
        self.projector = Linear(output_proj_dims, config.hidden_size, bias=False)

    def forward(
        self,
        op: builder.OpBuilder,
        input_features: ir.Value,
    ) -> ir.Value:
        # [B, T, input_size] → encoder → [B, T//4, output_proj_dims]
        audio_features = self.encoder(op, input_features)
        # Scale-free RMSNorm before projection (HF embed_audio.embedding_pre_projection_norm)
        audio_features = self.pre_projection_norm(op, audio_features)
        # → projector → [B, T//4, text_hidden_size]
        return self.projector(op, audio_features)


# ---------------------------------------------------------------------------
# Gemma4Model — unified vision-language (+ optional audio) model
# ---------------------------------------------------------------------------


class Gemma4Model(nn.Module):
    """Unified Gemma4 multimodal model (3- or 4-model split).

    Builds three or four separate ONNX models depending on whether the config
    includes an audio sub-config:

    Always produced:
    - ``decoder``: Gemma4 text decoder taking ``inputs_embeds``
    - ``vision``: SigLIP-style encoder + projector
    - ``embedding``: scaled word embedding + multimodal feature fusion

    Added when ``config.audio is not None``:
    - ``speech``: Conformer audio encoder + projection to text hidden size

    Covers all Gemma4 variants:
    - Vision-language (26B-A4B, 31B): ``audio=None``
    - Any-to-Any (E2B-it, E4B-it): ``audio=Gemma4AudioConfig(...)``

    Registered as ``gemma4`` (and ``gemma4_any_to_any`` for back-compat).
    """

    default_task: str = "gemma4"
    category: str = "Multimodal"

    def __init__(self, config: Gemma4Config):
        super().__init__()
        self.config = config
        self.decoder = _Gemma4DecoderModel(config)
        self.vision_encoder = _Gemma4VisionEncoderModel(config)
        self.embedding = Gemma4EmbeddingModel(config)
        self.audio_encoder: _Gemma4AudioEncoderModel | None = (
            _Gemma4AudioEncoderModel(config) if config.audio is not None else None
        )

    def forward(self, op: builder.OpBuilder, **kwargs):
        raise NotImplementedError(
            "Gemma4Model is a multi-model split; Gemma4Task builds each sub-module "
            "(decoder, vision_encoder, embedding, and optionally audio_encoder) separately."
        )

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Rename HuggingFace weight keys to ONNX initializer names.

        HF multimodal checkpoints prefix every key with ``model.``
        (e.g. ``model.language_model.*``, ``model.vision_tower.*``).

        Mapping (after stripping the leading ``model.``):

        - ``language_model.lm_head.*`` → ``decoder.lm_head.*``
        - ``language_model.*`` → ``decoder.model.*``
        - ``language_model.embed_tokens.weight`` also → ``embedding.embed_tokens.weight``
        - ``vision_tower.*`` → ``vision_encoder.encoder.*``
          (strips HF's extra ``encoder.`` level and ``.linear.`` infix from
          ``Gemma4ClippableLinear``)
        - ``embed_vision.embedding_projection.*`` → ``vision_encoder.projector.*``
        - ``embed_vision.embedding_pre_projection_norm.*`` → skip (scale-free, no weight)
        - ``audio_tower.*`` → ``audio_encoder.encoder.*``
          (strips ``.linear.`` infix; renames ``subsample_conv_projection.layerN.{conv,norm}``
          to ``{conv,norm}N``)
        - ``embed_audio.embedding_projection.*`` → ``audio_encoder.projector.*``
        - ``embed_audio.embedding_pre_projection_norm.*`` → skip (``Gemma4RMSNorm`` with
          ``with_scale=False``; normalises activations but has no learnable ``weight``
          parameter and produces no checkpoint key — verified against google/gemma-4-E2B-it)

        Note: the decoder sub-model takes ``inputs_embeds`` rather than
        ``input_ids``, so ``embed_tokens`` is not a decoder initializer — the
        token embedding lives only in the ``embedding`` sub-model.
        """
        # Strip top-level "model." prefix used by HF multimodal checkpoints.
        state_dict = {
            (key[len("model.") :] if key.startswith("model.") else key): value
            for key, value in state_dict.items()
        }

        # Synthesize lm_head from embed_tokens when weights are tied
        if self.config.tie_word_embeddings:
            embed_key = "language_model.embed_tokens.weight"
            head_key = "language_model.lm_head.weight"
            if head_key not in state_dict and embed_key in state_dict:
                state_dict[head_key] = state_dict[embed_key]

        renamed: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            if key.startswith("language_model."):
                suffix = key[len("language_model.") :]
                if suffix.startswith("lm_head"):
                    # lm_head lives directly under decoder (not decoder.model)
                    renamed["decoder." + suffix] = value
                else:
                    # All other text weights nest under decoder.model.*
                    renamed["decoder.model." + suffix] = value
                    if suffix == "embed_tokens.weight":
                        # Token embedding is shared with the embedding sub-model
                        renamed["embedding.embed_tokens.weight"] = value

            elif key.startswith("vision_tower."):
                new_key = "vision_encoder.encoder." + key[len("vision_tower.") :]
                # HF wraps encoder layers under an extra "encoder." sub-module; strip it
                new_key = new_key.replace(
                    "vision_encoder.encoder.encoder.", "vision_encoder.encoder.", 1
                )
                # HF uses Gemma4ClippableLinear which adds a ".linear." infix; strip it
                new_key = new_key.replace(".linear.weight", ".weight")
                new_key = new_key.replace(".linear.bias", ".bias")
                renamed[new_key] = value

            elif key.startswith("embed_vision.embedding_projection."):
                suffix = key[len("embed_vision.embedding_projection.") :]
                renamed["vision_encoder.projector." + suffix] = value

            elif key.startswith("embed_vision.embedding_pre_projection_norm."):
                pass  # Scale-free RMSNorm: normalizes without a learnable scale, no weight

            elif key.startswith("audio_tower."):
                new_key = "audio_encoder.encoder." + key[len("audio_tower.") :]
                # HF Conformer linear layers use a ".linear." infix; strip it
                new_key = new_key.replace(".linear.weight", ".weight")
                new_key = new_key.replace(".linear.bias", ".bias")
                # HF subsample_conv_projection uses "layerN.conv" / "layerN.norm" names;
                # our ONNX module uses "convN" / "normN" directly.
                new_key = new_key.replace(
                    ".subsample_conv_projection.layer0.conv.",
                    ".subsample_conv_projection.conv0.",
                )
                new_key = new_key.replace(
                    ".subsample_conv_projection.layer0.norm.",
                    ".subsample_conv_projection.norm0.",
                )
                new_key = new_key.replace(
                    ".subsample_conv_projection.layer1.conv.",
                    ".subsample_conv_projection.conv1.",
                )
                new_key = new_key.replace(
                    ".subsample_conv_projection.layer1.norm.",
                    ".subsample_conv_projection.norm1.",
                )
                renamed[new_key] = value

            elif key.startswith("embed_audio.embedding_projection."):
                # Learned audio-to-text projector (embed_audio.embedding_projection in HF)
                suffix = key[len("embed_audio.embedding_projection.") :]
                renamed["audio_encoder.projector." + suffix] = value

            elif key.startswith("embed_audio."):
                # embed_audio.embedding_pre_projection_norm.* — Gemma4RMSNorm(with_scale=False):
                # normalises activations but carries no learnable weight.  No checkpoint key
                # is saved, so nothing to map here.  The norm IS applied in the ONNX forward
                # pass via _Gemma4AudioEncoderModel.pre_projection_norm.
                pass

            else:
                renamed[key] = value

        return renamed
