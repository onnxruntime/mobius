# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Gemma 4 model implementations.

Architecture variants:
- **Gemma4CausalLMModel**: Text-only causal LM (model_type ``gemma4_text``).
- **Gemma4MultiModalModel**: Multimodal 3-model split — decoder + vision + embedding
  (model_type ``gemma4``).  For 26B-A4B and 31B variants (Image-Text-to-Text).

Key architectural differences from Gemma3:
- Standard ``RMSNorm`` throughout (no ``OffsetRMSNorm``).
- Dual head_dim: local sliding-window layers use ``config.head_dim``; global
  full-attention layers use ``config.global_head_dim``.
- Dual RoPE: different ``rope_theta`` and ``partial_rotary_factor`` per layer type.
- Per-layer input gating (disabled when ``hidden_size_per_layer_input == 0``).
- Vision encoder: pre-patchified input ``[B, N, 3*P^2]`` with 2D position lookup,
  bidirectional attention, 4-norm structure, and scale-then-project pooling.
- Vision projector: scale-free RMSNorm -> Linear (matches ``embed_vision`` weights).
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import numpy as np
import torch
from onnxscript import nn
from onnxscript._internal import builder

from mobius._configs import ArchitectureConfig, Gemma4Config
from mobius._weight_utils import vlm_decoder_weights, vlm_embedding_weights
from mobius.components import (
    MLP,
    Attention,
    Linear,
    RMSNorm,
    create_attention_bias,
    initialize_rope,
)
from mobius.models.base import CausalLMModel
from mobius.models.gemma3_text import Gemma3TextScaledWordEmbedding

if TYPE_CHECKING:
    import onnx_ir as ir


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
        # All-ones scale tensor — no learnable parameter to load
        scale = op.ConstantOfShape(op.Constant(value_ints=[self.dim]), value=1.0)
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

    def forward(self, op: builder.OpBuilder, hidden_states: ir.Value) -> ir.Value:
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

        # Bidirectional attention (no is_causal, no KV cache)
        attn_output = op.Attention(
            q, k, v,
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

    def __init__(self, hidden_size: int, intermediate_size: int, num_heads: int, norm_eps: float):
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

    def forward(self, op: builder.OpBuilder, hidden_states: ir.Value) -> ir.Value:
        residual = hidden_states
        hidden_states = self.input_layernorm(op, hidden_states)
        hidden_states = self.self_attn(op, hidden_states)
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
    ) -> ir.Value:
        # pixel_values in [0,1] -> normalize to [-1, 1]: 2*(v - 0.5) = 2v - 1
        pixel_values = op.Sub(
            op.Mul(pixel_values, op.Constant(value_float=2.0)),
            op.Constant(value_float=1.0),
        )
        hidden_states = self.input_proj(op, pixel_values)  # [B, N, hidden]

        # pixel_position_ids: [B, N, 2] — clamp -1 (padding) to 0
        clamped = op.Clip(pixel_position_ids, op.Constant(value_int=0))

        # Extract x and y coordinates: each [B, N]
        x_coords = op.Squeeze(op.Gather(clamped, op.Constant(value_int=0), axis=2), [-1])
        y_coords = op.Squeeze(op.Gather(clamped, op.Constant(value_int=1), axis=2), [-1])

        # Look up position embeddings from table
        x_table = op.Gather(self.position_embedding_table, op.Constant(value_int=0), axis=0)
        y_table = op.Gather(self.position_embedding_table, op.Constant(value_int=1), axis=0)
        x_emb = op.Gather(x_table, x_coords, axis=0)  # [B, N, hidden]
        y_emb = op.Gather(y_table, y_coords, axis=0)  # [B, N, hidden]

        return op.Add(hidden_states, op.Add(x_emb, y_emb))


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
            position_embedding_size=getattr(vc, "_position_embedding_size", 128),
        )
        self.encoder = nn.ModuleList(
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
        self.post_layernorm = RMSNorm(vc.hidden_size, eps=vc.norm_eps)

    def forward(
        self,
        op: builder.OpBuilder,
        pixel_values: ir.Value,
        pixel_position_ids: ir.Value,
    ) -> ir.Value:
        hidden_states = self.patch_embedder(op, pixel_values, pixel_position_ids)
        for layer in self.encoder:
            hidden_states = layer(op, hidden_states)
        return self.post_layernorm(op, hidden_states)  # [B, N, vision_hidden]


# ---------------------------------------------------------------------------
# Gemma4 text decoder layers
# ---------------------------------------------------------------------------


class Gemma4DecoderLayer(nn.Module):
    """Gemma4 text decoder layer.

    Uses standard ``RMSNorm`` (not ``OffsetRMSNorm`` as in Gemma3) and a
    4-norm structure.

    The ``config`` argument controls whether this is a local (sliding_attention)
    or global (full_attention) layer via ``head_dim``, ``partial_rotary_factor``,
    and ``rope_theta``.  Pass a ``dataclasses.replace``-d config for global layers.

    When ``config.hidden_size_per_layer_input > 0``, expects a
    ``per_layer_input [B, S, per_layer_dim]`` tensor and applies gated projection
    after the MLP block (matching HF ``Gemma4TextDecoderLayer``).
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.self_attn = Attention(config, rms_norm_class=RMSNorm)
        self.mlp = MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self._per_layer_dim = getattr(config, "hidden_size_per_layer_input", 0)
        if self._per_layer_dim:
            self.per_layer_input_gate = Linear(
                config.hidden_size, self._per_layer_dim, bias=False
            )
            self.per_layer_projection = Linear(
                self._per_layer_dim, config.hidden_size, bias=False
            )
            self.post_per_layer_input_norm = RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )
            # gelu_pytorch_tanh matches HF hidden_activation for Gemma4
            from mobius.components._activations import gelu_tanh
            self._act_fn = gelu_tanh

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        attention_bias: ir.Value,
        position_embeddings: tuple,
        past_key_value: tuple | None,
        per_layer_input: ir.Value | None = None,
    ) -> tuple[ir.Value, tuple]:
        # Attention block
        residual = hidden_states
        hidden_states = self.input_layernorm(op, hidden_states)
        attn_output, present_key_value = self.self_attn(
            op,
            hidden_states=hidden_states,
            attention_bias=attention_bias,
            position_embeddings=position_embeddings,
            past_key_value=past_key_value,
        )
        hidden_states = self.post_attention_layernorm(op, attn_output)
        hidden_states = op.Add(residual, hidden_states)

        # MLP block
        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(op, hidden_states)
        hidden_states = self.mlp(op, hidden_states)
        hidden_states = self.post_feedforward_layernorm(op, hidden_states)
        hidden_states = op.Add(residual, hidden_states)

        # Optional per-layer input gating (Gemma4-specific)
        if self._per_layer_dim and per_layer_input is not None:
            residual = hidden_states
            gated = self.per_layer_input_gate(op, hidden_states)
            gated = self._act_fn(op, gated)
            gated = op.Mul(gated, per_layer_input)
            gated = self.per_layer_projection(op, gated)
            gated = self.post_per_layer_input_norm(op, gated)
            hidden_states = op.Add(residual, gated)

        return hidden_states, present_key_value


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

        embed_scale = float(np.float16(config.hidden_size**0.5))
        self.embed_tokens = Gemma3TextScaledWordEmbedding(
            config.vocab_size,
            config.hidden_size,
            config.pad_token_id,
            embed_scale=embed_scale,
        )

        layer_types = config.layer_types or ["sliding_attention"] * config.num_hidden_layers
        self.layer_types = layer_types
        self.sliding_window = config.sliding_window

        # Local (sliding window) config — full rotation, local rope_theta
        local_config = dataclasses.replace(
            config,
            rope_type="default",
            rope_scaling=None,
            partial_rotary_factor=1.0,
        )
        # Global (full attention) config — larger head_dim, different RoPE
        global_head_dim = config.global_head_dim or config.head_dim
        global_config = dataclasses.replace(
            config,
            head_dim=global_head_dim,
            rope_theta=config.global_rope_theta,
            partial_rotary_factor=config.global_partial_rotary_factor,
            rope_type="default",
            rope_scaling=None,
            sliding_window=None,
        )

        self.layers = nn.ModuleList(
            [
                Gemma4DecoderLayer(
                    local_config if lt == "sliding_attention" else global_config
                )
                for lt in layer_types
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.rotary_emb_local = initialize_rope(local_config)
        self.rotary_emb_global = initialize_rope(global_config)

        # Per-layer input embeddings (optional feature)
        self._per_layer_dim = getattr(config, "hidden_size_per_layer_input", 0)
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

        # Project hidden states to per-layer space and normalise
        proj = self.per_layer_model_projection(op, inputs_embeds)
        # Scale by 1 / sqrt(num_layers) to stabilise training
        scale = float(self._num_layers**-0.5) * float(self._per_layer_dim**0.5)
        proj = op.Mul(proj, op.Constant(value_float=scale))
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

        return proj

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

        present_key_values = []
        past_kvs = past_key_values or [None] * len(self.layers)
        for i, (layer, layer_type, past_kv) in enumerate(
            zip(self.layers, self.layer_types, past_kvs)
        ):
            per_layer_input = (
                op.Gather(per_layer_inputs, op.Constant(value_int=i), axis=2)
                if per_layer_inputs is not None
                else None
            )
            hidden_states, present_kv = layer(
                op,
                hidden_states=hidden_states,
                attention_bias=attention_bias_dict[layer_type],
                position_embeddings=position_embeddings_dict[layer_type],
                past_key_value=past_kv,
                per_layer_input=per_layer_input,
            )
            present_key_values.append(present_kv)

        hidden_states = self.norm(op, hidden_states)
        return hidden_states, present_key_values


# ---------------------------------------------------------------------------
# Gemma4CausalLMModel (text-only)
# ---------------------------------------------------------------------------


class Gemma4CausalLMModel(CausalLMModel):
    """Gemma 4 text-only causal language model.

    Registered as ``gemma4_text`` in the model registry.  Uses hybrid
    local/global attention, standard ``RMSNorm``, and optional per-layer
    input embeddings.
    """

    config_class: type = Gemma4Config
    category: str = "Text"

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
        return super().preprocess_weights(state_dict)


# ---------------------------------------------------------------------------
# Gemma4 multimodal sub-models
# ---------------------------------------------------------------------------


class _Gemma4DecoderModel(nn.Module):
    """Gemma4 text decoder sub-model accepting ``inputs_embeds``."""

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
        past_key_values: list | None = None,
    ) -> tuple[ir.Value, list]:
        hidden_states, present_key_values = self.model(
            op,
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
        )
        logits = self.lm_head(op, hidden_states)
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
                new_key = "encoder." + key[len("vision_tower."):]
                # Flatten Gemma4ClippableLinear's .linear. wrapper
                new_key = new_key.replace(".linear.weight", ".weight")
                new_key = new_key.replace(".linear.bias", ".bias")
                renamed[new_key] = value
            elif key.startswith("embed_vision.embedding_projection."):
                suffix = key[len("embed_vision.embedding_projection."):]
                renamed["projector." + suffix] = value
            elif key.startswith("embed_vision.embedding_pre_projection_norm."):
                pass  # Scale-free norm: no learnable parameter
        return renamed


class _Gemma4EmbeddingModel(nn.Module):
    """Gemma4 embedding sub-model: scaled token lookup + image feature fusion.

    Scatters vision features into text embeddings at image-token positions.
    """

    def __init__(self, config: Gemma4Config):
        super().__init__()
        self.config = config
        embed_scale = float(np.float16(config.hidden_size**0.5))
        self.embed_tokens = Gemma3TextScaledWordEmbedding(
            config.vocab_size,
            config.hidden_size,
            config.pad_token_id,
            embed_scale=embed_scale,
        )
        self.image_token_id = config.image_token_id or 0

    def forward(
        self,
        op: builder.OpBuilder,
        input_ids: ir.Value,
        image_features: ir.Value,
    ) -> ir.Value:
        text_embeds = self.embed_tokens(op, input_ids)  # [B, S, hidden]

        image_mask = op.Equal(input_ids, op.Constant(value_int=self.image_token_id))
        image_mask_3d = op.Unsqueeze(image_mask, [-1])

        # Map each image-token position to its corresponding vision feature row
        mask_int = op.Cast(image_mask, to=7)  # INT64
        cumsum = op.CumSum(mask_int, op.Constant(value_int=1))
        indices = op.Sub(cumsum, op.Constant(value_int=1))
        indices = op.Clip(indices, op.Constant(value_int=0))

        gathered = op.Gather(image_features, indices, axis=0)
        return op.Where(image_mask_3d, gathered, text_embeds)

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        return vlm_embedding_weights(state_dict)


# ---------------------------------------------------------------------------
# Gemma4MultiModalModel (3-model split, Image-Text-to-Text)
# ---------------------------------------------------------------------------


class Gemma4MultiModalModel(nn.Module):
    """Gemma 4 vision-language model (3-model split).

    Builds three separate ONNX models:
    - ``decoder``: Gemma4 text decoder taking ``inputs_embeds``
    - ``vision_encoder``: SigLIP-style encoder + projector
    - ``embedding``: scaled word embedding + image feature fusion

    Used for the Image-Text-to-Text variants (26B-A4B, 31B).
    Registered as ``gemma4`` with task ``gemma4-multimodal``.
    """

    default_task: str = "gemma4-multimodal"
    category: str = "Multimodal"

    def __init__(self, config: Gemma4Config):
        super().__init__()
        self.config = config
        self.decoder = _Gemma4DecoderModel(config)
        self.vision_encoder = _Gemma4VisionEncoderModel(config)
        self.embedding = _Gemma4EmbeddingModel(config)

    def forward(self, op: builder.OpBuilder, **kwargs):
        raise NotImplementedError(
            "Gemma4MultiModalModel uses Gemma4VisionLanguageTask which calls "
            "each sub-module (decoder, vision_encoder, embedding) separately."
        )

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Rename HuggingFace weight keys to ONNX initializer names.

        Mapping:
        - ``language_model.*`` -> ``decoder.*``
        - ``vision_tower.*`` -> ``vision_encoder.encoder.*``
        - ``embed_vision.embedding_projection.*`` -> ``vision_encoder.projector.*``
        - ``embed_vision.embedding_pre_projection_norm.*`` -> skip (scale-free)
        - ``language_model.model.embed_tokens.weight`` also goes to
          ``embedding.embed_tokens.weight``
        - ``.linear.`` infix from ``Gemma4ClippableLinear`` is stripped
        """
        if self.config.tie_word_embeddings:
            embed_key = "language_model.model.embed_tokens.weight"
            head_key = "language_model.lm_head.weight"
            if head_key not in state_dict and embed_key in state_dict:
                state_dict[head_key] = state_dict[embed_key]

        renamed: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            if key.startswith("language_model."):
                new_key = "decoder." + key[len("language_model."):]
                renamed[new_key] = value
                if key == "language_model.model.embed_tokens.weight":
                    renamed["embedding.embed_tokens.weight"] = value

            elif key.startswith("vision_tower."):
                new_key = "vision_encoder.encoder." + key[len("vision_tower."):]
                new_key = new_key.replace(".linear.weight", ".weight")
                new_key = new_key.replace(".linear.bias", ".bias")
                renamed[new_key] = value

            elif key.startswith("embed_vision.embedding_projection."):
                suffix = key[len("embed_vision.embedding_projection."):]
                renamed["vision_encoder.projector." + suffix] = value

            elif key.startswith("embed_vision.embedding_pre_projection_norm."):
                pass  # Scale-free norm: no learnable parameter

            else:
                renamed[key] = value

        return renamed
