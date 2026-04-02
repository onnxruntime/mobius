# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Conditional DETR object detection model.

Conditional DETR extends DETR with learned spatial queries (reference points)
and a conditional cross-attention mechanism that separates content and
positional information.  This yields faster convergence and better performance
than vanilla DETR without changing the encoder.

Architecture differences from DETR
-----------------------------------
- **Learned reference points** — An MLP (``ref_point_head``) maps the query
  embeddings to 2-D reference points that are converted to sine positional
  embeddings at runtime.
- **Decoder self-attention** — Separate content and position linear
  projections are summed (``q_content + q_pos``, ``k_content + k_pos``)
  instead of adding position to the *input* before a single projection.
- **Decoder cross-attention** — Content and position projections are
  **concatenated**, doubling the Q/K head dimension.  This requires manual
  attention (``MatMul + Softmax + MatMul``) since the standard ONNX
  ``Attention`` op assumes Q/K/V share the same head dimension.
- **Query scale** — An MLP (``query_scale``) produces per-query scaling
  factors applied to the sine position embeddings from layer 2 onward.

Backbone (ResNet-50) and encoder are identical to DETR.

Architecture::

    pixel_values (B, 3, H, W)
      → ResNet-50 backbone (no GAP)        → (B, 2048, H/32, W/32)
      → 1x1 Conv input projection           → (B, d_model, H/32, W/32)
      → Flatten + sine position encoding    → (B, H'*W', d_model)
      → Transformer encoder (6 layers)      → memory (B, H'*W', d_model)
      → Conditional decoder (6 layers)      → (B, num_queries, d_model)
              ← learned reference points + sine pos embeddings
      → class head (Linear)                 → (B, num_queries, num_labels+1)
      → bbox head (3-layer MLP + sigmoid)   → (B, num_queries, 4)

Replicates HuggingFace ``ConditionalDetrForObjectDetection``.
"""

from __future__ import annotations

import math

import numpy as np
import onnx_ir as ir
import torch
from onnxscript import nn
from onnxscript._internal import builder

from mobius._configs import DetrConfig
from mobius.components import Conv2d, LayerNorm, Linear
from mobius.models.detr import (
    _compute_sine_pos_embed,
    _DetrEncoder,
    _DetrFFN,
    _DetrResNetBackbone,
    _MLPPredictionHead,
    _rename_detr_weight,
)

# ---------------------------------------------------------------------------
# Runtime sine position embeddings from 2-D reference points
# ---------------------------------------------------------------------------


def _gen_sine_position_embeddings(
    op: builder.OpBuilder,
    pos_xy: ir.Value,
    d_model: int,
    temperature: float = 10000.0,
) -> ir.Value:
    """Generate sine position embeddings from 2-D reference points.

    Replicates HuggingFace ``gen_sine_position_embeddings``.  Produces
    interleaved sin/cos pairs for each coordinate, concatenated y-then-x.

    Args:
        pos_xy: ``(B, num_queries, 2)`` — x/y coordinates in ``[0, 1]``.
        d_model: Embedding dimension (split equally between y and x halves).

    Returns:
        ``(B, num_queries, d_model)`` sine position embeddings.
    """
    scale = 2.0 * math.pi
    num_unique = d_model // 4  # unique frequency count per coordinate

    # Precompute unique frequency divisors: temperature^(i / num_unique)
    unique_freqs = temperature ** (np.arange(num_unique, dtype=np.float32) / num_unique)
    # Shape: (1, 1, num_unique) for broadcast with (B, nq, 1)
    freq_const = op.Constant(value=ir.tensor(unique_freqs.reshape(1, 1, num_unique)))

    # Extract x and y coordinates: (B, nq, 1) each
    x = op.Slice(
        pos_xy,
        op.Constant(value_ints=[0]),
        op.Constant(value_ints=[1]),
        op.Constant(value_ints=[2]),
    )  # (B, nq, 1)
    y = op.Slice(
        pos_xy,
        op.Constant(value_ints=[1]),
        op.Constant(value_ints=[2]),
        op.Constant(value_ints=[2]),
    )  # (B, nq, 1)

    # Scale by 2π: (B, nq, 1)
    scale_const = op.Constant(value=ir.tensor(np.float32(scale)))
    x = op.Mul(x, scale_const)
    y = op.Mul(y, scale_const)

    # Angles: (B, nq, 1) / (1, 1, num_unique) → (B, nq, num_unique)
    angles_x = op.Div(x, freq_const)
    angles_y = op.Div(y, freq_const)

    # Sine and cosine: (B, nq, num_unique) each
    sin_x = op.Sin(angles_x)
    cos_x = op.Cos(angles_x)
    sin_y = op.Sin(angles_y)
    cos_y = op.Cos(angles_y)

    # Interleave sin/cos pairs: [sin(θ₀), cos(θ₀), sin(θ₁), cos(θ₁), ...]
    # Unsqueeze to (B, nq, num_unique, 1), concat → (B, nq, num_unique, 2)
    pairs_x = op.Concat(
        op.Unsqueeze(sin_x, [-1]),
        op.Unsqueeze(cos_x, [-1]),
        axis=-1,
    )  # (B, nq, num_unique, 2)
    pairs_y = op.Concat(
        op.Unsqueeze(sin_y, [-1]),
        op.Unsqueeze(cos_y, [-1]),
        axis=-1,
    )  # (B, nq, num_unique, 2)

    # Flatten last two dims: (B, nq, num_unique*2) = (B, nq, d_model/2)
    batch_nq = op.Shape(pos_xy, start=0, end=2)  # [B, nq]
    half_d = d_model // 2
    flat_shape = op.Concat(batch_nq, op.Constant(value_ints=[half_d]), axis=0)
    embed_x = op.Reshape(pairs_x, flat_shape)  # (B, nq, d_model/2)
    embed_y = op.Reshape(pairs_y, flat_shape)  # (B, nq, d_model/2)

    # Concat y then x (matches HuggingFace convention): (B, nq, d_model)
    return op.Concat(embed_y, embed_x, axis=-1)


# ---------------------------------------------------------------------------
# Decoder self-attention: separate content and position projections
# ---------------------------------------------------------------------------


class _ConditionalDetrDecoderSelfAttention(nn.Module):
    """Conditional DETR decoder self-attention.

    Separates content and position into independent projections whose
    outputs are summed to form Q and K.  V uses only the content (hidden
    states).  The output projection is ``o_proj`` (matching HF naming).

    Q = q_content_proj(hidden) + q_pos_proj(query_pos)
    K = k_content_proj(hidden) + k_pos_proj(query_pos)
    V = v_proj(hidden)
    """

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q_content_proj = Linear(d_model, d_model, bias=True)
        self.q_pos_proj = Linear(d_model, d_model, bias=True)
        self.k_content_proj = Linear(d_model, d_model, bias=True)
        self.k_pos_proj = Linear(d_model, d_model, bias=True)
        self.v_proj = Linear(d_model, d_model, bias=True)
        self.o_proj = Linear(d_model, d_model, bias=True)

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        query_pos: ir.Value,
    ) -> ir.Value:
        # Separate content + position projections, then sum
        query = op.Add(
            self.q_content_proj(op, hidden_states),
            self.q_pos_proj(op, query_pos),
        )  # (B, nq, d_model)
        key = op.Add(
            self.k_content_proj(op, hidden_states),
            self.k_pos_proj(op, query_pos),
        )  # (B, nq, d_model)
        value = self.v_proj(op, hidden_states)  # (B, nq, d_model)

        # Standard multi-head attention (Q/K/V all have d_model dim)
        attn_output = op.Attention(
            query,
            key,
            value,
            q_num_heads=self.num_heads,
            kv_num_heads=self.num_heads,
            scale=float(self.head_dim**-0.5),
        )
        return self.o_proj(op, attn_output)


# ---------------------------------------------------------------------------
# Decoder cross-attention: content-position concatenation (doubled Q/K dim)
# ---------------------------------------------------------------------------


class _ConditionalDetrDecoderCrossAttention(nn.Module):
    """Conditional DETR decoder cross-attention.

    Content and position projections are **concatenated** along the head
    dimension, producing Q and K with 2x the normal head dimension while V
    keeps the standard dimension.  This requires manual attention
    (``MatMul + Softmax + MatMul``) since ``op.Attention`` assumes Q/K/V
    share the same head dimension.

    First layer only::

        Q = cat(q_content(h) + q_pos(query_pos), q_sine(sine_embed))
        K = cat(k_content(m) + k_pos(spatial_pos), k_pos(spatial_pos))

    Subsequent layers::

        Q = cat(q_content(h), q_sine(sine_embed))
        K = cat(k_content(m), k_pos(spatial_pos))

    V = v_proj(encoder_hidden_states) — always standard dimension.
    """

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.d_model = d_model
        self.q_content_proj = Linear(d_model, d_model, bias=True)
        self.q_pos_proj = Linear(d_model, d_model, bias=True)
        self.q_pos_sine_proj = Linear(d_model, d_model, bias=True)
        self.k_content_proj = Linear(d_model, d_model, bias=True)
        self.k_pos_proj = Linear(d_model, d_model, bias=True)
        self.v_proj = Linear(d_model, d_model, bias=True)
        self.o_proj = Linear(d_model, d_model, bias=True)

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        encoder_hidden_states: ir.Value,
        query_sine_embed: ir.Value,
        spatial_pos: ir.Value,
        *,
        is_first: bool = False,
        query_pos: ir.Value | None = None,
    ) -> ir.Value:
        d = self.d_model
        n_heads = self.num_heads
        head_dim = self.head_dim

        # Content projections
        q_content = self.q_content_proj(op, hidden_states)
        k_content = self.k_content_proj(op, encoder_hidden_states)
        value = self.v_proj(op, encoder_hidden_states)  # (B, S_k, d)
        k_pos = self.k_pos_proj(op, spatial_pos)  # (B, S_k, d)

        # First layer: add position embeddings to content before concat
        if is_first and query_pos is not None:
            q_content = op.Add(q_content, self.q_pos_proj(op, query_pos))
            k_content = op.Add(k_content, k_pos)

        # Sine embedding projection
        q_sine = self.q_pos_sine_proj(op, query_sine_embed)  # (B, nq, d)

        # Concatenate content and position: (B, S, 2*d)
        q = op.Concat(q_content, q_sine, axis=-1)  # (B, nq, 2*d)
        k = op.Concat(k_content, k_pos, axis=-1)  # (B, S_k, 2*d)

        # --- Manual multi-head attention (Q/K: 2*hd, V: hd) ---
        doubled_head_dim = 2 * head_dim
        scale_factor = float(doubled_head_dim**-0.5)

        # Extract dynamic batch dim
        batch = op.Shape(q, start=0, end=1)  # [B]
        seq_q = op.Shape(q, start=1, end=2)  # [nq]
        seq_k = op.Shape(k, start=1, end=2)  # [S_k]

        # Reshape to multi-head: (B, S, D) → (B, n_heads, S, hd)
        q_shape = op.Concat(
            batch,
            seq_q,
            op.Constant(value_ints=[n_heads, doubled_head_dim]),
            axis=0,
        )
        k_shape = op.Concat(
            batch,
            seq_k,
            op.Constant(value_ints=[n_heads, doubled_head_dim]),
            axis=0,
        )
        v_shape = op.Concat(
            batch,
            seq_k,
            op.Constant(value_ints=[n_heads, head_dim]),
            axis=0,
        )

        q = op.Transpose(op.Reshape(q, q_shape), perm=[0, 2, 1, 3])
        k = op.Transpose(op.Reshape(k, k_shape), perm=[0, 2, 1, 3])
        v = op.Transpose(op.Reshape(value, v_shape), perm=[0, 2, 1, 3])

        # Attention scores: Q @ K^T / sqrt(2 * head_dim)
        k_t = op.Transpose(k, perm=[0, 1, 3, 2])
        scores = op.MatMul(q, k_t)  # (B, n_heads, nq, S_k)
        scores = op.Mul(
            scores,
            op.Constant(value=ir.tensor(np.float32(scale_factor))),
        )
        attn_weights = op.Softmax(scores, axis=-1)

        # Weighted sum of values: (B, n_heads, nq, head_dim)
        attn_out = op.MatMul(attn_weights, v)

        # Reshape back: (B, n_heads, nq, hd) → (B, nq, d)
        attn_out = op.Transpose(attn_out, perm=[0, 2, 1, 3])
        out_shape = op.Concat(batch, seq_q, op.Constant(value_ints=[d]), axis=0)
        attn_out = op.Reshape(attn_out, out_shape)

        return self.o_proj(op, attn_out)


# ---------------------------------------------------------------------------
# Decoder layer
# ---------------------------------------------------------------------------


class _ConditionalDetrDecoderLayer(nn.Module):
    """Conditional DETR decoder layer.

    Structure (post-norm)::

        hidden + self_attn(hidden, query_pos)           → LayerNorm
        + cross_attn(hidden, memory, sine_embed, pos)   → LayerNorm
        + mlp(hidden)                                    → LayerNorm
    """

    def __init__(self, config: DetrConfig):
        super().__init__()
        d = config.d_model
        self.self_attn = _ConditionalDetrDecoderSelfAttention(
            d, config.decoder_attention_heads
        )
        self.self_attn_layer_norm = LayerNorm(d, eps=1e-5)
        self.encoder_attn = _ConditionalDetrDecoderCrossAttention(
            d, config.decoder_attention_heads
        )
        self.encoder_attn_layer_norm = LayerNorm(d, eps=1e-5)
        self.mlp = _DetrFFN(d, config.decoder_ffn_dim)
        self.final_layer_norm = LayerNorm(d, eps=1e-5)

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        encoder_hidden_states: ir.Value,
        query_pos: ir.Value,
        spatial_pos: ir.Value,
        query_sine_embed: ir.Value,
        *,
        is_first: bool = False,
    ) -> ir.Value:
        # Self-attention: object queries attend to each other
        residual = hidden_states
        attn_out = self.self_attn(op, hidden_states, query_pos)
        hidden_states = self.self_attn_layer_norm(op, op.Add(residual, attn_out))

        # Cross-attention: queries attend to encoder memory
        residual = hidden_states
        attn_out = self.encoder_attn(
            op,
            hidden_states,
            encoder_hidden_states,
            query_sine_embed,
            spatial_pos,
            is_first=is_first,
            query_pos=query_pos if is_first else None,
        )
        hidden_states = self.encoder_attn_layer_norm(op, op.Add(residual, attn_out))

        # FFN
        residual = hidden_states
        mlp_out = self.mlp(op, hidden_states)
        hidden_states = self.final_layer_norm(op, op.Add(residual, mlp_out))

        return hidden_states


# ---------------------------------------------------------------------------
# Decoder stack with reference points, sine embeddings, and query scale
# ---------------------------------------------------------------------------


class _ConditionalDetrDecoder(nn.Module):
    """Conditional DETR decoder.

    On top of the standard decoder layer stack this adds:

    - ``ref_point_head``: 2-layer MLP mapping query embeddings to 2-D
      reference points (sigmoid to [0, 1]).
    - ``query_scale``: 2-layer MLP that produces per-query scaling factors
      for the sine position embeddings (from layer 2 onward).
    - Runtime sine position embedding generation from reference points.
    """

    def __init__(self, config: DetrConfig):
        super().__init__()
        d = config.d_model

        self.layers = nn.ModuleList(
            [_ConditionalDetrDecoderLayer(config) for _ in range(config.decoder_layers)]
        )
        self.layernorm = LayerNorm(d, eps=1e-5)

        # Reference point prediction: query_embed → 2-D point in [0, 1]
        self.ref_point_head = _MLPPredictionHead(d, d, output_dim=2, num_layers=2)
        # Per-query position scale factor (from layer 2 onward)
        self.query_scale = _MLPPredictionHead(d, d, output_dim=1, num_layers=2)
        self.d_model = d

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        encoder_hidden_states: ir.Value,
        query_pos: ir.Value,
        spatial_pos: ir.Value,
    ) -> ir.Value:
        d = self.d_model

        # Compute 2-D reference points from query embeddings
        # ref_point_head: (B, nq, d) → (B, nq, 2) → sigmoid to [0, 1]
        reference_points = op.Sigmoid(self.ref_point_head(op, query_pos))  # (B, nq, 2)

        # Generate sine position embeddings from reference points
        sine_embed = _gen_sine_position_embeddings(op, reference_points, d)  # (B, nq, d)

        for idx, layer in enumerate(self.layers):
            if idx == 0:
                # First layer: no position scale transformation
                query_sine_embed = sine_embed
            else:
                # Subsequent layers: scale sine embed by query factor
                pos_scale = self.query_scale(op, hidden_states)  # (B, nq, 1)
                query_sine_embed = op.Mul(sine_embed, pos_scale)

            hidden_states = layer(
                op,
                hidden_states,
                encoder_hidden_states,
                query_pos,
                spatial_pos,
                query_sine_embed,
                is_first=(idx == 0),
            )

        return self.layernorm(op, hidden_states)


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------


class ConditionalDetrForObjectDetection(nn.Module):
    """Conditional DETR for object detection.

    Same backbone and encoder as standard DETR with a conditional decoder
    that uses learned reference points and separate content/position
    attention projections.  Default ``num_queries`` is 300 (vs 100 for DETR).

    Inputs:
        ``pixel_values``: ``(B, C, H, W)`` — pre-processed image pixels.

    Outputs:
        ``logits``: ``(B, num_queries, num_labels + 1)`` — class scores.
        ``pred_boxes``: ``(B, num_queries, 4)`` — sigmoid (cx, cy, w, h).

    Replicates HuggingFace ``ConditionalDetrForObjectDetection``.
    """

    default_task: str = "object-detection"
    category: str = "Object Detection"
    config_class: type = DetrConfig

    def __init__(self, config: DetrConfig):
        super().__init__()
        self.config = config
        d = config.d_model
        num_queries = config.num_queries
        backbone_out_channels = config.backbone_hidden_sizes[-1]

        # Feature map size: ResNet-50 stride-32
        image_size = getattr(config, "image_size", 800)
        feat_size = image_size // 32

        # ResNet-50 backbone (identical to DETR)
        self.backbone = _DetrResNetBackbone(config)

        # 1x1 projection from backbone channels to transformer d_model
        self.input_projection = Conv2d(backbone_out_channels, d, kernel_size=1)

        # Fixed sinusoidal 2-D spatial position encoding: (1, H'*W', d)
        pos_data = _compute_sine_pos_embed(feat_size, feat_size, d)
        self.spatial_pos_embed = nn.Parameter(
            list(pos_data.shape),
            data=ir.tensor(pos_data),
        )

        # Learned object query position embeddings: (num_queries, d)
        self.query_position_embeddings = nn.Parameter(
            [num_queries, d],
            data=ir.tensor(np.zeros((num_queries, d), dtype=np.float32)),
        )

        # Encoder (same as DETR — weight renaming handles o_proj→out_proj)
        self.encoder = _DetrEncoder(config)

        # Conditional decoder (different from DETR)
        self.decoder = _ConditionalDetrDecoder(config)

        # Detection heads (same as DETR)
        self.class_labels_classifier = Linear(d, config.num_labels + 1, bias=True)
        self.bbox_predictor = _MLPPredictionHead(d, d, 4, num_layers=3)

    def forward(
        self,
        op: builder.OpBuilder,
        pixel_values: ir.Value,
    ) -> tuple[ir.Value, ir.Value]:
        d = self.config.d_model
        num_queries = self.config.num_queries

        # ResNet backbone: (B, 3, H, W) → (B, 2048, H/32, W/32)
        features = self.backbone(op, pixel_values)

        # Input projection: → (B, d, H', W')
        features = self.input_projection(op, features)

        # Flatten spatial dims: (B, d, H', W') → (B, H'*W', d)
        batch = op.Shape(features, start=0, end=1)
        features = op.Transpose(features, perm=[0, 2, 3, 1])
        features = op.Reshape(
            features,
            op.Concat(batch, op.Constant(value_ints=[-1, d]), axis=0),
        )

        # Spatial position encoding: (1, H'*W', d)
        pos_embed = self.spatial_pos_embed

        # Encoder: (B, H'*W', d) → memory (B, H'*W', d)
        memory = self.encoder(op, features, pos_embed)

        # Query position embeddings: (nq, d) → (1, nq, d)
        query_pos = op.Unsqueeze(self.query_position_embeddings, [0])

        # Initial decoder target: zeros (B, nq, d)
        target = op.ConstantOfShape(
            op.Concat(
                batch,
                op.Constant(value_ints=[num_queries, d]),
                axis=0,
            ),
            value=ir.tensor(np.zeros(1, dtype=np.float32)),
        )

        # Conditional decoder: cross-attends to memory with ref points
        hs = self.decoder(op, target, memory, query_pos, pos_embed)

        # Class prediction: (B, nq, num_labels + 1)
        logits = self.class_labels_classifier(op, hs)

        # Box prediction: (B, nq, 4) in [0, 1] after sigmoid
        pred_boxes = op.Sigmoid(self.bbox_predictor(op, hs))

        return logits, pred_boxes

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Map HuggingFace ConditionalDETR weight names to ours.

        Handles:
        - Stripping the ``model.`` prefix.
        - Backbone renaming (via shared ``_rename_detr_weight``).
        - Encoder ``self_attn.o_proj`` → ``self_attn.out_proj`` (to reuse
          the DETR encoder which uses ``out_proj``).
        - ``query_position_embeddings.weight`` → bare parameter.
        - Decoder, detection heads: pass through unchanged.
        """
        new: dict[str, torch.Tensor] = {}
        for name, value in state_dict.items():
            # Strip top-level model. wrapper
            if name.startswith("model."):
                name = name[len("model.") :]

            # Backbone: delegate to shared DETR renaming
            if name.startswith("backbone."):
                mapped = _rename_detr_weight(name, value)
                if mapped is not None:
                    new[mapped[0]] = mapped[1]
                continue

            # Encoder: rename o_proj → out_proj (reusing DETR encoder)
            if name.startswith("encoder."):
                name = name.replace(".self_attn.o_proj.", ".self_attn.out_proj.")
                new[name] = value
                continue

            # Query position embeddings: nn.Embedding → bare Parameter
            if name == "query_position_embeddings.weight":
                new["query_position_embeddings"] = value
                continue

            # Decoder, input_projection, detection heads: pass through
            if name.startswith(
                (
                    "input_projection.",
                    "decoder.",
                    "class_labels_classifier.",
                    "bbox_predictor.",
                )
            ):
                new[name] = value
                continue

            # Drop unrecognised keys
        return new
