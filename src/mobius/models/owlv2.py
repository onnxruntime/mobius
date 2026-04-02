# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""OWLv2 (Open-World Localization v2) open-vocabulary object detection.

Implements ``google/owlv2-base-patch16-ensemble`` and compatible variants.
Combines a CLIP vision encoder + text encoder with detection heads for
class prediction, bounding box regression, and objectness scoring.

HuggingFace reference: ``Owlv2ForObjectDetection``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from onnxscript import nn
from onnxscript._internal import builder

from mobius._configs import ArchitectureConfig
from mobius.components._activations import get_activation
from mobius.components._common import LayerNorm, Linear
from mobius.models.clip import (
    _CLIPTextEmbeddings,
    _CLIPTextEncoderLayer,
    _CLIPVisionEmbeddings,
    _CLIPVisionEncoderLayer,
)

if TYPE_CHECKING:
    import onnx_ir as ir


# ---------------------------------------------------------------------------
# Sub-modules
# ---------------------------------------------------------------------------


class _Owlv2VisionEncoder(nn.Module):
    """OWLv2 vision encoder: CLIP ViT with pre/post layernorm.

    Returns the full sequence of hidden states (CLS + patches).
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.embeddings = _CLIPVisionEmbeddings(config)
        # Note: matches the deliberate CLIP typo ``pre_layrnorm``
        self.pre_layrnorm = LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.encoder = nn.ModuleList(
            [_CLIPVisionEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.post_layernorm = LayerNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, op: builder.OpBuilder, pixel_values: ir.Value) -> ir.Value:
        # pixel_values: (B, C, H, W) -> hidden_states: (B, 1+num_patches, D)
        hidden_states = self.embeddings(op, pixel_values)
        hidden_states = self.pre_layrnorm(op, hidden_states)

        for layer in self.encoder:
            hidden_states = layer(op, hidden_states)

        hidden_states = self.post_layernorm(op, hidden_states)
        return hidden_states


class _Owlv2TextEncoder(nn.Module):
    """OWLv2 text encoder: CLIP text transformer with causal masking.

    Accepts flattened ``input_ids`` of shape ``(N, seq_len)`` and returns
    the full last-hidden-state ``(N, seq_len, text_hidden)``.
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.embeddings = _CLIPTextEmbeddings(config)
        self.encoder = nn.ModuleList(
            [_CLIPTextEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.final_layer_norm = LayerNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, op: builder.OpBuilder, input_ids: ir.Value) -> ir.Value:
        # input_ids: (N, seq_len)
        hidden_states = self.embeddings(op, input_ids)

        # Build causal attention bias (lower-triangular)
        seq_len = op.Shape(input_ids, start=1, end=2)
        # Upper-tri filled with -10000, lower-tri + diag = 0
        neg_inf_mask = op.Trilu(
            op.Expand(
                op.Constant(value_float=-10000.0),
                op.Concat(seq_len, seq_len, axis=0),
            ),
            upper=1,
        )
        diag_mask = op.Trilu(neg_inf_mask, upper=0)
        causal_bias = op.Sub(neg_inf_mask, diag_mask)
        # (1, 1, seq, seq) for broadcast over batch & heads
        causal_bias = op.Unsqueeze(causal_bias, [0, 1])

        for layer in self.encoder:
            hidden_states = layer(op, hidden_states, causal_bias)

        hidden_states = self.final_layer_norm(op, hidden_states)
        return hidden_states  # (N, seq_len, text_hidden)


class _Owlv2ClassHead(nn.Module):
    """OWLv2 class prediction head.

    Projects image features to ``projection_dim``, L2-normalises both
    image and query embeddings, computes cosine-similarity logits, then
    applies a learned per-patch shift and scale.
    """

    def __init__(self, vision_hidden: int, projection_dim: int):
        super().__init__()
        self.dense0 = Linear(vision_hidden, projection_dim)
        self.logit_shift = Linear(vision_hidden, 1)
        self.logit_scale = Linear(vision_hidden, 1)

    def forward(
        self,
        op: builder.OpBuilder,
        image_feats: ir.Value,
        query_embeds: ir.Value,
    ) -> ir.Value:
        # image_feats: (B, P, vision_hidden)
        # query_embeds: (B, Q, projection_dim)

        # Project image features → (B, P, projection_dim)
        image_class_embeds = self.dense0(op, image_feats)

        # L2-normalise image class embeddings
        eps = op.Constant(value_float=1e-6)
        img_norm = op.ReduceL2(image_class_embeds, [-1], keepdims=True)
        image_class_embeds = op.Div(image_class_embeds, op.Add(img_norm, eps))

        # L2-normalise query embeddings
        q_norm = op.ReduceL2(query_embeds, [-1], keepdims=True)
        query_embeds_norm = op.Div(query_embeds, op.Add(q_norm, eps))

        # Cosine similarity: (B, P, D) @ (B, D, Q) → (B, P, Q)
        pred_logits = op.MatMul(
            image_class_embeds,
            op.Transpose(query_embeds_norm, perm=[0, 2, 1]),
        )

        # Learned per-patch shift and scale from raw image features
        logit_shift = self.logit_shift(op, image_feats)  # (B, P, 1)
        logit_scale = self.logit_scale(op, image_feats)  # (B, P, 1)
        # scale = ELU(logit_scale) + 1  (ensures positive)
        logit_scale = op.Add(op.Elu(logit_scale, alpha=1.0), op.Constant(value_float=1.0))
        pred_logits = op.Mul(op.Add(pred_logits, logit_shift), logit_scale)
        return pred_logits  # (B, P, Q)


class _Owlv2BoxHead(nn.Module):
    """OWLv2 bounding-box regression head.

    Three-layer MLP: dense0 → GELU → dense1 → GELU → dense2(→ 4).
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.dense0 = Linear(hidden_size, hidden_size)
        self.dense1 = Linear(hidden_size, hidden_size)
        self.dense2 = Linear(hidden_size, 4)
        self._act = get_activation("gelu")

    def forward(self, op: builder.OpBuilder, image_feats: ir.Value) -> ir.Value:
        x = self._act(op, self.dense0(op, image_feats))
        x = self._act(op, self.dense1(op, x))
        x = self.dense2(op, x)
        return x  # (B, P, 4) — before box_bias + sigmoid


class _Owlv2ObjectnessHead(nn.Module):
    """OWLv2 objectness prediction head.

    Three-layer MLP identical to box head but outputs a scalar per patch.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.dense0 = Linear(hidden_size, hidden_size)
        self.dense1 = Linear(hidden_size, hidden_size)
        self.dense2 = Linear(hidden_size, 1)
        self._act = get_activation("gelu")

    def forward(self, op: builder.OpBuilder, image_feats: ir.Value) -> ir.Value:
        x = self._act(op, self.dense0(op, image_feats))
        x = self._act(op, self.dense1(op, x))
        x = self.dense2(op, x)
        # Squeeze trailing dim → (B, P)
        x = op.Squeeze(x, [-1])
        return x


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------


class Owlv2ForObjectDetection(nn.Module):
    """OWLv2 open-vocabulary object detection model.

    Combines a CLIP-style dual encoder (vision + text) with three
    detection heads: class prediction, bounding-box regression, and
    objectness scoring.

    Inputs:
        pixel_values : (B, C, H, W) float
        input_ids    : (B, Q, S) int64  — Q text queries of length S

    Outputs:
        logits             : (B, P, Q) float  — class logits per patch
        pred_boxes         : (B, P, 4) float  — normalised box coords
        objectness_logits  : (B, P)    float  — objectness scores
    """

    default_task = "owlv2-object-detection"
    category = "vision"

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        hidden_size = config.hidden_size
        projection_dim = config.projection_dim or hidden_size
        text_hidden = getattr(config, "text_hidden_size", hidden_size)

        # --- Encoders ---
        self.vision_encoder = _Owlv2VisionEncoder(config)

        # Build a lightweight config for the text encoder dimensions
        text_config = ArchitectureConfig(
            hidden_size=text_hidden,
            intermediate_size=getattr(
                config, "text_intermediate_size", config.intermediate_size
            ),
            num_hidden_layers=getattr(
                config, "text_num_hidden_layers", config.num_hidden_layers
            ),
            num_attention_heads=getattr(
                config, "text_num_attention_heads", config.num_attention_heads
            ),
            max_position_embeddings=getattr(config, "text_max_position_embeddings", 16),
            vocab_size=getattr(config, "text_vocab_size", config.vocab_size),
            rms_norm_eps=config.rms_norm_eps,
            hidden_act=config.hidden_act,
        )
        self.text_encoder = _Owlv2TextEncoder(text_config)

        # Text projection: text_hidden → projection_dim (no bias)
        self.text_projection = Linear(text_hidden, projection_dim, bias=False)

        # --- Feature merging layer norm ---
        self.layer_norm = LayerNorm(hidden_size, eps=config.rms_norm_eps)

        # --- Detection heads ---
        self.class_head = _Owlv2ClassHead(hidden_size, projection_dim)
        self.box_head = _Owlv2BoxHead(hidden_size)
        self.objectness_head = _Owlv2ObjectnessHead(hidden_size)

        # Box bias: pre-computed (num_patches, 4) registered as parameter
        num_patches = (config.image_size // config.patch_size) ** 2
        self.box_bias = nn.Parameter((num_patches, 4))

    def forward(
        self,
        op: builder.OpBuilder,
        pixel_values: ir.Value,
        input_ids: ir.Value,
    ) -> tuple[ir.Value, ir.Value, ir.Value]:
        # ---- 1. Vision encoding ----
        # (B, 1+P, D)
        vision_out = self.vision_encoder(op, pixel_values)

        # ---- 2. Merge CLS token with patch tokens ----
        # class_token: (B, 1, D),  patch_tokens: (B, P, D)
        class_token = op.Slice(
            vision_out,
            op.Constant(value_ints=[0]),
            op.Constant(value_ints=[1]),
            op.Constant(value_ints=[1]),
        )
        patch_tokens = op.Slice(
            vision_out,
            op.Constant(value_ints=[1]),
            op.Constant(value_ints=[2147483647]),
            op.Constant(value_ints=[1]),
        )
        # Broadcast CLS across patch positions and multiply
        class_token_bc = op.Expand(class_token, op.Shape(patch_tokens))
        image_feats = op.Mul(patch_tokens, class_token_bc)
        # (B, P, D)
        image_feats = self.layer_norm(op, image_feats)

        # ---- 3. Text encoding ----
        # input_ids: (B, Q, S) → flatten to (B*Q, S)
        batch_dim = op.Shape(input_ids, start=0, end=1)
        num_queries_dim = op.Shape(input_ids, start=1, end=2)
        seq_len_dim = op.Shape(input_ids, start=2, end=3)

        flat_ids = op.Reshape(
            input_ids,
            op.Concat(op.Constant(value_ints=[-1]), seq_len_dim, axis=0),
        )
        # (B*Q, S) → (B*Q, S, text_hidden)
        text_hidden = self.text_encoder(op, flat_ids)

        # ---- 4. Pool at EOS position (argmax of input_ids) ----
        eos_pos = op.ArgMax(flat_ids, axis=-1, keepdims=False)  # (B*Q,)
        text_hidden_dim = op.Shape(text_hidden, start=2, end=3)
        eos_idx = op.Unsqueeze(eos_pos, [1, 2])
        eos_idx = op.Expand(eos_idx, op.Concat([1], [1], text_hidden_dim, axis=0))
        pooled = op.GatherElements(text_hidden, eos_idx, axis=1)
        pooled = op.Squeeze(pooled, [1])  # (B*Q, text_hidden)

        # Project to query space
        query_embeds = self.text_projection(op, pooled)  # (B*Q, proj_dim)

        # Reshape back to (B, Q, proj_dim)
        proj_dim = op.Shape(query_embeds, start=1, end=2)
        query_embeds = op.Reshape(
            query_embeds,
            op.Concat(batch_dim, num_queries_dim, proj_dim, axis=0),
        )

        # ---- 5. Detection heads ----
        logits = self.class_head(op, image_feats, query_embeds)
        pred_boxes = self.box_head(op, image_feats)
        pred_boxes = op.Add(pred_boxes, self.box_bias)
        pred_boxes = op.Sigmoid(pred_boxes)
        objectness_logits = self.objectness_head(op, image_feats)

        return logits, pred_boxes, objectness_logits

    # ------------------------------------------------------------------
    # Weight preprocessing
    # ------------------------------------------------------------------

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        new_sd: dict[str, torch.Tensor] = {}
        for name, tensor in state_dict.items():
            new_name = _rename_owlv2_weight(name)
            if new_name is not None:
                new_sd[new_name] = tensor
        return new_sd


# ---------------------------------------------------------------------------
# Weight name mapping helpers
# ---------------------------------------------------------------------------

# Shared rename table for CLIP-style encoder layers
_CLIP_ATTN_RENAMES = {
    "self_attn.q_proj.": "self_attn.q_proj.",
    "self_attn.k_proj.": "self_attn.k_proj.",
    "self_attn.v_proj.": "self_attn.v_proj.",
    "self_attn.out_proj.": "self_attn.out_proj.",
}


def _rename_encoder_layer(remainder: str, layer_idx: str) -> str | None:
    """Rename ``encoder.layers.<idx>.<remainder>`` to our convention."""
    for old, new in _CLIP_ATTN_RENAMES.items():
        if remainder.startswith(old):
            suffix = remainder[len(old) :]
            return f"encoder.{layer_idx}.{new}{suffix}"

    # MLP: fc1 → up_proj, fc2 → down_proj  (FCMLP naming)
    remainder = remainder.replace("mlp.fc1.", "mlp.up_proj.")
    remainder = remainder.replace("mlp.fc2.", "mlp.down_proj.")
    return f"encoder.{layer_idx}.{remainder}"


def _rename_owlv2_weight(name: str) -> str | None:
    """Map a HuggingFace OWLv2 weight name to our module naming."""
    # ---- Skip unused weights ----
    if name in ("owlv2.logit_scale",):
        return None
    if name.startswith("owlv2.visual_projection."):
        return None

    # ---- Text projection (no-bias Linear) ----
    if name == "owlv2.text_projection.weight":
        return "text_projection.weight"

    # ---- Vision encoder weights ----
    if name.startswith("owlv2.vision_model."):
        inner = name[len("owlv2.vision_model.") :]
        return _rename_vision(inner)

    # ---- Text encoder weights ----
    if name.startswith("owlv2.text_model."):
        inner = name[len("owlv2.text_model.") :]
        return _rename_text(inner)

    # ---- Detection heads + layer_norm + box_bias (no prefix) ----
    if name.startswith(
        (
            "class_head.",
            "box_head.",
            "objectness_head.",
            "layer_norm.",
        )
    ):
        return name
    if name == "box_bias":
        return "box_bias"

    return None


def _rename_vision(name: str) -> str | None:
    """Rename vision encoder weights under ``vision_encoder.``."""
    # Embeddings
    if name.startswith("embeddings."):
        return f"vision_encoder.{name}"

    # Pre/post layer norm (note: CLIP typo ``pre_layrnorm``)
    if name.startswith("pre_layernorm."):
        return f"vision_encoder.pre_layrnorm.{name[len('pre_layernorm.') :]}"
    if name.startswith("post_layernorm."):
        return f"vision_encoder.{name}"

    # Encoder layers
    if name.startswith("encoder.layers."):
        parts = name.split(".", 3)
        if len(parts) < 4:
            return None
        layer_idx, remainder = parts[2], parts[3]
        mapped = _rename_encoder_layer(remainder, layer_idx)
        return f"vision_encoder.{mapped}" if mapped else None

    return None


def _rename_text(name: str) -> str | None:
    """Rename text encoder weights under ``text_encoder.``."""
    # Embeddings — HF uses token_embedding, we use word_embeddings
    if name.startswith("embeddings."):
        inner = name.replace("embeddings.token_embedding.", "embeddings.word_embeddings.")
        return f"text_encoder.{inner}"

    # Final layer norm
    if name.startswith("final_layer_norm."):
        return f"text_encoder.{name}"

    # Encoder layers
    if name.startswith("encoder.layers."):
        parts = name.split(".", 3)
        if len(parts) < 4:
            return None
        layer_idx, remainder = parts[2], parts[3]
        mapped = _rename_encoder_layer(remainder, layer_idx)
        return f"text_encoder.{mapped}" if mapped else None

    return None
