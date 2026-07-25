# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""CLIP vision and text models for feature extraction."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from onnxscript import OpBuilder, nn

from mobius._configs import ArchitectureConfig
from mobius.components import FCMLP
from mobius.components._common import Embedding, LayerNorm
from mobius.components._conv import Conv2d, Conv2dNoBias
from mobius.components._encoder import EncoderAttention

if TYPE_CHECKING:
    import onnx_ir as ir

# Largest int64 value; used as an open-ended ``Slice`` end (ONNX clamps it to
# the actual dimension size).
_INT64_MAX = 9223372036854775807


class ClipVisionConfigView:
    """Adapter exposing a :class:`VisionConfig` under CLIP's field names.

    The CLIP vision modules read the encoder geometry from top-level attributes
    (``hidden_size``, ``num_hidden_layers``, ``num_channels``, ``rms_norm_eps``,
    ``hidden_act`` ...). In a multimodal checkpoint those live inside the nested
    :class:`~mobius._configs._sub_configs.VisionConfig` (e.g. Phi-3.5-Vision's
    ``img_processor`` dict). This view bridges the two so the CLIP tower can be
    reused without duplicating its module definitions.
    """

    def __init__(self, vision_config, *, default_hidden_act: str = "quick_gelu"):
        self._vision_config = vision_config
        self.hidden_size = vision_config.hidden_size
        self.intermediate_size = vision_config.intermediate_size
        self.num_hidden_layers = vision_config.num_hidden_layers
        self.num_attention_heads = vision_config.num_attention_heads
        self.image_size = vision_config.image_size
        self.patch_size = vision_config.patch_size
        self.num_channels = vision_config.in_channels
        self.rms_norm_eps = vision_config.norm_eps
        self.hidden_act = vision_config.hidden_act or default_hidden_act


class _CLIPVisionEmbeddings(nn.Module):
    """CLIP vision embeddings: Conv2d patch + CLS token + position embeddings."""

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        hidden_size = config.hidden_size
        patch_size = config.patch_size
        image_size = config.image_size

        self.class_embedding = nn.Parameter((hidden_size,))
        # CLIP's patch-embedding Conv2d has no bias (HuggingFace uses bias=False).
        self.patch_embedding = _Conv2dPatchEmbed(
            config.num_channels,
            hidden_size,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )
        num_patches = (image_size // patch_size) ** 2
        self.position_embedding = Embedding(num_patches + 1, hidden_size)

    def forward(self, op: OpBuilder, pixel_values: ir.Value):
        patch_embeds = self.patch_embedding(op, pixel_values)
        batch_size = op.Shape(pixel_values, start=0, end=1)

        cls_tokens = op.Unsqueeze(self.class_embedding, [0, 1])
        cls_tokens = op.Expand(
            cls_tokens, op.Concat(batch_size, [1], [self.class_embedding.shape[0]], axis=0)
        )
        embeddings = op.Concat(cls_tokens, patch_embeds, axis=1)

        seq_len = op.Shape(embeddings, start=1, end=2)
        position_ids = op.Range(op.Constant(value_int=0), seq_len, op.Constant(value_int=1))
        position_ids = op.Cast(position_ids, to=7)
        position_ids = op.Unsqueeze(position_ids, [0])
        embeddings = op.Add(embeddings, self.position_embedding(op, position_ids))
        return embeddings


class _Conv2dPatchEmbed(nn.Module):
    """Conv2d-based patch embedding with reshape and transpose.

    ``bias`` selects a biased (SigLIP) or bias-free (CLIP) patch convolution.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        bias: bool = True,
    ):
        super().__init__()
        conv_cls = Conv2d if bias else Conv2dNoBias
        self.projection = conv_cls(in_channels, out_channels, kernel_size, stride)

    def forward(self, op: OpBuilder, x: ir.Value):
        # Conv2d: [batch, channels, H, W] -> [batch, out_channels, H', W']
        conv_out = self.projection(op, x)
        # Reshape [batch, out_channels, H', W'] -> [batch, num_patches, out_channels]
        batch_size = op.Shape(conv_out, start=0, end=1)
        out_channels = op.Shape(conv_out, start=1, end=2)
        # Flatten spatial dims
        conv_out = op.Reshape(conv_out, op.Concat(batch_size, out_channels, [-1], axis=0))
        # Transpose to [batch, num_patches, channels]
        conv_out = op.Transpose(conv_out, perm=[0, 2, 1])
        return conv_out


class _CLIPVisionEncoderLayer(nn.Module):
    """CLIP vision encoder layer: pre-norm with LayerNorm."""

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.self_attn = EncoderAttention(config.hidden_size, config.num_attention_heads)
        self.layer_norm1 = LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = FCMLP(
            config.hidden_size,
            config.intermediate_size,
            activation=config.hidden_act or "quick_gelu",
        )
        self.layer_norm2 = LayerNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, op: OpBuilder, hidden_states: ir.Value):
        residual = hidden_states
        hidden_states = self.layer_norm1(op, hidden_states)
        hidden_states = self.self_attn(op, hidden_states)
        hidden_states = op.Add(residual, hidden_states)

        residual = hidden_states
        hidden_states = self.layer_norm2(op, hidden_states)
        hidden_states = self.mlp(op, hidden_states)
        hidden_states = op.Add(residual, hidden_states)
        return hidden_states


def resolve_clip_feature_num_layers(num_hidden_layers: int, feature_layer: int) -> int:
    """Number of encoder layers to run to reach ``hidden_states[feature_layer]``.

    HuggingFace CLIP exposes ``num_hidden_layers + 1`` hidden states: index 0 is
    the pre-encoder embedding output and index ``k`` is the output after ``k``
    encoder layers. A model such as Phi-3.5-Vision extracts features from an
    intermediate layer (``layer_idx = -2``), which corresponds to running one
    fewer encoder layer and skipping the final ``post_layernorm``.

    Args:
        num_hidden_layers: Total number of CLIP encoder layers in the checkpoint.
        feature_layer: The ``hidden_states`` index to extract (may be negative,
            using Python's from-the-end convention over the ``N + 1`` states).

    Returns:
        The number of encoder layers to actually instantiate and run.

    Raises:
        ValueError: If ``feature_layer`` is out of range for the encoder depth.
    """
    num_hidden_states = num_hidden_layers + 1
    index = feature_layer if feature_layer >= 0 else num_hidden_states + feature_layer
    if not 0 <= index <= num_hidden_layers:
        raise ValueError(
            f"feature_layer={feature_layer} is out of range for a CLIP encoder "
            f"with {num_hidden_layers} layers (valid hidden-state indices span "
            f"0..{num_hidden_layers})."
        )
    return index


class CLIPVisionModel(nn.Module):
    """CLIP vision model for standalone image feature extraction.

    By default this outputs the final ``last_hidden_state`` (all encoder layers
    followed by ``post_layernorm``).

    Two options support multimodal feature extraction used by models such as
    Phi-3.5-Vision:

    * ``feature_layer`` selects an intermediate ``hidden_states`` index
      (HuggingFace convention, e.g. ``-2``). When set, only the encoder layers
      needed to reach that hidden state are instantiated and run, and the final
      ``post_layernorm`` is skipped (it is only applied to the last hidden
      state in HuggingFace).
    * ``drop_class_token`` removes the leading CLS token from the output so that
      only the patch features remain (HuggingFace ``img_feature[:, 1:]``).
    """

    default_task = "image-classification"
    category = "vision"

    def __init__(
        self,
        config: ArchitectureConfig,
        *,
        feature_layer: int | None = None,
        drop_class_token: bool = False,
    ):
        super().__init__()
        self.feature_layer = feature_layer
        self.drop_class_token = drop_class_token

        if feature_layer is None:
            num_encoder_layers = config.num_hidden_layers
        else:
            num_encoder_layers = resolve_clip_feature_num_layers(
                config.num_hidden_layers, feature_layer
            )

        self.embeddings = _CLIPVisionEmbeddings(config)
        self.encoder = nn.ModuleList(
            [_CLIPVisionEncoderLayer(config) for _ in range(num_encoder_layers)]
        )
        self.pre_layrnorm = LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        # post_layernorm is only applied to the final hidden state; when we
        # extract an intermediate feature layer it is unused (and its weights
        # are intentionally left unmapped).
        if feature_layer is None:
            self.post_layernorm = LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.post_layernorm = None

    def forward(self, op: OpBuilder, pixel_values: ir.Value):
        hidden_states = self.embeddings(op, pixel_values)
        hidden_states = self.pre_layrnorm(op, hidden_states)

        for layer in self.encoder:
            hidden_states = layer(op, hidden_states)

        if self.post_layernorm is not None:
            hidden_states = self.post_layernorm(op, hidden_states)

        if self.drop_class_token:
            # Keep patch tokens only, dropping the leading CLS token:
            # (batch, 1 + num_patches, hidden) -> (batch, num_patches, hidden).
            hidden_states = op.Slice(
                hidden_states,
                [1],  # starts
                [_INT64_MAX],  # ends (clamped to sequence length)
                [1],  # axes: sequence dimension
            )

        return hidden_states

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        new_state_dict = {}
        for name, tensor in state_dict.items():
            new_name = _rename_clip_vision_weight(name)
            if new_name is not None:
                new_state_dict[new_name] = tensor
        return new_state_dict


# ---------------------------------------------------------------------------
# Weight name mapping
# ---------------------------------------------------------------------------

_CLIP_LAYER_RENAMES = {
    "self_attn.q_proj.": "self_attn.q_proj.",
    "self_attn.k_proj.": "self_attn.k_proj.",
    "self_attn.v_proj.": "self_attn.v_proj.",
    "self_attn.out_proj.": "self_attn.out_proj.",
}


def _rename_clip_vision_weight(name: str) -> str | None:
    """Rename a HF CLIP vision weight to our naming convention."""
    # Strip various prefixes
    for prefix in ("vision_model.", "clip.vision_model.", "model.vision_model."):
        if name.startswith(prefix):
            name = name[len(prefix) :]
            break

    # Skip text model and projection weights
    if name.startswith(
        ("text_model.", "text_projection.", "visual_projection.", "logit_scale")
    ):
        return None

    # Embeddings — HF stores a flat ``patch_embedding.{weight,bias}`` Conv, but
    # our ``_Conv2dPatchEmbed`` wraps the Conv in a ``.projection`` sub-module.
    if name.startswith("embeddings."):
        name = name.replace(
            "embeddings.patch_embedding.", "embeddings.patch_embedding.projection."
        )
        return name

    # Pre/post layer norm
    if name.startswith(("pre_layrnorm.", "post_layernorm.")):
        return name

    # Encoder layers
    if name.startswith("encoder.layers."):
        parts = name.split(".", 3)  # encoder, layers, idx, remainder
        if len(parts) < 4:
            return None
        layer_idx = parts[2]
        remainder = parts[3]

        for old, new in _CLIP_LAYER_RENAMES.items():
            if remainder.startswith(old):
                suffix = remainder[len(old) :]
                return f"encoder.{layer_idx}.{new}{suffix}"

        # MLP: fc1 → up_proj, fc2 → down_proj (FCMLP naming)
        remainder = remainder.replace("mlp.fc1.", "mlp.up_proj.")
        remainder = remainder.replace("mlp.fc2.", "mlp.down_proj.")

        # layer_norm1, layer_norm2, mlp pass through
        return f"encoder.{layer_idx}.{remainder}"

    return None


# ---------------------------------------------------------------------------
# SigLIP Vision Model (no CLS token, no pre-layernorm)
# ---------------------------------------------------------------------------


class _SigLIPVisionEmbeddings(nn.Module):
    """SigLIP vision embeddings: Conv2d patch + position embeddings (no CLS token)."""

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        hidden_size = config.hidden_size
        patch_size = config.patch_size
        image_size = config.image_size

        self.patch_embedding = _Conv2dPatchEmbed(
            config.num_channels,
            hidden_size,
            kernel_size=patch_size,
            stride=patch_size,
        )
        num_patches = (image_size // patch_size) ** 2
        self.position_embedding = Embedding(num_patches, hidden_size)

    def forward(self, op: OpBuilder, pixel_values: ir.Value):
        patch_embeds = self.patch_embedding(op, pixel_values)
        # Position IDs: [0, 1, ..., num_patches-1]
        seq_len = op.Shape(patch_embeds, start=1, end=2)
        position_ids = op.Range(op.Constant(value_int=0), seq_len, op.Constant(value_int=1))
        position_ids = op.Cast(position_ids, to=7)
        position_ids = op.Unsqueeze(position_ids, [0])
        embeddings = op.Add(patch_embeds, self.position_embedding(op, position_ids))
        return embeddings


class SigLIPVisionModel(nn.Module):
    """SigLIP vision model for standalone image feature extraction.

    SigLIP is similar to CLIP but without a CLS token and without
    pre-layernorm.  The position embedding size matches num_patches
    exactly (no +1 for CLS).
    """

    default_task = "image-classification"
    category = "vision"

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.embeddings = _SigLIPVisionEmbeddings(config)
        self.encoder = nn.ModuleList(
            [_CLIPVisionEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.post_layernorm = LayerNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, op: OpBuilder, pixel_values: ir.Value):
        hidden_states = self.embeddings(op, pixel_values)

        for layer in self.encoder:
            hidden_states = layer(op, hidden_states)

        hidden_states = self.post_layernorm(op, hidden_states)
        return hidden_states

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        new_state_dict = {}
        for name, tensor in state_dict.items():
            new_name = _rename_siglip_vision_weight(name)
            if new_name is not None:
                new_state_dict[new_name] = tensor
        return new_state_dict


def _rename_siglip_vision_weight(name: str) -> str | None:
    """Rename a HF SigLIP vision weight to our naming convention."""
    for prefix in ("vision_model.", "model.vision_model."):
        if name.startswith(prefix):
            name = name[len(prefix) :]
            break

    # Skip non-vision weights
    if name.startswith(
        ("text_model.", "text_projection.", "visual_projection.", "logit_scale")
    ):
        return None

    # Embeddings — SigLIP uses flat patch_embedding.weight/bias,
    # but our _Conv2dPatchEmbed wraps Conv2d in a .projection sub-module
    if name.startswith("embeddings."):
        name = name.replace(
            "embeddings.patch_embedding.", "embeddings.patch_embedding.projection."
        )
        return name

    # Post layer norm (SigLIP has no pre-layernorm)
    if name.startswith("post_layernorm."):
        return name
    if name.startswith("layernorm."):
        return name.replace("layernorm.", "post_layernorm.", 1)

    # Encoder layers (same as CLIP)
    if name.startswith("encoder.layers."):
        parts = name.split(".", 3)
        if len(parts) < 4:
            return None
        layer_idx = parts[2]
        remainder = parts[3]

        for old, new in _CLIP_LAYER_RENAMES.items():
            if remainder.startswith(old):
                suffix = remainder[len(old) :]
                return f"encoder.{layer_idx}.{new}{suffix}"

        remainder = remainder.replace("mlp.fc1.", "mlp.up_proj.")
        remainder = remainder.replace("mlp.fc2.", "mlp.down_proj.")
        return f"encoder.{layer_idx}.{remainder}"

    # Head weights (e.g., classifier head for image classification)
    if name.startswith("head."):
        return None

    return None


# ---------------------------------------------------------------------------
# CLIP Text Model
# ---------------------------------------------------------------------------


class _CLIPTextEmbeddings(nn.Module):
    """CLIP text embeddings: token + learned absolute position embeddings."""

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.word_embeddings = Embedding(config.vocab_size, config.hidden_size)
        self.position_embedding = Embedding(config.max_position_embeddings, config.hidden_size)

    def forward(self, op: OpBuilder, input_ids: ir.Value):
        token_embeds = self.word_embeddings(op, input_ids)
        seq_len = op.Shape(input_ids, start=1, end=2)
        position_ids = op.Range(
            op.Constant(value_int=0),
            op.Squeeze(seq_len),
            op.Constant(value_int=1),
        )
        position_ids = op.Cast(position_ids, to=7)  # INT64
        position_ids = op.Unsqueeze(position_ids, [0])
        position_embeds = self.position_embedding(op, position_ids)
        return op.Add(token_embeds, position_embeds)


class _CLIPTextEncoderLayer(nn.Module):
    """CLIP text encoder layer: pre-norm with LayerNorm and causal attention."""

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.self_attn = EncoderAttention(config.hidden_size, config.num_attention_heads)
        self.layer_norm1 = LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = FCMLP(
            config.hidden_size,
            config.intermediate_size,
            activation=config.hidden_act or "quick_gelu",
        )
        self.layer_norm2 = LayerNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        attention_mask: ir.Value | None = None,
    ):
        residual = hidden_states
        hidden_states = self.layer_norm1(op, hidden_states)
        hidden_states = self.self_attn(op, hidden_states, attention_mask)
        hidden_states = op.Add(residual, hidden_states)

        residual = hidden_states
        hidden_states = self.layer_norm2(op, hidden_states)
        hidden_states = self.mlp(op, hidden_states)
        hidden_states = op.Add(residual, hidden_states)
        return hidden_states


class CLIPTextModel(nn.Module):
    """CLIP text model with causal attention for feature extraction.

    Unlike BERT (bidirectional), CLIP's text encoder uses causal (triangular)
    attention masking. Outputs last_hidden_state.
    """

    default_task = "feature-extraction"
    category = "Encoder"

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.embeddings = _CLIPTextEmbeddings(config)
        self.encoder = nn.ModuleList(
            [_CLIPTextEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.final_layer_norm = LayerNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        op: OpBuilder,
        input_ids: ir.Value,
        attention_mask: ir.Value,
        token_type_ids: ir.Value,  # Unused but required by FeatureExtractionTask interface
    ):
        hidden_states = self.embeddings(op, input_ids)

        # Build causal attention bias (lower-triangular)
        # CastLike ensures the bias dtype matches hidden_states (fp16/bf16/fp32)
        # Cast the scalar *before* Expand so only a single element is cast
        seq_len = op.Shape(input_ids, start=1, end=2)
        _causal_mask = op.Trilu(
            op.Expand(
                op.CastLike(0.0, hidden_states),
                op.Concat(seq_len, seq_len, axis=0),
            ),
            upper=0,
        )
        # Fill upper triangle with -inf
        neg_inf_mask = op.Trilu(
            op.Expand(
                op.CastLike(-10000.0, hidden_states),
                op.Concat(seq_len, seq_len, axis=0),
            ),
            upper=1,
        )
        # Zero diagonal for upper-tri mask
        diag_mask = op.Trilu(neg_inf_mask, upper=0)
        causal_bias = op.Sub(neg_inf_mask, diag_mask)
        # Reshape to [1, 1, seq, seq] for attention
        causal_bias = op.Unsqueeze(causal_bias, [0, 1])

        for layer in self.encoder:
            hidden_states = layer(op, hidden_states, causal_bias)

        hidden_states = self.final_layer_norm(op, hidden_states)
        return hidden_states

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        new_state_dict = {}
        for name, tensor in state_dict.items():
            new_name = _rename_clip_text_weight(name)
            if new_name is not None:
                new_state_dict[new_name] = tensor
        return new_state_dict


def _rename_clip_text_weight(name: str) -> str | None:
    """Rename a HF CLIP text weight to our naming convention."""
    # Strip various prefixes
    for prefix in ("text_model.", "clip.text_model.", "model.text_model."):
        if name.startswith(prefix):
            name = name[len(prefix) :]
            break

    # Skip vision model and projection weights
    if name.startswith(
        ("vision_model.", "text_projection.", "visual_projection.", "logit_scale")
    ):
        return None

    # Embeddings — HF uses token_embedding, we use word_embeddings
    if name.startswith("embeddings."):
        name = name.replace("embeddings.token_embedding.", "embeddings.word_embeddings.")
        return name

    # Final layer norm
    if name.startswith("final_layer_norm."):
        return name

    # Encoder layers
    if name.startswith("encoder.layers."):
        parts = name.split(".", 3)  # encoder, layers, idx, remainder
        if len(parts) < 4:
            return None
        layer_idx = parts[2]
        remainder = parts[3]

        for old, new in _CLIP_LAYER_RENAMES.items():
            if remainder.startswith(old):
                suffix = remainder[len(old) :]
                return f"encoder.{layer_idx}.{new}{suffix}"

        # MLP: fc1 → up_proj, fc2 → down_proj (FCMLP naming)
        remainder = remainder.replace("mlp.fc1.", "mlp.up_proj.")
        remainder = remainder.replace("mlp.fc2.", "mlp.down_proj.")

        # layer_norm1, layer_norm2, mlp pass through
        return f"encoder.{layer_idx}.{remainder}"

    return None
