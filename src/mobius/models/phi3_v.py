# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Phi-3-Vision / Phi-3.5-Vision multimodal model (vision + text) — 3-model split.

Builds three ONNX models:

- **decoder**: Phi3 text decoder taking ``inputs_embeds``
- **vision**: CLIP ViT-L/14-336 + MLP projector
- **embedding**: token embedding + image feature fusion

Architecture:
    pixel_values (num_crops, 3, 336, 336)
        → CLIP ViT-L/14-336 (hidden_states[-2], CLS dropped) → (num_crops, 576, 1024)
        → [host-side HD transform: 2x2 merge + sub_GN/glb_GN + MLP projector]
        → concat + insert into input sequence
        → Phi3 text decoder

The vision encoder uses CLIP ViT-L/14-336 (patch_size=14, image_size=336),
which produces 576 patches per crop (after dropping the CLS token). Features
are taken from ``hidden_states[layer_idx]`` (default ``-2``): the last encoder
layer and ``post_layernorm`` are skipped, matching HuggingFace's
``get_img_features``.

HD transform note: the HuggingFace phi3_v model supports high-definition
tiling (sub_glb ordering with learnable separators) and the ``img_projection``
MLP whose input width is ``image_dim_out * 4``. Both are image-size dependent
and are expected to run host-side, outside ONNX — the ONNX vision encoder
emits the raw per-crop CLIP patch features that feed into them.

HuggingFace weight name prefixes::

    model.embed_tokens.*
        → decoder.model.embed_tokens.*
        → embedding.embed_tokens.*
    model.layers.N.* (fused qkv_proj, gate_up_proj)
        → decoder.model.layers.N.* (split q/k/v, gate/up)
    model.norm.* / lm_head.*
        → decoder.model.norm.* / decoder.lm_head.*
    model.vision_embed_tokens.img_processor.vision_model.*
        → vision_encoder.vision_tower.*  (CLIP tower only)

    (model.vision_embed_tokens.img_projection.* and sub_GN/glb_GN are
     host-side and intentionally not exported.)

Reference: ``microsoft/Phi-3-vision-128k-instruct``,
``microsoft/Phi-3.5-vision-instruct``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from onnxscript import nn
from onnxscript._internal import builder

from mobius._configs import ArchitectureConfig
from mobius._weight_utils import split_fused_qkv, split_gate_up_proj
from mobius.components import (
    Embedding,
    Linear,
)
from mobius.models.base import TextModel
from mobius.models.clip import (
    ClipVisionConfigView,
    CLIPVisionModel,
    _rename_clip_vision_weight,
)

if TYPE_CHECKING:
    import onnx_ir as ir

# Phi-3-Vision: <|image|> token id (kept for reference; the HF processor marks
# image slots with negative placeholder ids, not this positive id — see
# ``_Phi3VEmbeddingModel.forward``).
_IMAGE_TOKEN_ID = 32044

# Upper bound (magnitude) for negative image placeholder ids, mirroring
# ``modeling_phi3_v.MAX_INPUT_ID = int(1e9)``. Image positions satisfy
# ``-_MAX_INPUT_ID < input_ids < 0``.
_MAX_INPUT_ID = 1_000_000_000


class _Phi3VDecoderModel(nn.Module):
    """Phi3-V text decoder taking inputs_embeds.

    Identical to Phi3CausalLMModel but accepts inputs_embeds instead of
    input_ids, allowing the vision features to be fused before decoding.
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.config = config
        self.model = TextModel(config)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        op: builder.OpBuilder,
        inputs_embeds: ir.Value,
        attention_mask: ir.Value,
        position_ids: ir.Value,
        past_key_values: list | None = None,
    ):
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
        """Map Phi3-V decoder weights, splitting fused QKV and gate_up."""
        # phi3_v has no `language_model.` prefix — decoder weights live under `model.*`
        prefix = "model."
        renamed: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            if key.startswith("model.vision_embed_tokens."):
                continue
            if key.startswith(prefix):
                renamed[key] = value
            elif key == "lm_head.weight":
                renamed["lm_head.weight"] = value

        # Split fused QKV
        for key in list(renamed.keys()):
            if "qkv_proj" in key:
                q, k, v = split_fused_qkv(
                    renamed.pop(key),
                    self.config.num_attention_heads,
                    self.config.num_key_value_heads,
                    self.config.head_dim,
                )
                renamed[key.replace("qkv_proj", "q_proj")] = q
                renamed[key.replace("qkv_proj", "k_proj")] = k
                renamed[key.replace("qkv_proj", "v_proj")] = v
            elif "gate_up_proj" in key:
                gate, up = split_gate_up_proj(
                    renamed.pop(key),
                    self.config.intermediate_size,
                )
                renamed[key.replace("gate_up_proj", "gate_proj")] = gate
                renamed[key.replace("gate_up_proj", "up_proj")] = up

        return renamed


# Prefix under which the CLIP ViT weights live in a Phi-3-V checkpoint.
_VISION_TOWER_PREFIX = "model.vision_embed_tokens.img_processor."


def _rename_phi3v_vision_weight(name: str) -> str | None:
    """Map a Phi-3-V ``img_processor`` CLIP weight to ``vision_tower.*``.

    Strips the ``model.vision_embed_tokens.img_processor.`` prefix, then reuses
    the shared CLIP vision renamer (which handles the ``vision_model.`` prefix,
    the ``patch_embedding`` → ``patch_embedding.projection`` Conv wrapping, the
    ``mlp.fc1/fc2`` → ``mlp.up_proj/down_proj`` naming, and the per-layer
    ``encoder.layers.N`` → ``encoder.N`` flattening).

    Returns ``None`` for weights that are not part of the CLIP tower (e.g. the
    ``img_projection`` MLP and the learnable ``sub_GN``/``glb_GN`` separators,
    which are applied host-side as part of the HD feature transform).
    """
    if not name.startswith(_VISION_TOWER_PREFIX):
        return None
    clip_name = _rename_clip_vision_weight(name[len(_VISION_TOWER_PREFIX) :])
    if clip_name is None:
        return None
    return "vision_tower." + clip_name


class _Phi3VVisionEncoderModel(nn.Module):
    """Phi3-V vision encoder: CLIP ViT-L/14-336 patch-feature extractor.

    Faithfully reproduces HuggingFace ``Phi3ImageEmbedding.get_img_features``:
    run the CLIP tower to ``hidden_states[layer_idx]`` (default ``-2``, i.e. all
    but the last encoder layer and no ``post_layernorm``) and keep only the
    patch tokens, dropping the leading CLS token.

    The ``img_projection`` MLP and the HD 2x2 patch-merge with learnable
    ``sub_GN``/``glb_GN`` separators are intentionally *not* part of this ONNX
    graph: the projector's input width (``image_dim_out * 4``) only exists after
    the host-side HD feature transform, which is image-size dependent and cannot
    be expressed as a static graph. They run host-side alongside the pre-tiling
    of crops, consistent with the model's HD-transform design.

    Input shape:  (num_crops, 3, image_size, image_size)  — NCHW
    Output shape: (num_crops, num_patches, image_dim_out)  — raw patch features
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        assert config.vision is not None, "Phi3-V requires a vision config"
        clip_config = ClipVisionConfigView(config.vision)
        self.vision_tower = CLIPVisionModel(
            cast(ArchitectureConfig, clip_config),
            feature_layer=config.vision.feature_layer,
            drop_class_token=True,
        )

    def forward(self, op: builder.OpBuilder, pixel_values: ir.Value):
        # pixel_values: (num_crops, 3, H, W) -> (num_crops, num_patches, image_dim_out)
        return self.vision_tower(op, pixel_values)

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Extract CLIP vision-tower weights from the full checkpoint.

        HF path → ONNX path::

            model.vision_embed_tokens.img_processor.vision_model.*
                → vision_tower.*  (via the shared CLIP renamer)

        The ``img_projection`` and ``sub_GN``/``glb_GN`` tensors are dropped
        (host-side HD transform).
        """
        renamed: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            new_key = _rename_phi3v_vision_weight(key)
            if new_key is not None:
                renamed[new_key] = value
        return renamed


class _Phi3VEmbeddingModel(nn.Module):
    """Phi3-V embedding: token lookup + image feature fusion.

    Replaces image placeholder positions with projected vision features from
    the vision encoder. Phi-3/3.5-Vision processors mark those positions with
    *negative* placeholder ids (-1, -2, ...), matched here the same way HF's
    ``modeling_phi3_v`` does — see ``forward``.
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.embed_tokens = Embedding(
            config.vocab_size, config.hidden_size, config.pad_token_id
        )
        self.hidden_size = config.hidden_size

    def forward(self, op: builder.OpBuilder, input_ids: ir.Value, image_features: ir.Value):
        # (batch, seq_len, hidden_size)
        # Phi-3/3.5-Vision processors mark image placeholder positions with
        # *negative* token ids (-1 for the first image, -2 for the second, ...),
        # NOT with a positive ``image_token_id``. Detect them exactly as HF's
        # ``modeling_phi3_v`` does: ``(input_ids < 0) & (input_ids > -MAX_INPUT_ID)``.
        # Matching a positive id here would never fire for real processor output,
        # leaving image features unfused and the decoder running "blind".
        image_mask = op.And(
            op.Less(input_ids, op.Constant(value_int=0)),
            op.Greater(input_ids, op.Constant(value_int=-_MAX_INPUT_ID)),
        )
        # Replace image token positions with index 0 before the Gather so that
        # negative or out-of-range image token IDs don't cause undefined behavior.
        # These positions will be overwritten by image features via op.Where below.
        safe_ids = op.Where(image_mask, op.Constant(value_int=0), input_ids)
        text_embeds = self.embed_tokens(op, safe_ids)

        image_mask_3d = op.Unsqueeze(image_mask, [-1])

        mask_int = op.Cast(image_mask, to=7)  # INT64
        cumsum = op.CumSum(mask_int, op.Constant(value_int=1))
        indices = op.Sub(cumsum, op.Constant(value_int=1))
        indices = op.Clip(indices, op.Constant(value_int=0))

        # Pad ``image_features`` with a single trailing zero row so the Gather is
        # always valid — including at decode time, when there are no image
        # placeholders and the harness/runtime passes an empty (0-row) tensor.
        # Without the pad, ``Gather`` would index row 0 of a 0-row tensor and
        # raise an out-of-bounds error. The gathered padding is discarded by the
        # ``Where`` below (image_mask is all-false when there are no image rows).
        zero_row = op.ConstantOfShape(op.Constant(value_ints=[1, self.hidden_size]))
        zero_row = op.CastLike(zero_row, image_features)
        padded_features = op.Concat(image_features, zero_row, axis=0)

        gathered = op.Gather(padded_features, indices, axis=0)
        return op.Where(image_mask_3d, gathered, text_embeds)

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Extract embed_tokens weights for the embedding sub-model."""
        renamed: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            if "embed_tokens" in key:
                # model.embed_tokens.weight → embed_tokens.weight
                for pfx in ("model.", "language_model.model.", "language_model."):
                    if key.startswith(pfx):
                        renamed[key[len(pfx) :]] = value
                        break
        return renamed


class Phi3VModel(nn.Module):
    """Phi-3-Vision / Phi-3.5-Vision model (3-model split).

    Builds three separate ONNX models:
    - decoder: Phi3 text decoder taking inputs_embeds
    - vision_encoder: CLIP ViT-L/14-336 + MLP projector
    - embedding: token embedding + image feature fusion

    Supports ``microsoft/Phi-3-vision-128k-instruct`` and
    ``microsoft/Phi-3.5-vision-instruct``.
    """

    default_task: str = "vision-language"
    category: str = "Multimodal"

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.config = config
        self.decoder = _Phi3VDecoderModel(config)
        self.vision_encoder = _Phi3VVisionEncoderModel(config)
        self.embedding = _Phi3VEmbeddingModel(config)

    def forward(self, op: builder.OpBuilder, **kwargs):
        raise NotImplementedError(
            "Phi3VModel uses VisionLanguageTask which calls "
            "each sub-module (decoder, vision_encoder, embedding) separately."
        )

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Rename HuggingFace phi3_v weights to ONNX initializer names.

        HF checkpoint layout → ONNX initializer names::

            model.embed_tokens.*
                → decoder.model.embed_tokens.*
                → embedding.embed_tokens.*      (duplicated)
            model.layers.N.*  (fused qkv/gate_up)
                → decoder.model.layers.N.*       (split)
            model.norm.* / lm_head.*
                → decoder.model.norm.* / decoder.lm_head.*
            model.vision_embed_tokens.img_processor.vision_model.*
                → vision_encoder.vision_tower.*  (CLIP tower only)

        The ``img_projection`` MLP and the learnable ``sub_GN``/``glb_GN``
        separators are dropped — they are applied host-side as part of the HD
        feature transform (see :class:`_Phi3VVisionEncoderModel`).
        """
        renamed: dict[str, torch.Tensor] = {}

        for key, value in state_dict.items():
            vision_key = _rename_phi3v_vision_weight(key)
            if vision_key is not None:
                renamed["vision_encoder." + vision_key] = value
            elif key.startswith("model.vision_embed_tokens."):
                # img_projection / sub_GN / glb_GN and any other vision-embed
                # state is host-side (HD transform) — skip it.
                pass
            elif key == "model.embed_tokens.weight":
                renamed["decoder.model.embed_tokens.weight"] = value
                renamed["embedding.embed_tokens.weight"] = value
            elif key.startswith("model."):
                renamed["decoder." + key] = value
            elif key == "lm_head.weight":
                renamed["decoder.lm_head.weight"] = value
            else:
                renamed[key] = value

        # Split fused QKV and gate_up in decoder weights
        for key in list(renamed.keys()):
            if not key.startswith("decoder."):
                continue
            if "qkv_proj" in key:
                q, k, v = split_fused_qkv(
                    renamed.pop(key),
                    self.config.num_attention_heads,
                    self.config.num_key_value_heads,
                    self.config.head_dim,
                )
                renamed[key.replace("qkv_proj", "q_proj")] = q
                renamed[key.replace("qkv_proj", "k_proj")] = k
                renamed[key.replace("qkv_proj", "v_proj")] = v
            elif "gate_up_proj" in key:
                gate, up = split_gate_up_proj(
                    renamed.pop(key),
                    self.config.intermediate_size,
                )
                renamed[key.replace("gate_up_proj", "gate_proj")] = gate
                renamed[key.replace("gate_up_proj", "up_proj")] = up

        return renamed
