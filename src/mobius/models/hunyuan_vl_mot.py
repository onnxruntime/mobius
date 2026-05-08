# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""HunYuan VL-MoT vision-language model — 3-model split.

Replicates ``tencent/HY-Embodied-0.5-X`` (HunYuanVLMoTForConditionalGeneration).

Architecture:
- **Vision encoder**: 27-block ViT (fused QKV, LayerNorm) with spatial merger
- **Embedding**: Token lookup + image feature scatter at placeholder positions
- **Decoder**: 32-layer GQA (16Q/4KV heads, head_dim=128) with QK-norm

HuggingFace weight layout::

    model.visual.vision_tower.patch_embed.proj.{weight,bias}
    model.visual.vision_tower.pos_embed
    model.visual.vision_tower.blocks.{i}.attn.qkv.{weight,bias}
    model.visual.vision_tower.blocks.{i}.attn.proj.{weight,bias}
    model.visual.vision_tower.blocks.{i}.mlp.fc1/fc2.{weight,bias}
    model.visual.vision_tower.blocks.{i}.norm1/norm2.{weight,bias}
    model.visual.merger.proj1/proj2.{weight,bias}
    model.visual.merger.pooler.predictor.{0,2}.{weight,bias}
    model.language_model.model.layers.{i}.self_attn.{q,k,v,o}_proj.weight
    model.language_model.model.layers.{i}.self_attn.query_layernorm.weight
    model.language_model.model.layers.{i}.self_attn.key_layernorm.weight
    model.language_model.model.layers.{i}.{input,post_attention}_layernorm.weight
    model.language_model.model.layers.{i}.mlp.{gate,up,down}_proj.weight
    model.language_model.model.embed_tokens.weight
    model.language_model.model.norm.weight

.. note::

   The HF model also contains ``_v``-suffixed weights for a Mixture-of-Tokens
   (MoT) visual pathway (``q_proj_v``, ``mlp_v``, ``input_layernorm_v``, etc.)
   that routes vision tokens through separate projections per layer.  This
   pathway is not yet implemented — the ONNX decoder uses the standard text
   pathway only.  This is correct for text-only decode steps; full MoT prefill
   support is tracked as future work.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import torch
from onnxscript import OpBuilder, nn

from mobius._configs import ArchitectureConfig
from mobius.components import (
    Embedding,
    Linear,
    MLPMultiModalProjector,
    VisionModel,
)
from mobius.models.base import TextModel

if TYPE_CHECKING:
    import onnx_ir as ir


# ── Vision encoder ──────────────────────────────────────────────────────


class _HunYuanVLMoTVisionEncoderModel(nn.Module):
    """ViT vision tower + merger projector.

    The vision tower is a standard ViT (SigLIP-style) wrapped by
    :class:`VisionModel`.  The merger projects vision features from the
    vision hidden dimension to the text hidden dimension using a two-layer
    MLP (proj1 → GELU → proj2).

    HF weight prefix: ``model.visual.*``

    .. note::

       Weight renaming is handled entirely by
       :meth:`HunYuanVLMoTModel.preprocess_weights` — the sub-model's
       ``preprocess_weights`` is not called in the standard build flow.
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        vc = config.vision
        assert vc is not None, "VisionConfig is required"
        self.vision_tower = VisionModel(config)
        # Merger: two linear projections that map
        # vision_hidden → text_hidden
        self.multi_modal_projector = MLPMultiModalProjector(
            vision_hidden_size=vc.hidden_size or config.hidden_size,
            text_hidden_size=config.hidden_size,
        )

    def forward(self, op: OpBuilder, pixel_values: ir.Value):
        vision_features = self.vision_tower(op, pixel_values)
        return self.multi_modal_projector(op, vision_features)


# ── Embedding ───────────────────────────────────────────────────────────


class _HunYuanVLMoTEmbeddingModel(nn.Module):
    """Token embedding + image feature scatter.

    Replaces image placeholder tokens with projected vision features
    using the standard Gather + Where pattern.
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = Embedding(
            config.vocab_size, config.hidden_size, config.pad_token_id
        )
        self.image_token_id = (config.vision.image_token_id if config.vision else 0) or 0

    def forward(
        self,
        op: OpBuilder,
        input_ids: ir.Value,
        image_features: ir.Value,
    ):
        text_embeds = self.embed_tokens(op, input_ids)

        # Build a mask of image-token positions
        image_mask = op.Equal(input_ids, op.Constant(value_int=self.image_token_id))
        image_mask_3d = op.Unsqueeze(image_mask, [-1])

        # CumSum-based indexing into image_features
        mask_int = op.Cast(image_mask, to=7)  # INT64
        cumsum = op.CumSum(mask_int, 1)
        indices = op.Sub(cumsum, op.Constant(value_int=1))
        indices = op.Clip(indices, op.Constant(value_int=0))

        # Pad with one zero row so Gather is valid for text-only inputs
        pad_row = op.Expand(
            op.CastLike(0.0, image_features),
            op.Concat(
                op.Constant(value_ints=[1]),
                op.Shape(image_features, start=1, end=2),
                axis=0,
            ),
        )
        padded_features = op.Concat(image_features, pad_row, axis=0)

        gathered = op.Gather(padded_features, indices, axis=0)
        return op.Where(image_mask_3d, gathered, text_embeds)


# ── Decoder ─────────────────────────────────────────────────────────────


class _HunYuanVLMoTDecoderModel(nn.Module):
    """Text decoder (standard pathway only).

    Uses the text-pathway weights (no ``_v`` suffix).  QK-norm is enabled
    unconditionally to match HuggingFace's ``query_layernorm`` /
    ``key_layernorm``.

    .. note::

       Weight renaming is handled entirely by
       :meth:`HunYuanVLMoTModel.preprocess_weights`.
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.config = config
        # Force QK-norm on (HF always has query_layernorm / key_layernorm)
        config = dataclasses.replace(config, attn_qk_norm=True)
        self.model = TextModel(config)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        op: OpBuilder,
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


# ── Top-level model ─────────────────────────────────────────────────────


class HunYuanVLMoTModel(nn.Module):
    """HunYuan VL-MoT vision-language model (3-model split).

    Builds three ONNX models for ORT GenAI deployment:

    - **decoder**: text decoder taking ``inputs_embeds``
    - **vision_encoder**: ViT + merger projector
    - **embedding**: token embedding + image feature fusion
    """

    default_task: str = "vision-language"
    category: str = "Multimodal"

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.config = config
        self.decoder = _HunYuanVLMoTDecoderModel(config)
        self.vision_encoder = _HunYuanVLMoTVisionEncoderModel(config)
        self.embedding = _HunYuanVLMoTEmbeddingModel(config)

    def forward(self, op: OpBuilder, **kwargs):
        raise NotImplementedError(
            "HunYuanVLMoTModel uses VisionLanguageTask which calls "
            "each sub-module (decoder, vision_encoder, embedding) separately."
        )

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        # Top-level HF prefix: "model." wraps everything.
        # Strip it first: model.language_model.* → language_model.*
        #                  model.visual.*        → visual.*
        stripped: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            new_key = key[len("model.") :] if key.startswith("model.") else key
            stripped[new_key] = value

        result: dict[str, torch.Tensor] = {}

        for key, value in stripped.items():
            if key.startswith("visual."):
                self._route_vision_weight(key, value, result)
            elif key.startswith("language_model."):
                self._route_decoder_weight(key, value, result)

        return result

    def _route_vision_weight(
        self,
        key: str,
        value: torch.Tensor,
        result: dict[str, torch.Tensor],
    ) -> None:
        """Route visual.* weights to vision_encoder sub-model."""
        suffix = key[len("visual.") :]

        if suffix.startswith("merger."):
            # merger.proj1 → multi_modal_projector.linear_1
            # merger.proj2 → multi_modal_projector.linear_2
            merger_key = suffix[len("merger.") :]
            merger_key = merger_key.replace("proj1.", "linear_1.")
            merger_key = merger_key.replace("proj2.", "linear_2.")
            # Skip pooler weights (not in standard MLP projector)
            if "pooler." in merger_key:
                return
            result[f"vision_encoder.multi_modal_projector.{merger_key}"] = value
            return

        # Vision tower weight renames:
        # vision_tower.blocks.N → vision_tower.vision_model.encoder.layers.N
        new_key = suffix.replace(
            "vision_tower.blocks.",
            "vision_tower.vision_model.encoder.layers.",
        )
        # attn.proj → self_attn.out_proj
        new_key = new_key.replace(".attn.proj.", ".self_attn.out_proj.")
        # norm1/norm2 → layer_norm1/layer_norm2
        new_key = new_key.replace(".norm1.", ".layer_norm1.")
        new_key = new_key.replace(".norm2.", ".layer_norm2.")
        # mlp.fc1/fc2 → mlp.up_proj/down_proj
        new_key = new_key.replace(".mlp.fc1.", ".mlp.up_proj.")
        new_key = new_key.replace(".mlp.fc2.", ".mlp.down_proj.")
        # patch_embed.proj → embeddings.patch_embedding
        new_key = new_key.replace(
            "vision_tower.patch_embed.proj.",
            "vision_tower.vision_model.embeddings.patch_embedding.",
        )
        # pos_embed → embeddings.position_embedding.weight
        # HF pos_embed is [1, num_patches, hidden] — squeeze batch dim
        if "vision_tower.pos_embed" in suffix:
            result[
                "vision_encoder.vision_tower.vision_model.embeddings.position_embedding.weight"
            ] = value.squeeze(0)
            return

        # Split fused QKV into separate Q, K, V
        if ".attn.qkv." in suffix:
            layer_prefix = new_key.split(".attn.qkv.")[0]
            param = "weight" if "weight" in key else "bias"
            chunks = torch.chunk(value, 3, dim=0)
            result[f"vision_encoder.{layer_prefix}.self_attn.q_proj.{param}"] = chunks[0]
            result[f"vision_encoder.{layer_prefix}.self_attn.k_proj.{param}"] = chunks[1]
            result[f"vision_encoder.{layer_prefix}.self_attn.v_proj.{param}"] = chunks[2]
            return

        result[f"vision_encoder.{new_key}"] = value

    def _route_decoder_weight(
        self,
        key: str,
        value: torch.Tensor,
        result: dict[str, torch.Tensor],
    ) -> None:
        """Route language_model.* weights to decoder and embedding."""
        # Strip language_model. prefix → model.layers.0.*, lm_head.*, etc.
        suffix = key[len("language_model.") :]

        # Skip MoT _v pathway weights
        if "_v." in suffix or suffix.endswith("_v"):
            return

        # Duplicate embed_tokens to embedding sub-model
        if "embed_tokens" in suffix:
            embed_key = suffix[len("model.") :] if suffix.startswith("model.") else suffix
            result[f"embedding.{embed_key}"] = value

        # QK-norm rename: query_layernorm → q_norm, key_layernorm → k_norm
        renamed = suffix.replace(".query_layernorm.", ".q_norm.").replace(
            ".key_layernorm.", ".k_norm."
        )
        result[f"decoder.{renamed}"] = value

        # Weight tying: embed_tokens → lm_head
        if self.config.tie_word_embeddings and "embed_tokens.weight" in key:
            result["decoder.lm_head.weight"] = value
