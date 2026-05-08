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
from mobius._weight_utils import vlm_decoder_weights, vlm_embedding_weights
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

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        renamed: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            if not key.startswith("vision_tower.") and not key.startswith(
                "multi_modal_projector."
            ):
                continue
            new_key = key
            # ── Vision tower weight renames ──
            # HF: vision_tower.blocks.N.* → our: vision_tower.vision_model.encoder.layers.N.*
            new_key = new_key.replace(
                "vision_tower.blocks.", "vision_tower.vision_model.encoder.layers."
            )
            # HF: attn.qkv → separate q_proj/k_proj/v_proj (handled below)
            # HF: attn.proj → our: attention.out_proj
            new_key = new_key.replace(".attn.proj.", ".attention.out_proj.")
            # HF: norm1 → our: layer_norm1; norm2 → our: layer_norm2
            new_key = new_key.replace(".norm1.", ".layer_norm1.")
            new_key = new_key.replace(".norm2.", ".layer_norm2.")
            # HF: mlp.fc1 → our: mlp.up_proj; mlp.fc2 → our: mlp.down_proj
            new_key = new_key.replace(".mlp.fc1.", ".mlp.up_proj.")
            new_key = new_key.replace(".mlp.fc2.", ".mlp.down_proj.")
            # HF: patch_embed.proj → our: embeddings.patch_embedding
            new_key = new_key.replace(
                "vision_tower.patch_embed.proj.",
                "vision_tower.vision_model.embeddings.patch_embedding.",
            )
            # HF: pos_embed → our: embeddings.position_embedding
            new_key = new_key.replace(
                "vision_tower.pos_embed",
                "vision_tower.vision_model.embeddings.position_embedding",
            )

            # Split fused QKV into separate Q, K, V
            if ".attn.qkv." in key:
                layer_prefix = new_key.split(".attn.qkv.")[0]
                suffix = "weight" if "weight" in key else "bias"
                # Fused QKV: [3 * hidden, ...] → split into 3 equal parts
                chunks = torch.chunk(value, 3, dim=0)
                renamed[f"{layer_prefix}.attention.q_proj.{suffix}"] = chunks[0]
                renamed[f"{layer_prefix}.attention.k_proj.{suffix}"] = chunks[1]
                renamed[f"{layer_prefix}.attention.v_proj.{suffix}"] = chunks[2]
                continue

            renamed[new_key] = value
        return renamed


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

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        return vlm_embedding_weights(state_dict)


# ── Decoder ─────────────────────────────────────────────────────────────


class _HunYuanVLMoTDecoderModel(nn.Module):
    """Text decoder (standard pathway only).

    Uses the text-pathway weights (no ``_v`` suffix).  QK-norm is enabled
    unconditionally to match HuggingFace's ``query_layernorm`` /
    ``key_layernorm``.
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

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        # Extract decoder weights, stripping "language_model." prefix
        state_dict = vlm_decoder_weights(
            state_dict,
            prefix="language_model.",
            tie=self.config.tie_word_embeddings,
        )
        # Rename HF QK-norm keys to match Attention component naming
        # HF: .query_layernorm. → ours: .q_norm.
        # HF: .key_layernorm. → ours: .k_norm.
        renamed: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            # Skip MoT _v pathway weights — not used in text-only decoder
            if "_v." in key or key.endswith("_v"):
                continue
            new_key = key.replace(".query_layernorm.", ".q_norm.").replace(
                ".key_layernorm.", ".k_norm."
            )
            renamed[new_key] = value
        return renamed


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
            new_key = new_key.replace(
                "vision_tower.pos_embed",
                "vision_tower.vision_model.embeddings.position_embedding.weight",
            )
            result[f"vision_encoder.{new_key}"] = value.squeeze(0)
            return

        new_key = new_key.replace(
            "vision_tower.pos_embed",
            "vision_tower.vision_model.embeddings.position_embedding.weight",
        )

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
