# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""NVIDIA Cosmos 3 model support.

Supports the ``cosmos3_edge`` checkpoint (``nvidia/Cosmos3-Edge``,
``Cosmos3EdgeForConditionalGeneration``) both as a full vision-language model
(:class:`Cosmos3EdgeVLModel`, 3-model split) and as a standalone text reasoner
(:class:`Cosmos3EdgeTextModel`).

Cosmos3-Edge is a LLaVA-style VLM built from three towers:

- **Text reasoner** — a grouped-query-attention decoder with two
  Cosmos-specific traits:

  - **Non-gated feed-forward network** — ``down_proj(relu2(up_proj(x)))`` using
    a squared-ReLU activation (``hidden_act="relu2"``), rather than the
    GLU-style gated MLP used by Llama/Qwen. This maps onto :class:`FCMLP`.
  - **Interleaved 3D multimodal RoPE** (``mrope_section=[24, 20, 20]``).
    Frequency channel ``i`` is driven by the height axis when
    ``i % 3 == 1 and i < 3 * mrope_section[1]``, by the width axis when
    ``i % 3 == 2 and i < 3 * mrope_section[2]``, and by the temporal axis
    otherwise — *not* the contiguous chunking used by Qwen-VL. For text-only
    inference the three axes carry the same positions, so it reduces to
    standard 1D RoPE.

- **Vision encoder** — a *variable-resolution* SigLIP2 tower
  (:class:`Cosmos3EdgeVisionTower`, ``model.visual.*``) consuming
  pre-patchified, packed pixel values plus a ``grid_thw`` triple.
- **Merger projector** — a pixel-shuffle projector
  (:class:`Cosmos3EdgePatchMerger`, ``model.projector.*``) that merges each
  2x2 patch block and projects to the text hidden size.

Images and videos share the same tower, projector and token expansion: the
processor emits one ``<|vision_start|> ... <|vision_end|>`` span per image and
one span **per video frame**, with placeholder ids 19 (image) and 18 (video).
:class:`_Cosmos3EdgeEmbeddingModel` therefore scatters two independent feature
streams — ``image_features`` at id 19 and ``video_features`` at id 18 — exactly
mirroring ``Cosmos3EdgeModel.forward``'s two ``masked_scatter`` calls.

The HuggingFace weights use ``self_attn.to_{q,k,v,out}`` projection names and
place the text tower at the top level (``layers.*``, ``embed_tokens``,
``norm``, ``lm_head``) with the vision encoder and projector under
``model.visual.*`` / ``model.projector.*``. The VLM's
:meth:`Cosmos3EdgeVLModel.preprocess_weights` routes these to the three
sub-models; the text-only :meth:`Cosmos3EdgeTextModel.preprocess_weights`
drops the vision/projector weights.

The ``k_norm_und_for_gen`` per-layer key-norm weight is an artifact of the
two-tower (Mixture-of-Transformers) design: it normalizes the *understanding*
tower's keys for consumption by the *generator* (diffusion) tower, and is not
applied in the reasoner's own causal self-attention. It is therefore dropped
for both paths.

.. note::
   The Reasoner architecture is reproduced from the published
   ``transformers`` ``cosmos3_edge`` modeling code (cross-checked against
   vLLM's ``cosmos3_edge.py``). The *Generator*/Action towers that share the
   same checkpoint remain proprietary rectified-flow components and are not
   reproduced here.

The complete ``cosmos3_edge`` world-model package (this Reasoner plus the
shared MoT Generator, Wan VAE, and Action head) is composed by
``build_cosmos3_edge_world_model``. The ``cosmos3_omni`` variants use the same
Generator/VAE building blocks with a Qwen3-VL Reasoner.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from onnxscript import OpBuilder, nn

from mobius._configs import ArchitectureConfig
from mobius.components import (
    FCMLP,
    Cosmos3EdgePatchMerger,
    Cosmos3EdgeVisionTower,
    Embedding,
    Linear,
)
from mobius.models.base import CausalLMModel, TextModel

if TYPE_CHECKING:
    import onnx_ir as ir
    import torch


_DROPPED_UNIFIED_KEY_RE = re.compile(
    "|".join(
        (
            r"\.add_q_proj\.",
            r"\.add_k_proj\.",
            r"\.add_v_proj\.",
            r"\.to_add_out\.",
            r"\.norm_added_q\.",
            r"\.norm_added_k\.",
            r"moe_gen",
            r"^proj_out\.",
            r"^proj_in\.",
            r"^time_embedder\.",
            r"^audio_proj_out\.",
            r"^audio_proj_in\.",
            r"^audio_modality_embed$",
            r"^action_proj_out\.",
            r"^action_proj_in\.",
            r"^action_modality_embed$",
        )
    )
)


def _rename_cosmos_text_key(key: str) -> str:
    """Rename Cosmos3-Edge attention and QK-norm keys to mobius conventions."""
    return (
        key.replace("self_attn.to_out.0.", "self_attn.o_proj.")
        .replace("self_attn.to_q.", "self_attn.q_proj.")
        .replace("self_attn.to_k.", "self_attn.k_proj.")
        .replace("self_attn.to_v.", "self_attn.v_proj.")
        .replace("self_attn.to_out.", "self_attn.o_proj.")
        .replace("self_attn.norm_q.", "self_attn.q_norm.")
        .replace("self_attn.norm_k.", "self_attn.k_norm.")
    )


class Cosmos3EdgeTextModel(CausalLMModel):
    """Cosmos3-Edge text reasoner backbone (decoder-only).

    Extracts the language tower from the ``cosmos3_edge`` vision-language
    checkpoint. Replaces the gated MLP with a non-gated squared-ReLU
    :class:`FCMLP` and renames/strips weights so the standard
    :class:`CausalLMModel` backbone can consume them.
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__(config)
        # Cosmos3-Edge uses a non-gated FFN (up_proj -> relu2 -> down_proj),
        # unlike the GLU-style gated MLP of the base CausalLMModel.
        for layer in self.model.layers:
            layer.mlp = FCMLP(
                config.hidden_size,
                config.intermediate_size,
                activation=config.hidden_act or "relu2",
                bias=config.mlp_bias,
            )

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        renamed: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            # Drop the vision encoder and multimodal projector — this is the
            # standalone text decoder.
            if key.startswith(("model.visual.", "model.projector.")):
                continue
            # Drop the generator-tower key-norm (see module docstring).
            if "k_norm_und_for_gen" in key:
                continue

            new_key = _rename_cosmos_text_key(key)
            # The text tower is stored at the top level; the mobius backbone
            # nests it under ``model.``. ``lm_head`` stays at the top level.
            if new_key == "lm_head.weight":
                pass
            elif new_key.startswith(("layers.", "embed_tokens.")) or new_key == "norm.weight":
                new_key = f"model.{new_key}"

            renamed[new_key] = value
        return super().preprocess_weights(renamed)


# ──────────────────────────────────────────────────────────────────────────
# Cosmos3-Edge vision-language model (3-model split)
# ──────────────────────────────────────────────────────────────────────────


class _Cosmos3EdgeDecoderModel(nn.Module):
    """Cosmos3-Edge text decoder taking ``inputs_embeds``.

    Reuses the reasoner backbone (:class:`TextModel` with GQA) but swaps in a
    non-gated squared-ReLU :class:`FCMLP` and is driven through
    :class:`Cosmos3EdgeVLTask`, which feeds ``inputs_embeds`` (from the
    embedding sub-model) instead of ``input_ids``. The unused ``embed_tokens``
    initializer is pruned at build time, so the decoder only owns
    ``model.layers.*``, ``model.norm`` and ``lm_head``.
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.config = config
        self.model = TextModel(config)
        # Non-gated squared-ReLU FFN (up_proj -> relu2 -> down_proj).
        for layer in self.model.layers:
            layer.mlp = FCMLP(
                config.hidden_size,
                config.intermediate_size,
                activation=config.hidden_act or "relu2",
                bias=config.mlp_bias,
            )
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
        # state_dict here is the decoder-routed slice (text tower only): keys
        # such as ``layers.N.self_attn.to_q.weight``, ``norm.weight`` and
        # ``lm_head.weight`` at the HF top level (no ``model.`` prefix).
        renamed: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            new_key = _rename_cosmos_text_key(key)
            if new_key == "lm_head.weight":
                pass
            elif new_key.startswith(("layers.", "embed_tokens.")) or new_key == "norm.weight":
                new_key = f"model.{new_key}"
            renamed[new_key] = value
        return renamed


class _Cosmos3EdgeVisionEncoderModel(nn.Module):
    """Cosmos3-Edge vision encoder: SigLIP2 tower + pixel-shuffle projector.

    ``pixel_values [total_patches, patch_dim]`` + ``grid_thw [3]`` →
    :class:`Cosmos3EdgeVisionTower` → ``[total_patches, vision_hidden]`` →
    :class:`Cosmos3EdgePatchMerger` → ``[total_patches / merge², text_hidden]``.

    One call handles exactly one packed visual item: a single image
    (``grid_t == 1``) or all frames of one video (``grid_t == num_frames``),
    matching ``Cosmos3EdgeModel.get_image_features`` /
    ``get_video_features`` (which share the same code path).

    Sub-module attribute names (``visual`` / ``projector``) mirror the
    HuggingFace layout so weight mapping is a plain ``model.`` prefix strip.
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        vc = config.vision
        assert vc is not None, "Cosmos3-Edge requires a VisionConfig"
        assert vc.hidden_size is not None
        assert vc.intermediate_size is not None
        assert vc.num_hidden_layers is not None
        assert vc.num_attention_heads is not None
        assert vc.patch_size is not None
        assert vc.projector_intermediate_size is not None, (
            "Cosmos3-Edge projector requires projector_intermediate_size"
        )
        if vc.out_hidden_size is not None:
            assert vc.out_hidden_size == config.hidden_size, (
                "Cosmos3-Edge projector output must match the text hidden size, "
                f"got {vc.out_hidden_size} != {config.hidden_size}"
            )
        # ``num_patches`` sizes the learned position grid; fall back to the
        # nominal image_size//patch_size square when the config omits it.
        num_patches = vc.num_patches
        if num_patches is None:
            assert vc.image_size is not None
            if vc.image_size % vc.patch_size != 0:
                raise ValueError(
                    f"image_size ({vc.image_size}) must be divisible by "
                    f"patch_size ({vc.patch_size})"
                )
            num_patches = (vc.image_size // vc.patch_size) ** 2
        merge = vc.spatial_merge_size or 2
        self.patch_size = vc.patch_size
        self.temporal_patch_size = vc.temporal_patch_size or 1
        self.num_channels = vc.in_channels
        self.spatial_merge_size = merge
        self.patch_dim = (
            self.patch_size * self.patch_size * self.num_channels * self.temporal_patch_size
        )
        self.visual = Cosmos3EdgeVisionTower(
            hidden_size=vc.hidden_size,
            intermediate_size=vc.intermediate_size,
            num_hidden_layers=vc.num_hidden_layers,
            num_attention_heads=vc.num_attention_heads,
            patch_size=self.patch_size,
            num_channels=self.num_channels,
            num_patches=num_patches,
            norm_eps=vc.norm_eps,
            temporal_patch_size=self.temporal_patch_size,
            spatial_merge_size=merge,
        )
        self.projector = Cosmos3EdgePatchMerger(
            vision_hidden_size=vc.hidden_size,
            text_hidden_size=config.hidden_size,
            intermediate_size=vc.projector_intermediate_size,
            spatial_merge_size=merge,
            use_postshuffle_norm=vc.use_postshuffle_norm,
            norm_eps=vc.norm_eps,
        )

    def forward(self, op: OpBuilder, pixel_values: ir.Value, grid_thw: ir.Value):
        # grid_thw: [3] int64 = (frames, grid_h, grid_w) for this visual item.
        grid_t = op.Gather(grid_thw, op.Constant(value_int=0))
        grid_h = op.Gather(grid_thw, op.Constant(value_int=1))
        grid_w = op.Gather(grid_thw, op.Constant(value_int=2))
        vision_features = self.visual(op, pixel_values, grid_t, grid_h, grid_w)
        return self.projector(op, vision_features)

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        # state_dict here is the vision-routed slice: ``model.visual.*`` (SigLIP2
        # tower) and ``model.projector.*`` (merger projector). The module tree
        # mirrors those names, so only the ``model.`` prefix and the SigLIP
        # ``mlp.fc1/fc2`` -> ``FCMLP.up_proj/down_proj`` naming differ.
        renamed: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            if not key.startswith(("model.visual.", "model.projector.")):
                continue
            new_key = key[len("model.") :]
            new_key = new_key.replace(".mlp.fc1.", ".mlp.up_proj.").replace(
                ".mlp.fc2.", ".mlp.down_proj."
            )
            renamed[new_key] = value
        return renamed


class _Cosmos3EdgeEmbeddingModel(nn.Module):
    """Cosmos3-Edge embedding: token lookup + image/video feature fusion.

    Scatters projected vision features into the text embedding sequence at
    ``image_token_id`` (19) and ``video_token_id`` (18) positions. The two
    streams are independent, mirroring ``Cosmos3EdgeModel.forward``, which
    runs one ``masked_scatter`` per modality. Either stream may be empty
    (zero rows) for text-only, image-only or video-only prompts.
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = Embedding(
            config.vocab_size, config.hidden_size, config.pad_token_id
        )
        self.image_token_id = config.image_token_id or 0
        vision = config.vision
        self.video_token_id = (
            vision.video_token_id if vision is not None and vision.video_token_id else None
        )

    def _scatter(
        self,
        op: OpBuilder,
        embeds: ir.Value,
        input_ids: ir.Value,
        features: ir.Value,
        token_id: int,
    ) -> ir.Value:
        """Replace ``token_id`` positions with rows of ``features`` in order."""
        mask = op.Equal(input_ids, op.Constant(value_int=token_id))  # [B, S]
        mask_3d = op.Unsqueeze(mask, [-1])

        # Running index of the placeholder within the sequence.
        cumsum = op.CumSum(op.Cast(mask, to=7), 1)
        indices = op.Clip(op.Sub(cumsum, op.Constant(value_int=1)), op.Constant(value_int=0))

        # Pad the feature table with one zero row so Gather stays in-bounds
        # when the modality is absent (num_feature_tokens == 0); the mask
        # discards the padded row.
        pad_row = op.Expand(
            op.CastLike(0.0, features),
            op.Concat(
                op.Constant(value_ints=[1]),
                op.Shape(features, start=1, end=2),
                axis=0,
            ),
        )
        padded = op.Concat(features, pad_row, axis=0)
        gathered = op.Gather(padded, indices, axis=0)
        return op.Where(mask_3d, gathered, embeds)

    def forward(
        self,
        op: OpBuilder,
        input_ids: ir.Value,
        image_features: ir.Value,
        video_features: ir.Value | None = None,
    ):
        embeds = self.embed_tokens(op, input_ids)
        embeds = self._scatter(op, embeds, input_ids, image_features, self.image_token_id)
        if video_features is not None and self.video_token_id is not None:
            embeds = self._scatter(op, embeds, input_ids, video_features, self.video_token_id)
        return embeds

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        # HF stores the shared token table at the top level as
        # ``embed_tokens.weight``; keep it unchanged for the local Embedding.
        return {k: v for k, v in state_dict.items() if "embed_tokens" in k}


class Cosmos3EdgeVLModel(nn.Module):
    """NVIDIA Cosmos3-Edge vision-language model (3-model split).

    ``model_type: cosmos3_edge`` / ``Cosmos3EdgeForConditionalGeneration``.

    Builds three ONNX models:

    - **decoder**: squared-ReLU GQA text reasoner taking ``inputs_embeds`` and
      interleaved 3D M-RoPE ``position_ids [3, batch, seq]``.
    - **vision_encoder**: variable-resolution SigLIP2 tower + pixel-shuffle
      merger projector, driven by packed ``pixel_values [total_patches,
      patch_dim]`` and ``grid_thw [3]``. The *same* graph serves images
      (``grid_t == 1``) and videos (``grid_t == num_frames``).
    - **embedding**: token embedding + feature fusion at ``image_token_id``
      (19) and ``video_token_id`` (18).

    HuggingFace weight layout (single checkpoint, no ``language_model.``
    prefix):

    - ``model.visual.*`` → vision tower (SigLIP2)
    - ``model.projector.*`` → merger projector
    - ``embed_tokens.weight`` → embedding
    - ``layers.*`` / ``norm.weight`` / ``lm_head.weight`` → decoder
    - ``*k_norm_und_for_gen*`` → dropped (generator-tower artifact; see the
      module docstring)

    .. note::
       The Reasoner is validated numerically against a PyTorch transcription
       of the published ``transformers`` ``cosmos3_edge`` modeling code (see
       ``tests/_cosmos3_edge_reference.py``). The Generator/Action towers in
       the same checkpoint remain proprietary and are not reproduced.
    """

    default_task: str = "cosmos3-edge-vl"
    category: str = "Multimodal"

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.config = config
        self.decoder = _Cosmos3EdgeDecoderModel(config)
        self.vision_encoder = _Cosmos3EdgeVisionEncoderModel(config)
        self.embedding = _Cosmos3EdgeEmbeddingModel(config)

    def forward(self, op: OpBuilder, **kwargs):
        raise NotImplementedError(
            "Cosmos3EdgeVLModel uses Cosmos3EdgeVLTask, which builds the "
            "decoder, vision_encoder and embedding sub-models separately."
        )

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Route HF weights to the three sub-models and prefix by component.

        Sub-module parameters retain their root-relative names
        (``decoder.*`` / ``vision_encoder.*`` / ``embedding.*``), so the
        prefixed keys line up with each component graph's initializers when
        :meth:`ModelPackage.apply_weights` matches by name.
        """
        vision_sd: dict[str, torch.Tensor] = {}
        embedding_sd: dict[str, torch.Tensor] = {}
        decoder_sd: dict[str, torch.Tensor] = {}

        for key, value in state_dict.items():
            # Generator-tower key-norm is not used by the reasoner (see docstring).
            if "k_norm_und_for_gen" in key:
                continue
            if _DROPPED_UNIFIED_KEY_RE.search(key):
                continue
            if key.startswith(("model.visual.", "model.projector.")):
                vision_sd[key] = value
            elif "embed_tokens" in key:
                embedding_sd[key] = value
            else:
                decoder_sd[key] = value

        if self.config.tie_word_embeddings:
            embed_weight = next(iter(embedding_sd.values()), None)
            head_weight = decoder_sd.get("lm_head.weight")
            if embed_weight is None and head_weight is not None:
                embedding_sd["embed_tokens.weight"] = head_weight
            elif head_weight is None and embed_weight is not None:
                decoder_sd["lm_head.weight"] = embed_weight

        result: dict[str, torch.Tensor] = {}
        for k, v in self.vision_encoder.preprocess_weights(vision_sd).items():
            result[f"vision_encoder.{k}"] = v
        for k, v in self.embedding.preprocess_weights(embedding_sd).items():
            result[f"embedding.{k}"] = v
        for k, v in self.decoder.preprocess_weights(decoder_sd).items():
            result[f"decoder.{k}"] = v
        return result
