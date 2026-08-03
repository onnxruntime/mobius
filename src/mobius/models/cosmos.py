# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""NVIDIA Cosmos 3 model support.

Supports the ``cosmos3_edge`` checkpoint (``nvidia/Cosmos3-Edge``,
``Cosmos3EdgeForConditionalGeneration``) both as a full vision-language model
(:class:`Cosmos3EdgeVLModel`, 3-model onnxruntime-genai split) and as a
standalone text reasoner (:class:`Cosmos3EdgeTextModel`).

Cosmos3-Edge is a LLaVA-style VLM built from three towers:

- **Text reasoner** — a grouped-query-attention decoder with two
  Cosmos-specific traits:

  - **Non-gated feed-forward network** — ``down_proj(relu2(up_proj(x)))`` using
    a squared-ReLU activation (``hidden_act="relu2"``), rather than the
    GLU-style gated MLP used by Llama/Qwen. This maps onto :class:`FCMLP`.
  - **3D multimodal RoPE** (``mrope_section=[24, 20, 20]``). For text-only
    inference the three sections use the same positions, so it reduces to
    standard 1D RoPE (same simplification used by the Qwen-VL text decoders).

- **Vision encoder** — a SigLIP-style patch-embedding + transformer tower
  (``model.visual.*``).
- **Merger projector** — a pixel-shuffle projector
  (:class:`Cosmos3EdgeMultiModalProjector`, ``model.projector.*``) that merges
  each 2x2 patch block and projects to the text hidden size.

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
   NVIDIA does not publish modeling code for ``cosmos3_edge`` (it is not in
   ``transformers`` and the repo ships no remote-code module), so exact
   pixel-shuffle ordering and numerical parity are unverifiable. These builds
   are validated at graph-construction (L1) confidence only.

The ``cosmos3_omni`` variants (``nvidia/Cosmos3-Nano`` / ``-Super``) are
two-tower diffusion world models exported as diffusers pipelines and are
tracked separately.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from onnxscript import OpBuilder, nn

from mobius._configs import ArchitectureConfig
from mobius.components import (
    FCMLP,
    Cosmos3EdgeMultiModalProjector,
    Embedding,
    Linear,
    VisionModel,
)
from mobius.models.base import CausalLMModel, TextModel

if TYPE_CHECKING:
    import onnx_ir as ir
    import torch


def _rename_cosmos_text_key(key: str) -> str:
    """Rename a Cosmos3-Edge attention projection key to mobius conventions."""
    return (
        key.replace("self_attn.to_q.", "self_attn.q_proj.")
        .replace("self_attn.to_k.", "self_attn.k_proj.")
        .replace("self_attn.to_v.", "self_attn.v_proj.")
        .replace("self_attn.to_out.", "self_attn.o_proj.")
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
    """Cosmos3-Edge vision encoder: SigLIP tower + pixel-shuffle projector.

    ``pixel_values [B, 3, H, W]`` → SigLIP encoder → ``[B, num_patches, D]``
    → :class:`Cosmos3EdgeMultiModalProjector` → ``[B, num_merged, text_hidden]``.
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        vc = config.vision
        assert vc is not None, "Cosmos3-Edge requires a VisionConfig"
        assert vc.image_size is not None and vc.patch_size is not None
        assert vc.hidden_size is not None
        assert vc.projector_intermediate_size is not None, (
            "Cosmos3-Edge projector requires projector_intermediate_size"
        )
        self.vision_tower = VisionModel(config)
        # Fixed square patch grid (image_size // patch_size), e.g. 256 -> 16.
        grid_size = vc.image_size // vc.patch_size
        self.multi_modal_projector = Cosmos3EdgeMultiModalProjector(
            vision_hidden_size=vc.hidden_size,
            text_hidden_size=config.hidden_size,
            intermediate_size=vc.projector_intermediate_size,
            grid_size=grid_size,
            spatial_merge_size=vc.spatial_merge_size,
            norm_eps=vc.norm_eps,
        )

    def forward(self, op: OpBuilder, pixel_values: ir.Value):
        vision_features = self.vision_tower(op, pixel_values)
        return self.multi_modal_projector(op, vision_features)

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        # state_dict here is the vision-routed slice: ``model.visual.*`` (SigLIP
        # tower) and ``model.projector.*`` (merger projector).
        renamed: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            if key.startswith("model.visual."):
                new_key = "vision_tower.vision_model." + key[len("model.visual.") :]
                # SigLIP MLP: fc1/fc2 -> up_proj/down_proj (FCMLP convention).
                new_key = new_key.replace(".mlp.fc1.", ".mlp.up_proj.").replace(
                    ".mlp.fc2.", ".mlp.down_proj."
                )
                renamed[new_key] = value
            elif key.startswith("model.projector."):
                new_key = "multi_modal_projector." + key[len("model.projector.") :]
                renamed[new_key] = value
        return renamed


class _Cosmos3EdgeEmbeddingModel(nn.Module):
    """Cosmos3-Edge embedding: token lookup + image feature fusion.

    Scatters projected vision features into the text embedding sequence at
    ``image_token_id`` (19) positions, matching the LLaVA embedding contract.
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = Embedding(
            config.vocab_size, config.hidden_size, config.pad_token_id
        )
        self.image_token_id = config.image_token_id or 0

    def forward(self, op: OpBuilder, input_ids: ir.Value, image_features: ir.Value):
        text_embeds = self.embed_tokens(op, input_ids)

        image_mask = op.Equal(input_ids, op.Constant(value_int=self.image_token_id))
        image_mask_3d = op.Unsqueeze(image_mask, [-1])

        mask_int = op.Cast(image_mask, to=7)
        cumsum = op.CumSum(mask_int, 1)
        indices = op.Sub(cumsum, op.Constant(value_int=1))
        indices = op.Clip(indices, op.Constant(value_int=0))

        # Pad image_features with one zero row so Gather stays in-bounds for
        # text-only input (num_image_tokens == 0); the mask discards it.
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
        # HF stores the shared token table at the top level as
        # ``embed_tokens.weight``; keep it unchanged for the local Embedding.
        return {k: v for k, v in state_dict.items() if "embed_tokens" in k}


class Cosmos3EdgeVLModel(nn.Module):
    """NVIDIA Cosmos3-Edge vision-language model (3-model split).

    ``model_type: cosmos3_edge`` / ``Cosmos3EdgeForConditionalGeneration``.

    Builds three ONNX models for onnxruntime-genai deployment:

    - **decoder**: squared-ReLU GQA text reasoner taking ``inputs_embeds``.
    - **vision_encoder**: SigLIP vision tower + pixel-shuffle merger projector.
    - **embedding**: token embedding + image feature fusion at
      ``image_token_id`` (19).

    HuggingFace weight layout (single checkpoint, no ``language_model.``
    prefix):

    - ``model.visual.*`` → vision tower (SigLIP)
    - ``model.projector.*`` → merger projector
    - ``embed_tokens.weight`` → embedding
    - ``layers.*`` / ``norm.weight`` / ``lm_head.weight`` → decoder
    - ``*k_norm_und_for_gen*`` → dropped (generator-tower artifact; see the
      module docstring)

    .. note::
       NVIDIA does not publish modeling code for ``cosmos3_edge`` (it is not in
       ``transformers`` and the repo ships no remote-code module), so the exact
       pixel-shuffle ordering and numerical parity are unverifiable. This build
       is validated at graph-construction (L1) confidence only.
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
            if key.startswith(("model.visual.", "model.projector.")):
                vision_sd[key] = value
            elif "embed_tokens" in key:
                embedding_sd[key] = value
            else:
                decoder_sd[key] = value

        result: dict[str, torch.Tensor] = {}
        for k, v in self.vision_encoder.preprocess_weights(vision_sd).items():
            result[f"vision_encoder.{k}"] = v
        for k, v in self.embedding.preprocess_weights(embedding_sd).items():
            result[f"embedding.{k}"] = v
        for k, v in self.decoder.preprocess_weights(decoder_sd).items():
            result[f"decoder.{k}"] = v
        return result
