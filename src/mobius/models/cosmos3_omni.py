# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""NVIDIA Cosmos3-Omni understanding tower ("Reasoner").

Cosmos3-Omni (``nvidia/Cosmos3-Nano`` 16B, ``nvidia/Cosmos3-Super`` 64B and
their Text2Image / Image2Video / Policy-DROID variants) is a unified
world-foundation model that ships a **single** checkpoint containing three
towers:

1. **Reasoner** (understanding tower) — a vision-language decoder that is
   *architecturally identical to Qwen3-VL* (the HuggingFace modeling code is
   literally ``class Cosmos3OmniForConditionalGeneration(Qwen3VLForConditionalGeneration)``).
   Its ``text_config.model_type`` is ``qwen3_vl_text`` and its
   ``vision_config.model_type`` is ``qwen3_vl``.
2. **Generator** — a rectified-flow diffusion transformer
   (``Cosmos3OmniTransformer``) for image/video synthesis.
3. **Sound / Action** towers — cross-modal adapters for audio generation and
   robot-action prediction.

This module exports **only the Reasoner** as an ONNX-deployable
vision-language model, reusing the existing Qwen3-VL 3-model split
(``decoder`` + ``vision_encoder`` + ``embedding``).  The generator, sound and
action parameters that share the unified checkpoint are dropped during weight
preprocessing (mirroring HuggingFace's
``_keys_to_ignore_on_load_unexpected``).

The published checkpoint stores the Reasoner weights under NVIDIA's native
(diffusers-style) parameter names rather than the HuggingFace Qwen3-VL names.
:meth:`Cosmos3OmniReasonerModel.preprocess_weights` translates the native
names into the HuggingFace Qwen3-VL layout and then defers to the base
Qwen3-VL routing logic.
"""

from __future__ import annotations

import re

import torch

from mobius.models.qwen_vl import Qwen3VL3ModelCausalLMModel

# --- Unified-checkpoint keys that belong to the Generator / Sound / Action
#     towers (NOT the Reasoner).  Dropped when exporting the understanding
#     tower.  Mirrors transformers'
#     ``_COSMOS3_DROPPED_UNIFIED_CHECKPOINT_KEYS``.
_DROPPED_UNIFIED_KEY_PATTERNS = [
    # Generator (image/video diffusion) MoT expert branch + joint attention
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
    # Sound tower
    r"^audio_proj_out\.",
    r"^audio_proj_in\.",
    r"^audio_modality_embed$",
    # Action tower
    r"^action_proj_out\.",
    r"^action_proj_in\.",
    r"^action_modality_embed$",
]

_DROPPED_UNIFIED_KEY_RE = re.compile("|".join(_DROPPED_UNIFIED_KEY_PATTERNS))

# --- Reasoner self-attention: native (diffusers-style) -> HuggingFace Qwen3-VL
# NOTE: order matters — the loop below applies the first matching rename and
# stops, so the more specific ``to_out.0.`` (diffusers wraps the output
# projection in an ``nn.Sequential`` [Linear, Dropout]) must precede the plain
# ``to_out.`` entry.  Both collapse to the single HF ``o_proj`` Linear.
_TEXT_ATTN_RENAMES = {
    ".self_attn.to_out.0.": ".self_attn.o_proj.",
    ".self_attn.to_q.": ".self_attn.q_proj.",
    ".self_attn.to_k.": ".self_attn.k_proj.",
    ".self_attn.to_v.": ".self_attn.v_proj.",
    ".self_attn.to_out.": ".self_attn.o_proj.",
    ".self_attn.norm_q.": ".self_attn.q_norm.",
    ".self_attn.norm_k.": ".self_attn.k_norm.",
}

# --- Vision-tower top-level keys in the native checkpoint.  These are exactly
#     the Qwen3-VL vision submodule names *without* the ``model.visual.``
#     prefix that HuggingFace uses.
_VISION_TOP_LEVEL_PREFIXES = (
    "blocks.",
    "patch_embed.",
    "pos_embed",
    "merger.",
    "deepstack_merger_list.",
)


class Cosmos3OmniReasonerModel(Qwen3VL3ModelCausalLMModel):
    """Cosmos3-Omni understanding tower (Qwen3-VL 3-model split).

    Identical graph to :class:`Qwen3VL3ModelCausalLMModel`; only weight
    preprocessing differs because the unified Cosmos3 checkpoint uses NVIDIA's
    native parameter names and carries the extra Generator / Sound / Action
    towers.

    .. note::
       **DeepStack is preserved.** The Qwen3-VL 3-model split packs the
       intermediate maps (from ``deepstack_visual_indexes`` — ``[8, 16, 24]``
       for Cosmos3) into ``image_features``. The embedding model scatters and
       flattens them into ORT GenAI's rank-3 ``per_layer_inputs`` contract,
       and the decoder restores and injects one map into each of its first
       ``D`` layers — matching HuggingFace's per-layer injection. The 18
       ``deepstack_merger_list.*`` weights are therefore exported and routed
       to ``vision_encoder``.
    """

    default_task: str = "qwen-vl"
    category: str = "Multimodal"

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Translate the native Cosmos3 checkpoint to the Qwen3-VL layout.

        Steps:

        1. Drop Generator / Sound / Action tower parameters.
        2. Reasoner text tower -> ``model.language_model.*`` with
           ``to_{q,k,v,out}`` -> ``{q,k,v,o}_proj`` and
           ``norm_{q,k}`` -> ``{q,k}_norm``.
        3. Vision tower -> ``model.visual.*`` (names otherwise unchanged).

        The translated HuggingFace-style dict is then handed to the base
        Qwen3-VL routing (which prefixes each sub-model initializer with
        ``decoder.`` / ``vision_encoder.`` / ``embedding.`` and handles the
        ``mlp.linear_fc1/fc2`` -> ``up_proj/down_proj`` vision rename).
        """
        hf_style: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            # 1. Drop the non-Reasoner towers.
            if _DROPPED_UNIFIED_KEY_RE.search(key):
                continue

            # 2. Vision tower: prepend the HF ``model.visual.`` prefix.
            if key.startswith(_VISION_TOP_LEVEL_PREFIXES):
                hf_style[f"model.visual.{key}"] = value
                continue

            # 3. Text (Reasoner) tower.
            renamed = key
            for src, dst in _TEXT_ATTN_RENAMES.items():
                if src in renamed:
                    renamed = renamed.replace(src, dst)
                    break
            hf_style[f"model.language_model.{renamed}"] = value

        return super().preprocess_weights(hf_style)


__all__ = ["Cosmos3OmniReasonerModel"]
