# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""NVIDIA Cosmos 3 model support.

Currently supports the **text reasoner backbone** of the ``cosmos3_edge``
checkpoint (``nvidia/Cosmos3-Edge``, ``Cosmos3EdgeForConditionalGeneration``).

Cosmos3-Edge is a vision-language model whose language tower is a standard
grouped-query-attention decoder with two Cosmos-specific traits:

- **Non-gated feed-forward network** — ``down_proj(relu2(up_proj(x)))`` using
  a squared-ReLU activation (``hidden_act="relu2"``), rather than the
  GLU-style gated MLP used by Llama/Qwen. This maps onto :class:`FCMLP`.
- **3D multimodal RoPE** (``mrope_section=[24, 20, 20]``). For text-only
  inference the three sections are identical, so it reduces to standard 1D
  RoPE (same simplification used by the Qwen-VL text decoders).

The HuggingFace weights use ``self_attn.to_{q,k,v,out}`` projection names and
place the text tower at the top level (``layers.*``, ``embed_tokens``,
``norm``, ``lm_head``) with the vision encoder and projector under
``model.visual.*`` / ``model.projector.*``. :meth:`preprocess_weights` renames
the projections, prefixes the text tower with ``model.`` and drops the
vision/projector weights.

The ``k_norm_und_for_gen`` per-layer key-norm weight is an artifact of the
two-tower (Mixture-of-Transformers) design: it normalizes the *understanding*
tower's keys for consumption by the *generator* (diffusion) tower, and is not
applied in the reasoner's own causal self-attention. It is therefore dropped
for the standalone text decoder.

The ``cosmos3_omni`` variants (``nvidia/Cosmos3-Nano`` / ``-Super``) are
two-tower diffusion world models exported as diffusers pipelines and are out
of scope for this decoder-only path.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mobius._configs import ArchitectureConfig
from mobius.components import FCMLP
from mobius.models.base import CausalLMModel

if TYPE_CHECKING:
    import torch


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

            new_key = (
                key.replace("self_attn.to_q.", "self_attn.q_proj.")
                .replace("self_attn.to_k.", "self_attn.k_proj.")
                .replace("self_attn.to_v.", "self_attn.v_proj.")
                .replace("self_attn.to_out.", "self_attn.o_proj.")
            )
            # The text tower is stored at the top level; the mobius backbone
            # nests it under ``model.``. ``lm_head`` stays at the top level.
            if new_key == "lm_head.weight":
                pass
            elif new_key.startswith(("layers.", "embed_tokens.")) or new_key == "norm.weight":
                new_key = f"model.{new_key}"

            renamed[new_key] = value
        return super().preprocess_weights(renamed)
