# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""EmbeddingGemma 2 reuses Gemma4 media towers, not its causal text model."""

from __future__ import annotations

from mobius._configs._base import _as_attribute_config
from mobius._configs._extractors import register_audio_hook, register_vision_hook
from mobius._configs._sub_configs import Gemma4AudioConfig


@register_audio_hook("embedding_gemma2", "embedding_gemma2_text")
def _embedding_gemma2_audio(config, parent_config, model_type, fields):
    parent = parent_config or config
    ac = _as_attribute_config(getattr(parent, "audio_config", None))
    if ac is None:
        return None
    expected = {
        "conv_kernel_size": 5,
        "attention_chunk_size": 12,
        "attention_context_left": 13,
        "attention_context_right": 0,
        "use_clipped_linears": True,
        "residual_weight": 0.5,
    }
    for name, value in expected.items():
        if getattr(ac, name) != value:
            raise ValueError(
                f"EmbeddingGemma2 audio requires {name}={value!r}, got "
                f"{getattr(ac, name)!r}; use the pinned google/embeddinggemma-2 config."
            )
    return {
        "audio": Gemma4AudioConfig(
            num_layers=ac.num_hidden_layers,
            hidden_size=ac.hidden_size,
            attention_heads=ac.num_attention_heads,
            input_size=ac.subsampling_conv_channels[0],
            rms_norm_eps=ac.rms_norm_eps,
            subsampling_conv_channels=list(ac.subsampling_conv_channels),
            output_proj_dims=ac.output_proj_dims,
            audio_token_id=parent.audio_token_id,
        )
    }


@register_vision_hook("embedding_gemma2", "embedding_gemma2_text")
def _embedding_gemma2_vision(config, parent_config, model_type, fields):
    parent = parent_config or config
    vc = _as_attribute_config(getattr(parent, "vision_config", None))
    if vc is None:
        return None
    rope = vc.rope_parameters
    fields.update(
        pooling_kernel_size=vc.pooling_kernel_size,
        hidden_act=vc.hidden_activation,
        norm_eps=vc.rms_norm_eps,
        rope_theta=rope["rope_theta"] if isinstance(rope, dict) else rope.rope_theta,
    )
    return None
