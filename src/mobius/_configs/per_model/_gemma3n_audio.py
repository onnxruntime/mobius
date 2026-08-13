# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Gemma 3n audio extractor hook (uses :class:`Gemma3nAudioConfig`).

The HF ``Gemma3nAudioConfig`` field names are already the ones the encoder
components want (``conf_*`` for the Conformer stack, ``sscp_*`` for the
convolutional subsampler), so this is a near-verbatim copy rather than a
remapping.  Returning a fully-formed dict short-circuits the generic audio
extractor, whose field set (``attention_dim``, ``num_mel_bins``, ...) does not
apply to a USM encoder.
"""

from __future__ import annotations

from mobius._configs._extractors import register_audio_hook
from mobius._configs._sub_configs import Gemma3nAudioConfig

_GEMMA3N_TYPES = ("gemma3n", "gemma3n_text", "gemma3n_audio")


def _as_nested_ints(value):
    """Normalise a list-of-pairs config field (kernel/stride sizes)."""
    if value is None:
        return None
    return [[int(v) for v in pair] for pair in value]


@register_audio_hook
def _gemma3n_audio(config, parent_config, model_type: str, fields: dict):
    # No decorator filter: this hook must also fire when build() has resolved
    # to the text sub-config, whose model_type is "gemma3n_text" while the
    # parent is "gemma3n". The body's predicate covers both cases.
    #
    # Read the parent's model_type off ``parent_config`` rather than the
    # ``parent_config or config`` composite: falling back to *config* would
    # make the hook fire for any dispatched model_type whenever the config
    # itself happens to be a gemma3n one.
    parent_model_type = getattr(parent_config, "model_type", "") if parent_config else ""
    if model_type not in _GEMMA3N_TYPES and parent_model_type != "gemma3n":
        return None
    composite = parent_config or config
    hf_audio = getattr(composite, "audio_config", None)
    if hf_audio is None:
        return None
    ac = hf_audio if not isinstance(hf_audio, dict) else type("AC", (), hf_audio)()

    def _get(name, default=None):
        return getattr(ac, name, default)

    channels = _get("sscp_conv_channel_size")
    return {
        "audio": Gemma3nAudioConfig(
            hidden_size=_get("hidden_size", 1536),
            conf_num_hidden_layers=_get("conf_num_hidden_layers", 12),
            conf_num_attention_heads=_get("conf_num_attention_heads", 8),
            conf_attention_chunk_size=_get("conf_attention_chunk_size", 12),
            conf_attention_context_left=_get("conf_attention_context_left", 13),
            conf_attention_context_right=_get("conf_attention_context_right", 0),
            conf_attention_logit_cap=float(_get("conf_attention_logit_cap", 50.0)),
            conf_conv_kernel_size=_get("conf_conv_kernel_size", 5),
            conf_reduction_factor=_get("conf_reduction_factor", 4),
            conf_residual_weight=float(_get("conf_residual_weight", 0.5)),
            input_feat_size=_get("input_feat_size", 128),
            sscp_conv_channel_size=(list(channels) if channels is not None else None),
            sscp_conv_kernel_size=_as_nested_ints(_get("sscp_conv_kernel_size")),
            sscp_conv_stride_size=_as_nested_ints(_get("sscp_conv_stride_size")),
            sscp_conv_group_norm_eps=float(_get("sscp_conv_group_norm_eps", 1e-3)),
            gradient_clipping=float(_get("gradient_clipping", 1e10)),
            rms_norm_eps=_get("rms_norm_eps", 1e-6),
            vocab_offset=_get("vocab_offset"),
            vocab_size=_get("vocab_size"),
            audio_token_id=getattr(composite, "audio_token_id", None),
        )
    }
