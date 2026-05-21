# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Qwen3-ASR audio extractor hook (config lives under ``thinker_config``)."""

from __future__ import annotations

from mobius._configs._extractors import register_audio_hook


@register_audio_hook
def _qwen3_asr_audio(config, parent_config, model_type: str, fields: dict):
    thinker_source = parent_config or config
    hf_thinker_config = getattr(thinker_source, "thinker_config", None)
    if hf_thinker_config is None:
        return None

    tc = (
        hf_thinker_config
        if not isinstance(hf_thinker_config, dict)
        else type("TC", (), hf_thinker_config)()
    )
    hf_audio_config = getattr(tc, "audio_config", None)
    if hf_audio_config is not None:
        ac = (
            hf_audio_config
            if not isinstance(hf_audio_config, dict)
            else type("AC", (), hf_audio_config)()
        )
        fields.update(
            d_model=getattr(ac, "d_model", None),
            encoder_layers=getattr(ac, "encoder_layers", None),
            encoder_attention_heads=getattr(ac, "encoder_attention_heads", None),
            encoder_ffn_dim=getattr(ac, "encoder_ffn_dim", None),
            num_mel_bins=getattr(ac, "num_mel_bins", None),
            max_source_positions=getattr(ac, "max_source_positions", None),
            downsample_hidden_size=getattr(ac, "downsample_hidden_size", None),
            output_dim=getattr(ac, "output_dim", None),
            activation_function=getattr(ac, "activation_function", "gelu"),
            n_window=getattr(ac, "n_window", None),
            n_window_infer=getattr(ac, "n_window_infer", None),
        )
    # Special tokens from thinker config
    fields["audio_token_id"] = getattr(tc, "audio_token_id", None)
    fields["audio_start_token_id"] = getattr(tc, "audio_start_token_id", None)
    fields["audio_end_token_id"] = getattr(tc, "audio_end_token_id", None)
    fields["classify_num"] = getattr(tc, "classify_num", None)
    return None
