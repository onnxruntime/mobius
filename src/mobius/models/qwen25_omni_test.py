# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

from types import SimpleNamespace

import torch

from mobius._configs import ArchitectureConfig
from mobius.models.qwen25_omni import Qwen25OmniThinkerForConditionalGeneration


def _hf_config():
    text = SimpleNamespace(
        model_type="qwen2_5_omni_text",
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        hidden_act="silu",
        rms_norm_eps=1e-6,
        max_position_embeddings=128,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 1_000_000.0,
            "mrope_section": [4, 2, 2],
        },
        tie_word_embeddings=False,
    )
    thinker = SimpleNamespace(
        text_config=text,
        audio_config=SimpleNamespace(
            d_model=64,
            encoder_layers=2,
            encoder_attention_heads=4,
            encoder_ffn_dim=128,
            num_mel_bins=32,
            max_source_positions=128,
            n_window=8,
            output_dim=64,
        ),
        vision_config=SimpleNamespace(
            hidden_size=64,
            intermediate_size=128,
            depth=2,
            num_heads=4,
            patch_size=14,
            temporal_patch_size=2,
            in_channels=3,
            out_hidden_size=64,
            spatial_merge_size=2,
            fullatt_block_indexes=[0],
            window_size=112,
        ),
        audio_token_id=100,
        image_token_id=101,
        video_token_id=102,
    )
    return text, SimpleNamespace(thinker_config=thinker, tie_word_embeddings=False)


def test_qwen25_omni_extracts_nested_thinker_config():
    text, parent = _hf_config()
    config = ArchitectureConfig.from_transformers(text, parent_config=parent)

    assert config.attn_qkv_bias
    assert config.audio is not None
    assert config.audio.encoder_ffn_dim == 128
    assert config.audio.audio_token_id == 100
    assert config.vision is not None
    assert config.vision.hidden_size == 64
    assert config.image_token_id == 101
    assert config.video_token_id == 102


def test_qwen25_omni_preprocess_weights_routes_thinker_components():
    text, parent = _hf_config()
    config = ArchitectureConfig.from_transformers(text, parent_config=parent)
    model = Qwen25OmniThinkerForConditionalGeneration(config)
    weight = torch.randn(1)

    processed = model.preprocess_weights(
        {
            "thinker.audio_tower.conv1.weight": weight,
            "thinker.audio_tower.audio_bos_eos_token.weight": weight,
            "thinker.visual.blocks.0.attn.q.weight": weight,
            "thinker.visual.merger.mlp.0.weight": weight,
            "thinker.model.embed_tokens.weight": weight,
            "thinker.model.layers.0.self_attn.q_proj.bias": weight,
            "thinker.lm_head.weight": weight,
            "talker.model.layers.0.weight": weight,
            "token2wav.dit.weight": weight,
        }
    )

    assert set(processed) == {
        "audio_encoder.conv1.weight",
        "vision_encoder.visual.blocks.0.attn.q.weight",
        "vision_encoder.visual.merger.mlp_0.weight",
        "embedding.embed_tokens.weight",
        "decoder.layers.0.self_attn.q_proj.bias",
        "decoder.lm_head.weight",
    }
