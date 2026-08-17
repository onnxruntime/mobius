# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Synthetic stage-by-stage parity tests for GLM-ASR."""

from __future__ import annotations

import numpy as np
import torch

from mobius._builder import build_from_module
from mobius._configs import GlmAsrConfig
from mobius._testing.ort_inference import OnnxModelSession
from mobius.models.glm_asr import GlmAsrForConditionalGeneration
from mobius.tasks import GlmAsrSpeechLanguageTask


def _tiny_hf_model():
    from transformers import (
        GlmAsrConfig as HfGlmAsrConfig,
        GlmAsrForConditionalGeneration as HfGlmAsrForConditionalGeneration,
        LlamaConfig,
    )

    text_config = LlamaConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=256,
        rms_norm_eps=1e-5,
        rope_theta=10_000.0,
        tie_word_embeddings=False,
    )
    audio_config = {
        "hidden_size": 64,
        "intermediate_size": 256,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "head_dim": 16,
        "hidden_act": "gelu",
        "max_position_embeddings": 256,
        "rope_parameters": {
            "rope_type": "default",
            "rope_theta": 10_000.0,
            "partial_rotary_factor": 0.5,
        },
        "num_mel_bins": 128,
    }
    config = HfGlmAsrConfig(
        audio_config=audio_config,
        text_config=text_config,
        audio_token_id=100,
        projector_hidden_act="gelu",
        tie_word_embeddings=False,
    )
    torch.manual_seed(0)
    return HfGlmAsrForConditionalGeneration(config).eval()


def test_glmasr_three_stage_synthetic_parity():
    """Execute audio -> embedding -> decoder and compare every stage with HF."""
    hf_model = _tiny_hf_model()
    config = GlmAsrConfig.from_transformers(hf_model.config)
    module = GlmAsrForConditionalGeneration(config)
    package = build_from_module(module, config, task=GlmAsrSpeechLanguageTask())

    # Released safetensors omit the outer HF ``model.`` prefix, so exercise
    # the same route here rather than bypassing preprocess_weights().
    state_dict = {
        name.removeprefix("model."): value.detach()
        for name, value in hf_model.state_dict().items()
    }
    package.apply_weights(module.preprocess_weights(state_dict))

    rng = np.random.default_rng(0)
    input_features = rng.standard_normal((1, 128, 32)).astype(np.float32)
    input_features_mask = np.ones((1, 32), dtype=np.int64)
    torch_features = torch.from_numpy(input_features)
    torch_mask = torch.from_numpy(input_features_mask)

    with torch.no_grad():
        hf_audio = hf_model.model.get_audio_features(
            torch_features,
            torch_mask,
            return_dict=True,
        ).pooler_output.numpy()

    audio_session = OnnxModelSession(package["audio_encoder"])
    audio_outputs = audio_session.run(
        {
            "input_features": input_features,
            "input_features_mask": input_features_mask,
        }
    )
    short_mask = np.zeros((1, 8), dtype=np.int64)
    short_mask[:, :1] = 1
    short_outputs = audio_session.run(
        {
            "input_features": input_features[:, :, :8],
            "input_features_mask": short_mask,
        }
    )
    audio_session.close()
    np.testing.assert_allclose(audio_outputs["audio_features"], hf_audio, rtol=1e-4, atol=1e-4)
    assert audio_outputs["audio_feature_lengths"].tolist() == [hf_audio.shape[0]]
    assert short_outputs["audio_features"].shape == (0, config.hidden_size)
    assert short_outputs["audio_feature_lengths"].tolist() == [0]

    input_ids = np.array([[1, 2, *([100] * hf_audio.shape[0]), 3]], dtype=np.int64)
    with torch.no_grad():
        hf_inputs_embeds = hf_model.model.get_input_embeddings()(
            torch.from_numpy(input_ids)
        )
        audio_mask = torch.from_numpy(input_ids == 100).unsqueeze(-1).expand_as(
            hf_inputs_embeds
        )
        hf_inputs_embeds = hf_inputs_embeds.masked_scatter(
            audio_mask,
            torch.from_numpy(hf_audio),
        )

    embedding_session = OnnxModelSession(package["embedding"])
    inputs_embeds = embedding_session.run(
        {
            "input_ids": input_ids,
            "audio_features": audio_outputs["audio_features"],
        }
    )["inputs_embeds"]
    embedding_session.close()
    np.testing.assert_allclose(
        inputs_embeds,
        hf_inputs_embeds.numpy(),
        rtol=1e-4,
        atol=1e-4,
    )

    sequence_length = input_ids.shape[1]
    attention_mask = np.ones_like(input_ids)
    position_ids = np.arange(sequence_length, dtype=np.int64)[None, :]
    with torch.no_grad():
        hf_logits = hf_model.model.language_model(
            inputs_embeds=hf_inputs_embeds,
            attention_mask=torch.from_numpy(attention_mask),
            position_ids=torch.from_numpy(position_ids),
            use_cache=False,
        ).last_hidden_state
        hf_logits = hf_model.lm_head(hf_logits).numpy()

    decoder_feeds = {
        "inputs_embeds": inputs_embeds,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
    }
    for layer in range(config.num_hidden_layers):
        for cache_kind in ("key", "value"):
            decoder_feeds[f"past_key_values.{layer}.{cache_kind}"] = np.zeros(
                (1, config.num_key_value_heads, 0, config.head_dim),
                dtype=np.float32,
            )
    decoder_session = OnnxModelSession(package["decoder"])
    decoder_outputs = decoder_session.run(decoder_feeds)
    decoder_session.close()
    np.testing.assert_allclose(
        decoder_outputs["logits"],
        hf_logits,
        rtol=2e-4,
        atol=2e-4,
    )
