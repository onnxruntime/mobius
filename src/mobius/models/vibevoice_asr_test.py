# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Synthetic parity tests against Transformers' VibeVoice-ASR implementation."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from mobius._builder import build_from_module
from mobius._configs import VibeVoiceASRConfig
from mobius._model_package import ModelPackage
from mobius._testing.ort_inference import OnnxModelSession
from mobius.models.vibevoice_asr import VibeVoiceASRForConditionalGeneration
from mobius.tasks import VibeVoiceASRTask


def _make_tiny_hf_config():
    configuration = pytest.importorskip(
        "transformers.models.vibevoice_asr.configuration_vibevoice_asr"
    )
    tokenizer = {
        "model_type": "vibevoice_acoustic_tokenizer_encoder",
        "channels": 1,
        "hidden_size": 4,
        "kernel_size": 3,
        "num_filters": 4,
        "downsampling_ratios": [2, 2],
        "depths": [1, 1, 1],
        "ffn_expansion": 2,
        "hidden_act": "gelu",
        "rms_norm_eps": 1e-5,
        "layer_scale_init_value": 1e-6,
        "vae_std": 0.625,
    }
    return configuration.VibeVoiceAsrConfig(
        acoustic_tokenizer_encoder_config=tokenizer,
        semantic_tokenizer_encoder_config={**tokenizer, "hidden_size": 6},
        text_config={
            "model_type": "qwen2",
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "vocab_size": 64,
            "max_position_embeddings": 128,
            "rms_norm_eps": 1e-6,
            "hidden_act": "silu",
            "rope_theta": 10_000.0,
            "tie_word_embeddings": False,
        },
        audio_token_id=60,
        audio_bos_token_id=61,
        audio_eos_token_id=62,
        acoustic_tokenizer_chunk_size=8,
    )


def _make_models():
    modeling = pytest.importorskip("transformers.models.vibevoice_asr.modeling_vibevoice_asr")
    torch.manual_seed(7)
    hf_config = _make_tiny_hf_config()
    hf_model = modeling.VibeVoiceAsrForConditionalGeneration(hf_config).float().eval()
    config = VibeVoiceASRConfig.from_transformers(
        hf_config.text_config, parent_config=hf_config
    )
    module = VibeVoiceASRForConditionalGeneration(config)
    package = build_from_module(module, config, task=VibeVoiceASRTask())
    package.apply_weights(module.preprocess_weights(hf_model.state_dict()))
    return hf_model, module, package


def _run(
    package: ModelPackage, name: str, feeds: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    session = OnnxModelSession(package[name])
    try:
        return session.run(feeds)
    finally:
        session.close()


def _empty_conv_cache(module, batch: int) -> dict[str, np.ndarray]:
    return {
        f"past_conv.{index}": np.zeros((batch, channels, left_pad), dtype=np.float32)
        for index, (channels, left_pad) in enumerate(module.cache_specs)
    }


def _run_encoder_chunks(
    package: ModelPackage,
    name: str,
    module,
    input_values: np.ndarray,
    chunk_samples: int,
) -> np.ndarray:
    cache = _empty_conv_cache(module, input_values.shape[0])
    latents: list[np.ndarray] = []
    for start in range(0, input_values.shape[-1], chunk_samples):
        outputs = _run(
            package,
            name,
            {
                "input_values": input_values[:, :, start : start + chunk_samples],
                **cache,
            },
        )
        latents.append(outputs["audio_latents"])
        cache = {
            output.replace("present_conv.", "past_conv."): value
            for output, value in outputs.items()
            if output.startswith("present_conv.")
        }
    return np.concatenate(latents, axis=1)


def test_staged_asr_matches_transformers_with_batch_chunking_and_left_padding():
    """Match the pinned Transformers ASR source for every inference stage."""
    hf_model, module, package = _make_models()
    input_values = np.random.default_rng(4).standard_normal((2, 1, 16), dtype=np.float32)
    padding_mask = np.array(
        [[True] * 13 + [False] * 3, [True] * 9 + [False] * 7],
        dtype=np.bool_,
    )
    with torch.no_grad():
        torch.manual_seed(11)
        source_features = hf_model.model.get_audio_features(
            torch.from_numpy(input_values),
            padding_mask=torch.from_numpy(padding_mask),
            acoustic_tokenizer_chunk_size=8,
        ).pooler_output.numpy()

    acoustic = _run_encoder_chunks(
        package, "acoustic_encoder", module.acoustic_encoder, input_values, chunk_samples=8
    )
    semantic = _run_encoder_chunks(
        package, "semantic_encoder", module.semantic_encoder, input_values, chunk_samples=8
    )
    torch.manual_seed(11)
    acoustic_noise_scale = torch.randn(2).numpy()
    acoustic_latent_noise = torch.randn(*acoustic.shape).numpy()
    connector = _run(
        package,
        "connectors",
        {
            "acoustic_latents": acoustic,
            "semantic_latents": semantic,
            "padding_mask": padding_mask,
            "acoustic_noise_scale": acoustic_noise_scale,
            "acoustic_latent_noise": acoustic_latent_noise,
        },
    )
    np.testing.assert_allclose(
        connector["audio_features"], source_features, rtol=2e-4, atol=2e-5
    )
    assert connector["audio_feature_lengths"].tolist() == [4, 3]

    input_ids = np.array(
        [[1, 60, 60, 60, 60, 2], [0, 0, 1, 60, 60, 60]],
        dtype=np.int64,
    )
    attention_mask = (input_ids != 0).astype(np.int64)
    position_ids = attention_mask.cumsum(axis=-1, dtype=np.int64) - 1
    position_ids[attention_mask == 0] = 0
    with torch.no_grad():
        source_embeds = hf_model.model.get_input_embeddings()(torch.from_numpy(input_ids))
        source_embeds.masked_scatter_(
            torch.from_numpy(input_ids == 60).unsqueeze(-1),
            torch.from_numpy(source_features),
        )
        source_logits = hf_model.model.language_model(
            inputs_embeds=source_embeds,
            attention_mask=torch.from_numpy(attention_mask),
            position_ids=torch.from_numpy(position_ids),
            use_cache=True,
        ).last_hidden_state
        source_logits = hf_model.lm_head(source_logits).numpy()
    embedded = _run(
        package,
        "embedding",
        {"input_ids": input_ids, "audio_features": connector["audio_features"]},
    )["inputs_embeds"]
    np.testing.assert_allclose(embedded, source_embeds.numpy(), rtol=2e-4, atol=2e-5)
    empty_kv = np.zeros((2, 1, 0, 8), dtype=np.float32)
    decoded = _run(
        package,
        "decoder",
        {
            "inputs_embeds": embedded,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "past_key_values.0.key": empty_kv,
            "past_key_values.0.value": empty_kv,
        },
    )
    np.testing.assert_allclose(decoded["logits"], source_logits, rtol=3e-4, atol=3e-5)
