# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Synthetic numerical tests for every VibeVoice ONNX stage."""

from __future__ import annotations

import dataclasses

import numpy as np
import onnx_ir as ir
import pytest
import torch

from mobius._builder import build_from_module
from mobius._configs import (
    VibeVoiceConfig,
    VibeVoiceDiffusionConfig,
    VibeVoiceTokenizerConfig,
)
from mobius._model_package import ModelPackage
from mobius._pipeline_contract import (
    optional_input_contract,
    requires_arbitrary_attention_mask,
)
from mobius._registry import registry
from mobius._testing.ort_inference import OnnxModelSession
from mobius.models.vibevoice import (
    VIBEVOICE_MODEL_ID,
    VIBEVOICE_REVISION,
    VibeVoiceForConditionalGeneration,
)
from mobius.tasks import VibeVoiceTask


def _make_tiny_hf_config():
    configuration = pytest.importorskip(
        "transformers.models.vibevoice.configuration_vibevoice"
    )
    tokenizer = {
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
    return configuration.VibeVoiceConfig(
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
            "rope_parameters": {"rope_type": "default", "rope_theta": 10_000.0},
            "tie_word_embeddings": True,
        },
        audio_config={
            **tokenizer,
            "model_type": "vibevoice_acoustic_tokenizer",
        },
        semantic_model_config={
            **tokenizer,
            "hidden_size": 6,
            "model_type": "vibevoice_acoustic_tokenizer_encoder",
        },
        diffusion_head_config={
            "hidden_size": 16,
            "intermediate_size": 32,
            "latent_size": 4,
            "num_hidden_layers": 1,
            "rms_norm_eps": 1e-5,
            "hidden_act": "silu",
            "frequency_embedding_size": 8,
            "diffusion_max_period": 10_000,
            "mlp_bias": False,
        },
        audio_bos_token_id=61,
        audio_eos_token_id=62,
        audio_token_id=60,
        eos_token_id=63,
        pad_token_id=63,
    )


def _make_tiny_models():
    modeling = pytest.importorskip("transformers.models.vibevoice.modeling_vibevoice")
    torch.manual_seed(7)
    hf_config = _make_tiny_hf_config()
    hf_model = modeling.VibeVoiceForConditionalGeneration(hf_config).float().eval()
    with torch.no_grad():
        hf_model.model.latent_scaling_factor.fill_(0.375)
        hf_model.model.latent_bias_factor.fill_(-0.125)
    config = VibeVoiceConfig.from_transformers(
        hf_config.text_config,
        parent_config=hf_config,
    )
    module = VibeVoiceForConditionalGeneration(config)
    package = VibeVoiceTask().build(module, config)
    package.apply_weights(module.preprocess_weights(hf_model.state_dict()))
    return hf_model, module, package


def _run(package: ModelPackage, name: str, feeds: dict[str, np.ndarray]):
    session = OnnxModelSession(package[name])
    try:
        return session.run(feeds)
    finally:
        session.close()


def _zero_conv_cache(
    module,
    *,
    batch: int = 1,
) -> dict[str, np.ndarray]:
    return {
        f"past_conv.{index}": np.zeros((batch, channels, left_pad), dtype=np.float32)
        for index, (channels, left_pad) in enumerate(module.cache_specs)
    }


def run_vibevoice_synthetic_stage_parity():
    """Every ONNX boundary matches HF, including ONNX-to-ONNX audio feedback."""
    hf_model, module, package = _make_tiny_models()
    rng = np.random.default_rng(1)

    input_values = rng.standard_normal((2, 1, 16)).astype(np.float32)
    padding_mask = np.array(
        [[True] * 16, [True] * 12 + [False] * 4],
        dtype=np.bool_,
    )
    with torch.no_grad():
        raw = (
            hf_model.model.audio_tower.encode(
                torch.from_numpy(input_values),
                sample=False,
            )
            .latents.cpu()
            .numpy()
        )
    expected_valid = np.concatenate([raw[0, :4], raw[1, :3]], axis=0)
    encoder_outputs = _run(
        package,
        "audio_encoder",
        {
            "input_values": input_values,
            "padding_mask": padding_mask,
            "sample_noise": np.zeros(2, dtype=np.float32),
            "latent_noise": np.zeros_like(raw),
        },
    )
    np.testing.assert_allclose(
        encoder_outputs["audio_latents"],
        expected_valid,
        atol=1e-5,
        rtol=1e-5,
    )

    with torch.no_grad():
        scaled = (
            torch.from_numpy(expected_valid) + hf_model.model.latent_bias_factor
        ) * hf_model.model.latent_scaling_factor
        expected_audio_embeds = hf_model.model.multi_modal_projector(scaled).cpu().numpy()
    projection_outputs = _run(
        package,
        "audio_projection",
        {
            "audio_latents": encoder_outputs["audio_latents"],
            "latents_are_scaled": np.array(False),
        },
    )
    np.testing.assert_allclose(
        projection_outputs["scaled_audio_latents"],
        scaled.cpu().numpy(),
        atol=1e-5,
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        projection_outputs["audio_embeds"],
        expected_audio_embeds,
        atol=1e-5,
        rtol=1e-5,
    )
    generated_projection = _run(
        package,
        "audio_projection",
        {
            "audio_latents": projection_outputs["scaled_audio_latents"],
            "latents_are_scaled": np.array(True),
        },
    )
    np.testing.assert_array_equal(
        generated_projection["scaled_audio_latents"],
        projection_outputs["scaled_audio_latents"],
    )
    np.testing.assert_allclose(
        generated_projection["audio_embeds"],
        expected_audio_embeds,
        atol=1e-5,
        rtol=1e-5,
    )

    input_ids = np.array([[1, 60, 2, 60, 3, 60, 4]], dtype=np.int64)
    audio_embeds = projection_outputs["audio_embeds"][:3]
    with torch.no_grad():
        expected_embeds = hf_model.model.language_model.embed_tokens(
            torch.from_numpy(input_ids)
        )
        expected_embeds[torch.from_numpy(input_ids) == 60] = torch.from_numpy(audio_embeds)
    embedding_outputs = _run(
        package,
        "embedding",
        {
            "input_ids": input_ids,
            "audio_embeds": audio_embeds,
            "replace_audio_tokens": np.array(True),
        },
    )
    np.testing.assert_array_equal(
        embedding_outputs["inputs_embeds"],
        expected_embeds.cpu().numpy(),
    )

    attention_mask = np.ones_like(input_ids)
    position_ids = np.arange(input_ids.shape[1], dtype=np.int64)[None]
    with torch.no_grad():
        reference_decoder = hf_model.model.language_model(
            inputs_embeds=expected_embeds,
            attention_mask=torch.from_numpy(attention_mask),
            position_ids=torch.from_numpy(position_ids),
            use_cache=True,
        )
        reference_logits = hf_model.lm_head(reference_decoder.last_hidden_state)
    decoder_outputs = _run(
        package,
        "decoder",
        {
            "inputs_embeds": embedding_outputs["inputs_embeds"],
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "past_key_values.0.key": np.zeros((1, 1, 0, 8), dtype=np.float32),
            "past_key_values.0.value": np.zeros((1, 1, 0, 8), dtype=np.float32),
        },
    )
    np.testing.assert_allclose(
        decoder_outputs["last_hidden_state"],
        reference_decoder.last_hidden_state.cpu().numpy(),
        atol=2e-4,
        rtol=2e-4,
    )
    np.testing.assert_allclose(
        decoder_outputs["logits"],
        reference_logits.cpu().numpy(),
        atol=2e-4,
        rtol=2e-4,
    )

    noisy = rng.standard_normal((2, 4)).astype(np.float32)
    timesteps = np.array([10.0, 10.0], dtype=np.float32)
    condition = rng.standard_normal((2, 16)).astype(np.float32)
    with torch.no_grad():
        reference_velocity = hf_model.model.diffusion_head(
            torch.from_numpy(noisy),
            torch.from_numpy(timesteps),
            torch.from_numpy(condition),
        )
    diffusion_outputs = _run(
        package,
        "diffusion_head",
        {
            "noisy_audio_latents": noisy,
            "timesteps": timesteps,
            "condition": condition,
        },
    )
    np.testing.assert_allclose(
        diffusion_outputs["velocity"],
        reference_velocity.cpu().numpy(),
        atol=2e-4,
        rtol=2e-4,
    )

    generated_latent = rng.standard_normal((1, 1, 4)).astype(np.float32)
    with torch.no_grad():
        unscaled_latent = (
            torch.from_numpy(generated_latent) / hf_model.model.latent_scaling_factor
            - hf_model.model.latent_bias_factor
        )
        reference_waveform = (
            hf_model.model.audio_tower.decode(
                unscaled_latent,
                use_cache=True,
            )
            .audio.cpu()
            .numpy()
        )
    audio_decoder_outputs = _run(
        package,
        "audio_decoder",
        {
            "scaled_audio_latents": generated_latent,
            **_zero_conv_cache(module.audio_decoder),
        },
    )
    np.testing.assert_allclose(
        audio_decoder_outputs["waveform"],
        reference_waveform,
        atol=3e-4,
        rtol=3e-4,
    )

    # The semantic stage consumes the preceding ONNX waveform, not an HF intermediate.
    with torch.no_grad():
        reference_semantic = hf_model.model.semantic_tokenizer_encoder(
            torch.from_numpy(audio_decoder_outputs["waveform"]),
            use_cache=True,
        ).latents
    semantic_outputs = _run(
        package,
        "semantic_encoder",
        {
            "waveform": audio_decoder_outputs["waveform"],
            **_zero_conv_cache(module.semantic_encoder),
        },
    )
    np.testing.assert_allclose(
        semantic_outputs["semantic_latents"],
        reference_semantic.cpu().numpy(),
        atol=3e-4,
        rtol=3e-4,
    )
    with torch.no_grad():
        reference_semantic_embeds = hf_model.model.semantic_connector(
            torch.from_numpy(semantic_outputs["semantic_latents"])
        )
    semantic_projection_outputs = _run(
        package,
        "semantic_projection",
        {"semantic_latents": semantic_outputs["semantic_latents"]},
    )
    np.testing.assert_allclose(
        semantic_projection_outputs["semantic_embeds"],
        reference_semantic_embeds.cpu().numpy(),
        atol=2e-4,
        rtol=2e-4,
    )


def test_vibevoice_text_only_embedding_accepts_empty_audio_rows():
    """The processor's no-reference-audio contract needs no fake waveform."""
    hf_model, _, package = _make_tiny_models()
    input_ids = np.array([[1, 2, 3]], dtype=np.int64)
    outputs = _run(
        package,
        "embedding",
        {
            "input_ids": input_ids,
            "audio_embeds": np.zeros((0, 16), dtype=np.float32),
            "replace_audio_tokens": np.array(True),
        },
    )
    with torch.no_grad():
        expected = hf_model.model.language_model.embed_tokens(torch.from_numpy(input_ids))
    np.testing.assert_array_equal(outputs["inputs_embeds"], expected.cpu().numpy())


class TestVibeVoiceGraphContracts:
    """Verify VibeVoice's staged graph and explicit-state contracts."""

    @staticmethod
    def _config() -> VibeVoiceConfig:
        tokenizer = VibeVoiceTokenizerConfig(
            hidden_size=4,
            num_filters=4,
            downsampling_ratios=[2, 2],
            depths=[1, 1, 1],
            ffn_expansion=2,
        )
        return VibeVoiceConfig(
            model_type="vibevoice",
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            vocab_size=64,
            max_position_embeddings=128,
            rms_norm_eps=1e-6,
            hidden_act="silu",
            attn_qkv_bias=True,
            rope_type="default",
            audio_token_id=60,
            audio_bos_token_id=61,
            audio_eos_token_id=62,
            acoustic_tokenizer=tokenizer,
            semantic_tokenizer=dataclasses.replace(tokenizer, hidden_size=6),
            diffusion_head=VibeVoiceDiffusionConfig(
                hidden_size=16,
                intermediate_size=32,
                latent_size=4,
                num_hidden_layers=1,
                frequency_embedding_size=8,
            ),
        )

    def test_all_stage_graphs_build(self):
        config = self._config()
        package = build_from_module(
            VibeVoiceForConditionalGeneration(config),
            config,
            task=VibeVoiceTask(),
        )

        assert set(package) == {
            "audio_encoder",
            "audio_projection",
            "embedding",
            "decoder",
            "diffusion_head",
            "audio_decoder",
            "semantic_encoder",
            "semantic_projection",
        }
        assert all(model.graph.num_nodes() > 0 for model in package.values())

    def test_stage_io_and_explicit_state(self):
        config = self._config()
        package = VibeVoiceTask().build(VibeVoiceForConditionalGeneration(config), config)

        audio_encoder = package["audio_encoder"]
        assert {value.name for value in audio_encoder.graph.inputs} == {
            "input_values",
            "padding_mask",
            "sample_noise",
            "latent_noise",
        }
        assert {value.name for value in audio_encoder.graph.outputs} == {"audio_latents"}

        embedding_audio = next(
            value
            for value in package["embedding"].graph.inputs
            if value.name == "audio_embeds"
        )
        assert optional_input_contract(embedding_audio) == {
            "presence": "audio",
            "absent": {"kind": "zeros", "shape": [0, config.hidden_size]},
        }

        decoder_inputs = {value.name for value in package["decoder"].graph.inputs}
        decoder_outputs = {value.name for value in package["decoder"].graph.outputs}
        assert "past_key_values.0.key" in decoder_inputs
        assert {"logits", "last_hidden_state", "present.0.key"} <= decoder_outputs

        # Tiny tokenizer geometry has seven causal state slots. The production
        # [3, 3, 3, 3, 3, 3, 8] stack has 34, using the same explicit ABI.
        for name in ("audio_decoder", "semantic_encoder"):
            inputs = {value.name for value in package[name].graph.inputs}
            outputs = {value.name for value in package[name].graph.outputs}
            assert {f"past_conv.{index}" for index in range(7)} <= inputs
            assert {f"present_conv.{index}" for index in range(7)} <= outputs

    def test_registry_lookup_is_pinned(self):
        registration = registry.get_registration("vibevoice")
        assert registration.module_class is VibeVoiceForConditionalGeneration
        assert registration.task == "vibevoice-tts"
        assert registration.test_model_id == VIBEVOICE_MODEL_ID
        assert registration.test_revision == VIBEVOICE_REVISION

    def test_cuda_fp16_preserves_arbitrary_attention_mask(self):
        config = dataclasses.replace(self._config(), dtype=ir.DataType.FLOAT16)
        package = build_from_module(
            VibeVoiceForConditionalGeneration(config),
            config,
            task=VibeVoiceTask(),
            execution_provider="cuda",
        )
        decoder = package["decoder"]
        assert requires_arbitrary_attention_mask(decoder.graph)
        assert "position_ids" in {value.name for value in decoder.graph.inputs}
        assert any(node.op_type == "Attention" for node in decoder.graph)
        assert not any(node.op_type == "GroupQueryAttention" for node in decoder.graph)
