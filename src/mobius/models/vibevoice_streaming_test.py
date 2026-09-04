# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""L1 contracts for VibeVoice Realtime's host-orchestrated ONNX stages."""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import sys
import types

import numpy as np
import onnx_ir as ir
import pytest
import torch

from mobius._builder import build_from_module
from mobius._configs import (
    VibeVoiceStreamingConfig,
    VibeVoiceStreamingDiffusionConfig,
    VibeVoiceStreamingTokenizerConfig,
)
from mobius._registry import registry
from mobius._testing.ort_inference import OnnxModelSession
from mobius.models.vibevoice_streaming import (
    VIBEVOICE_STREAMING_MODEL_ID,
    VIBEVOICE_STREAMING_REVISION,
    VibeVoiceStreamingForConditionalGeneration,
)
from mobius.tasks import VibeVoiceStreamingTask


def _config() -> VibeVoiceStreamingConfig:
    return VibeVoiceStreamingConfig(
        model_type="vibevoice_streaming",
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=3,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        vocab_size=64,
        max_position_embeddings=128,
        rms_norm_eps=1e-6,
        hidden_act="silu",
        attn_qkv_bias=True,
        rope_type="default",
        acoustic_tokenizer=VibeVoiceStreamingTokenizerConfig(
            vae_dim=4,
            decoder_n_filters=4,
            decoder_ratios=[2, 2],
            encoder_depths=[1, 1, 1],
            kernel_size=7,
            ffn_expansion=4,
        ),
        diffusion_head=VibeVoiceStreamingDiffusionConfig(
            hidden_size=16,
            intermediate_size=32,
            latent_size=4,
            num_hidden_layers=1,
            frequency_embedding_size=8,
        ),
        tts_backbone_num_hidden_layers=2,
    )


def _source_classes():
    """Load the pinned Microsoft implementation on current Transformers releases."""
    if importlib.util.find_spec("vibevoice") is None:
        pytest.skip("requires the pinned Microsoft VibeVoice source package")

    import transformers
    from transformers import AutoModel

    # The source's processor imports the pre-5.16 private Qwen2 tokenizer path,
    # while its model registration predates an existing native registration.
    # Neither compatibility shim participates in neural execution.
    original_register = AutoModel.register

    def register_with_existing_mapping(_cls, config_class, model_class, _exist_ok=False):
        return original_register(config_class, model_class, exist_ok=True)

    module_name = "transformers.models.qwen2.tokenization_qwen2_fast"
    previous_tokenizer_module = sys.modules.get(module_name)
    tokenizer_module = types.ModuleType(module_name)
    tokenizer_module.Qwen2TokenizerFast = transformers.Qwen2TokenizerFast
    AutoModel.register = classmethod(register_with_existing_mapping)
    sys.modules[module_name] = tokenizer_module
    try:
        from vibevoice.modular.configuration_vibevoice_streaming import (
            VibeVoiceStreamingConfig as SourceConfig,
        )
        from vibevoice.modular.modeling_vibevoice_streaming_inference import (
            VibeVoiceStreamingForConditionalGenerationInference as SourceModelBase,
        )
        from vibevoice.modular.modular_vibevoice_tokenizer import (
            VibeVoiceTokenizerStreamingCache,
        )
        from vibevoice.processor.vibevoice_streaming_processor import (
            VibeVoiceStreamingProcessor,
        )
    finally:
        AutoModel.register = original_register
        if previous_tokenizer_module is None:
            del sys.modules[module_name]
        else:
            sys.modules[module_name] = previous_tokenizer_module

    class SourceModel(SourceModelBase):
        """Accept current Transformers' optional tie-weights keyword."""

        def tie_weights(self, *args, **kwargs):
            return super().tie_weights()

    return (
        SourceConfig,
        SourceModel,
        VibeVoiceTokenizerStreamingCache,
        VibeVoiceStreamingProcessor,
    )


def _source_config_values() -> dict[str, object]:
    return {
        "decoder_config": {
            "model_type": "qwen2",
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 3,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "vocab_size": 64,
            "max_position_embeddings": 128,
            "rms_norm_eps": 1e-6,
            "hidden_act": "silu",
            "rope_theta": 10_000.0,
            "attention_bias": True,
            "tie_word_embeddings": False,
            "use_sliding_window": False,
        },
        "acoustic_tokenizer_config": {
            "channels": 1,
            "vae_dim": 4,
            "encoder_n_filters": 4,
            "decoder_n_filters": 4,
            "encoder_ratios": [2, 2],
            "decoder_ratios": [2, 2],
            "encoder_depths": "1-1-1",
            "decoder_depths": "1-1-1",
            "kernel_size": 7,
            "layernorm_eps": 1e-5,
            "causal": True,
            "conv_bias": True,
            "conv_norm": "none",
            "pad_mode": "constant",
            "layernorm": "RMSNorm",
            "mixer_layer": "depthwise_conv",
            "disable_last_norm": True,
        },
        "diffusion_head_config": {
            "hidden_size": 16,
            "head_layers": 1,
            "head_ffn_ratio": 2.0,
            "rms_norm_eps": 1e-5,
            "latent_size": 4,
            "ddpm_num_steps": 1000,
            "ddpm_num_inference_steps": 20,
            "ddpm_beta_schedule": "cosine",
            "prediction_type": "v_prediction",
        },
        "tts_backbone_num_hidden_layers": 2,
    }


def _source_config(source_config_class):
    return source_config_class(**_source_config_values())


def test_vibevoice_streaming_config_uses_checkpoint_torch_dtype() -> None:
    """The composite config owns the BF16 checkpoint storage dtype."""
    source = _source_config_values() | {
        "model_type": "vibevoice_streaming",
        "torch_dtype": "bfloat16",
    }

    config = VibeVoiceStreamingConfig.from_transformers(source)

    assert config.dtype is ir.DataType.BFLOAT16


def _parity_package():
    source_config_class, source_model_class, source_cache_class, _ = _source_classes()
    torch.manual_seed(11)
    source_model = source_model_class(_source_config(source_config_class)).float().eval()
    with torch.no_grad():
        source_model.model.speech_scaling_factor.fill_(0.5)
        source_model.model.speech_bias_factor.fill_(-0.25)

    config = dataclasses.replace(
        _config(),
        diffusion_head=VibeVoiceStreamingDiffusionConfig(
            hidden_size=16,
            intermediate_size=32,
            latent_size=4,
            num_hidden_layers=1,
            # Microsoft hard-codes this source dimension rather than storing it in
            # VibeVoiceDiffusionHeadConfig.
            frequency_embedding_size=256,
        ),
    )
    module = VibeVoiceStreamingForConditionalGeneration(config)
    package = VibeVoiceStreamingTask().build(module, config)

    # The source constructs an acoustic encoder, but the released Realtime
    # checkpoint has no encoder tensors. It only decodes generated latents.
    source_state = {
        name: value
        for name, value in source_model.state_dict().items()
        if ".acoustic_tokenizer.encoder." not in name
    }
    targets = module.source_weight_targets(source_state)
    unused = {name for name, target in targets.items() if target is None}
    assert unused == {"model.tts_language_model.embed_tokens.weight"}

    parameter_names = {
        name
        for model in package.values()
        for name, initializer in model.graph.initializers.items()
        if initializer.const_value is None
    }
    routed = module.preprocess_weights(source_state)
    assert set(routed) == parameter_names
    package.apply_weights(routed)
    return source_model, source_cache_class, module, package


def _run(package, name: str, feeds: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    session = OnnxModelSession(package[name])
    try:
        return session.run(feeds)
    finally:
        session.close()


def _empty_kv_cache(num_layers: int) -> dict[str, np.ndarray]:
    return {
        f"past_key_values.{index}.{kind}": np.empty((1, 1, 0, 8), dtype=np.float32)
        for index in range(num_layers)
        for kind in ("key", "value")
    }


def _assert_source_close(actual: np.ndarray, expected: np.ndarray) -> None:
    """Allow only ordinary PyTorch/ORT float32 accumulation-order noise."""
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


def test_vibevoice_streaming_stage_graphs_build() -> None:
    """The source pipeline produces six independent, executable stage graphs."""
    config = _config()
    package = build_from_module(
        VibeVoiceStreamingForConditionalGeneration(config),
        config,
        task=VibeVoiceStreamingTask(),
    )

    assert set(package) == {
        "embedding",
        "lm_backbone",
        "tts_backbone",
        "speech_connector",
        "diffusion_head",
        "audio_decoder",
    }
    assert all(model.graph.num_nodes() > 0 for model in package.values())


def test_vibevoice_streaming_states_and_host_protocol_are_explicit() -> None:
    """Cache ownership is visible in graph I/O rather than implicit in a host session."""
    config = _config()
    package = VibeVoiceStreamingTask().build(
        VibeVoiceStreamingForConditionalGeneration(config),
        config,
    )

    lm_inputs = {value.name for value in package["lm_backbone"].graph.inputs}
    lm_outputs = {value.name for value in package["lm_backbone"].graph.outputs}
    assert {"past_key_values.0.key", "past_key_values.0.value"} <= lm_inputs
    assert {"present.0.key", "present.0.value"} <= lm_outputs
    assert not any(name.startswith("past_key_values.1.") for name in lm_inputs)

    tts_inputs = {value.name for value in package["tts_backbone"].graph.inputs}
    tts_outputs = {value.name for value in package["tts_backbone"].graph.outputs}
    assert {
        "lm_last_hidden_state",
        "tts_text_masks",
        "past_key_values.0.key",
        "past_key_values.1.key",
    } <= tts_inputs
    assert {"eos_logits", "last_hidden_state", "present.0.key", "present.1.key"} <= tts_outputs

    audio_inputs = {value.name for value in package["audio_decoder"].graph.inputs}
    audio_outputs = {value.name for value in package["audio_decoder"].graph.outputs}
    assert {f"past_conv.{index}" for index in range(7)} <= audio_inputs
    assert {f"present_conv.{index}" for index in range(7)} <= audio_outputs

    metadata = json.loads(
        package["tts_backbone"].graph.metadata_props[
            "mobius.vibevoice_streaming.host_protocol"
        ]
    )
    assert metadata == {
        "cfg_cache_families": ["lm", "tts_lm", "neg_lm", "neg_tts_lm"],
        "diffusion_scheduler": "DPMSolverMultistepScheduler",
        "onnxruntime_genai": "unsupported",
        "owner": "host",
        "speech_window_size": 6,
        "text_window_size": 5,
        "voice_prompt": "prefilled_cache_only",
    }


def test_vibevoice_streaming_cuda_fp16_uses_standard_gqa() -> None:
    """The prefix-valid causal mask permits the normal CUDA fp16 GQA lowering."""
    config = dataclasses.replace(_config(), dtype=ir.DataType.FLOAT16)
    package = build_from_module(
        VibeVoiceStreamingForConditionalGeneration(config),
        config,
        task=VibeVoiceStreamingTask(),
        execution_provider="cuda",
    )

    for name in ("lm_backbone", "tts_backbone"):
        graph = package[name].graph
        assert any(node.op_type == "GroupQueryAttention" for node in graph)
        assert all(node.op_type != "Attention" for node in graph)
        assert "mobius.attention.requires_arbitrary_mask" not in graph.metadata_props


@pytest.mark.parametrize("dtype", [ir.DataType.FLOAT16, ir.DataType.BFLOAT16])
def test_vibevoice_streaming_uses_bool_prefix_masks(dtype: ir.DataType) -> None:
    """Causal prefix masks do not need the generic float additive-bias path."""
    config = dataclasses.replace(_config(), dtype=dtype)
    package = VibeVoiceStreamingTask().build(
        VibeVoiceStreamingForConditionalGeneration(config),
        config,
    )

    for name in ("lm_backbone", "tts_backbone"):
        attention = next(node for node in package[name].graph if node.op_type == "Attention")
        bool_mask = attention.inputs[3]
        assert bool_mask.producer().op_type == "Expand"
        unsqueeze = bool_mask.producer().inputs[0]
        assert unsqueeze.producer().op_type == "Unsqueeze"
        cast = unsqueeze.producer().inputs[0].producer()
        assert cast.op_type == "Cast"
        assert cast.attributes["to"].value == int(ir.DataType.BOOL)
        assert all(node.op_type != "Where" for node in package[name].graph)


def test_vibevoice_streaming_registry_is_pinned_and_architecture_discriminated() -> None:
    """Realtime cannot fall through to the incompatible VibeVoice 1.5B registration."""
    registration = registry.get_registration("vibevoice_streaming")
    architecture = registry.get_registration(
        "VibeVoiceStreamingForConditionalGenerationInference"
    )
    assert registration.module_class is VibeVoiceStreamingForConditionalGeneration
    assert registration.task == "vibevoice-streaming-tts"
    assert registration.test_model_id == VIBEVOICE_STREAMING_MODEL_ID
    assert registration.test_revision == VIBEVOICE_STREAMING_REVISION
    assert architecture == registration


def test_vibevoice_streaming_runtime_metadata_is_advisory(tmp_path) -> None:
    """Neither runtime exporter may claim to orchestrate this multi-cache pipeline."""
    package = VibeVoiceStreamingTask().build(
        VibeVoiceStreamingForConditionalGeneration(_config()),
        _config(),
    )

    from mobius.integrations.onnx_genai import write_onnx_genai_config
    from mobius.integrations.ort_genai import write_ort_genai_config

    with pytest.raises(ValueError, match="host-owned"):
        write_ort_genai_config(package, str(tmp_path / "ort"))
    assert not (tmp_path / "ort").exists()

    artifacts = write_onnx_genai_config(package, str(tmp_path / "onnx"))
    assert "genai_config" not in artifacts
    with open(artifacts["runtime_compatibility"], encoding="utf-8") as handle:
        metadata = json.load(handle)
    assert metadata["runtime_validation_status"] == "unsupported-by-tested-runtime"
    assert "no onnx-genai runtime configuration is claimed" in metadata["warnings"][0]


def test_vibevoice_streaming_exact_source_stage_and_continuation_parity() -> None:
    """L3: Every executable source stage matches initial and cache-continuation steps."""
    source_model, source_cache_class, module, package = _parity_package()
    rng = np.random.default_rng(1)

    input_ids = np.array([[1, 2, 3]], dtype=np.int64)
    attention_mask = np.ones((1, 3), dtype=np.int64)
    position_ids = np.arange(3, dtype=np.int64)[None]
    embedding = _run(package, "embedding", {"input_ids": input_ids})["inputs_embeds"]
    with torch.no_grad():
        source_embedding = source_model.model.language_model.embed_tokens(
            torch.from_numpy(input_ids)
        ).numpy()
    np.testing.assert_array_equal(embedding, source_embedding)

    with torch.no_grad():
        source_lm = source_model.forward_lm(
            inputs_embeds=torch.from_numpy(embedding.copy()),
            attention_mask=torch.from_numpy(attention_mask),
            position_ids=torch.from_numpy(position_ids),
            use_cache=True,
        )
    lm = _run(
        package,
        "lm_backbone",
        {
            "inputs_embeds": embedding,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            **_empty_kv_cache(1),
        },
    )
    _assert_source_close(lm["last_hidden_state"], source_lm.last_hidden_state.numpy())

    next_ids = np.array([[4]], dtype=np.int64)
    next_embedding = _run(package, "embedding", {"input_ids": next_ids})["inputs_embeds"]
    with torch.no_grad():
        source_lm_next = source_model.forward_lm(
            inputs_embeds=torch.from_numpy(next_embedding.copy()),
            attention_mask=torch.ones((1, 4), dtype=torch.long),
            position_ids=torch.tensor([[3]]),
            past_key_values=source_lm.past_key_values,
            use_cache=True,
        )
    lm_next = _run(
        package,
        "lm_backbone",
        {
            "inputs_embeds": next_embedding,
            "attention_mask": np.ones((1, 4), dtype=np.int64),
            "position_ids": np.array([[3]], dtype=np.int64),
            "past_key_values.0.key": lm["present.0.key"],
            "past_key_values.0.value": lm["present.0.value"],
        },
    )
    _assert_source_close(
        lm_next["last_hidden_state"],
        source_lm_next.last_hidden_state.numpy(),
    )

    with torch.no_grad():
        source_tts = source_model.forward_tts_lm(
            inputs_embeds=torch.from_numpy(embedding.copy()),
            attention_mask=torch.from_numpy(attention_mask),
            position_ids=torch.from_numpy(position_ids),
            lm_last_hidden_state=source_lm.last_hidden_state,
            tts_text_masks=torch.ones((1, 1), dtype=torch.bool),
            use_cache=True,
        )
    tts = _run(
        package,
        "tts_backbone",
        {
            "inputs_embeds": embedding,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "lm_last_hidden_state": lm["last_hidden_state"],
            "tts_text_masks": np.ones((1, 1), dtype=np.bool_),
            **_empty_kv_cache(2),
        },
    )
    _assert_source_close(tts["last_hidden_state"], source_tts.last_hidden_state.numpy())
    _assert_source_close(tts["eos_logits"], source_tts.logits.numpy())

    with torch.no_grad():
        source_tts_next = source_model.forward_tts_lm(
            inputs_embeds=torch.from_numpy(next_embedding.copy()),
            attention_mask=torch.ones((1, 4), dtype=torch.long),
            position_ids=torch.tensor([[3]]),
            lm_last_hidden_state=source_lm_next.last_hidden_state,
            tts_text_masks=torch.zeros((1, 1), dtype=torch.bool),
            past_key_values=source_tts.past_key_values,
            use_cache=True,
        )
    tts_next = _run(
        package,
        "tts_backbone",
        {
            "inputs_embeds": next_embedding,
            "attention_mask": np.ones((1, 4), dtype=np.int64),
            "position_ids": np.array([[3]], dtype=np.int64),
            "lm_last_hidden_state": lm_next["last_hidden_state"],
            "tts_text_masks": np.zeros((1, 1), dtype=np.bool_),
            **{
                f"past_key_values.{index}.{kind}": tts[f"present.{index}.{kind}"]
                for index in range(2)
                for kind in ("key", "value")
            },
        },
    )
    _assert_source_close(
        tts_next["last_hidden_state"],
        source_tts_next.last_hidden_state.numpy(),
    )
    _assert_source_close(tts_next["eos_logits"], source_tts_next.logits.numpy())

    latents = rng.standard_normal((2, 4), dtype=np.float32)
    timesteps = np.array([1.0, 2.0], dtype=np.float32)
    condition = rng.standard_normal((2, 16), dtype=np.float32)
    with torch.no_grad():
        source_velocity = source_model.prediction_head(
            torch.from_numpy(latents),
            torch.from_numpy(timesteps),
            torch.from_numpy(condition),
        ).numpy()
        source_connector = source_model.acoustic_connector(
            torch.from_numpy(latents[:, None, :])
        ).numpy()
    diffusion = _run(
        package,
        "diffusion_head",
        {
            "noisy_speech_latents": latents,
            "timesteps": timesteps,
            "condition": condition,
        },
    )
    connector = _run(package, "speech_connector", {"speech_latents": latents[:, None, :]})
    _assert_source_close(diffusion["velocity"], source_velocity)
    _assert_source_close(connector["speech_embeds"], source_connector)

    first_latents = rng.standard_normal((1, 2, 4), dtype=np.float32)
    next_latents = rng.standard_normal((1, 1, 4), dtype=np.float32)
    source_cache = source_cache_class()
    sample_indices = torch.tensor([0])
    with torch.no_grad():
        source_audio = source_model.acoustic_tokenizer.decode(
            torch.from_numpy(first_latents / 0.5 + 0.25),
            cache=source_cache,
            sample_indices=sample_indices,
            use_cache=True,
        ).numpy()
    audio = _run(
        package,
        "audio_decoder",
        {
            "speech_latents": first_latents,
            **{
                f"past_conv.{index}": np.zeros((1, channels, left_pad), dtype=np.float32)
                for index, (channels, left_pad) in enumerate(module.audio_decoder.cache_specs)
            },
        },
    )
    _assert_source_close(audio["waveform"], source_audio)

    with torch.no_grad():
        source_audio_next = source_model.acoustic_tokenizer.decode(
            torch.from_numpy(next_latents / 0.5 + 0.25),
            cache=source_cache,
            sample_indices=sample_indices,
            use_cache=True,
        ).numpy()
    audio_next = _run(
        package,
        "audio_decoder",
        {
            "speech_latents": next_latents,
            **{
                f"past_conv.{index}": audio[f"present_conv.{index}"]
                for index in range(len(module.audio_decoder.cache_specs))
            },
        },
    )
    _assert_source_close(audio_next["waveform"], source_audio_next)


def test_vibevoice_streaming_source_processor_uses_cached_prompts_only() -> None:
    """The source processor emits pseudo-token slots, never arbitrary voice waveform input."""
    _, _, _, source_processor_class = _source_classes()

    class Tokenizer:
        pad_id = 0

        def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
            assert text == "Hello world\n"
            assert add_special_tokens is False
            return [7, 8]

    processor = source_processor_class(tokenizer=Tokenizer(), db_normalize=False)
    prompt = {
        name: {"last_hidden_state": torch.zeros((1, length, 16))}
        for name, length in (("lm", 3), ("tts_lm", 5), ("neg_lm", 3), ("neg_tts_lm", 5))
    }
    with pytest.raises(NotImplementedError, match="process_input_with_cached_prompt"):
        processor()
    encoded = processor.process_input_with_cached_prompt(
        "  Hello world  ",
        prompt,
        return_tensors="pt",
    )
    assert encoded["input_ids"].tolist() == [[0, 0, 0]]
    assert encoded["tts_lm_input_ids"].tolist() == [[0, 0, 0, 0, 0]]
    assert encoded["tts_text_ids"].tolist() == [[7, 8]]
    assert encoded["speech_input_mask"].tolist() == [[False, False, False, False, False]]
