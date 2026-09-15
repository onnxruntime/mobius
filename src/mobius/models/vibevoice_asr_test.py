# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Graph-contract and checkpoint-routing tests for VibeVoice streaming ASR."""

from __future__ import annotations

import dataclasses
import json
import subprocess
from importlib import metadata
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import onnx_ir as ir
import pytest
import torch

from mobius._builder import build_from_module
from mobius._configs import VibeVoiceASRConfig, VibeVoiceTokenizerConfig
from mobius._pipeline_contract import (
    optional_input_contract,
    requires_arbitrary_attention_mask,
)
from mobius._registry import registry
from mobius._testing.ort_inference import OnnxModelSession
from mobius.models.vibevoice import (
    VIBEVOICE_ASR_MODEL_REVISIONS,
    VIBEVOICE_ASR_SOURCE_REVISION,
    VibeVoiceASRForConditionalGeneration,
)
from mobius.tasks import VibeVoiceASRStreamingTask


def _config() -> VibeVoiceASRConfig:
    tokenizer = VibeVoiceTokenizerConfig(
        hidden_size=4,
        kernel_size=3,
        num_filters=4,
        downsampling_ratios=[2, 2],
        depths=[1, 1, 1],
        ffn_expansion=2,
        vae_std=0.5,
    )
    return VibeVoiceASRConfig(
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
        acoustic_tokenizer=tokenizer,
        semantic_tokenizer=dataclasses.replace(
            tokenizer,
            hidden_size=6,
            std_dist_type="none",
        ),
    )


def _source_key_for_parameter(parameter_name: str) -> str:
    """Invert the public ASR source-to-package mapping for exhaustive routing tests."""
    if parameter_name.startswith("audio_encoder.acoustic_tokenizer."):
        tower = "acoustic"
        suffix = parameter_name.removeprefix("audio_encoder.acoustic_tokenizer.")
    elif parameter_name.startswith("audio_encoder.semantic_tokenizer."):
        tower = "semantic"
        suffix = parameter_name.removeprefix("audio_encoder.semantic_tokenizer.")
    else:
        tower = None
        suffix = parameter_name

    if tower is not None:
        if suffix.startswith("stem.conv."):
            suffix = f"downsample_layers.0.0.{suffix.removeprefix('stem.')}"
        elif suffix.startswith("conv_layers."):
            _, index_text, remainder = suffix.split(".", maxsplit=2)
            if remainder.startswith("stage."):
                suffix = f"stages.{int(index_text) + 1}.{remainder.removeprefix('stage.')}"
            else:
                suffix = f"downsample_layers.{int(index_text) + 1}.0.{remainder}"
        elif suffix.startswith("stem.stage."):
            suffix = f"stages.0.{suffix.removeprefix('stem.stage.')}"
        elif not suffix.startswith("head."):
            raise AssertionError(f"Unexpected tokenizer parameter {parameter_name}")
        suffix = suffix.replace(".mixer.conv.", ".mixer.conv.conv.conv.")
        suffix = suffix.replace("head.conv.", "head.conv.conv.")
        return f"model.{tower}_tokenizer.encoder.{suffix}"

    if parameter_name.startswith("audio_encoder.acoustic_connector."):
        return "model.acoustic_connector." + parameter_name.removeprefix(
            "audio_encoder.acoustic_connector."
        )
    if parameter_name.startswith("audio_encoder.semantic_connector."):
        return "model.semantic_connector." + parameter_name.removeprefix(
            "audio_encoder.semantic_connector."
        )
    if parameter_name.startswith("embedding.embed_tokens."):
        return "model.language_model.embed_tokens." + parameter_name.removeprefix(
            "embedding.embed_tokens."
        )
    if parameter_name.startswith(("decoder.layers.", "decoder.norm.")):
        return "model.language_model." + parameter_name.removeprefix("decoder.")
    if parameter_name == "decoder.lm_head.weight":
        return "lm_head.weight"
    raise AssertionError(f"Unexpected ASR package parameter {parameter_name}")


class TestVibeVoiceASR:
    """Exercise every stage, runtime contract, and trained-weight route at tiny scale."""

    def test_stage_contract_and_explicit_state(self):
        config = _config()
        module = VibeVoiceASRForConditionalGeneration(config)
        package = VibeVoiceASRStreamingTask().build(module, config)

        assert set(package) == {"audio_encoder", "embedding", "decoder"}
        audio_inputs = {value.name for value in package["audio_encoder"].graph.inputs}
        audio_outputs = {value.name for value in package["audio_encoder"].graph.outputs}
        assert {
            "speech_tensors",
            "speech_masks",
            "acoustic_sample_noise",
            "acoustic_latent_noise",
            "is_final_chunk",
        } <= audio_inputs
        for prefix, names in (
            ("past_acoustic_conv", audio_inputs),
            ("present_acoustic_conv", audio_outputs),
            ("past_semantic_conv", audio_inputs),
            ("present_semantic_conv", audio_outputs),
        ):
            assert {f"{prefix}.{index}" for index in range(7)} <= names

        speech_embeds = next(
            value
            for value in package["embedding"].graph.inputs
            if value.name == "speech_embeds"
        )
        assert optional_input_contract(speech_embeds) == {
            "presence": "audio",
            "absent": {"kind": "zeros", "shape": [0, config.hidden_size]},
        }
        assert requires_arbitrary_attention_mask(package["decoder"].graph)

    def test_cuda_fp16_excludes_prefix_only_gqa_fusion(self):
        config = dataclasses.replace(_config(), dtype=ir.DataType.FLOAT16)
        package = build_from_module(
            VibeVoiceASRForConditionalGeneration(config),
            config,
            task=VibeVoiceASRStreamingTask(),
            execution_provider="cuda",
        )
        decoder_nodes = list(package["decoder"].graph.all_nodes())
        assert any(node.op_type == "Attention" for node in decoder_nodes)
        assert not any(node.op_type == "GroupQueryAttention" for node in decoder_nodes)

    def test_checkpoint_routes_every_inference_parameter_once(self):
        config = _config()
        module = VibeVoiceASRForConditionalGeneration(config)
        package = VibeVoiceASRStreamingTask().build(module, config)
        parameter_names = {
            value.name
            for model in package.values()
            for value in model.graph.initializers.values()
            if value.const_value is None
        }
        state_dict = {
            _source_key_for_parameter(name): torch.zeros(1) for name in parameter_names
        }
        # The acoustic VAE decoder is present in the publication checkpoint but
        # is provably not on the executable ASR encode_speech path.
        state_dict["model.acoustic_tokenizer.decoder.head.conv.conv.weight"] = torch.zeros(1)
        routed = module.preprocess_weights(state_dict)

        assert set(routed) == parameter_names
        assert len(routed) == len(parameter_names)
        assert not any("acoustic_tokenizer.decoder" in name for name in routed)

    def test_registration_is_pinned_and_architecture_specific(self):
        registration = registry.get_registration("VibeVoiceForASRStreamingTraining")
        assert registration.module_class is VibeVoiceASRForConditionalGeneration
        assert registration.task == "vibevoice-asr-streaming"
        assert registration.config_class is VibeVoiceASRConfig
        assert registration.test_model_id in VIBEVOICE_ASR_MODEL_REVISIONS
        assert (
            registration.test_revision
            == VIBEVOICE_ASR_MODEL_REVISIONS[registration.test_model_id]
        )

    def test_tied_checkpoint_lm_head_preserves_its_explicit_tensor(self):
        """A tied checkpoint's explicit LM head must not be overwritten by an embedding fallback."""
        module = VibeVoiceASRForConditionalGeneration(
            dataclasses.replace(_config(), tie_word_embeddings=True)
        )
        embedding_weight = torch.zeros(1)
        lm_head_weight = torch.ones(1)

        routed = module.preprocess_weights(
            {
                "model.language_model.embed_tokens.weight": embedding_weight,
                "lm_head.weight": lm_head_weight,
            }
        )

        assert set(routed) == {"embedding.embed_tokens.weight", "decoder.lm_head.weight"}
        assert routed["embedding.embed_tokens.weight"] is embedding_weight
        assert routed["decoder.lm_head.weight"] is lm_head_weight

    def test_vibevoice_dispatch_rejects_ambiguous_or_unknown_architectures(self):
        from mobius.integrations.transformers._builder import _resolve_module_class

        ambiguous = SimpleNamespace(model_type="vibevoice", architectures=[])
        with pytest.raises(ValueError, match="exactly one recognized architecture"):
            _resolve_module_class("vibevoice", ambiguous, None, None)

        unknown = SimpleNamespace(
            model_type="vibevoice",
            architectures=["VibeVoiceForSomethingElse"],
        )
        with pytest.raises(ValueError, match="Unsupported VibeVoice architecture"):
            _resolve_module_class("vibevoice", unknown, None, None)

        tts = SimpleNamespace(
            model_type="vibevoice",
            architectures=["VibeVoiceForConditionalGeneration"],
        )
        module_class, task, model_type = _resolve_module_class("vibevoice", tts, None, None)
        assert module_class.__name__ == "VibeVoiceForConditionalGeneration"
        assert task is None
        assert model_type == "vibevoice"

    def test_asr_config_rejects_non_executable_tokenizer_variants(self):
        from mobius._configs.vibevoice import _asr_tokenizer_config

        with pytest.raises(ValueError, match="mixer_layer='conv'"):
            _asr_tokenizer_config(
                SimpleNamespace(mixer_layer="conv"),
                default_hidden_size=64,
            )


def _require_pinned_reference():
    """Load the independently executable reference only from a verified local checkout."""
    import transformers

    if transformers.__version__ != "4.51.3":
        pytest.skip("Synthetic VibeVoice ASR parity requires transformers==4.51.3.")
    try:
        direct_url = metadata.distribution("vibevoice").read_text("direct_url.json")
    except metadata.PackageNotFoundError:
        direct_url = None
    source_is_pinned = False
    if direct_url is not None:
        vcs_info = json.loads(direct_url).get("vcs_info", {})
        if vcs_info.get("commit_id") != VIBEVOICE_ASR_SOURCE_REVISION:
            pytest.skip(
                "Synthetic VibeVoice ASR parity requires "
                f"VibeVoice@{VIBEVOICE_ASR_SOURCE_REVISION}."
            )
        source_is_pinned = True
    modeling = pytest.importorskip("vibevoice.modular.modeling_vibevoice_asr")
    configuration = pytest.importorskip("vibevoice.modular.configuration_vibevoice")
    tokenizer = pytest.importorskip("vibevoice.modular.modular_vibevoice_tokenizer")
    if source_is_pinned:
        return modeling, configuration, tokenizer
    source_root = Path(modeling.__file__).parents[2]
    try:
        source_revision = subprocess.run(
            ["git", "-C", str(source_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        pytest.skip(
            "Synthetic VibeVoice ASR parity requires installed VCS metadata or "
            f"VibeVoice@{VIBEVOICE_ASR_SOURCE_REVISION} in direct_url.json."
        )
    if source_revision != VIBEVOICE_ASR_SOURCE_REVISION:
        pytest.skip(
            "Synthetic VibeVoice ASR parity requires "
            f"VibeVoice@{VIBEVOICE_ASR_SOURCE_REVISION}."
        )
    return modeling, configuration, tokenizer


def _run_component(package, name: str, feeds: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    session = OnnxModelSession(package[name])
    try:
        return session.run(feeds)
    finally:
        session.close()


@pytest.mark.integration
def test_vibevoice_asr_synthetic_two_chunk_prefill_and_cached_decode_parity():
    """Compare all exported stages to the pinned executable source with random tiny weights."""
    modeling, configuration, tokenizer_module = _require_pinned_reference()
    source_tokenizer = {
        "channels": 1,
        "vae_dim": 4,
        "fix_std": 0.5,
        "encoder_n_filters": 4,
        "encoder_ratios": [2, 2],
        "encoder_depths": "1-1-1",
        "mixer_layer": "depthwise_conv",
        "conv_norm": "none",
        "pad_mode": "constant",
        "disable_last_norm": True,
        "layernorm": "RMSNorm",
        "layernorm_eps": 1e-5,
        "conv_bias": True,
    }
    decoder = {
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
    }
    torch.manual_seed(7)
    reference = (
        modeling.VibeVoiceASRForConditionalGeneration(
            configuration.VibeVoiceASRConfig(
                acoustic_tokenizer_config={**source_tokenizer, "std_dist_type": "gaussian"},
                semantic_tokenizer_config={
                    **source_tokenizer,
                    "vae_dim": 6,
                    "std_dist_type": "none",
                },
                decoder_config=decoder,
            )
        )
        .float()
        .eval()
    )

    mobius_tokenizer = VibeVoiceTokenizerConfig(
        hidden_size=4,
        kernel_size=7,
        num_filters=4,
        downsampling_ratios=[2, 2],
        depths=[1, 1, 1],
        ffn_expansion=4,
        vae_std=0.5,
        std_dist_type="gaussian",
    )
    config = VibeVoiceASRConfig(
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
        rope_theta=10_000.0,
        acoustic_tokenizer=mobius_tokenizer,
        semantic_tokenizer=dataclasses.replace(
            mobius_tokenizer,
            hidden_size=6,
            std_dist_type="none",
        ),
    )
    module = VibeVoiceASRForConditionalGeneration(config)
    package = VibeVoiceASRStreamingTask().build(module, config)
    package.apply_weights(module.preprocess_weights(reference.state_dict()))

    waveforms = (
        torch.linspace(-0.5, 0.5, 16).reshape(1, 16).repeat(2, 1),
        torch.linspace(0.25, -0.25, 12).reshape(1, 12).repeat(2, 1),
    )
    sample_noise = torch.tensor([-0.4, 0.7])
    latent_noises = (
        torch.linspace(-1, 1, 2 * 4 * 4).reshape(2, 4, 4),
        torch.linspace(1, -1, 2 * 3 * 4).reshape(2, 3, 4),
    )
    reference_acoustic_cache = tokenizer_module.VibeVoiceTokenizerStreamingCache()
    reference_semantic_cache = tokenizer_module.VibeVoiceTokenizerStreamingCache()

    def reference_audio_chunk(waveform, latent_noise, is_final_chunk):
        sample_indices = torch.arange(waveform.shape[0])
        acoustic_mean = reference.model.acoustic_tokenizer.encode(
            waveform.unsqueeze(1),
            cache=reference_acoustic_cache,
            sample_indices=sample_indices,
            use_cache=True,
            is_final_chunk=is_final_chunk,
        ).mean
        semantic_latents = reference.model.semantic_tokenizer.encode(
            waveform.unsqueeze(1),
            cache=reference_semantic_cache,
            sample_indices=sample_indices,
            use_cache=True,
            is_final_chunk=is_final_chunk,
        ).mean
        acoustic_latents = (
            acoustic_mean + sample_noise[:, None, None] * (0.5 / 0.8) * latent_noise
        )
        return reference.model.acoustic_connector(
            acoustic_latents
        ) + reference.model.semantic_connector(semantic_latents)

    with torch.no_grad():
        expected_audio = [
            reference_audio_chunk(waveforms[0], latent_noises[0], False),
            reference_audio_chunk(waveforms[1], latent_noises[1], True),
        ]
        input_ids = torch.tensor([[3, 4, 5, 6, 7], [3, 4, 5, 6, 7]])
        acoustic_input_mask = torch.tensor(
            [[False, True, True, True, True], [False, True, True, True, True]]
        )
        reference_embeds = reference.get_input_embeddings()(input_ids)
        reference_embeds[acoustic_input_mask] = expected_audio[0].reshape(-1, 16)
        reference_prefill = reference(
            inputs_embeds=reference_embeds,
            attention_mask=torch.ones(2, 5, dtype=torch.long),
            position_ids=torch.arange(5).repeat(2, 1),
            use_cache=True,
            return_dict=True,
        )
        decode_ids = torch.tensor([[8], [9]])
        reference_decode = reference(
            inputs_embeds=reference.get_input_embeddings()(decode_ids),
            attention_mask=torch.ones(2, 6, dtype=torch.long),
            position_ids=torch.full((2, 1), 5),
            past_key_values=reference_prefill.past_key_values,
            use_cache=True,
            return_dict=True,
        )

    def initial_cache(
        prefix: str, specs: tuple[tuple[int, int], ...]
    ) -> dict[str, np.ndarray]:
        return {
            f"{prefix}.{index}": np.zeros((2, channels, left_pad), dtype=np.float32)
            for index, (channels, left_pad) in enumerate(specs)
        }

    first = _run_component(
        package,
        "audio_encoder",
        {
            "speech_tensors": waveforms[0].numpy(),
            "speech_masks": np.ones((2, 4), dtype=bool),
            "acoustic_sample_noise": sample_noise.numpy(),
            "acoustic_latent_noise": latent_noises[0].numpy(),
            "is_final_chunk": np.asarray(False),
            **initial_cache("past_acoustic_conv", module.audio_encoder.acoustic_cache_specs),
            **initial_cache("past_semantic_conv", module.audio_encoder.semantic_cache_specs),
        },
    )
    second_feeds = {
        "speech_tensors": waveforms[1].numpy(),
        "speech_masks": np.ones((2, 3), dtype=bool),
        "acoustic_sample_noise": sample_noise.numpy(),
        "acoustic_latent_noise": latent_noises[1].numpy(),
        "is_final_chunk": np.asarray(True),
    }
    for index in range(len(module.audio_encoder.acoustic_cache_specs)):
        second_feeds[f"past_acoustic_conv.{index}"] = first[f"present_acoustic_conv.{index}"]
    for index in range(len(module.audio_encoder.semantic_cache_specs)):
        second_feeds[f"past_semantic_conv.{index}"] = first[f"present_semantic_conv.{index}"]
    second = _run_component(package, "audio_encoder", second_feeds)
    np.testing.assert_allclose(
        first["speech_embeds"],
        expected_audio[0].reshape(-1, 16).numpy(),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        second["speech_embeds"],
        expected_audio[1].reshape(-1, 16).numpy(),
        rtol=1e-5,
        atol=1e-5,
    )

    embeds = _run_component(
        package,
        "embedding",
        {
            "input_ids": input_ids.numpy(),
            "speech_embeds": first["speech_embeds"],
            "acoustic_input_mask": acoustic_input_mask.numpy(),
        },
    )
    prefill = _run_component(
        package,
        "decoder",
        {
            "inputs_embeds": embeds["inputs_embeds"],
            "attention_mask": np.ones((2, 5), dtype=np.int64),
            "position_ids": np.tile(np.arange(5, dtype=np.int64), (2, 1)),
            "past_key_values.0.key": np.zeros((2, 1, 0, 8), dtype=np.float32),
            "past_key_values.0.value": np.zeros((2, 1, 0, 8), dtype=np.float32),
        },
    )
    decode_embeds = _run_component(
        package,
        "embedding",
        {
            "input_ids": decode_ids.numpy(),
            "speech_embeds": np.zeros((0, 16), dtype=np.float32),
            "acoustic_input_mask": np.zeros((2, 1), dtype=bool),
        },
    )
    decode = _run_component(
        package,
        "decoder",
        {
            "inputs_embeds": decode_embeds["inputs_embeds"],
            "attention_mask": np.ones((2, 6), dtype=np.int64),
            "position_ids": np.full((2, 1), 5, dtype=np.int64),
            "past_key_values.0.key": prefill["present.0.key"],
            "past_key_values.0.value": prefill["present.0.value"],
        },
    )
    np.testing.assert_allclose(
        prefill["logits"],
        reference_prefill.logits.numpy(),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        decode["logits"],
        reference_decode.logits.numpy(),
        rtol=1e-5,
        atol=1e-5,
    )


@pytest.mark.integration
@pytest.mark.parametrize(
    ("model_id", "revision"),
    tuple(VIBEVOICE_ASR_MODEL_REVISIONS.items()),
)
def test_vibevoice_asr_pinned_processor_contract_for_hotwords_and_speakers(model_id, revision):
    """Validate processor rows, left padding, bilingual hotword prompts, and speaker JSON."""
    _require_pinned_reference()
    processor_module = pytest.importorskip("vibevoice.processor.vibevoice_asr_processor")
    processor = processor_module.VibeVoiceASRProcessor.from_pretrained(
        model_id,
        revision=revision,
    )
    english = np.linspace(-0.1, 0.1, 3200, dtype=np.float32)
    chinese = np.linspace(0.1, -0.1, 6500, dtype=np.float32)
    plain = processor(english, sampling_rate=24_000, return_tensors="pt")
    hotwords = processor(
        english,
        sampling_rate=24_000,
        return_tensors="pt",
        context_info="hotwords: Mobius, 你好",
    )
    batch = processor(
        [english, chinese],
        sampling_rate=24_000,
        return_tensors="pt",
        context_info="hotwords: Mobius, 你好",
    )

    assert set(batch) == {
        "input_ids",
        "attention_mask",
        "acoustic_input_mask",
        "speech_tensors",
        "speech_masks",
    }
    assert batch["speech_tensors"].shape == (2, 6500)
    assert batch["speech_masks"].sum(dim=1).tolist() == [1, 3]
    assert batch["attention_mask"][0, 0].item() == 0
    assert batch["attention_mask"][1, 0].item() == 1
    assert plain["input_ids"].tolist() != hotwords["input_ids"].tolist()
    assert {
        token: processor.tokenizer.convert_tokens_to_ids(token)
        for token in (
            "<|object_ref_start|>",
            "<|object_ref_end|>",
            "<|box_start|>",
            "<|text_chunk_end|>",
        )
    } == {
        "<|object_ref_start|>": 151646,
        "<|object_ref_end|>": 151647,
        "<|box_start|>": 151648,
        "<|text_chunk_end|>": 151665,
    }
    assert processor.post_process_transcription(
        '[{"Start time": 0.0, "End time": 1.2, "Speaker ID": "spk-1", '
        '"Content": "hello"}, {"Start": 1.2, "End": 2.0, "Speaker": "说话人2", '
        '"Content": "你好"}]'
    ) == [
        {"start_time": 0.0, "end_time": 1.2, "speaker_id": "spk-1", "text": "hello"},
        {"start_time": 1.2, "end_time": 2.0, "speaker_id": "说话人2", "text": "你好"},
    ]
