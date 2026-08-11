# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Numerical parity tests for Moonshine raw-waveform speech recognition."""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import pytest
import torch
import transformers

from mobius import build, build_from_module
from mobius._configs import MoonshineConfig
from mobius._testing.comparison import assert_logits_close
from mobius._testing.golden import (
    discover_test_cases,
    generation_json_path_for_case,
    load_generation_golden,
)
from mobius._testing.ort_inference import OnnxModelSession
from mobius.models import MoonshineForConditionalGeneration
from mobius.tasks import SpeechToTextTask

_MODEL_ID = "moonshine-ai/moonshine-tiny"
_AUDIO_PATH = Path(__file__).parent.parent / "testdata" / "652-129742-0006.flac"


def test_moonshine_l5_generation_reference_is_declared_and_valid():
    cases = [
        case
        for case in discover_test_cases(level="L5")
        if case.model_id == _MODEL_ID
    ]
    assert len(cases) == 1
    case = cases[0]
    generation_path = generation_json_path_for_case(case)
    assert generation_path.exists()
    generated_tokens = load_generation_golden(case)
    assert generated_tokens
    assert all(isinstance(token, int) for token in generated_tokens)


def _tiny_hf_config() -> transformers.MoonshineConfig:
    return transformers.MoonshineConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        encoder_num_hidden_layers=2,
        decoder_num_hidden_layers=2,
        encoder_num_attention_heads=4,
        decoder_num_attention_heads=4,
        encoder_num_key_value_heads=4,
        decoder_num_key_value_heads=4,
        max_position_embeddings=32,
        partial_rotary_factor=0.75,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10_000.0,
            "partial_rotary_factor": 0.75,
        },
        pad_head_dim_to_multiple_of=8,
        attention_bias=False,
        tie_word_embeddings=True,
    )


def _weighted_package(hf_model, hf_config):
    config = MoonshineConfig.from_transformers(hf_config)
    module = MoonshineForConditionalGeneration(config)
    package = build_from_module(module, config, task=SpeechToTextTask())
    state_dict = {
        name: value.detach().clone() for name, value in hf_model.state_dict().items()
    }
    package.apply_weights(module.preprocess_weights(state_dict))
    return package, config


def _run_encoder(package, input_values, attention_mask):
    session = OnnxModelSession(package["encoder"])
    try:
        return session.run(
            {
                "input_values": input_values.astype(np.float32),
                "attention_mask": attention_mask.astype(np.int64),
            }
        )
    finally:
        session.close()


def _empty_cache_feeds(config: MoonshineConfig, batch_size: int = 1):
    feeds = {}
    for layer_idx in range(config.num_hidden_layers):
        for kind in ("key", "value"):
            feeds[f"past_key_values.{layer_idx}.{kind}"] = np.zeros(
                (
                    batch_size,
                    config.num_key_value_heads,
                    0,
                    config.head_dim,
                ),
                dtype=np.float32,
            )
    return feeds


def _run_decoder(
    package,
    config,
    decoder_input_ids,
    encoder_hidden_states,
    encoder_attention_mask,
    past_key_values=None,
    position_offset=0,
):
    session = OnnxModelSession(package["decoder"])
    sequence_length = decoder_input_ids.shape[1]
    feeds = {
        "decoder_input_ids": decoder_input_ids.astype(np.int64),
        "encoder_hidden_states": encoder_hidden_states.astype(np.float32),
        "encoder_attention_mask": encoder_attention_mask.astype(np.int64),
        "position_ids": np.arange(
            position_offset,
            position_offset + sequence_length,
            dtype=np.int64,
        )[None, :],
        **(past_key_values or _empty_cache_feeds(config)),
    }
    try:
        return session.run(feeds)
    finally:
        session.close()


@pytest.fixture(scope="module")
def synthetic_models():
    torch.manual_seed(42)
    hf_config = _tiny_hf_config()
    hf_model = transformers.MoonshineForConditionalGeneration(hf_config).eval()
    package, config = _weighted_package(hf_model, hf_config)
    return hf_model, package, config


@pytest.fixture(scope="module")
def real_models():
    package = build(_MODEL_ID, dtype="f32", load_weights=True)
    hf_model = transformers.AutoModelForSpeechSeq2Seq.from_pretrained(_MODEL_ID).eval()
    processor = transformers.AutoProcessor.from_pretrained(_MODEL_ID)
    config = package.config
    assert isinstance(config, MoonshineConfig)
    return hf_model, processor, package, config


@pytest.fixture(scope="module")
def real_audio_inputs(real_models):
    _hf_model, processor, _package, _config = real_models
    audio, _sample_rate = librosa.load(str(_AUDIO_PATH), sr=16_000)
    return processor(audio, sampling_rate=16_000, return_tensors="np")


class TestMoonshineSyntheticParity:
    """L3 random-weight parity against a reduced Hugging Face Moonshine model."""

    def test_encoder_hidden_states_and_mask(self, synthetic_models):
        hf_model, package, _config = synthetic_models
        rng = np.random.default_rng(42)
        input_values = rng.standard_normal((1, 4096)).astype(np.float32)
        attention_mask = np.ones_like(input_values, dtype=np.int64)
        attention_mask[:, -256:] = 0

        with torch.no_grad():
            expected = hf_model.get_encoder()(
                input_values=torch.from_numpy(input_values),
                attention_mask=torch.from_numpy(attention_mask),
            )
        actual = _run_encoder(package, input_values, attention_mask)

        np.testing.assert_array_equal(
            actual["encoder_attention_mask"], expected.attention_mask.numpy()
        )
        assert_logits_close(
            actual["encoder_hidden_states"],
            expected.last_hidden_state.numpy(),
            rtol=1e-3,
            atol=1e-3,
        )

    def test_decoder_prefill_and_cached_decode(self, synthetic_models):
        hf_model, package, config = synthetic_models
        rng = np.random.default_rng(7)
        input_values = rng.standard_normal((1, 4096)).astype(np.float32)
        attention_mask = np.ones_like(input_values, dtype=np.int64)
        decoder_input_ids = np.array([[1, 17, 29]], dtype=np.int64)

        with torch.no_grad():
            encoder_output = hf_model.get_encoder()(
                input_values=torch.from_numpy(input_values),
                attention_mask=torch.from_numpy(attention_mask),
            )
            expected_prefill = hf_model.model.decoder(
                input_ids=torch.from_numpy(decoder_input_ids),
                encoder_hidden_states=encoder_output.last_hidden_state,
                encoder_attention_mask=encoder_output.attention_mask,
                use_cache=True,
            )
            expected_prefill_logits = hf_model.proj_out(
                expected_prefill.last_hidden_state
            ).numpy()

        actual_prefill = _run_decoder(
            package,
            config,
            decoder_input_ids,
            encoder_output.last_hidden_state.numpy(),
            encoder_output.attention_mask.numpy(),
        )
        assert_logits_close(
            actual_prefill["logits"],
            expected_prefill_logits,
            rtol=1e-3,
            atol=1e-3,
        )

        next_token = expected_prefill_logits[:, -1:].argmax(axis=-1).astype(np.int64)
        with torch.no_grad():
            expected_decode = hf_model.model.decoder(
                input_ids=torch.from_numpy(next_token),
                encoder_hidden_states=encoder_output.last_hidden_state,
                encoder_attention_mask=encoder_output.attention_mask,
                past_key_values=expected_prefill.past_key_values,
                use_cache=True,
            )
            expected_decode_logits = hf_model.proj_out(
                expected_decode.last_hidden_state
            ).numpy()

        cache = {}
        for layer_idx in range(config.num_hidden_layers):
            for kind in ("key", "value"):
                cache[f"past_key_values.{layer_idx}.{kind}"] = actual_prefill[
                    f"present.{layer_idx}.{kind}"
                ]
        actual_decode = _run_decoder(
            package,
            config,
            next_token,
            encoder_output.last_hidden_state.numpy(),
            encoder_output.attention_mask.numpy(),
            past_key_values=cache,
            position_offset=decoder_input_ids.shape[1],
        )
        assert_logits_close(
            actual_decode["logits"],
            expected_decode_logits,
            rtol=1e-3,
            atol=1e-3,
        )


@pytest.mark.integration
@pytest.mark.integration_fast
class TestMoonshineRealWeightParity:
    """L3 checkpoint parity for both Moonshine encoder and cached decoder."""

    def test_real_encoder_hidden_states(self, real_models, real_audio_inputs):
        hf_model, _processor, package, _config = real_models
        with torch.no_grad():
            expected = hf_model.get_encoder()(
                input_values=torch.from_numpy(real_audio_inputs["input_values"]),
                attention_mask=torch.from_numpy(real_audio_inputs["attention_mask"]),
            )
        actual = _run_encoder(
            package,
            real_audio_inputs["input_values"],
            real_audio_inputs["attention_mask"],
        )

        np.testing.assert_array_equal(
            actual["encoder_attention_mask"], expected.attention_mask.numpy()
        )
        assert_logits_close(
            actual["encoder_hidden_states"],
            expected.last_hidden_state.numpy(),
            rtol=1e-3,
            atol=1e-3,
        )

    def test_real_decoder_prefill_and_cached_decode(self, real_models, real_audio_inputs):
        hf_model, _processor, package, config = real_models
        decoder_input_ids = np.array([[1, 42, 57]], dtype=np.int64)
        with torch.no_grad():
            encoder_output = hf_model.get_encoder()(
                input_values=torch.from_numpy(real_audio_inputs["input_values"]),
                attention_mask=torch.from_numpy(real_audio_inputs["attention_mask"]),
            )
            expected_prefill = hf_model.model.decoder(
                input_ids=torch.from_numpy(decoder_input_ids),
                encoder_hidden_states=encoder_output.last_hidden_state,
                encoder_attention_mask=encoder_output.attention_mask,
                use_cache=True,
            )
            expected_prefill_logits = hf_model.proj_out(
                expected_prefill.last_hidden_state
            ).numpy()

        actual_prefill = _run_decoder(
            package,
            config,
            decoder_input_ids,
            encoder_output.last_hidden_state.numpy(),
            encoder_output.attention_mask.numpy(),
        )
        assert_logits_close(
            actual_prefill["logits"],
            expected_prefill_logits,
            rtol=1e-3,
            atol=1e-3,
        )

        next_token = expected_prefill_logits[:, -1:].argmax(axis=-1).astype(np.int64)
        with torch.no_grad():
            expected_decode = hf_model.model.decoder(
                input_ids=torch.from_numpy(next_token),
                encoder_hidden_states=encoder_output.last_hidden_state,
                encoder_attention_mask=encoder_output.attention_mask,
                past_key_values=expected_prefill.past_key_values,
                use_cache=True,
            )
            expected_decode_logits = hf_model.proj_out(
                expected_decode.last_hidden_state
            ).numpy()

        cache = {}
        for layer_idx in range(config.num_hidden_layers):
            for kind in ("key", "value"):
                cache[f"past_key_values.{layer_idx}.{kind}"] = actual_prefill[
                    f"present.{layer_idx}.{kind}"
                ]
        actual_decode = _run_decoder(
            package,
            config,
            next_token,
            encoder_output.last_hidden_state.numpy(),
            encoder_output.attention_mask.numpy(),
            past_key_values=cache,
            position_offset=decoder_input_ids.shape[1],
        )
        assert_logits_close(
            actual_decode["logits"],
            expected_decode_logits,
            rtol=1e-3,
            atol=1e-3,
        )
