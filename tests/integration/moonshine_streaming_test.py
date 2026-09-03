# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Numerical parity tests for Moonshine Streaming speech recognition.

Covers the pieces that make ``moonshine_streaming`` different from offline
Moonshine: the raw-waveform framing front end, causal strided convolutions with
mask propagation, per-layer asymmetric ``(left, right)`` sliding windows, the
unit-offset encoder LayerNorm, and the decoder's absolute position context
adapter. Real-weight cases run against a pinned checkpoint revision and real
nonzero speech, and chain the ONNX decoder off the ONNX encoder output rather
than a HuggingFace intermediate.
"""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import pytest
import torch
import transformers

from mobius import build, build_from_module
from mobius._configs import MoonshineStreamingConfig
from mobius._testing.comparison import assert_logits_close
from mobius._testing.golden import (
    discover_test_cases,
    generation_json_path_for_case,
    load_generation_golden,
)
from mobius._testing.ort_inference import OnnxModelSession
from mobius.models import MoonshineStreamingForConditionalGeneration
from mobius.tasks import SpeechToTextTask

_MODEL_ID = "moonshine-ai/moonshine-streaming-tiny"
_REVISION = "f8e9dfd8c562c257c151a907b7b7f2fe8ff8511a"
_AUDIO_PATH = Path(__file__).parent.parent.parent / "testdata" / "652-129742-0006.flac"
_EOS_TOKEN_ID = 2

pytestmark = pytest.mark.skipif(
    not hasattr(transformers, "MoonshineStreamingConfig"),
    reason="transformers build has no moonshine_streaming support",
)


def _tiny_hf_config(
    sliding_windows: list[list[int]] | None = None, attention_bias: bool = False
):
    """A production-shaped but tiny Moonshine Streaming config.

    Keeps the real frame length (80 samples), the real partial rotary factor and
    a mixed lookahead schedule so the window logic is genuinely exercised.
    """
    encoder_config_cls = transformers.models.moonshine_streaming.configuration_moonshine_streaming.MoonshineStreamingEncoderConfig
    encoder_config = encoder_config_cls(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=16,
        hidden_act="gelu",
        attention_bias=attention_bias,
        sliding_windows=sliding_windows or [[16, 4], [16, 0]],
        sample_rate=16_000,
        frame_ms=5.0,
    )
    return transformers.MoonshineStreamingConfig(
        encoder_config=encoder_config,
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=16,
        max_position_embeddings=128,
        attention_bias=attention_bias,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10_000.0,
            "partial_rotary_factor": 0.8,
        },
        tie_word_embeddings=False,
        decoder_start_token_id=1,
    )


def _weighted_package(hf_model, hf_config):
    config = MoonshineStreamingConfig.from_transformers(hf_config)
    module = MoonshineStreamingForConditionalGeneration(config)
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


def _empty_cache_feeds(
    config: MoonshineStreamingConfig,
    batch_size: int = 1,
    dtype: np.dtype | type = np.float32,
):
    feeds = {}
    for layer_idx in range(config.num_hidden_layers):
        for kind in ("key", "value"):
            feeds[f"past_key_values.{layer_idx}.{kind}"] = np.zeros(
                (batch_size, config.num_key_value_heads, 0, config.head_dim),
                dtype=dtype,
            )
    return feeds


def _decoder_feeds(
    config,
    decoder_input_ids,
    encoder_hidden_states,
    encoder_attention_mask,
    past_key_values=None,
    position_offset=0,
    dtype: np.dtype | type = np.float32,
):
    batch_size, sequence_length = decoder_input_ids.shape
    position_ids = np.arange(
        position_offset, position_offset + sequence_length, dtype=np.int64
    )
    return {
        "decoder_input_ids": decoder_input_ids.astype(np.int64),
        "encoder_hidden_states": encoder_hidden_states.astype(dtype),
        "encoder_attention_mask": encoder_attention_mask.astype(np.int64),
        "position_ids": np.broadcast_to(
            position_ids[None, :], (batch_size, sequence_length)
        ).copy(),
        **(past_key_values or _empty_cache_feeds(config, batch_size, dtype=dtype)),
    }


def _run_decoder(package, config, *args, **kwargs):
    session = OnnxModelSession(package["decoder"])
    try:
        return session.run(_decoder_feeds(config, *args, **kwargs))
    finally:
        session.close()


def _onnx_greedy_generate(package, config, encoder_outputs, max_new_tokens, start_id):
    """Greedy decode entirely through the exported ONNX decoder."""
    session = OnnxModelSession(package["decoder"])
    try:
        dtype = session.get_input_dtype("past_key_values.0.key") or np.float32
        cache = _empty_cache_feeds(config, dtype=dtype)
        current = np.array([[start_id]], dtype=np.int64)
        generated: list[int] = []
        for step in range(max_new_tokens):
            outputs = session.run(
                _decoder_feeds(
                    config,
                    current,
                    encoder_outputs["encoder_hidden_states"],
                    encoder_outputs["encoder_attention_mask"],
                    past_key_values=cache,
                    position_offset=step,
                    dtype=dtype,
                )
            )
            next_token = int(np.asarray(outputs["logits"], dtype=np.float32)[0, -1].argmax())
            generated.append(next_token)
            if next_token == _EOS_TOKEN_ID:
                break
            current = np.array([[next_token]], dtype=np.int64)
            cache = {
                f"past_key_values.{layer_idx}.{kind}": outputs[f"present.{layer_idx}.{kind}"]
                for layer_idx in range(config.num_hidden_layers)
                for kind in ("key", "value")
            }
        return generated
    finally:
        session.close()


@pytest.fixture(scope="module")
def synthetic_models():
    torch.manual_seed(42)
    hf_config = _tiny_hf_config()
    hf_model = transformers.MoonshineStreamingForConditionalGeneration(hf_config).eval()
    package, config = _weighted_package(hf_model, hf_config)
    return hf_model, package, config


def _nonmutating_hf_greedy(hf_model, encoder_hidden_states, encoder_attention_mask, steps):
    """Greedy decode with a pristine encoder context handed to every step.

    HuggingFace's ``MoonshineStreamingDecoder`` adds its absolute position table
    to ``encoder_hidden_states`` with ``+=``, mutating the caller's tensor. With
    cached cross-attention the mutated tensor is never read again, so the result
    is unaffected — but the reference this test compares against must not depend
    on that, so each step gets a fresh clone. This is the same semantics the
    exported ONNX decoder implements: encoder context is added exactly once.

    Returns ``(tokens, per_step_logits)``.
    """
    from transformers.cache_utils import DynamicCache, EncoderDecoderCache

    cache = EncoderDecoderCache(
        DynamicCache(config=hf_model.config), DynamicCache(config=hf_model.config)
    )
    current = torch.tensor([[hf_model.config.decoder_start_token_id]], dtype=torch.long)
    tokens: list[int] = []
    per_step: list[np.ndarray] = []
    for step in range(steps):
        with torch.no_grad():
            output = hf_model.model.decoder(
                input_ids=current,
                encoder_hidden_states=encoder_hidden_states.clone(),
                encoder_attention_mask=encoder_attention_mask,
                past_key_values=cache,
                position_ids=torch.tensor([[step]], dtype=torch.long),
                use_cache=True,
            )
            logits = hf_model.proj_out(output.last_hidden_state)
        cache = output.past_key_values
        per_step.append(logits[0, -1].float().numpy())
        token = int(logits[0, -1].argmax())
        tokens.append(token)
        if token == _EOS_TOKEN_ID:
            break
        current = torch.tensor([[token]], dtype=torch.long)
    return tokens, per_step


@pytest.fixture(scope="module")
def real_models():
    package = build(_MODEL_ID, dtype="f32", load_weights=True, revision=_REVISION)
    hf_model = transformers.AutoModelForSpeechSeq2Seq.from_pretrained(
        _MODEL_ID, revision=_REVISION
    ).eval()
    processor = transformers.AutoProcessor.from_pretrained(_MODEL_ID, revision=_REVISION)
    config = package.config
    assert isinstance(config, MoonshineStreamingConfig)
    return hf_model, processor, package, config


@pytest.fixture(scope="module")
def real_audio_inputs(real_models):
    _hf_model, processor, _package, _config = real_models
    audio, _sample_rate = librosa.load(str(_AUDIO_PATH), sr=16_000)
    assert np.abs(audio).max() > 0.1, "fixture must be real nonzero speech"
    return processor(audio, sampling_rate=16_000, return_tensors="np")


class TestMoonshineStreamingConfigExtraction:
    """Config extraction preserves the streaming-specific architecture fields."""

    def test_extracts_encoder_and_window_schedule(self):
        config = MoonshineStreamingConfig.from_transformers(_tiny_hf_config())
        assert config.model_type == "moonshine_streaming"
        assert config.encoder_sliding_windows == ((16, 4), (16, 0))
        assert config.encoder_hidden_act == "gelu"
        assert config.decoder_hidden_act == "silu"
        assert config.partial_rotary_factor == pytest.approx(0.8)
        assert config.rope_interleave is True
        assert config.tie_word_embeddings is False
        # 16 kHz * 5 ms = 80 raw samples per encoder frame.
        assert config.frame_length == 80
        assert config.encoder_output_size == config.encoder_hidden_size == 64

    def test_extracts_from_raw_pinned_json(self):
        """The pinned checkpoint's raw JSON (nested dict sub-config) resolves."""
        raw = transformers.AutoConfig.from_pretrained(_MODEL_ID, revision=_REVISION)
        config = MoonshineStreamingConfig.from_transformers(raw)
        assert config.encoder_num_hidden_layers == 6
        assert config.num_hidden_layers == 6
        assert config.encoder_sliding_windows == (
            (16, 4),
            (16, 4),
            (16, 0),
            (16, 0),
            (16, 4),
            (16, 4),
        )
        assert config.head_dim == 40
        assert config.encoder_head_dim == 40
        assert config.vocab_size == 32_768
        assert config.decoder_start_token_id == 1

    def test_rejects_other_model_types(self):
        config = _tiny_hf_config()
        config.model_type = "moonshine"
        with pytest.raises(ValueError, match="moonshine_streaming"):
            MoonshineStreamingConfig.from_transformers(config)


def test_moonshine_streaming_l5_generation_reference_is_declared_and_valid():
    cases = [case for case in discover_test_cases(level="L5") if case.model_id == _MODEL_ID]
    assert len(cases) == 1
    case = cases[0]
    assert case.revision == _REVISION
    generation_path = generation_json_path_for_case(case)
    assert generation_path.exists()
    generated_tokens = load_generation_golden(case)
    assert len(generated_tokens) >= 20
    assert all(isinstance(token, int) for token in generated_tokens)


class TestMoonshineStreamingSyntheticParity:
    """L3 random-weight parity against a reduced HuggingFace streaming model."""

    def test_encoder_hidden_states_and_mask(self, synthetic_models):
        hf_model, package, _config = synthetic_models
        rng = np.random.default_rng(42)
        # 200 frames of 80 samples; only the first 60 frames are real audio, so
        # the causal-conv mask propagation is genuinely exercised.
        input_values = (rng.standard_normal((1, 80 * 200)) * 0.05).astype(np.float32)
        attention_mask = np.ones_like(input_values, dtype=np.int64)
        attention_mask[:, 80 * 60 :] = 0

        with torch.no_grad():
            expected = hf_model.get_encoder()(
                input_values=torch.from_numpy(input_values),
                attention_mask=torch.from_numpy(attention_mask),
            )
        actual = _run_encoder(package, input_values, attention_mask)

        expected_mask = expected.attention_mask.numpy()
        np.testing.assert_array_equal(
            actual["encoder_attention_mask"].astype(bool), expected_mask
        )
        # Padded frames are excluded everywhere downstream, and HuggingFace and
        # ONNX fill fully-masked attention rows differently, so parity is
        # asserted on the valid frames the model actually consumes.
        valid = expected_mask[0].astype(bool)
        assert valid.sum() < expected_mask.shape[1], "padding should shrink the mask"
        assert_logits_close(
            actual["encoder_hidden_states"][:, valid],
            expected.last_hidden_state.numpy()[:, valid],
            rtol=1e-3,
            atol=1e-3,
        )

    @pytest.mark.parametrize(
        "sliding_windows",
        [[[2, 1], [3, 0]], [[16, 4], [16, 0]], [[64, 64], [64, 64]]],
        ids=["tight", "default", "global"],
    )
    def test_encoder_matches_each_window_schedule(self, sliding_windows):
        """Asymmetric left/right windows and lookahead reproduce upstream."""
        torch.manual_seed(7)
        hf_config = _tiny_hf_config(sliding_windows)
        hf_model = transformers.MoonshineStreamingForConditionalGeneration(hf_config).eval()
        package, _config = _weighted_package(hf_model, hf_config)

        rng = np.random.default_rng(3)
        input_values = (rng.standard_normal((1, 80 * 40)) * 0.05).astype(np.float32)
        attention_mask = np.ones_like(input_values, dtype=np.int64)
        with torch.no_grad():
            expected = hf_model.get_encoder()(
                input_values=torch.from_numpy(input_values),
                attention_mask=torch.from_numpy(attention_mask),
            )
        actual = _run_encoder(package, input_values, attention_mask)
        assert_logits_close(
            actual["encoder_hidden_states"],
            expected.last_hidden_state.numpy(),
            rtol=1e-3,
            atol=1e-3,
        )

    @pytest.mark.parametrize("attention_bias", [False, True], ids=["nobias", "bias"])
    def test_encoder_matches_with_and_without_attention_bias(self, attention_bias):
        """Upstream gates encoder q/k/v/o (and decoder q/k/v) on attention_bias."""
        torch.manual_seed(11)
        hf_config = _tiny_hf_config(attention_bias=attention_bias)
        hf_model = transformers.MoonshineStreamingForConditionalGeneration(hf_config).eval()
        package, config = _weighted_package(hf_model, hf_config)
        assert config.encoder_attention_bias is attention_bias
        assert config.attn_qkv_bias is attention_bias

        rng = np.random.default_rng(5)
        input_values = (rng.standard_normal((1, 80 * 40)) * 0.05).astype(np.float32)
        attention_mask = np.ones_like(input_values, dtype=np.int64)
        with torch.no_grad():
            expected = hf_model.get_encoder()(
                input_values=torch.from_numpy(input_values),
                attention_mask=torch.from_numpy(attention_mask),
            )
        actual = _run_encoder(package, input_values, attention_mask)
        assert_logits_close(
            actual["encoder_hidden_states"],
            expected.last_hidden_state.numpy(),
            rtol=1e-3,
            atol=1e-3,
        )

    def test_batched_ragged_padding_matches_huggingface(self, synthetic_models):
        """Two rows of different real length: mask, encoder and decoder all agree."""
        hf_model, package, config = synthetic_models
        rng = np.random.default_rng(21)
        input_values = (rng.standard_normal((3, 80 * 150)) * 0.05).astype(np.float32)
        attention_mask = np.ones_like(input_values, dtype=np.int64)
        # Ragged: 150, 90 and 40 valid frames.
        attention_mask[1, 80 * 90 :] = 0
        attention_mask[2, 80 * 40 :] = 0

        with torch.no_grad():
            expected = hf_model.get_encoder()(
                input_values=torch.from_numpy(input_values),
                attention_mask=torch.from_numpy(attention_mask),
            )
        actual = _run_encoder(package, input_values, attention_mask)

        expected_mask = expected.attention_mask.numpy()
        np.testing.assert_array_equal(
            actual["encoder_attention_mask"].astype(bool), expected_mask
        )
        assert len(set(expected_mask.sum(axis=-1).tolist())) == 3, (
            "each row should keep a different number of valid frames"
        )
        expected_hidden = expected.last_hidden_state.numpy()
        for row in range(input_values.shape[0]):
            valid = expected_mask[row].astype(bool)
            assert_logits_close(
                actual["encoder_hidden_states"][row][valid],
                expected_hidden[row][valid],
                rtol=1e-3,
                atol=1e-3,
            )

        decoder_input_ids = np.array([[1, 5], [1, 9], [1, 13]], dtype=np.int64)
        with torch.no_grad():
            prefill = hf_model.model.decoder(
                input_ids=torch.from_numpy(decoder_input_ids),
                encoder_hidden_states=expected.last_hidden_state.clone(),
                encoder_attention_mask=expected.attention_mask,
                use_cache=True,
            )
            expected_logits = hf_model.proj_out(prefill.last_hidden_state).numpy()

        session = OnnxModelSession(package["decoder"])
        try:
            actual_logits = session.run(
                _decoder_feeds(
                    config,
                    decoder_input_ids,
                    expected_hidden,
                    expected_mask.astype(np.int64),
                    past_key_values=_empty_cache_feeds(config, batch_size=3),
                )
            )["logits"]
        finally:
            session.close()
        assert_logits_close(actual_logits, expected_logits, rtol=1e-3, atol=1e-3)

    def test_decoder_prefill_and_cached_decode(self, synthetic_models):
        hf_model, package, config = synthetic_models
        rng = np.random.default_rng(7)
        input_values = (rng.standard_normal((1, 80 * 120)) * 0.05).astype(np.float32)
        attention_mask = np.ones_like(input_values, dtype=np.int64)
        decoder_input_ids = np.array([[1, 17, 29]], dtype=np.int64)

        with torch.no_grad():
            encoder_output = hf_model.get_encoder()(
                input_values=torch.from_numpy(input_values),
                attention_mask=torch.from_numpy(attention_mask),
            )
            # The decoder adds pos_emb to encoder_hidden_states in place, so
            # hand it a clone and keep the pristine tensor for the ONNX feed.
            expected_prefill = hf_model.model.decoder(
                input_ids=torch.from_numpy(decoder_input_ids),
                encoder_hidden_states=encoder_output.last_hidden_state.clone(),
                encoder_attention_mask=encoder_output.attention_mask,
                use_cache=True,
            )
            expected_prefill_logits = hf_model.proj_out(
                expected_prefill.last_hidden_state
            ).numpy()

        encoder_hidden_states = encoder_output.last_hidden_state.numpy()
        encoder_attention_mask = encoder_output.attention_mask.numpy()
        actual_prefill = _run_decoder(
            package,
            config,
            decoder_input_ids,
            encoder_hidden_states,
            encoder_attention_mask,
        )
        assert_logits_close(
            actual_prefill["logits"], expected_prefill_logits, rtol=1e-3, atol=1e-3
        )

        next_token = expected_prefill_logits[:, -1:].argmax(axis=-1).astype(np.int64)
        with torch.no_grad():
            expected_decode = hf_model.model.decoder(
                input_ids=torch.from_numpy(next_token),
                encoder_hidden_states=encoder_output.last_hidden_state.clone(),
                encoder_attention_mask=encoder_output.attention_mask,
                past_key_values=expected_prefill.past_key_values,
                use_cache=True,
            )
            expected_decode_logits = hf_model.proj_out(
                expected_decode.last_hidden_state
            ).numpy()

        cache = {
            f"past_key_values.{layer_idx}.{kind}": actual_prefill[
                f"present.{layer_idx}.{kind}"
            ]
            for layer_idx in range(config.num_hidden_layers)
            for kind in ("key", "value")
        }
        actual_decode = _run_decoder(
            package,
            config,
            next_token,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_values=cache,
            position_offset=decoder_input_ids.shape[1],
        )
        assert_logits_close(
            actual_decode["logits"], expected_decode_logits, rtol=1e-3, atol=1e-3
        )


@pytest.mark.integration
@pytest.mark.integration_fast
class TestMoonshineStreamingRealWeightParity:
    """L3 checkpoint parity on real speech, pinned to one revision."""

    def test_processor_contract(self, real_models, real_audio_inputs):
        """The audio processor emits frame-aligned raw samples plus a mask."""
        _hf_model, processor, _package, config = real_models
        assert set(real_audio_inputs) == {"input_values", "attention_mask"}
        input_values = real_audio_inputs["input_values"]
        assert input_values.ndim == 2
        assert input_values.dtype == np.float32
        # Framing reshape requires a multiple of the frame length.
        assert input_values.shape[1] % config.frame_length == 0
        assert real_audio_inputs["attention_mask"].shape == input_values.shape

        # The waveform is consumed raw: no normalisation, 16 kHz, and the frame
        # alignment the encoder needs is guaranteed by pad_to_multiple_of.
        extractor = processor.feature_extractor
        assert extractor.sampling_rate == config.encoder_sample_rate
        assert extractor.do_normalize is False
        assert extractor.return_attention_mask is True
        assert extractor.pad_to_multiple_of == config.frame_length

    def test_ort_genai_runtime_is_rejected(self, real_models, tmp_path):
        """ORT GenAI cannot host the variable-length raw-waveform encoder."""
        from mobius.integrations.ort_genai import write_ort_genai_config

        _hf_model, _processor, package, _config = real_models
        with pytest.raises(NotImplementedError, match="raw-waveform encoder"):
            write_ort_genai_config(package, str(tmp_path))

    def test_real_encoder_hidden_states(self, real_models, real_audio_inputs):
        hf_model, _processor, package, _config = real_models
        with torch.no_grad():
            expected = hf_model.get_encoder()(
                input_values=torch.from_numpy(real_audio_inputs["input_values"]),
                attention_mask=torch.from_numpy(
                    real_audio_inputs["attention_mask"].astype(np.int64)
                ),
            )
        actual = _run_encoder(
            package,
            real_audio_inputs["input_values"],
            real_audio_inputs["attention_mask"],
        )

        np.testing.assert_array_equal(
            actual["encoder_attention_mask"].astype(bool), expected.attention_mask.numpy()
        )
        assert_logits_close(
            actual["encoder_hidden_states"],
            expected.last_hidden_state.numpy(),
            rtol=1e-3,
            atol=1e-3,
        )

    def test_real_decoder_prefill_from_onnx_encoder(self, real_models, real_audio_inputs):
        """Chain ONNX encoder -> ONNX decoder and compare with the full HF model."""
        hf_model, _processor, package, config = real_models
        start_id = config.decoder_start_token_id
        decoder_input_ids = np.array([[start_id]], dtype=np.int64)

        with torch.no_grad():
            expected = hf_model(
                input_values=torch.from_numpy(real_audio_inputs["input_values"]),
                attention_mask=torch.from_numpy(
                    real_audio_inputs["attention_mask"].astype(np.int64)
                ),
                decoder_input_ids=torch.from_numpy(decoder_input_ids),
            )

        encoder_outputs = _run_encoder(
            package,
            real_audio_inputs["input_values"],
            real_audio_inputs["attention_mask"],
        )
        actual = _run_decoder(
            package,
            config,
            decoder_input_ids,
            encoder_outputs["encoder_hidden_states"],
            encoder_outputs["encoder_attention_mask"],
        )
        assert_logits_close(actual["logits"], expected.logits.numpy(), rtol=1e-3, atol=1e-3)

    def test_golden_reference_guard_blocks_encoder_mutation(self, real_models):
        """The golden generator's guard really stops the in-place ``+=``.

        Without it the reference decoder rewrites the caller's encoder context
        on every step, so this asserts the unguarded mutation exists (otherwise
        the guard would be silently vacuous) and that the guard removes it.
        """
        import sys

        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        try:
            from generate_golden import non_mutating_encoder_context
        finally:
            sys.path.pop(0)

        hf_model, _processor, _package, _config = real_models
        with torch.no_grad():
            encoder = hf_model.get_encoder()(
                input_values=torch.from_numpy(np.zeros((1, 80 * 40), dtype=np.float32) + 0.01),
                attention_mask=torch.ones((1, 80 * 40), dtype=torch.long),
            )
        pristine = encoder.last_hidden_state.clone()

        def decode_once(states):
            with torch.no_grad():
                hf_model.model.decoder(
                    input_ids=torch.tensor([[1]], dtype=torch.long),
                    encoder_hidden_states=states,
                    encoder_attention_mask=encoder.attention_mask,
                    use_cache=True,
                )

        unguarded = pristine.clone()
        decode_once(unguarded)
        assert not torch.equal(unguarded, pristine), (
            "upstream decoder is expected to mutate encoder_hidden_states in place"
        )

        guarded = pristine.clone()
        with non_mutating_encoder_context(hf_model):
            decode_once(guarded)
        assert torch.equal(guarded, pristine)

    def test_decoder_adds_encoder_context_exactly_once(self, real_models, real_audio_inputs):
        """The ONNX decoder must not accumulate encoder context across steps.

        HuggingFace's decoder adds ``pos_emb`` to ``encoder_hidden_states`` in
        place, so a decoder that re-used a mutated buffer would drift. The
        exported graph takes the encoder output as a read-only input and adds
        the position table on every call, so replaying step 0 after a full
        decode must reproduce the original step-0 logits bit for bit, and the
        input buffer must come back untouched.
        """
        _hf_model, _processor, package, config = real_models
        encoder_outputs = _run_encoder(
            package,
            real_audio_inputs["input_values"],
            real_audio_inputs["attention_mask"],
        )
        encoder_hidden_states = encoder_outputs["encoder_hidden_states"]
        pristine = encoder_hidden_states.copy()

        def step_zero_logits():
            session = OnnxModelSession(package["decoder"])
            try:
                return session.run(
                    _decoder_feeds(
                        config,
                        np.array([[config.decoder_start_token_id]], dtype=np.int64),
                        encoder_hidden_states,
                        encoder_outputs["encoder_attention_mask"],
                    )
                )["logits"]
            finally:
                session.close()

        first = step_zero_logits()
        _onnx_greedy_generate(
            package,
            config,
            encoder_outputs,
            max_new_tokens=20,
            start_id=config.decoder_start_token_id,
        )
        replayed = step_zero_logits()

        np.testing.assert_array_equal(encoder_hidden_states, pristine)
        np.testing.assert_array_equal(replayed, first)

    def test_stepwise_logit_parity_with_nonmutating_reference(
        self, real_models, real_audio_inputs
    ):
        """Every decode step matches a non-mutating HuggingFace reference.

        Token-level agreement can hide logit drift, so this compares the full
        vocabulary distribution at every step against a reference that is
        handed a pristine encoder context each call.
        """
        hf_model, processor, package, config = real_models
        encoder_outputs = _run_encoder(
            package,
            real_audio_inputs["input_values"],
            real_audio_inputs["attention_mask"],
        )

        with torch.no_grad():
            expected_encoder = hf_model.get_encoder()(
                input_values=torch.from_numpy(real_audio_inputs["input_values"]),
                attention_mask=torch.from_numpy(
                    real_audio_inputs["attention_mask"].astype(np.int64)
                ),
            )
        expected_tokens, expected_logits = _nonmutating_hf_greedy(
            hf_model,
            expected_encoder.last_hidden_state,
            expected_encoder.attention_mask,
            steps=50,
        )

        session = OnnxModelSession(package["decoder"])
        try:
            cache = _empty_cache_feeds(config)
            current = np.array([[config.decoder_start_token_id]], dtype=np.int64)
            actual_tokens: list[int] = []
            for step in range(len(expected_logits)):
                outputs = session.run(
                    _decoder_feeds(
                        config,
                        current,
                        encoder_outputs["encoder_hidden_states"],
                        encoder_outputs["encoder_attention_mask"],
                        past_key_values=cache,
                        position_offset=step,
                    )
                )
                assert_logits_close(
                    outputs["logits"][0, -1],
                    expected_logits[step],
                    rtol=1e-3,
                    atol=1e-3,
                )
                token = int(outputs["logits"][0, -1].argmax())
                actual_tokens.append(token)
                if token == _EOS_TOKEN_ID:
                    break
                current = np.array([[token]], dtype=np.int64)
                cache = {
                    f"past_key_values.{layer_idx}.{kind}": outputs[
                        f"present.{layer_idx}.{kind}"
                    ]
                    for layer_idx in range(config.num_hidden_layers)
                    for kind in ("key", "value")
                }
        finally:
            session.close()

        assert len(actual_tokens) == len(expected_tokens)
        assert actual_tokens == expected_tokens

        # The committed L5 golden must describe this same non-mutating decode.
        content = (
            expected_tokens[:-1]
            if expected_tokens and expected_tokens[-1] == _EOS_TOKEN_ID
            else expected_tokens
        )
        golden = load_generation_golden(
            next(
                case for case in discover_test_cases(level="L5") if case.model_id == _MODEL_ID
            )
        )
        assert len(content) == len(golden)
        assert content == golden
        assert processor.decode(content, skip_special_tokens=True)

    def test_float16_cpu_parity_and_transcript(self, real_audio_inputs, real_models):
        """fp16 keeps full-logit parity and an identical transcript on ORT CPU."""
        hf_model, processor, _package, _config = real_models
        package = build(_MODEL_ID, dtype="f16", load_weights=True, revision=_REVISION)
        config = package.config

        session = OnnxModelSession(package["encoder"])
        try:
            encoder_outputs = session.run(
                {
                    "input_values": real_audio_inputs["input_values"].astype(np.float16),
                    "attention_mask": real_audio_inputs["attention_mask"].astype(np.int64),
                }
            )
        finally:
            session.close()

        with torch.no_grad():
            expected = hf_model.get_encoder()(
                input_values=torch.from_numpy(real_audio_inputs["input_values"]),
                attention_mask=torch.from_numpy(
                    real_audio_inputs["attention_mask"].astype(np.int64)
                ),
            )
        actual_hidden = np.asarray(encoder_outputs["encoder_hidden_states"], dtype=np.float32)
        np.testing.assert_array_equal(
            encoder_outputs["encoder_attention_mask"].astype(bool),
            expected.attention_mask.numpy(),
        )
        assert not np.isnan(actual_hidden).any()
        np.testing.assert_allclose(
            actual_hidden, expected.last_hidden_state.numpy(), rtol=1e-2, atol=1e-1
        )

        tokens = _onnx_greedy_generate(
            package,
            config,
            encoder_outputs,
            max_new_tokens=50,
            start_id=config.decoder_start_token_id,
        )
        content = tokens[:-1] if tokens and tokens[-1] == _EOS_TOKEN_ID else tokens
        golden = load_generation_golden(
            next(
                case for case in discover_test_cases(level="L5") if case.model_id == _MODEL_ID
            )
        )
        assert len(content) == len(golden)
        assert content == golden
        assert processor.decode(content, skip_special_tokens=True)

    def test_float16_cuda_encoder_first_frame_matches_torch(
        self, real_audio_inputs, real_models
    ):
        """CUDA fp16 preserves the sparse-window first encoder frame."""
        ort = pytest.importorskip("onnxruntime")
        if "CUDAExecutionProvider" not in ort.get_available_providers():
            pytest.skip("CUDAExecutionProvider is not available")
        ort.preload_dlls()

        hf_model, _processor, _package, _config = real_models
        package = build(
            _MODEL_ID,
            dtype="f16",
            load_weights=True,
            revision=_REVISION,
            execution_provider="cuda",
        )
        with torch.no_grad():
            expected = hf_model.get_encoder()(
                input_values=torch.from_numpy(real_audio_inputs["input_values"]),
                attention_mask=torch.from_numpy(
                    real_audio_inputs["attention_mask"].astype(np.int64)
                ),
            )

        session = OnnxModelSession(package["encoder"], device="cuda")
        try:
            if session.providers[0] != "CUDAExecutionProvider":
                pytest.skip("CUDAExecutionProvider could not initialize on this host")
            outputs = session.run(
                {
                    "input_values": real_audio_inputs["input_values"].astype(np.float16),
                    "attention_mask": real_audio_inputs["attention_mask"].astype(np.int64),
                }
            )
            actual_hidden = outputs["encoder_hidden_states"]
            actual_mask = outputs["encoder_attention_mask"]
        finally:
            session.close()

        np.testing.assert_array_equal(
            actual_mask.astype(bool), expected.attention_mask.numpy()
        )
        # The first (16, 4)-window query sees only its four lookahead keys.
        np.testing.assert_allclose(
            actual_hidden[:, 0].astype(np.float32),
            expected.last_hidden_state.numpy()[:, 0],
            rtol=1e-2,
            atol=1e-1,
        )

    def test_real_generation_matches_huggingface(self, real_models, real_audio_inputs):
        """Full ONNX pipeline transcribes identically to HuggingFace generate()."""
        hf_model, processor, package, config = real_models
        encoder_outputs = _run_encoder(
            package,
            real_audio_inputs["input_values"],
            real_audio_inputs["attention_mask"],
        )
        actual_tokens = _onnx_greedy_generate(
            package,
            config,
            encoder_outputs,
            max_new_tokens=50,
            start_id=config.decoder_start_token_id,
        )

        with torch.no_grad():
            generated = hf_model.generate(
                input_values=torch.from_numpy(real_audio_inputs["input_values"]),
                attention_mask=torch.from_numpy(
                    real_audio_inputs["attention_mask"].astype(np.int64)
                ),
                max_new_tokens=50,
                do_sample=False,
            )
        expected_tokens = generated[0].tolist()
        if expected_tokens and expected_tokens[0] == config.decoder_start_token_id:
            expected_tokens = expected_tokens[1:]

        assert len(actual_tokens) == len(expected_tokens)
        assert actual_tokens == expected_tokens
        transcript = processor.decode(actual_tokens, skip_special_tokens=True)
        assert transcript == processor.decode(expected_tokens, skip_special_tokens=True)
        assert len(transcript.split()) > 5, transcript
