# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for the CTC ASR metadata contract and its metadata-driven runtime."""

from __future__ import annotations

import numpy as np
import pytest

from mobius._configs import MMSConfig
from mobius.integrations.onnx_genai import ctc_runtime
from mobius.integrations.onnx_genai.workflow_metadata import (
    build_ctc_asr_workflow_metadata,
)
from mobius.models.wav2vec2_ctc import Wav2Vec2ForCTCModel
from mobius.tasks._ctc_asr import CTCAsrTask


def _tiny_config(**overrides) -> MMSConfig:
    base = {
        "vocab_size": 32,
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "conv_dim": (32, 32, 64),
        "conv_kernel": (10, 3, 2),
        "conv_stride": (5, 2, 2),
        "conv_bias": False,
        "feat_extract_norm": "group",
        "do_stable_layer_norm": False,
        "pad_token_id": 0,
    }
    base.update(overrides)
    return MMSConfig(**base)


def _build(config: MMSConfig):
    return CTCAsrTask().build(Wav2Vec2ForCTCModel(config), config)


class TestFeatureExtractOutputLength:
    """The analytic frame count must mirror the convolution stack exactly."""

    def test_matches_manual_convolution_arithmetic(self):
        config = _tiny_config()
        samples = 4000
        expected = samples
        for kernel, stride in zip(config.conv_kernel, config.conv_stride):
            expected = (expected - kernel) // stride + 1
        assert config.feature_extract_output_length(samples) == expected

    def test_wav2vec2_base_geometry_downsamples_by_320(self):
        config = _tiny_config(
            conv_dim=(512,) * 7,
            conv_kernel=(10, 3, 3, 3, 3, 2, 2),
            conv_stride=(5, 2, 2, 2, 2, 2, 2),
        )
        # 16 kHz audio yields 50 frames per second.
        assert config.feature_extract_output_length(16_000) == 49

    def test_rejects_ragged_convolution_geometry(self):
        with pytest.raises(ValueError):
            _tiny_config(conv_kernel=(10, 3))

    def test_rejects_unknown_feature_normalization(self):
        with pytest.raises(ValueError):
            _tiny_config(feat_extract_norm="batch")


class TestArchitectureRouting:
    def test_config_class_survives_architecture_rerouting(self):
        # Config resolution reaches this module by re-routing model_type
        # "wav2vec2" to the "mms" registration, so the module must name its own
        # config class or the convolution geometry silently reverts to defaults.
        assert Wav2Vec2ForCTCModel.config_class is MMSConfig


class TestCtcAsrMetadata:
    """The emitted document must fully describe a frame-synchronous package."""

    @pytest.fixture
    def metadata(self):
        config = _tiny_config()
        return build_ctc_asr_workflow_metadata(_build(config), config)

    def test_declares_audio_preprocessing_bound_to_graph_inputs(self, metadata):
        audio = metadata["preprocessing"]["audio"]
        assert [t["op"] for t in audio["transforms"]] == [
            "decode",
            "resample",
            "downmix",
            "zero_mean_unit_variance",
            "pad",
        ]
        assert {o["name"] for o in audio["outputs"]} == {
            "input_values",
            "attention_mask",
        }

    def test_transcription_profile_carries_a_ctc_decoding_contract(self, metadata):
        decoding = metadata["profiles"]["transcription"]["decoding"]
        assert decoding["kind"] == "ctc"
        assert decoding["blank_id"] == 0
        assert decoding["collapse_repeats"] is True
        assert (decoding["time_axis"], decoding["class_axis"]) == (1, 2)

    def test_frame_lengths_bind_the_ctc_decode(self, metadata):
        profile = metadata["profiles"]["transcription"]
        assert profile["decoding"]["lengths"] == "frame_lengths"
        assert profile["outputs"]["frame_lengths"] == "frame_lengths"

    @pytest.mark.parametrize("normalization", ["group", "layer"])
    def test_batch_permission_is_never_inferred_from_shape_or_normalization(
        self, normalization
    ):
        config = _tiny_config(feat_extract_norm=normalization)
        metadata = build_ctc_asr_workflow_metadata(_build(config), config)
        profile = metadata["profiles"]["transcription"]
        component = metadata["pipeline"]["workflow"]["components"]["encoder"]
        assert "batch_invariance" not in profile
        assert "batch_capacity" not in component

    def test_workflow_is_a_plain_sequence_with_no_generation_loop(self, metadata):
        steps = metadata["pipeline"]["workflow"]["steps"]
        assert {step["kind"] for step in steps} <= {"invoke", "emit"}
        assert not metadata["pipeline"]["workflow"].get("state")

    def test_encoder_is_invoked_exactly_once(self, metadata):
        steps = metadata["pipeline"]["workflow"]["steps"]
        invocations = [s for s in steps if s["kind"] == "invoke"]
        assert sum(1 for s in invocations if s["component"] == "encoder") == 1


class TestCtcCollapse:
    """CTC collapsing must fold repeats before removing blanks."""

    def test_collapses_repeated_frames_into_one_token(self):
        assert ctc_runtime.collapse_ctc(
            [1, 1, 1, 2, 2], blank_id=0, collapse_repeats=True
        ) == [1, 2]

    def test_blank_separated_repeats_survive_as_two_tokens(self):
        assert ctc_runtime.collapse_ctc(
            [1, 1, 0, 1, 1], blank_id=0, collapse_repeats=True
        ) == [1, 1]

    def test_drops_blanks(self):
        assert ctc_runtime.collapse_ctc([0, 0, 3, 0], blank_id=0, collapse_repeats=True) == [3]

    def test_without_repeat_collapsing_every_non_blank_frame_is_kept(self):
        assert ctc_runtime.collapse_ctc([1, 1, 0, 2], blank_id=0, collapse_repeats=False) == [
            1,
            1,
            2,
        ]


def _decoding_metadata(**decoding_overrides) -> dict:
    decoding = {
        "kind": "ctc",
        "blank_id": 0,
        "collapse_repeats": True,
        "time_axis": 1,
        "class_axis": 2,
        "lengths": "frame_lengths",
        "vocabulary": {
            "source": "inline",
            "size": 5,
            "tokens": ["<pad>", "|", "A", "B", "<unk>"],
            "word_delimiter": "|",
            "ignored_tokens": ["<unk>"],
        },
    }
    decoding.update(decoding_overrides)
    return {
        "profiles": {
            "transcription": {
                "kind": "transcription",
                "outputs": {"logits": "logits", "frame_lengths": "frame_lengths"},
                "decoding": decoding,
            }
        }
    }


def _one_hot(rows: list[list[int]], classes: int = 5) -> np.ndarray:
    logits = np.full((len(rows), max(len(r) for r in rows), classes), -10.0, np.float32)
    for i, row in enumerate(rows):
        for t, class_id in enumerate(row):
            logits[i, t, class_id] = 10.0
    return logits


class TestMetadataDrivenDecoding:
    """Rendering is driven entirely by the declared vocabulary."""

    def test_word_delimiter_separates_words_without_adding_stray_spaces(self):
        # Leading, trailing and doubled delimiters must not create empty words.
        outputs = {
            "logits": _one_hot([[1, 2, 1, 1, 3, 1]]),
            "frame_lengths": np.array([6]),
        }
        decoded = ctc_runtime.decode_transcripts(_decoding_metadata(), outputs)
        assert decoded["transcripts"] == ["A B"]

    def test_ignored_tokens_are_dropped_before_word_splitting(self):
        outputs = {
            "logits": _one_hot([[2, 4, 3]]),
            "frame_lengths": np.array([3]),
        }
        decoded = ctc_runtime.decode_transcripts(_decoding_metadata(), outputs)
        assert decoded["transcripts"] == ["AB"]

    def test_lengths_binding_segments_a_padded_batch(self):
        outputs = {
            "logits": _one_hot([[2, 1, 3], [3, 0, 0]]),
            "frame_lengths": np.array([3, 1]),
        }
        decoded = ctc_runtime.decode_transcripts(_decoding_metadata(), outputs)
        assert decoded["argmax_ids"] == [[2, 1, 3], [3]]
        assert decoded["transcripts"] == ["A B", "B"]

    def test_absent_lengths_binding_decodes_every_frame(self):
        metadata = _decoding_metadata()
        del metadata["profiles"]["transcription"]["decoding"]["lengths"]
        outputs = {"logits": _one_hot([[2, 0, 0]])}
        decoded = ctc_runtime.decode_transcripts(metadata, outputs)
        assert decoded["argmax_ids"] == [[2, 0, 0]]

    def test_unbound_lengths_role_is_rejected(self):
        metadata = _decoding_metadata(lengths="missing_role")
        outputs = {"logits": _one_hot([[2]]), "frame_lengths": np.array([1])}
        with pytest.raises(ctc_runtime.MetadataContractError):
            ctc_runtime.decode_transcripts(metadata, outputs)

    def test_non_ctc_decoding_kind_is_rejected(self):
        metadata = _decoding_metadata(kind="beam_search")
        outputs = {"logits": _one_hot([[2]]), "frame_lengths": np.array([1])}
        with pytest.raises(ctc_runtime.MetadataContractError):
            ctc_runtime.decode_transcripts(metadata, outputs)


class TestAudioPreprocessingProgram:
    """Preprocessing must normalize per row and pad to the batch width."""

    @staticmethod
    def _program() -> dict:
        return {
            "transforms": [
                {"op": "decode"},
                {"op": "resample", "sample_rate": 16_000},
                {"op": "downmix", "channels": 1},
                {"op": "zero_mean_unit_variance", "epsilon": 1e-7},
                {"op": "pad", "mode": "right", "pad_value": 0.0},
            ],
            "outputs": [
                {
                    "name": "input_values",
                    "source": "values",
                    "content": "waveform",
                    "dtype": "float32",
                    "rank": 2,
                },
                {
                    "name": "attention_mask",
                    "source": "sample_mask",
                    "content": "validity_mask",
                    "dtype": "int64",
                    "rank": 2,
                },
            ],
        }

    def test_pads_to_the_longest_row_and_marks_validity(self):
        bound = ctc_runtime.run_audio_preprocessing(
            self._program(), [np.ones(10, np.float32), np.ones(4, np.float32)], 16_000
        )
        assert bound["input_values"].shape == (2, 10)
        assert bound["attention_mask"].tolist() == [[1] * 10, [1] * 4 + [0] * 6]

    def test_normalizes_each_row_independently(self):
        # A padded batch must not fold one row's statistics into another's.
        rows = [np.arange(10, dtype=np.float32), np.arange(4, dtype=np.float32) * 100]
        bound = ctc_runtime.run_audio_preprocessing(self._program(), rows, 16_000)
        solo = ctc_runtime.run_audio_preprocessing(self._program(), rows[:1], 16_000)
        np.testing.assert_allclose(
            bound["input_values"][0], solo["input_values"][0], atol=1e-6
        )

    def test_unknown_transform_is_rejected_rather_than_skipped(self):
        program = self._program()
        program["transforms"].append({"op": "spectrogram"})
        with pytest.raises(ctc_runtime.MetadataContractError):
            ctc_runtime.run_audio_preprocessing(program, [np.ones(4, np.float32)], 16_000)

    def test_output_reading_an_undeclared_value_is_rejected(self):
        program = self._program()
        program["outputs"][0]["source"] = "mel"
        with pytest.raises(ctc_runtime.MetadataContractError):
            ctc_runtime.run_audio_preprocessing(program, [np.ones(4, np.float32)], 16_000)
