# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for encoder-conditioned (Whisper-style) speech-to-text workflow metadata."""

from __future__ import annotations

import json
from types import SimpleNamespace

import jsonschema
import onnx_ir as ir
import pytest
import yaml

from mobius._model_package import ModelPackage
from mobius.integrations.onnx_genai._test_support import (
    _model,
    _onnx_genai_schema_path,
    _value,
)
from mobius.integrations.onnx_genai.auto_export import _audio_preprocessing_program
from mobius.integrations.onnx_genai.workflow_metadata import (
    build_speech_to_text_workflow_metadata,
    write_speech_to_text_workflow_metadata,
)

_WHISPER_EXTRACTOR = {
    "feature_extractor_type": "WhisperFeatureExtractor",
    "sampling_rate": 16000,
    "feature_size": 80,
    "n_fft": 400,
    "hop_length": 160,
    "chunk_length": 30,
    "n_samples": 480000,
    "padding_value": 0.0,
}


def _config() -> SimpleNamespace:
    return SimpleNamespace(eos_token_id=50257, max_position_embeddings=448)


def _speech_package(*, layers: int = 2) -> ModelPackage:
    """A whisper-tiny-shaped encoder/decoder pair with self-attention cache."""
    encoder = _model(
        "encoder",
        [_value("input_features", ir.DataType.FLOAT, ["batch", 80, "audio_seq_len"])],
        [("encoder_hidden_states", ir.DataType.FLOAT, ["batch", 1500, 384])],
    )
    decoder_inputs = [
        _value("decoder_input_ids", ir.DataType.INT64, ["batch", "sequence_len"]),
        _value("encoder_hidden_states", ir.DataType.FLOAT, ["batch", 1500, 384]),
        _value("position_ids", ir.DataType.INT64, ["batch", "sequence_len"]),
    ]
    decoder_outputs = [("logits", ir.DataType.FLOAT, ["batch", "sequence_len", 51865])]
    for layer in range(layers):
        for kind in ("key", "value"):
            decoder_inputs.append(
                _value(
                    f"past_key_values.{layer}.{kind}",
                    ir.DataType.FLOAT,
                    ["batch", 6, "past_sequence_len", 64],
                )
            )
            decoder_outputs.append(
                (
                    f"present.{layer}.{kind}",
                    ir.DataType.FLOAT,
                    ["batch", 6, "total_sequence_len", 64],
                )
            )
    return ModelPackage(
        {
            "encoder": encoder,
            "decoder": _model("decoder", decoder_inputs, decoder_outputs),
        }
    )


def _workflow(metadata: dict) -> dict:
    return metadata["pipeline"]["workflow"]


def _loop(metadata: dict) -> dict:
    return next(step for step in _workflow(metadata)["steps"] if step["kind"] == "loop")


def test_encoder_runs_once_in_loop_setup():
    metadata = build_speech_to_text_workflow_metadata(_speech_package(), _config())
    loop = _loop(metadata)
    setup = [node["component"] for node in loop["setup"]]

    # The encoder conditions the prefill, so it precedes the decoder and never
    # appears in the loop body.
    assert setup[0] == "encoder"
    assert setup.index("decoder") > 0
    assert setup.count("encoder") == 1
    body = [node.get("component") for node in loop["steps"]]
    assert "encoder" not in body


def test_cross_state_is_request_aligned_and_invariant():
    metadata = build_speech_to_text_workflow_metadata(_speech_package(), _config())
    workflow = _workflow(metadata)
    cross = workflow["state"]["cross.encoder_hidden_states"]

    assert cross["initializer"] == "encoder.encoder_hidden_states"
    assert cross["recurrence"] == {"kind": "invariant"}
    # Request alignment on axis 0 is what keeps encoder states and decoder rows
    # in step under batching and compaction.
    assert cross["contract"]["batch_layout"] == {"kind": "request_aligned", "axis": 0}
    assert cross["contract"]["shape"] == ["batch", 1500, 384]

    carry = next(
        item
        for item in _loop(metadata)["carried"]
        if item["cell"] == "cross.encoder_hidden_states"
    )
    assert carry["next"] == "cross.encoder_hidden_states"

    # Prefill reads the encoder value directly (same scope); every loop
    # iteration reads the carried cell instead of rerunning the encoder.
    prefill = next(
        node for node in _loop(metadata)["setup"] if node.get("component") == "decoder"
    )
    assert prefill["inputs"]["encoder_hidden_states"] == "encoder.encoder_hidden_states"
    body_decoder = next(
        node for node in _loop(metadata)["steps"] if node.get("component") == "decoder"
    )
    assert body_decoder["inputs"]["encoder_hidden_states"] == "cross.encoder_hidden_states"


def test_self_attention_cache_is_the_only_served_group():
    metadata = build_speech_to_text_workflow_metadata(_speech_package(), _config())
    groups = _workflow(metadata)["serving"]["state_service"]["groups"]

    # The cross state is loop-invariant, so nothing appends to it and it is not
    # served; only the growing self-attention cache is.
    assert set(groups) == {"decoder_cache"}
    ports = groups["decoder_cache"]["ports"]["decoder"]
    assert ports["cache_0"] == {
        "input": "past_key_values.0.key",
        "output": "present.0.key",
        "role": "key",
        "layer": 0,
    }
    assert len(ports) == 4


def test_audio_program_is_declared_and_bound_to_the_encoder_input(tmp_path):
    processor = tmp_path / "audio_processor.json"
    processor.write_text(json.dumps(_WHISPER_EXTRACTOR), encoding="utf-8")
    pkg = _speech_package()
    program = _audio_preprocessing_program(str(processor), pkg["encoder"])
    metadata = build_speech_to_text_workflow_metadata(
        pkg, _config(), audio_preprocessing=program
    )

    # The program is document-level data, a sibling of `pipeline`.
    audio = metadata["preprocessing"]["audio"]
    assert [transform["op"] for transform in audio["transforms"]] == [
        "decode",
        "resample",
        "pad",
        "log_mel",
        "normalize",
    ]
    assert audio["transforms"][2]["target_samples"] == 480000
    assert audio["transforms"][3]["num_mel_bins"] == 80
    output = audio["outputs"][0]
    assert output["name"] == "audio.input_features"
    assert output["content"] == "audio_features"
    assert output["contract"]["shape"] == ["batch", 80, "audio_seq_len"]

    workflow = _workflow(metadata)
    assert workflow["manifest"]["adapter_abis"] == {"onnx-genai.audio-preprocess": "1"}
    assert "audio_preprocessing_program" in workflow["manifest"]["capabilities"]
    adapter = workflow["components"]["audio_preprocess"]
    assert adapter["implementation"] == {
        "kind": "adapter",
        "abi": "onnx-genai.audio-preprocess",
        "version": "1",
    }
    assert adapter["ports"]["inputs"]["encoded"]["dtype"] == "uint8"

    # Encoded bytes enter as a workflow input; the adapter invoke precedes the encoder.
    assert workflow["inputs"]["request.audio"]["contract"]["dtype"] == "uint8"
    setup = _loop(metadata)["setup"]
    assert setup[0] == {
        "kind": "invoke",
        "component": "audio_preprocess",
        "inputs": {"encoded": "request.audio"},
        "outputs": {"input_features": "audio.input_features"},
    }
    assert setup[1]["component"] == "encoder"
    assert setup[1]["inputs"]["input_features"] == "audio.input_features"


def test_without_a_program_features_are_a_request_input():
    metadata = build_speech_to_text_workflow_metadata(_speech_package(), _config())
    workflow = _workflow(metadata)

    assert "preprocessing" not in metadata
    assert "audio_preprocess" not in workflow["components"]
    assert "adapter_abis" not in workflow["manifest"]
    features = workflow["inputs"]["encoder.input.input_features"]
    assert features["contract"]["shape"] == ["batch", 80, "audio_seq_len"]
    assert features["contract"]["batch_layout"] == {"kind": "request_aligned", "axis": 0}


def test_non_log_mel_extractor_declares_no_program(tmp_path):
    processor = tmp_path / "audio_processor.json"
    processor.write_text(
        json.dumps({"feature_extractor_type": "Wav2Vec2FeatureExtractor"}), encoding="utf-8"
    )
    assert _audio_preprocessing_program(str(processor), _speech_package()["encoder"]) is None


def test_speech_workflow_requires_encoder_and_decoder():
    pkg = ModelPackage(
        {
            "decoder": _speech_package()["decoder"],
        }
    )
    with pytest.raises(ValueError, match="encoder and decoder"):
        build_speech_to_text_workflow_metadata(pkg, _config())


def test_write_round_trips_the_built_metadata(tmp_path):
    path = write_speech_to_text_workflow_metadata(_speech_package(), str(tmp_path), _config())
    with open(path, encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    assert loaded == build_speech_to_text_workflow_metadata(_speech_package(), _config())


def test_speech_workflow_matches_producer_schema(tmp_path):
    with open(_onnx_genai_schema_path(), encoding="utf-8") as handle:
        schema = json.load(handle)
    processor = tmp_path / "audio_processor.json"
    processor.write_text(json.dumps(_WHISPER_EXTRACTOR), encoding="utf-8")
    pkg = _speech_package()
    metadata = build_speech_to_text_workflow_metadata(
        pkg,
        _config(),
        audio_preprocessing=_audio_preprocessing_program(str(processor), pkg["encoder"]),
    )
    jsonschema.validate(metadata, schema)
