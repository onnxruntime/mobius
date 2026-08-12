# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import json
import os

import jsonschema
import onnx_ir as ir
import pytest
import yaml

from mobius._model_package import ModelPackage
from mobius.integrations.onnx_genai.inference_metadata_test import (
    _model,
    _value,
)
from mobius.integrations.onnx_genai.workflow_metadata import (
    build_audio_codec_workflow_metadata,
    build_tts_workflow_metadata,
    write_audio_codec_workflow_metadata,
)


def _codec_package() -> ModelPackage:
    encoder = _model(
        "encoder",
        [_value("waveform", ir.DataType.FLOAT, ["batch", 1, "audio_samples"])],
        [("codes", ir.DataType.INT64, ["batch", 16, "frames"])],
    )
    decoder = _model(
        "decoder",
        [_value("codes", ir.DataType.INT64, ["batch", 16, "frames"])],
        [("waveform", ir.DataType.FLOAT, ["batch", 1, "audio_samples"])],
    )
    return ModelPackage({"encoder": encoder, "decoder": decoder})


def test_codec_workflow_has_typed_ssa_effects_and_audio_emit():
    metadata = build_audio_codec_workflow_metadata(_codec_package())
    pipeline = metadata["pipeline"]
    assert not {"models", "dataflow", "strategy", "phases"}.intersection(pipeline)

    workflow = pipeline["workflow"]
    assert workflow["inputs"]["request.waveform"]["contract"] == {
        "dtype": "float32",
        "rank": 3,
        "shape": ["batch", 1, "audio_samples"],
    }
    assert workflow["components"]["encoder"]["ports"]["outputs"]["codes"] == {
        "dtype": "int64",
        "rank": 3,
        "shape": ["batch", 16, "frames"],
    }
    assert workflow["components"]["encoder"]["effects"] == ["codec_encode"]
    assert workflow["components"]["decoder"]["effects"] == ["codec_decode"]

    encode, decode, emit = workflow["graph"]["nodes"]
    assert encode["outputs"] == {"codes": "codec.codes"}
    assert decode["inputs"] == {"codes": "codec.codes"}
    assert emit == {
        "kind": "emit",
        "value": "codec.waveform",
        "output": "waveform",
        "mode": "replace",
        "effect_name": "audio_emit",
        "effect": {"consumes": "audio_emit.0", "produces": "audio_emit.1"},
    }
    assert workflow["outputs"]["waveform"]["role"] == "audio"
    assert workflow["outputs"]["waveform"]["stage"] == "post_adapter"


def test_codec_workflow_rejects_incompatible_code_contracts():
    package = _codec_package()
    package["decoder"] = _model(
        "decoder",
        [_value("codes", ir.DataType.FLOAT, ["batch", 16, "frames"])],
        [("waveform", ir.DataType.FLOAT, ["batch", 1, "audio_samples"])],
    )
    with pytest.raises(ValueError, match="dtypes must match"):
        build_audio_codec_workflow_metadata(package)


def test_codec_workflow_roundtrips_yaml(tmp_path):
    path = write_audio_codec_workflow_metadata(_codec_package(), str(tmp_path))
    with open(path, encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    assert loaded == build_audio_codec_workflow_metadata(_codec_package())


def test_codec_workflow_matches_producer_schema():
    schema_path = os.environ.get("ONNX_GENAI_SCHEMA")
    if not schema_path:
        pytest.skip("set ONNX_GENAI_SCHEMA to the producer-contract schema")
    with open(schema_path, encoding="utf-8") as handle:
        schema = json.load(handle)
    jsonschema.validate(build_audio_codec_workflow_metadata(_codec_package()), schema)


def test_tts_reports_missing_nested_loop_induction_value():
    package = {
        "talker": object(),
        "code_predictor": object(),
        "talker_step_embedder": object(),
        "talker_prefill_embedder": object(),
    }
    with pytest.raises(NotImplementedError, match="code_predictor.step_index"):
        build_tts_workflow_metadata(package, object())
