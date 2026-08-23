# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import dataclasses
import json

import jsonschema
import onnx_ir as ir
import pytest
import yaml

from mobius._model_package import ModelPackage
from mobius.integrations.onnx_genai.inference_metadata_test import (
    _model,
    _onnx_genai_schema_path,
    _value,
)
from mobius.integrations.onnx_genai.workflow_metadata import (
    build_audio_codec_workflow_metadata,
    build_tts_workflow_metadata,
    write_audio_codec_workflow_metadata,
)
from mobius.models.qwen3_tts import Qwen3TTSForConditionalGeneration
from mobius.models.qwen3_tts_test import _TINY_CONFIG
from mobius.tasks import TTSTask


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


def test_codec_workflow_has_typed_ssa_and_audio_emit():
    metadata = build_audio_codec_workflow_metadata(_codec_package())
    pipeline = metadata["pipeline"]
    assert not {"models", "dataflow", "strategy", "phases"}.intersection(pipeline)

    workflow = pipeline["workflow"]
    assert workflow["inputs"]["request.waveform"]["contract"] == {
        "dtype": "float32",
        "rank": 3,
        "shape": ["batch", 1, "audio_samples"],
        "batch_layout": {"kind": "request_aligned", "axis": 0},
    }
    # An ONNX component does not restate what its artifact already says: the
    # encoder's ports, dtypes and shapes live in `encoder.onnx`, and the
    # workflow's own inputs are where the request contract is declared.
    assert not (workflow["components"]["encoder"].get("ports") or {})
    assert not (workflow["components"]["decoder"].get("ports") or {})
    assert "effects" not in workflow["components"]["encoder"]
    assert "effects" not in workflow["components"]["decoder"]

    encode, decode, emit = workflow["steps"]
    assert encode["outputs"] == {"codes": "codec.codes"}
    assert decode["inputs"] == {"codes": "codec.codes"}
    assert emit == {
        "kind": "emit",
        "value": "codec.waveform",
        "output": "waveform",
        "mode": "replace",
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
    with open(_onnx_genai_schema_path(), encoding="utf-8") as handle:
        schema = json.load(handle)
    jsonschema.validate(build_audio_codec_workflow_metadata(_codec_package()), schema)


@dataclasses.dataclass
class _TtsSubConfig:
    num_code_groups: int = 4


@dataclasses.dataclass
class _TtsConfig:
    tts: _TtsSubConfig = dataclasses.field(default_factory=_TtsSubConfig)


def _tts_package() -> ModelPackage:
    talker = _model(
        "talker",
        [_value("inputs_embeds", ir.DataType.FLOAT, ["batch", "sequence", 16])],
        [("last_hidden_state", ir.DataType.FLOAT, ["batch", 16])],
    )
    predictor = _model(
        "code_predictor",
        [
            _value("last_hidden_state", ir.DataType.FLOAT, ["batch", 16]),
            _value("step_index", ir.DataType.INT64, ["batch"]),
        ],
        [("logits", ir.DataType.FLOAT, ["batch", 64])],
    )
    step = _model(
        "talker_step_embedder",
        [_value("frame_codes", ir.DataType.INT64, ["batch", 4])],
        [("inputs_embeds", ir.DataType.FLOAT, ["batch", 1, 16])],
    )
    prefill = _model(
        "talker_prefill_embedder",
        [_value("text_ids", ir.DataType.INT64, ["batch", "sequence"])],
        [("prefill_embeds", ir.DataType.FLOAT, ["batch", "sequence", 16])],
    )
    codec = _model(
        "codec",
        [_value("codes", ir.DataType.INT64, ["batch", 4, "frames"])],
        [("waveform", ir.DataType.FLOAT, ["batch", 1, "samples"])],
    )
    return ModelPackage(
        {
            "talker": talker,
            "code_predictor": predictor,
            "talker_step_embedder": step,
            "talker_prefill_embedder": prefill,
            "codec": codec,
        }
    )


def test_tts_uses_nested_lexical_loop_induction_and_codec():
    workflow = build_tts_workflow_metadata(_tts_package(), _TtsConfig())["pipeline"][
        "workflow"
    ]
    outer = workflow["steps"][0]
    inner = outer["steps"][2]
    assert outer["iteration"]["value"] == "talker.iteration"
    assert inner["iteration"]["value"] == "code.iteration"
    assert inner["steps"][0]["inputs"]["step_index"] == "code.iteration"
    assert workflow["steps"][-2]["component"] == "codec"
    assert workflow["outputs"]["waveform"]["stage"] == "post_adapter"


def test_real_qwen3_tts_workflow_carries_trained_transitions_and_kv_state():
    package = TTSTask().build(Qwen3TTSForConditionalGeneration(_TINY_CONFIG), _TINY_CONFIG)
    package["codec"] = _model(
        "codec",
        [_value("codes", ir.DataType.INT64, ["batch", 4, "frames"])],
        [("waveform", ir.DataType.FLOAT, ["batch", 1, "samples"])],
    )

    workflow = build_tts_workflow_metadata(package, _TINY_CONFIG)["pipeline"]["workflow"]

    assert {
        "code_predictor_prefill",
        "code_predictor_step_embedder",
        "talker_text_step",
    }.issubset(workflow["components"])
    assert workflow["state"]["talker_cache_0"]["recurrence"]["kind"] == "bounded"
    assert workflow["state"]["talker_cache_0"]["service_group"] == "talker_cache"
    assert workflow["state"]["predictor_cache_0"]["service_group"] == "predictor_cache"
    assert workflow["serving"]["state_service"]["groups"]["talker_cache"]["kind"] == (
        "full_attention"
    )
    # Row identity is runtime-private: no published input carries a slot table.
    assert not any("slot_ids" in name for name in workflow["inputs"])
    assert workflow["serving"]["state_service"]["groups"]["talker_cache"]["ports"]["talker"][
        "talker_cache_0"
    ]["input"].startswith("past_key_values.")
    assert workflow["state"]["predictor_cache_0"]["scope"] == "invocation"
    assert (
        workflow["state"]["predictor_cache_0"]["recurrence"]["max"]
        == "package.predictor_context_limit"
    )
    outer = workflow["steps"][0]
    setup_history = next(
        node for node in outer["setup"] if node.get("component") == "code_history_append"
    )
    assert setup_history["inputs"]["frame"].startswith("setup.predictor.remaining_")

    assert outer["kind"] == "loop"
    inner = next(node for node in outer["steps"] if node["kind"] == "loop")
    assert inner["iteration"]["value"] == "code.iteration"
    assert all(carried["cell"] != "slot_ids" for carried in inner["carried"])
    assert any(
        node.get("component") == "code_predictor_step_embedder" for node in inner["steps"]
    )
