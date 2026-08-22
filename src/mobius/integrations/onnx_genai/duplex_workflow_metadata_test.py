# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for the full-duplex speech-to-speech workflow producer.

The workflow describes one frame of a Moshi-family full-duplex model
(PersonaPlex, Moshi): packed audio in, packed audio out, with the whole
conversation carried in session-scoped state between invocations.
"""

from __future__ import annotations

import dataclasses
import json
import os

import jsonschema
import onnx_ir as ir
import pytest
import yaml

from mobius._model_package import ModelPackage
from mobius.integrations.onnx_genai.inference_metadata_test import _model, _value
from mobius.integrations.onnx_genai.workflow_metadata import (
    build_full_duplex_workflow_metadata,
    write_full_duplex_workflow_metadata,
)

_DELAYS = [0, 0, 1, 1, 0, 1, 1]
_CHANNELS = len(_DELAYS)
_STREAMS = 3
_LAYERS = 2
_CONTEXT = 250
_DEP_LAYERS = 2


@dataclasses.dataclass
class _DuplexConfig:
    delays: list[int] = dataclasses.field(default_factory=lambda: list(_DELAYS))
    dep_q: int = _STREAMS
    n_q: int = _STREAMS
    frame_size: int = 1920
    context: int = _CONTEXT
    text_initial_token_id: int = 32000
    initial_token_id: int = 2048


def _duplex_package() -> ModelPackage:
    encoder = _model(
        "encoder",
        [_value("waveform", ir.DataType.FLOAT, ["batch", 1, "audio_samples"])],
        [("codes", ir.DataType.INT64, ["batch", _STREAMS, "frames"])],
    )
    decoder = _model(
        "decoder",
        [_value("codes", ir.DataType.INT64, ["batch", _STREAMS, "frames"])],
        [("waveform", ir.DataType.FLOAT, ["batch", 1, "audio_samples"])],
    )
    temporal_inputs = [
        _value("input_frame", ir.DataType.INT64, ["batch", _CHANNELS, "sequence_len"]),
        _value("attention_mask", ir.DataType.INT64, ["batch", "context"]),
        _value("position_ids", ir.DataType.INT64, ["batch", "sequence_len"]),
    ]
    temporal_outputs: list[tuple[str, ir.DataType, list[int | str]]] = [
        ("hidden", ir.DataType.FLOAT, ["batch", "sequence_len", 8]),
        ("text_logits", ir.DataType.FLOAT, ["batch", "sequence_len", 32]),
    ]
    for layer in range(_LAYERS):
        for port in ("key", "value"):
            temporal_inputs.append(
                _value(
                    f"past_key_values.{layer}.{port}",
                    ir.DataType.FLOAT,
                    ["batch", 2, "past_sequence_len", 4],
                )
            )
            temporal_outputs.append(
                (
                    f"present.{layer}.{port}",
                    ir.DataType.FLOAT,
                    ["batch", 2, "past_sequence_len + 1", 4],
                )
            )
    temporal = _model("temporal", temporal_inputs, temporal_outputs)

    depformer_inputs = [
        _value("hidden", ir.DataType.FLOAT, ["batch", 1, 8]),
        _value("prev_token", ir.DataType.INT64, ["batch", 1]),
        _value("substep_index", ir.DataType.INT64, []),
    ]
    depformer_outputs: list[tuple[str, ir.DataType, list[int | str]]] = [
        ("logits", ir.DataType.FLOAT, ["batch", 1, 16]),
    ]
    for layer in range(_DEP_LAYERS):
        for port in ("key", "value"):
            depformer_inputs.append(
                _value(
                    f"past_key_values.{layer}.{port}",
                    ir.DataType.FLOAT,
                    ["batch", 2, "past_substep_len", 4],
                )
            )
            depformer_outputs.append(
                (
                    f"present.{layer}.{port}",
                    ir.DataType.FLOAT,
                    ["batch", 2, "past_substep_len + 1", 4],
                )
            )
    depformer = _model("depformer", depformer_inputs, depformer_outputs)
    return ModelPackage(
        {
            "encoder": encoder,
            "decoder": decoder,
            "temporal": temporal,
            "depformer": depformer,
        }
    )


def _workflow() -> dict:
    metadata = build_full_duplex_workflow_metadata(_duplex_package(), _DuplexConfig())
    return metadata["pipeline"]["workflow"]


def _frame_loop(workflow: dict) -> dict:
    """The outer loop that carries session state across duplex frames."""
    loops = [step for step in workflow["steps"] if step["kind"] == "loop"]
    assert len(loops) == 1, "the frame loop is the only top-level step"
    return loops[0]


def test_duplex_workflow_is_one_event_per_invocation() -> None:
    workflow = _workflow()
    assert workflow["inputs"]["request.audio_chunk"]["role"]["role"] == "media"
    assert workflow["inputs"]["request.session_id"]["role"]["role"] == "session_id"
    output = workflow["outputs"]["audio_chunk"]
    assert output["role"] == "audio"
    frame_loop = _frame_loop(workflow)
    # One invocation is one duplex event by default; the loop exists because a
    # session-scoped cell is only readable through a loop carry.
    assert frame_loop["max_iterations"] == "package.frames_per_invocation"
    assert workflow["inputs"]["package.frames_per_invocation"]["default"] == 1
    emit = next(step for step in frame_loop["steps"] if step["kind"] == "emit")
    # A frame is only emitted once the delay ring has been primed, so the emit
    # is guarded rather than unconditional.
    assert emit["mode"] == "event"
    assert emit["when"] == "duplex.emit"


def test_duplex_conversation_state_is_session_scoped() -> None:
    workflow = _workflow()
    state = workflow["state"]
    conversational = [
        "token_cache",
        "token_provided",
        "offset",
        "attention_mask",
        "position_ids",
        *[f"temporal_cache_{index}" for index in range(_LAYERS * 2)],
    ]
    for name in conversational:
        cell = state[name]
        assert cell["scope"] == "session", name
        assert cell["release_boundary"] == "session", name
        assert cell["management"] == "runtime", name
        assert cell["session"]["policy"] == "exclusive", name
    # The acoustic transformer restarts every frame, so its cache must not
    # outlive the invocation.
    for index in range(_DEP_LAYERS * 2):
        cell = state[f"depformer_cache_{index}"]
        assert cell["scope"] == "invocation"
        assert "session" not in cell


def test_duplex_codec_prefix_state_releases_at_phase_boundary() -> None:
    """Stateless codec graphs replay a prefix; it is released before the session."""
    state = _workflow()["state"]
    for name in ("user_waveform", "agent_codes"):
        cell = state[name]
        assert cell["scope"] == "session"
        assert cell["release_boundary"] == "invocation"
        assert cell["recurrence"]["kind"] == "growing"
        assert cell["recurrence"]["axis"] == 2


def test_duplex_temporal_cache_is_bounded_by_the_context_window() -> None:
    workflow = _workflow()
    state = workflow["state"]
    for index in range(_LAYERS * 2):
        cell = state[f"temporal_cache_{index}"]
        assert cell["recurrence"] == {
            "kind": "bounded",
            "axis": 2,
            "max": "package.context_limit",
        }
        # A runtime-owned cache must be permutable for batch compaction.
        assert cell["contract"]["batch_layout"] == {"kind": "request_aligned", "axis": 0}
    assert workflow["inputs"]["package.context_limit"]["default"] == _CONTEXT


def test_duplex_acoustic_loop_iterates_once_per_stream() -> None:
    workflow = _workflow()
    frame_loop = _frame_loop(workflow)
    loops = [step for step in frame_loop["steps"] if step["kind"] == "loop"]
    assert len(loops) == 1, "the acoustic substep loop is the only loop in a frame"
    loop = loops[0]
    assert loop["max_iterations"] == "package.num_streams"
    assert workflow["inputs"]["package.num_streams"]["default"] == _STREAMS
    components = [step["component"] for step in loop["steps"] if step["kind"] == "invoke"]
    assert components == [
        "stream_index",
        "token_to_slot",
        "depformer",
        "last_acoustic_logits",
        "token_sampler",
        "frame_update",
        "teacher_select",
    ]


def test_duplex_frame_pipeline_order() -> None:
    workflow = _workflow()
    invokes = [
        step["component"]
        for step in _frame_loop(workflow)["steps"]
        if step["kind"] == "invoke"
    ]
    assert invokes == [
        "waveform_append",
        "encoder",
        "codes_tail",
        "user_stream_merge",
        "frame_assemble",
        "temporal",
        "last_text_logits",
        "token_sampler",
        "teacher_select",
        "target_frame",
        "text_frame_update",
        "frame_commit",
        "agent_frame_select",
        "codes_append",
        "decoder",
        "chunk_tail",
        "step_update",
        "cache_length_update",
    ]


def test_duplex_delay_pattern_is_published() -> None:
    workflow = _workflow()
    assert workflow["inputs"]["package.delays"]["default"] == _DELAYS
    assert workflow["inputs"]["package.initial_tokens"]["default"] == [32000] + [2048] * (
        _CHANNELS - 1
    )


def test_duplex_declares_temporal_kv_service() -> None:
    workflow = _workflow()
    serving = workflow["serving"]
    assert serving["active"] == "active"
    assert serving["done"] == "done"
    assert serving["accepted_len"] == "accepted_len"
    group = serving["state_service"]["groups"]["temporal_cache"]
    assert group["sequence_axis"] == 2
    assert group["layout"] == "bnsh"
    assert group["logical_lengths"] == "temporal_cache_lengths"
    assert len(group["ports"]["temporal"]) == _LAYERS * 2
    for index in range(_LAYERS * 2):
        assert workflow["state"][f"temporal_cache_{index}"]["service_group"] == (
            "temporal_cache"
        )


def test_duplex_requires_all_components() -> None:
    package = _duplex_package()
    del package["depformer"]
    with pytest.raises(ValueError, match="missing"):
        build_full_duplex_workflow_metadata(package, _DuplexConfig())


def test_duplex_requires_delay_pattern() -> None:
    @dataclasses.dataclass
    class _NoDelays:
        dep_q: int = _STREAMS

    with pytest.raises(ValueError, match="delay pattern"):
        build_full_duplex_workflow_metadata(_duplex_package(), _NoDelays())


def test_duplex_workflow_writes_policy_artifacts(tmp_path) -> None:
    package = _duplex_package()
    path = write_full_duplex_workflow_metadata(package, _DuplexConfig(), str(tmp_path))
    with open(path, encoding="utf-8") as handle:
        metadata = yaml.safe_load(handle)
    components = metadata["pipeline"]["workflow"]["components"]
    for name in ("frame_assemble", "frame_commit", "teacher_select", "token_sampler"):
        artifact = components[name]["implementation"]["artifact"]
        assert os.path.isfile(os.path.join(str(tmp_path), artifact)), name


def test_duplex_workflow_matches_producer_schema() -> None:
    schema_path = os.environ.get("ONNX_GENAI_SCHEMA")
    if not schema_path or not os.path.isfile(schema_path):
        pytest.skip("set ONNX_GENAI_SCHEMA to the producer-contract schema")
    with open(schema_path, encoding="utf-8") as handle:
        schema = json.load(handle)
    metadata = build_full_duplex_workflow_metadata(_duplex_package(), _DuplexConfig())
    jsonschema.validate(metadata, schema)
