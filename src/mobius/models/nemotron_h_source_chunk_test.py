# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Opt-in task/model contracts and owned successful-cache-phase regressions."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import onnx_ir as ir
import pytest
from onnx_ir.passes.common import CheckerPass, ShapeInferencePass

from mobius._builder import _cast_module_dtype, build_from_module
from mobius._configs import NemotronHOmniConfig, NemotronHPuzzleConfig
from mobius._configs.per_model._nemotron_h_omni_test import _omni_config
from mobius._configs.per_model._nemotron_h_puzzle_test import _source_config
from mobius._testing import create_test_builder, create_test_input
from mobius._testing.source_chunk import ExplicitCachePhaseCaller
from mobius.components import ChunkedPrefillMamba2Block, Mamba2Block
from mobius.integrations.ort_genai.auto_export import (
    _graph_input_names,
    _inspect_decoder_abi,
    _inspect_hybrid_multimodal_decoder_abi,
    _write_genai_config,
)
from mobius.models import (
    NemotronHOmniModel,
    NemotronHOmniSourceChunkModel,
    NemotronHPuzzleCausalLMModel,
    NemotronHPuzzleSourceChunkCausalLMModel,
)
from mobius.tasks import (
    NemotronHOmniTask,
    SourceChunkHybridCausalLMTask,
    SourceChunkNemotronHOmniTask,
)


def _config(omni=False, dtype=ir.DataType.BFLOAT16):
    source = _omni_config() if omni else _source_config()
    if omni:
        source.pop("quantization_config")
    cls = NemotronHOmniConfig if omni else NemotronHPuzzleConfig
    return dataclasses.replace(cls.from_transformers(source), dtype=dtype)


def _parameters(module):
    return [(name, list(value.shape)) for name, value in module.named_parameters()]


@pytest.mark.parametrize("omni", [False, True])
def test_direct_configuration_uses_fp32_cache_in_both_tasks(omni):
    converted = _config(omni)
    cls = type(converted)
    fields = {
        field.name: getattr(converted, field.name)
        for field in dataclasses.fields(converted)
        if field.init and field.name not in {"mamba_time_step_min", "mamba_ssm_dtype"}
    }
    config = cls(**fields)
    ordinary = NemotronHOmniModel(config) if omni else NemotronHPuzzleCausalLMModel(config)
    candidate = (
        NemotronHOmniSourceChunkModel(config)
        if omni
        else NemotronHPuzzleSourceChunkCausalLMModel(config)
    )
    key = "decoder" if omni else "model"
    for module, task in (
        (ordinary, NemotronHOmniTask() if omni else "hybrid-text-generation"),
        (
            candidate,
            SourceChunkNemotronHOmniTask() if omni else SourceChunkHybridCausalLMTask(),
        ),
    ):
        graph = build_from_module(module, config, task=task)[key].graph
        states = [
            value
            for value in (*graph.inputs, *graph.outputs)
            if value.name.endswith(".ssm_state")
        ]
        assert states
        assert all(value.dtype == ir.DataType.FLOAT for value in states)
        text = module.decoder.model if omni else module.model
        assert text.layers[0].mamba.time_step_min == pytest.approx(0.0, abs=0)


@pytest.mark.parametrize("omni", [False, True])
def test_phase_task_input_isolation_and_parameter_paths(omni, tmp_path):
    config = _config(omni)
    ordinary = NemotronHOmniModel(config) if omni else NemotronHPuzzleCausalLMModel(config)
    candidate = (
        NemotronHOmniSourceChunkModel(config)
        if omni
        else NemotronHPuzzleSourceChunkCausalLMModel(config)
    )
    assert _parameters(ordinary) == _parameters(candidate)
    original_text = ordinary.decoder.model if omni else ordinary.model
    candidate_text = candidate.decoder.model if omni else candidate.model
    for index in (1, 2, 3):
        assert type(candidate_text.layers[index]) is type(original_text.layers[index])
    assert type(candidate_text.layers[2].self_attn) is type(original_text.layers[2].self_attn)
    task = SourceChunkNemotronHOmniTask() if omni else SourceChunkHybridCausalLMTask()
    pkg = build_from_module(candidate, config, task=task)
    default = build_from_module(
        ordinary, config, task=NemotronHOmniTask() if omni else "hybrid-text-generation"
    )
    key = "decoder" if omni else "model"
    inputs = {v.name: v for v in pkg[key].graph.inputs}
    phase = inputs["has_previous_state"]
    assert phase.dtype == ir.DataType.BOOL and phase.shape == ir.Shape([])
    assert inputs["past_key_values.0.ssm_state"].dtype == ir.DataType.FLOAT
    assert inputs["past_key_values.0.ssm_state"].shape[-2:] == ir.Shape([4, 8])
    assert inputs["past_key_values.0.conv_state"].shape[-1] == 3
    outputs = {value.name: value for value in pkg[key].graph.outputs}
    assert (
        outputs["present.0.conv_state"].shape == inputs["past_key_values.0.conv_state"].shape
    )
    assert "has_previous_state" not in {v.name for v in default[key].graph.inputs}
    assert any(node.op_type == "If" for node in pkg[key].graph)
    assert not any(node.op_type == "If" for node in default[key].graph)
    abi = (
        _inspect_hybrid_multimodal_decoder_abi(pkg[key])
        if omni
        else _inspect_decoder_abi(pkg[key], model_type="decoder")
    )
    assert abi.inputs["has_previous_state"] == "has_previous_state"
    path = _write_genai_config(
        config,
        str(tmp_path),
        pkg=pkg,
        ort_model_type="nemotron_h_omni" if omni else "decoder",
        ep="cuda",
        context_length=128,
        bos_token_id=1,
        eos_token_id=[2, 11],
        pad_token_id=0,
        is_vlm=omni,
        has_speech=False,
    )
    metadata = json.loads(Path(path).read_text())
    assert metadata["model"]["decoder"]["inputs"]["has_previous_state"] == "has_previous_state"
    if omni:
        for name in ("embedding", "vision_encoder"):
            assert str(pkg[name]) == str(default[name])
        default_directory = tmp_path / "default"
        default_directory.mkdir()
        path = _write_genai_config(
            config,
            str(default_directory),
            pkg=default,
            ort_model_type="nemotron_h_omni",
            ep="cuda",
            context_length=128,
            bos_token_id=1,
            eos_token_id=[2, 11],
            pad_token_id=0,
            is_vlm=True,
            has_speech=False,
        )
        assert (
            "has_previous_state"
            not in json.loads(Path(path).read_text())["model"]["decoder"]["inputs"]
        )
    for model in pkg.values():
        CheckerPass()(model)
        ShapeInferencePass()(model)


def test_candidate_defers_dtype_until_build():
    config = _config(dtype=ir.DataType.FLOAT)
    module = NemotronHPuzzleSourceChunkCausalLMModel(config)
    _cast_module_dtype(module, ir.DataType.BFLOAT16)
    config.dtype = ir.DataType.BFLOAT16
    build_from_module(module, config, task=SourceChunkHybridCausalLMTask())
    with pytest.raises(TypeError, match="requires BF16"):
        build_from_module(
            NemotronHPuzzleSourceChunkCausalLMModel(_config(dtype=ir.DataType.FLOAT)),
            _config(dtype=ir.DataType.FLOAT),
            task=SourceChunkHybridCausalLMTask(),
        )


def test_task_rejects_a_configuration_that_would_silently_drop_phase():
    config = dataclasses.replace(
        _config(),
        num_hidden_layers=1,
        layer_types=["full_attention"],
        moe_block_configs=(None,),
    )
    with pytest.raises(ValueError, match="actual consumed"):
        build_from_module(
            NemotronHPuzzleSourceChunkCausalLMModel(config),
            config,
            task=SourceChunkHybridCausalLMTask(),
        )


def test_native_packed_projection_parameter_paths_are_preserved():
    source = _omni_config()
    source["llm_config"]["hidden_size"] = 256
    source["quantization_config"]["quantized_layers"] = {
        "language_model.backbone.layers.0.mixer.in_proj": {
            "quant_algo": "IQ2_XXS",
            "group_size": 256,
            "block_payload_bytes": 66,
            "packing": "ggml",
        },
        "language_model.backbone.layers.0.mixer.out_proj": {
            "quant_algo": "Q8_0",
            "group_size": 32,
            "block_payload_bytes": 34,
            "packing": "ggml",
        },
    }
    config = dataclasses.replace(
        NemotronHOmniConfig.from_transformers(source), dtype=ir.DataType.BFLOAT16
    )
    original, candidate = NemotronHOmniModel(config), NemotronHOmniSourceChunkModel(config)
    assert _parameters(original) == _parameters(candidate)
    paths = dict(candidate.named_parameters())
    assert paths["decoder.model.layers.0.mamba.in_proj.weight"].dtype == ir.DataType.UINT8
    assert paths["decoder.model.layers.0.mamba.out_proj.weight"].dtype == ir.DataType.UINT8
    package = build_from_module(candidate, config, task=SourceChunkNemotronHOmniTask())
    operators = {node.op_type for node in package["decoder"].graph}
    assert {"MatMulGGMLIQ2XXS", "MatMulGGMLQ8_0"} <= operators


def test_prefill_dt_clamp_and_threshold_do_not_enter_cached_branch():
    from mobius.components._chunked_mamba_test import _readout_graph

    values = {
        "A": np.asarray([-0.25], dtype=np.float32),
        "dt_bias": np.asarray([-2.5], dtype=np.float32),
        "D": np.asarray([0.5], dtype=ir.DataType.BFLOAT16.numpy()),
    }
    model = _readout_graph(values, clamp=(0.02, 0.2))
    node = next(node for node in model.graph if node.op_type == "If")
    cached = list(node.attributes["then_branch"].as_graph())
    prefill = list(node.attributes["else_branch"].as_graph())
    assert any(n.op_type == "Softplus" for n in cached)
    assert not any(n.op_type in ("Clip", "Log", "LessOrEqual") for n in cached)
    assert sum(n.op_type == "Clip" for n in prefill) == 1
    assert any(n.op_type == "LessOrEqual" for n in prefill)


@pytest.mark.parametrize(
    "dtype,shape",
    [
        (ir.DataType.FLOAT, []),
        (ir.DataType.BOOL, [1]),
        (ir.DataType.BOOL, None),
    ],
)
def test_metadata_rejects_noncanonical_phase_contract(dtype, shape):
    _, _, graph = create_test_builder()
    graph.inputs.append(
        ir.Value(
            name="has_previous_state",
            type=ir.TensorType(dtype),
            shape=ir.Shape(shape) if shape is not None else None,
        )
    )
    with pytest.raises(ValueError, match="BOOL scalar"):
        _graph_input_names(ir.Model(graph, ir_version=11))


def _block_inputs(op_builder, dtype=ir.DataType.BFLOAT16):
    return [
        create_test_input(op_builder, "hidden", [2, "time", 4], dtype),
        create_test_input(op_builder, "conv", [2, 6, 3], dtype),
        create_test_input(op_builder, "state", [2, 1, 1, 4], ir.DataType.FLOAT),
    ]


def test_public_positional_phase_append_and_default_rejection():
    builder, op, _ = create_test_builder()
    args = _block_inputs(builder)
    default = Mamba2Block(4, 4, 1, 4, 1)
    _cast_module_dtype(default, ir.DataType.BFLOAT16)
    default(op, *args)
    phase = create_test_input(builder, "has_previous_state", [], ir.DataType.BOOL)
    with pytest.raises(ValueError, match="phase-aware"):
        default(op, *args, None, phase)
    candidate = ChunkedPrefillMamba2Block(4, 4, 1, 4, 1)
    _cast_module_dtype(candidate, ir.DataType.BFLOAT16)
    with pytest.raises(ValueError, match="explicitly"):
        candidate(op, *args)
    candidate(op, *args, None, phase)


@pytest.mark.parametrize(
    "index,dtype,shape,message",
    [
        (0, ir.DataType.FLOAT, [2, 1, 4], "BF16 activations"),
        (1, ir.DataType.FLOAT, [2, 6, 3], "BF16 convolution"),
        (2, ir.DataType.BFLOAT16, [2, 1, 1, 4], "FP32 cached"),
        (2, ir.DataType.FLOAT, [2, 1, 4], "rank4"),
        (2, ir.DataType.FLOAT, [2, 1, 4, 1], "orientation"),
        (1, ir.DataType.BFLOAT16, [2, 7, 3], "window ABI"),
        (1, ir.DataType.BFLOAT16, [2, 6, 4], "window ABI"),
    ],
)
def test_candidate_rejects_wrong_native_dtype_shape_and_orientation(
    index, dtype, shape, message
):
    builder, op, _ = create_test_builder()
    args = _block_inputs(builder)
    args[index].dtype, args[index].shape = dtype, ir.Shape(shape)
    phase = create_test_input(builder, "has_previous_state", [], ir.DataType.BOOL)
    block = ChunkedPrefillMamba2Block(4, 4, 1, 4, 1)
    _cast_module_dtype(block, ir.DataType.BFLOAT16)
    with pytest.raises((TypeError, ValueError), match=message):
        block(op, *args, None, phase)


def test_phase_caller_advances_only_after_success_not_s1_or_offset():
    observed = []
    pointers = []
    fail = [True]

    def execute(feeds):
        observed.append(bool(feeds["has_previous_state"]))
        pointers.append(id(feeds["has_previous_state"]))
        if fail[0]:
            raise RuntimeError("deliberate decoder failure")
        return [np.ones((1,), dtype=np.float32)]

    caller = ExplicitCachePhaseCaller(execute)
    feeds = dict(
        input_ids=np.ones((2, 1), dtype=np.int64),
        position_ids=np.full((2, 1), 100, dtype=np.int64),
    )
    with pytest.raises(RuntimeError, match="deliberate"):
        caller(feeds)
    assert not caller.has_previous_state
    with pytest.raises(RuntimeError, match="poisoned"):
        caller(feeds)
    fail[0] = False
    caller = ExplicitCachePhaseCaller(execute)
    observed.clear()
    pointers.clear()
    caller(feeds)
    assert caller.has_previous_state
    feeds["position_ids"][:] = 0
    caller(feeds)
    assert observed == [False, True]
    assert len(set(pointers)) == 1
    fail[0] = True
    with pytest.raises(RuntimeError, match="deliberate"):
        caller(feeds)
    assert caller.has_previous_state
    with pytest.raises(RuntimeError, match="poisoned"):
        caller(feeds)
    guard = ExplicitCachePhaseCaller(lambda feeds: [])
    guard(feeds)
    with pytest.raises(ValueError, match="cached S>1"):
        guard({"input_ids": np.ones((2, 2), dtype=np.int64)})
    guard = ExplicitCachePhaseCaller(lambda feeds: [])
    guard(feeds)
    with pytest.raises(ValueError, match="contradicts"):
        guard({**feeds, "has_previous_state": np.asarray(False)})
    with pytest.raises(NotImplementedError, match="rewind"):
        caller.rewind()


@pytest.mark.parametrize(
    "options",
    [
        {"route": "Engine"},
        {"route": "scheduler"},
        {"prompt_chunk_size": 128},
        {"cuda_graph_capture": True},
    ],
)
def test_validation_caller_explicitly_rejects_unsafe_routes(options):
    with pytest.raises(ValueError, match="forbids"):
        ExplicitCachePhaseCaller(lambda feeds: [], **options)


@pytest.mark.parametrize(
    "mask",
    [
        np.ones((2, 1, 1), dtype=np.int64),
        np.ones((2, 1), dtype=np.float32),
        np.full((2, 1), 2, dtype=np.int64),
        np.ones((2, 0), dtype=np.int64),
    ],
)
def test_caller_rejects_noncontract_masks(mask):
    caller = ExplicitCachePhaseCaller(lambda feeds: [])
    with pytest.raises(ValueError, match="binary total mask"):
        caller({"input_ids": np.ones((2, 1), dtype=np.int64), "attention_mask": mask})
    assert not caller.has_previous_state


@pytest.mark.parametrize(
    "shape,dtype",
    [
        ([2, 1, 1], ir.DataType.INT64),
        ([2, 1], ir.DataType.FLOAT),
    ],
)
def test_model_rejects_silently_ignored_or_noncontract_mask_ranks(shape, dtype):
    builder, op, _ = create_test_builder()
    mask = create_test_input(builder, "mask", shape, dtype)
    phase = create_test_input(builder, "has_previous_state", [], ir.DataType.BOOL)
    positions = create_test_input(builder, "positions", [1], ir.DataType.INT64)
    hidden = create_test_input(builder, "hidden", [2, 1, 16], ir.DataType.BFLOAT16)
    with pytest.raises(TypeError, match="2D INT64/BOOL binary mask"):
        NemotronHPuzzleSourceChunkCausalLMModel(_config()).model(
            op, None, mask, positions, [], hidden, phase
        )
