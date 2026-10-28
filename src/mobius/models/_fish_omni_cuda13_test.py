# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""CPU IR/contract evidence only; no native kernel or CUDA13 admission."""

from __future__ import annotations

import dataclasses
import hashlib
import json

import numpy as np
import onnx_ir as ir
import pytest
from onnx_shape_inference import ShapeInferenceContext, infer_symbolic_shapes
from onnxscript import optimizer

from mobius import build_from_module
from mobius._build_context import build_context
from mobius._execution_providers import ep_registry
from mobius._testing.ort_inference import OnnxModelSession
from mobius.models import FishOmniModel
from mobius.models._fish_omni_cuda13 import (
    _RIGHT_CAUSAL,
    PROFILE_METADATA,
    Cuda13GenerationQualificationTask,
    Cuda13QualificationModel,
    Cuda13QualificationTask,
    _Cuda13Experts,
    _infer_causal_right,
    _infer_unit_lower,
    causal_right_attention,
    native_candidate_manifest,
    saved_native_candidate_manifest,
    unit_lower_triangular_solve,
    validate_cuda13_context,
)
from mobius.models.fish_omni import _FishFP8Experts
from mobius.models.fish_omni_test import learned_parameter_shapes, tiny_config
from mobius.tasks import FishOmniGenerationTask, FishOmniTask
from mobius.tasks._base import _make_graph, _make_model


def candidate_config(dtype=ir.DataType.FLOAT16):
    config = tiny_config()
    return dataclasses.replace(
        config,
        dtype=dtype,
        hidden_size=128,
        moe_intermediate_size=128,
        num_local_experts=256,
        num_experts_per_tok=8,
        fp8_experts=True,
        native_fp8_experts=True,
        head_dim=256,
        tower=dataclasses.replace(config.tower, hidden_size=128, head_dim=256),
    )


@pytest.mark.parametrize("dtype", [ir.DataType.FLOAT16, ir.DataType.BFLOAT16])
@pytest.mark.parametrize("generation", [False, True])
def test_candidate_model_emits_selected_ops_and_preserves_abi(dtype, generation, tmp_path):
    config = candidate_config(dtype)
    module = Cuda13QualificationModel(config)
    portable = FishOmniModel(config)
    assert learned_parameter_shapes(module) == learned_parameter_shapes(portable)
    task = Cuda13GenerationQualificationTask() if generation else Cuda13QualificationTask()
    package = build_from_module(module, config, task=task, execution_provider="cuda")
    manifest = native_candidate_manifest(package)
    assert manifest["readiness"] is False
    assert manifest["cuda_version"] == "13.0"
    assert not manifest["capture"] and not manifest["native_ordered_reductions"]
    assert manifest["disabled_optimizers"] == []
    assert not manifest["models"]["embedding"]["required_features"]
    decoder_nodes = list(package["decoder"].graph.all_nodes())
    qmoes = [n for n in decoder_nodes if n.op_type == "QMoE"]
    assert len(qmoes) == 2
    for node in qmoes:
        assert node.domain == "com.microsoft"
        assert node.attributes["fp8_execution_profile"].as_string() == "fp16_compatible"
        for attr, expected in {
            "activation_quantization": "dynamic_fp8",
            "quant_type": "fp8",
            "activation_type": "silu",
        }.items():
            assert node.attributes[attr].as_string() == expected
        for attr, expected in {
            "expert_weight_bits": 8,
            "block_size": 128,
            "k": 8,
            "normalize_routing_weights": 1,
            "swiglu_fusion": 0,
        }.items():
            assert node.attributes[attr].as_int() == expected
        assert node.inputs[0].dtype == node.inputs[1].dtype == node.outputs[0].dtype == dtype
        for index in (2, 5, 8):
            assert node.inputs[index].dtype == ir.DataType.FLOAT8E4M3FN
            assert node.inputs[index + 1].dtype == ir.DataType.FLOAT
        assert node.inputs[4] is node.inputs[7] is None
    solves = [n for n in decoder_nodes if n.op_type == "UnitLowerTriangularSolve"]
    assert len(solves) == 2
    assert all(not n.attributes for n in solves)
    for name, model in package.items():
        nodes = list(model.graph.all_nodes())
        assert not any(n.op_type.startswith("Ordered") for n in nodes)
        assert not any(n.op_type == "Attention" for n in nodes)
        for node in (n for n in nodes if n.op_type == "CausalRightAttention"):
            assert set(node.attributes) == {"scale"}
            assert len(node.inputs) == 3
            assert len(node.inputs[0].shape) == len(node.outputs[0].shape) == 4
        metadata = json.loads(model.metadata_props[PROFILE_METADATA])
        assert metadata["required_features"] == manifest["models"][name]["required_features"]
        assert "SM89" not in model.metadata_props.get("mobius.required_runtime_features", "")
        assert not any("candidate" in x.name for x in model.graph.inputs)
    # A portable qualification control has the identical public input/output contract.
    from mobius.models.fish_omni_test import reduced_precision_prototype_task

    control = build_from_module(
        portable,
        config,
        task=reduced_precision_prototype_task(
            "fish-omni-generation" if generation else "fish-omni"
        ),
        execution_provider="cuda",
    )
    for name in package:
        assert not any(
            n.op_type == "UnitLowerTriangularSolve" for n in control[name].graph.all_nodes()
        )
        assert [(v.name, v.dtype, v.shape) for v in package[name].graph.inputs] == [
            (v.name, v.dtype, v.shape) for v in control[name].graph.inputs
        ]
        assert [v.name for v in package[name].graph.outputs] == [
            v.name for v in control[name].graph.outputs
        ]
        assert all(
            v.dtype is not None and v.shape is not None for v in package[name].graph.outputs
        )
    logits = next(v for v in package["decoder"].graph.outputs if v.name == "logits")
    assert logits.dtype == dtype
    assert logits.shape == ir.Shape(["batch", "sequence_len", config.vocab_size])
    paths = {}
    for name, model in package.items():
        paths[name] = tmp_path / f"{name}.onnx"
        ir.save(model, paths[name])
    saved = saved_native_candidate_manifest(paths)
    assert not saved["readiness"]
    for name, path in paths.items():
        entry = saved["models"][name]
        assert entry["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert entry["required_features"] == manifest["models"][name]["required_features"]


@pytest.mark.parametrize(
    "changes",
    [
        {"dtype": ir.DataType.FLOAT},
        {"native_fp8_experts": False},
        {"num_local_experts": 128},
        {"num_experts_per_tok": 4},
        {"hidden_size": 129},
        {"moe_intermediate_size": 129},
        {"norm_topk_prob": False},
        {"head_dim": 128},
        {"num_attention_heads": 33},
    ],
)
def test_candidate_rejects_unsupported_geometry(changes):
    with pytest.raises(ValueError, match="CUDA13 candidate"):
        Cuda13QualificationModel(dataclasses.replace(candidate_config(), **changes))


@pytest.mark.parametrize("task", [FishOmniTask(), FishOmniGenerationTask()])
def test_public_admission_stays_closed(task):
    config = candidate_config()
    with pytest.raises(ValueError, match="disabled pending"):
        build_from_module(
            Cuda13QualificationModel(config), config, task=task, execution_provider="cuda"
        )


def test_candidate_task_requires_explicit_module_and_cuda():
    config = candidate_config()
    with pytest.raises(ValueError, match="explicit qualification model"):
        build_from_module(
            FishOmniModel(config),
            config,
            task=Cuda13QualificationTask(),
            execution_provider="cuda",
        )
    with pytest.raises(ValueError, match="execution_provider='cuda'"):
        build_from_module(
            Cuda13QualificationModel(config),
            config,
            task=Cuda13QualificationTask(),
            execution_provider="onnx-standard",
        )


def test_default_qmoe_execution_profile_is_unchanged():
    config = candidate_config()
    graph, builder = _make_graph("default_qmoe")
    hidden = builder.input("x", config.dtype, [1, 2, 128])
    logits = builder.input("logits", config.dtype, [1, 2, 256])
    with build_context(ep_registry.require("cuda"), config.dtype):
        output = _FishFP8Experts(config)(builder.op, hidden, hidden, hidden, logits)
    builder.add_output(output, "y")
    node = next(n for n in graph if n.op_type == "QMoE")
    assert "fp8_execution_profile" not in node.attributes
    manifest = native_candidate_manifest({"decoder": _make_model(graph)})
    assert manifest["qmoe_compute"] == "native profile; compute unqualified"
    assert manifest["models"]["decoder"]["required_features"] == [
        "com.microsoft::QMoE[fp8_execution_profile=native]"
    ]


def attention_graph(dtype=ir.DataType.FLOAT16, batch=2, length=3, past=4, heads=4):
    graph, builder = _make_graph("attention_adapter")
    q = builder.input("q", dtype, [batch, length, heads * 256])
    k = builder.input("k", dtype, [batch, length, 2 * 256])
    v = builder.input("v", dtype, [batch, length, 2 * 256])
    pk = builder.input("pk", dtype, [batch, 2, past, 256])
    pv = builder.input("pv", dtype, [batch, 2, past, 256])
    return graph, builder, (q, k, v, _RIGHT_CAUSAL, pk, pv)


@pytest.mark.parametrize(
    "batch,length,past,heads",
    [
        (1, 1, 0, 4),
        (2, 3, 4, 4),
        (8, 1, 255, 4),
        (1, 30, 0, 16),
        (2, 3, 4, 16),
    ],
)
def test_attention_adapter_preserves_real_cache_and_head_repeat(batch, length, past, heads):
    graph, builder, args = attention_graph(batch=batch, length=length, past=past, heads=heads)
    output, pk, pv = causal_right_attention(
        builder.op,
        *args,
        num_attention_heads=heads,
        num_key_value_heads=2,
        scale=0.0625,
        dtype=ir.DataType.FLOAT16,
    )
    assert output.shape == args[0].shape
    node = next(n for n in graph if n.op_type == "CausalRightAttention")
    # Execute only the portable reshapes/repeats/cache paths, NOT a stand-in native op.
    for tensor, name in zip([*node.inputs, pk, pv], ["q4", "k4", "v4", "pk4", "pv4"]):
        builder.add_output(tensor, name)
    model = _make_model(graph)
    optimizer.remove_unused_nodes(model)
    assert all(n.op_type != "CausalRightAttention" for n in model.graph)
    session = OnnxModelSession(model, graph_optimization_level="disabled")
    rng = np.random.default_rng(42)
    feed = {
        x.name: rng.standard_normal(tuple(x.shape)).astype(np.float16)
        for x in model.graph.inputs
    }
    actual = session.run(feed)
    expected_k = np.concatenate(
        (feed["pk"], feed["k"].reshape(batch, length, 2, 256).transpose(0, 2, 1, 3)), 2
    )
    expected_v = np.concatenate(
        (feed["pv"], feed["v"].reshape(batch, length, 2, 256).transpose(0, 2, 1, 3)), 2
    )
    np.testing.assert_array_equal(actual["pk4"], expected_k)
    np.testing.assert_array_equal(actual["pv4"], expected_v)
    np.testing.assert_array_equal(actual["k4"], np.repeat(expected_k, heads // 2, axis=1))
    np.testing.assert_array_equal(actual["v4"], np.repeat(expected_v, heads // 2, axis=1))
    np.testing.assert_array_equal(
        actual["q4"], feed["q"].reshape(batch, length, heads, 256).transpose(0, 2, 1, 3)
    )


@pytest.mark.parametrize(
    "case",
    [
        "mask",
        "float",
        "width",
        "batch",
        "empty",
        "total",
        "negative_past",
        "cache_dtype",
        "cache_rank",
        "cache_heads",
        "scale_zero",
        "scale_nan",
    ],
)
def test_attention_adapter_rejects_unsupported_inputs(case):
    _graph, builder, args = attention_graph()
    args = list(args)
    scale, dtype = 0.0625, ir.DataType.FLOAT16
    if case == "mask":
        args[3] = builder.input("bias", ir.DataType.FLOAT, [2, 1, 3, 7])
    elif case == "float":
        dtype = ir.DataType.FLOAT
    elif case in {"width", "batch", "empty", "total"}:
        args[0].shape = ir.Shape(
            {
                "width": [2, 3, 512],
                "batch": [9, 3, 1024],
                "empty": [2, 0, 1024],
                "total": [2, 256, 1024],
            }[case]
        )
        if case != "width":
            for value in args[1:3]:
                value.shape = ir.Shape([args[0].shape[0], args[0].shape[1], 512])
            for value in args[4:]:
                value.shape = ir.Shape([args[0].shape[0], 2, 4, 256])
    elif case == "negative_past":
        for value in args[4:]:
            value.shape = ir.Shape([2, 2, -1, 256])
    elif case == "cache_dtype":
        args[4].type = ir.TensorType(ir.DataType.FLOAT)
    elif case == "cache_rank":
        args[4].shape = ir.Shape([2, 4, 512])
    elif case == "cache_heads":
        for value in args[4:]:
            value.shape = ir.Shape([2, 1, 4, 256])
    else:
        scale = 0.0 if case == "scale_zero" else float("nan")
    with pytest.raises((ValueError, TypeError)):
        causal_right_attention(
            builder.op,
            *args,
            num_attention_heads=4,
            num_key_value_heads=2,
            scale=scale,
            dtype=dtype,
        )


@pytest.mark.parametrize("bad_operand", ["hidden", "logits", "unknown"])
def test_compatible_qmoe_rejects_non_low_or_unproven_operands(bad_operand):
    config = candidate_config()
    _graph, builder = _make_graph("qmoe_guard")
    hidden = builder.input("x", config.dtype, [1, 2, 128])
    logits = builder.input("logits", config.dtype, [1, 2, 256])
    if bad_operand == "hidden":
        hidden.type = ir.TensorType(ir.DataType.FLOAT)
    elif bad_operand == "logits":
        logits.type = ir.TensorType(ir.DataType.FLOAT)
    else:
        hidden.type = None
    with (
        build_context(ep_registry.require("cuda"), config.dtype),
        pytest.raises(ValueError, match="requires FLOAT16 X and logits"),
    ):
        _Cuda13Experts(config)(builder.op, hidden, hidden, hidden, logits)


def test_saved_inventory_hashes_actual_external_file_and_rejects_escape(tmp_path):
    graph, builder = _make_graph("external")
    weight = ir.Value(
        name="weight",
        type=ir.TensorType(ir.DataType.FLOAT),
        shape=ir.Shape([128]),
        const_value=ir.tensor(np.arange(128, dtype=np.float32)),
    )
    graph.initializers[weight.name] = weight
    builder.add_output(builder.op.Identity(weight), "y")
    path = tmp_path / "model.onnx"
    ir.save(_make_model(graph), path, external_data="weights.bin", size_threshold_bytes=0)
    manifest = saved_native_candidate_manifest({"codec": path})
    entry = manifest["components"]["codec"]
    assert entry["dtype"] == "f32"
    assert entry["required_runtime_features"] == ""
    assert entry["external_data"] == [
        {
            "path": str(tmp_path / "weights.bin"),
            "sha256": hashlib.sha256((tmp_path / "weights.bin").read_bytes()).hexdigest(),
            "bytes": 512,
        }
    ]
    loaded = ir.load(path)
    value = loaded.graph.initializers["weight"]
    value.const_value = ir.ExternalTensor(
        "../outside.bin",
        name="weight",
        offset=0,
        length=512,
        dtype=ir.DataType.FLOAT,
        shape=ir.Shape([128]),
        base_dir=str(tmp_path),
    )
    # Keep the location in IR; do not copy/read any file outside the component.
    ir.save(loaded, path)
    with pytest.raises(ValueError, match="escapes candidate component"):
        saved_native_candidate_manifest({"codec": path})


def test_saved_greedy_policy_inventory_dtype_describes_output_not_logits_input(tmp_path):
    graph, builder = _make_graph("greedy")
    logits = builder.input("logits", ir.DataType.FLOAT16, [1, 128])
    builder.add_output(builder.op.ArgMax(logits, axis=-1, keepdims=0), "next_token_ids")
    path = tmp_path / "policy.onnx"
    ir.save(_make_model(graph), path)
    manifest = saved_native_candidate_manifest({"main_policy": path})
    assert manifest["components"]["main_policy"]["dtype"] == "int64"
    assert manifest["qmoe_compute"] == "not emitted"


@pytest.mark.parametrize(
    "a,b,dtype,valid",
    [
        ([2, 3, 64, 64], [2, 3, 64, 128], ir.DataType.FLOAT, True),
        ([0, 0], [0, 128], ir.DataType.FLOAT, True),
        ([64, 64], [64, 0], ir.DataType.FLOAT, True),
        ([64, 64], [64, 128], ir.DataType.FLOAT16, False),
        ([2, 64, 64], [1, 64, 128], ir.DataType.FLOAT, False),
        ([64, 63], [64, 128], ir.DataType.FLOAT, False),
        ([64, 64], [1, 64, 128], ir.DataType.FLOAT, False),
        ([64], [64], ir.DataType.FLOAT, False),
        ([None, 64, 64], [None, 64, 128], ir.DataType.FLOAT, False),
        ([None, None], [None, 128], ir.DataType.FLOAT, False),
    ],
)
def test_native_solve_boundary(a, b, dtype, valid):
    _graph, builder = _make_graph("solve")
    system = builder.input("a", dtype, a)
    rhs = builder.input("b", dtype, b)
    if not valid:
        with pytest.raises(ValueError):
            unit_lower_triangular_solve(builder.op, system, rhs)
        return
    output = unit_lower_triangular_solve(builder.op, system, rhs)
    assert output.shape == rhs.shape and output.dtype == rhs.dtype
    node = output.producer()
    assert node.op_type == "UnitLowerTriangularSolve" and not node.attributes


@pytest.mark.parametrize("dtype", [ir.DataType.FLOAT16, ir.DataType.BFLOAT16])
def test_native_boundary_symbolic_inference_is_registered(dtype, caplog):
    graph, builder, args = attention_graph(dtype=dtype, batch="batch", length="S", past="P")
    output, _, _ = causal_right_attention(
        builder.op,
        *args,
        num_attention_heads=4,
        num_key_value_heads=2,
        scale=0.0625,
        dtype=dtype,
    )
    attention = next(n for n in graph if n.op_type == "CausalRightAttention")
    builder_diagnostics = dict(attention.metadata_props)
    attention.outputs[0].shape = attention.outputs[0].type = None
    builder.add_output(output, "y")
    model = _make_model(graph)
    infer_symbolic_shapes(model, policy="strict")
    assert attention.outputs[0].dtype == dtype
    assert attention.outputs[0].shape == attention.inputs[0].shape
    assert not any("No shape inference registered" in r.message for r in caplog.records)
    assert dict(attention.metadata_props) == builder_diagnostics


def test_native_solve_symbolic_inference_preserves_exact_rhs_contract(caplog):
    graph, builder = _make_graph("native_solve_inference")
    system = builder.input("a", ir.DataType.FLOAT, ["B", 32, "C", 64, 64])
    rhs = builder.input("b", ir.DataType.FLOAT, ["B", 32, "C", 64, 128])
    output = unit_lower_triangular_solve(builder.op, system, rhs)
    builder_diagnostics = dict(output.producer().metadata_props)
    output.shape = output.type = None
    builder.add_output(output, "y")
    infer_symbolic_shapes(_make_model(graph), policy="strict")
    assert output.shape == rhs.shape and output.dtype == ir.DataType.FLOAT
    assert not any("No shape inference registered" in r.message for r in caplog.records)
    assert dict(output.producer().metadata_props) == builder_diagnostics


@pytest.mark.parametrize("case", ["rhs_broadcast", "rhs_half", "causal_head", "causal_scale"])
def test_registered_native_inference_rejects_invalid_contract(case):
    _graph, builder = _make_graph("invalid_native_contract")
    if case.startswith("rhs"):
        a = builder.input("a", ir.DataType.FLOAT, [2, 64, 64])
        b = builder.input("b", ir.DataType.FLOAT, [2, 64, 128])
        output = builder.op.UnitLowerTriangularSolve(a, b, _domain="com.microsoft")
        if case == "rhs_broadcast":
            b.shape = ir.Shape([1, 64, 128])
        else:
            b.type = ir.TensorType(ir.DataType.FLOAT16)
        function = _infer_unit_lower
    else:
        q = builder.input("q", ir.DataType.FLOAT16, [2, 4, 3, 256])
        k = builder.input("k", ir.DataType.FLOAT16, [2, 4, 7, 256])
        v = builder.input("v", ir.DataType.FLOAT16, [2, 4, 7, 256])
        if case == "causal_head":
            k.shape = v.shape = ir.Shape([2, 2, 7, 256])
        output = builder.op.CausalRightAttention(
            q,
            k,
            v,
            scale=0.0 if case == "causal_scale" else 0.0625,
            _domain="com.microsoft",
        )
        function = _infer_causal_right
    context = ShapeInferenceContext({"": 24, "com.microsoft": 1}, policy="strict")
    with pytest.raises(ValueError):
        function(context, output.producer())


@pytest.mark.parametrize(
    "case",
    [
        "valid",
        "cuda12",
        "cuda133",
        "sm",
        "capture",
        "padding",
        "mixed",
        "length",
        "batch",
        "dtype",
        "cache_dtype",
        "append",
        "bf16",
        "conv_dtype",
        "recurrent_dtype",
        "unknown_cache",
    ],
)
def test_runtime_contract_guard(case):
    batch, length, start = (9 if case == "batch" else 2), 3, 4
    if case == "length":
        start = 254
    positions = np.broadcast_to(
        np.arange(start, start + length, dtype=np.int64), (batch, length)
    ).copy()
    mask = np.ones((batch, start + length), np.int64)
    inputs = np.zeros((batch, length, 128), np.float16)
    caches = {
        f"past_key_values.1.{suffix}": np.zeros((batch, 2, start, 256), np.float16)
        for suffix in ("key", "value")
    }
    caches["past_key_values.0.conv_state"] = np.zeros((batch, 32, 3), np.float16)
    caches["past_key_values.0.recurrent_state"] = np.zeros((batch, 2, 128, 128), np.float32)
    options = dict(cuda_version="13.0", sm=90, capture=False)
    if case in {"cuda12", "cuda133"}:
        options["cuda_version"] = "12.8" if case == "cuda12" else "13.3"
    elif case == "sm":
        options["sm"] = 89
    elif case == "capture":
        options["capture"] = True
    elif case == "padding":
        mask[0, 0] = 0
    elif case == "mixed":
        positions[1] += 1
    elif case == "dtype":
        inputs = inputs.astype(np.float32)
    elif case == "cache_dtype":
        caches["past_key_values.1.key"] = caches["past_key_values.1.key"].astype(np.float32)
    elif case == "append":
        caches["past_key_values.1.key"] = np.zeros((batch, 2, start + 1, 256), np.float16)
    elif case == "bf16":
        inputs = inputs.astype(ir.DataType.BFLOAT16.numpy())
        caches = {
            name: value.astype(ir.DataType.BFLOAT16.numpy())
            if not name.endswith(".recurrent_state")
            else value
            for name, value in caches.items()
        }
    elif case == "conv_dtype":
        caches["past_key_values.0.conv_state"] = caches["past_key_values.0.conv_state"].astype(
            np.float32
        )
    elif case == "recurrent_dtype":
        caches["past_key_values.0.recurrent_state"] = caches[
            "past_key_values.0.recurrent_state"
        ].astype(np.float16)
    elif case == "unknown_cache":
        caches["unknown"] = np.zeros((batch, 1), np.float16)
    if case in {"valid", "bf16"}:
        validate_cuda13_context(positions, mask, inputs, caches, **options)
    else:
        with pytest.raises(ValueError):
            validate_cuda13_context(positions, mask, inputs, caches, **options)
