# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Projection packing payload, cache, rejection, and adapter compatibility tests."""

from __future__ import annotations

import dataclasses
import json
from types import SimpleNamespace

import ml_dtypes
import numpy as np
import onnx_ir as ir
import pytest
import torch
from safetensors.numpy import save_file

from mobius import ModelPackage
from mobius._component_quantization import validate_quantized_component_bindings
from mobius._constants import QKV_PROJECTION_SOURCES_METADATA_KEY
from mobius._execution_providers import EpCapabilities, ep_registry, get_ep
from mobius._optimizations import (
    _get_optimization_passes,
    fold_initializers_after_weights,
    optimize_model,
)
from mobius._pipeline_contract import declare_arbitrary_attention_mask
from mobius._testing.ort_inference import OnnxModelSession
from mobius.adapter_io import _resolve_initializer_name, attach_peft_adapter
from mobius.adapters import AdapterSource
from mobius.functions import register_function_bodies
from mobius.functions.matmul_nbits import matmul_nbits
from mobius.rewrite_rules import pack_matmul_nbits_qkv_pass, pack_qkv_projections_pass

_DTYPES = {
    ir.DataType.FLOAT: np.float32,
    ir.DataType.FLOAT16: np.float16,
    ir.DataType.BFLOAT16: ml_dtypes.bfloat16,
}


def _model(
    *,
    quantized=False,
    shared=False,
    dtype=ir.DataType.FLOAT,
    deferred=False,
    transposed=True,
    zero_points=True,
    flat=False,
    hidden_size=64,
    block_size=32,
    q_width=32,
):
    rng = np.random.default_rng(42)
    nodes, initializers, weights, projections = [], [], {}, []

    def parameter(name, data):
        tensor = ir.tensor(data)
        value = ir.Value(
            name=name,
            shape=tensor.shape,
            type=ir.TensorType(tensor.dtype),
            const_value=None if deferred else tensor,
        )
        initializers.append(value)
        weights[name] = data
        return value

    def op(kind, inputs, name, shape, attrs=None):
        output = ir.Value(name=name, shape=ir.Shape(shape), type=ir.TensorType(dtype))
        nodes.append(
            ir.Node(
                "",
                kind,
                inputs,
                ir.convenience.convert_attributes(attrs or {}),
                outputs=[output],
                name=name,
            )
        )
        return output

    x = ir.Value(name="x", shape=ir.Shape([2, "S", hidden_size]), type=ir.TensorType(dtype))
    blocks = (hidden_size + block_size - 1) // block_size
    for index, (name, width) in enumerate(
        (("q", q_width), ("k", 16), *(() if shared else (("v", 16),)))
    ):
        output = ir.Value(
            name=name,
            shape=ir.Shape([2, "S", width]),
            type=ir.TensorType(dtype),
            doc_string=f"raw {name} projection",
        )
        output.metadata_props["test.projection"] = name
        if quantized:
            shape = (width, blocks, block_size // 2)
            inputs = [
                x,
                parameter(f"{name}.weight", rng.integers(0, 256, shape, dtype=np.uint8)),
                parameter(
                    f"{name}.scales",
                    np.full(
                        (width * blocks,) if flat else (width, blocks),
                        0.01,
                        dtype=_DTYPES[dtype],
                    ),
                ),
            ]
            if zero_points:
                inputs.append(
                    parameter(
                        f"{name}.zero_points",
                        np.full(
                            (width * ((blocks + 1) // 2),)
                            if flat
                            else (width, (blocks + 1) // 2),
                            0x88,
                            dtype=np.uint8,
                        ),
                    )
                )
            node = ir.Node(
                "com.microsoft",
                "MatMulNBits",
                inputs,
                ir.convenience.convert_attributes(
                    {"K": hidden_size, "N": width, "bits": 4, "block_size": block_size}
                ),
                outputs=[output],
            )
        else:
            transpose = transposed is True or (transposed == "mixed" and index == 0)
            shape = (width, hidden_size) if transpose else (hidden_size, width)
            weight = parameter(
                f"{name}.weight", (rng.standard_normal(shape) * 0.1).astype(_DTYPES[dtype])
            )
            if transpose:
                weight = op(
                    "Transpose", [weight], name + "_t", [hidden_size, width], {"perm": [1, 0]}
                )
            node = ir.Node("", "MatMul", [x, weight], outputs=[output])
        nodes.append(node)
        projections.append(node)
    q, k = projections[0].outputs[0], projections[1].outputs[0]
    v = k if shared else projections[2].outputs[0]
    # Normalization branches must remain separate; raw K also feeds V when shared.
    q = op(
        "RMSNormalization",
        [q, parameter("q.norm", np.ones(q_width, dtype=_DTYPES[dtype]))],
        "q_norm",
        [2, "S", q_width],
        {"axis": -1},
    )
    k = op(
        "RMSNormalization",
        [k, parameter("k.norm", np.ones(16, dtype=_DTYPES[dtype]))],
        "k_norm",
        [2, "S", 16],
        {"axis": -1},
    )
    mask = ir.Value(name="mask", shape=ir.Shape([2, 1, "S", "T"]), type=ir.TensorType(dtype))
    past = [
        ir.Value(name=name, shape=ir.Shape([2, 2, "P", 8]), type=ir.TensorType(dtype))
        for name in ("past_k", "past_v")
    ]
    outputs = [
        ir.Value(name="y", shape=ir.Shape([2, "S", q_width]), type=ir.TensorType(dtype)),
        *[
            ir.Value(name=name, shape=ir.Shape([2, 2, "T", 8]), type=ir.TensorType(dtype))
            for name in ("present_k", "present_v")
        ],
    ]
    attention = ir.Node(
        "",
        "Attention",
        [q, k, v, mask, *past],
        ir.convenience.convert_attributes(
            {"q_num_heads": q_width // 8, "kv_num_heads": 2, "is_causal": 1}
        ),
        outputs=outputs,
    )
    nodes.append(attention)
    graph = ir.Graph(
        [x, mask, *past],
        [*outputs, *[node.outputs[0] for node in projections]],
        nodes=nodes,
        initializers=initializers,
        opset_imports={"": 24, "com.microsoft": 1},
    )
    return ir.Model(graph, ir_version=10), weights, projections, attention


def _pass(quantized):
    return pack_matmul_nbits_qkv_pass() if quantized else pack_qkv_projections_pass()


@pytest.mark.parametrize("quantized", [False, True])
@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("dtype", list(_DTYPES))
def test_preserves_branches_metadata_and_is_idempotent(quantized, shared, dtype):
    model, _, projections, attention = _model(quantized=quantized, shared=shared, dtype=dtype)
    original_inputs = list(attention.inputs)
    assert _pass(quantized)(model).modified
    assert list(attention.inputs[:2]) == original_inputs[:2]  # Norm branches unchanged.
    assert list(attention.inputs[3:]) == original_inputs[3:]  # Mask/cache unchanged.
    split = next(node for node in model.graph if node.op_type == "Split")
    assert len(split.outputs) == (2 if shared else 3)
    for output in split.outputs:
        assert output.dtype == dtype
        assert output.metadata_props["test.projection"] == output.name
        assert output.doc_string == f"raw {output.name} projection"
    assert all(node not in model.graph for node in projections)
    assert not _pass(quantized)(model).modified


@pytest.mark.parametrize("quantized", [False, True])
@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("transposed", [False, True, "mixed"])
def test_cpu_prefill_and_reused_cache_parity(quantized, shared, transposed):
    original, _, _, _ = _model(quantized=quantized, shared=shared, transposed=transposed)
    packed, _, _, _ = _model(quantized=quantized, shared=shared, transposed=transposed)
    assert _pass(quantized)(packed).modified
    for model in (original, packed):
        fold_initializers_after_weights(model)
    sessions = [OnnxModelSession(model, device="cpu") for model in (original, packed)]
    rng = np.random.default_rng(7)
    caches = [[np.zeros((2, 2, 0, 8), dtype=np.float32) for _ in range(2)] for _ in range(2)]
    for length in (3, 1, 1):
        total = caches[0][0].shape[2] + length
        x = rng.standard_normal((2, length, 64)).astype(np.float32)
        mask = np.zeros((2, 1, length, total), dtype=np.float32)
        mask[1, :, :, -1] = -10000
        results = [
            session.run({"x": x, "mask": mask, "past_k": cache[0], "past_v": cache[1]})
            for session, cache in zip(sessions, caches, strict=True)
        ]
        for key in results[0]:
            np.testing.assert_allclose(results[0][key], results[1][key], atol=1e-5, rtol=1e-5)
        caches = [[result["present_k"], result["present_v"]] for result in results]


@pytest.mark.parametrize("flat", [False, True])
@pytest.mark.parametrize("zero_points", [False, True])
@pytest.mark.parametrize("dtype", list(_DTYPES))
@pytest.mark.parametrize("hidden_size", [64, 65])
def test_int4_payload_is_exact_without_requantization(flat, zero_points, dtype, hidden_size):
    model, weights, _, _ = _model(
        quantized=True,
        flat=flat,
        zero_points=zero_points,
        dtype=dtype,
        hidden_size=hidden_size,
    )
    assert pack_matmul_nbits_qkv_pass()(model).modified
    fold_initializers_after_weights(model)
    projection = next(node for node in model.graph if node.op_type == "MatMulNBits")
    for index, suffix in enumerate(
        ("weight", "scales", *(("zero_points",) if zero_points else ())), 1
    ):
        expected = np.concatenate([weights[f"{name}.{suffix}"] for name in "qkv"], axis=0)
        np.testing.assert_array_equal(projection.inputs[index].const_value.numpy(), expected)
    assert projection.attributes["N"].as_int() == 64


@pytest.mark.parametrize("zero_points", [False, True])
def test_registry_owned_fallback_remains_packable(zero_points):
    model, weights, projections, _ = _model(quantized=True, zero_points=zero_points)
    register_function_bodies(model)
    identifier = projections[0].op_identifier()
    assert identifier[2] == ("" if zero_points else "default_zero_points")
    function = model.functions[identifier]
    assert pack_matmul_nbits_qkv_pass()(model).modified
    assert model.functions[identifier] is function
    fold_initializers_after_weights(model)
    packed = next(node for node in model.graph if node.op_type == "MatMulNBits")
    assert packed.op_identifier() == identifier
    np.testing.assert_array_equal(
        packed.inputs[1].const_value.numpy(),
        np.concatenate([weights[f"{name}.weight"] for name in "qkv"], axis=0),
    )


@pytest.mark.parametrize("zero_points", [False, True])
def test_caller_factory_function_and_calls_are_not_packed_or_specialized(zero_points):
    model, _, projections, _ = _model(quantized=True, zero_points=zero_points)
    function = matmul_nbits()
    model.functions[function.identifier()] = function
    register_function_bodies(model)
    assert model.functions[function.identifier()] is function
    assert all(node.overload == "" for node in projections)
    assert not pack_matmul_nbits_qkv_pass()(model).modified
    assert model.functions[function.identifier()] is function
    assert all(node.overload == "" for node in projections)


@pytest.mark.parametrize("transposed", [False, True, "mixed"])
@pytest.mark.parametrize("quantized", [False, True])
def test_deferred_binding_and_package_roundtrip(tmp_path, transposed, quantized):
    model, weights, _, _ = _model(quantized=quantized, transposed=transposed, deferred=True)
    names = set(model.graph.initializers)
    assert _pass(quantized)(model).modified
    assert names <= set(model.graph.initializers)
    package = ModelPackage({"model": model})
    package.apply_weights({name: torch.from_numpy(value) for name, value in weights.items()})
    package.save(tmp_path / "package")
    restored = ModelPackage.load(tmp_path / "package")
    assert not any(node.op_type == "Concat" for node in restored["model"].graph)
    assert any(node.op_type == "Split" for node in restored["model"].graph)


@pytest.mark.parametrize("dtype", list(_DTYPES))
@pytest.mark.parametrize("transposed", [False, True, "mixed"])
@pytest.mark.parametrize("shared", [False, True])
def test_float_payload_is_exact_for_all_orientations(dtype, transposed, shared):
    model, weights, _, _ = _model(dtype=dtype, transposed=transposed, shared=shared)
    assert pack_qkv_projections_pass()(model).modified
    fold_initializers_after_weights(model)
    packed = next(node for node in model.graph if node.op_type == "MatMul").inputs[1]
    logical = []
    for index, name in enumerate(("q", "k") if shared else ("q", "k", "v")):
        weight = weights[f"{name}.weight"]
        if transposed is True or (transposed == "mixed" and index == 0):
            weight = weight.T
        logical.append(weight)
    assert packed.dtype == dtype
    np.testing.assert_array_equal(packed.const_value.numpy(), np.concatenate(logical, axis=1))


@pytest.mark.parametrize(
    "case",
    [
        "bits",
        "accuracy",
        "block",
        "unknown",
        "scale_dtype",
        "zp",
        "g_idx",
        "overridable",
        "different_input",
        "overlap",
        "custom_body",
    ],
)
def test_incompatible_int4_is_left_unchanged(case):
    model, _, projections, attention = _model(quantized=True)
    node = projections[0]
    if case in {"bits", "accuracy", "block", "unknown"}:
        key, value = {
            "bits": ("bits", 8),
            "accuracy": ("accuracy_level", 4),
            "block": ("block_size", 64),
            "unknown": ("weight_prepacked", 1),
        }[case]
        node.attributes[key] = ir.AttrInt64(key, value)
    elif case == "scale_dtype":
        node.inputs[2].type = ir.TensorType(ir.DataType.FLOAT16)
    elif case == "zp":
        node.replace_input_with(3, None)
    elif case == "g_idx":
        node.resize_inputs(5)
        node.replace_input_with(4, model.graph.inputs[0])
    elif case == "overridable":
        model.graph.inputs.append(node.inputs[1])
    elif case == "different_input":
        node.replace_input_with(0, model.graph.inputs[1])
    elif case == "custom_body":
        function = matmul_nbits()
        function.metadata_props.clear()
        model.functions[function.identifier()] = function
    else:
        model.graph.append(
            ir.Node(
                "",
                "Attention",
                [
                    attention.inputs[0],
                    attention.inputs[1],
                    attention.inputs[1],
                    *attention.inputs[3:],
                ],
                dict(attention.attributes),
                outputs=[ir.Value(name="other")],
            )
        )
    assert not pack_matmul_nbits_qkv_pass()(model).modified


@pytest.mark.parametrize(
    "case",
    [
        "attribute_type",
        "zero_k",
        "zero_n",
        "block_small",
        "block_non_power_two",
        "accuracy_presence",
        "unknown_overload",
        "mixed_overload",
        "default_overload_with_zp",
        "rank_two",
        "unknown_hidden",
        "output_shape",
        "output_dtype",
        "weight_shape",
        "weight_dtype",
        "tensor_shape",
        "tensor_dtype",
        "computed_weight",
        "foreign_weight",
        "fused_bias",
        "contrib_opset",
        "outside_consumer",
        "no_anchor",
    ],
)
def test_int4_correctness_guards_are_complete_noops(case):
    model, _, projections, attention = _model(quantized=True)
    q = projections[0]
    if case == "attribute_type":
        q.attributes["K"] = ir.AttrFloat32("K", 64.0)
    elif case in {"zero_k", "zero_n", "block_small", "block_non_power_two"}:
        key, value = {
            "zero_k": ("K", 0),
            "zero_n": ("N", 0),
            "block_small": ("block_size", 8),
            "block_non_power_two": ("block_size", 24),
        }[case]
        q.attributes[key] = ir.AttrInt64(key, value)
    elif case == "accuracy_presence":
        q.attributes["accuracy_level"] = ir.AttrInt64("accuracy_level", 0)
    elif case in {"unknown_overload", "mixed_overload", "default_overload_with_zp"}:
        q.overload = "unknown" if case == "unknown_overload" else "default_zero_points"
        if case == "default_overload_with_zp":
            for node in projections:
                node.overload = "default_zero_points"
    elif case in {"rank_two", "unknown_hidden"}:
        q.inputs[0].shape = ir.Shape([2, 64] if case == "rank_two" else [2, "S", "K"])
    elif case == "output_shape":
        q.outputs[0].shape = ir.Shape([2, "S", 31])
    elif case == "output_dtype":
        q.outputs[0].dtype = ir.DataType.FLOAT16
    elif case == "weight_shape":
        q.inputs[1].shape = ir.Shape([32, 32])
    elif case == "weight_dtype":
        q.inputs[1].dtype = ir.DataType.INT8
    elif case == "tensor_shape":
        q.inputs[1].const_value = ir.tensor(np.zeros((32, 2, 15), dtype=np.uint8))
    elif case == "tensor_dtype":
        q.inputs[2].const_value = ir.tensor(np.ones((32, 2), dtype=np.float16))
    elif case == "computed_weight":
        weight = q.inputs[1]
        output = ir.Value(name="computed", shape=weight.shape, type=weight.type)
        model.graph.insert_before(q, ir.Node("", "Identity", [weight], outputs=[output]))
        q.replace_input_with(1, output)
    elif case == "foreign_weight":
        del model.graph.initializers[q.inputs[1].name]
    elif case == "fused_bias":
        q.resize_inputs(6)
        q.replace_input_with(5, q.inputs[2])
    elif case == "contrib_opset":
        model.graph.opset_imports["com.microsoft"] = 2
    elif case == "outside_consumer":
        model.graph.append(ir.Node("", "Identity", [q.inputs[1]], num_outputs=1))
    elif case == "no_anchor":
        # Detach the public Attention outputs before safely removing their
        # producer. Keep raw projection outputs to exercise the no-anchor guard.
        for output in attention.outputs:
            model.graph.outputs.remove(output)
        model.graph.remove(attention, safe=True)
    before = tuple(model.graph)
    inputs = {node: tuple(node.inputs) for node in before}
    initializers = dict(model.graph.initializers)
    assert not pack_matmul_nbits_qkv_pass()(model).modified
    assert tuple(model.graph) == before
    assert {node: tuple(node.inputs) for node in before} == inputs
    assert dict(model.graph.initializers) == initializers


@pytest.mark.parametrize("quantized", [False, True])
@pytest.mark.parametrize("opset", [None, 12, 13])
def test_projection_split_opset_boundary(quantized, opset):
    model, _, _, _ = _model(quantized=quantized)
    if opset is None:
        del model.graph.opset_imports[""]
    else:
        model.graph.opset_imports[""] = opset
    before = tuple(model.graph)
    assert _pass(quantized)(model).modified == (opset == 13)
    if opset != 13:
        assert tuple(model.graph) == before


@pytest.mark.parametrize("quantized", [False, True])
def test_projection_matching_does_not_read_lazy_parameters(quantized):
    model, _, _, _ = _model(quantized=quantized)

    def unexpected_read():
        raise AssertionError("projection matching must not read weight payloads")

    for value in model.graph.initializers.values():
        value.const_value = ir.LazyTensor(
            unexpected_read, dtype=value.dtype, shape=value.shape, name=value.name
        )
    assert _pass(quantized)(model).modified
    assert not _pass(quantized)(model).modified


@pytest.mark.parametrize("transposed", [False, True, "mixed"])
@pytest.mark.parametrize("shared", [False, True])
def test_peft_sources_survive_fold_and_serialization(tmp_path, transposed, shared):
    model, weights, _, _ = _model(transposed=transposed, shared=shared)
    assert pack_qkv_projections_pass()(model).modified
    fold_initializers_after_weights(model)
    package = ModelPackage({"model": model})
    package.save(tmp_path / "base")
    restored = ModelPackage.load(tmp_path / "base")["model"]
    assert _resolve_initializer_name(restored, "q") == "q.weight"
    for index, name in enumerate(("q", "k") if shared else ("q", "k", "v")):
        original = weights[f"{name}.weight"]
        if transposed is False or (transposed == "mixed" and index != 0):
            original = original.T
        np.testing.assert_array_equal(
            restored.graph.initializers[f"{name}.weight"].const_value.numpy(), original
        )
    assert not any(
        QKV_PROJECTION_SOURCES_METADATA_KEY in node.metadata_props for node in restored.graph
    )


def test_invalid_peft_source_metadata_fails_closed():
    model, _, _, _ = _model()
    pack_qkv_projections_pass()(model)
    split = next(node for node in model.graph if node.op_type == "Split")
    split.metadata_props[QKV_PROJECTION_SOURCES_METADATA_KEY] = json.dumps({"version": 2})
    with pytest.raises(ValueError, match="unsupported"):
        _resolve_initializer_name(model, "q")


def _write_peft_adapter(directory, modules):
    directory.mkdir()
    (directory / "adapter_config.json").write_text(
        json.dumps(
            {
                "base_model_name_or_path": "offline/tiny",
                "r": 2,
                "lora_alpha": 4,
                "target_modules": list(modules),
            }
        )
    )
    tensors = {}
    for name, (output_size, input_size) in modules.items():
        tensors[f"base_model.model.{name}.lora_A.weight"] = np.ones(
            (2, input_size), dtype=np.float32
        )
        tensors[f"base_model.model.{name}.lora_B.weight"] = np.ones(
            (output_size, 2), dtype=np.float32
        )
    save_file(tensors, directory / "adapter_model.safetensors")


def test_float_initializer_with_outside_consumer_is_not_packed():
    model, _, projections, _ = _model(transposed=False)
    external = ir.Value(
        name="exposed_weight",
        shape=projections[0].inputs[1].shape,
        type=projections[0].inputs[1].type,
    )
    model.graph.append(ir.Node("", "Identity", [projections[0].inputs[1]], outputs=[external]))
    model.graph.outputs.append(external)
    assert not pack_qkv_projections_pass()(model).modified


@pytest.mark.parametrize("sharing", ["initializer", "logical_view"])
def test_tied_projection_groups_decline_packing_and_remain_publicly_attachable(
    tmp_path, sharing
):
    model, weights, projections, attention = _model(transposed=True)
    graph = model.graph
    first_q_view = projections[0].inputs[1]
    source = first_q_view.producer().inputs[0]
    second_outputs = []
    for name, width, original in (
        ("second_q", 32, "q"),
        ("second_key", 16, "k"),
        ("second_value", 16, "v"),
    ):
        if original == "q":
            parameter = source
        else:
            data = weights[f"{original}.weight"].copy()
            parameter = ir.Value(
                name=f"{name}.weight",
                shape=ir.Shape(data.shape),
                type=ir.TensorType(ir.DataType.FLOAT),
                const_value=ir.tensor(data),
            )
            graph.initializers.add(parameter)
        logical = parameter
        if original == "q" and sharing == "logical_view":
            logical = first_q_view
        else:
            logical = ir.Value(
                shape=ir.Shape([64, width]), type=ir.TensorType(ir.DataType.FLOAT)
            )
            graph.append(
                ir.Node(
                    "",
                    "Transpose",
                    [parameter],
                    {"perm": ir.AttrInt64s("perm", [1, 0])},
                    outputs=[logical],
                )
            )
        output = ir.Value(
            name=name,
            shape=ir.Shape([2, "S", width]),
            type=ir.TensorType(ir.DataType.FLOAT),
        )
        graph.append(ir.Node("", "MatMul", [graph.inputs[0], logical], outputs=[output]))
        second_outputs.append(output)
    outputs = [
        ir.Value(name=f"second_{value.name}", shape=value.shape, type=value.type)
        for value in attention.outputs
    ]
    graph.append(
        ir.Node(
            "",
            "Attention",
            [*second_outputs, *attention.inputs[3:]],
            dict(attention.attributes),
            outputs=outputs,
        )
    )
    graph.outputs.extend(outputs)
    packing_result = pack_qkv_projections_pass()(model)
    fold_initializers_after_weights(model)
    package = ModelPackage({"model": model})
    package.save(str(tmp_path / "base"), max_workers=1, progress_bar=False)
    restored = ModelPackage.load(str(tmp_path / "base"))
    adapter = tmp_path / "adapter"
    # Target BOTH groups. Before the outside-consumer guard, packing hid the
    # same q.weight in two metadata records: restoring k created q.weight, then
    # restoring second_key failed with "conflicting source weights". That
    # public attach failure precedes any later save/ownership guard.
    _write_peft_adapter(adapter, {"k": (16, 64), "second_key": (16, 64)})
    artifact = attach_peft_adapter(restored, adapter, name="tied")
    assert len(artifact.weights) == 2
    assert {weight.target.parameter for weight in artifact.weights} == {
        "k.weight",
        "second_key.weight",
    }
    assert not packing_result.modified
    restored.adapter_target_manifest.validate(restored)
    restored.save(str(tmp_path / "attached"), max_workers=1, progress_bar=False)
    roundtrip = ModelPackage.load(str(tmp_path / "attached"))
    assert not any(
        QKV_PROJECTION_SOURCES_METADATA_KEY in node.metadata_props
        for node in roundtrip["model"].graph
    )


@pytest.mark.parametrize("refold", [False, True])
def test_lazy_restoration_rejects_same_path_save_before_read_or_mutation(
    tmp_path, monkeypatch, refold
):
    model, weights, _, _ = _model(transposed=True)
    expected = {name: value.copy() for name, value in weights.items()}
    assert pack_qkv_projections_pass()(model).modified
    fold_initializers_after_weights(model)
    base = tmp_path / "base"
    ModelPackage({"model": model}).save(str(base), max_workers=1, progress_bar=False)
    before = {path.name: path.read_bytes() for path in base.iterdir()}
    restored = ModelPackage.load(str(base))
    assert _resolve_initializer_name(restored["model"], "q") == "q.weight"
    assert isinstance(
        restored["model"].graph.initializers["q.weight"].const_value, ir.LazyTensor
    )

    def unexpected_read(*args, **kwargs):
        raise AssertionError("alias guard must run before reading lazy source tensors")

    with monkeypatch.context() as patch:
        patch.setattr(ir.ExternalTensor, "numpy", unexpected_read)
        if refold:
            fold_initializers_after_weights(restored["model"])
        with pytest.raises(ValueError, match="aliases lazy source file"):
            restored.save(str(base), max_workers=1, progress_bar=False)
    assert before == {path.name: path.read_bytes() for path in base.iterdir()}
    restored.save(str(tmp_path / "safe"), max_workers=1, progress_bar=False)
    roundtrip = ModelPackage.load(str(tmp_path / "safe"))
    for name in ("q", "k", "v"):
        parameter = f"{name}.weight_t" if refold else f"{name}.weight"
        actual = roundtrip["model"].graph.initializers[parameter].const_value.numpy()
        np.testing.assert_array_equal(
            actual.T if refold else actual,
            expected[f"{name}.weight"],
        )


@pytest.mark.parametrize("alias", ["same_path", "hardlink", "symlink"])
def test_public_peft_attach_preserves_lazy_source_on_save_alias(tmp_path, alias):
    model, weights, _, _ = _model(transposed=True)
    expected = {name: value.copy() for name, value in weights.items()}
    assert pack_qkv_projections_pass()(model).modified
    fold_initializers_after_weights(model)
    base = tmp_path / "base"
    ModelPackage({"model": model}).save(str(base), max_workers=1, progress_bar=False)
    before = {path.name: path.read_bytes() for path in base.iterdir()}
    restored = ModelPackage.load(str(base))
    adapter = tmp_path / "adapter"
    _write_peft_adapter(adapter, {"q": (32, 64)})
    attach_peft_adapter(restored, adapter, name="packed")
    destination = base
    if alias != "same_path":
        destination = tmp_path / "alias"
        destination.mkdir()
        target = destination / "model.onnx.data"
        source = base / "model.onnx.data"
        if alias == "hardlink":
            target.hardlink_to(source)
        else:
            target.symlink_to(source)
    with pytest.raises(ValueError, match="aliases lazy source file"):
        restored.save(str(destination), max_workers=1, progress_bar=False)
    assert before == {path.name: path.read_bytes() for path in base.iterdir()}
    if alias != "same_path":
        assert list(destination.iterdir()) == [destination / "model.onnx.data"]
    restored.save(str(tmp_path / "safe"), max_workers=1, progress_bar=False)
    roundtrip = ModelPackage.load(str(tmp_path / "safe"))
    for name in ("q", "k", "v"):
        np.testing.assert_array_equal(
            roundtrip["model"].graph.initializers[f"{name}.weight"].const_value.numpy(),
            expected[f"{name}.weight"],
        )


@pytest.mark.parametrize(
    "artifact_file",
    [
        "adapter.json",
        "adapter_model.safetensors",
        "adapter_config.json",
        "adapter.onnx_adapter",
    ],
)
@pytest.mark.parametrize("alias", ["hardlink", "symlink"])
def test_adapter_artifact_alias_rejected_before_any_writes(tmp_path, artifact_file, alias):
    model, weights, _, _ = _model()
    assert pack_qkv_projections_pass()(model).modified
    fold_initializers_after_weights(model)
    base = tmp_path / "base"
    ModelPackage({"model": model}).save(str(base), max_workers=1, progress_bar=False)
    before = {path.name: path.read_bytes() for path in base.iterdir()}
    restored = ModelPackage.load(str(base))
    adapter = tmp_path / "adapter"
    _write_peft_adapter(adapter, {"q": (32, 64)})
    artifact = attach_peft_adapter(restored, adapter, name="packed")
    restored.adapter_service_options = dataclasses.replace(
        restored.adapter_service_options, preserve_source_format=True
    )
    if artifact_file == "adapter.onnx_adapter":
        # Persistence copies this format verbatim; no runtime/parser is needed
        # to exercise the exact destination naming and pre-write alias guard.
        source = tmp_path / "original.onnx_adapter"
        source.write_bytes(b"opaque adapter artifact")
        restored.adapter_artifacts["packed"] = dataclasses.replace(
            artifact, source=AdapterSource("onnx_adapter", path=str(source))
        )
    destination = tmp_path / "destination"
    artifact_directory = destination / "adapters" / "packed"
    artifact_directory.mkdir(parents=True)
    target = artifact_directory / artifact_file
    source = base / "model.onnx.data"
    if alias == "hardlink":
        target.hardlink_to(source)
    else:
        target.symlink_to(source)
    existing = {path.relative_to(destination) for path in destination.rglob("*")}
    with pytest.raises(ValueError, match="aliases lazy source file"):
        restored.save(str(destination), max_workers=1, progress_bar=False)
    assert before == {path.name: path.read_bytes() for path in base.iterdir()}
    assert existing == {path.relative_to(destination) for path in destination.rglob("*")}
    assert not (destination / "model.onnx").exists()
    safe = tmp_path / "safe"
    restored.save(str(safe), max_workers=1, progress_bar=False)
    assert (safe / "adapters" / "packed" / artifact_file).is_file()
    roundtrip = ModelPackage.load(str(safe))
    for name in ("q", "k", "v"):
        np.testing.assert_array_equal(
            roundtrip["model"].graph.initializers[f"{name}.weight"].const_value.numpy(),
            weights[f"{name}.weight"],
        )
    assert before == {path.name: path.read_bytes() for path in base.iterdir()}


def test_disabled_adapter_artifacts_do_not_block_safe_model_save(tmp_path):
    model, _, _, _ = _model()
    assert pack_qkv_projections_pass()(model).modified
    fold_initializers_after_weights(model)
    base = tmp_path / "base"
    ModelPackage({"model": model}).save(str(base), max_workers=1, progress_bar=False)
    restored = ModelPackage.load(str(base))
    adapter = tmp_path / "adapter"
    _write_peft_adapter(adapter, {"q": (32, 64)})
    attach_peft_adapter(restored, adapter, name="packed")
    output = tmp_path / "output"
    artifact_directory = output / "adapters" / "packed"
    artifact_directory.mkdir(parents=True)
    target = artifact_directory / "adapter.json"
    target.hardlink_to(base / "model.onnx.data")
    before = target.read_bytes()
    restored.save(
        str(output), include_adapter_artifacts=False, max_workers=1, progress_bar=False
    )
    assert target.read_bytes() == before
    assert (output / "model.onnx").is_file()


def test_ep_dtype_role_gating():
    for ep in ep_registry:
        for role in ("decoder", "masked-decoder", "encoder"):
            for dtype in _DTYPES:
                caps = get_ep(ep)
                stages, _ = _get_optimization_passes(caps, dtype, role)
                names = {name for name, _ in stages}
                assert not caps.matmul_nbits_qkv_pack_dtypes
                assert "PackMatMulNBitsQKV" not in names
                assert ("PackQKV" in names) == (
                    role == "decoder" and dtype in caps.qkv_pack_dtypes
                )
                assert ("PackQKVProjections" in names) == (
                    role != "encoder" and dtype in caps.qkv_pack_dtypes
                )


@pytest.mark.parametrize("dtype", list(_DTYPES))
@pytest.mark.parametrize("role", ["decoder", "masked-decoder", "encoder"])
def test_int4_capability_requires_explicit_opt_in(dtype, role):
    caps = dataclasses.replace(get_ep("cuda"), matmul_nbits_qkv_pack_dtypes=frozenset({dtype}))
    stages, _ = _get_optimization_passes(caps, dtype, role)
    assert ("PackMatMulNBitsQKV" in {name for name, _ in stages}) == (role != "encoder")


def test_new_capability_preserves_existing_positional_arguments():
    caps = EpCapabilities("legacy", frozenset(), frozenset(), False)
    assert not caps.supports_fused_rope
    assert not caps.matmul_nbits_qkv_pack_dtypes
    assert dataclasses.fields(EpCapabilities)[-1].name == "matmul_nbits_qkv_pack_dtypes"


@pytest.mark.parametrize("trace", [False, True])
@pytest.mark.parametrize("opt_in", [False, True])
def test_generic_pipeline_preserves_arbitrary_mask_contract(monkeypatch, trace, opt_in):
    model, _, _, _ = _model(quantized=True, deferred=True, dtype=ir.DataType.FLOAT16)
    declare_arbitrary_attention_mask(model.graph)
    caps = dataclasses.replace(
        get_ep("cuda"),
        name="projection-test",
        matmul_nbits_qkv_pack_dtypes=(
            frozenset({ir.DataType.FLOAT16}) if opt_in else frozenset()
        ),
    )
    monkeypatch.setitem(ep_registry._entries, caps.name, caps)
    optimize_model(model, ep=caps.name, dtype=ir.DataType.FLOAT16, trace=trace)
    attention = next(node for node in model.graph if node.op_type == "Attention")
    assert attention.inputs[3].name == "mask"
    assert not any(node.op_type == "GroupQueryAttention" for node in model.graph)
    assert sum(node.op_type == "MatMulNBits" for node in model.graph) == (1 if opt_in else 3)


@pytest.mark.parametrize("quantized", [False, True])
@pytest.mark.parametrize("transposed", [False, True, "mixed"])
@pytest.mark.parametrize("sharing", ["source", "projection"])
def test_repeated_qk_source_declines_before_graph_mutation(quantized, transposed, sharing):
    model, _, projections, attention = _model(
        quantized=quantized, transposed=transposed, q_width=16
    )
    # Use equal-width K/V sources for Q as well, leaving an independent V.
    q, k, _ = projections
    if sharing == "projection":
        attention.replace_input_with(0, attention.inputs[1])
    else:
        q.replace_input_with(1, k.inputs[1])
        if quantized:
            q.replace_input_with(2, k.inputs[2])
            q.replace_input_with(3, k.inputs[3])
            q.attributes["N"] = ir.AttrInt64("N", 16)
        q.outputs[0].shape = k.outputs[0].shape
    before_nodes = tuple(model.graph)
    before_inputs = {node: tuple(node.inputs) for node in before_nodes}
    before_initializers = dict(model.graph.initializers)
    assert not _pass(quantized)(model).modified
    assert tuple(model.graph) == before_nodes
    assert {node: tuple(node.inputs) for node in model.graph} == before_inputs
    assert dict(model.graph.initializers) == before_initializers
    assert not any(node.op_type == "Concat" for node in model.graph)


@pytest.mark.parametrize("sharing", ["same_transpose", "separate_transposes"])
def test_repeated_qk_weight_keeps_independent_v_publicly_attachable(tmp_path, sharing):
    model, _, projections, _ = _model(transposed=True, q_width=16)
    # Separate named Q/K nodes, but both consume k.weight.
    q, k, _ = projections
    if sharing == "same_transpose":
        q.replace_input_with(1, k.inputs[1])
    else:
        q.inputs[1].producer().replace_input_with(0, k.inputs[1].producer().inputs[0])
        q.inputs[1].shape = k.inputs[1].shape
    q.outputs[0].shape = k.outputs[0].shape
    nodes = tuple(model.graph)
    assert not pack_qkv_projections_pass()(model).modified
    assert tuple(model.graph) == nodes
    fold_initializers_after_weights(model)
    base = tmp_path / "base"
    ModelPackage({"model": model}).save(str(base), max_workers=1, progress_bar=False)
    restored = ModelPackage.load(str(base))
    adapter = tmp_path / "adapter"
    _write_peft_adapter(adapter, {"v": (16, 64)})
    artifact = attach_peft_adapter(restored, adapter, name="independent_v")
    assert {weight.target.parameter for weight in artifact.weights} == {"v.weight"}
    restored.adapter_target_manifest.validate(restored)
    restored.save(str(tmp_path / "attached"), max_workers=1, progress_bar=False)


@pytest.mark.parametrize("transposed", [False, True, "mixed"])
def test_independent_qk_adapters_restore_publicly_after_packing(tmp_path, transposed):
    independent, _, _, _ = _model(transposed=transposed)
    assert pack_qkv_projections_pass()(independent).modified
    fold_initializers_after_weights(independent)
    base = tmp_path / "base"
    ModelPackage({"model": independent}).save(str(base), max_workers=1, progress_bar=False)
    restored = ModelPackage.load(str(base))
    adapter = tmp_path / "adapter"
    _write_peft_adapter(adapter, {"q": (32, 64), "k": (16, 64)})
    artifact = attach_peft_adapter(restored, adapter, name="independent")
    assert {weight.target.parameter for weight in artifact.weights} == {
        "q.weight",
        "k.weight",
    }
    restored.adapter_target_manifest.validate(restored)


@pytest.mark.parametrize("fold", [False, True])
def test_deferred_quantized_concat_binding_validation(fold):
    model, weights, _, _ = _model(quantized=True, deferred=True)
    assert pack_matmul_nbits_qkv_pass()(model).modified
    config = SimpleNamespace()
    with pytest.raises(ValueError, match=r"unbound.*parameter"):
        validate_quantized_component_bindings({"model": model}, config)
    package = ModelPackage({"model": model})
    package.apply_weights(
        {name: torch.from_numpy(data) for name, data in weights.items()},
        fold_constants=fold,
    )
    validate_quantized_component_bindings(package, config)
    if not fold:
        package["model"].graph.initializers["k.scales"].const_value = None
        with pytest.raises(ValueError, match=r"unbound MatMulNBits parameter 'k.scales'"):
            validate_quantized_component_bindings(package, config)
