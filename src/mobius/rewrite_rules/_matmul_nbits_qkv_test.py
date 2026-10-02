# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Attention-anchored INT4 packing, storage preservation, and native parity."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import onnx_ir as ir
import onnxruntime as ort
import pytest

from mobius._optimizations import fold_initializers_after_weights
from mobius.functions import get_function, matmul_nbits, register_function_bodies
from mobius.rewrite_rules import pack_matmul_nbits_qkv_pass


def _value(name, dtype, shape, data=None):
    return ir.Value(
        name=name,
        type=ir.TensorType(dtype),
        shape=ir.Shape(shape),
        const_value=None if data is None else ir.tensor(data),
    )


def _model(*, k=65, block=32, widths=(8, 6, 10), zp=False, loaded=True, path=False):
    rng = np.random.default_rng(17)
    a = _value("A", ir.DataType.FLOAT16, ["batch", "sequence", k])
    blocks = (k + block - 1) // block
    initializers = []
    nodes = []
    projections = []

    def parameter(name, dtype, shape, data):
        value = _value(name, dtype, shape, data if loaded else None)
        initializers.append(value)
        return value

    for index, n in enumerate(widths):
        weight = parameter(
            f"w{index}",
            ir.DataType.UINT8,
            [n, blocks, block // 2],
            rng.integers(1, 256, (n, blocks, block // 2), dtype=np.uint8),
        )
        scales = parameter(
            f"s{index}",
            ir.DataType.FLOAT16,
            [n, blocks],
            rng.uniform(0.01, 0.1, (n, blocks)).astype(np.float16),
        )
        inputs = [a, weight, scales]
        if zp:
            inputs.append(
                parameter(
                    f"z{index}",
                    ir.DataType.UINT8,
                    [n, (blocks + 1) // 2],
                    rng.integers(1, 256, (n, (blocks + 1) // 2), dtype=np.uint8),
                )
            )
        y = _value(f"y{index}", ir.DataType.FLOAT16, ["batch", "sequence", n])
        y.metadata_props["test.projection"] = str(index)
        node = ir.Node(
            "com.microsoft",
            "MatMulNBits",
            inputs,
            outputs=[y],
            attributes=ir.convenience.convert_attributes(
                {"K": k, "N": n, "block_size": block, "bits": 4}
            ),
        )
        projections.append(node)
        nodes.append(node)
    attention_inputs = []
    for index, projection in enumerate(projections):
        value = projection.outputs[0]
        if path:
            width = widths[index]
            bias = parameter(
                f"b{index}",
                ir.DataType.FLOAT16,
                [width],
                rng.uniform(0.1, 1, width).astype(np.float16),
            )
            biased = _value(f"bias{index}", ir.DataType.FLOAT16, list(value.shape))
            nodes.append(
                ir.Node(
                    "", "Add", [bias, value] if index == 0 else [value, bias], outputs=[biased]
                )
            )
            value = biased
            if index < 2:
                shape4 = parameter(
                    f"shape4_{index}",
                    ir.DataType.INT64,
                    [4],
                    np.array([0, 0, -1, 2], dtype=np.int64),
                )
                shape3 = parameter(
                    f"shape3_{index}",
                    ir.DataType.INT64,
                    [3],
                    np.array([0, 0, -1], dtype=np.int64),
                )
                scale = parameter(
                    f"norm{index}",
                    ir.DataType.FLOAT16,
                    [2],
                    np.array([1, 2], dtype=np.float16),
                )
                reshaped = _value(
                    f"reshape4_{index}",
                    ir.DataType.FLOAT16,
                    ["batch", "sequence", width // 2, 2],
                )
                normed = _value(f"normed{index}", ir.DataType.FLOAT16, list(reshaped.shape))
                flattened = _value(
                    f"reshape3_{index}", ir.DataType.FLOAT16, ["batch", "sequence", width]
                )
                nodes.extend(
                    [
                        ir.Node("", "Reshape", [value, shape4], outputs=[reshaped]),
                        ir.Node(
                            "",
                            "RMSNormalization",
                            [reshaped, scale],
                            outputs=[normed],
                            attributes=ir.convenience.convert_attributes(
                                {"axis": -1, "epsilon": 1e-5}
                            ),
                        ),
                        ir.Node("", "Reshape", [normed, shape3], outputs=[flattened]),
                    ]
                )
                value = flattened
        attention_inputs.append(value)
    anchor = ir.Node(
        "",
        "Attention",
        attention_inputs,
        attributes=ir.convenience.convert_attributes(
            {"q_num_heads": 4, "kv_num_heads": 3, "is_causal": 1}
        ),
        outputs=[_value("attention", ir.DataType.FLOAT16, ["batch", "sequence", widths[0]])],
    )
    nodes.append(anchor)
    graph = ir.Graph(
        [a],
        [node.outputs[0] for node in projections],
        nodes=nodes,
        initializers=initializers,
        opset_imports={"": 24, "com.microsoft": 1},
        name="qkv",
    )
    return ir.Model(graph, ir_version=10), projections


def _count(model, op):
    return sum(node.op_type == op for node in model.graph)


@pytest.mark.parametrize("zp", [False, True])
@pytest.mark.parametrize("path", [False, True])
def test_pack_preserves_outputs_consumers_and_parameter_sections(zp, path):
    model, projections = _model(zp=zp, path=path)
    graph = model.graph
    original_parameters = dict(graph.initializers)
    downstream = [node for node in graph if node not in projections]
    attributes = [dict(node.attributes) for node in downstream]
    output_names = [value.name for value in graph.outputs]
    original_values = [
        value.const_value.numpy().copy() for value in graph.initializers.values()
    ]
    result = pack_matmul_nbits_qkv_pass()(model)
    assert result.modified
    assert _count(model, "MatMulNBits") == 1
    assert _count(model, "Split") == 1
    assert _count(model, "Concat") == 2 + zp
    assert [value.name for value in graph.outputs] == output_names
    assert [value.metadata_props["test.projection"] for value in graph.outputs] == [
        "0",
        "1",
        "2",
    ]
    assert all(node.graph is graph for node in downstream)
    assert [dict(node.attributes) for node in downstream] == attributes
    assert all(
        graph.initializers[name] is value for name, value in original_parameters.items()
    )
    for value, expected in zip(graph.initializers.values(), original_values, strict=True):
        np.testing.assert_array_equal(value.const_value.numpy(), expected)
    concats = [node for node in graph if node.op_type == "Concat"]
    expected_packed = [
        np.concatenate([value.const_value.numpy() for value in node.inputs], axis=0)
        for node in concats
    ]
    assert not pack_matmul_nbits_qkv_pass()(model).modified
    fold_initializers_after_weights(model)
    packed_node = next(node for node in graph if node.op_type == "MatMulNBits")
    for value, expected in zip(packed_node.inputs[1:], expected_packed, strict=True):
        np.testing.assert_array_equal(value.const_value.numpy(), expected)


@pytest.mark.parametrize("anchor_type", ["Attention", "GroupQueryAttention"])
@pytest.mark.parametrize("overload", ["", "default_zero_points"])
def test_supported_anchors_and_missing_output_metadata(anchor_type, overload):
    model, projections = _model()
    anchor = next(node for node in model.graph if node.op_type == "Attention")
    anchor.op_type = anchor_type
    if anchor_type == "GroupQueryAttention":
        anchor.domain = "com.microsoft"
    for node in projections:
        node.overload = overload
        node.outputs[0].shape = None
        node.outputs[0].type = None
    register_function_bodies(model)
    assert pack_matmul_nbits_qkv_pass()(model).modified
    assert [list(value.shape) for value in model.graph.outputs] == [
        ["batch", "sequence", width] for width in (8, 6, 10)
    ]
    assert all(value.dtype == ir.DataType.FLOAT16 for value in model.graph.outputs)


@pytest.mark.parametrize("opset", [None, 12, 13])
def test_standard_opset_split_boundary(opset):
    model, _ = _model()
    graph = model.graph
    anchor = next(node for node in graph if node.op_type == "Attention")
    anchor.domain = "com.microsoft"
    anchor.op_type = "GroupQueryAttention"
    if opset is None:
        del graph.opset_imports[""]
    else:
        graph.opset_imports[""] = opset
    original_nodes = list(graph)
    original_initializers = dict(graph.initializers)
    assert pack_matmul_nbits_qkv_pass()(model).modified == (opset == 13)
    if opset != 13:
        assert list(graph) == original_nodes
        assert dict(graph.initializers) == original_initializers
    else:
        split = next(node for node in graph if node.op_type == "Split")
        assert len(split.inputs) == 2
    assert graph.opset_imports.get("") == opset


_NEGATIVE_CASES = [
    "mixed_bits",
    "wrong_attribute_type",
    "zero_k",
    "zero_n",
    "multiple_outputs",
    "default_overload_with_zp",
    "all_eight",
    "different_a",
    "different_k",
    "different_block",
    "block_not_power_two",
    "block_small",
    "accuracy",
    "accuracy_presence",
    "unknown_attr",
    "unknown_overload",
    "mixed_overload",
    "user_function",
    "user_default_function",
    "float32_a",
    "bfloat16_a",
    "rank_two",
    "unknown_k",
    "output_shape",
    "output_dtype",
    "weight_shape",
    "scale_flat",
    "scale_dtype",
    "weight_dtype",
    "tensor_shape",
    "tensor_dtype",
    "mixed_zp",
    "float_zp",
    "mixed_zp_dtype",
    "zp_flat",
    "computed_weight",
    "foreign_weight",
    "overridable_weight",
    "g_idx",
    "fused_bias",
    "opset",
    "no_anchor",
    "gate",
    "dynamic_reshape",
    "extra_reshape",
    "decomposed_norm",
]


@pytest.mark.parametrize("case", _NEGATIVE_CASES)
def test_incompatible_triples_are_complete_noops(case):
    model, projections = _model(
        zp="zp" in case, path=case in {"dynamic_reshape", "extra_reshape", "decomposed_norm"}
    )
    graph = model.graph
    q, _, v = projections

    def attr(node, name, value):
        node.attributes[name] = ir.AttrInt64(name, value)

    if case == "mixed_bits":
        attr(v, "bits", 8)
    elif case == "wrong_attribute_type":
        v.attributes["K"] = ir.AttrFloat32("K", 65.0)
    elif case == "zero_k":
        for node in projections:
            attr(node, "K", 0)
    elif case == "zero_n":
        attr(v, "N", 0)
    elif case == "multiple_outputs":
        v.resize_outputs(2)
    elif case == "default_overload_with_zp":
        for node in projections:
            node.overload = "default_zero_points"
    elif case == "all_eight":
        for node in projections:
            attr(node, "bits", 8)
    elif case == "different_a":
        other = _value("other", ir.DataType.FLOAT16, list(q.inputs[0].shape))
        graph.inputs.append(other)
        v.replace_input_with(0, other)
    elif case == "different_k":
        attr(v, "K", 64)
    elif case == "different_block":
        attr(v, "block_size", 128)
    elif case in ("block_not_power_two", "block_small"):
        for node in projections:
            attr(node, "block_size", 24 if case == "block_not_power_two" else 8)
    elif case == "accuracy":
        for node in projections:
            attr(node, "accuracy_level", 4)
    elif case == "accuracy_presence":
        attr(v, "accuracy_level", 0)
    elif case == "unknown_attr":
        attr(v, "unknown", 1)
    elif case == "unknown_overload":
        for node in projections:
            node.overload = "unknown"
    elif case == "mixed_overload":
        v.overload = "default_zero_points"
    elif case in ("user_function", "user_default_function"):
        function = matmul_nbits(has_zero_points=case == "user_function")
        model.functions[function.identifier()] = function
        if case == "user_default_function":
            for node in projections:
                node.overload = "default_zero_points"
    elif case in ("float32_a", "bfloat16_a"):
        q.inputs[0].dtype = ir.DataType.FLOAT if case == "float32_a" else ir.DataType.BFLOAT16
    elif case == "rank_two":
        q.inputs[0].shape = ir.Shape([2, 65])
    elif case == "unknown_k":
        q.inputs[0].shape = ir.Shape(["batch", "sequence", "hidden"])
    elif case == "output_shape":
        v.outputs[0].shape = ir.Shape(["batch", "sequence", 9])
    elif case == "output_dtype":
        v.outputs[0].dtype = ir.DataType.FLOAT
    elif case == "weight_shape":
        v.inputs[1].shape = ir.Shape([10, 48])
    elif case == "scale_flat":
        v.inputs[2].shape = ir.Shape([30])
    elif case in ("scale_dtype", "weight_dtype"):
        v.inputs[2 if case == "scale_dtype" else 1].dtype = ir.DataType.FLOAT
    elif case == "tensor_shape":
        v.inputs[1].const_value = ir.tensor(np.ones((10, 3, 15), dtype=np.uint8))
    elif case == "tensor_dtype":
        v.inputs[2].const_value = ir.tensor(np.ones((10, 3), dtype=np.float32))
    elif case == "mixed_zp":
        v.replace_input_with(3, None)
    elif case in ("float_zp", "mixed_zp_dtype"):
        for node in projections if case == "float_zp" else [v]:
            node.inputs[3].dtype = ir.DataType.FLOAT
    elif case == "zp_flat":
        v.inputs[3].shape = ir.Shape([20])
    elif case == "computed_weight":
        weight = v.inputs[1]
        computed = _value("computed", weight.dtype, list(weight.shape))
        graph.insert_before(v, ir.Node("", "Identity", [weight], outputs=[computed]))
        v.replace_input_with(1, computed)
    elif case == "foreign_weight":
        graph.initializers.pop(v.inputs[1].name)
    elif case == "overridable_weight":
        graph.inputs.append(v.inputs[1])
    elif case in ("g_idx", "fused_bias"):
        v.resize_inputs(6)
        v.replace_input_with(4 if case == "g_idx" else 5, v.inputs[2])
    elif case == "opset":
        graph.opset_imports["com.microsoft"] = 2
    elif case == "no_anchor":
        graph.remove(next(node for node in graph if node.op_type == "Attention"), safe=True)
    elif case == "gate":
        anchor = next(node for node in graph if node.op_type == "Attention")
        gated = _value("gated", ir.DataType.FLOAT16, list(v.outputs[0].shape))
        graph.insert_before(
            anchor, ir.Node("", "Mul", [v.outputs[0], v.outputs[0]], outputs=[gated])
        )
        anchor.replace_input_with(2, gated)
    elif case == "dynamic_reshape":
        graph.initializers["shape3_0"].const_value = None
    elif case == "extra_reshape":
        anchor = next(node for node in graph if node.op_type == "Attention")
        value = anchor.inputs[0]
        extra = _value("extra", value.dtype, list(value.shape))
        graph.insert_before(
            anchor,
            ir.Node("", "Reshape", [value, graph.initializers["shape3_0"]], outputs=[extra]),
        )
        anchor.replace_input_with(0, extra)
    elif case == "decomposed_norm":
        next(node for node in graph if node.op_type == "RMSNormalization").op_type = "Mul"
    else:
        pytest.fail(f"Unhandled case {case}")
    original_nodes = list(graph)
    original_initializers = dict(graph.initializers)
    assert not pack_matmul_nbits_qkv_pass()(model).modified
    assert list(graph) == original_nodes
    assert dict(graph.initializers) == original_initializers
    assert _count(model, "MatMulNBits") == 3
    assert _count(model, "Split") == 0


def test_shared_parameters_external_consumers_and_trailing_none():
    model, projections = _model(widths=(8, 8, 8))
    q, k, v = projections
    for node in (k, v):
        node.replace_input_with(1, q.inputs[1])
    for node in projections:
        node.resize_inputs(7)
    external = _value("external", ir.DataType.UINT8, list(q.inputs[1].shape))
    model.graph.append(ir.Node("", "Identity", [q.inputs[1]], outputs=[external]))
    model.graph.outputs.append(external)
    assert pack_matmul_nbits_qkv_pass()(model).modified
    fold_initializers_after_weights(model)
    assert "w0" in model.graph.initializers
    assert model.graph.outputs[-1].name == "external"
    assert model.graph.outputs[-1].producer().inputs[0] is model.graph.initializers["w0"]


def test_duplicate_and_overlapping_attention_anchors():
    model, projections = _model(widths=(8, 8, 8))
    anchor = next(node for node in model.graph if node.op_type == "Attention")
    model.graph.append(
        ir.Node("", "Attention", list(anchor.inputs), outputs=[ir.Value(name="duplicate")])
    )
    assert pack_matmul_nbits_qkv_pass()(model).modified
    assert _count(model, "MatMulNBits") == 1

    model, projections = _model(widths=(8, 8, 8))
    q, k, v = [node.outputs[0] for node in projections]
    model.graph.append(ir.Node("", "Attention", [q, v, k], outputs=[ir.Value(name="overlap")]))
    assert not pack_matmul_nbits_qkv_pass()(model).modified
    assert _count(model, "MatMulNBits") == 3


def test_unloaded_parameters_survive_repeated_passes_and_premature_folding():
    model, _ = _model(loaded=False)
    names = set(model.graph.initializers)
    assert pack_matmul_nbits_qkv_pass()(model).modified
    for _ in range(2):
        fold_initializers_after_weights(model)
        assert not pack_matmul_nbits_qkv_pass()(model).modified
        assert set(model.graph.initializers) == names
        assert all(value.const_value is None for value in model.graph.initializers.values())
        assert _count(model, "Concat") == 2


def test_matcher_does_not_materialize_loaded_lazy_parameters():
    model, _ = _model()

    def materialize():
        pytest.fail("Matcher materialized a parameter")

    for value in model.graph.initializers.values():
        value.const_value = ir.LazyTensor(materialize, dtype=value.dtype, shape=value.shape)
    assert pack_matmul_nbits_qkv_pass()(model).modified


def test_output_unknown_dimensions_and_metadata_are_preserved():
    model, projections = _model()
    for node in projections:
        node.outputs[0].shape = ir.Shape([None, None, node.attributes.get_int("N")])
        node.outputs[0].doc_string = "Projection output"
        node.outputs[0].meta["test.meta"] = "preserve"
    assert pack_matmul_nbits_qkv_pass()(model).modified
    for value in model.graph.outputs:
        assert value.doc_string == "Projection output"
        assert value.meta["test.meta"] == "preserve"
        assert list(value.shape)[:2] == ["batch", "sequence"]


def test_packing_inserts_before_earliest_projection_and_external_user():
    model, projections = _model()
    q, k, _ = projections
    graph = model.graph
    graph.remove(q)
    graph.insert_after(k, q)
    external = _value("external", ir.DataType.FLOAT16, list(k.outputs[0].shape))
    graph.insert_before(q, ir.Node("", "Identity", [k.outputs[0]], outputs=[external]))
    graph.outputs.append(external)
    assert pack_matmul_nbits_qkv_pass()(model).modified
    indices = {node: index for index, node in enumerate(graph)}
    for node in graph:
        for value in node.inputs:
            if value is not None and value.producer() is not None:
                assert indices[value.producer()] < indices[node]
    assert external.producer().inputs[0].name == "y1"


def test_nested_graph_attention_is_not_packed():
    branch, _ = _model()
    condition = _value("condition", ir.DataType.BOOL, [])
    outputs = [ir.Value(name=f"branch_out_{index}") for index in range(3)]
    node = ir.Node(
        "",
        "If",
        [condition],
        outputs=outputs,
        attributes={"then_branch": ir.AttrGraph("then_branch", branch.graph)},
    )
    model = ir.Model(
        ir.Graph([condition], outputs, nodes=[node], opset_imports={"": 24}), ir_version=10
    )
    assert not pack_matmul_nbits_qkv_pass()(model).modified
    assert _count(branch, "MatMulNBits") == 3


def _reference(node, a):
    k = node.attributes.get_int("K")
    block = node.attributes.get_int("block_size")
    packed = node.inputs[1].const_value.numpy()
    codes = np.stack([packed & 15, packed >> 4], axis=-1).reshape(packed.shape[0], -1)
    scales = node.inputs[2].const_value.numpy().astype(np.float32)
    if len(node.inputs) > 3 and node.inputs[3] is not None:
        z = node.inputs[3].const_value.numpy()
        zero = np.stack([z & 15, z >> 4], axis=-1).reshape(z.shape[0], -1)
        zero = zero[:, : scales.shape[1]]
    else:
        zero = np.full(scales.shape, 8, dtype=np.float32)
    weight = (
        codes[:, :k].astype(np.float32) - np.repeat(zero, block, axis=1)[:, :k]
    ) * np.repeat(scales, block, axis=1)[:, :k]
    return (a.astype(np.float32) @ weight.T).astype(np.float16)


def _session(model, directory, label, provider):
    path = directory / f"{label}.onnx"
    ir.save(model, path)
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = 1
    options.enable_profiling = True
    options.profile_file_prefix = str(directory / label)
    return ort.InferenceSession(
        str(path),
        options,
        providers=[provider, "CPUExecutionProvider"]
        if provider != "CPUExecutionProvider"
        else [provider],
    )


def _assignments(session):
    with Path(session.end_profiling()).open() as stream:
        events = json.load(stream)
    return [
        event["args"]["provider"]
        for event in events
        if event.get("cat") == "Node"
        and event.get("args", {}).get("op_name") == "MatMulNBits"
        and "provider" in event["args"]
    ]


@pytest.mark.parametrize("provider", ["CPUExecutionProvider", "CUDAExecutionProvider"])
@pytest.mark.parametrize("k,block", [(64, 32), (65, 32), (384, 128), (257, 128)])
@pytest.mark.parametrize("zp", [False, True])
def test_projection_numerics_and_native_assignment(tmp_path, provider, k, block, zp):
    if provider not in ort.get_available_providers():
        pytest.skip(f"{provider} is not available")
    source, projections = _model(k=k, block=block, zp=zp)
    source.graph.remove(
        next(node for node in source.graph if node.op_type == "Attention"), safe=True
    )
    packed, _ = _model(k=k, block=block, zp=zp)
    assert pack_matmul_nbits_qkv_pass()(packed).modified
    packed.graph.remove(
        next(node for node in packed.graph if node.op_type == "Attention"), safe=True
    )
    fold_initializers_after_weights(packed)
    source_session = _session(source, tmp_path, "source", provider)
    if provider == "CUDAExecutionProvider" and provider not in source_session.get_providers():
        pytest.skip("Baseline CUDAExecutionProvider could not initialize on this host")
    packed_session = _session(packed, tmp_path, "packed", provider)
    if provider == "CUDAExecutionProvider":
        assert provider in packed_session.get_providers(), (
            "Packed session lost CUDAExecutionProvider after baseline CUDA initialization"
        )
    rng = np.random.default_rng(23)
    for sequence in (3, 1):
        a = rng.standard_normal((2, sequence, k)).astype(np.float16)
        original = source_session.run(None, {"A": a})
        optimized = packed_session.run(None, {"A": a})
        for node, before, after in zip(projections, original, optimized, strict=True):
            np.testing.assert_allclose(after, before, rtol=1e-2, atol=1e-2)
            np.testing.assert_allclose(after, _reference(node, a), rtol=1e-2, atol=1e-2)
    source_assignments = _assignments(source_session)
    packed_assignments = _assignments(packed_session)
    assert source_assignments == [provider] * 6, source_assignments
    assert packed_assignments == [provider] * 2, packed_assignments
    print(
        f"{provider} K={k} block={block} zp={zp}: "
        f"source={len(source_assignments)} packed={len(packed_assignments)} native events"
    )


@pytest.mark.parametrize("baseline_cuda", [False, True])
def test_native_cuda_skip_gate_requires_baseline_failure(tmp_path, monkeypatch, baseline_cuda):
    provider = "CUDAExecutionProvider"
    monkeypatch.setattr(ort, "get_available_providers", lambda: [provider])
    created = []

    def session(model, directory, label, requested_provider):
        created.append(label)
        providers = [provider] if baseline_cuda and label == "source" else []
        return SimpleNamespace(get_providers=lambda: providers)

    monkeypatch.setitem(
        test_projection_numerics_and_native_assignment.__globals__, "_session", session
    )
    expected = AssertionError if baseline_cuda else pytest.skip.Exception
    message = "Packed session lost CUDA" if baseline_cuda else "Baseline CUDA"
    with pytest.raises(expected, match=message):
        test_projection_numerics_and_native_assignment(
            tmp_path, provider, k=64, block=32, zp=False
        )
    assert created == (["source", "packed"] if baseline_cuda else ["source"])


def test_explicit_zero_accuracy_and_ordinary_builtin_overload():
    model, projections = _model(zp=True)
    for node in projections:
        node.attributes["accuracy_level"] = ir.AttrInt64("accuracy_level", 0)
    function = get_function(("com.microsoft", "MatMulNBits", ""))
    assert function is not None
    model.functions[function.identifier()] = function
    assert pack_matmul_nbits_qkv_pass()(model).modified
    packed = next(node for node in model.graph if node.op_type == "MatMulNBits")
    assert packed.attributes.get_int("accuracy_level") == 0
    assert packed.overload == ""
