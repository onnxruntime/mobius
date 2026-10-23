# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Value, laziness, and persistence coverage for static weight layout folding."""

from __future__ import annotations

import ml_dtypes
import numpy as np
import onnx_ir as ir
import pytest
from onnx_ir.passes import common

from mobius import ModelPackage
from mobius._optimizations import fold_initializers_after_weights
from mobius._passes import FoldWeightLayoutPass


def _initializer(name, data):
    tensor = ir.tensor(data, name=name)
    return ir.Value(
        name=name, shape=tensor.shape, type=ir.TensorType(tensor.dtype), const_value=tensor
    )


def _model(data, *, first=(2, 2, 4, 3), perm=(0, 2, 1, 3), last=(2, 8, 3)):
    weight = _initializer("weight", data)
    shape1 = _initializer("shape1", np.array(first, dtype=np.int64))
    shape2 = _initializer("shape2", np.array(last, dtype=np.int64))
    reshape1 = ir.Node("", "Reshape", [weight, shape1], num_outputs=1)
    transpose = ir.Node(
        "",
        "Transpose",
        [reshape1.outputs[0]],
        attributes=[ir.AttrInt64s("perm", perm)],
        num_outputs=1,
    )
    reshape2 = ir.Node("", "Reshape", [transpose.outputs[0], shape2], num_outputs=1)
    identity = ir.Node("", "Identity", [reshape2.outputs[0]], num_outputs=1)
    for node, name in zip(
        (reshape1, transpose, reshape2, identity),
        ("reshaped", "transposed", "layout", "output"),
        strict=True,
    ):
        node.outputs[0].name = name
        node.outputs[0].dtype = weight.dtype
    graph = ir.Graph(
        [],
        [identity.outputs[0]],
        nodes=[reshape1, transpose, reshape2, identity],
        initializers=[weight, shape1, shape2],
        opset_imports={"": 24},
        name="layout_test",
    )
    return ir.Model(graph, ir_version=10)


@pytest.mark.parametrize("dtype", [np.uint8, np.float16, np.float32, ml_dtypes.bfloat16])
def test_quantized_layout_values_and_package_roundtrip(dtype, tmp_path):
    data = np.arange(48).astype(dtype).reshape(2, 8, 3)
    model = _model(data)
    fold_initializers_after_weights(model)
    assert [node.op_type for node in model.graph] == ["Identity"]
    assert set(model.graph.initializers) == {"layout"}
    folded = model.graph.initializers["layout"]
    expected = data.reshape(2, 2, 4, 3).transpose(0, 2, 1, 3).reshape(2, 8, 3)
    assert folded.dtype == ir.tensor(data).dtype
    assert folded.const_value.numpy().tobytes() == expected.tobytes()
    package = ModelPackage({"model": model})
    package.save(tmp_path, progress_bar=False)
    loaded = ModelPackage.load(tmp_path)
    restored = loaded["model"].graph.initializers["layout"]
    assert restored.dtype == folded.dtype
    assert restored.shape == ir.Shape(expected.shape)
    assert restored.const_value.numpy().tobytes() == expected.tobytes()


def test_bfloat16_special_value_bits_preserved(tmp_path):
    bits = np.resize(
        np.array([0, 0x8000, 0x7F80, 0xFF80, 0x7FC1, 0x7FFF, 1, 0x8001], dtype=np.uint16),
        48,
    ).reshape(2, 8, 3)
    model = _model(bits.view(ml_dtypes.bfloat16))
    fold_initializers_after_weights(model)
    actual = model.graph.initializers["layout"].const_value.numpy()
    expected = bits.reshape(2, 2, 4, 3).transpose(0, 2, 1, 3).reshape(2, 8, 3)
    assert actual.tobytes() == expected.tobytes()
    ModelPackage({"model": model}).save(tmp_path, progress_bar=False)
    restored = ModelPackage.load(tmp_path)["model"].graph.initializers["layout"]
    assert restored.dtype == ir.DataType.BFLOAT16
    assert restored.const_value.numpy().tobytes() == expected.tobytes()


def test_weight_payload_remains_lazy_and_pass_is_idempotent():
    data = np.arange(48, dtype=np.uint8).reshape(2, 8, 3)
    model = _model(data)
    source = model.graph.initializers["weight"]
    calls = []

    def read():
        calls.append(True)
        return ir.tensor(data)

    source.const_value = ir.LazyTensor(read, dtype=source.dtype, shape=source.shape)
    assert FoldWeightLayoutPass()(model).modified
    assert not calls
    common.RemoveUnusedNodesPass()(model)
    assert set(model.graph.initializers) == {"layout"}
    assert not FoldWeightLayoutPass()(model).modified
    assert not calls
    actual = model.graph.initializers["layout"].const_value.numpy()
    assert len(calls) == 1
    assert (
        actual.tobytes()
        == data.reshape(2, 2, 4, 3).transpose(0, 2, 1, 3).reshape(2, 8, 3).tobytes()
    )


@pytest.mark.parametrize("dynamic", ["weight", "shape1"])
@pytest.mark.parametrize("overridable", [False, True])
def test_dynamic_or_overridable_inputs_are_not_folded(dynamic, overridable):
    model = _model(np.arange(48, dtype=np.uint8).reshape(2, 8, 3))
    value = model.graph.initializers[dynamic]
    model.graph.inputs.append(value)
    if not overridable:
        del model.graph.initializers[dynamic]
        value.const_value = None
    assert not FoldWeightLayoutPass()(model).modified
    assert len(model.graph) == 4


def test_shared_sources_are_not_duplicated():
    data = np.arange(48, dtype=np.uint8).reshape(2, 8, 3)
    model = _model(data)
    source = model.graph.initializers["weight"]
    extra = ir.Node("", "Identity", [source], num_outputs=1)
    extra.outputs[0].name = "original_weight"
    extra.outputs[0].dtype = source.dtype
    model.graph.append(extra)
    model.graph.outputs.append(extra.outputs[0])
    layout = next(node for node in model.graph if node.outputs[0].name == "layout")
    model.graph.outputs.append(layout.outputs[0])
    assert not FoldWeightLayoutPass()(model).modified
    common.RemoveUnusedNodesPass()(model)
    assert [output.name for output in model.graph.outputs] == [
        "output",
        "original_weight",
        "layout",
    ]
    assert extra.inputs[0] is source
    assert source.name in model.graph.initializers
    assert model.graph.outputs[2] is layout.outputs[0]
    assert set(model.graph.initializers) == {"weight", "shape1", "shape2"}
    assert source.const_value.numpy().tobytes() == data.tobytes()


def test_unshared_graph_output_layout_name_survives():
    model = _model(np.arange(48, dtype=np.uint8).reshape(2, 8, 3))
    layout = next(node for node in model.graph if node.outputs[0].name == "layout")
    model.graph.outputs.append(layout.outputs[0])
    assert FoldWeightLayoutPass()(model).modified
    assert model.graph.outputs[1] is model.graph.initializers["layout"]


@pytest.mark.parametrize("op_type", ["Reshape", "Transpose"])
def test_graph_output_source_is_not_duplicated(op_type):
    model = _model(np.arange(48, dtype=np.uint8).reshape(2, 8, 3))
    if op_type == "Transpose":
        first = next(iter(model.graph))
        first.op_type = "Transpose"
        first.replace_input_with(1, None)
        first.resize_inputs(1)
        first.attributes["perm"] = ir.AttrInt64s("perm", [2, 1, 0])
    model.graph.outputs.append(model.graph.initializers["weight"])
    assert not FoldWeightLayoutPass()(model).modified
    assert "layout" not in model.graph.initializers


@pytest.mark.parametrize("style", ["value", "value_ints"])
@pytest.mark.parametrize("shared", [False, True])
def test_post_weight_pipeline_folds_constant_shape_nodes(style, shared):
    data = np.arange(48, dtype=np.uint8).reshape(2, 8, 3)
    model = _model(data)
    if shared:
        extra = ir.Node("", "Identity", [model.graph.initializers["weight"]], num_outputs=1)
        extra.outputs[0].name = "shared"
        model.graph.append(extra)
        model.graph.outputs.append(extra.outputs[0])
    for name in ("shape1", "shape2"):
        shape = model.graph.initializers[name]
        attribute = (
            ir.AttrTensor("value", shape.const_value)
            if style == "value"
            else ir.AttrInt64s("value_ints", shape.const_value.numpy().tolist())
        )
        constant = ir.Node("", "Constant", [], attributes=[attribute], num_outputs=1)
        constant.outputs[0].name = name
        model.graph.insert_before(next(iter(model.graph)), constant)
        shape.replace_all_uses_with(constant.outputs[0])
        del model.graph.initializers[name]
    fold_initializers_after_weights(model)
    if shared:
        assert set(model.graph.initializers) == {"weight"}
        assert sum(node.op_type == "Reshape" for node in model.graph) == 2
        assert sum(node.op_type == "Transpose" for node in model.graph) == 1
        return
    assert [node.op_type for node in model.graph] == ["Identity"]
    assert set(model.graph.initializers) == {"layout"}
    expected = data.reshape(2, 2, 4, 3).swapaxes(1, 2).reshape(2, 8, 3)
    assert (
        model.graph.initializers["layout"].const_value.numpy().tobytes() == expected.tobytes()
    )


def test_tied_embedding_pipeline_keeps_one_table(tmp_path):
    weight = _initializer("weight", np.arange(24, dtype=np.float16).reshape(8, 3))
    indices = ir.Value(
        name="indices", shape=ir.Shape([2]), type=ir.TensorType(ir.DataType.INT64)
    )
    gather = ir.Node("", "Gather", [weight, indices], num_outputs=1)
    transpose = ir.Node(
        "", "Transpose", [weight], attributes=[ir.AttrInt64s("perm", [1, 0])], num_outputs=1
    )
    matmul = ir.Node("", "MatMul", [gather.outputs[0], transpose.outputs[0]], num_outputs=1)
    for node, name in ((gather, "embedding"), (transpose, "head"), (matmul, "output")):
        node.outputs[0].name = name
        node.outputs[0].dtype = ir.DataType.FLOAT16
    graph = ir.Graph(
        [indices],
        [matmul.outputs[0]],
        nodes=[gather, transpose, matmul],
        initializers=[weight],
        opset_imports={"": 24},
        name="tied_table",
    )
    model = ir.Model(graph, ir_version=10)
    fold_initializers_after_weights(model)
    assert set(graph.initializers) == {"weight"}
    assert transpose.inputs[0] is gather.inputs[0]
    ModelPackage({"model": model}).save(tmp_path, progress_bar=False)
    loaded = ModelPackage.load(tmp_path)["model"].graph
    assert set(loaded.initializers) == {"weight"}
    assert [node.op_type for node in loaded] == ["Gather", "Transpose", "MatMul"]


@pytest.mark.parametrize(
    ("source", "requested", "allowzero", "expected"),
    [
        ((2, 3), (0, -1), 0, (2, 3)),
        ((2, 3), (-1, 2), 0, (3, 2)),
        ((0, 3), (0, 3), 1, (0, 3)),
        ((0, 3), (-1, 3), 0, (0, 3)),
        ((1,), (), 0, ()),
    ],
)
def test_reshape_zero_inferred_and_scalar_shapes(source, requested, allowzero, expected):
    weight = _initializer("weight", np.zeros(source, dtype=np.float16))
    shape = _initializer("shape", np.array(requested, dtype=np.int64))
    node = ir.Node(
        "",
        "Reshape",
        [weight, shape],
        attributes=[ir.AttrInt64("allowzero", allowzero)],
        num_outputs=1,
    )
    node.outputs[0].name = "output"
    graph = ir.Graph(
        [],
        [node.outputs[0]],
        nodes=[node],
        initializers=[weight, shape],
        opset_imports={"": 24},
    )
    model = ir.Model(graph, ir_version=10)
    assert FoldWeightLayoutPass()(model).modified
    assert model.graph.outputs[0].shape == ir.Shape(expected)
    assert model.graph.outputs[0].const_value.numpy().shape == expected


@pytest.mark.parametrize("requested", [(0, -1), (-1, -1), (4,), (-2,)])
def test_invalid_static_reshape_fails_explicitly(requested):
    model = _model(np.zeros((0, 3), dtype=np.float16), first=requested)
    with pytest.raises(ValueError, match="Reshape"):
        FoldWeightLayoutPass()(model)


def test_default_transpose_and_missing_declared_dtype():
    data = np.arange(24, dtype=np.float16).reshape(2, 3, 4)
    weight = _initializer("weight", data)
    weight.type = None
    node = ir.Node("", "Transpose", [weight], num_outputs=1)
    node.outputs[0].name = "output"
    graph = ir.Graph([], [node.outputs[0]], nodes=[node], initializers=[weight])
    model = ir.Model(graph, ir_version=10)
    assert FoldWeightLayoutPass()(model).modified
    output = graph.outputs[0]
    assert output.dtype == ir.DataType.FLOAT16
    assert output.const_value.numpy().tobytes() == data.transpose().tobytes()


def test_invalid_permutation_and_conflicting_dtype_fail():
    model = _model(np.arange(48, dtype=np.uint8).reshape(2, 8, 3), perm=(0, 0, 1, 2))
    with pytest.raises(ValueError, match="permutation"):
        FoldWeightLayoutPass()(model)
    model = _model(np.arange(48, dtype=np.uint8).reshape(2, 8, 3))
    model.graph.initializers["weight"].dtype = ir.DataType.FLOAT
    with pytest.raises(ValueError, match="declares dtype"):
        FoldWeightLayoutPass()(model)


def test_custom_domain_layout_op_is_not_folded():
    model = _model(np.arange(48, dtype=np.uint8).reshape(2, 8, 3))
    transpose = next(node for node in model.graph if node.op_type == "Transpose")
    transpose.domain = "custom"
    FoldWeightLayoutPass()(model)
    assert transpose in list(model.graph)
    assert transpose.outputs[0].producer() is transpose


def test_missing_weights_are_warned_and_left_unmodified(caplog):
    model = _model(np.arange(48, dtype=np.uint8).reshape(2, 8, 3))
    model.graph.initializers["weight"].const_value = None
    assert not FoldWeightLayoutPass()(model).modified
    assert "has no data" in caplog.text


def test_invalid_shape_tensor_dtype_fails_explicitly():
    model = _model(np.arange(48, dtype=np.uint8).reshape(2, 8, 3))
    shape = model.graph.initializers["shape1"]
    shape.dtype = ir.DataType.INT32
    shape.const_value = ir.tensor(np.array([2, 2, 4, 3], dtype=np.int32))
    with pytest.raises(ValueError, match="INT64 shape"):
        FoldWeightLayoutPass()(model)


def test_nested_graph_initializers_are_registered_in_their_scope():
    first = _model(np.arange(48, dtype=np.uint8).reshape(2, 8, 3)).graph
    second = _model(np.arange(48, dtype=np.uint8).reshape(2, 8, 3)).graph
    condition = ir.Value(
        name="condition", shape=ir.Shape([]), type=ir.TensorType(ir.DataType.BOOL)
    )
    node = ir.Node(
        "",
        "If",
        [condition],
        attributes=[ir.AttrGraph("then_branch", first), ir.AttrGraph("else_branch", second)],
        num_outputs=1,
    )
    node.outputs[0].name = "result"
    graph = ir.Graph([condition], [node.outputs[0]], nodes=[node], opset_imports={"": 24})
    model = ir.Model(graph, ir_version=10)
    assert FoldWeightLayoutPass()(model).modified
    assert not graph.initializers
    for branch in (first, second):
        assert [item.op_type for item in branch] == ["Identity"]
        assert set(branch.initializers) == {"layout"}
