# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Value, laziness, and persistence coverage for static weight layout folding."""

from __future__ import annotations

from pathlib import Path

import ml_dtypes
import numpy as np
import onnx_ir as ir
import pytest
from onnx_ir.passes import common

from mobius import ModelPackage
from mobius._constants import LAZY_SOURCE_FILES_META_KEY
from mobius._optimizations import fold_initializers_after_weights
from mobius._passes import FoldWeightLayoutPass
from mobius._testing.ort_inference import OnnxModelSession


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
    # The shape-expression constant folder also resolves Identity symbolically,
    # preserving the public output name on the eventual lazy initializer.
    assert not list(model.graph)
    assert set(model.graph.initializers) == {"output"}
    folded = model.graph.initializers["output"]
    assert model.graph.outputs[0] is folded
    assert isinstance(folded.const_value, ir.LazyTensor)
    expected = data.reshape(2, 2, 4, 3).transpose(0, 2, 1, 3).reshape(2, 8, 3)
    assert folded.dtype == ir.tensor(data).dtype
    assert folded.const_value.numpy().tobytes() == expected.tobytes()
    package = ModelPackage({"model": model})
    package.save(tmp_path, progress_bar=False)
    loaded = ModelPackage.load(tmp_path)
    restored = loaded["model"].graph.initializers["output"]
    assert loaded["model"].graph.outputs[0] is restored
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
    folded = model.graph.initializers["output"]
    assert set(model.graph.initializers) == {"output"}
    assert model.graph.outputs[0] is folded
    assert isinstance(folded.const_value, ir.LazyTensor)
    actual = folded.const_value.numpy()
    expected = bits.reshape(2, 2, 4, 3).transpose(0, 2, 1, 3).reshape(2, 8, 3)
    assert actual.tobytes() == expected.tobytes()
    ModelPackage({"model": model}).save(tmp_path, progress_bar=False)
    restored_graph = ModelPackage.load(tmp_path)["model"].graph
    restored = restored_graph.initializers["output"]
    assert restored_graph.outputs[0] is restored
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


@pytest.mark.parametrize(
    ("domain", "op_type", "parameter_slot", "protected"),
    [
        ("com.microsoft", "MoE", 2, True),
        ("com.microsoft", "QMoE", 2, True),
        ("com.microsoft", "QMoE", 3, True),
        ("", "DequantizeLinear", 0, True),
        ("", "DequantizeLinear", 1, True),
        ("com.microsoft", "MatMulNBits", 1, False),
        ("", "MatMul", 1, False),
    ],
)
def test_post_weight_storage_layout_paths_preserve_named_sources(
    domain, op_type, parameter_slot, protected
):
    model = _model(np.arange(48, dtype=np.float32).reshape(2, 8, 3))
    source = model.graph.initializers["weight"]

    def unexpected_read():
        raise AssertionError("eligibility and layout folding must not read weight payloads")

    tensor = ir.LazyTensor(unexpected_read, dtype=source.dtype, shape=source.shape)
    source.const_value = tensor
    # Only the consumer's parameter position matters to this structural test;
    # complete model/schema and public save/load contracts are covered by the
    # existing QMoE/FP8/streaming model tests, unchanged by this regression fix.
    inputs = []
    for index in range(parameter_slot):
        argument = ir.Value(
            name=f"argument_{index}",
            type=ir.TensorType(ir.DataType.FLOAT),
            shape=ir.Shape([1]),
        )
        model.graph.inputs.append(argument)
        inputs.append(argument)
    inputs.append(model.graph.outputs[0])
    consumer = ir.Node(domain, op_type, inputs, num_outputs=1)
    consumer.outputs[0].name = "consumer_output"
    model.graph.append(consumer)
    model.graph.outputs.append(consumer.outputs[0])
    fold_initializers_after_weights(model)
    if protected:
        assert model.graph.initializers["weight"] is source
        assert source.const_value is tensor
        assert sum(node.op_type == "Reshape" for node in model.graph) == 2
        assert sum(node.op_type == "Transpose" for node in model.graph) == 1
    else:
        assert "weight" not in model.graph.initializers
        assert not any(node.op_type in ("Reshape", "Transpose") for node in model.graph)


@pytest.mark.parametrize("with_dependency", [False, True])
def test_qdq_scalar_scale_storage_survives_pipeline_and_package_roundtrip(
    tmp_path, with_dependency
):
    data = np.array([[0.5]], dtype=np.float32)
    scale = _initializer("checkpoint_scale", data)
    reads = []

    def read_scale():
        reads.append(True)
        return ir.tensor(data)

    tensor = ir.LazyTensor(read_scale, dtype=scale.dtype, shape=scale.shape)
    scale.const_value = tensor
    scalar_shape = _initializer("scalar_shape", np.array([], dtype=np.int64))
    reshape = ir.Node("", "Reshape", [scale, scalar_shape], num_outputs=1)
    reshape.outputs[0].name = "runtime_scalar_scale"
    reshape.outputs[0].dtype = ir.DataType.FLOAT
    reshape.outputs[0].shape = ir.Shape([])
    codes = ir.Value(name="codes", type=ir.TensorType(ir.DataType.UINT8), shape=ir.Shape([4]))
    inputs = [codes]
    nodes = [reshape]
    runtime_scale = reshape.outputs[0]
    if with_dependency:
        dependency = ir.Value(
            name="dependency", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([])
        )
        inputs.append(dependency)
        add = ir.Node("", "Add", [runtime_scale, dependency], num_outputs=1)
        add.outputs[0].name = "dependent_scale"
        add.outputs[0].dtype = ir.DataType.FLOAT
        add.outputs[0].shape = ir.Shape([])
        nodes.append(add)
        runtime_scale = add.outputs[0]
    dequantize = ir.Node("", "DequantizeLinear", [codes, runtime_scale], num_outputs=1)
    dequantize.outputs[0].name = "output"
    dequantize.outputs[0].dtype = ir.DataType.FLOAT
    dequantize.outputs[0].shape = ir.Shape([4])
    nodes.append(dequantize)
    model = ir.Model(
        ir.Graph(
            inputs,
            [dequantize.outputs[0]],
            nodes=nodes,
            initializers=[scale, scalar_shape],
            opset_imports={"": 21},
        ),
        ir_version=10,
    )
    fold_initializers_after_weights(model)
    assert model.graph.initializers["checkpoint_scale"] is scale
    assert scale.const_value is tensor
    assert not reads
    assert reshape.graph is model.graph
    assert reshape.inputs[0] is scale
    # The protected source stays in checkpoint orientation, not the scalar
    # runtime format. This is a storage contract, not just numerical parity.
    assert scale.shape == ir.Shape([1, 1])
    assert scale.dtype == ir.DataType.FLOAT
    fold_initializers_after_weights(model)
    assert model.graph.initializers["checkpoint_scale"] is scale
    assert not reads

    ModelPackage({"model": model}).save(tmp_path, progress_bar=False)
    restored = ModelPackage.load(tmp_path)["model"]
    restored_scale = restored.graph.initializers["checkpoint_scale"]
    assert restored_scale.shape == scale.shape
    assert restored_scale.dtype == scale.dtype
    assert restored_scale.const_value.numpy().tobytes() == data.tobytes()
    for candidate in (model, restored):
        session = OnnxModelSession(candidate, device="cpu")
        for dependency_value in (0.0, 0.25) if with_dependency else (0.0,):
            code_data = np.array([0, 1, 127, 255], dtype=np.uint8)
            feeds: dict[str, np.ndarray] = {"codes": code_data}
            if with_dependency:
                feeds["dependency"] = np.asarray(dependency_value, dtype=np.float32)
            actual = session.run(feeds)["output"]
            np.testing.assert_array_equal(
                actual, code_data.astype(np.float32) * (0.5 + dependency_value)
            )


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
    assert not list(model.graph)
    assert set(model.graph.initializers) == {"output"}
    folded = model.graph.initializers["output"]
    assert model.graph.outputs[0] is folded
    assert isinstance(folded.const_value, ir.LazyTensor)
    expected = data.reshape(2, 2, 4, 3).swapaxes(1, 2).reshape(2, 8, 3)
    assert folded.const_value.numpy().tobytes() == expected.tobytes()


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


def test_post_weight_pipeline_concat_layouts_are_isolated_in_if_branches(tmp_path):
    branches = []
    expected_outputs = []
    for offset in (0, 100):
        data = np.arange(24, dtype=np.float32).reshape(2, 3, 4) + offset
        zeros = np.zeros((1, 4, 3), dtype=np.float32)
        # Each lexical scope deliberately uses the same names for different
        # payloads. The rank-3 Transpose is folded by FoldWeightLayoutPass,
        # which exposes an all-initializer Concat in each branch.
        weight = _initializer("W", data)
        padding = _initializer("Z", zeros)
        transpose = ir.Node(
            "", "Transpose", [weight], [ir.AttrInt64s("perm", [0, 2, 1])], num_outputs=1
        )
        transpose.outputs[0].name = "layout"
        transpose.outputs[0].shape = ir.Shape([2, 4, 3])
        transpose.outputs[0].dtype = ir.DataType.FLOAT
        concat = ir.Node(
            "",
            "Concat",
            [transpose.outputs[0], padding],
            [ir.AttrInt64("axis", 0)],
            num_outputs=1,
        )
        concat.outputs[0].name = "branch_output"
        concat.outputs[0].shape = ir.Shape([3, 4, 3])
        concat.outputs[0].dtype = ir.DataType.FLOAT
        branches.append(
            ir.Graph(
                [],
                [concat.outputs[0]],
                nodes=[transpose, concat],
                initializers=[weight, padding],
                name=f"branch_{offset}",
            )
        )
        expected_outputs.append(np.concatenate([data.transpose(0, 2, 1), zeros], axis=0))
    condition = ir.Value(
        name="condition", shape=ir.Shape([]), type=ir.TensorType(ir.DataType.BOOL)
    )
    select = ir.Node(
        "",
        "If",
        [condition],
        [
            ir.AttrGraph("then_branch", branches[0]),
            ir.AttrGraph("else_branch", branches[1]),
        ],
        num_outputs=1,
    )
    select.outputs[0].name = "result"
    select.outputs[0].shape = ir.Shape([3, 4, 3])
    select.outputs[0].dtype = ir.DataType.FLOAT
    model = ir.Model(
        ir.Graph(
            [condition],
            [select.outputs[0]],
            nodes=[select],
            opset_imports={"": 24},
            name="layout_concat_if",
        ),
        ir_version=10,
    )

    fold_initializers_after_weights(model)
    assert not model.graph.initializers
    packed_name = "layout__Z__axis_0__concat"
    packed_values = []
    for branch, expected in zip(branches, expected_outputs, strict=True):
        assert not list(branch)
        assert set(branch.initializers) == {packed_name}
        packed = branch.initializers[packed_name]
        assert branch.outputs[0] is packed
        assert isinstance(packed.const_value, ir.LazyTensor)
        np.testing.assert_array_equal(packed.const_value.numpy(), expected)
        packed_values.append(packed)
    assert packed_values[0] is not packed_values[1]
    fold_initializers_after_weights(model)
    assert not model.graph.initializers
    assert all(
        branch.initializers[packed_name] is packed
        for branch, packed in zip(branches, packed_values, strict=True)
    )

    output_directory = tmp_path / "package"
    ModelPackage({"model": model}).save(
        str(output_directory), max_workers=1, progress_bar=False
    )
    restored = ModelPackage.load(str(output_directory))["model"]
    assert not restored.graph.initializers
    restored_select = next(iter(restored.graph))
    for attribute, expected in zip(
        ("then_branch", "else_branch"), expected_outputs, strict=True
    ):
        branch = restored_select.attributes[attribute].as_graph()
        assert not list(branch)
        assert set(branch.initializers) == {packed_name}
        packed = branch.initializers[packed_name]
        assert branch.outputs[0] is packed
        np.testing.assert_array_equal(packed.const_value.numpy(), expected)

    # Both branches must execute their own payload, before and after the public
    # package roundtrip, rather than reusing a same-named sibling initializer.
    for candidate in (model, restored):
        session = OnnxModelSession(candidate, device="cpu")
        for take_then, expected in zip((True, False), expected_outputs, strict=True):
            result = session.run({"condition": np.asarray(take_then, dtype=np.bool_)})
            np.testing.assert_array_equal(result["result"], expected)


@pytest.mark.parametrize("retained", [None, "shared", "graph_output"])
def test_parent_capture_initializer_ownership_and_persisted_payload(tmp_path, retained):
    data = np.arange(6, dtype=np.float32).reshape(2, 3)
    source = _initializer("parent_weight", data)
    transpose = ir.Node(
        "", "Transpose", [source], [ir.AttrInt64s("perm", [1, 0])], num_outputs=1
    )
    transpose.outputs[0].name = "captured_layout"
    transpose.outputs[0].shape = ir.Shape([3, 2])
    transpose.outputs[0].dtype = ir.DataType.FLOAT
    identity = ir.Node("", "Identity", [transpose.outputs[0]], num_outputs=1)
    identity.outputs[0].name = "branch_output"
    identity.outputs[0].shape = ir.Shape([3, 2])
    identity.outputs[0].dtype = ir.DataType.FLOAT
    branch = ir.Graph([], [identity.outputs[0]], nodes=[transpose, identity])
    other = _initializer("other_weight", np.zeros((3, 2), dtype=np.float32))
    other_branch = ir.Graph([], [other], nodes=[], initializers=[other])
    condition = ir.Value(
        name="condition", shape=ir.Shape([]), type=ir.TensorType(ir.DataType.BOOL)
    )
    select = ir.Node(
        "",
        "If",
        [condition],
        [ir.AttrGraph("then_branch", branch), ir.AttrGraph("else_branch", other_branch)],
        num_outputs=1,
    )
    select.outputs[0].name = "output"
    select.outputs[0].shape = ir.Shape([3, 2])
    select.outputs[0].dtype = ir.DataType.FLOAT
    graph = ir.Graph(
        [condition],
        [select.outputs[0]],
        nodes=[select],
        initializers=[source],
        opset_imports={"": 24},
    )
    if retained == "shared":
        extra = ir.Node("", "Identity", [source], num_outputs=1)
        extra.outputs[0].name = "shared_source"
        extra.outputs[0].shape = source.shape
        extra.outputs[0].dtype = source.dtype
        graph.append(extra)
        graph.outputs.append(extra.outputs[0])
    elif retained == "graph_output":
        graph.outputs.append(source)
    model = ir.Model(graph, ir_version=10)
    result = FoldWeightLayoutPass()(model)
    assert result.modified == (retained is None)
    if retained is None:
        assert "parent_weight" not in graph.initializers
        assert set(branch.initializers) == {"captured_layout"}
        assert not FoldWeightLayoutPass()(model).modified
        np.testing.assert_array_equal(
            branch.initializers["captured_layout"].const_value.numpy(), data.T
        )
    else:
        assert graph.initializers["parent_weight"] is source
        assert not branch.initializers
        assert transpose.inputs[0] is source
    # Exercise the public serializer, not a protobuf encoding. The retained
    # parent case must stay shared; the unshared case must serialize only the
    # folded child weight, not both it and the dead parent payload.
    fold_initializers_after_weights(model)
    ModelPackage({"model": model}).save(str(tmp_path), max_workers=1, progress_bar=False)
    restored = ModelPackage.load(str(tmp_path))["model"]
    persisted = [value for scope in restored.graphs() for value in scope.initializers.values()]
    assert sum(value.const_value.numpy().nbytes for value in persisted) == data.nbytes * 2
    if retained is None:
        assert "parent_weight" not in restored.graph.initializers
        restored_branch = next(iter(restored.graph)).attributes["then_branch"].as_graph()
        assert set(restored_branch.initializers) == {"captured_layout"}
        np.testing.assert_array_equal(
            restored_branch.initializers["captured_layout"].const_value.numpy(), data.T
        )
    else:
        np.testing.assert_array_equal(
            restored.graph.initializers["parent_weight"].const_value.numpy(), data
        )


@pytest.mark.parametrize("width", [3, 3 * 8192])
def test_shape_constant_folding_exposes_layout_without_reading_large_weights(width):
    data = np.arange(16 * width, dtype=np.float32).reshape(2, 8, width)
    model = _model(data, first=(2, 2, 4, width), last=(2, 8, width))
    source = model.graph.initializers["weight"]
    reads = []

    def read():
        reads.append(True)
        return ir.tensor(data)

    source.const_value = ir.LazyTensor(read, dtype=source.dtype, shape=source.shape)
    shape = model.graph.initializers["shape1"]
    offset = _initializer("shape_offset", np.zeros(4, dtype=np.int64))
    model.graph.register_initializer(offset)
    expression = ir.Node("", "Add", [shape, offset], num_outputs=1)
    expression.outputs[0].name = "computed_shape"
    expression.outputs[0].shape = shape.shape
    expression.outputs[0].dtype = ir.DataType.INT64
    first = next(iter(model.graph))
    model.graph.insert_before(first, expression)
    first.replace_input_with(1, expression.outputs[0])
    fold_initializers_after_weights(model)
    assert not reads
    assert set(model.graph.initializers) == {"output"}
    assert not list(model.graph)
    folded = model.graph.initializers["output"]
    assert model.graph.outputs[0] is folded
    assert isinstance(folded.const_value, ir.LazyTensor)
    fold_initializers_after_weights(model)
    assert not reads
    assert model.graph.initializers["output"] is folded
    assert model.graph.outputs[0] is folded
    expected = data.reshape(2, 2, 4, width).transpose(0, 2, 1, 3).reshape(2, 8, width)
    np.testing.assert_array_equal(folded.const_value.numpy(), expected)
    assert len(reads) == 1


def test_folded_layout_keeps_pruned_external_source_ownership(tmp_path, monkeypatch):
    # The 48-element fixture used elsewhere is only 192 bytes: ir.save keeps
    # tensors below its 256-byte default threshold inline, so it has no file
    # backing to protect. This fixture must exercise a real ExternalTensor.
    width = 24
    data = np.arange(16 * width, dtype=np.float32).reshape(2, 8, width)
    base = tmp_path / "base"
    model = _model(data, first=(2, 2, 4, width), last=(2, 8, width))
    ModelPackage({"model": model}).save(str(base), max_workers=1, progress_bar=False)
    before = {path.name: path.read_bytes() for path in base.iterdir()}
    package = ModelPackage.load(str(base))
    model = package["model"]
    source = model.graph.initializers["weight"].const_value
    assert isinstance(source, ir.ExternalTensor)
    source_path = Path(source.path).resolve()
    assert source_path == (base / "model.onnx.data").resolve()

    def unexpected_read(*args, **kwargs):
        raise AssertionError("source-alias refusal must precede lazy payload reads")

    with monkeypatch.context() as patch:
        patch.setattr(ir.ExternalTensor, "numpy", unexpected_read)
        fold_initializers_after_weights(model)
        folded_graph = model.graph
        assert set(folded_graph.initializers) == {"output"}
        folded = folded_graph.initializers["output"]
        assert folded_graph.outputs[0] is folded
        assert isinstance(folded.const_value, ir.LazyTensor)
        assert source_path in {
            Path(path).resolve() for path in model.meta[LAZY_SOURCE_FILES_META_KEY]
        }
        assert source_path in {
            Path(path).resolve()
            for path in folded.const_value.meta[LAZY_SOURCE_FILES_META_KEY]
        }
        with pytest.raises(ValueError, match="aliases lazy source file"):
            package.save(str(base), max_workers=1, progress_bar=False)
        assert source_path in package._native_streaming_source_files
    assert before == {path.name: path.read_bytes() for path in base.iterdir()}
    safe = tmp_path / "safe"
    package.save(str(safe), max_workers=1, progress_bar=False)
    restored_graph = ModelPackage.load(str(safe))["model"].graph
    assert set(restored_graph.initializers) == {"output"}
    restored = restored_graph.initializers["output"]
    assert restored_graph.outputs[0] is restored
    expected = data.reshape(2, 2, 4, width).transpose(0, 2, 1, 3).reshape(2, 8, width)
    np.testing.assert_array_equal(restored.const_value.numpy(), expected)
