# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""IR-only construction, independent scalar reference, and native QMoE parity."""

from __future__ import annotations

import itertools

import numpy as np
import onnx
import onnx_ir as ir
import pytest

from mobius._constants import OPSET_VERSION
from mobius._flags import override_flags
from mobius.functions import register_function_bodies
from mobius.functions._lowering_test_utils import (
    expand_grouped_matmul as _expand_grouped_matmul,
)
from mobius.functions._moe_test_utils import inline_native_calls

ort = pytest.importorskip("onnxruntime")
_requires_opset29 = pytest.mark.skipif(
    not onnx.defs.has("GroupedMatMul", 29),
    reason="Official ONNX opset-29 GroupedMatMul schema is required (ONNX >=1.24)",
)
_requires_native_cpu_qmoe = pytest.mark.skipif(
    not any(
        kernel.op_name == "QMoE" and kernel.provider == "CPUExecutionProvider"
        for kernel in ort.capi._pybind_state.get_all_opkernel_def()
    ),
    reason="ORT CPU QMoE kernel is required for native parity",
)


@pytest.fixture(autouse=True)
def _enable_experimental_opset29():
    with override_flags(experimental_opset29=True):
        yield


def _value(name, array):
    tensor = ir.tensor(array)
    return ir.Value(
        name=name,
        shape=ir.Shape(array.shape),
        type=ir.TensorType(tensor.dtype),
        const_value=tensor,
    )


def _pack(codes):
    if codes.shape[-1] % 2:
        codes = np.pad(codes, [(0, 0)] * (codes.ndim - 1) + [(0, 1)])
    return codes[..., ::2] | (codes[..., 1::2] << 4)


def _model(
    *,
    rank=2,
    k=2,
    zero_points=True,
    router_weights=True,
    normalize=1,
    clipped=False,
    dtype=ir.DataType.FLOAT,
    opset=29,
):
    rng = np.random.default_rng(42)
    experts, hidden, inter, block = 3, 32, 48, 16  # odd down-projection block count
    tokens = "tokens"
    shape: list[str | int] = [tokens, hidden] if rank == 2 else ["batch", "sequence", hidden]
    x = ir.Value(name="x", shape=ir.Shape(shape), type=ir.TensorType(dtype))
    router = ir.Value(
        name="router",
        shape=ir.Shape([tokens, experts]),
        type=ir.TensorType(dtype),
    )
    rw = ir.Value(name="routing_weights", shape=router.shape, type=ir.TensorType(dtype))
    inputs = [
        x,
        router,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        rw if router_weights else None,
    ]
    initializers: list[ir.Value] = []
    for wi, si, zi, n, width in ((2, 3, 11, 2 * inter, hidden), (5, 6, 12, hidden, inter)):
        codes = rng.integers(0, 16, (experts, n, width), dtype=np.uint8)
        scales = rng.uniform(0.005, 0.04, (experts, n, width // block)).astype(np.float32)
        if dtype == ir.DataType.FLOAT16:
            scales = scales.astype(np.float16)
        zps = rng.integers(2, 14, scales.shape, dtype=np.uint8)
        for index, array in ((wi, _pack(codes)), (si, scales)):
            value = _value(f"original_{index}", array)
            inputs[index] = value
            initializers.append(value)
        if zero_points:
            value = _value(f"original_{zi}", _pack(zps))
            inputs[zi] = value
            initializers.append(value)
    attrs = {
        "k": k,
        "expert_weight_bits": 4,
        "block_size": block,
        "quant_type": "int",
        "weights_prepacked": 0,
        "activation_type": "swiglu",
        "swiglu_fusion": 1,
        "normalize_routing_weights": normalize,
    }
    if clipped:
        attrs.update(activation_alpha=1.7, activation_beta=0.3, swiglu_limit=0.2)
    y = ir.Value(name="y", shape=x.shape, type=x.type)
    node = ir.Node(
        "com.microsoft",
        "QMoE",
        inputs=inputs,
        outputs=[y],
        attributes=ir.convenience.convert_attributes(attrs),
    )
    graph_inputs = [x, router] + ([rw] if router_weights else [])
    graph = ir.Graph(
        inputs=graph_inputs,
        outputs=[y],
        nodes=[node],
        initializers=initializers,
        opset_imports={"": opset, "com.microsoft": 1},
        name="qmoe",
    )
    return ir.Model(graph, ir_version=10)


def _feeds(*, rank=2, count=4, router_weights=True, case="normal", dtype=np.float32):
    rng = np.random.default_rng(53 + count)
    hidden_shape = (count, 32) if rank == 2 else (2, count, 32)
    x = rng.standard_normal(hidden_shape).astype(np.float32)
    tokens = x.size // 32
    logits = rng.standard_normal((tokens, 3)).astype(np.float32)
    weights = rng.uniform(0.1, 0.8, (tokens, 3)).astype(np.float32)
    if case == "ties":
        logits[:] = 1
    if case == "zero":
        weights[:] = 0
    if case == "negative":
        weights[:] = [-0.5, 0.2, 1e-10]
    if case == "cancellation":
        logits[:] = [3, 2, 1]
        weights[:] = [0.5, -0.5, 0]
    if case == "masked":
        logits[:, 0] = -np.inf
    feeds = {"x": x.astype(dtype), "router": logits.astype(dtype)}
    if router_weights:
        feeds["routing_weights"] = weights.astype(dtype)
    return feeds


def _reference(model, feeds):
    """Scalar expert-loop reference: decode each code by byte index and shift.

    Intentionally does not use the graph's nibble interleaving, GroupedMatMul,
    tensor reshape/deinterleave algorithm, or production lowering helpers.
    """
    node = next(iter(model.graph))
    attrs = node.attributes

    def weight(wi, si, zi):
        packed = node.inputs[wi].const_value.numpy()
        scales = node.inputs[si].const_value.numpy()
        zp_value = node.inputs[zi]
        zp = None if zp_value is None else zp_value.const_value.numpy()
        result = np.empty((*packed.shape[:-1], packed.shape[-1] * 2), dtype=np.float64)
        for expert, row, column in np.ndindex(result.shape):
            block = column // attrs.get_int("block_size")
            code = (int(packed[expert, row, column // 2]) >> (4 * (column % 2))) & 15
            zero = (
                8
                if zp is None
                else (int(zp[expert, row, block // 2]) >> (4 * (block % 2))) & 15
            )
            result[expert, row, column] = (code - zero) * float(scales[expert, row, block])
        return result

    fc1, fc2 = weight(2, 3, 11), weight(5, 6, 12)
    x = feeds["x"].reshape(-1, 32).astype(np.float64)
    result = np.zeros_like(x)
    limit = attrs.get_float("swiglu_limit", np.inf)
    alpha = attrs.get_float("activation_alpha", 1.0)
    beta = attrs.get_float("activation_beta", 0.0)
    for token, hidden in enumerate(x):
        logits = feeds["router"][token]
        selected = sorted(range(3), key=lambda expert: (logits[expert], expert), reverse=True)
        selected = selected[: attrs.get_int("k")]
        if "routing_weights" in feeds:
            weights = feeds["routing_weights"][token, selected].astype(np.float64)
            if attrs.get_int("normalize_routing_weights"):
                total = weights.sum()
                weights = weights / total if total != 0 else np.zeros_like(weights)
        else:
            weights = np.exp(logits[selected].astype(np.float64) - max(logits[selected]))
            weights /= weights.sum()
        for expert, route_weight in zip(selected, weights, strict=True):
            if route_weight <= 1e-8:
                continue
            middle = np.empty(48, dtype=np.float64)
            for row in range(48):
                gate = min(np.dot(fc1[expert, 2 * row], hidden), limit)
                up = np.clip(np.dot(fc1[expert, 2 * row + 1], hidden), -limit, limit)
                middle[row] = gate / (1 + np.exp(-alpha * gate)) * (up + beta)
            result[token] += route_weight * (fc2[expert] @ middle)
    return result.reshape(feeds["x"].shape).astype(feeds["x"].dtype)


def _run(model, feeds, path):
    ir.save(model, path)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
    return session.run(None, feeds)[0]


@pytest.mark.parametrize(
    "rank,k,zero_points,router_weights,normalize,clipped",
    list(
        itertools.product((2, 3), (1, 2), (False, True), (False, True), (0, 1), (False, True))
    ),
)
@pytest.mark.parametrize("dtype", [ir.DataType.FLOAT, ir.DataType.FLOAT16])
@_requires_opset29
def test_numerical_reference(
    tmp_path, rank, k, zero_points, router_weights, normalize, clipped, dtype
):
    model = _model(
        rank=rank,
        k=k,
        zero_points=zero_points,
        router_weights=router_weights,
        normalize=normalize,
        clipped=clipped,
        dtype=dtype,
    )
    numpy_dtype = np.float16 if dtype == ir.DataType.FLOAT16 else np.float32
    feeds = _feeds(rank=rank, router_weights=router_weights, dtype=numpy_dtype)
    expected = _reference(model, feeds)
    names = tuple(model.graph.initializers)
    assert inline_native_calls(model) is model
    assert model.graph.opset_imports[""] == 29
    assert tuple(model.graph.initializers) == names
    assert sum(n.op_type == "GroupedMatMul" for n in model.graph) == 2
    dq = [n for n in model.graph if n.op_type == "DequantizeLinear"]
    assert len(dq) == 2
    assert all(n.attributes.get_int("axis") == 2 for n in dq)
    ir.save(model, tmp_path / "standard.onnx")
    loaded = ir.load(tmp_path / "standard.onnx")
    expanded = _expand_grouped_matmul(loaded)
    actual = _run(expanded, feeds, tmp_path / "executable.onnx")
    assert actual.dtype == expected.dtype
    if dtype == ir.DataType.FLOAT16:
        np.testing.assert_allclose(actual, expected, rtol=2e-3, atol=2e-4)
    else:
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("shape", [(0, 32), (2, 0, 32), (0, 3, 32), (0, 0, 32)])
@pytest.mark.parametrize("dtype", [ir.DataType.FLOAT, ir.DataType.FLOAT16])
@_requires_opset29
def test_empty_token_dimensions_survive_roundtrip(tmp_path, shape, dtype):
    model = inline_native_calls(_model(rank=len(shape), dtype=dtype))
    path = tmp_path / "empty.onnx"
    ir.save(model, path)
    onnx.checker.check_model(str(path), full_check=True)
    executable = _expand_grouped_matmul(ir.load(path))
    numpy_dtype = np.float16 if dtype == ir.DataType.FLOAT16 else np.float32
    feeds = _feeds(count=0, dtype=numpy_dtype)
    feeds["x"] = np.empty(shape, dtype=numpy_dtype)

    actual = _run(executable, feeds, tmp_path / "empty_executable.onnx")

    assert actual.shape == shape
    assert actual.dtype == numpy_dtype
    assert actual.size == 0


@pytest.mark.parametrize("case", ["ties", "zero", "negative", "cancellation", "masked"])
@pytest.mark.parametrize("normalize", [0, 1])
@_requires_native_cpu_qmoe
@_requires_opset29
def test_routing_boundaries_and_dynamic_tokens(tmp_path, case, normalize):
    native = _model(normalize=normalize)
    lowered = inline_native_calls(native.clone())
    expanded = _expand_grouped_matmul(lowered)
    for count in (1, 5):
        feeds = _feeds(count=count, case=case)
        actual = _run(expanded, feeds, tmp_path / f"dynamic_{count}.onnx")
        np.testing.assert_allclose(actual, _reference(native, feeds), rtol=2e-5, atol=2e-6)
        native24 = _model(normalize=normalize, opset=24)
        native_result = _run(native24, feeds, tmp_path / f"native_dynamic_{count}.onnx")
        np.testing.assert_allclose(actual, native_result, rtol=3e-5, atol=3e-6)


@pytest.mark.parametrize(
    "router_weights,normalize,clipped",
    list(itertools.product((False, True), (0, 1), (False, True))),
)
@pytest.mark.parametrize("dtype", [ir.DataType.FLOAT, ir.DataType.FLOAT16])
@_requires_native_cpu_qmoe
@_requires_opset29
def test_native_cpu_parity(tmp_path, router_weights, normalize, clipped, dtype):
    native = _model(
        router_weights=router_weights, normalize=normalize, clipped=clipped, dtype=dtype
    )
    numpy_dtype = np.float16 if dtype == ir.DataType.FLOAT16 else np.float32
    feeds = _feeds(router_weights=router_weights, dtype=numpy_dtype)
    rtol, atol = (2e-3, 2e-4) if dtype == ir.DataType.FLOAT16 else (3e-5, 3e-6)
    native24 = _model(
        router_weights=router_weights,
        normalize=normalize,
        clipped=clipped,
        dtype=dtype,
        opset=24,
    )
    expected = _run(native24, feeds, tmp_path / "native.onnx")
    np.testing.assert_allclose(expected, _reference(native, feeds), rtol=rtol, atol=atol)
    actual = _run(
        _expand_grouped_matmul(inline_native_calls(native.clone())),
        feeds,
        tmp_path / "lowered.onnx",
    )
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)


@pytest.mark.parametrize(
    "attributes,message",
    [
        ({"quant_type": "fp4"}, "quant_type"),
        ({"weights_prepacked": 1}, "weights_prepacked"),
        ({"weights_prepacked": -1}, "weights_prepacked"),
        ({"accuracy_level": 4}, "accuracy_level"),
        ({"expert_weight_bits": 2}, "INT4"),
        ({"fc2_expert_weight_bits": 8}, "INT4"),
        ({"fc1_expert_weight_bits": 0}, "INT4"),
        ({"activation_type": "gelu"}, "activation_type"),
        ({"swiglu_fusion": 2}, "swiglu_fusion"),
        ({"use_sparse_mixer": 1}, "sparse_mixer"),
        ({"swiglu_limit_after_activation": 1}, "after_activation"),
        ({"k": 0}, "k must"),
        ({"k": 4}, "k must"),
        ({"block_size": 24}, "power of two"),
        ({"block_size": 64}, "divisible"),
        ({"normalize_routing_weights": 2}, "normalize_routing_weights"),
        ({"swiglu_limit": -1.0}, "nonnegative"),
        ({"activation_alpha": float("nan")}, "finite"),
        ({"unknown": 1}, "unsupported QMoE attributes"),
        ({"k": "two"}, "literal INT"),
    ],
)
def test_reject_attributes_before_any_mutation(tmp_path, attributes, message):
    model = _model()
    first = next(iter(model.graph))
    second = ir.Node(
        "com.microsoft",
        "QMoE",
        inputs=first.inputs,
        outputs=[ir.Value(name="second_output")],
        attributes=first.attributes,
    )
    for attr in ir.convenience.convert_attributes(attributes):
        second.attributes[attr.name] = attr
    model.graph.append(second)
    before = tmp_path / "before.onnx"
    after = tmp_path / "after.onnx"
    ir.save(model, before)
    with pytest.raises(ValueError, match=message):
        inline_native_calls(model)
    ir.save(model, after)
    assert before.read_bytes() == after.read_bytes()


@pytest.mark.parametrize("index", [4, 7, 8, 9, 10, 13, 15])
def test_reject_unsupported_optional_inputs(index):
    model = _model()
    node = next(iter(model.graph))
    if index == 15:
        # Reconstruct the node to add an extended optional input.
        replacement = ir.Node(
            node.domain,
            node.op_type,
            inputs=[*node.inputs, node.inputs[0]],
            num_outputs=1,
            attributes=node.attributes.values(),
        )
        model.graph.append(replacement)
    else:
        node.replace_input_with(index, node.inputs[0])
    with pytest.raises(ValueError, match=f"unsupported QMoE input {index}"):
        inline_native_calls(model)
    assert model.graph.opset_imports[""] == 29


@pytest.mark.parametrize(
    "index,shape,dtype,message",
    [
        (0, [2, 2, 2, 32], ir.DataType.FLOAT, "rank 2 or 3"),
        (0, [2, "hidden"], ir.DataType.FLOAT, "static hidden"),
        (0, [2, 32], ir.DataType.BFLOAT16, "only FLOAT"),
        (3, [3, 96, 2], ir.DataType.FLOAT16, "same FLOAT/FLOAT16"),
        (1, [2, 4], ir.DataType.FLOAT, "router_probs"),
        (2, [3, 96, 17], ir.DataType.UINT8, "raw"),
        (3, [3, 96, 1], ir.DataType.FLOAT, "scales"),
        (6, [3, 32], ir.DataType.FLOAT, "scales"),
        (11, [3, 96, 2], ir.DataType.UINT8, "zero points"),
        (12, [3, 32, 1], ir.DataType.UINT8, "zero points"),
        (14, [2, 4], ir.DataType.FLOAT, "router_weights"),
    ],
)
def test_reject_shapes_and_types(index, shape, dtype, message):
    model = _model()
    value = next(iter(model.graph)).inputs[index]
    value.shape = ir.Shape(shape)
    value.dtype = dtype
    with pytest.raises(ValueError, match=message):
        inline_native_calls(model)
    assert model.graph.opset_imports[""] == 29


def test_lowering_requires_experimental_opset29_before_mutation(tmp_path):
    model = _model()
    register_function_bodies(model)
    before, after = tmp_path / "before.onnx", tmp_path / "after.onnx"
    ir.save(model, before)

    with (
        override_flags(experimental_opset29=False),
        pytest.raises(ValueError, match="experimental_opset29"),
    ):
        inline_native_calls(model)

    ir.save(model, after)
    assert before.read_bytes() == after.read_bytes()


def test_native_serialization_no_automatic_opset29(tmp_path):
    model = _model(opset=24)
    path = tmp_path / "native.onnx"
    ir.save(model, path)
    before = path.read_bytes()
    register_function_bodies(model)
    assert OPSET_VERSION == 24
    assert model.graph.opset_imports[""] == 24
    assert next(iter(model.graph)).op_type == "QMoE"
    assert not any(fn.name == "QMoE" for fn in model.functions.values())
    assert all(fn.opset_imports.get("", 0) <= 24 for fn in model.functions.values())
    # Registration is allowed to attach existing non-QMoE bodies; the native
    # node itself, its optional slots, attributes, and weights remain untouched.
    model.functions.clear()
    ir.save(model, path)
    assert path.read_bytes() == before


@_requires_opset29
def test_output_annotations_survive_roundtrip(tmp_path):
    model = _model()
    original = model.graph.outputs[0]
    original.shape = ir.Shape([4, 32])
    original.doc_string = "Routed expert output."
    original.metadata_props["test.qmoe.annotation"] = "preserved"
    marker = object()
    original.meta["test.qmoe.marker"] = marker

    inline_native_calls(model)

    output = model.graph.outputs[0]
    assert output.type == original.type
    assert output.shape == original.shape
    assert output.meta["test.qmoe.marker"] is marker
    path = tmp_path / "annotated.onnx"
    ir.save(model, path)
    restored = ir.load(path).graph.outputs[0]
    assert restored.name == original.name
    assert restored.type == original.type
    assert restored.shape == original.shape
    assert restored.doc_string == original.doc_string
    assert restored.metadata_props["test.qmoe.annotation"] == "preserved"


@_requires_opset29
def test_unrelated_model_noop_and_idempotence(tmp_path):
    model = _model()
    inline_native_calls(model)
    ir.save(model, tmp_path / "once.onnx")
    inline_native_calls(model)
    ir.save(model, tmp_path / "twice.onnx")
    assert (tmp_path / "once.onnx").read_bytes() == (tmp_path / "twice.onnx").read_bytes()
    graph = ir.Graph([], [], nodes=[], opset_imports={"": 29})
    unrelated = ir.Model(graph, ir_version=10)
    assert inline_native_calls(unrelated) is unrelated
    assert graph.opset_imports[""] == 29
    assert unrelated.functions


@pytest.mark.parametrize("dtype", [ir.DataType.FLOAT, ir.DataType.FLOAT16])
@_requires_opset29
def test_opset29_checker_and_shape_inference(tmp_path, dtype):
    """Use file APIs, never protobuf construction or serialization helpers."""
    model = inline_native_calls(_model(rank=3, dtype=dtype))
    path = tmp_path / "standard.onnx"
    ir.save(model, path)
    onnx.checker.check_model(str(path), full_check=True)
    inferred = tmp_path / "inferred.onnx"
    onnx.shape_inference.infer_shapes_path(str(path), str(inferred), strict_mode=True)
    loaded = ir.load(inferred)
    assert loaded.graph.outputs[0].shape == ir.Shape(["batch", "sequence", 32])
    assert loaded.graph.outputs[0].dtype == dtype


@pytest.mark.parametrize("zp1,zp2", list(itertools.product((False, True), repeat=2)))
@_requires_opset29
def test_independent_optional_zero_points_and_all_experts(tmp_path, zp1, zp2):
    native = _model(k=3, router_weights=False)
    node = next(iter(native.graph))
    for index, present in ((11, zp1), (12, zp2)):
        if not present:
            node.replace_input_with(index, None)
    feeds = _feeds(router_weights=False)
    expected = _reference(native, feeds)
    actual = _run(
        _expand_grouped_matmul(inline_native_calls(native.clone())),
        feeds,
        tmp_path / "optional.onnx",
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)


@_requires_opset29
def test_truncated_optional_slots(tmp_path):
    original = _model(zero_points=False, router_weights=False)
    feeds = _feeds(router_weights=False)
    expected = _reference(original, feeds)
    node = next(iter(original.graph))
    output = ir.Value(name="y", shape=node.outputs[0].shape, type=node.outputs[0].type)
    short = ir.Node(
        node.domain,
        node.op_type,
        inputs=node.inputs[:7],
        outputs=[output],
        attributes=node.attributes,
    )
    output_before = node.outputs[0]
    original.graph.insert_before(node, short)
    output_before.replace_all_uses_with(output, replace_graph_outputs=True)
    original.graph.remove(node, safe=True)
    model = original
    actual = _run(
        _expand_grouped_matmul(inline_native_calls(model)), feeds, tmp_path / "short.onnx"
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("index", [0, 1, 2, 3, 5, 6])
def test_missing_required_inputs_are_rejected(index):
    model = _model()
    node = next(iter(model.graph))
    node.replace_input_with(index, None)
    with pytest.raises(ValueError, match=f"required QMoE input {index}"):
        inline_native_calls(model)
    assert model.graph.opset_imports[""] == 29


def test_missing_shape_metadata_is_rejected():
    model = _model()
    next(iter(model.graph)).inputs[0].shape = None
    with pytest.raises(ValueError, match="shape metadata"):
        inline_native_calls(model)


def test_prepacked_default_is_not_assumed_raw():
    model = _model()
    del next(iter(model.graph)).attributes["weights_prepacked"]
    with pytest.raises(ValueError, match="weights_prepacked must be 0"):
        inline_native_calls(model)


@_requires_opset29
def test_name_collisions_are_avoided(tmp_path):
    model = _model()
    model.graph.inputs[0].name = "qmoe29_0_0"
    inline_native_calls(model)
    names = (
        [v.name for n in model.graph for v in n.outputs]
        + [v.name for v in model.graph.inputs]
        + list(model.graph.initializers)
    )
    assert len(set(names)) == len(names)
    path = tmp_path / "collision.onnx"
    ir.save(model, path)
    onnx.checker.check_model(str(path))


@_requires_opset29
def test_function_scope_registration_uses_existing29(tmp_path):
    graph = _model().graph
    graph.inputs.extend(graph.initializers.values())
    graph.initializers.clear()
    function = ir.Function("test.qmoe", "Experts", graph=graph, attributes=[])
    outer = ir.Graph([], [], nodes=[], opset_imports={"": 29, "test.qmoe": 1}, name="outer")
    model = ir.Model(outer, ir_version=10, functions=[function])
    inline_native_calls(model)
    assert model.graph.opset_imports[""] == 29
    assert function.opset_imports[""] == 29
    path = tmp_path / "function.onnx"
    ir.save(model, path)
    loaded = ir.load(path)
    assert loaded.graph.opset_imports[""] == 29
    assert loaded.functions[function.identifier()].opset_imports[""] == 29
    onnx.checker.check_model(str(path))


def _caller_function(op_type="Identity", *, opset=24):
    x = ir.Value(
        name="caller_x", shape=ir.Shape([2, 3]), type=ir.TensorType(ir.DataType.FLOAT)
    )
    y = ir.Value(name="caller_y", shape=x.shape, type=x.type)
    inputs = [x]
    node = ir.Node("", op_type, inputs=inputs, outputs=[y])
    graph = ir.Graph(
        inputs, [y], nodes=[node], opset_imports={"": opset}, name="caller_function"
    )
    return ir.Function("caller", "PreserveMe", graph=graph, attributes=[])


@_requires_opset29
def test_unrelated_function_import_is_not_audited(tmp_path):
    model = _model()
    register_function_bodies(model)
    function = _caller_function("Softmax", opset=11)
    model.functions[function.identifier()] = function
    qmoe = next(iter(model.graph))
    assert qmoe.inputs[0] is not None
    uses_before = tuple(qmoe.inputs[0].uses())
    before, after = tmp_path / "before.onnx", tmp_path / "after.onnx"
    ir.save(model, before)
    original_imports = dict(function.opset_imports)
    register_function_bodies(model, moe_fallback=True)
    ir.save(model, after)
    assert dict(function.opset_imports) == original_imports
    assert model.functions[function.identifier()] is function
    assert tuple(qmoe.inputs[0].uses()) == uses_before


@_requires_opset29
def test_caller_function_preserved_without_relabeling(tmp_path):
    model = _model()
    function = _caller_function(opset=29)
    model.functions[function.identifier()] = function
    caller_input = ir.Value(
        name="caller_feed", shape=ir.Shape([2, 3]), type=ir.TensorType(ir.DataType.FLOAT)
    )
    caller_output = ir.Value(
        name="caller_result", shape=caller_input.shape, type=caller_input.type
    )
    call = ir.Node(
        function.domain, function.name, inputs=[caller_input], outputs=[caller_output]
    )
    model.graph.inputs.append(caller_input)
    model.graph.append(call)
    model.graph.outputs.insert(0, caller_output)
    model.graph.opset_imports[function.domain] = 1
    original_nodes = tuple(function)
    inline_native_calls(model)
    assert model.functions[function.identifier()] is function
    assert tuple(function) == original_nodes
    assert function.opset_imports[""] == 29
    assert model.graph.opset_imports[""] == 29
    path = tmp_path / "compatible.onnx"
    ir.save(model, path)
    onnx.checker.check_model(str(path), full_check=True)
    feeds = _feeds()
    feeds["caller_feed"] = np.arange(6, dtype=np.float32).reshape(2, 3)
    actual = _run(
        _expand_grouped_matmul(model.clone()), feeds, tmp_path / "compatible_executable.onnx"
    )
    np.testing.assert_array_equal(actual, feeds["caller_feed"])


@pytest.mark.parametrize("enabled", [False, True])
def test_no_qmoe_does_not_align_or_reject_caller_functions(tmp_path, enabled):
    function = _caller_function("Softmax", opset=11)
    graph = ir.Graph(
        [], [], nodes=[], name="no_qmoe", opset_imports={"": 29 if enabled else 24}
    )
    model = ir.Model(graph, ir_version=10, functions=[function])
    before, after = tmp_path / "before.onnx", tmp_path / "after.onnx"
    ir.save(model, before)
    with override_flags(experimental_opset29=enabled):
        register_function_bodies(model, moe_fallback=enabled)
    ir.save(model, after)
    assert model.functions[function.identifier()] is function
    assert len(model.functions) > 1
    assert function.opset_imports[""] == 11


def _family_model(op_type):
    if op_type == "MoE":
        from mobius.functions.moe_test import _model as float_model

        return float_model(), inline_native_calls
    return _model(), inline_native_calls


def _assert_atomic_error(model, lower, tmp_path, message):
    before, after = tmp_path / "before.onnx", tmp_path / "after.onnx"
    nodes = tuple(model.graph.all_nodes())
    uses = [(value, tuple(value.uses())) for value in model.graph.inputs]
    ir.save(model, before)
    with pytest.raises(ValueError, match=message):
        lower(model)
    ir.save(model, after)
    assert before.read_bytes() == after.read_bytes()
    assert tuple(model.graph.all_nodes()) == nodes
    assert all(tuple(value.uses()) == previous for value, previous in uses)


@pytest.mark.parametrize(
    "dtype,shape",
    [
        (ir.DataType.FLOAT16, [4, 32]),
        (ir.DataType.FLOAT, [5, 32]),
        (ir.DataType.FLOAT, [4, 33]),
        (ir.DataType.FLOAT, [4, 1, 32]),
    ],
)
def test_qmoe_contradictory_output_metadata_is_atomic(tmp_path, dtype, shape):
    model = _model()
    first = next(iter(model.graph))
    first.inputs[0].shape = ir.Shape([4, 32])
    model.graph.append(
        ir.Node(
            first.domain,
            first.op_type,
            inputs=first.inputs,
            attributes=first.attributes,
            outputs=[
                ir.Value(name="bad_output", shape=ir.Shape(shape), type=ir.TensorType(dtype))
            ],
        )
    )
    _assert_atomic_error(model, inline_native_calls, tmp_path, "output metadata")


@_requires_opset29
@pytest.mark.parametrize(
    "input_shape,output_shape,typed",
    [
        (["tokens", 32], None, False),
        (["tokens", 32], ["result_tokens", 32], True),
        (["tokens", 32], [4, 32], True),
        ([4, 32], ["result_tokens", 32], True),
    ],
)
def test_qmoe_compatible_output_annotations(tmp_path, input_shape, output_shape, typed):
    model = _model()
    x, output = model.graph.inputs[0], model.graph.outputs[0]
    x.shape = ir.Shape(input_shape)
    output.shape = None if output_shape is None else ir.Shape(output_shape)
    output.type = x.type if typed else None
    output.doc_string = "Compatible QMoE annotation."
    output.metadata_props["annotation"] = "preserved"
    marker = object()
    output.meta["marker"] = marker
    feeds = _feeds()
    expected = _reference(model, feeds)
    inline_native_calls(model)
    result = model.graph.outputs[0]
    assert result.shape == (x.shape if output_shape is None else ir.Shape(output_shape))
    assert result.type == x.type and result.meta["marker"] is marker
    path = tmp_path / "compatible_output.onnx"
    ir.save(model, path)
    onnx.checker.check_model(str(path), full_check=True)
    restored = ir.load(path)
    assert restored.graph.outputs[0].doc_string == output.doc_string
    assert restored.graph.outputs[0].metadata_props == output.metadata_props
    actual = _run(_expand_grouped_matmul(restored), feeds, tmp_path / "expanded.onnx")
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)


@_requires_opset29
@pytest.mark.parametrize(
    "router_tokens,weight_tokens",
    [
        ("selection_tokens", "weight_tokens"),
        (4, "weight_tokens"),
        ("selection_tokens", 4),
    ],
)
def test_router_weight_token_annotations_are_not_runtime_mismatches(
    tmp_path, router_tokens, weight_tokens
):
    model = _model()
    node = next(iter(model.graph))
    node.inputs[1].shape = ir.Shape([router_tokens, 3])
    node.inputs[14].shape = ir.Shape([weight_tokens, 3])
    feeds = _feeds()
    expected = _reference(model, feeds)
    inline_native_calls(model)
    path = tmp_path / "refined_router.onnx"
    ir.save(model, path)
    onnx.checker.check_model(str(path), full_check=True)
    actual = _run(_expand_grouped_matmul(ir.load(path)), feeds, tmp_path / "expanded.onnx")
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("static_input", [False, True])
def test_static_router_weight_token_mismatch_is_atomic(tmp_path, static_input):
    model = _model()
    node = next(iter(model.graph))
    if static_input:
        node.inputs[0].shape = ir.Shape([4, 32])
    else:
        node.inputs[1].shape = ir.Shape([4, 3])
    node.inputs[14].shape = ir.Shape([5, 3])
    _assert_atomic_error(model, inline_native_calls, tmp_path, "router_weights")


@pytest.mark.parametrize("op_type", ["MoE", "QMoE"])
def test_schema_absent_flag_noop_and_abi_validation(tmp_path, op_type):
    # Remove the actual schema, not just a capability predicate. This also runs
    # unchanged on released ONNX installations that never registered schema 29.
    schema = (
        onnx.defs.get_schema("GroupedMatMul", 29)
        if onnx.defs.has("GroupedMatMul", 29)
        else None
    )
    if schema is not None:
        onnx.defs.deregister_schema("GroupedMatMul", 29, "")
    try:
        assert not onnx.defs.has("GroupedMatMul", 29)
        model, lower = _family_model(op_type)
        with override_flags(experimental_opset29=False):
            _assert_atomic_error(model, lower, tmp_path, "experimental")
        next(iter(model.graph)).attributes["k"] = ir.AttrInt64("k", 0)
        _assert_atomic_error(model, lower, tmp_path, "k must")
        model, lower = _family_model(op_type)
        _assert_atomic_error(model, lower, tmp_path, "opset-29 GroupedMatMul schemas")
        # Preserve old unaudited bodies on no-target.
        function = _caller_function("Softmax", opset=11)
        unrelated = ir.Model(
            ir.Graph([], [], nodes=[], opset_imports={"": 29}),
            functions=[function],
            ir_version=10,
        )
        before, after = tmp_path / "noop_before.onnx", tmp_path / "noop_after.onnx"
        ir.save(unrelated, before)
        for enabled in (False, True):
            with override_flags(experimental_opset29=enabled):
                register_function_bodies(unrelated, moe_fallback=enabled)
            ir.save(unrelated, after)
            assert unrelated.functions[function.identifier()] is function
            assert function.opset_imports[""] == 11
    finally:
        if schema is not None:
            onnx.defs.register_schema(schema)


@_requires_opset29
def test_nested_qmoe_updates_containing_imports(tmp_path):
    then_graph = _model().graph
    else_graph = _model().graph
    outer_inputs = list(then_graph.inputs)
    for old, replacement in zip(else_graph.inputs, outer_inputs, strict=True):
        old.replace_all_uses_with(replacement)
    then_graph.inputs.clear()
    else_graph.inputs.clear()
    condition = ir.Value(
        name="condition", shape=ir.Shape([]), type=ir.TensorType(ir.DataType.BOOL)
    )
    result = ir.Value(
        name="result", shape=then_graph.outputs[0].shape, type=then_graph.outputs[0].type
    )
    branch = ir.Node(
        "",
        "If",
        inputs=[condition],
        outputs=[result],
        attributes=[
            ir.AttrGraph("then_branch", then_graph),
            ir.AttrGraph("else_branch", else_graph),
        ],
    )
    model = ir.Model(
        ir.Graph(
            [condition, *outer_inputs],
            [result],
            nodes=[branch],
            opset_imports={"": 29, "com.microsoft": 1},
            name="outer",
        ),
        ir_version=10,
    )
    inline_native_calls(model)
    assert model.graph.opset_imports[""] == 29
    assert then_graph.opset_imports[""] == 29
    assert else_graph.opset_imports[""] == 29
    assert not any(n.op_type == "QMoE" for n in model.graph.all_nodes())
    path = tmp_path / "nested.onnx"
    ir.save(model, path)
    onnx.checker.check_model(str(path), full_check=True)


@pytest.mark.parametrize("rank", [2, 3])
@pytest.mark.parametrize("dtype", [ir.DataType.FLOAT, ir.DataType.FLOAT16])
@_requires_opset29
def test_direct_grouped_matmul29_runtime(tmp_path, monkeypatch, rank, dtype):
    monkeypatch.setenv("ALLOW_RELEASED_ONNX_OPSET_ONLY", "0")
    native = _model(rank=rank, dtype=dtype)
    numpy_dtype = np.float16 if dtype == ir.DataType.FLOAT16 else np.float32
    feeds = _feeds(rank=rank, dtype=numpy_dtype)
    expected = _reference(native, feeds)
    model = inline_native_calls(native)
    path = tmp_path / "direct29.onnx"
    ir.save(model, path)
    onnx.checker.check_model(str(path), full_check=True)
    try:
        actual = _run(model, feeds, path)
    except Exception as error:
        message = str(error)
        if (
            "No Op registered for GroupedMatMul" in message
            or ("No Op registered for" in message and "domain_version of 29" in message)
            or "Could not find an implementation for GroupedMatMul" in message
            or ("support for domain ai.onnx is till opset" in message and "29" in message)
        ):
            pytest.skip(f"ORT {ort.__version__} lacks opset-29 runtime support: {message}")
        raise
    rtol, atol = (2e-3, 2e-4) if dtype == ir.DataType.FLOAT16 else (2e-5, 2e-6)
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)


@pytest.mark.parametrize("symmetric", [False, True])
@_requires_native_cpu_qmoe
@_requires_opset29
def test_real_mobius_moe_layer_lowering(tmp_path, symmetric):
    from onnx_shape_inference import infer_symbolic_shapes

    from mobius._build_context import build_context
    from mobius._configs import QuantizationConfig
    from mobius._execution_providers import ep_registry
    from mobius._optimizations import optimize_model
    from mobius._testing import create_test_input, make_config
    from mobius.components import MoELayer
    from mobius.tasks._base import _make_graph

    config = make_config(
        hidden_size=64,
        intermediate_size=32,
        moe_intermediate_size=32,
        num_local_experts=3,
        num_experts_per_tok=2,
        quantization=QuantizationConfig(
            bits=4, group_size=32, quant_method="olive", sym=symmetric
        ),
    )

    def build(ep):
        with build_context(ep_registry.require(ep)):
            graph, builder = _make_graph()
            hidden = create_test_input(builder, "hidden", [2, "sequence", 64])
            output = MoELayer(config)(builder.op, hidden)
            builder._adapt_outputs([output], "")
            graph.outputs.append(output)
        model = ir.Model(graph, ir_version=10)
        rng = np.random.default_rng(108)
        for value in graph.initializers.values():
            shape = tuple(value.shape)
            if value.dtype == ir.DataType.UINT8:
                array = rng.integers(0, 256, shape, dtype=np.uint8)
            elif "scales" in value.name:
                array = rng.uniform(0.005, 0.03, shape).astype(np.float32)
            else:
                array = rng.normal(0, 0.1, shape).astype(np.float32)
            value.const_value = ir.tensor(array)
        infer_symbolic_shapes(model)
        return model

    model = build("onnx-standard")
    native = build("cpu")
    graph = model.graph
    assert graph.opset_imports[""] == 29
    assert native.graph.opset_imports[""] == 24
    register_function_bodies(model)
    qmoe = next(n for n in graph if n.op_type == "QMoE")
    assert qmoe.attributes.get_string("activation_type") == "swiglu"
    assert qmoe.attributes.get_int("swiglu_fusion") == 1
    original_names = tuple(graph.initializers)
    register_function_bodies(model, moe_fallback=True)
    assert tuple(graph.initializers) == original_names
    optimize_model(model, ep="onnx-standard")
    assert not any(n.op_type == "QMoE" for n in graph)
    assert graph.opset_imports[""] == 29
    assert all(fn.name not in {"MoE", "QMoE"} for fn in model.functions.values())
    assert all(function.opset_imports[""] == 29 for function in model.functions.values())
    path = tmp_path / "real_registered29.onnx"
    ir.save(model, path)
    onnx.checker.check_model(str(path), full_check=True)
    executable = _expand_grouped_matmul(model)
    rng = np.random.default_rng(109)
    for count in (1, 4):
        feeds = {"hidden": rng.normal(size=(2, count, 64)).astype(np.float32)}
        expected = _run(native, feeds, tmp_path / f"real_native_{count}.onnx")
        actual = _run(executable, feeds, tmp_path / f"real_lowered_{count}.onnx")
        np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=3e-6)
