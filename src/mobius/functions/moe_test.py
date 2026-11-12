# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Native MoE CPU parity, independent scalar reference, and atomic IR contracts."""

from __future__ import annotations

import numpy as np
import onnx
import onnx_ir as ir
import pytest

from mobius._constants import OPSET_VERSION
from mobius._flags import override_flags
from mobius.functions import get_function, register_function_bodies
from mobius.functions._lowering_test_utils import expand_grouped_matmul
from mobius.functions._moe_test_utils import inline_native_calls
from mobius.functions.moe import MoEFunctionConfig, moe

ort = pytest.importorskip("onnxruntime")
_requires_opset29 = pytest.mark.skipif(
    not onnx.defs.has("GroupedMatMul", 29),
    reason="Official ONNX opset-29 GroupedMatMul schema is required (ONNX >=1.24)",
)


@pytest.mark.parametrize(
    "changes,message",
    [
        ({"activation": "geglu"}, "activation_type"),
        ({"experts": 0}, "dimensions"),
        ({"hidden": -1}, "dimensions"),
        ({"intermediate": 1.5}, "dimensions"),
        ({"k": 0}, "k must"),
        ({"k": 4}, "k must"),
        ({"alpha": float("inf")}, "finite"),
        ({"beta": float("nan")}, "finite"),
        ({"alpha": 1e100}, "finite"),
        ({"limit": float("nan")}, "nonnegative"),
        ({"limit": -0.1}, "nonnegative"),
        ({"limit": -1e-100}, "nonnegative"),
        ({"dtype": ir.DataType.BFLOAT16}, "FLOAT"),
    ],
)
def test_direct_factory_rejects_invalid_config_before_build(changes, message):
    kwargs = dict(
        experts=3,
        hidden=4,
        intermediate=3,
        k=2,
        activation="swiglu",
        normalize=True,
        alpha=1.0,
        beta=0.0,
        limit=float("inf"),
    )
    kwargs.update(changes)
    with pytest.raises(ValueError, match=message):
        moe(MoEFunctionConfig(**kwargs))


def _require_native_cpu(dtype):
    # ORT can silently insert FLOAT casts around half MoE without a half kernel.
    # Such execution cannot qualify native HALF rounding (and old FLOAT kernels
    # have a MayInplace alias bug on the inserted internal cast buffer).
    tensor_type = f"tensor({dtype.name.lower()})"
    if not any(
        kernel.op_name == "MoE"
        and kernel.provider == "CPUExecutionProvider"
        and tensor_type in kernel.type_constraints.get("T", ())
        for kernel in ort.capi._pybind_state.get_all_opkernel_def()
    ):
        pytest.skip(
            f"ORT {ort.__version__} has no native CPU MoE kernel for {tensor_type}; "
            "automatic FLOAT fallback does not qualify native dtype arithmetic"
        )


@pytest.fixture(autouse=True)
def _enable_opset29():
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


def _model(
    *,
    activation="swiglu",
    dtype=ir.DataType.FLOAT,
    rank=2,
    k=2,
    normalize=0,
    biases=(True, True),
    clipped=False,
    hidden=4,
    inter=3,
    opset=29,
):
    rng = np.random.default_rng(42)
    shape = ["tokens", hidden] if rank == 2 else ["batch", "sequence", hidden]
    x = ir.Value(name="x", shape=ir.Shape(shape), type=ir.TensorType(dtype))
    router = ir.Value(name="router", shape=ir.Shape(["tokens", 3]), type=ir.TensorType(dtype))
    numpy_dtype = np.float16 if dtype == ir.DataType.FLOAT16 else np.float32
    width = inter * (2 if activation == "swiglu" else 1)
    arrays = [
        rng.normal(0, 0.6, (3, width, hidden)),
        rng.normal(0, 0.4, (3, width)),
        rng.normal(0, 0.6, (3, hidden, inter)),
        rng.normal(0, 0.4, (3, hidden)),
    ]
    values = [_value(f"parameter_{i}", a.astype(numpy_dtype)) for i, a in enumerate(arrays)]
    inputs = [
        x,
        router,
        values[0],
        values[1] if biases[0] else None,
        values[2],
        values[3] if biases[1] else None,
    ]
    attrs = {"k": k, "activation_type": activation, "normalize_routing_weights": normalize}
    if activation == "swiglu":
        attrs["swiglu_fusion"] = 1
    if clipped:
        attrs.update(activation_alpha=1.7, activation_beta=0.3, swiglu_limit=0.2)
    y = ir.Value(name="y", shape=x.shape, type=x.type)
    node = ir.Node(
        "com.microsoft",
        "MoE",
        inputs=inputs,
        outputs=[y],
        attributes=ir.convenience.convert_attributes(attrs),
    )
    return ir.Model(
        ir.Graph(
            [x, router],
            [y],
            nodes=[node],
            initializers=[v for v in inputs[2:] if v is not None],
            opset_imports={"": opset, "com.microsoft": 1},
            name="moe",
        ),
        ir_version=10,
    )


def _feeds(*, shape=(5, 4), dtype=np.float32, case="normal"):
    rng = np.random.default_rng(53)
    x = rng.normal(size=shape).astype(dtype)
    tokens = x.size // shape[-1]
    logits = rng.normal(size=(tokens, 3)).astype(dtype)
    if case == "ties":
        logits[:] = 0
    elif case == "rounded_ties":
        logits[:] = [0, 1e-8, 0]
    elif case == "tiny":
        logits[:] = [0, -25, -24]
    elif case == "extreme":
        logits[:] = [1000, -1000, 999]
    elif case == "masked":
        logits[:] = [-np.inf, 1, 0]
    elif case == "underflow":
        logits[:] = [0, -200, -200]
    return {"x": x, "router": logits}


def _reference(model, feeds, *, round_stages=True):
    """Scalar expert-loop reference, without GMM, reshape, or production helpers."""
    node = next(iter(model.graph))
    attrs = node.attributes
    half = feeds["x"].dtype == np.float16
    hidden = feeds["x"].shape[-1]

    def store(value):
        if half and round_stages:
            return np.asarray(value, dtype=np.float16).astype(np.float32)
        return np.asarray(value, dtype=np.float32)

    def parameter(index):
        value = node.inputs[index]
        return None if value is None else value.const_value.numpy().astype(np.float32)

    w1, b1, w2, b2 = (parameter(i) for i in (2, 3, 4, 5))
    activation = attrs.get_string("activation_type")
    result = np.zeros((feeds["x"].size // hidden, hidden), dtype=np.float32)
    for token, state in enumerate(feeds["x"].reshape(-1, hidden).astype(np.float32)):
        logits = feeds["router"][token].astype(np.float32)
        probs = np.exp(logits - max(logits))
        probs *= np.float32(1) / probs.sum(dtype=np.float32)
        selected = sorted(range(3), key=lambda i: (probs[i], i), reverse=True)
        selected = selected[: attrs.get_int("k")]
        routing = probs[selected].copy()
        if attrs.get_int("normalize_routing_weights", 0):
            routing *= np.float32(1) / routing.sum(dtype=np.float32)
        for expert, weight in zip(selected, routing, strict=True):
            if weight <= 0:
                continue
            first = store([np.dot(row, state) for row in w1[expert]])
            if b1 is not None:
                first = store(first + b1[expert])
            if activation == "swiglu":
                activated = []
                for i in range(w2.shape[-1]):
                    gate = min(first[2 * i], attrs.get_float("swiglu_limit", np.inf))
                    up = np.clip(
                        first[2 * i + 1],
                        -attrs.get_float("swiglu_limit", np.inf),
                        attrs.get_float("swiglu_limit", np.inf),
                    )
                    arg = attrs.get_float("activation_alpha", 1.0) * gate
                    sigmoid = 1 / (1 + np.exp(-arg))
                    activated.append(
                        gate * sigmoid * (up + attrs.get_float("activation_beta", 0.0))
                    )
            elif activation == "relu":
                activated = [max(v, 0) for v in first]
            elif activation == "gelu":
                activated = [
                    0.5 * v * (1 + np.tanh(0.7978845608 * (v + 0.044715 * v * v * v)))
                    for v in first
                ]
            elif activation == "silu":
                activated = [v / (1 + np.exp(-v)) for v in first]
            else:
                activated = first
            middle = store(activated)
            down = store([np.dot(row, middle) for row in w2[expert]])
            if b2 is not None:
                down = store(down + b2[expert])
            result[token] += weight * down
    return result.reshape(feeds["x"].shape).astype(feeds["x"].dtype)


def _run(model, feeds, path):
    ir.save(model, path)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
    return session.run(None, feeds)[0]


def _assert_close(actual, expected):
    assert actual.dtype == expected.dtype
    rtol, atol = (2e-3, 5e-4) if expected.dtype == np.float16 else (3e-5, 3e-6)
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)


@pytest.mark.parametrize("activation", ["relu", "gelu", "silu", "identity", "swiglu"])
@pytest.mark.parametrize("dtype", [ir.DataType.FLOAT, ir.DataType.FLOAT16])
@_requires_opset29
def test_scalar_reference_and_schema_roundtrip(tmp_path, activation, dtype):
    native = _model(activation=activation, dtype=dtype, rank=3, clipped=activation == "swiglu")
    feeds = _feeds(
        shape=(2, 3, 4), dtype=np.float16 if dtype == ir.DataType.FLOAT16 else np.float32
    )
    expected = _reference(native, feeds)
    initializers = tuple(native.graph.initializers)
    model = inline_native_calls(native.clone())
    assert tuple(model.graph.initializers) == initializers
    assert sum(n.op_type == "GroupedMatMul" for n in model.graph) == 2
    assert model.graph.opset_imports[""] == 29
    path = tmp_path / "standard.onnx"
    ir.save(model, path)
    onnx.checker.check_model(str(path), full_check=True)
    inferred = tmp_path / "inferred.onnx"
    onnx.shape_inference.infer_shapes_path(str(path), str(inferred), strict_mode=True)
    restored = ir.load(inferred)
    assert restored.graph.outputs[0].shape == native.graph.outputs[0].shape
    assert restored.graph.outputs[0].dtype == dtype
    actual = _run(expand_grouped_matmul(ir.load(path)), feeds, tmp_path / "expanded.onnx")
    _assert_close(actual, expected)


def _compare_native_cpu(tmp_path, activation, dtype, normalize):
    _require_native_cpu(dtype)
    native = _model(
        activation=activation,
        dtype=dtype,
        normalize=normalize,
        clipped=activation == "swiglu",
    )
    feeds = _feeds(dtype=np.float16 if dtype == ir.DataType.FLOAT16 else np.float32)
    native24 = _model(
        activation=activation,
        dtype=dtype,
        normalize=normalize,
        clipped=activation == "swiglu",
        opset=24,
    )
    expected = _run(native24, feeds, tmp_path / "native.onnx")
    _assert_close(expected, _reference(native, feeds))
    actual = _run(
        expand_grouped_matmul(inline_native_calls(native.clone())),
        feeds,
        tmp_path / "expanded.onnx",
    )
    _assert_close(actual, expected)


@pytest.mark.parametrize("activation", ["relu", "gelu", "silu", "identity", "swiglu"])
@pytest.mark.parametrize("dtype", [ir.DataType.FLOAT, ir.DataType.FLOAT16])
@_requires_opset29
def test_native_cpu_parity(tmp_path, activation, dtype):
    _compare_native_cpu(tmp_path, activation, dtype, normalize=0)


@_requires_opset29
def test_normalized_float_native_cpu_parity(tmp_path):
    _compare_native_cpu(tmp_path, "swiglu", ir.DataType.FLOAT, normalize=1)


@pytest.mark.parametrize(
    "biases", [(False, False), (False, True), (True, False), (True, True)]
)
@pytest.mark.parametrize("k,normalize", [(1, 0), (1, 1), (3, 0), (3, 1)])
@_requires_opset29
def test_optional_biases_routing_modes_and_all_experts(tmp_path, biases, k, normalize):
    native = _model(biases=biases, k=k, normalize=normalize, activation="identity")
    feeds = _feeds()
    actual = _run(
        expand_grouped_matmul(inline_native_calls(native.clone())),
        feeds,
        tmp_path / "expanded.onnx",
    )
    _assert_close(actual, _reference(native, feeds))


@pytest.mark.parametrize("shape", [(0, 4), (2, 0, 4), (0, 3, 4), (0, 0, 4)])
@pytest.mark.parametrize("dtype", [ir.DataType.FLOAT, ir.DataType.FLOAT16])
@_requires_opset29
def test_empty_tokens(tmp_path, shape, dtype):
    model = inline_native_calls(_model(rank=len(shape), dtype=dtype))
    path = tmp_path / "empty.onnx"
    ir.save(model, path)
    onnx.checker.check_model(str(path), full_check=True)
    feeds = _feeds(
        shape=shape, dtype=np.float16 if dtype == ir.DataType.FLOAT16 else np.float32
    )
    actual = _run(expand_grouped_matmul(ir.load(path)), feeds, tmp_path / "expanded.onnx")
    assert actual.shape == shape and actual.dtype == feeds["x"].dtype and actual.size == 0


@pytest.mark.parametrize(
    "case", ["ties", "rounded_ties", "tiny", "extreme", "masked", "underflow"]
)
@pytest.mark.parametrize("normalize", [0, 1])
@_requires_opset29
def test_routing_boundaries_and_dynamic_tokens(tmp_path, case, normalize):
    _require_native_cpu(ir.DataType.FLOAT)
    native = _model(k=1 if "ties" in case else 3, normalize=normalize)
    executable = expand_grouped_matmul(inline_native_calls(native.clone()))
    for count in (1, 7):
        feeds = _feeds(shape=(count, 4), case=case)
        actual = _run(executable, feeds, tmp_path / f"expanded_{count}.onnx")
        _assert_close(actual, _reference(native, feeds))
        native24 = _model(k=1 if "ties" in case else 3, normalize=normalize, opset=24)
        _assert_close(actual, _run(native24, feeds, tmp_path / f"native_{count}.onnx"))


@_requires_opset29
def test_higher_expert_id_on_probability_rounding_tie(tmp_path):
    native = _model(activation="identity", k=1, biases=(False, True))
    node = next(iter(native.graph))
    node.inputs[2].const_value = ir.tensor(np.zeros((3, 3, 4), np.float32))
    node.inputs[5].const_value = ir.tensor(np.array([[1] * 4, [2] * 4, [3] * 4], np.float32))
    feeds = _feeds(shape=(1, 4), case="rounded_ties")
    # Logit 1 is larger, but all FLOAT probabilities round to 1/3. CPU picks ID 2.
    actual = _run(
        expand_grouped_matmul(inline_native_calls(native)), feeds, tmp_path / "ties.onnx"
    )
    np.testing.assert_allclose(actual, np.ones((1, 4), np.float32), rtol=1e-6)


@_requires_opset29
def test_tiny_positive_routes_are_not_qmoe_thresholded(tmp_path):
    native = _model(activation="identity", k=3, biases=(False, True))
    node = next(iter(native.graph))
    node.inputs[2].const_value = ir.tensor(np.zeros((3, 3, 4), np.float32))
    node.inputs[5].const_value = ir.tensor(
        np.array([[0] * 4, [1e10] * 4, [1e10] * 4], np.float32)
    )
    feeds = _feeds(shape=(1, 4), case="tiny")
    expected = _reference(native, feeds)
    assert np.all(expected > 0.1)
    actual = _run(
        expand_grouped_matmul(inline_native_calls(native)), feeds, tmp_path / "tiny.onnx"
    )
    _assert_close(actual, expected)


@_requires_opset29
def test_inactive_overflowing_expert_is_masked(tmp_path):
    native = _model(activation="identity", k=3, biases=(False, False))
    node = next(iter(native.graph))
    w1 = node.inputs[2].const_value.numpy().copy()
    w2 = node.inputs[4].const_value.numpy().copy()
    w1[1:] = np.finfo(np.float32).max
    w2[1:] = 0  # overflow followed by 0*inf can generate NaN, but inactive
    node.inputs[2].const_value = ir.tensor(w1)
    node.inputs[4].const_value = ir.tensor(w2)
    feeds = _feeds(shape=(1, 4), case="underflow")
    feeds["x"][:] = 2
    expected = _reference(native, feeds)
    actual = _run(
        expand_grouped_matmul(inline_native_calls(native)), feeds, tmp_path / "masked.onnx"
    )
    _assert_close(actual, expected)
    assert np.all(np.isfinite(actual))


@_requires_opset29
def test_half_rounding_boundaries_are_observable(tmp_path):
    native = _model(dtype=ir.DataType.FLOAT16)
    feeds = _feeds(dtype=np.float16)
    expected = _reference(native, feeds)
    float_through = _reference(native, feeds, round_stages=False)
    assert not np.array_equal(expected, float_through)
    lowered = inline_native_calls(native)
    # GEMM1, bias1, activation, GEMM2, bias2, final output each store to half.
    stores = [
        n for n in lowered.graph if n.op_type == "Cast" and n.attributes.get_int("to") == 10
    ]
    assert len(stores) == 6
    actual = _run(expand_grouped_matmul(lowered), feeds, tmp_path / "half.onnx")
    _assert_close(actual, expected)
    assert np.max(np.abs(actual - expected)) < np.max(np.abs(actual - float_through))


def _assert_atomic(model, tmp_path, message):
    before, after = tmp_path / "before.onnx", tmp_path / "after.onnx"
    ir.save(model, before)
    nodes_before = tuple(model.graph.all_nodes())
    uses = [(v, tuple(v.uses())) for v in model.graph.inputs]
    with pytest.raises(ValueError, match=message):
        inline_native_calls(model)
    ir.save(model, after)
    assert before.read_bytes() == after.read_bytes()
    assert tuple(model.graph.all_nodes()) == nodes_before
    assert all(tuple(v.uses()) == previous for v, previous in uses)


@pytest.mark.parametrize(
    "attrs,message",
    [
        ({"activation_type": "geglu"}, "activation_type"),
        ({"activation_type": "erf"}, "activation_type"),
        ({"swiglu_fusion": 0}, "swiglu_fusion"),
        ({"swiglu_fusion": 2}, "swiglu_fusion"),
        ({"use_sparse_mixer": 1}, "sparse_mixer"),
        ({"swiglu_limit_after_activation": 1}, "after_activation"),
        ({"normalize_routing_weights": -1}, "normalize_routing_weights"),
        ({"normalize_routing_weights": 2}, "normalize_routing_weights"),
        ({"k": 0}, "k must"),
        ({"k": 4}, "k must"),
        ({"k": "2"}, "literal INT"),
        ({"activation_alpha": np.inf}, "finite"),
        ({"activation_beta": np.nan}, "finite"),
        ({"swiglu_limit": np.nan}, "nonnegative"),
        ({"swiglu_limit": -1.0}, "nonnegative"),
        ({"unexpected": 1}, "unsupported MoE attributes"),
    ],
)
def test_mixed_valid_invalid_calls_reject_atomically(tmp_path, attrs, message):
    model = _model()
    first = next(iter(model.graph))
    second = ir.Node(
        first.domain,
        first.op_type,
        inputs=first.inputs,
        outputs=[
            ir.Value(name="other", shape=first.outputs[0].shape, type=first.outputs[0].type)
        ],
        attributes=first.attributes,
    )
    for attr in ir.convenience.convert_attributes(attrs):
        second.attributes[attr.name] = attr
    model.graph.append(second)
    _assert_atomic(model, tmp_path, message)


@pytest.mark.parametrize(
    "index,shape,dtype,message",
    [
        (0, [1, 2, 3, 4], ir.DataType.FLOAT, "rank 2 or 3"),
        (0, [2, "H"], ir.DataType.FLOAT, "static hidden"),
        (0, [2, 0], ir.DataType.FLOAT, "static hidden"),
        (0, [2, 4], ir.DataType.BFLOAT16, "only FLOAT"),
        (1, [2, 4], ir.DataType.FLOAT, "router_probs"),
        (1, [2, 3], ir.DataType.FLOAT16, "same FLOAT"),
        (2, [3, 4, 6], ir.DataType.FLOAT, "modern"),
        (4, [3, 3, 4], ir.DataType.FLOAT, "modern"),
        (2, [3, 0, 4], ir.DataType.FLOAT, "positive static"),
        (4, [3, 4, "I"], ir.DataType.FLOAT, "positive static"),
        (3, [3, 3], ir.DataType.FLOAT, "bias 3"),
        (5, [3, 3], ir.DataType.FLOAT, "bias 5"),
    ],
)
def test_invalid_shapes_types(tmp_path, index, shape, dtype, message):
    model = _model()
    value = next(iter(model.graph)).inputs[index]
    value.shape, value.dtype = ir.Shape(shape), dtype
    _assert_atomic(model, tmp_path, message)


@pytest.mark.parametrize("index", [0, 1, 2, 3, 4, 5])
@pytest.mark.parametrize("missing", ["shape", "type"])
def test_missing_metadata(tmp_path, index, missing):
    model = _model()
    node = next(iter(model.graph))
    value = node.inputs[index]
    setattr(value, missing, None)
    _assert_atomic(model, tmp_path, "metadata")


@pytest.mark.parametrize("index", [0, 1, 2, 4])
def test_missing_required_input(tmp_path, index):
    model = _model()
    next(iter(model.graph)).replace_input_with(index, None)
    _assert_atomic(model, tmp_path, f"required MoE input {index}")


@_requires_opset29
def test_truncated_optional_slots(tmp_path):
    model = _model(biases=(False, False))
    feeds = _feeds()
    expected = _reference(model, feeds)
    original = next(iter(model.graph))
    output = ir.Value(
        name="short_output", shape=original.outputs[0].shape, type=original.outputs[0].type
    )
    replacement = ir.Node(
        original.domain,
        original.op_type,
        inputs=original.inputs[:5],
        outputs=[output],
        attributes=original.attributes,
    )
    model.graph.insert_before(original, replacement)
    original.outputs[0].replace_all_uses_with(output, replace_graph_outputs=True)
    model.graph.remove(original, safe=True)
    actual = _run(
        expand_grouped_matmul(inline_native_calls(model)), feeds, tmp_path / "short_slots.onnx"
    )
    _assert_close(actual, expected)


def test_reference_attributes_and_malformed_output_are_rejected(tmp_path):
    model = _model()
    node = next(iter(model.graph))
    node.attributes["k"] = ir.Attr("k", ir.AttributeType.INT, None, ref_attr_name="caller_k")
    _assert_atomic(model, tmp_path, "literal INT")
    node.attributes["k"] = ir.AttrInt64("k", 2)
    node.outputs[0].type = ir.TensorType(ir.DataType.FLOAT16)
    _assert_atomic(model, tmp_path, "output metadata")


@pytest.mark.parametrize("index", [6, 7, 8])
def test_fc3_and_extended_slots(tmp_path, index):
    model = _model()
    node = next(iter(model.graph))
    extra = list(node.inputs) + [None] * (index + 1 - len(node.inputs))
    extra[index] = node.inputs[2]
    replacement = ir.Node(
        node.domain,
        node.op_type,
        inputs=extra,
        outputs=[
            ir.Value(name="extra", shape=node.outputs[0].shape, type=node.outputs[0].type)
        ],
        attributes=node.attributes,
    )
    model.graph.append(replacement)
    _assert_atomic(model, tmp_path, "FC3|extended")


def test_token_count_output_and_overload_validation(tmp_path):
    model = _model()
    node = next(iter(model.graph))
    node.inputs[0].shape = ir.Shape([2, 4])
    node.inputs[1].shape = ir.Shape([3, 3])
    _assert_atomic(model, tmp_path, "token dimension")
    node.inputs[1].shape = ir.Shape([2, 3])
    node.outputs[0].shape = ir.Shape([3, 4])
    _assert_atomic(model, tmp_path, "output metadata")
    node.outputs[0].shape = node.inputs[0].shape
    node.overload = "custom"
    register_function_bodies(model, moe_fallback=True)
    assert node.overload == "custom"


@pytest.mark.parametrize("activation", ["identity", "swiglu"])
@_requires_opset29
def test_square_modern_weights_are_qualified(tmp_path, activation):
    native = _model(hidden=4, inter=4, activation=activation)
    feeds = _feeds()
    expected = _reference(native, feeds)
    actual = _run(
        expand_grouped_matmul(inline_native_calls(native)), feeds, tmp_path / "square.onnx"
    )
    _assert_close(actual, expected)


@pytest.mark.parametrize("limit", [0.0, np.inf])
@_requires_opset29
def test_swiglu_limit_boundaries_and_default_normalization(tmp_path, limit):
    native = _model()
    node = next(iter(native.graph))
    node.attributes["swiglu_limit"] = ir.AttrFloat32("swiglu_limit", limit)
    del node.attributes["normalize_routing_weights"]
    feeds = _feeds()
    expected = _reference(native, feeds)
    actual = _run(
        expand_grouped_matmul(inline_native_calls(native)), feeds, tmp_path / "limit.onnx"
    )
    _assert_close(actual, expected)


def test_flag_off_and_flag_alone_do_not_change_native(tmp_path):
    model = _model(opset=24)
    with override_flags(experimental_opset29=False):
        _assert_atomic(model, tmp_path, "experimental_opset29")
    ir.save(model, tmp_path / "native.onnx")
    before = (tmp_path / "native.onnx").read_bytes()
    register_function_bodies(model)
    assert OPSET_VERSION == 24 and model.graph.opset_imports[""] == 24
    assert get_function(("com.microsoft", "MoE", "")) is None
    assert not any(fn.name == "MoE" for fn in model.functions.values())
    assert next(iter(model.graph)).op_type == "MoE"
    model.functions.clear()
    ir.save(model, tmp_path / "after.onnx")
    assert (tmp_path / "after.onnx").read_bytes() == before


@pytest.mark.parametrize("enabled", [False, True])
def test_no_target_uses_normal_registry_behavior(tmp_path, enabled):
    # Unrelated caller functions are never audited or relabeled.
    x = ir.Value(name="x", shape=ir.Shape([2, 3]), type=ir.TensorType(ir.DataType.FLOAT))
    y = ir.Value(name="y", shape=x.shape, type=x.type)
    function = ir.Function(
        "caller",
        "Old",
        attributes=[],
        graph=ir.Graph(
            [x],
            [y],
            nodes=[ir.Node("", "Softmax", inputs=[x], outputs=[y])],
            opset_imports={"": 11},
        ),
    )
    version = 29 if enabled else 24
    model = ir.Model(ir.Graph([], [], nodes=[], opset_imports={"": version}), ir_version=10)
    model.functions[function.identifier()] = function
    ir.save(model, tmp_path / "before.onnx")
    with override_flags(experimental_opset29=enabled):
        register_function_bodies(model, moe_fallback=enabled)
    ir.save(model, tmp_path / "after.onnx")
    assert len(model.functions) > 1
    assert model.functions[function.identifier()] is function
    assert function.opset_imports[""] == 11
    assert model.graph.opset_imports[""] == version


@_requires_opset29
def test_output_metadata_name_collision_and_idempotence(tmp_path):
    model = _model()
    original = model.graph.outputs[0]
    original.shape = ir.Shape([5, 4])
    original.doc_string = "Native expert output."
    original.metadata_props["test.annotation"] = "preserved"
    marker = object()
    original.meta["marker"] = marker
    model.graph.inputs[0].name = "moe29_0_0"
    assert inline_native_calls(model) is model
    output = model.graph.outputs[0]
    assert output.type == original.type and output.shape == original.shape
    assert output.meta["marker"] is marker
    path = tmp_path / "annotations.onnx"
    ir.save(model, path)
    onnx.checker.check_model(str(path), full_check=True)
    restored = ir.load(path).graph.outputs[0]
    assert restored.name == original.name and restored.doc_string == original.doc_string
    assert restored.metadata_props == original.metadata_props
    before = path.read_bytes()
    inline_native_calls(model)
    ir.save(model, path)
    assert path.read_bytes() == before


@pytest.mark.parametrize("dtype", [ir.DataType.FLOAT, ir.DataType.FLOAT16])
@pytest.mark.parametrize("consumer", ["CastLike", "Identity"])
@_requires_opset29
def test_untyped_unshaped_native_intermediate(tmp_path, dtype, consumer):
    model = _model(dtype=dtype)
    feeds = _feeds(dtype=np.float16 if dtype == ir.DataType.FLOAT16 else np.float32)
    expected = _reference(model, feeds)
    intermediate = model.graph.outputs[0]
    intermediate.type = intermediate.shape = None
    intermediate.doc_string = "Native custom-op intermediate."
    intermediate.metadata_props["annotation"] = "preserved"
    x = model.graph.inputs[0]
    output = ir.Value(name="typed_output", type=x.type, shape=x.shape)
    inputs = [intermediate, x] if consumer == "CastLike" else [intermediate]
    model.graph.outputs.clear()
    model.graph.append(ir.Node("", consumer, inputs=inputs, outputs=[output]))
    model.graph.outputs.append(output)
    inline_native_calls(model)
    inferred = next(n.outputs[0] for n in model.graph if n.outputs[0].name == "y")
    assert inferred.type == x.type and inferred.shape == x.shape
    assert inferred.doc_string == intermediate.doc_string
    assert inferred.metadata_props == intermediate.metadata_props
    path = tmp_path / "untyped.onnx"
    ir.save(model, path)
    onnx.checker.check_model(str(path), full_check=True)
    restored = ir.load(path)
    actual = _run(expand_grouped_matmul(restored), feeds, tmp_path / "expanded.onnx")
    _assert_close(actual, expected)


@_requires_opset29
def test_registration_does_not_convert_unrelated_function_imports():
    model = _model()
    x = ir.Value(name="audit_x", shape=ir.Shape([2, 3]), type=ir.TensorType(ir.DataType.FLOAT))
    y = ir.Value(name="audit_y", shape=x.shape, type=x.type)
    node = ir.Node("", "Softmax", inputs=[x], outputs=[y])
    graph = ir.Graph([x], [y], nodes=[node], opset_imports={"": 11})
    function = ir.Function("caller", "Audit", graph=graph, attributes=[])
    model.functions[function.identifier()] = function
    imports = dict(graph.opset_imports)
    register_function_bodies(model, moe_fallback=True)
    assert dict(graph.opset_imports) == imports
    assert model.functions[function.identifier()] is function


@_requires_opset29
def test_function_scope_and_registry_import_alignment(tmp_path):
    inner = _model().graph
    inner.inputs.extend(inner.initializers.values())
    inner.initializers.clear()
    function = ir.Function("test.moe", "Experts", graph=inner, attributes=[])
    model = ir.Model(
        ir.Graph([], [], nodes=[], opset_imports={"": 29, "test.moe": 1}, name="outer"),
        functions=[function],
        ir_version=10,
    )
    register_function_bodies(model)
    inline_native_calls(model)
    assert model.graph.opset_imports[""] == 29
    assert all(fn.opset_imports[""] == 29 for fn in model.functions.values())
    assert not any(n.op_type == "MoE" for n in function.graph.all_nodes())
    path = tmp_path / "functions.onnx"
    ir.save(model, path)
    onnx.checker.check_model(str(path), full_check=True)


@_requires_opset29
def test_existing24_model_must_be_rebuilt(tmp_path):
    model = _model(opset=24)
    x = model.graph.inputs[0]
    output = ir.Value(name="existing", shape=x.shape, type=x.type)
    existing = ir.Node("", "Identity", inputs=[x], outputs=[output], version=24)
    model.graph.append(existing)
    _assert_atomic(model, tmp_path, "rebuild under experimental opset29")
    assert model.graph.opset_imports[""] == existing.version == 24


def test_caller_owned_moe_function_rejected(tmp_path):
    model = _model()
    graph = _model().graph
    graph.inputs.extend(graph.initializers.values())
    graph.initializers.clear()
    function = ir.Function("com.microsoft", "MoE", graph=graph, attributes=[])
    model.functions[function.identifier()] = function
    _assert_atomic(model, tmp_path, "caller-owned source")


@_requires_opset29
def test_nested_subgraphs_are_lowered_and_aligned(tmp_path):
    then_graph, else_graph = _model().graph, _model().graph
    outer_inputs = list(then_graph.inputs)
    for old, new in zip(else_graph.inputs, outer_inputs, strict=True):
        old.replace_all_uses_with(new)
    then_graph.inputs.clear()
    else_graph.inputs.clear()
    condition = ir.Value(
        name="condition", shape=ir.Shape([]), type=ir.TensorType(ir.DataType.BOOL)
    )
    output = ir.Value(
        name="output", shape=then_graph.outputs[0].shape, type=then_graph.outputs[0].type
    )
    branch = ir.Node(
        "",
        "If",
        inputs=[condition],
        outputs=[output],
        attributes=[
            ir.AttrGraph("then_branch", then_graph),
            ir.AttrGraph("else_branch", else_graph),
        ],
    )
    model = ir.Model(
        ir.Graph(
            [condition, *outer_inputs],
            [output],
            nodes=[branch],
            opset_imports={"": 29, "com.microsoft": 1},
            name="outer",
        ),
        ir_version=10,
    )
    inline_native_calls(model)
    assert not any(n.op_type == "MoE" for n in model.graph.all_nodes())
    assert all(g.opset_imports[""] == 29 for g in (model.graph, then_graph, else_graph))
    path = tmp_path / "nested.onnx"
    ir.save(model, path)
    onnx.checker.check_model(str(path), full_check=True)


@_requires_opset29
def test_mixed_moe_qmoe_calls_register_and_inline_together(tmp_path):
    from mobius.functions.qmoe_test import _model as qmodel

    model = qmodel(router_weights=False, zero_points=False)
    native = _model()
    # Keep one tiny float node alongside the existing quantized fixture.
    initializers = list(native.graph.initializers.values())
    native.graph.initializers.clear()
    for value in [
        *native.graph.inputs,
        *initializers,
        native.graph.outputs[0],
    ]:
        value.name = "float_" + value.name
    inputs = list(native.graph.inputs)
    native.graph.inputs.clear()
    model.graph.inputs.extend(inputs)
    for value in initializers:
        model.graph.initializers[value.name] = value
    original = next(iter(native.graph))
    output = ir.Value(
        name=original.outputs[0].name,
        type=original.outputs[0].type,
        shape=original.outputs[0].shape,
    )
    node = ir.Node(
        original.domain,
        original.op_type,
        inputs=original.inputs,
        outputs=[output],
        attributes=original.attributes,
    )
    native.graph.outputs.clear()
    native.graph.remove(original, safe=True)
    model.graph.append(node)
    model.graph.outputs.append(node.outputs[0])
    inline_native_calls(model)
    inline_native_calls(model)  # Repeated registration/inlining is harmless.
    assert sum(n.op_type == "GroupedMatMul" for n in model.graph) == 4
    assert not any(n.op_type in {"MoE", "QMoE"} for n in model.graph)
    path = tmp_path / "composed.onnx"
    ir.save(model, path)
    onnx.checker.check_model(str(path), full_check=True)
