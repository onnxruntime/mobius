# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Default component function registration, closure, and precision regressions."""

from __future__ import annotations

import math

import ml_dtypes
import numpy as np
import onnx
import onnx_ir as ir
import pytest
from onnx.reference import ReferenceEvaluator
from onnx_ir.passes import common
from onnxscript._internal.builder import build_function

from mobius._build_context import build_context
from mobius._builder import _cast_module_dtype
from mobius._execution_providers import ep_registry
from mobius._flags import override_flags
from mobius._model_package import ModelPackage
from mobius._optimizations import optimize_model, validate_standard_onnx
from mobius._testing import create_test_input
from mobius.components._qwen3_vl_vision import Qwen3VLVisionAttention
from mobius.functions import register_function_bodies
from mobius.functions._component_functions import is_component_function
from mobius.models.qwen4_exp import (
    Qwen4ExpGroupedRMSNorm,
    Qwen4ExpOffsetRMSNorm,
    _Qwen4ExpDepthwiseConv1d,
    _Qwen4ExpPostGatedRMSNorm,
)
from mobius.tasks._base import _make_graph, _make_model

pytestmark = pytest.mark.opset29


def _norm_model(*, experimental=True, ep="onnx-standard"):
    with (
        override_flags(experimental_opset29=experimental),
        build_context(ep_registry.require(ep)),
    ):
        graph, builder = _make_graph()
        x = create_test_input(builder, "x", [2, 8])
        weight = create_test_input(builder, "weight", [8])

        output = builder.op.OffsetRMSNorm(x, weight, epsilon=1e-6, _domain="com.microsoft")
        output.type, output.shape = x.type, x.shape
        graph.outputs.append(output)
        model = _make_model(graph)
        register_function_bodies(model)
        common.RemoveUnusedFunctionsPass()(model)
        return model


@pytest.mark.parametrize(
    "experimental,ep",
    [(False, "default"), (True, "default"), (False, "onnx-standard"), (True, "onnx-standard")],
)
def test_default_portable_export_retains_registered_bodies(experimental, ep):
    model = _norm_model(experimental=experimental, ep=ep)
    optimize_model(model, ep=ep, dtype=ir.DataType.FLOAT)
    assert any(is_component_function(fn) for fn in model.functions.values())
    validate_standard_onnx(model)


@pytest.mark.parametrize("ep", ["cpu", "cuda", "dml", "openvino"])
def test_hardware_ep_exposes_standard_component_patterns_before_fusion(ep):
    model = _norm_model(ep=ep)
    optimize_model(model, ep=ep, dtype=ir.DataType.FLOAT)
    assert not any(is_component_function(fn) for fn in model.functions.values())
    assert all(node.op_type != "OffsetRMSNorm" for node in model.graph)


def test_shared_function_dedup_specialization_and_parameter_ownership():
    with (
        override_flags(experimental_opset29=True),
        build_context(ep_registry.require("onnx-standard")),
    ):
        graph, builder = _make_graph()
        x = create_test_input(builder, "x", [2, 8])
        for index, epsilon in enumerate([1e-6, 1e-6, 1e-5]):
            norm = Qwen4ExpOffsetRMSNorm(8, epsilon)
            norm._set_name(f"norm_{index}")
            graph.outputs.append(norm(builder.op, x))
        model = _make_model(graph)
        register_function_bodies(model)
        common.RemoveUnusedFunctionsPass()(model)
    assert set(model.graph.initializers) == {f"norm_{i}.weight" for i in range(3)}
    assert len(model.functions) == 2
    calls = list(model.graph)
    assert calls[0].op_identifier() == calls[1].op_identifier()
    assert calls[0].op_identifier() != calls[2].op_identifier()
    assert calls[0].inputs[1] is model.graph.initializers["norm_0.weight"]
    assert calls[1].inputs[1] is model.graph.initializers["norm_1.weight"]
    assert all(not fn.graph.initializers for fn in model.functions.values())
    other = _norm_model()
    assert all(
        function is not other.functions.get(identifier)
        for identifier, function in model.functions.items()
    )


def test_nested_registration_never_overwrites_caller_owned_destination():
    model = _norm_model()
    original = next(iter(model.functions.values()))
    # A wrapper is caller-owned; its nested component call must not replace an
    # existing caller definition at the specialization's destination identifier.
    wrapper = build_function(
        lambda op, x, w: op.OffsetRMSNorm(x, w, epsilon=1e-6, _domain="com.microsoft"),
        [ir.Value(name="x"), ir.Value(name="w")],
        domain="test.caller",
        name="Wrapper",
        opset_imports={"": 29, "com.microsoft": 1},
    )
    model.functions[wrapper.identifier()] = wrapper
    original.metadata_props.clear()
    functions = dict(model.functions)
    with pytest.raises(ValueError, match="conflicts with caller-owned definition"):
        register_function_bodies(model)
    assert dict(model.functions) == functions
    assert next(iter(wrapper)).overload == ""


def test_compatible_nested_registration_reuses_existing_definition(tmp_path):
    model = _norm_model()
    original = next(iter(model.functions.values()))
    wrapper = build_function(
        lambda op, x, w: op.OffsetRMSNorm(x, w, epsilon=1e-6, _domain="com.microsoft"),
        [ir.Value(name="x"), ir.Value(name="w")],
        domain="test.caller",
        name="Wrapper",
        opset_imports={"": 29, "com.microsoft": 1},
    )
    model.functions[wrapper.identifier()] = wrapper
    register_function_bodies(model)
    assert model.functions[original.identifier()] is original
    assert next(iter(wrapper)).op_identifier() == original.identifier()
    assert model.functions[wrapper.identifier()] is wrapper
    call = next(iter(model.graph))
    call.domain, call.op_type, call.overload = wrapper.identifier()
    call.attributes.clear()
    model.graph.opset_imports[wrapper.domain] = 1
    common.RemoveUnusedFunctionsPass()(model)
    ModelPackage({"model": model}).save(str(tmp_path), progress_bar=False)
    restored = ModelPackage.load(str(tmp_path))["model"]
    assert set(restored.functions) == set(model.functions)
    onnx.checker.check_model(ir.serde.serialize_model(restored), full_check=True)
    common.InlinePass()(restored)
    assert not restored.functions
    validate_standard_onnx(restored)


def test_registration_preserves_conflicting_domain_import():
    model = _norm_model()
    model.graph.opset_imports["com.microsoft"] = 2
    before = dict(model.functions)
    with pytest.raises(ValueError, match=r"requires com\.microsoft opset 1"):
        register_function_bodies(model)
    assert model.graph.opset_imports["com.microsoft"] == 2
    assert dict(model.functions) == before


def test_flash_next_fixture_skips_when_schema_is_absent(monkeypatch):
    from mobius.models.qwen4_exp_functions_test import _require_flash_next_schema

    def missing_schema(*args, **kwargs):
        raise AssertionError("get_schema must not be called for an absent schema")

    monkeypatch.setattr(onnx.defs, "has", lambda *args: False)
    monkeypatch.setattr(onnx.defs, "get_schema", missing_schema)
    with pytest.raises(pytest.skip.Exception, match="Requires opset-29 LinearAttention"):
        _require_flash_next_schema.__wrapped__()


def test_function_body_survives_package_roundtrip_and_public_inlining(tmp_path):
    model = _norm_model()
    package = ModelPackage({"model": model})
    package.save(str(tmp_path), progress_bar=False)
    restored = ModelPackage.load(str(tmp_path))["model"]
    assert set(restored.functions) == set(model.functions)
    before = dict(restored.functions)
    register_function_bodies(restored)
    assert all(restored.functions[key] is fn for key, fn in before.items())
    common.RemoveUnusedFunctionsPass()(restored)
    validate_standard_onnx(restored)
    onnx.checker.check_model(ir.serde.serialize_model(restored), full_check=True)
    feeds = {
        "x": np.arange(16, dtype=np.float32).reshape(2, 8) / 8,
        "weight": np.linspace(-0.2, 0.2, 8, dtype=np.float32),
    }
    expected = ReferenceEvaluator(ir.serde.serialize_model(restored)).run(None, feeds)
    common.InlinePass()(restored)
    assert not restored.functions
    validate_standard_onnx(restored)
    actual = ReferenceEvaluator(ir.serde.serialize_model(restored)).run(None, feeds)
    np.testing.assert_array_equal(actual[0], expected[0])


@pytest.mark.parametrize(
    "corruption", ["missing_body", "opaque_body", "recursive", "missing_import", "wrong_opset"]
)
def test_compact_function_validation_rejects_broken_closure(corruption):
    model = _norm_model()
    call = next(iter(model.graph))
    inner = model.functions[call.op_identifier()]
    if corruption == "missing_body":
        del model.functions[inner.identifier()]
    elif corruption == "opaque_body":
        next(iter(inner)).domain = "test.opaque"
    elif corruption == "recursive":
        node = next(iter(inner))
        node.domain, node.op_type, node.overload = inner.identifier()
        inner.opset_imports[node.domain] = 1
    elif corruption == "missing_import":
        del model.graph.opset_imports["com.microsoft"]
    else:
        inner.opset_imports[""] = 0
    with pytest.raises(ValueError, match="onnx-standard component"):
        validate_standard_onnx(model)


@pytest.mark.parametrize("kind", ["offset", "grouped", "sigmoid", "silu", "conv"])
@pytest.mark.parametrize(
    "dtype", [ir.DataType.FLOAT, ir.DataType.FLOAT16, ir.DataType.BFLOAT16]
)
def test_compact_component_rounding_and_state_match_inline(kind, dtype):
    rng = np.random.default_rng(42)
    numpy_dtype = ml_dtypes.bfloat16 if dtype == ir.DataType.BFLOAT16 else dtype.numpy()
    feeds = {"x": rng.normal(size=(2, 3, 8)).astype(numpy_dtype)}
    if kind == "conv":
        feeds["state"] = rng.normal(size=(2, 3, 4)).astype(numpy_dtype)
    elif kind in {"sigmoid", "silu"}:
        feeds["gate"] = rng.normal(size=(2, 3, 8)).astype(numpy_dtype)

    def build(compact):
        module = {
            "offset": lambda: Qwen4ExpOffsetRMSNorm(8, 1e-6),
            "grouped": lambda: Qwen4ExpGroupedRMSNorm(8, 4, 1e-6),
            "sigmoid": lambda: _Qwen4ExpPostGatedRMSNorm(8, 1e-6, "sigmoid"),
            "silu": lambda: _Qwen4ExpPostGatedRMSNorm(8, 1e-6, "silu"),
            "conv": lambda: _Qwen4ExpDepthwiseConv1d(3, 4, "silu"),
        }[kind]()
        _cast_module_dtype(module, dtype)
        with (
            override_flags(experimental_opset29=True),
            build_context(ep_registry.require("onnx-standard"), dtype),
        ):
            graph, builder = _make_graph()
            inputs = [
                create_test_input(builder, name, list(array.shape), dtype)
                for name, array in feeds.items()
            ]
            result = module(builder.op, *inputs)
            graph.outputs.extend([result] if isinstance(result, ir.Value) else result)
            model = _make_model(graph)
            register_function_bodies(model)
            common.RemoveUnusedFunctionsPass()(model)
            if not compact:
                common.InlinePass()(model)
        for value in model.graph.initializers.values():
            if value.const_value is not None:
                continue
            value.const_value = ir.tensor(
                np.linspace(-0.7, 0.8, math.prod(value.shape), dtype=np.float32)
                .reshape(tuple(value.shape))
                .astype(numpy_dtype)
            )
        return model

    compact = build(True)
    inline = build(False)
    assert len(compact.functions) == 1
    actual = ReferenceEvaluator(ir.serde.serialize_model(compact)).run(None, feeds)
    expected = ReferenceEvaluator(ir.serde.serialize_model(inline)).run(None, feeds)
    for lhs, rhs in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(lhs, rhs)
    if kind == "conv":
        np.testing.assert_array_equal(
            actual[1], np.concatenate([feeds["state"], feeds["x"]], axis=2)[:, :, -4:]
        )
    else:
        x = feeds["x"].astype(np.float32)
        weight = inline.graph.initializers["weight"].const_value.numpy()
        grouped = x.reshape(-1, 2, 4) if kind == "grouped" else x
        normed = grouped / np.sqrt(
            np.mean(grouped * grouped, axis=-1, keepdims=True) + np.float32(1e-6)
        )
        if kind in {"offset", "grouped"}:
            scale = weight.astype(np.float32) + 1
            scale = scale.reshape(2, 4) if kind == "grouped" else scale
            reference = (normed * scale).reshape(x.shape)
        else:
            normed = (normed.astype(numpy_dtype) * weight).astype(np.float32)
            gate = feeds["gate"].astype(np.float32)
            activated = 1 / (1 + np.exp(-gate))
            if kind == "silu":
                activated *= gate
            reference = normed * activated
        np.testing.assert_allclose(
            actual[0].astype(np.float32),
            reference.astype(numpy_dtype).astype(np.float32),
            rtol=2e-6,
            atol=2e-6,
        )


@pytest.mark.parametrize(
    "dtype", [ir.DataType.FLOAT, ir.DataType.FLOAT16, ir.DataType.BFLOAT16]
)
def test_compact_vision_attention_preserves_rotary_and_packed_segments(dtype):
    rng = np.random.default_rng(42)
    numpy_dtype = ml_dtypes.bfloat16 if dtype == ir.DataType.BFLOAT16 else dtype.numpy()
    angles = rng.normal(size=(4, 2)).astype(np.float32)
    feeds = {
        "hidden": rng.normal(size=(4, 8)).astype(numpy_dtype),
        "cos": np.cos(angles),
        "sin": np.sin(angles),
        "cu_seqlens": np.asarray([0, 2, 4], dtype=np.int64),
    }

    def build(compact):
        module = Qwen3VLVisionAttention(8, 2)
        _cast_module_dtype(module, dtype)
        with (
            override_flags(experimental_opset29=True),
            build_context(ep_registry.require("onnx-standard"), dtype),
        ):
            graph, builder = _make_graph()
            values = {
                name: create_test_input(
                    builder, name, list(array.shape), ir.tensor(array).dtype
                )
                for name, array in feeds.items()
            }
            graph.outputs.append(
                module(
                    builder.op,
                    values["hidden"],
                    values["cu_seqlens"],
                    (values["cos"], values["sin"]),
                )
            )
            model = _make_model(graph)
            register_function_bodies(model)
            common.RemoveUnusedFunctionsPass()(model)
            if not compact:
                common.InlinePass()(model)
        for value in model.graph.initializers.values():
            if value.const_value is None:
                value.const_value = ir.tensor(
                    np.linspace(-0.2, 0.3, math.prod(value.shape), dtype=np.float32)
                    .reshape(tuple(value.shape))
                    .astype(numpy_dtype)
                )
        return ReferenceEvaluator(ir.serde.serialize_model(model))

    compact, inline = build(True), build(False)
    actual, expected = compact.run(None, feeds)[0], inline.run(None, feeds)[0]
    np.testing.assert_array_equal(actual, expected)
    changed = dict(feeds)
    changed["hidden"] = feeds["hidden"].copy()
    changed["hidden"][2:] *= 19
    np.testing.assert_array_equal(compact.run(None, changed)[0][:2], actual[:2])
