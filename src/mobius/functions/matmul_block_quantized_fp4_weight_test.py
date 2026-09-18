# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for the inlineable Microsoft NVFP4 weight-only MatMul function."""

from __future__ import annotations

import ml_dtypes
import numpy as np
import onnx_ir as ir
import onnxruntime as ort
from onnx_ir.passes.common import InlinePass

from mobius.functions import (
    get_function,
    matmul_block_quantized_fp4_weight,
    register_function_bodies,
)
from mobius.functions.matmul_block_quantized_fp4_weight import (
    _E2M1_TABLE,
    _E4M3_TABLE,
)

_OP_ID = ("com.microsoft", "MatMulBlockQuantizedFp4Weight", "")


def _const(name: str, value: np.ndarray) -> ir.Value:
    tensor = ir.tensor(value, name=name)
    return ir.Value(
        name=name,
        shape=tensor.shape,
        type=ir.TensorType(tensor.dtype),
        const_value=tensor,
    )


def _build_model(
    packed: np.ndarray,
    raw_scales: np.ndarray,
    global_scale: np.ndarray,
) -> ir.Model:
    k = packed.shape[1] * 2
    activation = ir.Value(
        name="A",
        shape=ir.Shape([2, k]),
        type=ir.TensorType(ir.DataType.FLOAT),
    )
    packed_value = _const("B", packed)
    scale_value = _const("weight_scale", raw_scales)
    global_value = _const("weight_scale_2", global_scale)
    output = ir.Value(
        name="Y",
        shape=ir.Shape([2, packed.shape[0]]),
        type=ir.TensorType(ir.DataType.FLOAT),
    )
    node = ir.Node(
        "com.microsoft",
        "MatMulBlockQuantizedFp4Weight",
        inputs=[activation, packed_value, scale_value, global_value],
        outputs=[output],
        attributes=ir.convenience.convert_attributes({"block_size": 16}),
    )
    graph = ir.Graph(
        inputs=[activation],
        outputs=[output],
        nodes=[node],
        initializers=[packed_value, scale_value, global_value],
        opset_imports={"": 24, "com.microsoft": 1},
        name="nvfp4",
    )
    return ir.Model(graph, ir_version=11)


def _run(model: ir.Model, activation: np.ndarray, tmp_path) -> np.ndarray:
    path = tmp_path / "model.onnx"
    ir.save(model, path)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return session.run(None, {"A": activation})[0]


def test_e2m1_table_preserves_low_high_code_semantics():
    assert _E2M1_TABLE.tolist() == [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ]
    assert np.signbit(_E2M1_TABLE[8])
    assert np.isfinite(_E2M1_TABLE).all()


def test_e4m3_lookup_preserves_every_raw_code():
    decoded = np.arange(256, dtype=np.uint8).view(ml_dtypes.float8_e4m3fn)
    finite = np.isfinite(decoded)

    np.testing.assert_array_equal(_E4M3_TABLE[finite], decoded[finite].astype(np.float32))
    np.testing.assert_array_equal(np.isnan(_E4M3_TABLE), np.isnan(decoded))


def test_function_identity_and_registry_freshness():
    function = matmul_block_quantized_fp4_weight()
    first = get_function(_OP_ID)
    second = get_function(_OP_ID)

    assert function.identifier() == _OP_ID
    assert len(function.inputs) == 4
    assert function.attributes["block_size"].value == 16
    assert first is not None and second is not None and first is not second


def test_registration_is_lazy_for_unrelated_models():
    output = ir.Value(name="Y")
    model = ir.Model(
        ir.Graph(
            [],
            [output],
            nodes=[ir.Node("", "Constant", [], outputs=[output])],
            opset_imports={"": 24},
            name="unrelated",
        ),
        ir_version=11,
    )

    register_function_bodies(model)

    assert _OP_ID not in model.functions


def test_inlined_function_matches_explicit_nvfp4_reconstruction(tmp_path):
    codes = np.array(
        [
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
            [15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
        ],
        dtype=np.uint8,
    )
    packed = codes[:, 0::2] | (codes[:, 1::2] << 4)
    block_scales = np.array([[0.5], [2.0]], dtype=ml_dtypes.float8_e4m3fn)
    raw_scales = block_scales.view(np.uint8)
    global_scale = np.array([0.25], dtype=np.float32)
    activation = np.arange(32, dtype=np.float32).reshape(2, 16) / 8
    model = _build_model(packed, raw_scales, global_scale)
    register_function_bodies(model)

    InlinePass(
        criteria=lambda function: function.identifier() == _OP_ID,
    )(model)

    assert all(node.op_type != _OP_ID[1] for node in model.graph)
    actual = _run(model, activation, tmp_path)
    weight = _E2M1_TABLE[codes] * block_scales.astype(np.float32) * global_scale
    expected = activation @ weight.T
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
