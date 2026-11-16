# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Unit tests for the LinearAttention ir.Function builder."""

from __future__ import annotations

import numpy as np
import onnx
import onnx_ir as ir
import pytest
from onnxscript import GraphBuilder

from mobius._testing.ort_inference import OnnxModelSession
from mobius.functions.linear_attention import linear_attention

pytestmark = pytest.mark.opset29


@pytest.mark.parametrize("trajectory", [False, True])
def test_merged_value_head_schema_checks_and_infers(trajectory: bool) -> None:
    if "v_num_heads" not in onnx.defs.get_schema("LinearAttention", 29).attributes:
        pytest.skip("Requires ONNX containing merged onnx/onnx#8693")
    shapes = [
        [1, 3, 4],
        [1, 3, 4],
        [1, 3, 8],
        [1, 4, 2, 2],
        [1, 3, 4],
        [1, 3, 4],
    ]
    inputs = [
        ir.Value(name=name, type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape(shape))
        for name, shape in zip(
            ("query", "key", "value", "state", "decay", "beta"), shapes, strict=True
        )
    ]
    expected = [[1, 3, 8], [1, 4, 2, 2]]
    if trajectory:
        expected.append([1, 3, 4, 2, 2])
    outputs = [
        ir.Value(
            name=f"output{index}",
            type=ir.TensorType(ir.DataType.FLOAT),
            shape=ir.Shape([None] * len(shape)),
        )
        for index, shape in enumerate(expected)
    ]
    node = ir.Node(
        "",
        "LinearAttention",
        inputs,
        attributes=ir.convenience.convert_attributes(
            {
                "q_num_heads": 2,
                "kv_num_heads": 2,
                "v_num_heads": 4,
                "update_rule": "gated_delta",
            }
        ),
        outputs=outputs,
    )
    model = ir.Model(
        ir.Graph(
            inputs, outputs, nodes=[node], opset_imports={"": 29}, name="flash_next_schema"
        ),
        ir_version=10,
    )
    serialized = ir.serde.serialize_model(model)
    onnx.checker.check_model(serialized, full_check=True)
    inferred = onnx.shape_inference.infer_shapes(serialized, strict_mode=True, data_prop=True)
    restored = ir.serde.deserialize_model(inferred)
    assert [list(value.shape) for value in restored.graph.outputs] == expected


def _run_flash_next(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    past_state: np.ndarray,
    decay: np.ndarray,
    beta: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    inputs_data = (query, key, value, past_state, decay, beta)
    inputs = [
        ir.Value(
            name=name,
            shape=ir.Shape(array.shape),
            type=ir.TensorType(ir.DataType.FLOAT),
        )
        for name, array in zip(
            ("query", "key", "value", "past_state", "decay", "beta"),
            inputs_data,
            strict=True,
        )
    ]
    test_domain = "mobius.test"
    graph = ir.Graph(inputs, [], nodes=[], opset_imports={"": 24, test_domain: 1})
    op = GraphBuilder(graph).op
    outputs = op.LinearAttention(
        *inputs,
        q_num_heads=1,
        kv_num_heads=1,
        v_num_heads=2,
        update_rule="gated_delta",
        scale=1.0,
        _domain=test_domain,
        _outputs=3,
    )
    for name, output in zip(("output", "present_state", "states"), outputs, strict=True):
        output.name = name
        graph.outputs.append(output)
    model = ir.Model(graph, ir_version=10)
    function = linear_attention(
        q_num_heads=1,
        kv_num_heads=1,
        v_num_heads=2,
        update_rule="gated_delta",
        scale=1.0,
        output_states=True,
        domain=test_domain,
    )
    model.functions[function.identifier()] = function
    session = OnnxModelSession(model)
    try:
        result = session.run(
            dict(zip((value.name for value in inputs), inputs_data, strict=True))
        )
    finally:
        session.close()
    return result["output"], result["present_state"], result["states"]


def _reference(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    state: np.ndarray,
    decay: np.ndarray,
    beta: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    outputs = []
    states = []
    q = np.repeat(query.reshape(1, query.shape[1], 1, 2), 2, axis=2)
    k = np.repeat(key.reshape(1, key.shape[1], 1, 2), 2, axis=2)
    v = value.reshape(1, value.shape[1], 2, 1)
    for index in range(query.shape[1]):
        state = state * np.exp(decay[:, index, :, None, None])
        retrieved = np.einsum("bhkm,bhk->bhm", state, k[:, index])
        delta = beta[:, index, :, None] * (v[:, index] - retrieved)
        state = state + np.einsum("bhk,bhm->bhkm", k[:, index], delta)
        outputs.append(np.einsum("bhk,bhkm->bhm", q[:, index], state))
        states.append(state.copy())
    return (
        np.stack(outputs, axis=1).reshape(1, query.shape[1], 2),
        state,
        np.stack(states, axis=1),
    )


def _run_ordinary_gqa(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    past_state: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    unused = np.empty((1, query.shape[1], 0), dtype=np.float32)
    inputs_data = (query, key, value, past_state, unused, unused)
    inputs = [
        ir.Value(
            name=name,
            shape=ir.Shape(array.shape),
            type=ir.TensorType(ir.DataType.FLOAT),
        )
        for name, array in zip(
            ("query", "key", "value", "past_state", "decay", "beta"),
            inputs_data,
            strict=True,
        )
    ]
    test_domain = "mobius.test"
    graph = ir.Graph(inputs, [], nodes=[], opset_imports={"": 24, test_domain: 1})
    outputs = GraphBuilder(graph).op.LinearAttention(
        *inputs,
        q_num_heads=4,
        kv_num_heads=2,
        update_rule="linear",
        scale=1.0,
        _domain=test_domain,
        _outputs=2,
    )
    for name, output in zip(("output", "present_state"), outputs, strict=True):
        output.name = name
        graph.outputs.append(output)
    model = ir.Model(graph, ir_version=10)
    function = linear_attention(
        q_num_heads=4,
        kv_num_heads=2,
        update_rule="linear",
        scale=1.0,
        domain=test_domain,
    )
    model.functions[function.identifier()] = function
    session = OnnxModelSession(model)
    try:
        result = session.run(
            dict(zip((value.name for value in inputs), inputs_data, strict=True))
        )
    finally:
        session.close()
    return result["output"], result["present_state"]


class TestLinearAttentionSeparateQKV:
    """Tests for LinearAttention with separate Q, K, V inputs."""

    def test_function_has_six_inputs(self):
        func = linear_attention(q_num_heads=2, kv_num_heads=2, v_num_heads=4, scale=0.25)
        assert len(func.graph.inputs) == 6
        names = [inp.name for inp in func.graph.inputs]
        assert names == [
            "query",
            "key",
            "value",
            "past_state",
            "decay",
            "beta",
        ]

    def test_function_has_two_outputs(self):
        func = linear_attention(q_num_heads=2, kv_num_heads=2, v_num_heads=4, scale=0.25)
        assert len(func.graph.outputs) == 2
        names = [out.name for out in func.graph.outputs]
        assert "output" in names[0]
        assert "present_state" in names[1]

    def test_function_name_and_domain(self):
        func = linear_attention(q_num_heads=2, kv_num_heads=2, v_num_heads=4, scale=0.25)
        assert func.name == "LinearAttention"
        assert func.domain == "com.microsoft"

    def test_attributes_present(self):
        func = linear_attention(
            q_num_heads=2,
            kv_num_heads=2,
            v_num_heads=4,
            scale=0.25,
            update_rule="gated_delta",
        )
        attr_names = set(func.attributes.keys())
        assert "q_num_heads" in attr_names
        assert "kv_num_heads" in attr_names
        assert "v_num_heads" in attr_names
        assert "scale" in attr_names
        assert "update_rule" in attr_names

    def test_graph_contains_scan(self):
        func = linear_attention(q_num_heads=2, kv_num_heads=2, v_num_heads=4, scale=0.25)
        op_types = {n.op_type for n in func.graph}
        assert "Scan" in op_types

    def test_gqa_ratio_1_no_tile(self):
        """When q_num_heads == kv_num_heads, no Tile is needed."""
        func = linear_attention(q_num_heads=4, kv_num_heads=4, scale=0.25)
        op_types = [n.op_type for n in func.graph]
        assert "Tile" not in op_types

    def test_gqa_ratio_gt1_has_tile(self):
        """When kv_num_heads > q_num_heads, Tile expands heads."""
        func = linear_attention(q_num_heads=2, kv_num_heads=2, v_num_heads=8, scale=0.25)
        op_types = [n.op_type for n in func.graph]
        assert "Tile" in op_types

    def test_invalid_gqa_ratio_raises(self):
        with pytest.raises(ValueError, match="divisible"):
            linear_attention(q_num_heads=3, kv_num_heads=2, scale=0.25)
        with pytest.raises(ValueError, match="divisible"):
            linear_attention(q_num_heads=3, kv_num_heads=4, scale=0.25)

    def test_invalid_update_rule_raises(self):
        with pytest.raises(ValueError, match="Unknown update_rule"):
            linear_attention(
                q_num_heads=2,
                kv_num_heads=2,
                update_rule="invalid",
            )


class TestLinearAttentionUpdateRules:
    """Tests for different update rule variants."""

    @pytest.mark.parametrize(
        "rule",
        ["linear", "gated", "delta", "gated_delta"],
    )
    def test_valid_update_rules_build(self, rule):
        func = linear_attention(
            q_num_heads=2,
            kv_num_heads=2,
            v_num_heads=4,
            scale=0.25,
            update_rule=rule,
        )
        assert func.attributes["update_rule"].value == rule
        # All variants should produce a valid graph with Scan
        op_types = {n.op_type for n in func.graph}
        assert "Scan" in op_types

    def test_gated_delta_scan_body_has_exp(self):
        """gated_delta uses Exp for decay in the Scan body."""
        func = linear_attention(
            q_num_heads=2,
            kv_num_heads=2,
            v_num_heads=4,
            scale=0.25,
            update_rule="gated_delta",
        )
        # Find the Scan node and inspect its body
        scan_nodes = [n for n in func.graph if n.op_type == "Scan"]
        assert len(scan_nodes) == 1
        body = scan_nodes[0].attributes["body"].value
        body_ops = {n.op_type for n in body}
        assert "Exp" in body_ops  # decay uses Exp
        assert "Sub" in body_ops  # delta uses Sub(v, retrieval)

    def test_linear_scan_body_no_exp_no_sub(self):
        """Linear rule: no decay (Exp) and no beta (Sub)."""
        func = linear_attention(
            q_num_heads=2,
            kv_num_heads=2,
            v_num_heads=4,
            scale=0.25,
            update_rule="linear",
        )
        scan_nodes = [n for n in func.graph if n.op_type == "Scan"]
        assert len(scan_nodes) == 1
        body = scan_nodes[0].attributes["body"].value
        body_ops = {n.op_type for n in body}
        assert "Exp" not in body_ops
        assert "Sub" not in body_ops


class TestLinearAttentionDecayShapes:
    """Tests verifying decay shape handling (3D → 4D internally)."""

    def test_decay_reshape_present(self):
        """Function should Reshape decay from 3D to 4D."""
        func = linear_attention(q_num_heads=2, kv_num_heads=2, v_num_heads=4, scale=0.25)
        # Count Reshape ops — should have reshapes for
        # q, k, v, decay (3D→4D) and output (4D→3D)
        reshape_count = sum(1 for n in func.graph if n.op_type == "Reshape")
        assert reshape_count >= 5  # q, k, v, decay, output

    def test_beta_transpose_present(self):
        """Function should Transpose beta from (B,T,H) to (B,H,T)."""
        func = linear_attention(q_num_heads=2, kv_num_heads=2, v_num_heads=4, scale=0.25)
        transpose_count = sum(1 for n in func.graph if n.op_type == "Transpose")
        # At least: q, k, v (3D→4D), decay (3D→4D), beta,
        # q/k/v/decay/beta (4D→T-first for Scan), output (T→B first)
        assert transpose_count >= 6


def test_flash_next_prefill_decode_and_state_trajectory():
    query = np.array([[[0.2, 0.8], [0.6, -0.1], [0.3, 0.5]]], dtype=np.float32)
    key = np.array([[[0.5, 0.5], [0.8, 0.2], [0.1, 0.9]]], dtype=np.float32)
    value = np.array([[[1.0, -0.5], [0.25, 0.75], [-0.2, 0.4]]], dtype=np.float32)
    decay = np.array([[[-0.1, -0.2], [-0.3, -0.1], [-0.2, -0.4]]], dtype=np.float32)
    beta = np.array([[[0.7, 0.4], [0.5, 0.8], [0.9, 0.6]]], dtype=np.float32)
    initial = np.zeros((1, 2, 2, 1), dtype=np.float32)

    expected = _reference(query, key, value, initial.copy(), decay, beta)
    actual = _run_flash_next(query, key, value, initial, decay, beta)
    for actual_value, expected_value in zip(actual, expected, strict=True):
        np.testing.assert_allclose(actual_value, expected_value, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(actual[2][:, -1], actual[1], rtol=0, atol=0)

    prefill = _run_flash_next(
        query[:, :2],
        key[:, :2],
        value[:, :2],
        initial,
        decay[:, :2],
        beta[:, :2],
    )
    decode = _run_flash_next(
        query[:, 2:],
        key[:, 2:],
        value[:, 2:],
        prefill[1],
        decay[:, 2:],
        beta[:, 2:],
    )
    np.testing.assert_allclose(decode[0], actual[0][:, 2:], rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(decode[1], actual[1], rtol=1e-5, atol=1e-6)


def test_ordinary_gqa_q4_kv2_matches_independent_reference():
    query = np.array(
        [[[0.2, 0.8, 0.6, -0.1, 0.3, 0.5, -0.4, 0.7], [0.1] * 8]],
        dtype=np.float32,
    )
    key = np.array(
        [[[0.5, 0.5, 0.8, 0.2], [0.1, 0.9, -0.3, 0.4]]],
        dtype=np.float32,
    )
    value = np.array([[[1.0, -0.5], [0.25, 0.75]]], dtype=np.float32)
    initial = np.array([[[[0.1], [0.2]], [[-0.2], [0.3]]]], dtype=np.float32)

    state = initial.copy()
    expected_outputs = []
    q4 = query.reshape(1, 2, 4, 2)
    k4 = key.reshape(1, 2, 2, 2)
    v4 = value.reshape(1, 2, 2, 1)
    for token in range(2):
        state += np.einsum("bhk,bhm->bhkm", k4[:, token], v4[:, token])
        state_for_read = np.repeat(state, 2, axis=1)
        expected_outputs.append(np.einsum("bhk,bhkm->bhm", q4[:, token], state_for_read))
    expected_output = np.stack(expected_outputs, axis=1).reshape(1, 2, 4)

    actual_output, actual_state = _run_ordinary_gqa(query, key, value, initial)
    np.testing.assert_allclose(actual_output, expected_output, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(actual_state, state, rtol=1e-5, atol=1e-6)


def test_state_trajectory_is_opt_in():
    assert (
        len(linear_attention(q_num_heads=1, kv_num_heads=1, v_num_heads=2).graph.outputs) == 2
    )
    function = linear_attention(
        q_num_heads=1, kv_num_heads=1, v_num_heads=2, output_states=True
    )
    assert [output.name for output in function.graph.outputs] == [
        "output",
        "present_state",
        "states",
    ]
