# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for the SelectiveScan (S6) component."""

from __future__ import annotations

import onnx_ir as ir

from mobius._testing import (
    count_op_type,
    create_test_builder,
    create_test_input,
)
from mobius.components._ssm import SelectiveScan, SequenceSelectiveScan


class TestSelectiveScan:
    """Tests for SelectiveScan graph construction."""

    def test_parameters_created(self):
        """All expected parameters are created."""
        ssm = SelectiveScan(d_inner=64, d_state=16, dt_rank=4)
        params = {p.name for p in ssm.parameters()}
        # x_proj (Linear, no bias), dt_proj (Linear, with bias), A_log, D
        assert any("weight" in n for n in params)  # x_proj.weight
        assert ssm.A_log is not None
        assert ssm.D is not None

    def test_parameter_shapes(self):
        """Parameter shapes match the specified dimensions."""
        ssm = SelectiveScan(d_inner=64, d_state=16, dt_rank=4)
        assert list(ssm.A_log.shape) == [64, 16]
        assert list(ssm.D.shape) == [64]
        # x_proj: (dt_rank + 2*d_state, d_inner) = (36, 64)
        assert list(ssm.x_proj.weight.shape) == [36, 64]
        # dt_proj: (d_inner, dt_rank) = (64, 4)
        assert list(ssm.dt_proj.weight.shape) == [64, 4]

    def test_forward_builds_graph(self):
        """Forward pass constructs a valid ONNX graph."""
        ssm = SelectiveScan(d_inner=64, d_state=16, dt_rank=4)
        test_builder, op, _graph = create_test_builder()
        x = create_test_input(test_builder, "x", [2, 1, 64])
        state = create_test_input(test_builder, "ssm_state", [2, 64, 16])

        y, new_state = ssm(op, x, state)

        assert y is not None
        assert new_state is not None

    def test_discretization_ops_present(self):
        """Graph contains Exp ops for A discretization."""
        ssm = SelectiveScan(d_inner=32, d_state=8, dt_rank=2)
        test_builder, op, graph = create_test_builder()
        x = create_test_input(test_builder, "x", [1, 1, 32])
        state = create_test_input(test_builder, "ssm_state", [1, 32, 8])

        ssm(op, x, state)

        # Exp is used for both A discretization and softplus
        assert count_op_type(graph, "Exp") >= 1
        # Softplus for dt
        assert count_op_type(graph, "Softplus") >= 1
        # Split for dt/B/C from x_proj output
        assert count_op_type(graph, "Split") >= 1

    def test_skip_connection_ops(self):
        """Graph contains Add for skip connection (D * x)."""
        ssm = SelectiveScan(d_inner=32, d_state=8, dt_rank=2)
        test_builder, op, graph = create_test_builder()
        x = create_test_input(test_builder, "x", [1, 1, 32])
        state = create_test_input(test_builder, "ssm_state", [1, 32, 8])

        ssm(op, x, state)

        # Multiple Add ops: state update + skip connection
        assert count_op_type(graph, "Add") >= 2

    def test_different_d_state(self):
        """Works with different state dimensions."""
        for d_state in (4, 16, 64):
            ssm = SelectiveScan(d_inner=32, d_state=d_state, dt_rank=2)
            assert list(ssm.A_log.shape) == [32, d_state]
            # x_proj output size: dt_rank + 2*d_state
            assert list(ssm.x_proj.weight.shape) == [
                2 + 2 * d_state,
                32,
            ]

    def test_state_input_dtype(self):
        """State input accepts FLOAT dtype."""
        ssm = SelectiveScan(d_inner=16, d_state=4, dt_rank=2)
        test_builder, op, _graph = create_test_builder()
        x = create_test_input(test_builder, "x", [1, 1, 16], dtype=ir.DataType.FLOAT)
        state = create_test_input(
            test_builder, "ssm_state", [1, 16, 4], dtype=ir.DataType.FLOAT
        )

        y, _new_state = ssm(op, x, state)
        assert y is not None


class TestSequenceSelectiveScan:
    """Tests for the full-sequence selective scan."""

    def test_shares_parameters_with_decode_variant(self):
        """Parameter names and shapes match the single-token SelectiveScan."""
        decode = SelectiveScan(d_inner=64, d_state=16, dt_rank=4)
        sequence = SequenceSelectiveScan(d_inner=64, d_state=16, dt_rank=4)

        def spec(module):
            return {name: list(p.shape) for name, p in module.named_parameters()}

        assert spec(sequence) == spec(decode)

    def test_forward_builds_graph(self):
        """Forward pass constructs a graph with no carried state."""
        ssm = SequenceSelectiveScan(d_inner=32, d_state=8, dt_rank=2)
        test_builder, op, graph = create_test_builder()
        x = create_test_input(test_builder, "x", [2, 5, 32])

        y = ssm(op, x)

        assert y is not None
        # The recurrence is expressed as a single Scan over the sequence.
        assert count_op_type(graph, "Scan") == 1

    def test_scan_body_carries_state_and_decay(self):
        """The Scan body takes 2 carries + 4 per-step inputs and emits 3 values."""
        ssm = SequenceSelectiveScan(d_inner=16, d_state=4, dt_rank=2)
        test_builder, op, graph = create_test_builder()
        x = create_test_input(test_builder, "x", [1, 3, 16])

        ssm(op, x)

        scan = next(node for node in graph if node.op_type == "Scan")
        assert scan.attributes["num_scan_inputs"].value == 4
        body = scan.attributes["body"].value
        # ssm_state + a_neg carries, then dt/B/C/x scan inputs.
        assert len(body.inputs) == 6
        # ssm_state_out + a_neg_out carries, then the per-step output.
        assert len(body.outputs) == 3

    def test_scan_iterates_over_the_sequence_axis(self):
        """Sequence axis is 1 on both the scan inputs and the scan output."""
        ssm = SequenceSelectiveScan(d_inner=16, d_state=4, dt_rank=2)
        test_builder, op, graph = create_test_builder()
        x = create_test_input(test_builder, "x", [1, 3, 16])

        ssm(op, x)

        scan = next(node for node in graph if node.op_type == "Scan")
        assert list(scan.attributes["scan_input_axes"].value) == [1, 1, 1, 1]
        assert list(scan.attributes["scan_output_axes"].value) == [1]

    def test_recurrence_runs_in_float32(self):
        """The scan body stays in float32 even for a float16 activation."""
        ssm = SequenceSelectiveScan(d_inner=16, d_state=4, dt_rank=2)
        test_builder, op, graph = create_test_builder()
        x = create_test_input(test_builder, "x", [1, 3, 16], ir.DataType.FLOAT16)

        ssm(op, x)

        scan = next(node for node in graph if node.op_type == "Scan")
        for inp in scan.inputs:
            producer = inp.producer()
            if producer is not None and producer.op_type == "Cast":
                assert producer.attributes["to"].value == ir.DataType.FLOAT
            else:
                # ConstantOfShape / Neg / Softplus already operate on float32.
                assert inp.dtype in (ir.DataType.FLOAT, None)

        # The scan result is handed back in the activation's dtype.
        assert count_op_type(graph, "CastLike") >= 1
