# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for the MambaBlock component."""

from __future__ import annotations

import numpy as np
import onnx_ir as ir
import pytest
from onnxscript import GraphBuilder

from mobius._constants import OPSET_VERSION
from mobius._testing import (
    count_op_type,
    create_test_builder,
    create_test_input,
)
from mobius.components._mamba_block import (
    FloatSwiGLU,
    MambaBlock,
    SequenceMambaBlock,
    StatefulMambaBlock,
)


class TestMambaBlock:
    """Tests for MambaBlock graph construction."""

    def test_default_dt_rank(self):
        """dt_rank defaults to ceil(d_model / 16)."""
        block = MambaBlock(d_model=64, d_inner=128)
        # ceil(64 / 16) = 4
        assert block.dt_rank == 4

    def test_default_dt_rank_non_divisible(self):
        """dt_rank rounds up for non-divisible d_model."""
        block = MambaBlock(d_model=100, d_inner=200)
        # ceil(100 / 16) = 7
        assert block.dt_rank == 7

    def test_custom_dt_rank(self):
        """dt_rank can be overridden."""
        block = MambaBlock(d_model=64, d_inner=128, dt_rank=8)
        assert block.dt_rank == 8

    def test_parameters_created(self):
        """All sub-module parameters are created."""
        block = MambaBlock(d_model=64, d_inner=128, d_state=16)
        params = list(block.parameters())
        # in_proj.weight, conv1d.weight, conv1d.bias,
        # ssm.x_proj.weight, ssm.dt_proj.weight, ssm.dt_proj.bias,
        # ssm.A_log, ssm.D, out_proj.weight
        assert len(params) == 9

    def test_in_proj_shape(self):
        """Input projection maps d_model → 2*d_inner."""
        block = MambaBlock(d_model=64, d_inner=128)
        # in_proj: (2*d_inner, d_model) = (256, 64)
        assert list(block.in_proj.weight.shape) == [256, 64]

    def test_out_proj_shape(self):
        """Output projection maps d_inner → d_model."""
        block = MambaBlock(d_model=64, d_inner=128)
        # out_proj: (d_model, d_inner) = (64, 128)
        assert list(block.out_proj.weight.shape) == [64, 128]

    def test_conv1d_shape(self):
        """Conv1D weight has correct shape."""
        block = MambaBlock(d_model=64, d_inner=128, conv_kernel=4)
        # conv1d.weight: (d_inner, 1, conv_kernel) = (128, 1, 4)
        assert list(block.conv1d.weight.shape) == [128, 1, 4]

    def test_forward_builds_graph(self):
        """Forward pass constructs a valid ONNX graph."""
        block = MambaBlock(d_model=64, d_inner=128, d_state=16, conv_kernel=4)
        test_builder, op, _graph = create_test_builder()
        hidden = create_test_input(test_builder, "hidden_states", [2, 1, 64])
        conv_state = create_test_input(test_builder, "conv_state", [2, 128, 3])
        ssm_state = create_test_input(test_builder, "ssm_state", [2, 128, 16])

        output, new_conv, new_ssm = block(op, hidden, conv_state, ssm_state)

        assert output is not None
        assert new_conv is not None
        assert new_ssm is not None

    def test_conv_op_present(self):
        """Graph contains Conv op from depthwise conv1d."""
        block = MambaBlock(d_model=32, d_inner=64, d_state=8)
        test_builder, op, graph = create_test_builder()
        hidden = create_test_input(test_builder, "hidden_states", [1, 1, 32])
        conv_state = create_test_input(test_builder, "conv_state", [1, 64, 3])
        ssm_state = create_test_input(test_builder, "ssm_state", [1, 64, 8])

        block(op, hidden, conv_state, ssm_state)

        assert count_op_type(graph, "Conv") >= 1

    def test_split_ops_present(self):
        """Graph contains Split ops for in_proj and SSM projections."""
        block = MambaBlock(d_model=32, d_inner=64, d_state=8)
        test_builder, op, graph = create_test_builder()
        hidden = create_test_input(test_builder, "hidden_states", [1, 1, 32])
        conv_state = create_test_input(test_builder, "conv_state", [1, 64, 3])
        ssm_state = create_test_input(test_builder, "ssm_state", [1, 64, 8])

        block(op, hidden, conv_state, ssm_state)

        # Two splits: in_proj (x/z) and SSM x_proj (dt/B/C)
        assert count_op_type(graph, "Split") >= 2

    def test_silu_activation_uses_fused_swish(self):
        """Both SiLU activations use the fused Swish op."""
        block = MambaBlock(d_model=32, d_inner=64, d_state=8)
        test_builder, op, graph = create_test_builder()
        hidden = create_test_input(test_builder, "hidden_states", [1, 1, 32])
        conv_state = create_test_input(test_builder, "conv_state", [1, 64, 3])
        ssm_state = create_test_input(test_builder, "ssm_state", [1, 64, 8])

        block(op, hidden, conv_state, ssm_state)

        # SiLU appears in the convolution path and output gate.
        assert count_op_type(graph, "Swish") == 2
        assert count_op_type(graph, "Sigmoid") == 0

    def test_matmul_ops_for_projections(self):
        """Graph contains MatMul ops for linear projections."""
        block = MambaBlock(d_model=32, d_inner=64, d_state=8)
        test_builder, op, graph = create_test_builder()
        hidden = create_test_input(test_builder, "hidden_states", [1, 1, 32])
        conv_state = create_test_input(test_builder, "conv_state", [1, 64, 3])
        ssm_state = create_test_input(test_builder, "ssm_state", [1, 64, 8])

        block(op, hidden, conv_state, ssm_state)

        # in_proj, x_proj, dt_proj, out_proj = at least 4 MatMul ops
        assert count_op_type(graph, "MatMul") >= 4

    def test_custom_conv_kernel(self):
        """Custom conv_kernel size is reflected in parameters."""
        block = MambaBlock(d_model=32, d_inner=64, conv_kernel=8)
        assert list(block.conv1d.weight.shape) == [64, 1, 8]
        assert block.conv_kernel == 8


class TestSequenceMambaBlock:
    """Tests for the full-sequence (stateless) Mamba1 block."""

    def test_parameters_match_decode_block(self):
        """Parameter names and shapes are identical to MambaBlock's."""

        def spec(module):
            return {name: list(p.shape) for name, p in module.named_parameters()}

        assert spec(SequenceMambaBlock(d_model=32, d_inner=64, d_state=8)) == spec(
            MambaBlock(d_model=32, d_inner=64, d_state=8)
        )

    def test_forward_needs_no_state(self):
        """The whole sequence is consumed in one call, with no carried state."""
        block = SequenceMambaBlock(d_model=32, d_inner=64, d_state=8)
        test_builder, op, graph = create_test_builder()
        hidden = create_test_input(test_builder, "hidden_states", [2, 7, 32])

        out = block(op, hidden)

        assert out is not None
        assert count_op_type(graph, "Scan") == 1
        # in_proj, x_proj, dt_proj, out_proj
        assert count_op_type(graph, "MatMul") >= 4

    def test_conv_is_left_padded_for_causality(self):
        """Conv1D is left-padded by kernel_size - 1 so it stays causal."""
        block = SequenceMambaBlock(d_model=16, d_inner=32, d_state=4, conv_kernel=4)
        test_builder, op, graph = create_test_builder()
        hidden = create_test_input(test_builder, "hidden_states", [1, 5, 16])

        block(op, hidden)

        pad = next(node for node in graph if node.op_type == "Pad")
        pads = pad.inputs[1].const_value.numpy().tolist()
        # (batch, d_inner, seq): 3 begin values then 3 end values.
        assert pads == [0, 0, 3, 0, 0, 0]

    def test_matches_reference_recurrence(self):
        """ONNX output matches an explicit PyTorch selective-scan reference."""
        pytest.importorskip("onnxruntime")
        import onnxruntime as ort
        import torch
        import torch.nn.functional as torch_f

        d_model, d_state, conv_kernel, expand = 8, 4, 4, 2
        d_inner = d_model * expand
        dt_rank = -(-d_model // 16)
        batch, seq_len = 2, 6

        torch.manual_seed(0)
        weights = {
            "in_proj.weight": torch.randn(2 * d_inner, d_model) * 0.2,
            "conv1d.weight": torch.randn(d_inner, 1, conv_kernel) * 0.2,
            "conv1d.bias": torch.randn(d_inner) * 0.2,
            "ssm.x_proj.weight": torch.randn(dt_rank + 2 * d_state, d_inner) * 0.2,
            "ssm.dt_proj.weight": torch.randn(d_inner, dt_rank) * 0.2,
            "ssm.dt_proj.bias": torch.randn(d_inner) * 0.2,
            "ssm.A_log": torch.randn(d_inner, d_state) * 0.2,
            "ssm.D": torch.randn(d_inner) * 0.2,
            "out_proj.weight": torch.randn(d_model, d_inner) * 0.2,
        }

        def reference(u):
            """``mamba_ssm.Mamba`` forward, written out with plain torch ops."""
            x, z = (u @ weights["in_proj.weight"].t()).chunk(2, dim=-1)
            x = torch_f.conv1d(
                torch_f.pad(x.transpose(1, 2), (conv_kernel - 1, 0)),
                weights["conv1d.weight"],
                weights["conv1d.bias"],
                groups=d_inner,
            )
            x = torch_f.silu(x).transpose(1, 2)

            x_dbl = x @ weights["ssm.x_proj.weight"].t()
            dt_raw, b_mat, c_mat = torch.split(x_dbl, [dt_rank, d_state, d_state], dim=-1)
            dt = torch_f.softplus(
                dt_raw @ weights["ssm.dt_proj.weight"].t() + weights["ssm.dt_proj.bias"]
            )

            a_neg = -torch.exp(weights["ssm.A_log"])
            state = torch.zeros(u.shape[0], d_inner, d_state)
            outputs = []
            for t in range(u.shape[1]):
                dt_col = dt[:, t].unsqueeze(-1)
                decay = torch.exp(dt_col * a_neg.unsqueeze(0))
                update = dt_col * x[:, t].unsqueeze(-1) * b_mat[:, t].unsqueeze(1)
                state = decay * state + update
                outputs.append((state * c_mat[:, t].unsqueeze(1)).sum(-1))
            y = torch.stack(outputs, dim=1) + weights["ssm.D"] * x
            return (y * torch_f.silu(z)) @ weights["out_proj.weight"].t()

        graph = ir.Graph([], [], nodes=[], name="g", opset_imports={"": OPSET_VERSION})
        builder = GraphBuilder(graph)
        hidden = builder.input(
            "hidden_states", dtype=ir.DataType.FLOAT, shape=["batch", "seq", d_model]
        )
        block = SequenceMambaBlock(d_model, d_inner, d_state, dt_rank, conv_kernel)
        for name, param in block.named_parameters():
            param.const_value = ir.tensor(weights[name].numpy())
        builder.add_output(block(builder.op, hidden), "y")

        session = ort.InferenceSession(
            ir.to_proto(ir.Model(graph, ir_version=11)).SerializeToString(),
            providers=["CPUExecutionProvider"],
        )
        hidden_states = torch.randn(batch, seq_len, d_model)
        (got,) = session.run(None, {"hidden_states": hidden_states.numpy()})

        np.testing.assert_allclose(got, reference(hidden_states).detach().numpy(), atol=1e-5)


class TestStatefulMambaBlock:
    """Verify the externally threaded, chunk-capable Mamba-1 variant."""

    def test_public_k_wide_convolution_state_and_recurrence_are_explicit(self) -> None:
        block = StatefulMambaBlock(
            d_model=32,
            d_inner=64,
            d_state=8,
            dt_rank=4,
            conv_kernel=4,
            conv_state_width=4,
        )
        builder, op, graph = create_test_builder()
        hidden = create_test_input(builder, "hidden_states", [2, 5, 32])
        conv_state = create_test_input(builder, "conv_state", [2, 64, 4])
        ssm_state = create_test_input(builder, "ssm_state", [2, 64, 8])

        output, present_conv, present_ssm, raw_ssm = block(op, hidden, conv_state, ssm_state)
        builder._adapt_outputs([output, present_conv, present_ssm, raw_ssm], "")

        assert list(block.A_log.shape) == [64, 8]
        assert list(block.D.shape) == [64]
        assert block.A_log._keep_float32
        assert block.D._keep_float32
        assert count_op_type(graph, "Conv") == 1
        assert count_op_type(graph, "LinearAttention") == 1

    def test_dt_bias_is_added_after_float32_rank_projection(self) -> None:
        block = StatefulMambaBlock(d_model=32, d_inner=64, d_state=8, dt_rank=4)
        builder, op, graph = create_test_builder()
        hidden = create_test_input(builder, "hidden_states", [1, 2, 32], ir.DataType.BFLOAT16)
        conv_state = create_test_input(builder, "conv_state", [1, 64, 4], ir.DataType.BFLOAT16)
        ssm_state = create_test_input(builder, "ssm_state", [1, 64, 8], ir.DataType.BFLOAT16)

        block(op, hidden, conv_state, ssm_state)

        dt_bias_cast = next(
            node
            for node in graph
            if node.op_type == "Cast" and node.inputs[0].name == "dt_proj.bias"
        )
        assert dt_bias_cast.attributes["to"].value == int(ir.DataType.FLOAT)


class TestFloatSwiGLU:
    """Verify the Jiterator-compatible gated activation uses fp32 intermediates."""

    def test_keeps_fused_expression_precision_order(self) -> None:
        component = FloatSwiGLU()
        builder, op, graph = create_test_builder()
        gate = create_test_input(builder, "gate", [1, 2, 8], ir.DataType.BFLOAT16)
        value = create_test_input(builder, "value", [1, 2, 8], ir.DataType.BFLOAT16)

        builder.add_output(component(op, gate, value), "output")

        assert count_op_type(graph, "Sigmoid") == 1
        assert count_op_type(graph, "Swish") == 0
