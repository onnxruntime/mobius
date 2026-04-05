# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for ShortConv: gated causal depthwise Conv1d (LFM2 conv layers)."""

from __future__ import annotations

from mobius._testing import count_op_type, create_test_builder, create_test_input
from mobius.components._short_conv import ShortConv

_HIDDEN = 32
_KERNEL = 3
_BATCH = 2
_SEQ = 5


class TestShortConvParameters:
    """Verify parameter shapes match HuggingFace Lfm2ShortConv."""

    def test_in_proj_shape(self):
        conv = ShortConv(hidden_size=_HIDDEN, kernel_size=_KERNEL)
        assert list(conv.in_proj.weight.shape) == [3 * _HIDDEN, _HIDDEN]

    def test_out_proj_shape(self):
        conv = ShortConv(hidden_size=_HIDDEN, kernel_size=_KERNEL)
        assert list(conv.out_proj.weight.shape) == [_HIDDEN, _HIDDEN]

    def test_conv_weight_shape(self):
        """Depthwise conv: (hidden_size, 1, kernel_size) matches Conv1d(groups=hidden)."""
        conv = ShortConv(hidden_size=_HIDDEN, kernel_size=_KERNEL)
        assert list(conv.conv_weight.shape) == [_HIDDEN, 1, _KERNEL]

    def test_no_bias_by_default(self):
        conv = ShortConv(hidden_size=_HIDDEN, kernel_size=_KERNEL, bias=False)
        assert conv.conv_bias is None
        assert conv.in_proj.bias is None
        assert conv.out_proj.bias is None

    def test_bias_creates_parameters(self):
        conv = ShortConv(hidden_size=_HIDDEN, kernel_size=_KERNEL, bias=True)
        assert conv.conv_bias is not None
        assert list(conv.conv_bias.shape) == [_HIDDEN]
        assert conv.in_proj.bias is not None
        assert conv.out_proj.bias is not None


class TestShortConvPrefill:
    """Prefill path: conv_state=None, uses left-padding."""

    def test_output_shape(self):
        """Output should be (B, S, hidden_size)."""
        conv = ShortConv(hidden_size=_HIDDEN, kernel_size=_KERNEL)
        builder, op, graph = create_test_builder()
        x = create_test_input(builder, "x", [_BATCH, _SEQ, _HIDDEN])

        out, new_state = conv(op, x, conv_state=None)
        graph.outputs.extend([out, new_state])

        assert out.shape is not None
        # (B, S, hidden_size)
        assert list(out.shape) == [_BATCH, _SEQ, _HIDDEN]

    def test_conv_state_shape(self):
        """new_conv_state should be (B, hidden_size, kernel_size-1)."""
        conv = ShortConv(hidden_size=_HIDDEN, kernel_size=_KERNEL)
        builder, op, graph = create_test_builder()
        x = create_test_input(builder, "x", [_BATCH, _SEQ, _HIDDEN])

        out, new_state = conv(op, x, conv_state=None)
        graph.outputs.extend([out, new_state])

        # kernel_size - 1 = 2 for kernel=3
        assert new_state.shape is not None
        assert list(new_state.shape) == [_BATCH, _HIDDEN, _KERNEL - 1]

    def test_pad_op_used_for_causal(self):
        """Prefill path pads left by kernel_size-1 using ONNX Pad."""
        conv = ShortConv(hidden_size=_HIDDEN, kernel_size=_KERNEL)
        builder, op, graph = create_test_builder()
        x = create_test_input(builder, "x", [_BATCH, _SEQ, _HIDDEN])

        out, new_state = conv(op, x, conv_state=None)
        graph.outputs.extend([out, new_state])

        assert count_op_type(graph, "Pad") >= 1

    def test_conv_op_present(self):
        """Graph must contain ONNX Conv node (depthwise)."""
        conv = ShortConv(hidden_size=_HIDDEN, kernel_size=_KERNEL)
        builder, op, graph = create_test_builder()
        x = create_test_input(builder, "x", [_BATCH, _SEQ, _HIDDEN])

        out, new_state = conv(op, x, conv_state=None)
        graph.outputs.extend([out, new_state])

        assert count_op_type(graph, "Conv") >= 1

    def test_single_step_prefill(self):
        """Single-token prefill (S=1) still works: state shape (B, H, K-1)."""
        conv = ShortConv(hidden_size=_HIDDEN, kernel_size=_KERNEL)
        builder, op, graph = create_test_builder()
        x = create_test_input(builder, "x", [_BATCH, 1, _HIDDEN])

        out, new_state = conv(op, x, conv_state=None)
        graph.outputs.extend([out, new_state])

        assert list(out.shape) == [_BATCH, 1, _HIDDEN]
        assert list(new_state.shape) == [_BATCH, _HIDDEN, _KERNEL - 1]


class TestShortConvIncremental:
    """Incremental (decode) path: conv_state provided, uses Concat instead of Pad."""

    def test_output_shape_decode_step(self):
        """Single decode step: output (B, 1, hidden_size)."""
        conv = ShortConv(hidden_size=_HIDDEN, kernel_size=_KERNEL)
        builder, op, graph = create_test_builder()
        x = create_test_input(builder, "x", [_BATCH, 1, _HIDDEN])
        state = create_test_input(builder, "conv_state", [_BATCH, _HIDDEN, _KERNEL - 1])

        out, new_state = conv(op, x, conv_state=state)
        graph.outputs.extend([out, new_state])

        assert list(out.shape) == [_BATCH, 1, _HIDDEN]

    def test_conv_state_shape_decode(self):
        """new_conv_state stays (B, hidden_size, kernel_size-1)."""
        conv = ShortConv(hidden_size=_HIDDEN, kernel_size=_KERNEL)
        builder, op, graph = create_test_builder()
        x = create_test_input(builder, "x", [_BATCH, 1, _HIDDEN])
        state = create_test_input(builder, "conv_state", [_BATCH, _HIDDEN, _KERNEL - 1])

        out, new_state = conv(op, x, conv_state=state)
        graph.outputs.extend([out, new_state])

        assert list(new_state.shape) == [_BATCH, _HIDDEN, _KERNEL - 1]

    def test_concat_used_not_pad(self):
        """Incremental path uses Concat (not Pad) to prepend conv_state."""
        conv = ShortConv(hidden_size=_HIDDEN, kernel_size=_KERNEL)
        builder, op, graph = create_test_builder()
        x = create_test_input(builder, "x", [_BATCH, 1, _HIDDEN])
        state = create_test_input(builder, "conv_state", [_BATCH, _HIDDEN, _KERNEL - 1])

        out, new_state = conv(op, x, conv_state=state)
        graph.outputs.extend([out, new_state])

        assert count_op_type(graph, "Concat") >= 1
        assert count_op_type(graph, "Pad") == 0

    def test_multi_step_decode(self):
        """Multi-step decode (S>1 with conv_state) works correctly."""
        conv = ShortConv(hidden_size=_HIDDEN, kernel_size=_KERNEL)
        builder, op, graph = create_test_builder()
        x = create_test_input(builder, "x", [_BATCH, 3, _HIDDEN])
        state = create_test_input(builder, "conv_state", [_BATCH, _HIDDEN, _KERNEL - 1])

        out, new_state = conv(op, x, conv_state=state)
        graph.outputs.extend([out, new_state])

        assert list(out.shape) == [_BATCH, 3, _HIDDEN]
        assert list(new_state.shape) == [_BATCH, _HIDDEN, _KERNEL - 1]


class TestShortConvGating:
    """Verify the B/C/x gating structure is present in the graph."""

    def test_mul_ops_for_gating(self):
        """Graph must have at least 2 Mul nodes: B*x and C*conv_out."""
        conv = ShortConv(hidden_size=_HIDDEN, kernel_size=_KERNEL)
        builder, op, graph = create_test_builder()
        x = create_test_input(builder, "x", [_BATCH, _SEQ, _HIDDEN])

        out, new_state = conv(op, x, conv_state=None)
        graph.outputs.extend([out, new_state])

        assert count_op_type(graph, "Mul") >= 2

    def test_in_proj_splits_into_three(self):
        """in_proj expands H → 3H; verified by Slice ops (one per chunk B/C/x)."""
        conv = ShortConv(hidden_size=_HIDDEN, kernel_size=_KERNEL)
        builder, op, graph = create_test_builder()
        x = create_test_input(builder, "x", [_BATCH, _SEQ, _HIDDEN])

        out, new_state = conv(op, x, conv_state=None)
        graph.outputs.extend([out, new_state])

        # Three Slice ops to extract B, C, x from in_proj output
        assert count_op_type(graph, "Slice") >= 3


class TestShortConvKernelSizes:
    """Verify correctness with different kernel sizes."""

    def test_kernel_4(self):
        """kernel_size=4: conv_state shape (B, H, 3)."""
        conv = ShortConv(hidden_size=_HIDDEN, kernel_size=4)
        builder, op, graph = create_test_builder()
        x = create_test_input(builder, "x", [_BATCH, _SEQ, _HIDDEN])

        out, new_state = conv(op, x, conv_state=None)
        graph.outputs.extend([out, new_state])

        assert list(out.shape) == [_BATCH, _SEQ, _HIDDEN]
        assert list(new_state.shape) == [_BATCH, _HIDDEN, 3]

    def test_kernel_2(self):
        """kernel_size=2: conv_state shape (B, H, 1) — minimum rolling window."""
        conv = ShortConv(hidden_size=_HIDDEN, kernel_size=2)
        builder, op, graph = create_test_builder()
        x = create_test_input(builder, "x", [_BATCH, _SEQ, _HIDDEN])

        out, new_state = conv(op, x, conv_state=None)
        graph.outputs.extend([out, new_state])

        assert list(out.shape) == [_BATCH, _SEQ, _HIDDEN]
        assert list(new_state.shape) == [_BATCH, _HIDDEN, 1]
