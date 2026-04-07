# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for Linear, Embedding, and attention bias utilities."""

from __future__ import annotations

import onnx_ir as ir

from mobius._build_context import build_context
from mobius._execution_providers import EpCapabilities
from mobius._testing import count_op_type, create_test_builder, create_test_input
from mobius.components._common import (
    Embedding,
    Linear,
    _mask_seq_len,
    create_attention_bias,
    create_padding_mask,
)

_WEBGPU_CAPS = EpCapabilities(
    name="webgpu",
    gqa_dtypes=frozenset(),
    supports_shape=False,
)


class TestLinear:
    def test_linear_with_bias(self):
        linear = Linear(64, 128, bias=True)
        params = list(linear.parameters())
        assert len(params) == 2  # weight + bias
        assert list(linear.weight.shape) == [128, 64]
        assert list(linear.bias.shape) == [128]

    def test_linear_without_bias(self):
        linear = Linear(64, 128, bias=False)
        params = list(linear.parameters())
        assert len(params) == 1  # weight only
        assert linear.bias is None

    def test_linear_forward(self):
        builder, op, graph = create_test_builder()
        x = create_test_input(builder, "x", [2, 3, 64])
        linear = Linear(64, 128, bias=True)
        result = linear(op, x)
        assert result is not None
        # Linear emits FusedMatMul(x, weight, transB=1) directly
        assert count_op_type(graph, "FusedMatMul") >= 1
        assert count_op_type(graph, "Add") >= 1

    def test_linear_no_bias_forward(self):
        builder, op, graph = create_test_builder()
        x = create_test_input(builder, "x", [2, 3, 64])
        linear = Linear(64, 128, bias=False)
        result = linear(op, x)
        assert result is not None
        assert count_op_type(graph, "FusedMatMul") >= 1
        assert count_op_type(graph, "Add") == 0


class TestEmbedding:
    def test_embedding_params(self):
        emb = Embedding(1000, 64)
        params = list(emb.parameters())
        assert len(params) == 1
        assert list(emb.weight.shape) == [1000, 64]

    def test_embedding_forward(self):
        builder, op, graph = create_test_builder()
        input_ids = create_test_input(builder, "input_ids", [2, 4], dtype=ir.DataType.INT64)
        emb = Embedding(1000, 64)
        result = emb(op, input_ids)
        assert result is not None
        assert count_op_type(graph, "Gather") >= 1

    def test_embedding_with_padding_idx(self):
        emb = Embedding(1000, 64, padding_idx=0)
        assert emb.padding_idx == 0


class TestCreateAttentionBias:
    def test_creates_bias(self):
        builder, op, graph = create_test_builder()
        input_ids = create_test_input(builder, "input_ids", [2, 4], dtype=ir.DataType.INT64)
        attention_mask = create_test_input(
            builder, "attention_mask", [2, 8], dtype=ir.DataType.INT64
        )
        bias = create_attention_bias(op, input_ids, attention_mask)
        assert bias is not None
        assert graph.num_nodes() > 0

    def test_creates_bias_with_sliding_window(self):
        builder, op, graph = create_test_builder()
        input_ids = create_test_input(builder, "input_ids", [2, 4], dtype=ir.DataType.INT64)
        attention_mask = create_test_input(
            builder, "attention_mask", [2, 8], dtype=ir.DataType.INT64
        )
        bias = create_attention_bias(op, input_ids, attention_mask, sliding_window=4)
        assert bias is not None
        assert count_op_type(graph, "Less") >= 1

    def test_query_length_from_input_ids_not_attention_mask(self):
        """Shape(input_ids, 1) must provide query_length, not Shape(attention_mask, 1).

        During decode, input_ids is (batch, 1) and attention_mask is (batch, total_len).
        Both shapes must produce separate Shape nodes so the Slice picks only the
        last query row.  If both came from attention_mask, start=0 and the full
        sequence would be used as queries, producing wrong attention scores.
        """
        builder, op, graph = create_test_builder()
        # Simulate decode: q_len=1, total_len=8 (7 past + 1 current token)
        input_ids = create_test_input(builder, "input_ids", [2, 1], dtype=ir.DataType.INT64)
        attention_mask = create_test_input(
            builder, "attention_mask", [2, 8], dtype=ir.DataType.INT64
        )
        create_attention_bias(op, input_ids, attention_mask)

        # query_length must come from input_ids (dim 1 = 1), not attention_mask.
        # Verify there is a Shape node that reads from input_ids.
        shape_inputs = [
            n.inputs[0].name for n in graph if n.op_type == "Shape" and n.inputs[0] is not None
        ]
        assert any(name == "input_ids" for name in shape_inputs), (
            "Shape(input_ids, 1) must be present to extract query_length correctly"
        )


class TestCreatePaddingMask:
    def test_creates_bool_mask_with_2d_input_ids(self):
        """Standard path: input_ids is 2D [batch, q_len]."""
        builder, op, graph = create_test_builder()
        input_ids = create_test_input(builder, "input_ids", [2, 4], dtype=ir.DataType.INT64)
        attention_mask = create_test_input(
            builder, "attention_mask", [2, 8], dtype=ir.DataType.INT64
        )
        mask = create_padding_mask(op, input_ids, attention_mask)
        assert mask is not None
        assert count_op_type(graph, "Cast") >= 1
        assert count_op_type(graph, "Unsqueeze") >= 1
        assert count_op_type(graph, "Expand") >= 1

    def test_creates_bool_mask_with_3d_hidden_states(self):
        """inputs_embeds path: input_ids is actually 3D hidden_states."""
        builder, op, graph = create_test_builder()
        hidden_states = create_test_input(
            builder, "hidden_states", [2, 4, 256], dtype=ir.DataType.FLOAT
        )
        attention_mask = create_test_input(
            builder, "attention_mask", [2, 8], dtype=ir.DataType.INT64
        )
        mask = create_padding_mask(op, hidden_states, attention_mask)
        assert mask is not None
        # Should still produce a valid graph (no crash from 3D input).
        assert count_op_type(graph, "Cast") >= 1
        assert count_op_type(graph, "Expand") >= 1

    def test_uses_simpler_ops_than_attention_bias(self):
        """Padding mask uses simple broadcast ops, not the causal CumSum chain."""
        builder_pad, op_pad, graph_pad = create_test_builder()
        input_ids_pad = create_test_input(
            builder_pad, "input_ids", [2, 4], dtype=ir.DataType.INT64
        )
        mask_pad = create_test_input(
            builder_pad, "attention_mask", [2, 8], dtype=ir.DataType.INT64
        )
        create_padding_mask(op_pad, input_ids_pad, mask_pad)

        # Padding mask avoids the CumSum/GreaterOrEqual/Where causal-bias chain
        assert count_op_type(graph_pad, "Cast") >= 1
        assert count_op_type(graph_pad, "Unsqueeze") >= 1
        assert count_op_type(graph_pad, "Expand") >= 1
        assert count_op_type(graph_pad, "CumSum") == 0
        assert count_op_type(graph_pad, "GreaterOrEqual") == 0
        assert count_op_type(graph_pad, "Where") == 0

        builder_bias, op_bias, graph_bias = create_test_builder()
        input_ids_bias = create_test_input(
            builder_bias, "input_ids", [2, 4], dtype=ir.DataType.INT64
        )
        mask_bias = create_test_input(
            builder_bias, "attention_mask", [2, 8], dtype=ir.DataType.INT64
        )
        create_attention_bias(op_bias, input_ids_bias, mask_bias)

        # Full causal bias requires CumSum + GreaterOrEqual + Where chain
        assert count_op_type(graph_bias, "CumSum") >= 1
        assert count_op_type(graph_bias, "GreaterOrEqual") >= 1
        assert count_op_type(graph_bias, "Where") >= 1


class TestMaskSeqLen:
    """_mask_seq_len() must emit different ops depending on ep_capabilities().supports_shape."""

    def test_supports_shape_true_emits_shape_op(self):
        """Default EP (supports_shape=True): must emit Shape, not ReduceSum/ReduceMax."""
        builder, op, graph = create_test_builder()
        attention_mask = create_test_input(
            builder, "attention_mask", [2, 8], dtype=ir.DataType.INT64
        )
        # Default build context has supports_shape=True
        result = _mask_seq_len(op, attention_mask)
        assert result is not None
        assert count_op_type(graph, "Shape") == 1
        assert count_op_type(graph, "ReduceSum") == 0
        assert count_op_type(graph, "ReduceMax") == 0

    def test_supports_shape_false_emits_reduce_ops_not_shape(self):
        """WebGPU EP (supports_shape=False): must emit ReduceSum+ReduceMax, no Shape."""
        builder, op, graph = create_test_builder()
        attention_mask = create_test_input(
            builder, "attention_mask", [2, 8], dtype=ir.DataType.INT64
        )
        with build_context(_WEBGPU_CAPS, ir.DataType.FLOAT16):
            result = _mask_seq_len(op, attention_mask)
        assert result is not None
        assert count_op_type(graph, "Shape") == 0
        assert count_op_type(graph, "ReduceSum") == 1
        assert count_op_type(graph, "ReduceMax") == 1
