# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for Linear, Embedding, and attention bias utilities."""

from __future__ import annotations

import numpy as np
import onnx_ir as ir

from mobius._testing import count_op_type, create_test_builder, create_test_input
from mobius._testing.ort_inference import OnnxModelSession
from mobius.components._common import (
    Embedding,
    Linear,
    build_packed_token_offset,
    create_attention_bias,
    create_padding_mask,
    create_sliding_window_mask,
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
        # Linear emits Transpose(weight, perm=[1,0]) + MatMul(x, w_t)
        assert count_op_type(graph, "MatMul") >= 1
        assert count_op_type(graph, "Add") >= 1

    def test_linear_no_bias_forward(self):
        builder, op, graph = create_test_builder()
        x = create_test_input(builder, "x", [2, 3, 64])
        linear = Linear(64, 128, bias=False)
        result = linear(op, x)
        assert result is not None
        assert count_op_type(graph, "MatMul") >= 1
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


class TestBlockwiseAttentionBias:
    """Numerically verify the Gemma4 vision-block bidirectional overlay.

    ``create_attention_bias(block_sequence_ids=...)`` must bake the FULL mask
    (causal [+ sliding] OR same-block, AND padding) so the Attention op can be
    called with ``is_causal=0``. We build the graph, run it via ORT, and
    compare the attended pattern (bias == 0) against a numpy reference.
    """

    @staticmethod
    def _build(sliding):
        b, op, g = create_test_builder()
        input_ids = create_test_input(b, "input_ids", [1, "S"], dtype=ir.DataType.INT64)
        attn = create_test_input(b, "attention_mask", [1, "T"], dtype=ir.DataType.INT64)
        bsid = create_test_input(b, "block_sequence_ids", [1, "S"], dtype=ir.DataType.INT64)
        bias = create_attention_bias(
            op,
            input_ids,
            attn,
            sliding_window=sliding,
            dtype=ir.DataType.FLOAT,
            block_sequence_ids=bsid,
        )
        bias.name = "bias"
        g.outputs.append(bias)
        return ir.Model(g, ir_version=10)

    @staticmethod
    def _ref(block_ids, attn, sliding):
        cumsum = np.cumsum(attn)
        qi = cumsum[:, None]
        ki = cumsum[None, :]
        m = qi >= ki
        if sliding is not None:
            m = m & ((qi - ki) < sliding)
        qg = np.array(block_ids)[:, None]
        kg = np.array(block_ids)[None, :]
        m = m | ((qg == kg) & (qg >= 0))
        return m & (np.array(attn)[None, :].astype(bool))

    def _run(self, sliding, block_ids, attn):
        sess = OnnxModelSession(self._build(sliding), device="cpu")
        block_ids = np.array([block_ids], dtype=np.int64)
        attn = np.array([attn], dtype=np.int64)
        out = sess.run(
            {
                "input_ids": np.zeros_like(block_ids),
                "attention_mask": attn,
                "block_sequence_ids": block_ids,
            }
        )["bias"]
        attended = out[0, 0] > -1.0
        expected = self._ref(block_ids[0], attn[0], sliding)
        return attended, expected

    def test_multi_block_full_attention(self):
        # Two vision blocks (pos 1-2 and 4-5) separated by text.
        attended, expected = self._run(None, [-1, 0, 0, -1, 1, 1, -1, -1], [1] * 8)
        assert np.array_equal(attended, expected)
        # A vision token attends to a LATER token in the same block (bidirectional).
        assert attended[1, 2]
        # But text stays causal: position 3 cannot see position 4.
        assert not attended[3, 4]

    def test_block_wider_than_sliding_window(self):
        # Single block spanning positions 1..4 with a window of 2: the block
        # must escape the sliding window (same-block OR overrides the window).
        attended, expected = self._run(2, [-1, 0, 0, 0, 0, -1, -1, -1], [1] * 8)
        assert np.array_equal(attended, expected)
        assert attended[4, 1]  # distance 3 >= window, allowed via same block

    def test_padding_still_masked(self):
        # Last two positions are padding (attention_mask == 0).
        attended, expected = self._run(
            2, [-1, 0, 0, -1, -1, -1, -1, -1], [1, 1, 1, 1, 1, 1, 0, 0]
        )
        assert np.array_equal(attended, expected)
        assert not attended[:, 6:].any()  # nothing attends to padding

    def test_decode_single_query_is_causal(self):
        # Decode step: q_len=1 (new text token, group -1), kv total length 8.
        sess = OnnxModelSession(self._build(None), device="cpu")
        out = sess.run(
            {
                "input_ids": np.zeros((1, 1), dtype=np.int64),
                "attention_mask": np.ones((1, 8), dtype=np.int64),
                "block_sequence_ids": np.array([[-1]], dtype=np.int64),
            }
        )["bias"]
        assert out.shape == (1, 1, 1, 8)
        # Text decode token attends to all past positions (pure causal row).
        assert bool((out[0, 0, 0] > -1.0).all())


class TestBuildPackedTokenOffset:
    """``build_packed_token_offset`` must reproduce ORT's GetPaddingOffset.

    ORT's ``PackedMultiHeadAttention`` derives ``batch_size`` from
    ``token_offset.shape[0]`` and requires ``cumulative_sequence_length`` to
    have length ``batch_size + 1``.  ``token_offset`` lists the padded-layout
    indices (``b * max_len + s``) of valid tokens (packed order) first, then
    the padding slots.  We run the helper through ORT and compare exact values.
    """

    @staticmethod
    def _build():
        b, op, g = create_test_builder()
        cu = create_test_input(b, "cu_seqlens", ["K"], dtype=ir.DataType.INT64)
        token_offset = build_packed_token_offset(op, cu)
        token_offset.name = "token_offset"
        g.outputs.append(token_offset)
        return ir.Model(g, ir_version=10)

    def _run(self, cu_seqlens):
        sess = OnnxModelSession(self._build(), device="cpu")
        return sess.run({"cu_seqlens": np.array(cu_seqlens, dtype=np.int64)})["token_offset"]

    def test_ort_reference_example(self):
        # ORT test data: cu=[0,1,3] (lengths 1,2; max_len=2) -> [[0,2],[3,1]].
        out = self._run([0, 1, 3])
        assert out.dtype == np.int32
        assert np.array_equal(out, np.array([[0, 2], [3, 1]], dtype=np.int32))

    def test_padding_indices_exceed_token_count(self):
        # cu=[0,2,5]: lengths [2,3], max_len=3, token_count=5.
        # Padded grid pos = b*3 + s -> row0 valid cols {0,1} pad col {2};
        # row1 valid cols {3,4,5}. valid (packed order) = [0,1,3,4,5];
        # padding slot = [2]. token_offset = [[0,1,3],[4,5,2]].
        out = self._run([0, 2, 5])
        assert np.array_equal(out, np.array([[0, 1, 3], [4, 5, 2]], dtype=np.int32))
        # Padding value (2) is a padded-layout index, here < token_count, but
        # the construction may yield values >= token_count for other shapes.

    def test_single_subsequence_is_identity(self):
        # cu=[0,4]: one sub-sequence -> shape (1,4), identity [0,1,2,3].
        out = self._run([0, 4])
        assert out.shape == (1, 4)
        assert np.array_equal(out, np.array([[0, 1, 2, 3]], dtype=np.int32))

    def test_uniform_windows(self):
        # Three windows of equal length 2: max_len=2, no padding.
        out = self._run([0, 2, 4, 6])
        assert out.shape == (3, 2)
        assert np.array_equal(out, np.array([[0, 1], [2, 3], [4, 5]], dtype=np.int32))


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


class TestMaskHeadDimRank:
    """Padding / sliding-window bool masks must be 4-D ``(B, 1, q, total)``.

    A 3-D ``(B, q, total)`` mask is right-aligned by the ONNX Attention op so
    the batch axis is misread as ``q_num_heads`` — it happens to work for
    ``batch == 1`` (head dim broadcasts) but is rejected once ``batch > 1``.
    The explicit singleton head dim keeps the contract batch-agnostic.
    """

    @staticmethod
    def _build_padding():
        b, op, g = create_test_builder()
        input_ids = create_test_input(b, "input_ids", ["B", "S"], dtype=ir.DataType.INT64)
        attn = create_test_input(b, "attention_mask", ["B", "T"], dtype=ir.DataType.INT64)
        mask = create_padding_mask(op, input_ids, attn)
        mask.name = "mask"
        g.outputs.append(mask)
        return ir.Model(g, ir_version=10)

    @staticmethod
    def _build_sliding(window):
        b, op, g = create_test_builder()
        input_ids = create_test_input(b, "input_ids", ["B", "S"], dtype=ir.DataType.INT64)
        attn = create_test_input(b, "attention_mask", ["B", "T"], dtype=ir.DataType.INT64)
        mask = create_sliding_window_mask(op, input_ids, attn, window)
        mask.name = "mask"
        g.outputs.append(mask)
        return ir.Model(g, ir_version=10)

    def test_padding_mask_is_4d_with_singleton_head_dim(self):
        sess = OnnxModelSession(self._build_padding(), device="cpu")
        out = sess.run(
            {
                "input_ids": np.zeros((3, 4), dtype=np.int64),
                "attention_mask": np.ones((3, 8), dtype=np.int64),
            }
        )["mask"]
        # (batch, 1, q_len, total_len)
        assert out.shape == (3, 1, 4, 8)

    def test_sliding_mask_is_4d_with_singleton_head_dim(self):
        sess = OnnxModelSession(self._build_sliding(2), device="cpu")
        out = sess.run(
            {
                "input_ids": np.zeros((3, 4), dtype=np.int64),
                "attention_mask": np.ones((3, 8), dtype=np.int64),
            }
        )["mask"]
        assert out.shape == (3, 1, 4, 8)

    def test_padding_mask_per_row_independent(self):
        """With batch>1, each row's padding is honoured independently."""
        sess = OnnxModelSession(self._build_padding(), device="cpu")
        attn = np.ones((2, 5), dtype=np.int64)
        attn[1, :2] = 0  # row 1 has two leading padding tokens
        out = sess.run(
            {
                "input_ids": np.zeros((2, 5), dtype=np.int64),
                "attention_mask": attn,
            }
        )["mask"]
        assert out.shape == (2, 1, 5, 5)
        # Row 0 attends to every kv position; row 1 masks the two pad columns.
        assert bool(out[0, 0].all())
        assert not bool(out[1, 0, :, :2].any())
        assert bool(out[1, 0, :, 2:].all())
