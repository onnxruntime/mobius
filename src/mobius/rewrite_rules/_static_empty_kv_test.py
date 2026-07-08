# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for the DynamicEmptyKVToStaticConstant rewrite rule."""

from __future__ import annotations

from collections import Counter

import onnx_ir as ir
from onnxscript import GraphBuilder
from onnxscript.rewriter import rewrite
from onnxscript.rewriter._rewrite_rule import RewriteRuleSet

from mobius._constants import OPSET_VERSION
from mobius.rewrite_rules._static_empty_kv import static_empty_kv_rules


def _count_ops(model: ir.Model) -> Counter:
    return Counter(n.op_type for n in model.graph)


def _make_castlike_model(kv_hidden: int, dtype: ir.DataType = ir.DataType.FLOAT16) -> ir.Model:
    """Build a minimal model with the CastLike-based dynamic empty-KV pattern.

    Graph:
        query_states [batch, seq, hidden] ->
            batch_dim   = Shape(query_states, start=0, end=1)
            shape_tail  = Constant(value_ints=[0, kv_hidden])
            empty_shape = Concat(batch_dim, shape_tail, axis=0)
            cos         = ConstantOfShape(empty_shape)
            empty_kv    = CastLike(cos, query_states)  <- graph output
    """
    query_states = ir.Value(
        name="query_states",
        shape=ir.Shape([None, None, 64]),
        type=ir.TensorType(dtype),
    )
    graph = ir.Graph(
        inputs=[query_states],
        outputs=[],
        nodes=[],
        name="castlike_pattern",
        opset_imports={"": OPSET_VERSION},
    )
    op = GraphBuilder(graph).op
    batch_dim = op.Shape(query_states, start=0, end=1)
    shape_tail = op.Constant(value_ints=[0, kv_hidden])
    empty_shape = op.Concat(batch_dim, shape_tail, axis=0)
    cos = op.ConstantOfShape(empty_shape)
    empty_kv = op.CastLike(cos, query_states)
    empty_kv.name = "empty_kv"
    graph.outputs.append(empty_kv)
    return ir.Model(graph, ir_version=11)


def _make_cast_model(kv_hidden: int, dtype: ir.DataType = ir.DataType.FLOAT16) -> ir.Model:
    """Build a minimal model with the Cast-based dynamic empty-KV pattern.

    This is the post-cleanup variant (e.g. after quantization) where CastLike
    has been materialized to Cast with a concrete target dtype.

    Graph:
        query_states [batch, seq, hidden] ->
            batch_dim   = Shape(query_states, start=0, end=1)
            shape_tail  = Constant(value_ints=[0, kv_hidden])
            empty_shape = Concat(batch_dim, shape_tail, axis=0)
            cos         = ConstantOfShape(empty_shape)
            empty_kv    = Cast(cos, to=dtype)  <- graph output
    """
    query_states = ir.Value(
        name="query_states",
        shape=ir.Shape([None, None, 64]),
        type=ir.TensorType(dtype),
    )
    graph = ir.Graph(
        inputs=[query_states],
        outputs=[],
        nodes=[],
        name="cast_pattern",
        opset_imports={"": OPSET_VERSION},
    )
    op = GraphBuilder(graph).op
    batch_dim = op.Shape(query_states, start=0, end=1)
    shape_tail = op.Constant(value_ints=[0, kv_hidden])
    empty_shape = op.Concat(batch_dim, shape_tail, axis=0)
    cos = op.ConstantOfShape(empty_shape)
    empty_kv = op.Cast(cos, to=dtype)
    empty_kv.name = "empty_kv"
    graph.outputs.append(empty_kv)
    return ir.Model(graph, ir_version=11)


class TestStaticEmptyKVRules:
    def test_returns_rule_set(self):
        assert isinstance(static_empty_kv_rules(), RewriteRuleSet)

    def test_replaces_castlike_pattern_with_constant(self):
        kv_hidden = 256
        model = _make_castlike_model(kv_hidden)
        counts_before = _count_ops(model)
        assert counts_before["Shape"] == 1
        assert counts_before["ConstantOfShape"] == 1
        assert counts_before["CastLike"] == 1

        rewrite(model, pattern_rewrite_rules=static_empty_kv_rules())

        counts_after = _count_ops(model)
        assert counts_after.get("Shape", 0) == 0, "Shape should be removed"
        assert counts_after.get("ConstantOfShape", 0) == 0, "ConstantOfShape should be removed"
        assert counts_after.get("CastLike", 0) == 0, "CastLike should be removed"
        assert counts_after.get("Constant", 0) >= 1, "static Constant should be present"

    def test_output_constant_has_correct_shape(self):
        kv_hidden = 128
        model = _make_castlike_model(kv_hidden)
        rewrite(model, pattern_rewrite_rules=static_empty_kv_rules())

        const_nodes = [n for n in model.graph if n.op_type == "Constant"]
        assert const_nodes, "no Constant node found after rewrite"
        const_node = const_nodes[-1]
        value_attr = const_node.attributes.get("value")
        assert value_attr is not None
        tensor = value_attr.value
        assert list(tensor.shape) == [1, 0, kv_hidden]

    def test_does_not_match_non_batch_shape(self):
        """Rule should not fire when Shape slices a non-batch dimension."""
        query_states = ir.Value(
            name="query_states",
            shape=ir.Shape([None, None, 64]),
            type=ir.TensorType(ir.DataType.FLOAT16),
        )
        graph = ir.Graph(
            inputs=[query_states],
            outputs=[],
            nodes=[],
            name="non_batch_shape",
            opset_imports={"": OPSET_VERSION},
        )
        op = GraphBuilder(graph).op
        # start=1, end=2 — seq dim, not batch dim — should NOT match
        seq_dim = op.Shape(query_states, start=1, end=2)
        shape_tail = op.Constant(value_ints=[0, 256])
        empty_shape = op.Concat(seq_dim, shape_tail, axis=0)
        cos = op.ConstantOfShape(empty_shape)
        empty_kv = op.CastLike(cos, query_states)
        empty_kv.name = "empty_kv"
        graph.outputs.append(empty_kv)
        model = ir.Model(graph, ir_version=11)

        counts_before = _count_ops(model)
        rewrite(model, pattern_rewrite_rules=static_empty_kv_rules())
        counts_after = _count_ops(model)

        assert counts_after.get("Shape", 0) == counts_before.get("Shape", 0), (
            "rule should not fire for non-batch Shape slice"
        )
        assert counts_after.get("ConstantOfShape", 0) == counts_before.get(
            "ConstantOfShape", 0
        )


class TestStaticEmptyKVCastVariant:
    """Tests for the Cast-based pattern (post-cleanup, e.g. after quantization)."""

    def test_replaces_cast_pattern_with_constant(self):
        kv_hidden = 256
        model = _make_cast_model(kv_hidden)
        counts_before = _count_ops(model)
        assert counts_before["ConstantOfShape"] == 1
        assert counts_before["Cast"] == 1

        rewrite(model, pattern_rewrite_rules=static_empty_kv_rules())

        counts_after = _count_ops(model)
        assert counts_after.get("Shape", 0) == 0, "Shape should be removed"
        assert counts_after.get("ConstantOfShape", 0) == 0, "ConstantOfShape should be removed"
        assert counts_after.get("Cast", 0) == 0, "Cast should be removed"
        assert counts_after.get("Constant", 0) >= 1, "static Constant should be present"

    def test_cast_output_constant_has_correct_shape_and_dtype(self):
        kv_hidden = 512
        model = _make_cast_model(kv_hidden, dtype=ir.DataType.FLOAT16)
        rewrite(model, pattern_rewrite_rules=static_empty_kv_rules())

        const_nodes = [n for n in model.graph if n.op_type == "Constant"]
        assert const_nodes, "no Constant node found after rewrite"
        const_node = const_nodes[-1]
        value_attr = const_node.attributes.get("value")
        assert value_attr is not None
        tensor = value_attr.value
        assert list(tensor.shape) == [1, 0, kv_hidden]
        assert tensor.dtype == ir.DataType.FLOAT16, f"expected FLOAT16, got {tensor.dtype}"

    def test_cast_variant_supports_bfloat16(self):
        kv_hidden = 64
        model = _make_cast_model(kv_hidden, dtype=ir.DataType.BFLOAT16)
        rewrite(model, pattern_rewrite_rules=static_empty_kv_rules())

        counts_after = _count_ops(model)
        assert counts_after.get("Shape", 0) == 0
        assert counts_after.get("ConstantOfShape", 0) == 0

        # Accept either a BFLOAT16 Constant directly, or Constant(float32) + Cast(to=BFLOAT16).
        cast_nodes = [n for n in model.graph if n.op_type == "Cast"]
        if cast_nodes:
            to_attr = cast_nodes[-1].attributes.get("to")
            assert to_attr is not None
            assert ir.DataType(to_attr.value) == ir.DataType.BFLOAT16
        else:
            const_nodes = [n for n in model.graph if n.op_type == "Constant"]
            assert const_nodes, "no Constant node found after rewrite"
            value_attr = const_nodes[-1].attributes.get("value")
            assert value_attr is not None
            assert value_attr.value.dtype == ir.DataType.BFLOAT16
