# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for the DynamicEmptyKVToStaticConstant rewrite rule."""

from __future__ import annotations

from collections import Counter

import numpy as np
import onnx_ir as ir
import pytest
from onnxscript.rewriter import rewrite
from onnxscript.rewriter._rewrite_rule import RewriteRuleSet

from mobius.rewrite_rules._static_empty_kv import (
    _DynamicEmptyKVCast,
    _DynamicEmptyKVCastLike,
    static_empty_kv_rules,
)


def _count_ops(model: ir.Model) -> Counter:
    return Counter(n.op_type for n in model.graph)


def _make_empty_kv_model(kv_hidden: int, dtype: ir.DataType = ir.DataType.FLOAT16) -> ir.Model:
    """Build a minimal ONNX model containing the dynamic empty-KV pattern.

    Graph:
        query_states [batch, seq, hidden] →
            batch_dim  = Shape(query_states, start=0, end=1)
            empty_shape = Concat(batch_dim, Constant([0, kv_hidden]), axis=0)
            cos        = ConstantOfShape(empty_shape)
            empty_kv   = CastLike(cos, query_states)  ← graph output
    """
    import onnx

    # Build via onnx helper for portability
    batch_dim_node = onnx.helper.make_node(
        "Shape",
        inputs=["query_states"],
        outputs=["batch_dim"],
        start=0,
        end=1,
    )
    tail_const_node = onnx.helper.make_node(
        "Constant",
        inputs=[],
        outputs=["shape_tail"],
        value_ints=[0, kv_hidden],
    )
    concat_node = onnx.helper.make_node(
        "Concat",
        inputs=["batch_dim", "shape_tail"],
        outputs=["empty_shape"],
        axis=0,
    )
    cos_node = onnx.helper.make_node(
        "ConstantOfShape",
        inputs=["empty_shape"],
        outputs=["cos"],
    )
    castlike_node = onnx.helper.make_node(
        "CastLike",
        inputs=["cos", "query_states"],
        outputs=["empty_kv"],
    )

    np_dtype = np.float16 if dtype == ir.DataType.FLOAT16 else np.float32
    onnx_dtype = onnx.TensorProto.FLOAT16 if dtype == ir.DataType.FLOAT16 else onnx.TensorProto.FLOAT

    graph = onnx.helper.make_graph(
        [batch_dim_node, tail_const_node, concat_node, cos_node, castlike_node],
        "empty_kv_graph",
        inputs=[onnx.helper.make_tensor_value_info("query_states", onnx_dtype, [None, None, 64])],
        outputs=[onnx.helper.make_tensor_value_info("empty_kv", onnx_dtype, [None, 0, kv_hidden])],
    )
    proto = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 18)])
    return ir.serde.deserialize_model(proto)


class TestStaticEmptyKVRules:
    def test_returns_rule_set(self):
        assert isinstance(static_empty_kv_rules(), RewriteRuleSet)

    def test_replaces_dynamic_pattern_with_constant(self):
        kv_hidden = 256
        model = _make_empty_kv_model(kv_hidden)
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
        model = _make_empty_kv_model(kv_hidden)
        rewrite(model, pattern_rewrite_rules=static_empty_kv_rules())

        # Find the Constant node and verify its tensor shape
        const_nodes = [n for n in model.graph if n.op_type == "Constant"]
        assert const_nodes, "no Constant node found after rewrite"
        # The output Constant should be [1, 0, kv_hidden]
        const_node = const_nodes[-1]
        value_attr = const_node.attributes.get("value")
        assert value_attr is not None
        tensor = value_attr.value
        assert list(tensor.shape) == [1, 0, kv_hidden]

    def test_does_not_match_non_batch_shape(self):
        """Rule should not fire when Shape slices a different dimension."""
        import onnx

        # Shape(x, start=1, end=2) — not start=0, end=1
        batch_dim_node = onnx.helper.make_node(
            "Shape", inputs=["query_states"], outputs=["dim"], start=1, end=2
        )
        tail_const_node = onnx.helper.make_node(
            "Constant", inputs=[], outputs=["shape_tail"], value_ints=[0, 256]
        )
        concat_node = onnx.helper.make_node(
            "Concat", inputs=["dim", "shape_tail"], outputs=["empty_shape"], axis=0
        )
        cos_node = onnx.helper.make_node(
            "ConstantOfShape", inputs=["empty_shape"], outputs=["cos"]
        )
        castlike_node = onnx.helper.make_node(
            "CastLike", inputs=["cos", "query_states"], outputs=["empty_kv"]
        )
        graph = onnx.helper.make_graph(
            [batch_dim_node, tail_const_node, concat_node, cos_node, castlike_node],
            "non_batch_shape",
            inputs=[onnx.helper.make_tensor_value_info(
                "query_states", onnx.TensorProto.FLOAT16, [None, None, 64]
            )],
            outputs=[onnx.helper.make_tensor_value_info(
                "empty_kv", onnx.TensorProto.FLOAT16, [None, 0, 256]
            )],
        )
        proto = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 18)])
        model = ir.serde.deserialize_model(proto)

        counts_before = _count_ops(model)
        rewrite(model, pattern_rewrite_rules=static_empty_kv_rules())
        counts_after = _count_ops(model)

        assert counts_after.get("Shape", 0) == counts_before.get("Shape", 0), \
            "rule should not fire for non-batch Shape slice"
        assert counts_after.get("ConstantOfShape", 0) == counts_before.get("ConstantOfShape", 0)


class TestStaticEmptyKVCastVariant:
    """Tests for the Cast-based pattern (post-cleanup, e.g. after quantization)."""

    def test_replaces_cast_pattern_with_constant(self):
        import onnx

        kv_hidden = 256
        batch_dim_node = onnx.helper.make_node(
            "Shape", inputs=["query_states"], outputs=["batch_dim"], start=0, end=1
        )
        tail_const_node = onnx.helper.make_node(
            "Constant", inputs=[], outputs=["shape_tail"], value_ints=[0, kv_hidden]
        )
        concat_node = onnx.helper.make_node(
            "Concat", inputs=["batch_dim", "shape_tail"], outputs=["empty_shape"], axis=0
        )
        cos_node = onnx.helper.make_node(
            "ConstantOfShape", inputs=["empty_shape"], outputs=["cos"]
        )
        # Cast to FLOAT16 (to=10) — what quantization/cleanup emits instead of CastLike
        cast_node = onnx.helper.make_node(
            "Cast", inputs=["cos"], outputs=["empty_kv"], to=10
        )
        graph = onnx.helper.make_graph(
            [batch_dim_node, tail_const_node, concat_node, cos_node, cast_node],
            "cast_variant",
            inputs=[onnx.helper.make_tensor_value_info(
                "query_states", onnx.TensorProto.FLOAT16, [None, None, 64]
            )],
            outputs=[onnx.helper.make_tensor_value_info(
                "empty_kv", onnx.TensorProto.FLOAT16, [None, 0, kv_hidden]
            )],
        )
        proto = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 18)])
        model = ir.serde.deserialize_model(proto)

        counts_before = _count_ops(model)
        assert counts_before["ConstantOfShape"] == 1
        assert counts_before["Cast"] == 1

        rewrite(model, pattern_rewrite_rules=static_empty_kv_rules())

        counts_after = _count_ops(model)
        assert counts_after.get("Shape", 0) == 0, "Shape should be removed"
        assert counts_after.get("ConstantOfShape", 0) == 0, "ConstantOfShape should be removed"
        assert counts_after.get("Cast", 0) == 0, "Cast should be removed"
        assert counts_after.get("Constant", 0) >= 1, "static Constant should be present"

    def test_cast_output_constant_has_correct_shape(self):
        import onnx

        kv_hidden = 512
        batch_dim_node = onnx.helper.make_node(
            "Shape", inputs=["query_states"], outputs=["batch_dim"], start=0, end=1
        )
        tail_const_node = onnx.helper.make_node(
            "Constant", inputs=[], outputs=["shape_tail"], value_ints=[0, kv_hidden]
        )
        concat_node = onnx.helper.make_node(
            "Concat", inputs=["batch_dim", "shape_tail"], outputs=["empty_shape"], axis=0
        )
        cos_node = onnx.helper.make_node(
            "ConstantOfShape", inputs=["empty_shape"], outputs=["cos"]
        )
        cast_node = onnx.helper.make_node(
            "Cast", inputs=["cos"], outputs=["empty_kv"], to=10
        )
        graph = onnx.helper.make_graph(
            [batch_dim_node, tail_const_node, concat_node, cos_node, cast_node],
            "cast_shape_test",
            inputs=[onnx.helper.make_tensor_value_info(
                "query_states", onnx.TensorProto.FLOAT16, [None, None, 64]
            )],
            outputs=[onnx.helper.make_tensor_value_info(
                "empty_kv", onnx.TensorProto.FLOAT16, [None, 0, kv_hidden]
            )],
        )
        proto = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 18)])
        model = ir.serde.deserialize_model(proto)

        rewrite(model, pattern_rewrite_rules=static_empty_kv_rules())

        const_nodes = [n for n in model.graph if n.op_type == "Constant"]
        assert const_nodes, "no Constant node found after rewrite"
        const_node = const_nodes[-1]
        value_attr = const_node.attributes.get("value")
        assert value_attr is not None
        tensor = value_attr.value
        assert list(tensor.shape) == [1, 0, kv_hidden]
        assert tensor.dtype == ir.DataType.FLOAT16, f"expected FLOAT16, got {tensor.dtype}"
