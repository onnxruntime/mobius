# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Graph-construction coverage for differential grouped-query attention."""

from __future__ import annotations

import numpy as np
import onnx_ir as ir

from mobius._testing import count_op_type, create_test_builder, create_test_input
from mobius._testing.ort_inference import OnnxModelSession
from mobius.components._differential_attention import DifferentialGQAAttention


class TestDifferentialGQAAttention:
    """Verify the reusable four-branch differential-attention primitive."""

    def test_builds_four_striped_attention_reads(self) -> None:
        component = DifferentialGQAAttention(
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            depth=3,
            local_window_size=4,
        )
        builder, op, graph = create_test_builder()
        query = create_test_input(builder, "query", [2, 3, 4, 8], ir.DataType.BFLOAT16)
        key = create_test_input(builder, "key", [2, 5, 2, 8], ir.DataType.BFLOAT16)
        value = create_test_input(builder, "value", [2, 5, 2, 8], ir.DataType.BFLOAT16)
        attention_mask = create_test_input(builder, "attention_mask", [2, 5], ir.DataType.INT64)

        output = component(op, query, key, value, attention_mask)
        builder.add_output(output, "output")

        # Two Q/K stripes each read both V stripes, exactly matching the
        # source's four FlashAttention calls. The runtime selects an unmasked
        # compact or source-faithful padded branch for each read.
        assert count_op_type(graph, "If") == 4
        assert sum(node.op_type == "GroupQueryAttention" for node in graph.all_nodes()) == 4
        assert sum(node.op_type == "Attention" for node in graph.all_nodes()) == 4
        assert count_op_type(graph, "Exp") == 2
        assert "subln.weight" in dict(component.named_parameters())

    def test_lambda_vectors_remain_float32_when_model_is_lowered(self) -> None:
        component = DifferentialGQAAttention(
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            depth=0,
        )

        for parameter in (
            component.lambda_q1,
            component.lambda_k1,
            component.lambda_q2,
            component.lambda_k2,
        ):
            assert parameter._keep_float32

    def test_native_attention_selects_valid_runtime_mask_rank_for_batch_prefill(self) -> None:
        """All-valid and padded batch prefills execute through their respective branches."""
        component = DifferentialGQAAttention(
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            depth=3,
            local_window_size=4,
        )
        builder, op, graph = create_test_builder()
        query = create_test_input(builder, "query", [2, 3, 4, 8])
        key = create_test_input(builder, "key", [2, 3, 2, 8])
        value = create_test_input(builder, "value", [2, 3, 2, 8])
        attention_mask = create_test_input(builder, "attention_mask", [2, 3], ir.DataType.INT64)
        for parameter in component.parameters():
            parameter.const_value = ir.tensor(np.zeros(parameter.shape, dtype=np.float32))
        builder.add_output(component(op, query, key, value, attention_mask), "output")

        session = OnnxModelSession(ir.Model(graph, ir_version=10), device="cpu")
        feeds = {
            "query": np.ones((2, 3, 4, 8), dtype=np.float32),
            "key": np.ones((2, 3, 2, 8), dtype=np.float32),
            "value": np.ones((2, 3, 2, 8), dtype=np.float32),
        }
        for mask in (np.ones((2, 3), dtype=np.int64), np.array([[1, 1, 1], [0, 1, 1]])):
            assert session.run({**feeds, "attention_mask": mask})["output"].shape == (2, 3, 4, 8)

    def test_native_attention_keeps_batched_decode_cache_in_b_n_s_h_layout(self) -> None:
        """A batch-two decode accepts striped 4-D past KV through both mask branches."""
        component = DifferentialGQAAttention(
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            depth=3,
            local_window_size=4,
        )
        builder, op, graph = create_test_builder()
        query = create_test_input(builder, "query", [2, 1, 4, 8])
        key = create_test_input(builder, "key", [2, 1, 2, 8])
        value = create_test_input(builder, "value", [2, 1, 2, 8])
        attention_mask = create_test_input(builder, "attention_mask", [2, 4], ir.DataType.INT64)
        past_key = create_test_input(builder, "past_key", [2, 3, 2, 8])
        past_value = create_test_input(builder, "past_value", [2, 3, 2, 8])
        for parameter in component.parameters():
            parameter.const_value = ir.tensor(np.zeros(parameter.shape, dtype=np.float32))
        builder.add_output(
            component(op, query, key, value, attention_mask, (past_key, past_value)),
            "output",
        )

        session = OnnxModelSession(ir.Model(graph, ir_version=10), device="cpu")
        feeds = {
            "query": np.ones((2, 1, 4, 8), dtype=np.float32),
            "key": np.ones((2, 1, 2, 8), dtype=np.float32),
            "value": np.ones((2, 1, 2, 8), dtype=np.float32),
            "past_key": np.ones((2, 3, 2, 8), dtype=np.float32),
            "past_value": np.ones((2, 3, 2, 8), dtype=np.float32),
        }
        for mask in (np.ones((2, 4), dtype=np.int64), np.array([[1, 1, 1, 1], [0, 1, 1, 1]])):
            assert session.run({**feeds, "attention_mask": mask})["output"].shape == (2, 1, 4, 8)

        gqa = next(node for node in graph.all_nodes() if node.op_type == "GroupQueryAttention")
        assert gqa.inputs[5].producer().inputs[0].producer().op_type == "Sub"
        assert gqa.inputs[5].producer().inputs[0].producer().inputs[0].producer().op_type == "ReduceSum"
