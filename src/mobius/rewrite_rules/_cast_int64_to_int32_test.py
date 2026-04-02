# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import onnx_ir as ir
from onnxscript.rewriter import rewrite
from onnxscript.rewriter._rewrite_rule import RewriteRuleSet

from mobius._builder import build_from_module
from mobius._configs import ArchitectureConfig
from mobius._registry import registry
from mobius._testing.ort_inference import OnnxModelSession
from mobius.rewrite_rules import cast_int64_to_int32_rules
from mobius.rewrite_rules._testing_utils import (
    count_ops,
    fill_random_weights,
    make_prefill_feeds,
)


def _tiny_qwen3_config() -> ArchitectureConfig:
    return ArchitectureConfig(
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        num_hidden_layers=2,
        vocab_size=256,
        max_position_embeddings=128,
        hidden_act="silu",
        rms_norm_eps=1e-6,
        rope_type="default",
        rope_theta=10000.0,
        pad_token_id=0,
    )


class TestCastInt64ToInt32Rules:
    def test_returns_rule_set(self):
        rules = cast_int64_to_int32_rules()
        assert isinstance(rules, RewriteRuleSet)

    def test_casts_gather_indices_to_int32(self):
        """INT64 Gather indices are replaced with Cast(INT64 → INT32) + Gather."""
        config = _tiny_qwen3_config()
        model_module = registry.get("qwen3")(config)
        pkg = build_from_module(model_module, config)
        model = pkg["model"]

        ops_before = count_ops(model)
        gather_before = ops_before["Gather"]
        cast_before = ops_before["Cast"]
        assert gather_before >= 1, "Expected at least one Gather node before rewrite"

        # Count how many Gather nodes have INT64 indices
        int64_gather_count = sum(
            1
            for n in model.graph
            if n.op_type == "Gather"
            and n.inputs[1] is not None
            and n.inputs[1].dtype == ir.DataType.INT64
        )
        assert int64_gather_count >= 1, "Expected at least one Gather with INT64 indices"

        rewrite(model, pattern_rewrite_rules=cast_int64_to_int32_rules())

        ops_after = count_ops(model)
        # Gather count is unchanged — only the index dtype changes
        assert ops_after["Gather"] == gather_before, (
            f"Gather count changed unexpectedly: {gather_before} → {ops_after['Gather']}"
        )
        # Each INT64 Gather got a new Cast(INT64→INT32) node
        assert ops_after["Cast"] == cast_before + int64_gather_count, (
            f"Expected {cast_before + int64_gather_count} Cast nodes after rewrite, "
            f"got {ops_after['Cast']}"
        )

    def test_does_not_cast_int32_gather_indices(self):
        """Gather nodes whose indices are already INT32 are not modified."""
        config = _tiny_qwen3_config()
        model_module = registry.get("qwen3")(config)
        pkg = build_from_module(model_module, config)
        model = pkg["model"]

        # Apply the rule
        rewrite(model, pattern_rewrite_rules=cast_int64_to_int32_rules())

        # After the rewrite, no Gather node should have INT64 index inputs
        int64_gather_remaining = sum(
            1
            for n in model.graph
            if n.op_type == "Gather"
            and n.inputs[1] is not None
            and n.inputs[1].dtype == ir.DataType.INT64
        )
        assert int64_gather_remaining == 0, (
            f"Found {int64_gather_remaining} Gather nodes with INT64 indices "
            "after applying cast_int64_to_int32_rules"
        )

    def test_rewritten_model_runs_with_ort(self):
        """INT32-index model can be serialized and run with ORT."""
        config = _tiny_qwen3_config()
        model_module = registry.get("qwen3")(config)
        pkg = build_from_module(model_module, config)
        fill_random_weights(pkg["model"])

        rewrite(pkg["model"], pattern_rewrite_rules=cast_int64_to_int32_rules())

        session = OnnxModelSession(pkg["model"])
        feeds = make_prefill_feeds(session, seq_len=3)
        result = session.run(feeds)
        assert result is not None

    def test_ort_output_matches_original(self):
        """INT32 Gather indices produce identical outputs to INT64 indices."""
        config = _tiny_qwen3_config()
        model_module = registry.get("qwen3")(config)
        pkg = build_from_module(model_module, config)

        # Fill with deterministic weights once
        rng = np.random.default_rng(42)
        for init in pkg["model"].graph.initializers.values():
            if init.const_value is None:
                shape = list(init.shape or [1])
                init.const_value = ir.Tensor(rng.standard_normal(shape).astype(np.float32))

        # Run original (INT64 indices) to get baseline
        session_orig = OnnxModelSession(pkg["model"])
        feeds = make_prefill_feeds(session_orig, seq_len=3)
        logits_orig = session_orig.run(feeds)["logits"]
        session_orig.close()

        # Apply the cast rule in-place and re-run
        rewrite(pkg["model"], pattern_rewrite_rules=cast_int64_to_int32_rules())
        session_rewritten = OnnxModelSession(pkg["model"])
        logits_rewritten = session_rewritten.run(feeds)["logits"]

        np.testing.assert_allclose(
            logits_orig,
            logits_rewritten,
            atol=1e-5,
            rtol=1e-4,
            err_msg="Logits differ after INT64→INT32 cast — replacement is not equivalent",
        )
