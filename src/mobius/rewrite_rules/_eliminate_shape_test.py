# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import numpy as np
import onnx_ir as ir
from onnxscript.rewriter import rewrite
from onnxscript.rewriter._rewrite_rule import RewriteRuleSet

from mobius._builder import build_from_module
from mobius._configs import ArchitectureConfig
from mobius._registry import registry
from mobius._testing.ort_inference import OnnxModelSession
from mobius.rewrite_rules import eliminate_shape_rules
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


class TestEliminateShapeRules:
    def test_returns_rule_set(self):
        rules = eliminate_shape_rules()
        assert isinstance(rules, RewriteRuleSet)

    def test_eliminates_attention_mask_shape(self):
        """Shape+Gather on attention_mask is replaced by ReduceSum+ReduceMax."""
        config = _tiny_qwen3_config()
        model_module = registry.get("qwen3")(config)
        pkg = build_from_module(model_module, config)
        model = pkg["model"]

        # The model may use Shape(x, start=1, end=2) or Shape(x)+Gather(x,1);
        # both patterns should be matched.
        mask_shapes = sum(
            1
            for n in model.graph
            if n.op_type == "Shape"
            and n.inputs[0] is not None
            and n.inputs[0].name is not None
            and "mask" in n.inputs[0].name
        )
        assert mask_shapes >= 1, "Expected at least one Shape node on attention_mask"

        rewrite(model, pattern_rewrite_rules=eliminate_shape_rules())

        ops_after = count_ops(model)
        # All Shape nodes referencing attention_mask should be gone
        mask_shapes_after = sum(
            1
            for n in model.graph
            if n.op_type == "Shape"
            and n.inputs[0] is not None
            and n.inputs[0].name is not None
            and "mask" in n.inputs[0].name
        )
        assert mask_shapes_after == 0, (
            f"Expected all attention_mask Shape nodes to be eliminated, "
            f"but {mask_shapes_after} remain"
        )
        # ReduceSum and ReduceMax are inserted as replacement
        assert ops_after["ReduceSum"] >= 1, "Expected ReduceSum inserted by rule"
        assert ops_after["ReduceMax"] >= 1, "Expected ReduceMax inserted by rule"

    def test_input_ids_seq_len_shape_present_and_attention_mask_shape_present(self):
        """create_padding_mask() reads q_len from input_ids and total_len from attention_mask.

        Shape(input_ids, start=1, end=2) must be present (for the query length) and
        Shape(attention_mask, start=1, end=2) must also be present (for total length).
        Both are correct: q_len != total_len during decode (dynamic-cache) steps.
        """
        config = _tiny_qwen3_config()
        model_module = registry.get("qwen3")(config)
        pkg = build_from_module(model_module, config)
        model = pkg["model"]

        input_ids_seq_len_shapes = sum(
            1
            for n in model.graph
            if n.op_type == "Shape"
            and n.inputs[0] is not None
            and n.inputs[0].name == "input_ids"
            and n.attributes.get_int("start", 0) == 1
            and n.attributes.get_int("end", -1) == 2
        )
        assert input_ids_seq_len_shapes >= 1, (
            f"Expected at least one Shape(input_ids, start=1, end=2) for q_len, "
            f"got {input_ids_seq_len_shapes}"
        )

        mask_seq_len_shapes = sum(
            1
            for n in model.graph
            if n.op_type == "Shape"
            and n.inputs[0] is not None
            and n.inputs[0].name is not None
            and "mask" in n.inputs[0].name
            and n.attributes.get_int("start", 0) == 1
            and n.attributes.get_int("end", -1) == 2
        )
        assert mask_seq_len_shapes >= 1, (
            f"Expected at least one Shape(attention_mask, start=1, end=2) for total_len, "
            f"got {mask_seq_len_shapes}"
        )

    def test_rewritten_model_runs_with_ort(self):
        """Shape-eliminated model can be run with ORT (semantically equivalent)."""
        config = _tiny_qwen3_config()
        model_module = registry.get("qwen3")(config)
        pkg = build_from_module(model_module, config)
        fill_random_weights(pkg["model"])

        rewrite(pkg["model"], pattern_rewrite_rules=eliminate_shape_rules())

        session = OnnxModelSession(pkg["model"])
        # Use all-ones attention_mask: ReduceSum gives seq_len which equals
        # Shape(attention_mask)[1] — the replacement is semantically exact.
        feeds = make_prefill_feeds(session, seq_len=3)
        result = session.run(feeds)
        assert result is not None

    def test_ort_output_matches_original(self):
        """Rewritten model produces identical logits to the original graph."""
        config = _tiny_qwen3_config()
        model_module = registry.get("qwen3")(config)
        pkg = build_from_module(model_module, config)

        # Fill with deterministic weights once
        rng = np.random.default_rng(42)
        for init in pkg["model"].graph.initializers.values():
            if init.const_value is None:
                shape = list(init.shape or [1])
                init.const_value = ir.Tensor(rng.standard_normal(shape).astype(np.float32))

        # Run original model to get baseline logits
        session_orig = OnnxModelSession(pkg["model"])
        feeds = make_prefill_feeds(session_orig, seq_len=3)
        logits_orig = session_orig.run(feeds)["logits"]
        session_orig.close()

        # Apply the rule to the same model in-place and re-run
        rewrite(pkg["model"], pattern_rewrite_rules=eliminate_shape_rules())
        session_rewritten = OnnxModelSession(pkg["model"])
        logits_rewritten = session_rewritten.run(feeds)["logits"]

        np.testing.assert_allclose(
            logits_orig,
            logits_rewritten,
            atol=1e-5,
            rtol=1e-4,
            err_msg="Logits differ after Shape elimination — replacement is not equivalent",
        )
