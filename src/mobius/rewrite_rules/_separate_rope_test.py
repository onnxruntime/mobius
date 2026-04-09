# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

from onnxscript.rewriter import rewrite
from onnxscript.rewriter._rewrite_rule import RewriteRuleSet

from mobius._builder import build_from_module
from mobius._configs import ArchitectureConfig
from mobius.models.base import CausalLMModel
from mobius.rewrite_rules import group_query_attention_rules, separate_rope_rules
from mobius.rewrite_rules._testing_utils import count_ops

# Tiny GQA-friendly config — no QK norm so GQA fusion works
_CONFIG = ArchitectureConfig(
    hidden_size=64,
    intermediate_size=128,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=16,
    num_hidden_layers=1,
    vocab_size=256,
    max_position_embeddings=128,
    hidden_act="silu",
    rms_norm_eps=1e-6,
    rope_type="default",
    rope_theta=10000.0,
    pad_token_id=0,
)


def _build_gqa_model() -> object:
    """Build a model with GQA fusion applied (do_rotary=1), without QKV packing.

    SeparateRoPE operates on a GQA-fused model that still has separate Q/K/V
    projections — not the packed form.  Build with the default EP (no
    optimization pipeline), then manually apply only the GQA fusion rules so
    that do_rotary=1 is set but k/v inputs remain non-None.
    """
    mod = CausalLMModel(_CONFIG)
    pkg = build_from_module(mod, _CONFIG, execution_provider="default")
    model = pkg["model"]
    # Apply only GQA fusion (not PackQKV) so k/v inputs remain non-None.
    rewrite(model, pattern_rewrite_rules=group_query_attention_rules())
    return model


class TestSeparateRopeRules:
    def test_returns_rule_set(self):
        rules = separate_rope_rules()
        assert isinstance(rules, RewriteRuleSet)

    def test_separate_rope_decomposes_do_rotary(self):
        """GQA with do_rotary=1 → RotaryEmbedding + GQA with do_rotary=0."""
        model = _build_gqa_model()
        counts_before = count_ops(model)
        # Default CPU build fuses RoPE into GQA
        assert counts_before.get("GroupQueryAttention", 0) == 1
        # Verify do_rotary=1
        gqa_node = next(n for n in model.graph if n.op_type == "GroupQueryAttention")
        assert gqa_node.attributes.get_int("do_rotary", 0) == 1

        rewrite(model, pattern_rewrite_rules=separate_rope_rules())

        counts_after = count_ops(model)
        # GQA still present, do_rotary should now be 0
        assert counts_after.get("GroupQueryAttention", 0) == 1
        gqa_after = next(n for n in model.graph if n.op_type == "GroupQueryAttention")
        assert gqa_after.attributes.get_int("do_rotary", 0) == 0

        # RotaryEmbedding nodes should appear (one for Q, one for K)
        assert counts_after.get("RotaryEmbedding", 0) == 2

    def test_separate_rope_adds_gather_nodes(self):
        """SeparateRoPE inserts Gather nodes to index cos/sin from cache tables."""
        model = _build_gqa_model()
        gather_before = count_ops(model).get("Gather", 0)

        rewrite(model, pattern_rewrite_rules=separate_rope_rules())

        gather_after = count_ops(model).get("Gather", 0)
        # Two new Gather nodes for cos and sin cache lookups
        assert gather_after == gather_before + 2

    def test_separate_rope_idempotent(self):
        """Applying SeparateRoPE twice does not change anything on the second pass."""
        model = _build_gqa_model()
        rewrite(model, pattern_rewrite_rules=separate_rope_rules())
        counts_after_first = dict(count_ops(model))

        rewrite(model, pattern_rewrite_rules=separate_rope_rules())
        counts_after_second = dict(count_ops(model))

        assert counts_after_first == counts_after_second

    def test_no_match_without_do_rotary(self):
        """SeparateRoPE does not match a GQA that already has do_rotary=0."""
        model = _build_gqa_model()
        # Apply SeparateRoPE to get do_rotary=0
        rewrite(model, pattern_rewrite_rules=separate_rope_rules())
        counts_before = dict(count_ops(model))

        # Applying again should be a no-op
        rewrite(model, pattern_rewrite_rules=separate_rope_rules())
        assert dict(count_ops(model)) == counts_before
