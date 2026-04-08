# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Rewrite rule for separating fused RoPE from GroupQueryAttention.

DML's ``GroupQueryAttention`` kernel does not support ``do_rotary=1``
(fused rotary position embeddings).  This rule decomposes a fused GQA
node back to explicit ``RotaryEmbedding`` ops applied before GQA.

**Matched pattern:**

.. code-block:: text

    out, pk, pv = GroupQueryAttention(
        q, k, v, past_key, past_value, seqlens_k, total_seq,
        cos_cache, sin_cache, do_rotary=1, ...
    )

**Replacement:**

.. code-block:: text

    gathered_cos = Gather(cos_cache, position_ids)
    gathered_sin = Gather(sin_cache, position_ids)
    q_rot = RotaryEmbedding(q, gathered_cos, gathered_sin, num_heads=q_num_heads)
    k_rot = RotaryEmbedding(k, gathered_cos, gathered_sin, num_heads=kv_num_heads)
    out, pk, pv = GroupQueryAttention(
        q_rot, k_rot, v, past_key, past_value, seqlens_k, total_seq,
        do_rotary=0, ...
    )

The ``position_ids`` graph input is looked up by name and used to index
into the cosine/sine cache tables before applying the standard
``RotaryEmbedding`` op (standard ONNX opset 23).

These rules are applied automatically by
:func:`~mobius._optimizations.optimize_model` for EPs that do not support
fused RoPE (``supports_fused_rope=False``, e.g. DML).  They can also be
applied manually::

    from mobius.rewrite_rules import separate_rope_rules
    from onnxscript.rewriter import rewrite

    model = build("Qwen/Qwen3-0.6B", execution_provider="dml")
    rewrite(model, pattern_rewrite_rules=separate_rope_rules())
"""

from __future__ import annotations

from onnxscript.rewriter._basics import MatchResult
from onnxscript.rewriter._rewrite_rule import RewriteRuleClassBase, RewriteRuleSet


class GQASeparateRoPE(RewriteRuleClassBase):
    """Decompose fused GQA+RoPE into separate RotaryEmbedding + GQA.

    Matches ``GroupQueryAttention`` with ``do_rotary=1`` and separates the
    rotary embeddings into explicit ``RotaryEmbedding`` ops before GQA.

    Used for DML which does not support fused rotary in the GQA kernel.
    """

    # ------------------------------------------------------------------ pattern

    def pattern(self, op, q, k, v):
        # Match any GQA; past_kv, seqlens_k, cos/sin cache are accessed
        # via the producer node in check() and rewrite().
        return op.GroupQueryAttention(
            q,
            k,
            v,
            _domain="com.microsoft",
            _allow_other_inputs=True,
            _allow_other_attributes=True,
            _outputs=["gqa_out", "present_key", "present_value"],
        )

    # ------------------------------------------------------------------ check

    def check(self, context, q, k, v, gqa_out, **_):
        result = MatchResult()
        gqa_node = gqa_out.producer()
        if gqa_node is None:
            return result.fail("No GQA producer")

        # Only decompose when do_rotary=1 (fused RoPE)
        do_rotary = gqa_node.attributes.get_int("do_rotary", 0)
        if do_rotary != 1:
            return result.fail("do_rotary != 1 — RoPE already separate")

        # cos_cache and sin_cache must be present (inputs at indices 7 and 8)
        inputs = gqa_node.inputs
        if len(inputs) < 9:
            return result.fail("GQA has fewer than 9 inputs — no cos/sin cache")
        if inputs[7] is None or inputs[8] is None:
            return result.fail("cos_cache or sin_cache is None")

        # position_ids must exist as a named graph input (needed for re-gathering)
        if not any(gi.name == "position_ids" for gi in gqa_node.graph.inputs):
            return result.fail("No position_ids graph input")

        return result

    # ------------------------------------------------------------------ rewrite

    def rewrite(self, op, q, k, v, gqa_out, present_key, present_value, **_):
        gqa_node = gqa_out.producer()

        # Collect all GQA attributes; update do_rotary to 0.
        attrs = {key: gqa_node.attributes[key].value for key in gqa_node.attributes}
        attrs["do_rotary"] = 0
        num_heads = attrs.get("num_heads", 1)
        kv_num_heads = attrs.get("kv_num_heads", 1)

        cos_cache = gqa_node.inputs[7]
        sin_cache = gqa_node.inputs[8]

        # Look up the position_ids graph input by name.
        position_ids = next(gi for gi in gqa_node.graph.inputs if gi.name == "position_ids")

        # Gather per-position cos/sin from the cache tables.
        # position_ids: (batch, seq_len)
        # cos_cache:    (max_pos, rotary_dim)
        # → Gather axis=0 → (batch, seq_len, rotary_dim)
        gathered_cos = op.Gather(cos_cache, position_ids)
        gathered_sin = op.Gather(sin_cache, position_ids)

        # Apply standard ONNX RotaryEmbedding to Q and K.
        q_rot = op.RotaryEmbedding(q, gathered_cos, gathered_sin, num_heads=num_heads)
        k_rot = op.RotaryEmbedding(k, gathered_cos, gathered_sin, num_heads=kv_num_heads)

        # Rebuild GQA with rotated Q/K.  Retain inputs [3..6]:
        # past_key, past_value, seqlens_k, total_sequence_length.
        remaining = list(gqa_node.inputs[3:7])

        outputs = op.op_multi_out(
            "GroupQueryAttention",
            inputs=[q_rot, k_rot, v, *remaining],
            domain="com.microsoft",
            attributes=attrs,
            num_outputs=3,
        )
        return outputs[0], outputs[1], outputs[2]


def separate_rope_rules() -> RewriteRuleSet:
    """Return a rule set that separates fused GQA+RoPE into distinct ops.

    Used for DML which does not support ``do_rotary=1`` in GQA.

    Returns:
        :class:`RewriteRuleSet` containing the :class:`GQASeparateRoPE` rule.
    """
    return RewriteRuleSet([GQASeparateRoPE().rule()])
