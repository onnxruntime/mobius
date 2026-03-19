# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Rewrite rules for fusing RotaryEmbedding + Attention into GroupQueryAttention.

The standard decoder layer pattern applies RotaryEmbedding to Q and K
before feeding them into the ONNX ``Attention`` op (with KV cache).
The ``com.microsoft::GroupQueryAttention`` custom op fuses rotary
embedding into the attention kernel and replaces the explicit attention
bias with ``seqlens_k`` / ``total_sequence_length`` inputs computed
from the ``attention_mask`` graph input.

When the Q, K, and V projections are separate MatMuls that share the
same hidden_states input (and no QK norm is applied), the rule also
packs the three weight matrices into a single ``W_qkv`` and emits one
packed MatMul.  The packed QKV tensor is passed in the ``query`` slot
of ``GroupQueryAttention`` with ``key`` and ``value`` set to ``None``.
Models with QK norm (e.g. Qwen3) fall back to separate Q/K/V inputs.

These rules are **not applied by default**.  Apply them post-export::

    from mobius.rewrite_rules import group_query_attention_rules
    from onnxscript.rewriter import rewrite

    model = build("Qwen/Qwen3-0.6B")
    rewrite(model, pattern_rewrite_rules=group_query_attention_rules())
"""

from __future__ import annotations

import numpy as np
import onnx_ir as ir
from onnxscript.rewriter._basics import MatchResult
from onnxscript.rewriter._rewrite_rule import (
    RewriteRuleClassBase,
    RewriteRuleSet,
)


def _extract_weight(matmul_node):
    """Extract the weight numpy array from a MatMul node.

    Handles the ``Transpose(weight, perm=[1,0]) → MatMul`` pattern
    commonly emitted by ``Linear`` layers.

    Returns:
        A tuple ``(numpy_array, is_transposed)`` where
        ``is_transposed`` is ``True`` when the weight was preceded by a
        ``Transpose``.  Returns ``(None, False)`` if the weight cannot
        be extracted (e.g. not a constant initializer).
    """
    weight_input = matmul_node.inputs[1]
    producer = weight_input.producer()

    if producer is not None and producer.op_type == "Transpose":
        perm = producer.attributes.get("perm", None)
        if perm is not None and list(perm.value) == [1, 0]:
            raw_weight = producer.inputs[0]
            if raw_weight.const_value is None:
                return None, False
            return raw_weight.const_value.numpy(), True

    # Direct weight (no Transpose)
    if weight_input.const_value is None:
        return None, False
    return weight_input.const_value.numpy(), False


class RotaryAttentionToGQA(RewriteRuleClassBase):
    """Replace RotaryEmbedding + Attention with GroupQueryAttention.

    **Matched pattern:**

    .. code-block:: text

        q_rot = RotaryEmbedding(q_pre, cos, sin)
        k_rot = RotaryEmbedding(k_pre, cos, sin)
        attn_out, present_key, present_value = Attention(
            q_rot, k_rot, v, attention_bias, past_key, past_value,
        )

    Where ``past_key`` and ``past_value`` are graph inputs (decoder
    attention with KV cache), and ``cos`` / ``sin`` are position-gathered
    from rotary cache tables via ``Gather``.

    **Replacement (packed QKV, when Q/K/V share the same hidden_states
    input and no QK norm is present):**

    .. code-block:: text

        W_qkv = concatenate([W_q, W_k, W_v])
        packed_qkv = MatMul(hidden_states, W_qkv)
        attn_out, present_key, present_value = GroupQueryAttention(
            packed_qkv, None, None, past_key, past_value,
            seqlens_k, total_seq_len, cos_cache, sin_cache,
            num_heads=..., kv_num_heads=..., do_rotary=1,
        )

    **Replacement (separate Q/K/V, fallback for models with QK norm):**

    .. code-block:: text

        attn_out, present_key, present_value = GroupQueryAttention(
            q_pre, k_pre, v, past_key, past_value,
            seqlens_k, total_seq_len, cos_cache, sin_cache,
            num_heads=..., kv_num_heads=..., do_rotary=1,
        )

    ``cos_cache`` and ``sin_cache`` are the original rotary embedding
    tables (traced back through the ``Gather`` nodes).
    """

    def __init__(self):
        super().__init__()
        # Cached graph-level values shared across all GQA replacements
        self._seqlens_k = None
        self._total_seq_len = None
        self._cos_cache = None
        self._sin_cache = None

    # ------------------------------------------------------------------ pattern

    def pattern(self, op, q_pre, k_pre, v, attention_bias, past_key, past_value, cos, sin):
        q_rot = op.RotaryEmbedding(
            q_pre,
            cos,
            sin,
            _allow_other_attributes=True,
        )
        k_rot = op.RotaryEmbedding(
            k_pre,
            cos,
            sin,
            _allow_other_attributes=True,
        )
        return op.Attention(
            q_rot,
            k_rot,
            v,
            attention_bias,
            past_key,
            past_value,
            _allow_other_attributes=True,
            _outputs=["attn_out", "present_key", "present_value"],
        )

    # ------------------------------------------------------------------ check

    def check(self, context, attn_out, cos, sin, past_key, past_value, **_):
        result = MatchResult()

        attn = attn_out.producer()
        if attn.attributes.get_float("scale", None) is None:
            return result.fail("Missing scale attribute on Attention")
        if attn.attributes.get_int("q_num_heads", None) is None:
            return result.fail("Missing q_num_heads on Attention")
        if attn.attributes.get_int("kv_num_heads", None) is None:
            return result.fail("Missing kv_num_heads on Attention")

        # cos/sin must come from Gather (position-indexed cache tables)
        cos_prod = cos.producer()
        sin_prod = sin.producer()
        if cos_prod is None or cos_prod.op_type != "Gather":
            return result.fail("cos must be Gather-produced")
        if sin_prod is None or sin_prod.op_type != "Gather":
            return result.fail("sin must be Gather-produced")

        # past_key/past_value must be graph inputs (not None, not computed)
        # This distinguishes decoder attention from vision-encoder attention
        if past_key is None or past_value is None:
            return result.fail("No KV cache inputs")
        if past_key.producer() is not None:
            return result.fail("past_key is not a graph input")
        if past_value.producer() is not None:
            return result.fail("past_value is not a graph input")

        return result

    # ------------------------------------------------------------ pack QKV

    def _try_pack_qkv(self, op, q_pre, k_pre, v):
        """Try to pack Q/K/V projections into a single MatMul.

        Traces back from ``q_pre``, ``k_pre``, ``v`` to their producing
        MatMul ops.  If all three share the same hidden_states input and
        no QK norm is present, concatenates the weight matrices and
        returns a single packed QKV output.

        Returns:
            The packed QKV ``ir.Value`` on success, or ``None`` when
            packing is not possible (e.g. QK norm present, weights not
            materialised, or different hidden_states inputs).
        """
        # 1. All three must come directly from MatMul (no QK norm)
        q_matmul = q_pre.producer()
        k_matmul = k_pre.producer()
        v_matmul = v.producer()
        for node in (q_matmul, k_matmul, v_matmul):
            if node is None or node.op_type != "MatMul":
                return None

        # 2. All three MatMuls must share the same first input
        if not (q_matmul.inputs[0] is k_matmul.inputs[0] is v_matmul.inputs[0]):
            return None

        hidden_states = q_matmul.inputs[0]

        # 3. Extract weights, handling optional Transpose
        weights_np = []
        is_transposed = None
        for matmul in (q_matmul, k_matmul, v_matmul):
            w_value, transposed = _extract_weight(matmul)
            if w_value is None:
                return None
            # All weights must follow the same pattern
            if is_transposed is None:
                is_transposed = transposed
            elif is_transposed != transposed:
                return None
            weights_np.append(w_value)

        # 4. Concatenate along the output dimension
        # Transposed weights: shape (out_features, hidden_size) → axis=0
        # Non-transposed weights: shape (hidden_size, out_features) → axis=1
        concat_axis = 0 if is_transposed else 1
        w_qkv_np = np.concatenate(weights_np, axis=concat_axis)

        # 5. Create the packed MatMul
        packed_weight = op.Constant(value=ir.Tensor(w_qkv_np))
        if is_transposed:
            # Emit Transpose + MatMul (same pattern as the original)
            packed_weight_t = op.Transpose(packed_weight, perm=[1, 0])
            return op.MatMul(hidden_states, packed_weight_t)
        return op.MatMul(hidden_states, packed_weight)

    # ------------------------------------------------------------------ rewrite

    def rewrite(
        self,
        op,
        q_pre,
        k_pre,
        v,
        attention_bias,
        past_key,
        past_value,
        cos,
        sin,
        attn_out,
        present_key,
        present_value,
        **_,
    ):
        attn = attn_out.producer()
        scale = attn.attributes.get_float("scale")
        q_num_heads = attn.attributes.get_int("q_num_heads")
        kv_num_heads = attn.attributes.get_int("kv_num_heads")

        # Trace cos/sin back through Gather to the cache table initializers
        if self._cos_cache is None:
            self._cos_cache = cos.producer().inputs[0]
            self._sin_cache = sin.producer().inputs[0]

        # Build seqlens_k and total_sequence_length once (shared)
        if self._seqlens_k is None:
            graph = attn.graph
            attention_mask = None
            for gi in graph.inputs:
                if gi.name == "attention_mask":
                    attention_mask = gi
                    break

            # seqlens_k = Cast(ReduceSum(attention_mask, axis=1) - 1, INT32)
            axis = op.Constant(value_ints=[1])
            reduce_sum = op.ReduceSum(attention_mask, axis)
            one = op.Constant(value_ints=[1])
            self._seqlens_k = op.Cast(
                op.Sub(reduce_sum, one),
                to=6,
            )

            # total_seq_len = Cast(Gather(Shape(attention_mask), 1), INT32)
            mask_shape = op.Shape(attention_mask)
            idx_1 = op.Constant(value_int=1)
            self._total_seq_len = op.Cast(
                op.Gather(mask_shape, idx_1),
                to=6,
            )

        # Try packed QKV path; fall back to separate Q/K/V
        packed_qkv = self._try_pack_qkv(op, q_pre, k_pre, v)
        if packed_qkv is not None:
            gqa_inputs = [
                packed_qkv,
                None,
                None,
                past_key,
                past_value,
                self._seqlens_k,
                self._total_seq_len,
                self._cos_cache,
                self._sin_cache,
            ]
        else:
            gqa_inputs = [
                q_pre,
                k_pre,
                v,
                past_key,
                past_value,
                self._seqlens_k,
                self._total_seq_len,
                self._cos_cache,
                self._sin_cache,
            ]

        # Create GroupQueryAttention (3 outputs match Attention's 3)
        outputs = op.op_multi_out(
            "GroupQueryAttention",
            inputs=gqa_inputs,
            domain="com.microsoft",
            attributes={
                "num_heads": q_num_heads,
                "kv_num_heads": kv_num_heads,
                "scale": scale,
                "do_rotary": 1,
                "rotary_interleaved": 0,
            },
            num_outputs=3,
        )

        return outputs[0], outputs[1], outputs[2]


def group_query_attention_rules() -> RewriteRuleSet:
    """Return rules that fuse RotaryEmbedding + Attention into GQA.

    These rules match the RotaryEmbedding -> Attention pattern common
    in decoder layers and replace it with the Microsoft
    ``GroupQueryAttention`` custom op with ``do_rotary=1``.

    Returns:
        :class:`RewriteRuleSet` containing the RotaryEmbedding+Attention
        fusion rule.
    """
    return RewriteRuleSet([RotaryAttentionToGQA().rule()])
