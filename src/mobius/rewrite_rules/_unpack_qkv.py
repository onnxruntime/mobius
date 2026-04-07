# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Rewrite rule for unpacking packed QKV back into separate Q/K/V projections.

DML's ``GroupQueryAttention`` kernel does not support packed QKV (a single
fused ``MatMul`` for Q+K+V combined, passed in the query slot with ``key``
and ``value`` set to ``None``).  This rule reverses the ``PackQKVForGQA``
and ``PackQKVWithBiasForGQA`` fusions by splitting the packed weight (and
optional packed bias) back into three separate weight matrices.

**Matched pattern (no-bias, from PackQKVForGQA):**

.. code-block:: text

    packed_qkv = MatMul(hidden, Transpose(W_qkv))
    out, pk, pv = GroupQueryAttention(packed_qkv, None, None, ...)

**Matched pattern (with bias, from PackQKVWithBiasForGQA):**

.. code-block:: text

    packed_qkv = Add(MatMul(hidden, Transpose(W_qkv)), bias_qkv)
    out, pk, pv = GroupQueryAttention(packed_qkv, None, None, ...)

Where ``W_qkv`` concatenates W_q, W_k, W_v along axis=0, and ``bias_qkv``
concatenates bias_q, bias_k, bias_v along axis=0.

**Replacement (no-bias):**

.. code-block:: text

    q = MatMul(hidden, Transpose(W_q))
    k = MatMul(hidden, Transpose(W_k))
    v = MatMul(hidden, Transpose(W_v))
    out, pk, pv = GroupQueryAttention(q, k, v, ...)

**Replacement (with bias):**

.. code-block:: text

    q = Add(MatMul(hidden, Transpose(W_q)), bias_q)
    k = Add(MatMul(hidden, Transpose(W_k)), bias_k)
    v = Add(MatMul(hidden, Transpose(W_v)), bias_v)
    out, pk, pv = GroupQueryAttention(q, k, v, ...)

The weight split sizes are derived from ``num_heads``, ``kv_num_heads``, and
the packed weight shape:
- ``q_size = num_heads * head_dim``
- ``k_size = kv_num_heads * head_dim``
- ``v_size = kv_num_heads * head_dim``
- ``head_dim = total_out // (num_heads + 2 * kv_num_heads)``

These rules are applied automatically by
:func:`~mobius._optimizations.optimize_model` for EPs that do not support
fused RoPE (``supports_fused_rope=False``, e.g. DML).  They can also be
applied manually::

    from mobius.rewrite_rules import unpack_qkv_rules
    from onnxscript.rewriter import rewrite

    model = build("Qwen/Qwen2.5-0.5B", execution_provider="dml")
    rewrite(model, pattern_rewrite_rules=unpack_qkv_rules())
"""

from __future__ import annotations

import numpy as np
import onnx_ir as ir
from onnxscript.rewriter._basics import MatchResult
from onnxscript.rewriter._rewrite_rule import RewriteRuleClassBase, RewriteRuleSet


class GQAUnpackQKV(RewriteRuleClassBase):
    """Split packed QKV in GQA into separate Q/K/V MatMul (±Add bias) projections.

    Matches ``GroupQueryAttention`` nodes that use a single packed MatMul
    projection (``key=None``, ``value=None``) and splits the packed weight
    back into three independent projections.  Also handles the biased form
    produced by :class:`PackQKVWithBiasForGQA`, where the packed projection
    is ``Add(MatMul(hidden, packed_w), packed_bias)`` with ``packed_bias``
    being a ``Concat`` of three individual bias parameters.

    Used for DML which does not support packed QKV in GQA.
    """

    _counter: int
    _split_weights: tuple[np.ndarray, np.ndarray, np.ndarray] | None
    # Set in check() when the packed projection includes a bias Add node.
    _bias_concat_node: object | None  # ir.Node | None

    def __init__(self):
        super().__init__()
        self._counter = 0
        self._split_weights = None
        self._bias_concat_node = None

    # ------------------------------------------------------------------ pattern

    def pattern(self, op, packed_qkv):
        # Match GQA with only the packed_qkv (first) input specified.
        # k=None and v=None are checked in check() via the producer node.
        return op.GroupQueryAttention(
            packed_qkv,
            _domain="com.microsoft",
            _allow_other_inputs=True,
            _allow_other_attributes=True,
            _outputs=["gqa_out", "present_key", "present_value"],
        )

    # ------------------------------------------------------------------ check

    def check(self, context, packed_qkv, gqa_out, **_):
        result = MatchResult()
        gqa_node = gqa_out.producer()
        if gqa_node is None:
            return result.fail("No GQA producer")

        # Must be packed mode: k (input[1]) and v (input[2]) must be absent/None
        inputs = gqa_node.inputs
        if len(inputs) < 3:
            return result.fail("GQA has fewer than 3 inputs")
        if inputs[1] is not None or inputs[2] is not None:
            return result.fail("GQA not in packed mode — k or v is present")

        # Detect bias wrapping: packed_qkv may be Add(FusedMatMul, Concat(biases)).
        self._bias_concat_node = None
        qkv_producer = packed_qkv.producer()
        if qkv_producer is not None and qkv_producer.op_type == "Add":
            add_inputs = qkv_producer.inputs
            if len(add_inputs) < 2 or None in add_inputs:
                return result.fail("Add has missing inputs")
            # Find which Add input is the matmul-like op and which is the bias Concat.
            left, right = add_inputs[0], add_inputs[1]
            left_prod = left.producer()
            right_prod = right.producer()
            if left_prod is not None and left_prod.op_type in ("MatMul", "FusedMatMul"):
                matmul, bias_val = left_prod, right
            elif right_prod is not None and right_prod.op_type in ("MatMul", "FusedMatMul"):
                matmul, bias_val = right_prod, left
            else:
                return result.fail("Add inputs do not include a MatMul or FusedMatMul")
            # Validate the bias Concat node.
            bias_concat = bias_val.producer() if bias_val else None
            if bias_concat is None or bias_concat.op_type != "Concat":
                return result.fail("Bias is not produced by a Concat node")
            if len(bias_concat.inputs) != 3:
                return result.fail("Bias Concat does not have exactly 3 inputs")
            for bias_input in bias_concat.inputs:
                if bias_input is None or bias_input.producer() is not None:
                    return result.fail("Bias Concat input is not a graph parameter")
            self._bias_concat_node = bias_concat
        else:
            # No-bias path: packed_qkv must come from MatMul or FusedMatMul directly.
            matmul = qkv_producer
            if matmul is None or matmul.op_type not in ("MatMul", "FusedMatMul"):
                return result.fail("packed_qkv not produced by MatMul or FusedMatMul")

        # Validate the matmul weight structure (shared for both no-bias and biased paths).
        if len(matmul.inputs) < 2:
            return result.fail("MatMul/FusedMatMul has fewer than 2 inputs")
        w_input = matmul.inputs[1]
        if w_input is None:
            return result.fail("MatMul/FusedMatMul weight input is None")

        # FusedMatMul form: weight is Concat(W_q, W_k, W_v) directly (transB=1 handles transpose).
        # MatMul form (legacy): weight is Transpose(Concat(W_q, W_k, W_v)).
        if matmul.op_type == "FusedMatMul":
            w_inner = w_input
        else:
            # MatMul: expect Transpose(Concat(...))
            w_transpose = w_input.producer()
            if w_transpose is None or w_transpose.op_type != "Transpose":
                return result.fail("MatMul weight not produced by Transpose")
            w_inner = w_transpose.inputs[0]
            if w_inner is None:
                return result.fail("Transpose input is None")

        # Case 1 (new): Transpose(Concat(W_q, W_k, W_v, axis=0))
        concat_node = w_inner.producer()
        if concat_node is not None and concat_node.op_type == "Concat":
            if len(concat_node.inputs) != 3:
                return result.fail("Concat does not have exactly 3 inputs")
            for w in concat_node.inputs:
                if w is None or w.producer() is not None:
                    return result.fail("Concat input is not a graph parameter")
            return result

        # Case 2 (legacy): Transpose(W_qkv) where W_qkv is a constant initializer.
        # Keep this path so models packed with the old numpy-based approach still work.
        w_tensor = ir.convenience.get_const_tensor(w_inner)
        if w_tensor is None:
            return result.fail("Packed weight is neither Concat nor a constant")

        num_heads = gqa_node.attributes.get_int("num_heads", None)
        kv_num_heads = gqa_node.attributes.get_int("kv_num_heads", None)
        if num_heads is None or kv_num_heads is None:
            return result.fail("Missing num_heads or kv_num_heads attributes")

        total_out = w_tensor.numpy().shape[0]
        total_heads = num_heads + 2 * kv_num_heads
        if total_out % total_heads != 0:
            return result.fail(
                f"Cannot determine head_dim: total_out={total_out} "
                f"not divisible by total_heads={total_heads}"
            )

        # Pre-compute split weights for use in rewrite()
        w_np = w_tensor.numpy()  # shape: (total_out, hidden)
        head_dim = total_out // total_heads
        q_size = num_heads * head_dim
        k_size = kv_num_heads * head_dim
        self._split_weights = (
            w_np[:q_size, :],
            w_np[q_size : q_size + k_size, :],
            w_np[q_size + k_size :, :],
        )
        return result

    # ------------------------------------------------------------------ rewrite

    def rewrite(self, op, packed_qkv, gqa_out, present_key, present_value, **_):
        gqa_node = gqa_out.producer()
        # Navigate through optional Add wrapper to find the matmul-like op.
        qkv_producer = packed_qkv.producer()
        if qkv_producer is not None and qkv_producer.op_type == "Add":
            add_inputs = qkv_producer.inputs
            left_prod = add_inputs[0].producer()
            matmul = (
                left_prod
                if left_prod is not None and left_prod.op_type in ("MatMul", "FusedMatMul")
                else add_inputs[1].producer()
            )
        else:
            matmul = qkv_producer
        hidden_states = matmul.inputs[0]

        # Extract w_inner: for FusedMatMul the weight is the direct second input;
        # for legacy MatMul the weight goes through a Transpose first.
        w_input = matmul.inputs[1]
        if matmul.op_type == "FusedMatMul":
            w_inner = w_input
        else:
            w_inner = w_input.producer().inputs[0]  # Transpose → inner Concat/param

        concat_node = w_inner.producer()

        self._counter += 1
        suffix = self._counter

        if concat_node is not None and concat_node.op_type == "Concat":
            # Graph-level form: Concat(w_q, w_k, w_v) — rewire directly.
            w_q, w_k, w_v = concat_node.inputs
            if matmul.op_type == "FusedMatMul":
                # FusedMatMul: emit FusedMatMul(hidden, w_i, transB=1) for each
                q_mm = op.FusedMatMul(hidden_states, w_q, _domain="com.microsoft", transB=1)
                k_mm = op.FusedMatMul(hidden_states, w_k, _domain="com.microsoft", transB=1)
                v_mm = op.FusedMatMul(hidden_states, w_v, _domain="com.microsoft", transB=1)
            else:
                # Legacy MatMul: emit MatMul(hidden, Transpose(w_i))
                q_mm = op.MatMul(hidden_states, op.Transpose(w_q, perm=[1, 0]))
                k_mm = op.MatMul(hidden_states, op.Transpose(w_k, perm=[1, 0]))
                v_mm = op.MatMul(hidden_states, op.Transpose(w_v, perm=[1, 0]))
        else:
            # Legacy form: single constant initializer — split with numpy.
            assert self._split_weights is not None
            w_q_np, w_k_np, w_v_np = self._split_weights
            self._split_weights = None

            def _proj(w: np.ndarray, name: str) -> ir.Value:
                init = op.initializer(ir.Tensor(w, name=name), name=name)
                if matmul.op_type == "FusedMatMul":
                    return op.FusedMatMul(
                        hidden_states, init, _domain="com.microsoft", transB=1
                    )
                return op.MatMul(hidden_states, op.Transpose(init, perm=[1, 0]))

            q_mm = _proj(w_q_np, f"unpack_q_weight_{suffix}")
            k_mm = _proj(w_k_np, f"unpack_k_weight_{suffix}")
            v_mm = _proj(w_v_np, f"unpack_v_weight_{suffix}")

        # If bias was packed, redistribute bias_q/bias_k/bias_v.
        bias_concat = self._bias_concat_node
        self._bias_concat_node = None
        if bias_concat is not None:
            bias_q, bias_k, bias_v = bias_concat.inputs
            q = op.Add(q_mm, bias_q)
            k = op.Add(k_mm, bias_k)
            v = op.Add(v_mm, bias_v)
        else:
            q, k, v = q_mm, k_mm, v_mm

        # Rebuild GQA with separate Q/K/V — preserve all existing attributes
        # and remaining inputs (past_key, past_value, seqlens_k, total_seq, ...).
        attrs = {key: gqa_node.attributes[key].value for key in gqa_node.attributes}
        remaining = list(gqa_node.inputs[3:])  # everything after the packed slot

        outputs = op.op_multi_out(
            "GroupQueryAttention",
            inputs=[q, k, v, *remaining],
            domain="com.microsoft",
            attributes=attrs,
            num_outputs=3,
        )
        return outputs[0], outputs[1], outputs[2]


def unpack_qkv_rules() -> RewriteRuleSet:
    """Return a rule set that unpacks packed QKV in GQA into 3 separate MatMuls.

    Used for DML which does not support packed QKV in GQA.

    Returns:
        :class:`RewriteRuleSet` containing the :class:`GQAUnpackQKV` rule.
    """
    return RewriteRuleSet([GQAUnpackQKV().rule()])
