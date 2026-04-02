# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Rewrite rule for unpacking packed QKV back into separate Q/K/V projections.

DML's ``GroupQueryAttention`` kernel does not support packed QKV (a single
fused ``MatMul`` for Q+K+V combined, passed in the query slot with ``key``
and ``value`` set to ``None``).  This rule reverses the ``PackQKVForGQA``
fusion by splitting the packed weight initializer back into three separate
weight matrices.

**Matched pattern:**

.. code-block:: text

    packed_qkv = MatMul(hidden, Transpose(W_qkv))
    out, pk, pv = GroupQueryAttention(packed_qkv, None, None, ...)

Where ``W_qkv`` is a constant initializer that concatenates W_q, W_k, W_v
along axis=0 (out_features dimension).

**Replacement:**

.. code-block:: text

    q = MatMul(hidden, Transpose(W_q))
    k = MatMul(hidden, Transpose(W_k))
    v = MatMul(hidden, Transpose(W_v))
    out, pk, pv = GroupQueryAttention(q, k, v, ...)

The split sizes are derived from ``num_heads``, ``kv_num_heads``, and the
packed weight shape:
- ``q_size = num_heads * head_dim``
- ``k_size = kv_num_heads * head_dim``
- ``v_size = kv_num_heads * head_dim``
- ``head_dim = total_out // (num_heads + 2 * kv_num_heads)``

These rules are **not applied by default**.  Apply them post-export::

    from mobius.rewrite_rules import unpack_qkv_rules
    from onnxscript.rewriter import rewrite

    model = build("Qwen/Qwen3-0.6B", execution_provider="dml")
    rewrite(model, pattern_rewrite_rules=unpack_qkv_rules())
"""

from __future__ import annotations

import numpy as np
import onnx_ir as ir
from onnxscript.rewriter._basics import MatchResult
from onnxscript.rewriter._rewrite_rule import RewriteRuleClassBase, RewriteRuleSet

_counter: int = 0


class GQAUnpackQKV(RewriteRuleClassBase):
    """Split packed QKV in GQA into separate Q/K/V MatMul projections.

    Matches ``GroupQueryAttention`` nodes that use a single packed MatMul
    projection (``key=None``, ``value=None``) and splits the packed weight
    back into three independent projections.

    Used for DML which does not support packed QKV in GQA.
    """

    _counter: int
    _split_weights: tuple[np.ndarray, np.ndarray, np.ndarray] | None

    def __init__(self):
        super().__init__()
        self._counter = 0
        self._split_weights = None

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

        # packed_qkv must come from MatMul(hidden, Transpose(W))
        matmul = packed_qkv.producer()
        if matmul is None or matmul.op_type != "MatMul":
            return result.fail("packed_qkv not produced by MatMul")
        if len(matmul.inputs) < 2:
            return result.fail("MatMul has fewer than 2 inputs")

        # The second MatMul input must be Transpose(W_constant)
        w_t = matmul.inputs[1]
        if w_t is None:
            return result.fail("MatMul weight input is None")
        w_transpose = w_t.producer()
        if w_transpose is None or w_transpose.op_type != "Transpose":
            return result.fail("MatMul weight not produced by Transpose")

        w_const = w_transpose.inputs[0]
        if w_const is None:
            return result.fail("Transpose input is None")
        w_tensor = ir.convenience.get_const_tensor(w_const)
        if w_tensor is None:
            return result.fail("Packed weight is not a constant")

        # Validate we can split: need num_heads + kv_num_heads to be set
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

        # Pre-compute split weights so they're available in rewrite()
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
        assert self._split_weights is not None
        w_q, w_k, w_v = self._split_weights
        self._split_weights = None

        gqa_node = gqa_out.producer()
        matmul = packed_qkv.producer()
        hidden_states = matmul.inputs[0]

        # Store split weights as initializers and project separately.
        self._counter += 1
        suffix = self._counter

        def _proj(w: np.ndarray, name: str) -> ir.Value:
            init = op.initializer(ir.Tensor(w, name=name), name=name)
            return op.MatMul(hidden_states, op.Transpose(init, perm=[1, 0]))

        q = _proj(w_q, f"unpack_q_weight_{suffix}")
        k = _proj(w_k, f"unpack_k_weight_{suffix}")
        v = _proj(w_v, f"unpack_v_weight_{suffix}")

        # Rebuild GQA with separate Q/K/V — keep all existing attributes
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
