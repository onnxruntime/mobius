# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Rewrite ``TensorScatter`` (static-cache KV write) into ``ScatterND``.

The Qualcomm Hexagon HTP (QNN EP) has no kernel for the opset-24
``TensorScatter`` op, so the static-cache in-place KV writes are forced onto the
CPU EP. This rule rewrites them into ``ScatterND``, which HTP *does* support.

**Matched op** (as mobius emits it in the static-cache attention path):

.. code-block:: text

    updated = TensorScatter(cache, update, write_indices, axis=1)
    # semantics: updated[b, write_indices[b] + t, :] = update[b, t, :]

with ``cache`` shape ``(B, max_seq_len, kv_hidden)`` (a concrete ``max_seq_len``),
``update`` shape ``(B, S_q, kv_hidden)`` and ``write_indices`` shape ``(B,)``.

**Replacement** (batch = 1 — the on-device inference case):

.. code-block:: text

    cache0   = Squeeze(cache, 0)                  # (max, D)
    update0  = Squeeze(update, 0)                 # (S_q, D)
    t_range  = arange(max)[:S_q]                  # Constant + Slice (no Range op)
    pos      = write_indices[0] + t_range         # (S_q,)
    nd_idx   = Unsqueeze(pos, -1)                 # (S_q, 1)
    out0     = ScatterND(cache0, nd_idx, update0) # (max, D)
    updated  = Unsqueeze(out0, 0)                 # (1, max, D)

Only ``axis=1`` scatters over a statically-sized cache axis are rewritten; other
forms are left unchanged. QNN HTP inference is batch=1, so the batch axis is
squeezed and re-added.
"""

from __future__ import annotations

import numpy as np
import onnx_ir as ir
from onnxscript.rewriter._basics import MatchResult
from onnxscript.rewriter._rewrite_rule import RewriteRuleClassBase, RewriteRuleSet


class TensorScatterToScatterND(RewriteRuleClassBase):
    """Lower a static-cache ``TensorScatter`` (axis=1) to ``ScatterND`` (batch=1)."""

    def pattern(self, op, cache, update, write_indices):
        return op.TensorScatter(
            cache,
            update,
            write_indices,
            _allow_other_attributes=True,
            _outputs=["ts_out"],
        )

    def check(self, context, cache, ts_out, **_):
        result = MatchResult()
        node = ts_out.producer()
        if node is None or node.op_type != "TensorScatter":
            return result.fail("not a TensorScatter op")
        if node.attributes.get_int("axis", 0) != 1:
            return result.fail("only axis=1 static-cache scatter is rewritten")
        # The cache axis being scattered must be a concrete size (static cache).
        if cache.shape is None or len(cache.shape) != 3:
            return result.fail("expected rank-3 cache (B, max, D)")
        if not isinstance(cache.shape[1], int):
            return result.fail("cache axis 1 (max_seq_len) must be a concrete int")
        return result

    def rewrite(self, op, cache, update, write_indices, ts_out, **_):
        max_len = int(cache.shape[1])

        zero = op.Constant(value_ints=[0])
        neg1 = op.Constant(value_ints=[-1])
        axis0 = op.Constant(value_ints=[0])

        # Squeeze the (size-1) batch axis: (1, max, D) -> (max, D).
        cache0 = op.Squeeze(cache, zero)
        update0 = op.Squeeze(update, zero)  # (S_q, D)

        # t_range = arange(max_seq_len)[:S_q]  (Constant + Slice, no CPU-only Range).
        full_range = op.Constant(value=ir.tensor(np.arange(max_len, dtype=np.int64)))
        seq_len = op.Shape(update, start=1, end=2)  # (1,) == [S_q]
        t_range = op.Slice(full_range, zero, seq_len, axis0)  # (S_q,)

        # pos = write_indices[0] + t_range  (write_indices is (1,) -> scalar).
        base = op.Squeeze(write_indices, zero)  # scalar
        pos = op.Add(t_range, base)  # (S_q,)
        nd_idx = op.Unsqueeze(pos, neg1)  # (S_q, 1)

        out0 = op.ScatterND(cache0, nd_idx, update0)  # (max, D)
        return op.Unsqueeze(out0, zero)  # (1, max, D)


def tensor_scatter_to_scatternd_rules() -> RewriteRuleSet:
    """Return a rule set rewriting static-cache ``TensorScatter`` to ``ScatterND``.

    Used for EPs without a ``TensorScatter`` kernel
    (``supports_tensor_scatter=False``; QNN HTP). Numerically identical for the
    batch=1 on-device inference case.
    """
    return RewriteRuleSet([TensorScatterToScatterND().rule()])
