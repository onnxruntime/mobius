# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Decompose the opset-24 ``Attention`` op into primitive ONNX ops.

The Qualcomm Hexagon HTP (QNN EP) has no kernel for the fused opset-24
``Attention`` op, so any decoder that emits it is forced onto the CPU EP.
This pass rewrites ``Attention`` into the scaled-dot-product primitives HTP
*does* support (``Reshape``/``Transpose``/``MatMul``/``Softmax``/``Add`` and,
for group-query attention, ``Tile``), so the attention blocks can be claimed
by the QNN partitioner.

**Input op (as mobius emits it for the decoder):**

.. code-block:: text

    Y, present_key, present_value = Attention(
        Q, K, V, attn_mask?, past_key?, past_value?,
        q_num_heads=Nq, kv_num_heads=Nkv, scale=s, softcap=c, is_causal=b,
    )

``Q``/``K``/``V`` are rank-3 ``(B, S, N*H)`` (the layout mobius emits for the
standard-Attention path).  ``attn_mask`` is an optional float additive bias,
``past_key``/``past_value`` are optional rank-4 ``(B, Nkv, S_past, H)`` caches.

**Replacement** faithfully reproduces the op semantics:

1. Reshape/transpose ``Q``/``K``/``V`` to ``(B, N, S, H)``.
2. Concatenate ``past_key``/``past_value`` along the sequence axis (these
   concatenated tensors are also the ``present_key``/``present_value`` outputs).
3. Group-query attention: ``Tile`` the KV heads by ``Nq // Nkv``.
4. ``scores = (Q @ Kᵀ) * scale``; optional ``softcap * tanh(scores/softcap)``;
   add ``attn_mask``; add a bottom-right-aligned causal bias when
   ``is_causal=1``; ``Softmax``; ``@ V``.
5. Transpose/reshape back to ``(B, S, N*H)``.

Implemented as an :class:`onnx_ir.passes.InPlacePass` (not a pattern rewrite
rule) so it can faithfully rewire the ``Attention`` op's three outputs — the
``present_key``/``present_value`` KV-cache tensors are *graph outputs* on
cache-producing layers but *dead* on shared-KV layers, and the pattern
rewriter cannot express that mixed output arity.

Applied automatically by :func:`~mobius._optimizations.optimize_model` for EPs
that lack an ``Attention`` kernel (``supports_attention=False``; QNN).  Can also
be applied manually::

    from mobius.rewrite_rules import decompose_attention_pass
    from onnx_ir.passes import PassManager

    model = build("google/gemma-4-E2B-it", execution_provider="qnn")
    PassManager([decompose_attention_pass()])(model)
"""

from __future__ import annotations

import onnx_ir as ir
from onnx_ir import tape as _tape

# Large negative additive-bias value used for masked-out positions. Matches the
# magnitude ORT's Attention kernel uses for float masks; after softmax these
# positions contribute ~0. Kept finite (not -inf) so ``0 * -inf = NaN`` cannot
# arise for fully-masked rows.
_MASK_NEG = -3.0e38


def _is_decomposable(node: ir.Node) -> bool:
    """Return True if *node* is a fused ``Attention`` op this pass can lower."""
    if node.op_type != "Attention" or node.domain not in ("", "ai.onnx"):
        return False
    q, k, v = node.inputs[0], node.inputs[1], node.inputs[2]
    for val in (q, k, v):
        if val is None or val.shape is None or len(val.shape) != 3:
            return False
    q_heads = node.attributes.get_int("q_num_heads", 0)
    kv_heads = node.attributes.get_int("kv_num_heads", 0)
    if q_heads <= 0 or kv_heads <= 0 or q_heads % kv_heads != 0:
        return False
    # qk_matmul_output_mode!=0 (returning QK before/after softmax) is not a
    # plain SDPA and is not decomposed here.
    if node.attributes.get_int("qk_matmul_output_mode", 0) != 0:
        return False
    return True


def _build_replacement(node: ir.Node) -> tuple[list[ir.Node], list[ir.Value]]:
    """Build the SDPA decomposition for *node*.

    Returns ``(new_nodes, [y, present_key, present_value])``.
    """
    tape = _tape.Tape()

    def c_ints(vals):
        return tape.op("Constant", [], {"value_ints": list(vals)})

    def c_floats(vals):
        return tape.op("Constant", [], {"value_floats": [float(x) for x in vals]})

    inputs = node.inputs
    q, k, v = inputs[0], inputs[1], inputs[2]
    attn_mask = inputs[3] if len(inputs) > 3 else None
    past_key = inputs[4] if len(inputs) > 4 else None
    past_value = inputs[5] if len(inputs) > 5 else None

    q_heads = node.attributes.get_int("q_num_heads")
    kv_heads = node.attributes.get_int("kv_num_heads")
    group = q_heads // kv_heads
    scale = node.attributes.get_float("scale", None)
    softcap = node.attributes.get_float("softcap", 0.0) or 0.0
    is_causal = node.attributes.get_int("is_causal", 0)

    def split_heads(x, n_heads):
        # (B, S, N*H) -> (B, S, N, H) -> (B, N, S, H). ``-1`` infers the head
        # dim, so no static head_dim is required (it may be a SymbolicDim).
        x4 = tape.op("Reshape", [x, c_ints([0, 0, n_heads, -1])])
        return tape.op("Transpose", [x4], {"perm": [0, 2, 1, 3]})

    q_bnsh = split_heads(q, q_heads)
    k_bnsh = split_heads(k, kv_heads)
    v_bnsh = split_heads(v, kv_heads)

    # Concatenate the past cache along the sequence axis (axis=2). The
    # concatenated tensors are the present_key/present_value outputs.
    if past_key is not None:
        k_full = tape.op("Concat", [past_key, k_bnsh], {"axis": 2})
    else:
        k_full = k_bnsh
    if past_value is not None:
        v_full = tape.op("Concat", [past_value, v_bnsh], {"axis": 2})
    else:
        v_full = v_bnsh
    present_k, present_v = k_full, v_full

    # Group-query attention: replicate each KV head ``group`` times.
    # (B, Nkv, S, H) -Unsqueeze-> (B, Nkv, 1, S, H) -Expand-> (B, Nkv, G, S, H)
    # -Reshape-> (B, Nq, S, H). Target built from runtime S/H (no static head).
    # NOTE: Expand (broadcast) is used instead of Tile because the QNN HTP
    # backend has no Tile kernel (forces the node onto CPU); Expand with a
    # size-G broadcast on the inserted axis produces the identical result and
    # runs on HTP.
    def repeat_kv(x):
        if group == 1:
            return x
        x5 = tape.op("Unsqueeze", [x, c_ints([2])])
        # Broadcast the size-1 group axis to ``group`` via Expand.
        x5 = tape.op("Expand", [x5, c_ints([1, 1, group, 1, 1])])
        shp = tape.op("Shape", [x])  # (4,) == [B, Nkv, S, H]
        batch = tape.op("Slice", [shp, c_ints([0]), c_ints([1])])
        s_h = tape.op("Slice", [shp, c_ints([2]), c_ints([4])])
        target = tape.op("Concat", [batch, c_ints([q_heads]), s_h], {"axis": 0})
        return tape.op("Reshape", [x5, target])

    k_rep = repeat_kv(k_full)
    v_rep = repeat_kv(v_full)

    # scores = (Q @ Kᵀ) * scale  -> (B, Nq, Sq, Sk)
    k_t = tape.op("Transpose", [k_rep], {"perm": [0, 1, 3, 2]})
    scores = tape.op("MatMul", [q_bnsh, k_t])
    if scale is not None:
        scale_c = tape.op("CastLike", [c_floats([scale]), scores])
        scores = tape.op("Mul", [scores, scale_c])
    else:
        # scale = 1/sqrt(head_dim); derive head_dim from the runtime shape.
        hd = tape.op(
            "Squeeze",
            [
                tape.op("Slice", [tape.op("Shape", [q_bnsh]), c_ints([3]), c_ints([4])]),
                c_ints([0]),
            ],
        )
        hd_f = tape.op("CastLike", [hd, scores])
        scores = tape.op("Div", [scores, tape.op("Sqrt", [hd_f])])

    # Optional attention-logit soft-capping: cap * tanh(scores / cap).
    if softcap:
        cap = tape.op("CastLike", [c_floats([softcap]), scores])
        scores = tape.op("Mul", [tape.op("Tanh", [tape.op("Div", [scores, cap])]), cap])

    # Additive float mask (broadcasts over the head axis).
    if attn_mask is not None:
        scores = tape.op("Add", [scores, tape.op("CastLike", [attn_mask, scores])])

    # Built-in causal masking (bottom-right aligned so a decode step of Sq
    # queries attends to all Sk = Sq + past keys up to its own position).
    if is_causal:
        scores = tape.op(
            "Add", [scores, _causal_bias(tape, c_ints, c_floats, q_bnsh, k_full, scores)]
        )

    probs = tape.op("Softmax", [scores], {"axis": -1})
    out = tape.op("MatMul", [probs, v_rep])  # (B, Nq, Sq, H)
    out = tape.op("Transpose", [out], {"perm": [0, 2, 1, 3]})  # (B, Sq, Nq, H)
    y = tape.op("Reshape", [out, c_ints([0, 0, -1])])  # (B, Sq, Nq*H)

    return tape.nodes, [y, present_k, present_v]


def _causal_bias(tape, c_ints, c_floats, q_bnsh, k_full, scores):
    """Bottom-right-aligned ``(1, 1, Sq, Sk)`` causal additive bias.

    Query row ``i`` (within the current chunk of ``Sq``) may attend to key
    columns ``j <= i + (Sk - Sq)``; later columns get ``_MASK_NEG``.  Uses
    ``Range`` over the dynamic Sq/Sk so it is correct for both prefill
    (Sk == Sq) and decode (Sk == Sq + past).
    """
    sq = tape.op(
        "Squeeze",
        [
            tape.op("Slice", [tape.op("Shape", [q_bnsh]), c_ints([2]), c_ints([3])]),
            c_ints([0]),
        ],
    )
    sk = tape.op(
        "Squeeze",
        [
            tape.op("Slice", [tape.op("Shape", [k_full]), c_ints([2]), c_ints([3])]),
            c_ints([0]),
        ],
    )
    zero_s = tape.op("Constant", [], {"value_int": 0})
    one_s = tape.op("Constant", [], {"value_int": 1})
    rows = tape.op("Range", [zero_s, sq, one_s])  # (Sq,)
    cols = tape.op("Range", [zero_s, sk, one_s])  # (Sk,)
    offset = tape.op("Sub", [sk, sq])  # scalar Sk - Sq
    rows_off = tape.op("Add", [tape.op("Unsqueeze", [rows, c_ints([1])]), offset])  # (Sq, 1)
    cols_2d = tape.op("Unsqueeze", [cols, c_ints([0])])  # (1, Sk)
    allowed = tape.op("LessOrEqual", [cols_2d, rows_off])  # (Sq, Sk) bool
    zero_f = tape.op("CastLike", [c_floats([0.0]), scores])
    neg_f = tape.op("CastLike", [c_floats([_MASK_NEG]), scores])
    bias = tape.op("Where", [allowed, zero_f, neg_f])  # (Sq, Sk)
    return tape.op("Unsqueeze", [bias, c_ints([0, 1])])  # (1, 1, Sq, Sk)


class DecomposeAttentionPass(ir.passes.InPlacePass):
    """Lower every fused opset-24 ``Attention`` op to SDPA primitives.

    Numerically equivalent to the fused op. Leaves non-decomposable forms
    (rank-4 inputs, ``qk_matmul_output_mode != 0``) unchanged.
    """

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        graph = model.graph
        targets = [n for n in graph if _is_decomposable(n)]
        for node in targets:
            new_nodes, new_values = _build_replacement(node)
            # KV-shared layers emit a 1-output Attention (no present KV cache),
            # while cache-producing layers emit 3 (Y, present_key, present_value).
            # Return exactly as many replacements as the node has outputs; the
            # output order is [Y, present_key, present_value], so truncation is
            # safe. present_k/present_v for a 1-output node are simply unused.
            n_out = len(node.outputs)
            ir.convenience.replace_nodes_and_values(
                graph,
                insertion_point=node,
                old_nodes=[node],
                new_nodes=new_nodes,
                old_values=list(node.outputs),
                new_values=new_values[:n_out],
            )
        return ir.passes.PassResult(model, modified=bool(targets))


def decompose_attention_pass() -> DecomposeAttentionPass:
    """Return the :class:`DecomposeAttentionPass` instance.

    Used for EPs without an ``Attention`` kernel (``supports_attention=False``;
    QNN HTP). The transform is numerically equivalent to the fused op.
    """
    return DecomposeAttentionPass()
