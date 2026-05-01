# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Reference ir.Function for the proposed LinearAttention operator.

Implements the gated delta-rule linear attention recurrence as a
self-contained function: 3D→4D reshape + GQA head expansion + query
scaling + sequential Scan over the time dimension.  The component
(GatedDeltaNet) simply calls
``op.LinearAttention(q, k, v, state, decay, beta)`` — all complexity
lives here.

The gated-delta variant (used by Qwen3.5 GatedDeltaNet) computes:

    S_t = exp(g_t) * S_{t-1} + beta_t * k_t (x) (v_t - exp(g_t) * S_{t-1}^T k_t)
    o_t = q_t^T S_t

All activations are 3D ``[B, T, H*D]`` (matching the ONNX Attention op
convention).  ``q_num_heads`` and ``kv_num_heads`` attributes tell
the function how to reshape to 4D internally.

The function has 6 required inputs:
  ``(query, key, value, past_state, decay, beta)``

GQA support: when Q/K have fewer heads (q_num_heads) than
V/state (kv_num_heads), the function expands Q/K heads internally
via Tile+Reshape.

Op spec: https://github.com/onnx/onnx/issues/7689
"""

from __future__ import annotations

import onnx_ir as ir
from onnxscript._internal import builder

from mobius._constants import OPSET_VERSION
from mobius.components._scan_utils import (
    create_body_graph,
    rename_subgraph_values,
)

DOMAIN = "com.microsoft"


def linear_attention() -> ir.Function:
    """Build a static ``ir.Function`` for LinearAttention.

    Returns a single function definition whose body implements the
    gated-delta recurrence (the most general variant).  Callers pass
    model-specific attribute values as ``op.call()`` kwargs::

        op.call(fn, q, k, v, state, decay, beta,
                scale=0.5, q_num_heads=8, kv_num_heads=8,
                update_rule="gated_delta", _outputs=2)

    Head counts are derived from tensor shapes inside the body.
    The ``scale``, ``update_rule``, ``q_num_heads``, and
    ``kv_num_heads`` attributes appear on the calling node for
    runtime kernels to consume.

    Inputs:
        query:      (B, T, q_num_heads * d_k)
        key:        (B, T, q_num_heads * d_k)
        value:      (B, T, kv_num_heads * d_v)
        past_state: (B, kv_num_heads, d_k, d_v) — recurrent state
        decay:      (B, T, kv_num_heads * d_k)
        beta:       (B, T, kv_num_heads)

    Outputs:
        output:        (B, T, kv_num_heads * d_v) — attention output (3D)
        present_state: (B, kv_num_heads, d_k, d_v) — updated state
    """
    # Static body uses gated_delta (superset) with FLOAT precision.
    # Runtime kernels use the node-level attributes for correct dispatch.
    update_rule = "gated_delta"
    scale = 1.0
    stash_type = ir.DataType.FLOAT

    uses_decay = update_rule in ("gated", "gated_delta")
    uses_beta = update_rule in ("delta", "gated_delta")

    # --- Define function inputs ---
    # Always declare the full 6-input signature:
    # (query, key, value, past_state, decay, beta).
    # Update-rule-specific paths inside the function body ignore unused
    # trailing inputs so every variant exposes the same ir.Function
    # interface.  Call sites that don't need trailing inputs (decay, beta)
    # simply omit them — onnx-shape-inference >=0.1.9 accepts trailing
    # optional inputs (matching ONNX C++ behavior).
    inputs: list[ir.Value] = [
        ir.Value(name="query"),  # (B, T, q_num_heads * d_k)
        ir.Value(name="key"),  # (B, T, q_num_heads * d_k)
        ir.Value(name="value"),  # (B, T, kv_num_heads * d_v)
        ir.Value(name="past_state"),
        ir.Value(name="decay"),
        ir.Value(name="beta"),
    ]

    def body(op, query_v, key_v, value_v, past_state_v, decay_v, beta_v):

        # --- Derive head counts from tensor shapes ---
        # past_state: (B, kv_num_heads, d_k, d_v)
        kv_num_heads_dim = op.Shape(past_state_v, start=1, end=2)  # [kv_num_heads]
        d_k_dim = op.Shape(past_state_v, start=2, end=3)  # [d_k]

        # q_num_heads = query.shape[2] / d_k
        q_last_dim = op.Shape(query_v, start=2, end=3)  # [q_num_heads * d_k]
        q_num_heads_dim = op.Div(q_last_dim, d_k_dim)  # [q_num_heads]

        # --- Reshape 3D → 4D using derived head counts ---
        b_dim = op.Shape(query_v, start=0, end=1)
        t_dim = op.Shape(query_v, start=1, end=2)

        # Q/K: [B, T, q_num_heads*d_k] → [B, T, q_num_heads, d_k]
        #     → transpose to [B, q_num_heads, T, d_k]
        qk_4d_shape = op.Concat(
            b_dim,
            t_dim,
            q_num_heads_dim,
            op.Constant(value_ints=[-1]),
            axis=0,
        )
        query_4d = op.Transpose(
            op.Reshape(query_v, qk_4d_shape), perm=[0, 2, 1, 3]
        )  # [B, q_num_heads, T, d_k]
        key_4d = op.Transpose(
            op.Reshape(key_v, qk_4d_shape), perm=[0, 2, 1, 3]
        )  # [B, q_num_heads, T, d_k]

        # V: [B, T, kv_num_heads*d_v] → [B, kv_num_heads, T, d_v]
        kv_4d_shape = op.Concat(
            b_dim,
            t_dim,
            kv_num_heads_dim,
            op.Constant(value_ints=[-1]),
            axis=0,
        )
        value_4d = op.Transpose(
            op.Reshape(value_v, kv_4d_shape), perm=[0, 2, 1, 3]
        )  # [B, kv_num_heads, T, d_v]

        # --- GQA: expand Q/K heads to match V head count ---
        # Compute ratio dynamically: gqa_ratio = kv_num_heads / q_num_heads
        query_expanded, key_expanded = _expand_kv_heads_dynamic(
            op,
            query_4d,
            key_4d,
            q_num_heads_dim=q_num_heads_dim,
            kv_num_heads_dim=kv_num_heads_dim,
        )

        # --- Reshape decay/beta 3D → 4D (only when used) ---
        if uses_decay:
            decay_4d = op.Transpose(
                op.Reshape(decay_v, kv_4d_shape), perm=[0, 2, 1, 3]
            )  # [B, kv_num_heads, T, d_k]
        if uses_beta:
            beta_3d = op.Transpose(beta_v, perm=[0, 2, 1])  # [B, kv_num_heads, T]

        # --- Apply query scale ---
        scaled_query = op.Mul(
            query_expanded,
            op.CastLike(op.Constant(value_float=scale), query_expanded),
        )

        # --- Build Scan for sequential recurrence ---
        scan_body = _build_recurrence_body(uses_decay, uses_beta, stash_type=stash_type)

        # Transpose to T-first for Scan: (B, H, T, D) -> (T, B, H, D)
        q_t = op.Transpose(scaled_query, perm=[2, 0, 1, 3])
        k_t = op.Transpose(key_expanded, perm=[2, 0, 1, 3])
        v_t = op.Transpose(value_4d, perm=[2, 0, 1, 3])

        scan_inputs = [q_t, k_t, v_t]
        if uses_decay:
            decay_t = op.Transpose(decay_4d, perm=[2, 0, 1, 3])
            scan_inputs.append(decay_t)
        if uses_beta:
            beta_t = op.Transpose(beta_3d, perm=[2, 0, 1])
            scan_inputs.append(beta_t)

        present_state_v, output_t = op.Scan(
            past_state_v,
            *scan_inputs,
            body=scan_body,
            num_scan_inputs=len(scan_inputs),
            _outputs=2,
        )

        # --- Reshape output 4D → 3D ---
        output_bthd = op.Transpose(output_t, perm=[1, 0, 2, 3])
        out_3d_shape = op.Concat(b_dim, t_dim, op.Constant(value_ints=[-1]), axis=0)
        output = op.Reshape(output_bthd, out_3d_shape)  # [B, T, H*d_v]

        output.name = "output"
        present_state_v.name = "present_state"
        return output, present_state_v

    # --- Build the ir.Function ---
    return builder.build_function(
        body,
        inputs,
        domain=DOMAIN,
        name="LinearAttention",
        attributes=[
            ir.Attr(
                "update_rule",
                ir.AttributeType.STRING,
                update_rule,
            ),
            ir.Attr(
                "scale",
                ir.AttributeType.FLOAT,
                scale,
            ),
            ir.Attr(
                "q_num_heads",
                ir.AttributeType.INT,
                0,
            ),
            ir.Attr(
                "kv_num_heads",
                ir.AttributeType.INT,
                0,
            ),
        ],
        opset_imports={"": OPSET_VERSION},
    )


def _build_recurrence_body(
    uses_decay: bool,
    uses_beta: bool,
    *,
    stash_type: ir.DataType = ir.DataType.FLOAT,
) -> ir.Graph:
    """Build the Scan body for single-token delta-rule recurrence.

    The body operates in ``stash_type`` precision.

    Body inputs (in order):
        1. state: (B, H, d_k, d_v) [carry]
        2. q_t: (B, H, d_k) [scan input]
        3. k_t: (B, H, d_k) [scan input]
        4. v_t: (B, H, d_v) [scan input]
        5. decay_t: (B, H, d_k) [scan input]
        6. beta_t: (B, H) [scan input]

    Body outputs:
        1. new_state: (B, H, d_k, d_v) [carry]
        2. output_t: (B, H, d_v) [scan output]
    """
    dtype = ir.TensorType(stash_type)

    # Only specify dtype, not shape. Shapes can be inferred from Scan op's
    # inputs. Hard-coding locally chosen symbolic dim names is incorrect
    # since ONNX has a global scope for symbolic dims. Even the dtype is
    # optional in that it can be inferred, but specifying it is safe and
    # can help as long as we know the exact type.
    state_in = ir.Value(name="state", type=dtype)
    q_t = ir.Value(name="q_t", type=dtype)
    k_t = ir.Value(name="k_t", type=dtype)
    v_t = ir.Value(name="v_t", type=dtype)
    scan_in_vals = [q_t, k_t, v_t]
    decay_t: ir.Value | None = None
    beta_t: ir.Value | None = None
    if uses_decay:
        decay_t = ir.Value(name="decay_t", type=dtype)
        scan_in_vals.append(decay_t)
    if uses_beta:
        beta_t = ir.Value(name="beta_t", type=dtype)
        scan_in_vals.append(beta_t)

    body_graph, body_builder = create_body_graph(
        state_inputs=[state_in],
        scan_inputs=scan_in_vals,
        name="delta_recurrence",
    )
    bop = body_builder.op

    # Shared axes constants (deduplicated)
    axes_neg2 = bop.Constant(value_ints=[-2])
    axes_neg1 = bop.Constant(value_ints=[-1])

    # --- State decay: state = exp(g) * past_state ---
    if uses_decay:
        # decay_t: (B, H, d_k) -> (B, H, d_k, 1) for broadcasting with state (B, H, d_k, d_v)
        g_exp = bop.Exp(bop.Unsqueeze(decay_t, axes_neg1))
        state = bop.Mul(state_in, g_exp)
    else:
        state = state_in

    # --- Retrieval: k @ state -> (B, H, d_v) ---
    # k_row: (B, H, 1, d_k) @ state: (B, H, d_k, d_v) -> (B, H, 1, d_v)
    k_row = bop.Unsqueeze(k_t, axes_neg2)
    retrieval = bop.Squeeze(bop.MatMul(k_row, state), axes_neg2)

    # --- State update ---
    if uses_beta:
        # delta = (v - retrieval) * beta
        delta = bop.Sub(v_t, retrieval)
        beta_expanded = bop.Unsqueeze(beta_t, axes_neg1)  # (B, H, 1)
        delta = bop.Mul(delta, beta_expanded)
    else:
        delta = v_t

    # Outer product: k^T @ delta -> (B, H, d_k, d_v)
    k_col = bop.Unsqueeze(k_t, axes_neg1)  # (B, H, d_k, 1)
    delta_row = bop.Unsqueeze(delta, axes_neg2)  # (B, H, 1, d_v)
    outer = bop.MatMul(k_col, delta_row)
    new_state = bop.Add(state, outer)
    new_state.name = "new_state"

    # --- Output: q @ new_state -> (B, H, d_v) ---
    q_row = bop.Unsqueeze(q_t, axes_neg2)  # (B, H, 1, d_k)
    output_t = bop.Squeeze(bop.MatMul(q_row, new_state), axes_neg2)
    output_t.name = "output_t"

    body_graph.outputs.extend([new_state, output_t])
    rename_subgraph_values(body_graph, "dn_")

    return body_graph


def _expand_kv_heads(op, query, key, *, gqa_ratio: int):
    """Expand Q/K heads to match V head count for GQA.

    When gqa_ratio > 1, each Q/K head is tiled ``gqa_ratio`` times along
    a new dim and then reshaped to merge heads.
    When gqa_ratio == 1, this is a no-op (returns inputs unchanged).

    The expansion ratio is computed at graph-build time so the Tile
    repeats are a static Constant — this avoids Shape→Gather→Div ops
    whose int64 outputs cause CUDA EP memory placement issues.

    Args:
        op: ONNX op builder.
        query: (B, H_kv, T, d_k)
        key: (B, H_kv, T, d_k)
        gqa_ratio: ``num_v_heads // num_k_heads`` (computed at build time).

    Returns:
        query: (B, H, T, d_k)
        key: (B, H, T, d_k)
    """
    if gqa_ratio == 1:
        return query, key

    # (B, H_kv, T, d_k) -> (B, H_kv, 1, T, d_k)
    axes_2 = op.Constant(value_ints=[2])
    q_5d = op.Unsqueeze(query, axes_2)
    k_5d = op.Unsqueeze(key, axes_2)

    # Tile along dim 2 by the static ratio
    repeat_vec = op.Constant(value_ints=[1, 1, gqa_ratio, 1, 1])
    q_tiled = op.Tile(q_5d, repeat_vec)
    k_tiled = op.Tile(k_5d, repeat_vec)

    # Reshape: (B, H_kv, ratio, T, d_k) -> (B, H_kv*ratio, T, d_k)
    b_dim = op.Shape(query, start=0, end=1)
    t_dim = op.Shape(query, start=2, end=3)
    dk_dim = op.Shape(query, start=3, end=4)
    expanded_shape = op.Concat(
        b_dim,
        op.Constant(value_ints=[-1]),
        t_dim,
        dk_dim,
        axis=0,
    )
    return op.Reshape(q_tiled, expanded_shape), op.Reshape(k_tiled, expanded_shape)


def _expand_kv_heads_dynamic(op, query, key, *, q_num_heads_dim, kv_num_heads_dim):
    """Expand Q/K heads to match V head count for GQA (dynamic ratio).

    Like :func:`_expand_kv_heads` but computes the expansion ratio
    dynamically from shape tensors, allowing the function body to be
    independent of specific head counts.

    Args:
        op: ONNX op builder.
        query: (B, q_num_heads, T, d_k)
        key: (B, q_num_heads, T, d_k)
        q_num_heads_dim: 1-element tensor with q_num_heads value.
        kv_num_heads_dim: 1-element tensor with kv_num_heads value.

    Returns:
        query: (B, kv_num_heads, T, d_k)
        key: (B, kv_num_heads, T, d_k)
    """
    # gqa_ratio = kv_num_heads / q_num_heads
    gqa_ratio = op.Div(kv_num_heads_dim, q_num_heads_dim)  # [ratio]

    # (B, q_num_heads, T, d_k) -> (B, q_num_heads, 1, T, d_k)
    axes_2 = op.Constant(value_ints=[2])
    q_5d = op.Unsqueeze(query, axes_2)
    k_5d = op.Unsqueeze(key, axes_2)

    # Tile along dim 2 by the dynamic ratio
    ones = op.Constant(value_ints=[1, 1])
    ones2 = op.Constant(value_ints=[1, 1])
    repeat_vec = op.Concat(ones, gqa_ratio, ones2, axis=0)  # [1, 1, R, 1, 1]
    q_tiled = op.Tile(q_5d, repeat_vec)
    k_tiled = op.Tile(k_5d, repeat_vec)

    # Reshape: (B, q_num_heads, ratio, T, d_k) -> (B, kv_num_heads, T, d_k)
    b_dim = op.Shape(query, start=0, end=1)
    t_dim = op.Shape(query, start=2, end=3)
    dk_dim = op.Shape(query, start=3, end=4)
    expanded_shape = op.Concat(
        b_dim,
        op.Constant(value_ints=[-1]),
        t_dim,
        dk_dim,
        axis=0,
    )
    return op.Reshape(q_tiled, expanded_shape), op.Reshape(k_tiled, expanded_shape)
