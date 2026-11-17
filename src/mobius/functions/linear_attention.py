# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Reference ir.Function for the ONNX opset-29 LinearAttention operator.

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

``v_num_heads`` represents Flash-Next's independent value/state head
count exactly.  It is distinct from ordinary query GQA.

Op spec: https://github.com/onnx/onnx/issues/7689
"""

from __future__ import annotations

import onnx_ir as ir
from onnxscript._internal.builder import build_function

from mobius._constants import OPSET_VERSION
from mobius.components import (
    create_body_graph,
    rename_subgraph_values,
)

DOMAIN = "com.microsoft"


def linear_attention(
    *,
    q_num_heads: int,
    kv_num_heads: int,
    v_num_heads: int = 0,
    update_rule: str = "gated_delta",
    scale: float = 0.0,
    output_states: bool = False,
    stash_type: ir.DataType = ir.DataType.FLOAT,
    opset_version: int = OPSET_VERSION,
    domain: str = DOMAIN,
) -> ir.Function:
    """Build an ir.Function for LinearAttention.

    Inputs (conditional):
        query:      (B, T, q_num_heads * d_k)
        key:        (B, T, kv_num_heads * d_k)
        value:      (B, T, state_num_heads * d_v)
        past_state: (B, state_num_heads, d_k, d_v)
        decay:      (B, T, state_num_heads * d_k) or (B, T, state_num_heads)
        beta:       (B, T, state_num_heads) or (B, T, 1)

    Outputs:
        output:        (B, T, output_num_heads * d_v)
        present_state: (B, state_num_heads, d_k, d_v)

    Args:
        q_num_heads: Query head count.
        kv_num_heads: Compact key/value/state head count for ordinary GQA.
        v_num_heads: Independent value/state/output head count. Zero preserves
            ordinary GQA. A positive value requires equal Q/K heads and must be
            a multiple of ``kv_num_heads``.
        update_rule: One of "linear", "gated", "delta", "gated_delta".
        scale: Scalar multiplier applied to query. Zero derives
            ``1/sqrt(d_k)`` as specified by opset 29.
        output_states: Emit the optional post-update state trajectory
            ``(B, T, H_state, d_k, d_v)`` as a third function output.
        domain: Function domain. Primarily useful for isolated reference tests
            where an installed runtime already reserves ``com.microsoft``.
        stash_type: Element type for the Scan body's internal
            computation.  Must match the precision of the inputs
            passed at the call site.  Defaults to ``FLOAT``.

    The function body handles 3D→4D reshape, GQA expansion, query
    scaling, and uses an ONNX Scan op for the sequential recurrence.
    Output is reshaped back to 3D.  A fused kernel can replace the
    entire function for 10-50x speedup.
    """
    valid_rules = ("linear", "gated", "delta", "gated_delta")
    if update_rule not in valid_rules:
        raise ValueError(
            f"Unknown update_rule: {update_rule!r}. Expected one of {valid_rules}."
        )
    if q_num_heads <= 0:
        raise ValueError(f"q_num_heads must be > 0; got {q_num_heads}")
    if kv_num_heads <= 0:
        raise ValueError(f"kv_num_heads must be > 0; got {kv_num_heads}")
    legacy_expanded_heads = v_num_heads == 0 and q_num_heads < kv_num_heads
    if legacy_expanded_heads and kv_num_heads % q_num_heads:
        raise ValueError(
            f"Legacy kv_num_heads ({kv_num_heads}) must be divisible by "
            f"q_num_heads ({q_num_heads}); use explicit v_num_heads for opset 29."
        )
    if not legacy_expanded_heads and q_num_heads % kv_num_heads != 0:
        raise ValueError(
            f"q_num_heads ({q_num_heads}) must be divisible by kv_num_heads ({kv_num_heads})"
        )
    if v_num_heads < 0 or (
        v_num_heads > 0 and (q_num_heads != kv_num_heads or v_num_heads % kv_num_heads != 0)
    ):
        raise ValueError(
            "v_num_heads must be zero, or a positive multiple of kv_num_heads "
            "with q_num_heads == kv_num_heads"
        )
    state_num_heads = v_num_heads or kv_num_heads

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

        # --- Reshape 3D → 4D using head counts ---
        b_dim = op.Shape(query_v, start=0, end=1)
        t_dim = op.Shape(query_v, start=1, end=2)

        # Q/K retain their own head counts until the recurrent read/write.
        qk_4d_shape = op.Concat(
            b_dim,
            t_dim,
            op.Constant(value_ints=[q_num_heads, -1]),
            axis=0,
        )
        query_4d = op.Transpose(
            op.Reshape(query_v, qk_4d_shape), perm=[0, 2, 1, 3]
        )  # [B, q_num_heads, T, d_k]
        key_4d = op.Transpose(
            op.Reshape(
                key_v,
                op.Concat(
                    b_dim,
                    t_dim,
                    op.Constant(
                        value_ints=[q_num_heads if legacy_expanded_heads else kv_num_heads, -1]
                    ),
                    axis=0,
                ),
            ),
            perm=[0, 2, 1, 3],
        )  # [B, H_kv, T, d_k]

        # V/state gates use the independent state head count.
        state_4d_shape = op.Concat(
            b_dim,
            t_dim,
            op.Constant(value_ints=[state_num_heads, -1]),
            axis=0,
        )
        value_4d = op.Transpose(
            op.Reshape(value_v, state_4d_shape), perm=[0, 2, 1, 3]
        )  # [B, H_state, T, d_v]

        if v_num_heads or legacy_expanded_heads:
            # Flash-Next: compact equal-head Q/K are repeated onto value/state heads.
            query_4d, key_4d = _expand_kv_heads(
                op,
                query_4d,
                key_4d,
                gqa_ratio=(
                    v_num_heads // kv_num_heads if v_num_heads else kv_num_heads // q_num_heads
                ),
            )

        # --- Reshape decay/beta 3D → 4D (only when used) ---
        # decay: (B, T, kv_num_heads * d_k) → (B, T, kv_num_heads, d_k)
        #     → transpose to (B, kv_num_heads, T, d_k)
        if uses_decay:
            decay_4d = op.Transpose(
                op.Reshape(decay_v, state_4d_shape), perm=[0, 2, 1, 3]
            )  # [B, H_state, T, d_k or 1]
        if uses_beta:
            beta_3d = op.Transpose(beta_v, perm=[0, 2, 1])

        # --- Apply query scale (matches op spec default of 1/sqrt(d_k)) ---
        # CastLike ensures scale constant matches the input dtype.
        if not scale:
            d_k = op.CastLike(op.Shape(query_4d, start=3, end=4), query_4d)
            query_scale = op.Reciprocal(op.Sqrt(d_k))
        else:
            query_scale = op.CastLike(op.Constant(value_float=scale), query_4d)
        scaled_query = op.Mul(query_4d, query_scale)

        # --- Build Scan for sequential recurrence ---
        scan_body = _build_recurrence_body(
            uses_decay,
            uses_beta,
            read_gqa_ratio=(
                1 if v_num_heads or legacy_expanded_heads else q_num_heads // kv_num_heads
            ),
            output_states=output_states,
            stash_type=stash_type,
            opset_version=opset_version,
        )

        # Transpose to T-first for Scan: (B, H, T, D) -> (T, B, H, D)
        q_t = op.Transpose(scaled_query, perm=[2, 0, 1, 3])
        k_t = op.Transpose(key_4d, perm=[2, 0, 1, 3])
        v_t = op.Transpose(value_4d, perm=[2, 0, 1, 3])

        scan_inputs = [q_t, k_t, v_t]
        if uses_decay:
            decay_t = op.Transpose(decay_4d, perm=[2, 0, 1, 3])  # (T, B, H, d_k)
            scan_inputs.append(decay_t)
        if uses_beta:
            beta_t = op.Transpose(beta_3d, perm=[2, 0, 1])  # (T, B, H)
            scan_inputs.append(beta_t)

        scan_outputs = op.Scan(
            past_state_v,  # carry: (B, H, d_k, d_v)
            *scan_inputs,
            body=scan_body,
            num_scan_inputs=len(scan_inputs),
            _outputs=3 if output_states else 2,
        )
        present_state_v, output_t = scan_outputs[:2]
        # present_state_v: (B, H, d_k, d_v)
        # output_t: (T, B, H, d_v)

        # --- Reshape output 4D → 3D ---
        # (T, B, H, d_v) → (B, T, H, d_v) → (B, T, H*d_v)
        output_bthd = op.Transpose(output_t, perm=[1, 0, 2, 3])
        out_3d_shape = op.Concat(b_dim, t_dim, op.Constant(value_ints=[-1]), axis=0)
        output = op.Reshape(output_bthd, out_3d_shape)  # [B, T, H_output*d_v]

        output.name = "output"
        present_state_v.name = "present_state"
        if output_states:
            states_t = scan_outputs[2]  # (T, B, H_state, d_k, d_v)
            states = op.Transpose(states_t, perm=[1, 0, 2, 3, 4])
            states.name = "states"
            return output, present_state_v, states
        return output, present_state_v

    # --- Build the ir.Function ---
    return build_function(
        body,
        inputs,
        domain=domain,
        name="LinearAttention",
        attributes=[
            ir.Attr(
                "update_rule",
                ir.AttributeType.STRING,
                update_rule,
                ref_attr_name="update_rule",
            ),
            ir.Attr(
                "scale",
                ir.AttributeType.FLOAT,
                scale,
                ref_attr_name="scale",
            ),
            ir.Attr(
                "q_num_heads",
                ir.AttributeType.INT,
                q_num_heads,
                ref_attr_name="q_num_heads",
            ),
            ir.Attr(
                "kv_num_heads",
                ir.AttributeType.INT,
                kv_num_heads,
                ref_attr_name="kv_num_heads",
            ),
            ir.Attr(
                "v_num_heads",
                ir.AttributeType.INT,
                v_num_heads,
                ref_attr_name="v_num_heads",
            ),
        ],
        opset_imports={"": opset_version},
    )


def _build_recurrence_body(
    uses_decay: bool,
    uses_beta: bool,
    *,
    read_gqa_ratio: int = 1,
    output_states: bool = False,
    stash_type: ir.DataType = ir.DataType.FLOAT,
    opset_version: int = OPSET_VERSION,
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
        opset_version=opset_version,
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

    # Ordinary GQA repeats each compact state head across adjacent query heads.
    state_for_read = new_state
    if read_gqa_ratio > 1:
        axes_2 = bop.Constant(value_ints=[2])
        state_5d = bop.Unsqueeze(new_state, axes_2)
        state_tiled = bop.Tile(
            state_5d,
            bop.Constant(value_ints=[1, 1, read_gqa_ratio, 1, 1]),
        )
        state_shape = bop.Shape(new_state)
        state_for_read = bop.Reshape(
            state_tiled,
            bop.Concat(
                bop.Slice(state_shape, [0], [1]),
                bop.Constant(value_ints=[-1]),
                bop.Slice(state_shape, [2], [4]),
                axis=0,
            ),
        )

    # --- Output: q @ state -> (B, H_q, d_v) ---
    q_row = bop.Unsqueeze(q_t, axes_neg2)  # (B, H, 1, d_k)
    output_t = bop.Squeeze(bop.MatMul(q_row, state_for_read), axes_neg2)
    output_t.name = "output_t"

    body_graph.outputs.extend([new_state, output_t])
    if output_states:
        state_step = bop.Identity(new_state)
        state_step.name = "state_step"
        body_graph.outputs.append(state_step)
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
    # Extract B, T, d_k from the original 4D query to build the target shape.
    # We cannot use '0' sentinels because ONNX Reshape copies from the same
    # positional index, and the 5D→4D dimension mapping doesn't align.
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
