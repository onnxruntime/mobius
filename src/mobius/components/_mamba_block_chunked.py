# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Mamba2 chunked SSD (Structured State Space Duality) implementation.

Processes all tokens in parallel within chunks of ``chunk_size``, then
propagates SSM state across chunk boundaries.  Matches HuggingFace's
``torch_forward`` multi-token path.

This module contains the ``Mamba2BlockChunkedSSD`` subclass and its
helper functions (``_segment_sum``, ``_segment_sum_dynamic``).
"""

from __future__ import annotations

import numpy as np
import onnx_ir as ir
from onnxscript._internal import builder

from mobius.components._common import INT64_MAX
from mobius.components._mamba_block import Mamba2BlockBase

# -----------------------------------------------------------------------
# Chunked SSD helpers — ONNX equivalents of the PyTorch helper functions
# -----------------------------------------------------------------------


def _segment_sum(
    op: builder.OpBuilder,
    x: ir.Value,
    chunk_size: int,
):
    """Stable segment sum via cumulative sums and masking.

    Input:  x with last dim = chunk_size  (..., chunk_size)
    Output: (..., chunk_size, chunk_size) lower-triangular cumsum.

    Matches HF's ``segment_sum()`` function.  ``chunk_size`` must be a
    compile-time constant so we can build the fixed triangular masks.
    """
    # Expand: (..., chunk_size) → (..., chunk_size, chunk_size)
    x_expanded = op.Unsqueeze(x, [-1])
    x_tiled = op.Expand(
        x_expanded,
        op.Concat(
            op.Shape(x),
            op.Constant(value_ints=[chunk_size]),
            axis=0,
        ),
    )

    # Strict lower-triangular mask (diagonal=-1)
    mask_strict = op.Constant(
        value=ir.tensor(
            np.tril(
                np.ones((chunk_size, chunk_size), dtype=np.float32),
                k=-1,
            )
        )
    )
    x_masked = op.Mul(x_tiled, mask_strict)

    # CumSum along the second-to-last axis
    cumsum = op.CumSum(x_masked, op.Constant(value_int=-2))

    # Full lower-triangular mask (including diagonal)
    mask_full_bool = op.Constant(
        value=ir.tensor(
            np.tril(
                np.ones((chunk_size, chunk_size), dtype=np.bool_),
                k=0,
            )
        )
    )
    neg_inf = op.Constant(
        value=ir.tensor(
            np.full(
                (chunk_size, chunk_size),
                float("-inf"),
                dtype=np.float32,
            )
        )
    )
    result = op.Where(mask_full_bool, cumsum, neg_inf)
    return result


def _segment_sum_dynamic(op: builder.OpBuilder, x: ir.Value):
    """Segment sum for dynamic-sized last dimension.

    Same logic as ``_segment_sum`` but uses ONNX ops (``Trilu``) to
    build masks dynamically, since nc+1 (number of chunks + 1) is not
    known at graph-build time.

    Input:  x with shape (..., K) where K is dynamic.
    Output: (..., K, K) lower-triangular cumsum.
    """
    K = op.Shape(x, start=-1, end=None)  # [K] as 1-d tensor

    # Expand: (..., K) → (..., K, K)
    x_expanded = op.Unsqueeze(x, [-1])
    new_shape = op.Concat(op.Shape(x), K, axis=0)
    x_tiled = op.Expand(x_expanded, new_shape)

    # Build KxK ones, then lower-triangular masks via Trilu
    KK_shape = op.Concat(K, K, axis=0)
    ones_KK = op.ConstantOfShape(
        KK_shape,
        value=ir.tensor(np.array([1.0], dtype=np.float32)),
    )
    # Strict lower triangle (diagonal=-1)
    lower_strict = op.Trilu(
        ones_KK,
        op.Constant(value_int=-1),
        upper=0,
    )
    x_masked = op.Mul(x_tiled, lower_strict)

    # CumSum along second-to-last axis
    cumsum = op.CumSum(x_masked, op.Constant(value_int=-2))

    # Full lower triangle (including diagonal)
    lower_full = op.Trilu(ones_KK, upper=0)
    lower_bool = op.Cast(lower_full, to=ir.DataType.BOOL)

    # Fill non-lower-triangle with -inf
    neg_inf_KK = op.ConstantOfShape(
        KK_shape,
        value=ir.tensor(np.array([float("-inf")], dtype=np.float32)),
    )
    result = op.Where(lower_bool, cumsum, neg_inf_KK)
    return result


class Mamba2BlockChunkedSSD(Mamba2BlockBase):
    """Mamba2 block using the chunked SSD algorithm.

    Processes the full sequence in parallel within chunks of
    ``chunk_size`` tokens, then propagates SSM state across chunk
    boundaries.  Matches HF's ``torch_forward`` multi-token path.

    Note: seq_len must be divisible by chunk_size (the task layer
    is responsible for padding if necessary).
    """

    def _chunked_ssd(
        self,
        op: builder.OpBuilder,
        x: ir.Value,
        dt: ir.Value,
        B_mat: ir.Value,
        C_mat: ir.Value,
        ssm_state_in: ir.Value,
    ):
        """Chunked SSD computation in ONNX ops.

        Implements the "ssd naive implementation without einsums" from
        HuggingFace's ``torch_forward``, translated to ONNX operations.

        All 5 stages of the SSD algorithm:
          1. Intra-chunk diagonal blocks (attention-like within each
             chunk).
          2. Inter-chunk state computation (B terms: how each chunk
             contributes to the running state).
          3. Inter-chunk SSM recurrence (A terms: decay across chunk
             boundaries).
          4. State-to-output per chunk (C terms: readout from
             propagated state).
          5. Combine intra-chunk and inter-chunk contributions.

        Args:
            op: ONNX op builder.
            x: (B, T, H, D) float32 — activated hidden (discretised).
            dt: (B, T, H) float32 — softplus(dt_raw + dt_bias).
            B_mat: (B, T, H, N) float32.
            C_mat: (B, T, H, N) float32.
            ssm_state_in: (B, H, D, N) — carry state.

        Returns:
            y: (B, T, H, D) float32 — output.
            new_ssm_state: (B, H, D, N).
        """
        CS = self.chunk_size
        H = self.num_heads
        D = self.d_head
        N = self.d_state

        # --- D residual: D[..., None] * x (before discretisation) ---
        D_param = op.Cast(self.ssm.D, to=ir.DataType.FLOAT)
        D_4d = op.Unsqueeze(D_param, [0, 1, 3])  # (1, 1, H, 1)
        D_residual = op.Mul(D_4d, x)  # (B, T, H, D)

        # --- Discretise ---
        # x = x * dt[..., None]: (B, T, H, D) * (B, T, H, 1)
        dt_4d = op.Unsqueeze(dt, [-1])
        x_disc = op.Mul(x, dt_4d)

        # A = -exp(A_log) in float32
        A_neg = op.Neg(op.Exp(op.Cast(self.ssm.A_log, to=ir.DataType.FLOAT)))
        # A_dt = A * dt: (H,) * (B, T, H) → (B, T, H)
        A_2d = op.Unsqueeze(A_neg, [0, 1])
        A_dt = op.Mul(A_2d, dt)

        # --- Reshape into chunks ---
        # Requires T divisible by CS (caller pads).
        x_chunked = op.Reshape(
            x_disc,
            op.Constant(value_ints=[0, -1, CS, H, D]),
        )  # (B, nc, CS, H, D)
        A_chunked = op.Reshape(
            A_dt,
            op.Constant(value_ints=[0, -1, CS, H]),
        )  # (B, nc, CS, H)
        B_chunked = op.Reshape(
            B_mat,
            op.Constant(value_ints=[0, -1, CS, H, N]),
        )
        C_chunked = op.Reshape(
            C_mat,
            op.Constant(value_ints=[0, -1, CS, H, N]),
        )

        # A_cumsum: cumsum of A within each chunk
        # (B, nc, CS, H) → permute to (B, H, nc, CS) for cumsum
        A_perm = op.Transpose(A_chunked, perm=[0, 3, 1, 2])
        A_cumsum = op.CumSum(A_perm, op.Constant(value_int=-1))

        # =============================================
        # 1. Intra-chunk (diagonal blocks)
        # =============================================
        # L = exp(segment_sum(A)): causal decay within chunk
        L = op.Exp(_segment_sum(op, A_perm, CS))  # (B,H,nc,CS,CS)

        # G = sum_n(C[l,n] * B[s,n]) — contraction over state dim
        C_exp = op.Unsqueeze(C_chunked, [3])  # (B,nc,CS,1,H,N)
        B_exp = op.Unsqueeze(B_chunked, [2])  # (B,nc,1,CS,H,N)
        G = op.ReduceSum(
            op.Mul(C_exp, B_exp),
            [-1],
            keepdims=False,
        )  # (B,nc,CS,CS,H)

        # M = G * L (permuted to match G layout)
        L_perm = op.Transpose(L, perm=[0, 2, 3, 4, 1])
        M = op.Mul(G, L_perm)  # (B, nc, CS, CS, H)

        # Y_diag = (M[...,None] * x_chunked[:,None]).sum(dim=3)
        M_exp = op.Unsqueeze(M, [-1])  # (B,nc,l,s,H,1)
        x_exp = op.Unsqueeze(x_chunked, [2])  # (B,nc,1,s,H,D)
        Y_diag = op.ReduceSum(
            op.Mul(M_exp, x_exp),
            [3],
            keepdims=False,
        )  # (B, nc, CS, H, D)

        # =============================================
        # 2. Inter-chunk state computation (B terms)
        # =============================================
        # decay_states = exp(A_last - A_cumsum)
        A_last = op.Slice(
            A_cumsum,
            starts=[-1],
            ends=[INT64_MAX],
            axes=[-1],
        )  # (B, H, nc, 1)
        decay_states = op.Exp(op.Sub(A_last, A_cumsum))

        # B_decay = B * decay_states (permuted)
        decay_perm = op.Transpose(
            decay_states,
            perm=[0, 2, 3, 1],
        )  # (B, nc, CS, H)
        decay_exp = op.Unsqueeze(decay_perm, [-1])
        B_decay = op.Mul(B_chunked, decay_exp)  # (B, nc, CS, H, N)

        # states = sum_over_chunk(B_decay * x_disc)
        B_decay_exp = op.Unsqueeze(B_decay, [-2])  # (B,nc,CS,H,1,N)
        x_disc_exp = op.Unsqueeze(x_chunked, [-1])  # (B,nc,CS,H,D,1)
        states = op.ReduceSum(
            op.Mul(B_decay_exp, x_disc_exp),
            [2],
            keepdims=False,
        )  # (B, nc, H, D, N)

        # =============================================
        # 3. Inter-chunk SSM recurrence (A terms)
        # =============================================
        # Prepend previous state along chunk dim
        prev = op.Unsqueeze(
            op.Cast(ssm_state_in, to=ir.DataType.FLOAT),
            [1],
        )  # (B, 1, H, D, N)
        states_cat = op.Concat(
            prev,
            states,
            axis=1,
        )  # (B, nc+1, H, D, N)

        # decay_chunk = exp(segment_sum(pad(A_ends, (1,0))))
        A_ends = op.Squeeze(
            op.Slice(
                A_cumsum,
                starts=[-1],
                ends=[INT64_MAX],
                axes=[-1],
            ),
            [-1],
        )  # (B, H, nc)
        A_ends_padded = op.Pad(
            A_ends,
            op.Constant(value_ints=[0, 0, 1, 0, 0, 0]),
            op.Constant(value_float=0.0),
        )  # (B, H, nc+1)
        decay_chunk = op.Exp(
            _segment_sum_dynamic(op, A_ends_padded),
        )  # (B, H, nc+1, nc+1)

        # Propagate state across chunks
        decay_chunk_t = op.Transpose(
            decay_chunk,
            perm=[0, 2, 3, 1],
        )  # (B, nc+1, nc+1, H)
        decay_exp2 = op.Unsqueeze(decay_chunk_t, [-1, -2])
        states_exp = op.Unsqueeze(states_cat, [1])
        new_states = op.ReduceSum(
            op.Mul(decay_exp2, states_exp),
            [2],
            keepdims=False,
        )  # (B, nc+1, H, D, N)

        # Split: first nc for output, last for carry
        states_out = op.Slice(
            new_states,
            starts=[0],
            ends=[-1],
            axes=[1],
        )  # (B, nc, H, D, N)
        new_ssm_state = op.Squeeze(
            op.Slice(
                new_states,
                starts=[-1],
                ends=[INT64_MAX],
                axes=[1],
            ),
            [1],
        )  # (B, H, D, N)

        # =============================================
        # 4. State → output per chunk (C terms)
        # =============================================
        state_decay_out = op.Exp(A_cumsum)  # (B, H, nc, CS)

        C_exp2 = op.Unsqueeze(C_chunked, [-2])  # (B,nc,CS,H,1,N)
        states_exp2 = op.Unsqueeze(states_out, [2])  # (B,nc,1,H,D,N)
        C_states_sum = op.ReduceSum(
            op.Mul(C_exp2, states_exp2),
            [-1],
            keepdims=False,
        )  # (B, nc, CS, H, D)

        sdo_perm = op.Transpose(
            state_decay_out,
            perm=[0, 2, 3, 1],
        )  # (B, nc, CS, H)
        sdo_exp = op.Unsqueeze(sdo_perm, [-1])
        Y_off = op.Mul(C_states_sum, sdo_exp)  # (B, nc, CS, H, D)

        # =============================================
        # 5. Combine
        # =============================================
        y = op.Add(Y_diag, Y_off)  # (B, nc, CS, H, D)

        # Reshape back to (B, T, H, D)
        bt_shape = op.Shape(x, start=0, end=2)
        y_shape = op.Concat(
            bt_shape,
            op.Constant(value_ints=[H, D]),
            axis=0,
        )
        y_reshaped = op.Reshape(y, y_shape)

        y_out = op.Add(y_reshaped, D_residual)  # (B, T, H, D)

        return y_out, new_ssm_state

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        conv_state: ir.Value,
        ssm_state: ir.Value,
    ):
        """Multi-token forward using the chunked SSD algorithm.

        Processes the full sequence in parallel within chunks of
        ``chunk_size`` tokens, then propagates SSM state across chunk
        boundaries.  Matching HF's ``torch_forward`` multi-token path.

        Note: seq_len must be divisible by chunk_size (the task layer
        is responsible for padding if necessary).
        """
        # Realize SSM parameters so they are visible as graph
        # initializers when referenced directly (not via self.ssm()).
        self._realize_submodule(op.builder, self.ssm)
        H = self.num_heads
        D = self.d_head
        N = self.d_state

        # Dtype-matched zero for Pad ops (avoids f32/f16 mismatch)
        pad_zero = op.CastLike(op.Constant(value_float=0.0), hidden_states)

        # Step 1: Batch-project all tokens at once
        projected = self.in_proj(op, hidden_states)  # (B, T, proj)
        gate, x_bc, dt_raw = op.Split(
            projected,
            [self.d_inner, self.conv_dim, self.num_heads],
            axis=-1,
            _outputs=3,
        )

        # Step 2: Causal Conv1D over full sequence
        x_bc_t = op.Transpose(x_bc, perm=[0, 2, 1])
        # Pad left with K-1 zeros for causal convolution
        padded = op.Pad(
            x_bc_t,
            op.Constant(
                value_ints=[0, 0, self.conv_kernel - 1, 0, 0, 0],
            ),
            pad_zero,
        )  # (B, conv_dim, T+K-1)
        conv_out_raw = self.conv1d(op, padded)
        # SiLU activation
        conv_activated = op.Mul(
            conv_out_raw,
            op.Sigmoid(conv_out_raw),
        )
        hidden_B_C = op.Transpose(conv_activated, perm=[0, 2, 1])

        # Extract new conv_state: last K-1 positions from the combined
        # old state + new tokens.  For T >= K-1 we could just slice
        # x_bc_t, but for T < K-1 (e.g. T=1 during decode) we need
        # to include history from the old conv_state.
        conv_combined = op.Concat(
            conv_state,
            x_bc_t,
            axis=2,
        )  # (B, conv_dim, K-1+T)
        new_conv_state = op.Slice(
            conv_combined,
            starts=[-(self.conv_kernel - 1)],
            ends=[INT64_MAX],
            axes=[2],
        )  # (B, conv_dim, K-1)

        # Step 3: Split xBC → x, B, C
        gs = self.n_groups * self.d_state
        x_split, B_split, C_split = op.Split(
            hidden_B_C,
            [self.d_inner, gs, gs],
            axis=-1,
            _outputs=3,
        )

        # Step 4: Prepare SSM inputs
        dt_bias = op.Cast(self.ssm.dt_bias, to=ir.DataType.FLOAT)
        dt_bias_3d = op.Unsqueeze(dt_bias, [0, 1])
        dt = op.Softplus(
            op.Add(
                op.Cast(dt_raw, to=ir.DataType.FLOAT),
                dt_bias_3d,
            )
        )
        # Clamp dt to time_step_min (matches HF torch.clamp(dt, min=...))
        if self.time_step_min > 0.0:
            dt = op.Clip(dt, op.Constant(value_float=self.time_step_min))

        # Reshape x to (B, T, H, D)
        x_4d = op.Reshape(
            op.Cast(x_split, to=ir.DataType.FLOAT),
            op.Constant(value_ints=[0, 0, H, D]),
        )
        # Reshape B, C to (B, T, n_groups, N) then expand to heads
        B_grouped = op.Reshape(
            op.Cast(B_split, to=ir.DataType.FLOAT),
            op.Constant(value_ints=[0, 0, self.n_groups, N]),
        )
        C_grouped = op.Reshape(
            op.Cast(C_split, to=ir.DataType.FLOAT),
            op.Constant(value_ints=[0, 0, self.n_groups, N]),
        )
        # Expand groups → heads
        B_exp = op.Unsqueeze(B_grouped, [3])
        B_expand = op.Expand(
            B_exp,
            op.Concat(
                op.Shape(B_exp, start=0, end=3),
                op.Constant(value_ints=[self.heads_per_group]),
                op.Shape(B_exp, start=4, end=5),
                axis=0,
            ),
        )
        B_heads = op.Reshape(
            B_expand,
            op.Constant(value_ints=[0, 0, H, N]),
        )

        C_exp = op.Unsqueeze(C_grouped, [3])
        C_expand = op.Expand(
            C_exp,
            op.Concat(
                op.Shape(C_exp, start=0, end=3),
                op.Constant(value_ints=[self.heads_per_group]),
                op.Shape(C_exp, start=4, end=5),
                axis=0,
            ),
        )
        C_heads = op.Reshape(
            C_expand,
            op.Constant(value_ints=[0, 0, H, N]),
        )

        # Step 5: Pad seq_len to a multiple of chunk_size
        # pad_size = (CS - T % CS) % CS
        CS_c = op.Constant(value_int=self.chunk_size)
        T_val = op.Shape(x_4d, start=1, end=2)  # [T]
        T_scalar = op.Squeeze(T_val, [0])
        pad_size = op.Mod(
            op.Sub(CS_c, op.Mod(T_scalar, CS_c)),
            CS_c,
        )
        # Pad along dim=1: pads = [0, 0, 0, pad_size, 0...0]
        # For 4D: (B, T, H, D) → pad format [d0b,d1b,d2b,d3b,
        #                                     d0e,d1e,d2e,d3e]
        pad_size_1d = op.Reshape(pad_size, op.Constant(value_ints=[1]))
        zero_1d = op.Constant(value_ints=[0])
        pads_4d = op.Concat(
            zero_1d,
            zero_1d,
            zero_1d,
            zero_1d,  # begins
            zero_1d,
            pad_size_1d,
            zero_1d,
            zero_1d,  # ends
            axis=0,
        )
        pads_3d = op.Concat(
            zero_1d,
            zero_1d,
            zero_1d,
            zero_1d,
            pad_size_1d,
            zero_1d,
            axis=0,
        )
        pad_zero_f32 = op.Constant(value_float=0.0)
        x_4d = op.Pad(x_4d, pads_4d, pad_zero_f32)
        dt = op.Pad(dt, pads_3d, pad_zero_f32)
        B_heads = op.Pad(B_heads, pads_4d, pad_zero_f32)
        C_heads = op.Pad(C_heads, pads_4d, pad_zero_f32)

        # Step 6: Chunked SSD
        y, new_ssm_state = self._chunked_ssd(
            op,
            x_4d,
            dt,
            B_heads,
            C_heads,
            ssm_state,
        )

        # Trim padding: (B, T_padded, H, D) → (B, T, H, D)
        y = op.Slice(
            y,
            starts=[0],
            ends=T_val,
            axes=[1],
        )

        # Step 7: GatedRMSNorm (expects 2D — flatten B*T)
        y_flat = op.Reshape(
            y,
            op.Constant(value_ints=[0, 0, self.d_inner]),
        )
        y_2d = op.Reshape(
            y_flat,
            op.Constant(value_ints=[-1, self.d_inner]),
        )
        gate_2d = op.Reshape(
            gate,
            op.Constant(value_ints=[-1, self.d_inner]),
        )
        y_normed = self.norm(op, y_2d, gate_2d)

        # Reshape back to (B, T, d_inner)
        bt_shape = op.Shape(hidden_states, start=0, end=2)
        y_shape = op.Concat(
            bt_shape,
            op.Constant(value_ints=[self.d_inner]),
            axis=0,
        )
        y_3d = op.Reshape(y_normed, y_shape)

        # Step 7: Output projection — cast back to input dtype
        y_proj = op.CastLike(y_3d, hidden_states)
        output = self.out_proj(op, y_proj)

        return output, new_conv_state, new_ssm_state
