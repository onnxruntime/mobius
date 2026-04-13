# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Mamba2 ONNX Scan-based multi-token implementation.

Uses an ONNX Scan op to iterate over the sequence token-by-token,
carrying conv and SSM states across tokens.  Supports arbitrary
seq_len but is sequential (no intra-sequence parallelism).

This module contains the ``Mamba2BlockScan`` subclass.
"""

from __future__ import annotations

import onnx_ir as ir
from onnxscript._internal import builder
from onnxscript.onnx_types import FLOAT

from mobius.components._common import INT64_MAX
from mobius.components._mamba_block import Mamba2BlockBase


class Mamba2BlockScan(Mamba2BlockBase):
    """Mamba2 block using ONNX Scan (token-by-token iteration).

    Multi-token path that supports arbitrary seq_len but is sequential.
    The Scan body performs one conv state update + SSM step per token.
    """

    def _scan_body(
        self,
        op: builder.OpBuilder,
        conv_state: ir.Value,
        ssm_state: ir.Value,
        xbc_t: ir.Value,
        dt_t: ir.Value,
    ):
        """Scan body: per-token conv state update + SSM step.

        Carry states (updated each iteration):
            conv_state: (B, conv_dim, conv_kernel-1)
            ssm_state:  (B, num_heads, d_head, d_state)

        Scan inputs (per-token, sliced along axis 1):
            xbc_t: (B, conv_dim) -- projected xBC for this token
            dt_t:  (B, num_heads) -- time step for this token

        Scan outputs (per-token, stacked along axis 1):
            y_t: (B, num_heads * d_head) -- SSM output before norm
        """
        # --- Conv state update (shift register) ---
        xbc_3d = op.Unsqueeze(xbc_t, [2])
        conv_cat = op.Concat(conv_state, xbc_3d, axis=2)
        new_conv_state = op.Slice(
            conv_cat,
            starts=[1],
            ends=[INT64_MAX],
            axes=[2],
        )

        # --- Apply conv1d (params already realized as implicit inputs) ---
        conv_out = self.conv1d(op, conv_cat)

        # SiLU activation, then squeeze: (B, conv_dim, 1) -> (B, conv_dim)
        conv_out = op.Mul(conv_out, op.Sigmoid(conv_out))
        x_bc_act = op.Squeeze(conv_out, [2])

        # --- Split xBC -> hidden_x, B, C ---
        gs = self.n_groups * self.d_state
        hidden_x, b_mat, c_mat = op.Split(
            x_bc_act,
            [self.d_inner, gs, gs],
            axis=-1,
            _outputs=3,
        )

        # --- SSM step (params already realized as implicit inputs) ---
        y_flat, new_ssm = self.ssm(
            op,
            hidden_x,
            dt_t,
            b_mat,
            c_mat,
            ssm_state,
        )

        return new_conv_state, new_ssm, y_flat

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        conv_state: ir.Value,
        ssm_state: ir.Value,
    ):
        """Multi-token forward via ONNX Scan (token-by-token iteration).

        Uses an ONNX Scan op to iterate over the sequence, carrying conv
        and SSM states across tokens.  Supports arbitrary seq_len but is
        sequential (no intra-sequence parallelism).
        """
        # Realize conv1d and ssm parameters in the parent graph so they
        # are visible as implicit inputs to the Scan body.
        parent_builder = op.builder
        self._realize_submodule(parent_builder, self.conv1d)
        self._realize_submodule(parent_builder, self.ssm)

        # Step 1: Batch-project all tokens at once
        projected = self.in_proj(op, hidden_states)  # (B, T, proj_size)
        gate, x_bc, dt = op.Split(
            projected,
            [self.d_inner, self.conv_dim, self.num_heads],
            axis=-1,
            _outputs=3,
        )

        # Step 2: Scan over axis 1 (time)
        body = parent_builder.subgraph(
            self._scan_body,
            inputs={
                "conv_state": FLOAT[...],
                "ssm_state": FLOAT[...],
                "xbc_t": FLOAT[...],
                "dt_t": FLOAT[...],
            },
            outputs={
                "new_conv_state": FLOAT[...],
                "new_ssm_state": FLOAT[...],
                "y_t": FLOAT[...],
            },
            name="mamba2_recurrence",
        )
        new_conv_state, new_ssm_state, y_all = op.Scan(
            conv_state,
            ssm_state,
            x_bc,
            dt,
            body=body,
            num_scan_inputs=2,
            scan_input_axes=[1, 1],
            scan_output_axes=[1],
            _outputs=3,
        )
        # y_all: (B, T, d_inner)

        # Step 3: GatedRMSNorm (expects 2D -- flatten B*T)
        y_flat = op.Reshape(y_all, [-1, self.d_inner])
        gate_flat = op.Reshape(gate, [-1, self.d_inner])
        y_normed = self.norm(op, y_flat, gate_flat)

        # Reshape back to (B, T, d_inner)
        bt_shape = op.Shape(hidden_states, start=0, end=2)
        y_shape = op.Concat(bt_shape, [self.d_inner], axis=0)
        y_3d = op.Reshape(y_normed, y_shape)

        # Step 4: Batch output projection
        output = self.out_proj(op, y_3d)

        return output, new_conv_state, new_ssm_state
