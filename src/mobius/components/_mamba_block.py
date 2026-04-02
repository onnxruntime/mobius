# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Mamba block component: Conv1D → SSM → gated output projection.

Implements the standard Mamba layer used in Mamba, Jamba, Bamba,
FalconMamba, and related architectures. Composes a causal depthwise
Conv1D with the SelectiveScan SSM and a SiLU-gated output path.

Architecture per layer:
    1. in_proj: x → (x_branch, z_gate)  [expansion to d_inner]
    2. Causal depthwise Conv1D on x_branch
    3. SiLU activation
    4. Selective scan (SSM) with recurrent state
    5. Output gating: y * SiLU(z)
    6. out_proj: project back to d_model

State carried across steps:
    conv_state:  (batch, d_inner, conv_kernel - 1)
    ssm_state:   (batch, d_inner, d_state)

HuggingFace reference: ``MambaMixer``.
"""

from __future__ import annotations

import onnx_ir as ir
from onnxscript import nn
from onnxscript._internal import builder
from onnxscript.onnx_types import FLOAT

from mobius._flags import flags
from mobius.components._common import INT64_MAX, Linear
from mobius.components._rms_norm import GatedRMSNorm
from mobius.components._ssm import Mamba2Scan, SelectiveScan


class _DepthwiseConv1d(nn.Module):
    """Depthwise 1D convolution with optional bias.

    Each input channel is convolved with its own kernel (groups=channels).
    Used for causal convolution in the Mamba block.
    """

    def __init__(self, channels: int, kernel_size: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter([channels, 1, kernel_size])
        self.bias = nn.Parameter([channels]) if bias else None
        self._kernel_size = kernel_size
        self._channels = channels

    def forward(self, op: builder.OpBuilder, x: ir.Value):
        # x: (batch, channels, seq_len)
        result = op.Conv(
            x,
            self.weight,
            kernel_shape=[self._kernel_size],
            strides=[1],
            pads=[0, 0],
            group=self._channels,
        )
        if self.bias is not None:
            # bias: (channels,) → (1, channels, 1) for broadcasting
            bias_3d = op.Unsqueeze(self.bias, [0, 2])
            result = op.Add(result, bias_3d)
        return result


class MambaBlock(nn.Module):
    """Standard Mamba layer: input projection → Conv1D → SSM → gated output.

    Args:
        d_model: Model hidden dimension.
        d_inner: Expanded inner dimension (typically ``expand * d_model``).
        d_state: SSM state dimension (typically 16).
        dt_rank: Rank of the SSM time-step projection.
            Defaults to ``ceil(d_model / 16)`` (Mamba convention).
        conv_kernel: Causal Conv1D kernel size (typically 4).
    """

    def __init__(
        self,
        d_model: int,
        d_inner: int,
        d_state: int = 16,
        dt_rank: int | None = None,
        conv_kernel: int = 4,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_inner
        self.d_state = d_state
        self.conv_kernel = conv_kernel
        # Default dt_rank: ceil(d_model / 16) per Mamba convention
        self.dt_rank = dt_rank if dt_rank is not None else -(-d_model // 16)

        # Input projection: d_model → 2*d_inner (x_branch + z_gate)
        self.in_proj = Linear(d_model, 2 * d_inner, bias=False)

        # Causal depthwise Conv1D (with bias, matching HuggingFace)
        self.conv1d = _DepthwiseConv1d(d_inner, conv_kernel, bias=True)

        # Core SSM component
        self.ssm = SelectiveScan(d_inner, d_state, self.dt_rank)

        # Output projection: d_inner → d_model
        self.out_proj = Linear(d_inner, d_model, bias=False)

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        conv_state: ir.Value,
        ssm_state: ir.Value,
    ):
        """Single-token forward pass for the Mamba layer.

        Args:
            op: ONNX op builder.
            hidden_states: (batch, 1, d_model) — single token input.
            conv_state: (batch, d_inner, conv_kernel-1) — carry state.
            ssm_state: (batch, d_inner, d_state) — carry state.

        Returns:
            output: (batch, 1, d_model)
            new_conv_state: (batch, d_inner, conv_kernel-1)
            new_ssm_state: (batch, d_inner, d_state)
        """
        # --- Step 1: Input projection ---
        # projected: (batch, 1, 2*d_inner)
        projected = self.in_proj(op, hidden_states)

        # Split into x_branch and z_gate along last dim
        x_branch, z_gate = op.Split(
            projected,
            [self.d_inner, self.d_inner],
            axis=-1,
            _outputs=2,
        )
        # x_branch: (batch, 1, d_inner)
        # z_gate:   (batch, 1, d_inner)

        # --- Step 2: Causal Conv1D with state update ---
        # Transpose for conv: (batch, d_inner, 1)
        x_t = op.Transpose(x_branch, perm=[0, 2, 1])

        # Concatenate conv state + new token: (batch, d_inner, conv_kernel)
        conv_input = op.Concat(conv_state, x_t, axis=2)

        # Update conv state: drop oldest, keep last (conv_kernel-1)
        new_conv_state = op.Slice(
            conv_input, starts=[1], ends=[INT64_MAX], axes=[2],
        )

        # Apply depthwise conv: (batch, d_inner, 1)
        conv_out = self.conv1d(op, conv_input)

        # --- Step 3: SiLU activation ---
        conv_out = op.Mul(conv_out, op.Sigmoid(conv_out))

        # Transpose back: (batch, 1, d_inner)
        x_ssm = op.Transpose(conv_out, perm=[0, 2, 1])

        # --- Step 4: Selective scan ---
        y, new_ssm_state = self.ssm(op, x_ssm, ssm_state)
        # y: (batch, 1, d_inner)

        # --- Step 5: Output gating: y * SiLU(z) ---
        z_activated = op.Mul(z_gate, op.Sigmoid(z_gate))
        gated = op.Mul(y, z_activated)

        # --- Step 6: Output projection ---
        # output: (batch, 1, d_model)
        output = self.out_proj(op, gated)

        return output, new_conv_state, new_ssm_state


class Mamba2Block(nn.Module):
    """Mamba2/SSD block: in_proj -> Conv1D -> multi-head SSM -> gated norm.

    Supports arbitrary sequence lengths via an ONNX Scan that iterates
    token-by-token over the conv + SSM recurrence.  The input projection
    and output gating/projection are batched over the full sequence.

    Key differences from MambaBlock (Mamba1):
    - in_proj outputs [gate, xBC, dt] instead of [x, z]
    - Conv1D on wider xBC (conv_dim channels)
    - Multi-head SSM with grouped B/C
    - GatedRMSNorm instead of SiLU gating
    - dt direct from in_proj (no rank reduction), just bias

    HuggingFace reference: ``BambaMixer``.
    """

    def __init__(
        self,
        d_model: int,
        d_inner: int,
        num_heads: int,
        d_head: int,
        d_state: int,
        n_groups: int = 1,
        conv_kernel: int = 4,
        conv_bias: bool = True,
        proj_bias: bool = False,
        eps: float = 1e-5,
        norm_group_size: int | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_inner
        self.num_heads = num_heads
        self.d_head = d_head
        self.d_state = d_state
        self.n_groups = n_groups
        self.conv_kernel = conv_kernel
        self.heads_per_group = num_heads // n_groups

        self.conv_dim = d_inner + 2 * n_groups * d_state

        proj_size = d_inner + self.conv_dim + num_heads
        self.in_proj = Linear(d_model, proj_size, bias=proj_bias)
        self.conv1d = _DepthwiseConv1d(self.conv_dim, conv_kernel, bias=conv_bias)
        self.ssm = Mamba2Scan(num_heads, d_head, d_state, n_groups)
        self.norm = GatedRMSNorm(d_inner, eps=eps, group_size=norm_group_size)
        self.out_proj = Linear(d_inner, d_model, bias=proj_bias)

    @staticmethod
    def _realize_submodule(
        parent_builder: builder.GraphBuilder,
        submodule: nn.Module,
    ) -> None:
        """Register a submodule's parameters as parent-graph initializers.

        When Scan body graphs reference ``nn.Parameter`` objects from
        submodules, those parameters must be realized (name-qualified and
        registered) in the parent graph so they are visible as implicit
        inputs to the body subgraph.

        This pushes the submodule's naming context on the parent builder
        and calls ``_realize()`` on every parameter.  Since ``_realize``
        is idempotent, calling it again inside the Scan body trace
        function (via ``nn.Module.__call__``) is a harmless no-op.
        """
        name = submodule._name or ""
        parent_builder.push_module(name, type(submodule).__qualname__)
        for param in submodule._parameters.values():
            param._realize(parent_builder)
        for child in submodule._modules.values():
            Mamba2Block._realize_submodule(parent_builder, child)
        parent_builder.pop_module()

    def _scan_body(
        self,
        op: builder.OpBuilder,
        conv_state_in: ir.Value,
        ssm_state_in: ir.Value,
        xbc_in: ir.Value,
        dt_in: ir.Value,
    ):
        """Scan body: per-token conv state update + SSM step.

        Carry states (updated each iteration):
            conv_state: (B, conv_dim, conv_kernel-1)
            ssm_state:  (B, num_heads, d_head, d_state)

        Scan inputs (per-token, sliced along axis 1):
            xbc_in: (B, conv_dim) — projected xBC for this token
            dt_in:  (B, num_heads) — time step for this token

        Scan outputs (per-token, stacked along axis 1):
            y_t: (B, num_heads * d_head) — SSM output before norm

        Implicit inputs from parent graph (pre-realized by forward):
            conv1d.weight, conv1d.bias, ssm.A_log, ssm.D, ssm.dt_bias
        """
        # --- Conv state update (shift register) ---
        # xbc_in: (B, conv_dim) → (B, conv_dim, 1) for concat
        xbc_3d = op.Unsqueeze(xbc_in, [2])
        # Append new token to conv state: (B, conv_dim, K)
        conv_cat = op.Concat(conv_state_in, xbc_3d, axis=2)
        # New state = last K-1 positions
        new_conv_state = op.Slice(
            conv_cat, starts=[1], ends=[INT64_MAX], axes=[2],
        )

        # --- Apply conv1d (params already realized as implicit inputs) ---
        conv_out = self.conv1d(op, conv_cat)

        # SiLU activation, then squeeze: (B, conv_dim, 1) → (B, conv_dim)
        conv_out = op.Mul(conv_out, op.Sigmoid(conv_out))
        x_bc_act = op.Squeeze(conv_out, [2])

        # --- Split xBC → hidden_x, B, C ---
        gs = self.n_groups * self.d_state
        hidden_x, b_mat, c_mat = op.Split(
            x_bc_act,
            [self.d_inner, gs, gs],
            axis=-1,
            _outputs=3,
        )

        # --- SSM step (params already realized as implicit inputs) ---
        y_flat, new_ssm = self.ssm(
            op, hidden_x, dt_in, b_mat, c_mat, ssm_state_in,
        )

        return new_conv_state, new_ssm, y_flat

    def _forward_single_token(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        conv_state: ir.Value,
        ssm_state: ir.Value,
    ):
        """Single-token forward pass (no Scan, seq_len must be 1).

        This is the original implementation before the Scan-based
        multi-token support.  Useful for debugging.
        """
        # Step 1: Input projection -> gate, xBC, dt
        projected = self.in_proj(op, hidden_states)  # (B, 1, proj_size)
        gate, x_bc, dt = op.Split(
            projected,
            [self.d_inner, self.conv_dim, self.num_heads],
            axis=-1,
            _outputs=3,
        )

        # Step 2: Causal Conv1D with state update
        # x_bc: (B, 1, conv_dim) → (B, conv_dim, 1)
        x_bc_t = op.Transpose(x_bc, perm=[0, 2, 1])
        conv_input = op.Concat(conv_state, x_bc_t, axis=2)
        new_conv_state = op.Slice(
            conv_input, starts=[1], ends=[INT64_MAX], axes=[2],
        )
        conv_out = self.conv1d(op, conv_input)

        # Step 3: SiLU activation
        conv_out = op.Mul(conv_out, op.Sigmoid(conv_out))
        # (B, conv_dim, 1) → (B, 1, conv_dim) → squeeze → (B, conv_dim)
        x_bc_activated = op.Transpose(conv_out, perm=[0, 2, 1])

        # Step 4: Split xBC -> hidden, B, C
        gs = self.n_groups * self.d_state
        hidden_x, b_mat, c_mat = op.Split(
            x_bc_activated,
            [self.d_inner, gs, gs],
            axis=-1,
            _outputs=3,
        )

        # Squeeze seq dim for SSM: (B, 1, D) → (B, D)
        hidden_flat = op.Squeeze(hidden_x, [1])
        dt_flat = op.Squeeze(dt, [1])
        b_flat = op.Squeeze(b_mat, [1])
        c_flat = op.Squeeze(c_mat, [1])

        # Step 5: Multi-head selective scan
        y, new_ssm_state = self.ssm(op, hidden_flat, dt_flat, b_flat, c_flat, ssm_state)

        # Step 6: Gated RMSNorm
        gate_flat = op.Squeeze(gate, [1])
        y_normed = self.norm(op, y, gate_flat)

        # Restore seq dim: (B, d_inner) → (B, 1, d_inner)
        y_3d = op.Unsqueeze(y_normed, [1])

        # Step 7: Output projection
        output = self.out_proj(op, y_3d)  # (B, 1, d_model)

        return output, new_conv_state, new_ssm_state

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        conv_state: ir.Value,
        ssm_state: ir.Value,
    ):
        """Forward pass supporting arbitrary sequence length.

        When ``flags.mamba_scan`` is True (default), uses an ONNX Scan
        to iterate token-by-token, supporting multi-token sequences.
        When False, falls back to a single-token path (seq_len must be 1).

        Args:
            op: ONNX op builder.
            hidden_states: (batch, seq_len, d_model)
            conv_state: (batch, conv_dim, conv_kernel-1)
            ssm_state: (batch, num_heads, d_head, d_state)

        Returns:
            output: (batch, seq_len, d_model)
            new_conv_state: (batch, conv_dim, conv_kernel-1)
            new_ssm_state: (batch, num_heads, d_head, d_state)
        """
        if not flags.mamba_scan:
            return self._forward_single_token(
                op, hidden_states, conv_state, ssm_state,
            )
        # Realize conv1d and ssm parameters in the parent graph so they
        # are visible as implicit inputs to the Scan body.  _realize is
        # idempotent, so the re-call inside subgraph is a no-op.
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
        # gate: (B, T, d_inner), x_bc: (B, T, conv_dim), dt: (B, T, H)

        # Step 2: Scan over axis 1 (time).  The body graph is built via
        # GraphBuilder.subgraph which creates a child builder — its
        # scoped value names avoid SSA collisions with the parent graph.
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

        # Step 3: GatedRMSNorm (expects 2D input — flatten B*T)
        y_flat = op.Reshape(y_all, [-1, self.d_inner])
        gate_flat = op.Reshape(gate, [-1, self.d_inner])
        y_normed = self.norm(op, y_flat, gate_flat)  # (B*T, d_inner)

        # Reshape back to (B, T, d_inner) using input shape
        bt_shape = op.Shape(hidden_states, start=0, end=2)  # [B, T]
        y_shape = op.Concat(bt_shape, [self.d_inner], axis=0)
        y_3d = op.Reshape(y_normed, y_shape)

        # Step 4: Batch output projection
        output = self.out_proj(op, y_3d)  # (B, T, d_model)

        return output, new_conv_state, new_ssm_state
