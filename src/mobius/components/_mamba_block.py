# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Mamba block components: Conv1D → SSM → gated output projection.

This module provides:

- **MambaBlock** (Mamba1): standard Mamba layer for Mamba, Jamba,
  FalconMamba, etc.
- **Mamba2Block**: Mamba2 layer using ``com.microsoft.LinearAttention``
  with ``update_rule="gated"`` for the SSM recurrence and
  ``com.microsoft.CausalConvWithState`` for depthwise Conv1D.
  Supports both single-token decode (T=1) and multi-token prefill
  (T>1) in a single code path.

HuggingFace reference: ``MambaMixer``, ``BambaMixer``,
``NemotronHMamba2Mixer``.
"""

from __future__ import annotations

import onnx_ir as ir
from onnxscript import nn
from onnxscript._internal import builder

from mobius.components._common import INT64_MAX, Linear
from mobius.components._rms_norm import GatedRMSNorm
from mobius.components._ssm import SelectiveScan


class _DepthwiseConv1d(nn.Module):
    """Depthwise 1D convolution with optional bias.

    Each input channel is convolved with its own kernel (groups=channels).
    Used for causal convolution in the Mamba1 block.
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
            conv_input,
            starts=[1],
            ends=[INT64_MAX],
            axes=[2],
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


# =====================================================================
# Mamba2 block using LinearAttention
# =====================================================================


class _Mamba2DepthwiseConv1d(nn.Module):
    """Depthwise 1D convolution via CausalConvWithState function op.

    Wraps ``weight`` and optional ``bias`` parameters so that
    HuggingFace weight names (``conv1d.weight``, ``conv1d.bias``)
    automatically align with ONNX initializer names.

    The ``forward()`` method calls the ``CausalConvWithState``
    function op in the ``com.microsoft`` domain.
    """

    def __init__(self, channels: int, kernel_size: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter([channels, 1, kernel_size])
        self.bias = nn.Parameter([channels]) if bias else None
        self._channels = channels

    def forward(
        self,
        op: builder.OpBuilder,
        input_val: ir.Value,
        conv_state: ir.Value,
    ):
        """Run CausalConvWithState function op.

        Args:
            op: ONNX op builder.
            input_val: (B, D, T) — channels-first input.
            conv_state: (B, D, K-1) — carry state.

        Returns:
            output: (B, D, T) — convolution output with SiLU.
            present_state: (B, D, K-1) — updated carry state.
        """
        if self.bias is not None:
            conv_bias = self.bias
        else:
            # Zero bias — the function requires a bias input.
            conv_bias = op.Expand(
                op.CastLike(op.Constant(value_float=0.0), self.weight),
                op.Constant(value_ints=[self._channels]),
            )
        return op.CausalConvWithState(
            input_val,
            self.weight,
            conv_bias,
            conv_state,
            activation="silu",
            _domain="com.microsoft",
            _outputs=2,
        )


class Mamba2Block(nn.Module):
    """Mamba2 block using LinearAttention for the SSM recurrence.

    Uses ``com.microsoft.LinearAttention`` with ``update_rule="gated"``
    to express the Mamba2 SSD recurrence, and
    ``com.microsoft.CausalConvWithState`` for the depthwise Conv1D.
    Supports both single-token decode (T=1) and multi-token prefill
    (T>1) in a single unified code path.

    The Mamba2 SSD recurrence maps to LinearAttention as:

    - **query** = C (readout matrix), ``(B, T, num_heads * d_state)``
    - **key** = B (input matrix), ``(B, T, num_heads * d_state)``
    - **value** = dt * x (discretized input), ``(B, T, num_heads * d_head)``
    - **decay** = A * dt (log-decay), ``(B, T, num_heads)``

    B and C are expanded from ``n_groups`` to ``num_heads`` before
    passing to LinearAttention (each group's B/C is shared across
    ``heads_per_group`` heads).

    Key differences from MambaBlock (Mamba1):

    - in_proj outputs [gate, xBC, dt] instead of [x, z]
    - Conv1D on wider xBC (conv_dim channels)
    - Multi-head SSM with grouped B/C
    - GatedRMSNorm instead of SiLU gating
    - dt direct from in_proj (no rank reduction), just bias

    Args:
        d_model: Model hidden dimension.
        d_inner: Expanded inner dimension (``num_heads * d_head``).
        num_heads: Number of SSM heads.
        d_head: Per-head hidden dimension.
        d_state: SSM state dimension.
        n_groups: Number of B/C groups (``num_heads // n_groups``
            heads share the same B/C).
        chunk_size: Chunk size hint for LinearAttention (does not
            affect correctness).
        conv_kernel: Causal Conv1D kernel size (typically 4).
        conv_bias: Whether conv1d has a bias.
        proj_bias: Whether in_proj/out_proj have biases.
        eps: RMSNorm epsilon.
        norm_group_size: If set, GatedRMSNorm normalizes within
            groups of this size.
        time_step_min: Minimum dt clamp value (0 = no clamp).

    HuggingFace reference: ``Mamba2Mixer``, ``BambaMixer``,
    ``NemotronHMamba2Mixer``.
    """

    def __init__(
        self,
        d_model: int,
        d_inner: int,
        num_heads: int,
        d_head: int,
        d_state: int,
        n_groups: int = 1,
        chunk_size: int = 256,
        conv_kernel: int = 4,
        conv_bias: bool = True,
        proj_bias: bool = False,
        eps: float = 1e-5,
        norm_group_size: int | None = None,
        time_step_min: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_inner
        self.num_heads = num_heads
        self.d_head = d_head
        self.d_state = d_state
        self.n_groups = n_groups
        self.chunk_size = chunk_size
        self.conv_kernel = conv_kernel
        self.heads_per_group = num_heads // n_groups
        self.time_step_min = time_step_min

        self.conv_dim = d_inner + 2 * n_groups * d_state

        proj_size = d_inner + self.conv_dim + num_heads
        self.in_proj = Linear(d_model, proj_size, bias=proj_bias)
        self.conv1d = _Mamba2DepthwiseConv1d(
            self.conv_dim,
            conv_kernel,
            bias=conv_bias,
        )
        # SSM parameters directly on this module so they appear as
        # graph initializers. Weight paths: mamba.{A_log,D,dt_bias}
        # matching HuggingFace naming (no extra nesting needed).
        self.A_log = nn.Parameter([num_heads])
        self.D = nn.Parameter([num_heads])
        self.dt_bias = nn.Parameter([num_heads])
        self.norm = GatedRMSNorm(
            d_inner,
            eps=eps,
            group_size=norm_group_size,
        )
        self.out_proj = Linear(d_inner, d_model, bias=proj_bias)

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        conv_state: ir.Value,
        ssm_state: ir.Value,
    ):
        """Forward pass for the Mamba2 block.

        Args:
            op: ONNX op builder.
            hidden_states: (batch, seq_len, d_model)
            conv_state: (batch, conv_dim, conv_kernel-1)
            ssm_state: (batch, num_heads, d_state, d_head) — matches
                LinearAttention state layout (B, H, d_k, d_v).

        Returns:
            output: (batch, seq_len, d_model)
            new_conv_state: (batch, conv_dim, conv_kernel-1)
            new_ssm_state: (batch, num_heads, d_state, d_head)
        """
        # Step 1: Input projection → gate, xBC, dt
        # projected: (B, T, d_inner + conv_dim + num_heads)
        projected = self.in_proj(op, hidden_states)
        gate, x_bc, dt_raw = op.Split(
            projected,
            [self.d_inner, self.conv_dim, self.num_heads],
            axis=-1,
            _outputs=3,
        )
        # gate: (B, T, d_inner) — gating signal for GatedRMSNorm
        # x_bc: (B, T, conv_dim) — input to conv1d
        # dt_raw: (B, T, num_heads) — raw time step

        # Step 2: CausalConvWithState + SiLU
        # Transpose to channels-first: (B, conv_dim, T)
        x_bc_t = op.Transpose(x_bc, perm=[0, 2, 1])
        conv_out, new_conv_state = self.conv1d(op, x_bc_t, conv_state)
        # Transpose back: (B, T, conv_dim)
        x_bc_activated = op.Transpose(conv_out, perm=[0, 2, 1])

        # Step 3: Split xBC → x, B, C
        gs = self.n_groups * self.d_state
        x_hidden, b_mat, c_mat = op.Split(
            x_bc_activated,
            [self.d_inner, gs, gs],
            axis=-1,
            _outputs=3,
        )
        # x_hidden: (B, T, d_inner = num_heads * d_head)
        # b_mat: (B, T, n_groups * d_state)
        # c_mat: (B, T, n_groups * d_state)

        # Step 4: Compute dt and decay for LinearAttention.
        # Upcast to fp32 for softplus/exp to match HuggingFace which
        # computes the SSM recurrence in float32.
        dt_raw_f32 = op.Cast(dt_raw, to=ir.DataType.FLOAT)
        dt_bias_f32 = op.Cast(self.dt_bias, to=ir.DataType.FLOAT)
        a_log_f32 = op.Cast(self.A_log, to=ir.DataType.FLOAT)

        # dt = softplus(dt_raw + dt_bias): (B, T, num_heads)
        dt = op.Softplus(op.Add(dt_raw_f32, dt_bias_f32))
        if self.time_step_min > 0.0:
            dt = op.Clip(dt, op.Constant(value_float=self.time_step_min))

        # decay = A * dt in log-space: g_t where exp(g_t) is the decay
        # A = -exp(A_log), so decay = -exp(A_log) * dt
        a_neg = op.Neg(op.Exp(a_log_f32))  # (num_heads,)
        decay = op.Mul(a_neg, dt)  # (B, T, num_heads) in f32

        # Step 5: Prepare value = dt * x (absorb dt into input, in f32)
        # dt: (B, T, num_heads) → (B, T, num_heads, 1)
        dt_4d = op.Unsqueeze(dt, [-1])
        # x: (B, T, d_inner) → (B, T, num_heads, d_head), cast to f32
        x_f32 = op.Cast(x_hidden, to=ir.DataType.FLOAT)
        x_4d = op.Reshape(x_f32, [0, 0, self.num_heads, self.d_head])
        # value = dt * x: (B, T, num_heads, d_head)
        value_4d = op.Mul(dt_4d, x_4d)
        # Pack back: (B, T, num_heads * d_head)
        value = op.Reshape(value_4d, [0, 0, self.num_heads * self.d_head])

        # Step 6: Expand B and C from n_groups to num_heads (in f32)
        # Each group's B/C vector is shared across heads_per_group heads.
        b_expanded = self._expand_groups(op, op.Cast(b_mat, to=ir.DataType.FLOAT))
        c_expanded = self._expand_groups(op, op.Cast(c_mat, to=ir.DataType.FLOAT))

        # Step 7: Call LinearAttention (gated mode, all inputs in f32)
        # query = C: (B, T, num_heads * d_state) — d_k = d_state
        # key = B: (B, T, num_heads * d_state)
        # value = dt*x: (B, T, num_heads * d_head) — d_v = d_head
        # decay: (B, T, num_heads) — per-head scalar in log-space
        # state: (B, num_heads, d_state, d_head) = (B, H, d_k, d_v)
        ssm_state_f32 = op.Cast(ssm_state, to=ir.DataType.FLOAT)
        la_output, new_ssm_state = op.LinearAttention(
            c_expanded,
            b_expanded,
            value,
            ssm_state_f32,
            decay,
            scale=1.0,
            q_num_heads=self.num_heads,
            kv_num_heads=self.num_heads,
            update_rule="gated",
            _domain="com.microsoft",
            _outputs=2,
        )
        # la_output: (B, T, num_heads * d_head) in f32
        # new_ssm_state: (B, num_heads, d_state, d_head) in f32
        # Cast back to model dtype for downstream ops and state output
        la_output = op.CastLike(la_output, hidden_states)
        new_ssm_state = op.CastLike(new_ssm_state, ssm_state)

        # Step 8: D skip connection — y += D * x (per-head broadcast)
        # D: (num_heads,) → (1, 1, num_heads, 1) for broadcast
        # x_4d is still f32; cast D to f32 for the multiply, result cast
        # back via la_output's dtype.
        d_f32 = op.Cast(self.D, to=ir.DataType.FLOAT)
        d_4d = op.Reshape(d_f32, [1, 1, self.num_heads, 1])
        d_skip = op.CastLike(
            op.Reshape(
                op.Mul(d_4d, x_4d),
                [0, 0, self.num_heads * self.d_head],
            ),
            hidden_states,
        )
        y = op.Add(la_output, d_skip)

        # Step 9: GatedRMSNorm
        y_normed = self.norm(op, y, gate)

        # Step 10: Output projection
        output = self.out_proj(op, y_normed)

        return output, new_conv_state, new_ssm_state

    def _expand_groups(self, op: builder.OpBuilder, x: ir.Value) -> ir.Value:
        """Expand grouped B or C from n_groups to num_heads.

        Args:
            x: (B, T, n_groups * d_state)

        Returns:
            (B, T, num_heads * d_state) with each group's d_state
            vector replicated ``heads_per_group`` times.
        """
        if self.n_groups == self.num_heads:
            return x  # No expansion needed
        # (B, T, n_groups, 1, d_state)
        x_5d = op.Reshape(x, [0, 0, self.n_groups, 1, self.d_state])
        # Expand → (B, T, n_groups, heads_per_group, d_state)
        x_expanded = op.Expand(x_5d, [1, 1, 1, self.heads_per_group, 1])
        # Flatten → (B, T, num_heads * d_state)
        return op.Reshape(x_expanded, [0, 0, self.num_heads * self.d_state])
