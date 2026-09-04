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
- **StatefulMambaBlock**: source-compatible Mamba-1 with sequence prefill
  and an externally threaded convolution/SSM state ABI.

HuggingFace reference: ``MambaMixer``, ``BambaMixer``,
``NemotronHMamba2Mixer``.
"""

from __future__ import annotations

import math

import onnx_ir as ir
from onnxscript import OpBuilder, nn

from mobius.components._common import INT64_MAX, Linear
from mobius.components._rms_norm import GatedRMSNorm, PostGatedRMSNorm
from mobius.components._ssm import SelectiveScan, SequenceSelectiveScan


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

    def forward(self, op: OpBuilder, x: ir.Value):
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
        op: OpBuilder,
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
        conv_out = op.Swish(conv_out)

        # Transpose back: (batch, 1, d_inner)
        x_ssm = op.Transpose(conv_out, perm=[0, 2, 1])

        # --- Step 4: Selective scan ---
        y, new_ssm_state = self.ssm(op, x_ssm, ssm_state)
        # y: (batch, 1, d_inner)

        # --- Step 5: Output gating: y * SiLU(z) ---
        z_activated = op.Swish(z_gate)
        gated = op.Mul(y, z_activated)

        # --- Step 6: Output projection ---
        # output: (batch, 1, d_model)
        output = self.out_proj(op, gated)

        return output, new_conv_state, new_ssm_state


class _FloatBiasLinear(nn.Module):
    """Linear projection that accumulates its bias in float32."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter([out_features, in_features])
        self.bias = nn.Parameter([out_features])

    def forward(self, op: OpBuilder, value: ir.Value) -> ir.Value:
        product = op.MatMul(value, op.Transpose(self.weight))
        return op.Add(
            op.Cast(product, to=ir.DataType.FLOAT),
            op.Cast(self.bias, to=ir.DataType.FLOAT),
        )


class StatefulMambaBlock(nn.Module):
    """Mamba-1 over a token chunk with caller-owned convolution and SSM state.

    This is the portable sequence counterpart to :class:`MambaBlock`. It
    implements the ordinary Mamba-1 equations without Jamba's B/C/dt norms,
    and exposes the full recurrent state after every chunk. ``conv_state_width``
    may be either ``conv_kernel - 1`` (the conventional CausalConv state) or
    ``conv_kernel`` for references that retain the current raw convolution
    input in their public cache ABI.

    Inputs:
        hidden_states: ``[B, T, d_model]``.
        conv_state: ``[B, d_inner, conv_state_width]``.
        ssm_state: ``[B, d_inner, d_state]``.

    Returns:
        Output ``[B, T, d_model]``, updated convolution/SSM states, and the
        ungated SSM readout ``[B, T, d_inner]``. The final value lets a
        topology share Mamba memory without exposing it as persistent state.
    """

    def __init__(
        self,
        d_model: int,
        d_inner: int,
        d_state: int = 16,
        dt_rank: int | None = None,
        conv_kernel: int = 4,
        conv_state_width: int | None = None,
        conv_bias: bool = True,
        proj_bias: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_inner
        self.d_state = d_state
        self.dt_rank = dt_rank if dt_rank is not None else -(-d_model // 16)
        self.conv_kernel = conv_kernel
        self.conv_state_width = (
            conv_state_width if conv_state_width is not None else conv_kernel - 1
        )
        if self.conv_state_width not in (conv_kernel - 1, conv_kernel):
            raise ValueError(
                "StatefulMambaBlock conv_state_width must be conv_kernel - 1 or conv_kernel"
            )

        # Keep source-compatible parameter paths directly on the block:
        # in_proj → conv1d → (x_proj, dt_proj, A_log, D) → out_proj.
        self.in_proj = Linear(d_model, 2 * d_inner, bias=proj_bias)
        self.conv1d = _DepthwiseConv1d(d_inner, conv_kernel, bias=conv_bias)
        self.x_proj = Linear(d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = _FloatBiasLinear(self.dt_rank, d_inner)
        self.A_log = nn.Parameter([d_inner, d_state])
        self.D = nn.Parameter([d_inner])
        # Phi3Mamba declares its S4D spectrum and skip vector as fp32 even
        # when the surrounding checkpoint is bf16.
        self.A_log._keep_float32 = True
        self.D._keep_float32 = True
        self.out_proj = Linear(d_inner, d_model, bias=proj_bias)
        self.activation = FloatSwiGLU()

    def _repeat_for_channels(self, op: OpBuilder, value: ir.Value) -> ir.Value:
        """Expand token-wise B/C vectors across independent SSM channels."""
        value = op.Unsqueeze(value, [2])  # (B, T, 1, d_state)
        value = op.Tile(value, [1, 1, self.d_inner, 1])
        return op.Reshape(value, [0, 0, self.d_inner * self.d_state])

    def _conv_history(self, op: OpBuilder, conv_state: ir.Value) -> ir.Value:
        """Select the K-1 inputs needed by the causal depthwise convolution."""
        if self.conv_state_width == self.conv_kernel - 1:
            return conv_state
        return op.Slice(
            conv_state,
            starts=[1],
            ends=[INT64_MAX],
            axes=[2],
        )

    def _last_conv_state(self, op: OpBuilder, conv_input: ir.Value) -> ir.Value:
        """Preserve the caller's ABI width after appending this token chunk."""
        total_length = op.Shape(conv_input, start=2, end=3)
        starts = op.Sub(total_length, self.conv_state_width)
        return op.Slice(
            conv_input,
            starts=starts,
            ends=total_length,
            axes=[2],
        )

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        conv_state: ir.Value,
        ssm_state: ir.Value,
        padding_mask: ir.Value | None = None,
    ):
        """Process a prefill or decode chunk and return output plus all states."""
        projected = self.in_proj(op, hidden_states)  # (B, T, 2 * d_inner)
        x_branch, z_gate = op.Split(
            projected,
            [self.d_inner, self.d_inner],
            axis=-1,
            _outputs=2,
        )
        x_channels_first = op.Transpose(x_branch, perm=[0, 2, 1])  # (B, d_inner, T)
        if padding_mask is not None:
            x_channels_first = op.Mul(
                x_channels_first,
                op.Unsqueeze(op.CastLike(padding_mask, x_channels_first), [1]),
            )

        # The convolution receives the last K-1 raw inputs plus the current
        # chunk. A K-wide public cache retains one extra historical value so
        # references can update it in-place during single-token decode.
        conv_input = op.Concat(self._conv_history(op, conv_state), x_channels_first, axis=2)
        present_conv_state = self._last_conv_state(op, conv_input)
        x_ssm = op.Swish(self.conv1d(op, conv_input))  # (B, d_inner, T)
        if padding_mask is not None:
            x_ssm = op.Mul(x_ssm, op.Unsqueeze(op.CastLike(padding_mask, x_ssm), [1]))
        x_ssm = op.Transpose(x_ssm, perm=[0, 2, 1])  # (B, T, d_inner)

        # Mamba's selective recurrence is a gated linear attention where C
        # reads state, B writes state, and dt*x is the value vector.
        dt_raw, b_mat, c_mat = op.Split(
            self.x_proj(op, x_ssm),
            [self.dt_rank, self.d_state, self.d_state],
            axis=-1,
            _outputs=3,
        )
        # The source computes the BFloat16 rank projection first, then supplies
        # ``dt_proj.bias.float()`` as the selective-scan delta bias. This module
        # keeps the bias out of low-precision accumulation before fp32 Softplus
        # determines the recurrent decay.
        dt = op.Softplus(self.dt_proj(op, dt_raw))
        a_neg = op.Neg(op.Exp(op.Cast(self.A_log, to=ir.DataType.FLOAT)))
        decay = op.Reshape(
            op.Mul(op.Unsqueeze(dt, [-1]), op.Unsqueeze(a_neg, [0, 1])),
            [0, 0, self.d_inner * self.d_state],
        )
        ssm_output, present_state = op.LinearAttention(
            self._repeat_for_channels(op, op.Cast(c_mat, to=ir.DataType.FLOAT)),
            self._repeat_for_channels(op, op.Cast(b_mat, to=ir.DataType.FLOAT)),
            op.Mul(dt, op.Cast(x_ssm, to=ir.DataType.FLOAT)),
            op.Unsqueeze(op.Cast(ssm_state, to=ir.DataType.FLOAT), [-1]),
            decay,
            scale=1.0,
            q_num_heads=self.d_inner,
            kv_num_heads=self.d_inner,
            update_rule="gated",
            _domain="com.microsoft",
            _outputs=2,
        )
        ssm_output = op.Add(
            ssm_output,
            op.Mul(
                op.Cast(x_ssm, to=ir.DataType.FLOAT), op.Cast(self.D, to=ir.DataType.FLOAT)
            ),
        )
        ssm_output = op.CastLike(ssm_output, x_ssm)  # (B, T, d_inner)
        present_ssm_state = op.CastLike(op.Squeeze(present_state, [-1]), ssm_state)

        # The raw SSM readout is separately returned for shared-memory
        # topologies; the normal Mamba output still applies the SiLU z gate.
        output = self.out_proj(op, self.activation(op, z_gate, ssm_output))
        return output, present_conv_state, present_ssm_state, ssm_output


class FloatSwiGLU(nn.Module):
    """SwiGLU with float32 intermediate arithmetic and activation-shaped output.

    CUDA Jiterator implementations commonly evaluate
    ``float(gate) * float(value) * sigmoid(float(gate))`` before storing the
    result in the activation dtype. This stateless primitive preserves that
    behavior for model families whose source uses the fused kernel.
    """

    def forward(self, op: OpBuilder, gate: ir.Value, value: ir.Value) -> ir.Value:
        gate_f32 = op.Cast(gate, to=ir.DataType.FLOAT)
        value_f32 = op.Cast(value, to=ir.DataType.FLOAT)
        activated = op.Mul(op.Mul(gate_f32, value_f32), op.Sigmoid(gate_f32))
        return op.CastLike(activated, gate)


class GatedMemoryMixer(nn.Module):
    """Cross-memory gate ``out_proj(SiLU(in_proj(x)) * memory)``.

    It is useful for YOCO-style second-stage layers: the producer memory is
    transient within one forward call, while this mixer has no recurrent cache
    of its own.
    """

    def __init__(self, d_model: int, d_inner: int, bias: bool = False):
        super().__init__()
        self.in_proj = Linear(d_model, d_inner, bias=bias)
        self.out_proj = Linear(d_inner, d_model, bias=bias)
        self.activation = FloatSwiGLU()

    def forward(self, op: OpBuilder, hidden_states: ir.Value, memory: ir.Value) -> ir.Value:
        gate = self.in_proj(op, hidden_states)  # (B, T, d_inner)
        return self.out_proj(op, self.activation(op, gate, memory))


class SequenceMambaBlock(nn.Module):
    """Mamba1 layer applied to a whole sequence, with no carried state.

    Same parameters and math as :class:`MambaBlock` (input projection →
    causal Conv1D → selective scan → SiLU gate → output projection), but the
    entire ``(batch, seq_len, d_model)`` sequence is processed in one call
    and both the conv and SSM states start at zero.  This mirrors the
    reference ``mamba_ssm.Mamba`` module's offline forward pass and is what
    non-autoregressive backbones (e.g. speech enhancement) need.

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
        self.dt_rank = dt_rank if dt_rank is not None else -(-d_model // 16)

        self.in_proj = Linear(d_model, 2 * d_inner, bias=False)
        self.conv1d = _DepthwiseConv1d(d_inner, conv_kernel, bias=True)
        self.ssm = SequenceSelectiveScan(d_inner, d_state, self.dt_rank)
        self.out_proj = Linear(d_inner, d_model, bias=False)

    def forward(self, op: OpBuilder, hidden_states: ir.Value):
        """Run the Mamba layer over a full sequence.

        Args:
            op: ONNX op builder.
            hidden_states: (batch, seq_len, d_model).

        Returns:
            (batch, seq_len, d_model) output.
        """
        # --- Step 1: Input projection, split into SSM branch and gate ---
        projected = self.in_proj(op, hidden_states)  # (batch, seq_len, 2*d_inner)
        x_branch, z_gate = op.Split(
            projected,
            [self.d_inner, self.d_inner],
            axis=-1,
            _outputs=2,
        )

        # --- Step 2: Causal depthwise Conv1D over the sequence ---
        # (batch, seq_len, d_inner) → (batch, d_inner, seq_len) for Conv.
        x_t = op.Transpose(x_branch, perm=[0, 2, 1])
        # Left-pad by (conv_kernel - 1) so output position t only sees inputs
        # up to t; the conv itself uses pads=[0, 0] and keeps seq_len.
        x_padded = op.Pad(
            x_t,
            op.Constant(value_ints=[0, 0, self.conv_kernel - 1, 0, 0, 0]),
        )
        conv_out = op.Swish(self.conv1d(op, x_padded))  # (batch, d_inner, seq_len)

        # --- Step 3: Selective scan over the sequence ---
        x_ssm = op.Transpose(conv_out, perm=[0, 2, 1])  # (batch, seq_len, d_inner)
        y = self.ssm(op, x_ssm)  # (batch, seq_len, d_inner)

        # --- Step 4: Gating and output projection ---
        gated = op.Mul(y, op.Swish(z_gate))
        return self.out_proj(op, gated)


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
        op: OpBuilder,
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
        time_step_max: float | None = None,
        in_proj_bias: bool | None = None,
        out_proj_bias: bool | None = None,
        use_norm: bool = True,
        norm_before_gate: bool = False,
        input_multiplier: float = 1.0,
        projection_multipliers: tuple[float, float, float, float, float] = (
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
        ),
        output_multiplier: float = 1.0,
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
        self.time_step_max = time_step_max
        self.input_multiplier = input_multiplier
        self.projection_multipliers = projection_multipliers
        self.output_multiplier = output_multiplier

        self.conv_dim = d_inner + 2 * n_groups * d_state

        proj_size = d_inner + self.conv_dim + num_heads
        self.in_proj = Linear(
            d_model,
            proj_size,
            bias=proj_bias if in_proj_bias is None else in_proj_bias,
        )
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
        if use_norm:
            norm_cls = PostGatedRMSNorm if norm_before_gate else GatedRMSNorm
            self.norm = norm_cls(d_inner, eps=eps, group_size=norm_group_size)
        else:
            self.norm = None
        self.out_proj = Linear(
            d_inner,
            d_model,
            bias=proj_bias if out_proj_bias is None else out_proj_bias,
        )

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        conv_state: ir.Value,
        ssm_state: ir.Value,
        padding_mask: ir.Value | None = None,
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
        mamba_input = hidden_states
        if not math.isclose(self.input_multiplier, 1.0):
            mamba_input = op.Mul(mamba_input, self.input_multiplier)
        projected = self.in_proj(op, mamba_input)
        gate, x_bc, dt_raw = op.Split(
            projected,
            [self.d_inner, self.conv_dim, self.num_heads],
            axis=-1,
            _outputs=3,
        )
        gate_scale, x_scale, b_scale, c_scale, dt_scale = self.projection_multipliers
        if not math.isclose(gate_scale, 1.0):
            gate = op.Mul(gate, gate_scale)
        if not all(math.isclose(scale, 1.0) for scale in (x_scale, b_scale, c_scale)):
            x_hidden, b_mat, c_mat = op.Split(
                x_bc,
                [self.d_inner, self.n_groups * self.d_state, self.n_groups * self.d_state],
                axis=-1,
                _outputs=3,
            )
            if not math.isclose(x_scale, 1.0):
                x_hidden = op.Mul(x_hidden, x_scale)
            if not math.isclose(b_scale, 1.0):
                b_mat = op.Mul(b_mat, b_scale)
            if not math.isclose(c_scale, 1.0):
                c_mat = op.Mul(c_mat, c_scale)
            x_bc = op.Concat(x_hidden, b_mat, c_mat, axis=-1)
        if not math.isclose(dt_scale, 1.0):
            dt_raw = op.Mul(dt_raw, dt_scale)
        # gate: (B, T, d_inner) — gating signal for GatedRMSNorm
        # x_bc: (B, T, conv_dim) — input to conv1d
        # dt_raw: (B, T, num_heads) — raw time step

        # Step 2: CausalConvWithState + SiLU
        # Transpose to channels-first: (B, conv_dim, T)
        x_bc_t = op.Transpose(x_bc, perm=[0, 2, 1])
        conv_out, new_conv_state = self.conv1d(op, x_bc_t, conv_state)
        # Transpose back: (B, T, conv_dim)
        x_bc_activated = op.Transpose(conv_out, perm=[0, 2, 1])
        if padding_mask is not None:
            x_bc_activated = op.Mul(
                x_bc_activated,
                op.CastLike(padding_mask, x_bc_activated),
            )

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
        if self.time_step_min > 0.0 or self.time_step_max is not None:
            min_value = (
                op.Constant(value_float=self.time_step_min)
                if self.time_step_min > 0.0
                else None
            )
            max_value = (
                op.Constant(value_float=self.time_step_max)
                if self.time_step_max is not None
                else None
            )
            dt = op.Clip(dt, min_value, max_value)

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
        y_normed = (
            self.norm(op, y, gate) if self.norm is not None else op.Mul(y, op.Swish(gate))
        )

        # Step 10: Output projection
        output = self.out_proj(op, y_normed)
        if not math.isclose(self.output_multiplier, 1.0):
            output = op.Mul(output, self.output_multiplier)

        return output, new_conv_state, new_ssm_state

    def _expand_groups(self, op: OpBuilder, x: ir.Value) -> ir.Value:
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
