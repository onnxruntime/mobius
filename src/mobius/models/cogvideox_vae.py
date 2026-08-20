# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""CogVideoX 3D causal video autoencoder.

Replicates HuggingFace diffusers' ``AutoencoderKLCogVideoX`` decoder path
(``CogVideoXDecoder3D``), the autoencoder used by every CogVideoX text-to-video
and image-to-video pipeline.

The decoder is *temporally causal*: each ``CogVideoXCausalConv3d`` pads only the
past side of the time axis (the first frame is replicated ``kernel_t - 1``
times), so output frame ``t`` never depends on latent frames after ``t``. That
property is what lets the reference implementation decode a long video in
latent-frame chunks while carrying a small ``conv_cache`` across the chunk
boundary.

Chunking is *not* transparent, however: the ``CogVideoXSpatialNorm3D`` group
normalizations reduce over the whole time axis of whatever is passed in, so the
statistics depend on the chunk. ``AutoencoderKLCogVideoX._decode`` always walks
the clip two latent frames at a time, and this module reproduces that boundary
exactly by exposing the same ``conv_cache`` tensors as graph inputs and outputs.
Decoding a clip in one call is only equivalent when it fits in a single
reference chunk (``T_latent <= 3``).

Inputs / outputs
----------------
- Decoder input ``latent_sample``: ``[B, latent_channels, T_latent, H, W]``
- Decoder output ``sample``: ``[B, out_channels, T_pixels, H * s, W * s]``

with ``T_pixels = 2 ** log2(temporal_compression_ratio) * (T_latent - 1) + 1``
for the odd latent-frame counts CogVideoX pipelines produce, and ``s`` the
spatial compression ratio.

Temporal resampling in the reference model is ``F.interpolate(..., mode
="nearest")``, which is index arithmetic; this module reproduces it with
``Gather``/``Resize`` so the graph stays fully dynamic in ``T``, ``H`` and ``W``
and never bakes in a frame count.
"""

from __future__ import annotations

import dataclasses
import math
import typing
from typing import TYPE_CHECKING

from onnxscript import OpBuilder, nn

from mobius.components import GroupNorm as _GroupNorm
from mobius.components import SiLU as _SiLU

if TYPE_CHECKING:
    import onnx_ir as ir
    import torch


@dataclasses.dataclass
class CogVideoXVAEConfig:
    """Configuration for ``AutoencoderKLCogVideoX``."""

    in_channels: int = 3
    out_channels: int = 3
    latent_channels: int = 16
    block_out_channels: tuple[int, ...] = (128, 256, 256, 512)
    layers_per_block: int = 3
    norm_num_groups: int = 32
    norm_eps: float = 1e-6
    temporal_compression_ratio: int = 4
    scaling_factor: float = 1.15258426
    use_post_quant_conv: bool = False
    use_quant_conv: bool = False

    @classmethod
    def from_diffusers(cls, config: dict) -> CogVideoXVAEConfig:
        if hasattr(config, "to_dict"):
            config = dict(config.items())
        return cls(
            in_channels=config.get("in_channels", 3),
            out_channels=config.get("out_channels", 3),
            latent_channels=config.get("latent_channels", 16),
            block_out_channels=tuple(config.get("block_out_channels", [128, 256, 256, 512])),
            layers_per_block=config.get("layers_per_block", 3),
            norm_num_groups=config.get("norm_num_groups", 32),
            norm_eps=config.get("norm_eps", 1e-6),
            temporal_compression_ratio=config.get("temporal_compression_ratio", 4),
            scaling_factor=config.get("scaling_factor", 1.15258426),
            use_post_quant_conv=config.get("use_post_quant_conv", False),
            use_quant_conv=config.get("use_quant_conv", False),
        )

    @property
    def spatial_compression_ratio(self) -> int:
        """Spatial downsampling factor implied by the block count."""
        return 2 ** (len(self.block_out_channels) - 1)


# ---------------------------------------------------------------------------
# Shape / resampling helpers
# ---------------------------------------------------------------------------

# ``Slice`` end sentinel for "to the end of the axis".
_INT64_MAX = 2**63 - 1


def _dim(op: OpBuilder, value: ir.Value, axis: int) -> ir.Value:
    """Single dimension of ``value`` as an int64 tensor of shape ``[1]``."""
    return op.Shape(value, start=axis, end=axis + 1)


def _nearest_indices(op: OpBuilder, out_len: ir.Value, in_len: ir.Value) -> ir.Value:
    """``floor(i * in_len / out_len)`` for ``i`` in ``[0, out_len)``.

    This is exactly PyTorch's ``mode="nearest"`` source-index rule
    (asymmetric coordinate transform with a floor rounding mode).
    """
    zero = op.Constant(value_ints=[0])
    one = op.Constant(value_ints=[1])
    positions = op.Range(op.Squeeze(zero, [0]), op.Squeeze(out_len, [0]), op.Squeeze(one, [0]))
    ratio = op.Div(op.Cast(in_len, to=1), op.Cast(out_len, to=1))
    source = op.Floor(op.Mul(op.Cast(positions, to=1), ratio))
    return op.Cast(source, to=7)


def _causal_nearest_indices(op: OpBuilder, out_len: ir.Value, in_len: ir.Value) -> ir.Value:
    """Temporal source indices used by ``CogVideoXSpatialNorm3D``.

    When the conditioned feature map has an odd number of frames greater than
    one, the reference implementation resamples the first frame on its own and
    the remaining frames as a separate group::

        idx[0] = 0
        idx[t] = 1 + floor((t - 1) * (T_in - 1) / (T_out - 1))    for t >= 1

    Otherwise it resamples the whole clip uniformly. Both index programs are
    computed and selected with ``Where`` so no ``If`` subgraph (and no static
    frame count) is needed.
    """
    zero = op.Constant(value_ints=[0])
    one = op.Constant(value_ints=[1])
    two = op.Constant(value_ints=[2])
    positions = op.Range(op.Squeeze(zero, [0]), op.Squeeze(out_len, [0]), op.Squeeze(one, [0]))
    uniform = _nearest_indices(op, out_len, in_len)

    # Split-first-frame program. ``max(out_len - 1, 1)`` keeps the divisor
    # well defined for the degenerate single-frame case, where only index 0 is
    # ever materialized anyway.
    out_rest = op.Max(op.Sub(out_len, one), one)
    in_rest = op.Sub(in_len, one)
    ratio = op.Div(op.Cast(in_rest, to=1), op.Cast(out_rest, to=1))
    shifted = op.Cast(op.Sub(positions, op.Squeeze(one, [0])), to=1)
    split = op.Add(
        op.Squeeze(one, [0]),
        op.Cast(op.Floor(op.Mul(shifted, op.Squeeze(ratio, [0]))), to=7),
    )
    split = op.Where(op.Equal(positions, op.Squeeze(zero, [0])), op.Squeeze(zero, [0]), split)

    is_odd = op.And(
        op.Equal(op.Mod(out_len, two), one),
        op.Greater(out_len, one),
    )
    return op.Where(op.Squeeze(is_odd, [0]), split, uniform)


def _align_conditioning(op: OpBuilder, zq: ir.Value, target: ir.Value) -> ir.Value:
    """Resample ``zq`` onto ``target``'s ``(T, H, W)`` grid, causally in time."""
    target_t = _dim(op, target, 2)
    target_h = _dim(op, target, 3)
    target_w = _dim(op, target, 4)

    time_indices = _causal_nearest_indices(op, target_t, _dim(op, zq, 2))
    zq = op.Gather(zq, time_indices, axis=2)

    # Spatial nearest resize; the temporal extent is already correct so the
    # requested size keeps it unchanged.
    sizes = op.Concat(
        op.Shape(zq, start=0, end=2),
        target_t,
        target_h,
        target_w,
        axis=0,
    )
    return op.Resize(
        zq,
        None,
        None,
        sizes,
        mode="nearest",
        coordinate_transformation_mode="asymmetric",
        nearest_mode="floor",
    )


# ---------------------------------------------------------------------------
# Layers
# ---------------------------------------------------------------------------


class _SafeConv3d(nn.Module):
    """Plain 3D convolution (HF ``CogVideoXSafeConv3d``, no temporal padding)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int, int] = (1, 1, 1),
        padding: tuple[int, int, int] = (0, 0, 0),
    ):
        super().__init__()
        self.weight = nn.Parameter((out_channels, in_channels, *kernel_size))
        self.bias = nn.Parameter((out_channels,))
        self._kernel_size = list(kernel_size)
        self._padding = list(padding)

    def forward(self, op: OpBuilder, x: ir.Value) -> ir.Value:
        return op.Conv(
            x,
            self.weight,
            self.bias,
            kernel_shape=self._kernel_size,
            strides=[1, 1, 1],
            pads=self._padding + self._padding,
        )


class _ConvCacheScope:
    """Carries HuggingFace's ``conv_cache`` tensors through the decoder modules.

    A CogVideoX clip is decoded a few latent frames at a time. Each causal
    convolution therefore has to start from the tail of the previous chunk
    instead of replicating its own first frame, exactly as the reference
    implementation's ``conv_cache`` dictionary does.
    """

    def __init__(self, inputs: dict[str, ir.Value]):
        self.inputs = inputs
        self.outputs: dict[str, ir.Value] = {}


class _CausalConv3d(nn.Module):
    """Temporally causal 3D convolution.

    The ``conv`` attribute mirrors HuggingFace's ``CogVideoXCausalConv3d.conv``
    so parameter names match the checkpoint exactly.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        spatial_pad = (kernel_size - 1) // 2
        self.conv = _SafeConv3d(
            in_channels,
            out_channels,
            (kernel_size, kernel_size, kernel_size),
            padding=(0, spatial_pad, spatial_pad),
        )
        self._time_pad = kernel_size - 1

    def forward(
        self,
        op: OpBuilder,
        x: ir.Value,
        scope: _ConvCacheScope | None = None,
        path: str = "",
    ) -> ir.Value:
        if self._time_pad == 0:
            return self.conv(op, x)

        previous = None if scope is None else scope.inputs.get(path)
        if previous is None:
            # Start of a clip: replicate the first latent frame ``k - 1`` times
            # so frame ``t`` only ever sees frames ``<= t``.
            first = op.Slice(
                x,
                op.Constant(value_ints=[0]),
                op.Constant(value_ints=[1]),
                op.Constant(value_ints=[2]),
            )
            x = op.Concat(*([first] * self._time_pad), x, axis=2)
        else:
            # A zero-length cache means "no previous chunk"; edge padding then
            # falls back to replicating frame 0, which is the same branch-free
            # expression as the clip-start case above.
            x = op.Concat(previous, x, axis=2)
            deficit = op.Sub(
                op.Constant(value_ints=[self._time_pad]),
                op.Min(_dim(op, previous, 2), op.Constant(value_ints=[self._time_pad])),
            )
            x = op.Pad(
                x,
                op.Concat(deficit, op.Constant(value_ints=[0]), axis=0),
                None,
                op.Constant(value_ints=[2]),
                mode="edge",
            )

        if scope is not None:
            # The reference caches the tail of the temporally padded input.
            scope.outputs[path] = op.Slice(
                x,
                op.Constant(value_ints=[-self._time_pad]),
                op.Constant(value_ints=[_INT64_MAX]),
                op.Constant(value_ints=[2]),
            )
        return self.conv(op, x)


class _SpatialNorm3D(nn.Module):
    """Latent-conditioned normalization (HF ``CogVideoXSpatialNorm3D``).

    The reference implementation always builds its ``GroupNorm`` with
    ``eps=1e-6``, independently of the autoencoder's ``norm_eps``.
    """

    def __init__(self, f_channels: int, zq_channels: int, groups: int):
        super().__init__()
        self.norm_layer = _GroupNorm(groups, f_channels, eps=1e-6)
        self.conv_y = _CausalConv3d(zq_channels, f_channels, kernel_size=1)
        self.conv_b = _CausalConv3d(zq_channels, f_channels, kernel_size=1)

    def forward(self, op: OpBuilder, f: ir.Value, zq: ir.Value) -> ir.Value:
        zq = _align_conditioning(op, zq, f)
        scale = self.conv_y(op, zq)
        shift = self.conv_b(op, zq)
        return op.Add(op.Mul(self.norm_layer(op, f), scale), shift)


class _ResnetBlock3D(nn.Module):
    """Latent-conditioned causal 3D residual block (HF ``CogVideoXResnetBlock3D``)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        zq_channels: int,
        groups: int,
    ):
        super().__init__()
        self.norm1 = _SpatialNorm3D(in_channels, zq_channels, groups)
        self.conv1 = _CausalConv3d(in_channels, out_channels, kernel_size=3)
        self.norm2 = _SpatialNorm3D(out_channels, zq_channels, groups)
        self.conv2 = _CausalConv3d(out_channels, out_channels, kernel_size=3)
        self.conv_shortcut = (
            _SafeConv3d(in_channels, out_channels, (1, 1, 1))
            if in_channels != out_channels
            else None
        )
        self._silu = _SiLU()

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        zq: ir.Value,
        scope: _ConvCacheScope | None = None,
        path: str = "",
    ) -> ir.Value:
        residual = hidden_states
        hidden_states = self.norm1(op, hidden_states, zq)
        hidden_states = self._silu(op, hidden_states)
        hidden_states = self.conv1(op, hidden_states, scope, f"{path}conv1")
        hidden_states = self.norm2(op, hidden_states, zq)
        hidden_states = self._silu(op, hidden_states)
        hidden_states = self.conv2(op, hidden_states, scope, f"{path}conv2")
        if self.conv_shortcut is not None:
            residual = self.conv_shortcut(op, residual)
        return op.Add(hidden_states, residual)


class _Upsample3D(nn.Module):
    """Spatial (and optionally temporal) 2x upsample (HF ``CogVideoXUpsample3D``).

    The trailing convolution is a per-frame ``Conv2d`` in HuggingFace. It is
    emitted here as a ``(1, 3, 3)`` 3D convolution, which is the same linear
    map without the reshape round trip; ``preprocess_weights`` inserts the
    singleton temporal axis into the checkpoint weight.
    """

    def __init__(self, channels: int, compress_time: bool):
        super().__init__()
        self.conv = _SafeConv3d(channels, channels, (1, 3, 3), padding=(0, 1, 1))
        self._compress_time = compress_time

    def forward(self, op: OpBuilder, hidden_states: ir.Value) -> ir.Value:
        if self._compress_time:
            frames = _dim(op, hidden_states, 2)
            one = op.Constant(value_ints=[1])
            two = op.Constant(value_ints=[2])
            is_odd = op.Equal(op.Mod(frames, two), one)
            # Odd clips keep frame 0 fixed and double the remainder, matching
            # the reference implementation's split-first-frame branch.
            odd_len = op.Sub(op.Mul(frames, two), one)
            even_len = op.Mul(frames, two)
            out_len = op.Where(is_odd, odd_len, even_len)

            positions = op.Range(
                op.Squeeze(op.Constant(value_ints=[0]), [0]),
                op.Squeeze(out_len, [0]),
                op.Squeeze(one, [0]),
            )
            zero_scalar = op.Squeeze(op.Constant(value_ints=[0]), [0])
            one_scalar = op.Squeeze(one, [0])
            two_scalar = op.Squeeze(two, [0])
            odd_indices = op.Where(
                op.Equal(positions, zero_scalar),
                zero_scalar,
                op.Add(one_scalar, op.Div(op.Sub(positions, one_scalar), two_scalar)),
            )
            even_indices = op.Div(positions, two_scalar)
            indices = op.Where(op.Squeeze(is_odd, [0]), odd_indices, even_indices)
            hidden_states = op.Gather(hidden_states, indices, axis=2)

        hidden_states = op.Resize(
            hidden_states,
            None,
            op.Constant(value_floats=[1.0, 1.0, 1.0, 2.0, 2.0]),
            None,
            mode="nearest",
            coordinate_transformation_mode="asymmetric",
            nearest_mode="floor",
        )
        return self.conv(op, hidden_states)


class _MidBlock3D(nn.Module):
    """Decoder mid block: two latent-conditioned residual blocks."""

    def __init__(self, channels: int, zq_channels: int, groups: int, layers: int):
        super().__init__()
        self.resnets = nn.ModuleList(
            [_ResnetBlock3D(channels, channels, zq_channels, groups) for _ in range(layers)]
        )

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        zq: ir.Value,
        scope: _ConvCacheScope | None = None,
        path: str = "",
    ) -> ir.Value:
        for index, resnet in enumerate(self.resnets):
            hidden_states = resnet(op, hidden_states, zq, scope, f"{path}resnets.{index}.")
        return hidden_states


class _UpBlock3D(nn.Module):
    """Decoder up block: residual blocks plus an optional upsampler."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        zq_channels: int,
        groups: int,
        layers: int,
        add_upsample: bool,
        compress_time: bool,
    ):
        super().__init__()
        self.resnets = nn.ModuleList(
            [
                _ResnetBlock3D(
                    in_channels if index == 0 else out_channels,
                    out_channels,
                    zq_channels,
                    groups,
                )
                for index in range(layers)
            ]
        )
        self.upsamplers = (
            nn.ModuleList([_Upsample3D(out_channels, compress_time)]) if add_upsample else None
        )

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        zq: ir.Value,
        scope: _ConvCacheScope | None = None,
        path: str = "",
    ) -> ir.Value:
        for index, resnet in enumerate(self.resnets):
            hidden_states = resnet(op, hidden_states, zq, scope, f"{path}resnets.{index}.")
        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(op, hidden_states)
        return hidden_states


class _CogVideoXDecoder3D(nn.Module):
    """Causal 3D decoder (HF ``CogVideoXDecoder3D``)."""

    def __init__(self, config: CogVideoXVAEConfig):
        super().__init__()
        reversed_channels = list(reversed(config.block_out_channels))
        zq_channels = config.latent_channels
        groups = config.norm_num_groups

        self.conv_in = _CausalConv3d(zq_channels, reversed_channels[0], kernel_size=3)
        self.mid_block = _MidBlock3D(reversed_channels[0], zq_channels, groups, layers=2)

        temporal_levels = int(math.log2(config.temporal_compression_ratio))
        self.up_blocks = nn.ModuleList()
        output_channel = reversed_channels[0]
        for index, channels in enumerate(reversed_channels):
            prev_output_channel = output_channel
            output_channel = channels
            self.up_blocks.append(
                _UpBlock3D(
                    prev_output_channel,
                    output_channel,
                    zq_channels,
                    groups,
                    layers=config.layers_per_block + 1,
                    add_upsample=index != len(reversed_channels) - 1,
                    compress_time=index < temporal_levels,
                )
            )

        self.norm_out = _SpatialNorm3D(reversed_channels[-1], zq_channels, groups)
        self.conv_out = _CausalConv3d(
            reversed_channels[-1], config.out_channels, kernel_size=3
        )
        self._silu = _SiLU()

    def forward(
        self,
        op: OpBuilder,
        latent_sample: ir.Value,
        scope: _ConvCacheScope | None = None,
    ) -> ir.Value:
        # The un-scaled latent conditions every normalization layer, so it is
        # threaded through the whole decoder as ``zq``.
        zq = latent_sample
        hidden_states = self.conv_in(op, latent_sample, scope, "conv_in")
        hidden_states = self.mid_block(op, hidden_states, zq, scope, "mid_block.")
        for index, up_block in enumerate(self.up_blocks):
            hidden_states = up_block(op, hidden_states, zq, scope, f"up_blocks.{index}.")
        hidden_states = self.norm_out(op, hidden_states, zq)
        hidden_states = self._silu(op, hidden_states)
        return self.conv_out(op, hidden_states, scope, "conv_out")


class ConvCacheEntry(typing.NamedTuple):
    """Shape contract of one carried ``conv_cache`` tensor.

    ``channels`` and ``spatial_scale`` are relative to the latent grid, so the
    tensor is ``[batch, channels, frames, latent_height * spatial_scale,
    latent_width * spatial_scale]``.
    """

    name: str
    channels: int
    frames: int
    spatial_scale: int


class AutoencoderKLCogVideoXModel(nn.Module):
    """CogVideoX 3D causal video autoencoder (decode path).

    Decodes ``[B, latent_channels, T_latent, H, W]`` latents into
    ``[B, out_channels, T_pixels, H * s, W * s]`` video frames.

    Replicates HuggingFace diffusers' ``AutoencoderKLCogVideoX``.
    """

    default_task: str = "video-vae"
    config_class = CogVideoXVAEConfig
    category: str = "Diffusion"

    def __init__(self, config: CogVideoXVAEConfig):
        super().__init__()
        self.config = config
        if config.use_post_quant_conv or config.use_quant_conv:
            raise NotImplementedError(
                "AutoencoderKLCogVideoX quant/post-quant convolutions are not built; "
                "every published CogVideoX checkpoint sets use_quant_conv=False and "
                "use_post_quant_conv=False."
            )
        self.decoder = _CogVideoXDecoder3D(config)

    def conv_cache_spec(self) -> list[ConvCacheEntry]:
        """Ordered contract of the carried ``conv_cache`` tensors.

        Mirrors the traversal order of :class:`_CogVideoXDecoder3D.forward`, so
        callers can size the state cells for a chunked decode without inspecting
        the graph. Only ``kernel_t > 1`` convolutions carry state; the ``k=1``
        convolutions inside the spatial norms consume no temporal context.
        """
        config = self.config
        reversed_channels = list(reversed(config.block_out_channels))
        frames = 2  # kernel_t - 1 for every cached convolution
        entries = [ConvCacheEntry("conv_in", config.latent_channels, frames, 1)]

        head = reversed_channels[0]
        for index in range(2):
            entries.append(ConvCacheEntry(f"mid_block.resnets.{index}.conv1", head, frames, 1))
            entries.append(ConvCacheEntry(f"mid_block.resnets.{index}.conv2", head, frames, 1))

        scale = 1
        output_channel = head
        for block, channels in enumerate(reversed_channels):
            prev_output_channel = output_channel
            output_channel = channels
            for layer in range(config.layers_per_block + 1):
                in_channels = prev_output_channel if layer == 0 else output_channel
                prefix = f"up_blocks.{block}.resnets.{layer}"
                entries.append(ConvCacheEntry(f"{prefix}.conv1", in_channels, frames, scale))
                entries.append(
                    ConvCacheEntry(f"{prefix}.conv2", output_channel, frames, scale)
                )
            if block != len(reversed_channels) - 1:
                # The upsampler doubles the spatial grid for every later block.
                scale *= 2

        entries.append(ConvCacheEntry("conv_out", reversed_channels[-1], frames, scale))
        return entries

    def forward(
        self,
        op: OpBuilder,
        latent_sample: ir.Value,
        conv_cache: dict[str, ir.Value] | None = None,
    ):
        """Decode one chunk of latent frames.

        Args:
            op: ONNX op builder.
            latent_sample: ``[B, latent_channels, T_latent, H, W]``.
            conv_cache: Carried temporal context per cached convolution. Pass
                zero-length (``frames == 0``) tensors for the first chunk of a
                clip; omit entirely to decode a clip in one call.

        Returns:
            The decoded frames, or ``(frames, updated_conv_cache)`` when
            ``conv_cache`` is supplied.
        """
        if conv_cache is None:
            return self.decoder(op, latent_sample)
        scope = _ConvCacheScope(conv_cache)
        sample = self.decoder(op, latent_sample, scope)
        return sample, scope.outputs

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Adapt the diffusers checkpoint to this module's parameter names.

        - Encoder parameters are dropped; only the decode path is built.
        - ``upsamplers.*.conv.weight`` gains a singleton temporal axis because
          the per-frame ``Conv2d`` is emitted as a ``(1, 3, 3)`` 3D convolution.
        """
        processed: dict[str, torch.Tensor] = {}
        for name, tensor in state_dict.items():
            if name.startswith("encoder."):
                continue
            if ".upsamplers." in name and name.endswith(".conv.weight") and tensor.ndim == 4:
                tensor = tensor.unsqueeze(2)
            processed[name] = tensor
        return processed
