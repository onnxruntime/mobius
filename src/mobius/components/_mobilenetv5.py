# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""MobileNet-V5 vision encoder (the Gemma 3n vision tower).

Gemma 3n's vision tower is not a ViT: HF maps ``model_type="gemma3n_vision"``
to ``TimmWrapperModel`` wrapping timm's ``mobilenetv5_300m_enc``, and the
weights live under ``model.vision_tower.timm_model.``.  timm is not a mobius
dependency, so the block layout is reproduced here from the architecture
definition in ``timm/models/mobilenetv5.py`` (``_gen_mobilenet_v5``).

Structure: a stride-2 stem, then 84 blocks across 4 stages, then a
Multi-Scale Fusion Adapter (MSFA) that fuses the last two stage outputs into
a fixed 16x16 grid of ``hidden_size`` channels.

    768x768 -> stem -> 384 -> stage0 -> 192 -> stage1 -> 96
            -> stage2 -> 48 (640ch) -> stage3 -> 24 (1280ch)
            -> MSFA(concat 1920ch, nearest-upsample stage3 to 48) -> 16x16x2048

Three block types appear, all named for weight compatibility with timm:

* ``EdgeResidual`` (stage 0) — ``conv_exp -> bn1 -> act -> conv_pwl -> bn2``.
* ``UniversalInvertedResidual`` / UIB — optional ``dw_start``, ``pw_exp``,
  optional ``dw_mid``, ``pw_proj``, ``layer_scale``.
* ``MobileAttention`` — pre-norm multi-query attention over the spatial grid,
  optionally downsampling K/V with a depthwise stride-2 conv.

Note on norms: timm names every norm ``.bn``, but with ``norm_layer=RmsNorm2d``
these are channel-axis RMSNorm with a scale only — see
:class:`~mobius.components._conv.RmsNorm2d`.  The checkpoint confirms this: 116
``bn.weight`` tensors and zero ``bn.bias`` / ``running_mean`` / ``running_var``.

Weight name alignment (HF -> ONNX), with the ``timm_model.`` prefix stripped:

- ``conv_stem.{conv.weight,conv.bias,bn.weight}``
- ``blocks.{stage}.{idx}.`` + per-type names above
- ``msfa.{ffn.pw_exp,ffn.pw_proj,norm}.*``
"""

from __future__ import annotations

from typing import NamedTuple

import onnx_ir as ir
from onnxscript import OpBuilder, nn

from mobius.components._activations import gelu_tanh
from mobius.components._conv import Conv2d, Conv2dNoBias, RmsNorm2d


class _EdgeResidualSpec(NamedTuple):
    """Stage-0 fused inverted bottleneck (``er_`` in the timm arch string)."""

    out_chs: int
    exp_chs: int
    kernel_size: int
    stride: int


class _UIBSpec(NamedTuple):
    """Universal Inverted Bottleneck (``uir_`` in the timm arch string).

    ``dw_start_kernel`` / ``dw_mid_kernel`` of 0 mean that depthwise conv is
    absent (timm uses ``nn.Identity``), which is why many blocks in the
    checkpoint have no ``dw_start`` or ``dw_mid`` tensors at all.
    """

    out_chs: int
    exp_chs: int
    dw_start_kernel: int
    dw_mid_kernel: int
    stride: int


class _MQASpec(NamedTuple):
    """Multi-query attention block (``mqa_`` in the timm arch string).

    ``kv_stride`` > 1 adds the depthwise ``down_conv`` + ``norm`` pair on the
    key and value branches that spatially downsamples K/V before projection.
    """

    out_chs: int
    num_heads: int
    kv_dim: int
    kv_stride: int
    dw_kernel_size: int


_BlockSpec = _EdgeResidualSpec | _UIBSpec | _MQASpec

_STEM_CHS = 64
_MSFA_OUT_RESOLUTION = 16
_MSFA_EXPANSION_RATIO = 2.0

# Per-stage block specs for ``mobilenetv5_300m_enc``, transcribed from the
# ``arch_def`` in timm's ``_gen_mobilenet_v5`` (the ``else`` branch, i.e. the
# 300m variant) and cross-checked against every tensor shape in the
# google/gemma-3n-E4B-it checkpoint.  There is no config source for this
# layout: HF's ``vision_config`` carries only the architecture *name*.
#
# Stages 2 and 3 alternate MQA and UIB blocks after their first block. Only
# stage 2's MQA blocks downsample K/V (kv_stride=2); stage 3's do not.
_MOBILENETV5_300M_ENC_BLOCKS: tuple[tuple[_BlockSpec, ...], ...] = (
    # Stage 0: 384x384 in -> 192x192 out, 128 channels.
    (
        _EdgeResidualSpec(out_chs=128, exp_chs=256, kernel_size=3, stride=2),
        _EdgeResidualSpec(out_chs=128, exp_chs=512, kernel_size=3, stride=1),
        _EdgeResidualSpec(out_chs=128, exp_chs=512, kernel_size=3, stride=1),
    ),
    # Stage 1: 192x192 in -> 96x96 out, 256 channels.
    (
        _UIBSpec(
            out_chs=256, exp_chs=768, dw_start_kernel=3, dw_mid_kernel=5, stride=2
        ),
        _UIBSpec(
            out_chs=256, exp_chs=1024, dw_start_kernel=5, dw_mid_kernel=0, stride=1
        ),
        _UIBSpec(
            out_chs=256, exp_chs=1024, dw_start_kernel=3, dw_mid_kernel=0, stride=1
        ),
        _UIBSpec(
            out_chs=256, exp_chs=1024, dw_start_kernel=5, dw_mid_kernel=0, stride=1
        ),
        _UIBSpec(
            out_chs=256, exp_chs=1024, dw_start_kernel=3, dw_mid_kernel=0, stride=1
        ),
    ),
    # Stage 2: 96x96 in -> 48x48 out, 640 channels. First MSFA input.
    (
        _UIBSpec(
            out_chs=640, exp_chs=1536, dw_start_kernel=5, dw_mid_kernel=5, stride=2
        ),
        *(
            _UIBSpec(
                out_chs=640, exp_chs=2560, dw_start_kernel=5, dw_mid_kernel=0, stride=1
            )
            for _ in range(7)
        ),
        _UIBSpec(
            out_chs=640, exp_chs=640, dw_start_kernel=0, dw_mid_kernel=0, stride=1
        ),
        *(
            spec
            for _ in range(14)
            for spec in (
                _MQASpec(
                    out_chs=640,
                    num_heads=12,
                    kv_dim=64,
                    kv_stride=2,
                    dw_kernel_size=3,
                ),
                _UIBSpec(
                    out_chs=640,
                    exp_chs=1280,
                    dw_start_kernel=0,
                    dw_mid_kernel=0,
                    stride=1,
                ),
            )
        ),
    ),
    # Stage 3: 48x48 in -> 24x24 out, 1280 channels. Second MSFA input.
    (
        _UIBSpec(
            out_chs=1280, exp_chs=3840, dw_start_kernel=5, dw_mid_kernel=5, stride=2
        ),
        *(
            spec
            for _ in range(19)
            for spec in (
                _MQASpec(
                    out_chs=1280,
                    num_heads=16,
                    kv_dim=96,
                    kv_stride=1,
                    dw_kernel_size=3,
                ),
                _UIBSpec(
                    out_chs=1280,
                    exp_chs=2560,
                    dw_start_kernel=0,
                    dw_mid_kernel=0,
                    stride=1,
                ),
            )
        ),
    ),
)


def _same_padding(
    kernel_size: int, stride: int, input_size: int
) -> tuple[int, int, int, int]:
    """TensorFlow-style ``SAME`` padding as an ONNX ``[top, left, bottom, right]``.

    timm builds ``mobilenetv5_300m_enc`` with ``pad_type="same"``, which pads
    asymmetrically (extra padding goes on the *end*) whenever the total pad is
    odd.  Symmetric ``padding=k // 2`` is therefore wrong in general.

    This needs ``input_size`` because SAME padding depends on the input
    extent, not just the kernel and stride.  For MobileNet-V5 at 768x768
    every stride-2 conv sees an even input, so the amounts are static and can
    be folded into the ONNX ``Conv`` ``pads`` attribute (no ``Pad`` node),
    keeping shape inference intact.
    """
    total = max((-(-input_size // stride) - 1) * stride + kernel_size - input_size, 0)
    begin, end = total // 2, total - total // 2
    return begin, begin, end, end


class _EdgeResidual(nn.Module):
    """Fused inverted bottleneck: expansion conv then pointwise projection.

    Unlike a standard inverted residual, the expansion is a full k x k conv
    (not depthwise), so there is no separate depthwise stage.

    timm class: ``EdgeResidual`` in ``timm/models/_efficientnet_blocks.py``.
    """

    def __init__(
        self,
        in_chs: int,
        spec: _EdgeResidualSpec,
        input_size: int,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.conv_exp = Conv2dNoBias(
            in_chs,
            spec.exp_chs,
            kernel_size=spec.kernel_size,
            stride=spec.stride,
            padding=_same_padding(spec.kernel_size, spec.stride, input_size),
        )
        self.bn1 = RmsNorm2d(spec.exp_chs, eps=norm_eps)
        self.conv_pwl = Conv2dNoBias(spec.exp_chs, spec.out_chs, kernel_size=1)
        self.bn2 = RmsNorm2d(spec.out_chs, eps=norm_eps)
        # Residual only when the block preserves both shape and channel count.
        self.has_skip = in_chs == spec.out_chs and spec.stride == 1

    def forward(self, op: OpBuilder, x: ir.Value) -> ir.Value:
        shortcut = x
        x = self.conv_exp(op, x)
        # bn1 applies the activation (timm's norm_act layer); bn2 does not.
        x = gelu_tanh(op, self.bn1(op, x))
        x = self.conv_pwl(op, x)
        x = self.bn2(op, x)
        if self.has_skip:
            x = op.Add(x, shortcut)
        return x


class _UniversalInvertedBottleneck(nn.Module):
    """UIB block: optional depthwise, pointwise expand, optional depthwise, project.

    The two optional depthwise convs are what makes the block "universal" —
    each of the four (start, mid) presence combinations yields a different
    classic block shape.  ``layer_scale`` scales the branch output before the
    residual add.

    timm class: ``UniversalInvertedResidual``.
    """

    def __init__(
        self,
        in_chs: int,
        spec: _UIBSpec,
        input_size: int,
        norm_eps: float = 1e-6,
        layer_scale: bool = True,
        noskip: bool = False,
    ):
        super().__init__()
        self.has_dw_start = spec.dw_start_kernel > 0
        self.has_dw_mid = spec.dw_mid_kernel > 0
        # timm applies the block stride to dw_mid when present, else dw_start.
        dw_start_stride = spec.stride if not self.has_dw_mid else 1

        if self.has_dw_start:
            self.dw_start = _ConvNorm(
                in_chs,
                in_chs,
                kernel_size=spec.dw_start_kernel,
                stride=dw_start_stride,
                groups=in_chs,
                padding=_same_padding(
                    spec.dw_start_kernel, dw_start_stride, input_size
                ),
                norm_eps=norm_eps,
            )
            input_size //= dw_start_stride

        self.pw_exp = _ConvNorm(in_chs, spec.exp_chs, norm_eps=norm_eps)

        if self.has_dw_mid:
            self.dw_mid = _ConvNorm(
                spec.exp_chs,
                spec.exp_chs,
                kernel_size=spec.dw_mid_kernel,
                stride=spec.stride,
                groups=spec.exp_chs,
                padding=_same_padding(spec.dw_mid_kernel, spec.stride, input_size),
                norm_eps=norm_eps,
            )

        self.pw_proj = _ConvNorm(spec.exp_chs, spec.out_chs, norm_eps=norm_eps)
        # The MSFA's FFN is built with layer_scale_init_value=None and
        # noskip=True, so it has neither a gamma tensor nor a residual.
        self.has_layer_scale = layer_scale
        if layer_scale:
            self.layer_scale = _LayerScale2d(spec.out_chs)
        self.has_skip = (
            in_chs == spec.out_chs and spec.stride == 1 and not noskip
        )

    def forward(self, op: OpBuilder, x: ir.Value) -> ir.Value:
        shortcut = x
        if self.has_dw_start:
            # dw_start is linear (apply_act=False in timm).
            x = self.dw_start(op, x)
        x = gelu_tanh(op, self.pw_exp(op, x))
        if self.has_dw_mid:
            x = gelu_tanh(op, self.dw_mid(op, x))
        # pw_proj is linear (apply_act=False in timm).
        x = self.pw_proj(op, x)
        if self.has_layer_scale:
            x = self.layer_scale(op, x)
        if self.has_skip:
            x = op.Add(x, shortcut)
        return x


class _ConvNorm(nn.Module):
    """``Conv2dNoBias`` + :class:`RmsNorm2d`, named ``conv`` / ``bn``.

    Mirrors timm's ``ConvNormAct`` weight layout (``.conv`` and ``.bn``); the
    activation is applied by the caller so that this stays a pure
    conv-then-norm pair.
    """

    def __init__(
        self,
        in_chs: int,
        out_chs: int,
        kernel_size: int = 1,
        stride: int = 1,
        groups: int = 1,
        padding: int | tuple[int, int, int, int] = 0,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.conv = Conv2dNoBias(
            in_chs,
            out_chs,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
        )
        self.bn = RmsNorm2d(out_chs, eps=norm_eps)

    def forward(self, op: OpBuilder, x: ir.Value) -> ir.Value:
        return self.bn(op, self.conv(op, x))


class _LayerScale2d(nn.Module):
    """Per-channel learnable scale for NCHW tensors (timm ``LayerScale2d``).

    Distinct from :class:`~mobius.components._codec_conv.LayerScale`, which
    names its parameter ``scale`` and broadcasts over the last axis; timm
    names it ``gamma`` and broadcasts over the channel axis.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.gamma = nn.Parameter([dim])

    def forward(self, op: OpBuilder, x: ir.Value) -> ir.Value:
        gamma = op.Reshape(op.CastLike(self.gamma, x), [1, -1, 1, 1])
        return op.Mul(x, gamma)


class _MultiQueryAttention2d(nn.Module):
    """Multi-query attention over a 2D feature map.

    Multi-query: ``num_heads`` query heads share a *single* key head and a
    single value head, so K/V project to ``kv_dim`` rather than
    ``num_heads * kv_dim``.  All projections are 1x1 convs on NCHW input.

    When ``kv_stride`` > 1, K and V are first spatially downsampled by a
    depthwise stride-2 conv (``down_conv``) followed by a norm, cutting the
    attention key/value length by 4x.

    timm class: ``MultiQueryAttention2d`` in ``timm/layers/attention2d.py``.
    """

    def __init__(
        self,
        dim: int,
        spec: _MQASpec,
        input_size: int,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.num_heads = spec.num_heads
        self.kv_dim = spec.kv_dim
        self.kv_stride = spec.kv_stride
        self.scale = spec.kv_dim**-0.5

        self.query = _MQAQuery(dim, spec.num_heads * spec.kv_dim)
        self.key = _MQAKeyValue(
            dim,
            spec.kv_dim,
            kv_stride=spec.kv_stride,
            dw_kernel_size=spec.dw_kernel_size,
            input_size=input_size,
            norm_eps=norm_eps,
        )
        self.value = _MQAKeyValue(
            dim,
            spec.kv_dim,
            kv_stride=spec.kv_stride,
            dw_kernel_size=spec.dw_kernel_size,
            input_size=input_size,
            norm_eps=norm_eps,
        )
        self.output = _MQAOutput(spec.num_heads * spec.kv_dim, spec.out_chs)
        # The query branch is never strided in this architecture, so the
        # output grid matches the input grid.
        self._resolution = input_size

    def forward(self, op: OpBuilder, x: ir.Value) -> ir.Value:
        # Q: [B, H*K, h, w] -> [B, h*w, H, K], the layout attention expects.
        q = self.query(op, x)
        q = op.Reshape(q, [0, self.num_heads, self.kv_dim, -1])
        q = op.Transpose(q, perm=[0, 3, 1, 2])

        # K/V are single-head: [B, K, h', w'] -> [B, m, 1, K].
        k = _flatten_spatial(op, self.key(op, x), self.kv_dim)
        v = _flatten_spatial(op, self.value(op, x), self.kv_dim)

        attn_out = _multi_query_attention(op, q, k, v, scale=self.scale)

        # [B, n, H, K] -> [B, H*K, h, w]; n == h*w as the query is not strided.
        attn_out = op.Transpose(attn_out, perm=[0, 2, 3, 1])
        attn_out = op.Reshape(
            attn_out,
            [0, self.num_heads * self.kv_dim, self._resolution, self._resolution],
        )
        return self.output(op, attn_out)


def _flatten_spatial(op: OpBuilder, x: ir.Value, dim: int) -> ir.Value:
    """NCHW ``[B, dim, h, w]`` -> ``[B, h*w, 1, dim]`` (a single KV head)."""
    x = op.Reshape(x, [0, dim, -1])
    x = op.Transpose(x, perm=[0, 2, 1])
    return op.Unsqueeze(x, [2])


def _multi_query_attention(
    op: OpBuilder,
    q: ir.Value,
    k: ir.Value,
    v: ir.Value,
    scale: float,
) -> ir.Value:
    """Unmasked attention with ``num_heads`` query heads over one KV head.

    Inputs and output are ``[batch, seq, heads, dim]``.  Emitted as explicit
    MatMul/Softmax rather than an ``Attention`` op: the fused kernels take
    ``[B, H, S, D]`` and, more importantly, would need GQA-style broadcasting
    from 1 KV head to ``num_heads``, which not every EP supports.
    """
    # [B, S, H, D] -> [B, H, S, D]
    q = op.Transpose(q, perm=[0, 2, 1, 3])
    k = op.Transpose(k, perm=[0, 2, 1, 3])
    v = op.Transpose(v, perm=[0, 2, 1, 3])

    # Attention in float32: 48x48 grids make the logit sums large enough to
    # lose precision in float16, and Softmax is sensitive to that.
    q = op.Cast(q, to=ir.DataType.FLOAT)
    k_f32 = op.Cast(k, to=ir.DataType.FLOAT)
    v_f32 = op.Cast(v, to=ir.DataType.FLOAT)

    # [B, H, n, D] @ [B, 1, D, m] -> [B, H, n, m]; the single KV head
    # broadcasts across all query heads.
    logits = op.MatMul(op.Mul(q, scale), op.Transpose(k_f32, perm=[0, 1, 3, 2]))
    probs = op.Softmax(logits, axis=-1)
    out = op.MatMul(probs, v_f32)

    out = op.CastLike(out, v)
    # Back to [B, S, H, D]
    return op.Transpose(out, perm=[0, 2, 1, 3])


class _MQAQuery(nn.Module):
    """Query branch: a single 1x1 projection conv (no norm, no downsampling).

    A ``nn.Module`` rather than a bare conv so the weight name is
    ``query.proj.weight``, matching timm's ``nn.Sequential`` layout.
    """

    def __init__(self, dim: int, out_dim: int):
        super().__init__()
        self.proj = Conv2dNoBias(dim, out_dim, kernel_size=1)

    def forward(self, op: OpBuilder, x: ir.Value) -> ir.Value:
        return self.proj(op, x)


class _MQAKeyValue(nn.Module):
    """Key or value branch: optional depthwise downsample, then 1x1 projection.

    ``down_conv`` and ``norm`` exist only when ``kv_stride > 1``; the
    checkpoint has them on stage 2's MQA blocks and not on stage 3's.
    """

    def __init__(
        self,
        dim: int,
        kv_dim: int,
        kv_stride: int,
        dw_kernel_size: int,
        input_size: int,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.has_down = kv_stride > 1
        if self.has_down:
            self.down_conv = Conv2dNoBias(
                dim,
                dim,
                kernel_size=dw_kernel_size,
                stride=kv_stride,
                padding=_same_padding(dw_kernel_size, kv_stride, input_size),
                groups=dim,
            )
            self.norm = RmsNorm2d(dim, eps=norm_eps)
        self.proj = Conv2dNoBias(dim, kv_dim, kernel_size=1)

    def forward(self, op: OpBuilder, x: ir.Value) -> ir.Value:
        if self.has_down:
            x = self.norm(op, self.down_conv(op, x))
        return self.proj(op, x)


class _MQAOutput(nn.Module):
    """Output branch: a single 1x1 projection conv back to ``out_chs``."""

    def __init__(self, in_dim: int, out_chs: int):
        super().__init__()
        self.proj = Conv2dNoBias(in_dim, out_chs, kernel_size=1)

    def forward(self, op: OpBuilder, x: ir.Value) -> ir.Value:
        return self.proj(op, x)


class _MobileAttention(nn.Module):
    """Pre-norm multi-query attention block with a residual connection.

    ``norm -> attn -> layer_scale -> + shortcut``.  timm class:
    ``MobileAttention`` (with ``use_multi_query=True``).
    """

    def __init__(
        self,
        in_chs: int,
        spec: _MQASpec,
        input_size: int,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.norm = RmsNorm2d(in_chs, eps=norm_eps)
        self.attn = _MultiQueryAttention2d(
            in_chs, spec, input_size=input_size, norm_eps=norm_eps
        )
        self.layer_scale = _LayerScale2d(spec.out_chs)
        self.has_skip = in_chs == spec.out_chs

    def forward(self, op: OpBuilder, x: ir.Value) -> ir.Value:
        shortcut = x
        # The pre-norm here has apply_act=False in timm — no activation.
        out = self.norm(op, x)
        out = self.attn(op, out)
        out = self.layer_scale(op, out)
        if self.has_skip:
            out = op.Add(out, shortcut)
        return out


class _MobileNetV5MSFA(nn.Module):
    """Multi-Scale Fusion Adapter: fuse two stage outputs into a fixed grid.

    The lower-resolution input is nearest-upsampled to the higher resolution,
    the two are concatenated on the channel axis, passed through a UIB-style
    FFN (no depthwise convs), and then average-pooled down to
    ``output_resolution``.  For Gemma 3n that final grid is 16x16 = 256, which
    is exactly ``vision_soft_tokens_per_image``.

    timm class: ``MobileNetV5MultiScaleFusionAdapter``.
    """

    def __init__(
        self,
        in_chs: int,
        out_chs: int,
        input_resolutions: tuple[int, ...],
        output_resolution: int = _MSFA_OUT_RESOLUTION,
        expansion_ratio: float = _MSFA_EXPANSION_RATIO,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        fusion_resolution = input_resolutions[0]
        self.ffn = _UniversalInvertedBottleneck(
            in_chs,
            _UIBSpec(
                out_chs=out_chs,
                exp_chs=int(in_chs * expansion_ratio),
                dw_start_kernel=0,
                dw_mid_kernel=0,
                stride=1,
            ),
            input_size=fusion_resolution,
            norm_eps=norm_eps,
            layer_scale=False,
            noskip=True,
        )
        self.norm = RmsNorm2d(out_chs, eps=norm_eps)
        # Upsample factor per trailing input, relative to the fusion resolution.
        self._upsample_factors = tuple(
            fusion_resolution // res for res in input_resolutions[1:]
        )
        for res, factor in zip(input_resolutions[1:], self._upsample_factors):
            if res * factor != fusion_resolution:
                raise ValueError(
                    f"MSFA input resolution {res} does not divide the fusion "
                    f"resolution {fusion_resolution} evenly"
                )
        self._fusion_resolution = fusion_resolution
        self._output_resolution = output_resolution

    def forward(self, op: OpBuilder, features: list[ir.Value]) -> ir.Value:
        high, *rest = features
        resized = [high]
        for feat, factor in zip(rest, self._upsample_factors):
            # Nearest upsample to the highest input resolution. The factor is
            # a compile-time integer (each stage halves the grid), so constant
            # `scales` keep shape inference exact — and unlike `sizes` they
            # need no value for the dynamic batch axis.
            resized.append(
                op.Resize(
                    feat,
                    None,
                    op.Constant(value_floats=[1.0, 1.0, float(factor), float(factor)]),
                    mode="nearest",
                    nearest_mode="floor",
                    coordinate_transformation_mode="asymmetric",
                )
            )
        x = op.Concat(*resized, axis=1) if len(resized) > 1 else resized[0]
        x = self.ffn(op, x)

        stride = self._fusion_resolution // self._output_resolution
        if stride > 1:
            # Input resolution is an exact multiple of the output, so timm
            # takes the average-pool path rather than bilinear interpolation.
            x = op.AveragePool(
                x, kernel_shape=[stride, stride], strides=[stride, stride]
            )
        return self.norm(op, x)


class MobileNetV5Encoder(nn.Module):
    """MobileNet-V5-300m vision encoder producing a 16x16 grid of features.

    Consumes a fixed-size NCHW image (768x768 for Gemma 3n — MobileNet-V5 has
    no dynamic-resolution path) and returns ``[batch, hidden_size, 16, 16]``.

    Args:
        hidden_size: Output channel count (``vision_config.hidden_size``, 2048).
        image_size: Input spatial resolution. Must be divisible by 32 so that
            every stride-2 stage sees an even extent and SAME padding stays
            static.
        norm_eps: Epsilon for every :class:`RmsNorm2d` in the tower.
    """

    def __init__(
        self,
        hidden_size: int = 2048,
        image_size: int = 768,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        # 5 stride-2 reductions (stem + one per stage) must leave an integer
        # grid, and the MSFA average-pool needs its input to be a multiple of
        # the 16x16 output.
        if image_size % 32 != 0:
            raise ValueError(
                f"MobileNet-V5 image_size must be divisible by 32, got {image_size}"
            )
        # Stage 2 (the first MSFA input, and the resolution it fuses at) sits
        # 4 stride-2 reductions below the input.
        if (image_size // 16) % _MSFA_OUT_RESOLUTION != 0:
            raise ValueError(
                f"image_size {image_size} gives a stage-2 resolution of "
                f"{image_size // 16}, which is not a multiple of the "
                f"{_MSFA_OUT_RESOLUTION}x{_MSFA_OUT_RESOLUTION} MSFA output grid"
            )

        self.conv_stem = _ConvStem(
            _STEM_CHS,
            padding=_same_padding(3, 2, image_size),
            norm_eps=norm_eps,
        )

        num_stages = len(_MOBILENETV5_300M_ENC_BLOCKS)
        stages: list[nn.ModuleList] = []
        in_chs = _STEM_CHS
        resolution = image_size // 2
        # The MSFA fuses the last two stage outputs. Their channel counts sum,
        # and the fusion happens at the *first* (higher) of the two
        # resolutions — the lower one is upsampled to meet it.
        msfa_in_chs = 0
        msfa_resolutions: list[int] = []
        for stage_idx, stage_specs in enumerate(_MOBILENETV5_300M_ENC_BLOCKS):
            blocks = []
            for spec in stage_specs:
                blocks.append(_make_block(in_chs, spec, resolution, norm_eps))
                in_chs = spec.out_chs
                # MQA blocks never change the spatial extent; only the strided
                # depthwise convs inside EdgeResidual/UIB blocks do.
                if not isinstance(spec, _MQASpec):
                    resolution //= spec.stride
            stages.append(nn.ModuleList(blocks))
            if stage_idx >= num_stages - 2:
                msfa_in_chs += in_chs
                msfa_resolutions.append(resolution)
        self.blocks = nn.ModuleList(stages)

        self.msfa = _MobileNetV5MSFA(
            msfa_in_chs,
            hidden_size,
            input_resolutions=tuple(msfa_resolutions),
            norm_eps=norm_eps,
        )
        self.hidden_size = hidden_size
        self.image_size = image_size

    def forward(self, op: OpBuilder, pixel_values: ir.Value) -> ir.Value:
        """Encode a batch of images.

        Args:
            pixel_values: ``[batch, 3, image_size, image_size]``.

        Returns:
            ``[batch, hidden_size, 16, 16]`` spatial features.
        """
        x = self.conv_stem(op, pixel_values)
        # The MSFA consumes the outputs of the last two stages.
        msfa_inputs: list[ir.Value] = []
        num_stages = len(self.blocks)
        for stage_idx, stage in enumerate(self.blocks):
            for block in stage:
                x = block(op, x)
            if stage_idx >= num_stages - 2:
                msfa_inputs.append(x)
        return self.msfa(op, msfa_inputs)


class _ConvStem(nn.Module):
    """Stride-2 3x3 stem conv (the one conv in the tower that has a bias)."""

    def __init__(
        self,
        out_chs: int,
        padding: int | tuple[int, int, int, int] = 0,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.conv = Conv2d(3, out_chs, kernel_size=3, stride=2, padding=padding)
        self.bn = RmsNorm2d(out_chs, eps=norm_eps)

    def forward(self, op: OpBuilder, x: ir.Value) -> ir.Value:
        return gelu_tanh(op, self.bn(op, self.conv(op, x)))


def _make_block(
    in_chs: int,
    spec: _BlockSpec,
    input_size: int,
    norm_eps: float,
) -> nn.Module:
    """Instantiate the block class matching ``spec``'s type."""
    if isinstance(spec, _EdgeResidualSpec):
        return _EdgeResidual(in_chs, spec, input_size=input_size, norm_eps=norm_eps)
    if isinstance(spec, _UIBSpec):
        return _UniversalInvertedBottleneck(
            in_chs, spec, input_size=input_size, norm_eps=norm_eps
        )
    if isinstance(spec, _MQASpec):
        return _MobileAttention(in_chs, spec, input_size=input_size, norm_eps=norm_eps)
    raise TypeError(f"unknown MobileNet-V5 block spec: {type(spec).__name__}")
