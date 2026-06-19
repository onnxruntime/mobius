# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Mimi neural audio codec (Kyutai), as used by ``nvidia/personaplex-7b-v1``.

This builds the Mimi codec used by the Moshi / PersonaPlex full-duplex
speech model as two ONNX graphs (encoder + decoder), wired through
:class:`mobius.tasks.CodecTask`:

* **encoder**: waveform ``(B, 1, samples)`` -> codes ``(B, 8, T)``
* **decoder**: codes ``(B, 8, T)`` -> waveform ``(B, 1, samples)``

Architecture (mirrors ``moshi.models.compression.MimiModel`` with the
``nvidia/personaplex-7b-v1`` tokenizer hyper-parameters):

* SEANet encoder/decoder: causal Conv1d / ConvTranspose1d stacks with ELU
  residual blocks, ``ratios=[8, 6, 5, 4]`` (24 kHz, 1920 samples/frame).
* A causal Transformer (LayerNorm + RoPE + LayerScale + GELU MLP, 8 layers,
  d=512) on each side, operating at the 25 Hz encoder frame rate.
* A learnt strided downsample (25 Hz -> 12.5 Hz) and a channel-wise
  transposed-conv upsample (12.5 Hz -> 25 Hz).
* A split residual vector quantizer (1 semantic + 7 acoustic codebooks of
  the 32 trained, i.e. 8 active codebooks), bins=2048, dim=256.

Causal-conv padding convention (matches ``moshi.modules.conv``):
left pad = ``effective_kernel - stride`` (``extra_padding`` is 0 for
frame-aligned input), ConvTranspose trims ``kernel - stride`` from the right.

Weights load from the native Kyutai ``tokenizer-*.safetensors`` checkpoint;
:meth:`MimiModel.preprocess_weights` precomputes codebook embeddings from
``embedding_sum / cluster_usage``, splits the fused attention ``in_proj``,
and permutes Q/K rows from the interleaved RoPE convention to the half-split
convention used by mobius's RoPE.
"""

from __future__ import annotations



import numpy as np
import onnx_ir as ir
import torch
from onnxscript import OpBuilder, nn

from mobius._configs import ArchitectureConfig
from mobius.components._codec_transformer import CodecEncoderTransformerModel
from mobius.components._codec_vq import SplitResidualVectorQuantizer

# ---------------------------------------------------------------------------
# PersonaPlex / Moshi Mimi hyper-parameters (fixed; config.json has none).
# ---------------------------------------------------------------------------

_DIMENSION = 512
_N_FILTERS = 64
_RATIOS = [8, 6, 5, 4]
_COMPRESS = 2

_TR_DIM = 512
_TR_LAYERS = 8
_TR_HEADS = 8
_TR_HEAD_DIM = 64
_TR_FFN = 2048
_TR_THETA = 10000.0
_TR_CONTEXT = 250

_CODEBOOK_DIM = 512   # quantizer in/out projection dimension
_VQ_DIM = 256         # internal codebook dimension
_BINS = 2048
_N_SEMANTIC = 1
_N_ACTIVE = 8         # active codebooks (1 semantic + 7 acoustic)

_RESAMPLE_STRIDE = 2  # 25 Hz <-> 12.5 Hz


# ---------------------------------------------------------------------------
# Low-level conv primitives (causal, Kyutai padding convention).
#
# Module nesting reproduces the Kyutai weight names exactly so that no
# renames are needed for conv weights:
#   StreamingConv1d(.conv) -> NormConv1d(.conv) -> Conv1d(.weight)
#   => "<prefix>.conv.conv.weight"
# ---------------------------------------------------------------------------


class _RawConv(nn.Module):
    """Innermost Conv1d parameter holder (Kyutai ``NormConv1d.conv``)."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel: int,
        *,
        stride: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self._kernel = kernel
        self._stride = stride
        self._groups = groups
        self.weight = nn.Parameter([out_ch, in_ch // groups, kernel])
        self.bias = nn.Parameter([out_ch]) if bias else None

    def forward(self, op: OpBuilder, x: ir.Value):
        args = (x, self.weight) if self.bias is None else (x, self.weight, self.bias)
        return op.Conv(
            *args,
            kernel_shape=[self._kernel],
            strides=[self._stride],
            dilations=[1],
            pads=[0, 0],
            group=self._groups,
        )


class _NormConv(nn.Module):
    """Kyutai ``NormConv1d`` wrapper (norm is folded away -> identity)."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.conv = _RawConv(*args, **kwargs)

    def forward(self, op: OpBuilder, x: ir.Value):
        return self.conv(op, x)


class _Conv(nn.Module):
    """Causal ``StreamingConv1d``: left-pad then convolve.

    Left pad = ``effective_kernel - stride`` (dilation == 1 here), which is
    the Kyutai causal convention. ``pad_mode`` is ``constant`` for SEANet
    convs and ``edge`` (replicate) for the learnt downsample.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel: int,
        *,
        stride: int = 1,
        groups: int = 1,
        bias: bool = True,
        pad_mode: str = "constant",
    ):
        super().__init__()
        self._pad_left = kernel - stride
        self._pad_mode = pad_mode
        self.conv = _NormConv(
            in_ch, out_ch, kernel, stride=stride, groups=groups, bias=bias
        )

    def forward(self, op: OpBuilder, x: ir.Value):
        if self._pad_left > 0:
            x = op.Pad(
                x,
                op.Constant(value_ints=[0, 0, self._pad_left, 0, 0, 0]),
                mode=self._pad_mode,
            )
        return self.conv(op, x)


class _RawConvTr(nn.Module):
    """Innermost ConvTranspose1d holder (Kyutai ``NormConvTranspose1d.convtr``)."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel: int,
        stride: int,
        *,
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self._kernel = kernel
        self._stride = stride
        self._groups = groups
        # ConvTranspose weight: (in_ch, out_ch // groups, kernel)
        self.weight = nn.Parameter([in_ch, out_ch // groups, kernel])
        self.bias = nn.Parameter([out_ch]) if bias else None

    def forward(self, op: OpBuilder, x: ir.Value):
        args = (x, self.weight) if self.bias is None else (x, self.weight, self.bias)
        return op.ConvTranspose(
            *args,
            kernel_shape=[self._kernel],
            strides=[self._stride],
            dilations=[1],
            pads=[0, 0],
            group=self._groups,
        )


class _NormConvTr(nn.Module):
    """Kyutai ``NormConvTranspose1d`` wrapper."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.convtr = _RawConvTr(*args, **kwargs)

    def forward(self, op: OpBuilder, x: ir.Value):
        return self.convtr(op, x)


class _ConvTr(nn.Module):
    """Causal ``StreamingConvTranspose1d``: convolve then trim right.

    Trims ``kernel - stride`` samples from the right (``trim_right_ratio`` is
    1.0 for the Mimi decoder), preserving causality.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel: int,
        stride: int,
        *,
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self._trim_right = kernel - stride
        self.convtr = _NormConvTr(
            in_ch, out_ch, kernel, stride, groups=groups, bias=bias
        )

    def forward(self, op: OpBuilder, x: ir.Value):
        y = self.convtr(op, x)
        if self._trim_right > 0:
            y = op.Slice(
                y,
                op.Constant(value_ints=[0]),
                op.Constant(value_ints=[-self._trim_right]),
                op.Constant(value_ints=[2]),  # time axis
            )
        return y


class _ELU(nn.Module):
    """ELU activation (alpha=1.0); paramless, kept in the module list to
    preserve Kyutai ``model.<i>`` indexing."""

    def forward(self, op: OpBuilder, x: ir.Value):
        return op.Elu(x, alpha=1.0)


class _ResnetBlock(nn.Module):
    """SEANet residual block: ELU -> Conv(k3) -> ELU -> Conv(k1), + skip.

    Kyutai names: ``block.1.conv.conv.*`` (k3), ``block.3.conv.conv.*`` (k1);
    ``block.0`` / ``block.2`` are ELU activations (paramless). ``true_skip``
    is True, so the shortcut is identity.
    """

    def __init__(self, dim: int):
        super().__init__()
        hidden = dim // _COMPRESS
        self.block = nn.ModuleList(
            [
                _ELU(),
                _Conv(dim, hidden, 3),
                _ELU(),
                _Conv(hidden, dim, 1),
            ]
        )

    def forward(self, op: OpBuilder, x: ir.Value):
        residual = x
        h = x
        for layer in self.block:
            h = layer(op, h)
        return op.Add(residual, h)


# ---------------------------------------------------------------------------
# SEANet encoder / decoder (flat ``model.<i>`` module lists).
# ---------------------------------------------------------------------------


class _SEANetEncoder(nn.Module):
    """SEANet encoder: ``(B, 1, samples)`` -> ``(B, 512, T)``."""

    def __init__(self):
        super().__init__()
        layers: list[nn.Module] = []
        mult = 1
        layers.append(_Conv(1, _N_FILTERS, 7))  # model.0
        for ratio in reversed(_RATIOS):  # [4, 5, 6, 8]
            ch = mult * _N_FILTERS
            layers.append(_ResnetBlock(ch))            # resnet
            layers.append(_ELU())                      # ELU
            layers.append(_Conv(ch, ch * 2, ratio * 2, stride=ratio))  # downsample
            mult *= 2
        layers.append(_ELU())
        layers.append(_Conv(mult * _N_FILTERS, _DIMENSION, 3))  # model.14
        self.model = nn.ModuleList(layers)

    def forward(self, op: OpBuilder, x: ir.Value):
        for layer in self.model:
            x = layer(op, x)
        return x


class _SEANetDecoder(nn.Module):
    """SEANet decoder: ``(B, 512, T)`` -> ``(B, 1, samples)``."""

    def __init__(self):
        super().__init__()
        layers: list[nn.Module] = []
        mult = 2 ** len(_RATIOS)  # 16
        layers.append(_Conv(_DIMENSION, mult * _N_FILTERS, 7))  # model.0
        for ratio in _RATIOS:  # [8, 6, 5, 4]
            ch = mult * _N_FILTERS
            layers.append(_ELU())
            layers.append(_ConvTr(ch, ch // 2, ratio * 2, ratio))  # upsample
            layers.append(_ResnetBlock(ch // 2))
            mult //= 2
        layers.append(_ELU())
        layers.append(_Conv(_N_FILTERS, 1, 3))  # model.14
        self.model = nn.ModuleList(layers)

    def forward(self, op: OpBuilder, x: ir.Value):
        for layer in self.model:
            x = layer(op, x)
        return x


# ---------------------------------------------------------------------------
# Down / up sample (extra ``.conv`` nesting level for name alignment).
# ---------------------------------------------------------------------------


class _Downsample(nn.Module):
    """Learnt strided downsample (25 Hz -> 12.5 Hz), replicate-padded.

    Name: ``downsample.conv.conv.conv.weight``.
    """

    def __init__(self):
        super().__init__()
        self.conv = _Conv(
            _DIMENSION,
            _DIMENSION,
            _RESAMPLE_STRIDE * 2,
            stride=_RESAMPLE_STRIDE,
            bias=False,
            pad_mode="edge",
        )

    def forward(self, op: OpBuilder, x: ir.Value):
        return self.conv(op, x)


class _Upsample(nn.Module):
    """Channel-wise (depthwise) transposed-conv upsample (12.5 Hz -> 25 Hz).

    Name: ``upsample.convtr.convtr.convtr.weight``. groups == dimension.
    """

    def __init__(self):
        super().__init__()
        self.convtr = _ConvTr(
            _DIMENSION,
            _DIMENSION,
            _RESAMPLE_STRIDE * 2,
            _RESAMPLE_STRIDE,
            groups=_DIMENSION,
            bias=False,
        )

    def forward(self, op: OpBuilder, x: ir.Value):
        return self.convtr(op, x)


# ---------------------------------------------------------------------------
# Transformer wrapper (adds the Kyutai ``.transformer`` name level).
# ---------------------------------------------------------------------------


class _Transformer(nn.Module):
    """Causal Mimi transformer; wraps :class:`CodecEncoderTransformerModel`.

    The ``.transformer`` attribute reproduces the Kyutai
    ``<enc|dec>_transformer.transformer.layers.<i>`` naming.
    """

    def __init__(self):
        super().__init__()
        self.transformer = CodecEncoderTransformerModel(
            hidden_size=_TR_DIM,
            num_hidden_layers=_TR_LAYERS,
            num_attention_heads=_TR_HEADS,
            num_key_value_heads=_TR_HEADS,
            intermediate_size=_TR_FFN,
            head_dim=_TR_HEAD_DIM,
            rope_theta=_TR_THETA,
            max_position_embeddings=4096,
            layer_norm_eps=1e-5,
        )

    def forward(self, op: OpBuilder, x: ir.Value):
        """Apply the transformer to ``(B, C, T)`` (channels-first) input.

        Internally transposes to ``(B, T, C)``, builds position ids and a
        causal mask, runs the layers, and transposes back.
        """
        # (B, C, T) -> (B, T, C)
        h = op.Transpose(x, perm=[0, 2, 1])
        seq_len = op.Squeeze(op.Shape(h, start=1, end=2), [0])  # scalar T
        position_ids = op.Unsqueeze(
            op.Range(op.Constant(value_int=0), seq_len, op.Constant(value_int=1)),
            [0],
        )  # (1, T)
        mask = _causal_mask(op, seq_len)
        h = self.transformer(op, h, position_ids, mask)
        # (B, T, C) -> (B, C, T)
        return op.Transpose(h, perm=[0, 2, 1])


def _causal_mask(op: OpBuilder, seq_len: ir.Value) -> ir.Value:
    """Build a ``(1, 1, T, T)`` boolean causal mask (True = attend).

    Lower-triangular (including the diagonal) is True; positions above the
    diagonal are masked out. Broadcasts across batch and heads in the ORT
    Attention op.
    """
    t = op.Reshape(seq_len, op.Constant(value_ints=[1]))  # (1,)
    shape = op.Concat(t, t, axis=0)  # (T, T)
    ones = op.ConstantOfShape(shape, value=ir.tensor(np.ones(1, dtype=np.float32)))
    tril = op.Trilu(ones, upper=0)  # lower-tri incl diagonal
    mask = op.Cast(tril, to=ir.DataType.BOOL)
    return op.Unsqueeze(mask, [0, 1])  # (1, 1, T, T)


# ---------------------------------------------------------------------------
# Encoder-side quantizer (argmin nearest-codebook search).
# ---------------------------------------------------------------------------


class _CodebookTable(nn.Module):
    """Holds a codebook embedding table as ``embedding.weight``.

    Shared name with the decoder's :class:`EuclideanCodebook` so a single
    preprocessed weight serves both the encoder and decoder graphs.
    """

    def __init__(self, codebook_size: int, dim: int):
        super().__init__()
        self.embedding = _EmbeddingWeight(codebook_size, dim)

    def forward(self, op: OpBuilder):
        return self.embedding(op)


class _EmbeddingWeight(nn.Module):
    """Tiny holder exposing a ``weight`` parameter (matches ``Embedding``)."""

    def __init__(self, codebook_size: int, dim: int):
        super().__init__()
        self.weight = nn.Parameter([codebook_size, dim])

    def forward(self, op: OpBuilder):
        return op.Identity(self.weight)


class _EncVQ(nn.Module):
    """Single encoder VQ layer: nearest codebook entry via argmin."""

    def __init__(self, codebook_size: int, dim: int):
        super().__init__()
        self._codebook = _CodebookTable(codebook_size, dim)

    def forward(self, op: OpBuilder, x: ir.Value):
        # x: (B, dim, T) -> (B, T, dim)
        x_t = op.Transpose(x, perm=[0, 2, 1])
        embedding = self._codebook(op)  # (codebook_size, dim)
        # torch.cdist uses the direct (non-matmul) Euclidean computation for
        # small inputs; replicate it via broadcast difference to match argmin
        # tie-breaking at deep residual layers exactly.
        x_b = op.Unsqueeze(x_t, [2])  # (B, T, 1, dim)
        e_b = op.Unsqueeze(embedding, [0, 1])  # (1, 1, codebook, dim)
        diff = op.Sub(x_b, e_b)  # (B, T, codebook, dim)
        distances = op.ReduceSumSquare(diff, [-1], keepdims=0)  # (B, T, codebook)
        codes = op.ArgMin(distances, axis=-1, keepdims=0)  # (B, T)
        quantized = op.Gather(embedding, codes, axis=0)     # (B, T, dim)
        quantized = op.Transpose(quantized, perm=[0, 2, 1])  # (B, dim, T)
        return codes, quantized


class _EncVQList(nn.Module):
    """``vq`` container: runs the residual argmin loop over ``layers.<i>``."""

    def __init__(self, num_quantizers: int, codebook_size: int, dim: int):
        super().__init__()
        self.layers = nn.ModuleList(
            [_EncVQ(codebook_size, dim) for _ in range(num_quantizers)]
        )

    def forward(self, op: OpBuilder, projected: ir.Value):
        """Returns ``(codes (B, K, T), total_quantized (B, dim, T))``."""
        all_codes = []
        residual = projected
        for layer in self.layers:
            codes_i, quantized_i = layer(op, residual)
            all_codes.append(op.Unsqueeze(codes_i, [1]))  # (B, 1, T)
            residual = op.Sub(residual, quantized_i)
        codes = op.Concat(*all_codes, axis=1)  # (B, K, T)
        total_quantized = op.Sub(projected, residual)  # (B, dim, T)
        return codes, total_quantized


class _ProjParams(nn.Module):
    """1x1 conv projection holder named ``<name>.weight`` (no bias)."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.weight = nn.Parameter([out_dim, in_dim, 1])

    def forward(self, op: OpBuilder, x: ir.Value):
        return op.Conv(x, self.weight, kernel_shape=[1], pads=[0, 0])


class _EncRVQ(nn.Module):
    """Encoder RVQ: input_proj -> residual argmin loop -> codes.

    Names align with Kyutai ``rvq_first`` / ``rvq_rest``:
    ``input_proj.weight``, ``output_proj.weight``,
    ``vq.layers.<i>._codebook.embedding.weight``.
    """

    def __init__(self, num_quantizers: int, codebook_size: int):
        super().__init__()
        self.input_proj = _ProjParams(_CODEBOOK_DIM, _VQ_DIM)
        self.output_proj = _ProjParams(_VQ_DIM, _CODEBOOK_DIM)
        self.vq = _EncVQList(num_quantizers, codebook_size, _VQ_DIM)

    def forward(self, op: OpBuilder, hidden: ir.Value):
        """Returns ``(codes (B, K, T), residual (B, 512, T))``."""
        projected = self.input_proj(op, hidden)  # (B, 256, T)
        codes, total_quantized = self.vq(op, projected)
        total_quantized = self.output_proj(op, total_quantized)
        output_residual = op.Sub(hidden, total_quantized)
        return codes, output_residual


class _EncSplitRVQ(nn.Module):
    """Split encoder RVQ: ``rvq_first`` (semantic) + ``rvq_rest`` (acoustic)."""

    def __init__(self):
        super().__init__()
        self.rvq_first = _EncRVQ(_N_SEMANTIC, _BINS)
        self.rvq_rest = _EncRVQ(_N_ACTIVE - _N_SEMANTIC, _BINS)

    def forward(self, op: OpBuilder, hidden: ir.Value):
        # rvq_first and rvq_rest both quantize the ORIGINAL input independently
        # (Kyutai SplitResidualVectorQuantizer.encode), not a shared residual.
        sem_codes, _ = self.rvq_first(op, hidden)
        acou_codes, _ = self.rvq_rest(op, hidden)
        return op.Concat(sem_codes, acou_codes, axis=1)  # (B, 8, T)


# ---------------------------------------------------------------------------
# Encoder / decoder model wrappers and the composite MimiModel.
# ---------------------------------------------------------------------------


class MimiEncoderModel(nn.Module):
    """Waveform -> codes. Mirrors ``MimiModel.encode``."""

    def __init__(self):
        super().__init__()
        self.encoder = _SEANetEncoder()
        self.encoder_transformer = _Transformer()
        self.downsample = _Downsample()
        self.quantizer = _EncSplitRVQ()

    def forward(self, op: OpBuilder, waveform: ir.Value):
        emb = self.encoder(op, waveform)            # (B, 512, T@25Hz)
        emb = self.encoder_transformer(op, emb)     # causal transformer
        emb = self.downsample(op, emb)              # (B, 512, T@12.5Hz)
        codes = self.quantizer(op, emb)             # (B, 8, T)
        return codes


class MimiDecoderModel(nn.Module):
    """Codes -> waveform. Mirrors ``MimiModel.decode``."""

    def __init__(self):
        super().__init__()
        self.quantizer = SplitResidualVectorQuantizer(
            num_quantizers=_N_ACTIVE,
            codebook_size=_BINS,
            codebook_dim=_CODEBOOK_DIM,
        )
        self.upsample = _Upsample()
        self.decoder_transformer = _Transformer()
        self.decoder = _SEANetDecoder()

    def forward(self, op: OpBuilder, codes: ir.Value):
        emb = self.quantizer(op, codes)             # (B, 512, T@12.5Hz)
        emb = self.upsample(op, emb)                # (B, 512, T@25Hz)
        emb = self.decoder_transformer(op, emb)     # causal transformer
        waveform = self.decoder(op, emb)            # (B, 1, samples)
        return waveform


class MimiModel(nn.Module):
    """Mimi neural audio codec (Kyutai) for ``nvidia/personaplex-7b-v1``.

    Produces two ONNX graphs via :class:`mobius.tasks.CodecTask`:
    ``encoder`` (waveform -> codes) and ``decoder`` (codes -> waveform).
    """

    default_task: str = "codec"
    category: str = "Audio"

    def __init__(self, config: ArchitectureConfig | None = None):
        super().__init__()
        self.config = config
        self.encoder = MimiEncoderModel()
        self.decoder = MimiDecoderModel()

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        return _preprocess_mimi_weights(state_dict)


def mimi_default_config() -> ArchitectureConfig:
    """Return the fixed :class:`ArchitectureConfig` for the Mimi codec.

    The Mimi checkpoint ships no ``config.json``; all dimensions are fixed by
    the architecture and reproduced as module-level constants. The config only
    needs to pass :meth:`ArchitectureConfig.validate` (positive dims) and carry
    the compute ``dtype`` — the model ignores most fields.
    """
    return ArchitectureConfig(
        model_type="mimi",
        hidden_size=_TR_DIM,
        num_hidden_layers=_TR_LAYERS,
        num_attention_heads=_TR_HEADS,
        num_key_value_heads=_TR_HEADS,
        head_dim=_TR_HEAD_DIM,
        intermediate_size=_TR_FFN,
        vocab_size=_BINS,
        max_position_embeddings=8000,
        dtype=ir.DataType.FLOAT,
    )


# ---------------------------------------------------------------------------
# Weight preprocessing: native Kyutai checkpoint -> mobius module names.
# ---------------------------------------------------------------------------


def _interleaved_to_halfsplit(w: torch.Tensor, head_dim: int) -> torch.Tensor:
    """Permute Q/K projection rows from interleaved-pair RoPE to half-split.

    Kyutai rotates adjacent ``(2i, 2i+1)`` pairs; mobius RoPE rotates
    ``(i, i + D/2)`` halves. Reordering rows so even indices precede odd
    indices (per head) makes the two conventions numerically identical.

    Args:
        w: ``(num_heads * head_dim, in_features)`` projection weight.
        head_dim: per-head dimension.
    """
    out_features, in_features = w.shape
    num_heads = out_features // head_dim
    w = w.reshape(num_heads, head_dim, in_features)
    half = head_dim // 2
    idx = torch.empty(head_dim, dtype=torch.long)
    idx[:half] = torch.arange(0, head_dim, 2)
    idx[half:] = torch.arange(1, head_dim, 2)
    w = w[:, idx, :]
    return w.reshape(out_features, in_features)


def _convert_transformer_layer(
    out: dict[str, torch.Tensor], prefix: str, lprefix: str, sd: dict
) -> None:
    """Convert one Kyutai transformer layer (``lprefix``) into mobius names
    under ``prefix`` (the fully-qualified ``<enc|dec>oder.<...>.layers.<i>``)."""
    # Norms
    out[f"{prefix}.input_layernorm.weight"] = sd[f"{lprefix}.norm1.weight"]
    out[f"{prefix}.input_layernorm.bias"] = sd[f"{lprefix}.norm1.bias"]
    out[f"{prefix}.post_attention_layernorm.weight"] = sd[f"{lprefix}.norm2.weight"]
    out[f"{prefix}.post_attention_layernorm.bias"] = sd[f"{lprefix}.norm2.bias"]
    # Fused QKV -> q/k/v; permute q,k for RoPE convention.
    in_proj = sd[f"{lprefix}.self_attn.in_proj_weight"]  # (3*D, D)
    dim = in_proj.shape[1]
    q, k, v = in_proj[:dim], in_proj[dim : 2 * dim], in_proj[2 * dim :]
    out[f"{prefix}.self_attn.q_proj.weight"] = _interleaved_to_halfsplit(q, _TR_HEAD_DIM)
    out[f"{prefix}.self_attn.k_proj.weight"] = _interleaved_to_halfsplit(k, _TR_HEAD_DIM)
    out[f"{prefix}.self_attn.v_proj.weight"] = v
    out[f"{prefix}.self_attn.o_proj.weight"] = sd[f"{lprefix}.self_attn.out_proj.weight"]
    # MLP (linear1/linear2 -> up_proj/down_proj)
    out[f"{prefix}.mlp.up_proj.weight"] = sd[f"{lprefix}.linear1.weight"]
    out[f"{prefix}.mlp.down_proj.weight"] = sd[f"{lprefix}.linear2.weight"]
    # LayerScale
    out[f"{prefix}.self_attn_layer_scale.scale"] = sd[f"{lprefix}.layer_scale_1.scale"]
    out[f"{prefix}.mlp_layer_scale.scale"] = sd[f"{lprefix}.layer_scale_2.scale"]


def _codebook_embedding(sd: dict, prefix: str) -> torch.Tensor:
    """Reconstruct a codebook embedding table from ``embedding_sum`` and
    ``cluster_usage`` (Kyutai stores running statistics, not the table)."""
    embedding_sum = sd[f"{prefix}.embedding_sum"].float()  # (bins, dim)
    cluster_usage = sd[f"{prefix}.cluster_usage"].float()  # (bins,)
    denom = cluster_usage.clamp(min=1e-8).unsqueeze(-1)
    return embedding_sum / denom


def _preprocess_mimi_weights(
    sd: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    """Map native Kyutai Mimi weights to mobius module parameter names.

    Names are fully qualified with the ``encoder.``/``decoder.`` component
    prefix that mobius derives from the :class:`MimiModel` attribute tree.
    Quantizer weights are shared by both graphs, so they are emitted under
    both ``encoder.quantizer.*`` and ``decoder.quantizer.*``.

    * conv / convtr weights pass through (inner names already align);
    * fused attention ``in_proj`` is split and Q/K rows are RoPE-permuted;
    * codebook tables are reconstructed from ``embedding_sum/cluster_usage``;
    * only the first :data:`_N_ACTIVE` codebooks are kept.
    """
    out: dict[str, torch.Tensor] = {}

    # SEANet conv stacks (double-nested: MimiModel.<side>.<seanet>.model.*).
    for key, value in sd.items():
        if key.startswith("encoder.model."):
            out[f"encoder.{key}"] = value
        elif key.startswith("decoder.model."):
            out[f"decoder.{key}"] = value
        elif key.startswith("downsample."):
            out[f"encoder.{key}"] = value
        elif key.startswith("upsample."):
            out[f"decoder.{key}"] = value

    # Transformers.
    for side, comp in (("encoder_transformer", "encoder"), ("decoder_transformer", "decoder")):
        for i in range(_TR_LAYERS):
            lprefix = f"{side}.transformer.layers.{i}"
            prefix = f"{comp}.{side}.transformer.layers.{i}"
            _convert_transformer_layer(out, prefix, lprefix, sd)

    # Quantizer projections (shared by encoder + decoder graphs).
    for rvq in ("rvq_first", "rvq_rest"):
        for proj in ("input_proj", "output_proj"):
            src = f"quantizer.{rvq}.{proj}.weight"
            if src in sd:
                for comp in ("encoder", "decoder"):
                    out[f"{comp}.quantizer.{rvq}.{proj}.weight"] = sd[src]

    # Quantizer codebooks (shared): rebuild tables, keep only active codebooks.
    def _emit_codebook(rvq: str, idx: int) -> None:
        src = f"quantizer.{rvq}.vq.layers.{idx}._codebook"
        table = _codebook_embedding(sd, src)
        for comp in ("encoder", "decoder"):
            dst = f"{comp}.quantizer.{rvq}.vq.layers.{idx}._codebook.embedding.weight"
            out[dst] = table

    _emit_codebook("rvq_first", 0)
    for i in range(_N_ACTIVE - _N_SEMANTIC):
        _emit_codebook("rvq_rest", i)

    return out
