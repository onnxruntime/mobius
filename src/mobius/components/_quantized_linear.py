# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Quantized linear and embedding layers (com.microsoft domain).

``QuantizedLinear`` uses ``MatMulNBits``; ``QuantizedEmbedding`` uses
``GatherBlockQuantized``.
"""

from __future__ import annotations

import math

import numpy as np
import onnx_ir as ir
from onnxscript import OpBuilder, nn

from mobius._build_context import ep_capabilities

# MatMulNBits packs weights into uint8 blobs.  The packed shape depends
# on bits and block_size:
#   packed_weights: [N, n_blocks, blob_size]  (uint8)
#     where n_blocks = ceil(K / block_size)
#           blob_size = block_size * bits / 8
#   scales:         [N, n_blocks]             (float16/float32)
#   zero_points:    [N, ceil(n_blocks*bits/8)] (uint8, optional, bit-packed)

_MICROSOFT_DOMAIN = "com.microsoft"


def _accuracy_level_attrs() -> dict[str, int]:
    """Return the ``accuracy_level`` attribute for ``MatMulNBits``, if any.

    ORT's MLAS CPU ``MatMulNBits`` kernel selects its compute path from the
    ``accuracy_level`` attribute: unset/0 keeps the highest-precision fp32
    dequant + fp32 GEMM path, while ``4`` dynamically quantizes activations to
    int8 and uses int8 dot-products (SDOT/NEON on ARM, AVX-VNNI on x86) — the
    same class of kernel llama.cpp uses, and typically 2-4x faster on CPU with
    no observable quality loss for Q4 weights. The value is sourced from the
    active EP's :attr:`EpCapabilities.default_int4_accuracy_level` (4 for CPU /
    WebGPU). When it is 0 the attribute is omitted so ORT keeps its default.
    """
    level = ep_capabilities().default_int4_accuracy_level
    return {"accuracy_level": level} if level else {}


class QuantizedLinear(nn.Module):
    """Linear layer backed by the MatMulNBits custom op.

    Replaces a standard ``Linear`` for weight-only quantized models
    (GPTQ, AWQ, etc.).  The op performs::

        y = x @ dequantize(packed_weights, scales, zero_points)^T

    entirely inside a single fused kernel at inference time.

    Args:
        in_features: Input dimension (K).
        out_features: Output dimension (N).
        bits: Quantization bit-width (2, 4, or 8).
        block_size: Number of elements per quantization group.
        has_zero_point: Whether asymmetric zero-point is used.
        zero_point_dtype: Dtype for the zero_points parameter when
            ``has_zero_point`` is true. ``UINT8`` (default) uses ORT's
            bit-packed integer zero-points (same bit width as the
            weights). Float dtypes (``FLOAT``/``FLOAT16``/``BFLOAT16``)
            produce one un-packed float per block — required for
            codebooks whose offset is not an integer (e.g. Tencent SEQ
            uses ``1.5``).
        bias: Whether to include a bias term.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bits: int = 4,
        block_size: int = 32,
        has_zero_point: bool = False,
        zero_point_dtype: ir.DataType = ir.DataType.UINT8,
        bias: bool = False,
    ):
        super().__init__()
        if bits not in (2, 4, 8):
            raise ValueError(f"bits must be 2, 4, or 8, got {bits}")
        if block_size < 16 or (block_size & (block_size - 1)):
            raise ValueError(f"block_size must be a power of 2 >= 16, got {block_size}")

        self._bits = bits
        self._block_size = block_size
        self._k = in_features
        self._n = out_features

        n_blocks = math.ceil(in_features / block_size)
        blob_size = block_size * bits // 8

        # Packed quantized weight tensor (uint8)
        self.weight = nn.Parameter(
            [out_features, n_blocks, blob_size],
            dtype=ir.DataType.UINT8,
        )
        # Per-block scale factors
        self.scales = nn.Parameter([out_features, n_blocks])
        # Optional per-block zero points (asymmetric quantization).
        # UINT8 zero_points use the same bit-packing as the weights, so
        # the packed last dimension is ceil(n_blocks * bits / 8):
        #   bits=2 → 4 zero-points per byte
        #   bits=4 → 2 zero-points per byte
        #   bits=8 → 1 zero-point per byte (no packing)
        # Float zero_points are one value per block, dtype as specified.
        if has_zero_point:
            if zero_point_dtype == ir.DataType.UINT8:
                zp_dim = math.ceil(n_blocks * bits / 8)
                self.zero_points = nn.Parameter(
                    [out_features, zp_dim],
                    dtype=ir.DataType.UINT8,
                )
            else:
                self.zero_points = nn.Parameter(
                    [out_features, n_blocks],
                    dtype=zero_point_dtype,
                )
        else:
            self.zero_points = None
        self.bias = nn.Parameter([out_features]) if bias else None

    def forward(self, op: OpBuilder, x: ir.Value) -> ir.Value:
        """Compute quantized matmul: y = x @ dequant(W).

        Args:
            op: ONNX op builder.
            x: Input tensor of shape ``[*, K]``.

        Returns:
            Output tensor of shape ``[*, N]``.
        """
        inputs: list[ir.Value | None] = [x, self.weight, self.scales]
        if self.zero_points is not None:
            inputs.append(self.zero_points)

        result = op.MatMulNBits(
            *inputs,
            K=self._k,
            N=self._n,
            bits=self._bits,
            block_size=self._block_size,
            **_accuracy_level_attrs(),
            _domain=_MICROSOFT_DOMAIN,
        )
        if self.bias is not None:
            result = op.Add(result, self.bias)
        return result


class QuantizedEmbedding(nn.Module):
    """Embedding backed by the GatherBlockQuantized custom op.

    Looks up rows of a block-wise quantized embedding table and dequantizes
    the gathered rows inside a single fused kernel.  Replaces a standard
    :class:`~mobius.components.Embedding` for weight-only quantized models
    whose embedding table is quantized (e.g. Olive RTN exports with
    ``embeds: true``).

    Uses the uint8-packed layout shared with Olive/ORT exports
    (``gather_axis=0``)::

        qweight:     [num_embeddings, embedding_dim * bits // 8]  uint8
        scales:      [num_embeddings, embedding_dim // block_size]
        zero_points: [num_embeddings, ceil(n_blocks * bits / 8)]  uint8 (optional)

    The op dequantizes each gathered row block-wise as
    ``(q - zero_point) * scale``.  The output dtype matches ``scales``
    (the model compute dtype), so the gathered embeddings drop straight
    into the decoder.

    Args:
        num_embeddings: Vocabulary size (gather axis).
        embedding_dim: Embedding dimension (quantized axis).
        bits: Quantization bit-width (2, 4, or 8).
        block_size: Number of elements per quantization group.
        has_zero_point: Whether asymmetric zero-points are used.
        padding_idx: Optional padding index (stored for parity; the Gather
            does not special-case it, matching ``Embedding``).
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bits: int = 4,
        block_size: int = 32,
        has_zero_point: bool = True,
        padding_idx: int | None = None,
    ):
        super().__init__()
        if bits not in (2, 4, 8):
            raise ValueError(f"bits must be 2, 4, or 8, got {bits}")
        if block_size < 16 or (block_size & (block_size - 1)):
            raise ValueError(f"block_size must be a power of 2 >= 16, got {block_size}")
        if embedding_dim % block_size != 0:
            raise ValueError(
                f"embedding_dim ({embedding_dim}) must be divisible by "
                f"block_size ({block_size})"
            )

        self._bits = bits
        self._block_size = block_size
        self.padding_idx = padding_idx

        n_blocks = embedding_dim // block_size
        packed = embedding_dim * bits // 8

        # Packed quantized embedding table (uint8); rows gathered by token id.
        self.qweight = nn.Parameter([num_embeddings, packed], dtype=ir.DataType.UINT8)
        # Per-block scales (FLOAT here; cast to the model dtype before build).
        self.scales = nn.Parameter([num_embeddings, n_blocks])
        # Optional bit-packed uint8 zero-points (same packing as the weights).
        if has_zero_point:
            self.zero_points = nn.Parameter(
                [num_embeddings, math.ceil(n_blocks * bits / 8)],
                dtype=ir.DataType.UINT8,
            )
        else:
            self.zero_points = None

    def forward(self, op: OpBuilder, input_ids: ir.Value) -> ir.Value:
        """Gather and dequantize embedding rows for ``input_ids``.

        Args:
            op: ONNX op builder.
            input_ids: Integer indices of any shape ``[*]``.

        Returns:
            Dequantized embeddings of shape ``[*, embedding_dim]`` with the
            same dtype as ``scales``.
        """
        inputs: list[ir.Value | None] = [self.qweight, input_ids, self.scales]
        if self.zero_points is not None:
            inputs.append(self.zero_points)

        return op.GatherBlockQuantized(
            *inputs,
            bits=self._bits,
            block_size=self._block_size,
            gather_axis=0,
            quantize_axis=1,
            _domain=_MICROSOFT_DOMAIN,
        )


class TiedQuantizedLMHead(nn.Module):
    """LM head tied to a :class:`QuantizedEmbedding`, sharing one packed table.

    When the input embedding and the LM head are tied **and** both are
    block-wise quantized, the two ops want different layouts of the *same*
    bytes: the embedding uses ``GatherBlockQuantized`` with a **2-D**
    ``[vocab, dim*bits//8]`` table, while the head uses ``MatMulNBits`` with a
    **3-D** ``[vocab, n_blocks, blob_size]`` weight.

    Rather than store a second copy of the table (~the embedding size again),
    this head **shares the embedding's** ``qweight``/``scales``/``zero_points``
    Parameters — yielding a single ONNX initializer for each — and reshapes the
    shared 2-D packed table to the MatMulNBits layout at graph-build time.  The
    ``Reshape`` is a no-op view over identical bytes and is left in the graph
    (mobius does not constant-fold inputs this large); ONNX Runtime folds it at
    session creation.  Net effect: one copy of the tied table on disk.

    Args:
        embedding: The quantized input embedding to tie to.
        hidden_size: Input feature size (MatMulNBits ``K``).
        vocab_size: Output size / vocabulary (MatMulNBits ``N``).
    """

    def __init__(
        self,
        embedding: QuantizedEmbedding,
        hidden_size: int,
        vocab_size: int,
    ):
        super().__init__()
        bits = embedding._bits
        block_size = embedding._block_size
        if hidden_size % block_size != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by block_size ({block_size})"
            )

        self._bits = bits
        self._block_size = block_size
        self._k = hidden_size
        self._n = vocab_size

        n_blocks = hidden_size // block_size
        blob_size = block_size * bits // 8
        # MatMulNBits weight layout for the shared packed table.
        self._weight_shape = np.array([vocab_size, n_blocks, blob_size], dtype=np.int64)

        # Share the embedding's Parameters (same objects -> one initializer
        # each). The embedding's 2-D qweight is reshaped in forward().
        self.qweight = embedding.qweight
        self.scales = embedding.scales
        self.zero_points = embedding.zero_points

    def forward(self, op: OpBuilder, x: ir.Value) -> ir.Value:
        """Project hidden states to logits with the shared quantized table.

        Args:
            op: ONNX op builder.
            x: Hidden states of shape ``[*, hidden_size]``.

        Returns:
            Logits of shape ``[*, vocab_size]``.
        """
        # Reshape the shared 2-D packed table to the 3-D MatMulNBits layout.
        weight = op.Reshape(self.qweight, op.Constant(value=ir.tensor(self._weight_shape)))
        inputs: list[ir.Value | None] = [x, weight, self.scales]
        if self.zero_points is not None:
            inputs.append(self.zero_points)

        return op.MatMulNBits(
            *inputs,
            K=self._k,
            N=self._n,
            bits=self._bits,
            block_size=self._block_size,
            **_accuracy_level_attrs(),
            _domain=_MICROSOFT_DOMAIN,
        )


def make_quantized_linear_factory(
    bits: int = 4,
    block_size: int = 32,
    has_zero_point: bool = False,
    zero_point_dtype: ir.DataType = ir.DataType.UINT8,
) -> type:
    """Create a QuantizedLinear factory compatible with the linear_class pattern.

    Returns a class whose ``__init__(in_features, out_features, bias=True)``
    signature matches ``Linear`` so it can be injected via ``linear_class``
    in ``DecoderLayer``, ``Attention``, and ``MLP``.

    Args:
        bits: Quantization bit-width (typically 4).
        block_size: Number of elements per quantization group.
        has_zero_point: Whether to include zero-point parameters.
        zero_point_dtype: Dtype for the zero_points parameter (see
            :class:`QuantizedLinear`).

    Returns:
        A class that constructs QuantizedLinear instances.
    """

    class _Factory(QuantizedLinear):
        def __init__(
            self,
            in_features: int,
            out_features: int,
            bias: bool = True,
        ):
            super().__init__(
                in_features=in_features,
                out_features=out_features,
                bias=bias,
                bits=bits,
                block_size=block_size,
                has_zero_point=has_zero_point,
                zero_point_dtype=zero_point_dtype,
            )

    _Factory.__name__ = "QuantizedLinear"
    _Factory.__qualname__ = "QuantizedLinear"
    return _Factory
