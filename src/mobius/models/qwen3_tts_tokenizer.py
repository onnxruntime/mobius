# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Qwen3-TTS Codec Tokenizer 12Hz — ONNX model classes.

Implements the ``Qwen3TTSTokenizerV2Model`` architecture as a 2-model
ONNX package (decoder + encoder).

HuggingFace model: ``Qwen/Qwen3-TTS-Tokenizer-12Hz``
HuggingFace class: ``Qwen3TTSTokenizerV2Model`` in ``qwen_tts``

Decoder (codes -> waveform):
    VQ decode -> causal conv -> transformer -> upsample -> decoder blocks -> clamp

Encoder (waveform -> codes):
    Conv encoder -> transformer -> downsample -> quantize (argmin)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from onnxscript import OpBuilder, nn

from mobius.components._codec_conv import (
    CausalConv1d,
    CausalTransConv1d,
    ConvNeXtBlock,
    DecoderBlock,
    SnakeBeta,
    _Conv1dParams,
)
from mobius.components._codec_transformer import (
    CodecDecoderTransformerModel,
    CodecEncoderTransformerModel,
)
from mobius.components._codec_vq import (
    SplitResidualVectorQuantizer,
    _Conv1dProjParams,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    import onnx_ir as ir
    import torch

    from mobius._configs import ArchitectureConfig


# ---------------------------------------------------------------------------
# Codec Decoder: codes -> waveform
# ---------------------------------------------------------------------------


class Qwen3TTSCodecDecoderModel(nn.Module):
    """Qwen3-TTS codec decoder: 16-group audio codes -> waveform.

    Architecture:
        1. VQ decode: SplitRVQ codebook lookup -> (B, codebook_dim, T)
        2. Pre-conv: CausalConv1d -> (B, latent_dim, T)
        3. Transformer: input_proj -> 8 layers -> norm -> output_proj
        4. Upsample: Nx(CausalTransConv + ConvNeXt)
        5. Decoder: InitConv -> NxDecoderBlock -> SnakeBeta -> FinalConv
        6. Clamp: [-1, 1]

    HF class: ``Qwen3TTSTokenizerV2Decoder``.

    Parameters:
        config: Architecture config with codec_decoder sub-config.
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        dec = config.codec_decoder
        codebook_dim = dec.codebook_dim if dec else 512
        latent_dim = dec.latent_dim if dec else 1024
        hidden_size = dec.hidden_size if dec else 512
        num_layers = dec.num_hidden_layers if dec else 8
        num_heads = dec.num_attention_heads if dec else 16
        num_kv_heads = dec.num_key_value_heads if dec else 16
        head_dim = dec.head_dim if dec else 64
        intermediate = dec.intermediate_size if dec else 1024
        rms_eps = dec.rms_norm_eps if dec else 1e-5
        decoder_dim = dec.decoder_dim if dec else 1536
        upsample_rates = dec.upsample_rates if dec else (8, 5, 4, 3)
        upsampling_ratios = dec.upsampling_ratios if dec else (2, 2)
        num_quantizers = dec.num_quantizers if dec else 16
        codebook_size = dec.codebook_size if dec else 2048

        # 1. VQ decode
        self.quantizer = SplitResidualVectorQuantizer(
            num_quantizers=num_quantizers,
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
        )

        # 2. Pre-conv: codebook_dim -> latent_dim
        self.pre_conv = CausalConv1d(codebook_dim, latent_dim, kernel_size=3)

        # 3. Transformer
        self.pre_transformer = CodecDecoderTransformerModel(
            latent_dim=latent_dim,
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            num_key_value_heads=num_kv_heads,
            intermediate_size=intermediate,
            head_dim=head_dim,
            rms_norm_eps=rms_eps,
        )

        # 4. Upsample blocks — nn.Sequential of [TransConv, ConvNeXt]
        # so names become upsample.i.0 / upsample.i.1 matching HF
        self.upsample = nn.ModuleList(
            [
                nn.Sequential(
                    CausalTransConv1d(
                        latent_dim,
                        latent_dim,
                        kernel_size=ratio,
                        stride=ratio,
                    ),
                    ConvNeXtBlock(latent_dim),
                )
                for ratio in upsampling_ratios
            ]
        )

        # 5. Decoder blocks: progressive channel reduction + upsampling
        # CausalConv1d placed directly in list (no wrapper) so names
        # are decoder.0.conv.weight matching HF decoder.0.conv.weight
        self._num_decoder_blocks = len(upsample_rates)
        decoder_modules: list[nn.Module] = [
            CausalConv1d(latent_dim, decoder_dim, kernel_size=7),
        ]
        for i, rate in enumerate(upsample_rates):
            in_dim = decoder_dim // (2**i)
            out_dim = decoder_dim // (2 ** (i + 1))
            decoder_modules.append(DecoderBlock(in_dim, out_dim, rate))
        # Final: SnakeBeta -> Conv1d(out_dim -> 1, k=7)
        final_dim = decoder_dim // (2 ** len(upsample_rates))
        decoder_modules.append(SnakeBeta(final_dim))
        decoder_modules.append(
            CausalConv1d(final_dim, 1, kernel_size=7),
        )
        self.decoder = nn.Sequential(*decoder_modules)

    def forward(self, op: OpBuilder, codes: ir.Value):
        """Decode audio codes to waveform.

        Args:
            codes: (B, num_quantizers, T) int64 audio codes.

        Returns:
            waveform: (B, 1, T * total_upsample) float32, clamped to [-1, 1].
        """
        # 1. VQ decode: (B, 16, T) -> (B, codebook_dim, T)
        hidden = self.quantizer(op, codes)

        # 2. Pre-conv: (B, codebook_dim, T) -> (B, latent_dim, T)
        hidden = self.pre_conv(op, hidden)

        # 3. Transformer: channels-last -> transform -> channels-first
        # (B, latent_dim, T) -> (B, T, latent_dim)
        hidden = op.Transpose(hidden, perm=[0, 2, 1])
        # Generate position IDs: 0..T-1
        seq_len = op.Shape(hidden, start=1, end=2)
        position_ids = op.Unsqueeze(
            op.Range(
                op.Constant(value_int=0),
                op.Squeeze(seq_len, [0]),
                op.Constant(value_int=1),
            ),
            [0],
        )
        hidden = self.pre_transformer(op, hidden, position_ids)
        # (B, T, latent_dim) -> (B, latent_dim, T)
        hidden = op.Transpose(hidden, perm=[0, 2, 1])

        # 4. Upsample blocks (Sequential handles iteration)
        for block in self.upsample:
            hidden = block(op, hidden)

        # 5. Decoder blocks (Sequential handles iteration)
        hidden = self.decoder(op, hidden)

        # 6. Clamp to [-1, 1]
        hidden = op.Clip(
            hidden,
            -1.0,
            1.0,
        )

        return hidden


# ---------------------------------------------------------------------------
# Codec Encoder: waveform -> codes
# ---------------------------------------------------------------------------


class Qwen3TTSCodecEncoderModel(nn.Module):
    """Qwen3-TTS codec encoder: waveform -> 16-group audio codes.

    Architecture (based on MimiModel encoder):
        1. Conv encoder: progressive downsampling + channel increase
        2. Transformer: LayerNorm + Attention + GELU MLP + LayerScale
        3. Downsample: strided conv
        4. Quantize: find nearest codebook entries (argmin)

    HF class: ``Qwen3TTSTokenizerV2Encoder`` (extends ``MimiModel``).

    Parameters:
        config: Architecture config with codec_encoder sub-config.
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        enc = config.codec_encoder
        hidden_size = enc.hidden_size if enc else 512
        num_layers = enc.num_hidden_layers if enc else 8
        num_heads = enc.num_attention_heads if enc else 8
        num_kv_heads = enc.num_key_value_heads if enc else 8
        head_dim = enc.head_dim if enc else 64
        intermediate = enc.intermediate_size if enc else 2048

        # Conv encoder: series of Conv1d + residual blocks, derived from
        # config exactly like HF ``MimiEncoder.__init__``. With the
        # checkpoint's defaults (num_filters=64, upsampling_ratios=[8,6,5,4],
        # num_residual_layers=1, kernel_size=7, last_kernel_size=3) this
        # yields 1->64->128->256->512->1024->hidden_size.
        self.encoder = _MimiConvEncoder(
            hidden_size=hidden_size,
            audio_channels=enc.audio_channels if enc else 1,
            num_filters=enc.num_filters if enc else 64,
            num_residual_layers=enc.num_residual_layers if enc else 1,
            kernel_size=enc.kernel_size if enc else 7,
            last_kernel_size=enc.last_kernel_size if enc else 3,
            residual_kernel_size=enc.residual_kernel_size if enc else 3,
            compress=enc.compress if enc else 2,
            upsampling_ratios=tuple(enc.upsampling_ratios) if enc else (8, 6, 5, 4),
        )

        # Transformer
        self.encoder_transformer = CodecEncoderTransformerModel(
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            num_key_value_heads=num_kv_heads,
            intermediate_size=intermediate,
            head_dim=head_dim,
        )

        # Downsample: Conv1d(hidden_size->hidden_size, k=4, stride=2)
        self.downsample = _DownsampleConv(hidden_size, hidden_size, 4, 2)

        # Quantizer (encoder-side): uses argmin for encoding
        # We reuse the same SplitRVQ structure but add encode logic
        self.quantizer = _EncoderSplitRVQ(config)

    def forward(self, op: OpBuilder, waveform: ir.Value):
        """Encode waveform to audio codes.

        Args:
            waveform: (B, 1, audio_samples) float32.

        Returns:
            codes: (B, 16, T) int64 audio codes.
        """
        # 1. Conv encoder: (B, 1, samples) -> (B, hidden_size, T')
        hidden = self.encoder(op, waveform)

        # 2. Transformer: channels-last
        # (B, hidden_size, T') -> (B, T', hidden_size)
        hidden = op.Transpose(hidden, perm=[0, 2, 1])
        seq_len = op.Shape(hidden, start=1, end=2)
        position_ids = op.Unsqueeze(
            op.Range(
                op.Constant(value_int=0),
                op.Squeeze(seq_len, [0]),
                op.Constant(value_int=1),
            ),
            [0],
        )
        hidden = self.encoder_transformer(op, hidden, position_ids)
        # (B, T', hidden_size) -> (B, hidden_size, T')
        hidden = op.Transpose(hidden, perm=[0, 2, 1])

        # 3. Downsample: (B, hidden_size, T') -> (B, hidden_size, T'/2)
        hidden = self.downsample(op, hidden)

        # 4. Quantize: (B, hidden_size, T) -> (B, 16, T)
        codes = self.quantizer(op, hidden)

        return codes


class _MimiConvEncoder(nn.Module):
    """Mimi-style convolutional encoder, derived from config.

    Mirrors HF ``MimiEncoder.__init__``: one leading conv, then for each
    upsampling ratio (in reverse) ``num_residual_layers`` residual blocks,
    an ELU and a strided conv that doubles the channel count, then a
    trailing ELU and a conv down to ``hidden_size``.

    The parameterless ELU modules occupy list slots so that the resulting
    parameter names line up with the checkpoint's
    ``encoder.encoder.layers.*`` numbering. With the checkpoint defaults
    (``num_filters=64``, ``upsampling_ratios=[8, 6, 5, 4]``,
    ``num_residual_layers=1``, ``kernel_size=7``, ``last_kernel_size=3``)
    the stack is 1->64->128->256->512->1024->512 and numbers as:

        0: Conv1d(1->64, k=7)
        1: ResBlock(64): block.1 Conv1d(64->32,k=3), block.3 Conv1d(32->64,k=1)
        3: Conv1d(64->128, k=8, stride=4)
        4: ResBlock(128)
        6: Conv1d(128->256, k=10, stride=5)
        7: ResBlock(256)
        9: Conv1d(256->512, k=12, stride=6)
       10: ResBlock(512)
       12: Conv1d(512->1024, k=16, stride=8)
       14: Conv1d(1024->512, k=3)

    Parameters:
        hidden_size: Output channels of the final conv.
        audio_channels: Input channels of the waveform (1 for mono).
        num_filters: Channel count after the leading conv.
        num_residual_layers: Residual blocks per downsampling stage.
        kernel_size: Kernel of the leading conv.
        last_kernel_size: Kernel of the final conv.
        residual_kernel_size: Kernel of the first conv inside a residual block.
        compress: Channel-reduction factor inside a residual block.
        upsampling_ratios: Downsampling strides, applied in reverse order.
    """

    def __init__(
        self,
        hidden_size: int = 512,
        audio_channels: int = 1,
        num_filters: int = 64,
        num_residual_layers: int = 1,
        kernel_size: int = 7,
        last_kernel_size: int = 3,
        residual_kernel_size: int = 3,
        compress: int = 2,
        upsampling_ratios: Sequence[int] = (8, 6, 5, 4),
    ):
        super().__init__()
        layers: list[nn.Module] = [
            _EncoderConvLayer(audio_channels, num_filters, kernel_size, 1),
        ]
        scaling = 1
        # Encoder downsamples in reverse of the decoder's upsampling order:
        # ratios [8, 6, 5, 4] -> strides 4, 5, 6, 8.
        for ratio in reversed(upsampling_ratios):
            current_scale = scaling * num_filters
            for _ in range(num_residual_layers):
                layers.append(
                    _EncoderResBlock(
                        current_scale,
                        kernel=residual_kernel_size,
                        compress=compress,
                    )
                )
            layers.append(_ELUModule())
            # Strided conv doubles channels; kernel is 2x the stride.
            layers.append(
                _EncoderConvLayer(current_scale, current_scale * 2, ratio * 2, ratio)
            )
            scaling *= 2
        layers.append(_ELUModule())
        layers.append(
            _EncoderConvLayer(scaling * num_filters, hidden_size, last_kernel_size, 1)
        )
        self.layers = nn.Sequential(*layers)
        self.output_channels = hidden_size

    def forward(self, op: OpBuilder, x: ir.Value):
        """Encode waveform through conv layers.

        Args:
            x: (B, audio_channels, audio_samples).

        Returns:
            (B, hidden_size, T).
        """
        return self.layers(op, x)


class _ELUModule(nn.Module):
    """ELU activation layer.

    The Mimi encoder places ELU activations between ResBlocks and
    strided downsampling convolutions. These are essential nonlinearities
    that must be computed (not skipped).
    """

    def forward(self, op: OpBuilder, x: ir.Value):
        return op.Elu(x, alpha=1.0)


class _EncoderConvLayer(nn.Module):
    """Single causal 1-D convolution with left-padding.

    Pads the input on the left by ``kernel - 1`` to preserve causality,
    then applies the convolution (optionally strided for downsampling).

    Parameters:
        in_ch: Input channels.
        out_ch: Output channels.
        kernel: Convolution kernel size.
        stride: Convolution stride (>1 for downsampling).
    """

    def __init__(self, in_ch: int, out_ch: int, kernel: int, stride: int):
        super().__init__()
        self._kernel = kernel
        self.conv = _Conv1dParams(
            in_ch,
            out_ch,
            kernel,
            bias=True,
            stride=stride,
        )

    def forward(self, op: OpBuilder, x: ir.Value):
        # Causal padding: left-pad by (kernel - 1)
        pad_left = self._kernel - 1
        if pad_left > 0:
            x = op.Pad(
                x,
                op.Constant(value_ints=[0, 0, pad_left, 0, 0, 0]),
                mode="constant",
            )
        return self.conv(op, x)


class _EncoderResBlock(nn.Module):
    """Encoder residual block: ELU -> Conv(k=3) -> ELU -> Conv(k=1) + skip.

    HF weight structure: block.1.conv (dilated), block.3.conv (pointwise).
    block.0 and block.2 are ELU activations.

    Parameters:
        dim: Input/output channels.
        kernel: Kernel of the first (channel-reducing) conv.
        compress: Channel-reduction factor for the block's inner width.
    """

    def __init__(self, dim: int, kernel: int = 3, compress: int = 2):
        super().__init__()
        half = dim // compress
        self.block = nn.Sequential(
            _ELUModule(),  # 0: ELU
            _EncoderConvLayer(dim, half, kernel, 1),  # 1: dilated conv
            _ELUModule(),  # 2: ELU
            _EncoderConvLayer(half, dim, 1, 1),  # 3: pointwise conv
        )

    def forward(self, op: OpBuilder, x: ir.Value):
        residual = x
        return op.Add(residual, self.block(op, x))


class _DownsampleConv(nn.Module):
    """Causal strided convolution for downsampling in the encoder VQ path.

    Same as :class:`_EncoderConvLayer` but without bias, used inside
    :class:`_EncoderSplitRVQ` for the semantic/acoustic RVQ sub-encoders.
    """

    def __init__(self, in_ch: int, out_ch: int, kernel: int, stride: int):
        super().__init__()
        self._kernel = kernel
        self.conv = _Conv1dParams(
            in_ch,
            out_ch,
            kernel,
            bias=False,
            stride=stride,
        )

    def forward(self, op: OpBuilder, x: ir.Value):
        # Causal left-pad: (B, C, T) -> (B, C, T + kernel-1)
        pad_left = self._kernel - 1
        if pad_left > 0:
            x = op.Pad(
                x,
                op.Constant(value_ints=[0, 0, pad_left, 0, 0, 0]),
                mode="constant",
            )
        return self.conv(op, x)


class _EncoderSplitRVQ(nn.Module):
    """Encoder-side split RVQ: encodes continuous features to discrete codes.

    Uses argmin to find the nearest codebook entry for each frame.
    Split into semantic (1 quantizer) + acoustic (N-1 quantizers).

    The encode path is: input_proj -> for each layer: find nearest -> subtract.
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        enc = config.codec_encoder
        codebook_dim = enc.codebook_dim if enc else 256
        codebook_size = enc.codebook_size if enc else 2048
        num_quantizers = enc.num_quantizers if enc else 32
        num_semantic = enc.num_semantic_quantizers if enc else 1
        hidden_size = enc.hidden_size if enc else 512

        # The RVQ consumes the encoder's hidden features and projects them
        # down to the codebook dimension, matching HF ``MimiResidualVector
        # Quantizer``: input_proj is Conv1d(hidden_size -> codebook_dim, k=1).
        dim = codebook_dim

        # Semantic quantizer
        self.semantic_residual_vector_quantizer = _EncoderRVQ(
            num_quantizers=num_semantic,
            codebook_size=codebook_size,
            dim=dim,
            input_dim=hidden_size,
            output_dim=hidden_size,
        )
        # Acoustic quantizer
        self.acoustic_residual_vector_quantizer = _EncoderRVQ(
            num_quantizers=num_quantizers - num_semantic,
            codebook_size=codebook_size,
            dim=dim,
            input_dim=hidden_size,
            output_dim=hidden_size,
        )
        self._num_semantic = num_semantic
        self._num_valid = config.num_quantizers if hasattr(config, "num_quantizers") else 16

    def forward(self, op: OpBuilder, hidden: ir.Value):
        """Encode features to discrete codes.

        Args:
            hidden: (B, hidden_size, T) continuous features.

        Returns:
            codes: (B, num_valid_quantizers, T) int64.
        """
        # Semantic codes
        sem_codes, sem_residual = self.semantic_residual_vector_quantizer(op, hidden)
        # Acoustic codes (from residual)
        acou_codes, _ = self.acoustic_residual_vector_quantizer(op, sem_residual)
        # Concatenate: (B, 1, T) + (B, N-1, T) -> (B, N, T)
        codes = op.Concat(sem_codes, acou_codes, axis=1)
        # Only return first num_valid quantizers
        codes = op.Slice(
            codes,
            op.Constant(value_ints=[0]),
            op.Constant(value_ints=[self._num_valid]),
            op.Constant(value_ints=[1]),
        )
        return codes


class _EncoderRVQ(nn.Module):
    """Encoder-side RVQ with input/output projections.

    For each quantizer layer: project -> find nearest (argmin) -> subtract.
    """

    def __init__(
        self,
        num_quantizers: int,
        codebook_size: int,
        dim: int,
        input_dim: int,
        output_dim: int,
    ):
        super().__init__()
        # Use _Conv1dProjParams to get .weight suffix matching HF
        self.input_proj = _Conv1dProjParams(input_dim, dim)
        self.output_proj = _Conv1dProjParams(dim, output_dim)
        self.layers = nn.ModuleList(
            [_EncoderVQ(codebook_size, dim) for _ in range(num_quantizers)]
        )

    def forward(self, op: OpBuilder, hidden: ir.Value):
        """Encode to codes with residual quantization.

        Args:
            hidden: (B, input_dim, T).

        Returns:
            codes: (B, num_quantizers, T) int64.
            residual: (B, input_dim, T) remaining after quantization.
        """
        # Input projection: (B, input_dim, T) -> (B, dim, T)
        projected = self.input_proj(op, hidden)

        # Residual quantization
        all_codes = []
        residual = projected
        for layer in self.layers:
            codes_i, quantized_i = layer(op, residual)
            all_codes.append(op.Unsqueeze(codes_i, [1]))  # (B, 1, T)
            residual = op.Sub(residual, quantized_i)

        # Stack codes: (B, num_quantizers, T)
        codes = op.Concat(*all_codes, axis=1)

        # Compute full quantized for output residual
        # Re-project back to input_dim via output_proj
        total_quantized = op.Sub(projected, residual)
        total_quantized = self.output_proj(op, total_quantized)
        output_residual = op.Sub(hidden, total_quantized)

        return codes, output_residual


class _EncoderVQ(nn.Module):
    """Single encoder VQ layer: find nearest codebook entry via argmin.

    The codebook embeddings are precomputed in preprocess_weights
    (embedding_sum / cluster_usage).
    """

    def __init__(self, codebook_size: int, dim: int):
        super().__init__()
        # codebook.embed_sum -> precomputed to codebook.embedding
        self.codebook = _EncoderCodebook(codebook_size, dim)

    def forward(self, op: OpBuilder, x: ir.Value):
        """Find nearest codebook entry.

        Args:
            x: (B, dim, T) continuous features.

        Returns:
            codes: (B, T) int64 indices.
            quantized: (B, dim, T) quantized values.
        """
        # Transpose to (B, T, dim) for distance computation
        x_t = op.Transpose(x, perm=[0, 2, 1])

        # Get embedding table via codebook module call
        # (codebook_size, dim)
        embedding = self.codebook(op)

        # Find nearest: ||x - e||² = ||x||² - 2*x·e^T + ||e||²
        x_sq = op.ReduceSumSquare(x_t, [-1], keepdims=1)  # (B,T,1)
        e_sq = op.ReduceSumSquare(
            op.Unsqueeze(embedding, [0]), [-1], keepdims=0
        )  # (1, codebook_size)
        dot = op.MatMul(x_t, op.Transpose(embedding, perm=[1, 0]))
        distances = op.Add(op.Sub(x_sq, op.Mul(dot, 2.0)), e_sq)

        # ArgMin across codebook dimension
        codes = op.ArgMin(distances, axis=-1, keepdims=0)  # (B, T)

        # Look up quantized values (no project_out — dim == codebook_dim)
        quantized = op.Gather(embedding, codes, axis=0)  # (B, T, dim)
        quantized = op.Transpose(quantized, perm=[0, 2, 1])  # (B, dim, T)

        return codes, quantized


class _EncoderCodebook(nn.Module):
    """Encoder codebook: holds the precomputed embedding table.

    Must be called via forward() so onnxscript registers the parameter.
    """

    def __init__(self, codebook_size: int, dim: int):
        super().__init__()
        self.embedding = nn.Parameter([codebook_size, dim])

    def forward(self, op: OpBuilder):
        """Return the embedding table as an identity op.

        This ensures onnxscript registers the parameter.
        """
        return op.Identity(self.embedding)


# ---------------------------------------------------------------------------
# Composite model: routes weights to decoder + encoder
# ---------------------------------------------------------------------------


class Qwen3TTSTokenizerV2Model(nn.Module):
    """Composite Qwen3-TTS codec tokenizer model.

    Holds both decoder and encoder sub-models. The composite
    ``preprocess_weights`` routes HF weights and precomputes
    VQ codebook embeddings from ``embedding_sum / cluster_usage``.

    HF class: ``Qwen3TTSTokenizerV2Model`` in qwen_tts.

    Parameters:
        config: Architecture config.
    """

    # Runtime HF RVQ codec encoder/decoder sub-trees.
    HF_COMPONENT_SOURCES: ClassVar[dict[str, tuple[str, ...]]] = {
        "encoder": ("encoder",),
        "decoder": ("decoder",),
    }

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.decoder = Qwen3TTSCodecDecoderModel(config)
        self.encoder = Qwen3TTSCodecEncoderModel(config)

    def preprocess_weights(
        self,
        state_dict: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Route HF weights and precompute VQ codebook embeddings.

        Key transformations:
        1. ``embedding_sum / cluster_usage`` -> ``embedding.weight``
           for all codebook entries (decoder + encoder).
        2. Rename ``embed_sum`` -> ``embedding`` for encoder codebooks.
        3. Route ``decoder.*`` and ``encoder.*`` weights.
        """
        cleaned: dict[str, torch.Tensor] = {}

        for key, value in state_dict.items():
            # --- Decoder VQ codebooks ---
            # decoder.quantizer.rvq_{first,rest}.vq.layers.{i}._codebook.embedding_sum
            # decoder.quantizer.rvq_{first,rest}.vq.layers.{i}._codebook.cluster_usage
            if "_codebook.embedding_sum" in key:
                usage_key = key.replace("embedding_sum", "cluster_usage")
                if usage_key in state_dict:
                    usage = state_dict[usage_key]
                    # Precompute: embedding = embedding_sum / cluster_usage
                    embedding = value / usage.clamp(min=1e-7).unsqueeze(-1)
                    # Map to our Embedding weight name
                    # HF: ...vq.layers.{i}._codebook.embedding_sum
                    # Ours: ...vq.layers.{i}._codebook.embedding.weight
                    emb_key = key.replace(
                        "_codebook.embedding_sum", "_codebook.embedding.weight"
                    )
                    cleaned[emb_key] = embedding
                continue

            if "_codebook.cluster_usage" in key:
                # Already consumed above
                continue

            # --- Encoder VQ codebooks ---
            # encoder.quantizer.{semantic,acoustic}_residual_vector_quantizer
            #   .layers.{i}.codebook.embed_sum
            #   .layers.{i}.codebook.cluster_usage
            if "codebook.embed_sum" in key:
                usage_key = key.replace("embed_sum", "cluster_usage")
                if usage_key in state_dict:
                    usage = state_dict[usage_key]
                    embedding = value / usage.clamp(min=1e-7).unsqueeze(-1)
                    emb_key = key.replace("codebook.embed_sum", "codebook.embedding")
                    cleaned[emb_key] = embedding
                continue

            if "codebook.cluster_usage" in key:
                continue

            # Skip encoder codebook initialized flags
            if "codebook.initialized" in key:
                continue

            # Rename encoder transformer MLP weights:
            # HF uses fc1/fc2; ONNX uses up_proj/down_proj
            key = key.replace(".mlp.fc1.", ".mlp.up_proj.")
            key = key.replace(".mlp.fc2.", ".mlp.down_proj.")
            # Pass through everything else
            cleaned[key] = value

        return cleaned
