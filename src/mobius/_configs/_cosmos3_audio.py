# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Configuration for the Cosmos3 AVAE audio tokenizer (``sound_tokenizer``).

Mirrors the ``Cosmos3AVAEAudioTokenizer`` config surface from
``diffusers.models.autoencoders.autoencoder_cosmos3_audio`` as published for
``nvidia/Cosmos3-Nano`` / ``nvidia/Cosmos3-Super`` (``sound_tokenizer/config.json``).

The shipped checkpoint configuration is::

    model_type      = "autoencoder_v2"    sampling_rate = 48000
    enc_type        = "spec_convnext"     dec_type      = "oobleck"
    bottleneck_type = "vae"               activation    = "snakebeta"
    hop_size        = 1920                vocoder_input_dim = 64
    stereo          = True                dec_out_channels  = 2

Only that configuration is supported; every other variant raises
:class:`NotImplementedError` from :meth:`Cosmos3AudioConfig.validate`, exactly
like the upstream ``__init__`` guards.

Encoder presence is **not** a config property
---------------------------------------------

``nvidia/Cosmos3-Nano`` and ``nvidia/Cosmos3-Super`` ship full
encoder+decoder AVAE weights (249 tensors: 67 ``encoder.*`` + 182
``decoder.*``), while ``nvidia/Cosmos3-Super-Text2Image`` ships decoder-only
weights (182 tensors, zero ``encoder.*``). All three ``sound_tokenizer/
config.json`` files are **byte-identical**: every ``enc_*`` field is present
and none of them carries an ``encoder_enabled`` key.

Build-time config therefore cannot reveal encoder absence, which is why
:func:`state_dict_has_encoder` / :meth:`Cosmos3AudioConfig.from_diffusers`
(``weight_names=``) / :meth:`Cosmos3AudioConfig.with_encoder_from_state_dict`
exist, and why the model and task layers expose separate decoder-only and
encoder-decoder paths rather than one flag-driven class.

Two invariants tie the encoder and decoder together and are checked here:

* ``enc_latent_dim == 2 * vocoder_input_dim`` — the encoder emits
  ``[mean, scale]`` moments that the VAE bottleneck splits in half.
* ``prod(enc_strides) * enc_hop_length == prod(dec_strides) == hop_size`` — the
  waveform/latent compression factor must agree between the STFT front-end,
  the ConvNeXt downsampling stack and the Oobleck upsampling stack.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Iterable, Mapping, Sequence

from mobius._configs._base import BaseModelConfig

__all__ = ["Cosmos3AudioConfig", "state_dict_has_encoder"]

_ENCODER_WEIGHT_PREFIX = "encoder."


def state_dict_has_encoder(weight_names: Iterable[str]) -> bool:
    """Return ``True`` when a checkpoint carries Cosmos3 AVAE encoder weights.

    The upstream ``Cosmos3AVAEAudioTokenizer._fix_state_dict_keys_on_load``
    drops the encoder whenever no ``encoder.*`` key is present, because AVAE
    checkpoints are sometimes shipped decoder-only for sound generation. Mobius
    needs the same signal *before* graph construction so it never emits
    encoder initializers that no weight can fill.

    Args:
        weight_names: Iterable of checkpoint tensor names (or any mapping whose
            iteration yields them, such as a ``state_dict``).

    Returns:
        ``True`` if at least one name starts with ``"encoder."``.
    """
    return any(str(name).startswith(_ENCODER_WEIGHT_PREFIX) for name in weight_names)


def _as_int_tuple(values: Sequence[int] | int, field: str) -> tuple[int, ...]:
    """Coerce a config list/tuple of positive ints into a tuple."""
    if isinstance(values, int):
        values = (values,)
    result = tuple(int(v) for v in values)
    if not result:
        raise ValueError(f"{field} must contain at least one entry")
    if any(v <= 0 for v in result):
        raise ValueError(f"{field} must contain only positive integers, got {result}")
    return result


@dataclasses.dataclass
class Cosmos3AudioConfig(BaseModelConfig):
    """Configuration for ``Cosmos3AVAEAudioTokenizer``.

    Field names and defaults are copied verbatim from the upstream diffusers
    ``__init__`` signature so a ``sound_tokenizer/config.json`` can be splatted
    in without renaming.

    Attributes:
        model_type: AVAE variant; only ``"autoencoder_v2"`` is supported.
        sampling_rate: Waveform sample rate in Hz (48000 for Cosmos3).
        vocoder_input_dim: Latent channel count consumed by the decoder; equals
            the transformer ``sound_dim``.
        dec_dim: Base decoder channel count.
        dec_c_mults: Decoder channel multipliers (low → high resolution).
        dec_strides: Decoder strides; the decoder consumes them reversed.
        dec_out_channels: Output waveform channels (2 = stereo).
        stereo: Whether audio is stereo; doubles the encoder input channels.
        use_wav_as_input: Whether the encoder consumes raw waveforms.
        normalize_volume: Whether ``encode`` peak-normalizes before encoding.
        hop_size: Waveform→latent compression factor; defaults to
            ``prod(dec_strides)`` when ``None``.
        input_channels: Per-channel encoder input count before ``stereo``
            doubling.
        enc_type: Encoder type; only ``"spec_convnext"`` is supported.
        enc_dim: Base encoder channel count.
        enc_intermediate_dim: Unused upstream (ConvNeXt blocks use
            ``input_dim * 4``); retained for config fidelity.
        enc_num_layers: Unused upstream (depth derives from ``enc_num_blocks``);
            retained for config fidelity.
        enc_num_blocks: ConvNeXt blocks per encoder downsampling stage.
        enc_n_fft: STFT size of the encoder spectrogram front-end.
        enc_hop_length: STFT hop length of the encoder front-end.
        enc_latent_dim: Encoder output channels (``2 * vocoder_input_dim``).
        enc_c_mults: Encoder channel multipliers per stage.
        enc_strides: Encoder downsampling strides per stage.
        enc_identity_init: Zero-init flag for the ConvNeXt residual 1x1 conv.
            Training-time only; it does not change the exported graph.
        enc_use_snake: Whether ConvNeXt blocks use SnakeBeta (else GELU).
        dec_type: Decoder type; only ``"oobleck"`` is supported.
        dec_use_snake: Whether the decoder uses SnakeBeta; must be ``True``.
        dec_final_tanh: Vestigial decoder flag; must be ``False``.
        dec_anti_aliasing: Decoder anti-aliasing flag; must be ``False``.
        dec_use_nearest_upsample: Decoder upsample flag; must be ``False``.
        dec_use_tanh_at_final: Decoder final-tanh flag; must be ``False``.
        bottleneck_type: Bottleneck type; only ``"vae"`` is supported.
        bottleneck: Optional bottleneck dict whose ``"type"`` must be ``"vae"``.
        activation: Activation family; only ``"snakebeta"`` is supported.
        snake_logscale: Whether SnakeBeta parameters are log-scaled; must be
            ``True``.
        anti_aliasing: Global anti-aliasing flag; must be ``False``.
        use_cuda_kernel: Fused-CUDA-kernel flag; must be ``False``.
        causal: Whether convolutions are causal; must be ``False``.
        padding_mode: Convolution padding mode; only ``"zeros"`` is supported.
        latent_mean: Latent normalization mean; must be ``None`` (upstream does
            not implement latent normalization).
        latent_std: Latent normalization std; must be ``None``.
        encoder_enabled: Whether the encoder exists in this checkpoint. Set to
            ``False`` for decoder-only AVAE weights so no encoder initializer
            is ever created.
    """

    model_type: str = "autoencoder_v2"
    sampling_rate: int = 48000
    vocoder_input_dim: int = 64
    dec_dim: int = 320
    dec_c_mults: tuple[int, ...] = (1, 2, 4, 8, 16)
    dec_strides: tuple[int, ...] = (2, 4, 5, 6, 8)
    dec_out_channels: int = 2
    stereo: bool = True
    use_wav_as_input: bool = True
    normalize_volume: bool = True
    hop_size: int | None = None
    input_channels: int = 1
    enc_type: str = "spec_convnext"
    enc_dim: int = 192
    enc_intermediate_dim: int = 768
    enc_num_layers: int = 12
    enc_num_blocks: int = 2
    enc_n_fft: int = 64
    enc_hop_length: int = 16
    enc_latent_dim: int = 128
    enc_c_mults: tuple[int, ...] = (1, 2, 4)
    enc_strides: tuple[int, ...] = (4, 5, 6)
    enc_identity_init: bool = False
    enc_use_snake: bool = True
    dec_type: str = "oobleck"
    dec_use_snake: bool = True
    dec_final_tanh: bool = False
    dec_anti_aliasing: bool = False
    dec_use_nearest_upsample: bool = False
    dec_use_tanh_at_final: bool = False
    bottleneck_type: str = "vae"
    bottleneck: dict | None = None
    activation: str = "snakebeta"
    snake_logscale: bool = True
    anti_aliasing: bool = False
    use_cuda_kernel: bool = False
    causal: bool = False
    padding_mode: str = "zeros"
    latent_mean: float | list[float] | None = None
    latent_std: float | list[float] | None = None
    encoder_enabled: bool = True

    def __post_init__(self) -> None:
        """Normalize sequence fields and resolve the default ``hop_size``."""
        self.dec_c_mults = _as_int_tuple(self.dec_c_mults, "dec_c_mults")
        self.dec_strides = _as_int_tuple(self.dec_strides, "dec_strides")
        self.enc_c_mults = _as_int_tuple(self.enc_c_mults, "enc_c_mults")
        self.enc_strides = _as_int_tuple(self.enc_strides, "enc_strides")
        if self.hop_size is None:
            self.hop_size = math.prod(self.dec_strides)
        else:
            self.hop_size = int(self.hop_size)

    # -- Derived geometry --------------------------------------------------

    @property
    def encoder_input_channels(self) -> int:
        """Waveform channels the encoder expects (``input_channels`` x stereo)."""
        return self.input_channels * (2 if self.stereo else 1)

    @property
    def stft_num_bins(self) -> int:
        """One-sided STFT bin count (``n_fft // 2 + 1``)."""
        return self.enc_n_fft // 2 + 1

    @property
    def stft_pad_left(self) -> int:
        """Left zero-pad applied before the (``center=False``) STFT."""
        return (self.enc_n_fft - self.enc_hop_length) // 2

    @property
    def stft_pad_right(self) -> int:
        """Right zero-pad applied before the (``center=False``) STFT."""
        return (self.enc_n_fft - self.enc_hop_length) - self.stft_pad_left

    @property
    def spectrogram_channels(self) -> int:
        """Channels of the packed real/imaginary spectrogram fed to ``layers.0``.

        ``(n_fft + 2)`` equals ``2 * stft_num_bins`` (real bins followed by
        imaginary bins), multiplied by the number of waveform channels.
        """
        return (self.enc_n_fft + 2) * self.encoder_input_channels

    @property
    def latent_channels(self) -> int:
        """Latent channels after the VAE bottleneck split (``vocoder_input_dim``)."""
        return self.vocoder_input_dim

    @property
    def moments_channels(self) -> int:
        """Channels of the un-split posterior moments (``enc_latent_dim``)."""
        return self.enc_latent_dim

    @property
    def audio_channels(self) -> int:
        """Waveform channels produced by the decoder (``dec_out_channels``)."""
        return self.dec_out_channels

    @property
    def decoder_upsampling_ratios(self) -> tuple[int, ...]:
        """Decoder strides in application order (``reversed(dec_strides)``)."""
        return tuple(reversed(self.dec_strides))

    @property
    def decoder_channel_multiples(self) -> tuple[int, ...]:
        """``[1] + dec_c_mults`` — the multiplier table the decoder indexes."""
        return (1, *self.dec_c_mults)

    @property
    def decoder_upsample_factor(self) -> int:
        """Latent frames → waveform samples ratio (``prod(dec_strides)``)."""
        return math.prod(self.dec_strides)

    @property
    def encoder_downsample_factor(self) -> int:
        """Waveform samples → latent frames ratio for the spec-ConvNeXt encoder."""
        return math.prod(self.enc_strides) * self.enc_hop_length

    @property
    def resolved_hop_size(self) -> int:
        """The effective ``hop_size`` (never ``None`` after ``__post_init__``)."""
        assert self.hop_size is not None
        return self.hop_size

    # -- Validation --------------------------------------------------------

    def validate(self) -> None:
        """Validate that this config matches the supported Cosmos3 AVAE variant.

        Raises:
            NotImplementedError: For any architecture variant the upstream
                ``Cosmos3AVAEAudioTokenizer.__init__`` also rejects.
            ValueError: For structurally inconsistent dimensions (channel
                multiplier / stride length mismatch, latent-dim mismatch, or a
                ``hop_size`` that disagrees with the stride products).
        """
        if self.model_type != "autoencoder_v2":
            raise NotImplementedError(
                f"Cosmos3 AVAE model type {self.model_type!r} is not supported."
            )
        if not self.use_wav_as_input:
            raise NotImplementedError("Cosmos3 AVAE tokenizer only supports waveform input.")
        if self.enc_type != "spec_convnext":
            raise NotImplementedError(
                f"Cosmos3 AVAE encoder type {self.enc_type!r} is not supported."
            )
        if self.bottleneck is not None:
            declared = self.bottleneck.get("type", self.bottleneck_type)
            if declared != "vae":
                raise NotImplementedError(
                    "Cosmos3 AVAE tokenizer only supports the VAE bottleneck, got "
                    f"bottleneck={{'type': {declared!r}}}."
                )
        if self.bottleneck_type != "vae":
            raise NotImplementedError(
                "Cosmos3 AVAE tokenizer only supports the VAE bottleneck, got "
                f"bottleneck_type={self.bottleneck_type!r}."
            )
        if self.dec_type != "oobleck":
            raise NotImplementedError(
                f"Cosmos3 AVAE decoder type {self.dec_type!r} is not supported."
            )
        if (
            not self.dec_use_snake
            or self.dec_final_tanh
            or self.dec_anti_aliasing
            or self.dec_use_nearest_upsample
            or self.dec_use_tanh_at_final
        ):
            raise NotImplementedError(
                "Cosmos3 AVAE decoder only supports the shipped Oobleck decoder configuration "
                "(dec_use_snake=True and every dec_* toggle False)."
            )
        if (
            self.activation != "snakebeta"
            or not self.snake_logscale
            or self.anti_aliasing
            or self.use_cuda_kernel
        ):
            raise NotImplementedError(
                "Cosmos3 AVAE tokenizer only supports the shipped SnakeBeta configuration "
                "(activation='snakebeta', snake_logscale=True, anti_aliasing=False, "
                "use_cuda_kernel=False)."
            )
        if self.causal:
            raise NotImplementedError(
                "Cosmos3 AVAE causal audio encoder is not supported yet."
            )
        if self.padding_mode != "zeros":
            raise NotImplementedError(
                f"Cosmos3 AVAE only supports padding_mode='zeros', got {self.padding_mode!r}."
            )
        if self.latent_mean is not None or self.latent_std is not None:
            raise NotImplementedError(
                "Cosmos3 AVAE tokenizer does not apply latent normalization; "
                "`latent_mean`/`latent_std` must be None."
            )

        if len(self.enc_c_mults) != len(self.enc_strides):
            raise ValueError(
                "`enc_c_mults` and `enc_strides` must have the same length, got "
                f"{len(self.enc_c_mults)} and {len(self.enc_strides)}."
            )
        if len(self.dec_c_mults) != len(self.dec_strides):
            raise ValueError(
                "`dec_c_mults` and `dec_strides` must have the same length, got "
                f"{len(self.dec_c_mults)} and {len(self.dec_strides)}."
            )
        if self.enc_latent_dim != 2 * self.vocoder_input_dim:
            raise ValueError(
                "Cosmos3 AVAE VAE bottleneck splits the encoder output into mean/scale, so "
                f"enc_latent_dim must be 2 * vocoder_input_dim; got enc_latent_dim="
                f"{self.enc_latent_dim} and vocoder_input_dim={self.vocoder_input_dim}."
            )
        if self.enc_n_fft <= 0 or self.enc_n_fft % 2 != 0:
            raise ValueError(
                f"enc_n_fft must be a positive even number, got {self.enc_n_fft}."
            )
        if not 0 < self.enc_hop_length <= self.enc_n_fft:
            raise ValueError(
                "enc_hop_length must satisfy 0 < enc_hop_length <= enc_n_fft; got "
                f"enc_hop_length={self.enc_hop_length}, enc_n_fft={self.enc_n_fft}."
            )
        for name in (
            "sampling_rate",
            "enc_dim",
            "dec_dim",
            "vocoder_input_dim",
            "enc_num_blocks",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}.")
        if self.input_channels <= 0:
            raise ValueError(f"input_channels must be positive, got {self.input_channels}.")
        if self.dec_out_channels != self.encoder_input_channels:
            raise ValueError(
                "Cosmos3 AVAE round-trips audio, so dec_out_channels must equal "
                f"input_channels * (2 if stereo else 1); got dec_out_channels="
                f"{self.dec_out_channels} and encoder_input_channels="
                f"{self.encoder_input_channels}."
            )
        if self.decoder_upsample_factor != self.resolved_hop_size:
            raise ValueError(
                "hop_size must equal prod(dec_strides); got hop_size="
                f"{self.resolved_hop_size} and prod(dec_strides)="
                f"{self.decoder_upsample_factor}."
            )
        if self.encoder_downsample_factor != self.resolved_hop_size:
            raise ValueError(
                "The encoder compression factor prod(enc_strides) * enc_hop_length must equal "
                f"hop_size; got {self.encoder_downsample_factor} != {self.resolved_hop_size}."
            )

    # -- Construction ------------------------------------------------------

    @classmethod
    def from_diffusers(
        cls,
        config: Mapping[str, object] | object,
        *,
        encoder_enabled: bool | None = None,
        weight_names: Iterable[str] | None = None,
    ) -> Cosmos3AudioConfig:
        """Create a config from a diffusers ``sound_tokenizer/config.json`` dict.

        Unknown/private diffusers bookkeeping keys (``_class_name``,
        ``_diffusers_version``, ...) are ignored.

        .. warning::
            The published ``sound_tokenizer/config.json`` files **cannot** tell
            you whether an encoder exists. ``nvidia/Cosmos3-Nano``,
            ``nvidia/Cosmos3-Super`` (both full) and
            ``nvidia/Cosmos3-Super-Text2Image`` (decoder-only) all ship the
            *byte-identical* config: every ``enc_*`` field is present and there
            is no ``encoder_enabled`` key. Encoder presence is only observable
            from the checkpoint, so pass ``weight_names`` (preferred) or
            ``encoder_enabled``. Leaving both unset assumes a full checkpoint
            and will emit encoder initializers that decoder-only weights cannot
            fill.

        Args:
            config: Parsed ``sound_tokenizer/config.json`` mapping, or any
                object exposing ``to_dict()`` / ``items()``.
            encoder_enabled: Explicit override for encoder presence.
            weight_names: Checkpoint tensor names (or a ``state_dict``) from
                which encoder presence is detected. Mutually exclusive with
                ``encoder_enabled``.

        Returns:
            A validated :class:`Cosmos3AudioConfig`.

        Raises:
            ValueError: If both ``encoder_enabled`` and ``weight_names`` are given.
        """
        if encoder_enabled is not None and weight_names is not None:
            raise ValueError(
                "Pass either `encoder_enabled` or `weight_names`, not both — they are two "
                "ways to answer the same question."
            )
        if hasattr(config, "to_dict"):
            data = dict(config.to_dict())  # type: ignore[attr-defined]
        elif isinstance(config, Mapping):
            data = dict(config)
        else:
            data = dict(config.items())  # type: ignore[attr-defined]

        known = {field.name for field in dataclasses.fields(cls)}
        kwargs = {key: value for key, value in data.items() if key in known}
        if weight_names is not None:
            kwargs["encoder_enabled"] = state_dict_has_encoder(weight_names)
        elif encoder_enabled is not None:
            kwargs["encoder_enabled"] = encoder_enabled
        resolved = cls(**kwargs)  # type: ignore[arg-type]
        resolved.validate()
        return resolved

    def with_encoder_from_state_dict(
        self,
        weight_names: Iterable[str],
    ) -> Cosmos3AudioConfig:
        """Return a copy whose ``encoder_enabled`` reflects the checkpoint.

        This is the mobius equivalent of the upstream
        ``_fix_state_dict_keys_on_load`` hook: decoder-only AVAE checkpoints
        must never build an encoder graph, because every encoder initializer
        would be left without weight data.

        Args:
            weight_names: Checkpoint tensor names (or a ``state_dict``).

        Returns:
            ``self`` when the flag already matches, otherwise a new config.
        """
        has_encoder = state_dict_has_encoder(weight_names)
        if has_encoder == self.encoder_enabled:
            return self
        return dataclasses.replace(self, encoder_enabled=has_encoder)
