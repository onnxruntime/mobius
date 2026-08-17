# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for the Cosmos3 AVAE audio tokenizer (``sound_tokenizer``).

Coverage:

* parsing/validating the real published ``sound_tokenizer/config.json``;
* every ``NotImplementedError``/``ValueError`` guard in
  :meth:`Cosmos3AudioConfig.validate`;
* the decoder-only graph (no encoder initializers at all) and the full
  encoder+decoder package, including exact I/O contracts and initializer names;
* encoder-weight detection from checkpoint key names;
* ``weight_norm`` folding against ``torch.nn.utils.weight_norm``;
* numerical parity with a PyTorch reference transcribed from
  ``diffusers.models.autoencoders.autoencoder_cosmos3_audio`` for SnakeBeta, a
  residual unit, the whole tiny decoder, and the whole tiny STFT encoder.
"""

from __future__ import annotations

import math

import numpy as np
import onnx_ir as ir
import pytest
import torch
from onnxscript import GraphBuilder, nn
from torch import nn as tnn

from mobius._configs._cosmos3_audio import Cosmos3AudioConfig, state_dict_has_encoder
from mobius._constants import OPSET_VERSION
from mobius._model_package import ModelPackage
from mobius._testing.ort_inference import OnnxModelSession
from mobius.models.cosmos3_audio import (
    Cosmos3AudioConvNeXtBlock,
    Cosmos3AudioResidualUnit,
    Cosmos3AVAEAudioDecoderOnlyTokenizer,
    Cosmos3AVAEAudioTokenizer,
    _Snake1d,
    create_cosmos3_avae_audio_tokenizer,
    fold_weight_norm,
)
from mobius.tasks._cosmos3_audio import (
    Cosmos3AVAEAudioDecoderTask,
    Cosmos3AVAEAudioTokenizerTask,
    select_cosmos3_audio_task,
)

# The published nvidia/Cosmos3-Nano and nvidia/Cosmos3-Super
# ``sound_tokenizer/config.json`` (byte-identical between the two releases).
_REAL_CONFIG: dict = {
    "model_type": "autoencoder_v2",
    "sampling_rate": 48000,
    "stereo": True,
    "use_wav_as_input": True,
    "normalize_volume": True,
    "hop_size": 1920,
    "input_channels": 1,
    "enc_type": "spec_convnext",
    "enc_dim": 192,
    "enc_intermediate_dim": 768,
    "enc_num_layers": 12,
    "enc_num_blocks": 2,
    "enc_n_fft": 64,
    "enc_hop_length": 16,
    "enc_latent_dim": 128,
    "enc_c_mults": [1, 2, 4],
    "enc_strides": [4, 5, 6],
    "enc_identity_init": False,
    "enc_use_snake": True,
    "dec_type": "oobleck",
    "dec_dim": 320,
    "dec_c_mults": [1, 2, 4, 8, 16],
    "dec_strides": [2, 4, 5, 6, 8],
    "dec_use_snake": True,
    "dec_final_tanh": False,
    "dec_out_channels": 2,
    "dec_anti_aliasing": False,
    "dec_use_nearest_upsample": False,
    "dec_use_tanh_at_final": False,
    "bottleneck_type": "vae",
    "bottleneck": {"type": "vae"},
    "activation": "snakebeta",
    "snake_logscale": True,
    "anti_aliasing": False,
    "use_cuda_kernel": False,
    "causal": False,
    "padding_mode": "zeros",
    "vocoder_input_dim": 64,
    "latent_mean": None,
    "latent_std": None,
}

# Tiny but *structurally faithful* config: two encoder stages, two decoder
# blocks, stereo I/O, and a consistent compression factor of 12 samples/frame
# (prod(dec_strides) == prod(enc_strides) * enc_hop_length == hop_size).
_TINY_KWARGS: dict = {
    "vocoder_input_dim": 2,
    "dec_dim": 4,
    "dec_c_mults": (1, 2),
    "dec_strides": (3, 4),
    "dec_out_channels": 2,
    "hop_size": 12,
    "enc_dim": 4,
    "enc_num_blocks": 1,
    "enc_n_fft": 8,
    "enc_hop_length": 2,
    "enc_latent_dim": 4,
    "enc_c_mults": (1, 2),
    "enc_strides": (2, 3),
}


def _tiny_config(**overrides) -> Cosmos3AudioConfig:
    """Build the shared tiny config, applying *overrides*."""
    config = Cosmos3AudioConfig(**{**_TINY_KWARGS, **overrides})
    config.validate()
    return config


# ---------------------------------------------------------------------------
# PyTorch reference — a direct transcription of the upstream diffusers module
# ---------------------------------------------------------------------------


class _RefSnake1d(tnn.Module):
    """Reference ``Snake1d``."""

    def __init__(self, hidden_dim: int, logscale: bool = True):
        super().__init__()
        self.alpha = tnn.Parameter(torch.zeros(1, hidden_dim, 1))
        self.beta = tnn.Parameter(torch.zeros(1, hidden_dim, 1))
        self.logscale = logscale

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        alpha = self.alpha if not self.logscale else torch.exp(self.alpha)
        beta = self.beta if not self.logscale else torch.exp(self.beta)
        return hidden_states + (beta + 1e-9).reciprocal() * torch.sin(
            alpha * hidden_states
        ).pow(2)


class _RefFP32LayerNorm(tnn.LayerNorm):
    """Reference ``FP32LayerNorm`` (diffusers)."""

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        origin_dtype = inputs.dtype
        return torch.nn.functional.layer_norm(
            inputs.float(),
            self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        ).to(origin_dtype)


class _RefConvNeXtBlock(tnn.Module):
    """Reference ``Cosmos3AudioConvNeXtBlock`` (non-causal, SnakeBeta)."""

    def __init__(self, hidden_dim: int, intermediate_dim: int):
        super().__init__()
        self.dwconv = tnn.Sequential(
            tnn.ConstantPad1d((3, 3), 0),
            tnn.Conv1d(hidden_dim, hidden_dim, kernel_size=7, groups=hidden_dim),
        )
        self.norm = _RefFP32LayerNorm(hidden_dim, eps=1e-5, bias=False)
        self.pwconv1 = tnn.Conv1d(hidden_dim, intermediate_dim, kernel_size=1)
        self.act = _RefSnake1d(intermediate_dim)
        self.pwconv2 = tnn.Conv1d(intermediate_dim, hidden_dim, kernel_size=1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.dwconv(hidden_states)
        hidden_states = self.norm(hidden_states.permute(0, 2, 1)).permute(0, 2, 1)
        hidden_states = self.pwconv1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.pwconv2(hidden_states)
        return residual + hidden_states


class _RefEncoder(tnn.Module):
    """Reference ``Cosmos3AudioSpectrogramConvNeXtEncoder`` (weight-norm folded)."""

    def __init__(self, config: Cosmos3AudioConfig):
        super().__init__()
        self.input_channels = config.encoder_input_channels
        self.n_fft = config.enc_n_fft
        self.hop_length = config.enc_hop_length
        channels = config.enc_dim
        multiples = config.enc_c_mults
        strides = config.enc_strides

        layers: list[tnn.Module] = [
            tnn.Conv1d(config.spectrogram_channels, multiples[0] * channels, 1, bias=False)
        ]
        for index, stride in enumerate(strides):
            input_dim = multiples[index] * channels
            output_dim = (
                multiples[index + 1] * channels
                if index < len(multiples) - 1
                else multiples[-1] * channels
            )
            for _ in range(config.enc_num_blocks):
                layers.append(_RefConvNeXtBlock(input_dim, input_dim * 4))
            layers.append(
                tnn.Conv1d(
                    input_dim,
                    output_dim,
                    kernel_size=2 * stride,
                    stride=stride,
                    padding=math.ceil(stride / 2),
                )
            )
        layers.append(
            tnn.Conv1d(multiples[-1] * channels, config.enc_latent_dim, 1, bias=False)
        )
        self.layers = tnn.Sequential(*layers)

    def _spectrogram(self, waveform: torch.Tensor) -> torch.Tensor:
        pad_left = (self.n_fft - self.hop_length) // 2
        pad_right = (self.n_fft - self.hop_length) - pad_left
        waveform = torch.nn.functional.pad(waveform, (pad_left, pad_right)).float()
        window = torch.hann_window(self.n_fft, dtype=waveform.dtype)
        return torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=window,
            center=False,
            normalized=False,
            onesided=True,
            return_complex=True,
        )

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, num_samples = audio.shape
        if num_channels > 1:
            audio = audio.reshape(batch_size * num_channels, 1, num_samples)
        spectrogram = self._spectrogram(audio.squeeze(1))
        real, imaginary = torch.view_as_real(spectrogram).chunk(2, dim=-1)
        spectrogram = torch.cat([real, imaginary], dim=1).squeeze(-1)
        spectrogram = spectrogram.to(audio.dtype)
        if num_channels > 1:
            spectrogram = spectrogram.reshape(
                batch_size, num_channels * spectrogram.shape[1], spectrogram.shape[2]
            )
        return self.layers(spectrogram).transpose(1, 2)


class _RefResidualUnit(tnn.Module):
    """Reference ``Cosmos3AudioResidualUnit`` (weight-norm folded)."""

    def __init__(self, dimension: int = 16, dilation: int = 1):
        super().__init__()
        pad = ((7 - 1) * dilation) // 2
        self.snake1 = _RefSnake1d(dimension)
        self.conv1 = tnn.Conv1d(dimension, dimension, 7, dilation=dilation, padding=pad)
        self.snake2 = _RefSnake1d(dimension)
        self.conv2 = tnn.Conv1d(dimension, dimension, 1)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        output_tensor = self.conv1(self.snake1(hidden_state))
        output_tensor = self.conv2(self.snake2(output_tensor))
        padding = (hidden_state.shape[-1] - output_tensor.shape[-1]) // 2
        if padding > 0:
            hidden_state = hidden_state[..., padding:-padding]
        return hidden_state + output_tensor


class _RefDecoderBlock(tnn.Module):
    """Reference ``Cosmos3AudioDecoderBlock`` (weight-norm folded)."""

    def __init__(self, input_dim: int, output_dim: int, stride: int, output_padding: int):
        super().__init__()
        self.snake1 = _RefSnake1d(input_dim)
        self.conv_t1 = tnn.ConvTranspose1d(
            input_dim,
            output_dim,
            kernel_size=2 * stride,
            stride=stride,
            padding=math.ceil(stride / 2),
            output_padding=output_padding,
        )
        self.res_unit1 = _RefResidualUnit(output_dim, dilation=1)
        self.res_unit2 = _RefResidualUnit(output_dim, dilation=3)
        self.res_unit3 = _RefResidualUnit(output_dim, dilation=9)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        hidden_state = self.snake1(hidden_state)
        hidden_state = self.conv_t1(hidden_state)
        hidden_state = self.res_unit1(hidden_state)
        hidden_state = self.res_unit2(hidden_state)
        return self.res_unit3(hidden_state)


class _RefDecoder(tnn.Module):
    """Reference ``Cosmos3AudioDecoder`` (weight-norm folded)."""

    def __init__(self, config: Cosmos3AudioConfig):
        super().__init__()
        channels = config.dec_dim
        strides = config.decoder_upsampling_ratios
        multiples = config.decoder_channel_multiples
        self.conv1 = tnn.Conv1d(
            config.vocoder_input_dim, channels * multiples[-1], kernel_size=7, padding=3
        )
        self.block = tnn.ModuleList(
            [
                _RefDecoderBlock(
                    input_dim=channels * multiples[len(strides) - i],
                    output_dim=channels * multiples[len(strides) - i - 1],
                    stride=stride,
                    output_padding=stride % 2,
                )
                for i, stride in enumerate(strides)
            ]
        )
        self.snake1 = _RefSnake1d(channels)
        self.conv2 = tnn.Conv1d(
            channels, config.dec_out_channels, kernel_size=7, padding=3, bias=False
        )

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        hidden_state = self.conv1(hidden_state)
        for layer in self.block:
            hidden_state = layer(hidden_state)
        hidden_state = self.snake1(hidden_state)
        return self.conv2(hidden_state)


def _randomize(module: tnn.Module, seed: int) -> dict[str, torch.Tensor]:
    """Fill every parameter with small random values and return the state dict.

    Snake ``alpha``/``beta`` are log-scaled, so keeping them near zero keeps
    ``exp()`` well conditioned and mirrors a trained checkpoint.
    """
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, param in module.named_parameters():
            scale = 0.1 if name.endswith((".alpha", ".beta")) else 0.3
            param.copy_(torch.randn(param.shape, generator=generator) * scale)
    return {name: value.detach().clone() for name, value in module.state_dict().items()}


def _session(model, state_dict: dict[str, torch.Tensor], config) -> OnnxModelSession:
    """Apply weights to a single-model package and open an ORT session.

    Every initializer must be filled — an unfilled one means the reference
    state-dict names and the graph initializer names have drifted apart, which
    is exactly the failure mode the HF-aligned naming exists to prevent.
    """
    package = ModelPackage({"model": model}, config=config)
    package.apply_weights(state_dict)
    unset = [
        name for name, init in model.graph.initializers.items() if init.const_value is None
    ]
    assert not unset, f"initializers without weights: {unset}"
    return OnnxModelSession(model)


def _wrap_module_graph(
    module: nn.Module,
    name: str,
    channels: int,
) -> ir.Model:
    """Build a single-module ONNX graph with a ``(B, channels, T)`` contract."""
    module._set_name(name)
    graph = ir.Graph([], [], nodes=[], name=name, opset_imports={"": OPSET_VERSION})
    builder = GraphBuilder(graph)
    x = builder.input("x", dtype=ir.DataType.FLOAT, shape=["batch", channels, "time"])
    builder.add_output(module(builder.op, x), "y")
    return ir.Model(graph, ir_version=11)


# ---------------------------------------------------------------------------
# Config parsing and validation
# ---------------------------------------------------------------------------


def test_real_config_parses_and_validates():
    """The published sound_tokenizer config round-trips through from_diffusers."""
    config = Cosmos3AudioConfig.from_diffusers(_REAL_CONFIG)

    assert config.model_type == "autoencoder_v2"
    assert config.sampling_rate == 48000
    assert config.stereo is True
    assert config.input_channels == 1
    assert config.encoder_input_channels == 2
    assert config.dec_out_channels == 2
    assert config.resolved_hop_size == 1920
    assert config.vocoder_input_dim == 64
    assert config.enc_latent_dim == 128
    assert config.enc_type == "spec_convnext"
    assert config.dec_type == "oobleck"
    assert config.bottleneck_type == "vae"
    assert config.activation == "snakebeta"
    assert config.snake_logscale is True
    assert config.latent_mean is None
    assert config.latent_std is None
    assert config.encoder_enabled is True
    # Tuples, not the lists that came out of JSON.
    assert config.enc_c_mults == (1, 2, 4)
    assert config.enc_strides == (4, 5, 6)
    assert config.dec_c_mults == (1, 2, 4, 8, 16)
    assert config.dec_strides == (2, 4, 5, 6, 8)


def test_real_config_derived_geometry():
    """Derived shapes agree with the checkpoint tensor shapes."""
    config = Cosmos3AudioConfig.from_diffusers(_REAL_CONFIG)

    # encoder.layers.0.weight is [192, 132, 1] in the published checkpoint.
    assert config.spectrogram_channels == 132
    assert config.stft_num_bins == 33
    assert config.stft_pad_left == 24
    assert config.stft_pad_right == 24
    assert config.latent_channels == 64
    assert config.moments_channels == 128
    assert config.audio_channels == 2
    # Encoder and decoder must agree on samples-per-latent-frame.
    assert config.decoder_upsample_factor == 1920
    assert config.encoder_downsample_factor == 1920
    assert config.decoder_upsampling_ratios == (8, 6, 5, 4, 2)
    assert config.decoder_channel_multiples == (1, 1, 2, 4, 8, 16)


def test_hop_size_defaults_to_decoder_stride_product():
    """An absent hop_size falls back to prod(dec_strides)."""
    raw = dict(_REAL_CONFIG)
    del raw["hop_size"]
    config = Cosmos3AudioConfig.from_diffusers(raw)
    assert config.resolved_hop_size == 1920


def test_from_diffusers_ignores_unknown_keys():
    """Diffusers bookkeeping keys do not break construction."""
    raw = {
        **_REAL_CONFIG,
        "_class_name": "Cosmos3AVAEAudioTokenizer",
        "_diffusers_version": "1",
    }
    assert Cosmos3AudioConfig.from_diffusers(raw).sampling_rate == 48000


def test_from_diffusers_encoder_override():
    """The encoder_enabled override reaches the parsed config."""
    config = Cosmos3AudioConfig.from_diffusers(_REAL_CONFIG, encoder_enabled=False)
    assert config.encoder_enabled is False


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"model_type": "autoencoder_v1"}, NotImplementedError),
        ({"use_wav_as_input": False}, NotImplementedError),
        ({"enc_type": "convnext"}, NotImplementedError),
        ({"dec_type": "hifigan"}, NotImplementedError),
        ({"bottleneck_type": "rvq"}, NotImplementedError),
        ({"bottleneck": {"type": "rvq"}}, NotImplementedError),
        ({"dec_use_snake": False}, NotImplementedError),
        ({"dec_final_tanh": True}, NotImplementedError),
        ({"dec_anti_aliasing": True}, NotImplementedError),
        ({"dec_use_nearest_upsample": True}, NotImplementedError),
        ({"dec_use_tanh_at_final": True}, NotImplementedError),
        ({"activation": "gelu"}, NotImplementedError),
        ({"snake_logscale": False}, NotImplementedError),
        ({"anti_aliasing": True}, NotImplementedError),
        ({"use_cuda_kernel": True}, NotImplementedError),
        ({"causal": True}, NotImplementedError),
        ({"padding_mode": "reflect"}, NotImplementedError),
        ({"latent_mean": 0.0}, NotImplementedError),
        ({"latent_std": [1.0, 2.0]}, NotImplementedError),
        ({"enc_c_mults": [1, 2]}, ValueError),
        ({"dec_c_mults": [1, 2]}, ValueError),
        ({"enc_latent_dim": 100}, ValueError),
        ({"enc_n_fft": 63}, ValueError),
        ({"enc_hop_length": 128}, ValueError),
        ({"input_channels": 0}, ValueError),
        ({"dec_dim": 0}, ValueError),
        ({"dec_out_channels": 1}, ValueError),
        ({"hop_size": 960}, ValueError),
        ({"enc_hop_length": 8}, ValueError),
    ],
)
def test_config_rejects_unsupported_variants(overrides, expected):
    """Every upstream ``__init__`` guard is reproduced by ``validate()``."""
    with pytest.raises(expected):
        Cosmos3AudioConfig.from_diffusers({**_REAL_CONFIG, **overrides})


def test_config_rejects_empty_and_non_positive_strides():
    """Sequence fields are validated eagerly in ``__post_init__``."""
    with pytest.raises(ValueError, match="at least one entry"):
        Cosmos3AudioConfig(dec_strides=())
    with pytest.raises(ValueError, match="positive integers"):
        Cosmos3AudioConfig(enc_strides=(4, 0, 6))


# ---------------------------------------------------------------------------
# Encoder-weight availability
# ---------------------------------------------------------------------------


def test_state_dict_has_encoder_detection():
    """Encoder presence is decided by the ``encoder.`` key prefix."""
    assert state_dict_has_encoder(["encoder.layers.0.weight_g", "decoder.conv1.bias"])
    assert not state_dict_has_encoder(["decoder.conv1.bias", "decoder.conv2.weight_g"])
    assert not state_dict_has_encoder([])


def test_with_encoder_from_state_dict_disables_for_decoder_only_weights():
    """A decoder-only checkpoint flips ``encoder_enabled`` off."""
    config = Cosmos3AudioConfig.from_diffusers(_REAL_CONFIG)
    assert config.encoder_enabled is True

    decoder_only = config.with_encoder_from_state_dict(["decoder.conv1.weight_g"])
    assert decoder_only.encoder_enabled is False
    assert decoder_only is not config
    # Everything else is preserved.
    assert decoder_only.resolved_hop_size == 1920
    assert decoder_only.dec_c_mults == (1, 2, 4, 8, 16)

    # Idempotent when the flag already matches.
    full = config.with_encoder_from_state_dict(["encoder.layers.0.weight_g"])
    assert full is config


def test_create_tokenizer_dispatches_on_encoder_enabled():
    """The factory returns the module variant matching the config flag."""
    full = create_cosmos3_avae_audio_tokenizer(_tiny_config())
    assert isinstance(full, Cosmos3AVAEAudioTokenizer)
    assert full.encoder_available is True
    assert hasattr(full, "encoder")
    assert hasattr(full, "bottleneck")

    decoder_only = create_cosmos3_avae_audio_tokenizer(_tiny_config(encoder_enabled=False))
    assert isinstance(decoder_only, Cosmos3AVAEAudioDecoderOnlyTokenizer)
    assert not isinstance(decoder_only, Cosmos3AVAEAudioTokenizer)
    assert decoder_only.encoder_available is False
    assert not hasattr(decoder_only, "encoder")
    assert not hasattr(decoder_only, "bottleneck")


def test_select_task_dispatches_on_encoder_enabled():
    """Task selection follows the same encoder-availability signal."""
    assert select_cosmos3_audio_task(_tiny_config()) is Cosmos3AVAEAudioTokenizerTask
    assert (
        select_cosmos3_audio_task(_tiny_config(encoder_enabled=False))
        is Cosmos3AVAEAudioDecoderTask
    )


def test_tokenizer_task_rejects_decoder_only_config():
    """Building the encoder path on decoder-only weights is a hard error."""
    config = _tiny_config()
    module = create_cosmos3_avae_audio_tokenizer(config)
    decoder_only_config = _tiny_config(encoder_enabled=False)
    with pytest.raises(ValueError, match="encoder_enabled=True"):
        Cosmos3AVAEAudioTokenizerTask().build(module, decoder_only_config)


def test_tokenizer_task_rejects_module_without_encoder():
    """The component spec catches a decoder-only module on the encoder path."""
    config = _tiny_config()
    module = Cosmos3AVAEAudioDecoderOnlyTokenizer(config)
    with pytest.raises(TypeError, match="encoder"):
        Cosmos3AVAEAudioTokenizerTask().build(module, config)


# ---------------------------------------------------------------------------
# Graph construction and I/O contracts
# ---------------------------------------------------------------------------


def test_decoder_only_package_has_no_encoder_initializers():
    """Decoder-only builds never emit an initializer no weight can fill."""
    config = _tiny_config(encoder_enabled=False)
    module = create_cosmos3_avae_audio_tokenizer(config)
    package = Cosmos3AVAEAudioDecoderTask().build(module, config)

    assert set(package) == {"decoder"}
    names = list(package["decoder"].graph.initializers)
    assert names, "decoder graph must have initializers"
    assert all(name.startswith("decoder.") for name in names)
    assert not any(name.startswith("encoder.") for name in names)
    assert not any("bottleneck" in name for name in names)


def test_decoder_graph_io_contract():
    """``latents -> waveform`` shapes, names and dtypes."""
    config = _tiny_config()
    module = create_cosmos3_avae_audio_tokenizer(config)
    model = Cosmos3AVAEAudioDecoderTask().build(module, config)["decoder"]

    (latents,) = model.graph.inputs
    assert latents.name == "latents"
    assert latents.dtype == config.dtype
    assert latents.shape[1] == config.latent_channels

    (waveform,) = model.graph.outputs
    assert waveform.name == "waveform"


def test_encoder_graph_io_contract():
    """``audio -> moments/latent_mean/latent_std`` names and shapes."""
    config = _tiny_config()
    module = create_cosmos3_avae_audio_tokenizer(config)
    package = Cosmos3AVAEAudioTokenizerTask().build(module, config)

    assert set(package) == {"encoder", "decoder"}
    encoder = package["encoder"]

    (audio,) = encoder.graph.inputs
    assert audio.name == "audio"
    assert audio.dtype == config.dtype
    assert audio.shape[1] == config.encoder_input_channels

    assert [out.name for out in encoder.graph.outputs] == [
        "moments",
        "latent_mean",
        "latent_std",
    ]

    encoder_names = list(encoder.graph.initializers)
    assert all(name.startswith("encoder.") for name in encoder_names)
    decoder_names = list(package["decoder"].graph.initializers)
    assert all(name.startswith("decoder.") for name in decoder_names)
    # The two graphs must not fight over the same weight names.
    assert not set(encoder_names) & set(decoder_names)


def test_encoder_graph_has_no_random_ops():
    """Posterior sampling stays outside ONNX so the graph is deterministic."""
    config = _tiny_config()
    module = create_cosmos3_avae_audio_tokenizer(config)
    encoder = Cosmos3AVAEAudioTokenizerTask().build(module, config)["encoder"]

    op_types = {node.op_type for node in encoder.graph}
    assert not {"RandomNormal", "RandomNormalLike", "RandomUniform", "Multinomial"} & op_types
    # The STFT front-end is part of the exported encoder.
    assert "STFT" in op_types
    assert "Softplus" in op_types


def test_real_config_initializer_names_match_checkpoint_layout():
    """Full-size initializer names reproduce the HF module paths exactly."""
    config = Cosmos3AudioConfig.from_diffusers(_REAL_CONFIG)
    module = create_cosmos3_avae_audio_tokenizer(config)
    package = Cosmos3AVAEAudioTokenizerTask().build(module, config)

    encoder = package["encoder"].graph.initializers
    decoder = package["decoder"].graph.initializers

    # Encoder: nn.Sequential indices, depthwise conv nested at dwconv.1.
    assert tuple(encoder["encoder.layers.0.weight"].shape) == (192, 132, 1)
    assert "encoder.layers.0.bias" not in encoder
    assert tuple(encoder["encoder.layers.1.dwconv.1.weight"].shape) == (192, 1, 7)
    assert tuple(encoder["encoder.layers.1.norm.weight"].shape) == (192,)
    assert "encoder.layers.1.norm.bias" not in encoder
    assert tuple(encoder["encoder.layers.1.act.alpha"].shape) == (1, 768, 1)
    assert tuple(encoder["encoder.layers.3.weight"].shape) == (384, 192, 8)
    assert tuple(encoder["encoder.layers.6.weight"].shape) == (768, 384, 10)
    assert tuple(encoder["encoder.layers.9.weight"].shape) == (768, 768, 12)
    assert tuple(encoder["encoder.layers.10.weight"].shape) == (128, 768, 1)
    assert "encoder.layers.10.bias" not in encoder

    # Decoder: conv1 / block.{i} / snake1 / conv2, ConvTranspose weight is (in, out, k).
    assert tuple(decoder["decoder.conv1.weight"].shape) == (5120, 64, 7)
    assert tuple(decoder["decoder.block.0.conv_t1.weight"].shape) == (5120, 2560, 16)
    assert tuple(decoder["decoder.block.0.res_unit2.conv1.weight"].shape) == (2560, 2560, 7)
    assert tuple(decoder["decoder.block.0.res_unit3.snake2.beta"].shape) == (1, 2560, 1)
    assert tuple(decoder["decoder.snake1.alpha"].shape) == (1, 320, 1)
    assert tuple(decoder["decoder.conv2.weight"].shape) == (2, 320, 7)
    assert "decoder.conv2.bias" not in decoder

    # Exactly the 207 tensors the published checkpoint holds after folding.
    assert len(encoder) + len(decoder) == 207


def test_published_config_cannot_reveal_encoder_presence():
    """The shipped config declares ``enc_*`` even for decoder-only checkpoints.

    ``nvidia/Cosmos3-Nano``, ``nvidia/Cosmos3-Super`` (full) and
    ``nvidia/Cosmos3-Super-Text2Image`` (decoder-only) ship byte-identical
    ``sound_tokenizer/config.json`` files. This pins the reason encoder
    presence must come from the checkpoint, not the config.
    """
    assert "encoder_enabled" not in _REAL_CONFIG
    assert _REAL_CONFIG["enc_type"] == "spec_convnext"
    assert _REAL_CONFIG["enc_dim"] == 192
    # Parsing alone therefore optimistically assumes a full checkpoint.
    assert Cosmos3AudioConfig.from_diffusers(_REAL_CONFIG).encoder_enabled is True


def test_from_diffusers_resolves_encoder_from_weight_names():
    """``weight_names=`` is the single-call, checkpoint-driven entry point."""
    full = Cosmos3AudioConfig.from_diffusers(
        _REAL_CONFIG, weight_names=["encoder.layers.0.weight_g", "decoder.conv1.weight_g"]
    )
    assert full.encoder_enabled is True

    decoder_only = Cosmos3AudioConfig.from_diffusers(
        _REAL_CONFIG, weight_names=["decoder.conv1.weight_g", "decoder.conv2.weight_v"]
    )
    assert decoder_only.encoder_enabled is False
    assert select_cosmos3_audio_task(decoder_only) is Cosmos3AVAEAudioDecoderTask


def test_from_diffusers_rejects_conflicting_encoder_signals():
    """``encoder_enabled`` and ``weight_names`` answer the same question."""
    with pytest.raises(ValueError, match="not both"):
        Cosmos3AudioConfig.from_diffusers(
            _REAL_CONFIG, encoder_enabled=True, weight_names=["decoder.conv1.weight_g"]
        )


def test_real_decoder_only_package_matches_published_layout():
    """A decoder-only build emits exactly the 145 folded ``decoder.*`` tensors.

    The published decoder-only checkpoint holds 182 tensors, 37 of which are
    ``weight_g``/``weight_v`` magnitude entries that fold away: 182 - 37 = 145.
    """
    config = Cosmos3AudioConfig.from_diffusers(
        _REAL_CONFIG, weight_names=["decoder.conv1.weight_g"]
    )
    package = Cosmos3AVAEAudioDecoderTask().build(
        create_cosmos3_avae_audio_tokenizer(config), config
    )

    assert set(package) == {"decoder"}
    names = list(package["decoder"].graph.initializers)
    assert len(names) == 145
    assert all(name.startswith("decoder.") for name in names)

    # The full build adds exactly the encoder half: 67 - 5 folded pairs = 62.
    full_config = Cosmos3AudioConfig.from_diffusers(_REAL_CONFIG)
    full = Cosmos3AVAEAudioTokenizerTask().build(
        create_cosmos3_avae_audio_tokenizer(full_config), full_config
    )
    assert len(full["encoder"].graph.initializers) == 62
    assert len(full["decoder"].graph.initializers) == 145


def test_graphs_contain_no_runtime_weight_norm_math():
    """``weight_norm`` is folded offline — the graph holds dense kernels only.

    A graph that reconstructed ``g * v / ||v||`` at runtime would need a
    norm (``ReduceL2``/``Pow`` + ``ReduceSum`` + ``Sqrt``) and a ``Div`` feeding
    each conv. None of those may appear.
    """
    weight_norm_ops = {"Div", "Sqrt", "ReduceL2", "ReduceSumSquare", "Pow", "ReduceSum"}

    config = Cosmos3AudioConfig.from_diffusers(_REAL_CONFIG)
    package = Cosmos3AVAEAudioTokenizerTask().build(
        create_cosmos3_avae_audio_tokenizer(config), config
    )

    decoder_ops = [node.op_type for node in package["decoder"].graph]
    assert not weight_norm_ops & set(decoder_ops)
    # 37 weight-normed convs in the decoder — matching the checkpoint's 37 pairs.
    assert decoder_ops.count("Conv") + decoder_ops.count("ConvTranspose") == 37

    # The encoder's only Div is the peak-volume normalization; disable it and
    # the encoder must likewise be free of every weight-norm op.
    plain = Cosmos3AudioConfig.from_diffusers({**_REAL_CONFIG, "normalize_volume": False})
    plain_encoder = Cosmos3AVAEAudioTokenizerTask().build(
        create_cosmos3_avae_audio_tokenizer(plain), plain
    )["encoder"]
    assert not weight_norm_ops & {node.op_type for node in plain_encoder.graph}

    encoder_ops = [node.op_type for node in package["encoder"].graph]
    assert encoder_ops.count("Div") == 1
    assert encoder_ops.count("ReduceMax") == 1


def test_default_dtype_is_float32_like_the_checkpoints():
    """Published sound-tokenizer weights are FP32, so that is the build default."""
    config = Cosmos3AudioConfig.from_diffusers(_REAL_CONFIG)
    assert config.dtype == ir.DataType.FLOAT


def test_config_dtype_propagates_to_graph_and_initializers():
    """``config.dtype`` reaches graph I/O, every initializer, and the FP32 norm."""
    config = _tiny_config(dtype=ir.DataType.FLOAT16)
    module = create_cosmos3_avae_audio_tokenizer(config)
    package = Cosmos3AVAEAudioTokenizerTask().build(module, config)

    for model in package.values():
        assert all(inp.dtype == ir.DataType.FLOAT16 for inp in model.graph.inputs)
        assert all(
            init.dtype == ir.DataType.FLOAT16 for init in model.graph.initializers.values()
        ), "every parameter must adopt the configured dtype"

    # FP32LayerNorm upcasts around the ConvNeXt norm when the graph is not fp32.
    encoder_ops = [node.op_type for node in package["encoder"].graph]
    assert "Cast" in encoder_ops
    assert "LayerNormalization" in encoder_ops

    # A float32 build needs no cast around the norm.
    fp32 = _tiny_config()
    fp32_encoder = Cosmos3AVAEAudioTokenizerTask().build(
        create_cosmos3_avae_audio_tokenizer(fp32), fp32
    )["encoder"]
    assert all(
        init.dtype == ir.DataType.FLOAT for init in fp32_encoder.graph.initializers.values()
    )


# ---------------------------------------------------------------------------
# weight_norm folding
# ---------------------------------------------------------------------------


def test_fold_weight_norm_matches_torch_parametrization():
    """Folding g/v reproduces the effective weight torch computes."""
    torch.manual_seed(0)
    conv = torch.nn.utils.parametrizations.weight_norm(tnn.Conv1d(5, 7, 3))
    transposed = torch.nn.utils.parametrizations.weight_norm(tnn.ConvTranspose1d(5, 7, 4))
    with torch.no_grad():
        for module in (conv, transposed):
            for param in module.parameters():
                param.copy_(torch.randn(param.shape))

    state_dict = {
        **{f"conv.{k}": v for k, v in conv.state_dict().items()},
        **{f"conv_t.{k}": v for k, v in transposed.state_dict().items()},
    }
    folded = fold_weight_norm(state_dict)

    assert set(folded) == {"conv.weight", "conv.bias", "conv_t.weight", "conv_t.bias"}
    torch.testing.assert_close(folded["conv.weight"], conv.weight, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(
        folded["conv_t.weight"], transposed.weight, atol=1e-6, rtol=1e-6
    )


def test_fold_weight_norm_handles_legacy_key_names():
    """Legacy ``weight_g``/``weight_v`` keys (the shipped format) also fold."""
    direction = torch.randn(4, 3, 5)
    magnitude = torch.randn(4, 1, 1)
    folded = fold_weight_norm(
        {
            "layers.0.weight_g": magnitude,
            "layers.0.weight_v": direction,
            "layers.0.bias": torch.zeros(4),
        }
    )
    expected = magnitude * direction / direction.pow(2).sum(dim=(1, 2), keepdim=True).sqrt()
    assert set(folded) == {"layers.0.weight", "layers.0.bias"}
    torch.testing.assert_close(folded["layers.0.weight"], expected, atol=1e-6, rtol=1e-6)


def test_fold_weight_norm_requires_matching_direction():
    """A dangling magnitude tensor is an error, not a silent drop."""
    with pytest.raises(ValueError, match="no matching direction"):
        fold_weight_norm({"conv.weight_g": torch.ones(2, 1, 1)})


def test_fold_weight_norm_reconstructs_real_checkpoint_kernels():
    """Folding real checkpoint-shaped ``weight_g``/``weight_v`` pairs is exact.

    Uses the published shapes for the first/last weight-normed conv of each
    half: ``encoder.layers.0`` (132 packed STFT channels in) and
    ``decoder.block.0.conv_t1`` (a ConvTranspose, whose weight-norm axis 0 is
    *input* channels, not output).
    """
    torch.manual_seed(0)
    cases = {
        # (out, in, k) for Conv1d; weight_norm dim=0 -> g is (out, 1, 1)
        "encoder.layers.0": ((192, 132, 1), (192, 1, 1)),
        "encoder.layers.10": ((128, 768, 1), (128, 1, 1)),
        # (in, out, k) for ConvTranspose1d; weight_norm dim=0 -> g is (in, 1, 1)
        "decoder.block.0.conv_t1": ((5120, 2560, 16), (5120, 1, 1)),
        "decoder.conv2": ((2, 320, 7), (2, 1, 1)),
    }
    state_dict: dict[str, torch.Tensor] = {}
    for prefix, (v_shape, g_shape) in cases.items():
        state_dict[f"{prefix}.weight_v"] = torch.randn(v_shape) * 0.05
        state_dict[f"{prefix}.weight_g"] = torch.randn(g_shape).abs() + 0.1

    folded = fold_weight_norm(state_dict)
    assert set(folded) == {f"{prefix}.weight" for prefix in cases}

    for prefix, (v_shape, _) in cases.items():
        direction = state_dict[f"{prefix}.weight_v"]
        magnitude = state_dict[f"{prefix}.weight_g"]
        expected = torch._weight_norm(direction, magnitude, 0)
        assert folded[f"{prefix}.weight"].shape == torch.Size(v_shape)
        torch.testing.assert_close(folded[f"{prefix}.weight"], expected, atol=1e-6, rtol=1e-6)


def test_fold_weight_norm_leaves_bias_free_convs_bias_free():
    """``decoder.conv2`` and ``encoder.layers.10`` ship without a bias."""
    config = Cosmos3AudioConfig.from_diffusers(_REAL_CONFIG)
    package = Cosmos3AVAEAudioTokenizerTask().build(
        create_cosmos3_avae_audio_tokenizer(config), config
    )
    assert "decoder.conv2.bias" not in package["decoder"].graph.initializers
    assert "encoder.layers.10.bias" not in package["encoder"].graph.initializers
    assert "encoder.layers.0.bias" not in package["encoder"].graph.initializers
    # ...while the strided encoder convs and decoder convs do carry one.
    assert "encoder.layers.3.bias" in package["encoder"].graph.initializers
    assert "decoder.conv1.bias" in package["decoder"].graph.initializers


def test_preprocess_weights_is_exposed_on_both_module_variants():
    """Both module paths expose the builder's ``preprocess_weights`` hook."""
    direction = torch.randn(2, 2, 3)
    magnitude = torch.randn(2, 1, 1)
    payload = {"decoder.conv1.weight_g": magnitude, "decoder.conv1.weight_v": direction}
    for module in (
        create_cosmos3_avae_audio_tokenizer(_tiny_config()),
        create_cosmos3_avae_audio_tokenizer(_tiny_config(encoder_enabled=False)),
    ):
        assert set(module.preprocess_weights(payload)) == {"decoder.conv1.weight"}


# ---------------------------------------------------------------------------
# Numerical parity against the PyTorch reference
# ---------------------------------------------------------------------------


def test_snake_beta_matches_pytorch():
    """SnakeBeta (log-scaled) matches ``Snake1d`` elementwise."""
    channels = 6
    reference = _RefSnake1d(channels)
    state_dict = {f"snake.{k}": v for k, v in _randomize(reference, seed=7).items()}

    model = _wrap_module_graph(_Snake1d(channels), "snake", channels)
    session = _session(model, state_dict, None)

    sample = np.random.default_rng(3).standard_normal((2, channels, 9)).astype(np.float32)
    got = session.run({"x": sample})["y"]
    expected = reference(torch.from_numpy(sample)).detach().numpy()
    np.testing.assert_allclose(got, expected, atol=1e-5, rtol=1e-5)


def test_snake_beta_without_logscale_matches_pytorch():
    """``snake_logscale=False`` uses the raw alpha/beta parameters."""
    channels = 5
    reference = _RefSnake1d(channels, logscale=False)
    _randomize(reference, seed=61)
    with torch.no_grad():
        # Keep beta away from -1e-9 so the reciprocal stays well conditioned.
        reference.beta.copy_(reference.beta.abs() + 0.5)
    state_dict = {f"snake.{k}": v.detach().clone() for k, v in reference.state_dict().items()}

    model = _wrap_module_graph(_Snake1d(channels, logscale=False), "snake", channels)
    session = _session(model, state_dict, None)

    sample = np.random.default_rng(67).standard_normal((1, channels, 8)).astype(np.float32)
    got = session.run({"x": sample})["y"]
    expected = reference(torch.from_numpy(sample)).detach().numpy()
    np.testing.assert_allclose(got, expected, atol=1e-5, rtol=1e-5)


def test_residual_unit_matches_pytorch():
    """One dilated residual unit matches the reference bit-for-bit (fp32 tol)."""
    dim, dilation = 5, 3
    reference = _RefResidualUnit(dim, dilation=dilation)
    state_dict = {f"unit.{k}": v for k, v in _randomize(reference, seed=11).items()}

    model = _wrap_module_graph(Cosmos3AudioResidualUnit(dim, dilation=dilation), "unit", dim)
    session = _session(model, state_dict, None)

    sample = np.random.default_rng(5).standard_normal((2, dim, 16)).astype(np.float32)
    got = session.run({"x": sample})["y"]
    expected = reference(torch.from_numpy(sample)).detach().numpy()

    assert got.shape == sample.shape
    np.testing.assert_allclose(got, expected, atol=1e-4, rtol=1e-4)


def test_convnext_block_matches_pytorch():
    """The SnakeBeta ConvNeXt block (depthwise conv + FP32 LayerNorm) matches."""
    hidden, intermediate = 4, 16
    reference = _RefConvNeXtBlock(hidden, intermediate)
    state_dict = {f"block.{k}": v for k, v in _randomize(reference, seed=71).items()}

    model = _wrap_module_graph(
        Cosmos3AudioConvNeXtBlock(hidden, intermediate), "block", hidden
    )
    session = _session(model, state_dict, None)

    sample = np.random.default_rng(73).standard_normal((2, hidden, 12)).astype(np.float32)
    got = session.run({"x": sample})["y"]
    expected = reference(torch.from_numpy(sample)).detach().numpy()
    np.testing.assert_allclose(got, expected, atol=1e-4, rtol=1e-4)


def test_decoder_graph_matches_pytorch():
    """The whole tiny decoder matches the reference, including the clamp."""
    config = _tiny_config()
    reference = _RefDecoder(config)
    state_dict = {f"decoder.{k}": v for k, v in _randomize(reference, seed=13).items()}

    module = create_cosmos3_avae_audio_tokenizer(config)
    model = Cosmos3AVAEAudioDecoderTask().build(module, config)["decoder"]
    session = _session(model, state_dict, config)

    frames = 5
    latents = (
        np.random.default_rng(17)
        .standard_normal((2, config.latent_channels, frames))
        .astype(np.float32)
    )
    got = session.run({"latents": latents})["waveform"]
    expected = reference(torch.from_numpy(latents)).clamp(-1.0, 1.0).detach().numpy()

    assert got.shape == (2, config.audio_channels, frames * config.resolved_hop_size)
    np.testing.assert_allclose(got, expected, atol=1e-4, rtol=1e-4)


def test_encoder_graph_matches_pytorch():
    """The whole tiny encoder — STFT front-end included — matches the reference."""
    config = _tiny_config(normalize_volume=False)
    reference = _RefEncoder(config)
    state_dict = {f"encoder.{k}": v for k, v in _randomize(reference, seed=19).items()}

    module = create_cosmos3_avae_audio_tokenizer(config)
    model = Cosmos3AVAEAudioTokenizerTask().build(module, config)["encoder"]
    session = _session(model, state_dict, config)

    num_samples = config.resolved_hop_size * 4
    audio = (
        np.random.default_rng(23)
        .standard_normal((2, config.encoder_input_channels, num_samples))
        .astype(np.float32)
    )
    outputs = session.run({"audio": audio})

    # Upstream `_encode` transposes the channels-last encoder output back.
    expected_moments = reference(torch.from_numpy(audio)).transpose(1, 2)
    expected_mean, expected_scale = expected_moments.chunk(2, dim=1)
    expected_std = torch.nn.functional.softplus(expected_scale) + 1e-4

    frames = num_samples // config.resolved_hop_size
    assert outputs["moments"].shape == (2, config.moments_channels, frames)
    assert outputs["latent_mean"].shape == (2, config.latent_channels, frames)
    assert outputs["latent_std"].shape == (2, config.latent_channels, frames)

    np.testing.assert_allclose(
        outputs["moments"], expected_moments.detach().numpy(), atol=1e-4, rtol=1e-4
    )
    np.testing.assert_allclose(
        outputs["latent_mean"], expected_mean.detach().numpy(), atol=1e-4, rtol=1e-4
    )
    np.testing.assert_allclose(
        outputs["latent_std"], expected_std.detach().numpy(), atol=1e-4, rtol=1e-4
    )


def test_encoder_graph_normalizes_volume_and_pads_to_hop_size():
    """Peak normalization and hop padding are applied inside the graph."""
    config = _tiny_config()
    reference = _RefEncoder(config)
    state_dict = {f"encoder.{k}": v for k, v in _randomize(reference, seed=29).items()}

    module = create_cosmos3_avae_audio_tokenizer(config)
    model = Cosmos3AVAEAudioTokenizerTask().build(module, config)["encoder"]
    session = _session(model, state_dict, config)

    hop = config.resolved_hop_size
    # Deliberately not a multiple of hop_size — the graph must right-pad.
    num_samples = 3 * hop + 5
    audio = (
        np.random.default_rng(31)
        .standard_normal((1, config.encoder_input_channels, num_samples))
        .astype(np.float32)
        * 4.0
    )

    torch_audio = torch.from_numpy(audio)
    normalized = torch_audio / (torch_audio.abs().max() + 1e-5) * 0.95
    padding = (hop - (num_samples % hop)) % hop
    padded = torch.nn.functional.pad(normalized, (0, padding))
    expected = reference(padded).transpose(1, 2)

    outputs = session.run({"audio": audio})
    assert outputs["moments"].shape == (1, config.moments_channels, 4)
    np.testing.assert_allclose(
        outputs["moments"], expected.detach().numpy(), atol=1e-4, rtol=1e-4
    )


def test_decoder_graph_handles_mono_and_variable_lengths():
    """A mono (non-stereo) config still round-trips through the decoder graph."""
    config = _tiny_config(stereo=False, dec_out_channels=1)
    reference = _RefDecoder(config)
    state_dict = {f"decoder.{k}": v for k, v in _randomize(reference, seed=37).items()}

    module = create_cosmos3_avae_audio_tokenizer(config)
    model = Cosmos3AVAEAudioDecoderTask().build(module, config)["decoder"]
    session = _session(model, state_dict, config)

    for frames in (1, 7):
        latents = (
            np.random.default_rng(41 + frames)
            .standard_normal((1, config.latent_channels, frames))
            .astype(np.float32)
        )
        got = session.run({"latents": latents})["waveform"]
        expected = reference(torch.from_numpy(latents)).clamp(-1.0, 1.0).detach().numpy()
        assert got.shape == (1, 1, frames * config.resolved_hop_size)
        np.testing.assert_allclose(got, expected, atol=1e-4, rtol=1e-4)


def test_mono_encoder_graph_matches_pytorch():
    """The single-channel STFT path (no channel folding) also matches."""
    config = _tiny_config(stereo=False, dec_out_channels=1, normalize_volume=False)
    reference = _RefEncoder(config)
    state_dict = {f"encoder.{k}": v for k, v in _randomize(reference, seed=43).items()}

    module = create_cosmos3_avae_audio_tokenizer(config)
    model = Cosmos3AVAEAudioTokenizerTask().build(module, config)["encoder"]
    session = _session(model, state_dict, config)

    num_samples = config.resolved_hop_size * 3
    audio = np.random.default_rng(47).standard_normal((2, 1, num_samples)).astype(np.float32)
    got = session.run({"audio": audio})["moments"]
    expected = reference(torch.from_numpy(audio)).transpose(1, 2).detach().numpy()

    assert got.shape == (2, config.moments_channels, 3)
    np.testing.assert_allclose(got, expected, atol=1e-4, rtol=1e-4)


def test_gelu_convnext_variant_matches_pytorch():
    """``enc_use_snake=False`` swaps SnakeBeta for exact GELU."""

    class _RefGeluBlock(_RefConvNeXtBlock):
        def __init__(self, hidden_dim: int, intermediate_dim: int):
            super().__init__(hidden_dim, intermediate_dim)
            self.act = tnn.GELU()

    hidden, intermediate = 4, 16
    reference = _RefGeluBlock(hidden, intermediate)
    state_dict = {f"block.{k}": v for k, v in _randomize(reference, seed=53).items()}

    model = _wrap_module_graph(
        Cosmos3AudioConvNeXtBlock(hidden, intermediate, use_snake=False), "block", hidden
    )
    session = _session(model, state_dict, None)

    sample = np.random.default_rng(59).standard_normal((2, hidden, 12)).astype(np.float32)
    got = session.run({"x": sample})["y"]
    expected = reference(torch.from_numpy(sample)).detach().numpy()
    np.testing.assert_allclose(got, expected, atol=1e-4, rtol=1e-4)
