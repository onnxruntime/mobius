# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Cosmos3 AVAE audio tokenizer tasks (``sound_tokenizer``).

Two explicit build paths, because AVAE checkpoints are shipped in two shapes:

* :class:`Cosmos3AVAEAudioDecoderTask` — decoder-only weights. Produces a
  single ``"decoder"`` graph. Nothing encoder-related is instantiated, so the
  package can never contain an initializer without weight data.
* :class:`Cosmos3AVAEAudioTokenizerTask` — encoder + decoder weights. Produces
  ``"encoder"`` and ``"decoder"`` graphs.

Use :func:`select_cosmos3_audio_task` to pick the right one from a config whose
``encoder_enabled`` flag was resolved from the checkpoint. Resolve it with
``Cosmos3AudioConfig.from_diffusers(config_json, weight_names=state_dict)`` —
the published configs are byte-identical for full and decoder-only releases, so
parsing the config alone always assumes an encoder is present::

    nvidia/Cosmos3-Nano             -> Cosmos3AVAEAudioTokenizerTask  (encoder+decoder)
    nvidia/Cosmos3-Super            -> Cosmos3AVAEAudioTokenizerTask  (encoder+decoder)
    nvidia/Cosmos3-Super-Text2Image -> Cosmos3AVAEAudioDecoderTask    (decoder-only)

Graph contracts
---------------

``decoder`` — latent → waveform::

    inputs   latents  [batch, vocoder_input_dim, latent_frames]   config.dtype
    outputs  waveform [batch, dec_out_channels, latent_frames * hop_size]

    latents are consumed as-is: no latent mean/std de-normalization is applied,
    matching upstream (which rejects non-null latent_mean/latent_std). The
    waveform is clamped to [-1, 1].

``encoder`` — waveform → posterior moments::

    inputs   audio       [batch, encoder_input_channels, num_samples]  config.dtype
    outputs  moments     [batch, 2 * vocoder_input_dim, latent_frames]
             latent_mean [batch, vocoder_input_dim, latent_frames]
             latent_std  [batch, vocoder_input_dim, latent_frames]

    ``latent_frames = ceil(num_samples / hop_size)``.

    Normalization semantics baked into the graph, in order:

    1. peak volume normalization ``x / (|x|.max() + 1e-5) * 0.95`` when
       ``config.normalize_volume`` is set. The maximum is global across the
       whole input tensor (batch included), exactly as upstream — use batch
       size 1 for per-clip normalization.
    2. right zero-padding to a multiple of ``hop_size``.
    3. VAE bottleneck: ``mean, scale = split(moments, 2, axis=1)`` and
       ``std = softplus(scale) + 1e-4``.

    Sampling is intentionally left outside the graph so the ONNX model is
    deterministic; draw ``z = mean + std * eps`` in the caller, or use ``mean``
    for the distribution mode.

Preprocessing boundary
----------------------

The STFT front-end is part of ``encoder.forward`` upstream, so it is emitted
inside the ONNX graph (as an ``STFT`` node) and the encoder contract stays
``waveform -> moments``. onnxruntime implements ``STFT`` on CPU only, so under
the CUDA EP that single node falls back to the CPU EP and adds a host/device
copy at the graph entry. Everything after it — the whole ConvNeXt stack — runs
on the accelerator. Deployments that need a pure-GPU encoder should call
:meth:`~mobius.models.cosmos3_audio.Cosmos3AudioSpectrogramConvNeXtEncoder.spectrogram`
out of band and feed ``encoder.layers`` directly.

Latent normalization (``latent_mean``/``latent_std``) is *not* applied: the
published configs leave both ``null`` and upstream rejects any other value, so
:meth:`Cosmos3AudioConfig.validate` raises rather than invent semantics.

Weight routing
--------------

Both graphs keep the HuggingFace module paths, so initializer names are already
disjoint (``encoder.*`` vs ``decoder.*``). Do **not** set a ``weight_prefix_map``
on the module: the default "try every weight against every component" routing in
:meth:`ModelPackage.apply_weights` is correct here, and stripping the prefix
would break the match. Run
:meth:`~mobius.models.cosmos3_audio.Cosmos3AVAEAudioDecoderOnlyTokenizer.preprocess_weights`
first to fold ``weight_g``/``weight_v`` into ``weight``.
"""

from __future__ import annotations

from typing import ClassVar

import onnx_ir as ir

from mobius._configs._cosmos3_audio import Cosmos3AudioConfig
from mobius._model_package import ModelPackage
from mobius.models.cosmos3_audio import (
    Cosmos3AVAEAudioDecoderOnlyTokenizer,
    Cosmos3AVAEAudioTokenizer,
)
from mobius.tasks._base import ComponentSpec, ModelTask, _make_graph, _make_model

__all__ = [
    "Cosmos3AVAEAudioDecoderTask",
    "Cosmos3AVAEAudioTokenizerTask",
    "select_cosmos3_audio_task",
]


class Cosmos3AVAEAudioDecoderTask(ModelTask):
    """Build the ``latents -> waveform`` graph of a Cosmos3 AVAE sound tokenizer.

    This is the decoder-only path used for checkpoints that ship without
    ``encoder.*`` weights.
    """

    model_roles: ClassVar[dict[str, str]] = {"decoder": "decoder"}
    components: ClassVar[ComponentSpec] = ComponentSpec(decoder="decoder")

    def build(
        self,
        module: Cosmos3AVAEAudioDecoderOnlyTokenizer,
        config: Cosmos3AudioConfig,
    ) -> ModelPackage:
        """Build a package containing only the ``"decoder"`` model."""
        self._validate_components(module)
        config.validate()
        return ModelPackage({"decoder": self._build_decoder(module, config)}, config=config)

    def _build_decoder(
        self,
        module: Cosmos3AVAEAudioDecoderOnlyTokenizer,
        config: Cosmos3AudioConfig,
    ) -> ir.Model:
        """Wire ``decode()`` into a graph: latents → clamped waveform."""
        batch = ir.SymbolicDim("batch")
        latent_frames = ir.SymbolicDim("latent_frames")

        graph, builder = _make_graph(name="cosmos3_audio_decoder")
        latents = builder.input(
            "latents",
            dtype=config.dtype,
            shape=[batch, config.latent_channels, latent_frames],
        )

        # (B, z, T) -> (B, audio_channels, T * hop_size)
        waveform = module.decode(builder.op, latents)

        builder.add_output(waveform, "waveform")
        return _make_model(graph)


class Cosmos3AVAEAudioTokenizerTask(Cosmos3AVAEAudioDecoderTask):
    """Build both the encoder and decoder graphs of a Cosmos3 AVAE tokenizer.

    Requires a module with encoder weights available
    (:class:`~mobius.models.cosmos3_audio.Cosmos3AVAEAudioTokenizer`).
    """

    model_roles: ClassVar[dict[str, str]] = {"encoder": "encoder", "decoder": "decoder"}
    components: ClassVar[ComponentSpec] = ComponentSpec(
        encoder="encoder", decoder="decoder", bottleneck="bottleneck"
    )

    def build(
        self,
        module: Cosmos3AVAEAudioTokenizer,
        config: Cosmos3AudioConfig,
    ) -> ModelPackage:
        """Build a package containing the ``"encoder"`` and ``"decoder"`` models."""
        self._validate_components(module)
        config.validate()
        if not config.encoder_enabled:
            raise ValueError(
                "Cosmos3AVAEAudioTokenizerTask requires config.encoder_enabled=True. "
                "Decoder-only AVAE checkpoints must use Cosmos3AVAEAudioDecoderTask so no "
                "encoder initializer is created without weight data."
            )
        return ModelPackage(
            {
                "encoder": self._build_encoder(module, config),
                "decoder": self._build_decoder(module, config),
            },
            config=config,
        )

    def _build_encoder(
        self,
        module: Cosmos3AVAEAudioTokenizer,
        config: Cosmos3AudioConfig,
    ) -> ir.Model:
        """Wire ``encode()`` into a graph: waveform → deterministic moments."""
        batch = ir.SymbolicDim("batch")
        num_samples = ir.SymbolicDim("num_samples")

        graph, builder = _make_graph(name="cosmos3_audio_encoder")
        audio = builder.input(
            "audio",
            dtype=config.dtype,
            shape=[batch, config.encoder_input_channels, num_samples],
        )

        # (B, C, N) -> moments (B, 2z, T) and the split (mean, std) pair.
        moments, mean, std = module.encode(builder.op, audio)

        builder.add_output(moments, "moments")
        builder.add_output(mean, "latent_mean")
        builder.add_output(std, "latent_std")
        return _make_model(graph)


def select_cosmos3_audio_task(config: Cosmos3AudioConfig) -> type[Cosmos3AVAEAudioDecoderTask]:
    """Return the task class matching ``config.encoder_enabled``.

    Args:
        config: A Cosmos3 AVAE audio config whose ``encoder_enabled`` flag has
            already been reconciled with the checkpoint (see
            :meth:`Cosmos3AudioConfig.with_encoder_from_state_dict`).

    Returns:
        :class:`Cosmos3AVAEAudioTokenizerTask` when an encoder is present,
        otherwise :class:`Cosmos3AVAEAudioDecoderTask`.
    """
    if config.encoder_enabled:
        return Cosmos3AVAEAudioTokenizerTask
    return Cosmos3AVAEAudioDecoderTask
