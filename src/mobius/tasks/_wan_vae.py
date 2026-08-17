# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Wan video VAE task: 5D encoder and decoder ONNX graphs.

Builds a :class:`~mobius._model_package.ModelPackage` with two graphs:

``encoder``
    ``sample`` ``(B, video_channels, frames, height, width)``
    -> ``latent_mean`` / ``latent_logvar`` ``(B, z_dim, T', H', W')``
    -> ``latent`` — the deterministic, pipeline-normalised latent
    ``(latent_mean - latents_mean) / latents_std``.

``decoder``
    ``latent`` ``(B, z_dim, T', H', W')`` (pipeline-normalised)
    -> ``sample`` ``(B, video_channels, frames, height, width)``.

Both graphs are fully dynamic in batch, temporal and spatial extent.  The
temporal extent must satisfy ``frames = scale_factor_temporal * k + 1`` with
``k >= 0`` (encoder) and ``latent_frames = k + 1 >= 1`` (decoder); ``k == 0``
is the single-frame image mode used by text-to-image pipelines such as Cosmos3.
See the "Single-frame (image) mode" note in :mod:`mobius.models.wan_vae`.
``height``/``width`` must be divisible by ``scale_factor_spatial``.

Why a dedicated task instead of reusing :class:`~mobius.tasks._vae.VAETask` or
:class:`~mobius.tasks._qwen_image_vae.QwenImageVAETask`:

* ``VAETask`` declares 4D image I/O (``batch, C, height, width``) and reads
  ``config.latent_channels``; the Wan VAE is inherently 5D video.
* Both existing tasks emit a single fused ``latent_dist`` tensor and perform no
  posterior split, so they cannot expose deterministic mean/logvar moments.
* Neither models the ``WanPipeline`` latent mean/std normalisation, which has to
  straddle the graph boundary (see below).
* Neither applies ``patchify``/``unpatchify``, which upstream performs inside
  ``AutoencoderKLWan._encode``/``._decode`` rather than inside the encoder or
  decoder sub-module.

**Where the latent statistics live.**  Upstream keeps ``latents_mean`` /
``latents_std`` out of ``AutoencoderKLWan`` entirely — the ``WanPipeline``
normalises after ``vae.encode`` and denormalises before ``vae.decode``.  Mobius
mirrors that split: the ``encoder``/``decoder`` ``nn.Module``s stay byte-identical
to the checkpoint, and the normalisation is emitted here, at the graph boundary,
via :meth:`~mobius.models.wan_vae.AutoencoderKLWanModel.normalize_latents` and
:meth:`~mobius.models.wan_vae.AutoencoderKLWanModel.denormalize_latents`.  The
raw ``latent_mean``/``latent_logvar`` moments are exported alongside the
normalised latent so callers that need to sample the posterior (or reuse the
un-normalised scale) can do so outside the graph.
"""

from __future__ import annotations

from typing import ClassVar

import onnx_ir as ir

from mobius._configs._wan_vae import WanVAEConfig
from mobius._model_package import ModelPackage
from mobius.tasks._base import ComponentSpec, ModelTask, _make_graph, _make_model


class WanVAETask(ModelTask):
    """Build the Wan 3D causal video VAE encoder and decoder graphs."""

    model_roles: ClassVar[dict[str, str]] = {"encoder": "encoder", "decoder": "decoder"}
    components: ClassVar[ComponentSpec] = ComponentSpec(encoder="encoder", decoder="decoder")

    def build(self, module, config: WanVAEConfig) -> ModelPackage:
        """Build both graphs.

        Args:
            module: An :class:`~mobius.models.wan_vae.AutoencoderKLWanModel`.
            config: The parsed Wan VAE configuration.

        Returns:
            A package with ``"encoder"`` and ``"decoder"`` models.
        """
        self._validate_components(module)
        return ModelPackage(
            {
                "encoder": self._build_encoder_graph(module, config),
                "decoder": self._build_decoder_graph(module, config),
            },
            config=config,
        )

    def _build_encoder_graph(self, module, config: WanVAEConfig) -> ir.Model:
        graph, builder = _make_graph(name="wan_vae_encoder")
        op = builder.op

        # Pixel-space video in [-1, 1]. ``frames`` must be
        # ``scale_factor_temporal * k + 1`` (k >= 0) so the temporal downsampling
        # stages line up with upstream's chunked encode; k == 0 is a single image.
        sample = builder.input(
            "sample",
            dtype=config.dtype,
            shape=["batch", config.video_channels, "frames", "height", "width"],
        )

        latent_mean, latent_logvar = module.encode(op, sample)
        latent = module.normalize_latents(op, latent_mean)

        # Annotate the latent grid symbolically: shape inference cannot see
        # through the dynamic Reshape/Pad chain of the resampling blocks.
        latent_shape = ir.Shape(
            ["batch", config.z_dim, "latent_frames", "latent_height", "latent_width"]
        )
        for value, name in (
            (latent_mean, "latent_mean"),
            (latent_logvar, "latent_logvar"),
            (latent, "latent"),
        ):
            value.shape = latent_shape
            builder.add_output(value, name)

        return _make_model(graph)

    def _build_decoder_graph(self, module, config: WanVAEConfig) -> ir.Model:
        graph, builder = _make_graph(name="wan_vae_decoder")
        op = builder.op

        # Pipeline-normalised latents, i.e. the ``latent`` output of the encoder
        # graph or the denoiser's sample. ``latent_frames`` may be 1 (image mode).
        latent = builder.input(
            "latent",
            dtype=config.dtype,
            shape=["batch", config.z_dim, "latent_frames", "latent_height", "latent_width"],
        )

        sample = module.decode(op, module.denormalize_latents(op, latent))

        sample.shape = ir.Shape(
            ["batch", config.decoded_video_channels, "frames", "height", "width"]
        )
        builder.add_output(sample, "sample")

        return _make_model(graph)
