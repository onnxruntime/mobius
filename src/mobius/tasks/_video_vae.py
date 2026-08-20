# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Video VAE decode task for 3D causal autoencoders.

Builds a decoder graph whose latent input and frame output are rank-5 so the
temporal axis stays explicit end to end:

- ``latent_sample``: ``[batch, latent_channels, latent_frames, height, width]``
- ``sample``: ``[batch, out_channels, frames, out_height, out_width]``

The frame count is a free dimension: a causal video decoder expands
``latent_frames`` by the temporal compression ratio, so nothing about the graph
may assume a single frame or a fixed clip length.

The decoder additionally exposes the reference implementation's ``conv_cache``
as paired ``conv_cache.<path>`` inputs and ``conv_cache_out.<path>`` outputs.
Long clips are decoded a few latent frames at a time, and those tensors are the
only state that crosses a chunk boundary; a caller that decodes a whole clip in
one call passes zero-length caches and drops the outputs.
"""

from __future__ import annotations

from typing import ClassVar

import onnx_ir as ir

from mobius._model_package import ModelPackage
from mobius.models.cogvideox_vae import CogVideoXVAEConfig
from mobius.tasks._base import ModelTask, _make_graph, _make_model

CONV_CACHE_INPUT_PREFIX = "conv_cache."
CONV_CACHE_SCALE_METADATA = "mobius.conv_cache.spatial_scale."
CONV_CACHE_OUTPUT_PREFIX = "conv_cache_out."


class VideoVAETask(ModelTask):
    """Build the decode graph of a 3D causal video autoencoder."""

    model_roles: ClassVar[dict[str, str]] = {"decoder": "decoder"}

    def build(
        self,
        module,
        config: CogVideoXVAEConfig,
    ) -> ModelPackage:
        graph, builder = _make_graph(name="video_vae_decoder")
        op = builder.op

        latent_sample = builder.input(
            "latent_sample",
            dtype=ir.DataType.FLOAT,
            shape=[
                "batch",
                config.latent_channels,
                "latent_frames",
                "latent_height",
                "latent_width",
            ],
        )

        conv_cache = {}
        for entry in module.conv_cache_spec():
            scale = entry.spatial_scale
            height = "latent_height" if scale == 1 else f"{scale}*latent_height"
            width = "latent_width" if scale == 1 else f"{scale}*latent_width"
            conv_cache[entry.name] = builder.input(
                f"{CONV_CACHE_INPUT_PREFIX}{entry.name}",
                dtype=ir.DataType.FLOAT,
                shape=["batch", entry.channels, "cache_frames", height, width],
            )

        sample, updated_cache = module(
            op, latent_sample=latent_sample, conv_cache=conv_cache
        )
        # The temporal and spatial extents are recovered from Shape ops inside
        # the decoder, which mints anonymous symbolic dimensions. Name them so
        # the published contract states the compression ratios instead of
        # leaking solver-internal dimension identities.
        spatial = 2**(len(config.block_out_channels) - 1)
        sample.shape = ir.Shape(
            [
                "batch",
                config.out_channels,
                "frames",
                f"{spatial}*latent_height" if spatial != 1 else "latent_height",
                f"{spatial}*latent_width" if spatial != 1 else "latent_width",
            ]
        )
        builder.add_output(sample, "sample")
        for name, value in updated_cache.items():
            builder.add_output(value, f"{CONV_CACHE_OUTPUT_PREFIX}{name}")

        model = _make_model(graph)
        # Record each cache port's resolution relative to the latent. A consumer
        # that has to allocate the empty first-chunk caches cannot recover this
        # from the port's symbolic dimensions, which shape inference may rename
        # when it unifies a declared name with an internal value.
        for entry in module.conv_cache_spec():
            key = f"{CONV_CACHE_SCALE_METADATA}{CONV_CACHE_INPUT_PREFIX}{entry.name}"
            model.metadata_props[key] = str(entry.spatial_scale)
        return ModelPackage({"decoder": model}, config=config)
