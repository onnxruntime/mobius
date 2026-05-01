# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""QwenImage 3D VAE task for encoder/decoder with 5D (video) inputs.

Builds a ModelPackage with separate "encoder" and "decoder" ONNX graphs:
- "encoder": sample (B, 3, T, H, W) → latent_dist (B, 2*z_dim, T', H', W')
- "decoder": latent_sample (B, z_dim, T', H', W') → sample (B, 3, T, H, W)
"""

from __future__ import annotations

from typing import ClassVar

import onnx_ir as ir

from mobius._diffusers_configs import QwenImageVAEConfig
from mobius._model_package import ModelPackage
from mobius.tasks._base import ComponentSpec, ModelTask, _make_graph, _make_model


class QwenImageVAETask(ModelTask):
    """Build 3D causal VAE encoder and decoder ONNX graphs."""

    model_roles: ClassVar[dict[str, str]] = {"encoder": "encoder", "decoder": "decoder"}
    components: ClassVar[ComponentSpec] = ComponentSpec(encoder="encoder", decoder="decoder")

    def build(
        self,
        module,
        config: QwenImageVAEConfig,
    ) -> ModelPackage:
        self._validate_components(module)
        encoder = self._build_encoder_graph(module, config)
        decoder = self._build_decoder_graph(module, config)
        return ModelPackage({"encoder": encoder, "decoder": decoder}, config=config)

    def _build_encoder_graph(
        self,
        module,
        config: QwenImageVAEConfig,
    ) -> ir.Model:
        graph, builder = _make_graph(name="vae_encoder")
        op = builder.op

        sample = builder.input("sample", dtype=ir.DataType.FLOAT, shape=["batch", 3, "frames", "height", "width"])

        hidden_states = module.encoder(op, sample)
        hidden_states = module.quant_conv(op, hidden_states)

        builder.add_output(hidden_states, "latent_dist")

        return _make_model(graph)

    def _build_decoder_graph(
        self,
        module,
        config: QwenImageVAEConfig,
    ) -> ir.Model:
        graph, builder = _make_graph(name="vae_decoder")
        op = builder.op

        latent_sample = builder.input("latent_sample", dtype=ir.DataType.FLOAT, shape=["batch", config.z_dim, "frames", "height", "width"])

        hidden_states = module.post_quant_conv(op, latent_sample)
        hidden_states = module.decoder(op, hidden_states)

        builder.add_output(hidden_states, "sample")

        return _make_model(graph)
