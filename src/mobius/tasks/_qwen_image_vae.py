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

    # Both graphs are bidirectional convolution/attention networks. The
    # "decoder" name describes reconstruction direction, not an autoregressive
    # decoder role eligible for causal GQA/KV-cache fusion.
    model_roles: ClassVar[dict[str, str]] = {"encoder": "encoder", "decoder": "encoder"}
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

        sample = builder.input(
            "sample", dtype=config.dtype, shape=["batch", 3, "frames", "height", "width"]
        )

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

        latent_sample = builder.input(
            "latent_sample",
            dtype=config.dtype,
            shape=["batch", config.z_dim, "frames", "height", "width"],
        )

        hidden_states = module.post_quant_conv(op, latent_sample)
        hidden_states = module.decoder(op, hidden_states)

        builder.add_output(hidden_states, "sample")

        return _make_model(graph)


class QwenImageEditVAETask(QwenImageVAETask):
    """Build VAE graphs with Qwen Image Edit latent normalization embedded."""

    @staticmethod
    def _validate_latent_statistics(config: QwenImageVAEConfig) -> None:
        for name, values in (
            ("latents_mean", config.latents_mean),
            ("latents_std", config.latents_std),
        ):
            if values is None:
                raise ValueError(f"Qwen Image Edit VAE config requires {name}")
            if len(values) != config.z_dim:
                raise ValueError(
                    f"Qwen Image Edit VAE {name} must contain {config.z_dim} values, "
                    f"but received {len(values)}"
                )

    def _build_encoder_graph(self, module, config: QwenImageVAEConfig) -> ir.Model:
        self._validate_latent_statistics(config)
        assert config.latents_mean is not None and config.latents_std is not None
        graph, builder = _make_graph(name="vae_encoder")
        sample = builder.input(
            "sample", dtype=config.dtype, shape=["batch", 3, "frames", "height", "width"]
        )
        moments = module.quant_conv(builder.op, module.encoder(builder.op, sample))
        mean, _ = builder.op.Split(moments, num_outputs=2, axis=1, _outputs=2)
        latent_mean = builder.op.Constant(value_floats=list(config.latents_mean))
        latent_std = builder.op.Constant(value_floats=list(config.latents_std))
        latent_mean = builder.op.Reshape(
            builder.op.CastLike(latent_mean, mean),
            builder.op.Constant(value_ints=[1, config.z_dim, 1, 1, 1]),
        )
        latent_std = builder.op.Reshape(
            builder.op.CastLike(latent_std, mean),
            builder.op.Constant(value_ints=[1, config.z_dim, 1, 1, 1]),
        )
        image_latents = builder.op.Div(builder.op.Sub(mean, latent_mean), latent_std)
        builder.add_output(image_latents, "image_latents")
        return _make_model(graph)

    def _build_decoder_graph(self, module, config: QwenImageVAEConfig) -> ir.Model:
        self._validate_latent_statistics(config)
        assert config.latents_mean is not None and config.latents_std is not None
        graph, builder = _make_graph(name="vae_decoder")
        latent_sample = builder.input(
            "latent_sample",
            dtype=config.dtype,
            shape=["batch", config.z_dim, "frames", "height", "width"],
        )
        latent_mean = builder.op.Constant(value_floats=list(config.latents_mean))
        latent_std = builder.op.Constant(value_floats=list(config.latents_std))
        latent_mean = builder.op.Reshape(
            builder.op.CastLike(latent_mean, latent_sample),
            builder.op.Constant(value_ints=[1, config.z_dim, 1, 1, 1]),
        )
        latent_std = builder.op.Reshape(
            builder.op.CastLike(latent_std, latent_sample),
            builder.op.Constant(value_ints=[1, config.z_dim, 1, 1, 1]),
        )
        latent_sample = builder.op.Add(builder.op.Mul(latent_sample, latent_std), latent_mean)
        sample = module.decoder(builder.op, module.post_quant_conv(builder.op, latent_sample))
        builder.add_output(sample, "sample")
        return _make_model(graph)
