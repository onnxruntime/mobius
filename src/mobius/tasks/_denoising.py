# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Denoising task for diffusion models (UNet, DiT, etc.).

Builds an ONNX graph that takes noisy latent + timestep + conditioning
and produces noise prediction.
"""

from __future__ import annotations

from typing import ClassVar

import onnx_ir as ir

from mobius._model_package import ModelPackage
from mobius.integrations.diffusers._configs import UNet2DConfig
from mobius.tasks._base import ModelTask, _make_graph, _make_model


class DenoisingTask(ModelTask):
    """Build ONNX graph for diffusion denoising."""

    model_roles: ClassVar[dict[str, str]] = {"model": "encoder"}

    def build(
        self,
        module,
        config: UNet2DConfig,
    ) -> ModelPackage:
        graph, builder = _make_graph()
        op = builder.op

        sample = builder.input(
            "sample",
            dtype=ir.DataType.FLOAT,
            shape=["batch", config.in_channels, "height", "width"],
        )
        timestep = builder.input("timestep", dtype=ir.DataType.INT64, shape=["batch"])
        encoder_hidden_states = builder.input(
            "encoder_hidden_states",
            dtype=ir.DataType.FLOAT,
            shape=["batch", "sequence_length", config.cross_attention_dim],
        )

        # Runtime LoRA gates: one scalar `lora_gate.{name}` input per baked
        # adapter (1.0 = active, 0.0 = inactive, or a blend strength), so a loaded
        # model can switch/blend LoRAs at run time with no rebuild. Only modules
        # that declare adapters (`_lora_adapter_names`) receive the gates.
        lora_gates = {}
        for name in getattr(module, "_lora_adapter_names", []):
            lora_gates[name] = builder.input(
                f"lora_gate.{name}", dtype=ir.DataType.FLOAT, shape=[]
            )

        extra_kwargs = {"lora_gates": lora_gates} if lora_gates else {}
        noise_pred = module(
            op,
            sample=sample,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            **extra_kwargs,
        )

        builder.add_output(noise_pred, "noise_pred")

        return ModelPackage({"model": _make_model(graph)}, config=config)
