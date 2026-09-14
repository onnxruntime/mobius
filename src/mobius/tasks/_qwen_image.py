# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Qwen Image packed-token denoising task."""

from __future__ import annotations

from typing import ClassVar

import onnx_ir as ir

from mobius._model_package import ModelPackage
from mobius.integrations.diffusers._configs import QwenImageConfig
from mobius.tasks._base import ModelTask, _make_graph, _make_model


class QwenImageDenoisingTask(ModelTask):
    """Build the Qwen Image denoiser with mask and externally prepared 3D RoPE."""

    model_roles: ClassVar[dict[str, str]] = {"model": "encoder"}

    def build(self, module, config: QwenImageConfig) -> ModelPackage:
        graph, builder = _make_graph(name="qwen_image_transformer")
        half_head_dim = config.attention_head_dim // 2

        sample = builder.input(
            "sample",
            dtype=config.dtype,
            shape=["batch", "image_sequence_length", config.in_channels],
        )
        timestep = builder.input("timestep", dtype=config.dtype, shape=["batch"])
        encoder_hidden_states = builder.input(
            "encoder_hidden_states",
            dtype=config.dtype,
            shape=["batch", "text_sequence_length", config.cross_attention_dim],
        )
        encoder_hidden_states_mask = builder.input(
            "encoder_hidden_states_mask",
            dtype=ir.DataType.BOOL,
            shape=["batch", "text_sequence_length"],
        )
        image_rotary_cos = builder.input(
            "image_rotary_cos",
            dtype=ir.DataType.FLOAT,
            shape=["image_sequence_length", half_head_dim],
        )
        image_rotary_sin = builder.input(
            "image_rotary_sin",
            dtype=ir.DataType.FLOAT,
            shape=["image_sequence_length", half_head_dim],
        )
        text_rotary_cos = builder.input(
            "text_rotary_cos",
            dtype=ir.DataType.FLOAT,
            shape=["text_sequence_length", half_head_dim],
        )
        text_rotary_sin = builder.input(
            "text_rotary_sin",
            dtype=ir.DataType.FLOAT,
            shape=["text_sequence_length", half_head_dim],
        )
        target_sequence_length = builder.input(
            "target_sequence_length",
            dtype=ir.DataType.INT64,
            shape=[1],
        )

        noise_pred = module(
            builder.op,
            sample=sample,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            encoder_hidden_states_mask=encoder_hidden_states_mask,
            image_rotary_cos=image_rotary_cos,
            image_rotary_sin=image_rotary_sin,
            text_rotary_cos=text_rotary_cos,
            text_rotary_sin=text_rotary_sin,
        )
        noise_pred = builder.op.Slice(
            noise_pred,
            builder.op.Constant(value_ints=[0]),
            target_sequence_length,
            builder.op.Constant(value_ints=[1]),
        )
        builder.add_output(noise_pred, "noise_pred")
        return ModelPackage({"model": _make_model(graph)}, config=config)
