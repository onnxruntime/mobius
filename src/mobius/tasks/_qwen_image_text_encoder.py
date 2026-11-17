# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Qwen2.5-VL prompt encoder split for Qwen Image Edit."""

from __future__ import annotations

from typing import ClassVar

import onnx_ir as ir
from onnxscript import nn

from mobius._configs import ArchitectureConfig
from mobius._model_package import ModelPackage
from mobius.tasks._base import (
    ComponentSpec,
    ModelTask,
    _make_graph,
    _make_model,
    build_embedding_from_features,
)
from mobius.tasks._vision_language_3model import QwenVLTask


class QwenImageTextEncoderTask(ModelTask):
    """Build image-aware Qwen2.5-VL prompt embedding components."""

    model_roles: ClassVar[dict[str, str]] = {
        "model": "encoder",
        "vision_encoder": "encoder",
        "embedding": "embedding",
    }
    components = ComponentSpec(
        decoder="decoder",
        vision_encoder="vision_encoder",
        embedding="embedding",
    )

    def build(self, module: nn.Module, config: ArchitectureConfig) -> ModelPackage:
        self._validate_components(module)
        models = {
            "model": self._build_text_encoder(module.decoder, config),
            "vision_encoder": QwenVLTask()._build_vision(module.vision_encoder, config),
            "embedding": build_embedding_from_features(
                module.embedding,
                config,
                feature_name="image_features",
                feature_dim=config.hidden_size,
            ),
        }
        return ModelPackage(models, config=config)

    @staticmethod
    def _build_text_encoder(decoder: nn.Module, config: ArchitectureConfig) -> ir.Model:
        graph, builder = _make_graph(name="qwen_image_prompt_encoder")
        inputs_embeds = builder.input(
            "inputs_embeds",
            dtype=config.dtype,
            shape=["batch", "sequence_length", config.hidden_size],
        )
        attention_mask = builder.input(
            "attention_mask",
            dtype=ir.DataType.INT64,
            shape=["batch", "sequence_length"],
        )
        position_ids = builder.input(
            "position_ids",
            dtype=ir.DataType.INT64,
            shape=[3, "batch", "sequence_length"],
        )
        hidden_states, _ = decoder(
            builder.op,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            return_hidden_states=True,
        )

        # The official edit prompt template contributes 64 system-prefix tokens.
        # Remove them after the final text layer, preserving the padding mask for
        # the denoiser's joint attention.
        prompt_embeds = builder.op.Slice(
            hidden_states,
            builder.op.Constant(value_ints=[64]),
            builder.op.Constant(value_ints=[9223372036854775807]),
            builder.op.Constant(value_ints=[1]),
        )
        prompt_mask = builder.op.Slice(
            attention_mask,
            builder.op.Constant(value_ints=[64]),
            builder.op.Constant(value_ints=[9223372036854775807]),
            builder.op.Constant(value_ints=[1]),
        )
        prompt_mask = builder.op.Cast(prompt_mask, to=ir.DataType.BOOL)
        builder.add_output(prompt_embeds, "prompt_embeds")
        builder.add_output(prompt_mask, "prompt_embeds_mask")
        return _make_model(graph)
