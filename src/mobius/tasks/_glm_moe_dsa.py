# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Graph tasks for GLM-5.2 IndexShare DSA and its MTP head."""

from __future__ import annotations

from typing import ClassVar

import onnx_ir as ir

from mobius._configs import ArchitectureConfig
from mobius._model_package import ModelPackage
from mobius.tasks._base import ModelTask, _make_graph, _make_model


class GlmMoeDsaTask(ModelTask):
    """Build the target decoder and optional GLM-5.2 MTP component."""

    model_roles: ClassVar[dict[str, str]] = {"model": "decoder", "mtp": "decoder"}

    @staticmethod
    def _cache_dims(config: ArchitectureConfig, layer_idx: int) -> tuple[int, int]:
        qk_dim = config.num_attention_heads * (
            int(config.qk_nope_head_dim or 0) + int(config.qk_rope_head_dim or 0)
        )
        value_dim = config.num_attention_heads * int(config.v_head_dim or 0)
        if config.use_dsa and config.indexer_types:
            if config.indexer_types[layer_idx] == "full":
                qk_dim += int(config.index_head_dim or 0)
        return qk_dim, value_dim

    def _build_target(self, module, config: ArchitectureConfig):
        graph, builder = _make_graph("glm_moe_dsa")
        input_ids = builder.input(
            "input_ids", ir.DataType.INT64, ["batch_size", "sequence_length"]
        )
        attention_mask = builder.input(
            "attention_mask", ir.DataType.INT64, ["batch_size", "total_sequence_length"]
        )
        position_ids = builder.input(
            "position_ids", ir.DataType.INT64, ["batch_size", "sequence_length"]
        )

        past_key_values = []
        for i in range(config.num_hidden_layers):
            key_dim, value_dim = self._cache_dims(config, i)
            if config.use_dsa:
                key_shape = ["batch_size", 1, "past_sequence_length", key_dim]
                value_shape = ["batch_size", 1, "past_sequence_length", value_dim]
            else:
                key_shape = [
                    "batch_size",
                    config.num_attention_heads,
                    "past_sequence_length",
                    int(config.qk_nope_head_dim or 0) + int(config.qk_rope_head_dim or 0),
                ]
                value_shape = [
                    "batch_size",
                    config.num_attention_heads,
                    "past_sequence_length",
                    int(config.v_head_dim or 0),
                ]
            key = builder.input(
                f"past_key_values.{i}.key",
                config.dtype,
                key_shape,
            )
            value = builder.input(
                f"past_key_values.{i}.value",
                config.dtype,
                value_shape,
            )
            past_key_values.append((key, value))

        logits, presents = module(
            builder.op,
            input_ids,
            attention_mask,
            position_ids,
            past_key_values,
        )
        builder.add_output(logits, "logits")
        for i, (key, value) in enumerate(presents):
            builder.add_output(key, f"present.{i}.key")
            builder.add_output(value, f"present.{i}.value")
        return _make_model(graph)

    def _build_mtp(self, module, config: ArchitectureConfig):
        graph, builder = _make_graph("glm_moe_dsa_mtp")
        inputs_embeds = builder.input(
            "inputs_embeds",
            config.dtype,
            ["batch_size", "sequence_length", config.hidden_size],
        )
        hidden_states = builder.input(
            "hidden_states",
            config.dtype,
            ["batch_size", "sequence_length", config.hidden_size],
        )
        attention_mask = builder.input(
            "attention_mask", ir.DataType.INT64, ["batch_size", "total_sequence_length"]
        )
        position_ids = builder.input(
            "position_ids", ir.DataType.INT64, ["batch_size", "sequence_length"]
        )
        key_dim = config.num_attention_heads * (
            int(config.qk_nope_head_dim or 0) + int(config.qk_rope_head_dim or 0)
        ) + int(config.index_head_dim or 0)
        value_dim = config.num_attention_heads * int(config.v_head_dim or 0)
        past_key = builder.input(
            "past_key_values.0.key",
            config.dtype,
            ["batch_size", 1, "past_sequence_length", key_dim],
        )
        past_value = builder.input(
            "past_key_values.0.value",
            config.dtype,
            ["batch_size", 1, "past_sequence_length", value_dim],
        )

        mtp_hidden, present, topk_indices = module.mtp(
            builder.op,
            inputs_embeds,
            hidden_states,
            attention_mask,
            position_ids,
            (past_key, past_value),
        )
        builder.add_output(mtp_hidden, "mtp_hidden")
        builder.add_output(present[0], "present.0.key")
        builder.add_output(present[1], "present.0.value")
        builder.add_output(topk_indices, "topk_indices")
        return _make_model(graph)

    def build(self, module, config: ArchitectureConfig) -> ModelPackage:
        models = {"model": self._build_target(module, config)}
        if getattr(module, "mtp", None) is not None:
            models["mtp"] = self._build_mtp(module, config)
        return ModelPackage(models, config=config)
