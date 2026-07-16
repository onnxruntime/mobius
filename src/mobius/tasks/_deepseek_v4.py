# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Graph tasks for the DeepSeek-V4 target decoder and MTP sidecar."""

from __future__ import annotations

from typing import ClassVar

import onnx_ir as ir

from mobius._configs import ArchitectureConfig
from mobius._model_package import ModelPackage
from mobius.tasks._base import ModelTask, _make_graph, _make_model
from mobius.tasks._cache_utils import _make_kv_cache_inputs, _register_kv_cache_outputs


class DeepSeekV4Task(ModelTask):
    """Build the target decoder plus the optional official single-layer MTP head."""

    model_roles: ClassVar[dict[str, str]] = {"model": "decoder", "mtp": "decoder"}

    @staticmethod
    def _inputs(builder, config: ArchitectureConfig, num_layers: int):
        batch = ir.SymbolicDim("batch")
        sequence_length = ir.SymbolicDim("sequence_length")
        past_sequence_length = ir.SymbolicDim("past_sequence_length")
        attention_mask = builder.input(
            "attention_mask",
            ir.DataType.INT64,
            [batch, "past_sequence_length + sequence_length"],
        )
        position_ids = builder.input(
            "position_ids", ir.DataType.INT64, [batch, sequence_length]
        )
        past_key_values = _make_kv_cache_inputs(
            builder,
            num_layers,
            1,
            config.head_dim,
            config.dtype,
            batch,
            past_sequence_length,
        )
        return (
            batch,
            sequence_length,
            attention_mask,
            position_ids,
            past_key_values,
        )

    @staticmethod
    def _outputs(
        builder,
        presents,
        config: ArchitectureConfig,
        batch,
    ) -> None:
        _register_kv_cache_outputs(
            builder,
            presents,
            batch=batch,
            num_kv_heads=1,
            key_head_dim=config.head_dim,
            value_head_dim=config.head_dim,
            total_seq_len="past_sequence_length + sequence_length",
            dtype=config.dtype,
        )

    def _build_target(self, module, config: ArchitectureConfig):
        graph, builder = _make_graph("deepseek_v4")
        batch, sequence_length, attention_mask, position_ids, past_key_values = self._inputs(
            builder, config, config.num_hidden_layers
        )
        input_ids = builder.input("input_ids", ir.DataType.INT64, [batch, sequence_length])
        hidden_states, presents, hc_states = module.model(
            builder.op,
            input_ids,
            attention_mask,
            position_ids,
            past_key_values,
        )
        builder.add_output(module.lm_head(builder.op, hidden_states), "logits")
        if len(module.mtp):
            builder.add_output(hc_states, "hidden_states")
        self._outputs(builder, presents, config, batch)
        return _make_model(graph)

    def _build_mtp(self, module, config: ArchitectureConfig):
        graph, builder = _make_graph("deepseek_v4_mtp")
        batch, sequence_length, attention_mask, position_ids, past_key_values = self._inputs(
            builder, config, 1
        )
        inputs_embeds = builder.input(
            "inputs_embeds",
            config.dtype,
            [batch, sequence_length, config.hidden_size],
        )
        hidden_states = builder.input(
            "hidden_states",
            config.dtype,
            [batch, sequence_length, config.hc_mult, config.hidden_size],
        )
        mtp_hidden, present = module.mtp[0](
            builder.op,
            inputs_embeds,
            hidden_states,
            attention_mask,
            position_ids,
            past_key_values[0],
        )
        builder.add_output(mtp_hidden, "mtp_hidden")
        self._outputs(builder, [present], config, batch)
        return _make_model(graph)

    def build(self, module, config: ArchitectureConfig) -> ModelPackage:
        models = {"model": self._build_target(module, config)}
        if len(module.mtp):
            models["mtp"] = self._build_mtp(module, config)
        return ModelPackage(models, config=config)
