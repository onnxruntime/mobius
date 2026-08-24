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

    @staticmethod
    def _compressed_inputs(builder, module, batch):
        """Create the native-CSA ``past_compressed_*`` graph inputs.

        Inspects each built attention layer's resolved ``csa_plan`` (the
        property gate's output) and, for every native-CSA (ratio-128) layer,
        registers the deterministic compressed-state inputs alongside the
        existing dense KV cache. Returns a per-layer list aligned with
        ``module.model.layers`` -- ``None`` for dense layers -- for the model's
        ``past_compressed_states`` argument.

        When ``config.native_csa`` is off every plan is ``None``, so no inputs
        are created and the returned list is all-``None`` (byte-identical to
        the pre-CSA graph). The records axis is a shared dynamic symbolic dim
        because every CSA layer advances its compressed cache in lockstep.
        """
        records = ir.SymbolicDim("past_compressed_records")
        past_compressed_states: list = []
        for layer in module.model.layers:
            plan = layer.self_attn.csa_plan
            if plan is None:
                past_compressed_states.append(None)
                continue
            past_compressed_kv = builder.input(
                plan.past_compressed_kv_name,
                dtype=ir.DataType.FLOAT,
                shape=[batch, records, plan.stored_width],
            )
            past_compression_carry = builder.input(
                plan.past_compression_carry_name,
                dtype=ir.DataType.FLOAT,
                shape=[batch, plan.carry_slots, plan.carry_planes, plan.head_dim],
            )
            past_compressed_states.append((past_compressed_kv, past_compression_carry))
        return past_compressed_states

    @staticmethod
    def _compressed_outputs(builder, module, present_compressed_states, batch):
        """Register native-CSA ``present_compressed_*`` outputs with explicit shapes.

        ``pkg.nxrt::CompressedSparseAttention`` has no Python/ONNX
        symbolic-shape-inference function (like ``pkg.nxrt::IndexShare`` in the
        GLM DSA task), so each present compressed output is stamped with an
        explicit type or it would export untyped. The records axis is a
        distinct dynamic symbolic dim (present record count = past + newly
        pooled blocks, not a simple ``past + sequence`` sum); the carry tensor
        is records-independent ``[batch, ratio, planes, head_dim]``.
        """
        present_records = ir.SymbolicDim("present_compressed_records")
        for layer, present in zip(module.model.layers, present_compressed_states):
            plan = layer.self_attn.csa_plan
            if plan is None:
                continue
            present_compressed_kv, present_compression_carry = present
            present_compressed_kv.shape = ir.Shape([batch, present_records, plan.stored_width])
            present_compressed_kv.type = ir.TensorType(ir.DataType.FLOAT)
            present_compression_carry.shape = ir.Shape(
                [batch, plan.carry_slots, plan.carry_planes, plan.head_dim]
            )
            present_compression_carry.type = ir.TensorType(ir.DataType.FLOAT)
            builder.add_output(present_compressed_kv, plan.present_compressed_kv_name)
            builder.add_output(present_compression_carry, plan.present_compression_carry_name)

    def _build_target(self, module, config: ArchitectureConfig):
        graph, builder = _make_graph("deepseek_v4")
        batch, sequence_length, attention_mask, position_ids, past_key_values = self._inputs(
            builder, config, config.num_hidden_layers
        )
        input_ids = builder.input("input_ids", ir.DataType.INT64, [batch, sequence_length])
        past_compressed_states = self._compressed_inputs(builder, module, batch)
        hidden_states, presents, present_compressed_states, hc_states = module.model(
            builder.op,
            input_ids,
            attention_mask,
            position_ids,
            past_key_values,
            past_compressed_states=past_compressed_states,
        )
        builder.add_output(module.lm_head(builder.op, hidden_states), "logits")
        if len(module.mtp):
            builder.add_output(hc_states, "hidden_states")
        self._outputs(builder, presents, config, batch)
        self._compressed_outputs(builder, module, present_compressed_states, batch)
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
