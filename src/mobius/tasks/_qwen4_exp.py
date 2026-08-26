# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Graph I/O contract for the exact Qwen4-Exp heterogeneous text cache."""

from __future__ import annotations

import onnx_ir as ir
from onnxscript import nn

from mobius._configs import Qwen4ExpConfig
from mobius._model_package import ModelPackage
from mobius.tasks._base import ModelTask, _make_graph, _make_model
from mobius.tasks._cache_utils import _register_linear_attention_functions


class Qwen4ExpCausalLMTask(ModelTask):
    """Build Qwen4-Exp with DeltaNet, PLE, QSA, and position cache states."""

    def build(self, module: nn.Module, config: Qwen4ExpConfig) -> ModelPackage:
        batch = ir.SymbolicDim("batch")
        seq_len = ir.SymbolicDim("sequence_len")
        past_seq_len = ir.SymbolicDim("past_sequence_len")
        graph, builder = _make_graph("qwen4_exp")

        input_ids = builder.input("input_ids", dtype=ir.DataType.INT64, shape=[batch, seq_len])
        attention_mask = builder.input(
            "attention_mask",
            dtype=ir.DataType.INT64,
            shape=[batch, "past_sequence_len + sequence_len"],
        )
        position_ids = builder.input(
            "position_ids", dtype=ir.DataType.INT64, shape=[batch, seq_len]
        )
        past_position_ids = builder.input(
            "past_position_ids",
            dtype=ir.DataType.INT64,
            shape=[batch, past_seq_len],
        )

        assert config.layer_types is not None
        linear_key_dim = (
            config.linear_num_key_heads * config.linear_key_head_dim
            if config.linear_num_key_heads is not None
            and config.linear_key_head_dim is not None
            else 0
        )
        linear_value_dim = (
            config.linear_num_value_heads * config.linear_value_head_dim
            if config.linear_num_value_heads is not None
            and config.linear_value_head_dim is not None
            else 0
        )
        linear_conv_dim = 2 * linear_key_dim + linear_value_dim
        states = []
        for layer_idx, layer_type in enumerate(config.layer_types):
            if layer_type == "linear_attention":
                if not linear_key_dim or not linear_value_dim:
                    raise ValueError(
                        "Qwen4-Exp linear-attention dimensions must be configured"
                    )
                conv_state = builder.input(
                    f"past_key_values.{layer_idx}.conv_state",
                    dtype=config.dtype,
                    shape=[
                        batch,
                        linear_conv_dim,
                        config.linear_conv_kernel_dim,
                    ],
                )
                recurrent_state = builder.input(
                    f"past_key_values.{layer_idx}.recurrent_state",
                    dtype=config.dtype,
                    shape=[
                        batch,
                        config.linear_num_value_heads,
                        config.linear_key_head_dim,
                        config.linear_value_head_dim,
                    ],
                )
                layer_states: tuple[ir.Value, ...] = (
                    conv_state,
                    recurrent_state,
                )
                if layer_idx + 1 in (config.ple_layer_ids or []):
                    ple_state_len = (config.ple_conv_kernel_size - 1) * config.ngram_size
                    ple_conv_state = builder.input(
                        f"past_key_values.{layer_idx}.ple_conv_state",
                        dtype=config.dtype,
                        shape=[
                            batch,
                            config.hc_count * config.hidden_size,
                            ple_state_len,
                        ],
                    )
                    ple_context = builder.input(
                        f"past_key_values.{layer_idx}.ple_context",
                        dtype=ir.DataType.INT64,
                        shape=[batch, config.ngram_size - 1],
                    )
                    eos_token_id = (
                        config.eos_token_id[0]
                        if isinstance(config.eos_token_id, list)
                        else config.eos_token_id
                    )
                    assert eos_token_id is not None
                    empty_prefix = builder.op.Equal(
                        builder.op.Shape(past_position_ids, start=1, end=2),
                        builder.op.Constant(value_ints=[0]),
                    )
                    ple_context = builder.op.Where(
                        empty_prefix,
                        builder.op.Expand(
                            builder.op.Constant(value_int=eos_token_id),
                            builder.op.Shape(ple_context),
                        ),
                        ple_context,
                    )
                    layer_states += (ple_conv_state, ple_context)
            else:
                key = builder.input(
                    f"past_key_values.{layer_idx}.key",
                    dtype=config.dtype,
                    shape=[
                        batch,
                        config.num_key_value_heads,
                        past_seq_len,
                        config.head_dim,
                    ],
                )
                value = builder.input(
                    f"past_key_values.{layer_idx}.value",
                    dtype=config.dtype,
                    shape=[
                        batch,
                        config.num_key_value_heads,
                        past_seq_len,
                        config.head_dim,
                    ],
                )
                index_key = builder.input(
                    f"past_key_values.{layer_idx}.index_key",
                    dtype=config.dtype,
                    shape=[batch, past_seq_len, config.indexer_head_dim],
                )
                layer_states = (key, value, index_key)
            states.append(layer_states)

        logits, presents, present_position_ids = module(
            builder.op,
            input_ids,
            attention_mask,
            position_ids,
            past_position_ids,
            states,
        )
        builder.add_output(logits, "logits")
        builder.add_output(present_position_ids, "present_position_ids")
        for layer_idx, (layer_type, present) in enumerate(zip(config.layer_types, presents)):
            if layer_type == "linear_attention":
                builder.add_output(present[0], f"present.{layer_idx}.conv_state")
                builder.add_output(present[1], f"present.{layer_idx}.recurrent_state")
                if len(present) == 4:
                    builder.add_output(present[2], f"present.{layer_idx}.ple_conv_state")
                    builder.add_output(present[3], f"present.{layer_idx}.ple_context")
            else:
                builder.add_output(present[0], f"present.{layer_idx}.key")
                builder.add_output(present[1], f"present.{layer_idx}.value")
                builder.add_output(present[2], f"present.{layer_idx}.index_key")

        model = _make_model(graph)
        _register_linear_attention_functions(model, config)
        model.metadata_props["mobius.cache_abi"] = (
            "qwen4-exp:position_ids;"
            "linear=conv_state,recurrent_state[,ple_conv_state,ple_context];"
            "qsa=key,value,index_key"
        )
        model.metadata_props["mobius.upstream_revision"] = (
            "Qwen/Qwen3.8-Flash-Next@f5d08274bafd880402bd16f5e3e6c514136ec06c;"
            "transformers@598d8ba8baaec7fec5a22da0e2844c7bf4ea20e1"
        )
        return ModelPackage({"model": model}, config=config)
