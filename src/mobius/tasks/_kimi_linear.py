# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Graph task for Kimi Linear's heterogeneous six-kind state ABI."""

from __future__ import annotations

import onnx_ir as ir
from onnxscript import nn

from mobius._configs import ArchitectureConfig
from mobius._model_package import ModelPackage
from mobius.functions import causal_conv_nd_with_state, linear_attention
from mobius.tasks._base import ModelTask, _make_graph, _make_model


class KimiLinearCausalLMTask(ModelTask):
    """Dynamic-cache task with three KDA convolution states plus matrix state."""

    def __init__(self, *, static_cache: bool = False, **_: object):
        if static_cache:
            raise ValueError(
                "Kimi Linear does not support static cache: its KDA layers require "
                "three convolution histories and one FP32 recurrent matrix state"
            )

    def build(self, module: nn.Module, config: ArchitectureConfig) -> ModelPackage:
        batch = ir.SymbolicDim("batch")
        seq_len = ir.SymbolicDim("sequence_len")
        past_len = ir.SymbolicDim("past_sequence_len")
        graph, builder = _make_graph()
        input_ids = builder.input("input_ids", dtype=ir.DataType.INT64, shape=[batch, seq_len])
        attention_mask = builder.input(
            "attention_mask",
            dtype=ir.DataType.INT64,
            shape=[batch, "past_seq_len + seq_len"],
        )
        position_ids = builder.input(
            "position_ids", dtype=ir.DataType.INT64, shape=[batch, seq_len]
        )

        projection = config.linear_num_key_heads * config.linear_key_head_dim
        history = config.linear_conv_kernel_dim - 1
        past_states = []
        for layer_idx, layer_type in enumerate(config.layer_types):
            prefix = f"past_key_values.{layer_idx}"
            if layer_type == "kimi_linear_attention":
                q = builder.input(
                    f"{prefix}.q_conv_state",
                    dtype=config.dtype,
                    shape=[batch, projection, history],
                )
                k = builder.input(
                    f"{prefix}.k_conv_state",
                    dtype=config.dtype,
                    shape=[batch, projection, history],
                )
                v = builder.input(
                    f"{prefix}.v_conv_state",
                    dtype=config.dtype,
                    shape=[batch, projection, history],
                )
                recurrent = builder.input(
                    f"{prefix}.recurrent_state",
                    dtype=ir.DataType.FLOAT,
                    shape=[
                        batch,
                        config.linear_num_key_heads,
                        config.linear_key_head_dim,
                        config.linear_value_head_dim,
                    ],
                )
                past_states.append((q, k, v, recurrent))
            else:
                key = builder.input(
                    f"{prefix}.key",
                    dtype=config.dtype,
                    shape=[
                        batch,
                        config.num_attention_heads,
                        past_len,
                        config.qk_nope_head_dim + config.qk_rope_head_dim,
                    ],
                )
                value = builder.input(
                    f"{prefix}.value",
                    dtype=config.dtype,
                    shape=[
                        batch,
                        config.num_attention_heads,
                        past_len,
                        config.v_head_dim,
                    ],
                )
                past_states.append((key, value))

        logits, present_states = module(
            builder.op,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_states,
        )
        builder.add_output(logits, "logits")
        for layer_idx, (layer_type, states) in enumerate(
            zip(config.layer_types, present_states)
        ):
            prefix = f"present.{layer_idx}"
            if layer_type == "kimi_linear_attention":
                names = ("q_conv_state", "k_conv_state", "v_conv_state", "recurrent_state")
            else:
                names = ("key", "value")
            for state, name in zip(states, names):
                builder.add_output(state, f"{prefix}.{name}")

        model = _make_model(graph)
        conv = causal_conv_nd_with_state(
            kernel_size=config.linear_conv_kernel_dim,
            channels=projection,
            activation="silu",
        )
        recurrence = linear_attention(
            q_num_heads=config.linear_num_key_heads,
            kv_num_heads=config.linear_num_value_heads,
            update_rule="gated_delta",
            scale=config.linear_key_head_dim**-0.5,
            stash_type=ir.DataType.FLOAT,
        )
        model.functions[conv.identifier()] = conv
        model.functions[recurrence.identifier()] = recurrence
        model.metadata_props["mobius.cache_abi"] = (
            "KDA:q_conv_state,k_conv_state,v_conv_state,recurrent_state;"
            "MLA:key,value;batch-axis=0;rollback=whole-state;replay=exact-prior-state"
        )
        model.metadata_props["mobius.runtime_support"] = (
            "Deferred: released generic OGA decoder cache schemas do not represent "
            "Kimi Linear's heterogeneous state ABI; tracked by "
            "https://github.com/onnxruntime/mobius/issues/605"
        )
        return ModelPackage({"model": model}, config=config)
