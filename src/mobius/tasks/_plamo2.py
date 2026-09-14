# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import onnx_ir as ir
from onnxscript import nn

from mobius._configs import BaseModelConfig, Plamo2Config
from mobius._model_package import ModelPackage
from mobius.tasks._base import ModelTask, _make_graph, _make_model


class Plamo2CausalLMTask(ModelTask):
    """Build PLaMo2 with per-layer KV or convolution/SSM state."""

    def __init__(
        self,
        *,
        static_cache: bool = False,
        max_seq_len: int | None = None,
    ) -> None:
        if static_cache or max_seq_len is not None:
            raise ValueError(
                "PLaMo2 requires heterogeneous dynamic KV and recurrent states; "
                "static cache packages cannot represent this mixed-state ABI"
            )

    def build(self, module: nn.Module, config: BaseModelConfig) -> ModelPackage:
        if not isinstance(config, Plamo2Config):
            raise TypeError("Plamo2CausalLMTask requires Plamo2Config")

        batch = ir.SymbolicDim("batch_size")
        sequence = ir.SymbolicDim("sequence_length")
        past_sequence = ir.SymbolicDim("past_sequence_length")
        total_sequence = "past_sequence_length + sequence_length"
        present_sequence = ir.SymbolicDim("present_sequence_length")
        graph, builder = _make_graph("plamo2")
        input_ids = builder.input("input_ids", ir.DataType.INT64, [batch, sequence])
        position_ids = builder.input("position_ids", ir.DataType.INT64, [batch, sequence])
        attention_mask = builder.input(
            "attention_mask",
            ir.DataType.INT64,
            [batch, total_sequence],
        )

        past_states: list[tuple[ir.Value, ir.Value]] = []
        for layer, layer_type in enumerate(config.layer_types or ()):
            if layer_type == "full_attention":
                state_a = builder.input(
                    f"past_key_values.{layer}.key",
                    config.dtype,
                    [
                        batch,
                        config.num_key_value_heads,
                        past_sequence,
                        config.head_dim,
                    ],
                )
                state_b = builder.input(
                    f"past_key_values.{layer}.value",
                    config.dtype,
                    [
                        batch,
                        config.num_key_value_heads,
                        past_sequence,
                        config.head_dim,
                    ],
                )
            else:
                state_a = builder.input(
                    f"past_key_values.{layer}.conv_state",
                    config.dtype,
                    [batch, config.mamba_inner_size, config.mamba_d_conv - 1],
                )
                state_b = builder.input(
                    f"past_key_values.{layer}.recurrent_state",
                    ir.DataType.FLOAT,
                    [
                        batch,
                        config.mamba_num_heads,
                        config.mamba_head_dim,
                        config.mamba_d_state,
                    ],
                )
            past_states.append((state_a, state_b))

        logits, present_states = module(
            builder.op,
            input_ids,
            position_ids,
            attention_mask,
            tuple(past_states),
        )
        logits.shape = ir.Shape([batch, sequence, config.vocab_size])
        logits.type = ir.TensorType(config.dtype)
        builder.add_output(logits, "logits")
        for layer, (layer_type, states) in enumerate(
            zip(config.layer_types or (), present_states)
        ):
            state_a, state_b = states
            if layer_type == "full_attention":
                state_a.shape = ir.Shape(
                    [
                        batch,
                        config.num_key_value_heads,
                        present_sequence,
                        config.head_dim,
                    ]
                )
                state_b.shape = ir.Shape(
                    [
                        batch,
                        config.num_key_value_heads,
                        present_sequence,
                        config.head_dim,
                    ]
                )
                names = ("key", "value")
                state_b.type = ir.TensorType(config.dtype)
            else:
                state_a.shape = ir.Shape(
                    [batch, config.mamba_inner_size, config.mamba_d_conv - 1]
                )
                state_b.shape = ir.Shape(
                    [
                        batch,
                        config.mamba_num_heads,
                        config.mamba_head_dim,
                        config.mamba_d_state,
                    ]
                )
                names = ("conv_state", "recurrent_state")
                state_b.type = ir.TensorType(ir.DataType.FLOAT)
            state_a.type = ir.TensorType(config.dtype)
            builder.add_output(state_a, f"present.{layer}.{names[0]}")
            builder.add_output(state_b, f"present.{layer}.{names[1]}")

        model = _make_model(graph)
        self._register_functions(model, config)
        model.metadata_props["mobius.cache_abi"] = (
            "per-layer:attention=key,value;mamba=conv_state,recurrent_state-f32"
        )
        model.metadata_props["mobius.state_semantics"] = (
            "batch-axis reorder;copy-for-rollback;deterministic replay"
        )
        model.metadata_props["mobius.max_verified_context"] = str(config.attention_window_size)
        model.metadata_props["mobius.runtime_support"] = (
            "ORT GenAI 0.15.2 state ABI; package requires GQA-specialized attention"
        )
        return ModelPackage({"model": model}, config=config)

    @staticmethod
    def _register_functions(model: ir.Model, config: Plamo2Config) -> None:
        from mobius.functions import causal_conv_nd_with_state, linear_attention

        conv = causal_conv_nd_with_state(
            kernel_size=config.mamba_d_conv,
            channels=config.mamba_inner_size,
            ndim=1,
            activation="silu",
        )
        scan = linear_attention(
            q_num_heads=config.mamba_num_heads,
            kv_num_heads=config.mamba_num_heads,
            update_rule="gated",
            scale=1.0,
            stash_type=ir.DataType.FLOAT,
        )
        model.functions[conv.identifier()] = conv
        model.functions[scan.identifier()] = scan
        model.graph.opset_imports["com.microsoft"] = 1
