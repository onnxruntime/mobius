# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import onnx_ir as ir
from onnxscript import nn

from mobius._configs import BaseModelConfig, FalconH1Config
from mobius._model_package import ModelPackage
from mobius.tasks._base import ModelTask, _make_graph, _make_model
from mobius.tasks._cache_utils import _register_linear_attention_functions


class FalconH1CausalLMTask(ModelTask):
    """Build Falcon-H1 with ordered K, V, convolution, and SSM state per layer."""

    def __init__(
        self,
        *,
        static_cache: bool = False,
        max_seq_len: int | None = None,
    ) -> None:
        if static_cache or max_seq_len is not None:
            raise ValueError(
                "Falcon-H1 requires heterogeneous dynamic K/V, convolution, and SSM "
                "states; static cache packages cannot represent this four-state ABI"
            )
        self._static_cache = False

    def build(
        self,
        module: nn.Module,
        config: BaseModelConfig,
    ) -> ModelPackage:
        if not isinstance(config, FalconH1Config):
            raise TypeError("FalconH1CausalLMTask requires FalconH1Config")

        batch = ir.SymbolicDim("batch_size")
        sequence_length = ir.SymbolicDim("sequence_length")
        past_sequence_length = ir.SymbolicDim("past_sequence_length")
        total_sequence_length = "past_sequence_length + sequence_length"

        graph, builder = _make_graph("falcon_h1")
        op = builder.op
        input_ids = builder.input(
            "input_ids",
            ir.DataType.INT64,
            [batch, sequence_length],
        )
        position_ids = builder.input(
            "position_ids",
            ir.DataType.INT64,
            [batch, sequence_length],
        )
        attention_mask = builder.input(
            "attention_mask",
            ir.DataType.INT64,
            [batch, total_sequence_length],
        )

        past_states: list[ir.Value] = []
        for layer_idx in range(config.num_hidden_layers):
            past_states.extend(
                [
                    builder.input(
                        f"past_key_values.{layer_idx}.key",
                        config.dtype,
                        [
                            batch,
                            config.num_key_value_heads,
                            past_sequence_length,
                            config.head_dim,
                        ],
                    ),
                    builder.input(
                        f"past_key_values.{layer_idx}.value",
                        config.dtype,
                        [
                            batch,
                            config.num_key_value_heads,
                            past_sequence_length,
                            config.head_dim,
                        ],
                    ),
                    builder.input(
                        f"past_key_values.{layer_idx}.conv_state",
                        config.dtype,
                        [
                            batch,
                            config.mamba_d_ssm
                            + 2 * config.mamba_n_groups * config.mamba_d_state,
                            config.mamba_d_conv - 1,
                        ],
                    ),
                    builder.input(
                        f"past_key_values.{layer_idx}.ssm_state",
                        config.dtype,
                        [
                            batch,
                            config.mamba_n_heads,
                            config.mamba_d_state,
                            config.mamba_d_head,
                        ],
                    ),
                ]
            )

        # Recurrent masking applies only to the current token block, not cached K/V.
        current_sequence_length = op.Shape(input_ids, start=1, end=2)
        mamba_mask = op.Slice(
            attention_mask,
            op.Neg(current_sequence_length),
            [9223372036854775807],
            [1],
        )
        mamba_mask = op.Unsqueeze(mamba_mask, [-1])

        logits, present_states = module(
            op,
            input_ids,
            position_ids,
            attention_mask,
            mamba_mask,
            tuple(past_states),
        )
        logits.shape = ir.Shape([batch, sequence_length, config.vocab_size])
        logits.type = ir.TensorType(config.dtype)
        builder.add_output(logits, "logits")
        state_kinds = ("key", "value", "conv_state", "ssm_state")
        for flat_idx, state in enumerate(present_states):
            layer_idx, state_idx = divmod(flat_idx, 4)
            if state_idx < 2:
                state.shape = ir.Shape(
                    [
                        batch,
                        config.num_key_value_heads,
                        total_sequence_length,
                        config.head_dim,
                    ]
                )
            elif state_idx == 2:
                state.shape = ir.Shape(
                    [
                        batch,
                        config.mamba_d_ssm + 2 * config.mamba_n_groups * config.mamba_d_state,
                        config.mamba_d_conv - 1,
                    ]
                )
            else:
                state.shape = ir.Shape(
                    [
                        batch,
                        config.mamba_n_heads,
                        config.mamba_d_state,
                        config.mamba_d_head,
                    ]
                )
            state.type = ir.TensorType(config.dtype)
            builder.add_output(
                state,
                f"present.{layer_idx}.{state_kinds[state_idx]}",
            )

        model = _make_model(graph)
        _register_linear_attention_functions(model, config)
        model.metadata_props["mobius.cache_abi"] = "per-layer:key,value,conv_state,ssm_state"
        model.metadata_props["mobius.runtime_support"] = (
            "deferred: downstream heterogeneous-state package schema is unavailable"
        )
        return ModelPackage({"model": model}, config=config)
