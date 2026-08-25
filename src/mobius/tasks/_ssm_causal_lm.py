# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""SSM causal language model tasks with conv_state + ssm_state carry.

Unlike CausalLMTask (for transformers), SSM models:
    - Do NOT use attention_mask or position_ids
    - Carry conv_state + ssm_state per layer instead of KV cache
    - Still produce input_ids → logits

State per layer:
    past_states.{i}.conv_state:  (batch, d_inner, conv_kernel - 1)
    past_states.{i}.ssm_state:   (batch, ..., state_size)
"""

from __future__ import annotations

from typing import ClassVar

import onnx_ir as ir
from onnxscript import nn

from mobius._configs import BaseModelConfig, Mamba2Config, MambaConfig
from mobius._model_package import ModelPackage
from mobius.tasks._base import (
    ModelTask,
    _make_graph,
    _make_model,
)


def _build_ssm_task(
    module: nn.Module,
    config: BaseModelConfig,
    conv_state_shape: list,
    ssm_state_shape: list,
    sequence_length: int | ir.SymbolicDim,
) -> ModelPackage:
    """Shared implementation for SSM causal LM tasks.

    Args:
        module: The SSM module to wire.
        config: Model configuration.
        conv_state_shape: Per-layer conv state shape (excluding batch).
        ssm_state_shape: Per-layer SSM state shape (excluding batch).

    Returns:
        A :class:`ModelPackage` containing the built model.
    """
    batch = ir.SymbolicDim("batch")
    graph, builder = _make_graph()

    input_ids = builder.input(
        "input_ids",
        dtype=ir.DataType.INT64,
        shape=[batch, sequence_length],
    )

    past_states: list[tuple[ir.Value, ir.Value]] = []
    for i in range(config.num_hidden_layers):
        conv_state = builder.input(
            f"past_states.{i}.conv_state",
            dtype=config.dtype,
            shape=[batch, *conv_state_shape],
        )
        ssm_state = builder.input(
            f"past_states.{i}.ssm_state",
            dtype=config.dtype,
            shape=[batch, *ssm_state_shape],
        )
        past_states.append((conv_state, ssm_state))

    logits, present_states = module(
        builder.op,
        input_ids=input_ids,
        past_states=past_states,
    )

    builder.add_output(logits, "logits")
    for i, (conv_state, ssm_state) in enumerate(present_states):
        builder.add_output(conv_state, f"present.{i}.conv_state")
        builder.add_output(ssm_state, f"present.{i}.ssm_state")

    return ModelPackage({"model": _make_model(graph)}, config=config)


class SSMCausalLMTask(ModelTask):
    """Causal language model with SSM state carry for Mamba-style models.

    Inputs:
        - input_ids: [batch, sequence_len] INT64
        - past_states.{i}.conv_state: [batch, d_inner, conv_kernel-1] FLOAT
        - past_states.{i}.ssm_state: [batch, d_inner, state_size] FLOAT

    Outputs:
        - logits: FLOAT
        - present.{i}.conv_state: [batch, d_inner, conv_kernel-1] FLOAT
        - present.{i}.ssm_state: [batch, d_inner, state_size] FLOAT

    The module's ``forward()`` must accept
    ``(op, input_ids, past_states)``
    and return ``(logits, list_of_(conv_state, ssm_state)_tuples)``.
    """

    model_roles: ClassVar[dict[str, str]] = {"model": "decoder"}

    def build(
        self,
        module: nn.Module,
        config: BaseModelConfig,
    ) -> ModelPackage:
        assert isinstance(config, MambaConfig), (
            f"SSMCausalLMTask requires MambaConfig, got {type(config).__name__}"
        )
        d_inner = config.intermediate_size
        return _build_ssm_task(
            module,
            config,
            conv_state_shape=[d_inner, config.conv_kernel - 1],
            ssm_state_shape=[d_inner, config.state_size],
            # Mamba1's selective-scan graph is a single recurrent step.
            # Prompt ingestion is token-by-token with state threading.
            sequence_length=1,
        )


class SSM2CausalLMTask(ModelTask):
    """Causal language model with Mamba2/SSD state carry.

    Like SSMCausalLMTask but with 4D SSM state for Mamba2 multi-head
    architecture and wider conv_dim.  Uses LinearAttention + CausalConvWithState
    function ops.

    Inputs:
        - input_ids: [batch, sequence_len] INT64
        - past_states.{i}.conv_state: [batch, conv_dim, conv_kernel-1]
        - past_states.{i}.ssm_state: [batch, num_heads, state_size, head_dim]

    Outputs:
        - logits: FLOAT
        - present.{i}.conv_state: [batch, conv_dim, conv_kernel-1]
        - present.{i}.ssm_state: [batch, num_heads, state_size, head_dim]
    """

    model_roles: ClassVar[dict[str, str]] = {"model": "decoder"}

    def build(
        self,
        module: nn.Module,
        config: BaseModelConfig,
    ) -> ModelPackage:
        assert isinstance(config, Mamba2Config), (
            f"SSM2CausalLMTask requires Mamba2Config, got {type(config).__name__}"
        )
        n_groups = config.n_groups
        state_size = config.state_size
        conv_dim = config.intermediate_size + 2 * n_groups * state_size
        pkg = _build_ssm_task(
            module,
            config,
            conv_state_shape=[conv_dim, config.conv_kernel - 1],
            # LinearAttention convention: (H, d_k, d_v) = (H, d_state, d_head)
            ssm_state_shape=[config.num_heads, state_size, config.head_dim],
            sequence_length=ir.SymbolicDim("sequence_len"),
        )
        # Register CausalConvWithState and LinearAttention function ops.
        from mobius.tasks._cache_utils import (
            _register_linear_attention_functions_for_ssm2,
        )

        _register_linear_attention_functions_for_ssm2(pkg["model"], config)
        return pkg
