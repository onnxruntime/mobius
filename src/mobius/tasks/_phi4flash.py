# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Task wiring for Phi-4 Flash's heterogeneous SambaY cache ABI."""

from __future__ import annotations

from typing import ClassVar

import onnx_ir as ir
from onnxscript import nn

from mobius._configs import BaseModelConfig, Phi4FlashConfig
from mobius._model_package import ModelPackage
from mobius.tasks._base import ModelTask, _make_graph, _make_model


class Phi4FlashCausalLMTask(ModelTask):
    """Build Phi-4 Flash with source-shaped recurrent and differential KV state.

    The source has no cache entries for cross layers. Slots 0 through 17 are
    physical layers 0 through 17: even slots contain K-wide Mamba convolution
    state plus SSM state, odd slots below 17 contain 512-token local KV, and
    slot 17 contains global KV. Cache inputs are dynamic chronological tensors
    for monotonic prefill/decode chaining; this is intentionally not the
    source's mutable pre-allocated ``SambaYCache`` object.
    """

    model_roles: ClassVar[dict[str, str]] = {"model": "decoder"}

    def __init__(
        self,
        *,
        static_cache: bool = False,
        max_seq_len: int | None = None,
    ) -> None:
        if static_cache or max_seq_len is not None:
            raise ValueError(
                "Phi-4 Flash only supports its dynamic monotonic-append SambaY cache ABI; "
                "static/paged cache allocation is unsupported."
            )

    def build(self, module: nn.Module, config: BaseModelConfig) -> ModelPackage:
        if not isinstance(config, Phi4FlashConfig):
            raise TypeError("Phi4FlashCausalLMTask requires Phi4FlashConfig")

        batch = ir.SymbolicDim("batch_size")
        sequence = ir.SymbolicDim("sequence_length")
        global_past = ir.SymbolicDim("global_past_sequence_length")
        local_past = ir.SymbolicDim("local_past_sequence_length")
        global_present = ir.SymbolicDim("global_present_sequence_length")
        local_present = ir.SymbolicDim("local_present_sequence_length")
        graph, builder = _make_graph("phi4flash")
        input_ids = builder.input("input_ids", ir.DataType.INT64, [batch, sequence])
        attention_mask = builder.input(
            "attention_mask",
            ir.DataType.INT64,
            [batch, "global_past_sequence_length + sequence_length"],
        )

        d_inner = config.hidden_size * config.mamba_expand
        past_states: list[tuple[ir.Value, ir.Value]] = []
        for slot in range(config.cache_slot_count):
            layer_type = config.layer_types[slot]
            if layer_type in {"mamba", "shared_memory_mamba"}:
                first = builder.input(
                    f"past_key_values.{slot}.conv_state",
                    config.dtype,
                    [batch, d_inner, config.mamba_d_conv],
                )
                second = builder.input(
                    f"past_key_values.{slot}.ssm_state",
                    config.dtype,
                    [batch, d_inner, config.mamba_d_state],
                )
            elif layer_type == "local_differential_attention":
                first = builder.input(
                    f"past_key_values.{slot}.key",
                    config.dtype,
                    [batch, config.num_key_value_heads, local_past, config.head_dim],
                )
                second = builder.input(
                    f"past_key_values.{slot}.value",
                    config.dtype,
                    [batch, config.num_key_value_heads, local_past, config.head_dim],
                )
            elif layer_type == "global_differential_attention":
                first = builder.input(
                    f"past_key_values.{slot}.key",
                    config.dtype,
                    [batch, config.num_key_value_heads, global_past, config.head_dim],
                )
                second = builder.input(
                    f"past_key_values.{slot}.value",
                    config.dtype,
                    [batch, config.num_key_value_heads, global_past, config.head_dim],
                )
            else:
                raise AssertionError(f"Unexpected Phi4Flash cache layer {slot}: {layer_type}")
            past_states.append((first, second))

        logits, present_states, captured_hidden_states = module(
            builder.op,
            input_ids,
            attention_mask,
            tuple(past_states),
        )
        if len(present_states) != config.cache_slot_count:
            raise ValueError(
                "Phi4Flash model returned an invalid cache slot count: "
                f"expected {config.cache_slot_count}, got {len(present_states)}"
            )
        logits.shape = ir.Shape([batch, sequence, config.vocab_size])
        logits.type = ir.TensorType(config.dtype)
        builder.add_output(logits, "logits")
        for layer_idx, hidden_states in captured_hidden_states:
            hidden_states.shape = ir.Shape([batch, sequence, config.hidden_size])
            builder.add_output(hidden_states, f"hidden_states.{layer_idx}")

        for slot, (first, second) in enumerate(present_states):
            layer_type = config.layer_types[slot]
            if layer_type in {"mamba", "shared_memory_mamba"}:
                first.shape = ir.Shape([batch, d_inner, config.mamba_d_conv])
                second.shape = ir.Shape([batch, d_inner, config.mamba_d_state])
                names = ("conv_state", "ssm_state")
            elif layer_type == "local_differential_attention":
                first.shape = ir.Shape(
                    [batch, config.num_key_value_heads, local_present, config.head_dim]
                )
                second.shape = ir.Shape(
                    [batch, config.num_key_value_heads, local_present, config.head_dim]
                )
                names = ("key", "value")
            elif layer_type == "global_differential_attention":
                first.shape = ir.Shape(
                    [batch, config.num_key_value_heads, global_present, config.head_dim]
                )
                second.shape = ir.Shape(
                    [batch, config.num_key_value_heads, global_present, config.head_dim]
                )
                names = ("key", "value")
            else:
                raise AssertionError(f"Unexpected Phi4Flash present layer {slot}: {layer_type}")
            first.type = ir.TensorType(config.dtype)
            second.type = ir.TensorType(config.dtype)
            builder.add_output(first, f"present.{slot}.{names[0]}")
            builder.add_output(second, f"present.{slot}.{names[1]}")

        model = _make_model(graph)
        self._register_functions(model, config)
        model.metadata_props["mobius.cache_abi"] = (
            f"slots 0..{config.cache_slot_count - 2} even:conv_state[B,{d_inner},"
            f"{config.mamba_d_conv}],ssm_state[B,{d_inner},{config.mamba_d_state}]; "
            f"slots 1..{config.cache_slot_count - 3} odd:key,value[B,"
            f"{config.num_key_value_heads},<={config.local_attention_window},{config.head_dim}]; "
            f"slot {config.cache_slot_count - 1}:key,value[B,{config.num_key_value_heads},T,"
            f"{config.head_dim}]"
        )
        model.metadata_props["mobius.state_semantics"] = (
            "dynamic chronological local tail and global append; recurrent state is required "
            "for both prefill and decode; layer-16 memory and layer-17 shared KV are transient"
        )
        model.metadata_props["mobius.runtime_support"] = (
            "custom ONNX Runtime Session only; ONNX Runtime GenAI mixed SambaY cache, "
            "beam reorder, rewind, assisted/speculative decoding, and static/paged caches unsupported"
        )
        model.metadata_props["mobius.execution_provider_feasibility"] = (
            "CUDA: supported when ONNX Runtime provides bf16 Attention, GroupQueryAttention, and Scan kernels; "
            "CPU and DirectML: unsupported because the pinned source requires bf16 differential "
            "attention; MLX: not an ONNX Runtime execution provider"
        )
        model.metadata_props["mobius.reference_precision"] = (
            "pinned remote source casts differential-attention Q/K/V to bfloat16"
        )
        model.metadata_props["mobius.reference_kernel_audit"] = (
            "unmasked flash-attn semantics use compact ORT GroupQueryAttention local windows and "
            "native causal Attention; source-faithful left-padding uses a standard ONNX BxQxK bool "
            "mask because ORT has no ragged dynamic-cache ABI, so padded 64K+ prompts may be infeasible; "
            "causal-conv1d is standard depthwise Conv; mamba selective_scan is "
            "the attached portable LinearAttention Scan function body"
        )
        model.metadata_props["mobius.quantization_assessment"] = (
            "Olive weight-only quantization is unvalidated for this heterogeneous dynamic-cache "
            "graph; no quantized runtime package is claimed"
        )
        model.metadata_props["mobius.semantic_reference_revision"] = (
            "microsoft/Phi-4-mini-flash-reasoning@1dff8163d28ec880ca2411c474ddc0a927792810"
        )
        return ModelPackage({"model": model}, config=config)

    @staticmethod
    def _register_functions(model: ir.Model, config: Phi4FlashConfig) -> None:
        """Attach the portable standard-ONNX reference body for Mamba recurrence."""
        from mobius.functions import linear_attention

        function = linear_attention(
            q_num_heads=config.hidden_size * config.mamba_expand,
            kv_num_heads=config.hidden_size * config.mamba_expand,
            update_rule="gated",
            scale=1.0,
            stash_type=ir.DataType.FLOAT,
        )
        model.functions[function.identifier()] = function
        model.graph.opset_imports["com.microsoft"] = 1
