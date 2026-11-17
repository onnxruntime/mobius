# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Graph task for Mistral4 GGUF's per-layer latent K-only cache."""

from __future__ import annotations

import onnx_ir as ir

from mobius._configs import ArchitectureConfig
from mobius._model_package import ModelPackage
from mobius.tasks._base import ModelTask, _make_graph, _make_model


class Mistral4GGUFCausalLMTask(ModelTask):
    """Build Mistral4 with one ``[latent | RoPE]`` cache tensor per layer."""

    def __init__(self, *, static_cache: bool = False):
        if static_cache:
            raise ValueError(
                "Mistral4 GGUF static cache is not implemented; the dedicated task "
                "currently owns only the dynamic latent K-only cache contract"
            )

    def build(self, module, config: ArchitectureConfig) -> ModelPackage:
        width_fn = getattr(module, "latent_cache_width", None)
        if not callable(width_fn):
            raise TypeError(
                f"{type(module).__name__} must implement latent_cache_width() "
                "for Mistral4GGUFCausalLMTask"
            )
        cache_width = int(width_fn())
        if cache_width <= 0:
            raise ValueError("Mistral4 latent cache width must be positive")

        graph, builder = _make_graph("mistral4_gguf")
        batch = "batch"
        sequence = "sequence_len"
        past_sequence = "past_sequence_len"
        input_ids = builder.input(
            "input_ids",
            dtype=ir.DataType.INT64,
            shape=[batch, sequence],
        )
        attention_mask = builder.input(
            "attention_mask",
            dtype=ir.DataType.INT64,
            shape=[batch, "past_sequence_len + sequence_len"],
        )
        position_ids = builder.input(
            "position_ids",
            dtype=ir.DataType.INT64,
            shape=[batch, sequence],
        )
        past_key_values = [
            builder.input(
                f"past_key_values.{layer}.key",
                dtype=config.dtype,
                shape=[batch, 1, past_sequence, cache_width],
            )
            for layer in range(config.num_hidden_layers)
        ]

        logits, present_key_values = module(
            builder.op,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
        )
        builder.add_output(logits, "logits")
        total_sequence = "past_sequence_len + sequence_len"
        for layer, present in enumerate(present_key_values):
            present.type = ir.TensorType(config.dtype)
            present.shape = ir.Shape([batch, 1, total_sequence, cache_width])
            builder.add_output(present, f"present.{layer}.key")

        model = _make_model(graph)
        model.metadata_props["mobius.cache_abi"] = (
            "mistral4:per-layer=latent_key[normalized_kv|rotated_rope];"
            "dynamic-concat;no-value-cache"
        )
        model.metadata_props["mobius.runtime_support"] = (
            "deferred: real-weight parity and runtime packaging are not evidenced"
        )
        return ModelPackage({"model": model}, config=config)
