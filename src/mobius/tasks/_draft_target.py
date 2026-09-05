# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Target decoder and shared-weight bridge graphs for speculative drafting."""

from __future__ import annotations

import onnx_ir as ir
from onnxscript import nn

from mobius._configs import ArchitectureConfig
from mobius._model_package import ModelPackage
from mobius.tasks._base import ModelTask, _make_graph, _make_model
from mobius.tasks._causal_lm import CausalLMTask


class DraftTargetCausalLMTask(ModelTask):
    """Build a target decoder plus token-embedding and LM-head bridge graphs."""

    @staticmethod
    def bind_shared_initializers(package: ModelPackage) -> None:
        """Bind bridge graph weight inputs to the initialized decoder values."""
        decoder_initializers = package["model"].graph.initializers
        for component_name in ("embedding", "lm_head"):
            graph = package[component_name].graph
            bound: dict[str, ir.Value] = {}
            for node in graph:
                for input_index, value in enumerate(node.inputs):
                    if value is None or value.name is None:
                        continue
                    name = value.name
                    if name not in decoder_initializers:
                        continue
                    source = decoder_initializers[name]
                    initializer = bound.get(name)
                    if initializer is None:
                        initializer = ir.Value(
                            name=source.name,
                            type=source.type,
                            shape=source.shape,
                            const_value=source.const_value,
                        )
                        graph.initializers[initializer.name] = initializer
                        bound[initializer.name] = initializer
                    node.replace_input_with(input_index, initializer)

    def build(self, module: nn.Module, config: ArchitectureConfig) -> ModelPackage:
        decoder = CausalLMTask().build(module, config)["model"]
        text_model = getattr(module, "model", None)
        embedding = getattr(text_model, "embed_tokens", None)
        lm_head = getattr(module, "lm_head", None)
        if embedding is None or lm_head is None:
            raise TypeError(
                "DraftTargetCausalLMTask requires module.model.embed_tokens and module.lm_head"
            )

        graph, builder = _make_graph(name="target_embedding")
        input_ids = builder.input(
            "input_ids",
            dtype=ir.DataType.INT64,
            shape=["batch", "sequence"],
        )
        inputs_embeds = embedding(builder.op, input_ids)
        inputs_embeds.shape = ir.Shape(["batch", "sequence", config.hidden_size])
        builder.add_output(inputs_embeds, "inputs_embeds")

        head_graph, head_builder = _make_graph(name="target_lm_head")
        hidden_states = head_builder.input(
            "hidden_states",
            dtype=config.dtype,
            shape=["batch", "sequence", config.hidden_size],
        )
        logits = lm_head(head_builder.op, hidden_states)
        logits.shape = ir.Shape(["batch", "sequence", config.vocab_size])
        head_builder.add_output(logits, "logits")

        return ModelPackage(
            {
                "model": decoder,
                "embedding": _make_model(graph),
                "lm_head": _make_model(head_graph),
            },
            config=config,
        )
