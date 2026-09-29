# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Export tasks for the CLM-v0.1-8B and Kev-4B decision models."""

from __future__ import annotations

import onnx_ir as ir

from mobius._configs import BaseModelConfig
from mobius.models.decision import (
    CLM_PROVENANCE,
    KEV_PROVENANCE,
    CLMProjectionHead,
    CLMScorer,
    KevPointerHead,
    provenance_json,
)
from mobius.tasks._base import (
    ComponentConfig,
    ComponentRole,
    ComponentSpec,
    MultiComponentModelTask,
    _make_graph,
    _make_model,
)
from mobius.tasks._causal_lm import CausalLMTask, HybridCausalLMTask


def _stamp(model: ir.Model, provenance, contract: str) -> ir.Model:
    """Attach reproducibility and component-contract metadata to a model."""
    model.metadata_props["mobius.provenance"] = provenance_json(provenance)
    model.metadata_props["mobius.decision_contract"] = contract
    return model


class CLMTask(MultiComponentModelTask):
    """Export Qwen3 encoder, two projection heads, and scorer as one package."""

    components = ComponentSpec(
        encoder=ComponentConfig("encoder", ComponentRole.ENCODER),
        state_head=ComponentConfig("state_head", ComponentRole.HEAD),
        action_head=ComponentConfig("action_head", ComponentRole.HEAD),
        scorer=ComponentConfig("scorer", ComponentRole.HEAD),
    )

    def __init__(self, provenance=CLM_PROVENANCE):
        """Create the task with pinned or explicitly unpinned provenance."""
        self.provenance = provenance

    def build(self, module, config):
        """Build one package while honoring provenance selected by the module."""
        self.provenance = getattr(module, "provenance", self.provenance)
        return super().build(module, config)

    def build_component(self, name, component, module, config):
        """Build a headless encoder, projection head, or grouped scorer graph."""
        if name == "encoder":
            package = CausalLMTask().build(module, config)
            model = package["model"]
            model.graph.name = name
            model.graph.outputs[0].name = "token_hidden_states"
            return _stamp(
                model,
                self.provenance,
                "qwen3-token-hidden-states;caller-selects-last-token",
            )
        if isinstance(module, CLMProjectionHead):
            graph, builder = _make_graph(name)
            embeddings = builder.input(
                "embeddings",
                config.dtype,
                [ir.SymbolicDim("items"), config.hidden_size],
            )
            builder.add_output(module(builder.op, embeddings), "projections")
            return _stamp(
                _make_model(graph), self.provenance, "l2-normalized-projection"
            )
        if isinstance(module, CLMScorer):
            graph, builder = _make_graph(name)
            states = builder.input(
                "state_projections",
                config.dtype,
                [ir.SymbolicDim("questions"), ir.SymbolicDim("projection_dim")],
            )
            actions = builder.input(
                "action_projections",
                config.dtype,
                [ir.SymbolicDim("candidates"), ir.SymbolicDim("projection_dim")],
            )
            temperature = builder.input("temperature", config.dtype, [])
            candidate_owners = builder.input(
                "candidate_owners",
                ir.DataType.INT64,
                [ir.SymbolicDim("candidates")],
            )
            logits, probabilities = module(
                builder.op, states, actions, candidate_owners, temperature
            )
            builder.add_output(logits, "logits")
            builder.add_output(probabilities, "probabilities")
            return _stamp(_make_model(graph), self.provenance, "scaled-cosine")
        raise TypeError(f"unsupported CLM component {name!r}")


class KevTask(MultiComponentModelTask):
    """Export the Qwen3.5 hybrid backbone and grouped pointer head together."""

    components = ComponentSpec(
        backbone=ComponentConfig("backbone", ComponentRole.BACKBONE),
        pointer_head=ComponentConfig("pointer_head", ComponentRole.HEAD),
    )

    def build_component(
        self,
        name: str,
        component: ComponentConfig,
        module: object,
        config: BaseModelConfig,
    ) -> ir.Model:
        """Build the headless hybrid backbone or Kev pointer graph.

        The backbone emits token-level hidden states. The pointer component
        consumes per-row decide/option indices and returns flat logits plus a
        stable probability distribution within each question.
        """
        if name == "backbone":
            package = HybridCausalLMTask().build(module, config)
            model = package["model"]
            model.graph.name = name
            model.graph.outputs[0].name = "token_hidden_states"
            return _stamp(model, KEV_PROVENANCE, "one-causal-row-per-question")
        if not isinstance(module, KevPointerHead):
            raise TypeError("Kev pointer_head has an unexpected module type")
        graph, builder = _make_graph(name)
        hidden_states = builder.input(
            "hidden_states",
            config.dtype,
            [
                ir.SymbolicDim("questions"),
                ir.SymbolicDim("sequence_length"),
                config.hidden_size,
            ],
        )
        decide_indices = builder.input(
            "decide_indices", ir.DataType.INT64, [ir.SymbolicDim("questions")]
        )
        option_indices = builder.input(
            "option_indices", ir.DataType.INT64, [ir.SymbolicDim("options")]
        )
        option_owners = builder.input(
            "option_owners", ir.DataType.INT64, [ir.SymbolicDim("options")]
        )
        logits, probabilities = module(
            builder.op,
            hidden_states,
            decide_indices,
            option_indices,
            option_owners,
        )
        builder.add_output(logits, "logits")
        builder.add_output(probabilities, "probabilities")
        return _stamp(_make_model(graph), KEV_PROVENANCE, "grouped-pointer-softmax")
