# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Base class for model tasks and shared graph construction helpers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

import onnx_ir as ir
from onnxscript import nn
from onnxscript._internal.builder import GraphBuilder

import mobius
from mobius._configs import BaseModelConfig
from mobius._constants import OPSET_VERSION
from mobius._model_package import ModelPackage


class ComponentSpec:
    """Declares which sub-module attributes a multi-component task requires.

    Used by multi-component tasks (e.g. :class:`VisionLanguageTask`) to
    validate that a module exposes the expected sub-module attributes before
    building begins.  This produces a clear :exc:`TypeError` instead of the
    cryptic ``AttributeError`` that would otherwise surface deep inside
    ``build()``.

    Map output model names to the module attribute that builds each component::

        ComponentSpec(
            decoder="decoder",
            vision="vision_encoder",
            embedding="embedding",
        )

    The keys are the names used in the output :class:`ModelPackage`; the
    values are the attribute names on the ``nn.Module`` passed to
    ``task.build()``.  Dot notation is supported for nested attributes
    (e.g. ``"model.encoder"``).

    Args:
        **components: Keyword arguments mapping output name → module attribute
            name.  For example, ``vision="vision_encoder"`` means the task
            expects ``module.vision_encoder`` and will store the result as
            ``package["vision"]``.
    """

    def __init__(self, **components: str) -> None:
        self._components: dict[str, str] = dict(components)

    def validate(self, module: nn.Module, task_name: str) -> None:
        """Check that all required sub-module attributes exist on *module*.

        Args:
            module: The module passed to ``task.build()``.
            task_name: Name of the task class (for the error message).

        Raises:
            TypeError: If any required attribute is absent from *module*.
        """

        def _has_nested(obj: object, dotted: str) -> bool:
            for part in dotted.split("."):
                if not hasattr(obj, part):
                    return False
                obj = getattr(obj, part)
            return True

        missing = [
            (output_name, attr_name)
            for output_name, attr_name in self._components.items()
            if not _has_nested(module, attr_name)
        ]
        if not missing:
            return
        lines = "\n".join(
            f"  '{output_name}' component expects module.{attr_name}"
            for output_name, attr_name in missing
        )
        raise TypeError(
            f"{task_name} requires sub-module attribute(s) that are missing "
            f"from {type(module).__name__}:\n{lines}\n"
            f"Ensure each attribute is assigned in the module's __init__()."
        )

    def items(self):
        """Iterate over ``(output_name, attribute_name)`` pairs."""
        return self._components.items()

    def __repr__(self) -> str:
        parts = ", ".join(f"{k}={v!r}" for k, v in self._components.items())
        return f"ComponentSpec({parts})"


def _make_graph(
    inputs: list[ir.Value],
    name: str = "main_graph",
) -> tuple[ir.Graph, GraphBuilder]:
    """Create an empty graph and its builder.

    Returns:
        ``(graph, builder)`` — call ``builder.op`` to get the op handle.
    """
    graph = ir.Graph(
        inputs,
        [],
        nodes=[],
        name=name,
        opset_imports={"": OPSET_VERSION, "com.microsoft": 1},
    )
    return graph, GraphBuilder(graph)


def _make_model(graph: ir.Graph) -> ir.Model:
    """Create an ``ir.Model`` with standard producer metadata."""
    model = ir.Model(graph, ir_version=11)
    model.producer_name = "mobius"
    model.producer_version = mobius.__version__
    return model


class ModelTask(ABC):
    """Abstract base defining how to wire a module into an ONNX graph.

    Subclass this to support new model tasks (e.g. feature extraction,
    sequence classification). Each task defines its own graph I/O contract.

    Multi-component tasks should declare a class-level :class:`ComponentSpec`
    and call :meth:`_validate_components` at the start of ``build()``::

        class MyMultiModelTask(ModelTask):
            components = ComponentSpec(decoder="decoder", vision="vision_encoder")

            def build(self, module, config):
                self._validate_components(module)
                ...
    """

    #: Maps package key → optimization role for each model produced by this task.
    #: The role controls which fusion passes run (e.g. only ``"decoder"`` gets
    #: GQA fusion). Override in subclasses that produce non-decoder outputs.
    #: Unrecognised keys fall back to ``_MODEL_ROLE_MAP`` in ``_builder.py``,
    #: then default to ``"decoder"``.
    model_roles: ClassVar[dict[str, str]] = {"model": "decoder"}

    #: Optional component spec for multi-component tasks.  When set,
    #: :meth:`_validate_components` checks that all declared attributes
    #: exist on the module before building begins.
    components: ClassVar[ComponentSpec | None] = None

    def _validate_components(self, module: nn.Module) -> None:
        """Validate that *module* exposes all attributes declared in :attr:`components`.

        Call at the start of :meth:`build` in multi-component tasks.

        Raises:
            TypeError: If any required sub-module attribute is missing.
        """
        if self.components is not None:
            self.components.validate(module, type(self).__name__)

    @abstractmethod
    def build(
        self,
        module: nn.Module,
        config: BaseModelConfig,
    ) -> ModelPackage:
        """Build a :class:`ModelPackage` for this task.

        Single-component tasks return a package with one ``"model"`` entry.
        Multi-component tasks (e.g. encoder-decoder) return a package with
        separate entries for each component.

        Args:
            module: The onnxscript.nn.Module to wire into the graph.
            config: Architecture configuration.

        Returns:
            A :class:`ModelPackage` containing the built model(s).
        """
        ...


# ---------------------------------------------------------------------------
# Shared graph-builder helpers for multi-component tasks
# ---------------------------------------------------------------------------


def build_decoder_from_embeds(
    decoder,
    config: BaseModelConfig,
    *,
    mrope: bool = False,
    hybrid: bool = False,
) -> ir.Model:
    """Build an ``inputs_embeds → logits + KV cache`` decoder ONNX graph.

    This is the shared implementation for the ``_build_decoder`` method that
    is common to :class:`VisionLanguageTask`, :class:`QwenVLTask`,
    :class:`HybridQwenVLTask`, :class:`SpeechLanguageTask`, and
    :class:`Phi4MMMultiModalTask`.

    Args:
        decoder: The decoder sub-module to invoke.
        config: Architecture configuration.
        mrope: If ``True``, uses 3D MRoPE position_ids
            ``[3, batch, seq_len]`` instead of the standard
            ``[batch, seq_len]``.
        hybrid: If ``True``, uses hybrid KV + DeltaNet cache inputs/outputs
            (for Qwen3.5-VL and similar).  Requires ``config.layer_types``.

    Returns:
        A built :class:`ir.Model` for the decoder.
    """
    # Import here rather than at module top to keep _base.py focused on base
    # class definitions.  _cache_utils does NOT import from _base.py, so there
    # is no circular dependency — this is purely a namespace-clarity choice.
    from mobius.tasks._cache_utils import (
        _make_hybrid_cache_inputs,
        _make_kv_cache_inputs,
        _register_hybrid_cache_outputs,
        _register_kv_cache_outputs,
        _register_linear_attention_functions,
    )

    batch = ir.SymbolicDim("batch")
    seq_len = ir.SymbolicDim("sequence_len")
    past_seq_len = ir.SymbolicDim("past_sequence_len")

    inputs_embeds = ir.Value(
        name="inputs_embeds",
        shape=ir.Shape([batch, seq_len, config.hidden_size]),
        type=ir.TensorType(config.dtype),
    )
    attention_mask = ir.Value(
        name="attention_mask",
        shape=ir.Shape([batch, "past_seq_len + seq_len"]),
        type=ir.TensorType(ir.DataType.INT64),
    )
    # MRoPE: 3D position IDs (temporal, height, width) — shape [3, batch, seq_len]
    # Standard: shape [batch, seq_len]
    position_ids = ir.Value(
        name="position_ids",
        shape=ir.Shape([3, batch, seq_len] if mrope else [batch, seq_len]),
        type=ir.TensorType(ir.DataType.INT64),
    )

    graph_inputs = [inputs_embeds, attention_mask, position_ids]

    if hybrid:
        cache_inputs, past_key_values = _make_hybrid_cache_inputs(
            config,
            config.dtype,
            batch,
            past_seq_len,
        )
    else:
        cache_inputs, past_key_values = _make_kv_cache_inputs(
            config.num_hidden_layers,
            config.num_key_value_heads,
            config.head_dim,
            config.dtype,
            batch,
            past_seq_len,
        )
    graph_inputs.extend(cache_inputs)

    graph, builder = _make_graph(graph_inputs)
    logits, present_key_values = decoder(
        builder.op,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
    )

    logits.name = "logits"
    graph.outputs.append(logits)

    if hybrid:
        _register_hybrid_cache_outputs(
            graph,
            present_key_values,
            config.layer_types or [],
        )
        model = _make_model(graph)
        _register_linear_attention_functions(model, config)
        return model
    else:
        _register_kv_cache_outputs(graph, present_key_values)
        return _make_model(graph)


def build_embedding_from_features(
    embedding,
    config: BaseModelConfig,
    *,
    feature_name: str,
    feature_dim: int,
) -> ir.Model:
    """Build an ``input_ids + features → inputs_embeds`` embedding ONNX graph.

    This is the shared implementation for ``_build_embedding`` in
    :class:`VisionLanguageTask` (image features) and
    :class:`SpeechLanguageTask` (audio features).

    Args:
        embedding: The embedding sub-module to invoke.
        config: Architecture configuration.
        feature_name: Name of the second input (e.g. ``"image_features"`` or
            ``"audio_features"``).
        feature_dim: Feature dimension for the second input's last axis.

    Returns:
        A built :class:`ir.Model` for the embedding model.
    """
    batch = ir.SymbolicDim("batch")
    seq_len = ir.SymbolicDim("sequence_len")
    num_feature_tokens = ir.SymbolicDim("num_feature_tokens")

    input_ids = ir.Value(
        name="input_ids",
        shape=ir.Shape([batch, seq_len]),
        type=ir.TensorType(ir.DataType.INT64),
    )
    features = ir.Value(
        name=feature_name,
        shape=ir.Shape([num_feature_tokens, feature_dim]),
        type=ir.TensorType(config.dtype),
    )

    graph, builder = _make_graph([input_ids, features], name="embedding")
    inputs_embeds = embedding(
        builder.op,
        input_ids=input_ids,
        **{feature_name: features},
    )

    inputs_embeds.name = "inputs_embeds"
    graph.outputs.append(inputs_embeds)
    return _make_model(graph)
