# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Base class for model tasks and shared graph construction helpers."""

from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, ClassVar

import onnx_ir as ir
from onnxscript import GraphBuilder, nn

import mobius
from mobius._configs import BaseModelConfig
from mobius._constants import OPSET_VERSION
from mobius._model_package import ModelPackage

if TYPE_CHECKING:
    from mobius._component_manifest import ComponentManifest


class ComponentRole(str, Enum):
    """Neutral roles for non-generative package components."""

    BACKBONE = "backbone"
    ENCODER = "encoder"
    HEAD = "head"


@dataclasses.dataclass(frozen=True)
class ComponentConfig:
    """Configuration for one named component of a multi-component task.

    ``role`` accepts a :class:`ComponentRole` or one of its string values:
    ``"backbone"``, ``"encoder"``, and ``"head"``.
    """

    module_attribute_path: str
    role: str | ComponentRole | None = None

    def __post_init__(self) -> None:
        """Validate and normalize the dotted path and optional neutral role."""
        path = self.module_attribute_path
        if not isinstance(path, str):
            raise TypeError("component module_attribute_path must be a string")
        if not path or any(not part or not part.strip() for part in path.split(".")):
            raise ValueError(
                "component module_attribute_path must be a non-empty dotted attribute path"
            )
        role = self.role
        if role is None:
            return
        if not isinstance(role, (str, ComponentRole)):
            raise TypeError("component role must be a ComponentRole or string")
        try:
            normalized_role = ComponentRole(role)
        except ValueError:
            supported = ", ".join(role.value for role in ComponentRole)
            raise ValueError(
                f"unsupported component role {role!r}; expected one of: {supported}"
            ) from None
        object.__setattr__(self, "role", normalized_role.value)


class ComponentSpec:
    """Declares which sub-module attributes a multi-component task requires.

    Used by multi-component tasks (e.g. :class:`VisionLanguageTask`) to
    validate that a module exposes the expected sub-module attributes before
    building begins.  This produces a clear :exc:`TypeError` instead of the
    cryptic ``AttributeError`` that would otherwise surface deep inside
    ``build()``.

    Map output model names to the module attribute that builds each component.
    A :class:`ComponentConfig` additionally declares a neutral component role::

        ComponentSpec(
            encoder=ComponentConfig("encoder", ComponentRole.ENCODER),
            classifier=ComponentConfig("heads.classifier", ComponentRole.HEAD),
        )

    The keys are the names used in the output :class:`ModelPackage`; the
    values are attribute names or component configurations for the ``nn.Module``
    passed to ``task.build()``. Dot notation is supported for nested attributes.

    Args:
        **components: Keyword arguments mapping output name to a module
            attribute name or :class:`ComponentConfig`.
    """

    def __init__(self, **components: str | ComponentConfig) -> None:
        """Store component declarations, normalizing plain paths to configs."""
        self._components = {
            name: value if isinstance(value, ComponentConfig) else ComponentConfig(value)
            for name, value in components.items()
        }

    def validate(self, module: nn.Module, task_name: str) -> None:
        """Check that all required sub-module attributes exist on *module*.

        Args:
            module: The module passed to ``task.build()``.
            task_name: Name of the task class (for the error message).

        Raises:
            TypeError: If any required attribute is absent from *module*.
        """

        def _has_nested(obj: object, dotted: str) -> bool:
            """Return whether *obj* exposes every segment of a dotted path."""
            for part in dotted.split("."):
                if not hasattr(obj, part):
                    return False
                obj = getattr(obj, part)
            return True

        missing = [
            (output_name, component.module_attribute_path)
            for output_name, component in self._components.items()
            if not _has_nested(module, component.module_attribute_path)
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
        return (
            (name, component.module_attribute_path)
            for name, component in self._components.items()
        )

    def configs(self):
        """Iterate over ``(output_name, component_config)`` pairs."""
        return self._components.items()

    def roles(self) -> dict[str, str]:
        """Return roles explicitly declared by component configurations."""
        return {
            name: str(component.role)
            for name, component in self._components.items()
            if component.role is not None
        }

    def resolve(self, module: nn.Module, name: str) -> object:
        """Resolve a declared component module from the root module."""
        value: object = module
        for part in self._components[name].module_attribute_path.split("."):
            value = getattr(value, part)
        return value

    def keys(self):
        """Return the output model names declared by this spec."""
        return self._components.keys()

    def __contains__(self, item: str) -> bool:
        """Return ``True`` if *item* is a declared output model name."""
        return item in self._components

    def __repr__(self) -> str:
        """Return a constructor-like representation of this component spec."""
        parts = ", ".join(
            f"{name}={component.module_attribute_path!r}"
            if component.role is None
            else f"{name}={component!r}"
            for name, component in self._components.items()
        )
        return f"ComponentSpec({parts})"


def _make_graph(
    name: str = "main_graph",
) -> tuple[ir.Graph, GraphBuilder]:
    """Create an empty graph and its builder.

    Inputs should be added after creation via ``builder.input()``.
    Outputs should be registered via ``builder.add_output()``.

    Returns:
        ``(graph, builder)`` — call ``builder.op`` to get the op handle.
    """
    graph = ir.Graph(
        [],
        [],
        nodes=[],
        name=name,
        opset_imports={"": OPSET_VERSION, "com.microsoft": 1},
    )
    return graph, GraphBuilder(graph)


def _make_model(graph: ir.Graph) -> ir.Model:
    """Create an ``ir.Model`` with standard producer metadata."""
    model = ir.Model(graph, ir_version=12)
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

    def component_manifest(
        self,
        *,
        module_class: type | None = None,
        model_type: str | None = None,
        hf_config: object | None = None,
    ) -> ComponentManifest:
        """Resolve canonical metadata for every component produced by this task."""
        from mobius._component_manifest import resolve_component_manifest

        return resolve_component_manifest(
            self,
            module_class=module_class,
            model_type=model_type,
            hf_config=hf_config,
        )

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


class MultiComponentModelTask(ModelTask):
    """Base task for a named backbone/encoder and one or more named heads.

    Subclasses declare :attr:`components` with :class:`ComponentConfig` values
    and implement :meth:`build_component`. The common build implementation
    validates the layout and returns every graph in one :class:`ModelPackage`.
    """

    model_roles: ClassVar[dict[str, str]] = {}
    components: ClassVar[ComponentSpec | None] = None

    def __init_subclass__(cls, **kwargs) -> None:
        """Derive optimization roles from the subclass component declaration."""
        super().__init_subclass__(**kwargs)
        components = cls.components
        if components is None:
            return
        declared_roles = components.roles()
        explicit_roles = cls.__dict__.get("model_roles")
        if explicit_roles is not None:
            declared_roles.update(explicit_roles)
        cls.model_roles = declared_roles

    def build(
        self,
        module: nn.Module,
        config: BaseModelConfig,
    ) -> ModelPackage:
        """Build every declared component into one atomic model package.

        The declaration must contain exactly one backbone or encoder and at
        least one head. Component paths are validated before graph creation;
        each :meth:`build_component` result must be an ``ir.Model``.
        """
        components = self.components
        if components is None:
            raise TypeError(f"{type(self).__name__} must declare components")
        roles = components.roles()
        backbone_names = [
            name
            for name, role in roles.items()
            if role in {ComponentRole.BACKBONE.value, ComponentRole.ENCODER.value}
        ]
        head_names = [
            name for name, role in roles.items() if role == ComponentRole.HEAD.value
        ]
        if len(backbone_names) != 1 or not head_names:
            raise ValueError(
                f"{type(self).__name__} components must declare exactly one "
                "backbone/encoder and at least one head"
            )

        self._validate_components(module)
        models: dict[str, ir.Model] = {}
        for name, component in components.configs():
            model = self.build_component(
                name,
                component,
                components.resolve(module, name),
                config,
            )
            if not isinstance(model, ir.Model):
                raise TypeError(
                    f"{type(self).__name__}.build_component({name!r}) "
                    "must return an onnx_ir.Model"
                )
            models[name] = model
        return ModelPackage(models, config=config)

    @abstractmethod
    def build_component(
        self,
        name: str,
        component: ComponentConfig,
        module: object,
        config: BaseModelConfig,
    ) -> ir.Model:
        """Build one graph for a declared component."""
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
    deepstack: bool = False,
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
        deepstack: If ``True`` (Qwen3-VL family with
            ``deepstack_visual_indexes``), adds a ``per_layer_inputs`` input
            ``[batch, seq_len, D * hidden_size]`` that the decoder reshapes and
            injects into its first ``D`` layers.

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

    graph, builder = _make_graph()
    inputs_embeds = builder.input(
        "inputs_embeds",
        dtype=config.dtype,
        shape=[batch, seq_len, config.hidden_size],
    )
    attention_mask = builder.input(
        "attention_mask",
        dtype=ir.DataType.INT64,
        shape=[batch, "past_seq_len + seq_len"],
    )
    # MRoPE: 3D position IDs (temporal, height, width) — shape [3, batch, seq_len]
    # Standard: shape [batch, seq_len]
    position_ids = builder.input(
        "position_ids",
        dtype=ir.DataType.INT64,
        shape=[3, batch, seq_len] if mrope else [batch, seq_len],
    )

    # DeepStack intermediate vision features, pre-scattered and flattened by
    # the embedding model. Shape [batch, seq_len, D * hidden_size], matching
    # ORT GenAI's generic per_layer_inputs contract.
    per_layer_inputs = None
    num_deepstack = len(getattr(config, "deepstack_visual_indexes", None) or [])
    if deepstack and num_deepstack > 0:
        per_layer_inputs = builder.input(
            "per_layer_inputs",
            dtype=config.dtype,
            shape=[batch, seq_len, num_deepstack * config.hidden_size],
        )

    if hybrid:
        past_key_values = _make_hybrid_cache_inputs(
            builder,
            config,
            config.dtype,
            batch,
            past_seq_len,
        )
    else:
        past_key_values = _make_kv_cache_inputs(
            builder,
            config.num_hidden_layers,
            config.num_key_value_heads,
            config.head_dim,
            config.dtype,
            batch,
            past_seq_len,
        )

    decoder_kwargs = {}
    if per_layer_inputs is not None:
        decoder_kwargs["per_layer_inputs"] = per_layer_inputs

    logits, present_key_values = decoder(
        builder.op,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        **decoder_kwargs,
    )

    builder.add_output(logits, "logits")

    if hybrid:
        _register_hybrid_cache_outputs(
            builder,
            present_key_values,
            config.layer_types or [],
        )
        model = _make_model(graph)
        _register_linear_attention_functions(model, config)
        return model
    else:
        _register_kv_cache_outputs(builder, present_key_values)
        return _make_model(graph)


def build_embedding_from_features(
    embedding,
    config: BaseModelConfig,
    *,
    feature_name: str,
    feature_dim: int,
    deepstack: bool = False,
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
        deepstack: If ``True`` (Qwen3-VL family with
            ``deepstack_visual_indexes``), expects ``image_features`` to pack
            the final and intermediate maps as
            ``[num_feature_tokens, (D + 1) * feature_dim]`` and emits a second
            ``per_layer_inputs`` output
            ``[batch, seq_len, D * hidden_size]``.

    Returns:
        A built :class:`ir.Model` for the embedding model.
    """
    batch = ir.SymbolicDim("batch")
    seq_len = ir.SymbolicDim("sequence_len")
    num_feature_tokens = ir.SymbolicDim("num_feature_tokens")

    graph, builder = _make_graph(name="embedding")
    input_ids = builder.input(
        "input_ids",
        dtype=ir.DataType.INT64,
        shape=[batch, seq_len],
    )
    num_deepstack = len(getattr(config, "deepstack_visual_indexes", None) or [])
    has_deepstack = deepstack and num_deepstack > 0
    packed_feature_dim = (num_deepstack + 1) * feature_dim if has_deepstack else feature_dim
    packed_features = builder.input(
        feature_name,
        dtype=config.dtype,
        shape=[num_feature_tokens, packed_feature_dim],
    )

    embedding_kwargs = {}
    if has_deepstack:
        features = builder.op.Slice(
            packed_features,
            builder.op.Constant(value_ints=[0]),
            builder.op.Constant(value_ints=[feature_dim]),
            builder.op.Constant(value_ints=[1]),
        )
        deepstack_flat = builder.op.Slice(
            packed_features,
            builder.op.Constant(value_ints=[feature_dim]),
            builder.op.Constant(value_ints=[packed_feature_dim]),
            builder.op.Constant(value_ints=[1]),
        )
        deepstack_features = builder.op.Transpose(
            builder.op.Reshape(
                deepstack_flat,
                builder.op.Constant(value_ints=[0, num_deepstack, feature_dim]),
            ),
            perm=[1, 0, 2],
        )
        embedding_kwargs["deepstack_features"] = deepstack_features
    else:
        features = packed_features
    embedding_kwargs[feature_name] = features

    outputs = embedding(
        builder.op,
        input_ids=input_ids,
        **embedding_kwargs,
    )

    if isinstance(outputs, tuple):
        inputs_embeds, deepstack_embeds = outputs
        builder.add_output(inputs_embeds, "inputs_embeds")
        per_layer_inputs = builder.op.Reshape(
            builder.op.Transpose(deepstack_embeds, perm=[1, 2, 0, 3]),
            builder.op.Constant(value_ints=[0, 0, num_deepstack * config.hidden_size]),
        )
        builder.add_output(per_layer_inputs, "per_layer_inputs")
    else:
        builder.add_output(outputs, "inputs_embeds")
    return _make_model(graph)
