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
    ``task.build()``.

    Args:
        **components: Keyword arguments mapping output name → module attribute
            name.  For example, ``vision="vision_encoder"`` means the task
            expects ``module.vision_encoder`` and will store the result as
            ``package["vision"]``.
    """

    def __init__(self, **components: str) -> None:
        # output_name -> module attribute name
        self._components: dict[str, str] = dict(components)

    def validate(self, module: nn.Module, task_name: str) -> None:
        """Check that all required sub-module attributes exist on *module*.

        Attribute names may use dot notation to reference nested attributes,
        e.g. ``"model.encoder"`` checks that ``module.model.encoder`` exists.

        Args:
            module: The module passed to ``task.build()``.
            task_name: Name of the task class (for the error message).

        Raises:
            TypeError: If any required attribute is absent from *module*, with
                a message that lists every missing component and the attribute
                name expected for each.
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
    #: Unrecognised keys default to ``"decoder"``.
    model_roles: ClassVar[dict[str, str]] = {"model": "decoder"}

    #: Optional component spec for multi-component tasks.  When set,
    #: :meth:`_validate_components` checks that all declared attributes
    #: exist on the module before building begins.
    components: ClassVar[ComponentSpec | None] = None

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

    def _validate_components(self, module: nn.Module) -> None:
        """Validate that *module* exposes all components declared in ``self.components``.

        No-op when ``self.components`` is ``None``.  Multi-component tasks
        should call this at the start of ``build()`` so that missing
        sub-module attributes are caught with a helpful error rather than an
        ``AttributeError`` deep inside the build.

        Raises:
            TypeError: If any required sub-module attribute is absent from
                *module*.
        """
        if self.components is not None:
            self.components.validate(module, type(self).__name__)
