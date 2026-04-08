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
    """

    #: Maps package key → optimization role for each model produced by this task.
    #: The role controls which fusion passes run (e.g. only ``"decoder"`` gets
    #: GQA fusion). Override in subclasses that produce non-decoder outputs.
    #: Unrecognised keys default to ``"decoder"``.
    model_roles: ClassVar[dict[str, str]] = {"model": "decoder"}

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
