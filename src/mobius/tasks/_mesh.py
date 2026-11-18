# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Task contract for parametric articulated mesh generation."""

from __future__ import annotations

from typing import ClassVar

import onnx_ir as ir

from mobius._model_package import ModelPackage
from mobius.tasks._base import ModelTask, _make_graph, _make_model


class MeshGenerationTask(ModelTask):
    """Generate full 3D vertex positions from batched articulation parameters."""

    model_roles: ClassVar[dict[str, str]] = {"model": "head"}

    def build(self, module, config) -> ModelPackage:
        config.validate()
        graph, builder = _make_graph()
        inputs = [
            builder.input(name, dtype=ir.DataType.FLOAT, shape=shape)
            for name, shape in (
                ("identity", ["batch", config.identity_dim]),
                ("expression", ["batch", config.expression_dim]),
                ("rotations", ["batch", len(config.joint_parent_indices), 3]),
                ("translation", ["batch", 3]),
            )
        ]
        vertices = module(builder.op, *inputs)
        vertices.type = ir.TensorType(ir.DataType.FLOAT)
        vertices.shape = ir.Shape(["batch", config.num_vertices, 3])
        builder.add_output(vertices, "vertices")
        return ModelPackage({"model": _make_model(graph)}, config=config)
