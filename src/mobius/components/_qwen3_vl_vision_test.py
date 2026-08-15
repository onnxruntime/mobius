# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import onnx_ir as ir

from mobius._testing import count_op_type, create_test_builder, create_test_input
from mobius.components._qwen3_vl_vision import Qwen3VLVisionModel

_PATCH_DIM = 3 * 2 * 16 * 16


def _build_vision_graph() -> ir.Graph:
    module = Qwen3VLVisionModel(
        depth=1,
        hidden_size=32,
        intermediate_size=64,
        num_heads=4,
        patch_size=16,
        temporal_patch_size=2,
        in_channels=3,
        out_hidden_size=64,
        spatial_merge_size=2,
        num_position_embeddings=16,
        deepstack_visual_indexes=[],
    )
    builder, op, graph = create_test_builder()
    pixel_values = create_test_input(
        builder,
        "pixel_values",
        ["total_patches", _PATCH_DIM],
        dtype=ir.DataType.FLOAT,
    )
    grid_thw = create_test_input(
        builder,
        "grid_thw",
        ["num_media", 3],
        dtype=ir.DataType.INT64,
    )
    image_features = module(op, pixel_values, grid_thw)[0]
    image_features.name = "image_features"
    graph.outputs.append(image_features)
    return graph


def test_packed_coordinates_are_linear_and_shared():
    graph = _build_vision_graph()

    # Media ownership uses boundary scatter + prefix sum, not a quadratic
    # [total_patches, num_media] comparison matrix.
    assert count_op_type(graph, "ScatterElements") == 1
    # The only remaining comparison belongs to the single attention block.
    assert count_op_type(graph, "GreaterOrEqual") == 1

    # The two row/column coordinate values each feed both interpolation (Cast)
    # and rotary IDs (Unsqueeze), proving the coordinate graph is emitted once.
    shared_coordinates = []
    for node in graph:
        if node.op_type != "Add":
            continue
        for output in node.outputs:
            consumer_types = {consumer.op_type for consumer, _ in output.uses()}
            if {"Cast", "Unsqueeze"} <= consumer_types:
                shared_coordinates.append(output)
    assert len(shared_coordinates) == 2
