# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Graph-construction tests for the Qwen2.5-VL vision encoder packed path.

These exercise the EP-gated ``_emit_packed_mha`` branch (only taken when the
active EP advertises ``supports_packed_multi_head_attention``, e.g. CUDA),
which the default CPU build does not reach.
"""

from __future__ import annotations

import onnx_ir as ir

from mobius._build_context import build_context
from mobius._execution_providers import ep_registry
from mobius._testing import count_op_type, create_test_builder, create_test_input
from mobius.components._qwen25_vl_vision import Qwen25VLVisionModel

_PATCH_DIM = 3 * 2 * 14 * 14  # in_channels * temporal_patch * patch * patch


def _build_vision_graph(dtype: ir.DataType) -> ir.Graph:
    """Build the Qwen2.5-VL vision encoder graph under the CUDA EP.

    ``fullatt_block_indexes=[1]`` makes block 0 windowed and block 1 full, so
    both attention variants flow through the packed path.
    """
    module = Qwen25VLVisionModel(
        depth=2,
        hidden_size=64,
        intermediate_size=128,
        num_heads=2,
        patch_size=14,
        temporal_patch_size=2,
        in_channels=3,
        out_hidden_size=64,
        spatial_merge_size=2,
        fullatt_block_indexes=[1],
        window_size=112,
    )
    builder, op, graph = create_test_builder()
    pixel_values = create_test_input(builder, "pixel_values", ["N", _PATCH_DIM], dtype=dtype)
    grid = create_test_input(
        builder, "image_grid_thw", ["num_images", 3], dtype=ir.DataType.INT64
    )
    with build_context(ep_registry.require("cuda"), dtype):
        out = module(op, pixel_values, grid)
    out.name = "image_features"
    graph.outputs.append(out)
    return graph


def _count_cast_to(graph: ir.Graph, target: ir.DataType) -> int:
    count = 0
    for node in graph:
        if node.op_type == "Cast" and int(node.attributes["to"].value) == int(target):
            count += 1
    return count


class TestQwen25VLVisionPackedPath:
    def test_cuda_emits_packed_mha_with_helper_token_offset(self):
        graph = _build_vision_graph(ir.DataType.FLOAT)
        # One PackedMultiHeadAttention per transformer block, no standard
        # Attention fallback on CUDA.
        assert count_op_type(graph, "PackedMultiHeadAttention") == 2
        assert count_op_type(graph, "Attention") == 0
        # token_offset comes from build_packed_token_offset (Compress-based
        # valid/padding split), not the old (1, N) identity Range.
        assert count_op_type(graph, "Compress") >= 4  # 2 blocks x (valid + padding)

    def test_float32_build_keeps_native_dtype(self):
        # f32 is supported natively by the kernel: no down-cast to float16.
        graph = _build_vision_graph(ir.DataType.FLOAT)
        assert _count_cast_to(graph, ir.DataType.FLOAT16) == 0

    def test_bfloat16_build_casts_qkv_to_float16(self):
        # bf16 is unsupported by the kernel, so q/k/v are cast to float16:
        # 3 casts (q, k, v) per block x 2 blocks.
        graph = _build_vision_graph(ir.DataType.BFLOAT16)
        assert count_op_type(graph, "PackedMultiHeadAttention") == 2
        assert _count_cast_to(graph, ir.DataType.FLOAT16) == 6
