# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Graph and precision-contract checks for the LFM3 Audio frontend."""

from __future__ import annotations

import onnx_ir as ir
import pytest

from mobius._configs import Lfm2Config
from mobius._testing import create_test_builder, create_test_input
from mobius.models.lfm3_audio import Lfm3AudioEncoder


def _config(**kwargs):
    return Lfm2Config(
        hidden_size=32,
        intermediate_size=64,
        block_auto_adjust_ff_dim=False,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_hidden_layers=2,
        head_dim=8,
        layer_types=["conv", "full_attention"],
        short_conv_kernel=3,
        rope_type="default",
        max_position_embeddings=128,
        **kwargs,
    )


@pytest.mark.parametrize("dtype", [ir.DataType.FLOAT16, ir.DataType.BFLOAT16])
def test_encoder_rejects_unqualified_compute_dtype(dtype):
    with pytest.raises(ValueError, match="requires float32 compute"):
        Lfm3AudioEncoder(_config(dtype=dtype), conv_channels=4)


@pytest.mark.parametrize("dtype", [ir.DataType.FLOAT16, ir.DataType.BFLOAT16])
def test_encoder_rejects_reduced_precision_processor_boundary(dtype):
    module = Lfm3AudioEncoder(_config(), conv_channels=4)
    builder, op, _ = create_test_builder()
    features = create_test_input(builder, "input_features", [2, 128, 25], dtype=dtype)
    lengths = create_test_input(builder, "feature_lengths", [2], dtype=ir.DataType.INT64)
    with pytest.raises(ValueError, match="input_features must be float32"):
        module(op, features, lengths)


def test_encoder_builds_dynamic_batch_and_frames():
    module = Lfm3AudioEncoder(_config(), conv_channels=4)
    builder, op, graph = create_test_builder()
    features = create_test_input(builder, "input_features", ["batch", 128, "frames"])
    lengths = create_test_input(builder, "feature_lengths", ["batch"], dtype=ir.DataType.INT64)
    output, output_lengths = module(op, features, lengths)
    builder.add_output(output, "audio_features")
    builder.add_output(output_lengths, "audio_feature_lengths")
    assert len(graph.outputs) == 2
    assert any(node.op_type == "Attention" for node in graph)
    assert any(node.op_type == "CausalConvWithState" for node in graph)
