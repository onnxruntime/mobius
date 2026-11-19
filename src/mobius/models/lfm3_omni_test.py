# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Graph contract checks for LFM3-Omni."""

from __future__ import annotations

import sys
from pathlib import Path

import onnx_ir as ir
import pytest

from mobius import build_from_module
from mobius.models.lfm3_omni import Lfm3OmniForConditionalGeneration

# tests/_test_configs.py provides the tiny LFM3-Omni config.
_TESTS_DIR = Path(__file__).resolve().parents[3] / "tests"
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

from _test_configs import lfm3_omni_config  # noqa: E402


def test_omni_builds_five_component_package():
    config = lfm3_omni_config()
    package = build_from_module(
        Lfm3OmniForConditionalGeneration(config),
        config,
        task="lfm3-omni",
        execution_provider="cpu",
    )
    assert set(package) == {
        "vision_encoder",
        "audio_encoder",
        "embedding",
        "decoder",
        "depth_decoder",
    }
    assert [value.name for value in package["vision_encoder"].graph.inputs] == [
        "pixel_values",
        "image_grid_thw",
    ]
    assert [value.name for value in package["embedding"].graph.inputs] == [
        "input_ids",
        "image_features",
        "audio_features",
        "audio_out_codes",
    ]
    decoder = package["decoder"]
    assert [value.name for value in decoder.graph.outputs][:2] == ["logits", "hidden_states"]
    ops = [node.op_type for node in decoder.graph]
    assert ops.count("Attention") == 2
    assert "GroupQueryAttention" not in ops
    assert "Unpacked" in decoder.metadata_props["mobius.input_contract"]


@pytest.mark.parametrize("dtype", [ir.DataType.FLOAT16, ir.DataType.BFLOAT16])
def test_omni_rejects_reduced_precision(dtype):
    config = lfm3_omni_config()
    config.dtype = dtype
    with pytest.raises(ValueError, match="float32"):
        Lfm3OmniForConditionalGeneration(config)
