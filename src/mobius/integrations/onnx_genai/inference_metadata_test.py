# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import onnx_ir as ir
import pytest

from mobius._configs import ArchitectureConfig
from mobius._model_package import ModelPackage
from mobius.integrations.onnx_genai import (
    generate_inference_metadata,
    write_inference_metadata,
)


def _config(**kwargs) -> ArchitectureConfig:
    values = {
        "num_attention_heads": 14,
        "num_key_value_heads": 2,
        "head_dim": 64,
        "max_position_embeddings": 32768,
        "dtype": ir.DataType.FLOAT16,
    }
    values.update(kwargs)
    return ArchitectureConfig(**values)


def test_generate_gqa_fp16_metadata() -> None:
    metadata = generate_inference_metadata(_config(sliding_window=4096))

    assert metadata == {
        "required_capabilities": ["grouped_query_attention"],
        "model": {
            "attention": {
                "type": "group_query_attention",
                "num_kv_heads": 2,
                "num_attention_heads": 14,
                "head_dim": 64,
                "sliding_window": 4096,
            },
            "max_sequence_length": 4096,
            "runtime_configurable": {"kv_cache": {"dtype": ["float16"]}},
        },
        "kv_cache": {"native_dtype": "float16"},
    }


def test_generate_mha_float32_metadata() -> None:
    metadata = generate_inference_metadata(
        _config(
            num_attention_heads=8,
            num_key_value_heads=8,
            dtype=ir.DataType.FLOAT,
            sliding_window=None,
        )
    )

    assert metadata["required_capabilities"] == ["multi_head_attention"]
    assert metadata["model"]["attention"] == {
        "type": "multi_head_attention",
        "num_kv_heads": 8,
        "num_attention_heads": 8,
        "head_dim": 64,
    }
    assert metadata["kv_cache"]["native_dtype"] == "float32"


def test_write_inference_metadata_yaml(tmp_path) -> None:
    pkg = ModelPackage(config=_config())

    path = write_inference_metadata(pkg, str(tmp_path))

    assert path == str(tmp_path / "inference_metadata.yaml")
    assert (tmp_path / "inference_metadata.yaml").read_text() == (
        "required_capabilities:\n"
        "  - grouped_query_attention\n"
        "model:\n"
        "  attention:\n"
        "    type: group_query_attention\n"
        "    num_kv_heads: 2\n"
        "    num_attention_heads: 14\n"
        "    head_dim: 64\n"
        "  max_sequence_length: 4096\n"
        "  runtime_configurable:\n"
        "    kv_cache:\n"
        "      dtype:\n"
        "        - float16\n"
        "kv_cache:\n"
        "  native_dtype: float16\n"
    )


def test_max_sequence_length_override() -> None:
    metadata = generate_inference_metadata(_config(), max_sequence_length=2048)

    assert metadata["model"]["max_sequence_length"] == 2048


@pytest.mark.parametrize("max_sequence_length", [0, -1, 32769])
def test_invalid_max_sequence_length(max_sequence_length: int) -> None:
    with pytest.raises(ValueError, match="max_sequence_length"):
        generate_inference_metadata(_config(), max_sequence_length=max_sequence_length)


def test_write_requires_package_config(tmp_path) -> None:
    with pytest.raises(ValueError, match=r"ModelPackage\.config"):
        write_inference_metadata(ModelPackage(), str(tmp_path))
