# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import onnx_ir as ir
import pytest

from mobius._configs import ArchitectureConfig, Gemma4AssistantConfig
from mobius._model_package import ModelPackage
from mobius.integrations.onnx_genai import (
    generate_inference_metadata,
    write_inference_metadata,
)
from mobius.integrations.onnx_genai.inference_metadata import (
    _speculative_block,
    _target_layers_by_type,
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

    artifacts = write_inference_metadata(pkg, str(tmp_path))

    assert artifacts["inference_metadata"] == str(tmp_path / "inference_metadata.yaml")
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


# ---------------------------------------------------------------------------
# Helpers for building a minimal Gemma4AssistantConfig without HF weights
# ---------------------------------------------------------------------------

from types import SimpleNamespace  # noqa: E402


def _e2b_assistant_hf_config():
    """Minimal HF config stub for google/gemma-4-E2B-it-assistant."""
    text = SimpleNamespace(
        model_type="gemma4_text",
        vocab_size=262144,
        hidden_size=256,
        intermediate_size=2048,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=1,
        num_kv_shared_layers=4,
        head_dim=256,
        global_head_dim=512,
        layer_types=["sliding_attention"] * 3 + ["full_attention"],
        sliding_window=512,
        max_position_embeddings=131072,
        rms_norm_eps=1e-6,
        hidden_activation="gelu_pytorch_tanh",
        pad_token_id=0,
        bos_token_id=2,
        eos_token_id=1,
        tie_word_embeddings=True,
        attention_dropout=0.0,
        attention_bias=False,
        enable_moe_block=False,
        use_double_wide_mlp=False,
        hidden_size_per_layer_input=0,
        vocab_size_per_layer_input=0,
        rope_parameters={
            "full_attention": {
                "partial_rotary_factor": 0.25,
                "rope_theta": 1_000_000.0,
                "rope_type": "proportional",
            },
            "sliding_attention": {
                "rope_theta": 10_000.0,
                "rope_type": "default",
            },
        },
    )
    return SimpleNamespace(
        model_type="gemma4_assistant",
        text_config=text,
        backbone_hidden_size=1536,
        use_ordered_embeddings=True,
        num_centroids=2048,
        centroid_intermediate_top_k=32,
        tie_word_embeddings=True,
        architectures=["Gemma4AssistantForCausalLM"],
    )


def _assistant_config() -> Gemma4AssistantConfig:
    hf = _e2b_assistant_hf_config()
    return Gemma4AssistantConfig.from_transformers(hf, parent_config=hf)


# ---------------------------------------------------------------------------
# Unit tests: _target_layers_by_type
# ---------------------------------------------------------------------------


def test_target_layers_by_type_e2b_pattern() -> None:
    layer_types = ["sliding_attention", "sliding_attention", "sliding_attention", "full_attention"]
    result = _target_layers_by_type(layer_types)
    assert result == {"sliding_attention": [0, 1, 2], "full_attention": [3]}


def test_target_layers_by_type_all_sliding() -> None:
    assert _target_layers_by_type(["sliding_attention"] * 3) == {"sliding_attention": [0, 1, 2]}


def test_target_layers_by_type_empty() -> None:
    assert _target_layers_by_type([]) == {}


# ---------------------------------------------------------------------------
# Unit tests: _speculative_block
# ---------------------------------------------------------------------------


def test_speculative_block_e2b_defaults() -> None:
    cfg = _assistant_config()
    block = _speculative_block(cfg)

    assert block["proposal_type"] == "shared_kv"
    assert block["num_speculative_tokens"] == 3
    assert block["model"] == "model.onnx"
    assert block["backbone_hidden_size"] == 1536
    assert block["vocab_size"] == 262144
    assert block["projected_state_output"] == "projected_state"
    assert block["logits_output"] == "logits"

    shared_kv = block["shared_kv"]
    assert len(shared_kv) == 2
    assert shared_kv[0] == {"name": "sliding_attention", "target_layers": [0, 1, 2]}
    assert shared_kv[1] == {"name": "full_attention", "target_layers": [3]}


def test_speculative_block_custom_model_path() -> None:
    cfg = _assistant_config()
    block = _speculative_block(cfg, model_path="assistant/model.onnx", num_speculative_tokens=5)
    assert block["model"] == "assistant/model.onnx"
    assert block["num_speculative_tokens"] == 5


# ---------------------------------------------------------------------------
# Integration: generate_inference_metadata with Gemma4AssistantConfig
# ---------------------------------------------------------------------------


def test_generate_metadata_includes_speculative_for_assistant() -> None:
    cfg = _assistant_config()
    metadata = generate_inference_metadata(cfg)

    assert "speculative" in metadata
    spec = metadata["speculative"]
    assert spec["proposal_type"] == "shared_kv"
    assert spec["backbone_hidden_size"] == 1536
    assert spec["vocab_size"] == 262144
    shared_names = [s["name"] for s in spec["shared_kv"]]
    assert "sliding_attention" in shared_names
    assert "full_attention" in shared_names


def test_generate_metadata_no_speculative_for_plain_config() -> None:
    metadata = generate_inference_metadata(_config())
    assert "speculative" not in metadata


# ---------------------------------------------------------------------------
# Integration: _to_yaml serialises the speculative block correctly
# ---------------------------------------------------------------------------


def test_to_yaml_speculative_block() -> None:
    cfg = _assistant_config()
    metadata = generate_inference_metadata(cfg)
    from mobius.integrations.onnx_genai.inference_metadata import _to_yaml

    yaml_text = _to_yaml(metadata)

    assert "speculative:" in yaml_text
    assert "  proposal_type: shared_kv" in yaml_text
    assert "  num_speculative_tokens: 3" in yaml_text
    assert "  model: model.onnx" in yaml_text
    assert "  backbone_hidden_size: 1536" in yaml_text
    assert "  vocab_size: 262144" in yaml_text
    assert "  projected_state_output: projected_state" in yaml_text
    assert "  logits_output: logits" in yaml_text
    assert "    - name: sliding_attention" in yaml_text
    assert "      target_layers: [0, 1, 2]" in yaml_text
    assert "    - name: full_attention" in yaml_text
    assert "      target_layers: [3]" in yaml_text


def test_write_inference_metadata_yaml_for_assistant(tmp_path) -> None:
    cfg = _assistant_config()
    pkg = ModelPackage(config=cfg)

    artifacts = write_inference_metadata(pkg, str(tmp_path))

    assert "inference_metadata" in artifacts
    yaml_text = (tmp_path / "inference_metadata.yaml").read_text()
    assert "speculative:" in yaml_text
    assert "proposal_type: shared_kv" in yaml_text
    assert "backbone_hidden_size: 1536" in yaml_text


# ---------------------------------------------------------------------------
# Merged package: target model block + assistant speculative block
# ---------------------------------------------------------------------------


def _e2b_target_config():
    """A namespace mirroring the real Gemma-4 E2B text decoder config.

    35 hidden layers, 20 KV-shared → 15 exported KV layers; sliding/full
    interleave with sliding_window_pattern=5 (full at every 5th layer).
    """
    from types import SimpleNamespace

    layer_types = (["sliding_attention"] * 4 + ["full_attention"]) * 7
    return SimpleNamespace(
        layer_types=layer_types,
        num_hidden_layers=35,
        num_kv_shared_layers=20,
        num_attention_heads=8,
        num_key_value_heads=1,
        head_dim=256,
        max_position_embeddings=131072,
        sliding_window=512,
        dtype=ir.DataType.FLOAT16,
    )


def test_folded_shared_kv_groups_e2b() -> None:
    from mobius.integrations.onnx_genai.inference_metadata import _folded_shared_kv_groups

    groups = _folded_shared_kv_groups(_e2b_target_config())
    assert groups == [
        {"name": "sliding_attention", "target_layers": [0, 1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13]},
        {"name": "full_attention", "target_layers": [4, 9, 14]},
    ]
    # The runtime keys each group off target_layers.last(); it must point at the
    # last exported layer of that type (the KV-share source layer).
    assert groups[0]["target_layers"][-1] == 13
    assert groups[1]["target_layers"][-1] == 14


def test_generate_merged_inference_metadata_e2b() -> None:
    from mobius.integrations.onnx_genai import generate_merged_inference_metadata

    metadata = generate_merged_inference_metadata(_e2b_target_config(), _assistant_config())

    # model block describes the TARGET decoder
    assert metadata["model"]["attention"]["num_attention_heads"] == 8
    assert metadata["model"]["attention"]["head_dim"] == 256

    spec = metadata["speculative"]
    assert spec["model"] == "assistant/model.onnx"
    assert spec["backbone_hidden_size"] == 1536
    assert spec["vocab_size"] == 262144
    # shared_kv derives from the TARGET's folded layers, not the assistant's.
    assert spec["shared_kv"][0]["target_layers"] == [0, 1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13]
    assert spec["shared_kv"][1]["target_layers"] == [4, 9, 14]


def test_write_merged_inference_metadata_yaml(tmp_path) -> None:
    from mobius.integrations.onnx_genai import write_merged_inference_metadata

    artifacts = write_merged_inference_metadata(
        _e2b_target_config(), _assistant_config(), str(tmp_path)
    )
    assert "inference_metadata" in artifacts
    yaml_text = (tmp_path / "inference_metadata.yaml").read_text()
    assert "speculative:" in yaml_text
    assert "  model: assistant/model.onnx" in yaml_text
    assert "      target_layers: [0, 1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13]" in yaml_text
    assert "      target_layers: [4, 9, 14]" in yaml_text
