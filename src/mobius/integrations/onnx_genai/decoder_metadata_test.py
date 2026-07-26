# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for onnx-genai decoder (LLM) inference_metadata generation."""

from __future__ import annotations

import dataclasses
import os

import pytest
import yaml

from mobius.integrations.onnx_genai.decoder_metadata import (
    build_decoder_metadata,
    decoder_metadata_from_config,
    moe_metadata_from_config,
    write_decoder_metadata,
)


def _schema_path() -> str | None:
    for c in [
        os.environ.get("ONNX_GENAI_SCHEMA"),
        os.path.join(
            os.path.dirname(__file__),
            "../../../../../onnx-genai/schema/inference_metadata.schema.json",
        ),
        os.path.expanduser(
            "~/Documents/GitHub/onnx-genai/schema/inference_metadata.schema.json"
        ),
    ]:
        if c and os.path.exists(c):
            return c
    return None


@dataclasses.dataclass
class _FakeConfig:
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    hidden_size: int = 4096
    max_position_embeddings: int = 131072
    sliding_window: int | None = None
    model_type: str = "llama"


class TestDecoderMetadata:
    def test_grouped_query_attention(self):
        meta = build_decoder_metadata(
            num_attention_heads=32,
            num_kv_heads=8,
            head_dim=128,
            max_sequence_length=131072,
            kv_native_dtype="bf16",
        )
        assert meta["required_capabilities"] == ["kv_cache", "grouped_query_attention"]
        att = meta["model"]["attention"]
        assert att["type"] == "grouped_query"
        assert att["num_attention_heads"] == 32
        assert att["num_kv_heads"] == 8
        assert att["head_dim"] == 128
        assert meta["model"]["max_sequence_length"] == 131072
        assert meta["kv_cache"]["native_dtype"] == "bf16"

    def test_multi_head_when_kv_equals_heads(self):
        meta = build_decoder_metadata(num_attention_heads=16, num_kv_heads=16, head_dim=64)
        assert meta["model"]["attention"]["type"] == "multi_head"
        assert meta["required_capabilities"] == ["kv_cache", "multi_head_attention"]

    def test_sliding_window_and_sink(self):
        meta = build_decoder_metadata(
            num_attention_heads=8, head_dim=64, sliding_window=4096, sink_tokens=4
        )
        att = meta["model"]["attention"]
        assert att["sliding_window"] == 4096
        assert att["sink_tokens"] == 4

    def test_rejects_non_divisible_kv_heads(self):
        with pytest.raises(ValueError):
            build_decoder_metadata(num_attention_heads=12, num_kv_heads=5, head_dim=64)

    def test_from_config_reads_mobius_fields(self):
        meta = decoder_metadata_from_config(_FakeConfig(), kv_native_dtype="fp16")
        att = meta["model"]["attention"]
        assert att["type"] == "grouped_query"
        assert att["num_attention_heads"] == 32
        assert att["num_kv_heads"] == 8
        assert att["head_dim"] == 128
        assert meta["model"]["max_sequence_length"] == 131072
        assert meta["model"]["architecture"] == "llama"
        assert meta["kv_cache"]["native_dtype"] == "fp16"

    def test_from_config_derives_head_dim_and_drops_unset(self):
        cfg = _FakeConfig(head_dim=-42, sliding_window=-42)  # DEFAULT_INT sentinel
        meta = decoder_metadata_from_config(cfg)
        # head_dim = hidden_size / num_heads = 4096 / 32 = 128
        assert meta["model"]["attention"]["head_dim"] == 128
        assert "sliding_window" not in meta["model"]["attention"]

    def test_write_roundtrips_yaml(self, tmp_path):
        path = write_decoder_metadata(str(tmp_path), config=_FakeConfig())
        with open(path) as handle:
            loaded = yaml.safe_load(handle)
        assert loaded["model"]["attention"]["num_kv_heads"] == 8

    def test_matches_onnx_genai_schema(self):
        cfg = _FakeConfig(sliding_window=4096)
        cfg.num_local_experts = 8
        cfg.num_experts_per_tok = 2
        cfg.moe_intermediate_size = 256
        cfg.n_shared_experts = 1
        cfg.scoring_func = "sigmoid"
        cfg.topk_method = "noaux_tc"
        cfg.n_group = 4
        cfg.topk_group = 2
        meta = decoder_metadata_from_config(cfg, kv_native_dtype="bf16")

        schema_path = _schema_path()
        if schema_path is None:
            pytest.skip("onnx-genai schema not found")
        import json

        import jsonschema

        with open(schema_path) as handle:
            schema = json.load(handle)
        jsonschema.validate(instance=meta, schema=schema)

    def test_moe_metadata_from_config_is_structural(self):
        cfg = _FakeConfig()
        cfg.num_local_experts = 8
        cfg.num_experts_per_tok = 2
        cfg.moe_intermediate_size = 256
        cfg.n_shared_experts = 1
        cfg.scoring_func = "sigmoid"
        cfg.topk_method = "noaux_tc"
        cfg.n_group = 4
        cfg.topk_group = 2
        cfg.norm_topk_prob = True
        cfg.routed_scaling_factor = 2.5
        cfg.hidden_act = "silu"

        moe = moe_metadata_from_config(cfg)

        assert moe == {
            "representation": "dense_fallback",
            "routed_expert_count": 8,
            "shared_expert_count": 1,
            "experts_per_token": 2,
            "expert_intermediate_size": 256,
            "shared_expert_intermediate_size": 256,
            "activation": "silu",
            "router": {
                "score_function": "sigmoid",
                "selection_method": "grouped_top_k",
                "normalize_weights": True,
                "scaling_factor": 2.5,
                "group_count": 4,
                "groups_per_token": 2,
                "group_score": "top_2_sum",
            },
        }

    def test_decoder_metadata_embeds_moe_contract(self):
        cfg = _FakeConfig()
        cfg.num_local_experts = 4
        cfg.num_experts_per_tok = 2
        cfg.moe_intermediate_size = 128

        meta = decoder_metadata_from_config(cfg)

        assert meta["model"]["mixture_of_experts"]["routed_expert_count"] == 4
        assert meta["model"]["mixture_of_experts"]["representation"] == "dense_fallback"

    def test_moe_metadata_rejects_incomplete_contract(self):
        cfg = _FakeConfig()
        cfg.num_local_experts = 4
        cfg.num_experts_per_tok = None
        cfg.moe_intermediate_size = 128

        with pytest.raises(ValueError, match="num_experts_per_tok"):
            moe_metadata_from_config(cfg)

    def test_moe_metadata_rejects_invalid_expert_dimensions(self):
        cfg = _FakeConfig()
        cfg.num_local_experts = 4
        cfg.num_experts_per_tok = 5
        cfg.moe_intermediate_size = 128

        with pytest.raises(ValueError, match="exceeds num_local_experts"):
            moe_metadata_from_config(cfg)

        cfg.num_experts_per_tok = 2
        cfg.moe_intermediate_size = None
        cfg.intermediate_size = None
        with pytest.raises(ValueError, match="lacks both"):
            moe_metadata_from_config(cfg)

    def test_moe_metadata_handles_dense_and_simple_top_k_configs(self):
        assert moe_metadata_from_config(_FakeConfig()) is None

        cfg = _FakeConfig()
        cfg.num_local_experts = 4
        cfg.num_experts_per_tok = 1
        cfg.moe_intermediate_size = None
        cfg.intermediate_size = 512
        cfg.n_shared_experts = 2
        cfg.shared_expert_intermediate_size = 768
        cfg.n_group = 2
        cfg.topk_method = "greedy"

        moe = moe_metadata_from_config(cfg, representation="native")

        assert moe is not None
        assert moe["representation"] == "native"
        assert moe["expert_intermediate_size"] == 512
        assert moe["shared_expert_intermediate_size"] == 768
        assert moe["router"] == {
            "score_function": "softmax",
            "selection_method": "top_k",
            "normalize_weights": True,
            "scaling_factor": 1.0,
        }

    @pytest.mark.parametrize(
        ("field", "value", "expected"),
        [
            ("moe_num_shared_experts", 2, 2),
            ("num_shared_expert", 3, 3),
            ("moe_num_shared_experts", [0, 4], 4),
        ],
    )
    def test_moe_metadata_reads_shared_expert_aliases(self, field, value, expected):
        cfg = _FakeConfig()
        cfg.num_local_experts = 8
        cfg.num_experts_per_tok = 2
        cfg.moe_intermediate_size = 256
        setattr(cfg, field, value)

        moe = moe_metadata_from_config(cfg)

        assert moe is not None
        assert moe["shared_expert_count"] == expected
        assert moe["shared_expert_intermediate_size"] == expected * 256
