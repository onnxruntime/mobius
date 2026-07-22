# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for onnx-genai decoder (LLM) inference_metadata generation."""

from __future__ import annotations

import dataclasses
import os

import onnx_ir as ir
import pytest
import yaml

from mobius._configs import QuantizationConfig
from mobius.integrations.onnx_genai.decoder_metadata import (
    build_decoder_metadata,
    decoder_metadata_from_config,
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
        assert att["type"] == "grouped_query_attention"
        assert att["num_attention_heads"] == 32
        assert att["num_kv_heads"] == 8
        assert att["head_dim"] == 128
        assert meta["model"]["max_sequence_length"] == 131072
        assert meta["kv_cache"]["native_dtype"] == "bfloat16"

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
        assert att["type"] == "grouped_query_attention"
        assert att["num_attention_heads"] == 32
        assert att["num_kv_heads"] == 8
        assert att["head_dim"] == 128
        assert meta["model"]["max_sequence_length"] == 131072
        assert meta["model"]["architecture"] == "llama"
        assert meta["kv_cache"]["native_dtype"] == "float16"

    def test_from_config_infers_fp16_kv_dtype_for_int4_weights(self):
        cfg = _FakeConfig()
        cfg.dtype = ir.DataType.FLOAT16
        cfg.quantization = QuantizationConfig(bits=4, quant_method="rtn")

        meta = decoder_metadata_from_config(cfg)

        assert meta["kv_cache"]["native_dtype"] == "float16"

    def test_from_config_derives_head_dim_and_drops_unset(self):
        cfg = _FakeConfig(head_dim=-42, sliding_window=-42)  # DEFAULT_INT sentinel
        meta = decoder_metadata_from_config(cfg)
        # head_dim = hidden_size / num_heads = 4096 / 32 = 128
        assert meta["model"]["attention"]["head_dim"] == 128
        assert "sliding_window" not in meta["model"]["attention"]
        assert "kv_cache" not in meta

    def test_write_roundtrips_yaml(self, tmp_path):
        path = write_decoder_metadata(str(tmp_path), config=_FakeConfig())
        with open(path) as handle:
            loaded = yaml.safe_load(handle)
        assert loaded["model"]["attention"]["num_kv_heads"] == 8

    def test_matches_onnx_genai_schema(self):
        schema_path = _schema_path()
        if schema_path is None:
            pytest.skip("onnx-genai schema not found")
        import json

        import jsonschema

        with open(schema_path) as handle:
            schema = json.load(handle)
        meta = decoder_metadata_from_config(
            _FakeConfig(sliding_window=4096), kv_native_dtype="bf16"
        )
        jsonschema.validate(instance=meta, schema=schema)
