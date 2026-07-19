# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for the onnx-genai write_onnx_genai_config dispatcher."""

from __future__ import annotations

import dataclasses

import yaml

from mobius.integrations.onnx_genai import write_onnx_genai_config


@dataclasses.dataclass
class _Cfg:
    num_attention_heads: int = 16
    num_key_value_heads: int = 4
    head_dim: int = 64
    hidden_size: int = 1024
    max_position_embeddings: int = 8192
    sliding_window: int | None = None
    model_type: str = "qwen"


class _DiffusionPkg(dict):
    pass


def test_dispatch_decoder(tmp_path):
    arts = write_onnx_genai_config(object(), str(tmp_path), config=_Cfg(), kv_native_dtype="bf16")
    meta = yaml.safe_load(open(arts["inference_metadata"]))
    assert meta["model"]["attention"]["type"] == "grouped_query"
    assert meta["kv_cache"]["native_dtype"] == "bf16"


def test_dispatch_diffusion(tmp_path):
    pkg = _DiffusionPkg({"denoiser": object(), "vae": object()})
    arts = write_onnx_genai_config(
        pkg, str(tmp_path), num_inference_steps=20, vae_filename="vae.onnx",
    )
    meta = yaml.safe_load(open(arts["inference_metadata"]))
    assert meta["pipeline"]["strategy"]["kind"] == "iterative"
    assert "vae" in meta["pipeline"]["models"]
