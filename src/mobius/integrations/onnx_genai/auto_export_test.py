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
    arts = write_onnx_genai_config(
        object(), str(tmp_path), config=_Cfg(), kv_native_dtype="bf16"
    )
    with open(arts["inference_metadata"]) as handle:
        meta = yaml.safe_load(handle)
    assert meta["model"]["attention"]["type"] == "grouped_query"
    assert meta["kv_cache"]["native_dtype"] == "bf16"


def test_dispatch_diffusion(tmp_path):
    pkg = _DiffusionPkg({"denoiser": object(), "vae": object()})
    arts = write_onnx_genai_config(
        pkg,
        str(tmp_path),
        num_inference_steps=20,
        vae_filename="vae.onnx",
    )
    with open(arts["inference_metadata"]) as handle:
        meta = yaml.safe_load(handle)
    assert meta["pipeline"]["strategy"]["kind"] == "iterative"
    assert "vae" in meta["pipeline"]["models"]


def test_dispatch_diffusion_auto_reads_scheduler_from_source(tmp_path):
    import json

    src = tmp_path / "ckpt"
    (src / "scheduler").mkdir(parents=True)
    (src / "scheduler" / "scheduler_config.json").write_text(
        json.dumps({"_class_name": "EulerDiscreteScheduler", "beta_schedule": "scaled_linear"})
    )
    out = tmp_path / "out"
    pkg = _DiffusionPkg({"denoiser": object()})
    arts = write_onnx_genai_config(
        pkg,
        str(out),
        num_inference_steps=15,
        source=str(src),
    )
    with open(arts["inference_metadata"]) as handle:
        meta = yaml.safe_load(handle)
    assert meta["pipeline"]["strategy"]["scheduler_config"]["kind"] == "euler"
