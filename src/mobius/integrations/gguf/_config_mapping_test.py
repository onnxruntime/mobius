# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from mobius.integrations.gguf._config_mapping import gguf_to_config


@dataclass
class _FakeGGUFModel:
    architecture: str
    metadata: dict[str, Any]
    tensor_names: tuple[str, ...] = ("output.weight",)

    def get_metadata(self, key: str, default: Any = None) -> Any:
        return self.metadata.get(key, default)


def _metadata(architecture: str, **overrides: Any) -> dict[str, Any]:
    metadata = {
        f"{architecture}.embedding_length": 64,
        f"{architecture}.feed_forward_length": 128,
        f"{architecture}.block_count": 8,
        f"{architecture}.attention.head_count": 4,
        f"{architecture}.attention.head_count_kv": 2,
        f"{architecture}.attention.layer_norm_rms_epsilon": 1e-6,
        f"{architecture}.context_length": 512,
        f"{architecture}.vocab_size": 256,
        f"{architecture}.rope.freq_base": 1_000_000.0,
    }
    metadata.update(overrides)
    return metadata


def test_rope_scaling_factor_is_populated_from_gguf_metadata():
    model = _FakeGGUFModel(
        "llama",
        _metadata(
            "llama",
            **{
                "llama.rope.scaling.type": "linear",
                "llama.rope.scaling.factor": 8.0,
            },
        ),
    )

    config = gguf_to_config(model)

    assert config.rope_type == "linear"
    assert config.rope_scaling == {
        "rope_type": "linear",
        "type": "linear",
        "factor": 8.0,
    }


def test_rope_scaling_factor_defaults_to_identity_when_missing(
    caplog: pytest.LogCaptureFixture,
):
    model = _FakeGGUFModel(
        "llama",
        _metadata("llama", **{"llama.rope.scaling.type": "linear"}),
    )

    config = gguf_to_config(model)

    assert config.rope_scaling is not None
    assert config.rope_scaling["factor"] == pytest.approx(1.0)
    assert "using 1.0" in caplog.text


def test_gemma3_rope_local_base_freq_defaults_for_sliding_window_gguf():
    model = _FakeGGUFModel(
        "gemma3",
        _metadata(
            "gemma3",
            **{
                "gemma3.attention.sliding_window": 1024,
                "gemma3.rope.scaling.type": "linear",
                "gemma3.rope.scaling.factor": 8.0,
            },
        ),
    )

    config = gguf_to_config(model)

    assert config.rope_local_base_freq == pytest.approx(10_000.0)


def test_gemma3_rope_local_base_freq_prefers_swa_metadata():
    model = _FakeGGUFModel(
        "gemma3",
        _metadata(
            "gemma3",
            **{
                "gemma3.attention.sliding_window": 1024,
                "gemma3.rope.freq_base_swa": 12_345.0,
            },
        ),
    )

    config = gguf_to_config(model)

    assert config.rope_local_base_freq == pytest.approx(12_345.0)


def test_gemma3_layer_types_follow_hf_default_sliding_window_pattern():
    model = _FakeGGUFModel(
        "gemma3",
        _metadata("gemma3", **{"gemma3.attention.sliding_window": 1024}),
    )

    config = gguf_to_config(model)

    assert config.sliding_window == 1024
    assert config.attn_qk_norm is True
    assert config.hidden_act == "gelu_pytorch_tanh"
    assert config.layer_types == [
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "full_attention",
        "sliding_attention",
        "sliding_attention",
    ]
