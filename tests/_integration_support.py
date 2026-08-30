# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Shared helpers for the split integration test modules."""

from __future__ import annotations

import os

import numpy as np
import pytest
import transformers

from mobius._configs import ArchitectureConfig
from mobius._testing.ort_inference import OnnxModelSession


def _get_test_device() -> str:
    """Return 'cuda' if MOBIUS_TEST_DEVICE=cuda, else 'cpu'."""
    return os.environ.get("MOBIUS_TEST_DEVICE", "cpu").strip().lower()


def _make_session(model, **kwargs) -> OnnxModelSession:
    """Create an OnnxModelSession with the test device."""
    return OnnxModelSession(model, device=_get_test_device(), **kwargs)


def _model_accessible(model_id: str) -> bool:
    """Check if a HuggingFace model is accessible (not gated/private)."""
    try:
        from huggingface_hub import model_info

        model_info(model_id)
    except Exception:
        return False
    else:
        return True


_TEXT_MODELS = [
    # CausalLMModel (base: llama/mistral/qwen2 architecture)
    pytest.param("Qwen/Qwen2.5-0.5B", False, id="qwen2.5-0.5b"),
    pytest.param("LiquidAI/LFM2.5-230M", False, id="lfm2.5-230m"),
    pytest.param("HuggingFaceTB/SmolLM-135M", False, id="smollm-135m"),
    # SmolLM3 (per-layer RoPE gating via no_rope_layers)
    pytest.param("HuggingFaceTB/SmolLM3-3B", False, id="smollm3-3b"),
    # Gemma
    pytest.param("google/gemma-3-1b-pt", False, id="gemma3-1b"),
    # Granite
    pytest.param("ibm-granite/granite-3.3-2b-instruct", False, id="granite-3.3-2b"),
    # Phi3 (LongRoPE)
    pytest.param(
        "microsoft/Phi-3.5-mini-instruct",
        True,
        id="phi3.5-mini",
        marks=pytest.mark.skip(
            reason="Phi-3.5 uses trust_remote_code with DynamicCache.from_legacy_cache removed in transformers>=5.x"
        ),
    ),
    # Qwen3
    pytest.param("Qwen/Qwen3-0.6B", False, id="qwen3-0.6b"),
    # OLMo (post-norm)
    pytest.param("allenai/OLMo-1B-hf", False, id="olmo-1b"),
    # MoE (PhiMoE — Phi3MoECausalLMModel)
    pytest.param(
        "microsoft/Phi-tiny-MoE-instruct",
        True,
        id="phi-tiny-moe",
        marks=pytest.mark.skip(reason="Requires flash_attn package not available in CI"),
    ),
    # MoE (GraniteMoE — MoECausalLMModel with TopKGate)
    pytest.param("ibm-granite/granite-3.0-1b-a400m-instruct", False, id="granitemoe-1b"),
    # MoE (OLMoE — MoECausalLMModel with TopKGate, different expert count)
    pytest.param("allenai/OLMoE-1B-7B-0924", False, id="olmoe-1b"),
    # MoE (Qwen2-MoE — MoECausalLMModel with TopKGate, shared experts)
    pytest.param(
        "Qwen/Qwen1.5-MoE-A2.7B-Chat",
        False,
        id="qwen-moe-2.7b",
        marks=pytest.mark.skip(reason="2.7B model download too slow for CI"),
    ),
    # GPT-2 (absolute positional embeddings, no RoPE)
    pytest.param(
        "openai-community/gpt2",
        False,
        id="gpt2",
    ),
    # OPT (learned positional embeddings)
    pytest.param(
        "facebook/opt-125m",
        False,
        id="opt-125m",
        marks=pytest.mark.skip(reason="Model only has pytorch_model.bin, no safetensors"),
    ),
    # Bloom (ALiBi attention)
    pytest.param(
        "bigscience/bloom-560m",
        False,
        id="bloom-560m",
    ),
    # Falcon (ALiBi attention, multi-query)
    pytest.param(
        "tiiuae/falcon-rw-1b",
        False,
        id="falcon-rw-1b",
        marks=pytest.mark.skip(reason="Model only has pytorch_model.bin, no safetensors"),
    ),
    # Ministral3 (YaRN RoPE, text-only extraction of Mistral-3 VLM)
    pytest.param(
        "Aratako/Ministral-3-3B-Instruct-2512-BF16-TextOnly",
        False,
        id="ministral3-3b",
    ),
]


def _get_config(model_id: str, trust_remote_code: bool = False):
    """Load ArchitectureConfig for a model from HuggingFace."""
    hf_config = transformers.AutoConfig.from_pretrained(
        model_id, trust_remote_code=trust_remote_code
    )
    parent_config = hf_config
    if hasattr(hf_config, "text_config"):
        hf_config = hf_config.text_config
    return ArchitectureConfig.from_transformers(hf_config, parent_config=parent_config)


def _make_prefill_feeds(config, input_ids, attention_mask, position_ids):
    """Create ONNX session feeds for a prefill step."""
    feeds = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
    }
    for i in range(config.num_hidden_layers):
        layer_types = config.layer_types or []
        layer_type = layer_types[i] if i < len(layer_types) else "full_attention"
        if layer_type == "conv":
            feeds[f"past_key_values.{i}.conv_state"] = np.zeros(
                (1, config.hidden_size, config.short_conv_kernel - 1),
                dtype=np.float32,
            )
            continue
        feeds[f"past_key_values.{i}.key"] = np.zeros(
            (1, config.num_key_value_heads, 0, config.head_dim),
            dtype=np.float32,
        )
        feeds[f"past_key_values.{i}.value"] = np.zeros(
            (1, config.num_key_value_heads, 0, config.head_dim),
            dtype=np.float32,
        )
    return feeds


def _make_decode_feeds(
    config, decode_input_ids, decode_attention_mask, decode_position_ids, onnx_prefill_out
):
    """Create ONNX session feeds for a decode step using prior KV cache."""
    feeds = {
        "input_ids": decode_input_ids,
        "attention_mask": decode_attention_mask,
        "position_ids": decode_position_ids,
    }
    for i in range(config.num_hidden_layers):
        layer_types = config.layer_types or []
        layer_type = layer_types[i] if i < len(layer_types) else "full_attention"
        if layer_type == "conv":
            feeds[f"past_key_values.{i}.conv_state"] = onnx_prefill_out[
                f"present.{i}.conv_state"
            ]
            continue
        feeds[f"past_key_values.{i}.key"] = onnx_prefill_out[f"present.{i}.key"]
        feeds[f"past_key_values.{i}.value"] = onnx_prefill_out[f"present.{i}.value"]
    return feeds


def _fill_random_weights(model, rng: np.random.Generator) -> None:
    """Fill all unset graph initializers with random float32 values."""
    import onnx_ir as ir

    for init in model.graph.initializers.values():
        if init.const_value is not None:
            continue
        shape = tuple(d for d in init.shape)
        if init.dtype == ir.DataType.FLOAT:
            data = rng.standard_normal(shape).astype(np.float32) * 0.02
        elif init.dtype == ir.DataType.FLOAT16:
            data = (rng.standard_normal(shape) * 0.02).astype(np.float16)
        elif init.dtype == ir.DataType.BFLOAT16:
            data_f32 = (rng.standard_normal(shape) * 0.02).astype(np.float32)
            data_bf16 = (data_f32.view(np.uint32) >> 16).astype(np.uint16)
            init.const_value = ir.Tensor(data_bf16, dtype=ir.DataType.BFLOAT16)
            continue
        elif init.dtype in (ir.DataType.INT64, ir.DataType.INT32):
            data = rng.integers(0, 10, size=shape).astype(
                np.int64 if init.dtype == ir.DataType.INT64 else np.int32
            )
        else:
            data = rng.standard_normal(shape).astype(np.float32) * 0.02
        init.const_value = ir.Tensor(data)
