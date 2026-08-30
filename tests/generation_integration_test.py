# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Integration tests for real-weight and synthetic autoregressive generation."""

from __future__ import annotations

import numpy as np
import pytest
from _integration_support import (
    _TEXT_MODELS,
    _fill_random_weights,
    _get_config,
    _make_session,
)

from mobius import build
from mobius._configs import ArchitectureConfig
from mobius._testing.comparison import (
    assert_generation_match,
)
from mobius._testing.generation import OnnxGenerator, torch_generate_greedy
from mobius._testing.torch_reference import (
    load_torch_model,
)


@pytest.mark.integration
@pytest.mark.integration_fast
@pytest.mark.parametrize("model_id,trust_remote_code", _TEXT_MODELS)
class TestGreedyGeneration:
    """Compare greedy text generation between ONNX and PyTorch."""

    def test_generate_tokens_match(self, model_id: str, trust_remote_code: bool):
        """Generated token IDs should be identical for greedy decoding."""
        onnx_model = build(model_id, dtype="f32", load_weights=True)
        torch_model, tokenizer = load_torch_model(model_id)
        config = _get_config(model_id, trust_remote_code)

        prompt = "Once upon a time"
        tokens = tokenizer(prompt, return_tensors="np")
        input_ids = tokens["input_ids"].astype(np.int64)
        max_new = 20

        session = _make_session(onnx_model)
        generator = OnnxGenerator(session, config)
        onnx_ids = generator.generate(
            input_ids,
            max_new_tokens=max_new,
            eos_token_id=tokenizer.eos_token_id,
        )
        session.close()

        torch_ids = torch_generate_greedy(
            torch_model,
            input_ids,
            max_new_tokens=max_new,
            eos_token_id=tokenizer.eos_token_id,
        )

        onnx_text = tokenizer.decode(onnx_ids[0], skip_special_tokens=True)
        torch_text = tokenizer.decode(torch_ids[0], skip_special_tokens=True)
        print(f"\n[{model_id}] ONNX:  {onnx_text!r}")
        print(f"[{model_id}] Torch: {torch_text!r}")

        assert_generation_match(onnx_ids[0].tolist(), torch_ids[0].tolist())


_GEN_HIDDEN = 64


_GEN_INTERMEDIATE = 128


_GEN_HEADS = 4


_GEN_KV_HEADS = 2


_GEN_HEAD_DIM = _GEN_HIDDEN // _GEN_HEADS


_GEN_LAYERS = 2


_GEN_VOCAB = 256


_GEN_MAX_POS = 128


_GEN_STEPS = 5


def _run_generation_test(
    model_type: str,
    config_overrides: dict | None = None,
) -> None:
    """Build a tiny ONNX model, fill with random weights, run generation."""
    import onnx_ir as ir

    from mobius._registry import registry
    from mobius._testing.generation import OnnxGenerator
    from mobius.tasks import get_task

    overrides = dict(config_overrides or {})
    config_cls = overrides.pop("_config_cls", ArchitectureConfig)

    defaults = dict(
        hidden_size=_GEN_HIDDEN,
        intermediate_size=_GEN_INTERMEDIATE,
        num_attention_heads=_GEN_HEADS,
        num_key_value_heads=_GEN_KV_HEADS,
        head_dim=_GEN_HEAD_DIM,
        num_hidden_layers=_GEN_LAYERS,
        vocab_size=_GEN_VOCAB,
        max_position_embeddings=_GEN_MAX_POS,
        hidden_act="silu",
        rms_norm_eps=1e-6,
        rope_type="default",
        rope_theta=10_000.0,
        pad_token_id=0,
        dtype=ir.DataType.FLOAT,
    )
    defaults.update(overrides)
    config = config_cls(**defaults)

    model_cls = registry.get(model_type)
    assert model_cls is not None, f"Model type {model_type!r} not in registry"
    module = model_cls(config)
    task = get_task("text-generation")
    pkg = task.build(module, config)
    onnx_model = pkg["model"]

    # Fill parameters with random weights
    rng = np.random.default_rng(42)
    _fill_random_weights(onnx_model, rng)

    # Create ORT session and generator
    session = _make_session(onnx_model)
    generator = OnnxGenerator(session, config)

    # Prompt: 3 random tokens
    prompt = rng.integers(1, _GEN_VOCAB, size=(1, 3)).astype(np.int64)
    output_ids = generator.generate(prompt, max_new_tokens=_GEN_STEPS)

    # Verify output shape: [1, prompt_len + generated]
    assert output_ids.shape[0] == 1
    assert output_ids.shape[1] == 3 + _GEN_STEPS, (
        f"Expected {3 + _GEN_STEPS} tokens, got {output_ids.shape[1]}"
    )

    # Verify all generated token IDs are valid
    assert np.all(output_ids >= 0)
    assert np.all(output_ids < _GEN_VOCAB)

    # Run one more step manually to verify KV cache works
    # by checking logits are finite
    num_kv_heads = config.num_key_value_heads
    head_dim = config.head_dim
    past_kv = {}
    for i in range(_GEN_LAYERS):
        past_kv[f"past_key_values.{i}.key"] = np.zeros(
            (1, num_kv_heads, 0, head_dim), dtype=np.float32
        )
        past_kv[f"past_key_values.{i}.value"] = np.zeros(
            (1, num_kv_heads, 0, head_dim), dtype=np.float32
        )

    # Prefill step
    feeds = {
        "input_ids": prompt,
        "attention_mask": np.ones((1, 3), dtype=np.int64),
        "position_ids": np.arange(3, dtype=np.int64)[np.newaxis, :],
        **past_kv,
    }
    outputs = session.run(feeds)
    logits = outputs["logits"]

    # Logits should be finite
    assert np.all(np.isfinite(logits)), "Logits contain NaN or Inf"
    assert logits.shape == (1, 3, _GEN_VOCAB), (
        f"Expected logits shape (1, 3, {_GEN_VOCAB}), got {logits.shape}"
    )

    # KV cache should have grown to seq_len=3
    for i in range(_GEN_LAYERS):
        key_cache = outputs[f"present.{i}.key"]
        val_cache = outputs[f"present.{i}.value"]
        assert key_cache.shape[2] == 3, (
            f"Layer {i} key cache should have 3 entries, got {key_cache.shape[2]}"
        )
        assert val_cache.shape[2] == 3, (
            f"Layer {i} value cache should have 3 entries, got {val_cache.shape[2]}"
        )

    # Decode step: feed one token with updated KV cache
    next_token = np.argmax(logits[:, -1, :], axis=-1, keepdims=True)
    decode_past_kv = {}
    for i in range(_GEN_LAYERS):
        decode_past_kv[f"past_key_values.{i}.key"] = outputs[f"present.{i}.key"]
        decode_past_kv[f"past_key_values.{i}.value"] = outputs[f"present.{i}.value"]

    decode_feeds = {
        "input_ids": next_token.astype(np.int64),
        "attention_mask": np.ones((1, 4), dtype=np.int64),
        "position_ids": np.array([[3]], dtype=np.int64),
        **decode_past_kv,
    }
    decode_outputs = session.run(decode_feeds)
    decode_logits = decode_outputs["logits"]

    assert np.all(np.isfinite(decode_logits)), "Decode logits contain NaN or Inf"
    assert decode_logits.shape == (1, 1, _GEN_VOCAB)

    # KV cache grew by 1
    for i in range(_GEN_LAYERS):
        key_cache = decode_outputs[f"present.{i}.key"]
        assert key_cache.shape[2] == 4, (
            f"Layer {i} key cache should have 4 entries after decode, got {key_cache.shape[2]}"
        )

    session.close()
    print(f"Generation test passed for {model_type}: {_GEN_STEPS} steps, KV cache verified")


@pytest.mark.integration
@pytest.mark.integration_fast
class TestGeneration:
    """End-to-end generation loop tests for top 5 architectures.

    These tests build tiny ONNX models with random weights and verify
    the full autoregressive generation loop: prefill → decode → KV cache
    growth → finite logits at each step.
    """

    def test_generation_llama(self):
        _run_generation_test("llama")

    def test_generation_qwen2(self):
        _run_generation_test("qwen2")

    def test_generation_phi3(self):
        _run_generation_test(
            "phi3",
            {
                "partial_rotary_factor": 0.5,
                "rope_type": "longrope",
                "rope_scaling": {
                    "short_factor": [1.0] * ((_GEN_HEAD_DIM * 50 // 100) // 2),
                    "long_factor": [1.0] * ((_GEN_HEAD_DIM * 50 // 100) // 2),
                },
                "original_max_position_embeddings": 128,
            },
        )

    def test_generation_gemma2(self):
        from mobius._configs import Gemma2Config

        _run_generation_test(
            "gemma2",
            {
                "_config_cls": Gemma2Config,
                "attn_qkv_bias": True,
                "attn_o_bias": True,
                "attn_logit_softcapping": 50.0,
                "final_logit_softcapping": 30.0,
                "query_pre_attn_scalar": 256,
            },
        )

    def test_generation_mistral(self):
        _run_generation_test("mistral")
