# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Pinned full-checkpoint LFM3 logits, cache reuse, and greedy generation parity."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from integration._support import (
    _get_test_device,
    _make_decode_feeds,
    _make_prefill_feeds,
)
from mobius import build
from mobius._testing.comparison import assert_logits_close
from mobius._testing.generation import OnnxGenerator
from mobius._testing.ort_inference import OnnxModelSession
from mobius._testing.torch_reference import load_torch_model, torch_forward

_MODEL_ID = "LiquidAI/LFM3-1.7B"
_REVISION = "875094441ea6517efc0f44a5685f05e381b72b14"
_PROMPT = "Here is my poem:"
_GOLDEN_DIR = Path(__file__).resolve().parents[2] / "testdata" / "golden" / "causal-lm"

pytestmark = [pytest.mark.integration, pytest.mark.integration_fast]


@pytest.fixture(scope="module")
def lfm3_artifacts():
    """Load the checkpoint once; fail if requested CUDA execution falls back."""
    device = _get_test_device()
    reference, tokenizer = load_torch_model(
        _MODEL_ID,
        dtype=torch.float32,
        device=device,
        revision=_REVISION,
        trust_remote_code=True,
    )
    reference.config._attn_implementation = "eager"
    package = build(
        _MODEL_ID,
        dtype="f32",
        execution_provider="cuda" if device == "cuda" else "onnx-standard",
        revision=_REVISION,
        trust_remote_code=True,
        load_weights=True,
    )
    session = OnnxModelSession(package, device=device)
    assert session.providers[0] == (
        "CUDAExecutionProvider" if device == "cuda" else "CPUExecutionProvider"
    )
    try:
        yield reference, tokenizer, package, session
    finally:
        session.close()


def _prefill_feeds(package, session, input_ids, mask, positions):
    feeds = _make_prefill_feeds(package.config, input_ids, mask, positions)
    for name in feeds:
        if name.startswith("past_key_values."):
            dtype = session.get_input_dtype(name)
            assert dtype is not None, name
            feeds[name] = feeds[name].astype(dtype)
    return feeds


def test_lfm3_full_logits_and_twenty_cached_decode_steps(lfm3_artifacts):
    """Check all logits at prefill and each reused-state step, not only top-1."""
    reference, tokenizer, package, session = lfm3_artifacts
    encoded = tokenizer(_PROMPT, return_tensors="np")
    input_ids = encoded["input_ids"].astype(np.int64)
    mask = encoded["attention_mask"].astype(np.int64)
    positions = np.arange(input_ids.shape[1], dtype=np.int64)[None, :]
    feeds = _prefill_feeds(package, session, input_ids, mask, positions)
    hf_cache = None
    generated = []
    for _ in range(20):
        expected, hf_cache = torch_forward(
            reference, input_ids, mask, positions, past_key_values=hf_cache
        )
        actual = session.run(feeds)
        assert_logits_close(
            actual["logits"].astype(np.float32),
            expected.astype(np.float32),
            rtol=1e-3,
            atol=1e-3,
        )
        next_ids = np.argmax(actual["logits"][:, -1, :], axis=-1, keepdims=True)
        np.testing.assert_array_equal(
            next_ids, np.argmax(expected[:, -1, :], axis=-1, keepdims=True)
        )
        generated.append(int(next_ids[0, 0]))
        input_ids = next_ids.astype(np.int64)
        positions = np.array([[mask.shape[1]]], dtype=np.int64)
        mask = np.ones((1, mask.shape[1] + 1), dtype=np.int64)
        feeds = _make_decode_feeds(package.config, input_ids, mask, positions, actual)
    assert len(generated) == 20
    prompt_ids = encoded["input_ids"].astype(np.int64)
    generated_ids = OnnxGenerator(session, package.config).generate(
        prompt_ids, max_new_tokens=20, repetition_penalty=1.1
    )
    with torch.no_grad():
        expected_ids = (
            reference.generate(
                torch.from_numpy(prompt_ids).to(reference.device),
                max_new_tokens=20,
                do_sample=False,
                repetition_penalty=1.1,
            )
            .cpu()
            .numpy()
        )
    assert generated_ids.shape == expected_ids.shape == (1, prompt_ids.shape[1] + 20)
    np.testing.assert_array_equal(generated_ids, expected_ids)
    golden = json.loads((_GOLDEN_DIR / "lfm3-1_7b_generation.json").read_text())
    new_ids = generated_ids[0, prompt_ids.shape[1] :].tolist()
    assert len(golden["generated_tokens"]) == len(new_ids)
    assert new_ids == golden["generated_tokens"]


def test_lfm3_chunked_prefill_crosses_sliding_window(lfm3_artifacts):
    """Exercise width-7 convolution and sliding eviction with real checkpoint weights."""
    reference, tokenizer, package, session = lfm3_artifacts
    encoded = tokenizer("The river runs through the forest. " * 45, return_tensors="np")
    all_ids = encoded["input_ids"].astype(np.int64)
    window = package.config.sliding_window
    assert all_ids.shape[1] > window + 8
    all_ids = all_ids[:, : window + 9]
    hf_cache = None
    actual = None
    offset = 0
    for size in (window - 2, 5, 1, 5):
        input_ids = all_ids[:, offset : offset + size]
        mask = np.ones((1, offset + size), dtype=np.int64)
        positions = np.arange(offset, offset + size, dtype=np.int64)[None, :]
        expected, hf_cache = torch_forward(
            reference, input_ids, mask, positions, past_key_values=hf_cache
        )
        feeds = (
            _prefill_feeds(package, session, input_ids, mask, positions)
            if actual is None
            else _make_decode_feeds(package.config, input_ids, mask, positions, actual)
        )
        actual = session.run(feeds)
        assert_logits_close(
            actual["logits"].astype(np.float32),
            expected.astype(np.float32),
            rtol=1e-3,
            atol=1e-3,
        )
        for i, kind in enumerate(package.config.layer_types):
            if kind == "conv":
                assert actual[f"present.{i}.conv_state"].shape[-1] == 6
            else:
                assert actual[f"present.{i}.key"].shape[2] == offset + size
        offset += size
    assert offset == all_ids.shape[1]
