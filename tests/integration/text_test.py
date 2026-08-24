# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Integration tests for generic causal-language-model numerical parity."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from integration._support import (
    TEXT_MODELS,
    _get_config,
    _make_decode_feeds,
    _make_prefill_feeds,
    _make_session,
)
from mobius import build
from mobius._testing.comparison import (
    assert_logits_close,
)
from mobius._testing.torch_reference import (
    load_torch_model,
    torch_forward,
)


@pytest.mark.integration
@pytest.mark.integration_fast
@pytest.mark.parametrize("model_id,trust_remote_code", TEXT_MODELS)
class TestForwardNumerical:
    """Compare single forward pass logits between ONNX and PyTorch."""

    def test_prefill_logits_match(self, model_id: str, trust_remote_code: bool):
        """First forward pass (prefill) with a short prompt."""
        onnx_model = build(model_id, dtype="f32", load_weights=True)
        torch_model, tokenizer = load_torch_model(model_id)
        config = _get_config(model_id, trust_remote_code)

        prompt = "The capital of France is"
        tokens = tokenizer(prompt, return_tensors="np")
        input_ids = tokens["input_ids"].astype(np.int64)
        attention_mask = tokens["attention_mask"].astype(np.int64)
        seq_len = input_ids.shape[1]
        position_ids = np.arange(seq_len, dtype=np.int64)[np.newaxis, :]

        torch_logits, _ = torch_forward(torch_model, input_ids, attention_mask, position_ids)

        session = _make_session(onnx_model)
        feeds = _make_prefill_feeds(config, input_ids, attention_mask, position_ids)
        onnx_outputs = session.run(feeds)
        session.close()

        assert_logits_close(onnx_outputs["logits"], torch_logits, rtol=1e-3, atol=1e-3)

    def test_decode_step_logits_match(self, model_id: str, trust_remote_code: bool):
        """Second forward pass (single-token decode with KV cache)."""
        onnx_model = build(model_id, dtype="f32", load_weights=True)
        torch_model, tokenizer = load_torch_model(model_id)
        config = _get_config(model_id, trust_remote_code)

        prompt = "Hello world"
        tokens = tokenizer(prompt, return_tensors="np")
        input_ids = tokens["input_ids"].astype(np.int64)
        attention_mask = tokens["attention_mask"].astype(np.int64)
        seq_len = input_ids.shape[1]
        position_ids = np.arange(seq_len, dtype=np.int64)[np.newaxis, :]

        # Prefill
        torch_logits_1, torch_kv = torch_forward(
            torch_model, input_ids, attention_mask, position_ids
        )

        session = _make_session(onnx_model)
        feeds = _make_prefill_feeds(config, input_ids, attention_mask, position_ids)
        onnx_out_1 = session.run(feeds)

        # Decode step
        next_token = np.argmax(torch_logits_1[:, -1, :], axis=-1, keepdims=True)
        decode_input_ids = next_token.astype(np.int64)
        decode_attention_mask = np.ones((1, seq_len + 1), dtype=np.int64)
        decode_position_ids = np.array([[seq_len]], dtype=np.int64)

        torch_logits_2, _ = torch_forward(
            torch_model,
            decode_input_ids,
            decode_attention_mask,
            decode_position_ids,
            past_key_values=torch_kv,
        )

        decode_feeds = _make_decode_feeds(
            config, decode_input_ids, decode_attention_mask, decode_position_ids, onnx_out_1
        )
        onnx_out_2 = session.run(decode_feeds)
        session.close()

        assert_logits_close(onnx_out_2["logits"], torch_logits_2, rtol=1e-3, atol=1e-3)


# GraniteSWA's sliding layers only differ from full attention once the prompt
# exceeds the 128-token local window. Its learnable sink is also only expressed
# by HuggingFace's eager attention reference path.
_GRANITE_SWA_MODEL_ID = "ibm-granite/granite-swash-2b"
_GRANITE_SWA_REVISION = "af1e3227100b61088eead48389ab5409b5d0e39c"


def _granite_swa_long_prompt(tokenizer, min_tokens: int) -> np.ndarray:
    """Tokenize a prompt guaranteed to exceed ``min_tokens`` tokens."""
    sentence = (
        "The city archives recorded every harvest, every flood, and every "
        "quiet year in between, so that later readers could trace how the "
        "valley changed. "
    )
    text = sentence
    while True:
        input_ids = tokenizer(text, return_tensors="np")["input_ids"].astype(np.int64)
        if input_ids.shape[1] > min_tokens:
            return input_ids
        text += sentence


def _load_granite_swa_reference():
    """Load the pinned GraniteSWA checkpoint with the eager attention kernel."""
    model, tokenizer = load_torch_model(
        _GRANITE_SWA_MODEL_ID,
        revision=_GRANITE_SWA_REVISION,
        trust_remote_code=False,
        attn_implementation="eager",
    )
    assert model.config._attn_implementation == "eager"
    return model, tokenizer


@pytest.mark.integration
def test_granite_swa_prefill_logits_match():
    """Prefill past the sliding window matches HuggingFace eager attention."""
    onnx_model = build(
        _GRANITE_SWA_MODEL_ID,
        revision=_GRANITE_SWA_REVISION,
        dtype="f32",
        load_weights=True,
    )
    torch_model, tokenizer = _load_granite_swa_reference()
    config = _get_config(_GRANITE_SWA_MODEL_ID, revision=_GRANITE_SWA_REVISION)

    sliding_window = config.sliding_window
    assert sliding_window == 128
    assert config.layer_types is not None
    assert "sliding_attention" in config.layer_types
    assert "full_attention" in config.layer_types

    input_ids = _granite_swa_long_prompt(tokenizer, min_tokens=sliding_window + 32)
    seq_len = input_ids.shape[1]
    attention_mask = np.ones((1, seq_len), dtype=np.int64)
    position_ids = np.arange(seq_len, dtype=np.int64)[np.newaxis, :]
    torch_logits, _ = torch_forward(torch_model, input_ids, attention_mask, position_ids)

    session = _make_session(onnx_model)
    try:
        onnx_outputs = session.run(
            _make_prefill_feeds(config, input_ids, attention_mask, position_ids)
        )
    finally:
        session.close()

    assert_logits_close(onnx_outputs["logits"], torch_logits, rtol=1e-3, atol=1e-3)


@pytest.mark.integration
def test_granite_swa_decode_step_logits_match():
    """A cached decode step past the window still matches HuggingFace."""
    onnx_model = build(
        _GRANITE_SWA_MODEL_ID,
        revision=_GRANITE_SWA_REVISION,
        dtype="f32",
        load_weights=True,
    )
    torch_model, tokenizer = _load_granite_swa_reference()
    config = _get_config(_GRANITE_SWA_MODEL_ID, revision=_GRANITE_SWA_REVISION)

    input_ids = _granite_swa_long_prompt(tokenizer, min_tokens=config.sliding_window + 32)
    seq_len = input_ids.shape[1]
    attention_mask = np.ones((1, seq_len), dtype=np.int64)
    position_ids = np.arange(seq_len, dtype=np.int64)[np.newaxis, :]

    with torch.no_grad():
        hf_prefill = torch_model(
            input_ids=torch.from_numpy(input_ids),
            attention_mask=torch.from_numpy(attention_mask),
            position_ids=torch.from_numpy(position_ids),
            use_cache=True,
        )
    hf_cache = hf_prefill.past_key_values

    sliding_idx = config.layer_types.index("sliding_attention")
    full_idx = config.layer_types.index("full_attention")
    assert hf_cache.layers[sliding_idx].keys.shape[2] < seq_len
    assert hf_cache.layers[full_idx].keys.shape[2] == seq_len

    next_token = np.array([[int(np.argmax(hf_prefill.logits[0, -1].numpy()))]], dtype=np.int64)
    decode_attention_mask = np.ones((1, seq_len + 1), dtype=np.int64)
    decode_position_ids = np.array([[seq_len]], dtype=np.int64)
    with torch.no_grad():
        hf_decode = torch_model(
            input_ids=torch.from_numpy(next_token),
            attention_mask=torch.from_numpy(decode_attention_mask),
            position_ids=torch.from_numpy(decode_position_ids),
            past_key_values=hf_cache,
            use_cache=True,
        )
    torch_logits_2 = hf_decode.logits.numpy()

    session = _make_session(onnx_model)
    try:
        onnx_out_1 = session.run(
            _make_prefill_feeds(config, input_ids, attention_mask, position_ids)
        )
        onnx_out_2 = session.run(
            _make_decode_feeds(
                config,
                next_token,
                decode_attention_mask,
                decode_position_ids,
                onnx_out_1,
            )
        )
    finally:
        session.close()

    assert_logits_close(onnx_out_2["logits"], torch_logits_2, rtol=1e-3, atol=1e-3)
