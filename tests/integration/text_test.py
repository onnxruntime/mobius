# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Integration tests for generic causal-language-model numerical parity."""

from __future__ import annotations

import numpy as np
import pytest

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
