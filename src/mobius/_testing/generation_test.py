# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

from types import SimpleNamespace

import ml_dtypes
import numpy as np
import pytest

from mobius._testing.generation import OnnxGenerator, OnnxSeq2SeqGenerator


@pytest.mark.parametrize("penalty", [0.5, 1.0, 1.1, 2.0])
def test_causal_repetition_penalty_matches_transformers_per_row(penalty):
    import torch
    from transformers.generation.logits_process import RepetitionPenaltyLogitsProcessor

    logits = np.array([[[0.8, 1.1, -0.5, -0.9]], [[-0.8, -1.1, -0.5, -0.9]]], np.float32)
    config = SimpleNamespace(
        num_hidden_layers=0, num_key_value_heads=1, head_dim=1, layer_types=[]
    )
    session = SimpleNamespace(run=lambda feeds: {"logits": logits})
    prompt = np.array([[1, 1], [2, 3]], dtype=np.int64)
    actual = OnnxGenerator(session, config).generate(
        prompt, max_new_tokens=3, repetition_penalty=penalty
    )
    expected = torch.from_numpy(prompt)
    processor = RepetitionPenaltyLogitsProcessor(penalty)
    for _ in range(3):
        scores = processor(expected, torch.from_numpy(logits[:, -1, :].copy()))
        expected = torch.cat([expected, scores.argmax(-1, keepdim=True)], dim=-1)
    np.testing.assert_array_equal(actual, expected.numpy())


@pytest.mark.parametrize("penalty", [0.0, -1.0, float("inf"), float("nan")])
def test_causal_repetition_penalty_rejects_invalid_values(penalty):
    with pytest.raises(ValueError, match="finite and positive"):
        OnnxGenerator(None, None).generate(
            np.array([[1]], dtype=np.int64), repetition_penalty=penalty
        )


@pytest.mark.parametrize("hidden_dtype", [np.float32, np.float16, ml_dtypes.bfloat16])
def test_seq2seq_cache_uses_each_input_dtype_and_reuses_outputs(hidden_dtype):
    hidden = np.ones((1, 2, 4), dtype=hidden_dtype)
    dtypes = {
        "past_key_values.0.self.key": np.dtype(ml_dtypes.bfloat16),
        "past_key_values.0.self.value": np.dtype(np.float16),
        "past_key_values.0.cross.key": np.dtype(np.float32),
        "past_key_values.0.cross.value": None,
    }
    previous = {}
    calls = []

    def decode(feeds):
        calls.append(feeds)
        assert feeds["encoder_hidden_states"] is hidden
        assert feeds["input_ids"].dtype == np.int64
        assert feeds["attention_mask"].dtype == np.int64
        outputs = {"logits": np.array([[[0, 1, 2]]], dtype=np.float32)}
        for name, dtype in dtypes.items():
            value = feeds[name]
            assert value.dtype == (dtype or np.dtype(np.float32))
            if len(calls) == 1:
                assert value.shape == (1, 2, 0, 2)
            else:
                assert value is previous[name]
            present = np.ones((1, 2, 1, 2), dtype=value.dtype)
            outputs[name.replace("past_key_values.", "present.")] = present
            previous[name] = present
        return outputs

    encoder = SimpleNamespace(run=lambda feeds: {"last_hidden_state": hidden})
    decoder = SimpleNamespace(
        input_names=["input_ids", "encoder_hidden_states", "attention_mask", *dtypes],
        get_input_dtype=dtypes.get,
        run=decode,
    )
    config = SimpleNamespace(num_key_value_heads=2, head_dim=2)
    result = OnnxSeq2SeqGenerator(encoder, decoder, config).generate(
        np.array([[3, 4]], dtype=np.int64), max_new_tokens=2
    )
    np.testing.assert_array_equal(result, [[0, 2, 2]])
    assert len(calls) == 2
