# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

from types import SimpleNamespace

import ml_dtypes
import numpy as np
import pytest

from mobius._testing.generation import OnnxSeq2SeqGenerator


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
