"""Keep TRT's Q/K/V fixed; find which mask reproduces its attention output.

For the unpadded, batch-one Qwen3 diagnostic engine only.
HF hooks and RoPE alignment live in _attention_support.py.
"""

from __future__ import annotations

import numpy as np
import torch

from tensorrt_debug._attention_support import AttentionReference
from tensorrt_debug.comparison import compare_tensor
from tensorrt_debug.inspect_cache import snapshot_cache


class AttentionInspector(AttentionReference):
    """Compare after HF forward; hook setup and cleanup are inherited."""

    def compare(self, feeds, position):
        actual_query = self.runner.read_tensor("probe.query").astype(np.float32)
        actual_attention = self.runner.read_tensor("probe.attention").astype(np.float32)
        self.compare_with_hf(feeds, actual_query, actual_attention)
        self._compare_masks(actual_query, actual_attention, position)

    def _compare_masks(self, actual_query, actual_attention, position):
        key = torch.from_numpy(snapshot_cache(self.runner, "key_cache.0").astype(np.float32))
        value = torch.from_numpy(
            snapshot_cache(self.runner, "value_cache.0").astype(np.float32)
        )
        # GQA: repeat each KV head for its group of query heads.
        groups = actual_query.shape[1] // key.shape[1]
        key = key.repeat_interleave(groups, dim=1)
        value = value.repeat_interleave(groups, dim=1)
        scores = torch.from_numpy(actual_query) @ key.transpose(-1, -2)
        scores *= self.attention.scaling
        length = actual_query.shape[2]
        key_positions = torch.arange(key.shape[2])[None, :]
        query_positions = torch.arange(length)[:, None]
        # At cached decode position 22, these hypotheses allow slots 0..22 vs only 0.
        for hypothesis, offset in (
            ("correct cached causal mask", position),
            ("top-left causal mask", 0),
        ):
            mask = key_positions <= query_positions + offset
            masked_scores = scores.masked_fill(~mask, -torch.inf)
            probabilities = torch.softmax(masked_scores, dim=-1)
            reconstructed = probabilities @ value
            expected = reconstructed.transpose(1, 2).reshape(1, length, -1).numpy()
            compare_tensor(
                f"TRT attention vs own QKV + {hypothesis}", actual_attention, expected
            )
