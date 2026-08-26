# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mobius._registry import registry
from mobius._testing.ort_inference import OnnxModelSession
from mobius.integrations.gguf._arch_registry import get_arch_spec
from mobius.integrations.gguf._builder import (
    _raise_for_invalid_talkie_tensor_contract,
    build_from_gguf,
)
from mobius.integrations.gguf._config_mapping import gguf_to_config
from mobius.integrations.gguf._spec import Support
from mobius.integrations.gguf._tensor_mapping import map_gguf_to_hf_names
from mobius.models.talkie import TalkieForCausalLM
from mobius.tasks import CausalLMTask


class _FakeTalkieGGUF:
    architecture = "talkie"

    def __init__(self):
        self.metadata = {
            "talkie.context_length": 32,
            "talkie.embedding_length": 4,
            "talkie.feed_forward_length": 8,
            "talkie.block_count": 1,
            "talkie.attention.head_count": 1,
            "talkie.attention.head_count_kv": 1,
            "talkie.attention.layer_norm_rms_epsilon": 1e-5,
            "talkie.rope.freq_base": 10_000.0,
            "talkie.rope.dimension_count": 4,
            "talkie.logit_scale": 0.5,
            "talkie.vocab_size": 8,
        }
        self.tensors = _talkie_tensors()
        self.tensor_names = list(self.tensors)

    def get_metadata(self, key, default=None):
        return self.metadata.get(key, default)

    def get_tensor(self, name):
        return self.tensors[name]

    def tensor_items_raw(self):
        for name, value in self.tensors.items():
            yield name, None, SimpleNamespace(value=0, name="F32"), value.shape

    def tensor_items(self):
        return self.tensors.items()

    @property
    def num_tensors(self):
        return len(self.tensors)


def _talkie_tensors() -> dict[str, np.ndarray]:
    hidden = 4
    intermediate = 8
    return {
        "token_embd.weight": np.arange(32, dtype=np.float32).reshape(8, hidden) / 16,
        "output.weight": np.arange(32, dtype=np.float32).reshape(8, hidden) / 32,
        "blk.0.attn_q.weight": np.eye(hidden, dtype=np.float32),
        "blk.0.attn_k.weight": np.eye(hidden, dtype=np.float32),
        "blk.0.attn_v.weight": np.eye(hidden, dtype=np.float32),
        "blk.0.attn_output.weight": np.eye(hidden, dtype=np.float32),
        "blk.0.attn_q_norm.weight": np.array([[1.25]], dtype=np.float32),
        "blk.0.ffn_gate.weight": np.zeros((intermediate, hidden), dtype=np.float32),
        "blk.0.ffn_up.weight": np.zeros((intermediate, hidden), dtype=np.float32),
        "blk.0.ffn_down.weight": np.zeros((hidden, intermediate), dtype=np.float32),
        "blk.0.layer_output_scale.weight": np.array([0.25], dtype=np.float32),
    }


def _rms_norm(value: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    return value / np.sqrt(np.mean(value * value, axis=-1, keepdims=True) + eps)


def _talkie_reference(
    source: _FakeTalkieGGUF,
    input_ids: np.ndarray,
    *,
    past_key: np.ndarray | None = None,
    past_value: np.ndarray | None = None,
    start_position: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tensors = source.tensors
    hidden = _rms_norm(tensors["token_embd.weight"][input_ids])
    positions = np.arange(
        start_position, start_position + input_ids.shape[1], dtype=np.float32
    )
    frequencies = positions[:, None] * np.array([1.0, 0.01], dtype=np.float32)
    cos = np.concatenate([np.cos(frequencies), np.cos(frequencies)], axis=-1)[None]
    sin = -np.concatenate([np.sin(frequencies), np.sin(frequencies)], axis=-1)[None]

    def inverse_rope(value: np.ndarray) -> np.ndarray:
        first, second = np.split(value, 2, axis=-1)
        rotated = np.concatenate([-second, first], axis=-1)
        return value * cos + rotated * sin

    query = _rms_norm(inverse_rope(hidden)) * tensors["blk.0.attn_q_norm.weight"][0]
    key = _rms_norm(inverse_rope(hidden))
    value = hidden
    if past_key is not None:
        key = np.concatenate([past_key, key], axis=1)
        value = np.concatenate([past_value, value], axis=1)
    scores = query @ np.swapaxes(key, -1, -2) / np.sqrt(hidden.shape[-1])
    query_positions = np.arange(start_position, start_position + input_ids.shape[1])
    key_positions = np.arange(key.shape[1])
    scores = np.where(
        key_positions[None, None, :] <= query_positions[None, :, None],
        scores,
        -np.inf,
    )
    weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    weights /= weights.sum(axis=-1, keepdims=True)
    attended = weights @ value
    block_output = hidden + attended + hidden * tensors["blk.0.layer_output_scale.weight"]
    final_hidden = _rms_norm(block_output)
    logits = final_hidden @ tensors["output.weight"].T * 0.5
    return logits, key, value


def test_talkie_uses_dedicated_causal_graph_and_exact_float_policy() -> None:
    spec = get_arch_spec("talkie")
    assert spec.model_type == "talkie"
    assert spec.quantized_import is Support.REJECTED
    assert spec.runtime is Support.DEFERRED
    assert registry.get("talkie") is TalkieForCausalLM

    source = _FakeTalkieGGUF()
    _raise_for_invalid_talkie_tensor_contract(source)
    config = gguf_to_config(source)
    assert config.hidden_act == "silu"
    assert config.logit_scale == pytest.approx(0.5)
    assert not config.tie_word_embeddings


def test_talkie_build_from_gguf_returns_causal_cache_package() -> None:
    package = build_from_gguf(
        "talkie.gguf",
        keep_quantized=False,
        _gguf_model=_FakeTalkieGGUF(),
    )
    graph = package["model"].graph
    input_names = {value.name for value in graph.inputs}
    output_names = {value.name for value in graph.outputs}
    assert {"input_ids", "attention_mask", "position_ids"} <= input_names
    assert {"logits", "present.0.key", "present.0.value"} <= output_names


def test_talkie_weight_mapping_owns_every_and_only_graph_parameter() -> None:
    source = _FakeTalkieGGUF()
    config = gguf_to_config(source)
    graph = CausalLMTask().build(TalkieForCausalLM(config), config)["model"].graph
    mapped = {
        map_gguf_to_hf_names(name, "talkie") for name in source.tensor_names
    }
    graph_weights = {
        name.removesuffix("_t")
        for name in graph.initializers
        if not name.startswith("const_") and "rotary_emb" not in name
    }
    assert None not in mapped
    assert graph_weights == mapped
    assert not any("layernorm" in name.lower() for name in graph_weights)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing", "missing"),
        ("unexpected_norm", "unexpected"),
        ("vector_skip", "malformed"),
        ("scaled_rope", "unscaled"),
        ("gqa", "geometry"),
        ("missing_logit_scale", "missing"),
    ],
)
def test_talkie_tensor_contract_fails_closed(mutation: str, message: str) -> None:
    source = _FakeTalkieGGUF()
    if mutation == "missing":
        source.tensors.pop("blk.0.ffn_down.weight")
    elif mutation == "unexpected_norm":
        source.tensors["output_norm.weight"] = np.ones(4, dtype=np.float32)
    elif mutation == "vector_skip":
        source.tensors["blk.0.layer_output_scale.weight"] = np.ones(4, dtype=np.float32)
    elif mutation == "scaled_rope":
        source.metadata["talkie.rope.scaling.type"] = "linear"
    elif mutation == "gqa":
        source.metadata["talkie.attention.head_count"] = 2
    else:
        source.metadata.pop("talkie.logit_scale")
    source.tensor_names = list(source.tensors)
    with pytest.raises(ValueError, match=message):
        _raise_for_invalid_talkie_tensor_contract(source)


def test_talkie_nonzero_prefill_decode_and_logit_scale() -> None:
    source = _FakeTalkieGGUF()
    config = gguf_to_config(source)
    package = CausalLMTask().build(TalkieForCausalLM(config), config)
    package.apply_weights(
        {
            mapped: torch.from_numpy(value.copy())
            for name, value in source.tensors.items()
            if (mapped := map_gguf_to_hf_names(name, "talkie")) is not None
        }
    )
    session = OnnxModelSession(package["model"])
    try:
        prompt = np.array([[1, 2, 3]], dtype=np.int64)
        prefill = session.run(
            {
                "input_ids": prompt,
                "attention_mask": np.ones_like(prompt),
                "position_ids": np.arange(3, dtype=np.int64)[None, :],
                "past_key_values.0.key": np.zeros((1, 1, 0, 4), dtype=np.float32),
                "past_key_values.0.value": np.zeros((1, 1, 0, 4), dtype=np.float32),
            }
        )
        expected, reference_key, reference_value = _talkie_reference(source, prompt)
        np.testing.assert_allclose(prefill["logits"], expected, rtol=2e-4, atol=2e-4)
        np.testing.assert_allclose(
            prefill["present.0.key"][:, 0], reference_key, rtol=2e-4, atol=2e-4
        )
        np.testing.assert_allclose(
            prefill["present.0.value"][:, 0], reference_value, rtol=2e-4, atol=2e-4
        )
        assert prefill["present.0.key"].shape == (1, 1, 3, 4)
        assert prefill["present.0.value"].shape == (1, 1, 3, 4)

        token = np.array([[4]], dtype=np.int64)
        decode = session.run(
            {
                "input_ids": token,
                "attention_mask": np.ones((1, 4), dtype=np.int64),
                "position_ids": np.array([[3]], dtype=np.int64),
                "past_key_values.0.key": prefill["present.0.key"],
                "past_key_values.0.value": prefill["present.0.value"],
            }
        )
        expected_decode, reference_decode_key, _ = _talkie_reference(
            source,
            token,
            past_key=reference_key,
            past_value=reference_value,
            start_position=3,
        )
        np.testing.assert_allclose(
            decode["logits"], expected_decode, rtol=2e-4, atol=2e-4
        )
        assert decode["present.0.key"].shape == (1, 1, 4, 4)
        np.testing.assert_allclose(
            decode["present.0.key"][:, 0],
            reference_decode_key,
            rtol=2e-4,
            atol=2e-4,
        )
    finally:
        session.close()
