# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mobius._configs import QuantizationConfig
from mobius._registry import registry
from mobius.components import QuantizedLinear, RMSNormBias
from mobius.integrations.gguf._arch_registry import get_arch_spec
from mobius.integrations.gguf._builder import _validate_gguf_model
from mobius.integrations.gguf._config_mapping import gguf_to_config
from mobius.integrations.gguf._tensor_mapping import map_gguf_to_hf_names
from mobius.models.apertus import ApertusCausalLMModel
from mobius.tasks import CausalLMTask

_HIDDEN = 8
_HEADS = 2
_KV_HEADS = 1
_HEAD_DIM = 4
_INTERMEDIATE = 16
_LAYERS = 2
_VOCAB = 32


class _FakeGGUF:
    def __init__(
        self,
        metadata: dict,
        tensors: dict[str, np.ndarray],
        *,
        qtypes: dict[str, int] | None = None,
    ):
        self.architecture = "apertus"
        self.metadata = metadata
        self._tensors = tensors
        self.tensor_names = list(tensors)
        self._qtypes = qtypes or {}

    def get_metadata(self, key, default=None):
        return self.metadata.get(key, default)

    def get_tensor(self, name):
        return self._tensors[name]

    def tensor_items_raw(self):
        for name, tensor in self._tensors.items():
            yield name, None, SimpleNamespace(value=self._qtypes.get(name, 0)), tensor.shape


def _metadata() -> dict:
    return {
        "apertus.embedding_length": _HIDDEN,
        "apertus.feed_forward_length": _INTERMEDIATE,
        "apertus.block_count": _LAYERS,
        "apertus.attention.head_count": _HEADS,
        "apertus.attention.head_count_kv": _KV_HEADS,
        "apertus.attention.layer_norm_rms_epsilon": 1e-5,
        "apertus.context_length": 32,
        "apertus.rope.freq_base": 12_000_000.0,
        "apertus.rope.dimension_count": _HEAD_DIM,
        "apertus.vocab_size": _VOCAB,
        "apertus.xielu.alpha_p": [-0.2, -0.1],
        "apertus.xielu.alpha_n": [-1.0, -0.9],
        "apertus.xielu.beta": 0.5,
        "apertus.xielu.eps": -1e-6,
    }


def _tensors(*, biases: bool = True) -> dict[str, np.ndarray]:
    tensors = {
        "token_embd.weight": np.zeros((_VOCAB, _HIDDEN), np.float32),
        "output_norm.weight": np.ones((_HIDDEN,), np.float32),
        "output.weight": np.zeros((_VOCAB, _HIDDEN), np.float32),
        "rope_freqs.weight": np.asarray([1.0, 8.0], np.float32),
    }
    for layer in range(_LAYERS):
        prefix = f"blk.{layer}."
        tensors.update(
            {
                prefix + "attn_norm.weight": np.ones((_HIDDEN,), np.float32),
                prefix + "attn_q.weight": np.zeros((_HIDDEN, _HIDDEN), np.float32),
                prefix + "attn_k.weight": np.zeros((_HEAD_DIM, _HIDDEN), np.float32),
                prefix + "attn_v.weight": np.zeros((_HEAD_DIM, _HIDDEN), np.float32),
                prefix + "attn_output.weight": np.zeros((_HIDDEN, _HIDDEN), np.float32),
                prefix + "attn_q_norm.weight": np.ones((_HEAD_DIM,), np.float32),
                prefix + "attn_k_norm.weight": np.ones((_HEAD_DIM,), np.float32),
                prefix + "ffn_norm.weight": np.ones((_HIDDEN,), np.float32),
                prefix + "ffn_up.weight": np.zeros((_INTERMEDIATE, _HIDDEN), np.float32),
                prefix + "ffn_down.weight": np.zeros((_HIDDEN, _INTERMEDIATE), np.float32),
            }
        )
        if biases:
            tensors[prefix + "attn_q_norm.bias"] = np.zeros((_HEAD_DIM,), np.float32)
            tensors[prefix + "attn_k_norm.bias"] = np.zeros((_HEAD_DIM,), np.float32)
            tensors[prefix + "attn_output.bias"] = np.zeros((_HIDDEN,), np.float32)
    return tensors


def test_apertus_registry_promotes_exact_float_and_quantized_graph_import() -> None:
    spec = get_arch_spec("apertus")
    assert spec.is_importable
    assert spec.model_type == "apertus"
    assert spec.tensor_map_recipe == ("llama", "apertus_extras")
    assert spec.config_postprocessor == "apertus"
    assert spec.quantized_import.value == "supported"
    assert spec.runtime.value == "deferred"


def test_apertus_consumes_serialized_rope_and_xielu_values_exactly() -> None:
    model = _FakeGGUF(_metadata(), _tensors())
    config = gguf_to_config(model)

    assert config.rope_type == "longrope"
    assert config.original_max_position_embeddings == config.max_position_embeddings
    assert config.rope_scaling == {
        "short_factor": [1.0, 8.0],
        "long_factor": [1.0, 8.0],
    }
    assert config.xielu_alpha_p == (-0.2, -0.1)
    assert config.xielu_alpha_n == (-1.0, -0.9)
    assert config.xielu_beta == (0.5, 0.5)
    assert config.xielu_eps == (-1e-6, -1e-6)
    assert config.attn_q_norm_biases == (True, True)
    assert config.attn_k_norm_biases == (True, True)
    assert config.attn_o_bias


def test_apertus_maps_qk_norm_and_output_bias_without_value_transform() -> None:
    assert (
        map_gguf_to_hf_names("blk.1.attn_q_norm.bias", "apertus")
        == "model.layers.1.self_attn.q_norm.bias"
    )
    assert (
        map_gguf_to_hf_names("blk.1.attn_k_norm.weight", "apertus")
        == "model.layers.1.self_attn.k_norm.weight"
    )
    assert (
        map_gguf_to_hf_names("blk.1.attn_output.bias", "apertus")
        == "model.layers.1.self_attn.o_proj.bias"
    )
    assert map_gguf_to_hf_names("rope_freqs.weight", "apertus") is None


def test_apertus_rejects_quantized_or_malformed_serialized_rope_factors() -> None:
    tensors = _tensors()
    with pytest.raises(ValueError, match="F32/F16/BF16"):
        gguf_to_config(
            _FakeGGUF(
                _metadata(),
                tensors,
                qtypes={"rope_freqs.weight": 2},
            )
        )

    tensors = _tensors()
    tensors["rope_freqs.weight"] = np.ones((3,), np.float32)
    with pytest.raises(ValueError, match=r"shape \(2,\)"):
        gguf_to_config(_FakeGGUF(_metadata(), tensors))


def test_apertus_consumes_complete_longrope_factor_pair_exactly() -> None:
    tensors = _tensors()
    tensors.pop("rope_freqs.weight")
    tensors["rope_factors_short.weight"] = np.asarray([1.0, 2.0], np.float32)
    tensors["rope_factors_long.weight"] = np.asarray([4.0, 8.0], np.float32)
    metadata = _metadata()
    metadata["apertus.rope.scaling.original_context_length"] = 16

    config = gguf_to_config(_FakeGGUF(metadata, tensors))

    assert config.rope_scaling == {
        "short_factor": [1.0, 2.0],
        "long_factor": [4.0, 8.0],
    }
    assert config.original_max_position_embeddings == 16

    metadata.pop("apertus.rope.scaling.original_context_length")
    with pytest.raises(ValueError, match="original_context_length"):
        gguf_to_config(_FakeGGUF(metadata, tensors))


def test_apertus_tensor_contract_and_tiny_graph_close_all_owned_values() -> None:
    model = _FakeGGUF(_metadata(), _tensors())
    _validate_gguf_model(model, source="synthetic-apertus.gguf")
    config = gguf_to_config(model)
    module = ApertusCausalLMModel(config)

    assert isinstance(module.model.layers[0].self_attn.q_norm, RMSNormBias)
    assert isinstance(module.model.layers[0].self_attn.k_norm, RMSNormBias)
    package = CausalLMTask().build(module, config)
    initializers = package["model"].graph.initializers
    for layer in range(_LAYERS):
        prefix = f"model.layers.{layer}.mlp.act_fn."
        for name in ("alpha_p", "alpha_n", "beta", "eps"):
            assert initializers[prefix + name].const_value is not None
    assert initializers["model.rotary_emb.cos_cache"].const_value is not None
    assert initializers["model.rotary_emb.sin_cache"].const_value is not None


def test_apertus_quantized_mlp_keeps_packed_linear_modules() -> None:
    config = gguf_to_config(_FakeGGUF(_metadata(), _tensors()))
    config = dataclasses.replace(
        config,
        quantization=QuantizationConfig(
            bits=4,
            group_size=16,
            quant_method="gguf",
            sym=True,
        ),
    )
    module = ApertusCausalLMModel(config)
    assert isinstance(module.model.layers[0].mlp.up_proj, QuantizedLinear)
    assert isinstance(module.model.layers[0].mlp.down_proj, QuantizedLinear)


def test_apertus_tiny_prefill_and_decode_execute() -> None:
    from mobius._testing.ort_inference import OnnxModelSession

    tensors = _tensors()
    rng = np.random.default_rng(0)
    for name, tensor in tensors.items():
        if tensor.ndim == 2 and "rope_" not in name:
            tensors[name] = rng.normal(0.0, 0.02, tensor.shape).astype(np.float32)
    config = gguf_to_config(_FakeGGUF(_metadata(), tensors))
    module = registry.get("apertus")(config)
    package = CausalLMTask().build(module, config)
    state_dict = {
        mapped: torch.from_numpy(tensor)
        for name, tensor in tensors.items()
        if (mapped := map_gguf_to_hf_names(name, "apertus")) is not None
    }
    package.apply_weights(state_dict)

    session = OnnxModelSession(package["model"])
    try:
        prefill = session.run(
            {
                "input_ids": np.array([[1, 2]], np.int64),
                "attention_mask": np.ones((1, 2), np.int64),
                "position_ids": np.array([[0, 1]], np.int64),
                "past_key_values.0.key": np.empty((1, _KV_HEADS, 0, _HEAD_DIM), np.float32),
                "past_key_values.0.value": np.empty((1, _KV_HEADS, 0, _HEAD_DIM), np.float32),
                "past_key_values.1.key": np.empty((1, _KV_HEADS, 0, _HEAD_DIM), np.float32),
                "past_key_values.1.value": np.empty((1, _KV_HEADS, 0, _HEAD_DIM), np.float32),
            }
        )
        decode = session.run(
            {
                "input_ids": np.array([[3]], np.int64),
                "attention_mask": np.ones((1, 3), np.int64),
                "position_ids": np.array([[2]], np.int64),
                **{
                    name.replace("present.", "past_key_values."): value
                    for name, value in prefill.items()
                    if name.startswith("present.")
                },
            }
        )
    finally:
        session.close()

    assert prefill["logits"].shape == (1, 2, _VOCAB)
    assert decode["logits"].shape == (1, 1, _VOCAB)
    assert np.isfinite(prefill["logits"]).all()
    assert np.isfinite(decode["logits"]).all()
