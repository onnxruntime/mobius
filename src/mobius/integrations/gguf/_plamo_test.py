# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from mobius._configs import ArchitectureConfig
from mobius.integrations.gguf._arch_registry import get_arch_spec
from mobius.integrations.gguf._builder import _raise_for_invalid_plamo_tensor_contract
from mobius.integrations.gguf._config_mapping import gguf_to_config
from mobius.integrations.gguf._spec import Support
from mobius.integrations.gguf._tensor_mapping import map_gguf_to_hf_names
from mobius.integrations.gguf._tensor_processors import process_tensors


class _FakePlamoGGUF:
    architecture = "plamo"

    def __init__(self):
        self.metadata = {
            "plamo.context_length": 4096,
            "plamo.embedding_length": 5120,
            "plamo.block_count": 40,
            "plamo.feed_forward_length": 16640,
            "plamo.attention.head_count": 40,
            "plamo.attention.head_count_kv": 5,
            "plamo.attention.layer_norm_rms_epsilon": 1e-6,
            "plamo.vocab_size": 17,
        }
        self._tensors: dict[str, tuple[tuple[int, ...], object]] = {
            "token_embd.weight": ((17, 5120), SimpleNamespace(value=0, name="F32")),
            "output_norm.weight": ((5120,), SimpleNamespace(value=0, name="F32")),
            "output.weight": ((17, 5120), SimpleNamespace(value=0, name="F32")),
        }
        for layer in range(40):
            prefix = f"blk.{layer}."
            self._tensors.update(
                {
                    prefix + "attn_norm.weight": (
                        (5120,),
                        SimpleNamespace(value=0, name="F32"),
                    ),
                    prefix + "attn_q.weight": (
                        (5120, 5120),
                        SimpleNamespace(value=0, name="F32"),
                    ),
                    prefix + "attn_k.weight": (
                        (640, 5120),
                        SimpleNamespace(value=0, name="F32"),
                    ),
                    prefix + "attn_v.weight": (
                        (640, 5120),
                        SimpleNamespace(value=0, name="F32"),
                    ),
                    prefix + "attn_output.weight": (
                        (5120, 5120),
                        SimpleNamespace(value=0, name="F32"),
                    ),
                    prefix + "ffn_gate.weight": (
                        (16640, 5120),
                        SimpleNamespace(value=0, name="F32"),
                    ),
                    prefix + "ffn_up.weight": (
                        (16640, 5120),
                        SimpleNamespace(value=0, name="F32"),
                    ),
                    prefix + "ffn_down.weight": (
                        (5120, 16640),
                        SimpleNamespace(value=0, name="F32"),
                    ),
                }
            )
        self.tensor_names = list(self._tensors)

    def get_metadata(self, key, default=None):
        return self.metadata.get(key, default)

    def tensor_items_raw(self):
        for name, (shape, qtype) in self._tensors.items():
            yield name, None, qtype, shape


def _small_config() -> ArchitectureConfig:
    config = ArchitectureConfig(
        model_type="plamo",
        vocab_size=17,
        hidden_size=8,
        intermediate_size=12,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=2,
        hidden_act="silu",
        max_position_embeddings=16,
        rope_type="default",
    )
    config._gguf_arch = "plamo"
    return config


def test_plamo_registry_claims_are_fail_closed() -> None:
    spec = get_arch_spec("plamo")
    assert spec.module_type == "gguf_plamo"
    assert spec.tensor_map_recipe == ("plamo",)
    assert spec.tensor_processor == "plamo"
    assert spec.graph is Support.SUPPORTED
    assert spec.quantized_import is Support.REJECTED
    assert spec.runtime is Support.DEFERRED


def test_plamo_config_extracts_only_fixed_contract() -> None:
    model = _FakePlamoGGUF()
    config = gguf_to_config(model)
    assert (
        config.hidden_size,
        config.intermediate_size,
        config.num_hidden_layers,
        config.num_attention_heads,
        config.num_key_value_heads,
        config.head_dim,
    ) == (5120, 16640, 40, 40, 5, 128)
    assert config.rope_type == "default"
    assert config.hidden_act == "silu"
    assert not config.tie_word_embeddings

    model.metadata["plamo.attention.head_count_kv"] = 8
    with pytest.raises(ValueError, match="head_count_kv=5"):
        gguf_to_config(model)


def test_plamo_tensor_mapping_is_suffix_exact() -> None:
    assert (
        map_gguf_to_hf_names("blk.7.attn_norm.weight", "plamo")
        == "model.layers.7.input_layernorm.weight"
    )
    assert (
        map_gguf_to_hf_names("blk.7.attn_output.weight", "plamo")
        == "model.layers.7.self_attn.o_proj.weight"
    )
    assert map_gguf_to_hf_names("blk.7.attn_q.bias", "plamo") is None
    assert map_gguf_to_hf_names("blk.7.ffn_norm.weight", "plamo") is None
    assert map_gguf_to_hf_names("blk.7.attn_qkv.weight", "plamo") is None


def test_plamo_tensor_closure_rejects_missing_extra_and_malformed() -> None:
    model = _FakePlamoGGUF()
    _raise_for_invalid_plamo_tensor_contract(model, keep_quantized=False)

    model._tensors["blk.0.ffn_norm.weight"] = (
        (5120,),
        SimpleNamespace(value=0, name="F32"),
    )
    with pytest.raises(ValueError, match="unexpected"):
        _raise_for_invalid_plamo_tensor_contract(model, keep_quantized=False)
    del model._tensors["blk.0.ffn_norm.weight"]

    model._tensors["blk.0.attn_q.weight"] = (
        (640, 5120),
        SimpleNamespace(value=0, name="F32"),
    )
    with pytest.raises(ValueError, match="malformed"):
        _raise_for_invalid_plamo_tensor_contract(model, keep_quantized=False)


def test_plamo_rejects_packed_shuffle_preservation() -> None:
    model = _FakePlamoGGUF()
    model._tensors["blk.0.attn_q.weight"] = (
        (5120, 5120),
        SimpleNamespace(value=10, name="Q2_K"),
    )
    with pytest.raises(ValueError, match="packed Q/output"):
        _raise_for_invalid_plamo_tensor_contract(model, keep_quantized=True)
    _raise_for_invalid_plamo_tensor_contract(model, keep_quantized=False)


def test_plamo_q_and_output_inverse_shuffles_restore_source_values() -> None:
    config = _small_config()
    source_q = torch.arange(64, dtype=torch.float32).reshape(8, 8)
    source_output = torch.arange(64, dtype=torch.float32).reshape(8, 8) + 100
    converted_q = source_q.reshape(2, 2, 2, 8).permute(1, 0, 2, 3).reshape(8, 8)
    converted_output = source_output.reshape(8, 2, 2, 2).permute(0, 2, 1, 3).reshape(8, 8)
    state_dict = {
        "model.layers.0.self_attn.q_proj.weight": converted_q,
        "model.layers.0.self_attn.o_proj.weight": converted_output,
        "model.layers.0.self_attn.k_proj.weight": torch.arange(32).reshape(4, 8),
    }
    processed = process_tensors(state_dict, config)
    torch.testing.assert_close(
        processed["model.layers.0.self_attn.q_proj.weight"],
        source_q,
    )
    torch.testing.assert_close(
        processed["model.layers.0.self_attn.o_proj.weight"],
        source_output,
    )
    torch.testing.assert_close(
        processed["model.layers.0.self_attn.k_proj.weight"],
        torch.arange(32).reshape(4, 8),
    )
