# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Gemma4 GGUF imports must ignore only provably unused shared-KV weights."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from types import SimpleNamespace

import pytest

from mobius._configs import Gemma4Config, QuantizationConfig
from mobius.integrations.gguf import build_from_gguf
from mobius.integrations.gguf._builder import (
    _config_tensor_name_mapper,
    _detect_quant_params,
    _load_quantized_state_dict,
    _normalize_gguf_weights,
    _preflight_quantization_report,
)
from mobius.integrations.gguf._config_mapping import gguf_to_config
from mobius.integrations.gguf._reader import GGUFModel
from mobius.integrations.gguf._reader_test import _write_gemma4_test_gguf
from mobius.integrations.gguf._tensor_mapping import map_gguf_to_hf_names
from mobius.models.gemma4 import Gemma4CausalLMModel


def _write_quantized_gemma4(
    path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    shared_layers: int = 3,
    tied_embedding: bool = False,
) -> None:
    from gguf import GGMLQuantizationType, GGUFWriter, quantize

    add_tensor = GGUFWriter.add_tensor

    def add_quantized_tensor(self, name, tensor, **kwargs):
        if tied_embedding and name == "output.weight":
            return
        if tensor.ndim == 2 and (tied_embedding or name != "token_embd.weight"):
            qtype = (
                GGMLQuantizationType.Q8_0
                if name == "token_embd.weight"
                else GGMLQuantizationType.Q4_0
            )
            tensor = quantize(tensor, qtype)
            kwargs["raw_dtype"] = qtype
        return add_tensor(self, name, tensor, **kwargs)

    monkeypatch.setattr(GGUFWriter, "add_tensor", add_quantized_tensor)
    _write_gemma4_test_gguf(path, num_kv_shared_layers=shared_layers)


@pytest.mark.parametrize("shared_layers", [0, 3])
@pytest.mark.parametrize("keep_quantized", [False, True])
def test_shared_kv_import(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    shared_layers: int,
    keep_quantized: bool,
) -> None:
    path = tmp_path / "gemma4.gguf"
    _write_quantized_gemma4(path, monkeypatch, shared_layers=shared_layers)
    package = build_from_gguf(path, keep_quantized=keep_quantized)
    initializers = package["model"].graph.initializers
    assert package.config.num_kv_shared_layers == shared_layers
    records = {record.name for record in package.gguf_quantization_report.tensor_records}
    unused = {
        f"blk.{layer}.{projection}.weight"
        for layer in range(6 - shared_layers, 6)
        for projection in ("attn_k", "attn_v", "attn_k_norm")
    }
    assert not records.intersection(unused)
    assert sum(
        stat.tensor_count for stat in package.gguf_quantization_report.source_qtype_census
    ) == len(records) + len(unused)
    for layer in range(6):
        for projection in ("attn_k", "attn_v", "attn_k_norm"):
            source = f"blk.{layer}.{projection}.weight"
            target = map_gguf_to_hf_names(source, "gemma4")
            assert (source in records) == (layer < 6 - shared_layers)
            targets = [name for name in (target, f"{target}_t") if name in initializers]
            assert bool(targets) == (layer < 6 - shared_layers)
            for name in targets:
                assert initializers[name].const_value is not None
        query = f"model.layers.{layer}.self_attn.q_proj.weight"
        assert query in initializers or f"{query}_t" in initializers


def test_tied_quantized_head_loads_and_saves(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise GGUF binding and persistence without source-identity lifecycle checks."""
    import numpy as np

    from mobius import ModelPackage, build_from_module
    from mobius.components import TiedQuantizedLMHead

    path = tmp_path / "tied-gemma4.gguf"
    _write_quantized_gemma4(path, monkeypatch, tied_embedding=True)
    source = GGUFModel(path)
    try:
        config = gguf_to_config(source)
        assert config.tie_word_embeddings
        assert "output.weight" not in source.tensor_names
        bits, block_size, sym = _detect_quant_params(source, "gemma4")
        config = dataclasses.replace(
            config,
            quantization=QuantizationConfig(
                bits=bits,
                group_size=block_size,
                quant_method="gguf",
                sym=sym,
                quantize_embeddings=True,
                quantize_lm_head=True,
                tie_word_embeddings=True,
            ),
        )
        module = Gemma4CausalLMModel(config)
        package = build_from_module(module, config)
        weights = _load_quantized_state_dict(
            source,
            "gemma4",
            module,
            config,
            name_mapper=_config_tensor_name_mapper("gemma4", config),
        )
        weights = _normalize_gguf_weights(weights, "gemma4", config)
        package.apply_weights(module.preprocess_weights(weights))
    finally:
        source.close()

    output = tmp_path / "package"
    package.save(output, progress_bar=False)
    loaded = ModelPackage.load(output)
    assert isinstance(module.lm_head, TiedQuantizedLMHead)
    assert module.lm_head.qweight is module.model.embed_tokens.qweight
    assert module.lm_head.scales is module.model.embed_tokens.scales
    assert module.lm_head.zero_points is module.model.embed_tokens.zero_points
    assert not any(name.startswith("lm_head.") for name in package["model"].graph.initializers)
    for name, initializer in package["model"].graph.initializers.items():
        assert initializer.const_value is not None
        restored = loaded["model"].graph.initializers[name]
        assert restored.const_value is not None
        np.testing.assert_array_equal(
            restored.const_value.numpy(), initializer.const_value.numpy()
        )


@pytest.mark.parametrize("shared_layers", [0, 18])
def test_mixed_k_quant_preflight_preserves_source_layers(shared_layers: int) -> None:
    """Production layer indices and Q4_K/Q6_K headers need no payload conversion."""
    from gguf import GGMLQuantizationType

    config = Gemma4Config(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=42,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        global_head_dim=16,
        vocab_size=256,
        num_kv_shared_layers=shared_layers,
        hidden_act="gelu_pytorch_tanh",
        layer_types=["sliding_attention", "sliding_attention", "full_attention"] * 14,
        quantization=QuantizationConfig(bits=4, group_size=32, sym=False),
    )
    tensors = [
        SimpleNamespace(
            name=f"blk.{layer}.attn_{projection}.weight",
            tensor_type=qtype,
            shape=(64, 32),
            n_bytes=n_bytes,
        )
        for layer in range(42)
        for projection, qtype, n_bytes in (
            ("k", GGMLQuantizationType.Q4_K, 8 * 144),
            ("v", GGMLQuantizationType.Q6_K, 8 * 210),
        )
    ]
    source = SimpleNamespace(reader_tensors=lambda: iter(tensors), _reader=None)
    report = _preflight_quantization_report(
        source,
        "gemma4",
        Gemma4CausalLMModel(config),
        config,
        preserve_quantization=True,
        target_bits=4,
        target_block_size=32,
        execution_provider="default",
        name_mapper=_config_tensor_name_mapper("gemma4", config),
    )
    assert len(report.tensor_records) == 2 * (42 - shared_layers)
    assert sum(stat.tensor_count for stat in report.source_qtype_census) == 84
    assert all(
        int(record.name.split(".")[1]) < 42 - shared_layers for record in report.tensor_records
    )

    tensors.append(
        SimpleNamespace(
            name="blk.42.attn_k.weight",
            tensor_type=GGMLQuantizationType.Q4_K,
            shape=(64, 32),
            n_bytes=8 * 144,
        )
    )
    with pytest.raises(ValueError, match=r"blk.42.attn_k.weight.*no corresponding graph"):
        _preflight_quantization_report(
            source,
            "gemma4",
            Gemma4CausalLMModel(config),
            config,
            preserve_quantization=True,
            target_bits=4,
            target_block_size=32,
            execution_provider="default",
            name_mapper=_config_tensor_name_mapper("gemma4", config),
        )


def test_unused_projections_do_not_select_quantization_target() -> None:
    from gguf import GGMLQuantizationType

    source = SimpleNamespace(
        tensor_items_raw=lambda: iter(
            [
                ("blk.0.attn_q.weight", None, GGMLQuantizationType.Q8_0, (64, 64)),
                ("blk.1.attn_k.weight", None, GGMLQuantizationType.Q4_K, (32, 64)),
                ("blk.1.attn_v.weight", None, GGMLQuantizationType.Q6_K, (32, 64)),
            ]
        )
    )
    config = Gemma4Config(num_hidden_layers=2, num_kv_shared_layers=1)
    assert _detect_quant_params(
        source,
        "gemma4",
        name_mapper=_config_tensor_name_mapper("gemma4", config),
    )[:2] == (8, 32)


@pytest.mark.parametrize("shared_layers", [0, 1, 2])
def test_mapper_is_architecture_and_layer_bounded(shared_layers: int) -> None:
    config = Gemma4Config(num_hidden_layers=2, num_kv_shared_layers=shared_layers)
    mapper = _config_tensor_name_mapper("gemma4", config)
    for name in (
        "blk.0.attn_k.weight",
        "blk.2.attn_k.weight",
        "blk.1.attn_q.weight",
        "blk.1.attn_output.weight",
        "blk.1.ffn_down.weight",
        "blk.1.attn_k.bias",
    ):
        assert mapper(name, "gemma4") == map_gguf_to_hf_names(name, "gemma4")
    assert mapper("blk.1.attn_k.weight", "llama") == map_gguf_to_hf_names(
        "blk.1.attn_k.weight", "llama"
    )
    assert _config_tensor_name_mapper("llama", config) is map_gguf_to_hf_names
