# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Gemma4 GGUF imports must ignore only provably unused shared-KV weights."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from mobius._configs import Gemma4Config, QuantizationConfig
from mobius.integrations.gguf import build_from_gguf
from mobius.integrations.gguf._builder import (
    _config_tensor_name_mapper,
    _detect_quant_params,
    _preflight_quantization_report,
)
from mobius.integrations.gguf._reader_test import _write_gemma4_test_gguf
from mobius.integrations.gguf._tensor_mapping import map_gguf_to_hf_names
from mobius.models.gemma4 import Gemma4CausalLMModel


@pytest.mark.parametrize("shared_layers", [0, 3])
@pytest.mark.parametrize("keep_quantized", [False, True])
def test_shared_kv_import(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    shared_layers: int,
    keep_quantized: bool,
) -> None:
    from gguf import GGMLQuantizationType, GGUFWriter, quantize

    add_tensor = GGUFWriter.add_tensor

    def add_quantized_tensor(self, name, tensor, **kwargs):
        if tensor.ndim == 2 and name != "token_embd.weight":
            tensor = quantize(tensor, GGMLQuantizationType.Q4_0)
            kwargs["raw_dtype"] = GGMLQuantizationType.Q4_0
        return add_tensor(self, name, tensor, **kwargs)

    monkeypatch.setattr(GGUFWriter, "add_tensor", add_quantized_tensor)
    path = tmp_path / "gemma4.gguf"
    _write_gemma4_test_gguf(path, num_kv_shared_layers=shared_layers)

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
