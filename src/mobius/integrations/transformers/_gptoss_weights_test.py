# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for bounded native GPT-OSS MXFP4 safetensors loading."""

from __future__ import annotations

import gc
import json
import weakref

import onnx_ir as ir
import pytest
import safetensors.torch
import torch
from onnx_ir import tensor_adapters

from mobius._builder import build_from_module
from mobius._configs import QuantizationConfig, QuantizedWeightFormat
from mobius._testing import make_config
from mobius.integrations.transformers import _gptoss_weights
from mobius.models.gptoss import GPTOSSCausalLMModel, repack_gptoss_mxfp4_blocks

_E = 2
_H = 64
_I = 32
_ROOT = "model.layers.0.mlp"


def _config(**overrides):
    options = dict(
        model_type="gpt_oss",
        dtype=ir.DataType.FLOAT16,
        num_hidden_layers=1,
        hidden_size=_H,
        intermediate_size=_I,
        num_local_experts=_E,
        num_experts_per_tok=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        vocab_size=32,
        layer_types=["sliding_attention"],
        sliding_window=32,
        partial_rotary_factor=1.0,
        rope_interleave=False,
        attn_qkv_bias=True,
        attn_o_bias=True,
        quantization=QuantizationConfig(
            bits=4,
            group_size=32,
            quant_method="mxfp4",
            weight_format=QuantizedWeightFormat.MXFP4,
        ),
    )
    options.update(overrides)
    return make_config(**options)


def _package_and_state():
    config = _config()
    module = GPTOSSCausalLMModel(config)
    package = build_from_module(module, config, execution_provider="cuda")
    model = package["model"]
    state: dict[str, torch.Tensor] = {}
    special_targets = {
        f"{_ROOT}.fc1_experts_weights",
        f"{_ROOT}.fc1_scales",
        f"{_ROOT}.fc1_experts_bias",
        f"{_ROOT}.fc1_global_scales",
        f"{_ROOT}.fc2_experts_weights",
        f"{_ROOT}.fc2_scales",
        f"{_ROOT}.fc2_experts_bias",
        f"{_ROOT}.fc2_global_scales",
        f"{_ROOT}.gate.weight",
        f"{_ROOT}.gate.bias",
    }
    for name, initializer in model.graph.initializers.items():
        if initializer.const_value is not None or name in special_targets:
            continue
        dtype = tensor_adapters.to_torch_dtype(initializer.dtype)
        shape = tuple(int(dim) for dim in initializer.shape)
        state[name] = torch.zeros(shape, dtype=dtype)

    state.update(
        {
            f"{_ROOT}.experts.gate_up_proj_blocks": torch.randint(
                0, 256, (_E, 2 * _I, _H // 32, 16), dtype=torch.uint8
            ),
            f"{_ROOT}.experts.gate_up_proj_scales": torch.randint(
                0, 255, (_E, 2 * _I, _H // 32), dtype=torch.uint8
            ),
            f"{_ROOT}.experts.down_proj_blocks": torch.randint(
                0, 256, (_E, _H, _I // 32, 16), dtype=torch.uint8
            ),
            f"{_ROOT}.experts.down_proj_scales": torch.randint(
                0, 255, (_E, _H, _I // 32), dtype=torch.uint8
            ),
            f"{_ROOT}.experts.gate_up_proj_bias": torch.randn(_E, 2 * _I, dtype=torch.float16),
            f"{_ROOT}.experts.down_proj_bias": torch.randn(_E, _H, dtype=torch.float16),
            f"{_ROOT}.router.weight": torch.randn(_E, _H, dtype=torch.float16),
            f"{_ROOT}.router.bias": torch.randn(_E, dtype=torch.float16),
        }
    )
    return config, package, state


def _save_cross_sharded(state, directory):
    names = sorted(state)
    shard_a_names = [
        name for index, name in enumerate(names) if name.endswith("_blocks") or index % 2 == 0
    ]
    shard_b_names = [
        name
        for index, name in enumerate(names)
        if name.endswith("_scales") or (index % 2 == 1 and not name.endswith("_blocks"))
    ]
    # Explicitly put every blocks/scales pair in different files.
    shard_a_names = [name for name in shard_a_names if not name.endswith("_scales")]
    shard_b_names = [name for name in shard_b_names if not name.endswith("_blocks")]
    shards = {
        "model-00001-of-00002.safetensors": {name: state[name] for name in shard_a_names},
        "model-00002-of-00002.safetensors": {name: state[name] for name in shard_b_names},
    }
    weight_map = {}
    for filename, tensors in shards.items():
        safetensors.torch.save_file(tensors, str(directory / filename))
        weight_map.update(dict.fromkeys(tensors, filename))
    (directory / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map})
    )


def test_cross_shard_pairs_transform_and_bind_final_initializers(tmp_path, monkeypatch):
    config, package, state = _package_and_state()
    _save_cross_sharded(state, tmp_path)
    source_refs: list[weakref.ReferenceType[torch.Tensor]] = []

    def tracked_repack(tensor, _source_name):
        source_refs.append(weakref.ref(tensor))
        return repack_gptoss_mxfp4_blocks(tensor)

    monkeypatch.setattr(_gptoss_weights, "_repack_blocks", tracked_repack)

    report = _gptoss_weights.stream_gptoss_mxfp4_safetensors_to_package(
        package, str(tmp_path), config
    )

    model = package["model"]
    fc1 = model.graph.initializers[f"{_ROOT}.fc1_experts_weights"]
    fc2 = model.graph.initializers[f"{_ROOT}.fc2_experts_weights"]
    assert isinstance(fc1.const_value, ir.LazyTensor)
    assert isinstance(fc2.const_value, ir.LazyTensor)
    assert report["output_weight_format"] == "mxfp4"
    assert report["streaming_unit"] == "one_moe_projection"

    actual_fc1 = torch.from_numpy(fc1.const_value.numpy().copy())
    torch.testing.assert_close(
        actual_fc1,
        repack_gptoss_mxfp4_blocks(state[f"{_ROOT}.experts.gate_up_proj_blocks"]),
    )
    del actual_fc1
    gc.collect()
    assert source_refs[0]() is None

    actual_fc2 = torch.from_numpy(fc2.const_value.numpy().copy())
    torch.testing.assert_close(
        actual_fc2,
        repack_gptoss_mxfp4_blocks(state[f"{_ROOT}.experts.down_proj_blocks"]),
    )
    del actual_fc2
    gc.collect()
    assert all(reference() is None for reference in source_refs)

    scales = model.graph.initializers[f"{_ROOT}.fc1_scales"].const_value
    assert scales is not None
    actual_scale_bytes = torch.from_numpy(scales.numpy().view("uint8").copy())
    torch.testing.assert_close(
        actual_scale_bytes,
        state[f"{_ROOT}.experts.gate_up_proj_scales"],
    )
    global_scales = model.graph.initializers[f"{_ROOT}.fc1_global_scales"].const_value
    assert global_scales is not None
    assert global_scales.dtype == ir.DataType.FLOAT


@pytest.mark.parametrize("malformation", ["missing_pair", "invalid_scale"])
def test_incomplete_or_invalid_native_set_fails_before_assignment(tmp_path, malformation):
    config, package, state = _package_and_state()
    if malformation == "missing_pair":
        del state[f"{_ROOT}.experts.down_proj_scales"]
    else:
        state[f"{_ROOT}.experts.down_proj_scales"].reshape(-1)[-1] = 0xFF
    _save_cross_sharded(state, tmp_path)
    target = package["model"].graph.initializers[f"{_ROOT}.fc1_experts_weights"]

    with pytest.raises(ValueError, match=r"Malformed|0xff"):
        _gptoss_weights.stream_gptoss_mxfp4_safetensors_to_package(
            package, str(tmp_path), config
        )

    assert target.const_value is None


def test_native_legacy_checkpoint_fails_closed_without_eager_fallback(tmp_path):
    config, package, _state = _package_and_state()
    torch.save({"not": torch.ones(1)}, tmp_path / "pytorch_model.bin")

    with pytest.raises(ValueError, match="requires a safetensors checkpoint"):
        _gptoss_weights.stream_gptoss_mxfp4_safetensors_to_package(
            package, str(tmp_path), config
        )


@pytest.mark.parametrize(
    ("layers", "experts", "profile"),
    [(24, 32, "gpt-oss-20b"), (36, 128, "gpt-oss-120b")],
)
def test_official_profile_geometry_is_header_only(layers, experts, profile):
    config = _config(
        num_hidden_layers=layers,
        num_local_experts=experts,
        hidden_size=2880,
        intermediate_size=2880,
    )

    specs = _gptoss_weights._native_mxfp4_projection_specs(config)

    assert len(specs) == layers, profile
    assert specs["model.layers.0.mlp"]["gate_up_proj"][0] == (
        experts,
        5760,
        90,
        16,
    )
    assert specs[f"model.layers.{layers - 1}.mlp"]["down_proj"][0] == (
        experts,
        2880,
        90,
        16,
    )
