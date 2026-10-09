# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Offline draft INT3 QMoE export tests; no INT3 runtime execution is claimed."""

from __future__ import annotations

import dataclasses

import onnx_ir as ir
import pytest
import torch
from onnx_ir import tensor_adapters

from mobius._builder import build_from_module
from mobius._configs import QuantizationConfig, QuantizationOverride
from mobius._flags import override_flags
from mobius._model_package import ModelPackage
from mobius._testing import make_config
from mobius.models.moe import MoECausalLMModel

_ROOT = "model.layers.0.mlp"
_E, _H, _I, _BLOCK = 2, 64, 32, 32


def _pack_oracle(codes: torch.Tensor, bits: int) -> torch.Tensor:
    """Use independent arbitrary-width integers, not the exporter/Olive codec."""
    rows = codes.reshape(-1, codes.shape[-1]).tolist()
    row_bytes = (codes.shape[-1] * bits + 7) // 8
    raw = [
        list(
            sum(int(code) << (bits * k) for k, code in enumerate(row)).to_bytes(
                row_bytes, "little"
            )
        )
        for row in rows
    ]
    return torch.tensor(raw, dtype=torch.uint8).reshape(*codes.shape[:-1], row_bytes)


def _decode_oracle(packed: torch.Tensor, bits: int, k: int) -> torch.Tensor:
    """Extract logical codes independently from serialized bytes."""
    rows = packed.reshape(-1, packed.shape[-1]).tolist()
    codes = [
        [
            (int.from_bytes(bytes(row), "little") >> (bits * i)) & ((1 << bits) - 1)
            for i in range(k)
        ]
        for row in rows
    ]
    return torch.tensor(codes, dtype=torch.int32).reshape(*packed.shape[:-1], k)


def _config(fc1: int, fc2: int, dtype: ir.DataType = ir.DataType.FLOAT16):
    quantization = QuantizationConfig(
        bits=4,
        group_size=_BLOCK,
        quant_method="olive",
        overrides={
            f"{_ROOT}.experts.{projection}": QuantizationOverride(bits=bits)
            for projection, bits in (("gate_up_proj", fc1), ("down_proj", fc2))
        },
    )
    return make_config(
        dtype=dtype,
        num_hidden_layers=1,
        hidden_size=_H,
        intermediate_size=64,
        moe_intermediate_size=_I,
        num_local_experts=_E,
        num_experts_per_tok=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        attn_qk_norm=True,
        vocab_size=32,
        quantization=quantization,
    )


def _state(fc1: int, fc2: int, dtype: torch.dtype = torch.float16):
    state = {}
    for projection, bits, rows, k in (
        ("gate_up_proj", fc1, 2 * _I, _H),
        ("down_proj", fc2, _H, _I),
    ):
        stem = f"{_ROOT}.experts.{projection}"
        codes = (
            torch.arange(_E * rows * k).reshape(_E, rows, k)
            + torch.arange(rows).reshape(1, rows, 1)
        ) % (1 << bits)
        state[stem + "_qweight"] = _pack_oracle(codes, bits)
        state[stem + "_scales"] = (
            torch.arange(_E * rows * (k // _BLOCK)).reshape(_E, rows, k // _BLOCK) % 7
        ).to(dtype) / 8
    return state


@pytest.mark.parametrize("fc1,fc2", [(3, 4), (3, 3), (4, 3)])
@pytest.mark.parametrize("dtype", [ir.DataType.FLOAT16, ir.DataType.BFLOAT16])
def test_int3_fused_export_preserves_bytes_order_scales_and_external_data(
    tmp_path, fc1, fc2, dtype
):
    torch_dtype = tensor_adapters.to_torch_dtype(dtype)
    config = _config(fc1, fc2, dtype)
    raw = _state(fc1, fc2, torch_dtype)
    with override_flags(experimental_int3_qmoe_export=True):
        module = MoECausalLMModel(config)
        processed = module.preprocess_weights(dict(raw))
        package = build_from_module(module, config)
        for name, parameter in module.named_parameters():
            if name not in processed:
                processed[name] = torch.zeros(
                    tuple(parameter.shape),
                    dtype=tensor_adapters.to_torch_dtype(parameter.dtype),
                )
        package.apply_weights(processed)
        package.save(str(tmp_path), progress_bar=False)

    graph = ModelPackage.load(str(tmp_path))["model"].graph
    qmoe = next(node for node in graph if node.op_type == "QMoE")
    for name, value in (
        ("quant_type", "int"),
        ("weights_prepacked", 0),
        ("block_size", _BLOCK),
        ("fc1_expert_weight_bits", fc1),
        ("fc2_expert_weight_bits", fc2),
        ("fc3_expert_weight_bits", fc1),
        ("swiglu_fusion", 1),
    ):
        assert qmoe.attributes[name].value == value
    assert "draft" in qmoe.metadata_props["mobius.int3_qmoe.format"]
    assert "unqualified" in qmoe.metadata_props["mobius.int3_qmoe.runtime_support"]
    assert qmoe.inputs[11] is None
    assert qmoe.inputs[12] is None
    for projection, bits, slot, scale_slot, rows, k in (
        ("gate_up_proj", fc1, 2, 3, 2 * _I, _H),
        ("down_proj", fc2, 5, 6, _H, _I),
    ):
        stem = f"{_ROOT}.experts.{projection}"
        weight = qmoe.inputs[slot]
        scale = qmoe.inputs[scale_slot]
        assert weight is not None and weight.const_value is not None
        assert scale is not None and scale.const_value is not None
        actual = torch.from_numpy(weight.const_value.numpy().copy())
        actual_scales = torch.from_numpy(scale.const_value.numpy().astype("float32")).to(
            torch_dtype
        )
        expected = raw[stem + "_qweight"]
        expected_scales = raw[stem + "_scales"]
        if projection == "gate_up_proj":
            expected = expected.reshape(_E, 2, _I, -1).transpose(1, 2).reshape(_E, 2 * _I, -1)
            expected_scales = (
                expected_scales.reshape(_E, 2, _I, -1).transpose(1, 2).reshape(_E, 2 * _I, -1)
            )
        assert tuple(actual.shape) == (_E, rows, (k * bits + 7) // 8)
        assert torch.equal(actual, expected)
        assert torch.equal(actual_scales, expected_scales)
        codes = _decode_oracle(actual, bits, k)
        expected_codes = _decode_oracle(expected, bits, k)
        decoded = (codes - (1 << (bits - 1))) * actual_scales.repeat_interleave(_BLOCK, dim=-1)
        expected_decoded = (
            expected_codes - (1 << (bits - 1))
        ) * expected_scales.repeat_interleave(_BLOCK, dim=-1)
        assert torch.equal(decoded, expected_decoded)
    assert list(tmp_path.glob("*.data")), "packed weights must be externally serialized"


def test_int3_export_requires_opt_in():
    with (
        override_flags(experimental_int3_qmoe_export=False),
        pytest.raises(NotImplementedError, match="experimental_int3"),
    ):
        MoECausalLMModel(_config(3, 4))


@pytest.mark.parametrize("fc1,fc2", [(3, 2), (2, 3), (3, 8), (8, 3)])
def test_int3_export_rejects_unqualified_width_combinations(fc1, fc2):
    with (
        override_flags(experimental_int3_qmoe_export=True),
        pytest.raises(ValueError, match="expert widths"),
    ):
        MoECausalLMModel(_config(fc1, fc2))


@pytest.mark.parametrize("block", [16, 48, 256])
def test_int3_export_rejects_unqualified_block_sizes(block):
    config = _config(3, 4)
    config.quantization = dataclasses.replace(config.quantization, group_size=block)
    with (
        override_flags(experimental_int3_qmoe_export=True),
        pytest.raises(ValueError, match="block_size"),
    ):
        MoECausalLMModel(config)


def test_int3_export_rejects_unapproved_logical_tails():
    config = dataclasses.replace(_config(3, 4), moe_intermediate_size=33)
    with (
        override_flags(experimental_int3_qmoe_export=True),
        pytest.raises(ValueError, match="divisible"),
    ):
        MoECausalLMModel(config)


def test_int3_export_rejects_fp32_activations():
    with (
        override_flags(experimental_int3_qmoe_export=True),
        pytest.raises(ValueError, match="activations"),
    ):
        MoECausalLMModel(_config(3, 4, ir.DataType.FLOAT))


@pytest.mark.parametrize("fc1,fc2", [(3, 4), (3, 3), (4, 3)])
def test_int3_export_rejects_asymmetric_int3_config(fc1, fc2):
    config = _config(fc1, fc2)
    config.quantization.sym = False
    with (
        override_flags(experimental_int3_qmoe_export=True),
        pytest.raises(ValueError, match="symmetric"),
    ):
        MoECausalLMModel(config)


@pytest.mark.parametrize("mutation", ["dtype", "negative", "nan", "inf", "truncated"])
@pytest.mark.parametrize("projection", ["gate_up_proj", "down_proj"])
def test_int3_export_rejects_invalid_projection_payload(mutation, projection):
    raw = _state(3, 3)
    stem = f"{_ROOT}.experts.{projection}"
    if mutation == "dtype":
        raw[stem + "_scales"] = raw[stem + "_scales"].float()
    elif mutation == "truncated":
        raw[stem + "_qweight"] = raw[stem + "_qweight"][..., :-1]
    else:
        value = {"negative": -1, "nan": float("nan"), "inf": float("inf")}[mutation]
        raw[stem + "_scales"][0, 0, 0] = value
    with override_flags(experimental_int3_qmoe_export=True):
        model = MoECausalLMModel(_config(3, 3))
        with pytest.raises(ValueError, match=r"scales|shape"):
            model.preprocess_weights(raw)


@pytest.mark.parametrize(
    "projection,k,rows",
    [
        ("gate_up_proj", _H, 2 * _I),
        ("down_proj", _I, _H),
    ],
)
@pytest.mark.parametrize("invalid", [None, "non4", "padding"])
def test_int3_explicit_zero_points_validate_and_export_as_implicit(
    projection, k, rows, invalid
):
    raw = _state(3, 3)
    zeros = _pack_oracle(torch.full((_E, rows, k // _BLOCK), 4), 3)
    if invalid == "non4":
        zeros[0, 0, 0] ^= 1
    elif invalid == "padding":
        zeros[0, 0, -1] |= 128
    raw[f"{_ROOT}.experts.{projection}_qzeros"] = zeros
    with override_flags(experimental_int3_qmoe_export=True):
        module = MoECausalLMModel(_config(3, 3))
        if invalid:
            with pytest.raises(ValueError, match=r"zero.point"):
                module.preprocess_weights(raw)
        else:
            processed = module.preprocess_weights(raw)
            assert not any("zero_points" in key for key in processed)


@pytest.mark.parametrize(
    "codes,expected",
    [
        ([0, 1, 2, 3, 4, 5, 6, 7], [0x88, 0xC6, 0xFA]),
        ([0, 4, 7], [0xE0, 0x01]),
        ([4] * 8, [0x24, 0x49, 0x92]),
    ],
)
def test_int3_checkpoint_matches_independent_proposal_goldens(codes, expected):
    logical = torch.tensor([codes], dtype=torch.int32)
    reference = torch.tensor([expected], dtype=torch.uint8)
    assert torch.equal(_pack_oracle(logical, 3), reference)
    assert torch.equal(_decode_oracle(reference, 3, len(codes)), logical)


@pytest.mark.parametrize("k", [1, 3, 8, 31, 32, 33, 64, 128])
def test_olive_int3_checkpoint_bytes_match_independent_oracle(k):
    olive_quant = pytest.importorskip("olive.common.quant.utils")
    logical = torch.arange(2 * 5 * k).reshape(2, 5, k) % 8
    assert torch.equal(olive_quant.pack_to_uint8(logical, 3), _pack_oracle(logical, 3))


@pytest.mark.parametrize("fc1,fc2", [(3, 4), (3, 3), (4, 3)])
@pytest.mark.parametrize("dtype", [ir.DataType.FLOAT16, ir.DataType.BFLOAT16])
def test_public_build_loads_local_olive_int3_checkpoint(tmp_path, fc1, fc2, dtype):
    from safetensors.torch import save_file
    from transformers import Qwen3MoeConfig

    from mobius import build

    config = _config(fc1, fc2, dtype)
    hf_config = Qwen3MoeConfig(
        vocab_size=32,
        hidden_size=_H,
        intermediate_size=64,
        moe_intermediate_size=_I,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        num_experts=_E,
        num_experts_per_tok=2,
        max_position_embeddings=config.max_position_embeddings,
        tie_word_embeddings=False,
        quantization_config={
            "quant_method": "olive",
            "bits": 4,
            "group_size": _BLOCK,
            "sym": True,
            "overrides": {
                name: dataclasses.asdict(override)
                for name, override in config.quantization.overrides.items()
            },
        },
    )
    hf_config.save_pretrained(tmp_path)
    raw = _state(fc1, fc2, tensor_adapters.to_torch_dtype(dtype))
    with override_flags(experimental_int3_qmoe_export=True):
        module = MoECausalLMModel(config)
        for name, parameter in module.named_parameters():
            if ".fc1_" in name or ".fc2_" in name or name.startswith("model.rotary_emb."):
                continue
            tensor = torch.zeros(
                tuple(parameter.shape),
                dtype=tensor_adapters.to_torch_dtype(parameter.dtype),
            )
            if tensor.dtype == torch.uint8 and name.endswith(".weight"):
                raw[name + "_qweight"] = tensor.flatten(-2)
            elif name.endswith(".scales"):
                raw[name.removesuffix(".scales") + ".weight_scales"] = tensor
            else:
                raw[name] = tensor
        save_file(raw, str(tmp_path / "model.safetensors"), metadata={"format": "pt"})
        package = build(str(tmp_path), dtype="f16" if dtype == ir.DataType.FLOAT16 else "bf16")
        output = tmp_path / "export"
        package.save(str(output), progress_bar=False)
    graph = ModelPackage.load(str(output))["model"].graph
    qmoe = next(node for node in graph if node.op_type == "QMoE")
    assert qmoe.attributes["fc1_expert_weight_bits"].value == fc1
    assert qmoe.attributes["fc2_expert_weight_bits"].value == fc2
    assert qmoe.inputs[2].const_value is not None
    assert qmoe.inputs[5].const_value is not None
    expected = raw[f"{_ROOT}.experts.gate_up_proj_qweight"]
    expected = expected.reshape(_E, 2, _I, -1).transpose(1, 2).reshape(_E, 2 * _I, -1)
    assert qmoe.inputs[2].const_value.numpy().tobytes() == expected.numpy().tobytes()
    assert (
        qmoe.inputs[5].const_value.numpy().tobytes()
        == raw[f"{_ROOT}.experts.down_proj_qweight"].numpy().tobytes()
    )


@pytest.mark.parametrize("block", [32, 64, 128])
def test_int3_export_accepts_profile_block_sizes(block):
    config = dataclasses.replace(_config(3, 3), hidden_size=128, moe_intermediate_size=128)
    config.quantization = dataclasses.replace(config.quantization, group_size=block)
    with override_flags(experimental_int3_qmoe_export=True):
        module = MoECausalLMModel(config)
    layer = module.model.layers[0].mlp
    assert tuple(layer.fc1_experts_weights.shape) == (_E, 256, 48)
    assert tuple(layer.fc2_experts_weights.shape) == (_E, 128, 48)
    assert tuple(layer.fc1_scales.shape) == (_E, 256, 128 // block)
    assert tuple(layer.fc2_scales.shape) == (_E, 128, 128 // block)


@pytest.mark.parametrize("fc1,fc2", [(3, 4), (4, 3)])
def test_int3_mixed_export_preserves_asymmetric_int4_zero_points(fc1, fc2):
    config = _config(fc1, fc2)
    projection, label, rows, k = (
        ("gate_up_proj", "fc1", 2 * _I, _H) if fc1 == 4 else ("down_proj", "fc2", _H, _I)
    )
    name = f"{_ROOT}.experts.{projection}"
    config.quantization.overrides[name] = QuantizationOverride(bits=4, sym=False)
    raw = _state(fc1, fc2)
    zeros = _pack_oracle(
        torch.arange(_E * rows * (k // _BLOCK)).reshape(_E, rows, k // _BLOCK) % 16, 4
    )
    raw[name + "_qzeros"] = zeros
    with override_flags(experimental_int3_qmoe_export=True):
        module = MoECausalLMModel(config)
        processed = module.preprocess_weights(raw)
    expected = zeros
    if fc1 == 4:
        expected = zeros.reshape(_E, 2, _I, -1).transpose(1, 2).reshape(*zeros.shape)
    assert torch.equal(processed[f"{_ROOT}.{label}_experts_zero_points"], expected)
    assert getattr(module.model.layers[0].mlp, label + "_experts_zero_points") is not None
