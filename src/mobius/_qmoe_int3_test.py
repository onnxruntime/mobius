from __future__ import annotations

import dataclasses

import pytest
import torch

from mobius import _weight_utils
from mobius._configs import QuantizationConfig, QuantizationOverride


@pytest.mark.parametrize("bits", [(3, 3), (3, 4), (4, 3)])
@pytest.mark.parametrize("block", [32, 64, 128])
def test_int3_expert_layout_preserves_int4_fallback(monkeypatch, bits, block):
    fallback = QuantizationConfig(bits=4, group_size=block, sym=True, quant_method="olive")
    first = dataclasses.replace(fallback, bits=bits[0])
    second = dataclasses.replace(fallback, bits=bits[1])
    layouts = iter((first, second))
    monkeypatch.setattr(_weight_utils, "_resolve_qmoe_module", lambda *_: next(layouts))
    layout = _weight_utils.resolve_qmoe_quantization(fallback, ("model.layers.0.mlp",))
    assert layout is not None
    assert layout.fallback is fallback
    assert layout.fallback.bits == 4
    assert (layout.fc1.bits, layout.fc2.bits) == bits
    assert layout.requires_projection_preprocessing


@pytest.mark.parametrize(
    "change",
    [
        {"sym": False},
        {"group_size": 16},
        {"group_size": 256},
        {"quant_method": "gptq"},
        {"bits": 8},
    ],
)
def test_int3_rejects_unimplemented_expert_layout(monkeypatch, change):
    fallback = QuantizationConfig(bits=4, group_size=64, sym=True, quant_method="olive")
    first = dataclasses.replace(fallback, bits=3)
    second = dataclasses.replace(fallback, **change)
    layouts = iter((first, second))
    monkeypatch.setattr(_weight_utils, "_resolve_qmoe_module", lambda *_: next(layouts))
    with pytest.raises(ValueError, match="INT3 native QMoE requires"):
        _weight_utils.resolve_qmoe_quantization(fallback, ("model.layers.0.mlp",))


@pytest.mark.parametrize("bits", [(3, 3), (3, 4), (4, 3)])
def test_real_overrides_bind_raw_int3_sidecars(bits):
    root = "model.layers.0.mlp"
    block, hidden, intermediate, experts = 32, 64, 96, 2
    config = QuantizationConfig(
        bits=4,
        group_size=block,
        sym=True,
        quant_method="olive",
        overrides={
            f"{root}.experts.gate_up_proj": QuantizationOverride(bits=bits[0]),
            f"{root}.experts.down_proj": QuantizationOverride(bits=bits[1]),
        },
    )
    state = {}
    expected = {}
    for projection, outputs, features, width, label in (
        ("gate_up_proj", intermediate * 2, hidden, bits[0], "fc1"),
        ("down_proj", hidden, intermediate, bits[1], "fc2"),
    ):
        shape = (experts, outputs, features * width // 8)
        packed = torch.arange(shape[0] * shape[1] * shape[2]).to(torch.uint8).reshape(shape)
        scales = torch.full(
            (experts, outputs, features // block), 0.015625, dtype=torch.float16
        )
        state[f"{root}.experts.{projection}_qweight"] = packed
        state[f"{root}.experts.{projection}_scales"] = scales
        expected[f"{root}.{label}_experts_weights"] = packed
        expected[f"{root}.{label}_scales"] = scales
    actual = _weight_utils.preprocess_quantized_weights(
        state,
        config,
        qmoe_target_path=".mlp",
        qmoe_num_experts=experts,
        qmoe_hidden_size=hidden,
        qmoe_intermediate_size=intermediate,
        qmoe_expected_moe_paths=(root,),
        qmoe_source_moe_paths=(root,),
    )
    for name, tensor in expected.items():
        assert name in actual
        torch.testing.assert_close(actual[name], tensor, rtol=0, atol=0)


def test_int3_qmoe_graph_keeps_attention_int4():
    import onnx_ir as ir

    from mobius import build_from_module
    from mobius._testing import make_config
    from mobius.models.moe import MoECausalLMModel

    root = "model.layers.0.mlp"
    config = make_config(
        dtype=ir.DataType.FLOAT16,
        hidden_size=64,
        intermediate_size=96,
        moe_intermediate_size=96,
        num_hidden_layers=1,
        num_local_experts=3,
        num_experts_per_tok=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        vocab_size=32,
        quantization=QuantizationConfig(
            bits=4,
            group_size=32,
            sym=True,
            quant_method="olive",
            overrides={
                f"{root}.experts.{name}": QuantizationOverride(bits=3)
                for name in ("gate_up_proj", "down_proj")
            },
        ),
    )
    module = MoECausalLMModel(config)
    package = build_from_module(module, config)
    nodes = list(package["model"].graph)
    qmoe = [node for node in nodes if node.op_type == "QMoE"]
    assert len(qmoe) == 1
    root_bits = qmoe[0].attributes["expert_weight_bits"].as_int()
    for name in ("fc1_expert_weight_bits", "fc2_expert_weight_bits"):
        attribute = qmoe[0].attributes.get(name)
        assert (attribute.as_int() if attribute is not None else root_bits) == 3
    assert qmoe[0].attributes["weights_prepacked"].as_int() == 0
    linears = [node for node in nodes if node.op_type == "MatMulNBits"]
    assert linears
    assert all(node.attributes["bits"].as_int() == 4 for node in linears)
