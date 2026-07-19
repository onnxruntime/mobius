# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Graph-build tests for LoRA-adapted UNet attention.

These build the attention graph with a ``LoRALinear`` factory (no weights) and
verify the low-rank adapter branch — and its runtime gate — are wired in. This
is the foundation for runtime LoRA in the from-scratch diffusion UNet: adapters
are baked as ``lora_A``/``lora_B`` params and switched/blended at run time via a
per-adapter scalar gate input.
"""

from __future__ import annotations

import onnx_ir as ir

from mobius._testing import count_op_type, create_test_builder, create_test_input
from mobius.components import Linear, LoRALinear
from mobius.models.unet import _BasicAttention


def _lora_factory(gate_holder=None):
    def factory(in_features: int, out_features: int, bias: bool = True) -> LoRALinear:
        return LoRALinear(
            in_features,
            out_features,
            bias=bias,
            lora_adapters=[("test", 4, 1.0)],
            gate_holder=gate_holder,
        )

    return factory


def _referenced_names(graph: ir.Graph) -> set[str]:
    names: set[str] = set()
    for node in graph:
        for value in node.inputs:
            if value is not None and value.name:
                names.add(value.name)
    return names


def _build_attention(linear_class, gate_holder=None):
    builder, op, graph = create_test_builder()
    hidden = create_test_input(builder, "hidden", [1, 4, 8])
    context = create_test_input(builder, "context", [1, 4, 8])
    attention = _BasicAttention(8, 8, 2, linear_class=linear_class)
    output = attention(op, hidden, context)
    builder._adapt_outputs([output], "")
    return graph


def test_plain_attention_has_no_lora_branch():
    graph = _build_attention(Linear)
    assert count_op_type(graph, "Attention") >= 1
    assert not any("lora_" in name for name in _referenced_names(graph))


def test_lora_attention_wires_low_rank_branch():
    plain_graph = _build_attention(Linear)
    lora_graph = _build_attention(_lora_factory())

    # Still a valid attention graph.
    assert count_op_type(lora_graph, "Attention") >= 1
    # The low-rank branch adds two MatMuls per projection (q/k/v/out => +8).
    assert count_op_type(lora_graph, "MatMul") > count_op_type(plain_graph, "MatMul")
    # The adapter weights are present under the HuggingFace naming.
    names = _referenced_names(lora_graph)
    assert any("lora_A.test.weight" in name for name in names)
    assert any("lora_B.test.weight" in name for name in names)


def test_lora_gate_holder_applies_runtime_gate():
    # A shared gate_holder maps the adapter to a runtime scalar; the graph must
    # multiply the adapter contribution by that gate value (runtime on/off/blend).
    builder, op, graph = create_test_builder()
    gate = create_test_input(builder, "lora_gate.test", [])
    gate_holder = {"test": gate}
    hidden = create_test_input(builder, "hidden", [1, 4, 8])
    context = create_test_input(builder, "context", [1, 4, 8])
    attention = _BasicAttention(8, 8, 2, linear_class=_lora_factory(gate_holder))
    output = attention(op, hidden, context)
    builder._adapt_outputs([output], "")

    assert "lora_gate.test" in _referenced_names(graph)


def test_full_unet_declares_lora_gate_inputs():
    # Build a small full UNet with a baked adapter through the denoising task and
    # assert the runtime gate input + adapter params are present end-to-end.
    from mobius._diffusers_configs import UNet2DConfig
    from mobius.models.unet import UNet2DConditionModel
    from mobius.tasks._denoising import DenoisingTask

    config = UNet2DConfig(
        in_channels=4,
        out_channels=4,
        block_out_channels=(32, 64),
        layers_per_block=1,
        norm_num_groups=32,
        cross_attention_dim=16,
        attention_head_dim=8,
        lora_adapters=(("style", 4, 1.0),),
    )
    module = UNet2DConditionModel(config)
    package = DenoisingTask().build(module, config)
    graph = package["model"].graph

    input_names = {value.name for value in graph.inputs}
    assert "lora_gate.style" in input_names
    assert "sample" in input_names and "encoder_hidden_states" in input_names
    names = _referenced_names(graph)
    assert any("lora_A.style.weight" in name for name in names)
    assert any("lora_B.style.weight" in name for name in names)


def test_full_unet_without_lora_has_no_gate_inputs():
    from mobius._diffusers_configs import UNet2DConfig
    from mobius.models.unet import UNet2DConditionModel
    from mobius.tasks._denoising import DenoisingTask

    config = UNet2DConfig(
        block_out_channels=(32, 64),
        layers_per_block=1,
        cross_attention_dim=16,
    )
    module = UNet2DConditionModel(config)
    graph = DenoisingTask().build(module, config)["model"].graph
    assert not any(
        value.name and "lora_gate" in value.name for value in graph.inputs
    )


def test_remap_diffusers_lora_keys():
    from mobius.models.unet import remap_diffusers_unet_lora

    # Classic diffusers `lora.down`/`lora.up` spelling.
    src = {
        "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.lora.down.weight": 1,
        "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.lora.up.weight": 2,
        "mid_block.attentions.0.transformer_blocks.0.attn2.to_out.0.lora.down.weight": 3,
    }
    out = remap_diffusers_unet_lora(src, "style")
    assert out == {
        "down_blocks.0.attentions.0.attn1.to_q.lora_A.style.weight": 1,
        "down_blocks.0.attentions.0.attn1.to_q.lora_B.style.weight": 2,
        "mid_block.attentions.0.attn2.to_out.0.lora_A.style.weight": 3,
    }
    # Newer PEFT `lora_A`/`lora_B` spelling maps identically.
    peft_src = {
        "up_blocks.1.attentions.0.transformer_blocks.0.attn1.to_v.lora_A.weight": 4,
        "up_blocks.1.attentions.0.transformer_blocks.0.attn1.to_v.lora_B.weight": 5,
    }
    peft_out = remap_diffusers_unet_lora(peft_src, "style")
    assert peft_out == {
        "up_blocks.1.attentions.0.attn1.to_v.lora_A.style.weight": 4,
        "up_blocks.1.attentions.0.attn1.to_v.lora_B.style.weight": 5,
    }


def test_remapped_keys_match_baked_unet_param_names():
    # The remapped keys must land on real baked LoRALinear params of the UNet.
    from mobius._diffusers_configs import UNet2DConfig
    from mobius.models.unet import UNet2DConditionModel, remap_diffusers_unet_lora
    from mobius.tasks._denoising import DenoisingTask

    config = UNet2DConfig(
        in_channels=4,
        out_channels=4,
        block_out_channels=(32, 64),
        layers_per_block=1,
        norm_num_groups=32,
        cross_attention_dim=16,
        attention_head_dim=8,
        lora_adapters=(("style", 4, 1.0),),
    )
    graph = DenoisingTask().build(UNet2DConditionModel(config), config)["model"].graph
    baked = {name for name in _referenced_names(graph) if "lora_A.style" in name or "lora_B.style" in name}
    assert baked  # sanity

    # A diffusers key for a projection that exists in this tiny UNet.
    src = {
        "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.lora.down.weight": 0,
    }
    remapped_key = next(iter(remap_diffusers_unet_lora(src, "style")))
    assert remapped_key in baked, (remapped_key, sorted(baked)[:5])


def test_load_unet_lora_safetensors(tmp_path):
    import torch
    from safetensors.torch import save_file

    from mobius.models.unet import load_unet_lora_safetensors

    path = tmp_path / "style.safetensors"
    save_file(
        {
            "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.lora.down.weight": torch.zeros(
                4, 32
            ),
            "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.lora.up.weight": torch.zeros(
                32, 4
            ),
        },
        str(path),
    )
    loaded = load_unet_lora_safetensors(str(path), "style")
    assert set(loaded) == {
        "down_blocks.0.attentions.0.attn1.to_q.lora_A.style.weight",
        "down_blocks.0.attentions.0.attn1.to_q.lora_B.style.weight",
    }
    assert loaded["down_blocks.0.attentions.0.attn1.to_q.lora_A.style.weight"].shape == (4, 32)
