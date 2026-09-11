# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Packed-weight loading regressions for independently quantized components."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import onnx_ir as ir
import pytest
import torch

from mobius import build, build_from_module
from mobius._component_quantization import normalize_component_quantized_weights
from mobius._configs import (
    ArchitectureConfig,
    Gemma4Config,
    QuantizationConfig,
    QuantizationOverride,
    VisionConfig,
    WhisperConfig,
)
from mobius._testing import make_config
from mobius.components import Linear, QuantizedLinear
from mobius.models import CausalLMModel
from mobius.models.gemma4 import Gemma4Model
from mobius.models.moe import Qwen2MoECausalLMModel
from mobius.models.qwen3_next import Qwen3NextCausalLMModel
from mobius.models.qwen35 import (
    Qwen35CausalLMModel,
    Qwen35MoECausalLMModel,
    Qwen35MoEVL3ModelCausalLMModel,
    Qwen35VL3ModelCausalLMModel,
)
from mobius.models.t5 import T5ForConditionalGeneration
from mobius.models.whisper import WhisperForConditionalGeneration
from mobius.tasks import get_task
from mobius.weights import adapt_model_weights


def _quantization(**overrides) -> QuantizationConfig:
    return dataclasses.replace(
        QuantizationConfig(bits=4, group_size=16, quant_method="olive", sym=True),
        **overrides,
    )


def _bytes(*shape: int, offset: int = 0) -> torch.Tensor:
    return (
        (torch.arange(math.prod(shape)) + offset).remainder(251).add(1).to(torch.uint8)
    ).reshape(shape)


def _floats(*shape: int) -> torch.Tensor:
    return torch.linspace(0.125, 1.125, math.prod(shape)).reshape(shape)


def _build(module):
    return build_from_module(module, module.config, task=module.default_task)


def _canonical_checkpoint(package):
    return {
        name: torch.from_numpy(np.ones(tuple(value.shape), dtype=value.dtype.numpy()))
        for model in package.values()
        for name, value in model.graph.initializers.items()
        if value.const_value is None
    }


def _load(module, package, state_dict):
    task = get_task(module.default_task)
    manifest = task.component_manifest(
        module_class=type(module),
        model_type=module.config.model_type,
        hf_config=module.config,
    )
    weights = adapt_model_weights(module, state_dict, config=module.config, manifest=manifest)
    weights = normalize_component_quantized_weights(
        weights,
        module,
        module.config,
        package.keys(),
        manifest=manifest,
        task=task,
    )
    # Keep named parameters observable rather than folding float transposes.
    package.apply_weights(weights, fold_constants=False)
    return weights


def _assert_bound(package, component: str, name: str, expected: torch.Tensor) -> None:
    value = package[component].graph.initializers[name]
    assert tuple(value.shape) == tuple(expected.shape), name
    assert value.const_value is not None, f"{component}: {name} was not bound"
    np.testing.assert_array_equal(value.const_value.numpy(), expected.numpy(), strict=True)


def _assert_layout(package, component: str, name: str, bits: int, group_size: int) -> None:
    node = next(
        node
        for node in package[component].graph
        if node.op_type == "MatMulNBits" and node.inputs[1].name == name
    )
    assert (
        node.attributes["bits"].as_int(),
        node.attributes["block_size"].as_int(),
    ) == (bits, group_size)


def _t5_config(encoder, decoder) -> ArchitectureConfig:
    return make_config(
        num_hidden_layers=1,
        num_decoder_layers=1,
        hidden_act="gelu",
        quantization=decoder,
        component_quantization={"encoder": encoder, "decoder": decoder},
    )


@pytest.mark.parametrize(("method", "bits"), [("gptq", 8), ("awq", 4)])
def test_t5_raw_dotted_scales_survive_canonical_parameter_filter(method, bits):
    encoder = _quantization(quant_method=method, bits=bits, group_size=32)
    decoder = _quantization()
    module = T5ForConditionalGeneration(_t5_config(encoder, decoder))
    package = _build(module)
    encoder_source = "encoder.block.0.layer.0.SelfAttention.q"
    decoder_source = "decoder.block.0.layer.0.SelfAttention.q"
    encoder_target = "encoder.block.0.self_attn.q_proj"
    decoder_target = "decoder.block.0.self_attn.q_proj"

    qweight = (
        torch.arange(64 * 64 * bits // 32, dtype=torch.int32) * 0x010203 + 0x12345678
    ).reshape(64 * bits // 32, 64)
    scales = _floats(2, 64)
    decoder_qweight = _bytes(64, 32, offset=37)
    decoder_scales = _floats(64, 4)
    shared = _floats(100, 64)
    # Independently extract the four little-endian bytes in each packed word.
    expected = (
        torch.stack([(qweight >> shift) & 255 for shift in (0, 8, 16, 24)], dim=-1)
        .transpose(0, 1)
        .reshape(64, 2, 32 * bits // 8)
        .to(torch.uint8)
    )

    result = _load(
        module,
        package,
        {
            f"{encoder_source}.qweight": qweight,
            f"{encoder_source}.scales": scales,
            f"{decoder_source}.weight_qweight": decoder_qweight,
            f"{decoder_source}.weight_scales": decoder_scales,
            "shared.weight": shared,
        },
    )

    _assert_layout(package, "encoder", f"{encoder_target}.weight", bits, 32)
    _assert_layout(package, "decoder", f"{decoder_target}.weight", 4, 16)
    _assert_bound(package, "encoder", f"{encoder_target}.weight", expected)
    _assert_bound(package, "encoder", f"{encoder_target}.scales", scales.T)
    _assert_bound(
        package, "decoder", f"{decoder_target}.weight", decoder_qweight.reshape(64, 4, 8)
    )
    _assert_bound(package, "decoder", f"{decoder_target}.scales", decoder_scales)
    _assert_bound(package, "encoder", "encoder.embed_tokens.weight", shared)
    _assert_bound(package, "decoder", "decoder.embed_tokens.weight", shared)
    assert f"{encoder_target}.qweight" not in result
    assert f"{decoder_target}.weight_qweight" not in result


@pytest.mark.parametrize("canonical", [False, True], ids=["raw-olive", "already-canonical"])
def test_qwen35_vl_embedding_keeps_gather_layout_and_float_vision(canonical):
    decoder = _quantization()
    config = make_config(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        layer_types=["full_attention"],
        image_token_id=99,
        temporal_patch_size=1,
        vision=VisionConfig(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            patch_size=2,
            in_channels=3,
            out_hidden_size=32,
            num_position_embeddings=4,
        ),
        quantization=decoder,
        component_quantization={
            "decoder": decoder,
            "embedding": _quantization(bits=8, quantize_embeddings=True),
        },
    )
    module = Qwen35VL3ModelCausalLMModel(config)
    package = _build(module)
    table = _bytes(100, 32)
    scales = _floats(100, 2)
    projection = _bytes(64, 16, offset=73)
    projection_scales = _floats(64, 2)
    vision = _floats(32, 16)
    embedding_source = "model.language_model.embed_tokens"
    qweight_suffix = ".qweight" if canonical else ".weight_qweight"
    scales_suffix = ".scales" if canonical else ".weight_scales"
    decoder_source = "model.language_model.layers.0.self_attn.q_proj"

    result = _load(
        module,
        package,
        {
            f"{embedding_source}{qweight_suffix}": table,
            f"{embedding_source}{scales_suffix}": scales,
            f"{decoder_source}.weight_qweight": projection,
            f"{decoder_source}.weight_scales": projection_scales,
            "model.visual.blocks.0.mlp.linear_fc1.weight": vision,
        },
    )

    assert "embedding.embed_tokens.weight" not in result
    assert result["embedding.embed_tokens.qweight"] is table
    assert result["embedding.embed_tokens.scales"] is scales
    gather = next(
        node for node in package["embedding"].graph if node.op_type == "GatherBlockQuantized"
    )
    assert gather.inputs[0].name == "embedding.embed_tokens.qweight"
    assert (gather.attributes["bits"].as_int(), gather.attributes["block_size"].as_int()) == (
        8,
        16,
    )
    _assert_bound(package, "embedding", "embedding.embed_tokens.qweight", table)
    _assert_bound(package, "embedding", "embedding.embed_tokens.scales", scales)
    target = "decoder.model.layers.0.self_attn.q_proj"
    _assert_layout(package, "decoder", f"{target}.weight", 4, 16)
    _assert_bound(package, "decoder", f"{target}.weight", projection.reshape(64, 2, 8))
    _assert_bound(package, "decoder", f"{target}.scales", projection_scales)
    _assert_bound(
        package, "vision_encoder", "vision_encoder.visual.blocks.0.mlp.up_proj.weight", vision
    )
    assert not any(
        node.op_type in {"MatMulNBits", "GatherBlockQuantized"}
        for node in package["vision_encoder"].graph
    )


@pytest.mark.parametrize(
    ("float_name", "shape"),
    [
        ("model.layers.0.linear_attn.in_proj_qkv.weight", (48, 32)),
        ("model.layers.1.mlp.shared_expert_gate.weight", (1, 32)),
    ],
    ids=["ancestor-linear-attention-exclusion", "shared-expert-gate-exclusion"],
)
def test_qwen35_moe_component_rewrite_preserves_olive_float_modules(float_name, shape):
    quantization = _quantization()
    config = make_config(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        moe_intermediate_size=16,
        shared_expert_intermediate_size=16,
        num_local_experts=4,
        num_experts_per_tok=2,
        layer_types=["linear_attention", "full_attention"],
        linear_num_value_heads=2,
        linear_num_key_heads=1,
        linear_key_head_dim=8,
        linear_value_head_dim=16,
        linear_conv_kernel_dim=4,
        quantization=quantization,
        component_quantization={"decoder": quantization},
    )
    module = Qwen35MoECausalLMModel(config)
    package = _build(module)
    float_weight = _floats(*shape)
    target = "model.layers.1.self_attn.q_proj"
    qweight = _bytes(64, 16)
    scales = _floats(64, 2)

    _load(
        module,
        package,
        {
            float_name: float_weight,
            f"{target}.weight_qweight": qweight,
            f"{target}.weight_scales": scales,
        },
    )

    _assert_bound(package, "model", float_name, float_weight)
    assert package["model"].graph.initializers[float_name].dtype == ir.DataType.FLOAT
    assert f"{float_name.removesuffix('.weight')}.scales" not in (
        package["model"].graph.initializers
    )
    assert type(module.model.layers[0].linear_attn.in_proj_qkv) is Linear
    assert type(module.model.layers[1].mlp.shared_expert_gate) is Linear
    assert isinstance(module.model.layers[1].self_attn.q_proj, QuantizedLinear)
    _assert_layout(package, "model", f"{target}.weight", 4, 16)
    _assert_bound(package, "model", f"{target}.weight", qweight.reshape(64, 2, 8))
    _assert_bound(package, "model", f"{target}.scales", scales)


@pytest.mark.parametrize(
    "component_plan", [None, {}], ids=["global-default", "explicit-float"]
)
def test_llama_global_rules_distinguish_absent_and_empty_component_plan(component_plan):
    quantization = _quantization(
        quantize_lm_head=True,
        modules_to_not_convert=("lm_head",),
    )
    config = make_config(
        model_type="llama",
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        head_dim=8,
        quantization=quantization,
        component_quantization=component_plan,
    )
    module = CausalLMModel(config)
    package = _build(module)
    target = "model.layers.0.self_attn.q_proj"
    head = _floats(100, 32)
    weights = {"lm_head.weight": head}
    if component_plan is None:
        qweight = _bytes(32, 16)
        scales = _floats(32, 2)
        weights[f"{target}.weight_qweight"] = qweight
        weights[f"{target}.weight_scales"] = scales
        expected = qweight.reshape(32, 2, 8)
    else:
        expected = _floats(32, 32)
        weights[f"{target}.weight"] = expected

    _load(module, package, weights)

    _assert_bound(package, "model", "lm_head.weight", head)
    assert type(module.lm_head) is Linear
    assert "lm_head.scales" not in package["model"].graph.initializers
    _assert_bound(package, "model", f"{target}.weight", expected)
    if component_plan is None:
        _assert_layout(package, "model", f"{target}.weight", 4, 16)
        _assert_bound(package, "model", f"{target}.scales", scales)
    else:
        assert not any(node.op_type == "MatMulNBits" for node in package["model"].graph)


def test_public_transformers_build_loads_global_olive_rules_without_component_plan(
    monkeypatch,
):
    from transformers import LlamaConfig

    from mobius.integrations.transformers import _builder as transformers_builder

    hf_config = LlamaConfig(
        vocab_size=100,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
        tie_word_embeddings=False,
        quantization_config={
            "quant_method": "olive",
            "bits": 4,
            "group_size": 16,
            "sym": True,
            "lm_head": True,
            "modules_to_not_convert": ["lm_head"],
        },
    )
    qweight = _bytes(32, 16)
    scales = _floats(32, 2)
    head = _floats(100, 32)
    target = "model.layers.0.self_attn.q_proj"
    state_dict = {
        f"{target}.weight_qweight": qweight,
        f"{target}.weight_scales": scales,
        "lm_head.weight": head,
    }
    monkeypatch.setattr(
        transformers_builder,
        "_load_transformers_config",
        lambda *args, **kwargs: (hf_config, False),
    )
    checkpoint = _canonical_checkpoint(
        build("test/tiny-llama-global-olive", dtype="f32", load_weights=False)
    )
    del checkpoint[f"{target}.weight"]
    del checkpoint[f"{target}.scales"]
    checkpoint.update(state_dict)
    monkeypatch.setattr(
        transformers_builder,
        "_download_weights",
        lambda *args, **kwargs: dict(checkpoint),
    )

    package = build("test/tiny-llama-global-olive", dtype="f32")

    _assert_layout(package, "model", f"{target}.weight", 4, 16)
    _assert_bound(package, "model", f"{target}.weight", qweight.reshape(32, 2, 8))
    _assert_bound(package, "model", f"{target}.scales", scales)
    assert "lm_head.scales" not in package["model"].graph.initializers
    # Public loading folds the float head's transpose into an initializer.
    assert any(
        value.const_value is not None
        and tuple(value.shape) == (32, 100)
        and np.array_equal(value.const_value.numpy(), head.numpy().T)
        for value in package["model"].graph.initializers.values()
    )


def test_whisper_declared_output_head_stays_float_and_binds_tied_embedding():
    decoder = _quantization()
    config = WhisperConfig(
        vocab_size=100,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=16,
        hidden_act="gelu",
        pad_token_id=0,
        tie_word_embeddings=True,
        encoder_layers=1,
        encoder_attention_heads=4,
        encoder_ffn_dim=128,
        num_mel_bins=16,
        max_source_positions=16,
        max_target_positions=16,
        quantization=decoder,
        component_quantization={
            "encoder": _quantization(bits=8, group_size=32),
            "decoder": decoder,
        },
    )
    module = WhisperForConditionalGeneration(config)
    package = _build(module)
    table = _floats(100, 64)
    encoder_qweight = _bytes(64, 64)
    decoder_qweight = _bytes(64, 32, offset=53)
    encoder_scales = _floats(64, 2)
    decoder_scales = _floats(64, 4)
    encoder_target = "encoder.layers.0.self_attn.q_proj"
    decoder_target = "decoder.layers.0.self_attn.q_proj"

    result = _load(
        module,
        package,
        {
            "model.decoder.embed_tokens.weight": table,
            f"model.{encoder_target}.weight_qweight": encoder_qweight,
            f"model.{encoder_target}.weight_scales": encoder_scales,
            f"model.{decoder_target}.weight_qweight": decoder_qweight,
            f"model.{decoder_target}.weight_scales": decoder_scales,
        },
    )

    assert result["decoder.proj_out.weight"] is result["decoder.embed_tokens.weight"]
    _assert_bound(package, "decoder", "decoder.proj_out.weight", table)
    _assert_bound(package, "decoder", "decoder.embed_tokens.weight", table)
    assert type(module.model.decoder.proj_out) is Linear
    assert "decoder.proj_out.scales" not in package["decoder"].graph.initializers
    _assert_layout(package, "encoder", f"{encoder_target}.weight", 8, 32)
    _assert_layout(package, "decoder", f"{decoder_target}.weight", 4, 16)
    _assert_bound(
        package, "encoder", f"{encoder_target}.weight", encoder_qweight.reshape(64, 2, 32)
    )
    _assert_bound(package, "encoder", f"{encoder_target}.scales", encoder_scales)
    _assert_bound(
        package, "decoder", f"{decoder_target}.weight", decoder_qweight.reshape(64, 4, 8)
    )
    _assert_bound(package, "decoder", f"{decoder_target}.scales", decoder_scales)


def test_gemma4_vision_projection_override_reaches_adapter_normalizer_and_binding():
    source = "model.vision_tower.encoder.layers.0.self_attn"
    vision_quantization = _quantization(
        overrides={f"{source}.q_proj": QuantizationOverride(bits=8, group_size=32)},
    )
    config = Gemma4Config(
        model_type="gemma4",
        vocab_size=100,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        global_head_dim=32,
        hidden_act="gelu",
        layer_types=["full_attention"],
        attention_k_eq_v=True,
        num_global_key_value_heads=1,
        vision=VisionConfig(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=2,
            image_size=16,
            patch_size=4,
        ),
        component_quantization={"vision_encoder": vision_quantization},
    )
    module = Gemma4Model(config)
    package = _build(module)
    target = "vision_encoder.encoder.layers.0.self_attn"
    qweight = _bytes(32, 32)
    scales = _floats(32, 1)
    sibling_qweight = _bytes(32, 16, offset=101)
    sibling_scales = _floats(32, 2)
    decoder = _floats(128, 64)

    _assert_layout(package, "vision_encoder", f"{target}.q_proj.weight", 8, 32)
    _assert_layout(package, "vision_encoder", f"{target}.k_proj.weight", 4, 16)
    _load(
        module,
        package,
        {
            f"{source}.q_proj.linear.weight_qweight": qweight,
            f"{source}.q_proj.linear.weight_scales": scales,
            f"{source}.k_proj.linear.weight_qweight": sibling_qweight,
            f"{source}.k_proj.linear.weight_scales": sibling_scales,
            "model.language_model.layers.0.self_attn.q_proj.weight": decoder,
        },
    )

    _assert_bound(
        package, "vision_encoder", f"{target}.q_proj.weight", qweight.reshape(32, 1, 32)
    )
    _assert_bound(package, "vision_encoder", f"{target}.q_proj.scales", scales)
    _assert_bound(
        package,
        "vision_encoder",
        f"{target}.k_proj.weight",
        sibling_qweight.reshape(32, 2, 8),
    )
    _assert_bound(package, "vision_encoder", f"{target}.k_proj.scales", sibling_scales)
    _assert_bound(
        package, "decoder", "decoder.model.layers.0.self_attn.q_proj.weight", decoder
    )
    assert not any(node.op_type == "MatMulNBits" for node in package["decoder"].graph)


@pytest.mark.parametrize("scope", ["encoder", "encoder.block.0.layer.0.SelfAttention.q"])
def test_t5_hf_override_survives_component_rewrite_and_weight_loading(scope):
    decoder = _quantization()
    encoder = _quantization(
        overrides={scope: QuantizationOverride(bits=8, group_size=32)},
    )
    module = T5ForConditionalGeneration(_t5_config(encoder, decoder))
    package = _build(module)
    target = "encoder.block.0.self_attn.q_proj"
    sibling_target = "encoder.block.0.self_attn.k_proj"
    sibling_bits, sibling_group = (8, 32) if scope == "encoder" else (4, 16)
    qweight = _bytes(64, 64)
    scales = _floats(64, 2)
    sibling_qweight = _bytes(64, 64 * sibling_bits // 8, offset=29)
    sibling_scales = _floats(64, 64 // sibling_group)
    decoder_qweight = _bytes(64, 32, offset=109)
    decoder_scales = _floats(64, 4)

    _assert_layout(package, "encoder", f"{target}.weight", 8, 32)
    _assert_layout(package, "encoder", f"{sibling_target}.weight", sibling_bits, sibling_group)
    _load(
        module,
        package,
        {
            "encoder.block.0.layer.0.SelfAttention.q.weight_qweight": qweight,
            "encoder.block.0.layer.0.SelfAttention.q.weight_scales": scales,
            "encoder.block.0.layer.0.SelfAttention.k.weight_qweight": sibling_qweight,
            "encoder.block.0.layer.0.SelfAttention.k.weight_scales": sibling_scales,
            "decoder.block.0.layer.0.SelfAttention.q.weight_qweight": decoder_qweight,
            "decoder.block.0.layer.0.SelfAttention.q.weight_scales": decoder_scales,
        },
    )

    _assert_bound(package, "encoder", f"{target}.weight", qweight.reshape(64, 2, 32))
    _assert_bound(package, "encoder", f"{target}.scales", scales)
    _assert_bound(
        package,
        "encoder",
        f"{sibling_target}.weight",
        sibling_qweight.reshape(64, 64 // sibling_group, sibling_group * sibling_bits // 8),
    )
    _assert_bound(package, "encoder", f"{sibling_target}.scales", sibling_scales)
    _assert_layout(package, "decoder", "decoder.block.0.self_attn.q_proj.weight", 4, 16)
    _assert_bound(
        package,
        "decoder",
        "decoder.block.0.self_attn.q_proj.weight",
        decoder_qweight.reshape(64, 4, 8),
    )
    _assert_bound(
        package, "decoder", "decoder.block.0.self_attn.q_proj.scales", decoder_scales
    )


@pytest.mark.parametrize("rule", ["override", "exclusion"])
def test_t5_decoder_stack_rules_do_not_change_the_top_level_hf_head(rule):
    policy = (
        {"overrides": {"decoder": QuantizationOverride(bits=8, group_size=32)}}
        if rule == "override"
        else {"modules_to_not_convert": ("decoder",)}
    )
    decoder = _quantization(quantize_lm_head=True, **policy)
    module = T5ForConditionalGeneration(_t5_config(None, decoder))
    package = _build(module)
    head = _bytes(100, 32)
    head_scales = _floats(100, 4)
    source = "decoder.block.0.layer.0.SelfAttention.q"
    target = "decoder.block.0.self_attn.q_proj"
    state_dict = {
        "lm_head.weight_qweight": head,
        "lm_head.weight_scales": head_scales,
    }
    if rule == "override":
        stack = _bytes(64, 64)
        stack_scales = _floats(64, 2)
        state_dict[f"{source}.weight_qweight"] = stack
        state_dict[f"{source}.weight_scales"] = stack_scales
        expected_stack = stack.reshape(64, 2, 32)
    else:
        expected_stack = _floats(64, 64)
        state_dict[f"{source}.weight"] = expected_stack

    _load(module, package, state_dict)

    _assert_layout(package, "decoder", "decoder.lm_head.weight", 4, 16)
    _assert_bound(package, "decoder", "decoder.lm_head.weight", head.reshape(100, 4, 8))
    _assert_bound(package, "decoder", "decoder.lm_head.scales", head_scales)
    _assert_bound(package, "decoder", f"{target}.weight", expected_stack)
    if rule == "override":
        _assert_layout(package, "decoder", f"{target}.weight", 8, 32)
        _assert_bound(package, "decoder", f"{target}.scales", stack_scales)
    else:
        assert f"{target}.scales" not in package["decoder"].graph.initializers


def test_t5_symmetry_override_validates_zero_points_per_projection():
    source = "encoder.block.0.layer.0.SelfAttention"
    encoder = _quantization(
        sym=False,
        overrides={f"{source}.q": QuantizationOverride(sym=True)},
    )
    module = T5ForConditionalGeneration(_t5_config(encoder, _quantization()))
    package = _build(module)
    packed = _bytes(64, 32)
    scales = _floats(64, 4)
    zero_points = _bytes(64, 2, offset=47)

    _load(
        module,
        package,
        {
            f"{source}.q.weight_qweight": packed,
            f"{source}.q.weight_scales": scales,
            f"{source}.k.weight_qweight": packed,
            f"{source}.k.weight_scales": scales,
            f"{source}.k.weight_qzeros": zero_points,
        },
    )

    target = "encoder.block.0.self_attn"
    for projection in ("q_proj", "k_proj"):
        _assert_bound(
            package, "encoder", f"{target}.{projection}.weight", packed.reshape(64, 4, 8)
        )
        _assert_bound(package, "encoder", f"{target}.{projection}.scales", scales)
    assert f"{target}.q_proj.zero_points" not in package["encoder"].graph.initializers
    _assert_bound(package, "encoder", f"{target}.k_proj.zero_points", zero_points)


def test_t5_exact_hf_projection_exclusion_keeps_only_that_projection_float():
    encoder = _quantization(
        modules_to_not_convert=("encoder.block.0.layer.0.SelfAttention.q",),
    )
    module = T5ForConditionalGeneration(_t5_config(encoder, _quantization()))
    package = _build(module)
    qweight = _floats(64, 64)
    sibling_qweight = _bytes(64, 32)
    sibling_scales = _floats(64, 4)

    _load(
        module,
        package,
        {
            "encoder.block.0.layer.0.SelfAttention.q.weight": qweight,
            "encoder.block.0.layer.0.SelfAttention.k.weight_qweight": sibling_qweight,
            "encoder.block.0.layer.0.SelfAttention.k.weight_scales": sibling_scales,
        },
    )

    _assert_bound(package, "encoder", "encoder.block.0.self_attn.q_proj.weight", qweight)
    assert "encoder.block.0.self_attn.q_proj.scales" not in (
        package["encoder"].graph.initializers
    )
    _assert_layout(package, "encoder", "encoder.block.0.self_attn.k_proj.weight", 4, 16)
    _assert_bound(
        package,
        "encoder",
        "encoder.block.0.self_attn.k_proj.weight",
        sibling_qweight.reshape(64, 4, 8),
    )
    _assert_bound(
        package, "encoder", "encoder.block.0.self_attn.k_proj.scales", sibling_scales
    )


def _tiny_decoder_config(quantization, component_plan, **overrides):
    return make_config(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        layer_types=["full_attention"],
        num_local_experts=2,
        num_experts_per_tok=1,
        moe_intermediate_size=32,
        shared_expert_intermediate_size=32,
        tie_word_embeddings=False,
        quantization=quantization,
        component_quantization=component_plan,
        **overrides,
    )


def _tiny_qwen_vision_config():
    return VisionConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        patch_size=2,
        in_channels=3,
        out_hidden_size=32,
        num_position_embeddings=4,
    )


@pytest.mark.parametrize("method", ["olive", "gptq", "awq"])
@pytest.mark.parametrize("model_class", [Qwen2MoECausalLMModel, Qwen3NextCausalLMModel])
@pytest.mark.parametrize("component_plan", [False, True], ids=["legacy", "component"])
def test_shared_expert_gate_float_checkpoint_binds(model_class, component_plan, method):
    quantization = _quantization(quant_method=method)
    config = _tiny_decoder_config(
        quantization, {"model": quantization} if component_plan else None
    )
    module = model_class(config)
    assert type(module.model.layers[0].mlp.shared_expert_gate) is Linear
    package = _build(module)
    target = "model.layers.0.mlp.shared_expert_gate.weight"
    value = _floats(1, 32)

    _load(module, package, {target: value})

    assert type(module.model.layers[0].mlp.shared_expert_gate) is Linear
    _assert_bound(package, "model", target, value)


@pytest.mark.parametrize("quantize_embeddings", [False, True])
def test_whisper_float_position_table_still_binds(quantize_embeddings):
    quantization = _quantization(quantize_embeddings=quantize_embeddings)
    config = WhisperConfig(
        vocab_size=100,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=16,
        hidden_act="gelu",
        pad_token_id=0,
        tie_word_embeddings=False,
        encoder_layers=1,
        encoder_attention_heads=4,
        encoder_ffn_dim=128,
        num_mel_bins=16,
        max_source_positions=16,
        max_target_positions=16,
        quantization=quantization,
        component_quantization={"decoder": quantization},
    )
    module = WhisperForConditionalGeneration(config)
    package = _build(module)
    positions = _floats(16, 64)

    _load(module, package, {"model.decoder.embed_positions.weight": positions})

    _assert_bound(package, "decoder", "decoder.embed_positions.weight", positions)


@pytest.mark.parametrize(
    ("model_class", "component_plan"),
    [
        (CausalLMModel, False),
        (CausalLMModel, True),
        (Qwen35CausalLMModel, False),
        (Qwen35CausalLMModel, True),
        (Qwen35MoECausalLMModel, False),
        (Qwen35MoECausalLMModel, True),
        (Qwen35VL3ModelCausalLMModel, True),
        (Qwen35MoEVL3ModelCausalLMModel, True),
    ],
)
@pytest.mark.parametrize("override", [False, True], ids=["root-layout", "leaf-override"])
def test_projection_sidecars_follow_graph_layout(model_class, component_plan, override):
    split = model_class in (Qwen35VL3ModelCausalLMModel, Qwen35MoEVL3ModelCausalLMModel)
    source = (
        "model.language_model.layers.0.self_attn.q_proj"
        if split
        else "model.layers.0.self_attn.q_proj"
    )
    quantization = _quantization(
        overrides={source: QuantizationOverride(bits=8, group_size=32)} if override else {}
    )
    component = "decoder" if split else "model"
    config = _tiny_decoder_config(
        quantization,
        {component: quantization} if component_plan else None,
        image_token_id=99,
        temporal_patch_size=1,
        vision=_tiny_qwen_vision_config(),
    )
    module = model_class(config)
    package = _build(module)
    backbone = module.decoder.model if split else module.model
    projection = backbone.layers[0].self_attn.q_proj
    bits, group_size = (8, 32) if override else (4, 16)
    packed = _bytes(projection._n, projection._k * bits // 8)
    scales = _floats(projection._n, projection._k // group_size)
    target = (
        "decoder.model.layers.0.self_attn.q_proj"
        if split
        else "model.layers.0.self_attn.q_proj"
    )

    result = _load(
        module,
        package,
        {f"{source}.weight_qweight": packed, f"{source}.weight_scales": scales},
    )

    _assert_layout(package, component, f"{target}.weight", bits, group_size)
    _assert_bound(
        package, component, f"{target}.weight", packed.reshape(tuple(projection.weight.shape))
    )
    _assert_bound(package, component, f"{target}.scales", scales)
    assert not any(name.endswith("_qweight") for name in result)


@pytest.mark.parametrize("split", [False, True], ids=["text", "vl"])
@pytest.mark.parametrize("sym", [False, True], ids=["asymmetric", "symmetric"])
def test_qwen35_component_override_preserves_canonical_qmoe_groups(split, sym):
    source = "model.language_model.layers.0" if split else "model.layers.0"
    quantization = _quantization(
        sym=sym,
        overrides={
            f"{source}.self_attn.q_proj": QuantizationOverride(bits=8, group_size=32, sym=True)
        },
    )
    component = "decoder" if split else "model"
    config = _tiny_decoder_config(
        quantization,
        {component: quantization},
        image_token_id=99,
        temporal_patch_size=1,
        vision=_tiny_qwen_vision_config(),
    )
    model_class = Qwen35MoEVL3ModelCausalLMModel if split else Qwen35MoECausalLMModel
    module = model_class(config)
    package = _build(module)
    fc1 = _bytes(2, 64, 16)
    fc2 = _bytes(2, 32, 16, offset=19)
    fc1_scales = _floats(2, 64, 2)
    fc2_scales = _floats(2, 32, 2)
    projection = _bytes(64, 32, offset=37)
    weights = {
        f"{source}.mlp.experts.gate_up_proj_qweight": fc1,
        f"{source}.mlp.experts.gate_up_proj_scales": fc1_scales,
        f"{source}.mlp.experts.down_proj_qweight": fc2,
        f"{source}.mlp.experts.down_proj_scales": fc2_scales,
        f"{source}.self_attn.q_proj.weight_qweight": projection,
        f"{source}.self_attn.q_proj.weight_scales": _floats(64, 1),
    }
    if not sym:
        weights[f"{source}.mlp.experts.gate_up_proj_qzeros"] = _bytes(2, 64, 1)
        weights[f"{source}.mlp.experts.down_proj_qzeros"] = _bytes(2, 32, 1)

    result = _load(module, package, weights)

    target = "decoder.model.layers.0" if split else "model.layers.0"
    for suffix, expected in (
        ("fc1_experts_weights", fc1),
        ("fc2_experts_weights", fc2),
        ("fc1_scales", fc1_scales),
        ("fc2_scales", fc2_scales),
    ):
        _assert_bound(package, component, f"{target}.mlp.{suffix}", expected)
    if not sym:
        for name, rows in (("fc1", 64), ("fc2", 32)):
            _assert_bound(
                package,
                component,
                f"{target}.mlp.{name}_experts_zero_points",
                _bytes(2, rows, 1),
            )
    _assert_bound(
        package, component, f"{target}.self_attn.q_proj.weight", projection.reshape(64, 1, 32)
    )
    task = get_task(module.default_task)
    canonical = normalize_component_quantized_weights(
        result, module, config, package.keys(), task=task
    )
    assert canonical.keys() == result.keys()
    assert all(canonical[name] is value for name, value in result.items())
    incomplete = dict(result)
    del incomplete[f"{target}.mlp.fc1_experts_weights"]
    with pytest.raises(ValueError, match="no matching qweight"):
        normalize_component_quantized_weights(
            incomplete, module, config, package.keys(), task=task
        )


@pytest.mark.parametrize("missing", ["weight", "scales"])
@pytest.mark.parametrize("module_rules", [False, True])
def test_public_global_build_rejects_missing_packed_parameter(
    monkeypatch, missing, module_rules
):
    from transformers import LlamaConfig

    from mobius.integrations.transformers import _builder as transformers_builder

    hf_config = LlamaConfig(
        vocab_size=100,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
        tie_word_embeddings=False,
        quantization_config={
            "quant_method": "olive",
            "bits": 4,
            "group_size": 16,
            "sym": True,
            "modules_to_not_convert": ["lm_head"] if module_rules else [],
        },
    )
    monkeypatch.setattr(
        transformers_builder,
        "_load_transformers_config",
        lambda *args, **kwargs: (hf_config, False),
    )
    weights = _canonical_checkpoint(
        build("test/tiny-llama-global-missing-weight", dtype="f32", load_weights=False)
    )
    target = f"model.layers.0.self_attn.q_proj.{missing}"
    del weights[target]
    monkeypatch.setattr(
        transformers_builder, "_download_weights", lambda *args, **kwargs: dict(weights)
    )

    with pytest.raises(ValueError, match=r"unbound|no matching qweight") as error:
        build("test/tiny-llama-global-missing-weight", dtype="f32")
    assert "model.layers.0.self_attn.q_proj" in str(error.value)
