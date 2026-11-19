# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""LFM3-VL configuration, package contract, weight routing and export metadata tests."""

from __future__ import annotations

import json
from types import SimpleNamespace

import onnx_ir as ir
import pytest
import torch

from mobius import build_from_module
from mobius._configs import Lfm3VlConfig, VisionConfig
from mobius._registry import registry
from mobius.integrations.ort_genai import write_ort_genai_config
from mobius.integrations.transformers._builder import _resolve_module_class
from mobius.models.lfm3_vl import Lfm3VlForConditionalGeneration


def _pinned_composite(text_layer_types: list[str], vision_layer_types: list[str]):
    """Architectural fields of LiquidAI/LFM3-VL-2B@23387bed config.json."""
    text = SimpleNamespace(
        model_type="lfm3",
        hidden_size=2048,
        intermediate_size=7680,
        num_attention_heads=16,
        num_key_value_heads=4,
        num_hidden_layers=24,
        head_dim=128,
        vocab_size=128000,
        max_position_embeddings=128000,
        norm_eps=1e-5,
        conv_L_cache=7,
        conv_bias=False,
        tie_word_embeddings=True,
        use_sandwich_norm=True,
        sliding_window=256,
        pad_token_id=124893,
        bos_token_id=124894,
        eos_token_id=124900,
        rope_parameters={
            "full_attention": {
                "rope_type": "default",
                "rope_theta": 1e6,
                "partial_rotary_factor": 0.5,
            },
            "sliding_attention": {"rope_type": "default", "rope_theta": 10000},
        },
        layer_types=text_layer_types,
    )
    vision = SimpleNamespace(
        model_type="lfm3_vl_vision",
        hidden_size=1024,
        intermediate_size=2560,
        num_hidden_layers=14,
        num_attention_heads=16,
        num_key_value_heads=8,
        layer_types=vision_layer_types,
        patch_size=16,
        temporal_patch_size=2,
        in_channels=3,
        spatial_merge_size=2,
        conv_kernel_size=3,
        norm_eps=1e-5,
        rope_parameters={"rope_theta": 1e6, "rope_type": "axial"},
    )
    return SimpleNamespace(
        model_type="lfm3_vl",
        text_config=text,
        vision_config=vision,
        image_token_id=125030,
        video_token_id=125029,
        vision_start_token_id=125009,
        vision_end_token_id=125010,
        downsample_factor=2,
        projector_hidden_act="gelu",
        projector_hidden_size=4096,
        projector_bias=True,
        projector_layer_norm_eps=1e-6,
        tie_word_embeddings=True,
    )


_TEXT_LAYERS = [
    "full_attention" if i % 4 == 2 else "sliding_attention" if i and i % 4 == 0 else "conv"
    for i in range(24)
]
_VISION_LAYERS = ["conv" if i < 2 or i % 2 else "full_attention" for i in range(14)]


def _tiny_config(**overrides) -> Lfm3VlConfig:
    fields = dict(
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        num_hidden_layers=3,
        vocab_size=256,
        max_position_embeddings=128,
        layer_types=["conv", "full_attention", "sliding_attention"],
        sliding_window=8,
        pad_token_id=0,
        image_token_id=250,
        projector_hidden_size=64,
        vision_num_key_value_heads=2,
        vision_layer_types=["conv", "full_attention"],
        vision=VisionConfig(
            model_type="lfm3_vl_vision",
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            patch_size=4,
            norm_eps=1e-5,
            rope_theta=1e6,
        ),
    )
    fields.update(overrides)
    return Lfm3VlConfig(**fields)


@pytest.mark.parametrize("conv_name", ["conv", "linear_attention"])
def test_config_lifts_pinned_composite_fields(conv_name):
    # Transformers renames custom-code "conv" layers to "linear_attention".
    rename = {"conv": conv_name}
    composite = _pinned_composite(
        [rename.get(kind, kind) for kind in _TEXT_LAYERS],
        [rename.get(kind, kind) for kind in _VISION_LAYERS],
    )
    config = Lfm3VlConfig.from_transformers(composite.text_config, parent_config=composite)

    assert config.model_type == "lfm3_vl"
    assert config.layer_types == _TEXT_LAYERS
    assert config.max_position_embeddings == 128000
    assert config.short_conv_kernel == 7
    assert config.image_token_id == 125030
    assert config.vision_start_token_id == 125009
    assert config.projector_hidden_size == 4096
    assert config.projector_layer_norm_eps == pytest.approx(1e-6)
    assert config.vision_layer_types == _VISION_LAYERS
    assert config.vision_num_key_value_heads == 8
    assert config.vision_conv_kernel_size == 3
    assert config.vision.rope_theta == pytest.approx(1e6)
    assert config.vision.norm_eps == pytest.approx(1e-5)
    assert config.vision.temporal_patch_size == 2
    assert registry.get_config_class("lfm3_vl") is Lfm3VlConfig


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"vision_layer_types": ["conv"]}, "layer count"),
        ({"vision_layer_types": ["conv", "sliding_attention"]}, "Unsupported"),
        ({"vision_num_key_value_heads": 3}, "head counts"),
        ({"downsample_factor": 4}, "spatial_merge_size"),
    ],
)
def test_invalid_vision_config_fails_explicitly(overrides, match):
    with pytest.raises(ValueError, match=match):
        _tiny_config(**overrides)


def test_package_contract_one_image_per_vision_call():
    config = _tiny_config()
    pkg = build_from_module(Lfm3VlForConditionalGeneration(config), config, task="lfm3-vl")
    assert set(pkg) == {"decoder", "vision_encoder", "embedding"}

    vision = {value.name: value for value in pkg["vision_encoder"].graph.inputs}
    assert list(vision) == ["pixel_values", "image_grid_thw"]
    assert vision["pixel_values"].dtype == ir.DataType.FLOAT
    assert vision["pixel_values"].shape[1] == 3 * 2 * 4 * 4
    assert list(vision["image_grid_thw"].shape) == [1, 3]
    assert [value.name for value in pkg["vision_encoder"].graph.outputs] == ["image_features"]

    decoder = {value.name: value for value in pkg["decoder"].graph.inputs}
    assert list(decoder)[:3] == ["inputs_embeds", "attention_mask", "position_ids"]
    assert list(decoder["past_key_values.0.conv_state"].shape)[1:] == [64, 6]
    for i in (1, 2):
        assert list(decoder[f"past_key_values.{i}.key"].shape)[1::2] == [2, 16]
    assert "Unpacked" in pkg["decoder"].metadata_props["mobius.input_contract"]


def test_cuda_build_keeps_standard_decoder_attention():
    config = _tiny_config()
    pkg = build_from_module(
        Lfm3VlForConditionalGeneration(config),
        config,
        task="lfm3-vl",
        execution_provider="cuda",
    )
    attention = [node for node in pkg["decoder"].graph if "Attention" in node.op_type]
    assert [node.op_type for node in attention] == ["Attention", "Attention"]
    assert all(
        node.metadata_props["mobius.attention.preserve_standard"] == "true"
        for node in attention
    )


@pytest.mark.parametrize(
    "ep,op",
    [("cpu", ("com.microsoft", "MultiHeadAttention")), ("onnx-standard", ("", "Attention"))],
)
def test_vision_attention_repeats_key_value_heads(ep, op):
    config = _tiny_config()
    pkg = build_from_module(
        Lfm3VlForConditionalGeneration(config), config, task="lfm3-vl", execution_provider=ep
    )
    (node,) = [node for node in pkg["vision_encoder"].graph if "Attention" in node.op_type]
    assert (node.domain, node.op_type) == op
    heads = node.attributes.get_int("kv_num_heads", node.attributes.get_int("num_heads"))
    assert heads == config.vision.num_attention_heads > config.vision_num_key_value_heads


def test_qnn_vision_encoder_has_no_contrib_ops():
    config = _tiny_config()
    pkg = build_from_module(
        Lfm3VlForConditionalGeneration(config),
        config,
        task="lfm3-vl",
        execution_provider="qnn",
    )
    assert {node.domain for node in pkg["vision_encoder"].graph} <= {"", "ai.onnx"}


def test_vision_norm_epsilons_follow_config():
    # 2e-6 differs from the LayerNorm component's 1e-6 default.
    config = _tiny_config(projector_layer_norm_eps=2e-6)
    pkg = build_from_module(Lfm3VlForConditionalGeneration(config), config, task="lfm3-vl")
    expected = {
        "RMSNormalization": config.vision.norm_eps,
        "SkipSimplifiedLayerNormalization": config.vision.norm_eps,
        "LayerNormalization": config.projector_layer_norm_eps,
        "SkipLayerNormalization": config.projector_layer_norm_eps,
    }
    norms = [node for node in pkg["vision_encoder"].graph if node.op_type in expected]
    layers = config.vision_layer_types
    assert len(norms) == 1 + 2 * len(layers) + 2 * layers.count("full_attention") + 1
    for node in norms:
        assert node.attributes.get_float("epsilon") == pytest.approx(expected[node.op_type])


@pytest.mark.parametrize("dtype", [ir.DataType.FLOAT16, ir.DataType.BFLOAT16])
def test_rejects_unverified_reduced_precision(dtype):
    with pytest.raises(ValueError, match="float32 compute only"):
        Lfm3VlForConditionalGeneration(_tiny_config(dtype=dtype))


def test_preprocess_weights_routes_by_prefix_and_ties_lm_head():
    module = Lfm3VlForConditionalGeneration(_tiny_config())
    embed = torch.zeros(256, 64)
    state_dict = {
        "model.vision_tower.layers.1.self_attn.q_layernorm.weight": torch.zeros(8),
        "model.vision_tower.layers.0.conv.spatial_conv.weight": torch.zeros(32, 1, 3, 3),
        "model.multi_modal_projector.linear_1.bias": torch.zeros(64),
        "model.language_model.embed_tokens.weight": embed,
        "model.language_model.layers.1.self_attn.gate_proj.weight": torch.zeros(4, 64),
    }
    renamed = module.preprocess_weights(state_dict)

    assert set(renamed) == {
        "vision_encoder.vision_tower.layers.1.self_attn.q_layernorm.weight",
        "vision_encoder.vision_tower.layers.0.conv.spatial_conv.weight",
        "vision_encoder.multi_modal_projector.linear_1.bias",
        "decoder.model.embed_tokens.weight",
        "decoder.model.layers.1.self_attn.gate_proj.weight",
        "embedding.embed_tokens.weight",
        "decoder.lm_head.weight",
    }
    assert renamed["decoder.lm_head.weight"] is embed
    assert renamed["embedding.embed_tokens.weight"] is embed


def test_graph_derived_genai_metadata(tmp_path):
    config = _tiny_config()
    pkg = build_from_module(Lfm3VlForConditionalGeneration(config), config, task="lfm3-vl")
    write_ort_genai_config(pkg, str(tmp_path))
    metadata = json.loads((tmp_path / "genai_config.json").read_text())

    assert metadata["model"]["type"] == "lfm3_vl"
    decoder = metadata["model"]["decoder"]
    assert decoder["layer_types"] == ["conv", "full_attention", "sliding_attention"]
    assert decoder["conv_cache_size"] == 6
    assert decoder["inputs"]["inputs_embeds"] == "inputs_embeds"
    assert decoder["inputs"]["past_conv_names"] == "past_key_values.%d.conv_state"
    assert metadata["search"]["past_present_share_buffer"] is False
    vision = metadata["model"]["vision"]
    assert vision["inputs"] == {
        "pixel_values": "pixel_values",
        "image_grid_thw": "image_grid_thw",
    }
    assert vision["spatial_merge_size"] == 2
    assert vision["patch_size"] == 4
    assert vision["config_filename"] == "processor_config.json"
    assert metadata["model"]["image_token_id"] == 250
    assert not (tmp_path / "image_processor.json").exists()


def test_genai_image_pipeline_follows_the_hf_processor(tmp_path):
    # image_processor of LiquidAI/LFM3-VL-2B@23387bed processor_config.json, with the tiny patch size.
    image_processor = {
        "do_convert_rgb": True,
        "do_normalize": True,
        "do_rescale": True,
        "do_resize": True,
        "image_mean": [0.5, 0.5, 0.5],
        "image_std": [0.5, 0.5, 0.5],
        "merge_size": 2,
        "patch_size": 4,
        "resample": 3,
        "rescale_factor": 0.00392156862745098,
        "size": {"longest_edge": 16777216, "shortest_edge": 65536},
        "temporal_patch_size": 2,
    }
    hf_file = tmp_path / "processor_config.json"
    hf_file.write_text(json.dumps({"image_processor": image_processor}))
    config = _tiny_config()
    pkg = build_from_module(Lfm3VlForConditionalGeneration(config), config, task="lfm3-vl")
    write_ort_genai_config(pkg, str(tmp_path))

    steps = [
        step["operation"]
        for step in json.loads(hf_file.read_text())["processor"]["transforms"]
    ]
    assert [step["type"] for step in steps] == [
        "DecodeImage",
        "Resize",
        "Rescale",
        "Normalize",
        "PatchImage",
    ]
    assert steps[1]["attrs"] == {
        "interpolation": "CUBIC",
        "smart_resize": 1,
        "min_pixels": 65536,
        "max_pixels": 16777216,
        "patch_size": 4,
        "merge_size": 2,
    }
    assert steps[4]["attrs"] == {"patch_size": 4, "temporal_patch_size": 2, "merge_size": 2}
    converted = hf_file.read_text()
    write_ort_genai_config(pkg, str(tmp_path))
    assert hf_file.read_text() == converted

    hf_file.write_text(json.dumps({"image_processor": {**image_processor, "resample": 2}}))
    with pytest.raises(ValueError, match="resample"):
        write_ort_genai_config(pkg, str(tmp_path))


def test_registered_model_and_task():
    assert registry.get("lfm3_vl") is Lfm3VlForConditionalGeneration
    assert registry.get_task("lfm3_vl") == "lfm3-vl"


def test_d1_checkpoints_are_refused():
    # LiquidAI/d1-2B@a6bce774 config.json declares model_type lfm3_vl with architectures ["D1Model"].
    with pytest.raises(ValueError, match="D1Model"):
        _resolve_module_class(
            "lfm3_vl", SimpleNamespace(architectures=["D1Model"]), None, None
        )
    module_class, _, _ = _resolve_module_class(
        "lfm3_vl",
        SimpleNamespace(architectures=["Lfm3VlForConditionalGeneration"]),
        None,
        None,
    )
    assert module_class is Lfm3VlForConditionalGeneration
