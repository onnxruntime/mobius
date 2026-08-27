# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import hashlib
import json
import pathlib
import re
from types import SimpleNamespace
from unittest import mock

import numpy as np
import onnx
import onnx_ir as ir
import pytest
import safetensors.torch
import torch

from mobius._builder import build_from_module
from mobius._configs import Qwen4ExpConfig, VisionConfig
from mobius._registry import registry
from mobius._testing import create_test_builder, create_test_input
from mobius._testing.ort_inference import OnnxModelSession
from mobius.integrations._block_quant import BlockQuantScheme
from mobius.integrations._weight_loading import (
    apply_weights,
    stream_preprocessed_safetensors_to_model,
    stream_qdq_safetensors_to_model,
)
from mobius.models.qwen4_exp import (
    _PINNED_PLE_WEIGHT_SCALE,
    Qwen4ExpCausalLMModel,
    Qwen4ExpForConditionalGeneration,
    Qwen4ExpQSAIndexer,
    Qwen4ExpVLDecoderModel,
)
from mobius.tasks._qwen4_exp import Qwen4ExpVisionLanguageTask


def _config(**overrides) -> Qwen4ExpConfig:
    values = dict(
        model_type="qwen4_exp_text",
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        hidden_act="silu",
        max_position_embeddings=128,
        rms_norm_eps=1e-6,
        rope_type="default",
        rope_theta=10_000.0,
        partial_rotary_factor=0.25,
        pad_token_id=0,
        eos_token_id=1,
        layer_types=["linear_attention", "qwen_sparse_attention"],
        linear_conv_kernel_dim=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_num_key_heads=1,
        linear_num_value_heads=2,
        num_local_experts=2,
        num_experts_per_tok=1,
        moe_intermediate_size=8,
        shared_expert_intermediate_size=8,
        hc_count=2,
        hc_lowrank=4,
        indexer_n_heads=2,
        indexer_kv_heads=1,
        indexer_head_dim=8,
        indexer_budget=4,
        indexer_compress_ratio=2,
        ple_layer_ids=[1],
        ple_embed_dim=8,
        ple_conv_kernel_size=2,
        ngram_size=3,
        heads_per_ngram=2,
        ngram_vocab_size_base=31,
        make_ngram_vocab_size_divisible_by=8,
        split_ngram_parts=4,
        mtp_num_hidden_layers=0,
    )
    values.update(overrides)
    return Qwen4ExpConfig(**values)


def _build(config: Qwen4ExpConfig | None = None):
    config = config or _config()
    module = Qwen4ExpCausalLMModel(config)
    model = build_from_module(module, config, task="qwen4-exp-text-generation")["model"]
    return config, module, model


def _vl_config(**overrides) -> Qwen4ExpConfig:
    values = dict(
        model_type="qwen4_exp",
        vision=VisionConfig(
            model_type="qwen4_exp_vision",
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=2,
            patch_size=16,
            temporal_patch_size=2,
            spatial_merge_size=2,
            out_hidden_size=16,
            num_position_embeddings=16,
            hidden_act="gelu_pytorch_tanh",
            deepstack_visual_indexes=[],
        ),
        image_token_id=30,
        video_token_id=None,
        unsupported_video_token_id=31,
        vision_start_token_id=29,
        vision_end_token_id=28,
        mrope_section=[1, 1, 0],
        mrope_interleaved=True,
        deepstack_visual_indexes=[],
    )
    values.update(overrides)
    return _config(**values)


def test_pinned_config_fields_extract_and_normalize_schedule():
    text = SimpleNamespace(
        model_type="qwen4_exp_text",
        vocab_size=248320,
        hidden_size=2560,
        intermediate_size=10240,
        num_hidden_layers=4,
        num_attention_heads=24,
        num_key_value_heads=2,
        head_dim=256,
        hidden_act="silu",
        max_position_embeddings=262144,
        rms_norm_eps=1e-6,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10_000_000,
            "partial_rotary_factor": 0.25,
            "mrope_interleaved": True,
            "mrope_section": [11, 11, 10],
        },
        dtype="bfloat16",
        layer_types=[
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
        ],
        linear_conv_kernel_dim=4,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_num_key_heads=16,
        linear_num_value_heads=48,
        num_experts=512,
        num_experts_per_tok=10,
        moe_intermediate_size=640,
        shared_expert_intermediate_size=640,
        hc_count=4,
        hc_lowrank=320,
        ple_layer_ids=[2],
        ple_embed_dim=2560,
        ple_conv_kernel_size=4,
        ngram_size=3,
        heads_per_ngram=8,
        ngram_vocab_size_base=20_000_000,
        make_ngram_vocab_size_divisible_by=128,
        split_ngram_parts=128,
        indexer_n_heads=4,
        indexer_kv_heads=1,
        indexer_head_dim=128,
        indexer_budget=2048,
        indexer_compress_ratio=4,
        output_gate_type="sigmoid",
        mamba_ssm_dtype="float32",
        eos_token_id=248044,
        mtp_num_hidden_layers=0,
    )
    config = Qwen4ExpConfig.from_transformers(
        text,
        parent_config=SimpleNamespace(
            model_type="qwen4_exp",
            text_config=text,
            tie_word_embeddings=False,
            quantization_config={
                "quant_method": "fp8",
                "weight_block_size": [128, 128],
                "activation_scheme": "dynamic",
            },
        ),
    )

    assert config.model_type == "qwen4_exp_text"
    assert config.layer_types == [
        "linear_attention",
        "linear_attention",
        "linear_attention",
        "qwen_sparse_attention",
    ]
    assert config.linear_num_value_heads == 48
    assert config.num_local_experts == 512
    assert config.num_experts_per_tok == 10
    assert config.hc_count == 4
    assert config.ple_layer_ids == [2]
    assert config.indexer_budget == 2048
    assert config.dtype == ir.DataType.BFLOAT16
    assert config.mrope_interleaved
    assert config.mrope_section == [11, 11, 10]
    assert config.mamba_ssm_dtype == ir.DataType.FLOAT
    assert config.block_quant_scheme is not None
    assert config.block_quant_scheme.weight_block_size == (128, 128)

    parent = SimpleNamespace(
        model_type="qwen4_exp",
        architectures=["Qwen4ExpForConditionalGeneration"],
        language_model_only=False,
        text_config=text,
        vision_config=SimpleNamespace(
            model_type="qwen4_exp",
            hidden_size=1152,
            intermediate_size=4304,
            num_hidden_layers=27,
            num_attention_heads=16,
            in_channels=3,
            patch_size=16,
            temporal_patch_size=2,
            spatial_merge_size=2,
            num_position_embeddings=2304,
            out_hidden_size=2560,
            hidden_act="gelu_pytorch_tanh",
            deepstack_visual_indexes=[],
        ),
        image_token_id=248056,
        video_token_id=248057,
        vision_start_token_id=248053,
        vision_end_token_id=248054,
        tie_word_embeddings=False,
    )
    multimodal = Qwen4ExpConfig.from_transformers(parent)
    assert multimodal.model_type == "qwen4_exp"
    assert multimodal.vision is not None
    assert multimodal.vision.out_hidden_size == 2560
    assert multimodal.image_token_id == 248056
    assert multimodal.video_token_id is None
    assert multimodal.unsupported_video_token_id == 248057
    assert multimodal.vision_start_token_id == 248053
    assert multimodal.vision_end_token_id == 248054
    assert multimodal.deepstack_visual_indexes == []

    parent.vision_config.in_channels = 1
    with pytest.raises(ValueError, match="in_channels"):
        Qwen4ExpConfig.from_transformers(parent)
    parent.vision_config.in_channels = 3
    parent.video_token_id = 42
    with pytest.raises(ValueError, match="video_token_id 248057"):
        Qwen4ExpConfig.from_transformers(parent)


def test_immutable_fp8_schema_fixture_matches_integration_contract():
    fixture_path = (
        pathlib.Path(__file__).parents[3]
        / "testdata/evidence/causal-lm/qwen3.8-flash-next-fp8-schema.json"
    )
    fixture = json.loads(fixture_path.read_text())
    source = fixture["source_checkpoint"]
    quantization = fixture["quantization_contract"]
    export = fixture["mobius_export"]

    assert source["repository"] == "unsloth/Qwen3.8-Flash-Next-FP8"
    assert source["revision"] == "41cc25fe32cc20053a59c89716196897580cddf6"
    assert source["index"]["tensor_payload_bytes"] == 185_502_232_570
    assert source["shards"]["count"] == 131
    assert fixture["header_census"]["tensor_count"] == 152_089
    assert quantization["weight_block_size"] == [128, 128]
    assert quantization["scaled_fp8_pairs"]["text_core"] == 73_728
    assert quantization["scaled_fp8_pairs"]["invalid_or_missing_grids"] == 0
    ple = quantization["ple_scalar_scaled"]
    assert ple["count"] == 128
    assert ple["scale_tensor_count"] == 1
    assert ple["scale_bfloat16_bits"] == "0x3951"
    assert ple["scale_value_float32"] == _PINNED_PLE_WEIGHT_SCALE
    assert export["native_fp8"] is False
    assert export["multimodal_package_complete"] is False
    assert export["mtp_exported"] is False
    assert export["excluded_tensors"]["mtp"]["count"] == 3_101
    assert export["excluded_tensors"]["visual"]["count"] == 333


def test_mtp_metadata_is_preserved_but_dedicated_execution_fails_closed():
    assert _config(mtp_num_hidden_layers=1).mtp_num_hidden_layers == 1
    with pytest.raises(ValueError, match="dedicated MTP embeddings"):
        _config(mtp_use_dedicated_embeddings=True)


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"hc_count": 1}, "hc_count > 1"),
        ({"indexer_kv_heads": 2}, "indexer_kv_heads=1"),
        ({"indexer_budget": 3}, "divisible"),
        ({"ple_layer_ids": [2]}, "only supported on linear_attention"),
        (
            {"linear_num_key_heads": 2, "linear_num_value_heads": 3},
            "must be divisible",
        ),
        ({"linear_conv_kernel_dim": 0}, "must be positive"),
        ({"make_ngram_vocab_size_divisible_by": 0}, "must be > 0"),
        ({"mamba_ssm_dtype": ir.DataType.BFLOAT16}, "mamba_ssm_dtype=float32"),
    ],
)
def test_exact_config_guards(override, message):
    with pytest.raises(ValueError, match=message):
        _config(**override)


@pytest.mark.parametrize("interleaved", [False, True])
def test_qsa_rope_uses_full_rotary_width_from_half_width_frequency_cache(interleaved):
    config = _config(partial_rotary_factor=0.5)
    # The pinned model is half-split, but the reusable rotation path must
    # preserve ONNX RotaryEmbedding semantics for either layout.
    config.rope_interleave = interleaved
    indexer = Qwen4ExpQSAIndexer(config)
    builder, op, graph = create_test_builder()
    value = create_test_input(builder, "value", [1, 3, 2, 8])
    cos = create_test_input(builder, "cos", [1, 3, 2])
    sin = create_test_input(builder, "sin", [1, 3, 2])
    output = indexer._rotate(op, value, (cos, sin), num_heads=2)
    output.name = "output"
    graph.outputs.append(output)
    model = ir.Model(graph, ir_version=11)
    rotary_node = next(node for node in graph if node.op_type == "RotaryEmbedding")
    assert rotary_node.attributes["rotary_embedding_dim"].value == 4
    assert rotary_node.attributes["interleaved"].value == interleaved

    rng = np.random.default_rng(5)
    value_data = rng.normal(size=(1, 3, 2, 8)).astype(np.float32)
    cos_data = rng.normal(size=(1, 3, 2)).astype(np.float32)
    sin_data = rng.normal(size=(1, 3, 2)).astype(np.float32)
    session = OnnxModelSession(model)
    try:
        actual = session.run({"value": value_data, "cos": cos_data, "sin": sin_data})["output"]
    finally:
        session.close()

    rotary = value_data[..., :4]
    if interleaved:
        rotated_half = np.empty_like(rotary)
        rotated_half[..., 0::2] = -rotary[..., 1::2]
        rotated_half[..., 1::2] = rotary[..., 0::2]
        expanded_cos = np.repeat(cos_data[:, :, None, :], 2, axis=-1)
        expanded_sin = np.repeat(sin_data[:, :, None, :], 2, axis=-1)
    else:
        first, second = np.split(rotary, 2, axis=-1)
        rotated_half = np.concatenate((-second, first), axis=-1)
        expanded_cos = np.concatenate((cos_data, cos_data), axis=-1)[:, :, None, :]
        expanded_sin = np.concatenate((sin_data, sin_data), axis=-1)[:, :, None, :]
    expected_rotary = rotary * expanded_cos + rotated_half * expanded_sin
    expected = np.concatenate((expected_rotary, value_data[..., 4:]), axis=-1)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_registry_routes_composite_and_text_model_types():
    from mobius._registry import _TEXT_ONLY_MODEL_TYPE

    for architecture in ("qwen4_exp", "Qwen4ExpForConditionalGeneration"):
        registration = registry.get_registration(architecture)
        assert registration.module_class is Qwen4ExpForConditionalGeneration
        assert registration.config_class is Qwen4ExpConfig
        assert registration.task == "qwen4-exp-vision-language"
    registration = registry.get_registration("qwen4_exp_text")
    assert registration.module_class is Qwen4ExpCausalLMModel
    assert registration.config_class is Qwen4ExpConfig
    assert registration.task == "qwen4-exp-text-generation"
    assert _TEXT_ONLY_MODEL_TYPE["qwen4_exp"] == "qwen4_exp_text"


def test_multimodal_package_exposes_exact_three_model_io():
    config = _vl_config()
    module = Qwen4ExpForConditionalGeneration(config)
    package = build_from_module(
        module,
        config,
        task="qwen4-exp-vision-language",
    )

    assert set(package) == {"decoder", "vision_encoder", "embedding"}
    decoder_inputs = {value.name: value for value in package["decoder"].graph.inputs}
    decoder_outputs = {value.name for value in package["decoder"].graph.outputs}
    assert {"inputs_embeds", "ple_input_ids", "past_position_ids"} <= decoder_inputs.keys()
    assert "input_ids" not in decoder_inputs
    assert decoder_inputs["position_ids"].shape[0] == 4
    assert decoder_inputs["past_position_ids"].shape[0] == 4
    assert {
        "present_position_ids",
        "present.0.conv_state",
        "present.0.recurrent_state",
        "present.0.ple_conv_state",
        "present.0.ple_context",
        "present.1.key",
        "present.1.value",
        "present.1.index_key",
    } <= decoder_outputs
    assert {value.name for value in package["vision_encoder"].graph.inputs} == {
        "pixel_values",
        "image_grid_thw",
    }
    assert {value.name for value in package["embedding"].graph.inputs} == {
        "input_ids",
        "image_features",
    }
    assert json.loads(package["embedding"].metadata_props["mobius.unsupported_token_ids"]) == {
        "video": config.unsupported_video_token_id
    }


def test_multimodal_wrapper_fails_closed_without_exact_composite_shape():
    with pytest.raises(ValueError, match="requires a vision config"):
        Qwen4ExpForConditionalGeneration(_config(model_type="qwen4_exp"))
    with pytest.raises(ValueError, match="does not support DeepStack"):
        Qwen4ExpForConditionalGeneration(
            _vl_config(
                vision=VisionConfig(
                    hidden_size=32,
                    intermediate_size=64,
                    num_hidden_layers=1,
                    num_attention_heads=2,
                    patch_size=16,
                    out_hidden_size=16,
                    num_position_embeddings=16,
                    hidden_act="gelu_pytorch_tanh",
                    deepstack_visual_indexes=[0],
                )
            )
        )
    with pytest.raises(ValueError, match="video inputs are unsupported"):
        Qwen4ExpForConditionalGeneration(_vl_config(video_token_id=31))


def test_multimodal_embedding_preserves_global_image_order():
    import onnxruntime as ort

    from mobius.integrations._weight_loading import apply_weights

    config = _vl_config()
    package = build_from_module(
        Qwen4ExpForConditionalGeneration(config),
        config,
        task="qwen4-exp-vision-language",
    )
    embedding = package["embedding"]
    weight = torch.arange(config.vocab_size * config.hidden_size, dtype=torch.float32).reshape(
        config.vocab_size, config.hidden_size
    )
    apply_weights(embedding, {"embedding.embed_tokens.weight": weight})

    input_ids = np.array(
        [
            [2, config.image_token_id, 3, 4],
            [5, 6, config.image_token_id, 7],
        ],
        dtype=np.int64,
    )
    image_features = np.stack(
        [
            np.full(config.hidden_size, 101.0, dtype=np.float32),
            np.full(config.hidden_size, 202.0, dtype=np.float32),
        ]
    )
    guard_reshape = next(
        node
        for node in embedding.graph
        if node.op_type == "Reshape"
        and node.inputs[1] is not None
        and node.inputs[1].producer() is not None
        and node.inputs[1].producer().op_type == "Unsqueeze"
        and node.inputs[1].producer().inputs[0].producer() is not None
        and node.inputs[1].producer().inputs[0].producer().op_type == "Add"
    )
    assert (
        guard_reshape.inputs[1].producer().inputs[0].producer().inputs[0].producer()
        is not None
    )

    devices = ["cpu"]
    if "CUDAExecutionProvider" in ort.get_available_providers():
        devices.append("cuda")
    actual = None
    for device in devices:
        session = OnnxModelSession(embedding, device=device)
        try:
            device_actual = session.run(
                {
                    "input_ids": input_ids,
                    "image_features": image_features,
                }
            )["inputs_embeds"]
            if actual is None:
                actual = device_actual
            else:
                np.testing.assert_array_equal(device_actual, actual)
            video_input_ids = input_ids.copy()
            video_input_ids[0, 3] = config.unsupported_video_token_id
            with pytest.raises(
                (
                    ort.capi.onnxruntime_pybind11_state.Fail,
                    ort.capi.onnxruntime_pybind11_state.RuntimeException,
                ),
                match=r"cannot be reshaped|input_shape_size",
            ):
                session.run(
                    {
                        "input_ids": video_input_ids,
                        "image_features": image_features,
                    }
                )
        finally:
            session.close()

    assert actual is not None
    np.testing.assert_array_equal(actual[0, 1], image_features[0])
    np.testing.assert_array_equal(actual[1, 2], image_features[1])
    np.testing.assert_array_equal(actual[0, 0], weight[input_ids[0, 0]].numpy())


def test_state_manifest_preserves_layers_and_runtime_workflows_fail_closed(tmp_path):
    from mobius._model_package import ModelPackage
    from mobius.integrations.onnx_genai.auto_export import write_onnx_genai_config
    from mobius.integrations.onnx_genai.workflow_metadata import (
        build_vlm_workflow_metadata,
    )
    from mobius.integrations.ort_genai.auto_export import write_ort_genai_config

    config = _vl_config()
    package = Qwen4ExpVisionLanguageTask().build(
        Qwen4ExpForConditionalGeneration(config),
        config,
    )
    state = json.loads(package["decoder"].metadata_props["mobius.state_manifest"])
    assert state["position_state"] == {
        "input": "past_position_ids",
        "output": "present_position_ids",
        "axes": ["text", "temporal", "height", "width"],
        "update": "replace",
    }
    assert state["layers"] == [
        {
            "index": 0,
            "type": "linear_attention",
            "roles": [
                "conv_state",
                "recurrent_state",
                "ple_conv_state",
                "ple_context",
            ],
            "update": {
                "conv_state": "replace",
                "recurrent_state": "replace",
                "ple_conv_state": "replace",
                "ple_context": "replace",
            },
        },
        {
            "index": 1,
            "type": "qwen_sparse_attention",
            "roles": ["key", "value", "index_key"],
            "update": {"key": "append", "value": "append", "index_key": "replace"},
        },
    ]
    ort_output = tmp_path / "ort"
    with pytest.raises(ValueError, match="cannot represent Qwen4-Exp"):
        write_ort_genai_config(package, str(ort_output))
    assert not ort_output.exists()

    with pytest.raises(ValueError, match="cannot bind Qwen4-Exp"):
        build_vlm_workflow_metadata(package, config)
    onnx_genai_output = tmp_path / "onnx-genai"
    with pytest.raises(ValueError, match="cannot represent Qwen4-Exp"):
        write_onnx_genai_config(
            package,
            str(onnx_genai_output),
            config=config,
        )
    assert not onnx_genai_output.exists()

    override_output = tmp_path / "override"
    override_output.mkdir()
    sentinel = override_output / "sentinel.bin"
    sentinel.write_bytes(b"unchanged")
    snapshot = {path.name: path.read_bytes() for path in override_output.iterdir()}
    non_qwen_override = SimpleNamespace(model_type="qwen2")
    with pytest.raises(ValueError, match="cannot represent Qwen4-Exp"):
        write_onnx_genai_config(
            package,
            str(override_output),
            config=non_qwen_override,
        )
    assert {path.name: path.read_bytes() for path in override_output.iterdir()} == snapshot

    structurally_identical = ModelPackage(
        dict(package),
        config=non_qwen_override,
    )
    structural_output = tmp_path / "structural"
    with pytest.raises(ValueError, match="cannot represent Qwen4-Exp"):
        write_onnx_genai_config(
            structurally_identical,
            str(structural_output),
            config=non_qwen_override,
        )
    assert not structural_output.exists()


def test_ort_genai_text_export_also_fails_before_writing_artifacts(tmp_path):
    from mobius.integrations.ort_genai.auto_export import export_package

    config = _config()
    package = build_from_module(
        Qwen4ExpCausalLMModel(config),
        config,
        task="qwen4-exp-text-generation",
    )
    output_dir = tmp_path / "output"
    with pytest.raises(ValueError, match="cannot represent Qwen4-Exp"):
        export_package(package, str(output_dir))
    assert not output_dir.exists()


def test_processor_config_matches_qwen4exp_graph_contract(tmp_path):
    from mobius.integrations.ort_genai.auto_export import (
        _write_vision_processor_config,
    )

    image_processor = SimpleNamespace(
        image_mean=[0.5, 0.5, 0.5],
        image_std=[0.5, 0.5, 0.5],
        rescale_factor=1.0 / 255.0,
        resample=3,
        size={"shortest_edge": 65_536, "longest_edge": 16_777_216},
    )
    with mock.patch(
        "transformers.AutoProcessor.from_pretrained",
        return_value=SimpleNamespace(image_processor=image_processor),
    ):
        path = _write_vision_processor_config(
            _vl_config(),
            str(tmp_path),
            hf_model_id="Qwen/Qwen3.8-Flash-Next",
            revision="f5d08274bafd880402bd16f5e3e6c514136ec06c",
        )
    assert path is not None
    with open(path, encoding="utf-8") as handle:
        processor = json.load(handle)["processor"]
    transforms = processor["transforms"]
    assert processor["name"] == "qwen2_5_image_processor"
    assert [item["operation"]["type"] for item in transforms] == [
        "DecodeImage",
        "Resize",
        "Rescale",
        "Normalize",
        "PatchImage",
    ]
    assert transforms[1]["operation"]["attrs"]["min_pixels"] == 65_536
    assert transforms[1]["operation"]["attrs"]["max_pixels"] == 16_777_216
    assert transforms[3]["operation"]["attrs"]["mean"] == [0.5, 0.5, 0.5]
    assert transforms[3]["operation"]["attrs"]["std"] == [0.5, 0.5, 0.5]
    assert transforms[4]["operation"]["attrs"] == {
        "patch_size": 16,
        "temporal_patch_size": 2,
        "merge_size": 2,
    }

    fallback_dir = tmp_path / "fallback"
    fallback_dir.mkdir()
    with mock.patch(
        "transformers.AutoProcessor.from_pretrained",
        side_effect=OSError("offline"),
    ):
        fallback_path = _write_vision_processor_config(
            _vl_config(),
            str(fallback_dir),
            hf_model_id="Qwen/Qwen3.8-Flash-Next",
            revision="f5d08274bafd880402bd16f5e3e6c514136ec06c",
        )
    assert fallback_path is not None
    with open(fallback_path, encoding="utf-8") as handle:
        fallback = json.load(handle)["processor"]["transforms"]
    assert fallback[1]["operation"]["attrs"]["min_pixels"] == 65_536
    assert fallback[1]["operation"]["attrs"]["max_pixels"] == 16_777_216
    assert fallback[3]["operation"]["attrs"]["mean"] == [0.5, 0.5, 0.5]
    assert fallback[3]["operation"]["attrs"]["std"] == [0.5, 0.5, 0.5]


def test_bfloat16_vision_graph_casts_float_processor_input_and_loads_in_ort():
    config = _vl_config(dtype=ir.DataType.BFLOAT16)
    vision = build_from_module(
        Qwen4ExpForConditionalGeneration(config),
        config,
        task="qwen4-exp-vision-language",
    )["vision_encoder"]
    pixel_values = next(value for value in vision.graph.inputs if value.name == "pixel_values")
    assert pixel_values.dtype == ir.DataType.FLOAT
    cast = next(
        node
        for node in vision.graph
        if node.op_type in {"Cast", "CastLike"} and node.inputs[0] is pixel_values
    )
    assert cast.outputs[0].dtype == ir.DataType.BFLOAT16

    for initializer in vision.graph.initializers.values():
        if initializer.const_value is not None:
            continue
        shape = [int(dim) for dim in initializer.shape]
        if initializer.dtype == ir.DataType.BFLOAT16:
            initializer.const_value = ir.Tensor(
                np.zeros(shape, dtype=np.uint16),
                dtype=ir.DataType.BFLOAT16,
            )
        else:
            raise AssertionError(f"Unexpected uninitialized vision dtype {initializer.dtype}")
    import tempfile
    from pathlib import Path

    import onnxruntime as ort

    with tempfile.TemporaryDirectory() as directory:
        path = str(Path(directory) / "vision.onnx")
        ir.save(vision, path)
        if "CUDAExecutionProvider" in ort.get_available_providers():
            ort.InferenceSession(path, providers=["CUDAExecutionProvider"])
        else:
            # CPU ORT has no BF16 Conv kernel. Reaching kernel resolution proves
            # model loading passed graph type validation; before the Cast this
            # failed earlier with mixed FLOAT/BFLOAT16 Conv inputs.
            with pytest.raises(
                ort.capi.onnxruntime_pybind11_state.NotImplemented,
                match=r"implementation for Conv",
            ):
                ort.InferenceSession(path, providers=["CPUExecutionProvider"])


@pytest.mark.integration
def test_real_image_processor_outputs_feed_vision_graph():
    try:
        from transformers import AutoProcessor
    except ImportError:
        pytest.skip("Transformers with Qwen4-Exp processor support is unavailable")

    revision = "f5d08274bafd880402bd16f5e3e6c514136ec06c"
    try:
        processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen3.8-Flash-Next",
            revision=revision,
        )
    except (OSError, ValueError, KeyError) as error:
        pytest.skip(f"Pinned Qwen4-Exp processor is unavailable: {error}")

    frame = np.arange(64 * 64 * 3, dtype=np.uint8).reshape(64, 64, 3)
    batch = processor(
        text=["<|vision_start|><|image_pad|><|vision_end|>"],
        images=[frame],
        return_tensors="np",
    )
    pixels = np.asarray(batch["pixel_values"])
    grid = np.asarray(batch["image_grid_thw"])
    vision = build_from_module(
        Qwen4ExpForConditionalGeneration(_vl_config()),
        _vl_config(),
        task="qwen4-exp-vision-language",
    )["vision_encoder"]
    graph_inputs = {value.name: value for value in vision.graph.inputs}
    assert pixels.dtype == np.float32
    assert pixels.shape[-1] == 3 * 2 * 16 * 16
    assert grid.dtype == np.int64
    assert grid.shape[-1] == 3
    assert graph_inputs["pixel_values"].dtype == ir.DataType.FLOAT
    assert graph_inputs["pixel_values"].shape[-1] == pixels.shape[-1]
    assert graph_inputs["image_grid_thw"].dtype == ir.DataType.INT64


@pytest.mark.integration
def test_real_video_and_mixed_processor_outputs_have_no_export_route():
    import onnxruntime as ort

    from mobius.integrations._weight_loading import apply_weights

    try:
        from transformers import AutoProcessor
    except ImportError:
        pytest.skip("Transformers with Qwen4-Exp processor support is unavailable")

    revision = "f5d08274bafd880402bd16f5e3e6c514136ec06c"
    try:
        processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen3.8-Flash-Next",
            revision=revision,
        )
    except (OSError, ValueError, KeyError) as error:
        pytest.skip(f"Pinned Qwen4-Exp processor is unavailable: {error}")

    frame = np.arange(64 * 64 * 3, dtype=np.uint8).reshape(64, 64, 3)
    processed = processor(
        text=[
            (
                "<|vision_start|><|image_pad|><|vision_end|>"
                "<|vision_start|><|video_pad|><|vision_end|>"
            )
        ],
        images=[frame],
        videos=[[frame, frame]],
        return_tensors="np",
    )
    assert {"pixel_values_videos", "video_grid_thw"} <= processed.keys()

    config = _vl_config()
    package = build_from_module(
        Qwen4ExpForConditionalGeneration(config),
        config,
        task="qwen4-exp-vision-language",
    )
    assert config.video_token_id is None
    assert "video_features" not in {value.name for value in package["embedding"].graph.inputs}
    assert {"pixel_values_videos", "video_grid_thw"}.isdisjoint(
        value.name for value in package["vision_encoder"].graph.inputs
    )

    embedding = package["embedding"]
    apply_weights(
        embedding,
        {
            "embedding.embed_tokens.weight": torch.zeros(
                config.vocab_size,
                config.hidden_size,
            )
        },
    )
    processor_ids = np.asarray(processed["input_ids"], dtype=np.int64)
    tiny_ids = np.mod(processor_ids, config.vocab_size)
    tiny_ids[processor_ids == 248056] = config.image_token_id
    tiny_ids[processor_ids == 248057] = config.unsupported_video_token_id
    image_count = int(np.count_nonzero(tiny_ids == config.image_token_id))
    image_features = np.zeros(
        (image_count, config.hidden_size),
        dtype=np.float32,
    )
    session = OnnxModelSession(embedding)
    try:
        with pytest.raises(
            (
                ort.capi.onnxruntime_pybind11_state.Fail,
                ort.capi.onnxruntime_pybind11_state.RuntimeException,
            ),
            match=r"cannot be reshaped|input_shape_size",
        ):
            session.run(
                {
                    "input_ids": tiny_ids,
                    "image_features": image_features,
                }
            )
    finally:
        session.close()


def test_graph_exposes_exact_heterogeneous_state_abi():
    config, _module, model = _build()
    inputs = {value.name: value for value in model.graph.inputs}
    outputs = {value.name for value in model.graph.outputs}

    assert inputs["past_key_values.0.conv_state"].shape == ir.Shape(
        [ir.SymbolicDim("batch"), 32, config.linear_conv_kernel_dim]
    )
    assert inputs["past_key_values.0.recurrent_state"].shape[-3:] == (
        2,
        8,
        8,
    )
    assert inputs["past_key_values.0.recurrent_state"].dtype == ir.DataType.FLOAT
    assert inputs["past_key_values.0.ple_conv_state"].shape[-2:] == (
        config.hc_count * config.hidden_size,
        3,
    )
    assert inputs["past_key_values.0.ple_context"].shape[-1] == 2
    assert inputs["past_key_values.1.index_key"].shape[-1] == 8
    assert {
        "present_position_ids",
        "present.0.conv_state",
        "present.0.recurrent_state",
        "present.0.ple_conv_state",
        "present.0.ple_context",
        "present.1.key",
        "present.1.value",
        "present.1.index_key",
    } <= outputs
    assert model.metadata_props["mobius.cache_abi"].startswith("qwen4-exp:position_ids")
    state = json.loads(model.metadata_props["mobius.state_manifest"])
    assert state["position_state"]["axes"] == ["text"]


def test_bfloat16_graph_keeps_only_recurrent_math_and_state_in_float32():
    config, _module, model = _build(_config(dtype=ir.DataType.BFLOAT16))
    inputs = {value.name: value for value in model.graph.inputs}
    assert inputs["past_key_values.0.conv_state"].dtype == ir.DataType.BFLOAT16
    assert inputs["past_key_values.0.recurrent_state"].dtype == ir.DataType.FLOAT
    assert inputs["past_key_values.1.key"].dtype == ir.DataType.BFLOAT16

    linear_attention = next(node for node in model.graph if node.op_type == "LinearAttention")
    assert all(value.dtype == ir.DataType.FLOAT for value in linear_attention.inputs)
    assert linear_attention.outputs[1].dtype == config.mamba_ssm_dtype


def test_bfloat16_router_projects_before_float32_softmax():
    _config_value, _module, model = _build(_config(dtype=ir.DataType.BFLOAT16))
    router_matmul = next(
        node
        for node in model.graph
        if node.op_type == "MatMul"
        and node.outputs[0].name is not None
        and ".mlp.gate." in node.outputs[0].name
    )
    assert [value.dtype for value in router_matmul.inputs] == [
        ir.DataType.BFLOAT16,
        ir.DataType.BFLOAT16,
    ]
    router_cast = router_matmul.outputs[0].consumers()[0]
    assert router_cast.op_type == "Cast"
    assert router_cast.outputs[0].dtype == ir.DataType.FLOAT
    assert router_cast.outputs[0].consumers()[0].op_type == "Softmax"


def test_parameter_names_match_upstream_modules():
    _config_value, module, _model = _build()
    names = {name for name, _ in module.named_parameters()}
    assert "model.layers.0.attn_hyper_connection.input_mix_weight_down.weight" in names
    assert "model.layers.0.mlp_hyper_connection.block_inject_weight.weight" in names
    assert "model.layers.0.ple.ple_embedding.ngram_embedding.shard_0.weight" in names
    assert "model.layers.1.self_attn.indexer.index_qk_proj.weight" in names
    assert "model.layers.1.mlp.experts.gate_up_proj" in names
    assert "model.layers.1.mlp.experts.down_proj" in names
    assert "model.layers.1.mlp.shared_expert_gate.weight" in names


def test_ple_uses_one_logical_gather_independent_of_shard_count():
    _, _, two_shards = _build(_config(split_ngram_parts=2))
    _, _, four_shards = _build(_config(split_ngram_parts=4))

    def ple_ops(model):
        return [
            node.op_type
            for node in model.graph
            if node.name is not None and "/ple/ple_embedding/ngram_embedding/" in node.name
        ]

    assert ple_ops(two_shards) == ple_ops(four_shards) == ["Concat", "Gather"]


def test_ple_shard_divisibility_error_reports_remainder():
    with pytest.raises(
        ValueError,
        match=r"must be exactly divisible.*152 rows / 5 shards leaves remainder 2",
    ):
        _build(_config(split_ngram_parts=5))


def test_moe_executes_only_packed_topk_experts():
    _config_value, _module, model = _build()
    moe_nodes = [node for node in model.graph if node.op_type == "MoE"]
    assert len(moe_nodes) == _config_value.num_hidden_layers
    assert all(node.domain == "com.microsoft" for node in moe_nodes)
    assert all(
        node.attributes["k"].value == _config_value.num_experts_per_tok for node in moe_nodes
    )
    assert not any(node.op_type == "NonZero" for node in model.graph)
    assert not any(".mlp.experts.0." in name for name in model.graph.initializers)


def test_moe_graph_size_does_not_scale_with_expert_count():
    _, _, two_experts = _build(_config(num_local_experts=2))
    _, _, four_experts = _build(_config(num_local_experts=4))

    assert two_experts.graph.num_nodes() == four_experts.graph.num_nodes()
    assert sum(node.op_type == "MoE" for node in four_experts.graph) == 2


def test_preprocess_validates_packed_experts_and_joins_ple_shards():
    config = _config(split_ngram_parts=2)
    module = Qwen4ExpCausalLMModel(config)
    embedding = module.model.layers[0].ple.ple_embedding.ngram_embedding
    shard_rows = embedding.shard_0.weight.shape[0]
    embedding_width = embedding.shard_0.weight.shape[1]
    state = {
        "model.language_model.layers.0.mlp.experts.gate_up_proj": torch.arange(
            2 * 16 * 16, dtype=torch.float32
        ).reshape(2, 16, 16),
        "model.language_model.layers.0.mlp.experts.down_proj": torch.zeros(2, 16, 8),
        "model.language_model.layers.0.ple.ple_embedding.ngram_embedding.shard_0.weight": (
            torch.zeros(shard_rows, embedding_width)
        ),
        "model.language_model.layers.0.ple.ple_embedding.ngram_embedding.shard_1.weight": (
            torch.ones(shard_rows, embedding_width)
        ),
    }
    result = module.preprocess_weights(state)

    assert result["model.layers.0.mlp.experts.gate_up_proj"].shape == (2, 16, 16)
    assert result["model.layers.0.mlp.experts.down_proj"].shape == (2, 16, 8)
    assert torch.all(
        result["model.layers.0.ple.ple_embedding.ngram_embedding.shard_0.weight"] == 0
    )
    assert torch.all(
        result["model.layers.0.ple.ple_embedding.ngram_embedding.shard_1.weight"] == 1
    )
    assert not any(key.startswith("mtp.") for key in result)


def test_preprocess_fuses_exact_gguf_indexer_query_and_key_rows():
    config = _config()
    module = Qwen4ExpCausalLMModel(config)
    prefix = "model.layers.1.self_attn.indexer"
    query = torch.arange(16 * 16, dtype=torch.float32).reshape(16, 16)
    key = torch.arange(8 * 16, dtype=torch.float32).reshape(8, 16) + 1000

    result = module.preprocess_weights(
        {
            f"{prefix}.index_q_proj.weight": query,
            f"{prefix}.index_k_proj.weight": key,
        }
    )

    fused = result[f"{prefix}.index_qk_proj.weight"]
    assert fused.shape == (24, 16)
    torch.testing.assert_close(fused[:16], query)
    torch.testing.assert_close(fused[16:], key)


def test_preprocess_preserves_official_fused_hf_indexer_projection():
    module = Qwen4ExpCausalLMModel(_config())
    name = "model.layers.1.self_attn.indexer.index_qk_proj.weight"
    fused = torch.arange(24 * 16, dtype=torch.float32).reshape(24, 16)

    result = module.preprocess_weights({name: fused})

    assert result[name] is fused


def test_preprocess_fails_closed_on_incomplete_or_malformed_gguf_indexer_split():
    config = _config()
    module = Qwen4ExpCausalLMModel(config)
    prefix = "model.layers.1.self_attn.indexer"

    with pytest.raises(ValueError, match=r"missing parts \['k'\]"):
        module.preprocess_weights({f"{prefix}.index_q_proj.weight": torch.zeros(16, 16)})
    with pytest.raises(ValueError, match=r"query projection.*expected"):
        module.preprocess_weights(
            {
                f"{prefix}.index_q_proj.weight": torch.zeros(15, 16),
                f"{prefix}.index_k_proj.weight": torch.zeros(8, 16),
            }
        )


def test_multimodal_preprocess_routes_vision_projector_and_shared_embedding():
    module = Qwen4ExpForConditionalGeneration(_vl_config())
    embedding = torch.randn(32, 16)
    merger = torch.randn(128, 128)
    vision_mlp = torch.randn(64, 32)
    result = module.preprocess_weights(
        {
            "model.language_model.embed_tokens.weight": embedding,
            "model.visual.merger.linear_fc1.weight": merger,
            "model.visual.blocks.0.mlp.linear_fc1.weight": vision_mlp,
        }
    )
    assert result["decoder.model.embed_tokens.weight"] is embedding
    assert result["embedding.embed_tokens.weight"] is embedding
    assert result["vision_encoder.visual.merger.linear_fc1.weight"] is merger
    assert result["vision_encoder.visual.blocks.0.mlp.up_proj.weight"] is vision_mlp


def test_preprocess_ignores_nonexecuted_mtp_sidecar(caplog):
    result = Qwen4ExpCausalLMModel(_config()).preprocess_weights(
        {"mtp.fc_embedding.weight": torch.zeros(16, 16)}
    )
    assert result == {}
    assert "does not expose an unsupported NextN task" in caplog.text


def test_preprocess_splits_upstream_combined_ple_table():
    config = _config(split_ngram_parts=2)
    module = Qwen4ExpCausalLMModel(config)
    embedding = module.model.layers[0].ple.ple_embedding.ngram_embedding
    shard_shape = tuple(embedding.shard_0.weight.shape)
    combined = torch.arange(2 * shard_shape[0] * shard_shape[1], dtype=torch.float32).reshape(
        2 * shard_shape[0], shard_shape[1]
    )

    result = module.preprocess_weights(
        {
            "model.layers.0.ple.ple_embedding.ngram_embedding.weight": combined,
        }
    )

    assert torch.equal(
        result["model.layers.0.ple.ple_embedding.ngram_embedding.shard_0.weight"],
        combined[: shard_shape[0]],
    )
    assert torch.equal(
        result["model.layers.0.ple.ple_embedding.ngram_embedding.shard_1.weight"],
        combined[shard_shape[0] :],
    )


def test_preprocess_fails_closed_on_noncanonical_packed_or_deterministic_weights():
    config = _config()
    module = Qwen4ExpCausalLMModel(config)
    parameters = dict(module.named_parameters())
    buffer_key = "model.layers.0.ple.ple_embedding.layer_multipliers"
    canonical = torch.from_numpy(parameters[buffer_key]._const_value.numpy().copy())

    with pytest.raises(ValueError, match="does not match the pinned hash construction"):
        module.preprocess_weights({buffer_key: canonical + 2})
    with pytest.raises(ValueError, match="packed gate_up_proj has shape"):
        module.preprocess_weights(
            {"model.layers.0.mlp.experts.gate_up_proj": torch.zeros(2, 15, 16)}
        )


def test_preprocess_fails_closed_on_missing_or_unexpected_ple_shards():
    config = _config(split_ngram_parts=2)
    module = Qwen4ExpCausalLMModel(config)
    target = "model.layers.0.ple.ple_embedding.ngram_embedding"
    shard_shape = tuple(
        module.model.layers[0].ple.ple_embedding.ngram_embedding.shard_0.weight.shape
    )
    with pytest.raises(ValueError, match=r"missing shard indices \[1\]"):
        module.preprocess_weights({f"{target}.shard_0.weight": torch.zeros(shard_shape)})
    with pytest.raises(ValueError, match=r"Unexpected Qwen4-Exp PLE shard index 2"):
        module.preprocess_weights(
            {
                f"{target}.shard_0.weight": torch.zeros(shard_shape),
                f"{target}.shard_1.weight": torch.zeros(shard_shape),
                f"{target}.shard_2.weight": torch.zeros(shard_shape),
            }
        )


def _fp8_config(**overrides) -> Qwen4ExpConfig:
    return _config(
        block_quant_scheme=BlockQuantScheme(
            quant_method="fp8",
            weight_fmt="e4m3",
            weight_block_size=(128, 128),
            activation_scheme="dynamic",
        ),
        **overrides,
    )


def _source_name(target_name: str) -> str:
    if target_name.startswith("model."):
        return f"model.language_model.{target_name[len('model.') :]}"
    return target_name


def _graph_mutation_snapshot(model: ir.Model):
    nodes = tuple(
        (
            node.domain,
            node.op_type,
            node.name,
            tuple(value.name if value is not None else None for value in node.inputs),
            tuple(value.name for value in node.outputs),
            tuple(
                (name, repr(attribute)) for name, attribute in sorted(node.attributes.items())
            ),
        )
        for node in model.graph.all_nodes()
    )
    initializers = []
    for name, value in sorted(model.graph.initializers.items()):
        const_value = value.const_value
        initializers.append(
            (
                name,
                value.dtype,
                tuple(value.shape) if value.shape is not None else None,
                type(const_value).__name__ if const_value is not None else None,
                hashlib.sha256(const_value.tobytes()).hexdigest()
                if const_value is not None and not isinstance(const_value, ir.LazyTensor)
                else None,
            )
        )
    return nodes, tuple(initializers), tuple(sorted(model.metadata_props.items()))


def _quantize_blocks(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    rows, cols = weight.shape
    scales = torch.empty(
        ((rows + 127) // 128, (cols + 127) // 128),
        dtype=torch.bfloat16,
    )
    quantized = torch.empty_like(weight, dtype=torch.float8_e4m3fn)
    for block_row in range(scales.shape[0]):
        row_slice = slice(block_row * 128, min((block_row + 1) * 128, rows))
        for block_col in range(scales.shape[1]):
            col_slice = slice(block_col * 128, min((block_col + 1) * 128, cols))
            block = weight[row_slice, col_slice]
            scale = max(float(block.abs().max()) / 400.0, 2**-20)
            scales[block_row, block_col] = scale
            quantized[row_slice, col_slice] = (block / scale).to(torch.float8_e4m3fn)
    return quantized, scales


def _independent_dequantize_blocks(weight: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Reference reconstruction intentionally independent of production code."""
    result = torch.empty(weight.shape, dtype=torch.float32)
    for block_row in range(scales.shape[0]):
        row_slice = slice(block_row * 128, min((block_row + 1) * 128, weight.shape[0]))
        for block_col in range(scales.shape[1]):
            col_slice = slice(
                block_col * 128,
                min((block_col + 1) * 128, weight.shape[1]),
            )
            reconstructed = weight[row_slice, col_slice].to(torch.bfloat16) * scales[
                block_row, block_col
            ].to(torch.bfloat16)
            result[row_slice, col_slice] = reconstructed.to(torch.float32)
    return result


def _independent_tile_major_qdq(weight: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Execute the documented QDQ tile transform without production helpers."""
    rows, cols = weight.shape
    block_rows, block_cols = scales.shape
    padded = torch.zeros(
        (block_rows * 128, block_cols * 128),
        dtype=weight.dtype,
    )
    padded[:rows, :cols] = weight
    flat_tiles = (
        padded.reshape(block_rows, 128, block_cols, 128)
        .permute(0, 2, 1, 3)
        .reshape(block_rows * block_cols, 128 * 128)
    )
    dequantized = flat_tiles.to(torch.bfloat16) * scales.reshape(-1, 1).to(torch.bfloat16)
    restored = (
        dequantized.reshape(block_rows, block_cols, 128, 128)
        .permute(0, 2, 1, 3)
        .reshape(block_rows * 128, block_cols * 128)
    )
    return restored[:rows, :cols].to(torch.float32)


def _reduced_fp8_checkpoint(
    module: Qwen4ExpCausalLMModel,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    torch.manual_seed(41)
    source: dict[str, torch.Tensor] = {}
    dense_targets: dict[str, torch.Tensor] = {}
    for target_name, parameter in module.named_parameters():
        source_name = _source_name(target_name)
        if parameter._const_value is not None:
            array = parameter._const_value.numpy().copy()
            source[source_name] = (
                torch.from_numpy(array.view(np.uint16)).view(torch.bfloat16)
                if parameter._const_value.dtype == ir.DataType.BFLOAT16
                else torch.from_numpy(array)
            )
            continue
        shape = tuple(int(dim) for dim in parameter.shape)
        value = torch.randn(shape, dtype=torch.float32) * 0.02
        if target_name.endswith(".mlp.experts.gate_up_proj"):
            prefix = source_name[: -len("experts.gate_up_proj")]
            assert module.config.moe_intermediate_size is not None
            split = module.config.moe_intermediate_size
            reconstructed = torch.empty_like(value)
            for expert_index in range(value.shape[0]):
                for projection_name, projection in (
                    ("gate_proj", value[expert_index, :split]),
                    ("up_proj", value[expert_index, split:]),
                ):
                    quantized, scale = _quantize_blocks(projection)
                    weight_name = f"{prefix}experts.{expert_index}.{projection_name}.weight"
                    source[weight_name] = quantized
                    source[weight_name[: -len(".weight")] + ".weight_scale_inv"] = scale
                    row = 0 if projection_name == "gate_proj" else split
                    reconstructed[
                        expert_index,
                        row : row + projection.shape[0],
                    ] = _independent_dequantize_blocks(quantized, scale)
            dense_targets[target_name] = reconstructed
        elif target_name.endswith(".mlp.experts.down_proj"):
            prefix = source_name[: -len("experts.down_proj")]
            reconstructed = torch.empty_like(value)
            for expert_index in range(value.shape[0]):
                quantized, scale = _quantize_blocks(value[expert_index])
                weight_name = f"{prefix}experts.{expert_index}.down_proj.weight"
                source[weight_name] = quantized
                source[weight_name[: -len(".weight")] + ".weight_scale_inv"] = scale
                reconstructed[expert_index] = _independent_dequantize_blocks(
                    quantized,
                    scale,
                )
            dense_targets[target_name] = reconstructed
        elif ".mlp.experts." in target_name and target_name.endswith(".weight"):
            quantized, scale = _quantize_blocks(value)
            source[source_name] = quantized
            source[source_name[: -len(".weight")] + ".weight_scale_inv"] = scale
            dense_targets[target_name] = _independent_dequantize_blocks(quantized, scale)
        elif ".ple.ple_embedding.ngram_embedding.shard_" in target_name:
            stored = value.to(torch.float8_e4m3fn)
            source[source_name] = stored
            prefix, _suffix = source_name.split(".shard_", 1)
            source[f"{prefix}.weight_scale"] = torch.tensor(
                [_PINNED_PLE_WEIGHT_SCALE],
                dtype=torch.bfloat16,
            )
            dense_targets[target_name] = (
                stored.to(torch.bfloat16)
                * torch.tensor(_PINNED_PLE_WEIGHT_SCALE, dtype=torch.bfloat16)
            ).to(torch.float32)
        else:
            stored = value.to(torch.bfloat16)
            source[source_name] = stored
            dense_targets[target_name] = stored.to(torch.float32)
    source["model.visual.blocks.0.dummy.weight"] = torch.ones(1, dtype=torch.bfloat16)
    source["mtp.layers.0.dummy.weight"] = torch.ones(1, dtype=torch.bfloat16)
    return source, dense_targets


def test_fp8_streaming_plan_prefers_composite_keys_and_classifies_sidecars():
    config = _fp8_config()
    module = Qwen4ExpCausalLMModel(config)
    _config_value, _built_module, model = _build(config)
    source, _dense = _reduced_fp8_checkpoint(module)
    fallback = "model.layers.0.linear_attn.in_proj_qkv.weight"
    preferred = _source_name(fallback)
    source[fallback] = source[preferred]
    index = {
        name: ("shard.safetensors", list(tensor.shape), str(tensor.dtype))
        for name, tensor in source.items()
    }
    index = {
        name: (
            path,
            shape,
            {
                "torch.bfloat16": "BF16",
                "torch.float32": "F32",
                "torch.float8_e4m3fn": "F8_E4M3",
                "torch.int64": "I64",
            }[dtype],
        )
        for name, (path, shape, dtype) in index.items()
    }

    plan = module.build_fp8_streaming_plan(index, model.graph.initializers)

    assert plan.targets[fallback].source_name == preferred
    assert plan.ignored[fallback].startswith("lower-priority alias")
    assert any(reason.startswith("multimodal component") for reason in plan.ignored.values())
    assert any(reason.startswith("MTP sidecar") for reason in plan.ignored.values())
    assert plan.report["native_fp8_reason"]
    assert plan.report["multimodal_package_complete"] is False
    assert plan.report["mtp_exported"] is False
    assert plan.report["excluded_tensors"]["mtp"]["count"] == 1
    assert plan.report["excluded_tensors"]["visual"]["count"] == 1


def test_fp8_streaming_dense_fallback_matches_independent_reconstruction(tmp_path):
    config = _fp8_config()
    module = Qwen4ExpCausalLMModel(config)
    model = build_from_module(module, config, task="qwen4-exp-text-generation")["model"]
    source, dense_targets = _reduced_fp8_checkpoint(module)
    safetensors.torch.save_file(source, str(tmp_path / "model.safetensors"))

    report = stream_preprocessed_safetensors_to_model(
        model,
        str(tmp_path),
        module.build_fp8_streaming_plan,
        revision="reduced-independent-fixture",
    )

    reference_module = Qwen4ExpCausalLMModel(config)
    reference = build_from_module(reference_module, config, task="qwen4-exp-text-generation")[
        "model"
    ]
    apply_weights(reference, dense_targets)
    feeds = _initial_states() | {
        "input_ids": np.array([[2, 3, 4]], dtype=np.int64),
        "attention_mask": np.ones((1, 3), dtype=np.int64),
        "position_ids": np.arange(3, dtype=np.int64)[None],
    }
    streamed_session = OnnxModelSession(model)
    reference_session = OnnxModelSession(reference)
    try:
        streamed = streamed_session.run(feeds)["logits"]
        expected = reference_session.run(feeds)["logits"]
    finally:
        streamed_session.close()
        reference_session.close()

    np.testing.assert_allclose(streamed, expected, rtol=1e-5, atol=1e-6)
    assert report["native_fp8"] is False
    assert report["output_weight_format"] == "dense"
    assert report["scaled_fp8_tensors"] > 0
    assert report["scalar_scaled_fp8_tensors"] == config.split_ngram_parts
    assert report["checkpoint_scalar_scaled_fp8_tensors"] == config.split_ngram_parts
    assert report["mtp_exported"] is False
    assert report["largest_source_tensor_bytes"] > 0
    assert (
        report["largest_reconstruction_working_set_bytes"]
        > report["largest_source_tensor_bytes"]
    )


@pytest.mark.parametrize(
    ("dtype", "torch_dtype"),
    [
        (ir.DataType.FLOAT16, torch.float16),
        (ir.DataType.FLOAT, torch.float32),
    ],
)
def test_fp8_dense_fallback_reconstructs_bf16_before_output_cast(
    tmp_path,
    dtype,
    torch_dtype,
):
    config = _fp8_config(dtype=dtype)
    module = Qwen4ExpCausalLMModel(config)
    model = build_from_module(
        module,
        config,
        task="qwen4-exp-text-generation",
    )["model"]
    source, _dense_targets = _reduced_fp8_checkpoint(module)
    safetensors.torch.save_file(source, str(tmp_path / "model.safetensors"))

    stream_preprocessed_safetensors_to_model(
        model,
        str(tmp_path),
        module.build_fp8_streaming_plan,
    )

    target_name = "model.layers.0.ple.ple_embedding.ngram_embedding.shard_0.weight"
    source_name = _source_name(target_name)
    scale_name = source_name.split(".shard_", 1)[0] + ".weight_scale"
    expected = (
        source[source_name].to(torch.bfloat16) * source[scale_name].to(torch.bfloat16)
    ).to(torch_dtype)
    actual = torch.from_numpy(model.graph.initializers[target_name].const_value.numpy().copy())
    assert torch.equal(actual, expected)


def test_fp8_qdq_tile_recipe_matches_dense_reconstruction():
    torch.manual_seed(7)
    weight = (torch.randn(129, 257) * 0.02).to(torch.float32)
    quantized, scales = _quantize_blocks(weight)

    np.testing.assert_array_equal(
        _independent_tile_major_qdq(quantized, scales).numpy(),
        _independent_dequantize_blocks(quantized, scales).numpy(),
    )


def test_fp8_qdq_source_recipe_matches_all_dense_fallback_targets():
    config = _fp8_config()
    module = Qwen4ExpCausalLMModel(config)
    source, dense_targets = _reduced_fp8_checkpoint(module)

    for target_name, expected in dense_targets.items():
        source_name = _source_name(target_name)
        if target_name.endswith(".mlp.experts.gate_up_proj"):
            prefix = source_name[: -len("experts.gate_up_proj")]
            experts = []
            for expert_index in range(config.num_local_experts or 0):
                projections = []
                for projection_name in ("gate_proj", "up_proj"):
                    weight_name = f"{prefix}experts.{expert_index}.{projection_name}.weight"
                    scale_name = weight_name[: -len(".weight")] + ".weight_scale_inv"
                    projections.append(
                        _independent_tile_major_qdq(
                            source[weight_name],
                            source[scale_name],
                        )
                    )
                experts.append(torch.cat(projections, dim=0))
            actual = torch.stack(experts)
        elif target_name.endswith(".mlp.experts.down_proj"):
            prefix = source_name[: -len("experts.down_proj")]
            experts = []
            for expert_index in range(config.num_local_experts or 0):
                weight_name = f"{prefix}experts.{expert_index}.down_proj.weight"
                scale_name = weight_name[: -len(".weight")] + ".weight_scale_inv"
                experts.append(
                    _independent_tile_major_qdq(
                        source[weight_name],
                        source[scale_name],
                    )
                )
            actual = torch.stack(experts)
        elif ".ple.ple_embedding.ngram_embedding.shard_" in target_name:
            scale_name = source_name.split(".shard_", 1)[0] + ".weight_scale"
            actual = (
                source[source_name].to(torch.bfloat16) * source[scale_name].to(torch.bfloat16)
            ).to(torch.float32)
        else:
            continue
        np.testing.assert_array_equal(actual.numpy(), expected.numpy())


def test_fp8_qdq_preserves_storage_and_roundtrips_multishard(tmp_path):
    config = _fp8_config()
    module = Qwen4ExpCausalLMModel(config)
    package = build_from_module(
        module,
        config,
        task="qwen4-exp-text-generation",
    )
    source, _dense_targets = _reduced_fp8_checkpoint(module)
    names = sorted(source)
    weight_map = {}
    for shard_index, shard_names in enumerate((names[::2], names[1::2]), start=1):
        filename = f"model-{shard_index:05d}-of-00002.safetensors"
        safetensors.torch.save_file(
            {name: source[name] for name in shard_names},
            str(tmp_path / filename),
        )
        weight_map.update((name, filename) for name in shard_names)
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map})
    )

    model = package["model"]
    report = stream_qdq_safetensors_to_model(
        model,
        str(tmp_path),
        module.build_fp8_streaming_plan,
        revision="reduced-multishard-fixture",
    )
    package.weight_loading_report = report

    fp8_names = {
        name for name, tensor in source.items() if tensor.dtype == torch.float8_e4m3fn
    }
    scale_names = {
        name for name in source if name.endswith((".weight_scale", ".weight_scale_inv"))
    }
    assert fp8_names <= model.graph.initializers.keys()
    assert scale_names <= model.graph.initializers.keys()
    assert all(
        model.graph.initializers[name].dtype == ir.DataType.FLOAT8E4M3FN for name in fp8_names
    )
    assert all(
        model.graph.initializers[name].dtype == ir.DataType.BFLOAT16 for name in scale_names
    )
    assert "model.layers.0.mlp.experts.gate_up_proj" not in model.graph.initializers
    assert "model.layers.0.mlp.experts.down_proj" not in model.graph.initializers
    assert not any(
        name.startswith("model.layers.")
        and ".ple.ple_embedding.ngram_embedding.shard_" in name
        for name in model.graph.initializers
    )
    assert sum(node.op_type == "DequantizeLinear" for node in model.graph.all_nodes()) == len(
        fp8_names
    )
    assert report["output_weight_format"] == "fp8_qdq"
    assert report["storage_preserving"] is True
    assert report["native_fp8"] is False
    assert (
        report["stored_fp8_code_bytes"] + report["stored_scale_bytes"]
        < report["dense_equivalent_bytes"]
    )
    assert report["qdq_recipe"]["code_mapping"] == "bijective"
    assert report["qdq_recipe"]["source_code_tensors"] == len(fp8_names)
    assert len(report["qdq_recipe"]["canonical_code_mapping_sha256"]) == 64
    assert model.metadata_props["mobius.fp8_qdq_recipe"]

    output = tmp_path / "qdq"
    package.save(
        str(output),
        external_data="onnx",
        max_shard_size_bytes=4096,
        progress_bar=False,
    )
    output_shards = [
        *output.glob("model-*-of-*.onnx.data"),
        *output.glob("model.onnx-*-of-*.data"),
    ]
    assert len(output_shards) > 1
    onnx.checker.check_model(str(output / "model.onnx"), full_check=False)
    reloaded = ir.load(output / "model.onnx")
    for name in fp8_names:
        expected = source[name].view(torch.uint8).numpy().tobytes()
        assert reloaded.graph.initializers[name].const_value.tobytes() == expected
    for name in scale_names:
        expected = source[name].view(torch.uint16).numpy().tobytes()
        assert reloaded.graph.initializers[name].const_value.tobytes() == expected
    loaded_package = type(package).load(str(output))
    assert loaded_package.weight_loading_report["output_weight_format"] == "fp8_qdq"
    assert loaded_package.weight_loading_report["serializer_max_workers"] == 1


@pytest.mark.parametrize(
    ("dtype", "target_bytes"),
    [
        (ir.DataType.BFLOAT16, None),
        (ir.DataType.FLOAT16, 2),
        (ir.DataType.FLOAT, 4),
    ],
)
def test_fp8_qdq_reports_only_real_source_cast_overlap(
    tmp_path,
    dtype,
    target_bytes,
):
    config = _fp8_config(dtype=dtype)
    module = Qwen4ExpCausalLMModel(config)
    model = build_from_module(
        module,
        config,
        task="qwen4-exp-text-generation",
    )["model"]
    source, _dense_targets = _reduced_fp8_checkpoint(module)
    safetensors.torch.save_file(source, str(tmp_path / "model.safetensors"))

    report = stream_qdq_safetensors_to_model(
        model,
        str(tmp_path),
        module.build_fp8_streaming_plan,
    )

    if target_bytes is None:
        assert report["largest_source_cast_overlap_bytes"] == 0
    else:
        lm_head = source["lm_head.weight"]
        minimum_overlap = lm_head.numel() * (2 + target_bytes)
        assert report["largest_source_cast_overlap_bytes"] >= minimum_overlap
    assert report["qdq_recipe"]["source_codes_preserved"] is True


def test_fp8_qdq_rejects_logical_shape_before_graph_mutation(tmp_path):
    config = _fp8_config()
    module = Qwen4ExpCausalLMModel(config)
    model = build_from_module(
        module,
        config,
        task="qwen4-exp-text-generation",
    )["model"]
    source, _dense_targets = _reduced_fp8_checkpoint(module)
    source_name = (
        "model.language_model.layers.0.ple.ple_embedding.ngram_embedding.shard_0.weight"
    )
    original = source[source_name]
    source[source_name] = torch.cat(
        [original, torch.zeros((1, original.shape[1]), dtype=original.dtype)],
        dim=0,
    )
    safetensors.torch.save_file(source, str(tmp_path / "model.safetensors"))
    snapshot_before = _graph_mutation_snapshot(model)

    with pytest.raises(
        ValueError,
        match=rf"QDQ source '{re.escape(source_name)}'.*target.*expects",
    ):
        stream_qdq_safetensors_to_model(
            model,
            str(tmp_path),
            module.build_fp8_streaming_plan,
        )

    assert _graph_mutation_snapshot(model) == snapshot_before


def test_fp8_qdq_rejects_packed_expert_rows_before_graph_mutation(tmp_path):
    config = _fp8_config()
    module = Qwen4ExpCausalLMModel(config)
    model = build_from_module(
        module,
        config,
        task="qwen4-exp-text-generation",
    )["model"]
    source, _dense_targets = _reduced_fp8_checkpoint(module)
    source_name = "model.language_model.layers.0.mlp.experts.0.gate_proj.weight"
    original = source[source_name]
    malformed = torch.cat(
        [original, torch.zeros((1, original.shape[1]), dtype=original.dtype)],
        dim=0,
    )
    source[source_name] = malformed
    scale_name = source_name[: -len(".weight")] + ".weight_scale_inv"
    source[scale_name] = torch.ones(
        (
            (malformed.shape[0] + 127) // 128,
            (malformed.shape[1] + 127) // 128,
        ),
        dtype=torch.bfloat16,
    )
    safetensors.torch.save_file(source, str(tmp_path / "model.safetensors"))
    snapshot_before = _graph_mutation_snapshot(model)

    with pytest.raises(ValueError, match="source row sum"):
        stream_qdq_safetensors_to_model(
            model,
            str(tmp_path),
            module.build_fp8_streaming_plan,
        )

    assert _graph_mutation_snapshot(model) == snapshot_before


def test_fp8_qdq_rejects_changed_constant_before_graph_mutation(tmp_path):
    config = _fp8_config()
    module = Qwen4ExpCausalLMModel(config)
    model = build_from_module(
        module,
        config,
        task="qwen4-exp-text-generation",
    )["model"]
    source, _dense_targets = _reduced_fp8_checkpoint(module)
    constant_name = "model.language_model.layers.0.ple.ple_embedding.layer_multipliers"
    source[constant_name] = source[constant_name] + 2
    safetensors.torch.save_file(source, str(tmp_path / "model.safetensors"))
    snapshot_before = _graph_mutation_snapshot(model)

    with pytest.raises(ValueError, match="does not match the graph constant"):
        stream_qdq_safetensors_to_model(
            model,
            str(tmp_path),
            module.build_fp8_streaming_plan,
        )

    assert _graph_mutation_snapshot(model) == snapshot_before


def test_fp8_qdq_rejects_unclassified_source_before_graph_mutation(tmp_path):
    config = _fp8_config()
    module = Qwen4ExpCausalLMModel(config)
    model = build_from_module(
        module,
        config,
        task="qwen4-exp-text-generation",
    )["model"]
    source, _dense_targets = _reduced_fp8_checkpoint(module)
    source["model.language_model.unclassified.weight"] = torch.ones(
        (1, 1),
        dtype=torch.bfloat16,
    )
    safetensors.torch.save_file(source, str(tmp_path / "model.safetensors"))
    snapshot_before = _graph_mutation_snapshot(model)

    with pytest.raises(ValueError, match="unclassified by the QDQ plan"):
        stream_qdq_safetensors_to_model(
            model,
            str(tmp_path),
            module.build_fp8_streaming_plan,
        )

    assert _graph_mutation_snapshot(model) == snapshot_before


def test_fp8_dense_rejects_unclassified_source_before_graph_mutation(tmp_path):
    config = _fp8_config()
    module = Qwen4ExpCausalLMModel(config)
    model = build_from_module(
        module,
        config,
        task="qwen4-exp-text-generation",
    )["model"]
    source, _dense_targets = _reduced_fp8_checkpoint(module)
    source["model.language_model.unclassified.weight"] = torch.ones(
        (1, 1),
        dtype=torch.bfloat16,
    )
    safetensors.torch.save_file(source, str(tmp_path / "model.safetensors"))
    snapshot_before = _graph_mutation_snapshot(model)

    with pytest.raises(ValueError, match="unclassified by the streaming plan"):
        stream_preprocessed_safetensors_to_model(
            model,
            str(tmp_path),
            module.build_fp8_streaming_plan,
        )

    assert _graph_mutation_snapshot(model) == snapshot_before


def test_fp8_streaming_plan_rejects_unscaled_projection():
    config = _fp8_config()
    module = Qwen4ExpCausalLMModel(config)
    model = build_from_module(module, config, task="qwen4-exp-text-generation")["model"]
    source, _dense = _reduced_fp8_checkpoint(module)
    scale_name = "model.language_model.layers.0.mlp.experts.0.gate_proj.weight_scale_inv"
    del source[scale_name]
    index = {
        name: (
            "shard.safetensors",
            list(tensor.shape),
            "F8_E4M3"
            if tensor.dtype == torch.float8_e4m3fn
            else "BF16"
            if tensor.dtype == torch.bfloat16
            else "I64",
        )
        for name, tensor in source.items()
    }

    with pytest.raises(ValueError, match="has no scale"):
        module.build_fp8_streaming_plan(index, model.graph.initializers)


def test_fp8_streaming_plan_requires_ple_scalar():
    config = _fp8_config()
    module = Qwen4ExpCausalLMModel(config)
    model = build_from_module(module, config, task="qwen4-exp-text-generation")["model"]
    source, _dense = _reduced_fp8_checkpoint(module)
    scale_name = "model.language_model.layers.0.ple.ple_embedding.ngram_embedding.weight_scale"
    del source[scale_name]
    index = {
        name: (
            "shard.safetensors",
            list(tensor.shape),
            "F8_E4M3"
            if tensor.dtype == torch.float8_e4m3fn
            else "BF16"
            if tensor.dtype == torch.bfloat16
            else "I64",
        )
        for name, tensor in source.items()
    }

    with pytest.raises(ValueError, match="missing shared scalar"):
        module.build_fp8_streaming_plan(index, model.graph.initializers)


def test_fp8_streaming_plan_rejects_per_shard_ple_scale():
    config = _fp8_config()
    module = Qwen4ExpCausalLMModel(config)
    model = build_from_module(module, config, task="qwen4-exp-text-generation")["model"]
    source, _dense = _reduced_fp8_checkpoint(module)
    weight_name = (
        "model.language_model.layers.0.ple.ple_embedding.ngram_embedding.shard_0.weight"
    )
    weight = source[weight_name]
    source[weight_name[: -len(".weight")] + ".weight_scale_inv"] = torch.ones(
        (
            (weight.shape[0] + 127) // 128,
            (weight.shape[1] + 127) // 128,
        ),
        dtype=torch.bfloat16,
    )
    index = {
        name: (
            "shard.safetensors",
            list(tensor.shape),
            "F8_E4M3"
            if tensor.dtype == torch.float8_e4m3fn
            else "BF16"
            if tensor.dtype == torch.bfloat16
            else "I64",
        )
        for name, tensor in source.items()
    }

    with pytest.raises(ValueError, match="forbidden per-shard scale"):
        module.build_fp8_streaming_plan(index, model.graph.initializers)


def test_fp8_streaming_plan_rejects_e5m2_ple_storage():
    config = _fp8_config()
    module = Qwen4ExpCausalLMModel(config)
    model = build_from_module(module, config, task="qwen4-exp-text-generation")["model"]
    source, _dense = _reduced_fp8_checkpoint(module)
    weight_name = (
        "model.language_model.layers.0.ple.ple_embedding.ngram_embedding.shard_0.weight"
    )
    index = {
        name: (
            "shard.safetensors",
            list(tensor.shape),
            "F8_E4M3"
            if tensor.dtype == torch.float8_e4m3fn
            else "BF16"
            if tensor.dtype == torch.bfloat16
            else "I64",
        )
        for name, tensor in source.items()
    }
    path, shape, _dtype = index[weight_name]
    index[weight_name] = (path, shape, "F8_E5M2")

    with pytest.raises(ValueError, match="pinned checkpoint requires F8_E4M3"):
        module.build_fp8_streaming_plan(index, model.graph.initializers)


@pytest.mark.parametrize(
    "stream",
    [stream_preprocessed_safetensors_to_model, stream_qdq_safetensors_to_model],
)
def test_fp8_streaming_rejects_changed_ple_scalar(tmp_path, stream):
    config = _fp8_config()
    module = Qwen4ExpCausalLMModel(config)
    model = build_from_module(module, config, task="qwen4-exp-text-generation")["model"]
    source, _dense = _reduced_fp8_checkpoint(module)
    scale_name = "model.language_model.layers.0.ple.ple_embedding.ngram_embedding.weight_scale"
    source[scale_name] = torch.tensor([0.25], dtype=torch.bfloat16)
    safetensors.torch.save_file(source, str(tmp_path / "model.safetensors"))

    with pytest.raises(ValueError, match="expected"):
        stream(
            model,
            str(tmp_path),
            module.build_fp8_streaming_plan,
        )


def test_fp8_streaming_plan_validates_ignored_mtp_grid():
    config = _fp8_config()
    module = Qwen4ExpCausalLMModel(config)
    model = build_from_module(module, config, task="qwen4-exp-text-generation")["model"]
    source, _dense = _reduced_fp8_checkpoint(module)
    source["mtp.layers.0.dummy.weight"] = torch.ones((129, 129), dtype=torch.float32).to(
        torch.float8_e4m3fn
    )
    source["mtp.layers.0.dummy.weight_scale_inv"] = torch.ones((1, 1), dtype=torch.bfloat16)
    index = {
        name: (
            "shard.safetensors",
            list(tensor.shape),
            "F8_E4M3"
            if tensor.dtype == torch.float8_e4m3fn
            else "BF16"
            if tensor.dtype == torch.bfloat16
            else "I64",
        )
        for name, tensor in source.items()
    }

    with pytest.raises(ValueError, match="strict 128x128 blocks"):
        module.build_fp8_streaming_plan(index, model.graph.initializers)


def test_fp8_streaming_plan_rejects_orphan_scale():
    config = _fp8_config()
    module = Qwen4ExpCausalLMModel(config)
    model = build_from_module(module, config, task="qwen4-exp-text-generation")["model"]
    source, _dense = _reduced_fp8_checkpoint(module)
    source["model.language_model.orphan.weight_scale_inv"] = torch.ones(
        (1, 1), dtype=torch.bfloat16
    )
    index = {
        name: (
            "shard.safetensors",
            list(tensor.shape),
            "F8_E4M3"
            if tensor.dtype == torch.float8_e4m3fn
            else "BF16"
            if tensor.dtype == torch.bfloat16
            else "I64",
        )
        for name, tensor in source.items()
    }

    with pytest.raises(ValueError, match="orphan inverse scale"):
        module.build_fp8_streaming_plan(index, model.graph.initializers)


def _initial_states() -> dict[str, np.ndarray]:
    return {
        "past_position_ids": np.zeros((1, 0), dtype=np.int64),
        "past_key_values.0.conv_state": np.zeros((1, 32, 2), dtype=np.float32),
        "past_key_values.0.recurrent_state": np.zeros((1, 2, 8, 8), dtype=np.float32),
        "past_key_values.0.ple_conv_state": np.zeros((1, 32, 3), dtype=np.float32),
        # The graph must replace zero-initialized context with eos_token_id
        # when past_position_ids has zero length.
        "past_key_values.0.ple_context": np.zeros((1, 2), dtype=np.int64),
        "past_key_values.1.key": np.zeros((1, 1, 0, 8), dtype=np.float32),
        "past_key_values.1.value": np.zeros((1, 1, 0, 8), dtype=np.float32),
        "past_key_values.1.index_key": np.zeros((1, 0, 8), dtype=np.float32),
    }


def test_random_weight_prefill_matches_token_by_token_decode():
    _config_value, _module, model = _build()
    rng = np.random.default_rng(0)
    for value in model.graph.initializers.values():
        if value.const_value is None:
            value.const_value = ir.tensor(
                rng.normal(0.0, 0.02, [int(dim) for dim in value.shape]).astype(np.float32)
            )

    input_ids = np.array([[2, 3, 4, 5]], dtype=np.int64)
    session = OnnxModelSession(model)
    try:
        full = session.run(
            _initial_states()
            | {
                "input_ids": input_ids,
                "attention_mask": np.ones((1, 4), dtype=np.int64),
                "position_ids": np.arange(4, dtype=np.int64)[None],
            }
        )["logits"]

        states = _initial_states()
        decode_logits = []
        for token_index in range(input_ids.shape[1]):
            outputs = session.run(
                states
                | {
                    "input_ids": input_ids[:, token_index : token_index + 1],
                    "attention_mask": np.ones((1, token_index + 1), dtype=np.int64),
                    "position_ids": np.array([[token_index]], dtype=np.int64),
                }
            )
            decode_logits.append(outputs["logits"])
            states = {
                "past_position_ids": outputs["present_position_ids"],
                "past_key_values.0.conv_state": outputs["present.0.conv_state"],
                "past_key_values.0.recurrent_state": outputs["present.0.recurrent_state"],
                "past_key_values.0.ple_conv_state": outputs["present.0.ple_conv_state"],
                "past_key_values.0.ple_context": outputs["present.0.ple_context"],
                "past_key_values.1.key": outputs["present.1.key"],
                "past_key_values.1.value": outputs["present.1.value"],
                "past_key_values.1.index_key": outputs["present.1.index_key"],
            }
    finally:
        session.close()

    np.testing.assert_allclose(
        np.concatenate(decode_logits, axis=1),
        full,
        rtol=1e-5,
        atol=1e-6,
    )


def test_multimodal_decoder_matches_text_route_with_identical_positions():
    text_config, _text_module, text_model = _build()
    vl_config = _vl_config()
    vl_decoder = Qwen4ExpVLDecoderModel(vl_config)
    vl_model = Qwen4ExpVisionLanguageTask._build_decoder(vl_decoder, vl_config)

    rng = np.random.default_rng(7)
    values: dict[str, np.ndarray] = {}
    for initializer in text_model.graph.initializers.values():
        if initializer.const_value is None:
            value = rng.normal(0.0, 0.02, [int(dim) for dim in initializer.shape]).astype(
                np.float32
            )
            initializer.const_value = ir.tensor(value)
            values[initializer.name] = value
    for initializer in vl_model.graph.initializers.values():
        if initializer.const_value is None:
            initializer.const_value = ir.tensor(values[initializer.name])

    input_ids = np.array([[2, 3, 4, 5]], dtype=np.int64)
    attention_mask = np.ones((1, 4), dtype=np.int64)
    text_positions = np.arange(4, dtype=np.int64)[None]
    embedding_weight = values["model.embed_tokens.weight"]
    inputs_embeds = embedding_weight[input_ids]
    multimodal_positions = np.broadcast_to(
        text_positions[None],
        (4, 1, 4),
    ).copy()

    text_session = OnnxModelSession(text_model)
    vl_session = OnnxModelSession(vl_model)
    try:
        text_logits = text_session.run(
            _initial_states()
            | {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": text_positions,
            }
        )["logits"]
        vl_states = _initial_states()
        vl_states["past_position_ids"] = np.zeros((4, 1, 0), dtype=np.int64)
        vl_outputs = vl_session.run(
            vl_states
            | {
                "inputs_embeds": inputs_embeds,
                "ple_input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": multimodal_positions,
            }
        )
        vl_logits = vl_outputs["logits"]
        alternate_text_positions = multimodal_positions.copy()
        alternate_text_positions[0] += 97
        alternate_logits = vl_session.run(
            vl_states
            | {
                "inputs_embeds": inputs_embeds,
                "ple_input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": alternate_text_positions,
            }
        )["logits"]

        decode_states = {
            "past_position_ids": vl_outputs["present_position_ids"],
            "past_key_values.0.conv_state": vl_outputs["present.0.conv_state"],
            "past_key_values.0.recurrent_state": vl_outputs["present.0.recurrent_state"],
            "past_key_values.0.ple_conv_state": vl_outputs["present.0.ple_conv_state"],
            "past_key_values.0.ple_context": vl_outputs["present.0.ple_context"],
            "past_key_values.1.key": vl_outputs["present.1.key"],
            "past_key_values.1.value": vl_outputs["present.1.value"],
            "past_key_values.1.index_key": vl_outputs["present.1.index_key"],
        }
        decode_positions = np.full((4, 1, 1), 4, dtype=np.int64)
        decode_inputs = {
            "inputs_embeds": embedding_weight[np.array([[6]], dtype=np.int64)],
            "ple_input_ids": np.array([[6]], dtype=np.int64),
            "attention_mask": np.ones((1, 5), dtype=np.int64),
            "position_ids": decode_positions,
        }
        decode_logits = vl_session.run(decode_states | decode_inputs)["logits"]
        alternate_decode_states = dict(decode_states)
        alternate_decode_states["past_position_ids"] = decode_states[
            "past_position_ids"
        ].copy()
        alternate_decode_states["past_position_ids"][0] += 97
        alternate_decode_inputs = dict(decode_inputs)
        alternate_decode_inputs["position_ids"] = decode_positions.copy()
        alternate_decode_inputs["position_ids"][0] += 97
        alternate_decode_logits = vl_session.run(
            alternate_decode_states | alternate_decode_inputs
        )["logits"]
    finally:
        text_session.close()
        vl_session.close()

    np.testing.assert_allclose(vl_logits, text_logits, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(alternate_logits, vl_logits, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(alternate_decode_logits, decode_logits, rtol=1e-5, atol=1e-6)
    assert text_config.hidden_size == vl_config.hidden_size


def test_left_padding_matches_unpadded_prefill_and_following_decode():
    _config_value, _module, model = _build()
    rng = np.random.default_rng(1)
    for value in model.graph.initializers.values():
        if value.const_value is None:
            value.const_value = ir.tensor(
                rng.normal(0.0, 0.02, [int(dim) for dim in value.shape]).astype(np.float32)
            )

    session = OnnxModelSession(model)
    try:
        unpadded = session.run(
            _initial_states()
            | {
                "input_ids": np.array([[2, 3, 4]], dtype=np.int64),
                "attention_mask": np.ones((1, 3), dtype=np.int64),
                "position_ids": np.array([[0, 1, 2]], dtype=np.int64),
            }
        )
        padded = session.run(
            _initial_states()
            | {
                "input_ids": np.array([[0, 2, 3, 4]], dtype=np.int64),
                "attention_mask": np.array([[0, 1, 1, 1]], dtype=np.int64),
                "position_ids": np.array([[0, 0, 1, 2]], dtype=np.int64),
            }
        )

        def decode(outputs):
            return session.run(
                {
                    "input_ids": np.array([[5]], dtype=np.int64),
                    "attention_mask": np.ones((1, 4), dtype=np.int64),
                    "position_ids": np.array([[3]], dtype=np.int64),
                    "past_position_ids": outputs["present_position_ids"][:, -3:],
                    "past_key_values.0.conv_state": outputs["present.0.conv_state"],
                    "past_key_values.0.recurrent_state": outputs["present.0.recurrent_state"],
                    "past_key_values.0.ple_conv_state": outputs["present.0.ple_conv_state"],
                    "past_key_values.0.ple_context": outputs["present.0.ple_context"],
                    "past_key_values.1.key": outputs["present.1.key"][:, :, -3:],
                    "past_key_values.1.value": outputs["present.1.value"][:, :, -3:],
                    "past_key_values.1.index_key": outputs["present.1.index_key"][:, -3:],
                }
            )

        unpadded_decode = decode(unpadded)
        padded_decode = decode(padded)
    finally:
        session.close()

    np.testing.assert_allclose(
        padded["logits"][:, -3:],
        unpadded["logits"],
        rtol=1e-5,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        padded_decode["logits"],
        unpadded_decode["logits"],
        rtol=1e-5,
        atol=1e-6,
    )


@pytest.mark.integration
def test_reduced_random_weight_huggingface_prefill_and_decode_parity():
    """Compare the complete tiny core when the pinned Qwen4-Exp HF class is installed."""
    try:
        from transformers import (
            DynamicCache,
            Qwen4ExpForCausalLM,
            Qwen4ExpTextConfig,
        )
    except ImportError:
        pytest.skip(
            "Installed transformers does not contain the pinned experimental "
            "Qwen4-Exp implementation"
        )

    from mobius.integrations._weight_loading import apply_weights

    torch.manual_seed(0)
    hf_config = Qwen4ExpTextConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        hidden_act="silu",
        max_position_embeddings=128,
        rms_norm_eps=1e-6,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10_000.0,
            "partial_rotary_factor": 0.25,
        },
        layer_types=["linear_attention", "qwen_sparse_attention"],
        linear_conv_kernel_dim=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_num_key_heads=1,
        linear_num_value_heads=2,
        num_experts=2,
        num_experts_per_tok=1,
        moe_intermediate_size=8,
        shared_expert_intermediate_size=8,
        hc_count=2,
        hc_lowrank=4,
        indexer_n_heads=2,
        indexer_kv_heads=1,
        indexer_head_dim=8,
        indexer_budget=4,
        indexer_compress_ratio=2,
        ple_layer_ids=[1],
        ple_embed_dim=8,
        ple_conv_kernel_size=2,
        ngram_size=3,
        heads_per_ngram=2,
        ngram_vocab_size_base=31,
        make_ngram_vocab_size_divisible_by=8,
        split_ngram_parts=4,
        eos_token_id=1,
        pad_token_id=0,
        mtp_num_hidden_layers=0,
    )
    hf_model = Qwen4ExpForCausalLM(hf_config).eval()
    config = Qwen4ExpConfig.from_transformers(hf_config)
    module = Qwen4ExpCausalLMModel(config)
    model = build_from_module(module, config, task="qwen4-exp-text-generation")["model"]
    apply_weights(model, module.preprocess_weights(dict(hf_model.state_dict())))

    input_ids = np.array([[2, 3, 4, 5]], dtype=np.int64)
    with torch.no_grad():
        hf_full = hf_model(torch.from_numpy(input_ids), use_cache=False).logits.numpy()
        hf_cache = DynamicCache(config=hf_config)
        hf_decode = np.concatenate(
            [
                hf_model(
                    torch.from_numpy(input_ids[:, index : index + 1]),
                    past_key_values=hf_cache,
                    use_cache=True,
                ).logits.numpy()
                for index in range(input_ids.shape[1])
            ],
            axis=1,
        )
        masked_input_ids = np.array([[2, 7, 3, 4]], dtype=np.int64)
        masked_attention = np.array([[1, 0, 1, 1]], dtype=np.int64)
        masked_position_ids = np.array([[0, 0, 1, 2]], dtype=np.int64)
        hf_masked = hf_model(
            torch.from_numpy(masked_input_ids),
            attention_mask=torch.from_numpy(masked_attention),
            position_ids=torch.from_numpy(masked_position_ids),
            use_cache=False,
        ).logits.numpy()

    session = OnnxModelSession(model)
    try:
        onnx_full = session.run(
            _initial_states()
            | {
                "input_ids": input_ids,
                "attention_mask": np.ones((1, 4), dtype=np.int64),
                "position_ids": np.arange(4, dtype=np.int64)[None],
            }
        )["logits"]
        onnx_masked = session.run(
            _initial_states()
            | {
                "input_ids": masked_input_ids,
                "attention_mask": masked_attention,
                "position_ids": masked_position_ids,
            }
        )["logits"]
    finally:
        session.close()

    np.testing.assert_allclose(onnx_full, hf_full, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(onnx_full, hf_decode, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(onnx_masked, hf_masked, rtol=1e-3, atol=1e-3)

    # The multimodal decoder receives fused embeddings but must hash the
    # original lexical IDs through its independent PLE input. With all four
    # position channels equal (text-only), this route must remain identical.
    vl_config = _vl_config()
    vl_decoder = Qwen4ExpVLDecoderModel(vl_config)
    vl_model = Qwen4ExpVisionLanguageTask._build_decoder(vl_decoder, vl_config)
    apply_weights(
        vl_model,
        vl_decoder.preprocess_weights(dict(hf_model.state_dict())),
    )
    with torch.no_grad():
        inputs_embeds = hf_model.model.embed_tokens(torch.from_numpy(input_ids)).numpy()
    position_ids_4d = np.broadcast_to(
        np.arange(4, dtype=np.int64)[None, None, :],
        (4, 1, 4),
    ).copy()
    vl_states = _initial_states()
    vl_states["past_position_ids"] = np.zeros((4, 1, 0), dtype=np.int64)
    vl_session = OnnxModelSession(vl_model)
    try:
        vl_logits = vl_session.run(
            vl_states
            | {
                "inputs_embeds": inputs_embeds,
                "ple_input_ids": input_ids,
                "attention_mask": np.ones((1, 4), dtype=np.int64),
                "position_ids": position_ids_4d,
            }
        )["logits"]
        alternate_position_ids = position_ids_4d.copy()
        alternate_position_ids[0] += 97
        vl_alternate = vl_session.run(
            vl_states
            | {
                "inputs_embeds": inputs_embeds,
                "ple_input_ids": input_ids,
                "attention_mask": np.ones((1, 4), dtype=np.int64),
                "position_ids": alternate_position_ids,
            }
        )["logits"]
    finally:
        vl_session.close()
    with torch.no_grad():
        hf_positioned = hf_model(
            torch.from_numpy(input_ids),
            position_ids=torch.from_numpy(position_ids_4d),
            use_cache=False,
        ).logits.numpy()
        hf_alternate = hf_model(
            torch.from_numpy(input_ids),
            position_ids=torch.from_numpy(alternate_position_ids),
            use_cache=False,
        ).logits.numpy()

        def hf_cached_decode(prefill_positions, current_positions):
            cache = DynamicCache(config=hf_config)
            hf_model(
                torch.from_numpy(input_ids),
                position_ids=torch.from_numpy(prefill_positions),
                past_key_values=cache,
                use_cache=True,
            )
            return hf_model(
                torch.tensor([[6]], dtype=torch.int64),
                position_ids=torch.from_numpy(current_positions),
                past_key_values=cache,
                use_cache=True,
            ).logits.numpy()

        decode_position_ids = np.full((4, 1, 1), 4, dtype=np.int64)
        alternate_decode_position_ids = decode_position_ids.copy()
        alternate_decode_position_ids[0] += 97
        hf_decode_positioned = hf_cached_decode(position_ids_4d, decode_position_ids)
        hf_decode_alternate = hf_cached_decode(
            alternate_position_ids, alternate_decode_position_ids
        )
    np.testing.assert_allclose(vl_logits, hf_full, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(vl_alternate, vl_logits, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(hf_alternate, hf_positioned, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(hf_decode_alternate, hf_decode_positioned, rtol=1e-3, atol=1e-3)
