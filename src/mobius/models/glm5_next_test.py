# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Numerical and contract tests for the pinned GLM-5.3-Flash architecture."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import onnx_ir as ir
import pytest
import torch

from mobius import build_from_module
from mobius._configs import Glm5NextConfig, VisionConfig
from mobius._testing.ort_inference import OnnxModelSession
from mobius.integrations._weight_loading import apply_weights
from mobius.models.glm5_next import Glm5NextForConditionalGeneration

_MODEL_ID = "zai-org/GLM-5.3-Flash"
_REVISION = "03eb5366286afd40d2221b1d9c63a6dd1ba4832e"
_REPO_ROOT = Path(__file__).parents[3]


def _config(*, dtype: ir.DataType = ir.DataType.FLOAT) -> Glm5NextConfig:
    return Glm5NextConfig(
        model_type="glm5_next",
        vocab_size=40,
        hidden_size=16,
        intermediate_size=32,
        moe_intermediate_size=8,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=4,
        q_lora_rank=8,
        kv_lora_rank=8,
        qk_nope_head_dim=4,
        qk_rope_head_dim=0,
        v_head_dim=4,
        rms_norm_eps=1e-5,
        layer_types=[
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "deepseek_sparse_attention",
        ],
        mlp_layer_types=["dense", "dense", "dense", "sparse"],
        linear_num_heads=4,
        linear_head_dim=4,
        linear_conv_kernel_dim=2,
        linear_lower_bound=-5.0,
        num_local_experts=4,
        num_experts_per_tok=2,
        n_shared_experts=1,
        n_group=1,
        topk_group=1,
        norm_topk_prob=True,
        routed_scaling_factor=2.5,
        scoring_func="sigmoid",
        topk_method="noaux_tc",
        use_expert_bias=True,
        hc_mult=2,
        hc_sinkhorn_iters=2,
        hc_eps=1e-6,
        index_n_heads=2,
        index_head_dim=4,
        index_topk=4,
        index_kpool=2,
        indexer_types=["full"] * 4,
        swiglu_limit=10.0,
        pad_token_id=0,
        eos_token_id=[1],
        tie_word_embeddings=False,
        dtype=dtype,
        image_token_id=31,
        video_token_id=32,
        image_start_token_id=33,
        image_end_token_id=34,
        video_start_token_id=35,
        video_end_token_id=36,
        vision=VisionConfig(
            model_type="glm5_next_vision",
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            image_size=4,
            patch_size=2,
            norm_eps=1e-5,
            out_hidden_size=16,
            in_channels=3,
            spatial_merge_size=2,
            temporal_patch_size=2,
            hidden_act="silu",
            projector_intermediate_size=32,
            swiglu_limit=10.0,
        ),
    )


def _reduced_real_weight_config() -> Glm5NextConfig:
    return dataclasses.replace(
        _config(),
        num_hidden_layers=2,
        layer_types=["linear_attention", "deepseek_sparse_attention"],
        mlp_layer_types=["dense", "sparse"],
        indexer_types=["full", "full"],
    )


def _hf_config():
    try:
        from transformers.models.glm5_next.configuration_glm5_next import (
            Glm5NextConfig as HFGlm5NextConfig,
        )
    except ImportError:
        pytest.skip("Transformers with native glm5_next support is required")

    text = {
        "vocab_size": 40,
        "hidden_size": 16,
        "intermediate_size": 32,
        "moe_intermediate_size": 8,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "q_lora_rank": 8,
        "kv_lora_rank": 8,
        "qk_nope_head_dim": 4,
        "qk_rope_head_dim": 0,
        "v_head_dim": 4,
        "rms_norm_eps": 1e-5,
        "layer_types": [
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "deepseek_sparse_attention",
        ],
        "mlp_layer_types": ["dense", "dense", "dense", "sparse"],
        "linear_attn_config": {
            "num_heads": 4,
            "head_dim": 4,
            "short_conv_kernel_size": 2,
            "gate_lower_bound": -5.0,
            "safe_gate": True,
        },
        "n_routed_experts": 4,
        "n_shared_experts": 1,
        "num_experts_per_tok": 2,
        "n_group": 1,
        "topk_group": 1,
        "norm_topk_prob": True,
        "routed_scaling_factor": 2.5,
        "index_n_heads": 2,
        "index_head_dim": 4,
        "index_topk": 4,
        "index_kpool": 2,
        "indexer_types": ["full"] * 4,
        "hc_mult": 2,
        "hc_sinkhorn_iters": 2,
        "hc_eps": 1e-6,
        "swiglu_limit": 10.0,
        "pad_token_id": 0,
        "eos_token_id": [1],
        "tie_word_embeddings": False,
        "dtype": "float32",
    }
    vision = {
        "depth": 1,
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_heads": 4,
        "in_channels": 3,
        "image_size": 4,
        "patch_size": 2,
        "temporal_patch_size": 2,
        "spatial_merge_size": 2,
        "out_hidden_size": 16,
        "projection_intermediate_size": 32,
        "hidden_act": "silu",
        "swiglu_limit": 10.0,
        "rms_norm_eps": 1e-5,
        "attention_bias": True,
    }
    return HFGlm5NextConfig(
        text_config=text,
        vision_config=vision,
        image_token_id=31,
        video_token_id=32,
        image_start_token_id=33,
        image_end_token_id=34,
        video_start_token_id=35,
        video_end_token_id=36,
        tie_word_embeddings=False,
    )


def _empty_decoder_feeds(config: Glm5NextConfig, embeds: np.ndarray) -> dict[str, np.ndarray]:
    assert config.layer_types is not None
    assert config.linear_num_heads is not None
    assert config.linear_head_dim is not None
    assert config.qk_nope_head_dim is not None
    assert config.v_head_dim is not None
    assert config.index_head_dim is not None
    batch, sequence_length = embeds.shape[:2]
    projection = config.linear_num_heads * config.linear_head_dim
    feeds: dict[str, np.ndarray] = {
        "inputs_embeds": embeds,
        "attention_mask": np.ones((batch, sequence_length), dtype=np.int64),
        "position_ids": np.broadcast_to(
            np.arange(sequence_length, dtype=np.int64),
            (batch, sequence_length),
        ).copy(),
    }
    for layer_idx, layer_type in enumerate(config.layer_types):
        prefix = f"past_key_values.{layer_idx}"
        if layer_type == "linear_attention":
            feeds[f"{prefix}.conv_state"] = np.zeros(
                (
                    batch,
                    3 * projection,
                    config.linear_conv_kernel_dim,
                ),
                dtype=embeds.dtype,
            )
            feeds[f"{prefix}.recurrent_state"] = np.zeros(
                (
                    batch,
                    config.linear_num_heads,
                    config.linear_head_dim,
                    config.linear_head_dim,
                ),
                dtype=np.float32,
            )
        else:
            feeds[f"{prefix}.key"] = np.zeros(
                (batch, config.num_attention_heads, 0, config.qk_nope_head_dim),
                dtype=embeds.dtype,
            )
            feeds[f"{prefix}.value"] = np.zeros(
                (batch, config.num_attention_heads, 0, config.v_head_dim),
                dtype=embeds.dtype,
            )
            feeds[f"{prefix}.indexer_state"] = np.zeros(
                (batch, 0, 2 * config.index_head_dim + 1),
                dtype=embeds.dtype,
            )
    return feeds


def test_pinned_config_extracts_exact_hybrid_contract() -> None:
    text = SimpleNamespace(
        model_type="glm5_next_text",
        vocab_size=154880,
        hidden_size=4096,
        intermediate_size=12288,
        moe_intermediate_size=2048,
        num_hidden_layers=45,
        num_attention_heads=64,
        num_key_value_heads=64,
        head_dim=0,
        q_lora_rank=1536,
        kv_lora_rank=512,
        qk_nope_head_dim=256,
        qk_rope_head_dim=0,
        v_head_dim=256,
        rms_norm_eps=1e-5,
        layer_types=["linear_attention"] * 3 + ["deepseek_sparse_attention"],
        mlp_layer_types=["dense"] * 3 + ["sparse"] * 42,
        linear_attn_config={
            "num_heads": 64,
            "head_dim": 128,
            "short_conv_kernel_size": 4,
            "gate_lower_bound": -5.0,
        },
        n_routed_experts=288,
        n_shared_experts=1,
        num_experts_per_tok=8,
        n_group=1,
        topk_group=1,
        norm_topk_prob=True,
        routed_scaling_factor=2.5,
        index_n_heads=32,
        index_head_dim=128,
        index_topk=2048,
        index_kpool=4,
        indexer_types=["full"] * 45,
        hc_mult=4,
        hc_sinkhorn_iters=20,
        hc_eps=1e-6,
        swiglu_limit=10.0,
        tie_word_embeddings=False,
        dtype="bfloat16",
    )
    # Restore the full 45-entry schedule after the abbreviated source fixture.
    text.layer_types = [
        "linear_attention" if index % 4 != 3 else "deepseek_sparse_attention"
        for index in range(45)
    ]
    parent = SimpleNamespace(
        model_type="glm5_next",
        text_config=text,
        vision_config=SimpleNamespace(
            model_type="glm5_next_vision",
            depth=24,
            hidden_size=1024,
            intermediate_size=4096,
            num_heads=16,
            image_size=448,
            patch_size=14,
            temporal_patch_size=2,
            spatial_merge_size=2,
            out_hidden_size=4096,
            projection_intermediate_size=10240,
            hidden_act="silu",
            swiglu_limit=10.0,
            rms_norm_eps=1e-5,
            in_channels=3,
        ),
        image_token_id=154854,
        video_token_id=154855,
        image_start_token_id=154830,
        image_end_token_id=154831,
        video_start_token_id=154832,
        video_end_token_id=154833,
        tie_word_embeddings=False,
    )
    config = Glm5NextConfig.from_transformers(parent)

    assert config.head_dim == 256
    assert config.linear_num_heads == 64
    assert config.linear_head_dim == 128
    assert config.layer_types[3] == "deepseek_sparse_attention"
    assert config.layer_types[44] == "linear_attention"
    assert config.mlp_layer_types[:4] == ["dense", "dense", "dense", "sparse"]
    assert config.index_kpool == 4
    assert config.hc_mult == 4
    assert config.video_start_token_id == 154832
    assert config.vision is not None
    assert config.vision.projector_intermediate_size == 10240
    assert math.isclose(config.vision.swiglu_limit, 10.0)


def test_package_has_graph_derived_multimodal_and_cache_contracts() -> None:
    config = _config()
    module = Glm5NextForConditionalGeneration(config)
    package = build_from_module(module, config, task=module.default_task)

    assert set(package) == {"decoder", "vision_encoder", "embedding"}
    decoder_inputs = {value.name: value for value in package["decoder"].graph.inputs}
    decoder_outputs = {value.name for value in package["decoder"].graph.outputs}
    assert decoder_inputs["past_key_values.0.conv_state"].shape == [
        "batch",
        48,
        2,
    ]
    assert decoder_inputs["past_key_values.3.indexer_state"].shape == [
        "batch",
        "past_sequence_length",
        9,
    ]
    assert "present.0.recurrent_state" in decoder_outputs
    assert "present.3.indexer_state" in decoder_outputs
    assert "NoPE" in package["decoder"].metadata_props["mobius.cache_abi"]
    assert _REVISION in package["decoder"].metadata_props["mobius.semantic_reference_revision"]
    assert {value.name for value in package["vision_encoder"].graph.inputs} == {
        "pixel_values",
        "grid_thw",
    }
    assert {value.name for value in package["embedding"].graph.inputs} == {
        "input_ids",
        "image_features",
        "video_features",
    }
    sparse_attention_nodes = [
        node for node in package["decoder"].graph if "self_attn" in (node.name or "")
    ]
    assert not any(node.op_type == "OneHot" for node in sparse_attention_nodes)
    expert_gathers = [
        node
        for node in package["decoder"].graph
        if node.op_type == "Gather" and "/mlp/experts/" in (node.name or "")
    ]
    assert expert_gathers
    assert all(
        node.inputs[1].producer() is not None
        and node.inputs[1].producer().op_type == "Constant"
        for node in expert_gathers
    )


def test_ort_genai_metadata_fails_closed_for_heterogeneous_state(tmp_path) -> None:
    from mobius.integrations.ort_genai.auto_export import write_ort_genai_config

    config = _config()
    module = Glm5NextForConditionalGeneration(config)
    package = build_from_module(module, config, task=module.default_task)
    with pytest.raises(ValueError, match=r"cannot represent GLM-5\.3"):
        write_ort_genai_config(package, str(tmp_path))
    assert not (tmp_path / "genai_config.json").exists()


def test_reduced_real_weight_fixture_is_pinned_and_immutable() -> None:
    evidence_path = (
        _REPO_ROOT / "testdata/evidence/vision-language/glm5-next-reduced-real-weights.json"
    )
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    fixture_path = _REPO_ROOT / evidence["fixture"]["path"]

    assert evidence["status"] == "reduced-production-derived-real-weights"
    assert evidence["source"]["model_id"] == _MODEL_ID
    assert evidence["source"]["revision"] == _REVISION
    assert evidence["source"]["selected_decoder_layers"] == {"0": 0, "1": 3}
    assert len(evidence["range_reads"]) >= 80
    assert {item["dtype"] for item in evidence["range_reads"]} >= {
        "BF16",
        "F8_E4M3",
    }
    assert (
        hashlib.sha256(fixture_path.read_bytes()).hexdigest() == evidence["fixture"]["sha256"]
    )


def test_embedding_separates_shared_image_token_by_video_span() -> None:
    config = _config()
    module = Glm5NextForConditionalGeneration(config)
    embedding = build_from_module(
        module,
        config,
        task=module.default_task,
    )["embedding"]
    state = {
        name: torch.zeros(tuple(int(dim) for dim in value.shape))
        for name, value in embedding.graph.initializers.items()
        if value.const_value is None
    }
    apply_weights(embedding, state)
    input_ids = np.array(
        [
            [31, 2, 35, 31, 36],
            [3, 31, 4, 5, 6],
        ],
        dtype=np.int64,
    )
    image_features = np.stack(
        (
            np.full(16, 11.0, dtype=np.float32),
            np.full(16, 22.0, dtype=np.float32),
        )
    )
    video_features = np.full((1, 16), 33.0, dtype=np.float32)
    session = OnnxModelSession(embedding)
    try:
        actual = session.run(
            {
                "input_ids": input_ids,
                "image_features": image_features,
                "video_features": video_features,
            }
        )["inputs_embeds"]
    finally:
        session.close()

    np.testing.assert_array_equal(actual[0, 0], image_features[0])
    np.testing.assert_array_equal(actual[0, 3], video_features[0])
    np.testing.assert_array_equal(actual[1, 1], image_features[1])
    np.testing.assert_array_equal(actual[0, 1], np.zeros(16, dtype=np.float32))


def test_fp16_padding_stays_finite_across_dsa_then_kda() -> None:
    import onnxruntime as ort

    config = dataclasses.replace(
        _config(dtype=ir.DataType.FLOAT16),
        layer_types=[
            "linear_attention",
            "linear_attention",
            "deepseek_sparse_attention",
            "linear_attention",
        ],
        mlp_layer_types=["dense", "dense", "sparse", "sparse"],
        indexer_types=["full"] * 4,
    )
    module = Glm5NextForConditionalGeneration(config)
    decoder = build_from_module(module, config, task=module.default_task)["decoder"]
    generator = torch.Generator().manual_seed(19)
    weights = {}
    for name, value in decoder.graph.initializers.items():
        if value.const_value is not None:
            continue
        dtype = torch.float16 if value.dtype == ir.DataType.FLOAT16 else torch.float32
        weights[name] = (
            torch.randn(
                tuple(int(dim) for dim in value.shape),
                generator=generator,
                dtype=dtype,
            )
            * 0.02
        )
    apply_weights(decoder, weights)
    embeds = np.random.default_rng(20).normal(size=(2, 4, 16)).astype(np.float16)
    feeds = _empty_decoder_feeds(config, embeds)
    feeds["attention_mask"] = np.array(
        [[1, 1, 1, 1], [0, 0, 1, 1]],
        dtype=np.int64,
    )
    device = "cuda" if "CUDAExecutionProvider" in ort.get_available_providers() else "cpu"
    session = OnnxModelSession(decoder, device=device)
    try:
        logits = session.run(feeds)["logits"]
    finally:
        session.close()
    assert np.isfinite(logits).all()


@pytest.mark.integration
def test_tiny_hf_pipeline_prefill_and_cached_decode_match() -> None:
    try:
        from transformers.models.glm5_next.modeling_glm5_next import (
            Glm5NextForConditionalGeneration as HFGlm5Next,
        )
    except ImportError:
        pytest.skip("Transformers with native glm5_next support is required")

    hf_config = _hf_config()
    torch.manual_seed(7)
    reference = HFGlm5Next(hf_config).float().eval()
    config = Glm5NextConfig.from_transformers(hf_config)
    module = Glm5NextForConditionalGeneration(config)
    package = build_from_module(module, config, task=module.default_task)
    package.apply_weights(module.preprocess_weights(dict(reference.state_dict())))

    input_ids = torch.tensor([[2, 31, 3]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    generator = torch.Generator().manual_seed(9)
    pixel_values = torch.randn((4, 24), generator=generator)
    grid_thw = torch.tensor([[1, 2, 2]], dtype=torch.long)
    with torch.no_grad():
        reference_prefill = reference(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=grid_thw,
            use_cache=True,
        )

    vision_session = OnnxModelSession(package["vision_encoder"])
    embedding_session = OnnxModelSession(package["embedding"])
    decoder_session = OnnxModelSession(package["decoder"])
    try:
        image_features = vision_session.run(
            {
                "pixel_values": pixel_values.numpy(),
                "grid_thw": grid_thw.numpy(),
            }
        )["image_features"]
        inputs_embeds = embedding_session.run(
            {
                "input_ids": input_ids.numpy(),
                "image_features": image_features,
                "video_features": np.zeros((0, config.hidden_size), dtype=np.float32),
            }
        )["inputs_embeds"]
        prefill_feeds = _empty_decoder_feeds(config, inputs_embeds)
        prefill_outputs = decoder_session.run(prefill_feeds)
        np.testing.assert_allclose(
            prefill_outputs["logits"],
            reference_prefill.logits.detach().numpy(),
            rtol=1e-2,
            atol=5e-3,
        )

        next_token = reference_prefill.logits[:, -1].argmax(-1, keepdim=True)
        with torch.no_grad():
            reference_decode = reference(
                input_ids=next_token,
                attention_mask=torch.ones((1, 4), dtype=torch.long),
                past_key_values=reference_prefill.past_key_values,
                use_cache=True,
            ).logits
        decode_embeds = embedding_session.run(
            {
                "input_ids": next_token.numpy(),
                "image_features": np.zeros((0, config.hidden_size), dtype=np.float32),
                "video_features": np.zeros((0, config.hidden_size), dtype=np.float32),
            }
        )["inputs_embeds"]
        decode_feeds: dict[str, np.ndarray] = {
            "inputs_embeds": decode_embeds,
            "attention_mask": np.ones((1, 4), dtype=np.int64),
            "position_ids": np.array([[3]], dtype=np.int64),
        }
        assert config.layer_types is not None
        for layer_idx, layer_type in enumerate(config.layer_types):
            roles = (
                ("conv_state", "recurrent_state")
                if layer_type == "linear_attention"
                else ("key", "value", "indexer_state")
            )
            for role in roles:
                decode_feeds[f"past_key_values.{layer_idx}.{role}"] = prefill_outputs[
                    f"present.{layer_idx}.{role}"
                ]
        decode_logits = decoder_session.run(decode_feeds)["logits"]
        np.testing.assert_allclose(
            decode_logits,
            reference_decode.detach().numpy(),
            rtol=1e-2,
            atol=1e-3,
        )
        assert int(decode_logits[0, -1].argmax()) == int(reference_decode[0, -1].argmax())
    finally:
        decoder_session.close()
        embedding_session.close()
        vision_session.close()


@pytest.mark.integration
def test_pinned_processor_emits_nonzero_image_and_video_contract() -> None:
    try:
        from PIL import Image
        from transformers.models.glm5_next.processing_glm5_next import (
            Glm5NextProcessor,
        )
    except ImportError:
        pytest.skip("Transformers with native glm5_next support is required")

    processor = Glm5NextProcessor.from_pretrained(_MODEL_ID, revision=_REVISION)
    pixels = np.arange(28 * 28 * 3, dtype=np.uint8).reshape(28, 28, 3)
    image = Image.fromarray(pixels)
    image_inputs = processor(
        text=["<|begin_of_image|><|image|><|end_of_image|> describe"],
        images=[image],
        return_tensors="np",
    )
    video_inputs = processor(
        text=["<|begin_of_video|><|video|><|end_of_video|> describe"],
        videos=[[image, image]],
        videos_kwargs={"do_sample_frames": False},
        return_tensors="np",
    )

    assert image_inputs["pixel_values"].dtype == np.float32
    assert image_inputs["pixel_values"].shape[1] == 3 * 2 * 14 * 14
    assert image_inputs["image_grid_thw"].dtype == np.int64
    assert np.any(image_inputs["pixel_values"])
    assert set(np.unique(image_inputs["mm_token_type_ids"])) == {0, 1}
    assert video_inputs["pixel_values_videos"].dtype == np.float32
    assert video_inputs["video_grid_thw"].dtype == np.int64
    assert np.any(video_inputs["pixel_values_videos"])
    assert set(np.unique(video_inputs["mm_token_type_ids"])) == {0, 2}


@pytest.mark.integration
@pytest.mark.golden
@pytest.mark.generation
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_reduced_real_weight_l4_l5_golden(device: str) -> None:
    import onnxruntime as ort
    import safetensors.torch

    if device == "cuda" and "CUDAExecutionProvider" not in ort.get_available_providers():
        pytest.skip("CUDAExecutionProvider is required")

    config = _reduced_real_weight_config()
    module = Glm5NextForConditionalGeneration(config)
    package = build_from_module(module, config, task=module.default_task)
    fixture_path = (
        _REPO_ROOT / "testdata/evidence/vision-language/"
        "glm5-next-reduced-real-weights.safetensors"
    )
    package.apply_weights(
        module.preprocess_weights(safetensors.torch.load_file(str(fixture_path)))
    )
    l4 = json.loads(
        (_REPO_ROOT / "testdata/golden/vision-language/glm5-next-reduced.json").read_text(
            encoding="utf-8"
        )
    )
    l5 = json.loads(
        (
            _REPO_ROOT / "testdata/golden/vision-language/glm5-next-reduced_generation.json"
        ).read_text(encoding="utf-8")
    )

    input_ids = np.asarray([l4["input_ids"]], dtype=np.int64)
    pixel_values = np.arange(4 * 24, dtype=np.float32).reshape(4, 24) / 96.0
    grid_thw = np.array([[1, 2, 2]], dtype=np.int64)
    assert hashlib.sha256(pixel_values.tobytes()).hexdigest() == l4["pixel_values_sha256"]

    vision_session = OnnxModelSession(package["vision_encoder"], device=device)
    embedding_session = OnnxModelSession(package["embedding"], device=device)
    decoder_session = OnnxModelSession(package["decoder"], device=device)
    try:
        image_features = vision_session.run(
            {"pixel_values": pixel_values, "grid_thw": grid_thw}
        )["image_features"]
        inputs_embeds = embedding_session.run(
            {
                "input_ids": input_ids,
                "image_features": image_features,
                "video_features": np.zeros((0, config.hidden_size), dtype=np.float32),
            }
        )["inputs_embeds"]
        decoder_outputs = decoder_session.run(_empty_decoder_feeds(config, inputs_embeds))
        last_logits = decoder_outputs["logits"][0, -1].astype(np.float32)
        np.testing.assert_allclose(
            last_logits,
            np.asarray(l4["last_logits"], dtype=np.float32),
            rtol=1e-2,
            atol=5e-3,
        )
        assert np.argsort(last_logits)[-10:][::-1].tolist() == l4["top10_ids"]

        generated: list[int] = []
        next_token = np.array([[int(last_logits.argmax())]], dtype=np.int64)
        for step, expected_step_logits in enumerate(l5["step_logits"]):
            generated.append(int(next_token.item()))
            next_embeds = embedding_session.run(
                {
                    "input_ids": next_token,
                    "image_features": np.zeros((0, config.hidden_size), dtype=np.float32),
                    "video_features": np.zeros((0, config.hidden_size), dtype=np.float32),
                }
            )["inputs_embeds"]
            decode_feeds: dict[str, np.ndarray] = {
                "inputs_embeds": next_embeds,
                "attention_mask": np.ones((1, 4 + step), dtype=np.int64),
                "position_ids": np.array([[3 + step]], dtype=np.int64),
            }
            assert config.layer_types is not None
            for layer_idx, layer_type in enumerate(config.layer_types):
                roles = (
                    ("conv_state", "recurrent_state")
                    if layer_type == "linear_attention"
                    else ("key", "value", "indexer_state")
                )
                for role in roles:
                    decode_feeds[f"past_key_values.{layer_idx}.{role}"] = decoder_outputs[
                        f"present.{layer_idx}.{role}"
                    ]
            decoder_outputs = decoder_session.run(decode_feeds)
            step_logits = decoder_outputs["logits"][0, -1].astype(np.float32)
            np.testing.assert_allclose(
                step_logits,
                np.asarray(expected_step_logits, dtype=np.float32),
                rtol=1e-2,
                atol=5e-3,
            )
            next_token = np.array([[int(step_logits.argmax())]], dtype=np.int64)

        assert len(generated) == 24
        assert generated == l5["generated_tokens"]
    finally:
        decoder_session.close()
        embedding_session.close()
        vision_session.close()


@pytest.mark.integration
@pytest.mark.parametrize(
    ("onnx_dtype", "torch_dtype"),
    [
        (ir.DataType.FLOAT16, torch.float16),
        pytest.param(
            ir.DataType.BFLOAT16,
            torch.bfloat16,
            marks=pytest.mark.xfail(
                reason=(
                    "ORT 1.28 CUDA MemcpyTransformer rejects the mixed-provider "
                    "BF16 graph before execution (provider type for internal Less "
                    "node is unset); fp32/fp16 and BF16 graph construction pass"
                ),
                strict=True,
            ),
        ),
    ],
)
def test_reduced_precision_cuda_semantics_match_hf(
    onnx_dtype: ir.DataType,
    torch_dtype: torch.dtype,
) -> None:
    import onnxruntime as ort

    if "CUDAExecutionProvider" not in ort.get_available_providers():
        pytest.skip("CUDAExecutionProvider is required")
    try:
        from transformers.models.glm5_next.modeling_glm5_next import (
            Glm5NextForConditionalGeneration as HFGlm5Next,
        )
    except ImportError:
        pytest.skip("Transformers with native glm5_next support is required")

    hf_config = _hf_config()
    torch.manual_seed(11)
    reference = HFGlm5Next(hf_config).eval().to(dtype=torch_dtype)
    config = _config(dtype=onnx_dtype)
    module = Glm5NextForConditionalGeneration(config)
    package = build_from_module(
        module,
        config,
        task=module.default_task,
    )
    package.apply_weights(module.preprocess_weights(dict(reference.state_dict())))

    input_ids = torch.tensor([[2, 31, 3]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    pixel_values = torch.randn(
        (4, 24),
        generator=torch.Generator().manual_seed(12),
    )
    grid_thw = torch.tensor([[1, 2, 2]], dtype=torch.long)
    with torch.no_grad():
        expected = reference(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=grid_thw,
            use_cache=False,
        ).logits.float()

    vision_session = OnnxModelSession(package["vision_encoder"], device="cuda")
    embedding_session = OnnxModelSession(package["embedding"], device="cuda")
    decoder_session = OnnxModelSession(package["decoder"], device="cuda")
    try:
        image_features = vision_session.run(
            {
                "pixel_values": pixel_values.cpu().numpy(),
                "grid_thw": grid_thw.cpu().numpy(),
            }
        )["image_features"]
        inputs_embeds = embedding_session.run(
            {
                "input_ids": input_ids.cpu().numpy(),
                "image_features": image_features,
                "video_features": np.zeros(
                    (0, config.hidden_size),
                    dtype=image_features.dtype,
                ),
            }
        )["inputs_embeds"]
        feeds = _empty_decoder_feeds(config, inputs_embeds)
        actual = decoder_session.run(feeds)["logits"].astype(np.float32)
    finally:
        decoder_session.close()
        embedding_session.close()
        vision_session.close()

    np.testing.assert_allclose(
        actual,
        expected.cpu().numpy(),
        rtol=2e-2,
        atol=2e-2,
    )
    assert int(actual[0, -1].argmax()) == int(expected[0, -1].argmax())
