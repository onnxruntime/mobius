# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

from collections import Counter

import numpy as np
import onnx_ir as ir
import pytest
import torch

from mobius import build_from_module
from mobius._configs import MuseGlimmerConfig
from mobius._testing import create_test_builder, create_test_input
from mobius._testing.ort_inference import OnnxModelSession
from mobius.integrations._weight_loading import apply_weights
from mobius.models.muse_glimmer import (
    MuseGlimmerForConditionalGeneration,
    MuseGlimmerScaleFreeRMSNorm,
    MuseGlimmerTextCausalLMModel,
)
from mobius.tasks import MuseGlimmerVLTask


@pytest.mark.parametrize("dtype", [ir.DataType.FLOAT16, ir.DataType.BFLOAT16])
def test_scale_free_rms_norm_preserves_input_dtype(dtype):
    builder, op, _ = create_test_builder()
    hidden_states = create_test_input(builder, "hidden_states", [2, 3, 16], dtype)

    output = MuseGlimmerScaleFreeRMSNorm(16, 1e-5)(op, hidden_states)

    assert output.dtype == dtype
    rms_node = next(node for node in builder.graph if node.op_type == "RMSNormalization")
    assert rms_node.inputs[1].shape == ir.Shape([16])


def test_muse_glimmer_uses_fused_rms_normalization():
    config = MuseGlimmerConfig(
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        num_hidden_layers=4,
        vocab_size=256,
        max_position_embeddings=128,
        hidden_act="silu",
        layer_types=[
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention",
        ],
        layer_rope_theta=[500_000.0, 500_000.0, 500_000.0, 0],
        sliding_window=8,
        pad_token_id=0,
        rope_type="default",
        rope_theta=500_000.0,
        rms_norm_eps=1e-5,
        dtype=ir.DataType.BFLOAT16,
    )
    module = MuseGlimmerTextCausalLMModel(config)
    model = build_from_module(module, config, execution_provider="cuda")["model"]
    counts = Counter(node.op_type for node in model.graph)

    assert counts["RMSNormalization"] == 25
    assert counts["SkipSimplifiedLayerNormalization"] == 1
    assert counts["ReduceMean"] == 0
    assert counts["Pow"] == 0


def _hf_tiny_muse_glimmer():
    transformers = pytest.importorskip("transformers")
    if not hasattr(transformers, "MuseGlimmerForConditionalGeneration"):
        pytest.skip("Installed transformers does not include Muse Glimmer")

    text_config = transformers.MuseGlimmerTextConfig(
        attention_bias=False,
        bos_token_id=1,
        eos_token_id=2,
        final_logit_softcapping=20.0,
        head_dim=16,
        hidden_activation="silu",
        hidden_size=64,
        intermediate_size=128,
        layer_rope_theta=[500_000.0, 500_000.0, 500_000.0, 0],
        layer_types=[
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention",
        ],
        max_position_embeddings=128,
        num_attention_heads=4,
        num_hidden_layers=4,
        num_key_value_heads=2,
        output_multiplier=0.19611613513818404,
        pad_token_id=0,
        post_norm_eps=1e-8,
        qk_scale_factor=3.87,
        rms_norm_eps=1e-5,
        rope_parameters={"rope_theta": 500_000.0, "rope_type": "default"},
        sliding_window=8,
        tie_word_embeddings=False,
        vocab_size=256,
    )
    vision_config = transformers.MuseGlimmerVisionConfig(
        hidden_act="gelu",
        hidden_size=32,
        intermediate_size=64,
        layer_norm_eps=1e-5,
        layer_types=["window_attention", "full_attention"],
        max_position_embeddings=64,
        merge_size=2,
        num_attention_heads=4,
        num_hidden_layers=2,
        patch_size=2,
        patch_temporal=2,
        pos_emb_height=8,
        pos_emb_width=8,
        rope_parameters={"rope_theta": 10_000.0, "rope_type": "default"},
    )
    config = transformers.MuseGlimmerConfig(
        image_token_id=100,
        out_hidden_size=128,
        projector_hidden_act="gelu",
        projector_hidden_size=48,
        text_config=text_config,
        video_token_id=101,
        vision_config=vision_config,
    )
    torch.manual_seed(0)
    model = transformers.MuseGlimmerForConditionalGeneration(config).float().eval()
    return model, config


@pytest.mark.integration
def test_muse_glimmer_tiny_multimodal_prefill_matches_hf():
    hf_model, hf_config = _hf_tiny_muse_glimmer()
    config = MuseGlimmerConfig.from_transformers(
        hf_config.text_config,
        parent_config=hf_config,
    )
    config.dtype = ir.DataType.FLOAT
    module = MuseGlimmerForConditionalGeneration(config)
    package = MuseGlimmerVLTask().build(module, config)
    routed_weights = module.preprocess_weights(dict(hf_model.state_dict()))
    for model in package.values():
        apply_weights(model, routed_weights)

    rng = np.random.default_rng(0)
    image_grid_thw = np.array([[1, 4, 4]], dtype=np.int64)
    pixel_values = rng.standard_normal((16, 24)).astype(np.float32)
    input_ids = np.array([[5, 100, 100, 100, 100, 6]], dtype=np.int64)
    attention_mask = np.ones_like(input_ids)
    position_ids = np.arange(input_ids.shape[1], dtype=np.int64)[None, :]

    with torch.no_grad():
        hf_logits = hf_model(
            input_ids=torch.from_numpy(input_ids),
            attention_mask=torch.from_numpy(attention_mask),
            position_ids=torch.from_numpy(position_ids),
            pixel_values=torch.from_numpy(pixel_values),
            image_grid_thw=torch.from_numpy(image_grid_thw),
            use_cache=False,
        ).logits.numpy()

    vision_session = OnnxModelSession(package["vision_encoder"])
    embedding_session = OnnxModelSession(package["embedding"])
    decoder_session = OnnxModelSession(package["decoder"])
    try:
        image_features = vision_session.run(
            {
                "pixel_values": pixel_values,
                "image_grid_thw": image_grid_thw,
            }
        )["image_features"]
        inputs_embeds = embedding_session.run(
            {
                "input_ids": input_ids,
                "image_features": image_features,
            }
        )["inputs_embeds"]
        empty_decode_embeds = embedding_session.run(
            {
                "input_ids": np.array([[7]], dtype=np.int64),
                "image_features": np.empty((0, config.hidden_size), dtype=np.float32),
            }
        )["inputs_embeds"]
        assert np.isfinite(empty_decode_embeds).all()

        packed_features = np.stack(
            [
                np.full(config.hidden_size, 1.0, dtype=np.float32),
                np.full(config.hidden_size, 2.0, dtype=np.float32),
            ]
        )
        batched_embeds = embedding_session.run(
            {
                "input_ids": np.array([[100, 5], [100, 6]], dtype=np.int64),
                "image_features": packed_features,
            }
        )["inputs_embeds"]
        np.testing.assert_array_equal(batched_embeds[0, 0], packed_features[0])
        np.testing.assert_array_equal(batched_embeds[1, 0], packed_features[1])

        # The feature input is packed by modality (all images, then all videos),
        # while placeholders can appear in any order across batch rows.
        media_features = np.stack(
            [
                np.full(config.hidden_size, 10.0, dtype=np.float32),
                np.full(config.hidden_size, 20.0, dtype=np.float32),
                np.full(config.hidden_size, 30.0, dtype=np.float32),
                np.full(config.hidden_size, 40.0, dtype=np.float32),
            ]
        )
        mixed_embeds = embedding_session.run(
            {
                "input_ids": np.array([[101, 5, 100], [100, 6, 101]], dtype=np.int64),
                "image_features": media_features,
            }
        )["inputs_embeds"]
        np.testing.assert_array_equal(mixed_embeds[0, 0], media_features[2])
        np.testing.assert_array_equal(mixed_embeds[0, 2], media_features[0])
        np.testing.assert_array_equal(mixed_embeds[1, 0], media_features[1])
        np.testing.assert_array_equal(mixed_embeds[1, 2], media_features[3])

        decoder_feeds: dict[str, np.ndarray] = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }
        for layer_idx in range(config.num_hidden_layers):
            decoder_feeds[f"past_key_values.{layer_idx}.key"] = np.zeros(
                (1, config.num_key_value_heads, 0, config.head_dim),
                dtype=np.float32,
            )
            decoder_feeds[f"past_key_values.{layer_idx}.value"] = np.zeros(
                (1, config.num_key_value_heads, 0, config.head_dim),
                dtype=np.float32,
            )
        onnx_logits = decoder_session.run(decoder_feeds)["logits"]
    finally:
        vision_session.close()
        embedding_session.close()
        decoder_session.close()

    np.testing.assert_allclose(onnx_logits, hf_logits, rtol=1e-4, atol=1e-4)

    text_model = MuseGlimmerTextCausalLMModel(config)
    embed_weight = torch.ones(config.vocab_size, config.hidden_size)
    lm_head_weight = torch.ones(config.vocab_size, config.hidden_size)
    text_weights = text_model.preprocess_weights(
        {
            "model.language_model.embed_tokens.weight": embed_weight,
            "lm_head.weight": lm_head_weight,
            "model.vision_tower.patch_embedding.weight": torch.ones(1),
        }
    )
    assert text_weights == {
        "model.embed_tokens.weight": embed_weight,
        "lm_head.weight": lm_head_weight,
    }


def test_muse_glimmer_text_quantization_emits_quantized_ops():
    from mobius._configs import QuantizationConfig

    config = MuseGlimmerConfig(
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        num_hidden_layers=2,
        vocab_size=256,
        max_position_embeddings=128,
        hidden_act="silu",
        layer_types=["sliding_attention", "full_attention"],
        layer_rope_theta=[500_000.0, 0],
        sliding_window=8,
        pad_token_id=0,
        rope_type="default",
        rope_theta=500_000.0,
        rms_norm_eps=1e-5,
        dtype=ir.DataType.BFLOAT16,
        quantization=QuantizationConfig(
            bits=4,
            group_size=32,
            quant_method="gguf",
            sym=False,
            quantize_embeddings=True,
            quantize_lm_head=True,
        ),
    )
    module = MuseGlimmerTextCausalLMModel(config)
    model = build_from_module(module, config)["model"]
    counts = Counter(node.op_type for node in model.graph)

    # 5 attention projections + 3 MLP projections per layer, plus the LM head.
    assert counts["MatMulNBits"] == 2 * 8 + 1
    assert counts["GatherBlockQuantized"] == 1
    assert counts["MatMul"] == 0


def test_muse_glimmer_without_quantization_stays_float():
    config = MuseGlimmerConfig(
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        num_hidden_layers=2,
        vocab_size=256,
        max_position_embeddings=128,
        hidden_act="silu",
        layer_types=["sliding_attention", "full_attention"],
        layer_rope_theta=[500_000.0, 0],
        sliding_window=8,
        pad_token_id=0,
        rope_type="default",
        rope_theta=500_000.0,
        rms_norm_eps=1e-5,
        dtype=ir.DataType.BFLOAT16,
    )
    module = MuseGlimmerTextCausalLMModel(config)
    model = build_from_module(module, config)["model"]
    counts = Counter(node.op_type for node in model.graph)

    assert counts["MatMulNBits"] == 0
    assert counts["GatherBlockQuantized"] == 0
