# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Synthetic full-pipeline parity tests for GLM-OCR."""

from __future__ import annotations

import numpy as np
import onnx_ir as ir
import torch

from mobius._builder import _cast_module_dtype
from mobius._testing.ort_inference import OnnxModelSession
from mobius.integrations.transformers._config_resolver import _config_from_hf
from mobius.models.glm_ocr import GlmOcrForConditionalGeneration
from mobius.tasks import GlmOcrVLTask


def test_glm_ocr_synthetic_full_pipeline_matches_huggingface() -> None:
    """Vision, merger, embedding injection, M-RoPE, and decoder logits agree."""
    from transformers.models.glm_ocr.configuration_glm_ocr import (
        GlmOcrConfig,
        GlmOcrTextConfig,
        GlmOcrVisionConfig,
    )
    from transformers.models.glm_ocr.modeling_glm_ocr import (
        GlmOcrForConditionalGeneration as HfGlmOcrForConditionalGeneration,
    )

    torch.manual_seed(0)
    text_config = GlmOcrTextConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_nextn_predict_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=128,
        pad_token_id=0,
        attention_bias=False,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10_000.0,
            "mrope_section": [1, 1, 2],
            "partial_rotary_factor": 1.0,
        },
    )
    vision_config = GlmOcrVisionConfig(
        depth=1,
        hidden_size=16,
        intermediate_size=32,
        num_heads=2,
        image_size=28,
        patch_size=14,
        temporal_patch_size=2,
        spatial_merge_size=2,
        in_channels=3,
        out_hidden_size=32,
        attention_bias=True,
    )
    hf_config = GlmOcrConfig(
        text_config=text_config,
        vision_config=vision_config,
        image_start_token_id=120,
        image_end_token_id=121,
        image_token_id=122,
    )
    hf_model = HfGlmOcrForConditionalGeneration(hf_config).float().eval()

    config = _config_from_hf(text_config, parent_config=hf_config)
    module = GlmOcrForConditionalGeneration(config)
    bf16_module = GlmOcrForConditionalGeneration(config)
    _cast_module_dtype(bf16_module, ir.DataType.BFLOAT16)
    assert bf16_module.vision_encoder.visual.rotary_pos_emb.inv_freq.dtype == ir.DataType.FLOAT
    package = GlmOcrVLTask().build(module, config)
    package.apply_weights(module.preprocess_weights(dict(hf_model.state_dict())))

    input_ids = torch.tensor(
        [
            [
                1,
                120,
                122,
                122,
                122,
                122,
                121,
                2,
                120,
                122,
                122,
                122,
                122,
                121,
                3,
            ]
        ],
        dtype=torch.int64,
    )
    attention_mask = torch.ones_like(input_ids)
    mm_token_type_ids = torch.tensor(
        [[0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0]],
        dtype=torch.int64,
    )
    # Two differently shaped media rows validate packed feature ordering and
    # temporal repetition in the vectorized rotary/attention metadata.
    image_grid_thw = torch.tensor([[1, 4, 4], [2, 2, 4]], dtype=torch.int64)
    pixel_values = torch.randn(32, 3 * 2 * 14 * 14)

    with torch.no_grad():
        hf_outputs = hf_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            mm_token_type_ids=mm_token_type_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            use_cache=False,
        )
        text_embeddings = hf_model.model.language_model.get_input_embeddings()(input_ids)
        position_ids = hf_model.model.compute_3d_position_ids(
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=None,
            mm_token_type_ids=mm_token_type_ids,
            attention_mask=attention_mask,
            past_key_values=None,
            inputs_embeds=text_embeddings,
        )
        # The processor accepts extreme-aspect-ratio pages whose patch-grid axis
        # exceeds 512. Exercise dynamic rotary frequencies past that old table bound.
        wide_grid_thw = torch.tensor([[1, 2, 514]], dtype=torch.int64)
        wide_pixel_values = torch.randn(1028, 3 * 2 * 14 * 14)
        hf_wide_features = hf_model.model.visual(
            wide_pixel_values,
            grid_thw=wide_grid_thw,
        )

    vision_session = OnnxModelSession(package["vision_encoder"])
    embedding_session = OnnxModelSession(package["embedding"])
    decoder_session = OnnxModelSession(package["decoder"])
    try:
        image_features = vision_session.run(
            {
                "pixel_values": pixel_values.numpy(),
                "image_grid_thw": image_grid_thw.numpy(),
            }
        )["image_features"]
        wide_image_features = vision_session.run(
            {
                "pixel_values": wide_pixel_values.numpy(),
                "image_grid_thw": wide_grid_thw.numpy(),
            }
        )["image_features"]
        inputs_embeds = embedding_session.run(
            {
                "input_ids": input_ids.numpy(),
                "image_features": image_features,
            }
        )["inputs_embeds"]
        decoder_feeds: dict[str, np.ndarray] = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask.numpy(),
            "position_ids": position_ids.numpy(),
        }
        for layer_idx in range(text_config.num_hidden_layers):
            cache = np.zeros((1, text_config.num_key_value_heads, 0, text_config.head_dim))
            decoder_feeds[f"past_key_values.{layer_idx}.key"] = cache.astype(np.float32)
            decoder_feeds[f"past_key_values.{layer_idx}.value"] = cache.astype(np.float32)
        logits = decoder_session.run(decoder_feeds)["logits"]
    finally:
        vision_session.close()
        embedding_session.close()
        decoder_session.close()

    np.testing.assert_allclose(
        logits,
        hf_outputs.logits.detach().numpy(),
        rtol=1e-3,
        atol=1e-3,
    )
    np.testing.assert_allclose(
        wide_image_features,
        hf_wide_features.pooler_output.detach().numpy(),
        rtol=1e-3,
        atol=1e-3,
    )
