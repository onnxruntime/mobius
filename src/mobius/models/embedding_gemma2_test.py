# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import dataclasses

import numpy as np
import onnx_ir as ir
import pytest
import torch

from mobius import build_from_module
from mobius._testing.ort_inference import OnnxModelSession
from mobius.models.embedding_gemma2 import EmbeddingGemma2Config, EmbeddingGemma2Model


def tiny_config(**kwargs) -> EmbeddingGemma2Config:
    config = EmbeddingGemma2Config(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        global_head_dim=16,
        num_global_key_value_heads=1,
        hidden_size_per_layer_input=8,
        embedding_dim=24,
        hidden_act="gelu_pytorch_tanh",
        layer_types=["sliding_attention", "full_attention"],
        sliding_window=2,
        max_position_embeddings=32,
        rope_type="default",
        pad_token_id=0,
    )
    return dataclasses.replace(config, **kwargs)


def test_embedding_gemma2_builds_stateless_encoder():
    config = tiny_config()
    package = build_from_module(EmbeddingGemma2Model(config), config, task="embedding-gemma2")
    model = package["model"]
    assert {v.name for v in model.graph.outputs} == {"last_hidden_state", "sentence_embedding"}
    assert {v.name for v in model.graph.inputs} == {
        "input_ids",
        "attention_mask",
        "position_ids",
    }
    attention = [n for n in model.graph if n.op_type == "Attention"]
    assert len(attention) == 2
    assert all(n.attributes["is_causal"].as_int() == 0 for n in attention)


@pytest.mark.parametrize("dtype", [ir.DataType.FLOAT16, ir.DataType.BFLOAT16])
def test_embedding_gemma2_rejects_unqualified_precision(dtype):
    with pytest.raises(ValueError, match="float32 only"):
        EmbeddingGemma2Model(tiny_config(dtype=dtype))
    config = tiny_config()
    module = EmbeddingGemma2Model(config)
    config.dtype = dtype
    with pytest.raises(ValueError, match="float32 only"):
        build_from_module(module, config, task="embedding-gemma2")


@pytest.mark.integration
def test_embedding_gemma2_synthetic_parity():
    from transformers import EmbeddingGemma2Config as HFConfig
    from transformers import EmbeddingGemma2Model as HFModel
    from transformers import EmbeddingGemma2TextConfig as HFTextConfig

    torch.manual_seed(7)
    text = HFTextConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        global_head_dim=16,
        num_global_key_value_heads=1,
        hidden_size_per_layer_input=8,
        embedding_dim=24,
        layer_types=["sliding_attention", "full_attention"],
        sliding_window=2,
        max_position_embeddings=32,
    )
    hf_config = HFConfig(text_config=text, vision_config=None, audio_config=None)
    reference = HFModel(hf_config).eval()
    config = EmbeddingGemma2Config.from_transformers(hf_config)
    config.dtype = ir.DataType.FLOAT
    module = EmbeddingGemma2Model(config)
    package = build_from_module(module, config, task="embedding-gemma2")
    package.apply_weights(module.preprocess_weights(reference.state_dict()))
    session = OnnxModelSession(package)
    ids = np.array([[2, 5, 6, 9, 8, 3], [2, 4, 7, 1, 0, 0]], dtype=np.int64)
    mask = (ids != 0).astype(np.int64)
    positions = np.broadcast_to(np.arange(6, dtype=np.int64), ids.shape).copy()
    with torch.no_grad():
        expected = reference(
            input_ids=torch.from_numpy(ids),
            attention_mask=torch.from_numpy(mask),
            position_ids=torch.from_numpy(positions),
        ).last_hidden_state.numpy()
    actual = session.run({"input_ids": ids, "attention_mask": mask, "position_ids": positions})
    np.testing.assert_allclose(actual["last_hidden_state"], expected, atol=1e-4, rtol=1e-4)
    mean = (expected * mask[:, :, None]).sum(1) / mask.sum(1)[:, None]
    normalized = mean / np.linalg.norm(mean, axis=-1, keepdims=True)
    np.testing.assert_allclose(actual["sentence_embedding"], normalized, atol=1e-4, rtol=1e-4)


@pytest.mark.integration
def test_embedding_gemma2_media_pipeline():
    from transformers import EmbeddingGemma2Config as HFConfig
    from transformers import EmbeddingGemma2Model as HFModel
    from transformers import EmbeddingGemma2TextConfig as HFTextConfig
    from transformers import Gemma4AudioConfig, Gemma4VisionConfig

    torch.manual_seed(19)
    text = HFTextConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        global_head_dim=16,
        num_global_key_value_heads=1,
        hidden_size_per_layer_input=8,
        embedding_dim=24,
        layer_types=["sliding_attention", "full_attention"],
        sliding_window=2,
        max_position_embeddings=32,
    )
    vision = Gemma4VisionConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=8,
        global_head_dim=8,
        patch_size=2,
        pooling_kernel_size=3,
        position_embedding_size=32,
        default_output_length=1,
        max_position_embeddings=32,
        use_clipped_linears=False,
    )
    audio = Gemma4AudioConfig(
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        output_proj_dims=32,
        subsampling_conv_channels=[128, 4],
        use_clipped_linears=True,
    )
    hf_config = HFConfig(
        text_config=text,
        vision_config=vision,
        audio_config=audio,
        image_token_id=60,
        video_token_id=61,
        audio_token_id=62,
    )
    reference = HFModel(hf_config).eval()
    config = EmbeddingGemma2Config.from_transformers(hf_config)
    module = EmbeddingGemma2Model(config)
    package = build_from_module(module, config, task="embedding-gemma2")
    package.apply_weights(module.preprocess_weights(reference.state_dict()))
    vision_session = OnnxModelSession(package["vision_encoder"])
    audio_session = OnnxModelSession(package["audio_encoder"])
    encoder_session = OnnxModelSession(package["model"])
    patches = torch.rand(1, 9, 12)
    patch_positions = torch.tensor([[[x, y] for x in range(3) for y in range(3)]])
    sound = torch.randn(1, 16, 128)
    sound_mask = torch.ones(1, 16, dtype=torch.bool)
    image = vision_session.run(
        {"pixel_values": patches.numpy(), "pixel_position_ids": patch_positions.numpy()}
    )["image_features"]
    encoded_sound = audio_session.run(
        {"input_features": sound.numpy(), "input_features_mask": sound_mask.numpy()}
    )["audio_features"]
    with torch.no_grad():
        expected_image = reference.get_image_features(patches, patch_positions).pooler_output
        expected_sound = reference.get_audio_features(sound, sound_mask)
    expected_image = torch.cat(expected_image, dim=0).numpy()
    expected_sound = expected_sound.pooler_output[expected_sound.attention_mask].numpy()
    np.testing.assert_allclose(image, expected_image, atol=1e-3, rtol=1e-3)
    np.testing.assert_allclose(encoded_sound, expected_sound, atol=1e-3, rtol=1e-3)
    # Distinct rows exercise flattened-batch offsets and separate video ordering.
    sound_tokens = encoded_sound.shape[0]
    ids = torch.tensor(
        [[2, 60, *([62] * sound_tokens), 1], [2, 61, *([62] * sound_tokens), 1]]
    )
    mask = torch.ones_like(ids)
    positions = torch.arange(ids.shape[1]).expand_as(ids)
    with torch.no_grad():
        expected = reference(
            input_ids=ids,
            attention_mask=mask,
            position_ids=positions,
            pixel_values=patches,
            image_position_ids=patch_positions,
            pixel_values_videos=patches * 0.5,
            video_position_ids=patch_positions,
            num_frames_per_video=torch.tensor([1]),
            input_features=sound.repeat(2, 1, 1),
            input_features_mask=sound_mask.repeat(2, 1),
        ).last_hidden_state.numpy()
    video = vision_session.run(
        {"pixel_values": patches.numpy() * 0.5, "pixel_position_ids": patch_positions.numpy()}
    )["image_features"]
    actual = encoder_session.run(
        {
            "input_ids": ids.numpy(),
            "attention_mask": mask.numpy(),
            "position_ids": positions.numpy(),
            "image_features": image,
            "video_features": video,
            "audio_features": np.tile(encoded_sound, (2, 1)),
        }
    )
    np.testing.assert_allclose(actual["last_hidden_state"], expected, atol=1e-3, rtol=1e-3)
