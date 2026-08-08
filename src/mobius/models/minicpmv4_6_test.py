# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import dataclasses

import numpy as np
import onnx_ir as ir
import torch

from mobius._configs import ArchitectureConfig, VisionConfig
from mobius._testing.ort_inference import OnnxModelSession
from mobius._weight_loading import apply_weights
from mobius.models.minicpmv4_6 import MiniCPMV46ForConditionalGeneration
from mobius.tasks import MiniCPMVLTask


def _tiny_config() -> ArchitectureConfig:
    return ArchitectureConfig(
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        num_hidden_layers=4,
        vocab_size=256,
        max_position_embeddings=128,
        hidden_act="silu",
        rms_norm_eps=1e-6,
        rope_type="default",
        rope_theta=10_000.0,
        partial_rotary_factor=0.25,
        layer_types=[
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
        ],
        linear_num_key_heads=2,
        linear_key_head_dim=16,
        linear_num_value_heads=2,
        linear_value_head_dim=16,
        linear_conv_kernel_dim=4,
        image_token_id=250,
        video_token_id=251,
        vision=VisionConfig(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            image_size=56,
            patch_size=14,
            norm_eps=1e-6,
            in_channels=3,
            num_position_embeddings=16,
            insert_layer_id=0,
            window_kernel_size=(2, 2),
            merge_kernel_size=(2, 2),
            merger_times=1,
        ),
        dtype=ir.DataType.FLOAT,
    )


def test_minicpmv4_6_synthetic_vision_parity():
    """L3: packed vision + both mergers match Transformers with random weights."""
    from transformers.models.minicpmv4_6.configuration_minicpmv4_6 import (
        MiniCPMV4_6Config,
        MiniCPMV4_6VisionConfig,
    )
    from transformers.models.minicpmv4_6.modeling_minicpmv4_6 import (
        MiniCPMV4_6Merger,
        MiniCPMV4_6VisionModel,
    )

    torch.manual_seed(42)
    vision_config = MiniCPMV4_6VisionConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        image_size=56,
        patch_size=14,
    )
    hf_config = MiniCPMV4_6Config(
        vision_config=vision_config.to_dict(),
        insert_layer_id=0,
        text_config={
            "model_type": "qwen3_5_text",
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 16,
            "vocab_size": 256,
            "layer_types": [
                "linear_attention",
                "linear_attention",
                "linear_attention",
                "full_attention",
            ],
            "linear_num_key_heads": 2,
            "linear_key_head_dim": 16,
            "linear_num_value_heads": 2,
            "linear_value_head_dim": 16,
        },
    )
    hf_vision = MiniCPMV4_6VisionModel(hf_config.vision_config).eval()
    hf_merger = MiniCPMV4_6Merger(hf_config).eval()

    config = _tiny_config()
    module = MiniCPMV46ForConditionalGeneration(config)
    package = MiniCPMVLTask().build(module, config)
    state_dict = {
        **{
            f"model.vision_tower.{name}": value
            for name, value in hf_vision.state_dict().items()
        },
        **{f"model.merger.{name}": value for name, value in hf_merger.state_dict().items()},
    }
    processed_weights = module.preprocess_weights(state_dict)
    graph_parameters = {
        name
        for name in package["vision_encoder"].graph.initializers
        if name.startswith("vision_encoder.")
    }
    assert set(processed_weights) == graph_parameters
    apply_weights(
        package["vision_encoder"],
        processed_weights,
    )

    pixel_values = torch.randn(1, 3, 14, 224)
    target_sizes = torch.tensor([[4, 4]], dtype=torch.int32)
    with torch.no_grad():
        hidden_states = hf_vision(
            pixel_values,
            target_sizes=target_sizes,
        ).last_hidden_state
        expected = torch.cat(
            hf_merger(hidden_states, target_sizes // 2),
            dim=0,
        ).numpy()

    session = OnnxModelSession(package["vision_encoder"])
    actual = session.run(
        {
            "pixel_values": pixel_values.numpy(),
            "target_sizes": target_sizes.numpy(),
        }
    )["image_features"]
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)

    # Default slice-mode preprocessing commonly emits an overview crop and
    # detail crops with different grids. Compare the packed ONNX call against
    # independent HF calls, which are the ragged reference semantics.
    target_sizes = torch.tensor([[4, 4], [4, 8]], dtype=torch.int32)
    pixel_values = torch.randn(1, 3, 14, (4 * 4 + 4 * 8) * 14)
    expected_parts = []
    start = 0
    with torch.no_grad():
        for size in target_sizes:
            num_patches = int(size.prod())
            end = start + num_patches * 14
            unit_pixels = pixel_values[:, :, :, start:end]
            unit_size = size.unsqueeze(0)
            hidden_states = hf_vision(
                unit_pixels,
                target_sizes=unit_size,
            ).last_hidden_state
            expected_parts.extend(hf_merger(hidden_states, unit_size // 2))
            start = end
    expected = torch.cat(expected_parts, dim=0).numpy()
    actual = session.run(
        {
            "pixel_values": pixel_values.numpy(),
            "target_sizes": target_sizes.numpy(),
        }
    )["image_features"]
    session.close()
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)

    # The optional 4x mode skips only the in-tower merger, leaving the final
    # 2x2 projector to produce four times as many visual tokens as 16x mode.
    config_4x = dataclasses.replace(config, downsample_mode="4x")
    module_4x = MiniCPMV46ForConditionalGeneration(config_4x)
    package_4x = MiniCPMVLTask().build(module_4x, config_4x)
    processed_4x = module_4x.preprocess_weights(state_dict)
    graph_parameters_4x = set(package_4x["vision_encoder"].graph.initializers)
    apply_weights(
        package_4x["vision_encoder"],
        {name: value for name, value in processed_4x.items() if name in graph_parameters_4x},
    )
    target_sizes = torch.tensor([[4, 4]], dtype=torch.int32)
    pixel_values = torch.randn(1, 3, 14, 224)
    with torch.no_grad():
        hidden_states = hf_vision(
            pixel_values,
            target_sizes=target_sizes,
            use_vit_merger=False,
        ).last_hidden_state
        expected = torch.cat(
            hf_merger(hidden_states, target_sizes),
            dim=0,
        ).numpy()
    session = OnnxModelSession(package_4x["vision_encoder"])
    actual = session.run(
        {
            "pixel_values": pixel_values.numpy(),
            "target_sizes": target_sizes.numpy(),
        }
    )["image_features"]
    session.close()
    assert actual.shape[0] == 4
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)


def test_minicpmv4_6_embedding_mixes_image_and_video_tokens():
    """Image and video placeholders share the packed visual-feature stream."""
    config = _tiny_config()
    package = MiniCPMVLTask().build(
        MiniCPMV46ForConditionalGeneration(config),
        config,
    )
    rng = np.random.default_rng(42)
    for initializer in package["embedding"].graph.initializers.values():
        if initializer.const_value is None:
            initializer.const_value = ir.tensor(
                rng.standard_normal(initializer.shape).astype(np.float32)
            )

    input_ids = np.array([[1, 250, 2, 251]], dtype=np.int64)
    features = rng.standard_normal((2, config.hidden_size)).astype(np.float32)
    session = OnnxModelSession(package["embedding"])
    result = session.run(
        {
            "input_ids": input_ids,
            "image_features": features,
        }
    )["inputs_embeds"]
    session.close()

    np.testing.assert_array_equal(result[0, 1], features[0])
    np.testing.assert_array_equal(result[0, 3], features[1])
