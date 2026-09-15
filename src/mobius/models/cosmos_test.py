# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Unit tests for Cosmos3-Edge weight routing, vision I/O and token fusion."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import onnx_ir as ir
import onnxruntime as ort
import pytest
import torch

from mobius._configs import ArchitectureConfig, VisionConfig
from mobius._configs.per_model._cosmos3_edge_vision import _cosmos3_edge_vision
from mobius.models.cosmos import (
    Cosmos3EdgeTextModel,
    Cosmos3EdgeVLModel,
    _Cosmos3EdgeVisionEncoderModel,
)
from mobius.tasks import Cosmos3EdgeVLTask

IMAGE_TOKEN_ID = 19
VIDEO_TOKEN_ID = 18


def _tiny_config(
    *,
    tie_word_embeddings: bool = False,
    spatial_merge_size: int | None = 2,
    out_hidden_size: int | None = 64,
    num_patches: int | None = 16,
    patch_size: int = 14,
) -> ArchitectureConfig:
    return ArchitectureConfig(
        model_type="cosmos3_edge",
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        vocab_size=256,
        max_position_embeddings=128,
        hidden_act="relu2",
        rms_norm_eps=1e-6,
        tie_word_embeddings=tie_word_embeddings,
        mrope_section=[4, 2, 2],
        mrope_interleaved=True,
        image_token_id=IMAGE_TOKEN_ID,
        vision=VisionConfig(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=2,
            image_size=None,
            patch_size=patch_size,
            num_patches=num_patches,
            spatial_merge_size=spatial_merge_size,
            temporal_patch_size=1,
            out_hidden_size=out_hidden_size,
            projector_intermediate_size=64,
            image_token_id=IMAGE_TOKEN_ID,
            video_token_id=VIDEO_TOKEN_ID,
        ),
    )


def test_text_attention_weights_map_to_mobius_names():
    module = Cosmos3EdgeTextModel(_tiny_config())
    weights = {
        "layers.0.self_attn.to_q.weight": torch.zeros(1),
        "layers.0.self_attn.to_k.weight": torch.zeros(1),
        "layers.0.self_attn.to_v.weight": torch.zeros(1),
        "layers.0.self_attn.to_out.0.weight": torch.zeros(1),
        "layers.0.self_attn.norm_q.weight": torch.zeros(1),
        "layers.0.self_attn.norm_k.weight": torch.zeros(1),
    }

    result = module.preprocess_weights(weights)

    assert "model.layers.0.self_attn.q_proj.weight" in result
    assert "model.layers.0.self_attn.k_proj.weight" in result
    assert "model.layers.0.self_attn.v_proj.weight" in result
    assert "model.layers.0.self_attn.o_proj.weight" in result
    assert "model.layers.0.self_attn.q_norm.weight" in result
    assert "model.layers.0.self_attn.k_norm.weight" in result


def test_vl_tied_embedding_populates_decoder_lm_head():
    module = Cosmos3EdgeVLModel(_tiny_config(tie_word_embeddings=True))
    weight = torch.randn(256, 64)

    result = module.preprocess_weights({"embed_tokens.weight": weight})

    assert result["embedding.embed_tokens.weight"] is weight
    assert result["decoder.lm_head.weight"] is weight


def test_vl_tied_lm_head_populates_embedding():
    module = Cosmos3EdgeVLModel(_tiny_config(tie_word_embeddings=True))
    weight = torch.randn(256, 64)

    result = module.preprocess_weights({"lm_head.weight": weight})

    assert result["decoder.lm_head.weight"] is weight
    assert result["embedding.embed_tokens.weight"] is weight


def test_vl_drops_unified_generator_and_action_weights():
    module = Cosmos3EdgeVLModel(_tiny_config())
    weights = {
        "layers.0.self_attn.add_q_proj.weight": torch.zeros(1),
        "layers.0.mlp_moe_gen.up_proj.weight": torch.zeros(1),
        "time_embedder.linear_1.weight": torch.zeros(1),
        "proj_in.weight": torch.zeros(1),
        "action_proj_in.fc.weight": torch.zeros(1),
        "action_modality_embed": torch.zeros(1),
    }

    assert module.preprocess_weights(weights) == {}


def test_vision_weights_keep_hf_layout_and_linear_patch_embedding():
    """``model.visual``/``model.projector`` map with only a prefix strip.

    The checkpoint's ``patch_embedding`` is an ``nn.Linear`` over flattened
    ``(patch_h, patch_w, channel)`` values, so it must stay 2-D — reshaping it
    into a Conv2d ``[out, C, kH, kW]`` kernel would scramble the channel axis.
    """
    module = Cosmos3EdgeVLModel(_tiny_config())
    patch_weight = torch.arange(32 * 3 * 14 * 14, dtype=torch.float32).reshape(32, 3 * 14 * 14)
    weights = {
        "model.visual.embeddings.patch_embedding.weight": patch_weight,
        "model.visual.embeddings.patch_embedding.bias": torch.zeros(32),
        "model.visual.embeddings.position_embedding.weight": torch.zeros(16, 32),
        "model.visual.encoder.layers.0.mlp.fc1.weight": torch.zeros(64, 32),
        "model.visual.encoder.layers.0.mlp.fc2.weight": torch.zeros(32, 64),
        "model.visual.post_layernorm.weight": torch.zeros(32),
        "model.projector.norm.weight": torch.zeros(32),
        "model.projector.linear_fc1.weight": torch.zeros(64, 128),
        "model.projector.linear_fc2.weight": torch.zeros(64, 64),
    }

    result = module.preprocess_weights(weights)

    embed = result["vision_encoder.visual.embeddings.patch_embedding.weight"]
    assert embed.shape == (32, 3 * 14 * 14)
    assert torch.equal(embed, patch_weight)
    assert "vision_encoder.visual.embeddings.position_embedding.weight" in result
    assert "vision_encoder.visual.encoder.layers.0.mlp.up_proj.weight" in result
    assert "vision_encoder.visual.encoder.layers.0.mlp.down_proj.weight" in result
    assert "vision_encoder.visual.post_layernorm.weight" in result
    assert "vision_encoder.projector.linear_fc1.weight" in result


def test_vision_encoder_defaults_missing_spatial_merge_size():
    module = _Cosmos3EdgeVisionEncoderModel(_tiny_config(spatial_merge_size=None))

    assert module.projector.spatial_merge_size == 2


def test_vision_encoder_rejects_projector_width_mismatch():
    with pytest.raises(AssertionError, match="projector output must match"):
        _Cosmos3EdgeVisionEncoderModel(_tiny_config(out_hidden_size=32))


def test_vision_encoder_rejects_non_square_reference_grid():
    with pytest.raises(ValueError, match="square reference grid"):
        _Cosmos3EdgeVisionEncoderModel(_tiny_config(num_patches=15))


def test_vl_task_declares_packed_vision_and_dual_feature_embedding():
    config = _tiny_config()
    package = Cosmos3EdgeVLTask().build(Cosmos3EdgeVLModel(config), config)

    vision = package["vision_encoder"].graph
    pixel_values, grid_thw = vision.inputs
    assert pixel_values.name == "pixel_values"
    # patch_dim = patch * patch * channels * temporal_patch_size
    assert pixel_values.shape[1] == 14 * 14 * 3 * 1
    assert grid_thw.name == "grid_thw"
    assert grid_thw.dtype == ir.DataType.INT64
    assert list(grid_thw.shape) == [3]

    image_features = vision.outputs[0]
    assert len(image_features.shape) == 2
    assert image_features.shape[-1] == config.hidden_size

    embedding_inputs = {value.name: value for value in package["embedding"].graph.inputs}
    assert set(embedding_inputs) == {"input_ids", "image_features", "video_features"}
    for name in ("image_features", "video_features"):
        assert len(embedding_inputs[name].shape) == 2
        assert embedding_inputs[name].shape[-1] == config.hidden_size

    # Interleaved MRoPE decoder contract: position_ids is [3, batch, seq].
    decoder_inputs = {value.name: value for value in package["decoder"].graph.inputs}
    assert decoder_inputs["position_ids"].shape[0] == 3


def _run(model: ir.Model, feeds: dict[str, np.ndarray], tmp_path, name: str):
    path = tmp_path / f"{name}.onnx"
    ir.save(model, str(path))
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return session.run(None, feeds)


def _random_vision_weights(config: ArchitectureConfig) -> dict[str, torch.Tensor]:
    """Small random ``model.visual.*``/``model.projector.*`` checkpoint slice."""
    vc = config.vision
    assert vc is not None
    generator = torch.Generator().manual_seed(0)

    def randn(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=generator) * 0.02

    patch_dim = vc.patch_size * vc.patch_size * vc.in_channels
    weights: dict[str, torch.Tensor] = {
        "model.visual.embeddings.patch_embedding.weight": randn(vc.hidden_size, patch_dim),
        "model.visual.embeddings.patch_embedding.bias": randn(vc.hidden_size),
        "model.visual.embeddings.position_embedding.weight": randn(
            vc.num_patches, vc.hidden_size
        ),
        "model.visual.post_layernorm.weight": torch.ones(vc.hidden_size),
        "model.visual.post_layernorm.bias": torch.zeros(vc.hidden_size),
        "model.projector.norm.weight": torch.ones(vc.hidden_size),
        "model.projector.norm.bias": torch.zeros(vc.hidden_size),
        "model.projector.linear_fc1.weight": randn(
            vc.projector_intermediate_size, vc.hidden_size * 4
        ),
        "model.projector.linear_fc1.bias": randn(vc.projector_intermediate_size),
        "model.projector.linear_fc2.weight": randn(
            config.hidden_size, vc.projector_intermediate_size
        ),
        "model.projector.linear_fc2.bias": randn(config.hidden_size),
    }
    for layer in range(vc.num_hidden_layers):
        prefix = f"model.visual.encoder.layers.{layer}"
        for norm in ("layer_norm1", "layer_norm2"):
            weights[f"{prefix}.{norm}.weight"] = torch.ones(vc.hidden_size)
            weights[f"{prefix}.{norm}.bias"] = torch.zeros(vc.hidden_size)
        for proj in ("q_proj", "k_proj", "v_proj", "out_proj"):
            weights[f"{prefix}.self_attn.{proj}.weight"] = randn(
                vc.hidden_size, vc.hidden_size
            )
            weights[f"{prefix}.self_attn.{proj}.bias"] = randn(vc.hidden_size)
        weights[f"{prefix}.mlp.fc1.weight"] = randn(vc.intermediate_size, vc.hidden_size)
        weights[f"{prefix}.mlp.fc1.bias"] = randn(vc.intermediate_size)
        weights[f"{prefix}.mlp.fc2.weight"] = randn(vc.hidden_size, vc.intermediate_size)
        weights[f"{prefix}.mlp.fc2.bias"] = randn(vc.hidden_size)

    routed = Cosmos3EdgeVLModel(config).preprocess_weights(weights)
    return {key: value for key, value in routed.items() if key.startswith("vision_encoder.")}


def test_vision_encoder_token_count_scales_with_frames(tmp_path):
    """Video frames reuse the image path: tokens = t*h*w / merge**2."""
    config = _tiny_config()
    package = Cosmos3EdgeVLTask().build(Cosmos3EdgeVLModel(config), config)
    package.apply_weights_partial(_random_vision_weights(config))

    patch_dim = 14 * 14 * 3
    grid_h, grid_w = 4, 6
    rng = np.random.RandomState(0)
    for frames in (1, 3):
        total = frames * grid_h * grid_w
        features = _run(
            package["vision_encoder"],
            {
                "pixel_values": rng.randn(total, patch_dim).astype(np.float32),
                "grid_thw": np.array([frames, grid_h, grid_w], dtype=np.int64),
            },
            tmp_path,
            f"vision_{frames}",
        )[0]
        assert features.shape == (total // 4, config.hidden_size)


def _set_embedding_table(package, table: np.ndarray) -> None:
    package.apply_weights_partial({"embedding.embed_tokens.weight": torch.from_numpy(table)})


def test_embedding_scatters_image_and_video_tokens(tmp_path):
    config = _tiny_config()
    package = Cosmos3EdgeVLTask().build(Cosmos3EdgeVLModel(config), config)
    table = np.arange(config.vocab_size * config.hidden_size, dtype=np.float32).reshape(
        config.vocab_size, config.hidden_size
    )
    table /= table.max()
    _set_embedding_table(package, table)

    input_ids = np.array(
        [[7, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, 8, VIDEO_TOKEN_ID, VIDEO_TOKEN_ID, 9]],
        dtype=np.int64,
    )
    image_features = np.full((2, config.hidden_size), 3.0, dtype=np.float32)
    image_features[1] = 4.0
    video_features = np.full((2, config.hidden_size), -5.0, dtype=np.float32)
    video_features[1] = -6.0

    embeds = _run(
        package["embedding"],
        {
            "input_ids": input_ids,
            "image_features": image_features,
            "video_features": video_features,
        },
        tmp_path,
        "embedding",
    )[0]

    np.testing.assert_allclose(embeds[0, 1], image_features[0])
    np.testing.assert_allclose(embeds[0, 2], image_features[1])
    np.testing.assert_allclose(embeds[0, 4], video_features[0])
    np.testing.assert_allclose(embeds[0, 5], video_features[1])
    # Text positions keep their token embedding.
    np.testing.assert_allclose(embeds[0, 0], table[7])
    np.testing.assert_allclose(embeds[0, 3], table[8])
    np.testing.assert_allclose(embeds[0, 6], table[9])


def test_embedding_tolerates_empty_feature_streams(tmp_path):
    config = _tiny_config()
    package = Cosmos3EdgeVLTask().build(Cosmos3EdgeVLModel(config), config)
    _set_embedding_table(
        package, np.zeros((config.vocab_size, config.hidden_size), dtype=np.float32)
    )

    embeds = _run(
        package["embedding"],
        {
            "input_ids": np.array([[1, 2, 3]], dtype=np.int64),
            "image_features": np.zeros((0, config.hidden_size), dtype=np.float32),
            "video_features": np.zeros((0, config.hidden_size), dtype=np.float32),
        },
        tmp_path,
        "embedding_empty",
    )[0]

    assert embeds.shape == (1, 3, config.hidden_size)


def test_vision_config_rejects_non_square_num_patches():
    vision_config = SimpleNamespace(num_patches=255, patch_size=16)
    config = SimpleNamespace(vision_config=vision_config)

    with pytest.raises(ValueError, match="num_patches must form a square grid"):
        _cosmos3_edge_vision(config, None, "cosmos3_edge", {"image_size": None})


def test_vision_config_hook_extracts_edge_fields():
    config = SimpleNamespace(
        model_type="cosmos3_edge_text",
        vision_config=SimpleNamespace(
            num_patches=256, patch_size=16, hidden_act="gelu_pytorch_tanh"
        ),
        projector_config={
            "merger_intermediate_size": 128,
            "out_hidden_size": 64,
            "use_postshuffle_norm": False,
            "spatial_merge_size": 2,
        },
        image_token_id=19,
        video_token_id=18,
        vision_start_token_id=20,
        vision_end_token_id=21,
    )
    fields = {"image_size": None, "patch_size": 16, "out_hidden_size": None}

    _cosmos3_edge_vision(config, config, "cosmos3_edge_text", fields)

    assert fields["num_patches"] == 256
    assert fields["image_size"] == 256
    assert fields["projector_intermediate_size"] == 128
    assert fields["use_postshuffle_norm"] is False
    assert fields["temporal_patch_size"] == 1
    assert fields["image_token_id"] == 19
    assert fields["video_token_id"] == 18
    assert fields["vision_start_token_id"] == 20
    assert fields["vision_end_token_id"] == 21
    # Cosmos3-Edge M-RoPE assigns axes to interleaved channels, not chunks.
    assert fields["mrope_interleaved"] is True
