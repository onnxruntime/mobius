# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import numpy as np
import onnx_ir as ir
import onnxruntime as ort
import pytest

from mobius._builder import build_from_module
from mobius._configs import ArchitectureConfig
from mobius._configs._sub_configs import VisionConfig
from mobius.models import CausalLMModel
from mobius.models.gguf_projector import GenericGGUFProjectorModel
from mobius.tasks import GGUFProjectorVisionLanguageTask


def _legacy_clip_config(*, feature_layer: int | None = None) -> ArchitectureConfig:
    return ArchitectureConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=4,
        max_position_embeddings=128,
        rope_type="default",
        hidden_act="silu",
        vision=VisionConfig(
            image_size=28,
            patch_size=14,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=23,
            num_attention_heads=2,
            norm_eps=1e-5,
            hidden_act="quick_gelu",
            feature_layer=feature_layer,
        ),
    )


def _legacy_vision_model(config: ArchitectureConfig):
    module = GenericGGUFProjectorModel(
        config,
        CausalLMModel(config),
        projector_type="mlp",
        projector_hidden_size=16,
        image_token_id=-200,
    )
    model = build_from_module(
        module,
        config,
        task=GGUFProjectorVisionLanguageTask(),
    )["vision_encoder"]
    for name, initializer in model.graph.initializers.items():
        if initializer.const_value is None:
            rng = np.random.default_rng(sum(name.encode("utf-8")))
            initializer.const_value = ir.tensor(
                (rng.standard_normal(initializer.shape) * 0.05).astype(np.float32)
            )
    return module, model


def _embedding_session(
    tmp_path, *, hidden_size: int = 4
) -> tuple[ort.InferenceSession, np.ndarray]:
    config = ArchitectureConfig(
        vocab_size=8,
        hidden_size=hidden_size,
        intermediate_size=hidden_size * 2,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=hidden_size // 2,
        max_position_embeddings=16,
        rope_type="default",
        hidden_act="silu",
        vision=VisionConfig(
            image_size=28,
            patch_size=14,
            hidden_size=4,
            intermediate_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            norm_eps=1e-5,
            hidden_act="quick_gelu",
        ),
    )
    module = GenericGGUFProjectorModel(
        config,
        CausalLMModel(config),
        projector_type="mlp",
        projector_hidden_size=hidden_size,
        image_token_id=-200,
    )
    model = build_from_module(
        module,
        config,
        task=GGUFProjectorVisionLanguageTask(),
    )["embedding"]
    embedding = np.arange(8 * hidden_size, dtype=np.float32).reshape(8, hidden_size)
    model.graph.initializers["embedding.embed_tokens.weight"].const_value = ir.tensor(
        embedding
    )
    model_path = tmp_path / f"embedding-{hidden_size}.onnx"
    ir.save(model, model_path)
    return (
        ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"]),
        embedding,
    )


@pytest.mark.parametrize(
    ("projector_type", "image_size", "vision_width", "text_width", "kwargs"),
    [
        ("mlp", 28, 8, 16, {}),
        ("ldp", 336, 8, 16, {}),
        ("ldpv2", 336, 8, 16, {}),
        ("adapter", 28, 8, 16, {"projector_intermediate_size": 32}),
        ("resampler", 28, 8, 128, {"num_queries": 4}),
    ],
)
def test_generic_projector_package_builds(
    projector_type: str,
    image_size: int,
    vision_width: int,
    text_width: int,
    kwargs: dict[str, int],
):
    config = ArchitectureConfig(
        vocab_size=32,
        hidden_size=text_width,
        intermediate_size=text_width * 2,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=text_width // 4,
        max_position_embeddings=128,
        rope_type="default",
        hidden_act="silu",
        vision=VisionConfig(
            image_size=image_size,
            patch_size=14,
            hidden_size=vision_width,
            intermediate_size=vision_width * 2,
            num_hidden_layers=1,
            num_attention_heads=2,
            norm_eps=1e-5,
            hidden_act="quick_gelu",
        ),
    )
    module = GenericGGUFProjectorModel(
        config,
        CausalLMModel(config),
        projector_type=projector_type,
        projector_hidden_size=text_width,
        image_token_id=1,
        **kwargs,
    )
    package = build_from_module(
        module,
        config,
        task=GGUFProjectorVisionLanguageTask(),
    )

    assert set(package) == {"decoder", "vision_encoder", "embedding"}
    assert [value.name for value in package["vision_encoder"].graph.inputs] == ["pixel_values"]
    assert package["vision_encoder"].graph.inputs[0].dtype.name == "FLOAT"
    assert [value.name for value in package["vision_encoder"].graph.outputs] == [
        "image_features"
    ]
    assert [value.name for value in package["embedding"].graph.inputs] == [
        "input_ids",
        "image_features",
    ]


def test_legacy_clip_omits_final_serialized_block():
    config = _legacy_clip_config()
    module = GenericGGUFProjectorModel(
        config,
        CausalLMModel(config),
        projector_type="mlp",
        projector_hidden_size=16,
        image_token_id=-200,
    )

    assert len(module.vision_encoder.vision_tower.encoder) == 22


def test_explicit_final_feature_layer_executes_23rd_block_numerically():
    default_module, default_model = _legacy_vision_model(_legacy_clip_config())
    explicit_module, explicit_model = _legacy_vision_model(
        _legacy_clip_config(feature_layer=23)
    )
    assert len(default_module.vision_encoder.vision_tower.encoder) == 22
    assert len(explicit_module.vision_encoder.vision_tower.encoder) == 23
    pixel_values = np.linspace(-1, 1, 3 * 28 * 28, dtype=np.float32).reshape(1, 3, 28, 28)
    default_session = ort.InferenceSession(
        ir.serde.serialize_model(default_model).SerializeToString(),
        providers=["CPUExecutionProvider"],
    )
    explicit_session = ort.InferenceSession(
        ir.serde.serialize_model(explicit_model).SerializeToString(),
        providers=["CPUExecutionProvider"],
    )

    (default_output,) = default_session.run(None, {"pixel_values": pixel_values})
    (explicit_output,) = explicit_session.run(None, {"pixel_values": pixel_values})

    assert default_output.shape == explicit_output.shape == (4, 16)
    assert np.isfinite(explicit_output).all()
    assert not np.allclose(default_output, explicit_output, rtol=1e-5, atol=1e-5)


def test_minicpm_embedding_replaces_only_boundary_scoped_unknown_tokens(tmp_path):
    config = ArchitectureConfig(
        vocab_size=8,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=64,
        max_position_embeddings=16,
        rope_type="default",
        hidden_act="silu",
        vision=VisionConfig(
            image_size=28,
            patch_size=14,
            hidden_size=4,
            intermediate_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            norm_eps=1e-5,
            hidden_act="quick_gelu",
        ),
    )
    module = GenericGGUFProjectorModel(
        config,
        CausalLMModel(config),
        projector_type="resampler",
        projector_hidden_size=128,
        image_token_id=0,
        image_start_token_id=2,
        image_end_token_id=3,
        num_queries=2,
    )
    model = build_from_module(
        module,
        config,
        task=GGUFProjectorVisionLanguageTask(),
    )["embedding"]
    embedding = np.arange(8 * 128, dtype=np.float32).reshape(8, 128)
    model.graph.initializers["embedding.embed_tokens.weight"].const_value = ir.tensor(
        embedding
    )
    model_path = tmp_path / "embedding.onnx"
    ir.save(model, model_path)
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    image_features = np.arange(2 * 128, dtype=np.float32).reshape(2, 128) + 2000

    (actual,) = session.run(
        None,
        {
            "input_ids": np.array([[0, 2, 0, 0, 3, 0]], np.int64),
            "image_features": image_features,
        },
    )

    expected = embedding[[0, 2, 0, 0, 3, 0]][None]
    expected[0, 2:4] = image_features
    np.testing.assert_array_equal(actual, expected)


def test_embedding_sanitizes_negative_media_sentinel_before_gather(tmp_path):
    config = ArchitectureConfig(
        vocab_size=8,
        hidden_size=4,
        intermediate_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=2,
        max_position_embeddings=16,
        rope_type="default",
        hidden_act="silu",
        vision=VisionConfig(
            image_size=28,
            patch_size=14,
            hidden_size=4,
            intermediate_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            norm_eps=1e-5,
            hidden_act="quick_gelu",
        ),
    )
    module = GenericGGUFProjectorModel(
        config,
        CausalLMModel(config),
        projector_type="mlp",
        projector_hidden_size=4,
        image_token_id=-200,
    )
    model = build_from_module(
        module,
        config,
        task=GGUFProjectorVisionLanguageTask(),
    )["embedding"]
    embedding = np.arange(32, dtype=np.float32).reshape(8, 4)
    model.graph.initializers["embedding.embed_tokens.weight"].const_value = ir.tensor(
        embedding
    )
    model_path = tmp_path / "negative-sentinel.onnx"
    ir.save(model, model_path)
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    image_features = np.array([[100, 101, 102, 103]], np.float32)

    (actual,) = session.run(
        None,
        {
            "input_ids": np.array([[1, -200, 4]], np.int64),
            "image_features": image_features,
        },
    )

    expected = embedding[[1, 0, 4]][None]
    expected[0, 1] = image_features[0]
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    ("input_ids", "feature_rows"),
    [
        ([1, -200, 4], 0),
        ([-200, 2, -200], 1),
        ([1, -200, 4], 2),
    ],
)
def test_embedding_rejects_mismatched_placeholder_and_feature_counts(
    tmp_path, input_ids, feature_rows
):
    session, _ = _embedding_session(tmp_path)

    with pytest.raises(
        (
            ort.capi.onnxruntime_pybind11_state.Fail,
            ort.capi.onnxruntime_pybind11_state.InvalidArgument,
            ort.capi.onnxruntime_pybind11_state.RuntimeException,
        ),
        match="indices element out of data bounds",
    ):
        session.run(
            None,
            {
                "input_ids": np.array([input_ids], np.int64),
                "image_features": np.zeros((feature_rows, 4), np.float32),
            },
        )


def test_embedding_allows_text_only_input_with_zero_features(tmp_path):
    session, embedding = _embedding_session(tmp_path)

    (actual,) = session.run(
        None,
        {
            "input_ids": np.array([[1, 2, 4]], np.int64),
            "image_features": np.zeros((0, 4), np.float32),
        },
    )

    np.testing.assert_array_equal(actual, embedding[[1, 2, 4]][None])
