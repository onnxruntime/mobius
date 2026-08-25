# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import numpy as np
import onnx_ir as ir
import onnxruntime as ort
from onnxscript import GraphBuilder

from mobius._configs import ArchitectureConfig, VisionConfig
from mobius.components import Gemma3MultiModalProjector, OffsetRMSNorm
from mobius.models.gemma3 import Gemma3MultiModalModel, _Gemma3EmbeddingModel
from mobius.tasks import Gemma3VisionLanguageTask
from mobius.tasks._base import build_embedding_from_features


def _session(model: ir.Model) -> ort.InferenceSession:
    proto = ir.serde.serialize_model(model)
    return ort.InferenceSession(proto.SerializeToString(), providers=["CPUExecutionProvider"])


def _embedding_session() -> ort.InferenceSession:
    config = ArchitectureConfig(
        vocab_size=16,
        hidden_size=4,
        pad_token_id=0,
        image_token_id=9,
    )
    model = build_embedding_from_features(
        _Gemma3EmbeddingModel(config),
        config,
        feature_name="image_features",
        feature_dim=4,
    )
    weights = np.arange(64, dtype=np.float32).reshape(16, 4)
    model.graph.initializers["embed_tokens.weight"].const_value = ir.tensor(weights)
    return _session(model)


class TestGemma3Embedding:
    def test_zero_media_features_preserve_text_embeddings(self) -> None:
        session = _embedding_session()
        actual = session.run(
            None,
            {
                "input_ids": np.array([[1, 2]], dtype=np.int64),
                "image_features": np.empty((0, 4), dtype=np.float32),
            },
        )[0]
        expected = 2.0 * np.arange(64, dtype=np.float32).reshape(16, 4)[[1, 2]]
        np.testing.assert_allclose(actual, expected[None])

    def test_two_rows_use_flattened_feature_offsets(self) -> None:
        session = _embedding_session()
        features = np.array([[10.0] * 4, [20.0] * 4], dtype=np.float32)
        actual = session.run(
            None,
            {
                "input_ids": np.array([[9, 1], [2, 9]], dtype=np.int64),
                "image_features": features,
            },
        )[0]
        np.testing.assert_array_equal(actual[0, 0], features[0])
        np.testing.assert_array_equal(actual[1, 1], features[1])


class TestGemma3VisionEncoder:
    def test_projector_matches_numpy_reference(self) -> None:
        features = np.arange(32, dtype=np.float32).reshape(1, 16, 2) / 10
        norm_weight = np.array([0.25, -0.5], dtype=np.float32)
        projection = np.array([[1.0, 2.0, -1.0], [0.5, -0.25, 3.0]], dtype=np.float32)

        x = ir.Value(
            name="vision_features",
            shape=ir.Shape([1, 16, 2]),
            type=ir.TensorType(ir.DataType.FLOAT),
        )
        graph = ir.Graph(
            inputs=[x],
            outputs=[],
            nodes=[],
            name="gemma3_projector",
            opset_imports={"": 24},
        )
        builder = GraphBuilder(graph)
        projector = Gemma3MultiModalProjector(
            2,
            3,
            patches_per_image=4,
            tokens_per_image=1,
            norm=OffsetRMSNorm(2, eps=1e-6),
        )
        output = projector(builder.op, x)
        output.name = "image_features"
        graph.outputs.append(output)
        model = ir.Model(graph, ir_version=11)
        graph.initializers["mm_soft_emb_norm.weight"].const_value = ir.tensor(norm_weight)
        graph.initializers["mm_input_projection_weight"].const_value = ir.tensor(projection)

        actual = _session(model).run(None, {"vision_features": features})[0]
        pooled = features.mean(axis=1, keepdims=True)
        normalized = pooled / np.sqrt(np.mean(pooled**2, axis=-1, keepdims=True) + 1e-6)
        expected = (normalized * (1.0 + norm_weight)) @ projection
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)

    def test_canonical_three_model_package_keys(self) -> None:
        config = ArchitectureConfig(
            vocab_size=32,
            hidden_size=8,
            intermediate_size=16,
            hidden_act="gelu",
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=4,
            pad_token_id=0,
            max_position_embeddings=32,
            rope_type="default",
            rope_local_base_freq=10_000.0,
            layer_types=["full_attention", "sliding_attention"],
            sliding_window=8,
            attn_qk_norm=True,
            image_token_id=9,
            vision=VisionConfig(
                hidden_size=4,
                intermediate_size=8,
                num_hidden_layers=1,
                num_attention_heads=2,
                image_size=16,
                patch_size=4,
                mm_tokens_per_image=1,
            ),
        )
        package = Gemma3VisionLanguageTask().build(Gemma3MultiModalModel(config), config)
        assert set(package) == {"decoder", "vision_encoder", "embedding"}
        pixel_values = package["vision_encoder"].graph.inputs[0]
        assert pixel_values.dtype == ir.DataType.FLOAT
        assert list(pixel_values.shape) == [1, 3, 16, 16]
