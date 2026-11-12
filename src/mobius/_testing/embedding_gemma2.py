# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Checkpoint golden replay for a stateless multimodal embedding pipeline."""

from __future__ import annotations

import json
import os
from contextlib import closing
from pathlib import Path

import numpy as np
import onnxruntime as ort

from mobius import build
from mobius._testing.ort_inference import OnnxModelSession
from mobius.models.embedding_gemma2 import MODEL_ID, MODEL_REVISION

GOLDEN_ROOT = Path(__file__).resolve().parents[3] / "testdata" / "golden" / "embedding"


def golden_root() -> Path:
    """Allow a generator job to verify its artifact before committing it."""
    return Path(os.environ.get("MOBIUS_EMBEDDING_GOLDEN_DIR", GOLDEN_ROOT))


def replay(*, level: str) -> None:
    """Compare full outputs or semantic retrieval, never argmax token IDs."""
    root = golden_root()
    metadata = json.loads((root / "embeddinggemma-2.json").read_text())
    assert metadata["model_id"] == MODEL_ID
    assert metadata["revision"] == MODEL_REVISION
    assert metadata["dtype"] == "float32"
    device = os.environ.get("MOBIUS_TEST_DEVICE", "cpu")
    if device == "cuda":
        assert "CUDAExecutionProvider" in ort.get_available_providers()
    package = build(
        MODEL_ID,
        revision=MODEL_REVISION,
        dtype="f32",
        load_weights=True,
        execution_provider="cuda" if device == "cuda" else "cpu",
    )

    def run_component(name: str, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        # A10-4Q runners have 4 GiB: release each tower before loading the next.
        with closing(OnnxModelSession(package[name], device=device)) as session:
            if device == "cuda":
                assert session._session.get_providers()[0] == "CUDAExecutionProvider"
            return session.run(inputs)

    actual_embeddings = {}
    with np.load(root / "embeddinggemma-2.npz", allow_pickle=False) as arrays:
        for sample in metadata["samples"]:
            name = sample["name"]
            if level == "L5" and name != "retrieval":
                continue
            inputs = {key: arrays[f"{name}/input/{key}"] for key in sample["input_names"]}
            ids = inputs["input_ids"]
            empty = np.zeros((0, package.config.hidden_size), dtype=np.float32)
            features = {
                "image_features": empty,
                "video_features": empty,
                "audio_features": empty,
            }
            for pixels_name, positions_name, output_name in (
                ("pixel_values", "image_position_ids", "image_features"),
                ("pixel_values_videos", "video_position_ids", "video_features"),
            ):
                if pixels_name in inputs:
                    features[output_name] = np.concatenate(
                        [
                            run_component(
                                "vision_encoder",
                                {
                                    "pixel_values": pixels[None],
                                    "pixel_position_ids": positions[None],
                                },
                            )["image_features"]
                            for pixels, positions in zip(
                                inputs[pixels_name], inputs[positions_name], strict=True
                            )
                        ],
                        axis=0,
                    )
            if "input_features" in inputs:
                features["audio_features"] = run_component(
                    "audio_encoder",
                    {
                        "input_features": inputs["input_features"],
                        "input_features_mask": inputs["input_features_mask"],
                    },
                )["audio_features"]
            positions = np.broadcast_to(np.arange(ids.shape[1], dtype=np.int64), ids.shape)
            outputs = run_component(
                "model",
                {
                    "input_ids": ids,
                    "attention_mask": inputs["attention_mask"],
                    "position_ids": positions.copy(),
                    **features,
                },
            )
            assert np.isfinite(outputs["sentence_embedding"]).all()
            np.testing.assert_allclose(
                np.linalg.norm(outputs["sentence_embedding"], axis=-1),
                1.0,
                atol=1e-5,
            )
            for output_name in ("last_hidden_state", "sentence_embedding"):
                expected = arrays[f"{name}/output/{output_name}"]
                assert outputs[output_name].shape == expected.shape
                np.testing.assert_allclose(
                    outputs[output_name],
                    expected,
                    atol=1e-3,
                    rtol=1e-3,
                    err_msg=f"{name}/{output_name}",
                )
            actual_embeddings[name] = outputs["sentence_embedding"]
    if level == "L5":
        semantic = json.loads((root / "embeddinggemma-2_retrieval.json").read_text())
        embeddings = actual_embeddings["retrieval"]
        for dim in (128, 256, 512, 768):
            truncated = embeddings[:, :dim]
            truncated = truncated / np.linalg.norm(truncated, axis=-1, keepdims=True)
            scores = truncated[:3] @ truncated[3:].T
            ranking = np.argsort(-scores, axis=-1)
            expected = np.asarray(semantic["rankings"][str(dim)], dtype=np.int64)
            np.testing.assert_array_equal(ranking, expected)
            np.testing.assert_array_equal(ranking[:, 0], np.arange(3))
