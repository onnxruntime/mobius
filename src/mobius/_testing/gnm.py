# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Full-checkpoint GNM mesh fixtures and continuous-output golden replay."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np

GNM_REFERENCE_REVISION = "dfd3e4b96bda2f0061c6013a458b435cf3cb519f"
GNM_GOLDEN_DIR = Path("testdata/golden/3d")


def gnm_inputs(level: str) -> dict[str, np.ndarray]:
    """Exercise all basis rows, zero/tiny rotations, eye/head articulation and motion."""
    rng = np.random.default_rng(20261116)
    batch = 4 if level == "L4" else 24
    identity = rng.normal(0, 0.1, (batch, 253)).astype(np.float32)
    expression = rng.normal(0, 0.05, (batch, 383)).astype(np.float32)
    rotations = rng.uniform(-0.3, 0.3, (batch, 4, 3)).astype(np.float32)
    translation = rng.uniform(-0.1, 0.1, (batch, 3)).astype(np.float32)
    identity[0] = expression[0] = rotations[0] = translation[0] = 0
    rotations[1] = 1e-7
    if level == "L5":
        # A smooth 24-frame articulation, including independent eye rotations.
        phase = np.linspace(0, 2 * np.pi, batch, dtype=np.float32)
        identity[:] = identity[2]
        expression[:] = expression[2]
        rotations[:] = rotations[2]
        expression *= np.sin(phase[:, None])
        rotations *= np.sin(phase[:, None, None])
        translation[:] = 0
        translation[:, 0] = np.linspace(-0.1, 0.1, batch, dtype=np.float32)
    return {
        "identity": identity,
        "expression": expression,
        "rotations": rotations,
        "translation": translation,
    }


def gnm_semantic_inputs() -> dict[str, np.ndarray]:
    """Neutral, rigid translation, head, left-eye and right-eye articulation."""
    feeds = {
        "identity": np.zeros((5, 253), dtype=np.float32),
        "expression": np.zeros((5, 383), dtype=np.float32),
        "rotations": np.zeros((5, 4, 3), dtype=np.float32),
        "translation": np.zeros((5, 3), dtype=np.float32),
    }
    feeds["translation"][1] = [0.03, -0.02, 0.01]
    feeds["rotations"][2, 1, 1] = 0.2
    feeds["rotations"][3, 2, 1] = 0.2
    feeds["rotations"][4, 3, 0] = -0.2
    return feeds


def assert_gnm_golden(case, level: str) -> None:
    """Compare every generated vertex after package save/load on the selected EP."""
    import onnxruntime as ort

    from mobius import ModelPackage, build
    from mobius.models.gnm import GNM_SHA256

    suffix = "" if level == "L4" else "_generation"
    manifest_path = GNM_GOLDEN_DIR / f"{case.case_id}{suffix}.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["revision"] == case.revision
    assert manifest["checkpoint_sha256"] == GNM_SHA256
    assert manifest["reference_revision"] == GNM_REFERENCE_REVISION
    assert manifest["reference_device"].startswith("cuda")
    assert manifest["level"] == level
    golden_path = manifest_path.with_suffix(".npz")
    assert hashlib.sha256(golden_path.read_bytes()).hexdigest() == manifest["fixture_sha256"]
    with np.load(golden_path, allow_pickle=False) as archive:
        expected = archive["vertices"]
        feeds = {name: archive[name] for name in gnm_inputs(level)}
        if level == "L5":
            expected_semantics = archive["semantic_vertices"]
            left_eye, right_eye = archive["left_eye"], archive["right_eye"]
    for name, value in gnm_inputs(level).items():
        np.testing.assert_array_equal(feeds[name], value)
    requested_device = os.environ.get("MOBIUS_TEST_DEVICE", "cpu")
    provider = os.environ.get("MOBIUS_TEST_EP") or (
        "CUDAExecutionProvider" if requested_device == "cuda" else "CPUExecutionProvider"
    )
    if provider not in ort.get_available_providers():
        raise RuntimeError(
            f"GNM golden requires {provider}; install a compatible onnxruntime-gpu wheel "
            "and its CUDA/cuDNN libraries before retrying."
        )
    package = build(
        case.model_id,
        revision=case.revision,
        dtype="f32",
        execution_provider=os.environ.get("MOBIUS_TEST_BUILD_EP", "default"),
    )
    with tempfile.TemporaryDirectory(prefix="mobius-gnm-") as directory:
        package.save(directory, max_workers=1)
        restored = ModelPackage.load(directory)
        assert restored["model"].metadata_props["mobius.source_revision"] == case.revision
        restored_directory = Path(directory) / "restored"
        restored.save(restored_directory, max_workers=1)
        options = ort.SessionOptions()
        options.enable_profiling = requested_device == "cuda"
        options.profile_file_prefix = str(Path(directory) / "profile")
        session = ort.InferenceSession(
            str(restored_directory / "model.onnx"),
            sess_options=options,
            providers=[provider],
        )
        assert session.get_providers()[0] == provider
        chunks = []
        block_size = 4 if level == "L5" else 2
        for start in range(0, len(feeds["identity"]), block_size):
            block = {name: value[start : start + block_size] for name, value in feeds.items()}
            chunks.append(session.run(["vertices"], block)[0])
        actual = np.concatenate(chunks)
        assert actual.shape == expected.shape == (4 if level == "L4" else 24, 17821, 3)
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=2e-6)
        assert np.isfinite(actual).all()
        # Translation is a rigid motion, not an identity or articulation control.
        shifted = dict(block)
        shift = np.array([0.03, -0.02, 0.01], dtype=np.float32)
        shifted["translation"] = block["translation"] + shift
        shifted_vertices = session.run(["vertices"], shifted)[0]
        np.testing.assert_allclose(shifted_vertices, chunks[-1] + shift, rtol=1e-4, atol=2e-6)
        assert float(np.max(np.ptp(actual, axis=0))) > 0.01
        if level == "L5":
            centered = actual - feeds["translation"][:, None, :]
            np.testing.assert_allclose(centered[0], centered[-1], rtol=1e-4, atol=2e-6)
            amplitude = float(np.max(np.ptp(centered, axis=0)))
            assert amplitude > 1e-4
            assert float(np.max(np.abs(np.diff(centered, axis=0)))) < 0.4 * amplitude
            assert float(np.max(np.abs(np.diff(centered, n=2, axis=0)))) < 0.2 * amplitude
            semantic_feeds = gnm_semantic_inputs()
            semantics = session.run(["vertices"], semantic_feeds)[0]
            assert semantics.shape == expected_semantics.shape == (5, 17821, 3)
            np.testing.assert_allclose(semantics, expected_semantics, rtol=1e-4, atol=2e-6)
            np.testing.assert_allclose(
                semantics[1],
                semantics[0] + semantic_feeds["translation"][1],
                rtol=1e-4,
                atol=2e-6,
            )
            assert float(np.max(np.abs(semantics[2] - semantics[0]))) > 1e-3
            for moved, stationary, row in ((left_eye, right_eye, 3), (right_eye, left_eye, 4)):
                assert moved.any() and stationary.any()
                assert (
                    float(np.max(np.abs(semantics[row, moved] - semantics[0, moved]))) > 1e-4
                )
                np.testing.assert_allclose(
                    semantics[row, stationary], semantics[0, stationary], rtol=1e-4, atol=2e-6
                )
        if requested_device == "cuda":
            profile_path = session.end_profiling()
            profile = json.loads(Path(profile_path).read_text())
            assigned = {
                event["args"].get("provider")
                for event in profile
                if event.get("args", {}).get("op_name")
                in {"MatMul", "Einsum", "FusedMatMul", "Gemm"}
            }
            assert assigned == {"CUDAExecutionProvider"}, assigned
            Path(profile_path).unlink()
