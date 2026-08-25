# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest
from huggingface_hub import hf_hub_download

from mobius.integrations.gguf import build_from_gguf
from mobius.integrations.gguf._reader import GGUFModel


@dataclass(frozen=True)
class _Artifact:
    architecture: str
    repo_id: str
    revision: str
    filename: str
    size: int
    sha256: str
    qtypes: dict[str, int]
    greedy_token: int


_ARTIFACTS = (
    _Artifact(
        architecture="olmo",
        repo_id="QuantFactory/AMD-OLMo-1B-GGUF",
        revision="5f34243a42dbae2141b8f5286320bf63d51eeefb",
        filename="AMD-OLMo-1B.Q4_K_M.gguf",
        size=733_520_128,
        sha256="2a848051ef7a3edfd829ce915835794e789e6ed7f425066c242759b8dbc645b4",
        qtypes={"Q4_K": 96, "Q6_K": 17},
        greedy_token=187,
    ),
    _Artifact(
        architecture="olmo2",
        repo_id="allenai/OLMo-2-0425-1B-Instruct-GGUF",
        revision="62f8c199538474c3e33ed5d7e0580abd66686a27",
        filename="OLMo-2-0425-1B-Instruct-Q4_K_M.gguf",
        size=935_515_296,
        sha256="abd8187934a438fbf7cfff0a1de5b9d2793ce913f158794df1951dcba6c93cc6",
        qtypes={"F32": 65, "Q4_K": 97, "Q6_K": 17},
        greedy_token=16,
    ),
    _Artifact(
        architecture="smollm3",
        repo_id="ggml-org/SmolLM3-3B-GGUF",
        revision="4965cb60b150737b68a0408c36aeefb65078f894",
        filename="SmolLM3-Q4_K_M.gguf",
        size=1_915_305_312,
        sha256="8334b850b7bd46238c16b0c550df2138f0889bf433809008cc17a8b05761863e",
        qtypes={"F32": 73, "Q4_K": 216, "Q6_K": 37},
        greedy_token=12_286,
    ),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@pytest.mark.integration
@pytest.mark.parametrize("artifact", _ARTIFACTS, ids=lambda artifact: artifact.architecture)
def test_real_dense_gguf_artifact(artifact: _Artifact, tmp_path: Path) -> None:
    """Pinned mixed-qtype GGUFs import through the asserted affine-requantization route."""
    path = Path(
        hf_hub_download(
            repo_id=artifact.repo_id,
            revision=artifact.revision,
            filename=artifact.filename,
        )
    )
    assert path.stat().st_size == artifact.size
    assert _sha256(path) == artifact.sha256

    gguf_model = GGUFModel(path)
    qtypes = Counter(qtype.name for _, _, qtype, _ in gguf_model.tensor_items_raw())
    assert dict(sorted(qtypes.items())) == artifact.qtypes

    package = build_from_gguf(path)
    op_types = {node.op_type for node in package["model"].graph}
    # Q4_K/Q6_K cannot be preserved by MatMulNBits. They are dequantized and
    # affine-requantized to explicit-zero-point 4-bit/block-32 weights.
    assert "MatMulNBits" in op_types
    assert "GatherBlockQuantized" in op_types
    assert "BlockQuantizedMatMul" not in op_types

    output_dir = tmp_path / artifact.architecture
    package.save(output_dir, progress_bar=False)
    session = ort.InferenceSession(
        str(output_dir / "model.onnx"),
        providers=["CPUExecutionProvider"],
    )
    feeds: dict[str, np.ndarray] = {}
    for model_input in session.get_inputs():
        if model_input.name == "input_ids":
            feeds[model_input.name] = np.array([[1, 2]], dtype=np.int64)
        elif model_input.name == "attention_mask":
            feeds[model_input.name] = np.ones((1, 2), dtype=np.int64)
        elif model_input.name == "position_ids":
            feeds[model_input.name] = np.array([[0, 1]], dtype=np.int64)
        elif model_input.name.endswith((".key", ".value")):
            feeds[model_input.name] = np.empty(
                [1, model_input.shape[1], 0, model_input.shape[3]],
                dtype=np.float32,
            )

    first = session.run(["logits"], feeds)[0]
    second = session.run(["logits"], feeds)[0]
    assert np.isfinite(first).all()
    np.testing.assert_array_equal(first, second)
    assert int(first[0, -1].argmax()) == artifact.greedy_token
