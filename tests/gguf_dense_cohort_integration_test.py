# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

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


_ARTIFACTS = (
    _Artifact(
        architecture="olmo",
        repo_id="QuantFactory/AMD-OLMo-1B-GGUF",
        revision="5f34243a42dbae2141b8f5286320bf63d51eeefb",
        filename="AMD-OLMo-1B.Q4_K_M.gguf",
        size=733_520_128,
        sha256="2a848051ef7a3edfd829ce915835794e789e6ed7f425066c242759b8dbc645b4",
        qtypes={"Q4_K": 96, "Q6_K": 17},
    ),
    _Artifact(
        architecture="olmo2",
        repo_id="allenai/OLMo-2-0425-1B-Instruct-GGUF",
        revision="62f8c199538474c3e33ed5d7e0580abd66686a27",
        filename="OLMo-2-0425-1B-Instruct-Q4_K_M.gguf",
        size=935_515_296,
        sha256="abd8187934a438fbf7cfff0a1de5b9d2793ce913f158794df1951dcba6c93cc6",
        qtypes={"F32": 65, "Q4_K": 97, "Q6_K": 17},
    ),
    _Artifact(
        architecture="smollm3",
        repo_id="ggml-org/SmolLM3-3B-GGUF",
        revision="4965cb60b150737b68a0408c36aeefb65078f894",
        filename="SmolLM3-Q4_K_M.gguf",
        size=1_915_305_312,
        sha256="8334b850b7bd46238c16b0c550df2138f0889bf433809008cc17a8b05761863e",
        qtypes={"F32": 73, "Q4_K": 216, "Q6_K": 37},
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
def test_real_dense_gguf_artifact_reports_lossy_requantization(
    artifact: _Artifact, caplog
) -> None:
    """Pinned Q4_K_M artifacts remain packed without a source-fidelity claim."""
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

    with caplog.at_level("WARNING"):
        package = build_from_gguf(path)
    report = package.gguf_quantization_report
    assert caplog.text.count("GGUF QUANTIZATION FIDELITY WARNING") == 1
    assert report.storage_quantized is True
    assert report.source_fidelity is False
    assert report.converted_from == "Q4_K_M-like mixed GGUF"
    assert {stat.qtype for stat in report.source_qtype_census} == set(artifact.qtypes)
