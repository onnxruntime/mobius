# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Identity and import verdicts for the VibeVoice ASR BitNet GGUF artifacts.

The official VibeVoice ASR BitNet release contains an F32 safetensors
conversion source alongside two execution-only GGUF files. The latter are not
ordinary llama.cpp quantizations: ``I2_S`` needs VibeASR.cpp's packed ternary
and activation kernels, and ``I8_S`` needs its fused VAE kernels. This module
centralizes that artifact-level distinction without teaching generic ONNX
components about a model-specific execution format.
"""

from __future__ import annotations

__all__ = [
    "VIBEVOICE_ASR_BITNET_ARTIFACTS",
    "VIBEVOICE_ASR_BITNET_REPOSITORY",
    "VIBEVOICE_ASR_BITNET_REVISION",
    "VibeASRBitNetGGUFArtifact",
    "find_vibeasr_bitnet_gguf_artifact",
    "reject_vibeasr_bitnet_gguf",
]

from dataclasses import dataclass
from pathlib import PurePath

from mobius.integrations.gguf._errors import VibeASRBitNetGGUFImportError
from mobius.integrations.gguf._header import GGUFHeaderInfo

VIBEVOICE_ASR_BITNET_REPOSITORY = "microsoft/VibeVoice-ASR-BitNet"
VIBEVOICE_ASR_BITNET_REVISION = "66e78021ab8f5f06133d1ab421ba4d348bda97c9"
_VIBEASR_LM_FILE_TYPE = 40
_VIBEASR_VAE_FILE_TYPE = 41
_VIBEASR_I2_S_TYPE_ID = 36
_VIBEASR_I8_S_TYPE_ID = 37


@dataclass(frozen=True, slots=True)
class VibeASRBitNetGGUFArtifact:
    """Pinned fingerprint and native execution contract of one GGUF artifact."""

    filename: str
    role: str
    architecture: str
    tensor_count: int
    size_bytes: int
    sha256: str
    native_format: str
    blocker: str
    file_type: int
    tensor_type_ids: frozenset[int]


_LM_BLOCKER = (
    "The I2_S language-model projections use VibeASR.cpp's packed ternary code "
    "layout, tensor-sidecar scales, and ISA-specific I2_S-by-I8_S activation kernel. "
    "ORT MatMulNBits is affine and cannot preserve or execute this contract."
)
_VAE_BLOCKER = (
    "The I8_S acoustic/semantic encoder and connector weights rely on VibeASR.cpp's "
    "all-INT8 fused convolution, MatMul, RMSNorm, and residual kernels. There is no "
    "lossless ONNX Runtime import or execution provider for that contract."
)

VIBEVOICE_ASR_BITNET_ARTIFACTS = (
    VibeASRBitNetGGUFArtifact(
        filename="vibeasr-lm-i2_s-embed-q6_k.gguf",
        role="decoder",
        architecture="qwen2",
        tensor_count=339,
        size_bytes=992_877_600,
        sha256="fbe273d8dc2f2433bb25f849e19d77ea65aaa2188d12c20cee987ab6f321e002",
        native_format="I2_S ternary projections with Q6_K embedding/head",
        blocker=_LM_BLOCKER,
        file_type=_VIBEASR_LM_FILE_TYPE,
        tensor_type_ids=frozenset({0, 1, 14, _VIBEASR_I2_S_TYPE_ID}),
    ),
    VibeASRBitNetGGUFArtifact(
        filename="vibeasr-vae-encoder-i8_s.gguf",
        role="acoustic, semantic, and connector stages",
        architecture="vibeasr-vae",
        tensor_count=562,
        size_bytes=703_080_064,
        sha256="4941c82608c253ec066b5cc74d3dd11a5c8fef96cccbc5b87359ef0fe4338df6",
        native_format="I8_S fused VAE encoder and connector kernels",
        blocker=_VAE_BLOCKER,
        file_type=_VIBEASR_VAE_FILE_TYPE,
        tensor_type_ids=frozenset({0, _VIBEASR_I8_S_TYPE_ID}),
    ),
)


def _is_vibeasr_lm_header(header: GGUFHeaderInfo) -> bool:
    """Identify the custom LM by its model-owned header identity, not Qwen2 alone."""
    return (
        header.architecture == "qwen2"
        and header.file_type == _VIBEASR_LM_FILE_TYPE
        and _VIBEASR_I2_S_TYPE_ID in header.tensor_type_ids
    )


def _is_vibeasr_vae_header(header: GGUFHeaderInfo) -> bool:
    """Identify the VAE only when its custom all-INT8 storage is present."""
    return (
        header.architecture == "vibeasr-vae"
        and header.file_type == _VIBEASR_VAE_FILE_TYPE
        and _VIBEASR_I8_S_TYPE_ID in header.tensor_type_ids
    )


def find_vibeasr_bitnet_gguf_artifact(
    *,
    repository: str | None = None,
    revision: str | None = None,
    filename: str | None = None,
    size_bytes: int | None = None,
    sha256: str | None = None,
    header: GGUFHeaderInfo | None = None,
) -> VibeASRBitNetGGUFArtifact | None:
    """Return a verified VibeASR native artifact, never a generic Qwen2 alias.

    An exact Hub artifact is matched by repository, immutable revision, basename,
    and optional size/checksum. A local file is recognized only by VibeASR-owned
    header identity; the generic ``qwen2`` architecture is intentionally
    insufficient because ordinary Qwen2 GGUF files remain supported.
    """
    basename = PurePath(filename).name if filename else None
    if (
        repository == VIBEVOICE_ASR_BITNET_REPOSITORY
        and revision == VIBEVOICE_ASR_BITNET_REVISION
        and basename is not None
    ):
        for artifact in VIBEVOICE_ASR_BITNET_ARTIFACTS:
            if basename != artifact.filename:
                continue
            if size_bytes is not None and size_bytes != artifact.size_bytes:
                continue
            if sha256 is not None and sha256.casefold() != artifact.sha256:
                continue
            return artifact
    if header is not None:
        for artifact in VIBEVOICE_ASR_BITNET_ARTIFACTS:
            is_match = (
                header.architecture == artifact.architecture
                and header.file_type == artifact.file_type
                and (
                    _is_vibeasr_vae_header(header)
                    if artifact.architecture == "vibeasr-vae"
                    else _is_vibeasr_lm_header(header)
                )
            )
            if is_match:
                return artifact
    return None


def reject_vibeasr_bitnet_gguf(
    *,
    source: str,
    repository: str | None = None,
    revision: str | None = None,
    filename: str | None = None,
    size_bytes: int | None = None,
    sha256: str | None = None,
    header: GGUFHeaderInfo | None = None,
) -> None:
    """Fail closed before config extraction or tensor payload access."""
    artifact = find_vibeasr_bitnet_gguf_artifact(
        repository=repository,
        revision=revision,
        filename=filename,
        size_bytes=size_bytes,
        sha256=sha256,
        header=header,
    )
    if artifact is None:
        return
    raise VibeASRBitNetGGUFImportError(
        f"Direct GGUF import is unsupported for VibeVoice ASR BitNet {artifact.role} "
        f"artifact {artifact.filename!r} ({artifact.native_format}) from {source!r}. "
        f"{artifact.blocker} Build the pinned Hugging Face safetensors checkpoint with "
        "`mobius build` instead; that route uses the release's dense F32 conversion "
        "source and does not claim native BitNet/GGUF preservation or execution. "
        "No ONNX artifacts were emitted."
    )
