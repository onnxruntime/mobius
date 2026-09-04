# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Artifact identities and import verdicts for VibeVoice ASR BitNet.

The official VibeVoice ASR BitNet release contains an F32 safetensors
conversion source alongside two execution-only GGUF files. The latter are not
ordinary llama.cpp quantizations: ``I2_S`` needs VibeASR.cpp's packed ternary
and activation kernels, and ``I8_S`` needs its fused VAE kernels. This module
centralizes that artifact-level distinction without teaching generic ONNX
components about a model-specific execution format. The dense safetensors
conversion source is recorded separately so its ONNX path cannot be confused
with VibeASR.cpp-native quantized execution.
"""

from __future__ import annotations

__all__ = [
    "VIBEVOICE_ASR_BITNET_DENSE_F32_TENSOR_COUNT",
    "VIBEVOICE_ASR_BITNET_DENSE_F32_VALUE_COUNT",
    "VIBEVOICE_ASR_BITNET_DENSE_SAFETENSORS",
    "VIBEVOICE_ASR_BITNET_ARTIFACTS",
    "VIBEVOICE_ASR_BITNET_REPOSITORY",
    "VIBEVOICE_ASR_BITNET_REVISION",
    "build_vibeasr_bitnet_dense_weight_plan",
    "is_vibeasr_bitnet_conversion_source",
    "VibeASRBitNetGGUFArtifact",
    "VibeASRBitNetSafetensorsArtifact",
    "find_vibeasr_bitnet_gguf_artifact",
    "reject_vibeasr_bitnet_gguf",
]

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import PurePath
from typing import TYPE_CHECKING

from mobius.integrations.gguf._errors import VibeASRBitNetGGUFImportError
from mobius.integrations.gguf._header import GGUFHeaderInfo

if TYPE_CHECKING:
    import onnx_ir as ir

    from mobius.integrations._weight_loading import StreamingWeightPlan
    from mobius.models.vibevoice_asr import VibeVoiceASRForConditionalGeneration

VIBEVOICE_ASR_BITNET_REPOSITORY = "microsoft/VibeVoice-ASR-BitNet"
VIBEVOICE_ASR_BITNET_REVISION = "66e78021ab8f5f06133d1ab421ba4d348bda97c9"
VIBEVOICE_ASR_BITNET_DENSE_F32_TENSOR_COUNT = 1_177
VIBEVOICE_ASR_BITNET_DENSE_F32_VALUE_COUNT = 2_814_116_321
_VIBEVOICE_ASR_BITNET_INDEX_REPORTED_PARAMETER_COUNT = 322_592_829
_VIBEASR_LM_FILE_TYPE = 40
_VIBEASR_VAE_FILE_TYPE = 41
_VIBEASR_I2_S_TYPE_ID = 36
_VIBEASR_I8_S_TYPE_ID = 37


@dataclass(frozen=True, slots=True)
class VibeASRBitNetSafetensorsArtifact:
    """Immutable identity of one dense F32 conversion-source shard."""

    filename: str
    size_bytes: int
    sha256: str


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


VIBEVOICE_ASR_BITNET_DENSE_SAFETENSORS = (
    VibeASRBitNetSafetensorsArtifact(
        filename="model-00001-of-00003.safetensors",
        size_bytes=4_996_674_400,
        sha256="58cb328634bb4b7e5afcc4f14c43261a0c636c9031b0b085bd3ef53c131aaf19",
    ),
    VibeASRBitNetSafetensorsArtifact(
        filename="model-00002-of-00003.safetensors",
        size_bytes=4_963_184_724,
        sha256="410349401256a9a995424408f8f77a21cad5e53183e07207820d7b25e84682df",
    ),
    VibeASRBitNetSafetensorsArtifact(
        filename="model-00003-of-00003.safetensors",
        size_bytes=1_296_761_656,
        sha256="49c2d7591f9dcbbd6fb37baa58379211f21df30dbd3c6ff39cb8c81473660cdb",
    ),
)


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


def is_vibeasr_bitnet_conversion_source(model_id: str, revision: str | None) -> bool:
    """Return whether the requested immutable source is the audited F32 release."""
    return (
        model_id == VIBEVOICE_ASR_BITNET_REPOSITORY
        and revision == VIBEVOICE_ASR_BITNET_REVISION
    )


def build_vibeasr_bitnet_dense_weight_plan(
    model: VibeVoiceASRForConditionalGeneration,
    source_tensors: Mapping[str, tuple[str, list[int], str]],
    _initializers: Mapping[str, ir.Value],
) -> StreamingWeightPlan:
    """Classify every source tensor for the staged dense-F32 conversion route.

    The parent ASR module remains authoritative for HF-to-ONNX name alignment.
    Marker tensors exercise that mapping without materializing any checkpoint
    values; every non-decoder source must map to an exported initializer.
    """
    import torch

    from mobius.integrations._weight_loading import StreamingWeightPlan, StreamingWeightSource

    if len(source_tensors) != VIBEVOICE_ASR_BITNET_DENSE_F32_TENSOR_COUNT:
        raise ValueError(
            "VibeVoice ASR BitNet dense source tensor count changed: expected "
            f"{VIBEVOICE_ASR_BITNET_DENSE_F32_TENSOR_COUNT}, got {len(source_tensors)}."
        )
    non_f32 = {
        source_name: source_dtype
        for source_name, (_, _, source_dtype) in source_tensors.items()
        if source_dtype != "F32"
    }
    if non_f32:
        examples = sorted(non_f32.items())[:5]
        raise ValueError(
            "VibeVoice ASR BitNet dense conversion source must contain only F32 tensors; "
            f"found {examples}."
        )
    value_count = sum(math.prod(shape) for _, shape, _ in source_tensors.values())
    if value_count != VIBEVOICE_ASR_BITNET_DENSE_F32_VALUE_COUNT:
        raise ValueError(
            "VibeVoice ASR BitNet dense source value count changed: expected "
            f"{VIBEVOICE_ASR_BITNET_DENSE_F32_VALUE_COUNT}, got {value_count}."
        )

    markers = {name: torch.empty(0) for name in source_tensors}
    mapped = model.preprocess_weights(markers)
    marker_sources = {id(marker): name for name, marker in markers.items()}
    targets: dict[str, StreamingWeightSource] = {}
    for target_name, marker in mapped.items():
        source_name = marker_sources.get(id(marker))
        if source_name is None:
            raise ValueError(
                "VibeVoice ASR BitNet weight preprocessing transformed a source marker for "
                f"{target_name!r}; streaming requires a one-to-one source tensor mapping."
            )
        if target_name in targets:
            raise ValueError(
                f"VibeVoice ASR BitNet maps multiple source tensors to {target_name!r}."
            )
        targets[target_name] = StreamingWeightSource(
            source_name=source_name,
            expected_dtype="F32",
        )

    used_sources = {source.source_name for source in targets.values()}
    ignored: dict[str, str] = {}
    for source_name in source_tensors:
        if source_name in used_sources:
            continue
        if source_name.startswith("model.acoustic_tokenizer.decoder."):
            ignored[source_name] = (
                "The source acoustic VAE decoder is not an ASR inference stage."
            )
            continue
        raise ValueError(
            f"VibeVoice ASR BitNet source tensor {source_name!r} is not classified by "
            "the staged inference package."
        )

    return StreamingWeightPlan(
        targets=targets,
        ignored=ignored,
        report={
            "source_format": "safetensors",
            "source_storage_dtype": "float32",
            "source_tensor_count": len(source_tensors),
            "source_value_count": value_count,
            "source_parameter_count_status": (
                "The checkpoint index's reported 322592829 parameter count is inconsistent "
                "with the exact dense-F32 tensor-byte census (2814116321 values); the latter "
                "is authoritative for this export."
            ),
            "index_reported_parameter_count": _VIBEVOICE_ASR_BITNET_INDEX_REPORTED_PARAMETER_COUNT,
            "native_bitnet_execution": False,
            "native_gguf_disposition": (
                "not imported; VibeASR.cpp I2_S/I8_S kernels and packing are unsupported"
            ),
            "native_gguf_artifacts": [
                {
                    "filename": artifact.filename,
                    "sha256": artifact.sha256,
                    "disposition": "unsupported_native_execution",
                }
                for artifact in VIBEVOICE_ASR_BITNET_ARTIFACTS
            ],
        },
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
