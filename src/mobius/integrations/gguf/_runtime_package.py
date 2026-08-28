# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Emit a runnable package from a GGUF build.

Saving the ONNX graph is not enough to run a model: the runtime also needs a
tokenizer and its own configuration contract. Those come from two different
places — the tokenizer from the GGUF's embedded ggml metadata, the contract
from the built package — so a caller that saves the graph and stops produces a
directory that loads nowhere.

This module is the single place that knows the full artifact set, so the CLI
and the Python API cannot drift apart on what a complete package contains, and
so both supported runtimes are reachable from one entry point.
"""

from __future__ import annotations

import ctypes
import json
import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Literal

from mobius.integrations.gguf._arch_registry import get_arch_spec
from mobius.integrations.gguf._runtime_evidence import (
    gguf_artifact_identity,
    gguf_graph_package_identity,
    matching_runtime_evidence,
)
from mobius.integrations.gguf._shard_set import open_gguf_model
from mobius.integrations.gguf._spec import Support
from mobius.integrations.gguf._tokenizer import (
    GGUFTokenizerAsset,
    GGUFTokenizerSource,
    inspect_gguf_tokenizer,
    materialize_gguf_tokenizer,
    write_gguf_tokenizer_json,
)

__all__ = ["write_gguf_runtime_package"]

_LOGGER = logging.getLogger(__name__)


def _publish_directory_no_replace(stage: Path, destination: Path) -> None:
    """Atomically publish a directory while refusing an existing destination."""
    if sys.platform == "linux":
        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = getattr(libc, "renameat2", None)
        if renameat2 is None:
            raise OSError(
                "Atomic no-replace directory publication requires renameat2 on Linux"
            )
        renameat2.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameat2.restype = ctypes.c_int
        result = renameat2(
            -100,
            os.fsencode(stage),
            -100,
            os.fsencode(destination),
            1,
        )
    elif sys.platform == "darwin":
        libc = ctypes.CDLL(None, use_errno=True)
        renamex_np = libc.renamex_np
        renamex_np.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        renamex_np.restype = ctypes.c_int
        result = renamex_np(os.fsencode(stage), os.fsencode(destination), 0x00000004)
    elif os.name == "nt":
        os.rename(stage, destination)
        return
    else:
        raise OSError(
            f"Atomic no-replace directory publication is unsupported on {sys.platform!r}"
        )
    if result != 0:
        error_number = ctypes.get_errno()
        raise OSError(
            error_number,
            os.strerror(error_number),
            str(destination),
        )


Runtime = Literal["onnx-genai", "ort-genai"]


def write_gguf_runtime_package(
    pkg: Any,
    gguf_path: str | Path,
    output_dir: str | Path,
    *,
    runtime: Runtime = "onnx-genai",
    runtime_version: str | None = None,
    tokenizer_repository: str | None = None,
    tokenizer_revision: str | None = None,
    local_files_only: bool = False,
    save_model: bool = True,
    **save_kwargs: Any,
) -> dict[str, str]:
    """Write a faithful runtime package without making runtime evidence an export gate.

    The graph and configuration contract are always emitted when Mobius can
    represent them correctly. Runtime evidence and processor availability are
    recorded as validation metadata and warnings rather than admission checks.

    Args:
        pkg: The :class:`~mobius.ModelPackage` returned by
            :func:`~mobius.integrations.gguf.build_from_gguf`.
        gguf_path: The source ``.gguf`` file.
        output_dir: Destination directory.
        runtime: Which runtime contract to emit. ``"onnx-genai"`` writes
            ``inference_metadata.yaml``; ``"ort-genai"`` writes
            ``genai_config.json``.
        runtime_version: Optional downstream runtime version to assess.
        tokenizer_repository: Optional Hub repository holding tokenizer assets.
        tokenizer_revision: Optional immutable tokenizer repository revision.
        local_files_only: Resolve tokenizer assets only from the local Hub cache.
        save_model: Must remain ``True``. Existing graph directories cannot be
            associated with the build-time evidence transaction safely.
        **save_kwargs: Forwarded to :meth:`ModelPackage.save`.

    Returns:
        Mapping of artifact name to written path.

    Raises:
        ValueError: If ``runtime`` is not a supported runtime name.
    """
    if runtime not in ("onnx-genai", "ort-genai"):
        raise ValueError(f"Unknown runtime {runtime!r}; expected 'onnx-genai' or 'ort-genai'.")
    if not save_model:
        raise ValueError(
            "save_model=False is not supported for runtime-evidenced GGUF packages because "
            "an existing graph cannot be bound to the build-time evidence transaction."
        )
    if (tokenizer_repository is None) != (tokenizer_revision is None):
        raise ValueError(
            "tokenizer_repository and tokenizer_revision must be provided together."
        )
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(
            f"GGUF runtime package destination already exists: {output_dir}. "
            "Refusing a non-atomic directory replacement."
        )
    architecture = getattr(pkg, "gguf_architecture", None)
    if not architecture:
        raise ValueError(
            "GGUF runtime packaging requires the canonical source architecture captured "
            "during graph construction."
        )
    architecture_spec = get_arch_spec(architecture)
    import_route = getattr(pkg, "gguf_import_route", None)
    if not import_route:
        raise ValueError(
            "GGUF runtime packaging requires the exact import route captured during "
            "graph construction."
        )
    built_identity = getattr(pkg, "gguf_artifact_identity", None)
    if built_identity is None:
        raise ValueError(
            "GGUF runtime packaging requires the immutable source identity captured during "
            "graph construction."
        )
    if runtime == "ort-genai" and getattr(pkg, "gguf_reuse_plan", None) is not None:
        raise ValueError(
            "ORT GenAI packaging is not supported with reused GGUF weights because "
            "genai_config.json has no supported setting that disables ORT constant "
            "folding. Use direct ONNX Runtime with ORT_DISABLE_ALL, or build without "
            "reuse_gguf_weights."
        )
    draft_manifest = getattr(pkg, "draft_manifest", None)
    mtp_head = getattr(pkg, "mtp_head", None)

    source_path = Path(getattr(pkg, "gguf_source_path", gguf_path))
    source_model = open_gguf_model(source_path)
    if not source_model.source_matches_path():
        raise ValueError(
            "The GGUF source changed while the canonical reader was opening it; "
            "refusing runtime publication."
        )
    source_spec = get_arch_spec(source_model.architecture)
    source_architecture = getattr(source_spec, "gguf_arch", source_model.architecture)
    if source_architecture != architecture:
        raise ValueError(
            "The GGUF source architecture no longer matches the canonical architecture "
            f"captured during graph construction: built={architecture!r}, "
            f"current={source_architecture!r}."
        )
    current_identity = gguf_artifact_identity(
        source_path,
        source_model,
        architecture=architecture,
        filename=built_identity.filename,
    )
    if current_identity != built_identity:
        raise ValueError(
            "The GGUF source no longer matches the exact artifact identity captured during "
            f"graph construction: built={built_identity!r}, current={current_identity!r}."
        )
    if not source_model.source_matches_path():
        raise ValueError(
            "The GGUF source changed while its artifact identity was being validated; "
            "refusing runtime publication."
        )
    evidence = None
    if (
        tokenizer_repository is not None
        and tokenizer_revision is not None
        and architecture_spec.runtime_evidence_ids
    ):
        try:
            evidence = matching_runtime_evidence(
                architecture_spec.runtime_evidence_ids,
                architecture=architecture,
                runtime=runtime,
                source_path=source_path,
                gguf_model=source_model,
                built_identity=built_identity,
                import_route=import_route,
                runtime_version=runtime_version,
                tokenizer_repository=tokenizer_repository,
                tokenizer_revision=tokenizer_revision,
            )
        except ValueError as error:
            if not (
                str(error).startswith("No unique GGUF runtime evidence matches")
                or str(error)
                == "Runtime packaging requires the exact runtime version covered by evidence."
            ):
                raise
            _LOGGER.warning("%s Export continues without claiming runtime validation.", error)
    if not source_model.source_matches_path():
        raise ValueError(
            "The GGUF source changed while runtime evidence was being matched; "
            "refusing runtime publication."
        )
    source_metadata = source_model.metadata
    verdict = inspect_gguf_tokenizer(
        source_metadata,
        source=str(source_path),
        require_complete=False,
    )
    built_verdict = getattr(pkg, "gguf_tokenizer_verdict", None)
    if built_verdict is None or built_verdict.metadata_sha256 != verdict.metadata_sha256:
        raise ValueError(
            "The GGUF tokenizer metadata no longer matches the identity captured during "
            "graph construction; refusing to pair the graph with a replaced tokenizer source."
        )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", suffix=".tmp", dir=output_dir.parent)
    )
    try:
        artifacts: dict[str, str] = {}
        pkg.save(str(stage), **save_kwargs)
        graph_identity = gguf_graph_package_identity(stage)
        validation_warnings: list[str] = []
        if architecture_spec.runtime is not Support.SUPPORTED:
            validation_warnings.append(
                f"No real-artifact runtime validation is recorded for {architecture!r}: "
                f"{architecture_spec.reason}"
            )
        if evidence is not None and (
            graph_identity.files != evidence.graph_files
            or graph_identity.sha256 != evidence.graph_sha256
        ):
            validation_warnings.append(
                "The serialized graph does not match the recorded runtime-evidence identity."
            )
            evidence = None

        if evidence is not None:
            tokenizer_source = GGUFTokenizerSource(
                repository=evidence.tokenizer_repository,
                revision=evidence.tokenizer_revision,
                metadata_sha256=evidence.tokenizer_metadata_sha256,
                assets=tuple(
                    GGUFTokenizerAsset(filename, size, sha256)
                    for filename, size, sha256 in evidence.tokenizer_assets
                ),
            )
            tokenizer_path = materialize_gguf_tokenizer(
                source_path,
                stage,
                source=tokenizer_source,
                metadata=source_metadata,
                source_identity=(f"sha256:{built_identity.sha256}/{built_identity.filename}"),
                local_files_only=local_files_only,
            )
            artifacts["tokenizer"] = tokenizer_path
        elif verdict.materialized:
            artifacts["tokenizer"] = write_gguf_tokenizer_json(
                source_path,
                stage,
                metadata=source_metadata,
                expected_metadata_sha256=verdict.metadata_sha256,
                source_identity=f"sha256:{built_identity.sha256}/{built_identity.filename}",
            )
        else:
            validation_warnings.append(
                "No faithful tokenizer processor could be materialized from the GGUF source; "
                "the graph and runtime metadata were still exported."
            )

        if runtime == "ort-genai":
            from mobius.integrations.ort_genai import write_ort_genai_config

            execution_provider = getattr(pkg, "gguf_execution_provider", None)
            if not isinstance(execution_provider, str) or not execution_provider:
                raise ValueError(
                    "ORT GenAI runtime packaging requires the graph's non-empty execution "
                    "provider identity."
                )
            # The portable/default graph intentionally keeps standard ONNX operators.
            # ORT GenAI still needs a concrete provider for session construction.
            runtime_execution_provider = (
                "cpu" if execution_provider == "default" else execution_provider
            )
            if execution_provider not in {"default", "cpu", "cuda", "dml"}:
                validation_warnings.append(
                    f"ORT GenAI runtime execution with provider {execution_provider!r} has not "
                    "been validated; the provider identity is preserved in genai_config.json."
                )
            artifacts.update(
                write_ort_genai_config(
                    pkg,
                    str(stage),
                    ep=runtime_execution_provider,
                    runtime_version=runtime_version,
                )
            )
        else:
            from mobius.integrations.onnx_genai import write_onnx_genai_config

            artifacts.update(
                write_onnx_genai_config(
                    pkg, str(stage), config=getattr(pkg, "config", None), source=None
                )
            )
        if draft_manifest is not None:
            from mobius.integrations.gguf._draft import write_draft_manifest

            artifacts["draft_manifest"] = write_draft_manifest(draft_manifest, stage)
        if mtp_head is not None:
            from mobius.integrations.onnx_genai.inference_metadata import (
                write_mtp_speculator_metadata,
            )

            speculator_path = write_mtp_speculator_metadata(
                str(stage),
                backbone_config=getattr(pkg, "config", None),
                proposer_config=getattr(mtp_head, "config", None),
            )
            if speculator_path is not None:
                artifacts["speculator"] = str(speculator_path)
        runtime_identity = gguf_graph_package_identity(stage)
        if evidence is not None and (
            runtime_identity.files != evidence.runtime_package_files
            or runtime_identity.sha256 != evidence.runtime_package_sha256
        ):
            validation_warnings.append(
                "The completed runtime package does not match the recorded runtime-evidence "
                "identity."
            )
            evidence = None
        compatibility_path = stage / "runtime_compatibility.json"
        compatibility = (
            json.loads(compatibility_path.read_text(encoding="utf-8"))
            if compatibility_path.exists()
            else {"runtime": runtime}
        )
        existing_status = compatibility.get("runtime_validation_status")
        if existing_status == "unsupported-by-tested-runtime":
            validation_status = existing_status
        else:
            validation_status = "unvalidated"
        if evidence is not None:
            validation_warnings.append(
                "Exact graph/tokenizer evidence exists, but the emitted compatibility metadata "
                "changes the recorded package identity; new end-to-end evidence is required."
            )
        compatibility.update(
            {
                "runtime_validation_status": validation_status,
                "gguf_architecture": architecture,
                "execution_provider": getattr(pkg, "gguf_execution_provider", None),
                "gguf_graph_identity": {
                    "files": list(graph_identity.files),
                    "sha256": str(graph_identity.sha256),
                },
                "runtime_evidence_id": (
                    evidence.evidence_id if evidence is not None else None
                ),
                "warnings": [
                    *compatibility.get("warnings", []),
                    *validation_warnings,
                ],
            }
        )
        compatibility_path.write_text(
            json.dumps(compatibility, indent=2) + "\n", encoding="utf-8"
        )
        artifacts["runtime_compatibility"] = str(compatibility_path)
        for warning in validation_warnings:
            _LOGGER.warning("%s", warning)
        _publish_directory_no_replace(stage, output_dir)
        return {
            name: str(output_dir / Path(path).relative_to(stage))
            for name, path in artifacts.items()
        }
    finally:
        if stage.exists():
            shutil.rmtree(stage)
