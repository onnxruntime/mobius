# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Emit a validated runtime package or an accurate runtime-unvalidated model package.

Saving the ONNX graph is not enough to run a model: the runtime also needs a
tokenizer and its own configuration contract. Those come from two different
places — the tokenizer from the GGUF's embedded ggml metadata, the contract
from the built package — so a caller that saves the graph and stops produces a
directory that loads nowhere.

This module is the single place that knows the full artifact set. Exact runtime
evidence produces a complete package. Downstream runtime, version, registry, or
executor limitations preserve the model package with explicit unvalidated
metadata instead of blocking export.
"""

from __future__ import annotations

import ctypes
import hashlib
import importlib.metadata
import json
import logging
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Literal

from mobius.integrations.gguf._arch_registry import get_arch_spec
from mobius.integrations.gguf._runtime_evidence import (
    FINAL_RUNTIME_PACKAGE_SCHEMA,
    RuntimeEvidenceUnavailableError,
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
from mobius.integrations.gguf._tokenizer_evidence import tokenizer_evidence

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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _installed_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _cache_ports(model: Any) -> list[dict[str, str]]:
    output_names = {value.name for value in model.graph.outputs if value.name is not None}
    pairs: list[dict[str, str]] = []
    for value in model.graph.inputs:
        name = value.name
        if name is None or not name.startswith("past_key_values."):
            continue
        output = "present." + name.removeprefix("past_key_values.")
        if output not in output_names:
            raise ValueError(
                f"MTP package status cannot pair cache input {name!r} with {output!r}"
            )
        pairs.append({"input": name, "output": output})
    return sorted(pairs, key=lambda pair: pair["input"])


def _write_mtp_runtime_status(
    stage: Path,
    *,
    pkg: Any,
    built_identity: Any,
    graph_identity: Any,
    runtime_payload_identity: Any,
    runtime: Runtime,
    runtime_version: str | None,
    tokenizer_repository: str | None,
    tokenizer_revision: str | None,
    tokenizer_metadata_sha256: str | None,
) -> str:
    from mobius._model_package import _read_mtp_sidecar_name

    sidecar_name = _read_mtp_sidecar_name(str(stage))
    if sidecar_name is None or pkg.mtp_head is None:
        raise ValueError("Saved MTP package is missing its sidecar manifest identity")
    if set(pkg.mtp_head) != {"model"} or "model" not in pkg:
        raise ValueError("GGUF MTP runtime status requires one target and one MTP model")
    config_hashes = {
        path.name: _sha256_file(path)
        for path in sorted(stage.iterdir())
        if path.is_file()
        and path.name
        in {
            "genai_config.json",
            "inference_metadata.yaml",
            "mtp_config.json",
            "runtime_compatibility.json",
        }
    }
    tokenizer_hashes = {
        path.name: {"size": path.stat().st_size, "sha256": _sha256_file(path)}
        for path in sorted(stage.iterdir())
        if path.is_file()
        and path.name
        in {
            "added_tokens.json",
            "chat_template.jinja",
            "merges.txt",
            "special_tokens_map.json",
            "tokenizer.json",
            "tokenizer.jsonl",
            "tokenizer.model",
            "tokenizer_config.json",
            "vocab.json",
        }
    }
    payload = {
        "schema_version": 1,
        "status": "runtime_unvalidated",
        "artifact": {
            "architecture": built_identity.architecture,
            "filename": built_identity.filename,
            "size": built_identity.size,
            "sha256": built_identity.sha256,
            "tensor_count": built_identity.tensor_count,
            "tensor_qtypes": dict(built_identity.tensor_qtypes),
        },
        "graph_package": {
            "files": list(graph_identity.files),
            "sha256": graph_identity.sha256,
        },
        "runtime_payload": {
            "files": list(runtime_payload_identity.files),
            "sha256": runtime_payload_identity.sha256,
            "excludes": "mtp_runtime_status.json",
        },
        "config_sha256": config_hashes,
        "tokenizer": {
            "repository": tokenizer_repository,
            "revision": tokenizer_revision,
            "gguf_metadata_sha256": tokenizer_metadata_sha256,
            "assets": tokenizer_hashes,
            "status": (
                "best_effort_unvalidated"
                if tokenizer_repository is not None
                else "not_provided"
            ),
        },
        "cache_namespaces": {
            "target": {
                "namespace": "target",
                "ports": _cache_ports(pkg["model"]),
            },
            "mtp": {
                "namespace": "mtp",
                "model": f"{sidecar_name}/model.onnx",
                "ports": _cache_ports(pkg.mtp_head["model"]),
            },
        },
        "runtime": {
            "name": runtime,
            "requested_version": runtime_version,
            "installed_onnxruntime_version": _installed_version("onnxruntime"),
            "installed_ort_genai_version": _installed_version("onnxruntime-genai"),
            "execution_provider": getattr(pkg, "gguf_execution_provider", None),
            "orchestration": "external",
        },
        "validated_claims": {
            "artifact_identity": True,
            "graph_serialization": True,
            "cache_namespace_separation": True,
            "runtime_execution": False,
            "source_value_fidelity": False,
            "storage_fidelity": False,
            "target_only_output_equality": False,
        },
    }
    path = stage / "mtp_runtime_status.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    _LOGGER.warning(
        "Published target and MTP ONNX graphs with runtime_unvalidated external "
        "coordination metadata; no downstream MTP execution claim is implied."
    )
    return str(path)


def _runtime_package_matches_evidence(evidence: Any, identity: Any) -> bool:
    """Whether the complete final package bytes match one runtime evidence record."""
    return (
        identity.files == evidence.runtime_package_files
        and identity.sha256 == evidence.runtime_package_sha256
    )


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
    Attached MTP graphs additionally record separate target/MTP cache namespaces
    and package hashes without claiming downstream orchestration support.

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
    if (
        tokenizer_revision is not None
        and re.fullmatch(r"[0-9a-f]{40}", tokenizer_revision) is None
    ):
        raise ValueError("GGUF runtime packaging tokenizer_revision must be immutable 40-hex")
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
    evidence_warning: str | None = None
    if architecture_spec.runtime_evidence_ids:
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
                tokenizer_repository=None,
                tokenizer_revision=None,
            )
        except RuntimeEvidenceUnavailableError as error:
            evidence_warning = f"{error} Export continues without claiming runtime validation."
    if (
        evidence is not None
        and tokenizer_repository is not None
        and (
            tokenizer_repository != evidence.tokenizer_repository
            or tokenizer_revision != evidence.tokenizer_revision
        )
    ):
        raise ValueError(
            "The explicit tokenizer source conflicts with exact runtime evidence: "
            f"requested={tokenizer_repository}@{tokenizer_revision}, "
            f"evidence={evidence.tokenizer_repository}@{evidence.tokenizer_revision}."
        )
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
    from mobius.integrations.gguf._component_export import resolve_tokenizer_export_verdict

    verdict = resolve_tokenizer_export_verdict(
        source_model,
        source_path,
        verdict=verdict,
        artifact_identity=built_identity,
    )
    built_verdict = getattr(pkg, "gguf_tokenizer_verdict", None)
    if (
        built_verdict is None
        or built_verdict.metadata_sha256 != verdict.metadata_sha256
        or built_verdict.route != verdict.route
        or built_verdict.evidence_id != verdict.evidence_id
        or built_verdict.tokenizer_sha256 != verdict.tokenizer_sha256
    ):
        raise ValueError(
            "The GGUF tokenizer metadata no longer matches the identity captured during "
            "graph construction; refusing to pair the graph with a replaced tokenizer source."
        )
    if (
        getattr(built_verdict, "blocker_category", None) is not None
        and getattr(evidence, "runtime_package_schema", None) != FINAL_RUNTIME_PACKAGE_SCHEMA
    ):
        # Identifier-level blockers remain authoritative unless a final-package
        # record proves this exact artifact through the strict checks below.
        evidence = None

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", suffix=".tmp", dir=output_dir.parent)
    )
    previous_export_report = getattr(pkg, "export_report", None)
    published = False
    try:
        artifacts: dict[str, str] = {}
        pkg.save(str(stage), _atomic_export_report=False, **save_kwargs)
        if not source_model.source_matches_path():
            raise ValueError(
                "The GGUF source changed while target/MTP graphs were being serialized; "
                "refusing package publication."
            )
        graph_files = tuple(
            sorted(
                path.relative_to(stage).as_posix()
                for path in stage.rglob("*")
                if path.is_file() and path.name != "export_report.json"
            )
        )
        graph_identity = gguf_graph_package_identity(stage, files=graph_files)
        validation_warnings: list[str] = []
        if evidence_warning is not None:
            validation_warnings.append(evidence_warning)
        if architecture_spec.runtime is not Support.SUPPORTED:
            validation_warnings.append(
                f"No real-artifact runtime validation is recorded for {architecture!r}: "
                f"{architecture_spec.reason}"
            )
        if runtime == "ort-genai" and getattr(pkg, "gguf_reuse_plan", None) is not None:
            validation_warnings.append(
                "ORT GenAI cannot disable constant folding for reused GGUF weights; "
                "the faithful package is exported without claiming runtime validation."
            )
        if evidence is not None and (
            graph_identity.files != evidence.graph_files
            or graph_identity.sha256 != evidence.graph_sha256
        ):
            validation_warnings.append(
                "The serialized graph does not match the recorded runtime-evidence identity."
            )
            evidence = None

        tokenizer_exported = False
        tokenizer_warning: str | None = None
        tokenizer_route_evidence = (
            tokenizer_evidence(verdict.evidence_id)
            if verdict.route == "pinned-source" and verdict.evidence_id is not None
            else None
        )
        if verdict.route == "pinned-source":
            if tokenizer_route_evidence is None:
                raise ValueError(
                    "Pinned-source tokenizer verdict references missing exact evidence "
                    f"{verdict.evidence_id!r}."
                )
            tokenizer_path = materialize_gguf_tokenizer(
                source_path,
                stage,
                source=tokenizer_route_evidence.source,
                metadata=source_metadata,
                source_identity=(f"sha256:{built_identity.sha256}/{built_identity.filename}"),
                local_files_only=local_files_only,
            )
            artifacts["tokenizer"] = tokenizer_path
            tokenizer_exported = True
        elif evidence is not None:
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
            tokenizer_exported = True
        elif verdict.route == "copy":
            artifacts["tokenizer"] = write_gguf_tokenizer_json(
                source_path,
                stage,
                metadata=source_metadata,
                expected_metadata_sha256=verdict.metadata_sha256,
                source_identity=f"sha256:{built_identity.sha256}/{built_identity.filename}",
            )
            tokenizer_exported = True
        else:
            tokenizer_warning = (
                f"Tokenizer route {verdict.route_identifier!r} was omitted "
                f"({verdict.blocker_category or 'tokenizer-materialization-unvalidated'}"
                f"{f'; evidence={verdict.evidence_id}' if verdict.evidence_id else ''}): "
                f"{verdict.reason} Tokenizer semantics are unverified; provide and validate "
                "a tokenizer before end-to-end use."
            )
            validation_warnings.append(tokenizer_warning)

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
                )
            )
        else:
            from mobius.integrations.onnx_genai import write_onnx_genai_config

            artifacts.update(
                write_onnx_genai_config(
                    pkg,
                    str(stage),
                    config=getattr(pkg, "config", None),
                    source=None,
                    revision=None,
                )
            )
        if draft_manifest is not None:
            from mobius.integrations.gguf._draft import write_draft_manifest

            artifacts["draft_manifest"] = write_draft_manifest(draft_manifest, stage)
        if mtp_head is not None and runtime == "onnx-genai":
            from mobius._model_package import _read_mtp_sidecar_name
            from mobius.integrations.onnx_genai.inference_metadata import (
                write_mtp_speculator_metadata,
            )

            sidecar_name = _read_mtp_sidecar_name(str(stage))
            if sidecar_name is None:
                raise ValueError("Saved MTP package has no sidecar manifest")
            speculator_path = write_mtp_speculator_metadata(
                str(stage),
                backbone_config=getattr(pkg, "config", None),
                proposer_config=getattr(mtp_head, "config", None),
                model_path=f"{sidecar_name}/model.onnx",
            )
            if speculator_path is not None:
                artifacts["speculator"] = str(speculator_path)
        compatibility_path = stage / "runtime_compatibility.json"
        compatibility = (
            json.loads(compatibility_path.read_text(encoding="utf-8"))
            if compatibility_path.exists()
            else {"runtime": runtime}
        )
        compatibility_warnings = list(compatibility.get("warnings", []))
        existing_status = compatibility.get("runtime_validation_status")
        from mobius.integrations.gguf._component_export import (
            attach_runtime_unvalidated_report,
        )

        export_report_path = stage / "export_report.json"
        artifacts["export_report"] = str(export_report_path)
        artifacts["runtime_compatibility"] = str(compatibility_path)

        def write_final_metadata() -> None:
            if evidence is not None:
                validation_status = "validated"
                runtime_support: Literal["supported", "deferred", "blocked"] = "supported"
                blocker_category = None
                runtime_reason = None
                report_validation_status: Literal["validated", "unvalidated"] = "validated"
            elif existing_status == "unsupported-by-tested-runtime":
                validation_status = existing_status
                runtime_support = "blocked"
                blocker_category = "runtime-unsupported-by-tested-runtime"
                runtime_reason = "; ".join(validation_warnings)
                report_validation_status = "unvalidated"
            elif architecture_spec.runtime is Support.REJECTED:
                validation_status = "unvalidated"
                runtime_support = "blocked"
                blocker_category = "runtime-route-rejected"
                runtime_reason = "; ".join(validation_warnings)
                report_validation_status = "unvalidated"
            elif architecture_spec.runtime is Support.DEFERRED:
                validation_status = "unvalidated"
                runtime_support = "deferred"
                blocker_category = "runtime-route-deferred"
                runtime_reason = "; ".join(validation_warnings)
                report_validation_status = "unvalidated"
            else:
                validation_status = "unvalidated"
                runtime_support = "deferred"
                blocker_category = "runtime-validation-unavailable"
                runtime_reason = (
                    "; ".join(validation_warnings)
                    or "No exact end-to-end runtime validation is claimed for the "
                    "emitted package."
                )
                report_validation_status = "unvalidated"

            attach_runtime_unvalidated_report(
                pkg,
                runtime,
                blocker_category=blocker_category,
                reason=runtime_reason,
                evidence_id=(evidence.evidence_id if evidence is not None else None),
                support_status=runtime_support,
                runtime_output="exported",
                runtime_validation_status=report_validation_status,
                tokenizer_exported=tokenizer_exported,
                emit_warning=False,
            )
            assert pkg.export_report is not None
            pkg.export_report.write_json(export_report_path)
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
                        *compatibility_warnings,
                        *validation_warnings,
                    ],
                }
            )
            compatibility_path.write_text(
                json.dumps(compatibility, indent=2) + "\n",
                encoding="utf-8",
            )

        def write_mtp_status() -> None:
            if mtp_head is None:
                return
            status_path = stage / "mtp_runtime_status.json"
            if status_path.exists():
                status_path.unlink()
            runtime_payload_identity = gguf_graph_package_identity(stage)
            artifacts["mtp_runtime_status"] = _write_mtp_runtime_status(
                stage,
                pkg=pkg,
                built_identity=built_identity,
                graph_identity=graph_identity,
                runtime_payload_identity=runtime_payload_identity,
                runtime=runtime,
                runtime_version=runtime_version,
                tokenizer_repository=tokenizer_repository,
                tokenizer_revision=tokenizer_revision,
                tokenizer_metadata_sha256=verdict.metadata_sha256,
            )

        write_final_metadata()
        write_mtp_status()
        final_identity = gguf_graph_package_identity(stage)
        if evidence is not None and not _runtime_package_matches_evidence(
            evidence, final_identity
        ):
            validation_warnings.append(
                "The final staged runtime package, including export and compatibility "
                "metadata, does not match the recorded runtime-evidence identity."
            )
            evidence = None
            write_final_metadata()
            write_mtp_status()
        if not source_model.source_matches_path():
            raise ValueError(
                "The GGUF source changed while runtime metadata was being written; "
                "refusing package publication."
            )
        _publish_directory_no_replace(stage, output_dir)
        published = True
        for warning in validation_warnings:
            if (
                warning != tokenizer_warning
                or getattr(built_verdict, "blocker_category", None) is None
            ):
                _LOGGER.warning("%s", warning)
        return {
            name: str(output_dir / Path(path).relative_to(stage))
            for name, path in artifacts.items()
        }
    finally:
        if stage.exists():
            shutil.rmtree(stage)
        if not published:
            pkg.export_report = previous_export_report
