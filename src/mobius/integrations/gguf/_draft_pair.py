# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Build a self-contained target/draft GGUF package without claiming a downstream runtime."""

from __future__ import annotations

__all__ = ["build_draft_pair_from_gguf", "write_draft_pair_package"]

import copy
import dataclasses
import json
import logging
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from mobius._model_package import ModelPackage
from mobius.integrations.gguf._builder import _resolve_gguf_path, build_from_gguf
from mobius.integrations.gguf._draft import _tokenizer_digest, _validate_special_ids
from mobius.integrations.gguf._reader import GGUFModel
from mobius.integrations.gguf._runtime_evidence import gguf_artifact_identity
from mobius.integrations.gguf._runtime_package import (
    _cache_ports,
    _installed_version,
    _publish_directory_no_replace,
    _sha256_file,
)
from mobius.integrations.gguf._shard_set import open_gguf_model
from mobius.tasks import DraftTargetCausalLMTask

_LOGGER = logging.getLogger(__name__)


def _require_single_file_model(model: Any, *, role: str) -> GGUFModel:
    if not isinstance(model, GGUFModel):
        raise TypeError(f"{role} draft-pair evidence currently requires one GGUF file")
    return model


def _validate_target_artifact(
    target_model: GGUFModel,
    target_config: Any,
    draft_manifest: Mapping[str, Any],
) -> None:
    expected = draft_manifest["target"]
    actual = {
        "model_type": target_config.model_type,
        "hidden_size": target_config.hidden_size,
        "num_hidden_layers": target_config.num_hidden_layers,
        "vocab_size": target_config.vocab_size,
    }
    mismatches = {
        field: (expected[field], value)
        for field, value in actual.items()
        if expected[field] != value
    }
    if mismatches:
        raise ValueError(
            f"target GGUF config does not match the validated draft target: {mismatches}"
        )
    _validate_special_ids(
        target_model.metadata,
        {
            "bos_token_id": target_config.bos_token_id,
            "eos_token_id": target_config.eos_token_id,
            "pad_token_id": target_config.pad_token_id,
        },
    )
    tokens = [str(token) for token in target_model.metadata.get("tokenizer.ggml.tokens", ())]
    tokens_sha256 = _tokenizer_digest(tokens)
    if tokens_sha256 != expected["tokenizer_tokens_sha256"]:
        raise ValueError(
            "target GGUF tokenizer does not match the exact tokenizer bound to the draft"
        )


def build_draft_pair_from_gguf(
    target_gguf: str | Path,
    draft_gguf: str | Path,
    *,
    target_config: str | Path | Mapping[str, object],
    dtype: str | None = None,
    keep_quantized: bool = True,
    execution_provider: str = "default",
) -> ModelPackage:
    """Build exact target and draft graphs plus their orchestration manifest.

    The package is directly executable with ONNX Runtime and the coordinator in
    :mod:`mobius.integrations.gguf._draft_runtime`. It intentionally makes no
    claim that a released higher-level generation runtime consumes this package.
    """
    if dtype in {"bf16", "bfloat16"}:
        raise ValueError(
            "GGUF draft-pair coordination supports float32 and float16, not bfloat16 "
            "cache feeds"
        )
    resolved_target = _resolve_gguf_path(target_gguf, keep_quantized)
    resolved_draft = _resolve_gguf_path(draft_gguf, keep_quantized)
    target_source = _require_single_file_model(open_gguf_model(resolved_target), role="target")
    draft_source = _require_single_file_model(open_gguf_model(resolved_draft), role="draft")
    draft_package = build_from_gguf(
        resolved_draft,
        target_config=target_config,
        dtype=dtype,
        keep_quantized=keep_quantized,
        execution_provider=execution_provider,
        _gguf_model=draft_source,
    )
    if draft_package.draft_manifest is None:
        raise ValueError("draft GGUF did not produce a target-pairing manifest")
    draft_manifest = copy.deepcopy(draft_package.draft_manifest)
    target_layers = list(draft_manifest["target"]["target_layers"])
    needs_embedding = draft_manifest["orchestration"]["embedding_source"] == "target"
    needs_lm_head = draft_manifest["orchestration"]["lm_head_source"] == "target"
    target_task = DraftTargetCausalLMTask() if needs_embedding or needs_lm_head else None
    target_package = build_from_gguf(
        resolved_target,
        task=target_task,
        dtype=dtype,
        keep_quantized=keep_quantized,
        execution_provider=execution_provider,
        output_layer_indices=target_layers,
        _gguf_model=target_source,
    )

    if not target_source.source_matches_path() or not draft_source.source_matches_path():
        raise ValueError("target or draft GGUF changed while its pair graph was built")
    _validate_target_artifact(target_source, target_package.config, draft_manifest)

    components = {
        "target": target_package["model"],
        "draft": draft_package["model"],
    }
    component_manifest: dict[str, Any] = {
        "target": {
            "artifact": "target/model.onnx",
            "quantization_report": "target_quantization_report.json",
            "hidden_state_outputs": [f"hidden_states.{index}" for index in target_layers],
        },
        "draft": {
            "artifact": "draft/model.onnx",
            "quantization_report": "draft_quantization_report.json",
        },
    }
    if needs_embedding:
        components["target_embedding"] = target_package["embedding"]
        component_manifest["target_embedding"] = {
            "artifact": "target_embedding/model.onnx",
            "input": "input_ids",
            "output": "inputs_embeds",
        }
    if needs_lm_head:
        components["target_lm_head"] = target_package["lm_head"]
        component_manifest["target_lm_head"] = {
            "artifact": "target_lm_head/model.onnx",
            "input": "hidden_states",
            "output": "logits",
        }

    package = ModelPackage(components, config=target_package.config)
    package.draft_pair_quantization_reports = {
        "target": target_package.gguf_quantization_report,
        "draft": draft_package.gguf_quantization_report,
    }

    target_identity = gguf_artifact_identity(
        Path(resolved_target),
        target_source,
        architecture=target_source.architecture,
    )
    draft_identity = gguf_artifact_identity(
        Path(resolved_draft),
        draft_source,
        architecture=draft_source.architecture,
    )
    draft_manifest.update(
        {
            "runtime": "runtime_unvalidated",
            "runtime_warning": (
                "The target and draft graphs are directly executable with ONNX Runtime, "
                "but no released higher-level generation runtime is claimed."
            ),
            "components": component_manifest,
            "cache_namespaces": {
                "target": {
                    "namespace": "target",
                    "model": "target/model.onnx",
                    "ports": _cache_ports(components["target"]),
                },
                "draft": {
                    "namespace": "draft",
                    "model": "draft/model.onnx",
                    "ports": _cache_ports(components["draft"]),
                },
            },
            "artifacts": {
                "target": dataclasses.asdict(target_identity),
                "draft": dataclasses.asdict(draft_identity),
            },
            "generation": {
                "mode": "greedy",
                "batch_size": 1,
                "beam_reorder": "not supported by the reference coordinator",
            },
        }
    )
    package.draft_manifest = draft_manifest
    package.draft_config = draft_package.config
    package.gguf_architecture = draft_source.architecture
    package.gguf_execution_provider = execution_provider
    if not target_source.source_matches_path() or not draft_source.source_matches_path():
        raise ValueError("target or draft GGUF changed while its pair identity was bound")
    package.gguf_source_path = str(Path(resolved_draft).resolve())
    package.gguf_target_source_path = str(Path(resolved_target).resolve())
    return package


def _write_draft_runtime_status(
    stage: Path,
    *,
    package: ModelPackage,
    graph_identity: Any,
    runtime_payload_identity: Any,
    requested_runtime: str | None,
    runtime_version: str | None,
) -> str:
    manifest = package.draft_manifest
    if manifest is None:
        raise ValueError("Draft runtime status requires a pairing manifest")
    payload = {
        "schema_version": 1,
        "status": "runtime_unvalidated",
        "artifacts": manifest["artifacts"],
        "graph_package": {
            "files": list(graph_identity.files),
            "sha256": graph_identity.sha256,
        },
        "runtime_payload": {
            "files": list(runtime_payload_identity.files),
            "sha256": runtime_payload_identity.sha256,
            "excludes": "draft_runtime_status.json",
        },
        "config_sha256": {"draft_manifest.json": _sha256_file(stage / "draft_manifest.json")},
        "cache_namespaces": manifest["cache_namespaces"],
        "runtime": {
            "name": requested_runtime or "onnxruntime",
            "requested_version": runtime_version,
            "direct_executor": "onnxruntime",
            "installed_onnxruntime_version": _installed_version("onnxruntime"),
            "installed_ort_genai_version": _installed_version("onnxruntime-genai"),
            "execution_provider": package.gguf_execution_provider,
            "orchestration": "external",
            "higher_level_status": "runtime_unvalidated",
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
    path = stage / "draft_runtime_status.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _LOGGER.warning(
        "Published target and draft ONNX graphs with runtime_unvalidated higher-level "
        "coordination metadata; direct ORT evidence remains artifact-scoped."
    )
    return str(path)


def _validate_pair_sources(package: ModelPackage) -> tuple[GGUFModel, GGUFModel]:
    manifest = package.draft_manifest
    if manifest is None:
        raise ValueError("Draft package source validation requires a pairing manifest")
    paths = {
        "target": package.gguf_target_source_path,
        "draft": package.gguf_source_path,
    }
    models = {}
    for role, source_path in paths.items():
        if source_path is None:
            raise ValueError(f"Draft package lost its {role} GGUF source path")
        model = _require_single_file_model(open_gguf_model(source_path), role=role)
        expected = manifest["artifacts"][role]
        current = dataclasses.asdict(
            gguf_artifact_identity(
                Path(source_path),
                model,
                architecture=model.architecture,
                filename=expected["filename"],
            )
        )
        if current != expected:
            raise ValueError(f"Draft package {role} GGUF no longer matches its build identity")
        if not model.source_matches_path():
            raise ValueError(f"Draft package {role} GGUF changed during validation")
        models[role] = model
    return models["target"], models["draft"]


def write_draft_pair_package(
    package: ModelPackage,
    output_dir: str | Path,
    *,
    requested_runtime: str | None = None,
    runtime_version: str | None = None,
    **save_kwargs: Any,
) -> dict[str, str]:
    """Serialize a draft pair and its required orchestration manifest."""
    from mobius.integrations.gguf._draft import write_draft_manifest

    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"draft pair package already exists: {output}")
    if package.draft_manifest is None:
        raise ValueError("draft pair package has no draft_manifest")
    target_source, draft_source = _validate_pair_sources(package)
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(
            prefix=f".{output.name}.",
            suffix=".tmp",
            dir=output.parent,
        )
    )
    report_names: dict[str, str] = {}
    published = False
    try:
        if not target_source.source_matches_path() or not draft_source.source_matches_path():
            raise ValueError("target or draft GGUF changed before package serialization")
        package.save(str(stage), **save_kwargs)
        if not target_source.source_matches_path() or not draft_source.source_matches_path():
            raise ValueError("target or draft GGUF changed during package serialization")
        for role, report in package.draft_pair_quantization_reports.items():
            if report is None:
                continue
            report_name = f"{role}_quantization_report.json"
            report.write_json(stage / report_name)
            report_names[f"{role}_quantization_report"] = report_name
        from mobius.integrations.gguf._runtime_evidence import (
            gguf_graph_package_identity,
        )

        graph_identity = gguf_graph_package_identity(stage)
        manifest = package.draft_manifest
        manifest["graph_package"] = {
            "files": list(graph_identity.files),
            "sha256": graph_identity.sha256,
        }
        write_draft_manifest(manifest, stage)
        runtime_payload_identity = gguf_graph_package_identity(stage)
        status_path = _write_draft_runtime_status(
            stage,
            package=package,
            graph_identity=graph_identity,
            runtime_payload_identity=runtime_payload_identity,
            requested_runtime=requested_runtime,
            runtime_version=runtime_version,
        )
        if not target_source.source_matches_path() or not draft_source.source_matches_path():
            raise ValueError("target or draft GGUF changed while runtime metadata was written")
        _publish_directory_no_replace(stage, output)
        published = True
    finally:
        if not published:
            shutil.rmtree(stage, ignore_errors=True)
    return {
        "package": str(output),
        "manifest": str(output / "draft_manifest.json"),
        "draft_runtime_status": str(output / Path(status_path).name),
        **{key: str(output / name) for key, name in report_names.items()},
    }
