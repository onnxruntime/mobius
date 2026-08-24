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

import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Literal

from mobius.integrations.gguf._reader import GGUFModel
from mobius.integrations.gguf._tokenizer import (
    inspect_gguf_tokenizer,
    write_gguf_tokenizer_json,
)

__all__ = ["write_gguf_runtime_package"]

Runtime = Literal["onnx-genai", "ort-genai"]


def write_gguf_runtime_package(
    pkg: Any,
    gguf_path: str | Path,
    output_dir: str | Path,
    *,
    runtime: Runtime = "onnx-genai",
    save_model: bool = True,
    **save_kwargs: Any,
) -> dict[str, str]:
    """Write a complete, loadable package for a GGUF-built model.

    Emits the ONNX graph, an exact embedded ``tokenizer.huggingface.json`` copy,
    and the selected runtime's configuration contract as one staged directory.

    Args:
        pkg: The :class:`~mobius.ModelPackage` returned by
            :func:`~mobius.integrations.gguf.build_from_gguf`.
        gguf_path: The source ``.gguf`` file. Runtime packaging is rejected when
            it does not embed a complete, vocabulary-identical tokenizer JSON.
        output_dir: Destination directory.
        runtime: Which runtime contract to emit. ``"onnx-genai"`` writes
            ``inference_metadata.yaml``; ``"ort-genai"`` writes
            ``genai_config.json``.
        save_model: Save the ONNX graph too. Pass ``False`` when the caller has
            already saved it and only wants the runtime artifacts.
        **save_kwargs: Forwarded to :meth:`ModelPackage.save`.

    Returns:
        Mapping of artifact name to written path. The ``tokenizer`` key is
        absent when the GGUF carries no tokenizer metadata to rebuild from.

    Raises:
        ValueError: If ``runtime`` is not a supported runtime name.
    """
    if runtime not in ("onnx-genai", "ort-genai"):
        raise ValueError(f"Unknown runtime {runtime!r}; expected 'onnx-genai' or 'ort-genai'.")
    if runtime == "ort-genai" and getattr(pkg, "gguf_reuse_plan", None) is not None:
        raise ValueError(
            "ORT GenAI packaging is not supported with reused GGUF weights because "
            "genai_config.json has no supported setting that disables ORT constant "
            "folding. Use direct ONNX Runtime with ORT_DISABLE_ALL, or build without "
            "reuse_gguf_weights."
        )
    draft_manifest = getattr(pkg, "draft_manifest", None)
    if draft_manifest is not None:
        raise ValueError(
            f"{draft_manifest['architecture']} is a target-coupled speculative draft; "
            "standalone runtime packaging is unsupported. Save the ONNX auxiliary "
            "package and draft_manifest.json, then pair it with the exact validated target."
        )
    mtp_head = getattr(pkg, "mtp_head", None)
    if runtime == "ort-genai" and mtp_head is not None:
        raise ValueError(
            "ORT GenAI runtime packaging does not yet have a declared GGUF MTP sidecar "
            "contract; refusing to emit an unreachable mtp/model.onnx."
        )

    output_dir = Path(output_dir)
    if not save_model:
        if not output_dir.is_dir():
            raise ValueError(
                "save_model=False requires an existing package containing model.onnx; "
                "refusing to publish runtime artifacts without a graph."
            )
        component_names = list(pkg)
        expected_models = (
            [output_dir / "model.onnx"]
            if len(component_names) == 1
            else [output_dir / name / "model.onnx" for name in component_names]
        )
        missing_models = [path for path in expected_models if not path.is_file()]
        if not component_names or missing_models:
            raise ValueError(
                "save_model=False requires every existing primary package graph; "
                f"missing {[str(path) for path in missing_models] or ['package components']}."
            )
        if mtp_head is not None and not (output_dir / "mtp" / "model.onnx").is_file():
            raise ValueError(
                "save_model=False requires the existing package to contain mtp/model.onnx."
            )

    source_path = Path(getattr(pkg, "gguf_source_path", gguf_path))
    source_metadata = GGUFModel(source_path).metadata
    verdict = inspect_gguf_tokenizer(
        source_metadata,
        source=str(source_path),
        require_complete=True,
    )
    if not verdict.materialized:
        raise ValueError(
            f"Cannot emit a complete {runtime} package: {verdict.reason}. "
            "The GGUF graph remains buildable, but Mobius will not claim a runnable "
            "package without an exact tokenizer artifact."
        )
    built_verdict = getattr(pkg, "gguf_tokenizer_verdict", None)
    if (
        built_verdict is None
        or built_verdict.metadata_sha256 is None
        or built_verdict.metadata_sha256 != verdict.metadata_sha256
    ):
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
        if save_model:
            pkg.save(str(stage), **save_kwargs)
        elif output_dir.exists():
            shutil.copytree(output_dir, stage, dirs_exist_ok=True)

        tokenizer_path = write_gguf_tokenizer_json(
            source_path,
            stage,
            metadata=source_metadata,
            expected_metadata_sha256=built_verdict.metadata_sha256,
        )
        artifacts["tokenizer"] = tokenizer_path

        if runtime == "ort-genai":
            from mobius.integrations.ort_genai import write_ort_genai_config

            artifacts.update(write_ort_genai_config(pkg, str(stage)))
        else:
            from mobius.integrations.onnx_genai import write_onnx_genai_config

            artifacts.update(
                write_onnx_genai_config(
                    pkg, str(stage), config=getattr(pkg, "config", None), source=None
                )
            )
        if mtp_head is not None:
            from mobius.integrations.onnx_genai.inference_metadata import (
                write_mtp_speculator_metadata,
            )

            speculator_path = write_mtp_speculator_metadata(
                stage,
                backbone_config=getattr(pkg, "config", None),
                proposer_config=getattr(mtp_head, "config", None),
            )
            if speculator_path is not None:
                artifacts["speculator"] = str(speculator_path)
        backup: Path | None = None
        if output_dir.exists():
            backup = output_dir.with_name(f".{output_dir.name}.backup")
            if backup.exists():
                raise FileExistsError(f"Atomic package backup path already exists: {backup}")
            os.replace(output_dir, backup)
        try:
            os.replace(stage, output_dir)
        except Exception:
            if backup is not None:
                os.replace(backup, output_dir)
            raise
        if backup is not None:
            shutil.rmtree(backup)
        return {
            name: str(output_dir / Path(path).relative_to(stage))
            for name, path in artifacts.items()
        }
    finally:
        if stage.exists():
            shutil.rmtree(stage)
