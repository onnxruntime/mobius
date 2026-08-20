# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Import an existing ``genai_config.json`` package into native workflow metadata.

The onnx-genai runtime executes a single structural workflow IR
(``pipeline.workflow``). Packages published for onnxruntime-genai instead ship a
``genai_config.json`` that names ports through ``%d`` patterns and leaves the
control flow implicit. This module performs the one-way import: it reads the
declared port names, resolves them against the ONNX graphs actually present, and
emits an equivalent typed-SSA workflow plus the generation-policy ONNX
components the loop needs.

The import is structural. Nothing here keys on a model family, a model name, or
``model.type``: an encoder-conditioned package is recognised because the config
declares an encoder whose outputs the decoder consumes.
"""

from __future__ import annotations

import dataclasses
import json
import os
import shutil
from typing import Any

import onnx_ir as ir

from mobius._model_package import ModelPackage
from mobius.integrations.onnx_genai.workflow_metadata import (
    _dump_yaml,
    build_speech_to_text_workflow_metadata,
)


@dataclasses.dataclass(frozen=True)
class ImportedGenAiConfig:
    """Minimal generation contract a ``genai_config.json`` package declares."""

    eos_token_id: int
    pad_token_id: int | None
    bos_token_id: int | None
    max_position_embeddings: int
    vocab_size: int | None

    @property
    def num_hidden_layers(self) -> int:  # pragma: no cover - informational
        return 0


@dataclasses.dataclass(frozen=True)
class ImportResult:
    output_dir: str
    metadata_path: str
    components: dict[str, str]
    config: ImportedGenAiConfig


def _expand_pattern(pattern: str | None, count: int) -> list[str]:
    if pattern is None:
        return []
    return [pattern % index for index in range(count)]


def _declared_port_names(section: dict[str, Any], count: int) -> set[str]:
    """Expand every ``%d`` pattern and literal name a config section declares."""
    names: set[str] = set()
    for value in section.values():
        if not isinstance(value, str):
            continue
        if "%d" in value:
            names.update(_expand_pattern(value, count))
        else:
            names.add(value)
    return names


def load_genai_config_package(source_dir: str) -> tuple[ModelPackage, ImportedGenAiConfig]:
    """Load the ONNX components a ``genai_config.json`` package declares.

    Returns a package keyed by structural role (``encoder``/``decoder``) and the
    generation constants the config carries. Weights stay on disk: only the graph
    interface is needed to derive the workflow.
    """
    config_path = os.path.join(source_dir, "genai_config.json")
    with open(config_path, encoding="utf-8") as handle:
        raw = json.load(handle)
    model = raw["model"]
    search = raw.get("search", {})

    models: dict[str, ir.Model] = {}
    filenames: dict[str, str] = {}
    for role in ("encoder", "decoder"):
        section = model.get(role)
        if section is None:
            continue
        filename = section["filename"]
        filenames[role] = filename
        models[role] = ir.load(os.path.join(source_dir, filename))
    if "decoder" not in models:
        raise ValueError(f"{config_path} declares no decoder component")

    eos = model.get("eos_token_id", 0)
    if isinstance(eos, list):
        eos = eos[0] if eos else 0
    config = ImportedGenAiConfig(
        eos_token_id=int(eos),
        pad_token_id=model.get("pad_token_id"),
        bos_token_id=model.get("bos_token_id"),
        max_position_embeddings=int(
            model.get("context_length") or search.get("max_length") or 2048
        ),
        vocab_size=model.get("vocab_size"),
    )
    package = ModelPackage(models, config=config)
    package.imported_filenames = filenames  # type: ignore[attr-defined]
    package.genai_config = raw  # type: ignore[attr-defined]
    return package, config


def unbound_decoder_ports(package: ModelPackage) -> dict[str, list[str]]:
    """Report decoder graph ports the imported config never names.

    A port the config does not declare cannot be bound by an importer without
    guessing, so this is reported rather than silently defaulted.
    """
    raw = getattr(package, "genai_config", {})
    decoder_section = raw.get("model", {}).get("decoder", {})
    layers = int(decoder_section.get("num_hidden_layers", 0))
    declared_inputs = _declared_port_names(decoder_section.get("inputs", {}), layers)
    declared_outputs = _declared_port_names(decoder_section.get("outputs", {}), layers)
    decoder = package["decoder"]
    return {
        "inputs": sorted(
            value.name for value in decoder.graph.inputs if value.name not in declared_inputs
        ),
        "outputs": sorted(
            value.name for value in decoder.graph.outputs if value.name not in declared_outputs
        ),
    }


def import_genai_config_package(
    source_dir: str,
    output_dir: str,
    *,
    sampler: str = "greedy",
    audio_preprocessing: dict[str, Any] | None = None,
    link_artifacts: bool = True,
) -> ImportResult:
    """Emit native workflow metadata for an existing genai_config package.

    The ONNX artifacts are referenced under their original filenames and are
    materialised inside *output_dir* so the emitted package is self-contained.
    Symlinks are convenient during development but a loader that refuses to
    follow an artifact path outside the package root will reject them, so
    ``link_artifacts=False`` copies instead.
    """
    package, config = load_genai_config_package(source_dir)
    if "encoder" not in package:
        raise ValueError(
            "genai_config import currently covers encoder-conditioned packages; "
            "single-decoder packages already load through the bare model contract"
        )
    filenames: dict[str, str] = package.imported_filenames  # type: ignore[attr-defined]

    os.makedirs(output_dir, exist_ok=True)
    if os.path.abspath(source_dir) != os.path.abspath(output_dir):
        for filename in filenames.values():
            for candidate in (filename, f"{filename}.data"):
                source = os.path.join(source_dir, candidate)
                if not os.path.exists(source):
                    continue
                target = os.path.join(output_dir, candidate)
                if os.path.lexists(target):
                    continue
                if link_artifacts:
                    try:
                        os.symlink(os.path.abspath(source), target)
                        continue
                    except OSError:  # pragma: no cover - platform dependent
                        pass
                shutil.copy2(source, target)

    metadata = build_speech_to_text_workflow_metadata(
        package,
        config,
        sampler=sampler,
        audio_preprocessing=audio_preprocessing,
        artifacts=filenames,
    )
    package.save_policy_components(output_dir)
    metadata_path = os.path.join(output_dir, "inference_metadata.yaml")
    with open(metadata_path, "w", encoding="utf-8") as handle:
        _dump_yaml(metadata, handle)
    return ImportResult(
        output_dir=output_dir,
        metadata_path=metadata_path,
        components=dict(filenames),
        config=config,
    )
