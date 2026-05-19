# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Writers for ORT Model Package metadata files.

This module produces the three package metadata files that the
current ORT Model Package parser expects:

- ``<package>/manifest.json`` — top-level package descriptor
  (``schema_version`` + ``components`` array of component names).
- ``<package>/models/<component>/metadata.json`` — component descriptor
  whose ``variants`` is a *map* keyed by variant name; each entry
  carries an ``ep_compatibility`` array of
  ``{"ep", "device"?, "compatibility_string"?}`` objects.
- ``<package>/models/<component>/<variant>/variant.json`` — per-variant
  manifest listing files, optional per-file session/provider options
  (objects, not arrays), optional ``shared_files`` map, and the
  ``consumer_metadata.genai_config_overlay`` blob.

Mobius emits a *single-variant* package whose variant is named
``"base"``. The base variant is EP-agnostic: per-file session/provider
options are simply absent (the loader treats absent as default), and
the ``genai_config_overlay`` is empty. Downstream packagers (Olive,
EP-specific compilers) add additional variants alongside ``base``
when they want EP-specialized artifacts.

These writers do not import core model/task layers — they operate
on plain Python dicts and write JSON.
"""

from __future__ import annotations

import json
import os
from typing import Any

# Keep in sync with ORT's model-package descriptor parser.
SCHEMA_VERSION = 1

# The single variant name produced by mobius. Downstream packagers
# add more variants alongside this one (e.g. "cuda", "qnn-htp-v75").
BASE_VARIANT_NAME = "base"

# ep_compatibility entries for the ``base`` variant. The base variant
# carries an EP-agnostic ONNX graph: any major ORT EP that supports
# the opset and ops is expected to load it. Listing the major EPs by
# default keeps today's ``og.Model(path, ep="CUDAExecutionProvider")``
# UX intact. Producers who know their build is e.g. CUDA-only can
# narrow this list afterwards.
#
# Each entry follows the ORT Model Package schema:
#   { "ep": <canonical ORT EP name>,
#     "device"?: <optional device class hint>,
#     "compatibility_string"?: <opaque EP match string> }
# Omitting ``compatibility_string`` means "matches any device the EP
# claims to support" — the right default for an EP-agnostic graph.
DEFAULT_BASE_EP_COMPATIBILITY: list[dict[str, Any]] = [
    {"ep": "CPUExecutionProvider"},
    {"ep": "CUDAExecutionProvider"},
    {"ep": "DmlExecutionProvider"},
    {"ep": "WebGpuExecutionProvider"},
    {"ep": "NvTensorRTRTXExecutionProvider"},
]


def write_manifest(directory: str, components: list[str]) -> str:
    """Write ``manifest.json`` at the package root.

    Args:
        directory: Package root directory (created if missing).
        components: Ordered list of component names included in the
            package (e.g. ``["decoder"]`` for LLMs;
            ``["decoder", "vision_encoder", "embedding"]`` for VLMs).

    Returns the path to the written file.
    """
    os.makedirs(directory, exist_ok=True)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "components": list(components),
    }
    path = os.path.join(directory, "manifest.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=4)
    return path


def write_component_metadata(
    component_dir: str,
    *,
    variants: dict[str, dict[str, Any]] | None = None,
) -> str:
    """Write ``metadata.json`` in a component directory.

    Schema (per ORT-GenAI Model Package v4)::

        {
          "variants": {
            "<name>": {
              "ep_compatibility": [
                {"ep": "...", "device"?: "...", "compatibility_string"?: "..."}
              ]
            }
          }
        }

    Args:
        component_dir: Path to ``<package>/models/<component>/`` (created if
            missing).
        variants: Mapping from variant name to its descriptor dict.
            Each descriptor must contain ``"ep_compatibility"`` (a
            list of EP entries). When ``None``, defaults to a single
            ``base`` variant declaring
            :data:`DEFAULT_BASE_EP_COMPATIBILITY`.

    Returns the path to the written file.
    """
    os.makedirs(component_dir, exist_ok=True)
    if variants is None:
        variants = {
            BASE_VARIANT_NAME: {"ep_compatibility": DEFAULT_BASE_EP_COMPATIBILITY},
        }
    else:
        variants = _normalize_variants(variants)
    metadata = {"variants": variants}
    path = os.path.join(component_dir, "metadata.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
    return path


def write_variant_json(
    variant_dir: str,
    *,
    files: list[dict[str, Any]] | None = None,
    consumer_metadata: dict[str, Any] | None = None,
) -> str:
    """Write ``variant.json`` inside a variant directory.

    Schema (per ORT-GenAI Model Package v4)::

        {
          "files": [
            {"filename": "<rel-path>",
             "session_options"?: {<obj>},
             "provider_options"?: {<obj>},
             "shared_files"?: {"<graph-name>": "<sha256>"}}
          ],
          "consumer_metadata"?: {...}
        }

    The optional fields ``session_options``, ``provider_options``, and
    ``shared_files`` are *objects* when present. The loader rejects
    list-form values (``shared_files`` is explicitly checked, and
    ``session_options``/``provider_options`` are routed through
    ``DocumentToObject`` which throws on non-objects). We therefore
    *omit* these keys when there is nothing to declare, which is the
    cleanest representation for mobius's EP-agnostic ``base`` variant.

    Args:
        variant_dir: Path to ``<package>/models/<component>/<variant>/``
            (created if missing).
        files: List of file entries. Each entry has at minimum
            ``"filename"`` (relative to *variant_dir*); optional
            ``"session_options"`` / ``"provider_options"`` /
            ``"shared_files"`` must be objects when supplied. When
            ``None``, defaults to a single ``"model.onnx"`` file with
            no per-file options — the standard mobius single-file
            base variant.
        consumer_metadata: Opaque blob carried verbatim through the
            package API. ORT-GenAI looks for
            ``consumer_metadata.genai_config_overlay`` here. When
            ``None``, defaults to ``{"genai_config_overlay": {}}``.

    Returns the path to the written file.
    """
    os.makedirs(variant_dir, exist_ok=True)
    if files is None:
        files = [_make_default_file_entry("model.onnx")]
    else:
        files = [_normalize_file_entry(entry) for entry in files]
    if consumer_metadata is None:
        consumer_metadata = {"genai_config_overlay": {}}

    variant = {
        "files": files,
        "consumer_metadata": consumer_metadata,
    }
    path = os.path.join(variant_dir, "variant.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(variant, f, indent=4)
    return path


def _make_default_file_entry(filename: str) -> dict[str, Any]:
    """Return a file entry for the EP-agnostic base variant.

    The base variant ships only the file itself — no per-file session
    options, no provider options, no shared-weight references — so the
    optional fields are omitted entirely.
    """
    return {"filename": filename}


def _normalize_variants(variants: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Return a copy of variant descriptors using ORT's current field names."""
    normalized: dict[str, dict[str, Any]] = {}
    for variant_name, descriptor in variants.items():
        ep_entries = descriptor.get("ep_compatibility")
        if not isinstance(ep_entries, list) or not ep_entries:
            raise ValueError("metadata.json variant entry must contain a non-empty ep_compatibility list")

        out = dict(descriptor)
        out["ep_compatibility"] = [_normalize_ep_compatibility_entry(entry) for entry in ep_entries]
        normalized[variant_name] = out
    return normalized


def _normalize_ep_compatibility_entry(entry: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(entry, dict):
        raise TypeError("metadata.json ep_compatibility entries must be objects")

    ep = entry.get("ep")
    if not isinstance(ep, str) or not ep:
        raise ValueError("metadata.json ep_compatibility entries require a non-empty string 'ep'")

    out: dict[str, Any] = {"ep": ep}

    device = entry.get("device")
    if device is not None:
        if not isinstance(device, str):
            raise ValueError("metadata.json ep_compatibility 'device' must be a string")
        out["device"] = device

    compatibility_string = entry.get("compatibility_string")
    if compatibility_string is None and "compatibility" in entry:
        compatibility = entry["compatibility"]
        if not isinstance(compatibility, list) or not all(isinstance(item, str) for item in compatibility):
            raise ValueError("metadata.json ep_compatibility 'compatibility' must be a list of strings")
        compatibility_string = ",".join(compatibility)

    if compatibility_string is not None:
        if not isinstance(compatibility_string, str):
            raise ValueError("metadata.json ep_compatibility 'compatibility_string' must be a string")
        if compatibility_string:
            out["compatibility_string"] = compatibility_string

    return out


def _normalize_file_entry(entry: dict[str, Any]) -> dict[str, Any]:
    """Validate and copy a file entry, omitting empty optional fields.

    The loader requires ``session_options`` / ``provider_options`` /
    ``shared_files`` to be JSON objects (dicts) when present. We
    enforce that here and drop empty values to keep variant.json
    minimal and round-trip-stable.
    """
    if "filename" not in entry:
        raise ValueError("variant.json file entry is missing required 'filename'")

    out: dict[str, Any] = {"filename": entry["filename"]}
    for field in ("session_options", "provider_options", "shared_files"):
        if field in entry:
            value = entry[field]
            if not isinstance(value, dict):
                raise ValueError(
                    f"variant.json file entry '{field}' must be an object (dict), "
                    f"got {type(value).__name__}"
                )
            if value:
                out[field] = value
    return out
