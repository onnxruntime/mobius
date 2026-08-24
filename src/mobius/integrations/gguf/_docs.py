# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Generate the authoritative GGUF support census in the API documentation."""

from __future__ import annotations

__all__ = ["DOC_PATH", "check_document", "render_blocks", "update_document"]

from collections import Counter
from pathlib import Path
import re

from mobius.integrations.gguf._arch_registry import iter_arch_specs
from mobius.integrations.gguf._mmproj_registry import iter_projector_specs
from mobius.integrations.gguf._quant_registry import (
    iter_quant_specs,
    render_quant_support_matrix,
)
from mobius.integrations.gguf._spec import StorageRole, Support
from mobius.integrations.gguf._tokenizer_registry import tokenizer_pre_policies
from mobius.integrations.gguf._upstream import (
    UPSTREAM_COMMIT,
    UPSTREAM_DATE,
    upstream_architectures,
)

DOC_PATH = Path(__file__).resolve().parents[4] / "docs" / "api" / "build_from_gguf.md"

_MARKERS = {
    "summary": ("<!-- BEGIN GGUF CLOSURE SUMMARY -->", "<!-- END GGUF CLOSURE SUMMARY -->"),
    "architectures": (
        "<!-- BEGIN GGUF SUPPORT MATRIX (generated; see _arch_registry.py) -->",
        "<!-- END GGUF SUPPORT MATRIX -->",
    ),
    "qtypes": (
        "<!-- BEGIN GGUF QUANTIZATION MATRIX (generated; see _quant_registry.py) -->",
        "<!-- END GGUF QUANTIZATION MATRIX -->",
    ),
    "projectors": (
        "<!-- BEGIN GGUF MMPROJ SUPPORT MATRIX (generated; see _mmproj_registry.py) -->",
        "<!-- END GGUF MMPROJ SUPPORT MATRIX -->",
    ),
    "tokenizers": (
        "<!-- BEGIN GGUF TOKENIZER PRE SUPPORT MATRIX -->",
        "<!-- END GGUF TOKENIZER PRE SUPPORT MATRIX -->",
    ),
}


def _cell(value: object) -> str:
    return str(value).replace("|", r"\|").replace("\n", " ")


def _status(verdicts: dict[str, Support]) -> str:
    return "; ".join(f"{name}={verdict.value}" for name, verdict in verdicts.items())


def _summary() -> str:
    architectures = iter_arch_specs()
    qtypes = iter_quant_specs()
    projectors = iter_projector_specs()
    tokenizers = tokenizer_pre_policies()
    stored = [
        spec for spec in qtypes if spec.readable and spec.role is StorageRole.QUANTIZED
    ]
    routed = [
        spec
        for spec in stored
        if spec.native_preserve is not None
        or spec.affine_repack is not None
        or spec.dequantize is Support.SUPPORTED
    ]
    graph_counts = Counter(spec.graph.value for spec in architectures)
    runtime_counts = Counter(spec.runtime.value for spec in architectures)
    quantized_import_counts = Counter(
        spec.quantized_import.value for spec in architectures
    )
    projector_counts = {
        "graph-importable": sum(spec.is_importable for spec in projectors),
        "runtime-supported": sum(spec.runtime is Support.SUPPORTED for spec in projectors),
    }
    return "\n".join(
        (
            f"**Pinned source:** `ggml-org/llama.cpp@{UPSTREAM_COMMIT}` "
            f"({UPSTREAM_DATE}).",
            "",
            "| Census | Total | Closure |",
            "|---|---:|---|",
            f"| Architectures | {len(architectures)} | "
            f"graph verdicts: {dict(sorted(graph_counts.items()))}; "
            f"importable: {sum(spec.is_importable for spec in architectures)}; "
            f"quantized import: {dict(sorted(quantized_import_counts.items()))}; "
            f"runtime: {dict(sorted(runtime_counts.items()))} |",
            f"| Active stored qtypes | {len(stored)} | {len(routed)} have an import route; "
            f"{len(stored) - len(routed)} are explicitly deferred with no route |",
            f"| Serialized projector strings | {len(projectors)} | "
            f"{dict(sorted(projector_counts.items()))} |",
            f"| Tokenizer pre identifiers | {len(tokenizers)} | "
            f"{len({policy.canonical for policy in tokenizers.values()})} semantic groups; "
            "all default to deferred and become exact-copy only with a validated embedded "
            "`tokenizer.huggingface.json` |",
            "",
            "`SUPPORTED` means the named capability is implemented and mechanically tested. "
            "`DEFERRED` means it is intentionally unavailable pending the stated work. "
            "`REJECTED` means the input or route is invalid by policy. Graph support proves "
            "construction/execution only; runtime support additionally requires a pinned real "
            "artifact, independent parity, and deterministic generation or stateful semantics. "
            "Tokenizer `copy` delegates algorithm semantics to an embedded, vocabulary-identical "
            "tokenizer JSON; it is not a reconstructed or independently proven tokenizer.",
        )
    )


def _architectures() -> str:
    rows = [
        "| Canonical architecture | Aliases | Import route | Tensor exactness | "
        "Config/tensor/graph/runtime/quantized import | Restriction or evidence gap |",
        "|---|---|---|---|---|---|",
    ]
    upstream = upstream_architectures()
    for spec in sorted(iter_arch_specs(), key=lambda item: item.gguf_arch):
        aliases = ", ".join(f"`{alias}`" for alias in sorted(spec.aliases)) or "—"
        route_parts = []
        if spec.is_importable and spec.model_type:
            route_parts.append(f"model=`{spec.model_type}`")
        if spec.is_importable and spec.module_type:
            route_parts.append(f"module=`{spec.module_type}`")
        if spec.is_importable and spec.tensor_map_recipe:
            route_parts.append(
                "tensor=" + "+".join(f"`{name}`" for name in spec.tensor_map_recipe)
            )
        if spec.is_importable and spec.vlm_builder:
            route_parts.append(f"mmproj=`{spec.vlm_builder}`")
        if route_parts:
            route = "; ".join(route_parts)
        elif spec.config is not Support.SUPPORTED:
            route = "none (fails before config extraction)"
        elif spec.tensor_map is not Support.SUPPORTED:
            route = "none (no tensor mapping route)"
        else:
            route = "none (no graph construction route)"
        exactness = upstream[spec.gguf_arch].tensor_closure_status or "not claimed"
        reason = spec.reason or "No restriction; evidence record is registry-backed."
        rows.append(
            f"| `{spec.gguf_arch}` | {aliases} | {_cell(route)} | {_cell(exactness)} | "
            f"{_cell(_status(spec.capabilities))} | {_cell(reason)} |"
        )
    return "\n".join(rows)


def _qtypes() -> str:
    return render_quant_support_matrix()


def _projectors() -> str:
    rows = [
        "| Projector string | Modality | Paired text architecture | "
        "Metadata/tensor/graph/runtime | Exactness/evidence |",
        "|---|---|---|---|---|",
    ]
    for spec in sorted(iter_projector_specs(), key=lambda item: item.projector_type):
        modalities = ", ".join(
            modality.value for modality in sorted(spec.modalities, key=lambda item: item.value)
        )
        targets = (
            ", ".join(f"`{target}`" for target in sorted(spec.target_architectures)) or "—"
        )
        evidence = (
            "artifact pins=" + ", ".join(f"`{item}`" for item in spec.real_artifact_ids)
            if spec.real_artifact_ids
            else spec.reason
        )
        rows.append(
            f"| `{spec.projector_type}` | {modalities} | {targets} | "
            f"{_cell(_status(dict(spec.verdicts)))} | {_cell(evidence or 'none')} |"
        )
    return "\n".join(rows)


def _tokenizers() -> str:
    rows = [
        "| Exact identifier | Canonical semantic group | Pinned pre-type | Default route | "
        "Exactness/restriction |",
        "|---|---|---|---|---|",
    ]
    for identifier, policy in sorted(tokenizer_pre_policies().items()):
        rows.append(
            f"| `{identifier}` | `{policy.canonical}` | `{policy.pre_type}` | "
            f"`{policy.default_route}` | Exact-copy only after embedded tokenizer JSON and "
            "ordered vocabulary validation; otherwise runtime packaging is deferred. |"
        )
    return "\n".join(rows)


def render_blocks() -> dict[str, str]:
    """Render every generated documentation block from the live registries."""
    return {
        "summary": _summary(),
        "architectures": _architectures(),
        "qtypes": _qtypes(),
        "projectors": _projectors(),
        "tokenizers": _tokenizers(),
    }


def _replace(text: str, begin: str, end: str, body: str) -> str:
    if text.count(begin) != 1 or text.count(end) != 1:
        raise ValueError(f"Expected exactly one generated block delimited by {begin!r}/{end!r}")
    before, remainder = text.split(begin, 1)
    _, after = remainder.split(end, 1)
    return f"{before}{begin}\n\n{body}\n\n{end}{after}"


def update_document(path: Path = DOC_PATH) -> str:
    """Return the documentation with every generated block refreshed."""
    text = path.read_text(encoding="utf-8")
    for name, body in render_blocks().items():
        text = _replace(text, *_MARKERS[name], body)
    pins = set(
        re.findall(
            r"llama\.cpp(?: commit)?(?:@|\s|`)*([0-9a-f]{40})",
            text,
            flags=re.IGNORECASE,
        )
    )
    if pins - {UPSTREAM_COMMIT}:
        raise ValueError(f"Stale llama.cpp pins outside generated blocks: {sorted(pins)}")
    return text


def check_document(path: Path = DOC_PATH) -> bool:
    """Return whether *path* exactly matches the generated registry content."""
    return path.read_text(encoding="utf-8") == update_document(path)
