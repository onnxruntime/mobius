# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Generate the authoritative GGUF support census in the API documentation."""

from __future__ import annotations

__all__ = [
    "DOC_PATH",
    "check_document",
    "render_blocks",
    "render_document",
    "update_document",
]

import re
from collections import Counter
from pathlib import Path

from mobius.integrations.gguf._arch_registry import (
    _RUNTIME_VALIDATION_PENDING,
    iter_arch_specs,
)
from mobius.integrations.gguf._mmproj_registry import (
    MMPROJ_ARTIFACT_PINS,
    iter_projector_specs,
)
from mobius.integrations.gguf._quant_registry import (
    iter_quant_specs,
    render_quant_support_matrix,
)
from mobius.integrations.gguf._runtime_evidence import runtime_evidence
from mobius.integrations.gguf._spec import GGUFArchitectureSpec, StorageRole, Support
from mobius.integrations.gguf._tokenizer_alias_evidence import tokenizer_alias_evidence
from mobius.integrations.gguf._tokenizer_census import tokenizer_route_census
from mobius.integrations.gguf._tokenizer_evidence import iter_tokenizer_evidence
from mobius.integrations.gguf._tokenizer_registry import tokenizer_pre_policies
from mobius.integrations.gguf._upstream import (
    UPSTREAM_COMMIT,
    UPSTREAM_DATE,
    upstream_architectures,
)

DOC_PATH = Path(__file__).resolve().parents[4] / "docs" / "api" / "build_from_gguf.md"


def _cell(value: object) -> str:
    return str(value).replace("|", r"\|").replace("\n", " ")


def _status(verdicts: dict[str, Support]) -> str:
    return "; ".join(f"{name}={verdict.value}" for name, verdict in verdicts.items())


def _summary() -> str:
    architectures = iter_arch_specs()
    qtypes = iter_quant_specs()
    projectors = iter_projector_specs()
    tokenizers = tokenizer_pre_policies()
    tokenizer_statuses = Counter(record.current_status for record in tokenizer_route_census())
    stored = [spec for spec in qtypes if spec.readable and spec.role is StorageRole.QUANTIZED]
    routed = [
        spec
        for spec in stored
        if spec.native_preserve is not None
        or spec.affine_repack is not None
        or spec.dequantize is Support.SUPPORTED
    ]
    graph_counts = Counter(spec.graph.value for spec in architectures)
    runtime_counts = Counter(spec.runtime.value for spec in architectures)
    quantized_import_counts = Counter(spec.quantized_import.value for spec in architectures)
    projector_counts = {
        "graph-importable": sum(spec.is_importable for spec in projectors),
        "runtime-supported": sum(spec.runtime is Support.SUPPORTED for spec in projectors),
    }
    return "\n".join(
        (
            f"**Pinned source:** `ggml-org/llama.cpp@{UPSTREAM_COMMIT}` ({UPSTREAM_DATE}).",
            "",
            "| Census | Total | Closure |",
            "|---|---:|---|",
            (
                f"| Architectures | {len(architectures)} | "
                f"graph verdicts: {dict(sorted(graph_counts.items()))}; "
                f"importable: {sum(spec.is_importable for spec in architectures)}; "
                f"quantized import: {dict(sorted(quantized_import_counts.items()))}; "
                f"runtime: {dict(sorted(runtime_counts.items()))} |"
            ),
            (
                f"| Active stored qtypes | {len(stored)} | {len(routed)} have an import route; "
                f"{len(stored) - len(routed)} are explicitly deferred with no route |"
            ),
            (
                f"| Serialized projector strings | {len(projectors)} | "
                f"{dict(sorted(projector_counts.items()))} |"
            ),
            (
                f"| Tokenizer pre identifiers | {len(tokenizers)} | "
                f"{len({policy.canonical for policy in tokenizers.values()})} semantic groups; "
                f"route dispositions: {dict(sorted(tokenizer_statuses.items()))} |"
            ),
            "",
            (
                "`SUPPORTED` means the named capability is implemented and mechanically tested. "
                "`DEFERRED` means it is intentionally unavailable pending the stated work. "
                "`REJECTED` means the input or route is invalid by policy. Graph support proves "
                "construction/execution only; runtime support additionally requires a pinned real "
                "artifact, independent parity, and deterministic generation or stateful semantics. "
                "Tokenizer `copy` requires embedded ordered-vocabulary identity; `pinned-source` "
                "also binds the complete GGUF artifact, immutable Hub assets, reconstruction "
                "policy, semantic hashes, and representative token-ID vectors."
            ),
        )
    )


def _reason_code(verdicts: dict[str, Support]) -> str:
    config = verdicts.get("config", verdicts.get("metadata"))
    tensor = verdicts.get("tensor_map")
    graph = verdicts.get("graph")
    runtime = verdicts.get("runtime")
    quantized = verdicts.get("quantized_import")
    if config is Support.REJECTED:
        return (
            "CONFIG_REJECTED — The serialized architecture contract is deliberately refused."
        )
    if config is not Support.SUPPORTED:
        return "CONFIG_DEFERRED — Exact configuration ownership is not implemented."
    if tensor is Support.REJECTED:
        return "TENSOR_MAP_REJECTED — The serialized tensor contract is deliberately refused."
    if tensor is not Support.SUPPORTED:
        return "TENSOR_MAP_DEFERRED — Exact tensor-name closure is not implemented."
    if graph is Support.REJECTED:
        return "GRAPH_REJECTED — Executable graph construction is deliberately refused."
    if graph is not Support.SUPPORTED:
        return "GRAPH_DEFERRED — Executable graph construction is not implemented."
    if runtime is Support.REJECTED:
        return "RUNTIME_REJECTED — Runtime package publication is deliberately refused."
    if runtime is not Support.SUPPORTED and quantized is Support.REJECTED:
        return (
            "RUNTIME_EVIDENCE_PENDING / FLOAT_IMPORT_ONLY — Runtime evidence is incomplete, "
            "and packed quantized import is unavailable."
        )
    if runtime is not Support.SUPPORTED:
        return (
            "RUNTIME_EVIDENCE_PENDING — Exact real-artifact parity and deterministic runtime "
            "semantics are not yet evidenced."
        )
    if quantized is Support.REJECTED:
        return "FLOAT_IMPORT_ONLY — Runtime is evidenced only through explicit float import."
    return "EVIDENCED_SCOPE — Runtime publication is limited to registry-linked immutable evidence."


def _architecture_reason(spec: GGUFArchitectureSpec) -> str:
    capabilities = dict(spec.capabilities)
    code = _reason_code(capabilities).partition(" — ")[0]
    reason = spec.reason
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError(f"{spec.gguf_arch}: architecture registry reason must be non-empty")
    restriction_source = reason.removeprefix(_RUNTIME_VALIDATION_PENDING).strip() or reason
    restriction = re.split(r"(?<=[.!?])\s+", restriction_source, maxsplit=1)[0]
    return f"{code} — {restriction}"


def _architectures() -> str:
    rows = [
        (
            "| Canonical architecture | Aliases | Import route | Tensor exactness | "
            "Config/tensor/graph/runtime/quantized import | Restriction or evidence gap |"
        ),
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
        reason = _architecture_reason(spec)
        rows.append(
            f"| `{spec.gguf_arch}` | {aliases} | {_cell(route)} | {_cell(exactness)} | "
            f"{_cell(_status(spec.capabilities))} | {_cell(reason)} |"
        )
    return "\n".join(rows)


def _qtypes() -> str:
    return render_quant_support_matrix()


def _projectors() -> str:
    rows = [
        (
            "| Projector string | Modality | Paired text architecture | "
            "Metadata/tensor/graph/runtime | Exactness/evidence |"
        ),
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
            else _reason_code(dict(spec.verdicts))
        )
        rows.append(
            f"| `{spec.projector_type}` | {modalities} | {targets} | "
            f"{_cell(_status(dict(spec.verdicts)))} | {_cell(evidence or 'none')} |"
        )
    return "\n".join(rows)


def _tokenizers() -> str:
    rows = [
        (
            "| Exact identifier | Semantic group / pre-type | Default policy | Current status | "
            "Evidence / blocker |"
        ),
        "|---|---|---|---|---|",
    ]
    alias_proofs = tokenizer_alias_evidence()
    for record in tokenizer_route_census():
        disposition = (
            f"`{record.evidence_id}`"
            if record.evidence_id is not None
            else f"`{record.blocker_category}`"
        )
        if record.identifier in alias_proofs:
            disposition += (
                f"; `llama-vocab.cpp:L{alias_proofs[record.identifier].dispatch_line}`"
            )
        rows.append(
            f"| `{record.identifier}` | `{record.semantic_group}` / `{record.pre_type}` | "
            f"`{record.default_policy}` | `{record.current_status}` | {disposition} |"
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


def _runtime_evidence_table() -> str:
    records = []
    seen: set[str] = set()
    for spec in iter_arch_specs():
        for evidence_id in spec.runtime_evidence_ids:
            if evidence_id in seen:
                continue
            evidence = runtime_evidence(evidence_id)
            if evidence is None:
                raise ValueError(f"Missing runtime evidence {evidence_id!r}")
            seen.add(evidence_id)
            records.append(evidence)
    rows = [
        "| Evidence ID | GGUF identity | Config identity | Tokenizer identity | Runtime proof |",
        "|---|---|---|---|---|",
    ]
    for evidence in sorted(records, key=lambda item: item.evidence_id):
        assets = ", ".join(
            f"`{name}` {size:,} B `{sha256}`"
            for name, size, sha256 in evidence.tokenizer_assets
        )
        rows.append(
            f"| `{evidence.evidence_id}` | `{evidence.repository}@{evidence.revision}`<br>"
            f"`{evidence.filename}`<br>{evidence.size:,} B<br>`{evidence.lfs_sha256}` | "
            f"`{evidence.config_repository}@{evidence.config_revision}` | "
            f"`{evidence.tokenizer_repository}@{evidence.tokenizer_revision}`<br>{assets}<br>"
            f"metadata `{evidence.tokenizer_metadata_sha256}` | "
            f"{evidence.runtime} {evidence.runtime_version}; {evidence.parity_kind}; "
            f"{evidence.stateful_semantics} |"
        )
    return "\n".join(rows)


def _tokenizer_evidence_table() -> str:
    rows = [
        "| Evidence ID | GGUF identity | Official source | Exact tokenizer proof |",
        "|---|---|---|---|",
    ]
    for evidence in iter_tokenizer_evidence():
        assets = ", ".join(
            f"`{name}` {size:,} B `{sha256}`"
            for name, size, sha256 in evidence.tokenizer_assets
        )
        encodings = "<br>".join(
            f"`{text.encode('unicode_escape').decode()}` → `{list(token_ids)}`"
            for text, token_ids in evidence.representative_encodings
        )
        padding_start, padding_end = evidence.deterministic_padding_range
        padding = (
            f"unused `[PAD{{id}}]` IDs `{padding_start}..{padding_end}`"
            if padding_start <= padding_end
            else "no GGUF-only padding extension"
        )
        identity = (
            f"`{evidence.repository}@{evidence.revision}`<br>"
            f"`{evidence.filename}`<br>{evidence.size:,} B<br>`{evidence.lfs_sha256}`"
        )
        source = (
            f"`{evidence.tokenizer_repository}@{evidence.tokenizer_revision}`<br>"
            f"`{evidence.source_config_asset[0]}` {evidence.source_config_asset[1]:,} B "
            f"`{evidence.source_config_asset[2]}`<br>{assets}"
        )
        proof = (
            f"validated identifiers `{list(evidence.validated_identifiers)}`<br>"
            f"metadata `{evidence.tokenizer_metadata_sha256}`<br>"
            f"tokens {evidence.token_count:,} `{evidence.ordered_vocabulary_sha256}`<br>"
            f"merges {evidence.merge_count:,} `{evidence.ordered_merges_sha256}`<br>"
            f"types `{evidence.ordered_token_types_sha256}`; scores={evidence.score_count}<br>"
            f"source IDs `0..{evidence.source_token_count - 1}`; {padding}; "
            f"rows={evidence.embedding_vocabulary_size:,}<br>"
            f"materialized `{evidence.materialized_tokenizer_sha256}`<br>{encodings}"
        )
        rows.append(
            "| "
            + " | ".join(
                _cell(value)
                for value in (f"`{evidence.evidence_id}`", identity, source, proof)
            )
            + " |"
        )
    return "\n".join(rows)


def _projector_evidence_table() -> str:
    rows = [
        "| Artifact ID | Immutable sidecar | Bytes | SHA-256 | Projector types |",
        "|---|---|---:|---|---|",
    ]
    for pin in MMPROJ_ARTIFACT_PINS:
        rows.append(
            f"| `{pin.artifact_id}` | `{pin.repository}@{pin.revision}`<br>"
            f"`{pin.filename}` | {pin.size:,} | `{pin.lfs_sha256}` | "
            f"{', '.join(f'`{item}`' for item in pin.projector_types)} |"
        )
    return "\n".join(rows)


def render_document() -> str:
    """Render the complete concise API document from live registries and evidence."""
    blocks = render_blocks()
    return f"""# `build_from_gguf()`

Build ONNX packages directly from GGUF metadata and tensors without tracing PyTorch.
Support is capability-specific: graph import does not imply runtime packaging.

<!-- BEGIN GGUF CLOSURE SUMMARY -->

{blocks["summary"]}

<!-- END GGUF CLOSURE SUMMARY -->

## Usage

```python
from mobius.integrations.gguf import build_from_gguf

package = build_from_gguf("model.gguf")
package.save("output")
```

Use `keep_quantized=False` for explicit float import. Pass `mmproj=` only for an
evidenced multimodal sidecar. The CLI equivalent is `mobius build model.gguf -o output`.

## API

```python
build_from_gguf(
    gguf_path,
    *,
    task=None,
    dtype=None,
    keep_quantized=True,
    execution_provider="default",
    mmproj=None,
    static_cache=False,
    max_seq_len=None,
    allow_dense_moe=None,
    reuse_gguf_weights=False,
    target_config=None,
)
```

The function returns a `ModelPackage`. Import validates architecture metadata, exact tensor
closure, shapes, qtypes, and selected graph route before publication. Source reuse requires
the original immutable GGUF at runtime. Runtime packages additionally require an exact
artifact, graph, tokenizer, runtime version, parity proof, and deterministic state/generation
evidence match.

## Runtime evidence

{_runtime_evidence_table()}

Runtime support above is independent from tokenizer materialization support below.

## Tokenizer evidence

{_tokenizer_evidence_table()}

```python
from mobius.integrations.gguf import materialize_evidenced_gguf_tokenizer

materialize_evidenced_gguf_tokenizer("Qwen3.5-0.8B-Q4_0.gguf", "tokenizer")
```

Each row is independently artifact-scoped. Evidence proves ordered tokens, merges or scores,
token types, special IDs, source pipeline/config assets, representative encodings, embedding
alignment, any non-matchable padding extension, and the final materialized hash. Shared rows
additionally require every exact identifier to select the same implementation and flag overrides
in pinned `llama-vocab.cpp`; the tokenizer matrix links each promoted route to its dispatch line.
This does not claim graph or runtime support.

## Supported GGUF architectures

Reason codes are concise user-facing categories; detailed architecture audits remain in
`_arch_registry.py` and its tests.

<!-- BEGIN GGUF SUPPORT MATRIX (generated; see _arch_registry.py) -->

{blocks["architectures"]}

<!-- END GGUF SUPPORT MATRIX -->

## Stored quantization types

<!-- BEGIN GGUF QUANTIZATION MATRIX (generated; see _quant_registry.py) -->

{blocks["qtypes"]}

<!-- END GGUF QUANTIZATION MATRIX -->

## Multimodal projector sidecars

{_projector_evidence_table()}

<!-- BEGIN GGUF MMPROJ SUPPORT MATRIX (generated; see _mmproj_registry.py) -->

{blocks["projectors"]}

<!-- END GGUF MMPROJ SUPPORT MATRIX -->

## Tokenizer pre-types

The pre-type is never sufficient evidence by itself. The generated census preserves all aliases
and gives every route an exact evidence ID or concrete compiled-semantics blocker.

<!-- BEGIN GGUF TOKENIZER PRE SUPPORT MATRIX -->

{blocks["tokenizers"]}

<!-- END GGUF TOKENIZER PRE SUPPORT MATRIX -->

## Validation boundary

Normal tests are deterministic and network-free; committed registry records contain compact
immutable identities and semantic hashes. Real-artifact qualification is performed serially
with pinned revisions, full SHA-256 verification, at least twice the artifact size free, and
independent runtime evidence where runtime support is claimed.
"""


def update_document(path: Path = DOC_PATH) -> str:
    """Return the complete generated document after rejecting stale upstream pins."""
    text = path.read_text(encoding="utf-8")
    pins = set(
        re.findall(
            r"llama\.cpp(?: commit)?(?:@|\s|`)*([0-9a-f]{40})",
            text,
            flags=re.IGNORECASE,
        )
    )
    if pins - {UPSTREAM_COMMIT}:
        raise ValueError(f"Stale llama.cpp pins outside generated blocks: {sorted(pins)}")
    return render_document()


def check_document(path: Path = DOC_PATH) -> bool:
    """Return whether *path* exactly matches the generated registry content."""
    return path.read_text(encoding="utf-8") == update_document(path)
