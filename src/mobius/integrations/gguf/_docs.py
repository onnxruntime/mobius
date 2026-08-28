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

import json
import re
from collections import Counter
from pathlib import Path

from mobius.integrations.gguf._arch_registry import (
    _RUNTIME_VALIDATION_PENDING,
    iter_arch_specs,
)
from mobius.integrations.gguf._artifact_blocker_evidence import (
    iter_artifact_blocker_evidence,
)
from mobius.integrations.gguf._draft import has_direct_draft_runtime
from mobius.integrations.gguf._mmproj_registry import (
    LLAMA_CPP_MMPROJ_SHA,
    MMPROJ_ARTIFACT_AVAILABILITY_PINS,
    MMPROJ_ARTIFACT_PINS,
    iter_projector_source_evidence,
    iter_projector_specs,
)
from mobius.integrations.gguf._mtp_runtime_evidence import (
    iter_mtp_runtime_evidence,
)
from mobius.integrations.gguf._quant_registry import (
    iter_quant_specs,
    render_quant_support_matrix,
)
from mobius.integrations.gguf._route_census import (
    RECENT_PR_DEPENDENCIES,
    render_remaining_route_batches,
)
from mobius.integrations.gguf._runtime_blocker_evidence import (
    iter_runtime_blocker_evidence,
)
from mobius.integrations.gguf._runtime_evidence import runtime_evidence
from mobius.integrations.gguf._spec import GGUFArchitectureSpec, StorageRole, Support
from mobius.integrations.gguf._tokenizer_alias_evidence import tokenizer_alias_evidence
from mobius.integrations.gguf._tokenizer_census import tokenizer_route_census
from mobius.integrations.gguf._tokenizer_evidence import (
    iter_tokenizer_blocker_evidence,
    iter_tokenizer_evidence,
)
from mobius.integrations.gguf._tokenizer_registry import tokenizer_pre_policies
from mobius.integrations.gguf._upstream import (
    UPSTREAM_COMMIT,
    UPSTREAM_DATE,
    upstream_architectures,
)

_DRAFT_RUNTIME_EVIDENCE_PATH = (
    Path(__file__).parents[4] / "testdata" / "evidence" / "gguf_draft_runtime_evidence.json"
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
        "runtime-evidenced": sum(spec.runtime is Support.SUPPORTED for spec in projectors),
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
                "`REJECTED` means the input or route is invalid by policy. Graph support controls "
                "export. The separate runtime verdict records pinned real-artifact validation, "
                "independent parity, and deterministic generation or stateful semantics; it never "
                "gates export of a faithfully represented graph and package contract. "
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
        return "RUNTIME_EVIDENCE_REJECTED — The recorded runtime route is invalid."
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
    return "EVIDENCED_SCOPE — Registry-linked immutable runtime evidence is available."


def _architecture_reason(spec: GGUFArchitectureSpec) -> str:
    if has_direct_draft_runtime(spec.gguf_arch):
        return (
            "DIRECT_ORT_EVIDENCED / RUNTIME_UNVALIDATED — Exact target-coupled "
            "direct ORT acceptance, rollback, and deterministic generation are evidenced; "
            "higher-level runtime compatibility remains advisory."
        )
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
        if spec.preflight_only:
            route_parts.append("header/config/tensor preflight only")
        if (spec.is_importable or spec.preflight_only) and spec.model_type:
            route_parts.append(f"model=`{spec.model_type}`")
        if spec.is_importable and spec.module_type:
            route_parts.append(f"module=`{spec.module_type}`")
        if (spec.is_importable or spec.preflight_only) and spec.tensor_map_recipe:
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
            "| Projector string | Modality | Graph role / route | Paired text architecture | "
            "Metadata/tensor/graph/runtime | Exactness/evidence |"
        ),
        "|---|---|---|---|---|---|",
    ]
    for spec in sorted(iter_projector_specs(), key=lambda item: item.projector_type):
        modalities = ", ".join(
            modality.value for modality in sorted(spec.modalities, key=lambda item: item.value)
        )
        targets = (
            ", ".join(f"`{target}`" for target in sorted(spec.target_architectures)) or "—"
        )
        roles = ", ".join(role.value for role in spec.model_roles)
        route = (
            f"{roles or 'paired package'} via `{spec.sidecar_builder or spec.builder}`"
            if spec.sidecar_builder or spec.builder
            else "—"
        )
        evidence_parts = []
        if spec.real_artifact_ids:
            evidence_parts.append(
                "artifact pins=" + ", ".join(f"`{item}`" for item in spec.real_artifact_ids)
            )
        if spec.source_evidence_ids:
            evidence_parts.append(
                "source evidence="
                + ", ".join(f"`{item}`" for item in spec.source_evidence_ids)
            )
        evidence = "; ".join(evidence_parts) or _reason_code(dict(spec.verdicts))
        rows.append(
            f"| `{spec.projector_type}` | {modalities} | {_cell(route)} | {targets} | "
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
            else (
                f"`{record.blocker_category}`"
                + (
                    f" (`{record.blocker_evidence_id}`)"
                    if record.blocker_evidence_id is not None
                    else ""
                )
            )
        )
        if record.candidate_disposition is not None:
            route_identity = ""
            if record.artifact_architecture is not None:
                declared = record.declared_pre_identifier or "absent"
                effective = record.effective_pre_identifier or "unresolved"
                route_identity = (
                    f"; architecture `{record.artifact_architecture}`, declared pre "
                    f"`{declared}`, effective pre `{effective}`"
                )
            disposition += (
                f"; `{record.artifact_repository}@{record.artifact_revision}` / "
                f"`{record.artifact_filename}` vs "
                f"`{record.tokenizer_repository}@{record.tokenizer_revision}`: "
                f"{record.candidate_disposition}{route_identity}"
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
        "remaining_routes": render_remaining_route_batches(),
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
            f"ONNX Runtime {evidence.onnxruntime_version} "
            f"`{evidence.execution_provider}`; {evidence.runtime} "
            f"{evidence.runtime_version}; result={evidence.result}; "
            f"{evidence.parity_kind}; {evidence.stateful_semantics}"
            f"{'; ' + evidence.limitations if evidence.limitations else ''} |"
        )
    for mtp_evidence in iter_mtp_runtime_evidence():
        layouts = "<br>".join(
            (
                f"{layout.name}: "
                + ", ".join(
                    f"`{artifact.repository}@{artifact.revision}/{artifact.filename}` "
                    f"{artifact.size:,} B `{artifact.lfs_sha256}`"
                    for artifact in layout.artifacts
                )
                + f"<br>total {layout.total_size:,} B; "
                + (
                    "within 16 GiB"
                    if layout.within_bounded_artifact_policy
                    else "above 16 GiB"
                )
            )
            for layout in mtp_evidence.layouts
        )
        discriminator = mtp_evidence.target_only_discriminator
        if discriminator is not None:
            layouts += (
                f"<br>target-only discriminator: `{discriminator.filename}` "
                f"{discriminator.size:,} B `{discriminator.lfs_sha256}`"
            )
        limitations = "<br>".join(mtp_evidence.downstream_limitations)
        deferrals = "<br>".join(mtp_evidence.separate_deferrals)
        synthetic = (
            f"; reduced direct-ORT coordinator "
            f"{dict(mtp_evidence.synthetic_acceptance_statistics)}"
            if mtp_evidence.synthetic_coordinator_test is not None
            else ""
        )
        rows.append(
            f"| `{mtp_evidence.evidence_id}` | {layouts} | "
            f"`{mtp_evidence.config_repository}@{mtp_evidence.config_revision}` "
            f"`{mtp_evidence.config_sha256}` | "
            f"`{mtp_evidence.tokenizer_repository}@{mtp_evidence.tokenizer_revision}`; "
            f"separately deferred; metadata `{mtp_evidence.tokenizer_metadata_sha256}` | "
            f"status={mtp_evidence.result}; graph/package hashes unclaimed; "
            f"ORT {mtp_evidence.onnxruntime_version} "
            f"`{mtp_evidence.execution_provider}`; {mtp_evidence.runtime} "
            f"{mtp_evidence.runtime_version} source "
            f"`{mtp_evidence.runtime_source_revision}`{synthetic}; "
            f"{limitations}<br>{deferrals} |"
        )
    return "\n".join(rows)


def _runtime_blocker_evidence_table() -> str:
    rows = [
        "| Evidence ID | Pinned candidate | Bounded result | Withheld runtime claims |",
        "|---|---|---|---|",
    ]
    for evidence in iter_runtime_blocker_evidence():
        blockers = "<br>".join(evidence.blockers)
        withheld = ", ".join(evidence.withheld_checks)
        rows.append(
            f"| `{evidence.evidence_id}` | `{evidence.repository}@{evidence.revision}`<br>"
            f"`{evidence.filename}`<br>{evidence.size:,} B<br>`{evidence.lfs_sha256}` | "
            f"config/tokenizer `{evidence.config_repository}@{evidence.config_revision}`; "
            f"GGUF tokenizer metadata `{evidence.tokenizer_metadata_sha256}`; "
            f"result={evidence.result}; {evidence.tensor_count} tensors / "
            f"{evidence.logical_parameter_count:,} parameters; "
            f"ORT {evidence.onnxruntime_version} / {evidence.execution_provider}; "
            f"{evidence.runtime} {evidence.runtime_version}; {blockers} | {withheld} |"
        )
    for artifact in iter_artifact_blocker_evidence():
        files = "<br>".join(
            f"`{file.path}` {file.size:,} B `{file.lfs_sha256}`" for file in artifact.files
        )
        rows.append(
            f"| `{artifact.evidence_id}` | `{artifact.repository}@{artifact.revision}`<br>"
            f"{files}<br>total {artifact.total_size:,} B | blocked; "
            f"{artifact.blocker} | real-weight full-logit parity, runtime packaging, "
            "deterministic generation |"
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
        special_encodings = "<br>".join(
            f"`{text.encode('unicode_escape').decode()}` + specials → `{list(token_ids)}`"
            for text, token_ids in evidence.representative_special_encodings
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
            f"<br>architecture `{evidence.architecture}`; declared pre "
            f"`{'absent' if evidence.uses_model_pre_fallback else evidence.pre_identifier}`; "
            f"effective pre `{evidence.pre_identifier}`"
        )
        source = (
            f"`{evidence.tokenizer_repository}@{evidence.tokenizer_revision}`<br>"
            f"`{evidence.source_config_asset[0]}` {evidence.source_config_asset[1]:,} B "
            f"`{evidence.source_config_asset[2]}`<br>{assets}"
        )
        reconstruction = (
            f"<br>GGUF-native GPT4O reconstruction; {evidence.source_disposition}"
            if evidence.reconstruct_gpt4o_from_gguf
            else (
                f"<br>GGUF-native Gemma4 reconstruction; {evidence.source_disposition}"
                if evidence.reconstruct_gemma4_from_gguf
                else ""
            )
        )
        oracle = (
            f"<br>llama.cpp oracle `{evidence.llamacpp_oracle[0]}`: "
            f"{evidence.llamacpp_oracle[1]} cases "
            f"`{evidence.llamacpp_oracle[2]}`"
            if evidence.llamacpp_oracle is not None
            else ""
        )
        proof = (
            f"validated identifiers `{list(evidence.validated_identifiers)}`<br>"
            f"metadata `{evidence.tokenizer_metadata_sha256}`<br>"
            f"tokens {evidence.token_count:,} `{evidence.ordered_vocabulary_sha256}`<br>"
            f"merges {evidence.merge_count:,} `{evidence.ordered_merges_sha256}`<br>"
            f"types `{evidence.ordered_token_types_sha256}`; scores={evidence.score_count}<br>"
            f"user-defined IDs `{list(evidence.user_defined_token_ids)}`; "
            f"source added tokens={evidence.source_added_token_count} "
            f"`{evidence.ordered_source_added_tokens_sha256}`<br>"
            f"pipeline `{dict(evidence.pipeline_sha256)}`; "
            f"chat `{evidence.chat_template_sha256}`<br>"
            f"source IDs `0..{evidence.source_token_count - 1}`; {padding}; "
            f"rows={evidence.embedding_vocabulary_size:,}<br>"
            f"materialized `{evidence.materialized_tokenizer_sha256}`<br>{encodings}"
            f"{'<br>' + special_encodings if special_encodings else ''}"
            f"{reconstruction}{oracle}"
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


def _tokenizer_blocker_evidence_table() -> str:
    rows = []
    for evidence in iter_tokenizer_blocker_evidence():
        assets = ", ".join(
            f"`{name}` {size:,} B `{sha256}`"
            for name, size, sha256 in evidence.tokenizer_assets
        )
        text, llamacpp_ids, source_ids = evidence.mismatch
        identity = (
            f"`{evidence.repository}@{evidence.revision}`<br>"
            f"`{evidence.filename}`<br>{evidence.size:,} B<br>`{evidence.lfs_sha256}`<br>"
            f"`{evidence.tokenizer_repository}@{evidence.tokenizer_revision}`<br>{assets}"
            f"<br>`{evidence.source_config_asset[0]}` "
            f"{evidence.source_config_asset[1]:,} B `{evidence.source_config_asset[2]}`"
        )
        if evidence.bounded_header_bytes is not None:
            identity += (
                f"<br>first {evidence.bounded_header_bytes:,} B "
                f"`{evidence.bounded_header_sha256}`"
            )
        closure = (
            f"architecture `{evidence.architecture}`; pre `{evidence.pre_identifier}`<br>"
            f"metadata `{evidence.tokenizer_metadata_sha256}`<br>"
            f"tokens {evidence.token_count:,} `{evidence.ordered_vocabulary_sha256}`; "
            f"source {evidence.source_token_count:,} `{evidence.source_vocabulary_sha256}`<br>"
            f"merges {evidence.merge_count:,} `{evidence.ordered_merges_sha256}`; "
            f"source `{evidence.source_merges_sha256}`<br>"
            f"scores={evidence.score_count}; types `{evidence.ordered_token_types_sha256}`<br>"
            f"added tokens `{evidence.source_added_tokens_sha256}`; "
            f"chat `{evidence.chat_template_sha256}`<br>"
            f"normalizer `{evidence.source_normalizer}`; "
            f"pipeline `{evidence.source_pipeline_sha256}`"
        )
        if (
            evidence.source_model_token_count is not None
            and evidence.source_merge_count is not None
            and evidence.source_score_mismatch_count is not None
            and evidence.source_type_mismatch_count is not None
            and evidence.source_chat_template_sha256 is not None
        ):
            closure += (
                f"<br>source model tokens={evidence.source_model_token_count:,}; "
                f"source merges={evidence.source_merge_count:,}; "
                f"score mismatches={evidence.source_score_mismatch_count:,}; "
                f"type mismatches={evidence.source_type_mismatch_count:,}<br>"
                f"GGUF chat `{evidence.chat_template_sha256}` vs source "
                f"`{evidence.source_chat_template_sha256}`"
            )
        if evidence.blocked_identifiers:
            closure += (
                f"<br>exact aliases={list(evidence.blocked_identifiers)}; "
                f"materialized `{evidence.materialized_tokenizer_sha256}`; "
                f"config `{evidence.source_tokenizer_config_sha256}`<br>"
                f"pipeline components={dict(evidence.source_pipeline_component_sha256)}; "
                f"added-token type mismatches="
                f"{evidence.source_added_token_type_mismatch_count}"
            )
        witness = (
            f"{evidence.disposition}<br>"
            f"`{text.encode('unicode_escape').decode()}`: llama.cpp `{list(llamacpp_ids)}` "
            f"vs source `{list(source_ids)}`<br>"
            f"corpus `{evidence.oracle_corpus_sha256}`; "
            f"llama.cpp oracle `{evidence.llamacpp_oracle[0]}`: "
            f"{evidence.llamacpp_oracle[1]} cases `{evidence.llamacpp_oracle[2]}`"
        )
        if evidence.oracle_mismatch_count is not None:
            witness += (
                f"<br>{evidence.oracle_mismatch_count} mismatches "
                f"{list(evidence.oracle_mismatch_count_by_mode)} by mode; "
                f"source oracle `{evidence.source_oracle_sha256}`"
            )
        if evidence.blocked_identifiers:
            witness += (
                f"<br>dispatch oracles={dict(evidence.dispatch_oracles)}; "
                f"discriminator={evidence.dispatch_discriminator}; "
                f"detokenize mismatches={evidence.oracle_detokenize_mismatch_count} "
                f"{list(evidence.oracle_detokenize_mismatch_count_by_mode)} by mode"
            )
            if evidence.first_detokenize_mismatch is not None:
                detokenize_text, llamacpp_hex, source_hex = evidence.first_detokenize_mismatch
                witness += (
                    f"; `{detokenize_text.encode('unicode_escape').decode()}`: "
                    f"llama.cpp hex `{llamacpp_hex}` vs source hex `{source_hex}`"
                )
        rows.append(
            f"- `{evidence.evidence_id}` — **GGUF/source:** {_cell(identity)}; "
            f"**closure:** {_cell(closure)}; **witness:** {_cell(witness)}"
        )
    return "\n".join(rows)


def _projector_evidence_table() -> str:
    generic_types = {"adapter", "ldp", "ldpv2", "mlp", "resampler"}
    generic_total = sum(
        pin.size + (pin.paired_text_size or 0)
        for pin in MMPROJ_ARTIFACT_PINS
        if generic_types.intersection(pin.projector_types)
    )
    rows = [
        (
            "| Artifact ID | Immutable sidecar | Bytes | SHA-256 | Projector types | "
            "Paired text | Processor source |"
        ),
        "|---|---|---:|---|---|---|---|",
    ]
    for pin in MMPROJ_ARTIFACT_PINS:
        sidecar = f"`{pin.repository}@{pin.revision}`<br>`{pin.filename}`"
        if pin.bounded_header_bytes is not None and pin.bounded_header_sha256 is not None:
            sidecar += (
                f"<br>first {pin.bounded_header_bytes:,} B `{pin.bounded_header_sha256}`"
            )
        paired_text = f"`{pin.paired_text_target}`"
        if pin.paired_text_repository and pin.paired_text_revision:
            paired_text = (
                f"`{pin.paired_text_repository}@{pin.paired_text_revision}`<br>"
                f"`{pin.paired_text_target}`"
            )
            if pin.paired_text_size is not None:
                paired_text += f"<br>{pin.paired_text_size:,} bytes"
        processor = "—"
        if pin.processor_repository and pin.processor_revision:
            processor = f"`{pin.processor_repository}@{pin.processor_revision}`"
        rows.append(
            f"| `{pin.artifact_id}` | {sidecar} | {pin.size:,} | `{pin.lfs_sha256}` | "
            f"{', '.join(f'`{item}`' for item in pin.projector_types)} | "
            f"{paired_text} | {processor} |"
        )
    rows.extend(
        (
            "",
            (
                f"The five generic projector evidence pairs total **{generic_total:,} bytes** "
                "(sidecars plus paired text GGUFs), below the 16 GiB evidence budget. "
                "Runtime remains deferred; four routes have independent nonzero-weight graph "
                "parity, while MiniCPM resampler remains component-only and graph-deferred."
            ),
        )
    )
    return "\n".join(rows)


def _projector_source_evidence_table() -> str:
    rows = [
        "| Evidence ID | Immutable sources | Finding |",
        "|---|---|---|",
    ]
    for evidence in iter_projector_source_evidence():
        sources = "<br>".join(
            f"`{repository}@{revision}` `{path}`"
            for repository, revision, path in evidence.sources
        )
        rows.append(f"| `{evidence.evidence_id}` | {sources} | {evidence.finding} |")
    return "\n".join(rows)


def _projector_availability_table() -> str:
    rows = [
        "| Candidate route | Immutable available sidecar | Bytes | SHA-256 |",
        "|---|---|---:|---|",
    ]
    for pin in MMPROJ_ARTIFACT_AVAILABILITY_PINS:
        rows.append(
            f"| `{pin.projector_type}` | `{pin.repository}@{pin.revision}`<br>"
            f"`{pin.filename}` | {pin.size:,} | `{pin.lfs_sha256}` |"
        )
    return "\n".join(rows)


def _draft_runtime_evidence_summary() -> str:
    evidence = json.loads(_DRAFT_RUNTIME_EVIDENCE_PATH.read_text(encoding="utf-8"))
    summaries = []
    for record in evidence["routes"]:
        result = record["direct_ort_result"]
        fidelity = record["source_fidelity"]
        fidelity_summary = (
            "source tensors exact"
            if fidelity.get("all_types_shapes_values_equal")
            else (
                f"source cosine={fidelity['cosine']:.6f}, "
                f"relative-L2={fidelity['relative_l2']:.6f}"
            )
        )
        summaries.append(
            f"`{record['architecture']}`: "
            f"{result['generated_token_count']} target-only-equal greedy tokens; "
            f"{result['accepted_tokens']}/{result['proposed_tokens']} accepted; "
            f"{result['multi_token_rounds']} multi-token rounds; "
            f"{result['rollback_events']} rollbacks; {fidelity_summary}"
        )
    return (
        "Pinned DFlash/EAGLE3 source, artifact, tokenizer, graph, and package hashes live in "
        "`testdata/evidence/gguf_draft_runtime_evidence.json`: "
        + "; ".join(summaries)
        + ". Both use separate target/draft caches; higher-level runtime=`runtime_unvalidated`."
    )


def render_document() -> str:
    """Render the complete concise API document from live registries and evidence."""
    blocks = render_blocks()
    recent_prs = "; ".join(
        f"#{record.number} ({record.state_at_audit}) — {record.dependency}"
        for record in RECENT_PR_DEPENDENCIES
    )
    tokenizer_scope_note = " ".join(
        (
            "Each row is independently artifact-scoped and proves ordered tokenizer semantics,",
            "source assets, embedding alignment, and the final materialized hash.",
            "Shared rows also require identical pinned llama.cpp dispatch.",
            "A matching complete immutable GGUF is automatically promoted to the pinned-source",
            "route during model and runtime package export; identifier-only inspection remains",
            "deferred because an identifier cannot prove artifact identity.",
            "This does not claim graph or runtime support.",
        )
    )
    tokenizer_refresh_note = " ".join(
        (
            "The MiniCPM, Gemma4, and final alias-group fixtures are reproducible through",
            "`scripts/generate_*tokenizer*.py`, which validates immutable bounded headers",
            "and official tokenizer hashes, builds tokenizer-only GGUFs and the pinned",
            "llama.cpp helper, then recomputes exact outputs and mismatch witnesses.",
            "Committed Gemma4 and alias-group inputs replay materialized identities",
            "network-free; the alias oracle never calls the production reconstruction.",
        )
    )
    draft_usage_note = " ".join(
        (
            "For target-coupled DFlash/EAGLE3, use `build_draft_pair_from_gguf`,",
            "`write_draft_pair_package`, and `DraftPairRunner`; the package carries",
            "independent caches, required target bridges, and an MTP-aligned",
            "`draft_runtime_status.json`, while `runtime_unvalidated` warns about",
            "higher-level runtimes without gating direct ORT. CLI:",
            "`mobius build-gguf draft.gguf --target-gguf target.gguf",
            "--target-config target-config --output output`.",
            "The committed real-pair evidence also uses a test-only direct ORT",
            "coordinator that reads remapping metadata from the raw immutable draft GGUF",
            "and does not import `DraftPairRunner` or its transition helpers. Per-round",
            "DFlash and EAGLE3 traces bind proposal/remap tokens, proposal-logit hashes,",
            "accepted prefixes, correction tokens, target replay, target/draft cache",
            "states, final counters, and four execution-mutating discriminators. Target",
            "replay starts from an empty cache, and final speculative rounds never process",
            "past the requested token count. Beam reorder is reported unsupported for the",
            "batch-size-one reference coordinator rather than inferred.",
        )
    )
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

`keep_quantized=True` requests quantized target storage where supported; it does
not guarantee source fidelity. An authoritative tokenizer blocker does not invalidate
a proven graph: the API returns the model, emits one structured warning, omits assets,
and persists the exact component disposition in `export_report.json`. Lossy qtype
conversion remains in `quantization_report.json`; use `keep_quantized=False` for float.

{draft_usage_note}

Packed MatMulNBits storage may use a native op or portable nibble unpack,
`DequantizeLinear`, and float `MatMul`; neither implies dense storage or a specific kernel.
Use `mmproj=` only for evidenced sidecars; CLI: `mobius build model.gguf -o output`.
Split shards validate siblings and ownership; Hub references reject partial downloads.
Standalone projector graphs persist their processor ABI, output width, and runtime warning.

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
    image_token_id=None,
    static_cache=False,
    max_seq_len=None,
    allow_dense_moe=None,
    reuse_gguf_weights=False,
    target_config=None,
    output_layer_indices=None,
)
```

The function returns a `ModelPackage`. Mobius-side graph/semantic inability, corruption,
identity, I/O, malformed tokenizer, and unexpected errors remain fail-closed. Authoritative
tokenizer blockers become partial exports with exact reasons and no unverified assets.
Exact tokenizer evidence is matched independently from runtime evidence, so a proven tokenizer
is exported even when downstream runtime execution remains unvalidated.
Downstream runtime, version, registry, or executor limitations preserve the accurate model
package and record distinct `export_status` and `runtime_validation_status` fields instead of
blocking export. Only exact artifact, graph, tokenizer, version, parity, and state evidence
covering the final directory bytes (including `export_report.json` and
`runtime_compatibility.json`) marks a package validated and end-to-end runnable. Existing
pre-report runtime evidence remains unvalidated until its final-package hashes are regenerated.

Use `build_mmproj_from_gguf("mmproj.gguf", projector_type=..., target_architecture=...)`
to export a registry-evidenced standalone `vision_encoder`, `audio_encoder`, or
`speaker_encoder`. It never invents decoder, media-mixing, or generated-audio roles and
warns while downstream runtime orchestration remains unvalidated.

## Runtime evidence

The first low-cost architecture batch promotes GPT-2, GPT-NeoX/Pythia, MPT, OLMo, StarCoder, and StarCoder2 using 334,238,976 bytes of GGUF payload and 346,825,051
download bytes including tokenizer assets. Every route is explicit-float only. The
network-free selection, budget, exclusions, and fail-closed candidate reasons are recorded in
`testdata/evidence/gguf_low_cost_runtime_batch.json`.

{_runtime_evidence_table()}

{_draft_runtime_evidence_summary()}

Runtime support above is independent from tokenizer materialization support below.
### Fail-closed runtime evidence

{_runtime_blocker_evidence_table()}

## Remaining route work

Every unresolved route is classified once from its authoritative registry. Exact reasons
remain machine-readable in `_route_census.py`; this table groups only shared next work.

{blocks["remaining_routes"]}

Recent PR dependencies: {recent_prs}.

## Tokenizer evidence

{_tokenizer_evidence_table()}

```python
from mobius.integrations.gguf import materialize_evidenced_gguf_tokenizer

materialize_evidenced_gguf_tokenizer("Qwen3.5-0.8B-Q4_0.gguf", "tokenizer")
```

{tokenizer_scope_note}

### Fail-closed tokenizer evidence

{_tokenizer_blocker_evidence_table()}

{tokenizer_refresh_note}

## Supported GGUF architectures

Reason codes are concise user-facing categories; detailed architecture audits remain in
`_arch_registry.py` and its tests.

<!-- BEGIN GGUF SUPPORT MATRIX (generated; see _arch_registry.py) -->

{blocks["architectures"]}

<!-- END GGUF SUPPORT MATRIX -->

## Stored quantization types

The generated machine-readable source for this table is
`testdata/evidence/gguf_quantization_capabilities.json`. It records parse and exact
dequantization support separately from conversion, names the implementation transform and
operator ABI for every tensor role, and treats dequantize/requantize as non-preserving.

<!-- BEGIN GGUF QUANTIZATION MATRIX (generated; see _quant_registry.py) -->

{blocks["qtypes"]}

<!-- END GGUF QUANTIZATION MATRIX -->

## Multimodal projector sidecars

{_projector_evidence_table()}

Pinned source proofs cover graph semantics that cannot be inferred from tensor names,
including conversion-time permutations, co-resident modality roles, and processor boundaries.

{_projector_source_evidence_table()}

These additional immutable files prove artifact availability only. Their routes remain
governed by the capability matrix until tensor mapping and component parity are established.

{_projector_availability_table()}

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
    if pins - {UPSTREAM_COMMIT, LLAMA_CPP_MMPROJ_SHA}:
        raise ValueError(f"Stale llama.cpp pins outside generated blocks: {sorted(pins)}")
    return render_document()


def check_document(path: Path = DOC_PATH) -> bool:
    """Return whether *path* exactly matches the generated registry content."""
    return path.read_text(encoding="utf-8") == update_document(path)
