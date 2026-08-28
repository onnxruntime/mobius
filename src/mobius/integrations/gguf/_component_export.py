# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Component-level GGUF export diagnostics and partial-package policy."""

from __future__ import annotations

__all__ = [
    "attach_runtime_unvalidated_report",
    "attach_tokenizer_export_report",
    "emit_runtime_unvalidated_warning",
    "resolve_tokenizer_export_verdict",
    "tokenizer_export_is_partial",
]

import dataclasses
import json
import logging
from pathlib import Path
from typing import Any, Literal

from mobius._export_report import ComponentExportDisposition, ComponentExportReport
from mobius.integrations.gguf._tokenizer import GGUFTokenizerVerdict, inspect_gguf_tokenizer

logger = logging.getLogger(__name__)

_TOKENIZER_IMPACT = (
    "The proven model graph was exported, but the package is not end-to-end runnable "
    "because tokenizer assets were omitted."
)
_TOKENIZER_REMEDIATION = (
    "Tokenizer semantics are unverified. Provide and validate a tokenizer against the "
    "source GGUF before end-to-end use."
)
_RUNTIME_OMITTED_IMPACT = (
    "The accurate model package was exported, but runtime-specific assets were omitted "
    "because execution with the requested runtime has not been validated."
)
_RUNTIME_EXPORTED_IMPACT = (
    "The runtime configuration was exported faithfully, but execution with the requested "
    "runtime has not been validated."
)
_RUNTIME_REMEDIATION = (
    "Validate the exported package with the requested runtime and version before production "
    "use; export success is not a runtime support claim."
)
_RUNTIME_NOT_REQUESTED_IMPACT = (
    "No runtime-specific configuration was requested by this model-only export."
)
_RUNTIME_NOT_REQUESTED_REMEDIATION = (
    "Request runtime packaging separately when an end-to-end runtime package is needed."
)


def resolve_tokenizer_export_verdict(
    gguf_model: Any,
    source_path: str | Path,
    *,
    verdict: GGUFTokenizerVerdict | None = None,
    artifact_identity: Any | None = None,
) -> GGUFTokenizerVerdict:
    """Apply exact artifact evidence to promote or block a tokenizer verdict."""
    if verdict is None:
        verdict = inspect_gguf_tokenizer(gguf_model.metadata, source=str(source_path))
    if verdict.materialized or verdict.metadata_sha256 is None:
        return verdict

    from mobius.integrations.gguf._tokenizer_evidence import (
        find_matching_tokenizer_evidence,
        matching_tokenizer_blocker_evidence,
    )

    blocker = matching_tokenizer_blocker_evidence(
        Path(source_path),
        gguf_model,
        metadata_sha256=verdict.metadata_sha256,
        artifact_identity=artifact_identity,
    )
    if blocker is None:
        evidence = find_matching_tokenizer_evidence(
            Path(source_path),
            gguf_model,
            metadata_sha256=verdict.metadata_sha256,
            artifact_identity=artifact_identity,
        )
        if evidence is None:
            return verdict
        return dataclasses.replace(
            verdict,
            route="pinned-source",
            reason=(
                "complete immutable GGUF identity matches independently validated tokenizer "
                f"evidence {evidence.evidence_id!r}"
            ),
            tokenizer_sha256=evidence.materialized_tokenizer_sha256,
            audit_status="validated-pinned-source",
            blocker_category=None,
            evidence_id=evidence.evidence_id,
        )
    return dataclasses.replace(
        verdict,
        reason=blocker.disposition,
        audit_status="deferred-pinned-artifact-mismatch",
        blocker_category="pinned-candidate-source-semantic-mismatch",
        evidence_id=blocker.evidence_id,
    )


def attach_tokenizer_export_report(
    pkg: Any,
    verdict: GGUFTokenizerVerdict,
    *,
    model_route: str,
) -> None:
    """Record and warn once when a proven graph must omit a tokenizer component."""
    if verdict.blocker_category is None:
        return
    existing = getattr(pkg, "export_report", None)
    if (
        isinstance(existing, ComponentExportReport)
        and existing.component("tokenizer") is not None
    ):
        return

    support: Literal["blocked", "deferred"] = (
        "blocked"
        if verdict.audit_status == "deferred-pinned-artifact-mismatch"
        else "deferred"
    )
    tokenizer = ComponentExportDisposition(
        name="tokenizer",
        route=verdict.route_identifier,
        requested=True,
        discovered=True,
        support=support,
        output="omitted",
        runtime_validation_status="unvalidated",
        blocker_category=verdict.blocker_category,
        reason=verdict.reason,
        evidence_id=verdict.evidence_id,
        impact=_TOKENIZER_IMPACT,
        remediation=_TOKENIZER_REMEDIATION,
    )
    model = ComponentExportDisposition(
        name="model",
        route=model_route,
        requested=True,
        discovered=True,
        support="supported",
        output="exported",
    )
    runtime = ComponentExportDisposition(
        name="runtime",
        route="not-requested",
        requested=False,
        discovered=False,
        support="deferred",
        output="omitted",
        blocker_category="not-requested",
        reason="Runtime packaging was not requested during model graph construction.",
        impact=_RUNTIME_NOT_REQUESTED_IMPACT,
        remediation=_RUNTIME_NOT_REQUESTED_REMEDIATION,
    )
    pkg.export_report = ComponentExportReport.create(
        (model, runtime, tokenizer),
        end_to_end_runnable=False,
    )
    warning = {
        "blocker_category": tokenizer.blocker_category,
        "component": tokenizer.name,
        "evidence_id": tokenizer.evidence_id,
        "export_status": tokenizer.output,
        "impact": tokenizer.impact,
        "reason": tokenizer.reason,
        "remediation": tokenizer.remediation,
        "route": tokenizer.route,
        "runtime_validation_status": tokenizer.runtime_validation_status,
        "support_status": tokenizer.support,
    }
    logger.warning(
        "GGUF PARTIAL EXPORT WARNING: %s",
        json.dumps(warning, ensure_ascii=True, separators=(",", ":"), sort_keys=True),
        extra={
            "mobius_warning_code": "gguf_component_partial_export",
            "mobius_component": tokenizer.name,
            "mobius_component_route": tokenizer.route,
            "mobius_blocker_category": tokenizer.blocker_category,
            "mobius_evidence_id": tokenizer.evidence_id,
        },
    )


def tokenizer_export_is_partial(pkg: Any) -> bool:
    """Whether the package carries an omitted/deferred tokenizer disposition."""
    report = getattr(pkg, "export_report", None)
    if not isinstance(report, ComponentExportReport):
        return False
    tokenizer = report.component("tokenizer")
    return tokenizer is not None and (
        tokenizer.support != "supported" or tokenizer.output != "exported"
    )


def attach_runtime_unvalidated_report(
    pkg: Any,
    runtime: str,
    *,
    blocker_category: str | None,
    reason: str | None,
    evidence_id: str | None = None,
    support_status: Literal["supported", "deferred", "blocked"] = "deferred",
    runtime_output: Literal["exported", "omitted"] = "exported",
    runtime_validation_status: Literal["validated", "unvalidated"] = "unvalidated",
    tokenizer_exported: bool = False,
    emit_warning: bool = True,
) -> bool:
    """Record the exported runtime component and its independent validation status."""
    existing = getattr(pkg, "export_report", None)
    if isinstance(existing, ComponentExportReport):
        runtime_component = existing.component("runtime")
        if (
            runtime_component is not None
            and runtime_component.route == runtime
            and runtime_component.support == support_status
            and runtime_component.runtime_validation_status == runtime_validation_status
            and runtime_component.output == runtime_output
            and runtime_component.blocker_category == blocker_category
            and runtime_component.reason == reason
            and runtime_component.evidence_id == evidence_id
        ):
            return False

    model = (
        existing.component("model") if isinstance(existing, ComponentExportReport) else None
    )
    if model is None:
        model = ComponentExportDisposition(
            name="model",
            route=getattr(pkg, "gguf_architecture", "gguf-model"),
            requested=True,
            discovered=True,
            support="supported",
            output="exported",
        )
    tokenizer = (
        existing.component("tokenizer")
        if isinstance(existing, ComponentExportReport)
        else None
    )
    tokenizer_verdict = getattr(pkg, "gguf_tokenizer_verdict", None)
    if tokenizer is None:
        if tokenizer_verdict is None:
            raise ValueError(
                "Runtime-unvalidated packaging requires the tokenizer disposition captured "
                "during graph construction."
            )
        tokenizer_route = getattr(tokenizer_verdict, "route_identifier", None)
        if not isinstance(tokenizer_route, str) or not tokenizer_route:
            tokenizer_route = "embedded" if tokenizer_exported else "unresolved"
        if tokenizer_exported or tokenizer_verdict.materialized:
            tokenizer = ComponentExportDisposition(
                name="tokenizer",
                route=tokenizer_route,
                requested=True,
                discovered=True,
                support="supported",
                output="exported",
            )
        else:
            tokenizer = ComponentExportDisposition(
                name="tokenizer",
                route=tokenizer_route,
                requested=True,
                discovered=True,
                support="deferred",
                output="omitted",
                runtime_validation_status="unvalidated",
                blocker_category=(
                    getattr(tokenizer_verdict, "blocker_category", None)
                    or "tokenizer-materialization-unvalidated"
                ),
                reason=tokenizer_verdict.reason,
                evidence_id=getattr(tokenizer_verdict, "evidence_id", None),
                impact=_TOKENIZER_IMPACT,
                remediation=_TOKENIZER_REMEDIATION,
            )
    runtime_component = ComponentExportDisposition(
        name="runtime",
        route=runtime,
        requested=True,
        discovered=True,
        support=support_status,
        output=runtime_output,
        runtime_validation_status=runtime_validation_status,
        blocker_category=blocker_category,
        reason=reason,
        evidence_id=evidence_id,
        impact=(
            None
            if runtime_validation_status == "validated"
            else (
                _RUNTIME_EXPORTED_IMPACT
                if runtime_output == "exported"
                else _RUNTIME_OMITTED_IMPACT
            )
        ),
        remediation=(
            None if runtime_validation_status == "validated" else _RUNTIME_REMEDIATION
        ),
    )
    components = (model, runtime_component, tokenizer)
    if isinstance(existing, ComponentExportReport):
        report = existing
        for component in components:
            report = report.with_component(component, end_to_end_runnable=False)
        pkg.export_report = report
    else:
        pkg.export_report = ComponentExportReport.create(
            components,
            end_to_end_runnable=False,
        )
    end_to_end_runnable = (
        runtime_validation_status == "validated"
        and runtime_output == "exported"
        and tokenizer.output == "exported"
    )
    if end_to_end_runnable:
        pkg.export_report = ComponentExportReport.create(
            pkg.export_report.components,
            end_to_end_runnable=True,
        )
    if emit_warning and runtime_validation_status == "unvalidated":
        emit_runtime_unvalidated_warning(pkg, runtime)
    return True


def emit_runtime_unvalidated_warning(pkg: Any, runtime: str) -> None:
    """Emit the structured warning for a persisted runtime-unvalidated report."""
    report = getattr(pkg, "export_report", None)
    if not isinstance(report, ComponentExportReport):
        raise TypeError("Runtime validation warning requires a component export report.")
    runtime_component = report.component("runtime")
    if (
        runtime_component is None
        or runtime_component.runtime_validation_status != "unvalidated"
    ):
        raise ValueError(
            "Runtime validation warning requires an unvalidated runtime component."
        )
    warning = {
        "blocker_category": runtime_component.blocker_category,
        "component": "runtime",
        "evidence_id": runtime_component.evidence_id,
        "export_status": runtime_component.output,
        "impact": runtime_component.impact,
        "reason": runtime_component.reason,
        "remediation": runtime_component.remediation,
        "route": runtime,
        "runtime_validation_status": "unvalidated",
        "support_status": runtime_component.support,
    }
    logger.warning(
        "GGUF RUNTIME VALIDATION WARNING: %s",
        json.dumps(warning, ensure_ascii=True, separators=(",", ":"), sort_keys=True),
        extra={
            "mobius_warning_code": "gguf_runtime_unvalidated",
            "mobius_component": "runtime",
            "mobius_component_route": runtime,
            "mobius_blocker_category": runtime_component.blocker_category,
            "mobius_evidence_id": runtime_component.evidence_id,
        },
    )
