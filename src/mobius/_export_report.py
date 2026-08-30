# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Machine-readable component dispositions for partial model-package exports."""

from __future__ import annotations

__all__ = ["ComponentExportDisposition", "ComponentExportReport"]

import dataclasses
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

ComponentSupport = Literal["supported", "deferred", "blocked"]
ComponentOutput = Literal["exported", "omitted"]
ExportStatus = Literal["complete", "partial"]
RuntimeValidationStatus = Literal["validated", "unvalidated", "not-applicable"]

_FORMAT = "mobius.component-export-report.v1"


@dataclasses.dataclass(frozen=True, slots=True)
class ComponentExportDisposition:
    """Support and output status for one independently exportable component."""

    name: str
    route: str
    requested: bool
    discovered: bool
    support: ComponentSupport
    output: ComponentOutput
    runtime_validation_status: RuntimeValidationStatus = "not-applicable"
    blocker_category: str | None = None
    reason: str | None = None
    evidence_id: str | None = None
    impact: str | None = None
    remediation: str | None = None

    def __post_init__(self) -> None:
        if not self.name or not self.route:
            raise ValueError(
                "Component export dispositions require non-empty names and routes."
            )
        if self.support not in {"supported", "deferred", "blocked"}:
            raise ValueError(f"Invalid component support status: {self.support!r}.")
        if self.output not in {"exported", "omitted"}:
            raise ValueError(f"Invalid component export status: {self.output!r}.")
        if self.runtime_validation_status not in {
            "validated",
            "unvalidated",
            "not-applicable",
        }:
            raise ValueError(
                "Invalid component runtime validation status: "
                f"{self.runtime_validation_status!r}."
            )
        incomplete = (
            self.support != "supported"
            or self.output != "exported"
            or self.runtime_validation_status == "unvalidated"
        )
        diagnostics = (
            self.blocker_category,
            self.reason,
            self.impact,
            self.remediation,
        )
        if incomplete and any(not value for value in diagnostics):
            raise ValueError(
                "Deferred, blocked, or omitted components require category, reason, "
                "impact, and remediation diagnostics."
            )
        if not incomplete and any(value is not None for value in diagnostics):
            raise ValueError("A fully exported component cannot carry blocker diagnostics.")
        if self.evidence_id is not None and not self.evidence_id:
            raise ValueError("Component evidence IDs must be non-empty when present.")

    @property
    def supported(self) -> bool:
        """Whether this component has a proven support route."""
        return self.support == "supported"

    @property
    def exported(self) -> bool:
        """Whether this component was included in the package output."""
        return self.output == "exported"

    def to_dict(self) -> dict[str, object]:
        """Return a stable JSON-compatible representation."""
        return {
            "blocker_category": self.blocker_category,
            "discovered": self.discovered,
            "evidence_id": self.evidence_id,
            "export_status": self.output,
            "exported": self.exported,
            "impact": self.impact,
            "reason": self.reason,
            "remediation": self.remediation,
            "requested": self.requested,
            "route": self.route,
            "runtime_validation_status": self.runtime_validation_status,
            "support_status": self.support,
            "supported": self.supported,
        }

    @classmethod
    def from_dict(cls, name: str, payload: Mapping[str, object]) -> ComponentExportDisposition:
        """Parse and validate one component disposition."""

        def require_string(key: str) -> str:
            value = payload.get(key)
            if not isinstance(value, str) or not value:
                raise ValueError(f"Component {name!r} requires a non-empty {key!r} field.")
            return value

        def optional_string(key: str) -> str | None:
            value = payload.get(key)
            if value is not None and (not isinstance(value, str) or not value):
                raise ValueError(
                    f"Component {name!r} field {key!r} must be null or a non-empty string."
                )
            return value

        def require_bool(key: str) -> bool:
            value = payload.get(key)
            if type(value) is not bool:
                raise TypeError(f"Component {name!r} field {key!r} must be a boolean.")
            return value

        requested = require_bool("requested")
        discovered = require_bool("discovered")
        supported = payload.get("supported")
        exported = payload.get("exported")
        if type(supported) is not bool or type(exported) is not bool:
            raise TypeError(f"Component {name!r} supported/exported fields must be booleans.")
        raw_support = require_string("support_status")
        if raw_support == "supported":
            support: ComponentSupport = "supported"
        elif raw_support == "deferred":
            support = "deferred"
        elif raw_support == "blocked":
            support = "blocked"
        else:
            raise ValueError(f"Component {name!r} has invalid support status {raw_support!r}.")
        raw_output = require_string("export_status")
        if raw_output == "exported":
            output: ComponentOutput = "exported"
        elif raw_output == "omitted":
            output = "omitted"
        else:
            raise ValueError(f"Component {name!r} has invalid export status {raw_output!r}.")
        raw_runtime_validation = require_string("runtime_validation_status")
        if raw_runtime_validation == "validated":
            runtime_validation_status: RuntimeValidationStatus = "validated"
        elif raw_runtime_validation == "unvalidated":
            runtime_validation_status = "unvalidated"
        elif raw_runtime_validation == "not-applicable":
            runtime_validation_status = "not-applicable"
        else:
            raise ValueError(
                f"Component {name!r} has invalid runtime validation status "
                f"{raw_runtime_validation!r}."
            )
        if supported != (support == "supported") or exported != (output == "exported"):
            raise ValueError(f"Component {name!r} contains contradictory status booleans.")
        return cls(
            name=name,
            route=require_string("route"),
            requested=requested,
            discovered=discovered,
            support=support,
            output=output,
            runtime_validation_status=runtime_validation_status,
            blocker_category=optional_string("blocker_category"),
            reason=optional_string("reason"),
            evidence_id=optional_string("evidence_id"),
            impact=optional_string("impact"),
            remediation=optional_string("remediation"),
        )


@dataclasses.dataclass(frozen=True, slots=True)
class ComponentExportReport:
    """Deterministic package-level summary of independently handled components."""

    export_status: ExportStatus
    runtime_validation_status: RuntimeValidationStatus
    end_to_end_runnable: bool
    components: tuple[ComponentExportDisposition, ...]

    def __post_init__(self) -> None:
        if not self.components:
            raise ValueError("Component export reports must contain at least one component.")
        names = tuple(component.name for component in self.components)
        if names != tuple(sorted(names)) or len(set(names)) != len(names):
            raise ValueError("Component export report names must be unique and sorted.")
        expected_status = (
            "partial"
            if any(
                component.support != "supported" or component.output != "exported"
                for component in self.components
                if component.requested or component.discovered
            )
            else "complete"
        )
        if self.export_status != expected_status:
            raise ValueError(
                f"Component export report status must be {expected_status!r}, "
                f"got {self.export_status!r}."
            )
        component_runtime_statuses = {
            component.runtime_validation_status for component in self.components
        }
        expected_runtime_status: RuntimeValidationStatus
        if "unvalidated" in component_runtime_statuses:
            expected_runtime_status = "unvalidated"
        elif "validated" in component_runtime_statuses:
            expected_runtime_status = "validated"
        else:
            expected_runtime_status = "not-applicable"
        if self.runtime_validation_status != expected_runtime_status:
            raise ValueError(
                "Component export report runtime validation status must be "
                f"{expected_runtime_status!r}, got {self.runtime_validation_status!r}."
            )
        if self.end_to_end_runnable and self.export_status != "complete":
            raise ValueError("A partial component export cannot be end-to-end runnable.")
        if self.end_to_end_runnable and self.runtime_validation_status != "validated":
            raise ValueError("An end-to-end runnable export must be runtime validated.")

    @property
    def status(self) -> ExportStatus:
        """Backward-compatible alias for the explicit export status."""
        return self.export_status

    @classmethod
    def create(
        cls,
        components: tuple[ComponentExportDisposition, ...],
        *,
        end_to_end_runnable: bool,
    ) -> ComponentExportReport:
        """Create a report with a status derived from its component dispositions."""
        ordered = tuple(sorted(components, key=lambda component: component.name))
        status: ExportStatus = (
            "partial"
            if any(
                component.support != "supported" or component.output != "exported"
                for component in ordered
                if component.requested or component.discovered
            )
            else "complete"
        )
        runtime_statuses = {component.runtime_validation_status for component in ordered}
        runtime_validation_status: RuntimeValidationStatus
        if "unvalidated" in runtime_statuses:
            runtime_validation_status = "unvalidated"
        elif "validated" in runtime_statuses:
            runtime_validation_status = "validated"
        else:
            runtime_validation_status = "not-applicable"
        return cls(
            export_status=status,
            runtime_validation_status=runtime_validation_status,
            end_to_end_runnable=end_to_end_runnable,
            components=ordered,
        )

    def component(self, name: str) -> ComponentExportDisposition | None:
        """Return one named component disposition, if present."""
        return next(
            (component for component in self.components if component.name == name),
            None,
        )

    def with_component(
        self, component: ComponentExportDisposition, *, end_to_end_runnable: bool | None = None
    ) -> ComponentExportReport:
        """Return a report with one component inserted or replaced."""
        components_by_name = {existing.name: existing for existing in self.components}
        components_by_name[component.name] = component
        return self.create(
            tuple(components_by_name.values()),
            end_to_end_runnable=(
                self.end_to_end_runnable
                if end_to_end_runnable is None
                else end_to_end_runnable
            ),
        )

    def to_dict(self) -> dict[str, object]:
        """Return a stable JSON-compatible representation."""
        return {
            "components": {
                component.name: component.to_dict() for component in self.components
            },
            "end_to_end_runnable": self.end_to_end_runnable,
            "export_status": self.export_status,
            "format": _FORMAT,
            "runtime_validation_status": self.runtime_validation_status,
        }

    def to_json(self) -> str:
        """Serialize with deterministic key ordering and a trailing newline."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    def to_bytes(self) -> bytes:
        """Serialize to deterministic UTF-8 bytes with LF line endings."""
        return self.to_json().encode("utf-8")

    def write_json(self, path: str | Path) -> None:
        """Write the deterministic report without platform newline translation."""
        Path(path).write_bytes(self.to_bytes())

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> ComponentExportReport:
        """Parse and validate a component export report."""
        if payload.get("format") != _FORMAT:
            raise ValueError("Invalid component export report format.")
        status = payload.get("export_status")
        if status not in {"complete", "partial"}:
            raise ValueError("Invalid component export report status.")
        raw_runtime_validation = payload.get("runtime_validation_status")
        if raw_runtime_validation == "validated":
            runtime_validation_status: RuntimeValidationStatus = "validated"
        elif raw_runtime_validation == "unvalidated":
            runtime_validation_status = "unvalidated"
        elif raw_runtime_validation == "not-applicable":
            runtime_validation_status = "not-applicable"
        else:
            raise ValueError("Invalid component export report runtime validation status.")
        runnable = payload.get("end_to_end_runnable")
        if type(runnable) is not bool:
            raise ValueError("Component export report runnable status must be a boolean.")
        raw_components = payload.get("components")
        if not isinstance(raw_components, dict) or not raw_components:
            raise ValueError("Component export report components must be a non-empty object.")
        components: list[ComponentExportDisposition] = []
        for name, component_payload in sorted(raw_components.items()):
            if not isinstance(name, str) or not name:
                raise ValueError("Component export report names must be non-empty strings.")
            if not isinstance(component_payload, dict):
                raise TypeError(f"Component export report entry {name!r} must be an object.")
            components.append(ComponentExportDisposition.from_dict(name, component_payload))
        return cls(
            export_status=status,
            runtime_validation_status=runtime_validation_status,
            end_to_end_runnable=runnable,
            components=tuple(components),
        )

    @classmethod
    def read_json(cls, path: str | Path) -> ComponentExportReport:
        """Read and validate a component export report."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError("Component export report must contain a JSON object.")
        return cls.from_dict(payload)
