# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Typed, persistent fidelity reporting for GGUF quantization conversion."""

from __future__ import annotations

__all__ = [
    "GGUFQuantizationReport",
    "QuantizationDisposition",
    "QuantizationDispositionStat",
    "QuantizationTensorRecord",
    "QuantizationTypeStat",
    "disposition_for_import_route",
]

import dataclasses
import enum
import json
import os
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path

from mobius.integrations.gguf._spec import QuantImportRoute, RepackExactness


class QuantizationDisposition(enum.Enum):
    """How one mapped GGUF tensor is represented in the resulting package."""

    NATIVE_BYTES = "byte-preserved native"
    LOSSLESS_REPACK = "numerically lossless repack"
    LOSSY_REQUANTIZE = "lossy dequantize+requantize"
    DEQUANTIZED_FLOAT = "dequantized-to-float"
    SOURCE_FLOAT = "source float"
    REJECTED = "rejected"


def disposition_for_import_route(
    route: QuantImportRoute,
    exactness: RepackExactness | None,
) -> QuantizationDisposition:
    """Map the authoritative qtype route onto its report disposition."""
    if route is QuantImportRoute.NATIVE_BYTES:
        return QuantizationDisposition.NATIVE_BYTES
    if route is QuantImportRoute.AFFINE_REPACK:
        return (
            QuantizationDisposition.LOSSLESS_REPACK
            if exactness is RepackExactness.EXACT
            else QuantizationDisposition.LOSSY_REQUANTIZE
        )
    if route is QuantImportRoute.DEQUANTIZE_REQUANTIZE:
        return QuantizationDisposition.LOSSY_REQUANTIZE
    if route is QuantImportRoute.DEQUANTIZE_FLOAT:
        return QuantizationDisposition.DEQUANTIZED_FLOAT
    return QuantizationDisposition.REJECTED


@dataclasses.dataclass(frozen=True, slots=True)
class QuantizationTypeStat:
    """Tensor count and source payload bytes for one GGUF qtype."""

    qtype: str
    tensor_count: int
    source_bytes: int


@dataclasses.dataclass(frozen=True, slots=True)
class QuantizationDispositionStat:
    """Aggregate mapped-tensor statistics for one conversion disposition."""

    disposition: QuantizationDisposition
    tensor_count: int
    source_bytes: int
    qtypes: tuple[str, ...]


@dataclasses.dataclass(frozen=True, slots=True)
class QuantizationTensorRecord:
    """Fidelity decision for one mapped GGUF tensor."""

    name: str
    qtype: str
    source_bytes: int
    disposition: QuantizationDisposition
    target_storage: str
    reason: str


@dataclasses.dataclass(frozen=True, slots=True)
class GGUFQuantizationReport:
    """Machine-readable source/target quantization fidelity statement."""

    source_qtype_census: tuple[QuantizationTypeStat, ...]
    target_storage_format: str
    dispositions: tuple[QuantizationDispositionStat, ...]
    source_fidelity: bool
    storage_quantized: bool
    compute_mode: str
    compute_capability: str
    explicit_float_tensors: tuple[QuantizationTensorRecord, ...]
    tensor_records: tuple[QuantizationTensorRecord, ...]
    converted_from: str | None = None
    schema_version: int = 1

    @classmethod
    def create(
        cls,
        *,
        source_qtypes: Iterable[tuple[str, int]],
        tensor_records: Iterable[QuantizationTensorRecord],
        target_storage_format: str,
        compute_mode: str,
        compute_capability: str,
    ) -> GGUFQuantizationReport:
        """Create a deterministic report from header census and mapped decisions."""
        census_counts: Counter[str] = Counter()
        census_bytes: Counter[str] = Counter()
        for qtype, source_bytes in source_qtypes:
            if source_bytes < 0:
                raise ValueError(
                    f"GGUF tensor source bytes must be non-negative, got {source_bytes}"
                )
            census_counts[qtype] += 1
            census_bytes[qtype] += source_bytes
        census = tuple(
            QuantizationTypeStat(name, census_counts[name], census_bytes[name])
            for name in sorted(census_counts)
        )

        records = tuple(sorted(tensor_records, key=lambda record: record.name))
        grouped: dict[QuantizationDisposition, list[QuantizationTensorRecord]] = defaultdict(
            list
        )
        for record in records:
            grouped[record.disposition].append(record)
        dispositions = tuple(
            QuantizationDispositionStat(
                disposition,
                len(grouped[disposition]),
                sum(record.source_bytes for record in grouped[disposition]),
                tuple(sorted({record.qtype for record in grouped[disposition]})),
            )
            for disposition in QuantizationDisposition
            if grouped[disposition]
        )
        quantized_dispositions = {
            QuantizationDisposition.NATIVE_BYTES,
            QuantizationDisposition.LOSSLESS_REPACK,
            QuantizationDisposition.LOSSY_REQUANTIZE,
        }
        fidelity_breaks = {
            QuantizationDisposition.LOSSY_REQUANTIZE,
            QuantizationDisposition.DEQUANTIZED_FLOAT,
            QuantizationDisposition.REJECTED,
        }
        census_names = {stat.qtype for stat in census}
        converted_from = (
            "Q4_K_M-like mixed GGUF"
            if "Q4_K" in census_names and census_names.intersection({"Q5_K", "Q6_K"})
            else None
        )
        return cls(
            source_qtype_census=census,
            target_storage_format=target_storage_format,
            dispositions=dispositions,
            source_fidelity=not any(
                record.disposition in fidelity_breaks for record in records
            ),
            storage_quantized=any(
                record.disposition in quantized_dispositions for record in records
            ),
            compute_mode=compute_mode,
            compute_capability=compute_capability,
            explicit_float_tensors=tuple(
                record
                for record in records
                if record.disposition
                in {
                    QuantizationDisposition.DEQUANTIZED_FLOAT,
                    QuantizationDisposition.SOURCE_FLOAT,
                }
            ),
            tensor_records=records,
            converted_from=converted_from,
        )

    def warning_message(self) -> str | None:
        """Return the single deterministic warning for lossy quantized conversion."""
        lossy = [
            record
            for record in self.tensor_records
            if record.disposition is QuantizationDisposition.LOSSY_REQUANTIZE
        ]
        if not lossy:
            return None

        def summarize(records: Iterable[QuantizationTensorRecord]) -> str:
            counts: Counter[str] = Counter()
            source_bytes: Counter[str] = Counter()
            for record in records:
                counts[record.qtype] += 1
                source_bytes[record.qtype] += record.source_bytes
            return ", ".join(
                f"{qtype}: {counts[qtype]} tensor(s) / {source_bytes[qtype]} source bytes"
                for qtype in sorted(counts)
            )

        lossless = [
            record
            for record in self.tensor_records
            if record.disposition
            in {
                QuantizationDisposition.NATIVE_BYTES,
                QuantizationDisposition.LOSSLESS_REPACK,
            }
        ]
        lossless_summary = summarize(lossless) if lossless else "none"
        return (
            "GGUF QUANTIZATION FIDELITY WARNING: output target is "
            f"{self.target_storage_format}; lossy requantization will change represented "
            f"source values ({summarize(lossy)}). Losslessly preserved/repacked qtypes "
            f"in this artifact: {lossless_summary}. The output is quantized target storage "
            "but is NOT source-faithful and must not be labeled as the original GGUF preset."
        )

    @classmethod
    def combine(
        cls,
        *reports: GGUFQuantizationReport,
    ) -> GGUFQuantizationReport:
        """Combine component reports for one GGUF artifact into a root report."""
        if not reports:
            raise ValueError("At least one GGUF quantization report is required")
        census = reports[0].source_qtype_census
        if any(report.source_qtype_census != census for report in reports[1:]):
            raise ValueError("Cannot combine reports from different GGUF qtype censuses")
        records_by_name: dict[str, QuantizationTensorRecord] = {}
        for report in reports:
            for record in report.tensor_records:
                previous = records_by_name.setdefault(record.name, record)
                if previous != record:
                    raise ValueError(
                        f"Conflicting GGUF quantization dispositions for {record.name!r}"
                    )
        source_qtypes = [
            (stat.qtype, stat.source_bytes if index == 0 else 0)
            for stat in census
            for index in range(stat.tensor_count)
        ]
        target_formats = {
            target
            for report in reports
            for target in report.target_storage_format.split(" + ")
        }
        compute_modes = {report.compute_mode for report in reports}
        compute_capabilities = {report.compute_capability for report in reports}
        return cls.create(
            source_qtypes=source_qtypes,
            tensor_records=records_by_name.values(),
            target_storage_format=" + ".join(sorted(target_formats)),
            compute_mode=" + ".join(sorted(compute_modes)),
            compute_capability=" ".join(sorted(compute_capabilities)),
        )

    def to_dict(self) -> dict[str, object]:
        """Return the stable JSON-compatible representation."""
        return {
            "schema_version": self.schema_version,
            "converted_from": self.converted_from,
            "source_qtype_census": [
                dataclasses.asdict(stat) for stat in self.source_qtype_census
            ],
            "target_storage_format": self.target_storage_format,
            "dispositions": [
                {
                    **dataclasses.asdict(stat),
                    "disposition": stat.disposition.value,
                }
                for stat in self.dispositions
            ],
            "source_fidelity": self.source_fidelity,
            "storage_quantized": self.storage_quantized,
            "compute_mode": self.compute_mode,
            "compute_capability": self.compute_capability,
            "explicit_float_tensors": [
                self._record_to_dict(record) for record in self.explicit_float_tensors
            ],
            "tensor_records": [self._record_to_dict(record) for record in self.tensor_records],
        }

    @staticmethod
    def _record_to_dict(record: QuantizationTensorRecord) -> dict[str, object]:
        payload = dataclasses.asdict(record)
        payload["disposition"] = record.disposition.value
        return payload

    def write_json(self, path: str | Path) -> None:
        """Persist the stable report artifact."""
        payload = json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"
        flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags, 0o666)
        with os.fdopen(descriptor, "w", encoding="utf-8") as file:
            file.write(payload)

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> GGUFQuantizationReport:
        """Validate and restore a report from its JSON representation."""
        schema_version = payload.get("schema_version")
        if schema_version != 1:
            raise ValueError(
                f"Unsupported GGUF quantization report schema: {schema_version!r}"
            )

        def required_string(key: str) -> str:
            value = payload.get(key)
            if not isinstance(value, str):
                raise TypeError(f"GGUF quantization report {key!r} must be a string")
            return value

        def required_bool(key: str) -> bool:
            value = payload.get(key)
            if not isinstance(value, bool):
                raise TypeError(f"GGUF quantization report {key!r} must be a boolean")
            return value

        def nonnegative_int(value: object, *, field: str) -> int:
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"GGUF quantization report {field!r} must be an integer")
            if value < 0:
                raise ValueError(
                    f"GGUF quantization report {field!r} must be a non-negative integer"
                )
            return value

        def mapping_string(record: Mapping[str, object], key: str, *, field: str) -> str:
            value = record.get(key)
            if not isinstance(value, str):
                raise TypeError(f"GGUF quantization report {field!r} must be a string")
            return value

        def records(key: str) -> tuple[QuantizationTensorRecord, ...]:
            raw_records = payload.get(key)
            if not isinstance(raw_records, list):
                raise TypeError(f"GGUF quantization report {key!r} must be a list")
            if not all(isinstance(record, Mapping) for record in raw_records):
                raise TypeError(f"GGUF quantization report {key!r} entries must be objects")
            return tuple(
                QuantizationTensorRecord(
                    name=mapping_string(record, "name", field=f"{key}.name"),
                    qtype=mapping_string(record, "qtype", field=f"{key}.qtype"),
                    source_bytes=nonnegative_int(
                        record.get("source_bytes"), field=f"{key}.source_bytes"
                    ),
                    disposition=QuantizationDisposition(
                        mapping_string(
                            record,
                            "disposition",
                            field=f"{key}.disposition",
                        )
                    ),
                    target_storage=mapping_string(
                        record,
                        "target_storage",
                        field=f"{key}.target_storage",
                    ),
                    reason=mapping_string(record, "reason", field=f"{key}.reason"),
                )
                for record in raw_records
            )

        raw_census = payload.get("source_qtype_census")
        raw_dispositions = payload.get("dispositions")
        if not isinstance(raw_census, list) or not isinstance(raw_dispositions, list):
            raise TypeError("GGUF quantization report census/dispositions must be lists")
        if not all(isinstance(stat, Mapping) for stat in raw_census):
            raise TypeError("GGUF quantization report census entries must be objects")
        if not all(isinstance(stat, Mapping) for stat in raw_dispositions):
            raise TypeError("GGUF quantization report disposition entries must be objects")
        census = tuple(
            QuantizationTypeStat(
                qtype=mapping_string(
                    stat,
                    "qtype",
                    field="source_qtype_census.qtype",
                ),
                tensor_count=nonnegative_int(
                    stat.get("tensor_count"),
                    field="source_qtype_census.tensor_count",
                ),
                source_bytes=nonnegative_int(
                    stat.get("source_bytes"),
                    field="source_qtype_census.source_bytes",
                ),
            )
            for stat in raw_census
        )
        dispositions: list[QuantizationDispositionStat] = []
        for stat in raw_dispositions:
            raw_qtypes = stat.get("qtypes")
            if not isinstance(raw_qtypes, list) or not all(
                isinstance(item, str) for item in raw_qtypes
            ):
                raise TypeError(
                    "GGUF quantization report 'dispositions.qtypes' must be a list of strings"
                )
            dispositions.append(
                QuantizationDispositionStat(
                    disposition=QuantizationDisposition(
                        mapping_string(
                            stat,
                            "disposition",
                            field="dispositions.disposition",
                        )
                    ),
                    tensor_count=nonnegative_int(
                        stat.get("tensor_count"),
                        field="dispositions.tensor_count",
                    ),
                    source_bytes=nonnegative_int(
                        stat.get("source_bytes"),
                        field="dispositions.source_bytes",
                    ),
                    qtypes=tuple(raw_qtypes),
                )
            )
        converted_from = payload.get("converted_from")
        if converted_from is not None and not isinstance(converted_from, str):
            raise TypeError(
                "GGUF quantization report 'converted_from' must be a string or null"
            )
        report = cls(
            source_qtype_census=census,
            target_storage_format=required_string("target_storage_format"),
            dispositions=tuple(dispositions),
            source_fidelity=required_bool("source_fidelity"),
            storage_quantized=required_bool("storage_quantized"),
            compute_mode=required_string("compute_mode"),
            compute_capability=required_string("compute_capability"),
            explicit_float_tensors=records("explicit_float_tensors"),
            tensor_records=records("tensor_records"),
            converted_from=converted_from,
            schema_version=1,
        )
        source_qtypes = [
            (stat.qtype, stat.source_bytes if index == 0 else 0)
            for stat in census
            for index in range(stat.tensor_count)
        ]
        canonical = cls.create(
            source_qtypes=source_qtypes,
            tensor_records=report.tensor_records,
            target_storage_format=report.target_storage_format,
            compute_mode=report.compute_mode,
            compute_capability=report.compute_capability,
        )
        if report != canonical:
            raise ValueError(
                "GGUF quantization report summaries are inconsistent with tensor records"
            )
        return report

    @classmethod
    def read_json(cls, path: str | Path) -> GGUFQuantizationReport:
        """Load and validate a persisted report artifact."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise TypeError("GGUF quantization report root must be an object")
        return cls.from_dict(payload)
