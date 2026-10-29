# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Progressive Linux DSO mapping evidence, using only the standard library."""

from __future__ import annotations

import hashlib
from pathlib import Path


class LibraryIdentityError(RuntimeError):
    """Fail closed while preserving actual observed mappings and match decisions."""

    def __init__(self, message, evidence):
        super().__init__(message)
        self.evidence = evidence


def _digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def _resolve(path):
    return Path(path).resolve()


def observe_loaded_libraries(
    cuda_files, ort_files, *, maps_text=None, maps_path=None, cuda_source="runtime.libraries"
):
    """Collect all relevant map rows before raising; never approve an unknown path.

    Expected hashes are policy, not observations. Observed hashes are read from
    the actual resolved mapped files. Same-hash unapproved paths remain failures.
    """
    expected = {"cuda": {}, "ort": {}}
    evidence = {
        "status": "FAILED",
        "expected_sources": {"cuda": [], "ort": []},
        "observed_mappings": [],
        "violations": [],
    }
    for kind, descriptors in (("cuda", cuda_files), ("ort", ort_files)):
        for item in descriptors:
            try:
                path = str(_resolve(item["path"]))
            except (OSError, RuntimeError) as error:
                evidence["expected_sources"][kind].append(
                    {
                        "declared_path": item["path"],
                        "expected_sha256": item["sha256"],
                        "resolution_error": f"{type(error).__name__}: {error}",
                    }
                )
                evidence["violations"].append(
                    {
                        "reason": "EXPECTED_PATH_RESOLUTION_FAILED",
                        "kind": kind,
                        "declared_path": item["path"],
                    }
                )
                continue
            source = {
                "declared_path": item["path"],
                "resolved_path": path,
                "expected_sha256": item["sha256"],
                "source_field": "runtime.files" if kind == "ort" else cuda_source,
            }
            evidence["expected_sources"][kind].append(source)
            if path in expected[kind] and expected[kind][path] != item["sha256"]:
                evidence["violations"].append(
                    {
                        "reason": "CONFLICTING_EXPECTED_HASHES",
                        "resolved_path": path,
                        "kind": kind,
                    }
                )
            expected[kind][path] = item["sha256"]
    if maps_text is None:
        try:
            maps_text = (
                maps_path if maps_path is not None else Path("/proc/self/maps")
            ).read_text()
        except OSError as error:
            evidence["violations"].append(
                {
                    "reason": "MAPS_UNAVAILABLE",
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            raise LibraryIdentityError("Loaded library maps unavailable", evidence) from error
    by_mapping = {}
    observed_hashes = {}
    for line in maps_text.splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) != 6 or not fields[5].startswith("/"):
            continue
        mapped = fields[5]
        basename = Path(mapped).name
        try:
            path = None if mapped.endswith(" (deleted)") else _resolve(mapped)
        except (OSError, RuntimeError) as error:
            record = {
                "mapped_path": mapped,
                "library_basename": basename,
                "map_lines": [line],
                "resolved_path": None,
                "kind": "unknown",
                "match_decision": "REJECT",
                "reasons": ["MAPPED_PATH_RESOLUTION_FAILED"],
                "resolution_error": f"{type(error).__name__}: {error}",
            }
            evidence["observed_mappings"].append(record)
            evidence["violations"].append(
                {
                    "reason": "MAPPED_PATH_RESOLUTION_FAILED",
                    "mapped_path": mapped,
                }
            )
            continue
        # Match the resolved basename too: a neutral-name alias cannot hide Torch/CUDA.
        name = path.name.lower() if path is not None else basename.lower()
        if "libtorch" in name or name.startswith("libc10"):
            kind = "framework"
        elif "onnxruntime" in name:
            kind = "ort"
        elif any(
            key in name for key in ("libcuda", "libcublas", "libcudnn", "libnvrtc", "libnvjit")
        ):
            kind = "cuda"
        else:
            continue
        if mapped in by_mapping:
            by_mapping[mapped]["map_lines"].append(line)
            continue
        record = {
            "mapped_path": mapped,
            "library_basename": basename,
            "kind": kind,
            "resolved_library_basename": path.name if path is not None else None,
            "map_lines": [line],
            "match_decision": "UNKNOWN",
            "hash_scope": "resolved_file_bytes_not_resident_memory",
        }
        evidence["observed_mappings"].append(record)
        by_mapping[mapped] = record
        reasons = []
        if mapped.endswith(" (deleted)"):
            record["resolved_path"] = None
            reasons.append("MAPPED_FILE_DELETED")
        else:
            record["resolved_path"] = str(path)
            record["path_alias_resolved"] = str(path) != mapped
            try:
                if path not in observed_hashes:
                    observed_hashes[path] = _digest(path)
                record["observed_sha256"] = observed_hashes[path]
            except OSError as error:
                record["observed_hash_error"] = f"{type(error).__name__}: {error}"
                reasons.append("OBSERVED_FILE_HASH_UNAVAILABLE")
            if kind == "framework":
                reasons.append("FRAMEWORK_LIBRARY_LOADED")
            else:
                declared_hash = expected[kind].get(str(path))
                record["expected_sha256"] = declared_hash
                if declared_hash is None:
                    reasons.append("UNAPPROVED_RESOLVED_PATH")
                    record["same_hash_expected_other_paths"] = [
                        key
                        for key, value in expected[kind].items()
                        if record.get("observed_sha256") == value
                    ]
                elif record.get("observed_sha256") != declared_hash:
                    reasons.append("OBSERVED_HASH_MISMATCH")
                if (
                    kind == "cuda"
                    and name.startswith(
                        (
                            "libcudart.",
                            "libcublas.",
                            "libcublaslt.",
                            "libnvrtc",
                            "libnvjitlink.",
                        )
                    )
                    and ".so.13" not in name
                ):
                    reasons.append("PRE_CUDA13_LIBRARY_LOADED")
        record["match_decision"] = "MATCH" if not reasons else "REJECT"
        record["reasons"] = reasons
        for reason in reasons:
            evidence["violations"].append(
                {
                    "reason": reason,
                    "mapped_path": mapped,
                    "resolved_path": record["resolved_path"],
                    "kind": kind,
                }
            )
    if not any(
        item["kind"] == "cuda"
        and (item.get("resolved_library_basename") or "").startswith("libcudart.")
        for item in evidence["observed_mappings"]
    ):
        evidence["violations"].append({"reason": "CUDA_RUNTIME_MAPPING_MISSING"})
    if evidence["violations"]:
        first = evidence["violations"][0]
        raise LibraryIdentityError(
            f"Loaded library identity rejected: {first['reason']}: "
            f"{first.get('mapped_path', first.get('resolved_path', 'unavailable'))}",
            evidence,
        )
    evidence["status"] = "PASSED"
    return evidence
