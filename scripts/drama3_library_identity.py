# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Progressive Linux DSO mapping evidence, using only the standard library."""

from __future__ import annotations

import hashlib
import re
import struct
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


def _header(path):
    with path.open("rb") as stream:
        data = stream.read(64)
    result = {
        "bytes_read": len(data),
        "magic_hex": data[:4].hex(),
        "scope": "resolved_file_header_not_resident_memory",
    }
    if data[:4] != b"\x7fELF":
        result["file_type"] = "NON_ELF" if len(data) == 64 else "AMBIGUOUS_SHORT_HEADER"
        return result
    result["file_type"] = "AMBIGUOUS_ELF_HEADER"
    if len(data) < 16 or data[4] not in (1, 2) or data[5] not in (1, 2) or data[6] != 1:
        return result
    required = 52 if data[4] == 1 else 64
    if len(data) < required:
        return result
    endian = "<" if data[5] == 1 else ">"
    size_offset = 40 if data[4] == 1 else 52
    result.update(
        elf_class=32 if data[4] == 1 else 64,
        elf_byte_order="little" if data[5] == 1 else "big",
        elf_type=struct.unpack_from(endian + "H", data, 16)[0],
        elf_machine=struct.unpack_from(endian + "H", data, 18)[0],
        elf_header_version=struct.unpack_from(endian + "I", data, 20)[0],
    )
    if (
        struct.unpack_from(endian + "H", data, size_offset)[0] == required
        and result["elf_machine"] != 0
        and result["elf_header_version"] == 1
    ):
        result["file_type"] = "ELF"
    return result


def _name_kind(name):
    name = name.lower()
    if "libtorch" in name or name.startswith("libc10"):
        return "framework"
    if "onnxruntime" in name:
        return "ort"
    if any(
        key in name for key in ("libcuda", "libcublas", "libcudnn", "libnvrtc", "libnvjit")
    ):
        return "cuda"
    return None


def _installation_root(path, kind):
    parent = path.parent
    # A pinned system driver must not make all of /usr/lib part of CUDA scope.
    if kind == "cuda" and (
        path.name.lower().startswith("libcuda.so")
        or str(parent) in ("/usr/lib", "/usr/lib64", "/lib", "/lib64")
    ):
        return None
    if parent.name in ("lib", "lib64", "capi"):
        return parent.parent
    return parent


def _scope_kind(raw, resolved, expected, roots):
    for kind, allowed in expected.items():
        if resolved is not None and str(resolved) in allowed:
            return kind
    for name in (
        Path(raw.removesuffix(" (deleted)")).name,
        resolved.name if resolved is not None else "",
    ):
        kind = _name_kind(name)
        if kind is not None:
            return kind
    for candidate in (Path(raw.removesuffix(" (deleted)")), resolved):
        if candidate is not None:
            for kind, directories in roots.items():
                if any(candidate.is_relative_to(root) for root in directories):
                    return kind
    return None


def _objects(maps_text):
    """Group all file-backed segments before classification, independent of order."""
    objects = {}
    for line in maps_text.splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) != 6 or not fields[5].startswith("/"):
            continue
        # procfs device+inode identifies aliases of one mapped object. Synthetic
        # zero-inode rows have no object identity, so group only by their path.
        key = (
            (fields[3], fields[4]) if fields[4].isdigit() and int(fields[4]) else (fields[5],)
        )
        obj = objects.setdefault(key, {})
        obj.setdefault(fields[5], []).append({"raw": line, "permissions": fields[1]})
    return list(objects.values())


def observe_loaded_libraries(
    cuda_files, ort_files, *, maps_text=None, maps_path=None, cuda_source="runtime.libraries"
):
    """Collect all relevant map rows before raising; never approve an unknown path.

    Expected hashes are policy, not observations. Observed hashes are read from
    the actual resolved mapped files. Same-hash unapproved paths remain failures.
    """
    expected = {"cuda": {}, "ort": {}}
    roots = {"cuda": set(), "ort": set()}
    evidence = {
        "status": "FAILED",
        "expected_sources": {"cuda": [], "ort": []},
        "observed_mappings": [],
        "violations": [],
        "outside_attested_scope": [],
        "attestation_scope": (
            "ORT/CUDA canonical paths, installation roots and relevant raw/resolved names; "
            "not a full-process native-code or adversarial-renaming closure guarantee"
        ),
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
            for candidate in (Path(item["path"]), Path(path)):
                root = _installation_root(candidate, kind)
                if root is not None:
                    roots[kind].add(root)
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
    observed_hashes = {}
    for object_id, segments in enumerate(_objects(maps_text)):
        records = []
        for mapped, lines in segments.items():
            path = None
            error = None
            try:
                if not mapped.endswith(" (deleted)"):
                    path = _resolve(mapped)
            except (OSError, RuntimeError) as failure:
                error = f"{type(failure).__name__}: {failure}"
            records.append(
                {
                    "mapped_path": mapped,
                    "library_basename": Path(mapped).name,
                    "resolved_path": str(path) if path is not None else None,
                    "resolved_library_basename": path.name if path is not None else None,
                    "path_alias_resolved": path is not None and str(path) != mapped,
                    "object_id": object_id,
                    "map_lines": [line["raw"] for line in lines],
                    "map_permissions": [line["permissions"] for line in lines],
                    "kind": _scope_kind(mapped, path, expected, roots),
                    **({"resolution_error": error} if error else {}),
                }
            )
        scoped_kind = next((record["kind"] for record in records if record["kind"]), None)
        executable = any(
            "x" in line["permissions"] for lines in segments.values() for line in lines
        )
        permissions_valid = all(
            re.fullmatch(r"[r-][w-][x-][ps]", line["permissions"])
            for lines in segments.values()
            for line in lines
        )
        for record in records:
            mapped = record["mapped_path"]
            if scoped_kind is None:
                record.update(kind="outside_scope", match_decision="OUTSIDE_ATTESTED_SCOPE")
                evidence["outside_attested_scope"].append(record)
                continue
            kind = record["kind"] or scoped_kind
            record.update(
                kind=kind,
                object_has_executable_segment=executable,
                hash_scope="resolved_file_bytes_not_resident_memory",
            )
            evidence["observed_mappings"].append(record)
            reasons = []
            path = (
                Path(record["resolved_path"]) if record["resolved_path"] is not None else None
            )
            if record.get("resolution_error"):
                reasons.append("MAPPED_PATH_RESOLUTION_FAILED")
            elif mapped.endswith(" (deleted)"):
                reasons.append("MAPPED_FILE_DELETED")
            if not permissions_valid:
                reasons.append("AMBIGUOUS_MAP_PERMISSIONS")
            if path is not None:
                try:
                    record["observed_header"] = _header(path)
                except OSError as error:
                    record["observed_header_error"] = f"{type(error).__name__}: {error}"
                    reasons.append("OBSERVED_HEADER_UNAVAILABLE")
                try:
                    if path not in observed_hashes:
                        observed_hashes[path] = _digest(path)
                    record["observed_sha256"] = observed_hashes[path]
                except OSError as error:
                    record["observed_hash_error"] = f"{type(error).__name__}: {error}"
                    reasons.append("OBSERVED_FILE_HASH_UNAVAILABLE")
            header_type = record.get("observed_header", {}).get("file_type")
            if header_type is not None and header_type.startswith("AMBIGUOUS"):
                reasons.append(header_type)
            shared_name = any(
                re.search(r"\.so(?:\.|$)", name.lower())
                for name in (
                    Path(mapped.removesuffix(" (deleted)")).name,
                    record["resolved_library_basename"] or "",
                )
            )
            native = executable or bool(shared_name) or header_type == "ELF"
            record["native_candidate"] = native
            record["shared_library_name_evidence"] = bool(shared_name)
            if not reasons and not native and header_type == "NON_ELF":
                record.update(
                    match_decision="EXCLUDED_NON_NATIVE_DATA",
                    reasons=["EXCLUDED_NON_NATIVE_DATA"],
                )
                continue
            if header_type == "NON_ELF" and native:
                reasons.append("NON_ELF_NATIVE_CANDIDATE")
            if header_type == "ELF" and record["observed_header"]["elf_type"] not in (2, 3):
                reasons.append("UNSUPPORTED_ELF_OBJECT_TYPE")
            if kind == "framework":
                reasons.append("FRAMEWORK_LIBRARY_LOADED")
            else:
                declared_hash = expected[kind].get(str(path)) if path is not None else None
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
                name = (record["resolved_library_basename"] or "").lower()
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
        and item["match_decision"] == "MATCH"
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
