# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Metadata-only preflight for GGUF split sets (local or Hugging Face Hub).

A north-star checkpoint like ``unsloth/GLM-5.2-GGUF`` UD-IQ1_{S,M} is a
six-file GGUF split set weighing 200+ GiB. Before anyone downloads a single
tensor payload, this module answers the only questions that matter for an
export decision:

* Exactly which files make up the split set, their byte sizes, and their LFS
  sha256 checksums.
* The GGUF ``general.architecture`` and the canonical Mobius model type it
  bridges to (e.g. ``glm-dsa`` → ``glm_moe_dsa``).
* Whether the export is *blocked* — most importantly the sparse-MoE honesty
  blocker: routed IQ/MXFP4 experts have no sparse ``BlockQuantizedMoE`` fusion,
  so a build would be dense-all-expert.

Everything here is **metadata only**. For a Hub reference it uses
``HfApi.model_info(expand=["gguf"])`` for the architecture/expert counts and
``HfApi.get_paths_info`` for per-file sizes + LFS sha256 — no ``hf_hub_download``
of tensor data. For a local path it reads the (memory-mapped) GGUF headers via
:class:`GgufShardSet`, never the tensor bodies.

The report is JSON-serialisable and the computation is idempotent, so a caller
can persist it and resume without re-hitting the network (see ``cache_path``).

This is deliberately a small library of GGUF-specific functions, not a
``_weight_loading.py`` and not a duplicate of the generic export-preflight CLI:
it exposes :func:`preflight_gguf` / :func:`preflight_local_gguf` /
:func:`preflight_hf_gguf` for reuse by any front-end.
"""

from __future__ import annotations

__all__ = [
    "GgufFileMeta",
    "GgufPreflightReport",
    "preflight_gguf",
    "preflight_local_gguf",
    "preflight_hf_gguf",
]

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from mobius.integrations.gguf._config_mapping import (
    resolve_model_type,
)
from mobius.integrations.gguf._shard_set import (
    GgufShardManifest,
    open_gguf_model,
    parse_shard_filename,
)

logger = logging.getLogger(__name__)

# Canonical model types whose routed-expert stack is a Mixture-of-Experts.
# Used only to decide whether the sparse-MoE fusion blocker is relevant; the
# authoritative signal is the GGUF ``<arch>.expert_count`` metadata when present.
_MOE_MODEL_TYPES = frozenset(
    {
        "glm_moe_dsa",
        "deepseek4",
        "deepseek3",
        "deepseek2",
        "qwen3moe",
        "qwen2moe",
        "nemotron_h_moe",
    }
)

# GGUF quantization tokens whose routed experts lower to native
# ``pkg.nxrt::BlockQuantizedMatMul`` blocks (no sparse top-k fusion exists yet).
# Only int4 ``MatMulNBits`` experts fuse into ``com.microsoft::QMoE``.
_NATIVE_BLOCK_QUANT_RE = re.compile(
    r"(IQ1_[SM]|IQ2_(XXS|XS|S)|IQ3_(XXS|S)|IQ4_(NL|XS)|MXFP4|TQ1_0|TQ2_0)",
    re.IGNORECASE,
)


@dataclass
class GgufFileMeta:
    """One shard file's metadata-only descriptor (no tensor payload read)."""

    filename: str
    size_bytes: int
    sha256: str | None
    shard_index: int | None
    shard_count: int | None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GgufPreflightReport:
    """Metadata-only assessment of a GGUF split set.

    Attributes:
        source: The requested reference (local path or ``owner/repo[:file]``).
        location: ``"local"`` or ``"hf"``.
        is_sharded: True when more than one shard makes up the set.
        architecture: GGUF ``general.architecture`` string (``None`` if the
            architecture metadata could not be read without a download).
        resolved_model_type: Canonical Mobius registry key the architecture
            bridges to.
        quantization: Detected quantization label (e.g. ``"IQ1_S"``) when it can
            be read from metadata or the filename.
        split_count: Declared number of shards.
        total_files: Number of shard files found/enumerated.
        total_bytes: Sum of shard byte sizes.
        total_tensors: Total tensor count across the set (``None`` when only
            remote metadata is available and it does not expose the count).
        total_params: Total parameter count when the Hub exposes it (``None``
            for a local-only metadata read).
        num_experts: Routed-expert count when known (MoE detection signal).
        sparse_moe_fusion_supported: False when routed experts would lower to
            native block ``BlockQuantizedMatMul`` nodes with no sparse fusion.
        files: Per-file metadata (sizes + checksums).
        blockers: Hard export blockers (each is a human-readable reason).
        warnings: Non-fatal advisories.
    """

    source: str
    location: str
    is_sharded: bool
    architecture: str | None
    resolved_model_type: str | None
    quantization: str | None
    split_count: int | None
    total_files: int
    total_bytes: int
    total_tensors: int | None
    total_params: int | None
    num_experts: int | None
    sparse_moe_fusion_supported: bool
    files: list[GgufFileMeta] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def exportable(self) -> bool:
        """True when no hard blocker was recorded."""
        return not self.blockers

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["exportable"] = self.exportable
        return data

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.as_dict(), indent=indent, sort_keys=True)

    def render(self) -> str:
        """Human-readable multi-line summary (no tensor payloads touched)."""
        lines: list[str] = []
        lines.append(f"GGUF preflight: {self.source}  [{self.location}]")
        arch = self.architecture or "?"
        model_type = self.resolved_model_type or "?"
        lines.append(f"  architecture : {arch}  ->  {model_type}")
        if self.quantization:
            lines.append(f"  quantization : {self.quantization}")
        shard_word = "shards" if self.is_sharded else "file"
        lines.append(
            f"  split        : {self.total_files} {shard_word}"
            + (f" (declared split.count={self.split_count})" if self.split_count else "")
        )
        lines.append(
            f"  total bytes  : {self.total_bytes:,} ({_gib(self.total_bytes):.3f} GiB)"
        )
        if self.total_tensors is not None:
            lines.append(f"  total tensors: {self.total_tensors:,}")
        if self.total_params is not None:
            lines.append(
                f"  parameters   : {self.total_params:,} ({self.total_params / 1e9:.1f} B)"
            )
        if self.num_experts:
            lines.append(f"  moe experts  : {self.num_experts}")
        lines.append(
            "  sparse MoE   : "
            + ("supported" if self.sparse_moe_fusion_supported else "NOT supported")
        )
        lines.append("  files:")
        for f in self.files:
            sha = (f.sha256[:12] + "…") if f.sha256 else "(no sha256)"
            lines.append(f"    - {f.filename}  {f.size_bytes:,} B  sha256:{sha}")
        if self.blockers:
            lines.append("  BLOCKERS:")
            for b in self.blockers:
                lines.append(f"    ✗ {b}")
        else:
            lines.append("  BLOCKERS: none")
        if self.warnings:
            lines.append("  warnings:")
            for w in self.warnings:
                lines.append(f"    ! {w}")
        return "\n".join(lines)


def _gib(num_bytes: int) -> float:
    return num_bytes / (1024**3)


def _detect_quantization(*candidates: str | None) -> str | None:
    """Best-effort quantization label from a filename/name (reporting only).

    This is used solely for the human-readable report and the fusion blocker
    advisory — never for architecture resolution, which is driven exclusively by
    ``general.architecture`` metadata.
    """
    for text in candidates:
        if not text:
            continue
        match = _NATIVE_BLOCK_QUANT_RE.search(text)
        if match is not None:
            return match.group(0).upper()
    return None


def _is_moe(model_type: str | None, num_experts: int | None) -> bool:
    if num_experts and num_experts > 0:
        return True
    return model_type in _MOE_MODEL_TYPES


def _assess_sparse_moe(
    *,
    model_type: str | None,
    num_experts: int | None,
    quantization: str | None,
    source: str,
) -> tuple[bool, list[str]]:
    """Return ``(fusion_supported, blockers)`` for the routed-expert stack.

    Mobius has exactly one sparse MoE path: int4 ``MatMulNBits`` experts fused
    into ``com.microsoft::QMoE``. GGUF always tags ``quant_method="gguf"``, so
    routed experts follow the dense per-expert loop; when their block format is
    a native IQ/MXFP4 layout they lower to ``pkg.nxrt::BlockQuantizedMatMul``
    with no fusion, i.e. dense-all-expert compute. Report that as a blocker.
    """
    if not _is_moe(model_type, num_experts):
        return True, []

    native_block = _NATIVE_BLOCK_QUANT_RE.search(quantization or "") is not None
    if not native_block:
        # int4-class MoE experts can be repacked to MatMulNBits and fused.
        return True, []

    blocker = (
        f"sparse-MoE fusion blocker: {source} is a MoE architecture "
        f"('{model_type}') whose routed experts use '{quantization}' native "
        "blocks. No sparse top-k BlockQuantizedMoE fusion exists for these "
        "block formats — only int4 MatMulNBits experts fuse into "
        "com.microsoft::QMoE. A default export would build a dense-all-expert "
        "graph (every expert evaluated for every token) with no performance "
        "guarantee, so build_from_gguf fails closed. Next slice: sparse IQ-block "
        "BlockQuantizedMoE fusion (top-k gather over native-block expert weights)."
    )
    return False, [blocker]


def preflight_local_gguf(
    path: str | Path,
    *,
    verify_checksums: bool = False,
) -> GgufPreflightReport:
    """Metadata-only preflight of a local GGUF file or split set.

    Reads only the GGUF headers (memory-mapped) to enumerate shards, sizes,
    tensor counts, architecture, and the sparse-MoE blocker. Tensor payloads are
    never read. When *verify_checksums* is set, per-shard sha256 is computed
    (this reads file bytes but not through the tensor API).
    """
    path = Path(path)
    model = open_gguf_model(path, verify_checksums=verify_checksums)

    manifest: GgufShardManifest | None = getattr(model, "manifest", None)
    architecture = _safe_arch(model)
    model_type = resolve_model_type(architecture) if architecture else None
    num_experts = _read_expert_count(model, architecture)
    quantization = _detect_quantization(
        _read_str_metadata(model, "general.file_type_name"),
        path.name,
        _read_str_metadata(model, "general.name"),
    )

    files: list[GgufFileMeta] = []
    if manifest is not None:
        is_sharded = True
        split_count = manifest.split_count
        total_tensors = manifest.total_tensors
        total_bytes = manifest.total_bytes
        for shard in manifest.shards:
            files.append(
                GgufFileMeta(
                    filename=shard.path.name,
                    size_bytes=shard.size_bytes,
                    sha256=shard.sha256,
                    shard_index=shard.split_no,
                    shard_count=shard.split_count,
                )
            )
    else:
        is_sharded = False
        split_count = 1
        total_tensors = model.num_tensors
        size_bytes = path.stat().st_size
        total_bytes = size_bytes
        sha = None
        if verify_checksums:
            from mobius.integrations.gguf._shard_set import _sha256_of

            sha = _sha256_of(path)
        files.append(
            GgufFileMeta(
                filename=path.name,
                size_bytes=size_bytes,
                sha256=sha,
                shard_index=0,
                shard_count=1,
            )
        )

    fusion_ok, blockers = _assess_sparse_moe(
        model_type=model_type,
        num_experts=num_experts,
        quantization=quantization,
        source=str(path),
    )
    warnings: list[str] = []
    if architecture is None:
        warnings.append("architecture metadata missing from primary shard")

    return GgufPreflightReport(
        source=str(path),
        location="local",
        is_sharded=is_sharded,
        architecture=architecture,
        resolved_model_type=model_type,
        quantization=quantization,
        split_count=split_count,
        total_files=len(files),
        total_bytes=total_bytes,
        total_tensors=total_tensors,
        total_params=None,
        num_experts=num_experts,
        sparse_moe_fusion_supported=fusion_ok,
        files=files,
        blockers=blockers,
        warnings=warnings,
    )


def preflight_hf_gguf(
    repo_id: str,
    *,
    filename: str | None = None,
    revision: str | None = None,
    token: str | bool | None = None,
) -> GgufPreflightReport:
    """Metadata-only preflight of a GGUF split set on the Hugging Face Hub.

    Uses ``HfApi.get_paths_info`` for per-file sizes + LFS sha256 and
    ``HfApi.model_info(expand=["gguf"])`` for the architecture / expert count.
    No tensor payload is downloaded. When *filename* names one shard of a split
    set, its sibling shards are enumerated from the repo file listing (the
    ``-000i-of-000N`` pattern is validated, never blindly concatenated).
    """
    from huggingface_hub import HfApi

    api = HfApi(token=token if isinstance(token, str) else None)
    ref, sep, inline_file = repo_id.partition(":")
    if sep:
        repo_id = ref
        filename = filename or inline_file

    all_files = api.list_repo_files(repo_id, revision=revision, token=token)
    gguf_files = [f for f in all_files if f.lower().endswith(".gguf")]
    if not gguf_files:
        raise FileNotFoundError(f"No .gguf files found in {repo_id!r}.")

    shard_files = _select_shard_files(gguf_files, filename)
    is_sharded = len(shard_files) > 1

    paths_info = api.get_paths_info(
        repo_id, shard_files, revision=revision, token=token, expand=True
    )
    info_by_path = {getattr(p, "path", None): p for p in paths_info}

    files: list[GgufFileMeta] = []
    total_bytes = 0
    for name in shard_files:
        info = info_by_path.get(name)
        size = int(getattr(info, "size", 0) or 0)
        lfs = getattr(info, "lfs", None)
        sha = getattr(lfs, "sha256", None) if lfs is not None else None
        parsed = parse_shard_filename(name.rsplit("/", 1)[-1])
        idx = parsed[1] if parsed else None
        cnt = parsed[2] if parsed else None
        total_bytes += size
        files.append(
            GgufFileMeta(
                filename=name,
                size_bytes=size,
                sha256=sha,
                shard_index=idx,
                shard_count=cnt,
            )
        )

    architecture, num_experts, total_tensors, total_params = _hf_gguf_metadata(
        api, repo_id, revision, token
    )
    model_type = resolve_model_type(architecture) if architecture else None
    quantization = _detect_quantization(filename, *shard_files)

    fusion_ok, blockers = _assess_sparse_moe(
        model_type=model_type,
        num_experts=num_experts,
        quantization=quantization,
        source=f"{repo_id}{('/' + filename) if filename else ''}",
    )

    warnings: list[str] = []
    if architecture is None:
        warnings.append(
            "architecture not exposed by Hub metadata; resolve after a "
            "metadata-only header fetch of the primary shard"
        )
    split_count = files[0].shard_count if is_sharded and files else (1 if files else None)

    return GgufPreflightReport(
        source=repo_id if not filename else f"{repo_id}:{filename}",
        location="hf",
        is_sharded=is_sharded,
        architecture=architecture,
        resolved_model_type=model_type,
        quantization=quantization,
        split_count=split_count,
        total_files=len(files),
        total_bytes=total_bytes,
        total_tensors=total_tensors,
        total_params=total_params,
        num_experts=num_experts,
        sparse_moe_fusion_supported=fusion_ok,
        files=files,
        blockers=blockers,
        warnings=warnings,
    )


def preflight_gguf(
    source: str | Path,
    *,
    filename: str | None = None,
    revision: str | None = None,
    token: str | bool | None = None,
    verify_checksums: bool = False,
    cache_path: str | Path | None = None,
) -> GgufPreflightReport:
    """Preflight a GGUF split set from a local path or a Hub reference.

    ``source`` is treated as a local path when it exists on disk (or its
    directory does); otherwise it is treated as an ``owner/repo[:file]`` Hub
    reference. The whole operation is metadata-only.

    When *cache_path* is given the report is read from / written to that JSON
    file, making repeated preflights idempotent and resumable without re-hitting
    the network.
    """
    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists():
            logger.info("Loading cached GGUF preflight from %s", cache_path)
            return _report_from_dict(json.loads(cache_path.read_text()))

    report = _dispatch_preflight(
        source,
        filename=filename,
        revision=revision,
        token=token,
        verify_checksums=verify_checksums,
    )

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(report.to_json())
        logger.info("Wrote GGUF preflight cache to %s", cache_path)
    return report


def _dispatch_preflight(
    source: str | Path,
    *,
    filename: str | None,
    revision: str | None,
    token: str | bool | None,
    verify_checksums: bool,
) -> GgufPreflightReport:
    source_path = Path(source)
    looks_local = source_path.exists() or (
        source_path.parent.exists() and source_path.parent != source_path
    )
    if looks_local:
        return preflight_local_gguf(source, verify_checksums=verify_checksums)
    return preflight_hf_gguf(str(source), filename=filename, revision=revision, token=token)


def _select_shard_files(gguf_files: list[str], filename: str | None) -> list[str]:
    """Return the shard file list for *filename* (or the whole repo's set)."""
    if filename is not None:
        base = filename.rsplit("/", 1)[-1]
        parsed = parse_shard_filename(base)
        if parsed is None:
            # Single explicit file (not a shard); return just it.
            match = [f for f in gguf_files if f.rsplit("/", 1)[-1] == base]
            return match or [filename]
        prefix, _, count = parsed
        siblings = _shards_for_prefix(gguf_files, prefix)
        if len(siblings) != count:
            logger.warning(
                "Declared split.count=%d but found %d matching shard files for "
                "prefix %r; reporting the files that are present.",
                count,
                len(siblings),
                prefix,
            )
        return siblings

    # No filename: if the repo holds exactly one split set, return it; if it
    # holds a single plain file, return that; otherwise report all gguf files.
    shard_groups: dict[str, list[str]] = {}
    plain: list[str] = []
    for f in gguf_files:
        parsed = parse_shard_filename(f.rsplit("/", 1)[-1])
        if parsed is None:
            plain.append(f)
        else:
            shard_groups.setdefault(parsed[0], []).append(f)
    if len(shard_groups) == 1 and not plain:
        return sorted(next(iter(shard_groups.values())))
    if not shard_groups and len(plain) == 1:
        return plain
    return sorted(gguf_files)


def _shards_for_prefix(gguf_files: list[str], prefix: str) -> list[str]:
    out: list[str] = []
    for f in gguf_files:
        base = f.rsplit("/", 1)[-1]
        parsed = parse_shard_filename(base)
        if parsed is not None and parsed[0] == prefix:
            out.append(f)
    return sorted(out)


def _hf_gguf_metadata(
    api: Any, repo_id: str, revision: str | None, token: str | bool | None
) -> tuple[str | None, int | None, int | None, int | None]:
    """Return ``(architecture, num_experts, total_tensors, total_params)``.

    Uses ``model_info(expand=["gguf"])`` which surfaces the parsed GGUF header
    fields without downloading tensor bytes. The Hub's ``gguf.total`` is the
    *parameter* count (not the tensor count); any field the Hub does not expose
    comes back ``None`` (the caller records a warning).
    """
    architecture: str | None = None
    num_experts: int | None = None
    total_tensors: int | None = None
    total_params: int | None = None
    try:
        info = api.model_info(repo_id, revision=revision, token=token, expand=["gguf"])
    except Exception as error:
        logger.info("model_info(expand=gguf) unavailable for %s: %s", repo_id, error)
        return architecture, num_experts, total_tensors, total_params

    gguf_meta = getattr(info, "gguf", None)
    if isinstance(gguf_meta, dict):
        architecture = gguf_meta.get("architecture") or gguf_meta.get("general.architecture")
        total_params = gguf_meta.get("total")
        total_tensors = gguf_meta.get("total_tensors")
        num_experts = (
            gguf_meta.get("expert_count")
            or gguf_meta.get("num_local_experts")
            or gguf_meta.get("n_expert")
        )
    elif gguf_meta is not None:
        architecture = getattr(gguf_meta, "architecture", None)
        total_params = getattr(gguf_meta, "total", None)
        total_tensors = getattr(gguf_meta, "total_tensors", None)
        num_experts = getattr(gguf_meta, "expert_count", None)
    return (
        architecture,
        int(num_experts) if num_experts else None,
        int(total_tensors) if total_tensors else None,
        int(total_params) if total_params else None,
    )


def _safe_arch(model: Any) -> str | None:
    try:
        arch = model.architecture
    except Exception:
        return None
    return arch or None


def _read_str_metadata(model: Any, key: str) -> str | None:
    try:
        value = model.get_metadata(key)
    except Exception:
        return None
    if value is None:
        return None
    return str(value)


def _read_expert_count(model: Any, architecture: str | None) -> int | None:
    if not architecture:
        return None
    key = f"{architecture}.expert_count"
    try:
        value = model.get_metadata(key)
    except Exception:
        return None
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _report_from_dict(data: dict[str, Any]) -> GgufPreflightReport:
    files = [GgufFileMeta(**f) for f in data.get("files", [])]
    known = {
        "source",
        "location",
        "is_sharded",
        "architecture",
        "resolved_model_type",
        "quantization",
        "split_count",
        "total_files",
        "total_bytes",
        "total_tensors",
        "total_params",
        "num_experts",
        "sparse_moe_fusion_supported",
        "blockers",
        "warnings",
    }
    kwargs = {k: data[k] for k in known if k in data}
    return GgufPreflightReport(files=files, **kwargs)
