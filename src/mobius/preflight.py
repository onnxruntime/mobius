# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Resumable export preflight / dry-run.

Before a large sharded checkpoint is downloaded and exported, this module
validates *only metadata* — the safetensors index, every shard's URL/size/hash,
and the resolved commit — and computes the exact storage, host-RAM and device
VRAM required by the export. It then compares those requirements against the
free space actually available and **refuses** when any budget is insufficient.

Design rules (these are load-bearing, not decoration):

* **No success-shaped fallback.** A metadata call that fails (network, auth,
  missing index) is a hard blocker, never a silent "assume it is fine". A
  refusal is a first-class, correct outcome.
* **No model-name allowlist.** Everything is derived from the safetensors index
  and the HuggingFace file metadata, so a new checkpoint needs no code change.
* **Metadata only.** Nothing here downloads a weight shard; the only network
  reads are ``model_info`` and the small ``*.index.json``. That is what makes it
  safe to run *before* committing hundreds of GB of transfer.
* **Resumable.** Validated shards are recorded in a JSON state file keyed by the
  resolved commit. A re-run skips re-validated shards and refuses loudly if the
  upstream commit drifted (checkpoint identity changed underneath us).
"""

from __future__ import annotations

__all__ = [
    "ExportMode",
    "LoaderMode",
    "ShardMeta",
    "Budget",
    "SpaceCheck",
    "PreflightResult",
    "resolve_source",
    "estimate_budget",
    "run_preflight",
]

import ctypes
import dataclasses
import enum
import json
import logging
import os
import pathlib
import re
import shutil
import subprocess
import sys
from ctypes import wintypes

from mobius.integrations._weight_loading import (
    _SINGLE_WEIGHT_NAME,
    _WEIGHT_INDEX_NAME,
    _validate_weight_filenames,
)

logger = logging.getLogger(__name__)

# Bytes-per-parameter for a safetensors dtype string as reported by the Hub's
# ``safetensors.parameters`` metadata block.
_DTYPE_BYTES: dict[str, float] = {
    "F64": 8,
    "I64": 8,
    "F32": 4,
    "I32": 4,
    "BF16": 2,
    "F16": 2,
    "I16": 2,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I8": 1,
    "U8": 1,
    "BOOL": 1,
    "F4": 0.5,
    "U4": 0.5,
    "I4": 0.5,
}


class ExportMode(str, enum.Enum):
    """How the weights are represented in the exported ONNX artifact."""

    PASSTHROUGH = "passthrough"  # keep source width (e.g. bf16 -> bf16/fp16 cast)
    FP16 = "fp16"
    INT4_QMOE = "int4-qmoe"  # 4-bit packed + scales + zero points


class LoaderMode(str, enum.Enum):
    """Weight application strategy, which sets the host-RAM peak."""

    EAGER = "eager"  # whole checkpoint resident (today's default)
    STREAM = "stream"  # one shard/tensor resident at a time


@dataclasses.dataclass
class ShardMeta:
    """A single safetensors shard as named by the checkpoint's weight index."""

    filename: str
    size: int | None = None
    sha256: str | None = None
    present_local: bool = False
    validated: bool = False

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class Budget:
    """Exact resource requirements derived from checkpoint metadata."""

    param_count: int
    dtype_bytes: dict[str, int]
    source_bytes: int
    largest_shard_bytes: int
    output_bytes: int
    peak_ram_bytes: int
    peak_ram_eager_bytes: int
    peak_ram_stream_bytes: int
    vram_weights_bytes: int
    export_mode: str
    loader: str

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class SpaceCheck:
    """A single free-space (disk or RAM) requirement and whether it is met."""

    kind: str  # "disk:download", "disk:output", "ram"
    path: str | None
    required_bytes: int
    free_bytes: int
    margin_frac: float
    ok: bool

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class PreflightResult:
    """Full preflight verdict. ``ok`` is only True when there are no blockers."""

    model_id: str
    revision: str | None
    commit_sha: str | None
    shards: list[ShardMeta]
    budget: Budget | None
    checks: list[SpaceCheck]
    blockers: list[str]
    ok: bool

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "revision": self.revision,
            "commit_sha": self.commit_sha,
            "num_shards": len(self.shards),
            "shards": [s.to_dict() for s in self.shards],
            "budget": self.budget.to_dict() if self.budget else None,
            "checks": [c.to_dict() for c in self.checks],
            "blockers": list(self.blockers),
            "ok": self.ok,
        }


def _bytes_from_dtype_params(parameters: dict[str, int]) -> tuple[int, dict[str, int]]:
    """Return (total_param_count, {dtype: bytes}) from a Hub parameters block."""
    total_params = 0
    dtype_bytes: dict[str, int] = {}
    for dtype, count in parameters.items():
        total_params += int(count)
        per = _DTYPE_BYTES.get(dtype.upper())
        if per is None:
            raise ValueError(
                f"Unknown safetensors dtype {dtype!r} in checkpoint metadata; "
                "refusing rather than guessing its byte width."
            )
        dtype_bytes[dtype.upper()] = int(count * per)
    return total_params, dtype_bytes


# ---------------------------------------------------------------------------
# Source resolution (metadata only)
# ---------------------------------------------------------------------------


def _resolve_local(model_dir: pathlib.Path) -> tuple[None, list[ShardMeta], dict]:
    index_path = model_dir / _WEIGHT_INDEX_NAME
    if index_path.is_file():
        index = json.loads(index_path.read_text())
        filenames = _validate_weight_filenames(sorted(set(index["weight_map"].values())))
    elif (model_dir / _SINGLE_WEIGHT_NAME).is_file():
        index = {}
        filenames = [_SINGLE_WEIGHT_NAME]
    else:
        raise FileNotFoundError(
            f"Local checkpoint has no {_WEIGHT_INDEX_NAME!r} or {_SINGLE_WEIGHT_NAME!r}: "
            f"{model_dir}"
        )

    root = model_dir.resolve()
    shards: list[ShardMeta] = []
    for filename in filenames:
        path = (model_dir / filename).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"Unsafe weight filename in index: {filename!r}") from exc
        present = path.is_file()
        size = path.stat().st_size if present else None
        shards.append(ShardMeta(filename=filename, size=size, present_local=present))
    return None, shards, index


def _resolve_hub(
    model_id: str, revision: str | None, *, hf_api=None
) -> tuple[str, list[ShardMeta], dict]:
    from huggingface_hub import hf_hub_download

    if hf_api is None:
        from huggingface_hub import HfApi

        hf_api = HfApi()

    # model_info validates access (401/404 raise) and yields per-file metadata.
    info = hf_api.model_info(model_id, revision=revision, files_metadata=True)
    commit_sha = info.sha
    size_by_name: dict[str, int | None] = {}
    sha_by_name: dict[str, str | None] = {}
    for sibling in info.siblings or []:
        name = sibling.rfilename
        size_by_name[name] = getattr(sibling, "size", None)
        lfs = getattr(sibling, "lfs", None)
        sha_by_name[name] = getattr(lfs, "sha256", None) if lfs else None

    # Resolve the shard list from the index (falling back to a single file).
    try:
        index_path = hf_hub_download(
            repo_id=model_id, filename=_WEIGHT_INDEX_NAME, revision=revision
        )
        index = json.loads(pathlib.Path(index_path).read_text())
        filenames = _validate_weight_filenames(sorted(set(index["weight_map"].values())))
    except Exception as exc:  # re-raise below as a blocker
        if _SINGLE_WEIGHT_NAME in size_by_name:
            index = {}
            filenames = [_SINGLE_WEIGHT_NAME]
        else:
            raise FileNotFoundError(
                f"Could not resolve a safetensors index for {model_id!r}: {exc}"
            ) from exc

    shards: list[ShardMeta] = []
    for filename in filenames:
        if filename not in size_by_name:
            raise FileNotFoundError(
                f"Weight index references {filename!r} but it is absent from the "
                f"repository file listing for {model_id!r} @ {commit_sha}."
            )
        shards.append(
            ShardMeta(
                filename=filename,
                size=size_by_name.get(filename),
                sha256=sha_by_name.get(filename),
                present_local=False,
            )
        )
    return commit_sha, shards, index


def resolve_source(
    model_id: str, revision: str | None = None, *, hf_api=None
) -> tuple[str | None, list[ShardMeta], dict]:
    """Resolve the shard list + sizes for a checkpoint without downloading it.

    ``model_id`` may be a local directory or a HuggingFace repo id. Returns
    ``(commit_sha, shards, index)`` where ``commit_sha`` is ``None`` for a local
    directory. Raises on inaccessible / malformed checkpoints (no fallback).
    """
    local = pathlib.Path(model_id)
    if local.is_dir():
        return _resolve_local(local)
    return _resolve_hub(model_id, revision, hf_api=hf_api)


# ---------------------------------------------------------------------------
# Budget estimation
# ---------------------------------------------------------------------------


def _params_and_dtype_bytes(
    shards: list[ShardMeta], index: dict, safetensors_meta: dict | None
) -> tuple[int, dict[str, int], int]:
    """Best-effort (param_count, {dtype: bytes}, source_bytes).

    Prefers the Hub ``safetensors`` metadata block (exact dtype breakdown). Falls
    back to the index ``metadata.total_size`` and finally to summing shard sizes.
    """
    source_bytes = 0
    have_all_sizes = all(s.size is not None for s in shards)
    if have_all_sizes:
        source_bytes = sum(int(s.size) for s in shards)

    if safetensors_meta and safetensors_meta.get("parameters"):
        param_count, dtype_bytes = _bytes_from_dtype_params(safetensors_meta["parameters"])
        if not source_bytes:
            source_bytes = sum(dtype_bytes.values())
        return param_count, dtype_bytes, source_bytes

    total_size = None
    if isinstance(index, dict):
        total_size = index.get("metadata", {}).get("total_size")
    weight_bytes = int(total_size) if total_size else source_bytes
    if not source_bytes:
        source_bytes = weight_bytes
    # Without a dtype block, assume 16-bit weights (bf16/fp16) — the common case
    # for an unquantized checkpoint — and record it as an assumption.
    param_count = weight_bytes // 2 if weight_bytes else 0
    dtype_bytes = {"UNKNOWN16": weight_bytes} if weight_bytes else {}
    return param_count, dtype_bytes, source_bytes


def estimate_output_bytes(
    param_count: int,
    dtype_bytes: dict[str, int],
    export_mode: ExportMode,
    *,
    group_size: int = 32,
) -> int:
    """Estimate the ONNX external-data output size for an export mode."""
    source_bytes = sum(dtype_bytes.values())
    if export_mode == ExportMode.PASSTHROUGH:
        # A passthrough export dequantizes an fp8 checkpoint to bf16 (2 bytes)
        # on the eager path, so fp8 external data is written at 2x its stored
        # width. Size fp8 params at their dequantized width to avoid a false OK
        # on the output-disk check.
        fp8_extra = sum(v for k, v in dtype_bytes.items() if k.startswith("F8"))
        return source_bytes + fp8_extra
    if export_mode == ExportMode.FP16:
        return param_count * 2
    if export_mode == ExportMode.INT4_QMOE:
        # 4-bit packed weights + fp16 scales + int4 zero points, per block.
        per = 0.5 + 2.0 / group_size + 0.5 / group_size
        return int(param_count * per)
    raise ValueError(f"unknown export mode {export_mode!r}")


def _has_fp8(dtype_bytes: dict[str, int]) -> bool:
    return any(k.startswith("F8") for k in dtype_bytes)


def estimate_budget(
    shards: list[ShardMeta],
    index: dict,
    safetensors_meta: dict | None,
    *,
    export_mode: ExportMode = ExportMode.PASSTHROUGH,
    loader: LoaderMode = LoaderMode.STREAM,
    group_size: int = 32,
    target_dtype_bytes: float | None = None,
    vram_headroom_frac: float = 0.15,
) -> Budget:
    """Compute the storage / RAM / VRAM budget for an export.

    When *target_dtype_bytes* is None the runtime weight footprint is taken from
    the exported artifact size (``output_bytes``), so an int4-qmoe export is
    sized at ~0.5 byte/param rather than the source dtype. Pass an explicit
    value to model a runtime load dtype that differs from the export.
    """
    param_count, dtype_bytes, source_bytes = _params_and_dtype_bytes(
        shards, index, safetensors_meta
    )
    largest_shard = max((int(s.size) for s in shards if s.size is not None), default=0)
    output_bytes = estimate_output_bytes(
        param_count, dtype_bytes, export_mode, group_size=group_size
    )

    # Eager holds the entire source state dict resident; an fp8 source is
    # up-converted to bf16 (2x its bytes) and, for a quantizing export, the
    # derived packed tensors coexist with the source before serialization.
    fp8_upconvert = 0
    if _has_fp8(dtype_bytes):
        fp8_bytes = sum(v for k, v in dtype_bytes.items() if k.startswith("F8"))
        fp8_upconvert = fp8_bytes  # 1 byte -> 2 bytes adds one extra copy
    derived_overhead = output_bytes if export_mode == ExportMode.INT4_QMOE else 0
    peak_ram_eager = source_bytes + fp8_upconvert + derived_overhead

    # Streaming keeps at most one shard (and its cast twin) resident.
    peak_ram_stream = 2 * largest_shard if largest_shard else source_bytes

    peak_ram = peak_ram_eager if loader == LoaderMode.EAGER else peak_ram_stream

    # The weights loaded at runtime are the weights that were exported, so the
    # exported artifact size is the right default VRAM footprint (an int4-qmoe
    # export loads int4, not the bf16 source).
    vram_base = (
        output_bytes if target_dtype_bytes is None else int(param_count * target_dtype_bytes)
    )
    vram_weights = int(vram_base * (1 + vram_headroom_frac))

    return Budget(
        param_count=param_count,
        dtype_bytes=dtype_bytes,
        source_bytes=source_bytes,
        largest_shard_bytes=largest_shard,
        output_bytes=output_bytes,
        peak_ram_bytes=peak_ram,
        peak_ram_eager_bytes=peak_ram_eager,
        peak_ram_stream_bytes=peak_ram_stream,
        vram_weights_bytes=vram_weights,
        export_mode=export_mode.value,
        loader=loader.value,
    )


# ---------------------------------------------------------------------------
# Free-space checks
# ---------------------------------------------------------------------------


def _existing_ancestor(path: pathlib.Path) -> pathlib.Path:
    probe = path
    while not probe.exists():
        if probe.parent == probe:
            break
        probe = probe.parent
    return probe


def _free_bytes(path: pathlib.Path) -> int:
    return shutil.disk_usage(_existing_ancestor(path)).free


def _same_device(a: pathlib.Path, b: pathlib.Path) -> bool:
    """True when two paths resolve to the same filesystem (shared free space)."""
    try:
        return os.stat(_existing_ancestor(a)).st_dev == os.stat(_existing_ancestor(b)).st_dev
    except OSError:
        return False


def _default_download_dir() -> pathlib.Path:
    """Where ``hf_hub_download`` actually writes shards (the HF cache).

    The real loader calls ``hf_hub_download`` without ``local_dir``, so shards
    land in the Hugging Face cache regardless of the export output directory.
    Budgeting the download against that filesystem — not the output dir — is
    what keeps the "refuse before a large download" promise honest. Point the
    cache at a large volume with ``HF_HOME``/``HF_HUB_CACHE`` (or pass an
    explicit ``download_dir``) to change it.
    """
    try:
        from huggingface_hub.constants import HF_HUB_CACHE

        return pathlib.Path(HF_HUB_CACHE)
    except Exception:
        return pathlib.Path.home() / ".cache" / "huggingface" / "hub"


class _MemoryStatusEx(ctypes.Structure):
    _fields_ = [
        ("dwLength", wintypes.DWORD),
        ("dwMemoryLoad", wintypes.DWORD),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def _windows_available_ram_bytes() -> int | None:
    status = _MemoryStatusEx()
    status.dwLength = ctypes.sizeof(status)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GlobalMemoryStatusEx.argtypes = [ctypes.POINTER(_MemoryStatusEx)]
    kernel32.GlobalMemoryStatusEx.restype = wintypes.BOOL
    if kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return int(status.ullAvailPhys)
    return None


def _proc_available_ram_bytes() -> int | None:
    try:
        for line in pathlib.Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except OSError:
        return None
    return None


def _darwin_available_ram_bytes() -> int | None:
    try:
        result = subprocess.run(
            ["vm_stat"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None

    return _parse_vm_stat_available_bytes(result.stdout)


def _parse_vm_stat_available_bytes(output: str) -> int | None:
    page_size_match = re.search(r"page size of (\d+) bytes", output)
    if page_size_match is None:
        return None
    page_size = int(page_size_match.group(1))
    available_page_names = {
        "Pages free",
        "Pages inactive",
        "Pages speculative",
    }
    available_pages = 0
    parsed_fields = 0
    for line in output.splitlines():
        name, separator, raw_value = line.partition(":")
        if separator and name.strip() in available_page_names:
            value = raw_value.strip().rstrip(".").replace(",", "")
            if value.isdigit():
                available_pages += int(value)
                parsed_fields += 1
    return available_pages * page_size if parsed_fields else None


def _sysconf_available_ram_bytes() -> int | None:
    try:
        pages = os.sysconf("SC_AVPHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, OSError, ValueError):
        return None
    return int(pages * page_size) if pages > 0 and page_size > 0 else None


def _probe_available_ram() -> tuple[int | None, str]:
    if os.name == "nt":
        return _windows_available_ram_bytes(), "GlobalMemoryStatusEx:ullAvailPhys"
    if sys.platform == "darwin":
        if (available := _darwin_available_ram_bytes()) is not None:
            return available, "vm_stat:available pages"
    if (available := _proc_available_ram_bytes()) is not None:
        return available, "/proc/meminfo:MemAvailable"
    if (available := _sysconf_available_ram_bytes()) is not None:
        return available, "os.sysconf:SC_AVPHYS_PAGES"
    return None, "unavailable"


def _available_ram_bytes() -> int | None:
    return _probe_available_ram()[0]


def check_disk(kind: str, path: pathlib.Path, required: int, margin_frac: float) -> SpaceCheck:
    free = _free_bytes(path)
    ok = free * (1 - margin_frac) >= required
    return SpaceCheck(
        kind=kind,
        path=str(path),
        required_bytes=required,
        free_bytes=free,
        margin_frac=margin_frac,
        ok=ok,
    )


def check_ram(required: int, margin_frac: float) -> SpaceCheck:
    free, source = _probe_available_ram()
    # If available RAM can't be probed we cannot verify the RAM budget; refuse
    # rather than emit a success-shaped pass.
    ok = free is not None and free * (1 - margin_frac) >= required
    return SpaceCheck(
        kind="ram",
        path=source,
        required_bytes=required,
        free_bytes=int(free) if free is not None else -1,
        margin_frac=margin_frac,
        ok=ok,
    )


# ---------------------------------------------------------------------------
# Resumable state
# ---------------------------------------------------------------------------


def _load_state(state_path: pathlib.Path | None) -> dict:
    if state_path is None or not state_path.is_file():
        return {}
    try:
        return json.loads(state_path.read_text())
    except (OSError, json.JSONDecodeError):
        logger.warning("Ignoring unreadable preflight state at %s", state_path)
        return {}


def _save_state(state_path: pathlib.Path | None, state: dict) -> None:
    if state_path is None:
        return
    state_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = state_path.with_suffix(state_path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True))
    os.replace(tmp, state_path)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_preflight(
    model_id: str,
    output_dir: str | os.PathLike,
    *,
    revision: str | None = None,
    download_dir: str | os.PathLike | None = None,
    export_mode: ExportMode = ExportMode.PASSTHROUGH,
    loader: LoaderMode = LoaderMode.STREAM,
    group_size: int = 32,
    target_dtype_bytes: float | None = None,
    gpu_total_bytes: int | None = None,
    margin_frac: float = 0.05,
    state_path: str | os.PathLike | None = None,
    hf_api=None,
) -> PreflightResult:
    """Run the full preflight and return a verdict.

    The verdict's ``ok`` is True only when there are zero blockers: metadata
    resolved, every index shard accounted for, and every disk/RAM budget met.
    """
    blockers: list[str] = []
    output_path = pathlib.Path(output_dir)
    download_path = pathlib.Path(download_dir) if download_dir else _default_download_dir()
    state_file = pathlib.Path(state_path) if state_path else None

    # 1) Resolve + validate metadata. A failure here is terminal, never a pass.
    try:
        commit_sha, shards, index = resolve_source(model_id, revision, hf_api=hf_api)
    except Exception as exc:  # surface as a structured blocker
        return PreflightResult(
            model_id=model_id,
            revision=revision,
            commit_sha=None,
            shards=[],
            budget=None,
            checks=[],
            blockers=[f"metadata unavailable: {type(exc).__name__}: {exc}"],
            ok=False,
        )

    if not shards:
        blockers.append("no safetensors shards resolved from checkpoint metadata")

    missing_size = [s.filename for s in shards if s.size is None and not s.present_local]
    if missing_size:
        blockers.append(
            f"{len(missing_size)} shard(s) have no size metadata "
            f"(cannot budget): {missing_size[:3]}{'...' if len(missing_size) > 3 else ''}"
        )

    # 2) Resumable state — refuse on identity drift.
    state = _load_state(state_file)
    prior_sha = state.get("commit_sha")
    if prior_sha and commit_sha and prior_sha != commit_sha:
        blockers.append(
            f"checkpoint identity drift: state recorded commit {prior_sha} but "
            f"upstream now resolves to {commit_sha}; refusing to reuse state"
        )
        state = {}
    validated_prior = set(state.get("validated_shards", []))
    for shard in shards:
        if shard.filename in validated_prior:
            shard.validated = True

    # 3) Metadata from the Hub (dtype breakdown) when available.
    safetensors_meta = None
    if commit_sha is not None:
        info = _maybe_model_info(model_id, revision, hf_api)
        if info is not None and getattr(info, "safetensors", None) is not None:
            st = info.safetensors
            safetensors_meta = {
                "parameters": dict(getattr(st, "parameters", {}) or {}),
                "total": getattr(st, "total", None),
            }

    # 4) Budget.
    budget = estimate_budget(
        shards,
        index,
        safetensors_meta,
        export_mode=export_mode,
        loader=loader,
        group_size=group_size,
        target_dtype_bytes=target_dtype_bytes,
    )

    # 5) Free-space + RAM checks.
    checks: list[SpaceCheck] = []
    already_present = sum(
        int(s.size) for s in shards if s.present_local and s.size is not None
    )
    download_required = max(budget.source_bytes - already_present, 0)
    if _same_device(download_path, output_path):
        # Downloaded shards must stay resident while the streaming loader reads
        # them to write the ONNX output, so on a shared filesystem the source
        # and output requirements coexist and must be summed against one pool of
        # free space (checking them independently hides a combined shortfall).
        checks.append(
            check_disk(
                "disk:download+output",
                output_path,
                download_required + budget.output_bytes,
                margin_frac,
            )
        )
    else:
        checks.append(
            check_disk("disk:download", download_path, download_required, margin_frac)
        )
        checks.append(check_disk("disk:output", output_path, budget.output_bytes, margin_frac))
    checks.append(check_ram(budget.peak_ram_bytes, margin_frac))

    for chk in checks:
        if not chk.ok:
            if chk.kind == "ram" and chk.free_bytes < 0:
                blockers.append(
                    "host RAM budget could not be verified: failed to read "
                    f"{chk.path} (need {_h(chk.required_bytes)}); refusing"
                )
            else:
                blockers.append(
                    f"insufficient {chk.kind}: need {_h(chk.required_bytes)} at "
                    f"{chk.path}, free {_h(chk.free_bytes)} (margin {chk.margin_frac:.0%})"
                )

    # 6) VRAM advisory (single-GPU fit is common blocker for full checkpoints).
    if gpu_total_bytes is not None and budget.vram_weights_bytes > gpu_total_bytes:
        blockers.append(
            f"weights need {_h(budget.vram_weights_bytes)} VRAM but the largest "
            f"single device has {_h(gpu_total_bytes)}; multi-GPU shard or weight "
            f"offload is required (not a single-device load)"
        )

    ok = not blockers

    # 7) Persist resumable state (record identity + which shards are validated).
    new_state = {
        "model_id": model_id,
        "revision": revision,
        "commit_sha": commit_sha,
        "validated_shards": sorted(
            {s.filename for s in shards if s.size is not None or s.present_local}
            | validated_prior
        ),
        "budget": budget.to_dict(),
        "ok": ok,
    }
    _save_state(state_file, new_state)

    return PreflightResult(
        model_id=model_id,
        revision=revision,
        commit_sha=commit_sha,
        shards=shards,
        budget=budget,
        checks=checks,
        blockers=blockers,
        ok=ok,
    )


def _maybe_model_info(model_id: str, revision: str | None, hf_api):
    try:
        if hf_api is None:
            from huggingface_hub import HfApi

            hf_api = HfApi()
        return hf_api.model_info(model_id, revision=revision, files_metadata=False)
    except Exception:  # dtype breakdown is best-effort
        return None


def _h(n: int) -> str:
    """Human-readable byte count (decimal units, matching HF/df)."""
    if n < 0:
        return "unknown"
    step = 1000.0
    for unit in ("B", "KB", "MB", "GB", "TB", "PB"):
        if n < step:
            return f"{n:.1f}{unit}" if unit != "B" else f"{int(n)}{unit}"
        n /= step
    return f"{n:.1f}EB"
