# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Direct multi-shard GGUF reading (``*-00001-of-000NN.gguf`` split sets).

llama.cpp / Unsloth ship large GGUF checkpoints as a *split set*: N sibling
files ``<name>-00001-of-000NN.gguf`` … ``<name>-000NN-of-000NN.gguf``. Every
shard is itself a valid GGUF container, but the model's key-value metadata
(architecture, hyper-parameters, tokenizer, chat template) lives only in the
*primary* shard (``split.no == 0``); the remaining shards carry only the
``split.*`` bookkeeping keys plus their slice of the tensor table.

:class:`GgufShardSet` assembles these files into a single logical model that is
a drop-in for :class:`~mobius.integrations.gguf._reader.GGUFModel`. It never
concatenates the shards into a second on-disk GGUF: tensors are read on demand
straight from the original shard that owns them (each shard is memory-mapped by
``gguf.GGUFReader``), so memory stays bounded to one tensor/block chunk plus the
metadata tables regardless of the total checkpoint size.

Discovery is confined to the directory of the requested file — the split
count is read from authoritative GGUF ``split.*`` metadata and cross-checked
against the ``-000i-of-000N`` filenames, never blindly string-concatenated.
Validation fails closed on missing, duplicate, mixed-revision, mixed-quant, or
corrupt shards.

Example::

    from mobius.integrations.gguf._shard_set import open_gguf_model

    model = open_gguf_model("GLM-5.2-UD-IQ1_S-00001-of-00006.gguf")
    print(model.architecture)          # 'glm-dsa'
    print(model.num_tensors)           # tensors across all six shards
    w = model.get_tensor("blk.0.attn_q.weight")   # read from its owning shard
"""

from __future__ import annotations

__all__ = [
    "GgufShardError",
    "GgufShardSet",
    "ShardInfo",
    "GgufShardManifest",
    "discover_gguf_shards",
    "parse_shard_filename",
    "open_gguf_model",
    "SHARD_FILENAME_RE",
]

import hashlib
import logging
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from mobius.integrations.gguf._reader import GGUFModel

logger = logging.getLogger(__name__)

#: ``<name>-<index>-of-<count>.gguf`` where index/count are five-digit,
#: one-based. Matches the llama.cpp ``SHARD_NAME_FORMAT`` and the reader's
#: ``_GGUF_SHARD_FILENAME_RE`` in ``_builder.py``.
SHARD_FILENAME_RE = re.compile(
    r"^(?P<prefix>.+)-(?P<index>\d{5})-of-(?P<count>\d{5})\.gguf$",
    re.IGNORECASE,
)

# GGUF split bookkeeping keys (llama.cpp ``Keys.Split``).
_SPLIT_NO = "split.no"
_SPLIT_COUNT = "split.count"
_SPLIT_TENSORS_COUNT = "split.tensors.count"

# Metadata keys whose value must be identical on every shard that declares
# them. A divergence means the shards came from different exports (mixed
# revision) or different quantization presets (mixed quant), which would build
# a corrupt model if silently stitched together.
#
# NOTE ON THE RESIDUAL MIXED-QUANT GAP: in real llama.cpp/Unsloth split sets
# these keys are carried ONLY by the primary shard (``split.no == 0``);
# continuation shards store just ``split.*`` plus tensor data. Identity
# validation therefore cannot, on metadata alone, distinguish a continuation
# shard of ``…-IQ1_S`` from the same-index continuation shard of ``…-IQ1_M``
# of the same model (identical tensor names, identical ``split.tensors.count``,
# per-shard-consistent offsets). The byte-exact guard for that case is the
# download manifest: pass ``expected_sha256`` / ``expected_sizes`` (both are
# produced by :mod:`._preflight`), which are verified per shard in
# :func:`_build_shard_infos` and fail closed on any mismatch.
_IDENTITY_KEYS = (
    "general.architecture",
    "general.name",
    "general.quantization_version",
    "general.file_type",
    "tokenizer.ggml.model",
    "tokenizer.chat_template",
)


class GgufShardError(ValueError):
    """A GGUF split set failed structural validation (fail closed)."""


def parse_shard_filename(name: str) -> tuple[str, int, int] | None:
    """Parse ``<prefix>-<index>-of-<count>.gguf`` → ``(prefix, index, count)``.

    ``index`` and ``count`` are the one-based integers from the filename.
    Returns ``None`` when *name* is not a split-shard filename.
    """
    match = SHARD_FILENAME_RE.match(Path(name).name)
    if match is None:
        return None
    return (
        match.group("prefix"),
        int(match.group("index")),
        int(match.group("count")),
    )


def discover_gguf_shards(path: str | Path) -> list[Path]:
    """Return the ordered shard files for *path*, or ``[path]`` if not sharded.

    *path* may be one shard of a split set (any index is accepted) or a
    directory containing exactly one split set. Sibling shards are found by
    globbing the *same directory* for the ``<prefix>-*-of-<count>.gguf``
    pattern — discovery never leaves that directory and never invents file
    names by string concatenation.

    Raises:
        FileNotFoundError: *path* does not exist, or a directory holds no
            ``.gguf`` files.
        GgufShardError: a directory holds more than one distinct split set
            (ambiguous), or the discovered files do not form a complete,
            contiguous ``1..count`` set.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"GGUF path not found: {p}")

    if p.is_dir():
        return _discover_in_directory(p)

    parsed = parse_shard_filename(p.name)
    if parsed is None:
        # A plain, non-sharded single-file GGUF.
        return [p]
    prefix, _index, count = parsed
    return _collect_shard_group(p.parent, prefix, count)


def _discover_in_directory(directory: Path) -> list[Path]:
    groups: dict[tuple[str, int], list[Path]] = {}
    single_files: list[Path] = []
    for candidate in sorted(directory.glob("*.gguf")):
        parsed = parse_shard_filename(candidate.name)
        if parsed is None:
            single_files.append(candidate)
            continue
        prefix, _index, count = parsed
        groups.setdefault((prefix, count), []).append(candidate)

    if not groups:
        if len(single_files) == 1:
            return [single_files[0]]
        if not single_files:
            raise FileNotFoundError(f"No *.gguf files found in {directory}")
        raise GgufShardError(
            f"{directory} contains multiple non-sharded .gguf files "
            f"{[f.name for f in single_files]}; specify one explicitly."
        )
    if len(groups) > 1:
        summaries = ", ".join(f"{prefix}-*-of-{count:05d}" for prefix, count in sorted(groups))
        raise GgufShardError(
            f"{directory} contains more than one GGUF split set ({summaries}); "
            "point at a specific shard file to disambiguate."
        )
    (prefix, count), _files = next(iter(groups.items()))
    return _collect_shard_group(directory, prefix, count)


def _collect_shard_group(directory: Path, prefix: str, count: int) -> list[Path]:
    """Collect and order the ``count`` shards named ``<prefix>-i-of-count``."""
    if count < 1:
        raise GgufShardError(f"Invalid shard count {count} in split filename")

    by_index: dict[int, Path] = {}
    for candidate in directory.glob(f"{glob_escape(prefix)}-*-of-{count:05d}.gguf"):
        parsed = parse_shard_filename(candidate.name)
        if parsed is None:
            continue
        cand_prefix, index, cand_count = parsed
        if cand_prefix != prefix or cand_count != count:
            continue
        if index in by_index:
            raise GgufShardError(
                f"Duplicate shard index {index:05d} for split set {prefix!r}: "
                f"{by_index[index].name} and {candidate.name}"
            )
        by_index[index] = candidate

    missing = [i for i in range(1, count + 1) if i not in by_index]
    if missing:
        raise GgufShardError(
            f"Incomplete GGUF split set {prefix!r}: declared {count} shards but "
            f"{len(missing)} missing (indices {[f'{i:05d}' for i in missing]}). "
            "Fetch the whole set before building."
        )
    extra = [i for i in by_index if i < 1 or i > count]
    if extra:
        raise GgufShardError(
            f"Split set {prefix!r} has out-of-range shard indices {sorted(extra)} "
            f"for declared count {count}."
        )
    return [by_index[i] for i in range(1, count + 1)]


def glob_escape(value: str) -> str:
    """Escape glob metacharacters in a literal filename prefix."""
    # ``pathlib.Path.glob`` uses fnmatch semantics; escape the wildcard set so a
    # prefix containing ``[`` / ``*`` / ``?`` is matched literally.
    return re.sub(r"([\[\]\*\?])", r"[\1]", value)


@dataclass(frozen=True)
class ShardInfo:
    """One shard's authoritative, metadata-only descriptor."""

    path: Path
    filename_index: int
    filename_count: int
    split_no: int | None
    split_count: int | None
    split_tensors_count: int | None
    size_bytes: int
    tensor_count: int
    sha256: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "filename": self.path.name,
            "filename_index": self.filename_index,
            "filename_count": self.filename_count,
            "split_no": self.split_no,
            "split_count": self.split_count,
            "split_tensors_count": self.split_tensors_count,
            "size_bytes": self.size_bytes,
            "tensor_count": self.tensor_count,
            "sha256": self.sha256,
        }


@dataclass
class GgufShardManifest:
    """Validated, metadata-only summary of a GGUF split set.

    Reusable by the preflight API: it captures everything needed to report the
    exact files/bytes/checksums and the architecture without reading any tensor
    payloads beyond the (small) GGUF headers.
    """

    architecture: str | None
    split_count: int
    total_tensors: int
    total_bytes: int
    shards: list[ShardInfo] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "architecture": self.architecture,
            "split_count": self.split_count,
            "total_tensors": self.total_tensors,
            "total_bytes": self.total_bytes,
            "shards": [s.as_dict() for s in self.shards],
        }


class GgufShardSet:
    """A GGUF split set presented as a single logical :class:`GGUFModel`.

    The public surface mirrors :class:`GGUFModel` exactly so every downstream
    consumer (config mapping, tokenizer, repacker, builder) works unchanged.
    Tensors are read on demand from the shard that owns them; nothing is
    concatenated on disk or held fully in memory.

    Args:
        shard_paths: Ordered shard files (index ``1..count``). Usually produced
            by :func:`discover_gguf_shards`.
        verify_checksums: When ``True``, compute each shard's SHA-256 and store
            it on the manifest. Off by default because hashing a multi-hundred-GB
            set is expensive; the preflight path opts in.
        expected_sha256: Optional ``{filename: sha256}`` from a download manifest
            (e.g. Hugging Face LFS metadata). When present each listed shard is
            verified and a mismatch fails closed. This is the ONLY byte-exact
            guard against a same-model mixed-quant continuation-shard swap
            (``IQ1_S`` primary + ``IQ1_M`` continuation), whose continuation
            shards carry no identity metadata to compare — supply it (from
            :mod:`._preflight`) whenever manifest data is available.
        expected_sizes: Optional ``{filename: size_bytes}`` verified the same way.
    """

    def __init__(
        self,
        shard_paths: list[str | Path],
        *,
        verify_checksums: bool = False,
        expected_sha256: dict[str, str] | None = None,
        expected_sizes: dict[str, int] | None = None,
    ) -> None:
        if not shard_paths:
            raise GgufShardError("GgufShardSet requires at least one shard file")

        self._paths = [Path(p) for p in shard_paths]
        for path in self._paths:
            if not path.is_file():
                raise FileNotFoundError(f"GGUF shard not found: {path}")

        # One GGUFModel per shard (each memory-maps its own file lazily).
        shards = [GGUFModel(path) for path in self._paths]

        self._infos = _build_shard_infos(
            self._paths,
            shards,
            verify_checksums=verify_checksums or bool(expected_sha256),
            expected_sha256=expected_sha256,
            expected_sizes=expected_sizes,
        )
        _validate_shard_set(self._infos, shards)

        # Order shards by their authoritative ``split.no`` (order independence:
        # the caller may pass them in any order). Primary shard is split.no == 0.
        order = sorted(
            range(len(shards)),
            key=lambda i: (
                self._infos[i].split_no
                if self._infos[i].split_no is not None
                else self._infos[i].filename_index - 1
            ),
        )
        self._shards = [shards[i] for i in order]
        self._infos = [self._infos[i] for i in order]
        self._primary = self._shards[0]
        self._identity_paths = [Path(shard._path) for shard in self._shards]
        # Validate primary authority before exposing any tensor access. Delaying
        # this until ``metadata`` is first read would let callers consume a
        # malformed continuation-only semantic override.
        self._metadata: dict[str, Any] | None = _merge_metadata(
            self._infos,
            self._shards,
        )

        # Combined, order-preserving tensor index: name -> owning shard.
        self._owner: dict[str, GGUFModel] = {}
        self._names: list[str] = []
        for shard in self._shards:
            for name in shard.tensor_names:
                if name in self._owner:
                    raise GgufShardError(
                        f"Duplicate tensor {name!r} across shards "
                        f"({Path(self._owner[name]._path).name} and "
                        f"{Path(shard._path).name}); refusing to build from an "
                        "ambiguous split set."
                    )
                self._owner[name] = shard
                self._names.append(name)

        self._manifest = GgufShardManifest(
            architecture=self._safe_architecture(),
            split_count=len(self._shards),
            total_tensors=len(self._names),
            total_bytes=sum(info.size_bytes for info in self._infos),
            shards=list(self._infos),
        )
        logger.info(
            "Assembled GGUF split set: %d shards, %d tensors, %.3f GiB (arch=%s)",
            self._manifest.split_count,
            self._manifest.total_tensors,
            self._manifest.total_bytes / float(1 << 30),
            self._manifest.architecture,
        )

    # -- GGUFModel-compatible surface ------------------------------------

    @property
    def architecture(self) -> str:
        """Model architecture, read from the primary shard's metadata."""
        return self._primary.architecture

    @property
    def metadata(self) -> dict[str, Any]:
        """Deterministically merged metadata from every shard.

        Split bookkeeping is canonicalized to the logical set: ``split.no``
        remains zero while ``split.count`` and ``split.tensors.count`` describe
        the complete model. All other repeated keys must have identical values.
        """
        if self._metadata is None:
            self._metadata = _merge_metadata(self._infos, self._shards)
        return self._metadata

    def get_metadata(self, key: str, default: Any = None) -> Any:
        return self.metadata.get(key, default)

    @property
    def format_version(self) -> int:
        """GGUF container version shared by all shards."""
        return self._primary.format_version

    @property
    def tensor_names(self) -> list[str]:
        return list(self._names)

    @property
    def num_tensors(self) -> int:
        return len(self._names)

    def reader_tensors(self) -> list[Any]:
        """Underlying ``gguf.ReaderTensor`` records across all shards, in order."""
        records: list[Any] = []
        for shard in self._shards:
            records.extend(shard.reader_tensors())
        return records

    @property
    def is_little_endian(self) -> bool:
        """Whether every shard stores tensor payloads in little-endian order."""
        return all(shard.is_little_endian for shard in self._shards)

    def tensor_items(self) -> Iterator[tuple[str, np.ndarray]]:
        for shard in self._shards:
            yield from shard.tensor_items()

    def tensor_items_raw(self) -> Iterator[tuple[str, np.ndarray, Any, tuple[int, ...]]]:
        for shard in self._shards:
            yield from shard.tensor_items_raw()

    def get_tensor(self, name: str) -> np.ndarray:
        shard = self._owner.get(name)
        if shard is None:
            raise KeyError(
                f"Tensor {name!r} not found in split set. Available: {self._names[:10]}..."
            )
        return shard.get_tensor(name)

    def get_tensor_type(self, name: str) -> Any:
        shard = self._owner.get(name)
        if shard is None:
            raise KeyError(f"Tensor {name!r} not found in split set.")
        return shard.get_tensor_type(name)

    def get_tensor_shape(self, name: str) -> tuple[int, ...]:
        """Get a tensor's logical numpy shape from its owning shard."""
        shard = self._owner.get(name)
        if shard is None:
            raise KeyError(f"Tensor {name!r} not found in split set.")
        return shard.get_tensor_shape(name)

    def tensor_storage_range(self, name: str) -> tuple[int, int, str]:
        """Get a tensor's byte range within its owning shard."""
        shard = self._owner.get(name)
        if shard is None:
            raise KeyError(f"Tensor {name!r} not found in split set.")
        return shard.tensor_storage_range(name)

    def _tensor_source(self, name: str) -> tuple[Path, int, Any]:
        """Return the owning path, data-section offset, and reader record."""
        shard = self._owner.get(name)
        if shard is None:
            raise KeyError(f"Tensor {name!r} not found in split set.")
        return shard._tensor_source(name)

    def dequantize_raw_tensor(
        self,
        raw_data: np.ndarray,
        quant_type: Any,
        np_shape: tuple[int, ...],
    ) -> np.ndarray:
        return self._primary.dequantize_raw_tensor(raw_data, quant_type, np_shape)

    # -- shard-set specific ---------------------------------------------

    @property
    def manifest(self) -> GgufShardManifest:
        """The validated metadata-only manifest for this split set."""
        return self._manifest

    @property
    def shard_paths(self) -> list[Path]:
        return [Path(shard._path) for shard in self._shards]

    @property
    def identity_paths(self) -> list[Path]:
        """Regular files used for immutable identity hashing, in shard order."""
        return list(self._identity_paths)

    def _set_identity_paths(self, paths: list[Path]) -> None:
        """Bind trusted regular-file aliases of the opened shard sources."""
        if len(paths) != len(self._paths):
            raise ValueError(
                f"Expected {len(self._paths)} shard identity paths, got {len(paths)}."
            )
        ordered: list[Path] = []
        for shard, candidate in zip(self._shards, paths):
            original = Path(shard._path)
            if not original.samefile(candidate):
                raise ValueError(
                    f"Identity path does not name the opened shard {original.name!r}."
                )
            ordered.append(candidate)
        self._identity_paths = ordered

    def source_matches_path(self) -> bool:
        """Return whether every shard still names the exact file that was opened."""
        return all(shard.source_matches_path() for shard in self._shards)

    def _safe_architecture(self) -> str | None:
        try:
            return self._primary.architecture
        except ValueError:
            return None

    def __repr__(self) -> str:
        return (
            f"GgufShardSet(shards={self._manifest.split_count}, "
            f"tensors={self._manifest.total_tensors}, "
            f"arch='{self._manifest.architecture}')"
        )


def _sha256_of(path: Path, *, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _build_shard_infos(
    paths: list[Path],
    shards: list[GGUFModel],
    *,
    verify_checksums: bool,
    expected_sha256: dict[str, str] | None,
    expected_sizes: dict[str, int] | None,
) -> list[ShardInfo]:
    infos: list[ShardInfo] = []
    for path, shard in zip(paths, shards):
        parsed = parse_shard_filename(path.name)
        split_no = _int_or_none(shard.get_metadata(_SPLIT_NO))
        split_count = _int_or_none(shard.get_metadata(_SPLIT_COUNT))
        if parsed is None:
            if split_no is None or split_count is None:
                raise GgufShardError(
                    f"{path.name!r} is not a ``-000i-of-000N.gguf`` shard filename "
                    "and lacks authoritative split.no/split.count metadata."
                )
            index, count = split_no + 1, split_count
        else:
            _prefix, index, count = parsed
        size_bytes = path.stat().st_size

        expected_size = (expected_sizes or {}).get(path.name)
        if expected_size is not None and expected_size != size_bytes:
            raise GgufShardError(
                f"Shard {path.name} size mismatch: manifest expected "
                f"{expected_size} bytes, on disk {size_bytes} bytes (truncated "
                "or corrupt download)."
            )

        sha256: str | None = None
        want_sha = (expected_sha256 or {}).get(path.name)
        if verify_checksums:
            sha256 = _sha256_of(path)
            if want_sha is not None and sha256.lower() != want_sha.lower():
                raise GgufShardError(
                    f"Shard {path.name} SHA-256 mismatch: manifest expected "
                    f"{want_sha}, computed {sha256} (corrupt shard)."
                )

        infos.append(
            ShardInfo(
                path=path,
                filename_index=index,
                filename_count=count,
                split_no=split_no,
                split_count=split_count,
                split_tensors_count=_int_or_none(shard.get_metadata(_SPLIT_TENSORS_COUNT)),
                size_bytes=size_bytes,
                tensor_count=shard.num_tensors,
                sha256=sha256,
            )
        )
    return infos


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _validate_shard_set(infos: list[ShardInfo], shards: list[GGUFModel]) -> None:
    """Fail closed on any structural inconsistency in the split set."""
    n = len(infos)
    filename_count = infos[0].filename_count
    prefixes = {
        parsed[0]
        for info in infos
        if (parsed := parse_shard_filename(info.path.name)) is not None
    }
    if len(prefixes) > 1:
        raise GgufShardError(
            f"Shard filenames have inconsistent basenames {sorted(prefixes)}; "
            "the files belong to different split sets."
        )
    if any(info.filename_count != filename_count for info in infos):
        counts = sorted({info.filename_count for info in infos})
        raise GgufShardError(
            f"Shards disagree on the split count in their filenames: {counts}. "
            "The files belong to different split sets."
        )
    if n != filename_count:
        raise GgufShardError(
            f"Split set declares {filename_count} shards but {n} were provided."
        )

    # Authoritative split.count metadata must agree with the filename count.
    missing_counts = [info.path.name for info in infos if info.split_count is None]
    if missing_counts:
        raise GgufShardError(
            f"Every shard must declare split.count; missing from {missing_counts}."
        )
    declared_counts = {info.split_count for info in infos}
    if declared_counts != {filename_count}:
        raise GgufShardError(
            f"GGUF split.count metadata {sorted(declared_counts)} disagrees with the "
            f"{filename_count}-shard filenames; mixed or corrupt split set."
        )

    # split.no must be a contiguous 0..count-1 permutation when present.
    missing_nos = [info.path.name for info in infos if info.split_no is None]
    if missing_nos:
        raise GgufShardError(f"Every shard must declare split.no; missing from {missing_nos}.")
    split_nos = [info.split_no for info in infos]
    if sorted(split_nos) != list(range(n)):
        raise GgufShardError(
            f"GGUF split.no values {sorted(split_nos)} are not the contiguous "
            f"set 0..{n - 1} (missing, duplicate, or mixed shards)."
        )
    mismatched_nos = [
        (info.path.name, info.filename_index, info.split_no)
        for info in infos
        if info.split_no != info.filename_index - 1
    ]
    if mismatched_nos:
        raise GgufShardError(
            f"Shard filename indices do not match their split.no metadata: {mismatched_nos}."
        )

    # split.tensors.count (total across the set) must agree everywhere and equal
    # the observed tensor total.
    missing_tensor_totals = [
        info.path.name for info in infos if info.split_tensors_count is None
    ]
    if missing_tensor_totals:
        raise GgufShardError(
            "Every shard must declare split.tensors.count; missing from "
            f"{missing_tensor_totals}."
        )
    declared_tensor_totals = {info.split_tensors_count for info in infos}
    observed_total = sum(info.tensor_count for info in infos)
    if len(declared_tensor_totals) > 1:
        raise GgufShardError(
            f"Shards disagree on split.tensors.count {sorted(declared_tensor_totals)}; "
            "mixed-revision split set."
        )
    if declared_tensor_totals and next(iter(declared_tensor_totals)) != observed_total:
        raise GgufShardError(
            f"split.tensors.count={next(iter(declared_tensor_totals))} does not match "
            f"the {observed_total} tensors actually present across the shards "
            "(missing or extra shard)."
        )

    _validate_identity(infos, shards)
    _validate_container_compatibility(infos, shards)
    _validate_offsets(infos, shards)


def _validate_container_compatibility(infos: list[ShardInfo], shards: list[GGUFModel]) -> None:
    """Require one GGUF version and byte order across the complete set."""
    versions = {shard.format_version for shard in shards}
    if len(versions) != 1:
        raise GgufShardError(
            f"Shards use incompatible GGUF container versions {sorted(versions)}."
        )
    endian_values = {shard.is_little_endian for shard in shards}
    if len(endian_values) != 1:
        names = [info.path.name for info in infos]
        raise GgufShardError(f"Shards use mixed byte order across {names}.")


def _validate_identity(infos: list[ShardInfo], shards: list[GGUFModel]) -> None:
    """Every shard that declares an identity key must agree with the others.

    This catches the common mixed-revision / mixed-quant cases (a shard whose
    primary declares a different architecture, name, quantization version, or
    tokenizer/template). It CANNOT catch a same-model ``IQ1_S`` vs ``IQ1_M``
    continuation-shard swap, because continuation shards carry no identity
    keys; supply a manifest (``expected_sha256`` / ``expected_sizes``) for that
    case. See the note on :data:`_IDENTITY_KEYS`.
    """
    for key in _IDENTITY_KEYS:
        seen: dict[Any, str] = {}
        for info, shard in zip(infos, shards):
            value = shard.get_metadata(key)
            if value is None:
                continue
            if isinstance(value, list):
                value = tuple(value)
            if value not in seen:
                seen[value] = info.path.name
            if len(seen) > 1:
                where = ", ".join(f"{v!r} in {name}" for v, name in seen.items())
                raise GgufShardError(
                    f"Shards disagree on {key!r} ({where}); this is a mixed-revision "
                    "or mixed-quant split set and cannot be stitched together."
                )


def _merge_metadata(infos: list[ShardInfo], shards: list[GGUFModel]) -> dict[str, Any]:
    """Use primary semantics and reject continuation-only metadata injection."""
    if not shards:
        raise GgufShardError("Cannot merge metadata from an empty GGUF shard set.")
    merged: dict[str, Any] = {}
    owners: dict[str, str] = {}
    primary_info = infos[0]
    primary_metadata = shards[0].metadata
    for key, value in primary_metadata.items():
        if key == _SPLIT_NO:
            continue
        merged[key] = value
        owners[key] = primary_info.path.name

    for info, shard in zip(infos[1:], shards[1:]):
        for key, value in shard.metadata.items():
            if key in {_SPLIT_NO, _SPLIT_COUNT, _SPLIT_TENSORS_COUNT}:
                continue
            # gguf-py exposes container-local pseudo fields alongside serialized
            # metadata. Only primary pseudo fields describe the logical model.
            if key.startswith("GGUF."):
                continue
            if key not in primary_metadata:
                raise GgufShardError(
                    f"Continuation shard {info.path.name} declares semantic metadata "
                    f"key {key!r} that is absent from authoritative primary shard "
                    f"{primary_info.path.name}; refusing continuation-only metadata."
                )
            if not _metadata_values_equal(primary_metadata[key], value):
                raise GgufShardError(
                    f"Shards disagree on metadata key {key!r}: "
                    f"{merged[key]!r} in {owners[key]} versus {value!r} "
                    f"in {info.path.name}."
                )
    merged[_SPLIT_NO] = 0
    merged[_SPLIT_COUNT] = len(shards)
    merged[_SPLIT_TENSORS_COUNT] = sum(info.tensor_count for info in infos)
    merged["GGUF.tensor_count"] = sum(info.tensor_count for info in infos)
    return merged


def _metadata_values_equal(left: Any, right: Any) -> bool:
    """Compare parsed metadata values, including numpy-backed unknown fields."""
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return bool(np.array_equal(left, right))
    return bool(left == right)


def _validate_offsets(infos: list[ShardInfo], shards: list[GGUFModel]) -> None:
    """Reject shards whose tensor table points outside the file (corruption)."""
    for info, shard in zip(infos, shards):
        reader = shard._reader
        alignment = getattr(reader, "alignment", 32) or 32
        data_offset = getattr(reader, "data_offset", 0)
        if data_offset % alignment != 0:
            raise GgufShardError(
                f"Shard {info.path.name} data section offset {data_offset} is not "
                f"aligned to {alignment} bytes (corrupt header)."
            )
        for tensor in reader.tensors:
            end = int(tensor.data_offset) + int(tensor.n_bytes)
            if int(tensor.data_offset) < data_offset or end > info.size_bytes:
                raise GgufShardError(
                    f"Shard {info.path.name} tensor {tensor.name!r} data span "
                    f"[{tensor.data_offset}, {end}) lies outside the file "
                    f"(size {info.size_bytes}); shard is truncated or corrupt."
                )


def open_gguf_model(
    path: str | Path,
    *,
    shard_paths: list[str | Path] | None = None,
    verify_checksums: bool = False,
    expected_sha256: dict[str, str] | None = None,
    expected_sizes: dict[str, int] | None = None,
) -> GGUFModel | GgufShardSet:
    """Open *path* as a single-file :class:`GGUFModel` or a :class:`GgufShardSet`.

    A :class:`GgufShardSet` is returned when *path* is one shard of a split set
    (or a directory holding a single split set); otherwise a plain
    :class:`GGUFModel`. Callers get the same interface either way.
    """
    shards = (
        [Path(shard_path) for shard_path in shard_paths]
        if shard_paths is not None
        else discover_gguf_shards(path)
    )
    if len(shards) == 1 and shard_paths is None:
        single = shards[0]
        # A lone ``-00001-of-00001`` file is a degenerate single-shard set;
        # treat it as a plain file. GGUFModel still validates it.
        return GGUFModel(single)
    return GgufShardSet(
        shards,
        verify_checksums=verify_checksums,
        expected_sha256=expected_sha256,
        expected_sizes=expected_sizes,
    )
