# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""GGUF → ONNX build pipeline.

Converts ``.gguf`` model files to ONNX using the standard build
pipeline. Quantized target storage is the default: affine linear-layer weights
are repacked into MatMulNBits format and compatible token embeddings into
GatherBlockQuantized format. For text-only builds, runtime-supported native
IQ/MXFP4 projection blocks are preserved for BlockQuantizedMatMul. Lossy
normalization to a common packed target is allowed only with a deterministic
warning and persistent fidelity report. Set
``keep_quantized=False`` to request a fully float import explicitly.
"""

from __future__ import annotations

__all__ = ["build_from_gguf"]

import logging
import math
import re
import shutil
from collections import Counter
from collections.abc import Callable, Collection, Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

import numpy as np
import tqdm
from huggingface_hub import (
    HfApi,
    get_hf_file_metadata,
    get_session,
    hf_hub_download,
    hf_hub_url,
    try_to_load_from_cache,
)
from huggingface_hub.utils import build_hf_headers

from mobius._model_package import ModelPackage
from mobius.integrations.gguf._arch_registry import (
    MMPROJ_ARCHITECTURE,
    arch_names_with,
    try_get_arch_spec,
)
from mobius.integrations.gguf._errors import (
    DisabledGGUFArchitectureError,
    ShardedGGUFNotSupportedError,
    UnsupportedGGUFArchitectureError,
)
from mobius.integrations.gguf._header import (
    GGUFHeaderInfo,
    GGUFHeaderTruncatedError,
    _gguf_architecture_from_header,
    _gguf_header_info_from_header,
)
from mobius.integrations.gguf._shard_set import MAX_GGUF_SHARD_COUNT
from mobius.integrations.gguf._spec import Support, TensorRole

_HUB_PREFLIGHT_TRANSPORT_ERRORS: tuple[type[BaseException], ...] = (OSError,)
try:
    from httpx import HTTPError as _HttpxHTTPError
except ImportError:
    pass
else:
    _HUB_PREFLIGHT_TRANSPORT_ERRORS += (_HttpxHTTPError,)

if TYPE_CHECKING:
    from mobius.tasks import ModelTask

logger = logging.getLogger(__name__)

_GGUF_SHARD_FILENAME_RE = re.compile(
    r"-(?P<index>\d{5})-of-(?P<count>\d{5})\.gguf$",
    re.IGNORECASE,
)
_NEMOTRON_H_MOE_ARCHITECTURE = "nemotron_h_moe"
_GGUF_HEADER_RANGE_BYTES = 16 * 1024 * 1024
_GGUF_SPLIT_DISCOVERY_MULTIPLIER = 4


@dataclass(frozen=True, slots=True, eq=False)
class _GGUFPreflightRevision:
    """Immutable Hub revision carrying the selected file's bounded header."""

    revision: str
    header_info: GGUFHeaderInfo

    def __str__(self) -> str:
        return self.revision

    def __eq__(self, other: object) -> bool:
        if isinstance(other, _GGUFPreflightRevision):
            return self.revision == other.revision and self.header_info == other.header_info
        return isinstance(other, str) and self.revision == other

    def __hash__(self) -> int:
        return hash((self.revision, self.header_info))


@dataclass(frozen=True, slots=True, eq=False)
class _GGUFPreflightFallbackRevision:
    """Immutable revision whose bounded header requires local validation."""

    revision: str

    def __str__(self) -> str:
        return self.revision

    def __eq__(self, other: object) -> bool:
        if isinstance(other, _GGUFPreflightFallbackRevision):
            return self.revision == other.revision
        return isinstance(other, str) and self.revision == other

    def __hash__(self) -> int:
        return hash(self.revision)


@dataclass(frozen=True, slots=True, eq=False)
class _ResolvedGGUFPath:
    """Downloaded shard path carrying the trusted Hub manifest to the reader."""

    path: str
    expected_sha256: dict[str, str]
    expected_sizes: dict[str, int]
    shard_paths: list[str]

    def __fspath__(self) -> str:
        return self.path

    def __str__(self) -> str:
        return self.path

    def __eq__(self, other: object) -> bool:
        if isinstance(other, _ResolvedGGUFPath):
            return (
                self.path == other.path
                and self.expected_sha256 == other.expected_sha256
                and self.expected_sizes == other.expected_sizes
                and self.shard_paths == other.shard_paths
            )
        return isinstance(other, str) and self.path == other

    def __hash__(self) -> int:
        return hash(self.path)


def _summarize_nemotron_h_moe_layout(
    tensor_names: Iterable[str],
) -> tuple[Counter[str], tuple[int, ...], dict[int, frozenset[str]]]:
    """Summarize base-layer and MTP mixer types from Nemotron-H GGUF names."""
    layer_kinds: dict[int, set[str]] = {}
    mtp_blocks: set[int] = set()
    for name in tensor_names:
        match = re.match(r"^blk\.(\d+)\.(.+)$", name)
        if match is None:
            continue
        block_index = int(match.group(1))
        suffix = match.group(2)
        kinds = layer_kinds.setdefault(block_index, set())
        if suffix.startswith("nextn."):
            mtp_blocks.add(block_index)
        elif suffix.startswith("ssm_"):
            kinds.add("mamba")
        elif suffix.startswith(("ffn_", "exp_probs_")):
            kinds.add("moe")
        elif suffix.startswith(("attn_q.", "attn_k.", "attn_v.", "attn_output.")):
            kinds.add("attention")

    base_counts: Counter[str] = Counter()
    for block_index, kinds in layer_kinds.items():
        if block_index not in mtp_blocks:
            base_counts.update(kinds)
    mtp_kinds = {
        block_index: frozenset(layer_kinds.get(block_index, set()))
        for block_index in sorted(mtp_blocks)
    }
    return base_counts, tuple(sorted(mtp_blocks)), mtp_kinds


def _raise_for_unsupported_gguf_architecture(
    architecture: str,
    *,
    source: str,
    tensor_names: Iterable[str] | None = None,
    allow_mmproj_companion: bool = False,
    allow_preflight_only: bool = False,
) -> None:
    """Reject known architectures whose config, tensor map, or graph is unavailable.

    This gate runs immediately after the GGUF header is available so dtype,
    quantization, config extraction, registry lookup, and graph construction
    cannot obscure the architecture verdict. Runtime packaging is deliberately
    excluded: an importable graph may still have a deferred runtime contract.
    """
    spec = try_get_arch_spec(architecture)
    if spec is None:
        raise UnsupportedGGUFArchitectureError(
            f"GGUF architecture {architecture!r} has no immutable capability-registry "
            f"spec for {source!r}; refusing generic Hugging Face config/model dispatch "
            "before config extraction. No ONNX artifacts were emitted."
        )
    if spec.gguf_arch == MMPROJ_ARCHITECTURE and allow_mmproj_companion:
        # mmproj sidecars are opened deliberately by the multimodal path, which
        # pairs them with a text backbone and applies role-specific validation.
        return
    if spec.preflight_only and allow_preflight_only:
        return
    import_verdicts = {
        name: verdict
        for name, verdict in spec.verdicts.items()
        if name in {"config", "tensor_map", "graph"}
    }
    unavailable = [
        name for name, verdict in import_verdicts.items() if verdict is not Support.SUPPORTED
    ]
    if not unavailable:
        return

    layout = ""
    if architecture == _NEMOTRON_H_MOE_ARCHITECTURE and tensor_names is not None:
        counts, mtp_blocks, mtp_kinds = _summarize_nemotron_h_moe_layout(tensor_names)
        mtp_kind_names = {index: sorted(kinds) for index, kinds in mtp_kinds.items()}
        layout = (
            " Detected base schedule: "
            f"{counts['mamba']} Mamba + {counts['moe']} MoE + "
            f"{counts['attention']} attention layers; auxiliary MTP blocks: "
            f"{list(mtp_blocks)} with mixer types {mtp_kind_names}."
        )

    if any(verdict is Support.REJECTED for verdict in import_verdicts.values()):
        raise DisabledGGUFArchitectureError(
            f"Direct GGUF conversion for architecture {spec.gguf_arch!r} is intentionally "
            f"disabled for {source!r}.{layout} {spec.reason} No ONNX artifacts were emitted."
        )
    capabilities = ", ".join(unavailable)
    raise UnsupportedGGUFArchitectureError(
        f"GGUF architecture {spec.gguf_arch!r} is deferred for {source!r} before config "
        f"extraction because these import capabilities are unavailable: {capabilities}. "
        f"{spec.reason} No ONNX artifacts were emitted."
    )


def _raise_for_sharded_gguf(
    *,
    source: str,
    filename: str | None = None,
    split_count: int | None = None,
) -> None:
    """Reject one-shard inputs before they can produce an incomplete model."""
    shard_index: int | None = None
    if filename is not None:
        match = _GGUF_SHARD_FILENAME_RE.search(filename)
        if match is not None:
            shard_index = int(match.group("index"))
            split_count = int(match.group("count"))
    if split_count is None or split_count <= 1:
        return

    shard_detail = f" shard {shard_index} of {split_count}" if shard_index else ""
    raise ShardedGGUFNotSupportedError(
        f"Sharded GGUF input is not supported: {source!r} is{shard_detail}. "
        "The GGUF builder reads one file and cannot assemble split tensor tables; "
        "continuing would emit an incomplete ONNX model. Select a single-file GGUF "
        "variant, join the shards with a GGUF-aware tool, or build from the original "
        "Hugging Face checkpoint."
    )


class SparseMoEExportError(NotImplementedError):
    """Routed MoE experts have no sparse fusion for their quantization.

    Raised before export when a checkpoint's routed experts lower to per-expert
    native ``pkg.nxrt::BlockQuantizedMatMul`` nodes (e.g. GLM-5.2 UD-IQ1_{S,M})
    for which no sparse top-k ``BlockQuantizedMoE`` fusion exists — exporting
    anyway would recompute every expert for every token (dense-all-expert
    compute) with no performance guarantee.
    """


def _routed_dense_block_expert_paths(module) -> list[str]:
    """Return module paths of routed experts lowered as native block linears.

    A routed expert is a :class:`BlockQuantizedLinear` whose qualified name
    contains a ``.moe.experts.`` (or bare ``.experts.``) segment but is *not* a
    ``shared_expert``. These are the modules that, absent a sparse top-k MoE
    fusion, force a dense loop over all experts.
    """
    from mobius.components import BlockQuantizedLinear

    paths: list[str] = []
    for name, mod in module.named_modules():
        if not isinstance(mod, BlockQuantizedLinear):
            continue
        if "shared_expert" in name:
            continue
        if ".experts." not in name:
            continue
        paths.append(name)
    return paths


def _assert_sparse_moe_capability(module, config, *, source: str, allow_dense: bool) -> None:
    """Fail closed when routed IQ-block experts have no sparse MoE fusion.

    The int4 ``MatMulNBits`` dense-fallback → ``com.microsoft::QMoE`` rewrite is
    the only sparse MoE path today; native IQ/MXFP4 block experts stay as
    per-expert ``BlockQuantizedMatMul`` nodes. Exporting those yields a graph
    that evaluates every expert for every token. Refuse by default; only proceed
    (with a loud warning) when the caller explicitly opts in.
    """
    num_experts = int(getattr(config, "num_local_experts", 0) or 0)
    if num_experts <= 0:
        return
    dense_paths = _routed_dense_block_expert_paths(module)
    if not dense_paths:
        return

    example = ", ".join(sorted(dense_paths)[:3])
    detail = (
        f"{len(dense_paths)} routed expert projections (e.g. {example}) in "
        f"{source!r} lower to per-expert native BlockQuantizedMatMul nodes."
    )
    if allow_dense:
        logger.warning(
            "allow_dense_moe: exporting %d routed MoE expert(s) as independent "
            "BlockQuantizedMatMul nodes for %s. This recomputes EVERY expert for "
            "every token (dense-all-expert compute); it is NOT a performance path "
            "and makes no throughput claim. %s",
            len(dense_paths),
            source,
            detail,
        )
        return

    raise SparseMoEExportError(
        "Sparse-MoE export blocker: "
        f"{detail} No sparse top-k BlockQuantizedMoE fusion exists for these "
        "block formats — only int4 MatMulNBits experts are fused into "
        "com.microsoft::QMoE. Exporting anyway would build a dense-all-expert "
        "graph (every expert evaluated for every token) with no performance "
        "guarantee, so the build fails closed. To study the dense fallback, "
        "re-run with allow_dense_moe=True (or set "
        "MOBIUS_ALLOW_DENSE_MOE_EXPERTS=1); this makes no throughput claim. "
        "The supported fix is a sparse IQ-block BlockQuantizedMoE fusion "
        "(top-k gather over native-block expert weights)."
    )


def _fuse_native_block_moe(pkg, *, allow_dense: bool) -> int:
    """Collapse routed native-block expert storms into sparse ``BlockQuantizedMoE``.

    Runs on the final, fully-weighted graph so the fusion can stack each layer's
    per-expert native blocks byte-for-byte into one expert-major bank. Every
    candidate layer is validated before any node is emitted, so an unfusable
    layer (per-expert bias, ragged/incomplete group, ...) raises
    :class:`SparseMoEExportError` atomically with the graph untouched, unless
    ``allow_dense`` downgrades it to a warning + dense keep.

    A layer that mixes native formats across its fc1/fc2/fc3 banks (GLM-5.2
    UD-IQ1) can only be expressed with the ``block_layout_version=2``
    per-projection ABI, which no shipped onnx-genai runtime executes yet. The
    production builder therefore never enables v2: such layers always
    typed-reject here rather than emit an unrunnable node. There is no
    environment or CLI opt-in -- v2 stays a schema-construction test path until a
    real typed runtime-capability handshake exists.
    """
    # Imported lazily: the generic rewrite lives in the rewrite_rules package and
    # must not be pulled into the GGUF import graph at module load time.
    from mobius.rewrite_rules import fuse_block_quantized_moe

    fused = 0
    for model in pkg.values():
        # No ``_allow_perproj_v2_schema`` argument: the production authority path
        # always fails closed for mixed-format v2 (fail-safe default).
        fused += fuse_block_quantized_moe(model, allow_dense_moe=allow_dense)
    return fused


def _routed_dense_block_matmul_nodes(model) -> list:
    """Routed per-expert ``BlockQuantizedMatMul`` nodes surviving in a graph.

    A routed expert projection is a ``pkg.nxrt::BlockQuantizedMatMul`` whose
    packed-weight initializer path carries an ``.experts.`` segment and is not a
    ``shared_expert``. After :func:`_fuse_native_block_moe` runs, any that remain
    are an un-collapsed dense-all-expert storm.
    """
    hits = []
    for node in model.graph:
        if node.op_type != "BlockQuantizedMatMul":
            continue
        weight = node.inputs[1] if len(node.inputs) > 1 else None
        name = getattr(weight, "name", None) or ""
        if ".experts." in name and "shared_expert" not in name:
            hits.append(node)
    return hits


def _assert_sparse_moe_graph(pkg, *, source: str, allow_dense: bool) -> None:
    """Sparse-MoE honesty gate over the final (post-fusion) graph state.

    :func:`_fuse_native_block_moe` already fails closed with a precise reason for
    every routed native-block storm it recognises. This backstop catches any
    routed per-expert ``BlockQuantizedMatMul`` storm that survived fusion (e.g. a
    dispatch shape the rewrite did not recognise): shipping it silently would be
    a dense-all-expert graph with no throughput guarantee. Opting into the dense
    fallback (``allow_dense``) is already warned about by the fusion, so this
    gate only enforces the fail-closed default.
    """
    if allow_dense:
        return
    storm = [n for model in pkg.values() for n in _routed_dense_block_matmul_nodes(model)]
    if not storm:
        return
    raise SparseMoEExportError(
        "Sparse-MoE export blocker: "
        f"{len(storm)} routed expert projection(s) in {source!r} remain as "
        "per-expert pkg.nxrt::BlockQuantizedMatMul nodes after native-block MoE "
        "fusion (no sparse top-k BlockQuantizedMoE was applied). Exporting anyway "
        "would build a dense-all-expert graph (every expert evaluated for every "
        "token) with no performance guarantee, so the build fails closed. To "
        "study the dense fallback, re-run with allow_dense_moe=True (or set "
        "MOBIUS_ALLOW_DENSE_MOE_EXPERTS=1); this makes no throughput claim."
    )


def _preflight_hf_gguf(api: HfApi, repo_id: str, filename: str) -> None:
    """Use Hub metadata to reject known-bad inputs before a multi-GB download."""
    source = f"{repo_id}:{filename}"
    try:
        info = api.model_info(repo_id, expand=["gguf"])
    except TypeError as error:
        if "expand" not in str(error):
            raise
        logger.info(
            "Skipping Hub GGUF architecture preflight for %s because this "
            "huggingface_hub version has no model_info(expand=...) support; "
            "the downloaded or cached local header will still be validated.",
            source,
        )
        return
    except _HUB_PREFLIGHT_TRANSPORT_ERRORS as error:
        logger.warning(
            "Hub GGUF architecture preflight failed for %s (%s); continuing to "
            "hf_hub_download so an authenticated or cached file can still be used. "
            "The local header will be validated before graph construction.",
            source,
            error,
        )
        return
    gguf_metadata = getattr(info, "gguf", None)
    if isinstance(gguf_metadata, Mapping):
        architecture = gguf_metadata.get("architecture")
    else:
        architecture = getattr(gguf_metadata, "architecture", None)
    if isinstance(architecture, str):
        _raise_for_unsupported_gguf_architecture(
            architecture,
            source=source,
        )


def _gguf_architecture_from_header_prefix(data: bytes, *, source: str) -> str:
    """Read ``general.architecture`` from a bounded GGUF header prefix."""
    architecture = _gguf_architecture_from_header(data, source=source)
    assert architecture is not None
    return architecture


def _gguf_header_info_from_header_prefix(
    data: bytes,
    *,
    source: str,
) -> GGUFHeaderInfo:
    """Read architecture and split bookkeeping from a bounded GGUF header."""
    return _gguf_header_info_from_header(
        data,
        source=source,
        require_architecture=False,
    )


def _validate_preflight_split_header(info: GGUFHeaderInfo, *, source: str) -> None:
    """Require complete, internally consistent split bookkeeping when present."""
    split_values = (info.split_no, info.split_count, info.split_tensors_count)
    if all(value is None for value in split_values):
        return
    if any(value is None for value in split_values):
        raise ValueError(
            f"GGUF header {source!r} has incomplete split bookkeeping "
            f"(no={info.split_no}, count={info.split_count}, "
            f"tensors.count={info.split_tensors_count}). No payload was downloaded."
        )
    assert info.split_no is not None
    assert info.split_count is not None
    assert info.split_tensors_count is not None
    if (
        info.split_count <= 0
        or info.split_count > MAX_GGUF_SHARD_COUNT
        or not 0 <= info.split_no < info.split_count
    ):
        raise ValueError(
            f"GGUF header {source!r} has invalid split bookkeeping "
            f"(no={info.split_no}, count={info.split_count}, maximum="
            f"{MAX_GGUF_SHARD_COUNT}). "
            "No payload was downloaded."
        )
    if info.split_tensors_count < info.tensor_count:
        raise ValueError(
            f"GGUF header {source!r} declares split.tensors.count="
            f"{info.split_tensors_count}, smaller than its local tensor count "
            f"{info.tensor_count}. No payload was downloaded."
        )


def _preflight_hf_gguf_file(
    repo_id: str,
    filename: str,
    *,
    revision: str = "main",
    allow_mmproj_companion: bool = False,
    expected_architecture: str | None = None,
    dispatch_architecture: bool = True,
) -> str | _GGUFPreflightRevision | _GGUFPreflightFallbackRevision:
    """Validate the exact selected Hub file header and return its immutable revision."""
    source = f"{repo_id}@{revision}:{filename}"
    url = hf_hub_url(repo_id, filename, revision=revision)
    try:
        metadata = get_hf_file_metadata(url)
    except _HUB_PREFLIGHT_TRANSPORT_ERRORS as error:
        raise RuntimeError(
            f"Cannot resolve the exact selected GGUF file {source!r} to an immutable "
            "revision; refusing repository-level metadata or mutable-revision fallback"
        ) from error
    commit_hash = metadata.commit_hash
    if (
        not isinstance(commit_hash, str)
        or re.fullmatch(r"[0-9a-fA-F]{40}", commit_hash) is None
    ):
        raise ValueError(
            f"Hub did not resolve a 40-character immutable commit SHA for {source!r}."
        )

    headers = build_hf_headers()
    if urlparse(url).netloc != urlparse(metadata.location).netloc:
        headers.pop("authorization", None)
    headers["Range"] = f"bytes=0-{_GGUF_HEADER_RANGE_BYTES - 1}"

    def read_response(response) -> list[bytes]:
        response.raise_for_status()
        chunks: list[bytes] = []
        size = 0
        chunk_iterator = (
            response.iter_bytes()
            if hasattr(response, "iter_bytes")
            else response.iter_content(chunk_size=64 * 1024)
        )
        for chunk in chunk_iterator:
            remaining = _GGUF_HEADER_RANGE_BYTES - size
            if remaining <= 0:
                break
            chunks.append(chunk[:remaining])
            size += min(len(chunk), remaining)
            if size == _GGUF_HEADER_RANGE_BYTES:
                break
        return chunks

    try:
        session = get_session()
        stream = getattr(session, "stream", None)
        if callable(stream):
            with stream("GET", metadata.location, headers=headers) as response:
                chunks = read_response(response)
        else:
            with session.get(metadata.location, headers=headers, stream=True) as response:
                chunks = read_response(response)
    except _HUB_PREFLIGHT_TRANSPORT_ERRORS as error:
        logger.warning(
            "Bounded GGUF header range read failed for %s (%s); downloading the "
            "same immutable revision and validating its local header before dispatch.",
            source,
            error,
        )
        return _GGUFPreflightFallbackRevision(commit_hash)
    try:
        header_info = _gguf_header_info_from_header_prefix(
            b"".join(chunks),
            source=source,
        )
    except GGUFHeaderTruncatedError:
        logger.warning(
            "The selected GGUF metadata header for %s exceeds the bounded range; "
            "downloading the same immutable revision for full local validation.",
            source,
        )
        return _GGUFPreflightFallbackRevision(commit_hash)
    _validate_preflight_split_header(header_info, source=source)
    architecture = header_info.architecture
    if (
        dispatch_architecture
        and expected_architecture is not None
        and architecture != expected_architecture
    ):
        raise ValueError(
            f"Expected a {expected_architecture!r} mmproj GGUF for {source!r}, "
            f"got architecture {architecture!r}. No payload was downloaded."
        )
    if architecture is None and dispatch_architecture:
        if (
            header_info.split_count is None
            or header_info.split_count <= 1
            or header_info.split_no in {None, 0}
        ):
            raise ValueError(
                f"Selected GGUF header {source!r} has no general.architecture and "
                "is not a continuation shard in a complete split set. "
                "No payload was downloaded."
            )
    elif architecture is not None and dispatch_architecture:
        _raise_for_unsupported_gguf_architecture(
            architecture,
            source=source,
            allow_mmproj_companion=allow_mmproj_companion,
            allow_preflight_only=True,
        )
    if architecture == "qwen4exp" and dispatch_architecture:
        from mobius.integrations.gguf._qwen4_exp import reject_qwen4exp_payload

        reject_qwen4exp_payload()
    return _GGUFPreflightRevision(commit_hash, header_info)


def _preflight_hf_mmproj_companion_file(
    repo_id: str,
    filename: str,
    *,
    revision: str = "main",
) -> str | _GGUFPreflightRevision | _GGUFPreflightFallbackRevision:
    """Validate one exact Hub mmproj file and pin its immutable revision."""
    return _preflight_hf_gguf_file(
        repo_id,
        filename,
        revision=revision,
        allow_mmproj_companion=True,
        expected_architecture=MMPROJ_ARCHITECTURE,
    )


def _validate_gguf_model(
    gguf_model,
    *,
    source: str,
    allow_mmproj_companion: bool = False,
    keep_quantized: bool | None = None,
) -> None:
    """Validate a parsed GGUF before config extraction or graph construction."""
    from mobius.integrations.gguf._shard_set import GgufShardSet

    # A GgufShardSet has already assembled and structurally validated the whole
    # split set, so the single-file "sharded input is unsupported" guard must
    # not fire for it. Plain single-file GGUFModels still reject a lone shard.
    if not isinstance(gguf_model, GgufShardSet):
        split_count = int(gguf_model.get_metadata("split.count", 1))
        _raise_for_sharded_gguf(source=source, split_count=split_count)
    from mobius.integrations.gguf._qwen4_exp import validate_qwen4exp_tensor_contract

    validate_qwen4exp_tensor_contract(
        gguf_model,
        source=source,
        keep_quantized=keep_quantized,
    )
    _raise_for_unsupported_gguf_architecture(
        gguf_model.architecture,
        source=source,
        tensor_names=gguf_model.tensor_names,
        allow_mmproj_companion=allow_mmproj_companion,
    )
    _raise_for_invalid_bitnet_tensor_contract(gguf_model)
    _raise_for_invalid_talkie_tensor_contract(gguf_model)
    from mobius.integrations.gguf._mtp import validate_mtp_tensor_contract

    validate_mtp_tensor_contract(gguf_model)
    _raise_for_unsupported_auxiliary_quantization(gguf_model)
    _raise_for_invalid_falcon_h1_tensor_contract(gguf_model)
    _raise_for_invalid_plamo_tensor_contract(
        gguf_model,
        keep_quantized=keep_quantized,
    )
    _raise_for_invalid_plamo2_tensor_contract(gguf_model)
    _raise_for_invalid_minimax_tensor_contract(gguf_model)
    _raise_for_invalid_kimi_k3_tensor_contract(gguf_model)
    _raise_for_invalid_kimi_linear_tensor_contract(gguf_model)
    _raise_for_invalid_hybrid_tensor_contract(gguf_model)
    _raise_for_invalid_t5_tensor_contract(gguf_model)
    _raise_for_malformed_recurrent_tensors(gguf_model)
    _raise_for_unsupported_encoder_heads(gguf_model)
    _raise_for_invalid_encoder_tensor_contract(gguf_model)
    _raise_for_invalid_specialized_encoder_tensor_contract(gguf_model)
    _raise_for_invalid_minicpm_tensor_contract(gguf_model)
    _raise_for_invalid_embedding_tensor_contract(gguf_model)
    _raise_for_invalid_dense_c01_tensor_contract(gguf_model)
    _raise_for_invalid_conventional_decoder_tensor_contract(gguf_model)
    _raise_for_invalid_maincoder_tensor_contract(gguf_model)
    _raise_for_invalid_granite_tensor_contract(gguf_model)
    _raise_for_invalid_smallthinker_tensor_contract(gguf_model)
    _raise_for_invalid_conventional_moe_tensor_contract(gguf_model)
    _raise_for_invalid_moe_cohort_tensor_contract(gguf_model)
    from mobius.integrations.gguf._draft import validate_draft_tensor_contract

    validate_draft_tensor_contract(gguf_model)
    if not allow_mmproj_companion:
        from mobius.integrations.gguf._tokenizer import inspect_gguf_tokenizer

        # Validate tokenizer identity and tables before config extraction. A
        # known-but-deferred tokenizer is allowed for graph-only imports; an
        # unknown or contradictory tokenizer is not.
        inspect_gguf_tokenizer(gguf_model.metadata, source=source)


def _raise_for_invalid_bitnet_tensor_contract(gguf_model) -> None:
    """Validate the exact pinned BitNet metadata and tensor closure."""
    if gguf_model.architecture != "bitnet":
        return

    architecture = "bitnet"
    metadata = gguf_model.metadata
    required_geometry = (
        "context_length",
        "embedding_length",
        "feed_forward_length",
        "block_count",
        "attention.head_count",
        "attention.head_count_kv",
        "attention.layer_norm_rms_epsilon",
        "rope.freq_base",
        "rope.dimension_count",
    )
    missing_geometry = [
        f"{architecture}.{suffix}"
        for suffix in required_geometry
        if f"{architecture}.{suffix}" not in metadata
    ]
    if missing_geometry:
        raise ValueError(
            f"BitNet GGUF is missing required architecture metadata: {missing_geometry}"
        )

    context = int(metadata["bitnet.context_length"])
    hidden = int(metadata["bitnet.embedding_length"])
    intermediate = int(metadata["bitnet.feed_forward_length"])
    layers = int(metadata["bitnet.block_count"])
    heads = int(metadata["bitnet.attention.head_count"])
    kv_heads = int(metadata["bitnet.attention.head_count_kv"])
    rope_dim = int(metadata["bitnet.rope.dimension_count"])
    norm_eps = float(metadata["bitnet.attention.layer_norm_rms_epsilon"])
    rope_base = float(metadata["bitnet.rope.freq_base"])
    vocab = int(metadata.get("bitnet.vocab_size", 0))
    if not vocab:
        vocab = len(metadata.get("tokenizer.ggml.tokens", ()))

    if (
        min(context, hidden, intermediate, layers, heads, kv_heads, rope_dim, vocab) <= 0
        or hidden % heads
        or heads % kv_heads
        or not math.isfinite(norm_eps)
        or norm_eps <= 0
        or not math.isfinite(rope_base)
        or rope_base <= 0
    ):
        raise ValueError("BitNet GGUF has inconsistent architecture geometry or metadata")

    head_dim = hidden // heads
    key_dim = int(metadata.get("bitnet.attention.key_length", head_dim))
    value_dim = int(metadata.get("bitnet.attention.value_length", head_dim))
    if key_dim != head_dim or value_dim != head_dim or rope_dim != head_dim or rope_dim % 2:
        raise ValueError(
            "BitNet GGUF requires equal full attention/RoPE head widths: "
            f"embedding_length/head_count={head_dim}, key_length={key_dim}, "
            f"value_length={value_dim}, rope.dimension_count={rope_dim}"
        )

    required: dict[str, tuple[int, ...]] = {
        "token_embd.weight": (vocab, hidden),
        "output_norm.weight": (hidden,),
    }
    optional: dict[str, tuple[int, ...]] = {}
    q_width = heads * head_dim
    kv_width = kv_heads * head_dim
    for layer in range(layers):
        prefix = f"blk.{layer}."
        required.update(
            {
                prefix + "attn_norm.weight": (hidden,),
                prefix + "attn_sub_norm.weight": (hidden,),
                prefix + "attn_q.weight": (q_width, hidden),
                prefix + "attn_k.weight": (kv_width, hidden),
                prefix + "attn_v.weight": (kv_width, hidden),
                prefix + "attn_output.weight": (hidden, q_width),
                prefix + "ffn_norm.weight": (hidden,),
                prefix + "ffn_sub_norm.weight": (intermediate,),
                prefix + "ffn_gate.weight": (intermediate, hidden),
                prefix + "ffn_up.weight": (intermediate, hidden),
                prefix + "ffn_down.weight": (hidden, intermediate),
            }
        )
        for projection in (
            "attn_q",
            "attn_k",
            "attn_v",
            "attn_output",
            "ffn_gate",
            "ffn_up",
            "ffn_down",
        ):
            optional[prefix + projection + ".scale"] = (1,)

    # The pinned graph has no independent output tensor: token_embd.weight is
    # shared by input lookup and the final projection.
    actual = {
        name: tuple(int(dimension) for dimension in shape)
        for name, _raw, _qtype, shape in gguf_model.tensor_items_raw()
    }
    allowed = set(required) | set(optional)
    missing = sorted(set(required) - set(actual))
    unexpected = sorted(set(actual) - allowed)
    expected = {**required, **optional}
    malformed = {
        name: (expected[name], actual[name])
        for name in set(actual) & allowed
        if actual[name] != expected[name]
    }
    if missing or unexpected or malformed:
        raise ValueError(
            "Invalid BitNet GGUF tensor closure (the output projection must be tied to "
            "token_embd.weight): "
            f"missing={missing}, unexpected={unexpected}, malformed={malformed}"
        )


def _raise_for_invalid_talkie_tensor_contract(gguf_model) -> None:
    """Require the exact pinned scalar-sidecar Talkie graph and tensor closure."""
    if gguf_model.architecture != "talkie":
        return

    arch = "talkie"
    metadata = gguf_model.metadata
    required_metadata = (
        "context_length",
        "embedding_length",
        "feed_forward_length",
        "block_count",
        "attention.head_count",
        "attention.head_count_kv",
        "attention.layer_norm_rms_epsilon",
        "rope.freq_base",
        "rope.dimension_count",
        "logit_scale",
    )
    missing_metadata = [
        f"{arch}.{suffix}"
        for suffix in required_metadata
        if f"{arch}.{suffix}" not in metadata
    ]
    if missing_metadata:
        raise ValueError(f"Talkie GGUF is missing required metadata: {missing_metadata}")

    hidden = int(metadata[f"{arch}.embedding_length"])
    intermediate = int(metadata[f"{arch}.feed_forward_length"])
    layers = int(metadata[f"{arch}.block_count"])
    heads = int(metadata[f"{arch}.attention.head_count"])
    kv_heads = int(metadata[f"{arch}.attention.head_count_kv"])
    rope_dim = int(metadata[f"{arch}.rope.dimension_count"])
    context = int(metadata[f"{arch}.context_length"])
    eps = float(metadata[f"{arch}.attention.layer_norm_rms_epsilon"])
    rope_base = float(metadata[f"{arch}.rope.freq_base"])
    logit_scale = float(metadata[f"{arch}.logit_scale"])
    vocab = int(metadata.get(f"{arch}.vocab_size", 0))
    if not vocab:
        vocab = len(metadata.get("tokenizer.ggml.tokens", ()))
    if (
        min(hidden, intermediate, layers, heads, kv_heads, rope_dim, context, vocab) <= 0
        or hidden % heads
        or heads != kv_heads
        or rope_dim != hidden // heads
        or rope_dim % 2
        or not all(math.isfinite(value) for value in (eps, rope_base, logit_scale))
        or eps <= 0
        or rope_base <= 0
    ):
        raise ValueError("Talkie GGUF has inconsistent geometry or non-finite metadata")
    if metadata.get(f"{arch}.rope.scaling.type") not in {None, "", "none"}:
        raise ValueError("Talkie requires unscaled full-head NeoX RoPE")

    required: dict[str, tuple[int, ...]] = {
        "token_embd.weight": (vocab, hidden),
        "output.weight": (vocab, hidden),
    }
    for layer in range(layers):
        prefix = f"blk.{layer}."
        required.update(
            {
                prefix + "attn_q.weight": (hidden, hidden),
                prefix + "attn_k.weight": (hidden, hidden),
                prefix + "attn_v.weight": (hidden, hidden),
                prefix + "attn_output.weight": (hidden, hidden),
                prefix + "attn_q_norm.weight": (heads, 1),
                prefix + "ffn_gate.weight": (intermediate, hidden),
                prefix + "ffn_up.weight": (intermediate, hidden),
                prefix + "ffn_down.weight": (hidden, intermediate),
                prefix + "layer_output_scale.weight": (1,),
            }
        )

    actual = {
        name: tuple(int(dim) for dim in shape)
        for name, _raw, _qtype, shape in gguf_model.tensor_items_raw()
    }
    missing = sorted(set(required) - set(actual))
    unexpected = sorted(set(actual) - set(required))
    malformed = {
        name: (required[name], actual[name])
        for name in set(required) & set(actual)
        if actual[name] != required[name]
    }
    if missing or unexpected or malformed:
        raise ValueError(
            f"Invalid Talkie tensor closure: missing={missing}, "
            f"unexpected={unexpected}, malformed={malformed}"
        )


def _raise_for_invalid_minimax_tensor_contract(gguf_model) -> None:
    """Validate MiniMax-01 metadata, per-layer families, and exact tensor shapes."""
    if gguf_model.architecture != "minimax-01":
        return

    from mobius.integrations.gguf._config_mapping import _derive_hybrid_layout

    metadata = gguf_model.metadata
    layers, layer_types, mtp_count = _derive_hybrid_layout(
        "minimax-01", metadata, gguf_model.tensor_names
    )
    assert layer_types is not None
    if mtp_count:
        raise ValueError("MiniMax-01 GGUF does not support appended MTP blocks")

    hidden = int(metadata["minimax-01.embedding_length"])
    intermediate = int(metadata["minimax-01.feed_forward_length"])
    heads = int(metadata["minimax-01.attention.head_count"])
    kv_heads = int(metadata["minimax-01.attention.head_count_kv"])
    head_dim = int(metadata["minimax-01.attention.key_length"])
    value_dim = int(metadata["minimax-01.attention.value_length"])
    rope_dim = int(metadata["minimax-01.rope.dimension_count"])
    experts = int(metadata["minimax-01.expert_count"])
    top_k = int(metadata["minimax-01.expert_used_count"])
    residual_scale = float(metadata["minimax-01.residual_scale"])
    norm_eps = float(metadata["minimax-01.attention.layer_norm_rms_epsilon"])
    rope_freq_base = float(metadata["minimax-01.rope.freq_base"])
    vocab = int(metadata.get("minimax-01.vocab_size", 0))
    if not vocab:
        vocab = len(metadata.get("tokenizer.ggml.tokens", ()))
    if (
        min(hidden, intermediate, heads, kv_heads, head_dim, experts, top_k, vocab) <= 0
        or heads % kv_heads
        or value_dim != head_dim
        or rope_dim <= 0
        or rope_dim > head_dim
        or rope_dim % 2
        or experts <= 1
        or top_k > experts
        or not math.isfinite(residual_scale)
        or residual_scale <= 0
        or not math.isfinite(norm_eps)
        or norm_eps <= 0
        or not math.isfinite(rope_freq_base)
        or rope_freq_base <= 0
    ):
        raise ValueError("MiniMax-01 GGUF has inconsistent architecture metadata")
    if any(
        key in metadata
        for key in (
            "minimax-01.expert_shared_count",
            "minimax-01.expert_shared_feed_forward_length",
        )
    ):
        raise ValueError("MiniMax-01 pinned GGUF does not support shared experts")

    q_width = heads * head_dim
    kv_width = kv_heads * head_dim
    required: dict[str, tuple[int, ...]] = {
        "token_embd.weight": (vocab, hidden),
        "output_norm.weight": (hidden,),
    }
    optional: dict[str, tuple[int, ...]] = {"output.weight": (vocab, hidden)}
    for layer, layer_type in enumerate(layer_types):
        prefix = f"blk.{layer}."
        required.update(
            {
                prefix + "attn_norm.weight": (hidden,),
                prefix + "attn_output.weight": (hidden, q_width),
                prefix + "ffn_norm.weight": (hidden,),
                prefix + "ffn_gate_inp.weight": (experts, hidden),
                prefix + "ffn_gate_exps.weight": (experts, intermediate, hidden),
                prefix + "ffn_up_exps.weight": (experts, intermediate, hidden),
                prefix + "ffn_down_exps.weight": (experts, hidden, intermediate),
            }
        )
        if layer_type == "lightning_attention":
            required.update(
                {
                    prefix + "attn_qkv.weight": (3 * q_width, hidden),
                    prefix + "attn_gate.weight": (q_width, hidden),
                    prefix + "attn_norm_2.weight": (q_width,),
                }
            )
        else:
            required.update(
                {
                    prefix + "attn_q.weight": (q_width, hidden),
                    prefix + "attn_k.weight": (kv_width, hidden),
                    prefix + "attn_v.weight": (kv_width, hidden),
                }
            )

    actual = set(gguf_model.tensor_names)
    allowed = set(required) | set(optional)
    out_of_range = sorted(
        name
        for name in actual
        if (match := re.match(r"^blk\.(\d+)\.", name)) and int(match.group(1)) >= layers
    )
    missing = sorted(set(required) - actual)
    unexpected = sorted(actual - allowed)
    if missing or unexpected or out_of_range:
        raise ValueError(
            "Invalid MiniMax-01 GGUF tensor closure: "
            f"missing={missing}, unexpected={unexpected}, out_of_range={out_of_range}"
        )
    if not hasattr(gguf_model, "tensor_items_raw"):
        return
    shapes = {
        name: tuple(int(dimension) for dimension in shape)
        for name, _raw, _qtype, shape in gguf_model.tensor_items_raw()
    }
    malformed = {
        name: (shape, shapes.get(name))
        for name, shape in {**required, **optional}.items()
        if name in shapes and shapes[name] != shape
    }
    if malformed:
        raise ValueError(f"MiniMax-01 GGUF has invalid tensor shape(s): {malformed}")


def _raise_for_invalid_minicpm_tensor_contract(gguf_model) -> None:
    """Validate the exact dense MiniCPM/MiniCPM3 loader closure and geometry."""
    architecture = gguf_model.architecture
    if architecture not in {"minicpm", "minicpm3"}:
        return

    metadata = gguf_model.metadata
    required_geometry = (
        "context_length",
        "embedding_length",
        "feed_forward_length",
        "block_count",
        "attention.head_count",
    )
    missing_geometry = [
        f"{architecture}.{suffix}"
        for suffix in required_geometry
        if f"{architecture}.{suffix}" not in metadata
    ]
    if missing_geometry:
        raise ValueError(
            f"{architecture} GGUF is missing required geometry: {missing_geometry}"
        )

    hidden = int(metadata[f"{architecture}.embedding_length"])
    intermediate = int(metadata[f"{architecture}.feed_forward_length"])
    layers = int(metadata[f"{architecture}.block_count"])
    heads = int(metadata[f"{architecture}.attention.head_count"])
    kv_heads = int(metadata.get(f"{architecture}.attention.head_count_kv", heads))
    context = int(metadata[f"{architecture}.context_length"])
    vocab = int(metadata.get(f"{architecture}.vocab_size", 0)) or len(
        metadata.get("tokenizer.ggml.tokens", ())
    )
    if min(hidden, intermediate, layers, heads, kv_heads, context, vocab) <= 0:
        raise ValueError(f"{architecture} GGUF has non-positive model geometry")
    if hidden % heads or heads % kv_heads:
        raise ValueError(f"{architecture} GGUF has invalid attention head geometry")
    if int(metadata.get(f"{architecture}.expert_count", 0)):
        raise ValueError(
            f"{architecture} routed-expert GGUF is outside the exact dense graph subset"
        )
    if architecture == "minicpm3":
        mla_metadata = (
            "attention.key_length",
            "attention.q_lora_rank",
            "attention.kv_lora_rank",
            "rope.dimension_count",
        )
        missing_mla = [
            f"{architecture}.{suffix}"
            for suffix in mla_metadata
            if f"{architecture}.{suffix}" not in metadata
        ]
        if missing_mla:
            raise ValueError(f"minicpm3 GGUF is missing required MLA geometry: {missing_mla}")

    actual = {
        name: tuple(int(dimension) for dimension in shape)
        for name, _raw, _qtype, shape in gguf_model.tensor_items_raw()
    }
    required: dict[str, tuple[int, ...]] = {
        "token_embd.weight": (vocab, hidden),
        "output_norm.weight": (hidden,),
    }
    optional: dict[str, tuple[int, ...]] = {"output.weight": (vocab, hidden)}

    if architecture == "minicpm":
        head_dim = hidden // heads
        kv_dim = kv_heads * head_dim
        rope_dim = int(metadata.get("minicpm.rope.dimension_count", head_dim))
        if rope_dim <= 0 or rope_dim > head_dim or rope_dim % 2:
            raise ValueError("minicpm GGUF has invalid rotary dimension")
        if "rope_freqs.weight" in actual:
            raise ValueError(
                "MiniCPM GGUF with serialized rope_freqs.weight is unsupported: "
                "the exact per-dimension frequency factors are not representable "
                "by the current rotary graph"
            )
        for rope_name in (
            "rope_factors_long.weight",
            "rope_factors_short.weight",
        ):
            optional[rope_name] = (rope_dim // 2,)
        for layer in range(layers):
            prefix = f"blk.{layer}."
            if prefix + "attn_qkv.weight" in actual:
                raise ValueError(
                    "MiniCPM fused QKV is unsupported because its Q and K rows require "
                    "different exact permutations; split Q/K/V tensors are required"
                )
            required.update(
                {
                    prefix + "attn_norm.weight": (hidden,),
                    prefix + "attn_q.weight": (hidden, hidden),
                    prefix + "attn_k.weight": (kv_dim, hidden),
                    prefix + "attn_v.weight": (kv_dim, hidden),
                    prefix + "attn_output.weight": (hidden, hidden),
                    prefix + "ffn_norm.weight": (hidden,),
                    prefix + "ffn_gate.weight": (intermediate, hidden),
                    prefix + "ffn_up.weight": (intermediate, hidden),
                    prefix + "ffn_down.weight": (hidden, intermediate),
                }
            )
            optional.update(
                {
                    prefix + "attn_q.bias": (hidden,),
                    prefix + "attn_k.bias": (kv_dim,),
                    prefix + "attn_v.bias": (kv_dim,),
                    prefix + "attn_output.bias": (hidden,),
                    prefix + "ffn_gate.bias": (intermediate,),
                    prefix + "ffn_up.bias": (intermediate,),
                    prefix + "ffn_down.bias": (hidden,),
                }
            )
    else:
        if kv_heads != heads:
            raise ValueError("minicpm3 GGUF requires one expanded K/V head per query head")
        qk_dim = int(metadata["minicpm3.attention.key_length"])
        rope_dim = int(metadata["minicpm3.rope.dimension_count"])
        q_rank = int(metadata["minicpm3.attention.q_lora_rank"])
        kv_rank = int(metadata["minicpm3.attention.kv_lora_rank"])
        value_dim = hidden // heads
        nope_dim = qk_dim - rope_dim
        if min(q_rank, kv_rank, value_dim, nope_dim, rope_dim) <= 0 or rope_dim % 2:
            raise ValueError("minicpm3 GGUF has invalid MLA geometry")
        optional.update(
            {
                "rope_factors_long.weight": (rope_dim // 2,),
                "rope_factors_short.weight": (rope_dim // 2,),
            }
        )
        for layer in range(layers):
            prefix = f"blk.{layer}."
            required.update(
                {
                    prefix + "attn_norm.weight": (hidden,),
                    prefix + "attn_q_a.weight": (q_rank, hidden),
                    prefix + "attn_q_a_norm.weight": (q_rank,),
                    prefix + "attn_q_b.weight": (heads * qk_dim, q_rank),
                    prefix + "attn_kv_a_mqa.weight": (
                        kv_rank + rope_dim,
                        hidden,
                    ),
                    prefix + "attn_kv_a_norm.weight": (kv_rank,),
                    prefix + "attn_kv_b.weight": (
                        heads * (nope_dim + value_dim),
                        kv_rank,
                    ),
                    prefix + "attn_output.weight": (hidden, heads * value_dim),
                    prefix + "ffn_norm.weight": (hidden,),
                    prefix + "ffn_gate.weight": (intermediate, hidden),
                    prefix + "ffn_up.weight": (intermediate, hidden),
                    prefix + "ffn_down.weight": (hidden, intermediate),
                }
            )

    missing = sorted(set(required) - set(actual))
    unexpected = sorted(set(actual) - set(required) - set(optional))
    malformed = {
        name: (required.get(name, optional.get(name)), actual[name])
        for name in set(actual) & (set(required) | set(optional))
        if actual[name] != required.get(name, optional.get(name))
    }
    if missing or unexpected or malformed:
        raise ValueError(
            f"Invalid {architecture} GGUF tensor closure: missing={missing}, "
            f"unexpected={unexpected}, malformed={malformed}"
        )


def _raise_for_invalid_kimi_linear_tensor_contract(gguf_model) -> None:
    """Validate Kimi Linear's pinned metadata and exact heterogeneous closure."""
    if gguf_model.architecture != "kimi-linear":
        return

    import numpy as np

    from mobius.integrations.gguf._config_mapping import _derive_hybrid_layout
    from mobius.integrations.gguf._quant_registry import float_storage_type_ids

    if int(gguf_model.format_version) != 3:
        raise ValueError(
            f"Kimi Linear supports only pinned GGUF v3, got v{gguf_model.format_version}"
        )
    metadata = gguf_model.metadata
    arch = "kimi-linear"
    layers, layer_types, mtp_count = _derive_hybrid_layout(
        arch, metadata, gguf_model.tensor_names
    )
    assert layer_types is not None
    if mtp_count:
        raise ValueError("Kimi Linear GGUF does not support appended NextN blocks")

    hidden = int(metadata[f"{arch}.embedding_length"])
    dense_intermediate = int(metadata[f"{arch}.feed_forward_length"])
    heads = int(metadata[f"{arch}.attention.head_count"])
    qk_dim = int(metadata[f"{arch}.attention.key_length_mla"])
    value_dim = int(metadata[f"{arch}.attention.value_length_mla"])
    kv_rank = int(metadata[f"{arch}.attention.kv_lora_rank"])
    extra_dim = int(metadata[f"{arch}.rope.dimension_count"])
    kda_dim = int(metadata[f"{arch}.kda.head_dim"])
    conv = int(metadata[f"{arch}.ssm.conv_kernel"])
    experts = int(metadata[f"{arch}.expert_count"])
    top_k = int(metadata[f"{arch}.expert_used_count"])
    expert_intermediate = int(metadata[f"{arch}.expert_feed_forward_length"])
    shared = int(metadata[f"{arch}.expert_shared_count"])
    dense_layers = int(metadata[f"{arch}.leading_dense_block_count"])
    routed_scale = float(metadata[f"{arch}.expert_weights_scale"])
    epsilon = float(metadata[f"{arch}.attention.layer_norm_rms_epsilon"])
    if (
        min(
            layers,
            hidden,
            dense_intermediate,
            heads,
            qk_dim,
            value_dim,
            kv_rank,
            extra_dim,
            kda_dim,
            conv,
            experts,
            top_k,
            expert_intermediate,
        )
        <= 0
        or conv < 2
        or qk_dim <= extra_dim
        or top_k > experts
        or shared != 1
        or dense_layers != 1
        or not math.isfinite(routed_scale)
        or routed_scale <= 0
        or not math.isfinite(epsilon)
        or epsilon <= 0
    ):
        raise ValueError("Kimi Linear GGUF has inconsistent pinned architecture metadata")
    for key in ("expert_group_count", "expert_group_used_count"):
        if f"{arch}.{key}" in metadata and int(metadata[f"{arch}.{key}"]) != 1:
            raise ValueError(f"Kimi Linear requires {arch}.{key}=1")

    kv_counts = metadata[f"{arch}.attention.head_count_kv"]
    if not isinstance(kv_counts, (list, tuple, np.ndarray)):
        raise TypeError("kimi-linear.attention.head_count_kv must be an exact per-layer array")
    kv_counts = [int(value) for value in kv_counts]
    if len(kv_counts) != layers or any(value not in {0, 1} for value in kv_counts):
        raise ValueError(
            "Kimi Linear per-layer KV-head counts must contain block_count entries of 0 or 1"
        )
    if all(value == 0 for value in kv_counts) or all(value == 1 for value in kv_counts):
        raise ValueError("Kimi Linear requires both KDA and MLA layers")

    actual_shapes = {
        name: tuple(int(dim) for dim in gguf_model.get_tensor_shape(name))
        for name in gguf_model.tensor_names
    }
    vocab = int(
        metadata.get(
            f"{arch}.vocab_size",
            actual_shapes.get("token_embd.weight", (0,))[0],
        )
    )
    required: dict[str, tuple[int, ...]] = {
        "token_embd.weight": (vocab, hidden),
        "output_norm.weight": (hidden,),
        "output.weight": (vocab, hidden),
    }
    projection_width = heads * kda_dim
    nope_dim = qk_dim - extra_dim
    for layer, layer_type in enumerate(layer_types):
        prefix = f"blk.{layer}."
        required.update(
            {
                prefix + "attn_norm.weight": (hidden,),
                prefix + "ffn_norm.weight": (hidden,),
            }
        )
        if layer_type == "kimi_linear_attention":
            required.update(
                {
                    prefix + "attn_q.weight": (projection_width, hidden),
                    prefix + "attn_k.weight": (projection_width, hidden),
                    prefix + "attn_v.weight": (projection_width, hidden),
                    prefix + "ssm_conv1d_q.weight": (1, projection_width, 1, conv),
                    prefix + "ssm_conv1d_k.weight": (1, projection_width, 1, conv),
                    prefix + "ssm_conv1d_v.weight": (1, projection_width, 1, conv),
                    prefix + "ssm_f_a.weight": (kda_dim, hidden),
                    prefix + "ssm_f_b.weight": (projection_width, kda_dim),
                    prefix + "ssm_beta.weight": (heads, hidden),
                    prefix + "ssm_a": (1, 1, heads, 1),
                    prefix + "ssm_dt.bias": (projection_width,),
                    prefix + "ssm_g_a.weight": (kda_dim, hidden),
                    prefix + "ssm_g_b.weight": (projection_width, kda_dim),
                    prefix + "ssm_norm.weight": (kda_dim,),
                    prefix + "attn_output.weight": (hidden, projection_width),
                }
            )
        else:
            required.update(
                {
                    prefix + "attn_q.weight": (heads * qk_dim, hidden),
                    prefix + "attn_kv_a_mqa.weight": (kv_rank + extra_dim, hidden),
                    prefix + "attn_kv_a_norm.weight": (kv_rank,),
                    prefix + "attn_k_b.weight": (heads, kv_rank, nope_dim),
                    prefix + "attn_v_b.weight": (heads, value_dim, kv_rank),
                    prefix + "attn_output.weight": (hidden, heads * value_dim),
                }
            )
        if layer < dense_layers:
            required.update(
                {
                    prefix + "ffn_gate.weight": (dense_intermediate, hidden),
                    prefix + "ffn_up.weight": (dense_intermediate, hidden),
                    prefix + "ffn_down.weight": (hidden, dense_intermediate),
                }
            )
        else:
            shared_width = shared * expert_intermediate
            required.update(
                {
                    prefix + "ffn_gate_inp.weight": (experts, hidden),
                    prefix + "exp_probs_b.bias": (experts,),
                    prefix + "ffn_gate_exps.weight": (
                        experts,
                        expert_intermediate,
                        hidden,
                    ),
                    prefix + "ffn_up_exps.weight": (
                        experts,
                        expert_intermediate,
                        hidden,
                    ),
                    prefix + "ffn_down_exps.weight": (
                        experts,
                        hidden,
                        expert_intermediate,
                    ),
                    prefix + "ffn_gate_shexp.weight": (shared_width, hidden),
                    prefix + "ffn_up_shexp.weight": (shared_width, hidden),
                    prefix + "ffn_down_shexp.weight": (hidden, shared_width),
                }
            )

    actual = set(gguf_model.tensor_names)
    out_of_range = sorted(
        name
        for name in actual
        if (match := re.match(r"^blk\.(\d+)\.", name)) and int(match.group(1)) >= layers
    )
    missing = sorted(set(required) - actual)
    unexpected = sorted(actual - set(required))
    if missing or unexpected or out_of_range:
        raise ValueError(
            "Invalid Kimi Linear GGUF tensor closure: "
            f"missing={missing}, unexpected={unexpected}, out_of_range={out_of_range}"
        )
    malformed = {
        name: (shape, actual_shapes.get(name))
        for name, shape in required.items()
        if actual_shapes.get(name) != shape
    }
    if malformed:
        raise ValueError(f"Kimi Linear GGUF has invalid tensor shape(s): {malformed}")

    float_types = float_storage_type_ids()
    non_matmul = {
        name
        for name in required
        if name.endswith(("_norm.weight", "ssm_a", "ssm_dt.bias", "exp_probs_b.bias"))
        or "conv1d" in name
    }
    invalid_storage = []
    for name, _raw, qtype, _shape in gguf_model.tensor_items_raw():
        qtype_id = getattr(qtype, "value", qtype)
        if name in non_matmul and qtype_id not in float_types:
            invalid_storage.append(name)
    if invalid_storage:
        raise ValueError(
            "Kimi Linear recurrent, norm, and routing auxiliary tensors must remain float: "
            f"{sorted(invalid_storage)}"
        )

    for layer, layer_type in enumerate(layer_types):
        if layer_type != "kimi_linear_attention":
            continue
        name = f"blk.{layer}.ssm_a"
        decay = np.asarray(gguf_model.get_tensor(name))
        if not np.all(np.isfinite(decay)) or not np.all(decay < 0):
            raise ValueError(
                f"Malformed Kimi Linear decay tensor {name!r}: expected finite -exp(A_log)"
            )


def _raise_for_invalid_kimi_k3_tensor_contract(gguf_model) -> None:
    """Validate Kimi-K3 metadata, layer families, storage, and exact tensor closure."""
    if gguf_model.architecture != "kimi-k3":
        return

    import numpy as np

    from mobius.integrations.gguf._config_mapping import _derive_hybrid_layout
    from mobius.integrations.gguf._quant_registry import float_storage_type_ids

    if int(gguf_model.format_version) != 3:
        raise ValueError(
            f"Kimi-K3 supports only pinned GGUF v3, got v{gguf_model.format_version}"
        )
    metadata = gguf_model.metadata
    arch = "kimi-k3"
    layers, layer_types, mtp_count = _derive_hybrid_layout(
        arch, metadata, gguf_model.tensor_names
    )
    assert layer_types is not None
    if mtp_count:
        raise ValueError("Kimi-K3 GGUF does not support appended NextN blocks")

    hidden = int(metadata[f"{arch}.embedding_length"])
    dense_intermediate = int(metadata[f"{arch}.feed_forward_length"])
    heads = int(metadata[f"{arch}.attention.head_count"])
    q_rank = int(metadata[f"{arch}.attention.q_lora_rank"])
    qk_dim = int(metadata[f"{arch}.attention.key_length_mla"])
    value_dim = int(metadata[f"{arch}.attention.value_length_mla"])
    kv_rank = int(metadata[f"{arch}.attention.kv_lora_rank"])
    extra_dim = int(metadata[f"{arch}.rope.dimension_count"])
    kda_dim = int(metadata[f"{arch}.kda.head_dim"])
    conv = int(metadata[f"{arch}.ssm.conv_kernel"])
    experts = int(metadata[f"{arch}.expert_count"])
    top_k = int(metadata[f"{arch}.expert_used_count"])
    expert_intermediate = int(metadata[f"{arch}.expert_feed_forward_length"])
    shared = int(metadata.get(f"{arch}.expert_shared_count", 0))
    dense_layers = int(metadata.get(f"{arch}.leading_dense_block_count", 0))
    routed_scale = float(metadata.get(f"{arch}.expert_weights_scale", 0.0))
    expert_latent = int(metadata.get(f"{arch}.expert_latent_length", hidden))
    epsilon = float(metadata[f"{arch}.attention.layer_norm_rms_epsilon"])
    gate_lower_bound = float(metadata.get(f"{arch}.kda.gate_lower_bound", -math.inf))
    attn_res_block = int(metadata.get(f"{arch}.attn_res.block_size", 0))
    situ_beta = float(metadata.get(f"{arch}.activation.situ_beta", 1.0))
    situ_linear_beta = float(metadata.get(f"{arch}.activation.situ_linear_beta", 0.0))
    gating = int(metadata.get(f"{arch}.expert_gating_func", 0))
    if (
        min(
            layers,
            hidden,
            dense_intermediate,
            heads,
            q_rank,
            qk_dim,
            value_dim,
            kv_rank,
            extra_dim,
            kda_dim,
            conv,
            experts,
            top_k,
            expert_intermediate,
            shared,
            expert_latent,
            attn_res_block,
        )
        <= 0
        or conv < 2
        or qk_dim <= extra_dim
        or top_k > experts
        or dense_layers != 1
        or shared != 2
        or gating != 2
        or not bool(metadata.get(f"{arch}.expert_weights_norm", False))
        or not math.isfinite(routed_scale)
        or routed_scale <= 0
        or not math.isfinite(epsilon)
        or epsilon <= 0
        or not math.isfinite(gate_lower_bound)
        or gate_lower_bound >= 0
        or not math.isfinite(situ_beta)
        or not math.isclose(situ_beta, 4.0)
        or not math.isfinite(situ_linear_beta)
        or not math.isclose(situ_linear_beta, 25.0)
    ):
        raise ValueError("Kimi-K3 GGUF has inconsistent pinned architecture metadata")

    kv_counts = metadata[f"{arch}.attention.head_count_kv"]
    if not isinstance(kv_counts, (list, tuple, np.ndarray)):
        raise TypeError("kimi-k3.attention.head_count_kv must be an exact per-layer array")
    kv_counts = [int(value) for value in kv_counts]
    if len(kv_counts) != layers or any(value not in {0, 1} for value in kv_counts):
        raise ValueError(
            "Kimi-K3 per-layer KV-head counts must contain block_count entries of 0 or 1"
        )
    if all(value == 0 for value in kv_counts) or all(value == 1 for value in kv_counts):
        raise ValueError("Kimi-K3 requires both KDA and MLA layers")

    actual_shapes = {
        name: tuple(int(dim) for dim in gguf_model.get_tensor_shape(name))
        for name in gguf_model.tensor_names
    }
    vocab = int(
        metadata.get(
            f"{arch}.vocab_size",
            actual_shapes.get("token_embd.weight", (0,))[0],
        )
    )
    required: dict[str, tuple[int, ...] | tuple[tuple[int, ...], ...]] = {
        "token_embd.weight": (vocab, hidden),
        "output_norm.weight": (hidden,),
        "output.weight": (vocab, hidden),
        "output_res_score.weight": (hidden,),
    }
    projection_width = heads * kda_dim
    nope_dim = qk_dim - extra_dim
    for layer, layer_type in enumerate(layer_types):
        prefix = f"blk.{layer}."
        required.update(
            {
                prefix + "attn_norm.weight": (hidden,),
                prefix + "ffn_norm.weight": (hidden,),
                prefix + "attn_res_score.weight": (hidden,),
                prefix + "ffn_res_score.weight": (hidden,),
            }
        )
        if layer_type == "kimi_k3_attention":
            required.update(
                {
                    prefix + "attn_q.weight": (projection_width, hidden),
                    prefix + "attn_k.weight": (projection_width, hidden),
                    prefix + "attn_v.weight": (projection_width, hidden),
                    prefix + "ssm_conv1d_q.weight": (
                        (1, projection_width, 1, conv),
                        (1, projection_width, conv),
                    ),
                    prefix + "ssm_conv1d_k.weight": (
                        (1, projection_width, 1, conv),
                        (1, projection_width, conv),
                    ),
                    prefix + "ssm_conv1d_v.weight": (
                        (1, projection_width, 1, conv),
                        (1, projection_width, conv),
                    ),
                    prefix + "ssm_f_a.weight": (kda_dim, hidden),
                    prefix + "ssm_f_b.weight": (projection_width, kda_dim),
                    prefix + "ssm_beta.weight": (heads, hidden),
                    prefix + "ssm_a": (heads,),
                    prefix + "ssm_dt.bias": (projection_width,),
                    prefix + "ssm_g.weight": (projection_width, hidden),
                    prefix + "ssm_norm.weight": (kda_dim,),
                    prefix + "attn_output.weight": (hidden, projection_width),
                }
            )
        else:
            required.update(
                {
                    prefix + "attn_q_a.weight": (q_rank, hidden),
                    prefix + "attn_q_a_norm.weight": (q_rank,),
                    prefix + "attn_q_b.weight": (heads * qk_dim, q_rank),
                    prefix + "attn_kv_a_mqa.weight": (kv_rank + extra_dim, hidden),
                    prefix + "attn_kv_a_norm.weight": (kv_rank,),
                    prefix + "attn_gate.weight": (heads * value_dim, hidden),
                    prefix + "attn_output.weight": (hidden, heads * value_dim),
                }
            )
            fused = prefix + "attn_kv_b.weight"
            split_k = prefix + "attn_k_b.weight"
            split_v = prefix + "attn_v_b.weight"
            has_fused = fused in actual_shapes
            has_split = split_k in actual_shapes or split_v in actual_shapes
            if has_fused == has_split:
                raise ValueError(
                    f"Kimi-K3 layer {layer} must contain exactly one KV-B representation"
                )
            if has_fused:
                required[fused] = (heads * (nope_dim + value_dim), kv_rank)
            else:
                required[split_k] = (heads, kv_rank, nope_dim)
                required[split_v] = (heads, value_dim, kv_rank)
        if layer < dense_layers:
            required.update(
                {
                    prefix + "ffn_gate.weight": (dense_intermediate, hidden),
                    prefix + "ffn_up.weight": (dense_intermediate, hidden),
                    prefix + "ffn_down.weight": (hidden, dense_intermediate),
                }
            )
        else:
            shared_width = shared * expert_intermediate
            required.update(
                {
                    prefix + "ffn_gate_inp.weight": (experts, hidden),
                    prefix + "exp_probs_b.bias": (experts,),
                    prefix + "ffn_gate_exps.weight": (
                        experts,
                        expert_intermediate,
                        expert_latent,
                    ),
                    prefix + "ffn_up_exps.weight": (
                        experts,
                        expert_intermediate,
                        expert_latent,
                    ),
                    prefix + "ffn_down_exps.weight": (
                        experts,
                        expert_latent,
                        expert_intermediate,
                    ),
                    prefix + "ffn_routed_down.weight": (expert_latent, hidden),
                    prefix + "ffn_routed_up.weight": (hidden, expert_latent),
                    prefix + "ffn_routed_norm.weight": (expert_latent,),
                    prefix + "ffn_gate_shexp.weight": (shared_width, hidden),
                    prefix + "ffn_up_shexp.weight": (shared_width, hidden),
                    prefix + "ffn_down_shexp.weight": (hidden, shared_width),
                }
            )

    actual = set(gguf_model.tensor_names)
    out_of_range = sorted(
        name
        for name in actual
        if (match := re.match(r"^blk\.(\d+)\.", name)) and int(match.group(1)) >= layers
    )
    missing = sorted(set(required) - actual)
    unexpected = sorted(actual - set(required))
    if missing or unexpected or out_of_range:
        raise ValueError(
            "Invalid Kimi-K3 GGUF tensor closure: "
            f"missing={missing}, unexpected={unexpected}, out_of_range={out_of_range}"
        )
    malformed: dict[str, object] = {}
    for name, expected in required.items():
        actual_shape = actual_shapes.get(name)
        alternatives = expected if expected and isinstance(expected[0], tuple) else (expected,)
        if actual_shape not in alternatives:
            malformed[name] = (expected, actual_shape)
    if malformed:
        raise ValueError(f"Kimi-K3 GGUF has invalid tensor shape(s): {malformed}")

    float_types = float_storage_type_ids()
    non_matmul = {
        name
        for name in required
        if name.endswith(
            ("_norm.weight", "ssm_a", "ssm_dt.bias", "exp_probs_b.bias", "_res_score.weight")
        )
        or "conv1d" in name
    }
    invalid_storage = []
    for name, _raw, qtype, _shape in gguf_model.tensor_items_raw():
        qtype_id = getattr(qtype, "value", qtype)
        if name in non_matmul and qtype_id not in float_types:
            invalid_storage.append(name)
    if invalid_storage:
        raise ValueError(
            "Kimi-K3 recurrent, normalization, residual-score, and routing auxiliary "
            f"tensors must remain float: {sorted(invalid_storage)}"
        )

    for layer, layer_type in enumerate(layer_types):
        if layer_type != "kimi_k3_attention":
            continue
        name = f"blk.{layer}.ssm_a"
        decay = np.asarray(gguf_model.get_tensor(name))
        if not np.all(np.isfinite(decay)) or not np.all(decay < 0):
            raise ValueError(
                f"Malformed Kimi-K3 decay tensor {name!r}: expected finite -exp(A_log)"
            )


def _raise_for_invalid_dense_c01_tensor_contract(gguf_model) -> None:
    """Validate the exact pinned C01 dense profiles before config extraction."""
    import numpy as np

    from mobius.integrations.gguf._tensor_mapping import is_known_skip

    architecture = gguf_model.architecture
    if architecture not in {"apertus", "baichuan", "chatglm", "phi2", "seed_oss"}:
        return

    metadata = gguf_model.metadata
    required_geometry = (
        "context_length",
        "embedding_length",
        "feed_forward_length",
        "block_count",
        "attention.head_count",
    )
    missing_geometry = [
        f"{architecture}.{suffix}"
        for suffix in required_geometry
        if f"{architecture}.{suffix}" not in metadata
    ]
    if missing_geometry:
        raise ValueError(
            f"{architecture} GGUF is missing required dense metadata: {missing_geometry}"
        )
    hidden = int(metadata[f"{architecture}.embedding_length"])
    intermediate = int(metadata[f"{architecture}.feed_forward_length"])
    layers = int(metadata[f"{architecture}.block_count"])
    heads = int(metadata[f"{architecture}.attention.head_count"])
    context = int(metadata[f"{architecture}.context_length"])
    kv_heads = int(metadata.get(f"{architecture}.attention.head_count_kv", heads))
    if (
        context <= 0
        or hidden <= 0
        or intermediate <= 0
        or layers <= 0
        or heads <= 0
        or hidden % heads
    ):
        raise ValueError(f"{architecture} GGUF has invalid dense model geometry")
    default_head_dim = hidden // heads
    key_head_dim = int(metadata.get(f"{architecture}.attention.key_length", default_head_dim))
    value_head_dim = int(metadata.get(f"{architecture}.attention.value_length", key_head_dim))
    if key_head_dim <= 0 or value_head_dim <= 0 or key_head_dim != value_head_dim:
        raise ValueError(
            f"{architecture} GGUF requires equal positive key/value head widths, got "
            f"key_length={key_head_dim}, value_length={value_head_dim}"
        )
    head_dim = key_head_dim
    rope_dim = int(metadata.get(f"{architecture}.rope.dimension_count", head_dim))
    if kv_heads <= 0 or heads % kv_heads:
        raise ValueError(
            f"{architecture} GGUF has invalid grouped-query geometry: "
            f"head_count={heads}, head_count_kv={kv_heads}"
        )
    if architecture in {"baichuan", "chatglm", "phi2"} and heads * head_dim != hidden:
        raise ValueError(
            f"{architecture} requires attention key width * head_count == "
            f"embedding_length, got {head_dim} * {heads} != {hidden}"
        )
    if rope_dim <= 0 or rope_dim > head_dim or rope_dim % 2:
        raise ValueError(
            f"{architecture} GGUF has invalid rope.dimension_count={rope_dim} "
            f"for head_dim={head_dim}"
        )
    vocab = int(metadata.get(f"{architecture}.vocab_size", 0))
    if not vocab:
        vocab = len(metadata.get("tokenizer.ggml.tokens", ()))
    if vocab <= 0:
        raise ValueError(f"{architecture} GGUF has no positive vocabulary size")

    if architecture == "baichuan":
        if layers != 32:
            raise ValueError(
                "Baichuan GGUF import supports only block_count=32 (the pinned 7B RoPE "
                f"profile), got {layers}; block_count=40 uses unsupported hardcoded ALiBi."
            )
        if kv_heads != heads or rope_dim != head_dim:
            raise ValueError(
                "Baichuan 7B requires full MHA and full-head RoPE: "
                f"head_count={heads}, head_count_kv={kv_heads}, "
                f"rope.dimension_count={rope_dim}, head_dim={head_dim}"
            )
        if float(metadata.get("baichuan.attention.max_alibi_bias", 0.0)):
            raise ValueError("Baichuan 7B RoPE metadata contradicts a nonzero ALiBi bias")
    elif architecture == "phi2":
        if kv_heads != heads or intermediate != 4 * hidden:
            raise ValueError(
                "Phi-2 requires full MHA and feed_forward_length == 4 * embedding_length"
            )
    elif architecture == "seed_oss":
        if layers != 64:
            raise ValueError(f"Seed-OSS pinned profile requires block_count=64, got {layers}")
        if kv_heads <= 0 or heads % kv_heads or rope_dim != head_dim:
            raise ValueError(
                "Seed-OSS requires valid grouped-query geometry and full-head RoPE"
            )

    items = list(gguf_model.tensor_items_raw())
    actual = {
        name: tuple(int(dimension) for dimension in shape)
        for name, _raw, _qtype, shape in items
        if not is_known_skip(name)
    }
    required: dict[str, tuple[int, ...]] = {
        "token_embd.weight": (vocab, hidden),
        "output_norm.weight": (hidden,),
    }
    optional: dict[str, tuple[int, ...]] = {}
    if architecture in {"apertus", "baichuan", "phi2"}:
        required["output.weight"] = (vocab, hidden)
    elif architecture in {"chatglm", "seed_oss"}:
        optional["output.weight"] = (vocab, hidden)
    if architecture == "phi2":
        required.update(
            {
                "output_norm.bias": (hidden,),
                "output.bias": (vocab,),
            }
        )

    q_dim = heads * head_dim
    kv_dim = kv_heads * head_dim
    qkv_biases: list[str] = []
    for layer in range(layers):
        prefix = f"blk.{layer}."
        fused_weight = prefix + "attn_qkv.weight"
        separate_weights = {
            prefix + "attn_q.weight": (q_dim, hidden),
            prefix + "attn_k.weight": (kv_dim, hidden),
            prefix + "attn_v.weight": (kv_dim, hidden),
        }
        if architecture in {"chatglm", "phi2"}:
            has_fused = fused_weight in actual
            present_separate = set(separate_weights) & set(actual)
            if has_fused == bool(present_separate) or (
                present_separate and present_separate != set(separate_weights)
            ):
                raise ValueError(
                    f"{architecture} layer {layer} must contain exactly one complete QKV "
                    "layout: one fused attn_qkv tensor or all separate attn_q/attn_k/attn_v "
                    "tensors"
                )
        else:
            has_fused = False

        if not has_fused:
            required.update(separate_weights)
            selected_biases = {
                prefix + "attn_q.bias": (q_dim,),
                prefix + "attn_k.bias": (kv_dim,),
                prefix + "attn_v.bias": (kv_dim,),
            }
            alternate_biases = {prefix + "attn_qkv.bias"}
        else:
            required[fused_weight] = (q_dim + 2 * kv_dim, hidden)
            selected_biases = {prefix + "attn_qkv.bias": (q_dim + 2 * kv_dim,)}
            alternate_biases = {
                prefix + "attn_q.bias",
                prefix + "attn_k.bias",
                prefix + "attn_v.bias",
            }
        mismatched_biases = sorted(alternate_biases & set(actual))
        if mismatched_biases:
            raise ValueError(
                f"{architecture} layer {layer} QKV bias layout does not match its "
                f"weight layout: {mismatched_biases}"
            )

        if architecture == "chatglm":
            required.update(
                {
                    prefix + "attn_norm.weight": (hidden,),
                    prefix + "attn_output.weight": (hidden, q_dim),
                    prefix + "ffn_norm.weight": (hidden,),
                    prefix + "ffn_up.weight": (2 * intermediate, hidden),
                    prefix + "ffn_down.weight": (hidden, intermediate),
                }
            )
            optional.update(selected_biases)
            qkv_biases.extend(selected_biases)
        else:
            required.update(
                {
                    prefix + "attn_norm.weight": (hidden,),
                    prefix + "attn_output.weight": (hidden, q_dim),
                }
            )
            if architecture != "phi2":
                required.update(
                    {
                        prefix + "ffn_up.weight": (intermediate, hidden),
                        prefix + "ffn_down.weight": (hidden, intermediate),
                    }
                )
        if architecture in {"apertus", "baichuan"}:
            required[prefix + "ffn_norm.weight"] = (hidden,)
        if architecture in {"baichuan", "seed_oss"}:
            required[prefix + "ffn_gate.weight"] = (intermediate, hidden)
        if architecture == "phi2":
            required.update(
                {
                    prefix + "attn_norm.bias": (hidden,),
                    prefix + "attn_output.bias": (hidden,),
                    prefix + "ffn_up.weight": (intermediate, hidden),
                    prefix + "ffn_up.bias": (intermediate,),
                    prefix + "ffn_down.weight": (hidden, intermediate),
                    prefix + "ffn_down.bias": (hidden,),
                }
            )
            required.update(selected_biases)
        elif architecture == "seed_oss":
            required.update(
                {
                    prefix + "post_attention_norm.weight": (hidden,),
                    prefix + "ffn_up.weight": (intermediate, hidden),
                    prefix + "ffn_down.weight": (hidden, intermediate),
                }
            )
            optional.update(
                {
                    prefix + "attn_q.bias": (q_dim,),
                    prefix + "attn_k.bias": (kv_dim,),
                    prefix + "attn_v.bias": (kv_dim,),
                }
            )
        elif architecture == "apertus":
            required.update(
                {
                    prefix + "attn_q_norm.weight": (head_dim,),
                    prefix + "attn_k_norm.weight": (head_dim,),
                }
            )
            optional.update(
                {
                    prefix + "attn_q.bias": (q_dim,),
                    prefix + "attn_k.bias": (kv_dim,),
                    prefix + "attn_v.bias": (kv_dim,),
                    prefix + "attn_output.bias": (hidden,),
                    prefix + "attn_q_norm.bias": (head_dim,),
                    prefix + "attn_k_norm.bias": (head_dim,),
                }
            )

    if architecture == "seed_oss":
        attention_scale = float(metadata.get("seed_oss.attention.scale", 0.0))
        if not np.isfinite(attention_scale):
            raise ValueError("seed_oss.attention.scale must be finite")
        if attention_scale:
            raise ValueError(
                f"seed_oss.attention.scale={attention_scale} is not consumed by the "
                "pinned llama.cpp loader for this architecture"
            )
        qkv_biases = {
            f"blk.{layer}.attn_{projection}.bias"
            for layer in range(layers)
            for projection in ("q", "k", "v")
        }
        present_qkv_biases = qkv_biases & set(actual)
        if present_qkv_biases and present_qkv_biases != qkv_biases:
            raise ValueError(
                "Seed-OSS Q/K/V bias tensors must be present for every projection "
                "in every layer or absent entirely"
            )

    if architecture == "chatglm":
        present_biases = set(qkv_biases) & set(actual)
        if present_biases and present_biases != set(qkv_biases):
            raise ValueError("ChatGLM QKV bias must be present in every layer or none")

    if architecture == "apertus":
        for family in (
            ("attn_q.bias", "attn_k.bias", "attn_v.bias"),
            ("attn_output.bias",),
        ):
            family_names = {
                f"blk.{layer}.{suffix}" for layer in range(layers) for suffix in family
            }
            present = family_names & set(actual)
            if present and present != family_names:
                raise ValueError(
                    "Apertus projection biases must be present for every corresponding "
                    f"projection in every layer or absent entirely: {sorted(present)}"
                )

    allowed = set(required) | set(optional)
    shape_checked = set(allowed)
    missing = sorted(set(required) - set(actual))
    unexpected = sorted(set(actual) - allowed)
    malformed = {
        name: (required.get(name, optional.get(name)), actual[name])
        for name in shape_checked & set(actual)
        if actual[name] != required.get(name, optional.get(name))
    }
    if missing or unexpected or malformed:
        raise ValueError(
            f"Invalid {architecture} GGUF tensor closure: missing={missing}, "
            f"unexpected={unexpected}, malformed={malformed}"
        )


def _raise_for_invalid_plamo_tensor_contract(
    gguf_model,
    *,
    keep_quantized: bool | None,
) -> None:
    """Validate the fixed converter-emitted PLaMo-13B tensor/value contract."""
    if gguf_model.architecture != "plamo":
        return

    from mobius.integrations.gguf._quant_registry import (
        float_storage_type_ids,
        get_quant_spec,
    )

    metadata = gguf_model.metadata
    expected_metadata = {
        "plamo.context_length": 4096,
        "plamo.embedding_length": 5120,
        "plamo.block_count": 40,
        "plamo.feed_forward_length": 16640,
        "plamo.attention.head_count": 40,
        "plamo.attention.head_count_kv": 5,
    }
    missing_metadata = sorted(set(expected_metadata) - set(metadata))
    if missing_metadata:
        raise ValueError(
            f"PLaMo GGUF is missing required architecture metadata: {missing_metadata}"
        )
    invalid_metadata = {
        key: (expected, metadata[key])
        for key, expected in expected_metadata.items()
        if int(metadata[key]) != expected
    }
    if invalid_metadata:
        raise ValueError(f"PLaMo GGUF has unsupported fixed geometry: {invalid_metadata}")
    if int(metadata.get("plamo.rope.dimension_count", 128)) != 128:
        raise ValueError("PLaMo GGUF requires full-head rope.dimension_count=128")
    if not math.isclose(
        float(metadata.get("plamo.rope.freq_base", 10000.0)),
        10000.0,
        rel_tol=0.0,
        abs_tol=1e-6,
    ):
        raise ValueError("PLaMo GGUF requires rope.freq_base=10000")

    items = list(gguf_model.tensor_items_raw())
    actual = {
        name: tuple(int(dimension) for dimension in shape)
        for name, _raw, _qtype, shape in items
    }
    if "token_embd.weight" not in actual or len(actual["token_embd.weight"]) != 2:
        raise ValueError("PLaMo GGUF requires a rank-2 token_embd.weight tensor")
    vocab, hidden = actual["token_embd.weight"]
    if hidden != 5120 or vocab <= 0:
        raise ValueError(
            "PLaMo token_embd.weight must have shape [positive_vocab, 5120], "
            f"got {actual['token_embd.weight']}"
        )

    required: dict[str, tuple[int, ...]] = {
        "token_embd.weight": (vocab, 5120),
        "output_norm.weight": (5120,),
        "output.weight": (vocab, 5120),
    }
    for layer in range(40):
        prefix = f"blk.{layer}."
        required.update(
            {
                prefix + "attn_norm.weight": (5120,),
                prefix + "attn_q.weight": (5120, 5120),
                prefix + "attn_k.weight": (640, 5120),
                prefix + "attn_v.weight": (640, 5120),
                prefix + "attn_output.weight": (5120, 5120),
                prefix + "ffn_gate.weight": (16640, 5120),
                prefix + "ffn_up.weight": (16640, 5120),
                prefix + "ffn_down.weight": (5120, 16640),
            }
        )

    missing = sorted(set(required) - set(actual))
    unexpected = sorted(set(actual) - set(required))
    malformed = {
        name: (required[name], actual[name])
        for name in set(required) & set(actual)
        if required[name] != actual[name]
    }
    if missing or unexpected or malformed:
        raise ValueError(
            "Invalid PLaMo exact GGUF tensor closure: "
            f"missing={missing}, unexpected={unexpected}, malformed={malformed}"
        )

    float_types = float_storage_type_ids()
    unsupported_qtypes: dict[str, str] = {}
    packed_shuffle_tensors: list[str] = []
    for name, _raw, qtype, _shape in items:
        qtype_id = getattr(qtype, "value", qtype)
        quant_spec = get_quant_spec(qtype)
        if quant_spec is None or quant_spec.dequantize is not Support.SUPPORTED:
            unsupported_qtypes[name] = str(getattr(qtype, "name", qtype))
        if (
            keep_quantized
            and qtype_id not in float_types
            and name.endswith(("attn_q.weight", "attn_output.weight"))
        ):
            packed_shuffle_tensors.append(name)
    if unsupported_qtypes:
        raise ValueError(f"PLaMo GGUF contains unsupported qtypes: {unsupported_qtypes}")
    if packed_shuffle_tensors:
        raise ValueError(
            "PLaMo keep_quantized=True cannot preserve packed Q/output tensors that "
            "require value shuffles: " + ", ".join(sorted(packed_shuffle_tensors))
        )


def _raise_for_invalid_exact_legacy_decoder_tensor_contract(gguf_model) -> None:
    """Validate narrowed executable unions for six conventional GGUF decoders."""
    architecture = gguf_model.architecture
    supported = {"gptneox", "jais", "mpt", "refact", "ernie4_5", "openelm"}
    if architecture not in supported:
        return
    metadata = gguf_model.metadata
    common = (
        "context_length",
        "embedding_length",
        "feed_forward_length",
        "block_count",
        "attention.head_count",
    )
    missing_metadata = [
        f"{architecture}.{suffix}"
        for suffix in common
        if f"{architecture}.{suffix}" not in metadata
    ]
    if missing_metadata:
        raise ValueError(
            f"{architecture} GGUF is missing required decoder metadata: {missing_metadata}"
        )

    hidden = int(metadata[f"{architecture}.embedding_length"])
    layers = int(metadata[f"{architecture}.block_count"])
    context = int(metadata[f"{architecture}.context_length"])
    vocab = int(metadata.get(f"{architecture}.vocab_size", 0)) or len(
        metadata.get("tokenizer.ggml.tokens", ())
    )
    raw_heads = metadata[f"{architecture}.attention.head_count"]
    raw_kv_heads = metadata.get(f"{architecture}.attention.head_count_kv", raw_heads)
    raw_ffn = metadata[f"{architecture}.feed_forward_length"]
    if architecture == "openelm":
        if not all(
            isinstance(value, (list, tuple, np.ndarray))
            for value in (
                raw_heads,
                raw_kv_heads,
                raw_ffn,
            )
        ):
            raise ValueError("openelm head and feed-forward metadata must be per-layer arrays")
        heads_by_layer = tuple(int(value) for value in raw_heads)
        kv_heads_by_layer = tuple(int(value) for value in raw_kv_heads)
        ffn_by_layer = tuple(int(value) for value in raw_ffn)
        if not all(
            len(values) == layers
            for values in (
                heads_by_layer,
                kv_heads_by_layer,
                ffn_by_layer,
            )
        ):
            raise ValueError(
                "openelm per-layer arrays must contain exactly block_count entries"
            )
    else:
        heads = int(raw_heads)
        kv_heads = int(raw_kv_heads)
        intermediate = int(raw_ffn)
        heads_by_layer = (heads,) * layers
        kv_heads_by_layer = (kv_heads,) * layers
        ffn_by_layer = (intermediate,) * layers

    head_dim = int(
        metadata.get(
            f"{architecture}.attention.key_length",
            hidden // heads_by_layer[0] if heads_by_layer[0] else 0,
        )
    )
    if (
        min(hidden, layers, context, vocab, head_dim) <= 0
        or min(heads_by_layer) <= 0
        or min(kv_heads_by_layer) <= 0
        or min(ffn_by_layer) <= 0
        or any(heads % kv for heads, kv in zip(heads_by_layer, kv_heads_by_layer))
    ):
        raise ValueError(f"{architecture} GGUF has invalid exact decoder geometry")
    if architecture != "openelm" and hidden != heads_by_layer[0] * head_dim:
        raise ValueError(f"{architecture} attention heads do not span embedding_length")
    if architecture in {"gptneox", "jais"} and kv_heads_by_layer[0] != heads_by_layer[0]:
        raise ValueError(f"{architecture} exact GGUF subset requires multi-head attention")
    if architecture == "refact" and kv_heads_by_layer[0] != 1:
        raise ValueError("refact exact GGUF subset requires exactly one KV head")
    if architecture == "gptneox" and not bool(metadata["gptneox.use_parallel_residual"]):
        raise ValueError("gptneox exact GGUF subset requires use_parallel_residual=true")
    if architecture in {"jais", "mpt"}:
        max_bias = float(metadata[f"{architecture}.attention.max_alibi_bias"])
        if not math.isfinite(max_bias) or max_bias <= 0:
            raise ValueError(
                f"{architecture}.attention.max_alibi_bias must be finite and positive"
            )
    if architecture == "mpt":
        clip = float(metadata.get("mpt.attention.clamp_kqv", 0.0) or 0.0)
        if clip:
            raise ValueError("mpt exact GGUF subset rejects nonzero attention.clamp_kqv")
    if architecture in {"jais", "mpt", "refact"}:
        position_keys = sorted(
            key for key in metadata if key.startswith(f"{architecture}.rope.")
        )
        if position_keys:
            raise ValueError(
                f"{architecture} exact ALiBi subset rejects RoPE metadata: {position_keys}"
            )
    if architecture in {"gptneox", "ernie4_5", "openelm"}:
        rope_dim = int(metadata.get(f"{architecture}.rope.dimension_count", head_dim))
        if rope_dim <= 0 or rope_dim > head_dim or rope_dim % 2:
            raise ValueError(f"{architecture} has invalid rotary dimension")
        if architecture in {"ernie4_5", "openelm"} and rope_dim != head_dim:
            raise ValueError(f"{architecture} exact subset requires full-head RoPE")
    if "ernie4_5.rope.dimension_sections" in metadata:
        raise ValueError("ernie4_5 exact dense subset rejects rope.dimension_sections")
    if architecture in {"ernie4_5", "refact"}:
        forbidden_metadata = sorted(
            key
            for key in metadata
            if key.startswith(
                (
                    f"{architecture}.expert",
                    f"{architecture}.leading_dense",
                    f"{architecture}.interleave_moe",
                )
            )
        )
        if forbidden_metadata:
            raise ValueError(
                f"{architecture} exact dense subset rejects MoE metadata: {forbidden_metadata}"
            )

    actual = {
        name: tuple(int(dimension) for dimension in shape)
        for name, _raw, _qtype, shape in gguf_model.tensor_items_raw()
    }
    required: dict[str, tuple[int, ...]] = {
        "token_embd.weight": (vocab, hidden),
        "output_norm.weight": (hidden,),
    }
    optional: dict[str, tuple[int, ...]] = {}
    if architecture in {"gptneox", "jais"}:
        required["output.weight"] = (vocab, hidden)
        required["output_norm.bias"] = (hidden,)
    elif architecture != "openelm":
        optional["output.weight"] = (vocab, hidden)

    mpt_bias_names: list[str] = ["output_norm.bias"]
    for layer, (heads, kv_heads, intermediate) in enumerate(
        zip(heads_by_layer, kv_heads_by_layer, ffn_by_layer)
    ):
        prefix = f"blk.{layer}."
        q_dim = heads * head_dim
        kv_dim = kv_heads * head_dim
        required[prefix + "attn_norm.weight"] = (hidden,)
        required[prefix + "ffn_norm.weight"] = (hidden,)
        required[prefix + "attn_output.weight"] = (hidden, q_dim)
        if architecture in {"gptneox", "jais"}:
            required[prefix + "attn_norm.bias"] = (hidden,)
            required[prefix + "ffn_norm.bias"] = (hidden,)
            required[prefix + "attn_output.bias"] = (hidden,)
        elif architecture == "mpt":
            mpt_bias_names.extend(
                [
                    prefix + "attn_norm.bias",
                    prefix + "ffn_norm.bias",
                    prefix + "attn_qkv.bias",
                    prefix + "attn_output.bias",
                    prefix + "ffn_up.bias",
                    prefix + "ffn_down.bias",
                ]
            )

        if architecture in {"gptneox", "jais", "mpt", "openelm"}:
            required[prefix + "attn_qkv.weight"] = (q_dim + 2 * kv_dim, hidden)
            if architecture in {"gptneox", "jais"}:
                required[prefix + "attn_qkv.bias"] = (q_dim + 2 * kv_dim,)
        else:
            required.update(
                {
                    prefix + "attn_q.weight": (q_dim, hidden),
                    prefix + "attn_k.weight": (kv_dim, hidden),
                    prefix + "attn_v.weight": (kv_dim, hidden),
                }
            )
        if architecture == "openelm":
            required[prefix + "attn_q_norm.weight"] = (head_dim,)
            required[prefix + "attn_k_norm.weight"] = (head_dim,)

        if architecture in {"jais", "refact", "ernie4_5", "openelm"}:
            required[prefix + "ffn_gate.weight"] = (intermediate, hidden)
        required[prefix + "ffn_up.weight"] = (intermediate, hidden)
        required[prefix + "ffn_down.weight"] = (hidden, intermediate)
        if architecture == "gptneox":
            required[prefix + "ffn_up.bias"] = (intermediate,)
            required[prefix + "ffn_down.bias"] = (hidden,)
        elif architecture == "jais":
            required[prefix + "ffn_gate.bias"] = (intermediate,)
            required[prefix + "ffn_up.bias"] = (intermediate,)
            required[prefix + "ffn_down.bias"] = (hidden,)

    if architecture == "mpt":
        present = {name for name in mpt_bias_names if name in actual}
        if present and present != set(mpt_bias_names):
            raise ValueError(
                "mpt exact GGUF subset requires every optional bias family or none"
            )
        if present:
            for name in mpt_bias_names:
                if name.endswith("attn_qkv.bias"):
                    layer = int(name.split(".")[1])
                    shape = (
                        heads_by_layer[layer] * head_dim
                        + 2 * kv_heads_by_layer[layer] * head_dim,
                    )
                elif name.endswith("ffn_up.bias"):
                    layer = int(name.split(".")[1])
                    shape = (ffn_by_layer[layer],)
                else:
                    shape = (hidden,)
                required[name] = shape

    allowed = set(required) | set(optional)
    missing = sorted(set(required) - set(actual))
    unexpected = sorted(set(actual) - allowed)
    malformed = {
        name: (required.get(name, optional.get(name)), actual[name])
        for name in allowed & set(actual)
        if actual[name] != required.get(name, optional.get(name))
    }
    if missing or unexpected or malformed:
        raise ValueError(
            f"Invalid {architecture} exact GGUF tensor closure: missing={missing}, "
            f"unexpected={unexpected}, malformed={malformed}"
        )


def _raise_for_invalid_conventional_decoder_tensor_contract(gguf_model) -> None:
    """Validate the complete pinned tensor census for conventional legacy decoders."""
    architecture = gguf_model.architecture
    if architecture in {"gptneox", "jais", "mpt", "refact", "ernie4_5", "openelm"}:
        _raise_for_invalid_exact_legacy_decoder_tensor_contract(gguf_model)
        return
    supported = {
        "bloom",
        "codeshell",
        "command-r",
        "jais2",
        "orion",
        "pangu-embedded",
        "qwen",
        "starcoder",
        "xverse",
    }
    if architecture not in supported:
        return
    metadata = gguf_model.metadata
    suffixes = (
        "context_length",
        "embedding_length",
        "feed_forward_length",
        "block_count",
        "attention.head_count",
    )
    if architecture == "command-r":
        suffixes += ("logit_scale",)
    missing_metadata = [
        f"{architecture}.{suffix}"
        for suffix in suffixes
        if f"{architecture}.{suffix}" not in metadata
    ]
    if missing_metadata:
        raise ValueError(
            f"{architecture} GGUF is missing required decoder metadata: {missing_metadata}"
        )
    if architecture == "command-r":
        logit_scale = float(metadata["command-r.logit_scale"])
        if not math.isfinite(logit_scale) or logit_scale <= 0:
            raise ValueError(
                f"command-r.logit_scale must be a finite positive value, got {logit_scale!r}"
            )
    hidden = int(metadata[f"{architecture}.embedding_length"])
    serialized_ffn = int(metadata[f"{architecture}.feed_forward_length"])
    layers = int(metadata[f"{architecture}.block_count"])
    heads = int(metadata[f"{architecture}.attention.head_count"])
    kv_heads = int(metadata.get(f"{architecture}.attention.head_count_kv", heads))
    context = int(metadata[f"{architecture}.context_length"])
    vocab = int(metadata.get(f"{architecture}.vocab_size", 0)) or len(
        metadata.get("tokenizer.ggml.tokens", ())
    )
    if min(hidden, serialized_ffn, layers, heads, kv_heads, context, vocab) <= 0:
        raise ValueError(f"{architecture} GGUF has invalid decoder geometry")
    if hidden % heads or heads % kv_heads:
        raise ValueError(f"{architecture} GGUF has invalid grouped-query geometry")
    head_dim = hidden // heads
    kv_dim = kv_heads * head_dim
    intermediate = serialized_ffn // 2 if architecture == "qwen" else serialized_ffn
    if architecture == "qwen" and serialized_ffn % 2:
        raise ValueError("qwen feed_forward_length must be even (two fused SwiGLU halves)")
    if architecture == "command-r" and layers >= 64:
        raise ValueError(
            "command-r GGUF with block_count >= 64 requires distinct per-head Q/K "
            "LayerNorm weights, which the current Attention graph cannot represent"
        )

    actual = {
        name: tuple(int(dimension) for dimension in shape)
        for name, _raw, _qtype, shape in gguf_model.tensor_items_raw()
    }
    if architecture in {"command-r", "orion", "pangu-embedded", "xverse"}:
        fused = sorted(name for name in actual if name.endswith(".attn_qkv.weight"))
        if fused:
            raise ValueError(
                f"{architecture} GGUF fused QKV tensors are not supported; "
                f"split attn_q/attn_k/attn_v tensors are required: {fused}"
            )
    pangu_has_qkv_bias = architecture == "pangu-embedded" and any(
        name.endswith(("attn_q.bias", "attn_k.bias", "attn_v.bias")) for name in actual
    )
    required: dict[str, tuple[int, ...]] = {}
    optional: dict[str, tuple[int, ...]] = {}
    if architecture == "codeshell":
        optional["token_embd.weight"] = (vocab, hidden)
    else:
        required["token_embd.weight"] = (vocab, hidden)
    if architecture == "bloom":
        required.update(
            {
                "token_embd_norm.weight": (hidden,),
                "token_embd_norm.bias": (hidden,),
            }
        )
    if architecture == "starcoder":
        required["position_embd.weight"] = (context, hidden)
    required["output_norm.weight"] = (hidden,)
    if architecture in {"bloom", "codeshell", "jais2", "orion", "starcoder"}:
        required["output_norm.bias"] = (hidden,)
    output_shape = (vocab, hidden)
    if architecture == "codeshell":
        optional["output.weight"] = output_shape
    elif architecture in {"orion", "qwen", "xverse"}:
        required["output.weight"] = output_shape
    elif architecture in {"bloom", "jais2", "pangu-embedded", "starcoder"}:
        optional["output.weight"] = output_shape

    gated = architecture in {"command-r", "orion", "pangu-embedded", "qwen", "xverse"}
    fused_only = architecture in {"bloom", "qwen", "starcoder"}
    separate_only = architecture == "jais2"
    for layer in range(layers):
        prefix = f"blk.{layer}."
        required[prefix + "attn_norm.weight"] = (hidden,)
        if architecture in {"bloom", "codeshell", "jais2", "orion", "starcoder"}:
            required[prefix + "attn_norm.bias"] = (hidden,)
        required[prefix + "attn_output.weight"] = (hidden, hidden)
        if architecture in {"bloom", "codeshell", "jais2", "pangu-embedded", "starcoder"}:
            required[prefix + "attn_output.bias"] = (hidden,)

        fused = prefix + "attn_qkv.weight"
        split = {
            prefix + "attn_q.weight": (hidden, hidden),
            prefix + "attn_k.weight": (kv_dim, hidden),
            prefix + "attn_v.weight": (kv_dim, hidden),
        }
        use_fused = fused_only or (not separate_only and fused in actual)
        if use_fused:
            required[fused] = (hidden + 2 * kv_dim, hidden)
            if architecture in {"bloom", "codeshell", "qwen", "starcoder"}:
                required[prefix + "attn_qkv.bias"] = (hidden + 2 * kv_dim,)
        else:
            required.update(split)
            if architecture in {"codeshell", "jais2"}:
                required.update(
                    {
                        prefix + "attn_q.bias": (hidden,),
                        prefix + "attn_k.bias": (kv_dim,),
                        prefix + "attn_v.bias": (kv_dim,),
                    }
                )
            elif architecture == "pangu-embedded":
                qkv_bias_names = {
                    prefix + "attn_q.bias": (hidden,),
                    prefix + "attn_k.bias": (kv_dim,),
                    prefix + "attn_v.bias": (kv_dim,),
                }
                if pangu_has_qkv_bias:
                    required.update(qkv_bias_names)
        if architecture != "command-r":
            required[prefix + "ffn_norm.weight"] = (hidden,)
        if architecture in {"bloom", "codeshell", "jais2", "orion", "starcoder"}:
            required[prefix + "ffn_norm.bias"] = (hidden,)
        if gated:
            required[prefix + "ffn_gate.weight"] = (intermediate, hidden)
        required[prefix + "ffn_up.weight"] = (intermediate, hidden)
        required[prefix + "ffn_down.weight"] = (hidden, intermediate)
        if architecture in {"bloom", "codeshell", "jais2", "starcoder"}:
            required[prefix + "ffn_up.bias"] = (intermediate,)
            required[prefix + "ffn_down.bias"] = (hidden,)

    allowed = set(required) | set(optional)
    missing = sorted(set(required) - set(actual))
    if architecture == "codeshell" and not {
        "token_embd.weight",
        "output.weight",
    } & set(actual):
        missing.append("token_embd.weight or output.weight")
    unexpected = sorted(set(actual) - allowed)
    malformed = {
        name: (required.get(name, optional.get(name)), actual[name])
        for name in allowed & set(actual)
        if actual[name] != required.get(name, optional.get(name))
    }
    if missing or unexpected or malformed:
        raise ValueError(
            f"Invalid {architecture} GGUF tensor closure: missing={missing}, "
            f"unexpected={unexpected}, malformed={malformed}"
        )


def _raise_for_invalid_maincoder_tensor_contract(gguf_model) -> None:
    """Validate Maincoder metadata, geometry, qtypes, and tied tensor closure."""
    if gguf_model.architecture != "maincoder":
        return

    metadata = gguf_model.metadata
    arch = "maincoder"
    required_suffixes = (
        "context_length",
        "embedding_length",
        "feed_forward_length",
        "block_count",
        "attention.head_count",
        "attention.head_count_kv",
        "attention.key_length",
        "attention.value_length",
        "attention.layer_norm_rms_epsilon",
        "rope.freq_base",
        "rope.dimension_count",
    )
    missing_metadata = [
        f"{arch}.{suffix}"
        for suffix in required_suffixes
        if f"{arch}.{suffix}" not in metadata
    ]
    if missing_metadata:
        raise ValueError(f"Maincoder GGUF is missing required metadata: {missing_metadata}")

    def _positive_int(suffix: str) -> int:
        key = f"{arch}.{suffix}"
        value = metadata[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{key} must be a positive integer, got {value!r}")
        try:
            integer = int(value)
        except (ValueError, OverflowError) as error:
            raise ValueError(f"{key} must be a positive integer, got {value!r}") from error
        if integer <= 0 or integer != value:
            raise ValueError(f"{key} must be a positive integer, got {value!r}")
        return integer

    _positive_int("context_length")
    hidden = _positive_int("embedding_length")
    intermediate = _positive_int("feed_forward_length")
    blocks = _positive_int("block_count")
    query_heads = _positive_int("attention.head_count")
    kv_heads = _positive_int("attention.head_count_kv")
    key_length = _positive_int("attention.key_length")
    value_length = _positive_int("attention.value_length")
    rope_dimension = _positive_int("rope.dimension_count")
    epsilon = metadata[f"{arch}.attention.layer_norm_rms_epsilon"]
    rope_base = metadata[f"{arch}.rope.freq_base"]
    for key, value in (
        (f"{arch}.attention.layer_norm_rms_epsilon", epsilon),
        (f"{arch}.rope.freq_base", rope_base),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{key} must be a finite positive number, got {value!r}")
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f"{key} must be a finite positive number, got {value!r}")

    if query_heads % kv_heads:
        raise ValueError(
            "maincoder.attention.head_count must be divisible by "
            "maincoder.attention.head_count_kv"
        )
    if hidden != query_heads * key_length:
        raise ValueError(
            "Maincoder requires embedding_length == attention.head_count * "
            f"attention.key_length, got {hidden} != {query_heads} * {key_length}"
        )
    if value_length != key_length or rope_dimension != key_length:
        raise ValueError(
            "Maincoder requires equal full-head key/value/RoPE dimensions, got "
            f"key={key_length}, value={value_length}, rope={rope_dimension}"
        )
    scaling_keys = sorted(key for key in metadata if key.startswith(f"{arch}.rope.scaling."))
    if scaling_keys:
        raise ValueError(
            "Maincoder's promoted profile does not support scaled/sectioned RoPE "
            f"metadata: {scaling_keys}"
        )

    actual: dict[str, tuple[int, ...]] = {}
    qtypes: dict[str, Any] = {}
    for name, _raw, qtype, shape in gguf_model.tensor_items_raw():
        actual[name] = tuple(int(dim) for dim in shape)
        qtypes[name] = qtype

    kv_hidden = kv_heads * key_length
    required = {
        "token_embd.weight": (-1, hidden),
        "output_norm.weight": (hidden,),
    }
    for block in range(blocks):
        prefix = f"blk.{block}."
        required.update(
            {
                prefix + "attn_norm.weight": (hidden,),
                prefix + "attn_q.weight": (hidden, hidden),
                prefix + "attn_k.weight": (kv_hidden, hidden),
                prefix + "attn_v.weight": (kv_hidden, hidden),
                prefix + "attn_output.weight": (hidden, hidden),
                prefix + "attn_q_norm.weight": (key_length,),
                prefix + "attn_k_norm.weight": (key_length,),
                prefix + "ffn_norm.weight": (hidden,),
                prefix + "ffn_gate.weight": (intermediate, hidden),
                prefix + "ffn_up.weight": (intermediate, hidden),
                prefix + "ffn_down.weight": (hidden, intermediate),
            }
        )

    missing = sorted(set(required) - set(actual))
    unexpected = sorted(set(actual) - set(required))
    malformed = {
        name: (expected, actual[name])
        for name, expected in required.items()
        if name in actual
        and (
            len(expected) != len(actual[name])
            or any(want != -1 and want != got for want, got in zip(expected, actual[name]))
        )
    }
    vocab = actual.get("token_embd.weight", (0,))[0]
    if vocab <= 0:
        malformed["token_embd.weight"] = (
            required["token_embd.weight"],
            actual.get("token_embd.weight"),
        )
    if missing or unexpected or malformed:
        raise ValueError(
            "Invalid maincoder GGUF tensor closure: "
            f"missing={missing}, unexpected={unexpected}, malformed={malformed}"
        )

    from mobius.integrations.gguf._quant_registry import float_storage_type_ids

    float_qtypes = float_storage_type_ids()
    quantized_auxiliary = sorted(
        name
        for name, shape in actual.items()
        if len(shape) == 1 and getattr(qtypes[name], "value", qtypes[name]) not in float_qtypes
    )
    if quantized_auxiliary:
        raise ValueError(
            "Maincoder normalization tensors must use float GGUF storage: "
            f"{quantized_auxiliary}"
        )


def _raise_for_invalid_smallthinker_tensor_contract(gguf_model) -> None:
    """Validate SmallThinker's complete tensor, shape, and qtype closure."""
    if gguf_model.architecture != "smallthinker":
        return

    from mobius.integrations.gguf._quant_registry import (
        float_storage_type_ids,
        get_quant_spec,
    )
    from mobius.integrations.gguf._tensor_mapping import is_known_skip

    arch = "smallthinker"
    metadata = gguf_model.metadata
    required_suffixes = (
        "context_length",
        "embedding_length",
        "feed_forward_length",
        "block_count",
        "attention.head_count",
        "attention.head_count_kv",
        "attention.layer_norm_rms_epsilon",
        "rope.dimension_count",
        "rope.freq_base",
        "expert_count",
        "expert_used_count",
        "expert_feed_forward_length",
        "expert_gating_func",
    )
    missing_metadata = [
        f"{arch}.{suffix}"
        for suffix in required_suffixes
        if f"{arch}.{suffix}" not in metadata
    ]
    if missing_metadata:
        raise ValueError(f"smallthinker GGUF is missing required metadata: {missing_metadata}")

    hidden = int(metadata[f"{arch}.embedding_length"])
    intermediate = int(metadata[f"{arch}.feed_forward_length"])
    layers = int(metadata[f"{arch}.block_count"])
    heads = int(metadata[f"{arch}.attention.head_count"])
    kv_heads = int(metadata[f"{arch}.attention.head_count_kv"])
    experts = int(metadata[f"{arch}.expert_count"])
    top_k = int(metadata[f"{arch}.expert_used_count"])
    expert_intermediate = int(metadata[f"{arch}.expert_feed_forward_length"])
    context = int(metadata[f"{arch}.context_length"])
    vocab = int(metadata.get(f"{arch}.vocab_size", 0)) or len(
        metadata.get("tokenizer.ggml.tokens", ())
    )
    epsilon = float(metadata[f"{arch}.attention.layer_norm_rms_epsilon"])
    rope_base = float(metadata[f"{arch}.rope.freq_base"])
    head_dim = int(
        metadata.get(f"{arch}.attention.key_length", hidden // heads if heads > 0 else 0)
    )
    value_dim = int(metadata.get(f"{arch}.attention.value_length", head_dim))
    rope_dim = int(metadata[f"{arch}.rope.dimension_count"])
    if (
        min(
            hidden,
            intermediate,
            layers,
            heads,
            kv_heads,
            experts,
            top_k,
            expert_intermediate,
            context,
            vocab,
            head_dim,
        )
        <= 0
        or hidden != heads * head_dim
        or heads % kv_heads
        or value_dim != head_dim
        or rope_dim != head_dim
        or intermediate != expert_intermediate
        or top_k > experts
        or not math.isfinite(epsilon)
        or epsilon <= 0
        or not math.isfinite(rope_base)
        or rope_base <= 0
    ):
        raise ValueError("smallthinker GGUF has invalid model geometry")

    items = list(gguf_model.tensor_items_raw())
    actual = {
        name: tuple(int(dimension) for dimension in shape)
        for name, _raw, _qtype, shape in items
        if not is_known_skip(name)
    }
    qtypes = {name: qtype for name, _raw, qtype, _shape in items}
    required: dict[str, tuple[int, ...]] = {
        "token_embd.weight": (vocab, hidden),
        "output_norm.weight": (hidden,),
    }
    optional: dict[str, tuple[int, ...]] = {
        "output.weight": (vocab, hidden),
    }
    all_projection_biases: set[str] = set()
    q_width = heads * head_dim
    kv_width = kv_heads * head_dim
    for layer in range(layers):
        prefix = f"blk.{layer}."
        required.update(
            {
                prefix + "attn_norm.weight": (hidden,),
                prefix + "attn_output.weight": (hidden, q_width),
                prefix + "ffn_norm.weight": (hidden,),
                prefix + "ffn_gate_inp.weight": (experts, hidden),
                prefix + "ffn_gate_exps.weight": (
                    experts,
                    expert_intermediate,
                    hidden,
                ),
                prefix + "ffn_up_exps.weight": (
                    experts,
                    expert_intermediate,
                    hidden,
                ),
                prefix + "ffn_down_exps.weight": (
                    experts,
                    hidden,
                    expert_intermediate,
                ),
            }
        )

        fused_weight = prefix + "attn_qkv.weight"
        split_weights = {
            prefix + "attn_q.weight": (q_width, hidden),
            prefix + "attn_k.weight": (kv_width, hidden),
            prefix + "attn_v.weight": (kv_width, hidden),
        }
        has_fused = fused_weight in actual
        present_split = set(split_weights) & set(actual)
        if has_fused == bool(present_split) or (
            present_split and present_split != set(split_weights)
        ):
            raise ValueError(
                f"smallthinker layer {layer} must contain exactly one complete QKV "
                "layout: fused attn_qkv or split attn_q/attn_k/attn_v"
            )
        if has_fused:
            required[fused_weight] = (q_width + 2 * kv_width, hidden)
            selected_biases = {prefix + "attn_qkv.bias": (q_width + 2 * kv_width,)}
            alternate_biases = {
                prefix + "attn_q.bias",
                prefix + "attn_k.bias",
                prefix + "attn_v.bias",
            }
        else:
            required.update(split_weights)
            selected_biases = {
                prefix + "attn_q.bias": (q_width,),
                prefix + "attn_k.bias": (kv_width,),
                prefix + "attn_v.bias": (kv_width,),
            }
            alternate_biases = {prefix + "attn_qkv.bias"}
        if alternate_biases & set(actual):
            raise ValueError(
                f"smallthinker layer {layer} QKV bias layout does not match its weights"
            )
        present_biases = set(selected_biases) & set(actual)
        if present_biases and present_biases != set(selected_biases):
            raise ValueError(
                f"smallthinker layer {layer} has a partial attention Q/K/V bias set"
            )
        optional.update(selected_biases)
        all_projection_biases.update(selected_biases)

    present_biases = all_projection_biases & set(actual)
    if present_biases and present_biases != all_projection_biases:
        raise ValueError(
            "smallthinker attention Q/K/V biases must be present in every layer "
            "or absent entirely"
        )

    allowed = set(required) | set(optional)
    missing = sorted(set(required) - set(actual))
    unexpected = sorted(set(actual) - allowed)
    malformed = {
        name: (required.get(name, optional.get(name)), actual[name])
        for name in allowed & set(actual)
        if actual[name] != required.get(name, optional.get(name))
    }
    out_of_range = sorted(
        name
        for name in actual
        if (match := re.match(r"^blk\.(\d+)\.", name)) and int(match.group(1)) >= layers
    )
    if missing or unexpected or malformed or out_of_range:
        raise ValueError(
            "Invalid smallthinker GGUF tensor closure: "
            f"missing={missing}, unexpected={unexpected}, malformed={malformed}, "
            f"out_of_range={out_of_range}"
        )

    unsupported_qtypes = {}
    for name, qtype in qtypes.items():
        if name not in allowed:
            continue
        quant_spec = get_quant_spec(qtype)
        if quant_spec is None or quant_spec.dequantize is not Support.SUPPORTED:
            unsupported_qtypes[name] = getattr(qtype, "name", str(qtype))
    if unsupported_qtypes:
        raise ValueError(
            "smallthinker GGUF contains qtypes without a supported float "
            f"dequantization route: {unsupported_qtypes}"
        )

    float_types = float_storage_type_ids()
    non_float_norms = sorted(
        name
        for name, qtype in qtypes.items()
        if name.endswith(("attn_norm.weight", "ffn_norm.weight", "output_norm.weight"))
        and getattr(qtype, "value", qtype) not in float_types
    )
    if non_float_norms:
        raise ValueError(
            "smallthinker normalization tensors must use F32/F16/BF16 storage: "
            f"{non_float_norms}"
        )


def _raise_for_invalid_conventional_moe_tensor_contract(gguf_model) -> None:
    """Validate the exact BailingMoE/DeepSeek/Dots1 tensor closure."""
    from mobius.integrations.gguf._config_mapping import (
        _validate_conventional_moe_rope_scaling,
    )
    from mobius.integrations.gguf._tensor_mapping import is_known_skip

    architecture = gguf_model.architecture
    if architecture not in {"bailingmoe", "deepseek", "dots1"}:
        return

    metadata = gguf_model.metadata
    _validate_conventional_moe_rope_scaling(metadata, architecture)
    required_suffixes = (
        "context_length",
        "embedding_length",
        "feed_forward_length",
        "block_count",
        "attention.head_count",
        "attention.layer_norm_rms_epsilon",
    )
    missing_metadata = [
        f"{architecture}.{suffix}"
        for suffix in required_suffixes
        if f"{architecture}.{suffix}" not in metadata
    ]
    if missing_metadata:
        raise ValueError(
            f"{architecture} GGUF is missing required MoE metadata: {missing_metadata}"
        )

    hidden = int(metadata[f"{architecture}.embedding_length"])
    intermediate = int(metadata[f"{architecture}.feed_forward_length"])
    layers = int(metadata[f"{architecture}.block_count"])
    dense_prefix = int(metadata.get(f"{architecture}.leading_dense_block_count", 0))
    has_routed_layers = dense_prefix < layers
    if has_routed_layers:
        expert_suffixes = (
            "expert_count",
            "expert_used_count",
            "expert_feed_forward_length",
            "expert_shared_count",
        )
        missing_expert_metadata = [
            f"{architecture}.{suffix}"
            for suffix in expert_suffixes
            if f"{architecture}.{suffix}" not in metadata
        ]
        if missing_expert_metadata:
            raise ValueError(
                f"{architecture} GGUF is missing required MoE metadata: "
                f"{missing_expert_metadata}"
            )
    heads = int(metadata[f"{architecture}.attention.head_count"])
    kv_heads = int(metadata.get(f"{architecture}.attention.head_count_kv", heads))
    experts = int(metadata.get(f"{architecture}.expert_count", 0))
    top_k = int(metadata.get(f"{architecture}.expert_used_count", 0))
    expert_intermediate = int(metadata.get(f"{architecture}.expert_feed_forward_length", 0))
    shared_experts = int(metadata.get(f"{architecture}.expert_shared_count", 0))
    context = int(metadata[f"{architecture}.context_length"])
    vocab = int(metadata.get(f"{architecture}.vocab_size", 0)) or len(
        metadata.get("tokenizer.ggml.tokens", ())
    )
    if (
        min(
            hidden,
            intermediate,
            layers,
            heads,
            kv_heads,
            context,
            vocab,
        )
        <= 0
        or hidden % heads
        or heads % kv_heads
        or not 0 <= dense_prefix <= layers
        or (architecture == "bailingmoe" and dense_prefix)
        or (
            has_routed_layers
            and (
                min(experts, top_k, expert_intermediate, shared_experts) <= 0
                or top_k > experts
            )
        )
    ):
        raise ValueError(f"{architecture} GGUF has invalid conventional MoE geometry")

    head_dim = int(metadata.get(f"{architecture}.attention.key_length", hidden // heads))
    value_dim = int(metadata.get(f"{architecture}.attention.value_length", head_dim))
    rope_dim = int(metadata.get(f"{architecture}.rope.dimension_count", head_dim))
    if (
        head_dim <= 0
        or value_dim != head_dim
        or heads * head_dim != hidden
        or rope_dim <= 0
        or rope_dim != head_dim
        or (architecture == "dots1" and kv_heads != heads)
    ):
        raise ValueError(f"{architecture} GGUF has invalid attention geometry")
    norm_epsilon = float(metadata[f"{architecture}.attention.layer_norm_rms_epsilon"])
    if not math.isfinite(norm_epsilon) or norm_epsilon <= 0:
        raise ValueError(f"{architecture} GGUF has invalid normalization or routing scale")
    if has_routed_layers:
        route_scale = float(metadata.get(f"{architecture}.expert_weights_scale", 1.0))
        if math.isclose(route_scale, 0.0):
            route_scale = 1.0
        if not math.isfinite(route_scale) or route_scale <= 0:
            raise ValueError(f"{architecture} GGUF has invalid normalization or routing scale")

    actual = {
        name: tuple(int(dimension) for dimension in shape)
        for name, _raw, _qtype, shape in gguf_model.tensor_items_raw()
        if not is_known_skip(name)
    }
    required: dict[str, tuple[int, ...]] = {
        "token_embd.weight": (vocab, hidden),
        "output_norm.weight": (hidden,),
    }
    optional: dict[str, tuple[int, ...]] = {}
    if architecture == "deepseek":
        optional["output.weight"] = (vocab, hidden)
    else:
        required["output.weight"] = (vocab, hidden)

    q_width = heads * head_dim
    kv_width = kv_heads * head_dim
    shared_width = shared_experts * expert_intermediate
    all_qkv_biases: set[str] = set()
    routed_correction_biases: set[str] = set()
    for layer in range(layers):
        prefix = f"blk.{layer}."
        required.update(
            {
                prefix + "attn_norm.weight": (hidden,),
                prefix + "attn_output.weight": (hidden, q_width),
                prefix + "ffn_norm.weight": (hidden,),
            }
        )
        fused_weight = prefix + "attn_qkv.weight"
        split_weights = {
            prefix + "attn_q.weight": (q_width, hidden),
            prefix + "attn_k.weight": (kv_width, hidden),
            prefix + "attn_v.weight": (kv_width, hidden),
        }
        has_fused = fused_weight in actual
        present_split = set(split_weights) & set(actual)
        if has_fused == bool(present_split) or (
            present_split and present_split != set(split_weights)
        ):
            raise ValueError(
                f"{architecture} layer {layer} must contain exactly one complete QKV "
                "layout: fused attn_qkv or split attn_q/attn_k/attn_v"
            )
        if has_fused:
            required[fused_weight] = (q_width + 2 * kv_width, hidden)
            selected_biases = {prefix + "attn_qkv.bias": (q_width + 2 * kv_width,)}
            alternate_biases = {
                prefix + "attn_q.bias",
                prefix + "attn_k.bias",
                prefix + "attn_v.bias",
            }
        else:
            required.update(split_weights)
            selected_biases = {
                prefix + "attn_q.bias": (q_width,),
                prefix + "attn_k.bias": (kv_width,),
                prefix + "attn_v.bias": (kv_width,),
            }
            alternate_biases = {prefix + "attn_qkv.bias"}
        if alternate_biases & set(actual):
            raise ValueError(
                f"{architecture} layer {layer} QKV bias layout does not match its weights"
            )
        present_biases = set(selected_biases) & set(actual)
        if present_biases and present_biases != set(selected_biases):
            raise ValueError(
                f"{architecture} layer {layer} has a partial attention Q/K/V projection bias set"
            )
        optional.update(selected_biases)
        all_qkv_biases.update(selected_biases)

        if architecture == "dots1":
            required.update(
                {
                    prefix + "attn_q_norm.weight": (head_dim,),
                    prefix + "attn_k_norm.weight": (head_dim,),
                }
            )
        if layer < dense_prefix:
            required.update(
                {
                    prefix + "ffn_gate.weight": (intermediate, hidden),
                    prefix + "ffn_up.weight": (intermediate, hidden),
                    prefix + "ffn_down.weight": (hidden, intermediate),
                }
            )
            continue

        required.update(
            {
                prefix + "ffn_gate_inp.weight": (experts, hidden),
                prefix + "ffn_gate_exps.weight": (
                    experts,
                    expert_intermediate,
                    hidden,
                ),
                prefix + "ffn_up_exps.weight": (
                    experts,
                    expert_intermediate,
                    hidden,
                ),
                prefix + "ffn_down_exps.weight": (
                    experts,
                    hidden,
                    expert_intermediate,
                ),
                prefix + "ffn_gate_shexp.weight": (shared_width, hidden),
                prefix + "ffn_up_shexp.weight": (shared_width, hidden),
                prefix + "ffn_down_shexp.weight": (hidden, shared_width),
            }
        )
        if architecture == "dots1":
            bias_name = prefix + "exp_probs_b.bias"
            optional[bias_name] = (experts,)
            routed_correction_biases.add(bias_name)
        for projection in ("gate", "up", "down"):
            stem = prefix + f"ffn_{projection}_exps"
            optional[stem + ".scale"] = (experts,)
            optional[stem + ".input_scale"] = (experts,)

    present_qkv_biases = all_qkv_biases & set(actual)
    if present_qkv_biases and present_qkv_biases != all_qkv_biases:
        raise ValueError(
            f"{architecture} attention Q/K/V projection biases must be present in every "
            "layer or absent entirely"
        )
    present_correction_biases = routed_correction_biases & set(actual)
    if present_correction_biases and present_correction_biases != routed_correction_biases:
        raise ValueError(
            "dots1 correction bias must be present in every routed layer or absent entirely"
        )

    allowed = set(required) | set(optional)
    out_of_range = sorted(
        name
        for name in actual
        if (match := re.match(r"^blk\.(\d+)\.", name)) and int(match.group(1)) >= layers
    )
    missing = sorted(set(required) - set(actual))
    unexpected = sorted(set(actual) - allowed)
    malformed = {
        name: (required.get(name, optional.get(name)), actual[name])
        for name in allowed & set(actual)
        if actual[name] != required.get(name, optional.get(name))
    }
    if missing or unexpected or malformed or out_of_range:
        raise ValueError(
            f"Invalid {architecture} GGUF tensor closure: missing={missing}, "
            f"unexpected={unexpected}, malformed={malformed}, out_of_range={out_of_range}"
        )


def _raise_for_invalid_moe_cohort_tensor_contract(gguf_model) -> None:
    """Validate exact tensor ownership for dedicated GGUF MoE cohort graphs."""
    from mobius.integrations.gguf._tensor_mapping import is_known_skip

    architecture = gguf_model.architecture
    if architecture not in {"arctic", "dbrx", "ernie4_5-moe", "nomic-bert-moe"}:
        return

    metadata = gguf_model.metadata
    required_suffixes: tuple[str, ...]
    required: dict[str, tuple[int, ...]]
    optional: dict[str, tuple[int, ...]]
    output_biases: set[str]
    if architecture == "ernie4_5-moe":
        required_suffixes = (
            "context_length",
            "embedding_length",
            "feed_forward_length",
            "block_count",
            "attention.head_count",
            "attention.layer_norm_rms_epsilon",
            "expert_count",
            "expert_used_count",
            "expert_feed_forward_length",
            "interleave_moe_layer_step",
        )
        missing_metadata = [
            f"{architecture}.{suffix}"
            for suffix in required_suffixes
            if f"{architecture}.{suffix}" not in metadata
        ]
        if missing_metadata:
            raise ValueError(
                f"{architecture} GGUF is missing required MoE metadata: {missing_metadata}"
            )
        hidden = int(metadata[f"{architecture}.embedding_length"])
        dense_intermediate = int(metadata[f"{architecture}.feed_forward_length"])
        expert_intermediate = int(metadata[f"{architecture}.expert_feed_forward_length"])
        layers = int(metadata[f"{architecture}.block_count"])
        heads = int(metadata[f"{architecture}.attention.head_count"])
        kv_heads = int(metadata.get(f"{architecture}.attention.head_count_kv", heads))
        experts = int(metadata[f"{architecture}.expert_count"])
        top_k = int(metadata[f"{architecture}.expert_used_count"])
        frequency = int(metadata[f"{architecture}.interleave_moe_layer_step"])
        dense_prefix = int(metadata.get(f"{architecture}.leading_dense_block_count", 0))
        vocab = int(metadata.get(f"{architecture}.vocab_size", 0)) or len(
            metadata.get("tokenizer.ggml.tokens", ())
        )
        routed_layers = (
            [
                layer
                for layer in range(layers)
                if layer >= dense_prefix and (layer + 1) % frequency == 0
            ]
            if frequency > 0
            else []
        )
        if (
            min(
                hidden,
                dense_intermediate,
                expert_intermediate,
                layers,
                heads,
                kv_heads,
                experts,
                top_k,
                vocab,
                frequency,
            )
            <= 0
            or hidden % heads
            or heads % kv_heads
            or top_k > experts
            or not 0 <= dense_prefix <= layers
            or not routed_layers
        ):
            raise ValueError(f"{architecture} GGUF has invalid MoE geometry")
        eps = float(metadata[f"{architecture}.attention.layer_norm_rms_epsilon"])
        if not math.isfinite(eps) or eps <= 0:
            raise ValueError(f"{architecture} GGUF has invalid normalization epsilon")

        actual = {
            name: tuple(int(dimension) for dimension in shape)
            for name, _raw, _qtype, shape in gguf_model.tensor_items_raw()
            if not is_known_skip(name)
        }
        from mobius.integrations.gguf._config_mapping import (
            _ernie45_shared_expert_width,
        )

        shared_intermediate, _shared_count = _ernie45_shared_expert_width(
            metadata,
            actual,
            routed_layers,
        )
        required = {
            "token_embd.weight": (vocab, hidden),
            "output_norm.weight": (hidden,),
        }
        optional = {"output.weight": (vocab, hidden)}
        correction_biases: set[str] = set()
        head_dim = hidden // heads
        q_width = heads * head_dim
        kv_width = kv_heads * head_dim
        routed_set = set(routed_layers)
        for layer in range(layers):
            prefix = f"blk.{layer}."
            required.update(
                {
                    prefix + "attn_norm.weight": (hidden,),
                    prefix + "attn_q.weight": (q_width, hidden),
                    prefix + "attn_k.weight": (kv_width, hidden),
                    prefix + "attn_v.weight": (kv_width, hidden),
                    prefix + "attn_output.weight": (hidden, q_width),
                    prefix + "ffn_norm.weight": (hidden,),
                }
            )
            if layer not in routed_set:
                required.update(
                    {
                        prefix + "ffn_gate.weight": (dense_intermediate, hidden),
                        prefix + "ffn_up.weight": (dense_intermediate, hidden),
                        prefix + "ffn_down.weight": (hidden, dense_intermediate),
                    }
                )
                continue
            required.update(
                {
                    prefix + "ffn_gate_inp.weight": (experts, hidden),
                    prefix + "ffn_gate_exps.weight": (
                        experts,
                        expert_intermediate,
                        hidden,
                    ),
                    prefix + "ffn_up_exps.weight": (
                        experts,
                        expert_intermediate,
                        hidden,
                    ),
                    prefix + "ffn_down_exps.weight": (
                        experts,
                        hidden,
                        expert_intermediate,
                    ),
                }
            )
            correction_biases.add(prefix + "exp_probs_b.bias")
            optional[prefix + "exp_probs_b.bias"] = (experts,)
            if shared_intermediate is not None:
                required.update(
                    {
                        prefix + "ffn_gate_shexp.weight": (shared_intermediate, hidden),
                        prefix + "ffn_up_shexp.weight": (shared_intermediate, hidden),
                        prefix + "ffn_down_shexp.weight": (hidden, shared_intermediate),
                    }
                )
        present_correction_biases = correction_biases & set(actual)
        if present_correction_biases and present_correction_biases != correction_biases:
            raise ValueError(
                f"{architecture} correction bias tensors must be complete across routed layers"
            )
        allowed = set(required) | set(optional)
        missing = sorted(set(required) - set(actual))
        unexpected = sorted(set(actual) - allowed)
        malformed = {
            name: (required.get(name, optional.get(name)), actual[name])
            for name in allowed & set(actual)
            if actual[name] != required.get(name, optional.get(name))
        }
        if missing or unexpected or malformed:
            raise ValueError(
                f"Invalid {architecture} GGUF tensor closure: missing={missing}, "
                f"unexpected={unexpected}, malformed={malformed}"
            )
        return

    elif architecture == "dbrx":
        required_suffixes = (
            "context_length",
            "embedding_length",
            "feed_forward_length",
            "block_count",
            "attention.head_count",
            "attention.layer_norm_epsilon",
            "attention.clamp_kqv",
            "expert_count",
            "expert_used_count",
        )
        missing_metadata = [
            f"{architecture}.{suffix}"
            for suffix in required_suffixes
            if f"{architecture}.{suffix}" not in metadata
        ]
        if missing_metadata:
            raise ValueError(
                f"{architecture} GGUF is missing required MoE metadata: {missing_metadata}"
            )
        hidden = int(metadata[f"{architecture}.embedding_length"])
        intermediate = int(metadata[f"{architecture}.feed_forward_length"])
        layers = int(metadata[f"{architecture}.block_count"])
        heads = int(metadata[f"{architecture}.attention.head_count"])
        kv_heads = int(metadata.get(f"{architecture}.attention.head_count_kv", heads))
        experts = int(metadata[f"{architecture}.expert_count"])
        top_k = int(metadata[f"{architecture}.expert_used_count"])
        vocab = int(metadata.get(f"{architecture}.vocab_size", 0)) or len(
            metadata.get("tokenizer.ggml.tokens", ())
        )
        if (
            min(hidden, intermediate, layers, heads, kv_heads, experts, top_k, vocab) <= 0
            or hidden % heads
            or heads % kv_heads
            or top_k > experts
        ):
            raise ValueError(f"{architecture} GGUF has invalid MoE geometry")
        head_dim = hidden // heads
        query_width = heads * head_dim
        kv_width = kv_heads * head_dim
        qkv_width = query_width + 2 * kv_width

        actual = {
            name: tuple(int(dimension) for dimension in shape)
            for name, _raw, _qtype, shape in gguf_model.tensor_items_raw()
            if not is_known_skip(name)
        }
        required = {
            "token_embd.weight": (vocab, hidden),
            "output_norm.weight": (hidden,),
            "output.weight": (vocab, hidden),
        }
        for layer in range(layers):
            prefix = f"blk.{layer}."
            required.update(
                {
                    prefix + "attn_qkv.weight": (qkv_width, hidden),
                    prefix + "attn_output.weight": (hidden, query_width),
                    prefix + "attn_norm.weight": (hidden,),
                    prefix + "attn_output_norm.weight": (hidden,),
                    prefix + "ffn_gate_inp.weight": (experts, hidden),
                    prefix + "ffn_gate_exps.weight": (experts, intermediate, hidden),
                    prefix + "ffn_up_exps.weight": (experts, intermediate, hidden),
                    prefix + "ffn_down_exps.weight": (experts, hidden, intermediate),
                }
            )
        missing = sorted(set(required) - set(actual))
        unexpected = sorted(set(actual) - set(required))
        malformed = {
            name: (required[name], actual[name])
            for name in set(required) & set(actual)
            if actual[name] != required[name]
        }
        if missing or unexpected or malformed:
            raise ValueError(
                f"Invalid {architecture} GGUF tensor closure: missing={missing}, "
                f"unexpected={unexpected}, malformed={malformed}"
            )
        return

    elif architecture == "arctic":
        required_suffixes = (
            "context_length",
            "embedding_length",
            "feed_forward_length",
            "block_count",
            "attention.head_count",
            "attention.layer_norm_rms_epsilon",
            "expert_count",
            "expert_used_count",
        )
        missing_metadata = [
            f"{architecture}.{suffix}"
            for suffix in required_suffixes
            if f"{architecture}.{suffix}" not in metadata
        ]
        if missing_metadata:
            raise ValueError(
                f"{architecture} GGUF is missing required MoE metadata: {missing_metadata}"
            )
        hidden = int(metadata[f"{architecture}.embedding_length"])
        expert_intermediate = int(metadata[f"{architecture}.feed_forward_length"])
        layers = int(metadata[f"{architecture}.block_count"])
        heads = int(metadata[f"{architecture}.attention.head_count"])
        kv_heads = int(metadata.get(f"{architecture}.attention.head_count_kv", heads))
        experts = int(metadata[f"{architecture}.expert_count"])
        top_k = int(metadata[f"{architecture}.expert_used_count"])
        vocab = int(metadata.get(f"{architecture}.vocab_size", 0)) or len(
            metadata.get("tokenizer.ggml.tokens", ())
        )
        if (
            min(hidden, expert_intermediate, layers, heads, kv_heads, experts, top_k, vocab)
            <= 0
            or hidden % heads
            or heads % kv_heads
            or top_k > experts
        ):
            raise ValueError("arctic GGUF has invalid MoE geometry")
        eps = float(metadata[f"{architecture}.attention.layer_norm_rms_epsilon"])
        if not math.isfinite(eps) or eps <= 0:
            raise ValueError("arctic GGUF has invalid normalization epsilon")
        actual = {
            name: tuple(int(dimension) for dimension in shape)
            for name, _raw, _qtype, shape in gguf_model.tensor_items_raw()
            if not is_known_skip(name)
        }
        required = {
            "token_embd.weight": (vocab, hidden),
            "output_norm.weight": (hidden,),
        }
        optional = {"output.weight": (vocab, hidden)}
        head_dim = hidden // heads
        q_width = heads * head_dim
        kv_width = kv_heads * head_dim
        for layer in range(layers):
            prefix = f"blk.{layer}."
            required.update(
                {
                    prefix + "attn_norm.weight": (hidden,),
                    prefix + "attn_q.weight": (q_width, hidden),
                    prefix + "attn_k.weight": (kv_width, hidden),
                    prefix + "attn_v.weight": (kv_width, hidden),
                    prefix + "attn_output.weight": (hidden, q_width),
                    prefix + "ffn_norm.weight": (hidden,),
                    prefix + "ffn_gate.weight": (hidden, hidden),
                    prefix + "ffn_up.weight": (hidden, hidden),
                    prefix + "ffn_down.weight": (hidden, hidden),
                    prefix + "ffn_norm_exps.weight": (hidden,),
                    prefix + "ffn_gate_inp.weight": (experts, hidden),
                    prefix + "ffn_gate_exps.weight": (
                        experts,
                        expert_intermediate,
                        hidden,
                    ),
                    prefix + "ffn_up_exps.weight": (
                        experts,
                        expert_intermediate,
                        hidden,
                    ),
                    prefix + "ffn_down_exps.weight": (
                        experts,
                        hidden,
                        expert_intermediate,
                    ),
                }
            )
        allowed = set(required) | set(optional)
        missing = sorted(set(required) - set(actual))
        unexpected = sorted(set(actual) - allowed)
        malformed = {
            name: (required.get(name, optional.get(name)), actual[name])
            for name in allowed & set(actual)
            if actual[name] != required.get(name, optional.get(name))
        }
        if missing or unexpected or malformed:
            raise ValueError(
                f"Invalid arctic GGUF tensor closure: missing={missing}, "
                f"unexpected={unexpected}, malformed={malformed}"
            )
        return

    elif architecture == "nomic-bert-moe":
        required_suffixes = (
            "context_length",
            "embedding_length",
            "feed_forward_length",
            "block_count",
            "attention.head_count",
            "attention.layer_norm_epsilon",
            "attention.causal",
            "moe_every_n_layers",
            "expert_count",
            "expert_used_count",
            "rope.freq_base",
        )
        missing_metadata = [
            f"{architecture}.{suffix}"
            for suffix in required_suffixes
            if f"{architecture}.{suffix}" not in metadata
        ]
        if missing_metadata:
            raise ValueError(
                f"{architecture} GGUF is missing required MoE metadata: {missing_metadata}"
            )
        hidden = int(metadata[f"{architecture}.embedding_length"])
        intermediate = int(metadata[f"{architecture}.feed_forward_length"])
        layers = int(metadata[f"{architecture}.block_count"])
        heads = int(metadata[f"{architecture}.attention.head_count"])
        kv_heads = int(metadata.get(f"{architecture}.attention.head_count_kv", heads))
        experts = int(metadata[f"{architecture}.expert_count"])
        top_k = int(metadata[f"{architecture}.expert_used_count"])
        frequency = int(metadata[f"{architecture}.moe_every_n_layers"])
        token_types = int(metadata.get("tokenizer.ggml.token_type_count", 0))
        if token_types <= 0:
            raise ValueError(
                "nomic-bert-moe requires a positive tokenizer.ggml.token_type_count"
            )
        vocab = int(metadata.get(f"{architecture}.vocab_size", 0)) or len(
            metadata.get("tokenizer.ggml.tokens", ())
        )
        if (
            min(
                hidden,
                intermediate,
                layers,
                heads,
                kv_heads,
                experts,
                top_k,
                vocab,
            )
            <= 0
            or hidden % heads
            or kv_heads != heads
            or top_k > experts
            or frequency < 2
            or bool(metadata[f"{architecture}.attention.causal"])
        ):
            raise ValueError(f"{architecture} GGUF has invalid encoder MoE geometry")
        eps = float(metadata[f"{architecture}.attention.layer_norm_epsilon"])
        if not math.isfinite(eps) or eps <= 0:
            raise ValueError(f"{architecture} GGUF has invalid normalization epsilon")

        actual = {
            name: tuple(int(dimension) for dimension in shape)
            for name, _raw, _qtype, shape in gguf_model.tensor_items_raw()
            if not is_known_skip(name)
        }
        required = {
            "token_embd.weight": (vocab, hidden),
            "token_embd_norm.weight": (hidden,),
            "token_embd_norm.bias": (hidden,),
        }
        optional = {
            # llama.cpp stores NomicBERT-MoE's only token-type row as a vector.
            "token_types.weight": (hidden,),
        }
        qkv_biases: set[str] = set()
        output_biases = set()
        dense_up_biases: set[str] = set()
        dense_down_biases: set[str] = set()
        for layer in range(layers):
            prefix = f"blk.{layer}."
            required.update(
                {
                    prefix + "attn_qkv.weight": (3 * hidden, hidden),
                    prefix + "attn_output.weight": (hidden, hidden),
                    prefix + "attn_output_norm.weight": (hidden,),
                    prefix + "attn_output_norm.bias": (hidden,),
                    prefix + "layer_output_norm.weight": (hidden,),
                    prefix + "layer_output_norm.bias": (hidden,),
                }
            )
            qkv_biases.add(prefix + "attn_qkv.bias")
            output_biases.add(prefix + "attn_output.bias")
            optional[prefix + "attn_qkv.bias"] = (3 * hidden,)
            optional[prefix + "attn_output.bias"] = (hidden,)
            if layer % frequency == 1:
                required.update(
                    {
                        prefix + "ffn_gate_inp.weight": (experts, hidden),
                        prefix + "ffn_up_exps.weight": (experts, intermediate, hidden),
                        prefix + "ffn_down_exps.weight": (experts, hidden, intermediate),
                    }
                )
            else:
                required.update(
                    {
                        prefix + "ffn_up.weight": (intermediate, hidden),
                        prefix + "ffn_down.weight": (hidden, intermediate),
                    }
                )
                dense_up_biases.add(prefix + "ffn_up.bias")
                dense_down_biases.add(prefix + "ffn_down.bias")
                optional[prefix + "ffn_up.bias"] = (intermediate,)
                optional[prefix + "ffn_down.bias"] = (hidden,)

        for label, family in (
            ("fused QKV bias", qkv_biases),
            ("attention output bias", output_biases),
            ("dense FFN up bias", dense_up_biases),
            ("dense FFN down bias", dense_down_biases),
        ):
            present = family & set(actual)
            if present and present != family:
                raise ValueError(
                    f"{architecture} {label} tensors must be present for every applicable layer"
                )
        allowed = set(required) | set(optional)
        missing = sorted(set(required) - set(actual))
        unexpected = sorted(set(actual) - allowed)
        malformed = {
            name: (required.get(name, optional.get(name)), actual[name])
            for name in allowed & set(actual)
            if actual[name] != required.get(name, optional.get(name))
        }
        if missing or unexpected or malformed:
            raise ValueError(
                f"Invalid {architecture} GGUF tensor closure: missing={missing}, "
                f"unexpected={unexpected}, malformed={malformed}"
            )
        return


def _raise_for_invalid_granite_tensor_contract(gguf_model) -> None:
    """Validate Granite's architecture-wide dense-or-MoE tensor union."""
    if gguf_model.architecture != "granite":
        return

    from mobius.integrations.gguf._tensor_mapping import is_known_skip

    metadata = gguf_model.metadata
    arch = "granite"
    required_suffixes = (
        "context_length",
        "embedding_length",
        "feed_forward_length",
        "block_count",
        "attention.head_count",
        "attention.layer_norm_rms_epsilon",
        "logit_scale",
    )
    missing_metadata = [
        f"{arch}.{suffix}"
        for suffix in required_suffixes
        if f"{arch}.{suffix}" not in metadata
    ]
    if missing_metadata:
        raise ValueError(f"granite GGUF is missing required metadata: {missing_metadata}")

    hidden = int(metadata[f"{arch}.embedding_length"])
    intermediate = int(metadata[f"{arch}.feed_forward_length"])
    layers = int(metadata[f"{arch}.block_count"])
    heads = int(metadata[f"{arch}.attention.head_count"])
    kv_heads = int(metadata.get(f"{arch}.attention.head_count_kv", heads))
    context = int(metadata[f"{arch}.context_length"])
    vocab = int(metadata.get(f"{arch}.vocab_size", 0)) or len(
        metadata.get("tokenizer.ggml.tokens", ())
    )
    experts = int(metadata.get(f"{arch}.expert_count", 0))
    top_k = int(metadata.get(f"{arch}.expert_used_count", 0))
    raw_expert_intermediate = metadata.get(f"{arch}.expert_feed_forward_length")
    if raw_expert_intermediate is not None:
        if isinstance(raw_expert_intermediate, (bool, np.bool_)):
            serialized_expert_intermediate = None
        elif isinstance(raw_expert_intermediate, (int, np.integer)):
            serialized_expert_intermediate = int(raw_expert_intermediate)
        elif isinstance(raw_expert_intermediate, (float, np.floating)):
            numeric_expert_intermediate = float(raw_expert_intermediate)
            serialized_expert_intermediate = (
                int(numeric_expert_intermediate)
                if math.isfinite(numeric_expert_intermediate)
                and numeric_expert_intermediate.is_integer()
                else None
            )
        else:
            serialized_expert_intermediate = None
        if serialized_expert_intermediate != intermediate:
            raise ValueError(
                "granite.expert_feed_forward_length must equal feed_forward_length "
                "because the pinned loader sizes routed experts from feed_forward_length"
            )
    expert_intermediate = intermediate
    shared_width = int(metadata.get(f"{arch}.expert_shared_feed_forward_length", 0))
    if (
        min(hidden, intermediate, layers, heads, kv_heads, context, vocab) <= 0
        or hidden % heads
        or heads % kv_heads
        or experts < 0
        or top_k < 0
        or bool(experts) != bool(top_k)
        or (experts and (top_k > experts or expert_intermediate <= 0))
        or shared_width < 0
        or (not experts and shared_width)
    ):
        raise ValueError("granite GGUF has invalid dense/MoE geometry")

    head_dim = hidden // heads
    key_dim = int(metadata.get(f"{arch}.attention.key_length", head_dim))
    value_dim = int(metadata.get(f"{arch}.attention.value_length", head_dim))
    rope_dim = int(metadata.get(f"{arch}.rope.dimension_count", head_dim))
    if key_dim != head_dim or value_dim != head_dim or rope_dim != head_dim or rope_dim % 2:
        raise ValueError(
            "granite attention key/value/rotary dimensions must equal embedding_length / "
            "attention.head_count"
        )

    epsilon = float(metadata[f"{arch}.attention.layer_norm_rms_epsilon"])
    logit_scale = float(metadata[f"{arch}.logit_scale"])
    optional_scales = [
        float(metadata.get(f"{arch}.{suffix}", 0.0))
        for suffix in ("embedding_scale", "residual_scale", "attention.scale")
    ]
    if (
        not math.isfinite(epsilon)
        or epsilon <= 0
        or not math.isfinite(logit_scale)
        or math.isclose(logit_scale, 0.0, rel_tol=0.0, abs_tol=0.0)
        or not all(math.isfinite(value) for value in optional_scales)
    ):
        raise ValueError("granite GGUF has invalid normalization or scaling metadata")

    scaling_type = metadata.get(f"{arch}.rope.scaling.type")
    supported_scaling_types = {None, "", "none"}
    if scaling_type not in supported_scaling_types:
        raise ValueError(f"granite GGUF has unsupported rope.scaling.type={scaling_type!r}")

    deepstack = metadata.get(f"{arch}.deepstack_mapping")
    if deepstack is not None:
        if not isinstance(deepstack, (list, tuple, np.ndarray)):
            raise ValueError("granite.deepstack_mapping must be an integer array")
        mapping = [int(value) for value in deepstack]
        if mapping and len(mapping) != layers:
            raise ValueError("granite.deepstack_mapping must match block_count")
        if any(value != -1 for value in mapping):
            raise NotImplementedError(
                "granite deep-stack embedding injection is unsupported by the text task; "
                "only an absent, empty, or all -1 mapping is accepted"
            )

    raw_items = list(gguf_model.tensor_items_raw())
    actual = {
        name: tuple(int(dimension) for dimension in shape)
        for name, _raw, _qtype, shape in raw_items
        if not is_known_skip(name)
    }
    skipped_shapes = {
        name: tuple(int(dimension) for dimension in shape)
        for name, _raw, _qtype, shape in raw_items
        if is_known_skip(name) and not name.startswith("tokenizer.")
    }
    if "rope_freqs.weight" in skipped_shapes:
        raise ValueError("granite serialized rope_freqs.weight is not representable exactly")
    longrope_names = {"rope_factors_long.weight", "rope_factors_short.weight"}
    present_longrope = longrope_names & set(skipped_shapes)
    if present_longrope:
        raise ValueError(
            "granite tensor-backed LongRoPE is unsupported because the current rotary graph "
            "cannot preserve rope.scaling.attn_factor exactly"
        )

    required: dict[str, tuple[int, ...]] = {
        "token_embd.weight": (vocab, hidden),
        "output_norm.weight": (hidden,),
    }
    optional: dict[str, tuple[int, ...]] = {"output.weight": (vocab, hidden)}
    q_width = hidden
    kv_width = kv_heads * head_dim
    selected_qkv_biases: set[str] = set()
    attention_output_biases: set[str] = set()
    dense_ffn_biases: set[str] = set()
    for layer in range(layers):
        prefix = f"blk.{layer}."
        required.update(
            {
                prefix + "attn_norm.weight": (hidden,),
                prefix + "attn_output.weight": (hidden, q_width),
                prefix + "ffn_norm.weight": (hidden,),
            }
        )
        optional[prefix + "attn_output.bias"] = (hidden,)
        attention_output_biases.add(prefix + "attn_output.bias")

        fused_weight = prefix + "attn_qkv.weight"
        split_weights = {
            prefix + "attn_q.weight": (q_width, hidden),
            prefix + "attn_k.weight": (kv_width, hidden),
            prefix + "attn_v.weight": (kv_width, hidden),
        }
        has_fused = fused_weight in actual
        present_split = set(split_weights) & set(actual)
        if has_fused == bool(present_split) or (
            present_split and present_split != set(split_weights)
        ):
            raise ValueError(
                f"granite layer {layer} must contain exactly one complete QKV layout"
            )
        if has_fused:
            required[fused_weight] = (q_width + 2 * kv_width, hidden)
            selected_biases = {prefix + "attn_qkv.bias": (q_width + 2 * kv_width,)}
            alternate_biases = {
                prefix + "attn_q.bias",
                prefix + "attn_k.bias",
                prefix + "attn_v.bias",
            }
        else:
            required.update(split_weights)
            selected_biases = {
                prefix + "attn_q.bias": (q_width,),
                prefix + "attn_k.bias": (kv_width,),
                prefix + "attn_v.bias": (kv_width,),
            }
            alternate_biases = {prefix + "attn_qkv.bias"}
        if alternate_biases & set(actual):
            raise ValueError(
                f"granite layer {layer} QKV bias layout does not match its weights"
            )
        present_biases = set(selected_biases) & set(actual)
        if present_biases and present_biases != set(selected_biases):
            raise ValueError(f"granite layer {layer} has a partial QKV bias set")
        optional.update(selected_biases)
        selected_qkv_biases.update(selected_biases)

        if experts:
            required.update(
                {
                    prefix + "ffn_gate_inp.weight": (experts, hidden),
                    prefix + "ffn_gate_exps.weight": (
                        experts,
                        expert_intermediate,
                        hidden,
                    ),
                    prefix + "ffn_up_exps.weight": (
                        experts,
                        expert_intermediate,
                        hidden,
                    ),
                    prefix + "ffn_down_exps.weight": (
                        experts,
                        hidden,
                        expert_intermediate,
                    ),
                }
            )
            if shared_width:
                required.update(
                    {
                        prefix + "ffn_gate_shexp.weight": (shared_width, hidden),
                        prefix + "ffn_up_shexp.weight": (shared_width, hidden),
                        prefix + "ffn_down_shexp.weight": (hidden, shared_width),
                    }
                )
        else:
            dense_weights = {
                prefix + "ffn_gate.weight": (intermediate, hidden),
                prefix + "ffn_up.weight": (intermediate, hidden),
                prefix + "ffn_down.weight": (hidden, intermediate),
            }
            required.update(dense_weights)
            layer_biases = {
                prefix + "ffn_gate.bias": (intermediate,),
                prefix + "ffn_up.bias": (intermediate,),
                prefix + "ffn_down.bias": (hidden,),
            }
            present_biases = set(layer_biases) & set(actual)
            if present_biases and present_biases != set(layer_biases):
                raise ValueError(f"granite layer {layer} has a partial dense FFN bias set")
            optional.update(layer_biases)
            dense_ffn_biases.update(layer_biases)

    actual_names = set(actual)
    for family_name, family in (
        ("QKV projection biases", selected_qkv_biases),
        ("attention output biases", attention_output_biases),
        ("dense FFN biases", dense_ffn_biases),
    ):
        present = family & actual_names
        if present and present != family:
            raise ValueError(
                f"granite {family_name} must be present in every layer or absent entirely"
            )
    allowed = set(required) | set(optional)
    allowed = set(required) | set(optional)
    out_of_range = sorted(
        name
        for name in actual
        if (match := re.match(r"^blk\.(\d+)\.", name)) and int(match.group(1)) >= layers
    )
    missing = sorted(set(required) - actual_names)
    unexpected = sorted(actual_names - allowed)
    malformed = {
        name: (required.get(name, optional.get(name)), actual[name])
        for name in allowed & actual_names
        if actual[name] != required.get(name, optional.get(name))
    }
    if missing or unexpected or malformed or out_of_range:
        raise ValueError(
            "Invalid granite GGUF tensor closure: "
            f"missing={missing}, unexpected={unexpected}, malformed={malformed}, "
            f"out_of_range={out_of_range}"
        )


def _raise_for_invalid_falcon_h1_tensor_contract(gguf_model) -> None:
    """Validate Falcon-H1's complete parallel attention/Mamba2 tensor closure."""
    if gguf_model.architecture != "falcon-h1":
        return

    import numpy as np

    metadata = gguf_model.metadata
    arch = "falcon-h1"
    positive_fields = (
        "context_length",
        "embedding_length",
        "feed_forward_length",
        "block_count",
        "attention.head_count",
        "attention.head_count_kv",
        "attention.key_length",
        "attention.value_length",
        "ssm.conv_kernel",
        "ssm.inner_size",
        "ssm.state_size",
        "ssm.time_step_rank",
        "ssm.group_count",
    )
    required_fields = (*positive_fields, "attention.layer_norm_rms_epsilon", "rope.freq_base")
    missing_metadata = sorted(
        suffix for suffix in required_fields if f"{arch}.{suffix}" not in metadata
    )
    if missing_metadata:
        raise ValueError(
            f"falcon-h1 GGUF is missing required metadata field(s): {missing_metadata}"
        )
    values = {suffix: int(metadata[f"{arch}.{suffix}"]) for suffix in positive_fields}
    invalid = sorted(suffix for suffix, value in values.items() if value <= 0)
    if invalid:
        raise ValueError(f"falcon-h1 GGUF has non-positive metadata field(s): {invalid}")
    rms_epsilon = float(metadata[f"{arch}.attention.layer_norm_rms_epsilon"])
    rope_base = float(metadata[f"{arch}.rope.freq_base"])
    if not np.isfinite(rms_epsilon) or rms_epsilon <= 0:
        raise ValueError("falcon-h1 attention.layer_norm_rms_epsilon must be positive")
    if not np.isfinite(rope_base) or rope_base <= 0:
        raise ValueError("falcon-h1 rope.freq_base must be positive")

    hidden = values["embedding_length"]
    intermediate = values["feed_forward_length"]
    layers = values["block_count"]
    attn_heads = values["attention.head_count"]
    kv_heads = values["attention.head_count_kv"]
    key_dim = values["attention.key_length"]
    value_dim = values["attention.value_length"]
    conv_kernel = values["ssm.conv_kernel"]
    ssm_inner = values["ssm.inner_size"]
    state_size = values["ssm.state_size"]
    ssm_heads = values["ssm.time_step_rank"]
    groups = values["ssm.group_count"]
    if (
        key_dim != value_dim
        or hidden != attn_heads * key_dim
        or attn_heads % kv_heads
        or ssm_inner % ssm_heads
        or ssm_heads % groups
        or ssm_inner % groups
    ):
        raise ValueError("falcon-h1 GGUF has inconsistent attention or SSM geometry")

    vocab = int(metadata.get(f"{arch}.vocab_size", 0))
    if not vocab:
        vocab = len(metadata.get("tokenizer.ggml.tokens", ()))
    if vocab <= 0:
        raise ValueError("falcon-h1 GGUF has no positive vocabulary size")

    actual = {
        name: tuple(int(dimension) for dimension in shape)
        for name, _raw, _qtype, shape in gguf_model.tensor_items_raw()
    }
    required: dict[str, tuple[int, ...]] = {
        "token_embd.weight": (vocab, hidden),
        "output_norm.weight": (hidden,),
    }
    optional: dict[str, tuple[int, ...]] = {
        "output.weight": (vocab, hidden),
    }
    q_dim = attn_heads * key_dim
    kv_dim = kv_heads * key_dim
    conv_dim = ssm_inner + 2 * groups * state_size
    projection_size = 2 * ssm_inner + 2 * groups * state_size + ssm_heads
    for layer in range(layers):
        prefix = f"blk.{layer}."
        required.update(
            {
                prefix + "attn_norm.weight": (hidden,),
                prefix + "attn_q.weight": (q_dim, hidden),
                prefix + "attn_k.weight": (kv_dim, hidden),
                prefix + "attn_v.weight": (kv_dim, hidden),
                prefix + "attn_output.weight": (hidden, q_dim),
                prefix + "ffn_norm.weight": (hidden,),
                prefix + "ffn_gate.weight": (intermediate, hidden),
                prefix + "ffn_up.weight": (intermediate, hidden),
                prefix + "ffn_down.weight": (hidden, intermediate),
                prefix + "ssm_in.weight": (projection_size, hidden),
                prefix + "ssm_conv1d.weight": (conv_dim, conv_kernel),
                prefix + "ssm_dt.bias": (ssm_heads,),
                prefix + "ssm_a": (ssm_heads, 1),
                prefix + "ssm_d": (ssm_heads, 1),
                prefix + "ssm_out.weight": (hidden, ssm_inner),
            }
        )
        optional.update(
            {
                prefix + "attn_q.bias": (q_dim,),
                prefix + "attn_k.bias": (kv_dim,),
                prefix + "attn_v.bias": (kv_dim,),
                prefix + "attn_output.bias": (hidden,),
                prefix + "ffn_gate.bias": (intermediate,),
                prefix + "ffn_up.bias": (intermediate,),
                prefix + "ffn_down.bias": (hidden,),
                prefix + "ssm_conv1d.bias": (conv_dim,),
                prefix + "ssm_norm.weight": (groups, ssm_inner // groups),
                prefix + "rope_freqs.weight": (key_dim // 2,),
            }
        )

    families = {
        "attention projection biases": {
            f"blk.{layer}.attn_{projection}.bias"
            for layer in range(layers)
            for projection in ("q", "k", "v", "output")
        },
        "feed-forward biases": {
            f"blk.{layer}.ffn_{projection}.bias"
            for layer in range(layers)
            for projection in ("gate", "up", "down")
        },
        "Mamba convolution biases": {
            f"blk.{layer}.ssm_conv1d.bias" for layer in range(layers)
        },
        "Mamba RMSNorm weights": {f"blk.{layer}.ssm_norm.weight" for layer in range(layers)},
    }
    actual_names = set(actual)
    for family_name, family in families.items():
        present = family & actual_names
        if present and present != family:
            raise ValueError(
                f"falcon-h1 {family_name} must be present in every layer or absent entirely"
            )

    allowed = set(required) | set(optional)
    missing = sorted(set(required) - actual_names)
    unexpected = sorted(actual_names - allowed)
    malformed = {
        name: (required.get(name, optional.get(name)), actual[name])
        for name in allowed & actual_names
        if actual[name] != required.get(name, optional.get(name))
    }
    if missing or unexpected or malformed:
        raise ValueError(
            "Invalid falcon-h1 GGUF tensor closure: "
            f"missing={missing}, unexpected={unexpected}, malformed={malformed}"
        )

    for layer in range(layers):
        decay_name = f"blk.{layer}.ssm_a"
        decay = np.asarray(gguf_model.get_tensor(decay_name))
        if not np.all(np.isfinite(decay)) or not np.all(decay < 0):
            raise ValueError(
                f"Malformed Falcon-H1 GGUF Mamba decay tensor {decay_name!r}: "
                "ssm_a must contain only finite negative -exp(A_log) values"
            )


def _raise_for_invalid_plamo2_tensor_contract(gguf_model) -> None:
    """Reject any PLaMo2 metadata/tensor topology not implemented exactly."""
    if gguf_model.architecture != "plamo2":
        return

    import numpy as np

    metadata = gguf_model.metadata
    arch = "plamo2"
    if int(gguf_model.format_version) != 3:
        raise ValueError(
            f"PLaMo2 supports only pinned GGUF v3, got v{gguf_model.format_version}"
        )
    layers = int(metadata[f"{arch}.block_count"])
    hidden = int(metadata[f"{arch}.embedding_length"])
    intermediate = int(metadata[f"{arch}.feed_forward_length"])
    inner = int(metadata[f"{arch}.ssm.inner_size"])
    state = int(metadata[f"{arch}.ssm.state_size"])
    conv = int(metadata[f"{arch}.ssm.conv_kernel"])
    ssm_heads = int(metadata[f"{arch}.ssm.time_step_rank"])
    group_count = int(metadata[f"{arch}.ssm.group_count"])
    epsilon = float(metadata[f"{arch}.attention.layer_norm_rms_epsilon"])
    if min(layers, hidden, intermediate, inner, state, conv, ssm_heads) <= 0:
        raise ValueError("PLaMo2 dimensions must all be positive")
    if inner % ssm_heads:
        raise ValueError("PLaMo2 requires an exact SSM head split")
    if group_count != 0:
        raise ValueError("PLaMo2 supports only ssm.group_count=0")
    if not math.isclose(epsilon, 1e-6, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("PLaMo2 requires attention.layer_norm_rms_epsilon=1e-6")
    activation = metadata.get(f"{arch}.feed_forward.activation", "silu")
    if activation not in {"silu", "swiglu"}:
        raise ValueError(
            "PLaMo2 feed_forward.activation must be 'silu' or 'swiglu' "
            f"(both select fused SwiGLU), got {activation!r}"
        )
    if bool(metadata.get(f"{arch}.ssm.use_predefined_initial_state", False)):
        raise ValueError("PLaMo2 predefined initial state is unsupported")

    head_counts = metadata[f"{arch}.attention.head_count"]
    kv_counts = metadata[f"{arch}.attention.head_count_kv"]
    head_counts_are_arrays = isinstance(head_counts, (list, tuple, np.ndarray))
    kv_counts_are_arrays = isinstance(kv_counts, (list, tuple, np.ndarray))
    if head_counts_are_arrays and len(head_counts) != layers:
        raise ValueError("PLaMo2 attention head arrays must match block_count")
    if kv_counts_are_arrays and len(kv_counts) != layers:
        raise ValueError("PLaMo2 attention head arrays must match block_count")
    if kv_counts_are_arrays:
        attention_layers = [int(value) > 0 for value in kv_counts]
    elif head_counts_are_arrays:
        attention_layers = [int(value) > 0 for value in head_counts]
    else:
        # Early llama.cpp PLaMo2 converters serialized the attention dimensions
        # as scalars. The mutually exclusive tensor families still encode the
        # exact layer schedule, so expand it only when every layer is unambiguous.
        tensor_names = set(gguf_model.tensor_names)
        attention_layers = []
        for layer in range(layers):
            has_attention = f"blk.{layer}.attn_qkv.weight" in tensor_names
            has_mamba = f"blk.{layer}.ssm_in.weight" in tensor_names
            if has_attention == has_mamba:
                raise ValueError(
                    "PLaMo2 scalar head metadata requires exactly one attention or "
                    f"Mamba tensor family in layer {layer}"
                )
            attention_layers.append(has_attention)
    if not head_counts_are_arrays:
        attention_heads = int(head_counts)
        if attention_heads <= 0:
            raise ValueError("PLaMo2 scalar attention head_count must be positive")
        head_counts = metadata[f"{arch}.attention.head_count"] = [
            attention_heads if is_attention else 0 for is_attention in attention_layers
        ]
    if not kv_counts_are_arrays:
        attention_kv_heads = int(kv_counts)
        if attention_kv_heads <= 0:
            raise ValueError("PLaMo2 scalar attention head_count_kv must be positive")
        kv_counts = metadata[f"{arch}.attention.head_count_kv"] = [
            attention_kv_heads if is_attention else 0 for is_attention in attention_layers
        ]
    head_counts = [int(value) for value in head_counts]
    kv_counts = [int(value) for value in kv_counts]

    actual_shapes = {
        name: tuple(int(dim) for dim in gguf_model.get_tensor_shape(name))
        for name in gguf_model.tensor_names
    }
    from mobius.integrations.gguf._config_mapping import (
        _infer_plamo2_attention_widths,
    )

    key_width, value_width = _infer_plamo2_attention_widths(
        gguf_model,
        tuple(head_counts),
        tuple(kv_counts),
    )
    required: dict[str, tuple[int, ...]] = {
        "token_embd.weight": (
            int(
                metadata.get(
                    f"{arch}.vocab_size",
                    actual_shapes.get("token_embd.weight", (0,))[0],
                )
            ),
            hidden,
        ),
        "output_norm.weight": (hidden,),
    }
    optional: dict[str, tuple[int, ...]] = {
        "output.weight": (required["token_embd.weight"][0], hidden)
    }
    dt_rank = max(64, hidden // 16)
    for layer, (heads, kv_heads) in enumerate(zip(head_counts, kv_counts)):
        prefix = f"blk.{layer}."
        required.update(
            {
                prefix + "attn_norm.weight": (hidden,),
                prefix + "post_attention_norm": (hidden,),
                prefix + "ffn_norm.weight": (hidden,),
                prefix + "post_ffw_norm": (hidden,),
                prefix + "ffn_up.weight": (2 * intermediate, hidden),
                prefix + "ffn_down.weight": (hidden, intermediate),
            }
        )
        if kv_heads == 0:
            if heads != 0:
                raise ValueError(
                    f"PLaMo2 layer {layer} has head_count={heads} but head_count_kv=0"
                )
            required.update(
                {
                    prefix + "ssm_in.weight": (2 * inner, hidden),
                    prefix + "ssm_conv1d.weight": (inner, conv),
                    prefix + "ssm_x.weight": (2 * state + dt_rank, inner),
                    prefix + "ssm_dt.weight": (ssm_heads, dt_rank),
                    prefix + "ssm_dt.bias": (ssm_heads,),
                    prefix + "ssm_a": (ssm_heads,),
                    prefix + "ssm_d": (ssm_heads,),
                    prefix + "ssm_dt_norm": (dt_rank,),
                    prefix + "ssm_b_norm": (state,),
                    prefix + "ssm_c_norm": (state,),
                    prefix + "ssm_out.weight": (hidden, inner),
                }
            )
        else:
            if heads <= 0 or heads % kv_heads:
                raise ValueError(f"PLaMo2 layer {layer} has invalid attention head geometry")
            qkv_name = prefix + "attn_qkv.weight"
            output_name = prefix + "attn_output.weight"
            required.update(
                {
                    qkv_name: (
                        heads * key_width + kv_heads * key_width + kv_heads * value_width,
                        hidden,
                    ),
                    prefix + "attn_q_norm.weight": (heads, key_width),
                    prefix + "attn_k_norm.weight": (kv_heads, key_width),
                    output_name: (hidden, heads * value_width),
                }
            )

    if key_width != value_width or key_width * max(head_counts) != hidden:
        raise ValueError("PLaMo2 attention widths do not reconstruct embedding_length")
    if inner != ssm_heads * key_width:
        raise ValueError(
            "PLaMo2 ssm.inner_size must equal ssm.time_step_rank * attention key width"
        )
    for suffix, expected in (
        ("attention.key_length", key_width),
        ("attention.value_length", value_width),
    ):
        explicit = metadata.get(f"{arch}.{suffix}")
        if explicit is not None and int(explicit) != expected:
            raise ValueError(f"PLaMo2 {suffix} contradicts exact tensor shapes")

    actual = set(actual_shapes)
    allowed = set(required) | set(optional)
    missing = sorted(set(required) - actual)
    unexpected = sorted(actual - allowed)
    malformed = {
        name: (required.get(name, optional.get(name)), actual_shapes[name])
        for name in allowed & actual
        if actual_shapes[name] != required.get(name, optional.get(name))
    }
    if missing or unexpected or malformed:
        raise ValueError(
            "Invalid PLaMo2 GGUF tensor closure: "
            f"missing={missing}, unexpected={unexpected}, malformed={malformed}"
        )
    for layer, kv_heads in enumerate(kv_counts):
        if kv_heads:
            continue
        decay_name = f"blk.{layer}.ssm_a"
        decay = np.asarray(gguf_model.get_tensor(decay_name))
        if not np.all(np.isfinite(decay)) or not np.all(decay < 0):
            raise ValueError(
                f"Malformed PLaMo2 decay tensor {decay_name!r}: "
                "ssm_a must contain finite negative -exp(A_log) values"
            )


def _raise_for_invalid_hybrid_tensor_contract(gguf_model) -> None:
    """Enforce pinned per-layer mixer closure before any graph is constructed."""
    from mobius.integrations.gguf._config_mapping import _derive_hybrid_layout

    architecture = gguf_model.architecture
    if architecture in {"jamba", "nemotron_h", "nemotron_h_moe", "granitehybrid"}:
        _raise_for_invalid_mamba_hybrid_tensor_contract(gguf_model)
        return
    if architecture not in {"lfm2", "lfm2moe", "qwen35", "qwen35moe", "qwen3next"}:
        return

    metadata = gguf_model.metadata
    trunk_layers, layer_types, mtp_count = _derive_hybrid_layout(architecture, metadata)
    assert layer_types is not None
    if mtp_count > 1:
        raise ValueError(
            f"{architecture} GGUF has {mtp_count} MTP blocks; pinned llama.cpp "
            "supports exactly one appended MTP block"
        )
    if mtp_count and architecture in {"qwen35moe", "qwen3next"}:
        raise NotImplementedError(
            f"{architecture} GGUF MTP blocks use routed experts, but the Mobius MTP "
            "sidecar currently supports only dense Qwen3.5 heads; refusing to omit "
            "the MTP expert tensors"
        )

    actual = set(gguf_model.tensor_names)
    total_layers = int(metadata[f"{architecture}.block_count"])
    for name in actual:
        match = re.match(r"^blk\.(\d+)\.", name)
        if match is not None and int(match.group(1)) >= total_layers:
            raise ValueError(
                f"{architecture} GGUF tensor {name!r} references out-of-range layer "
                f"{match.group(1)} (block_count={total_layers})"
            )

    if architecture in {"lfm2", "lfm2moe"}:
        required_global = {"token_embd.weight", "token_embd_norm.weight"}
        auxiliary = sorted(
            name
            for name in actual
            if name.startswith("dense_2_out.") or name == "output_norm.weight"
        )
        if auxiliary:
            raise ValueError(
                "lfm2 causal-LM import does not support embedding/ColBERT head "
                f"tensor(s): {auxiliary}"
            )
        common_suffixes = {"attn_norm.weight", "ffn_norm.weight"}
        full_suffixes = {
            "attn_q.weight",
            "attn_k.weight",
            "attn_v.weight",
            "attn_output.weight",
            "attn_q_norm.weight",
            "attn_k_norm.weight",
        }
        recurrent_suffixes = {
            "shortconv.conv.weight",
            "shortconv.in_proj.weight",
            "shortconv.out_proj.weight",
        }
    else:
        required_global = {"token_embd.weight", "output_norm.weight"}
        common_suffixes = {"attn_norm.weight"}
        if architecture == "qwen3next":
            common_suffixes.add("attn_post_norm.weight")
        else:
            common_suffixes.add("post_attention_norm.weight")
        if architecture == "qwen35":
            common_suffixes.update({"ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"})
        else:
            common_suffixes.update(
                {
                    "ffn_gate_inp.weight",
                    "ffn_down_exps.weight",
                    "ffn_gate_inp_shexp.weight",
                    "ffn_gate_shexp.weight",
                    "ffn_up_shexp.weight",
                    "ffn_down_shexp.weight",
                }
            )
        full_suffixes = {
            "attn_q.weight",
            "attn_k.weight",
            "attn_v.weight",
            "attn_output.weight",
            "attn_q_norm.weight",
            "attn_k_norm.weight",
        }
        recurrent_suffixes = {
            "ssm_conv1d.weight",
            "ssm_dt.bias",
            "ssm_a",
            "ssm_norm.weight",
            "ssm_out.weight",
        }
        if architecture == "qwen3next":
            recurrent_suffixes.add("ssm_ba.weight")
        else:
            recurrent_suffixes.update({"ssm_beta.weight", "ssm_alpha.weight"})

    missing_global = sorted(required_global - actual)
    if missing_global:
        raise ValueError(
            f"{architecture} GGUF is missing required global tensor(s): {missing_global}"
        )

    for layer, layer_type in enumerate(layer_types):
        prefix = f"blk.{layer}."
        layer_names = {name[len(prefix) :] for name in actual if name.startswith(prefix)}
        required = set(common_suffixes)
        required.update(
            full_suffixes if layer_type == "full_attention" else recurrent_suffixes
        )
        if architecture == "lfm2":
            required.update({"ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"})
        elif architecture == "lfm2moe":
            dense_ffn = {"ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"}
            routed_ffn = {
                "ffn_gate_inp.weight",
                "ffn_gate_exps.weight",
                "ffn_up_exps.weight",
                "ffn_down_exps.weight",
                "exp_probs_b.bias",
            }
            dense_layers = int(metadata.get("lfm2moe.leading_dense_block_count", 0))
            if layer < dense_layers:
                required.update(dense_ffn)
                wrong_ffn = sorted(layer_names & routed_ffn)
            else:
                required.update(routed_ffn)
                wrong_ffn = sorted(layer_names & dense_ffn)
            if wrong_ffn:
                raise ValueError(
                    f"lfm2moe layer {layer} contains tensor(s) from the wrong "
                    f"{'routed' if layer < dense_layers else 'dense'} FFN family: "
                    f"{wrong_ffn}"
                )

        if architecture in {"qwen35moe", "qwen3next"}:
            fused = "ffn_gate_up_exps.weight" in layer_names
            separate_members = {
                "ffn_gate_exps.weight",
                "ffn_up_exps.weight",
            }
            separate = separate_members.issubset(layer_names)
            if fused and layer_names & separate_members:
                raise ValueError(
                    f"{architecture} layer {layer} mixes fused and separate routed-expert "
                    "gate/up tensors"
                )
            if not fused and not separate:
                raise ValueError(
                    f"{architecture} layer {layer} must contain exactly one routed-expert "
                    "gate/up representation (fused or separate)"
                )
            required.add("ffn_gate_up_exps.weight" if fused else "ffn_gate_exps.weight")
            if not fused:
                required.add("ffn_up_exps.weight")

        if layer_type != "full_attention":
            if architecture == "qwen3next":
                modern_members = {
                    "attn_qkv.weight",
                    "attn_gate.weight",
                }
                modern = modern_members.issubset(layer_names)
                legacy = "ssm_in.weight" in layer_names
                if legacy and layer_names & modern_members:
                    raise ValueError(
                        f"qwen3next layer {layer} mixes legacy ssm_in with modern "
                        "attn_qkv/attn_gate tensors"
                    )
                if not legacy and not modern:
                    raise ValueError(
                        f"qwen3next layer {layer} must contain exactly one recurrent input "
                        "representation (attn_qkv+attn_gate or ssm_in)"
                    )
                required.update(
                    {"attn_qkv.weight", "attn_gate.weight"} if modern else {"ssm_in.weight"}
                )
            elif architecture not in {"lfm2", "lfm2moe"}:
                required.update({"attn_qkv.weight", "attn_gate.weight"})

        missing = sorted(required - layer_names)
        if missing:
            raise ValueError(
                f"{architecture} {layer_type} layer {layer} is missing required "
                f"tensor(s): {missing}"
            )

        wrong_family = recurrent_suffixes if layer_type == "full_attention" else full_suffixes
        wrong = sorted(layer_names & wrong_family)
        if layer_type == "full_attention":
            wrong.extend(
                sorted(
                    layer_names
                    & {
                        "attn_qkv.weight",
                        "attn_gate.weight",
                        "ssm_in.weight",
                    }
                )
            )
        if wrong:
            raise ValueError(
                f"{architecture} layer {layer} contains tensor(s) from the wrong "
                f"{layer_type} mixer family: {sorted(set(wrong))}"
            )

    if mtp_count:
        mtp_layer = trunk_layers
        prefix = f"blk.{mtp_layer}."
        layer_names = {name[len(prefix) :] for name in actual if name.startswith(prefix)}
        wrong_mixer = sorted(layer_names & recurrent_suffixes)
        if wrong_mixer:
            raise ValueError(
                f"{architecture} MTP block contains recurrent-mixer tensor(s) "
                f"{wrong_mixer}; appended MTP blocks must be full attention"
            )
        required_block = set(common_suffixes) | set(full_suffixes)
        if architecture in {"qwen35moe", "qwen3next"}:
            fused = "ffn_gate_up_exps.weight" in layer_names
            separate_members = {
                "ffn_gate_exps.weight",
                "ffn_up_exps.weight",
            }
            separate = separate_members.issubset(layer_names)
            if fused and layer_names & separate_members:
                raise ValueError(
                    f"{architecture} MTP block mixes fused and separate routed-expert "
                    "gate/up tensors"
                )
            if not fused and not separate:
                raise ValueError(
                    f"{architecture} MTP block must contain exactly one routed-expert "
                    "gate/up representation (fused or separate)"
                )
            required_block.add("ffn_gate_up_exps.weight" if fused else "ffn_gate_exps.weight")
            if not fused:
                required_block.add("ffn_up_exps.weight")
        missing_block = sorted(required_block - layer_names)
        if missing_block:
            raise ValueError(
                f"{architecture} MTP block is missing full-attention/FFN tensor(s): "
                f"{missing_block}"
            )
        required_mtp = {
            f"blk.{mtp_layer}.nextn.eh_proj.weight",
            f"blk.{mtp_layer}.nextn.enorm.weight",
            f"blk.{mtp_layer}.nextn.hnorm.weight",
        }
        missing_mtp = sorted(required_mtp - actual)
        if missing_mtp:
            raise ValueError(
                f"{architecture} GGUF declares an MTP block but is missing tensor(s): "
                f"{missing_mtp}"
            )
        known_nextn = {
            "nextn.eh_proj.weight",
            "nextn.enorm.weight",
            "nextn.hnorm.weight",
            "nextn.embed_tokens.weight",
            "nextn.shared_head_norm.weight",
            "nextn.shared_head_head.weight",
        }
        unknown_nextn = sorted(
            name
            for name in layer_names
            if name.startswith("nextn.") and name not in known_nextn
        )
        if unknown_nextn:
            raise ValueError(
                f"{architecture} MTP block contains unsupported nextn tensor(s): "
                f"{unknown_nextn}"
            )


def _raise_for_unsupported_encoder_heads(gguf_model) -> None:
    """Reject optional llama.cpp encoder heads that the token-output graph omits."""
    if gguf_model.architecture not in {
        "bert",
        "modern-bert",
        "eurobert",
        "neo-bert",
        "nomic-bert",
        "nomic-bert-moe",
        "jina-bert-v2",
        "jina-bert-v3",
    }:
        return
    head_tensors = [
        name
        for name in gguf_model.tensor_names
        if name.startswith(("cls.", "cls_out.", "cls_norm."))
    ]
    if head_tensors:
        raise ValueError(
            f"{gguf_model.architecture} GGUF contains unsupported pooler/classifier "
            f"tensor(s): {head_tensors}. Mobius will not silently discard encoder heads."
        )


def _raise_for_invalid_encoder_tensor_contract(gguf_model) -> None:
    """Validate required encoder tensors and their exact logical ranks/shapes."""
    from mobius.integrations.gguf._quant_registry import float_storage_type_ids

    architecture = gguf_model.architecture
    if architecture not in {"bert", "modern-bert"}:
        return

    metadata = gguf_model.metadata
    required_geometry = (
        "context_length",
        "embedding_length",
        "feed_forward_length",
        "block_count",
        "attention.head_count",
    )
    missing_geometry = [
        f"{architecture}.{suffix}"
        for suffix in required_geometry
        if f"{architecture}.{suffix}" not in metadata
    ]
    if missing_geometry:
        raise ValueError(
            f"{architecture} GGUF is missing required encoder metadata: {missing_geometry}"
        )
    hidden = int(metadata[f"{architecture}.embedding_length"])
    intermediate = int(metadata[f"{architecture}.feed_forward_length"])
    layers = int(metadata[f"{architecture}.block_count"])
    num_heads = int(metadata[f"{architecture}.attention.head_count"])
    context = int(metadata[f"{architecture}.context_length"])
    if min(hidden, intermediate, layers, num_heads, context) <= 0 or hidden % num_heads:
        raise ValueError(
            f"{architecture} GGUF has invalid encoder geometry: "
            f"embedding_length={hidden}, feed_forward_length={intermediate}, "
            f"block_count={layers}, attention.head_count={num_heads}, "
            f"context_length={context}; all values must be positive and "
            "embedding_length must be divisible by attention.head_count"
        )
    num_kv_heads = int(metadata.get(f"{architecture}.attention.head_count_kv", num_heads))
    if num_kv_heads != num_heads:
        raise ValueError(
            f"{architecture} GGUF grouped-query attention is not supported: "
            f"attention.head_count={num_heads}, attention.head_count_kv={num_kv_heads}"
        )
    vocab = int(metadata.get(f"{architecture}.vocab_size", 0))
    if not vocab:
        vocab = len(metadata.get("tokenizer.ggml.tokens", ()))
    if vocab <= 0:
        raise ValueError(f"{architecture} GGUF has no positive vocabulary size")

    required: dict[str, tuple[int, ...]] = {
        "token_embd.weight": (vocab, hidden),
    }
    if architecture == "bert":
        token_types = int(metadata.get("tokenizer.ggml.token_type_count", 0))
        if token_types <= 0:
            raise ValueError("bert GGUF tokenizer.ggml.token_type_count must be positive")
        required.update(
            {
                "token_types.weight": (token_types, hidden),
                "position_embd.weight": (context, hidden),
                "token_embd_norm.weight": (hidden,),
                "token_embd_norm.bias": (hidden,),
            }
        )
        layer_shapes = {
            "attn_output.weight": (hidden, hidden),
            "attn_output.bias": (hidden,),
            "attn_output_norm.weight": (hidden,),
            "attn_output_norm.bias": (hidden,),
            "ffn_up.weight": (intermediate, hidden),
            "ffn_up.bias": (intermediate,),
            "ffn_down.weight": (hidden, intermediate),
            "ffn_down.bias": (hidden,),
            "layer_output_norm.weight": (hidden,),
            "layer_output_norm.bias": (hidden,),
        }
    else:
        required.update(
            {
                "token_embd_norm.weight": (hidden,),
                "output_norm.weight": (hidden,),
            }
        )
        layer_shapes = {
            "attn_qkv.weight": (3 * hidden, hidden),
            "attn_output.weight": (hidden, hidden),
            "ffn_up.weight": (2 * intermediate, hidden),
            "ffn_down.weight": (hidden, intermediate),
            "ffn_norm.weight": (hidden,),
        }

    actual_items = list(gguf_model.tensor_items_raw())
    actual = {
        name: tuple(int(dim) for dim in shape) for name, _raw, _qtype, shape in actual_items
    }
    qtypes = {name: qtype for name, _raw, qtype, _shape in actual_items}

    for layer in range(layers):
        for suffix, shape in layer_shapes.items():
            required[f"blk.{layer}.{suffix}"] = shape
        if architecture == "bert":
            fused_weight = f"blk.{layer}.attn_qkv.weight"
            fused_bias = f"blk.{layer}.attn_qkv.bias"
            split_names = {
                f"blk.{layer}.attn_{projection}.{suffix}"
                for projection in ("q", "k", "v")
                for suffix in ("weight", "bias")
            }
            has_fused = fused_weight in actual or fused_bias in actual
            has_split = bool(split_names & set(actual))
            if has_fused and has_split:
                raise ValueError(
                    f"bert GGUF layer {layer} contains both fused and split Q/K/V tensors; "
                    "the import contract requires one unambiguous representation"
                )
            if has_fused:
                required[fused_weight] = (3 * hidden, hidden)
                required[fused_bias] = (3 * hidden,)
                qtype = qtypes.get(fused_weight)
                qtype_id = getattr(qtype, "value", qtype)
                if qtype is not None and qtype_id not in float_storage_type_ids():
                    raise ValueError(
                        "Quantized fused BERT attn_qkv.weight cannot be split losslessly; "
                        "use a GGUF with split Q/K/V tensors or float fused QKV"
                    )
            else:
                for projection in ("q", "k", "v"):
                    required[f"blk.{layer}.attn_{projection}.weight"] = (hidden, hidden)
                    required[f"blk.{layer}.attn_{projection}.bias"] = (hidden,)
        if architecture == "modern-bert" and layer > 0:
            required[f"blk.{layer}.attn_norm.weight"] = (hidden,)

    missing = sorted(set(required) - set(actual))
    if missing:
        raise ValueError(
            f"{architecture} GGUF is missing required encoder tensor(s): {missing}"
        )
    malformed = {
        name: (required[name], actual[name])
        for name in required
        if actual[name] != required[name]
    }
    if malformed:
        raise ValueError(
            f"{architecture} GGUF has invalid encoder tensor shape(s): {malformed}"
        )
    if architecture == "modern-bert" and "blk.0.attn_norm.weight" in actual:
        raise ValueError(
            "modern-bert blk.0.attn_norm.weight is present, but Mobius models layer 0 "
            "as the pinned identity-norm variant and will not ignore the tensor"
        )


def _raise_for_invalid_specialized_encoder_tensor_contract(gguf_model) -> None:
    """Validate the exact llama.cpp tensor closure for promoted specialized encoders."""
    from mobius.integrations.gguf._tensor_mapping import is_known_skip

    arch = gguf_model.architecture
    if arch not in {
        "eurobert",
        "neo-bert",
        "nomic-bert",
        "jina-bert-v2",
        "jina-bert-v3",
    }:
        return
    metadata = gguf_model.metadata
    geometry = (
        "context_length",
        "embedding_length",
        "feed_forward_length",
        "block_count",
        "attention.head_count",
    )
    missing_metadata = [
        f"{arch}.{suffix}" for suffix in geometry if f"{arch}.{suffix}" not in metadata
    ]
    if missing_metadata:
        raise ValueError(f"{arch} GGUF is missing encoder metadata: {missing_metadata}")

    hidden = int(metadata[f"{arch}.embedding_length"])
    intermediate = int(metadata[f"{arch}.feed_forward_length"])
    layers = int(metadata[f"{arch}.block_count"])
    heads = int(metadata[f"{arch}.attention.head_count"])
    context = int(metadata[f"{arch}.context_length"])
    kv_heads = int(metadata.get(f"{arch}.attention.head_count_kv", heads))
    if min(hidden, intermediate, layers, heads, context) <= 0 or hidden % heads:
        raise ValueError(f"{arch} GGUF has invalid positive encoder geometry")
    if kv_heads != heads:
        raise ValueError(f"{arch} GGUF requires head_count_kv == head_count")
    head_dim = hidden // heads
    key_length = int(metadata.get(f"{arch}.attention.key_length", head_dim))
    value_length = int(metadata.get(f"{arch}.attention.value_length", key_length))
    if key_length != head_dim or value_length != head_dim:
        raise ValueError(f"{arch} GGUF requires full-width equal Q/K/V heads")
    if arch != "jina-bert-v2":
        rope_dim = int(metadata.get(f"{arch}.rope.dimension_count", head_dim))
        if rope_dim != head_dim:
            raise ValueError(f"{arch} GGUF requires full-head RoPE")

    vocab = int(metadata.get(f"{arch}.vocab_size", 0))
    if not vocab:
        vocab = len(metadata.get("tokenizer.ggml.tokens", ()))
    if vocab <= 0:
        raise ValueError(f"{arch} GGUF has no positive vocabulary size")
    token_types = int(metadata.get("tokenizer.ggml.token_type_count", 0))

    actual = {
        name: tuple(int(dim) for dim in shape)
        for name, _raw, _qtype, shape in gguf_model.tensor_items_raw()
        if not is_known_skip(name)
    }
    required: dict[str, tuple[int, ...]] = {"token_embd.weight": (vocab, hidden)}
    optional: dict[str, tuple[int, ...]] = {}

    if arch == "eurobert":
        required["output_norm.weight"] = (hidden,)
    elif arch == "neo-bert":
        required["enc.output_norm.weight"] = (hidden,)
    else:
        if token_types <= 0:
            raise ValueError(f"{arch} tokenizer.ggml.token_type_count must be positive")
        token_type_shape = (
            (hidden,) if arch == "jina-bert-v3" and token_types == 1 else (token_types, hidden)
        )
        if arch == "jina-bert-v2":
            required["token_types.weight"] = token_type_shape
        else:
            optional["token_types.weight"] = token_type_shape
        required.update(
            {
                "token_embd_norm.weight": (hidden,),
                "token_embd_norm.bias": (hidden,),
            }
        )

    optional_families: dict[str, set[str]] = {}
    moe_interval = int(metadata.get(f"{arch}.moe_every_n_layers", 0))
    if arch == "jina-bert-v3":
        expert_metadata = any(
            f"{arch}.{suffix}" in metadata
            for suffix in (
                "expert_count",
                "expert_used_count",
                "expert_feed_forward_length",
                "expert_weights_norm",
                "expert_weights_scale",
            )
        )
        if moe_interval != 0 or expert_metadata:
            raise ValueError(
                f"{arch} MoE metadata is unsupported because the pinned loader never "
                "loads moe_every_n_layers"
            )
    for layer in range(layers):
        prefix = f"blk.{layer}."
        if arch in {"eurobert", "neo-bert"}:
            required.update(
                {
                    prefix + "attn_norm.weight": (hidden,),
                    prefix + "attn_output.weight": (hidden, hidden),
                    prefix + "ffn_norm.weight": (hidden,),
                    prefix + "ffn_down.weight": (hidden, intermediate),
                }
            )
            if arch == "eurobert":
                required.update(
                    {
                        prefix + "attn_q.weight": (hidden, hidden),
                        prefix + "attn_k.weight": (hidden, hidden),
                        prefix + "attn_v.weight": (hidden, hidden),
                        prefix + "ffn_gate.weight": (intermediate, hidden),
                        prefix + "ffn_up.weight": (intermediate, hidden),
                    }
                )
            else:
                required.update(
                    {
                        prefix + "attn_qkv.weight": (3 * hidden, hidden),
                        prefix + "ffn_up.weight": (2 * intermediate, hidden),
                    }
                )
            continue

        if arch == "jina-bert-v3":
            required.update(
                {
                    prefix + "attn_output.weight": (hidden, hidden),
                    prefix + "attn_output_norm.weight": (hidden,),
                    prefix + "attn_output_norm.bias": (hidden,),
                    prefix + "layer_output_norm.weight": (hidden,),
                    prefix + "layer_output_norm.bias": (hidden,),
                }
            )
            fused = prefix + "attn_qkv.weight" in actual
            if fused:
                required[prefix + "attn_qkv.weight"] = (3 * hidden, hidden)
                optional[prefix + "attn_qkv.bias"] = (3 * hidden,)
            else:
                for projection in ("q", "k", "v"):
                    required[prefix + f"attn_{projection}.weight"] = (hidden, hidden)
                    optional[prefix + f"attn_{projection}.bias"] = (hidden,)
            optional[prefix + "attn_output.bias"] = (hidden,)

            required.update(
                {
                    prefix + "ffn_up.weight": (intermediate, hidden),
                    prefix + "ffn_down.weight": (hidden, intermediate),
                }
            )
            optional[prefix + "ffn_up.bias"] = (intermediate,)
            optional[prefix + "ffn_down.bias"] = (hidden,)
            continue

        required.update(
            {
                prefix + "attn_q.weight": (hidden, hidden),
                prefix + "attn_k.weight": (hidden, hidden),
                prefix + "attn_v.weight": (hidden, hidden),
                prefix + "attn_output.weight": (hidden, hidden),
                prefix + "attn_output_norm.weight": (hidden,),
                prefix + "attn_output_norm.bias": (hidden,),
                prefix + "ffn_down.weight": (hidden, intermediate),
                prefix + "layer_output_norm.weight": (hidden,),
                prefix + "layer_output_norm.bias": (hidden,),
            }
        )
        for suffix, shape in {
            "attn_q.bias": (hidden,),
            "attn_k.bias": (hidden,),
            "attn_v.bias": (hidden,),
            "ffn_up.bias": (intermediate,),
        }.items():
            name = prefix + suffix
            optional[name] = shape
            optional_families.setdefault(suffix, set()).add(name)

        if arch == "nomic-bert":
            required.update(
                {
                    prefix + "ffn_gate.weight": (intermediate, hidden),
                    prefix + "ffn_up.weight": (intermediate, hidden),
                }
            )
            for suffix, shape in {
                "attn_output.bias": (hidden,),
                "ffn_down.bias": (hidden,),
            }.items():
                name = prefix + suffix
                optional[name] = shape
                optional_families.setdefault(suffix, set()).add(name)
        else:
            required.update(
                {
                    prefix + "attn_output.bias": (hidden,),
                    prefix + "ffn_down.bias": (hidden,),
                }
            )
            for suffix in (
                "attn_q_norm.weight",
                "attn_q_norm.bias",
                "attn_k_norm.weight",
                "attn_k_norm.bias",
                "attn_norm_2.weight",
                "attn_norm_2.bias",
                "ffn_gate.weight",
            ):
                shape = (intermediate, hidden) if suffix == "ffn_gate.weight" else (hidden,)
                name = prefix + suffix
                optional[name] = shape
                optional_families.setdefault(suffix, set()).add(name)

    present = set(actual)
    for suffix, family in optional_families.items():
        selected = family & present
        if selected and selected != family:
            raise ValueError(
                f"{arch} optional tensor family {suffix!r} must be all-layers or absent"
            )

    if arch == "jina-bert-v2":
        q_norm = any(name.endswith("attn_q_norm.weight") for name in present)
        k_norm = any(name.endswith("attn_k_norm.weight") for name in present)
        q_norm_bias = any(name.endswith("attn_q_norm.bias") for name in present)
        k_norm_bias = any(name.endswith("attn_k_norm.bias") for name in present)
        if len({q_norm, k_norm, q_norm_bias, k_norm_bias}) != 1:
            raise ValueError("jina-bert-v2 Q/K LayerNorm weights and biases must co-occur")
        extra_weight = any(name.endswith("attn_norm_2.weight") for name in present)
        extra_bias = any(name.endswith("attn_norm_2.bias") for name in present)
        if extra_weight != extra_bias:
            raise ValueError("jina-bert-v2 extra attention norm needs weight and bias")
        has_gate = any(name.endswith("ffn_gate.weight") for name in present)
        up_width = intermediate if has_gate else 2 * intermediate
        for layer in range(layers):
            required[f"blk.{layer}.ffn_up.weight"] = (up_width, hidden)
            if f"blk.{layer}.ffn_up.bias" in actual:
                optional[f"blk.{layer}.ffn_up.bias"] = (up_width,)

    allowed = set(required) | set(optional)
    missing = sorted(set(required) - present)
    unexpected = sorted(present - allowed)
    malformed = {
        name: (required.get(name, optional.get(name)), actual[name])
        for name in allowed & present
        if actual[name] != required.get(name, optional.get(name))
    }
    if missing or unexpected or malformed:
        raise ValueError(
            f"Invalid {arch} GGUF tensor closure: missing={missing}, "
            f"unexpected={unexpected}, malformed={malformed}"
        )


def _raise_for_invalid_embedding_tensor_contract(gguf_model) -> None:
    """Validate an exact profile before conditional loader fields reach config mapping."""
    arch = gguf_model.architecture
    if arch not in {"gemma-embedding", "llama-embed"}:
        return
    metadata = gguf_model.metadata
    required_suffixes = (
        "context_length",
        "embedding_length",
        "feed_forward_length",
        "block_count",
        "attention.head_count",
        "attention.layer_norm_rms_epsilon",
        "rope.dimension_count",
        "rope.freq_base",
        "attention.causal",
    )
    missing_metadata = [
        f"{arch}.{suffix}"
        for suffix in required_suffixes
        if f"{arch}.{suffix}" not in metadata
    ]
    if missing_metadata:
        raise ValueError(f"{arch} GGUF is missing embedding metadata: {missing_metadata}")
    if arch == "gemma-embedding" and f"{arch}.attention.sliding_window" not in metadata:
        raise ValueError("gemma-embedding requires attention.sliding_window")
    if bool(metadata[f"{arch}.attention.causal"]):
        raise ValueError(f"{arch}.attention.causal must be false")

    hidden = int(metadata[f"{arch}.embedding_length"])
    intermediate = int(metadata[f"{arch}.feed_forward_length"])
    layers = int(metadata[f"{arch}.block_count"])
    heads = int(metadata[f"{arch}.attention.head_count"])
    kv_heads = int(metadata.get(f"{arch}.attention.head_count_kv", heads))
    if (
        min(hidden, intermediate, layers, heads, kv_heads) <= 0
        or hidden % heads
        or heads % kv_heads
    ):
        raise ValueError(f"{arch} has invalid embedding geometry")
    head_dim = hidden // heads
    if int(metadata[f"{arch}.rope.dimension_count"]) != head_dim:
        raise ValueError(f"{arch} requires full-head default RoPE")
    if int(metadata.get(f"{arch}.attention.key_length", head_dim)) != head_dim:
        raise ValueError(f"{arch} key_length variants are unsupported")
    if int(metadata.get(f"{arch}.attention.value_length", head_dim)) != head_dim:
        raise ValueError(f"{arch} value_length variants are unsupported")
    if int(metadata.get(f"{arch}.expert_count", 0)):
        raise ValueError(f"{arch} MoE loader profile is unsupported")
    rope_type = metadata.get(f"{arch}.rope.scaling.type")
    if rope_type not in {None, "none"}:
        raise ValueError(f"{arch} rope scaling variant {rope_type!r} is unsupported")

    names = set(gguf_model.tensor_names)
    forbidden = sorted(
        name
        for name in names
        if (
            "attn_qkv." in name
            or name.endswith(".bias")
            or "rope_factors_" in name
            or "rope_freqs." in name
            or "_exps." in name
            or "_shexp." in name
            or "ffn_gate_inp." in name
        )
    )
    if forbidden:
        raise ValueError(
            f"{arch} unsupported fused/bias/MoE/rope tensor variant(s): {forbidden}"
        )

    vocab = int(metadata.get(f"{arch}.vocab_size", 0))
    if not vocab:
        vocab = len(metadata.get("tokenizer.ggml.tokens", ()))
    if vocab <= 0:
        raise ValueError(f"{arch} has no positive vocabulary size")
    q_width = heads * head_dim
    kv_width = kv_heads * head_dim
    required: dict[str, tuple[int, ...]] = {
        "token_embd.weight": (vocab, hidden),
        "output_norm.weight": (hidden,),
    }
    optional: dict[str, tuple[int, ...]] = {
        # llama.cpp loads/ties this tensor but graph<true> never consumes it.
        "output.weight": (vocab, hidden),
    }
    for layer in range(layers):
        prefix = f"blk.{layer}."
        required.update(
            {
                prefix + "attn_norm.weight": (hidden,),
                prefix + "attn_q.weight": (q_width, hidden),
                prefix + "attn_k.weight": (kv_width, hidden),
                prefix + "attn_v.weight": (kv_width, hidden),
                prefix + "attn_output.weight": (hidden, q_width),
                prefix + "ffn_norm.weight": (hidden,),
                prefix + "ffn_gate.weight": (intermediate, hidden),
                prefix + "ffn_up.weight": (intermediate, hidden),
                prefix + "ffn_down.weight": (hidden, intermediate),
            }
        )
        if arch == "gemma-embedding":
            required.update(
                {
                    prefix + "attn_q_norm.weight": (head_dim,),
                    prefix + "attn_k_norm.weight": (head_dim,),
                    prefix + "post_attention_norm.weight": (hidden,),
                    prefix + "post_ffw_norm.weight": (hidden,),
                }
            )

    if arch == "gemma-embedding":
        has_dense_2 = "dense_2.weight" in names
        has_dense_3 = "dense_3.weight" in names
        if (has_dense_2 or has_dense_3) and int(metadata.get(f"{arch}.pooling_type", 0)) == 0:
            raise ValueError("gemma-embedding dense modules require a pooled output")
        dense_2_keys = (f"{arch}.dense_2_feat_in", f"{arch}.dense_2_feat_out")
        dense_3_keys = (f"{arch}.dense_3_feat_in", f"{arch}.dense_3_feat_out")
        if has_dense_2 != all(key in metadata for key in dense_2_keys):
            raise ValueError("gemma-embedding dense_2 tensor and metadata must co-occur")
        if has_dense_3 != all(key in metadata for key in dense_3_keys):
            raise ValueError("gemma-embedding dense_3 tensor and metadata must co-occur")
        current_width = hidden
        if has_dense_2:
            dense_in = int(metadata[dense_2_keys[0]])
            dense_out = int(metadata[dense_2_keys[1]])
            if dense_in != hidden or dense_out <= 0:
                raise ValueError("gemma-embedding dense_2 dimensions are incompatible")
            optional["dense_2.weight"] = (dense_out, dense_in)
            current_width = dense_out
        if has_dense_3:
            dense_in = int(metadata[dense_3_keys[0]])
            dense_out = int(metadata[dense_3_keys[1]])
            if dense_in != current_width or dense_out != hidden:
                raise ValueError("gemma-embedding dense_3 dimensions are incompatible")
            optional["dense_3.weight"] = (dense_out, dense_in)

    actual = {
        name: tuple(int(dim) for dim in shape)
        for name, _raw, _qtype, shape in gguf_model.tensor_items_raw()
    }
    allowed = set(required) | set(optional)
    missing = sorted(set(required) - set(actual))
    unexpected = sorted(set(actual) - allowed)
    malformed = {
        name: (required.get(name, optional.get(name)), actual[name])
        for name in allowed & set(actual)
        if actual[name] != required.get(name, optional.get(name))
    }
    if missing or unexpected or malformed:
        raise ValueError(
            f"Invalid {arch} embedding tensor closure: missing={missing}, "
            f"unexpected={unexpected}, malformed={malformed}"
        )


def _raise_for_invalid_mamba_hybrid_tensor_contract(gguf_model) -> None:
    """Require the exact dense tensor family for each audited hybrid layer."""
    import numpy as np

    from mobius.integrations.gguf._config_mapping import _derive_hybrid_layout

    architecture = gguf_model.architecture
    metadata = gguf_model.metadata
    layer_count, layer_types, mtp_count = _derive_hybrid_layout(
        architecture, metadata, gguf_model.tensor_names
    )
    assert layer_types is not None
    if mtp_count:
        raise ValueError(f"{architecture} GGUF auxiliary/MTP blocks are not supported")

    actual = set(gguf_model.tensor_names)
    required_global = {"token_embd.weight", "output_norm.weight"}
    optional_global = {"output.weight"}
    required_by_type: dict[str, set[str]]
    optional_by_type: dict[str, set[str]]
    common: set[str]
    if architecture == "jamba":
        common = {"attn_norm.weight", "ffn_norm.weight"}
        num_experts = int(metadata.get("jamba.expert_count", 0))
        top_k = int(metadata.get("jamba.expert_used_count", 0))
        if bool(num_experts) != bool(top_k):
            raise ValueError(
                "Jamba expert_count and expert_used_count must both be zero or both positive"
            )
        if num_experts and not 1 <= top_k <= num_experts:
            raise ValueError(
                f"Jamba expert_used_count must be in [1, {num_experts}], got {top_k}"
            )
        if num_experts == 1:
            raise ValueError(
                "Jamba expert_count=1 is not a routed-MoE layout; use dense FFN tensors"
            )
        required_by_type = {
            "mamba": {
                "ssm_in.weight",
                "ssm_conv1d.weight",
                "ssm_conv1d.bias",
                "ssm_x.weight",
                "ssm_dt_norm.weight",
                "ssm_dt.weight",
                "ssm_dt.bias",
                "ssm_b_norm.weight",
                "ssm_c_norm.weight",
                "ssm_a",
                "ssm_d",
                "ssm_out.weight",
            },
            "full_attention": {
                "attn_q.weight",
                "attn_k.weight",
                "attn_v.weight",
                "attn_output.weight",
            },
        }
        optional_by_type = {"mamba": set(), "full_attention": set()}
    elif architecture in {"nemotron_h", "nemotron_h_moe"}:
        common = {"attn_norm.weight"}
        required_by_type = {
            "mamba2": {
                "ssm_in.weight",
                "ssm_conv1d.weight",
                "ssm_dt.bias",
                "ssm_a",
                "ssm_d",
                "ssm_norm.weight",
                "ssm_out.weight",
            },
            "full_attention": {
                "attn_q.weight",
                "attn_k.weight",
                "attn_v.weight",
                "attn_output.weight",
            },
            "mlp": {"ffn_up.weight", "ffn_down.weight"},
            "moe": {
                "ffn_gate_inp.weight",
                "exp_probs_b.bias",
                "ffn_up_exps.weight",
                "ffn_down_exps.weight",
                "ffn_up_shexp.weight",
                "ffn_down_shexp.weight",
            },
        }
        optional_by_type = {
            "mamba2": {"ssm_conv1d.bias"},
            "full_attention": {"attn_output.bias"},
            "mlp": {"ffn_up.bias", "ffn_down.bias"},
            "moe": {
                "ffn_latent_down.weight",
                "ffn_latent_up.weight",
            },
        }
    else:
        common = {"attn_norm.weight", "ffn_norm.weight"}
        num_experts = int(metadata.get("granitehybrid.expert_count", 0))
        top_k = int(metadata.get("granitehybrid.expert_used_count", 0))
        if bool(num_experts) != bool(top_k):
            raise ValueError(
                "GraniteHybrid expert_count and expert_used_count must both be zero "
                "or both positive"
            )
        dense_ffn = {"ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"}
        dense_ffn_biases = {"ffn_gate.bias", "ffn_up.bias", "ffn_down.bias"}
        routed_ffn = {
            "ffn_gate_inp.weight",
            "ffn_gate_exps.weight",
            "ffn_up_exps.weight",
            "ffn_down_exps.weight",
        }
        shared_ffn = {
            "ffn_gate_shexp.weight",
            "ffn_up_shexp.weight",
            "ffn_down_shexp.weight",
        }
        shared_width = int(metadata.get("granitehybrid.expert_shared_feed_forward_length", 0))
        common |= routed_ffn if num_experts else dense_ffn
        if num_experts and shared_width > 0:
            common |= shared_ffn
        required_by_type = {
            "mamba2": {
                "ssm_in.weight",
                "ssm_conv1d.weight",
                "ssm_dt.bias",
                "ssm_a",
                "ssm_d",
                "ssm_norm.weight",
                "ssm_out.weight",
            },
            "full_attention": {
                "attn_q.weight",
                "attn_k.weight",
                "attn_v.weight",
                "attn_output.weight",
            },
        }
        optional_by_type = {
            "mamba2": {"ssm_conv1d.bias"},
            "full_attention": {
                "attn_q.bias",
                "attn_k.bias",
                "attn_v.bias",
                "attn_output.bias",
                "rope_freqs.weight",
            },
        }
        if not num_experts:
            for optional in optional_by_type.values():
                optional.update(dense_ffn_biases)
        forbidden_ffn = dense_ffn if num_experts else routed_ffn | shared_ffn
        present_forbidden = sorted(
            f"blk.{index}.{suffix}"
            for index in range(layer_count)
            for suffix in forbidden_ffn
            if f"blk.{index}.{suffix}" in actual
        )
        if present_forbidden:
            raise ValueError(
                "GraniteHybrid GGUF mixes dense and routed/shared MoE representations: "
                f"{present_forbidden}"
            )

    expected = set(required_global)
    allowed = required_global | optional_global

    def require_all_or_none(label: str, names: list[str]) -> None:
        present = sorted(set(names) & actual)
        if present and len(present) != len(names):
            missing_family = sorted(set(names) - actual)
            raise ValueError(
                f"{architecture} GGUF has a partial {label} bias family: "
                f"present={present}, missing={missing_family}"
            )

    recurrent_layers = [
        index
        for index, layer_type in enumerate(layer_types)
        if layer_type in {"mamba", "mamba2"}
    ]
    attention_layers = [
        index for index, layer_type in enumerate(layer_types) if layer_type == "full_attention"
    ]
    require_all_or_none(
        "recurrent convolution",
        [f"blk.{index}.ssm_conv1d.bias" for index in recurrent_layers],
    )
    if architecture in {"nemotron_h", "nemotron_h_moe", "granitehybrid"}:
        require_all_or_none(
            "attention output projection",
            [f"blk.{index}.attn_output.bias" for index in attention_layers],
        )
    if architecture == "granitehybrid":
        require_all_or_none(
            "attention Q/K/V projection",
            [
                f"blk.{index}.attn_{projection}.bias"
                for index in attention_layers
                for projection in ("q", "k", "v")
            ],
        )
    if architecture == "granitehybrid" and not int(
        metadata.get("granitehybrid.expert_count", 0)
    ):
        require_all_or_none(
            "dense shared-MLP",
            [
                f"blk.{index}.ffn_{projection}.bias"
                for index in range(layer_count)
                for projection in ("gate", "up", "down")
            ],
        )
    if architecture in {"nemotron_h", "nemotron_h_moe"}:
        mlp_layers = [
            index for index, layer_type in enumerate(layer_types) if layer_type == "mlp"
        ]
        require_all_or_none(
            "dense MLP",
            [
                f"blk.{index}.ffn_{projection}.bias"
                for index in mlp_layers
                for projection in ("up", "down")
            ],
        )
        moe_layers = [
            index for index, layer_type in enumerate(layer_types) if layer_type == "moe"
        ]
        require_all_or_none(
            "MoE latent projection",
            [
                f"blk.{index}.ffn_latent_{direction}.weight"
                for index in moe_layers
                for direction in ("down", "up")
            ],
        )
        has_latent_metadata = f"{architecture}.moe_latent_size" in metadata
        has_latent_tensors = any(
            f"blk.{index}.ffn_latent_down.weight" in actual for index in moe_layers
        )
        if has_latent_metadata != has_latent_tensors:
            raise ValueError(
                f"{architecture} moe_latent_size metadata and latent projection tensors "
                "must either both be present or both be absent"
            )

    for index, layer_type in enumerate(layer_types):
        prefix = f"blk.{index}."
        required = common | required_by_type[layer_type]
        optional = set(optional_by_type[layer_type])
        if architecture == "jamba":
            has_router = f"{prefix}ffn_gate_inp.weight" in actual
            dense_ffn = {"ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"}
            moe_ffn = {
                "ffn_gate_inp.weight",
                "ffn_gate_exps.weight",
                "ffn_up_exps.weight",
                "ffn_down_exps.weight",
            }
            if has_router:
                if not num_experts:
                    raise ValueError(
                        f"Jamba layer {index} has routed experts without expert metadata"
                    )
                required |= moe_ffn
                optional |= {
                    suffix
                    for stem in ("ffn_gate_exps", "ffn_up_exps", "ffn_down_exps")
                    for suffix in (f"{stem}.scale", f"{stem}.input_scale")
                }
            else:
                required |= dense_ffn
                optional |= {
                    suffix
                    for stem in ("ffn_gate", "ffn_up", "ffn_down")
                    for suffix in (f"{stem}.scale", f"{stem}.input_scale")
                }
        expected.update(prefix + suffix for suffix in required)
        allowed.update(prefix + suffix for suffix in required | optional)

    missing = sorted(expected - actual)
    unexpected = sorted(actual - allowed)
    out_of_range = sorted(
        name
        for name in actual
        if (match := re.match(r"^blk\.(\d+)\.", name)) and int(match.group(1)) >= layer_count
    )
    if missing or unexpected or out_of_range:
        raise ValueError(
            f"Invalid {architecture} GGUF tensor closure: missing={missing}, "
            f"unexpected={unexpected}, out_of_range={out_of_range}"
        )

    if not hasattr(gguf_model, "tensor_items_raw"):
        return

    shapes = {
        name: tuple(int(dimension) for dimension in shape)
        for name, _raw, _qtype, shape in gguf_model.tensor_items_raw()
    }
    if architecture in {"nemotron_h", "nemotron_h_moe"}:
        _validate_nemotron_h_tensor_shapes(gguf_model, layer_types, actual)
        return
    if architecture == "granitehybrid":
        _validate_granitehybrid_tensor_shapes(gguf_model, layer_types, actual)
        return

    hidden = int(metadata["jamba.embedding_length"])
    intermediate = int(metadata["jamba.feed_forward_length"])
    vocab = int(metadata.get("jamba.vocab_size", 0))
    if not vocab:
        vocab = len(metadata.get("tokenizer.ggml.tokens", ()))
    raw_head_counts = metadata["jamba.attention.head_count"]
    head_counts = (
        [int(value) for value in raw_head_counts]
        if isinstance(raw_head_counts, (list, tuple))
        else [int(raw_head_counts)]
    )
    positive_head_counts = {value for value in head_counts if value > 0}
    if len(positive_head_counts) != 1:
        raise ValueError("Jamba GGUF attention layers must use one consistent head count")
    heads = positive_head_counts.pop()
    if hidden <= 0 or intermediate <= 0 or vocab <= 0 or hidden % heads:
        raise ValueError(
            "Jamba GGUF has inconsistent embedding, FFN, vocabulary, or head geometry"
        )
    head_dim = hidden // heads
    state = int(metadata["jamba.ssm.state_size"])
    inner = int(metadata["jamba.ssm.inner_size"])
    rank = int(metadata["jamba.ssm.time_step_rank"])
    conv = int(metadata["jamba.ssm.conv_kernel"])
    if inner != 2 * hidden or min(state, rank, conv) <= 0:
        raise ValueError("Jamba GGUF has inconsistent Mamba-1 geometry")

    expected_shapes: dict[str, tuple[int, ...]] = {
        "token_embd.weight": (vocab, hidden),
        "output_norm.weight": (hidden,),
    }
    if "output.weight" in actual:
        expected_shapes["output.weight"] = (vocab, hidden)
    for index, layer_type in enumerate(layer_types):
        prefix = f"blk.{index}."
        expected_shapes[prefix + "attn_norm.weight"] = (hidden,)
        expected_shapes[prefix + "ffn_norm.weight"] = (hidden,)
        if layer_type == "mamba":
            expected_shapes.update(
                {
                    prefix + "ssm_in.weight": (2 * inner, hidden),
                    prefix + "ssm_conv1d.weight": (inner, conv),
                    prefix + "ssm_conv1d.bias": (inner,),
                    prefix + "ssm_x.weight": (rank + 2 * state, inner),
                    prefix + "ssm_dt_norm.weight": (rank,),
                    prefix + "ssm_dt.weight": (inner, rank),
                    prefix + "ssm_dt.bias": (inner,),
                    prefix + "ssm_b_norm.weight": (state,),
                    prefix + "ssm_c_norm.weight": (state,),
                    prefix + "ssm_a": (inner, state),
                    prefix + "ssm_d": (inner,),
                    prefix + "ssm_out.weight": (hidden, inner),
                }
            )
        else:
            kv_heads = int(metadata["jamba.attention.head_count_kv"][index])
            if kv_heads <= 0 or heads % kv_heads:
                raise ValueError(f"Jamba attention layer {index} has invalid KV head count")
            kv_width = kv_heads * head_dim
            expected_shapes.update(
                {
                    prefix + "attn_q.weight": (hidden, hidden),
                    prefix + "attn_k.weight": (kv_width, hidden),
                    prefix + "attn_v.weight": (kv_width, hidden),
                    prefix + "attn_output.weight": (hidden, hidden),
                }
            )
        if prefix + "ffn_gate_inp.weight" in actual:
            expected_shapes.update(
                {
                    prefix + "ffn_gate_inp.weight": (num_experts, hidden),
                    prefix + "ffn_gate_exps.weight": (
                        num_experts,
                        intermediate,
                        hidden,
                    ),
                    prefix + "ffn_up_exps.weight": (
                        num_experts,
                        intermediate,
                        hidden,
                    ),
                    prefix + "ffn_down_exps.weight": (
                        num_experts,
                        hidden,
                        intermediate,
                    ),
                }
            )
        else:
            expected_shapes.update(
                {
                    prefix + "ffn_gate.weight": (intermediate, hidden),
                    prefix + "ffn_up.weight": (intermediate, hidden),
                    prefix + "ffn_down.weight": (hidden, intermediate),
                }
            )
    malformed = sorted(
        f"{name}: expected {expected_shape}, got {shapes[name]}"
        for name, expected_shape in expected_shapes.items()
        if name in shapes and shapes[name] != expected_shape
    )
    if malformed:
        raise ValueError(f"Invalid Jamba GGUF tensor shape(s): {malformed}")
    for index, layer_type in enumerate(layer_types):
        if layer_type != "mamba":
            continue
        decay_name = f"blk.{index}.ssm_a"
        decay = np.asarray(gguf_model.get_tensor(decay_name))
        if not np.all(np.isfinite(decay)) or not np.all(decay < 0):
            raise ValueError(
                f"Malformed Jamba GGUF Mamba decay tensor {decay_name!r}: "
                "ssm_a must contain only finite negative -exp(A_log) values"
            )


def _validate_nemotron_h_tensor_shapes(gguf_model, layer_types, actual: set[str]) -> None:
    """Validate logical Nemotron-H tensor shapes before graph construction."""
    metadata = gguf_model.metadata
    arch = gguf_model.architecture
    shapes = {
        name: tuple(int(dimension) for dimension in shape)
        for name, _raw, _qtype, shape in gguf_model.tensor_items_raw()
    }
    hidden = int(metadata[f"{arch}.embedding_length"])
    vocab = int(metadata.get(f"{arch}.vocab_size", 0))
    if not vocab:
        vocab = len(metadata.get("tokenizer.ggml.tokens", ()))
    heads_raw = metadata[f"{arch}.attention.head_count"]
    heads_by_layer = (
        [int(value) for value in heads_raw]
        if isinstance(heads_raw, (list, tuple, np.ndarray))
        else [int(heads_raw)] * len(layer_types)
    )
    kv_raw = metadata[f"{arch}.attention.head_count_kv"]
    kv_by_layer = [int(value) for value in kv_raw]
    head_dim = int(
        metadata.get(
            f"{arch}.attention.key_length",
            hidden // next(value for value in heads_by_layer if value > 0),
        )
    )
    state = int(metadata[f"{arch}.ssm.state_size"])
    inner = int(metadata[f"{arch}.ssm.inner_size"])
    groups = int(metadata[f"{arch}.ssm.group_count"])
    ssm_heads = int(metadata[f"{arch}.ssm.time_step_rank"])
    conv = int(metadata[f"{arch}.ssm.conv_kernel"])

    expected: dict[str, tuple[int, ...]] = {
        "token_embd.weight": (vocab, hidden),
        "output_norm.weight": (hidden,),
    }
    if "output.weight" in actual:
        expected["output.weight"] = (vocab, hidden)
    for layer, layer_type in enumerate(layer_types):
        prefix = f"blk.{layer}."
        expected[prefix + "attn_norm.weight"] = (hidden,)
        if layer_type == "mamba2":
            conv_width = inner + 2 * groups * state
            expected.update(
                {
                    prefix + "ssm_in.weight": (
                        2 * inner + 2 * groups * state + ssm_heads,
                        hidden,
                    ),
                    prefix + "ssm_conv1d.weight": (conv_width, conv),
                    prefix + "ssm_dt.bias": (ssm_heads,),
                    prefix + "ssm_a": (ssm_heads, 1),
                    prefix + "ssm_d": (ssm_heads, 1),
                    prefix + "ssm_norm.weight": (groups, inner // groups),
                    prefix + "ssm_out.weight": (hidden, inner),
                }
            )
            if prefix + "ssm_conv1d.bias" in actual:
                expected[prefix + "ssm_conv1d.bias"] = (conv_width,)
        elif layer_type == "full_attention":
            heads = heads_by_layer[layer]
            kv_heads = kv_by_layer[layer]
            expected.update(
                {
                    prefix + "attn_q.weight": (heads * head_dim, hidden),
                    prefix + "attn_k.weight": (kv_heads * head_dim, hidden),
                    prefix + "attn_v.weight": (kv_heads * head_dim, hidden),
                    prefix + "attn_output.weight": (hidden, heads * head_dim),
                }
            )
            if prefix + "attn_output.bias" in actual:
                expected[prefix + "attn_output.bias"] = (hidden,)
        elif layer_type == "mlp":
            width = int(metadata[f"{arch}.feed_forward_length"][layer])
            expected.update(
                {
                    prefix + "ffn_up.weight": (width, hidden),
                    prefix + "ffn_down.weight": (hidden, width),
                }
            )
            for projection, size in (("up", width), ("down", hidden)):
                name = prefix + f"ffn_{projection}.bias"
                if name in actual:
                    expected[name] = (size,)
        else:
            experts = int(metadata[f"{arch}.expert_count"])
            expert_width = int(metadata[f"{arch}.expert_feed_forward_length"])
            shared_width = int(metadata[f"{arch}.expert_shared_feed_forward_length"])
            latent = metadata.get(f"{arch}.moe_latent_size")
            expert_input = int(latent) if latent is not None else hidden
            expected.update(
                {
                    prefix + "ffn_gate_inp.weight": (experts, hidden),
                    prefix + "exp_probs_b.bias": (experts,),
                    prefix + "ffn_up_exps.weight": (
                        experts,
                        expert_width,
                        expert_input,
                    ),
                    prefix + "ffn_down_exps.weight": (
                        experts,
                        expert_input,
                        expert_width,
                    ),
                    prefix + "ffn_up_shexp.weight": (shared_width, hidden),
                    prefix + "ffn_down_shexp.weight": (hidden, shared_width),
                }
            )
            if latent is not None:
                expected[prefix + "ffn_latent_down.weight"] = (int(latent), hidden)
                expected[prefix + "ffn_latent_up.weight"] = (hidden, int(latent))

    malformed = sorted(
        f"{name}: expected {expected_shape}, got {shapes[name]}"
        for name, expected_shape in expected.items()
        if name in shapes and shapes[name] != expected_shape
    )
    if malformed:
        raise ValueError(f"Invalid Nemotron-H GGUF tensor shape(s): {malformed}")


def _validate_granitehybrid_tensor_shapes(gguf_model, layer_types, actual: set[str]) -> None:
    """Validate logical GraniteHybrid tensor shapes before graph construction."""
    metadata = gguf_model.metadata
    arch = gguf_model.architecture
    shapes = {
        name: tuple(int(dimension) for dimension in shape)
        for name, _raw, _qtype, shape in gguf_model.tensor_items_raw()
    }
    hidden = int(metadata[f"{arch}.embedding_length"])
    vocab = int(metadata.get(f"{arch}.vocab_size", 0))
    if not vocab:
        vocab = len(metadata.get("tokenizer.ggml.tokens", ()))
    heads_raw = metadata[f"{arch}.attention.head_count"]
    heads_by_layer = (
        [int(value) for value in heads_raw]
        if isinstance(heads_raw, (list, tuple, np.ndarray))
        else [int(heads_raw)] * len(layer_types)
    )
    kv_by_layer = [int(value) for value in metadata[f"{arch}.attention.head_count_kv"]]
    attention_heads = next(
        (
            heads_by_layer[index]
            for index, kind in enumerate(layer_types)
            if kind == "full_attention"
        ),
        0,
    )
    head_dim = int(
        metadata.get(
            f"{arch}.attention.key_length",
            hidden // attention_heads if attention_heads else 0,
        )
    )
    state = int(metadata[f"{arch}.ssm.state_size"])
    inner = int(metadata[f"{arch}.ssm.inner_size"])
    groups = int(metadata[f"{arch}.ssm.group_count"])
    ssm_heads = int(metadata[f"{arch}.ssm.time_step_rank"])
    conv = int(metadata[f"{arch}.ssm.conv_kernel"])
    expert_count = int(metadata.get(f"{arch}.expert_count", 0))
    expert_width = int(metadata[f"{arch}.feed_forward_length"])
    shared_width = int(metadata.get(f"{arch}.expert_shared_feed_forward_length", 0))

    expected: dict[str, tuple[int, ...]] = {
        "token_embd.weight": (vocab, hidden),
        "output_norm.weight": (hidden,),
    }
    if "output.weight" in actual:
        expected["output.weight"] = (vocab, hidden)
    for layer, layer_type in enumerate(layer_types):
        prefix = f"blk.{layer}."
        expected[prefix + "attn_norm.weight"] = (hidden,)
        expected[prefix + "ffn_norm.weight"] = (hidden,)
        if layer_type == "mamba2":
            conv_width = inner + 2 * groups * state
            expected.update(
                {
                    prefix + "ssm_in.weight": (
                        2 * inner + 2 * groups * state + ssm_heads,
                        hidden,
                    ),
                    prefix + "ssm_conv1d.weight": (conv_width, conv),
                    prefix + "ssm_dt.bias": (ssm_heads,),
                    prefix + "ssm_a": (ssm_heads, 1),
                    prefix + "ssm_d": (ssm_heads, 1),
                    prefix + "ssm_norm.weight": (groups, inner // groups),
                    prefix + "ssm_out.weight": (hidden, inner),
                }
            )
            if prefix + "ssm_conv1d.bias" in actual:
                expected[prefix + "ssm_conv1d.bias"] = (conv_width,)
        else:
            heads = heads_by_layer[layer]
            kv_heads = kv_by_layer[layer]
            expected.update(
                {
                    prefix + "attn_q.weight": (heads * head_dim, hidden),
                    prefix + "attn_q.bias": (heads * head_dim,),
                    prefix + "attn_k.weight": (kv_heads * head_dim, hidden),
                    prefix + "attn_k.bias": (kv_heads * head_dim,),
                    prefix + "attn_v.weight": (kv_heads * head_dim, hidden),
                    prefix + "attn_v.bias": (kv_heads * head_dim,),
                    prefix + "attn_output.weight": (hidden, heads * head_dim),
                }
            )
            if prefix + "attn_output.bias" in actual:
                expected[prefix + "attn_output.bias"] = (hidden,)

        if expert_count:
            expected.update(
                {
                    prefix + "ffn_gate_inp.weight": (expert_count, hidden),
                    prefix + "ffn_gate_exps.weight": (
                        expert_count,
                        expert_width,
                        hidden,
                    ),
                    prefix + "ffn_up_exps.weight": (
                        expert_count,
                        expert_width,
                        hidden,
                    ),
                    prefix + "ffn_down_exps.weight": (
                        expert_count,
                        hidden,
                        expert_width,
                    ),
                }
            )
            if shared_width:
                expected.update(
                    {
                        prefix + "ffn_gate_shexp.weight": (shared_width, hidden),
                        prefix + "ffn_up_shexp.weight": (shared_width, hidden),
                        prefix + "ffn_down_shexp.weight": (hidden, shared_width),
                    }
                )
        else:
            expected.update(
                {
                    prefix + "ffn_gate.weight": (expert_width, hidden),
                    prefix + "ffn_up.weight": (expert_width, hidden),
                    prefix + "ffn_down.weight": (hidden, expert_width),
                }
            )
            for projection, size in (
                ("gate", expert_width),
                ("up", expert_width),
                ("down", hidden),
            ):
                name = prefix + f"ffn_{projection}.bias"
                if name in actual:
                    expected[name] = (size,)

    malformed = sorted(
        f"{name}: expected {expected_shape}, got {shapes[name]}"
        for name, expected_shape in expected.items()
        if name in shapes and shapes[name] != expected_shape
    )
    if malformed:
        raise ValueError(f"Invalid GraniteHybrid GGUF tensor shape(s): {malformed}")


def _raise_for_invalid_t5_tensor_contract(gguf_model) -> None:
    """Validate the pinned T5/T5-encoder tensor closure and logical shapes."""
    from mobius.integrations.gguf._quant_registry import float_storage_type_ids

    architecture = gguf_model.architecture
    if architecture not in {"t5", "t5encoder"}:
        return

    metadata = gguf_model.metadata
    hidden = int(metadata[f"{architecture}.embedding_length"])
    intermediate = int(metadata[f"{architecture}.feed_forward_length"])
    encoder_layers = int(metadata[f"{architecture}.block_count"])
    decoder_layers = (
        int(metadata.get("t5.decoder_block_count", encoder_layers))
        if architecture == "t5"
        else 0
    )
    heads = int(metadata[f"{architecture}.attention.head_count"])
    key_length = int(metadata.get(f"{architecture}.attention.key_length", hidden // heads))
    value_length = int(metadata.get(f"{architecture}.attention.value_length", hidden // heads))
    buckets = int(metadata[f"{architecture}.attention.relative_buckets_count"])
    vocab = int(metadata.get(f"{architecture}.vocab_size", 0))
    if not vocab:
        vocab = len(metadata.get("tokenizer.ggml.tokens", ()))

    actual_items = list(gguf_model.tensor_items_raw())
    actual = {
        name: tuple(int(dim) for dim in shape) for name, _raw, _qtype, shape in actual_items
    }
    qtypes = {name: qtype for name, _raw, qtype, _shape in actual_items}
    layer_limits = {"enc": encoder_layers, "dec": decoder_layers}
    invalid_layer_indices = []
    for name in actual:
        match = re.match(r"^(enc|dec)\.blk\.([^.]+)\.", name)
        if match is None:
            continue
        stack, raw_index = match.groups()
        if not re.fullmatch(r"0|[1-9][0-9]*", raw_index):
            invalid_layer_indices.append(
                f"{name} (layer index {raw_index!r} is not canonical)"
            )
            continue
        layer = int(raw_index)
        if layer >= layer_limits[stack]:
            invalid_layer_indices.append(
                f"{name} (layer index {layer} is outside declared {stack} layer "
                f"count {layer_limits[stack]})"
            )
    if invalid_layer_indices:
        raise ValueError(
            f"{architecture} GGUF has invalid T5 layer tensor index(es): "
            f"{sorted(invalid_layer_indices)}"
        )

    required: dict[str, tuple[int, ...]] = {
        "token_embd.weight": (vocab, hidden),
        "enc.output_norm.weight": (hidden,),
        "enc.blk.0.attn_rel_b.weight": (buckets, heads),
    }
    optional: dict[str, tuple[int, ...]] = {
        "output.weight": (vocab, hidden),
    }
    if architecture == "t5":
        required.update(
            {
                "dec.output_norm.weight": (hidden,),
                "dec.blk.0.attn_rel_b.weight": (buckets, heads),
            }
        )

    def add_stack(prefix: str, layers: int, *, decoder: bool) -> None:
        for layer in range(layers):
            base = f"{prefix}.blk.{layer}"
            required.update(
                {
                    f"{base}.attn_norm.weight": (hidden,),
                    f"{base}.attn_q.weight": (heads * key_length, hidden),
                    f"{base}.attn_k.weight": (heads * key_length, hidden),
                    f"{base}.attn_v.weight": (heads * value_length, hidden),
                    f"{base}.attn_o.weight": (hidden, heads * value_length),
                    f"{base}.ffn_norm.weight": (hidden,),
                    f"{base}.ffn_up.weight": (intermediate, hidden),
                    f"{base}.ffn_down.weight": (hidden, intermediate),
                }
            )
            optional[f"{base}.attn_rel_b.weight"] = (buckets, heads)
            optional[f"{base}.ffn_gate.weight"] = (intermediate, hidden)
            if decoder:
                required.update(
                    {
                        f"{base}.cross_attn_norm.weight": (hidden,),
                        f"{base}.cross_attn_q.weight": (heads * key_length, hidden),
                        f"{base}.cross_attn_k.weight": (heads * key_length, hidden),
                        f"{base}.cross_attn_v.weight": (heads * value_length, hidden),
                        f"{base}.cross_attn_o.weight": (hidden, heads * value_length),
                    }
                )
                # llama.cpp loads this tensor but deliberately does not consume
                # it in the cross-attention graph.
                optional[f"{base}.cross_attn_rel_b.weight"] = (buckets, heads)

    add_stack("enc", encoder_layers, decoder=False)
    add_stack("dec", decoder_layers, decoder=True)

    missing = sorted(set(required) - set(actual))
    if missing:
        raise ValueError(f"{architecture} GGUF is missing required T5 tensor(s): {missing}")
    expected = {**optional, **required}
    malformed = {
        name: (expected[name], actual[name])
        for name in actual
        if name in expected and actual[name] != expected[name]
    }
    if malformed:
        raise ValueError(f"{architecture} GGUF has invalid T5 tensor shape(s): {malformed}")

    float_type_ids = float_storage_type_ids()
    small_non_float = []
    for name, qtype in qtypes.items():
        if name.endswith(("_norm.weight", "attn_rel_b.weight")):
            qtype_id = getattr(qtype, "value", qtype)
            if qtype_id not in float_type_ids:
                small_non_float.append(name)
    if small_non_float:
        raise ValueError(
            f"{architecture} GGUF relative-bias and norm tensors must remain float: "
            f"{sorted(small_non_float)}"
        )
    ignored_names = (
        {"output.weight"}
        if architecture == "t5encoder"
        else {f"dec.blk.{layer}.cross_attn_rel_b.weight" for layer in range(decoder_layers)}
    )
    ignored = sorted(set(actual) & ignored_names)
    if ignored:
        logger.warning(
            "Ignoring pinned llama.cpp T5 tensor(s) that do not participate in "
            "the architecture's output graph: %s",
            ignored,
        )


def _raise_for_malformed_recurrent_tensors(gguf_model) -> None:
    """Reject suffixes not created by the pinned C++ tensor loaders."""
    from mobius.integrations.gguf._tensor_mapping import map_gguf_to_hf_names
    from mobius.integrations.gguf._upstream import upstream_architecture

    upstream = upstream_architecture(gguf_model.architecture)
    if upstream is None or not upstream.tensor_names:
        return

    expected = set(upstream.tensor_names)
    malformed = []
    for name in gguf_model.tensor_names:
        template = re.sub(
            r"^((?:enc\.|dec\.)?blk)\.\d+\.",
            r"\1.{bid}.",
            name,
        )
        mapped_sidecar = name.endswith((".scale", ".input_scale")) and (
            map_gguf_to_hf_names(name, gguf_model.architecture) is not None
        )
        if template not in expected and not mapped_sidecar:
            malformed.append(name)
    if malformed:
        raise ValueError(
            f"Malformed {gguf_model.architecture} GGUF tensor name(s): {malformed}. "
            "The suffixes do not match the pinned llama.cpp tensor creation sites."
        )


def _raise_for_unsupported_auxiliary_quantization(gguf_model) -> None:
    """Reject scale sidecars whose semantics the target quantization ABI cannot express."""
    from mobius.integrations.gguf._mtp import map_gguf_mtp_to_hf_names
    from mobius.integrations.gguf._tensor_mapping import map_gguf_to_hf_names

    architecture = gguf_model.architecture
    if architecture == MMPROJ_ARCHITECTURE:
        # The mmproj registry validates role-specific tensor closure. Running
        # text tensor mapping here would misclassify clip sidecars as standalone
        # language models before the more actionable projector error can fire.
        return
    expert_count = gguf_model.get_metadata(f"{architecture}.expert_count")
    block_count = int(gguf_model.get_metadata(f"{architecture}.block_count", 0))
    mtp_count = int(gguf_model.get_metadata(f"{architecture}.nextn_predict_layers", 0))
    mtp_block_indices = range(max(0, block_count - mtp_count), block_count)
    for gguf_name, _raw, _qtype, shape in gguf_model.tensor_items_raw():
        suffix = next(
            (
                candidate
                for candidate in (".input_scale", ".scale")
                if gguf_name.endswith(candidate)
            ),
            None,
        )
        if suffix is None:
            continue
        hf_name = map_gguf_to_hf_names(gguf_name, architecture)
        if hf_name is None:
            hf_name = next(
                (
                    mapped
                    for block_index in mtp_block_indices
                    if (mapped := map_gguf_mtp_to_hf_names(gguf_name, block_index)) is not None
                ),
                None,
            )
        if hf_name is None:
            continue

        is_expert_scale = ".mlp.experts." in hf_name
        expected = (int(expert_count),) if is_expert_scale and expert_count else (1,)
        actual = tuple(int(dim) for dim in shape)
        if actual != expected:
            raise ValueError(
                f"Invalid GGUF auxiliary quantization tensor {gguf_name!r}: "
                f"expected shape {expected}, got {actual}"
            )
        if architecture == "bitnet" and suffix == ".scale":
            # BitNet's pinned loader treats this optional [1] tensor as an
            # output multiplier for the paired projection. The architecture
            # processor folds it into the dequantized matrix on the explicit
            # float route; it is not a target quantization sidecar.
            continue
        raise ValueError(
            f"GGUF auxiliary quantization tensor {gguf_name!r} maps to {hf_name!r}, "
            "but Mobius cannot represent GGUF scale/input_scale sidecars "
            "(including NVFP4 scale2) in its expert/projection quantization ABI. "
            "This file is rejected before graph construction to avoid silently "
            "dropping required quantization data."
        )


def _looks_like_hf_repo_id(value: str) -> bool:
    """Heuristic: ``value`` matches ``owner/repo`` (no path separators, no .gguf suffix)."""
    if value.startswith((".", "/", "~")):
        return False
    parts = value.split("/")
    return len(parts) == 2 and all(p and not p.endswith(".gguf") for p in parts)


def _select_complete_hf_gguf_set(
    repo_files: Collection[str],
    filename: str | None,
) -> tuple[str, list[str]]:
    """Select one immutable logical GGUF and enumerate its complete shard set."""
    gguf_files = sorted(name for name in repo_files if name.lower().endswith(".gguf"))
    if not gguf_files:
        raise FileNotFoundError("The Hugging Face repository contains no *.gguf files.")

    if filename is None:
        plain_files = [
            name
            for name in gguf_files
            if _GGUF_SHARD_FILENAME_RE.search(PurePosixPath(name).name) is None
        ]
        shard_groups: dict[tuple[str, str, int], list[str]] = {}
        for name in gguf_files:
            path = PurePosixPath(name)
            match = _GGUF_SHARD_FILENAME_RE.search(path.name)
            if match is None:
                continue
            prefix = path.name[: match.start()]
            key = (path.parent.as_posix(), prefix, int(match.group("count")))
            shard_groups.setdefault(key, []).append(name)
        if len(plain_files) == 1 and not shard_groups:
            return plain_files[0], plain_files
        if len(shard_groups) == 1 and not plain_files:
            group = next(iter(shard_groups.values()))
            filename = sorted(group)[0]
        else:
            raise ValueError(
                "The Hugging Face repository contains multiple logical GGUF models: "
                f"{gguf_files}. Specify one via 'owner/repo:<filename.gguf>'."
            )

    if filename not in gguf_files:
        raise FileNotFoundError(
            f"GGUF file {filename!r} is not present in the selected Hugging Face revision."
        )
    selected_path = PurePosixPath(filename)
    selected_match = _GGUF_SHARD_FILENAME_RE.search(selected_path.name)
    if selected_match is None:
        return filename, [filename]

    prefix = selected_path.name[: selected_match.start()]
    count = int(selected_match.group("count"))
    if count > MAX_GGUF_SHARD_COUNT:
        raise ValueError(
            f"GGUF split set {filename!r} declares {count} shards, exceeding "
            f"the supported maximum {MAX_GGUF_SHARD_COUNT}. No payload was downloaded."
        )
    by_index: dict[int, str] = {}
    for candidate in gguf_files:
        candidate_path = PurePosixPath(candidate)
        if candidate_path.parent != selected_path.parent:
            continue
        match = _GGUF_SHARD_FILENAME_RE.search(candidate_path.name)
        if match is None:
            continue
        if (
            candidate_path.name[: match.start()] != prefix
            or int(match.group("count")) != count
        ):
            continue
        index = int(match.group("index"))
        if index in by_index:
            raise ValueError(
                f"Duplicate Hugging Face GGUF shard index {index:05d}: "
                f"{by_index[index]!r} and {candidate!r}."
            )
        by_index[index] = candidate

    missing = [index for index in range(1, count + 1) if index not in by_index]
    if missing:
        raise ValueError(
            f"Incomplete Hugging Face GGUF split set for {filename!r}: declared "
            f"{count} shards but missing indices {[f'{index:05d}' for index in missing]}. "
            "No payload was downloaded."
        )
    extra = sorted(index for index in by_index if index < 1 or index > count)
    if extra:
        raise ValueError(
            f"GGUF split set for {filename!r} has out-of-range shard indices {extra}."
        )
    return filename, [by_index[index] for index in range(1, count + 1)]


def _select_hf_gguf_set_from_split_headers(
    repo_files: Collection[str],
    *,
    repo_id: str,
    selected_filename: str,
    revision: str,
    selected_preflight: _GGUFPreflightRevision,
) -> list[str]:
    """Enumerate renamed split siblings from authoritative bounded headers."""
    selected_info = selected_preflight.header_info
    _validate_preflight_split_header(
        selected_info,
        source=f"{repo_id}@{revision}:{selected_filename}",
    )
    if selected_info.split_count is None or selected_info.split_count <= 1:
        return [selected_filename]
    assert selected_info.split_tensors_count is not None

    selected_path = PurePosixPath(selected_filename)
    candidates = sorted(
        name
        for name in repo_files
        if PurePosixPath(name).parent == selected_path.parent
        and name.lower().endswith(".gguf")
    )
    candidate_limit = selected_info.split_count * _GGUF_SPLIT_DISCOVERY_MULTIPLIER
    if len(candidates) > candidate_limit:
        raise ValueError(
            f"Renamed GGUF split discovery found {len(candidates)} candidate files, "
            f"exceeding the bounded limit {candidate_limit} for split.count="
            f"{selected_info.split_count}. No additional headers or payloads were read."
        )
    preflights: dict[str, _GGUFPreflightRevision] = {selected_filename: selected_preflight}
    by_split_no: dict[int, str] = {}
    primary_architecture: str | None = None
    declared_architectures: dict[str, str] = {}
    for name in candidates:
        preflight = preflights.get(name)
        if preflight is None:
            candidate_preflight = _preflight_hf_gguf_file(
                repo_id,
                name,
                revision=revision,
                dispatch_architecture=False,
            )
            if not isinstance(candidate_preflight, _GGUFPreflightRevision):
                raise ValueError(
                    f"Cannot enumerate renamed GGUF split sibling {name!r} without "
                    "bounded split metadata. No payload was downloaded."
                )
            preflight = candidate_preflight
            preflights[name] = preflight
        if str(preflight) != revision:
            raise ValueError(
                f"GGUF sibling {name!r} resolved to {str(preflight)!r}, expected "
                f"the pinned revision {revision!r}. No payload was downloaded."
            )
        info = preflight.header_info
        if (
            info.split_count != selected_info.split_count
            or info.split_tensors_count != selected_info.split_tensors_count
        ):
            continue
        assert info.split_no is not None
        if info.split_no in by_split_no:
            raise ValueError(
                "Ambiguous GGUF split set discovered from bounded headers: "
                f"{by_split_no[info.split_no]!r} and {name!r} both declare "
                f"split.no={info.split_no}. No payload was downloaded."
            )
        by_split_no[info.split_no] = name
        if info.split_no == 0:
            if info.architecture is None:
                raise ValueError(
                    f"Primary GGUF shard {name!r} has no general.architecture. "
                    "No payload was downloaded."
                )
            primary_architecture = info.architecture
        if info.architecture is not None:
            declared_architectures[name] = info.architecture

    observed_nos = sorted(by_split_no)
    complete_nos = len(observed_nos) == selected_info.split_count and all(
        split_no == expected for expected, split_no in enumerate(observed_nos)
    )
    if not complete_nos:
        raise ValueError(
            "Incomplete GGUF split set discovered from bounded headers: "
            f"expected split.no values 0..{selected_info.split_count - 1}, got "
            f"{observed_nos}. No payload was downloaded."
        )
    if primary_architecture is None:
        raise ValueError(
            "GGUF split set has no authoritative primary architecture. "
            "No payload was downloaded."
        )
    primary_filename = by_split_no[0]
    _raise_for_unsupported_gguf_architecture(
        primary_architecture,
        source=f"{repo_id}@{revision}:{primary_filename}",
        allow_preflight_only=True,
    )
    if primary_architecture == "qwen4exp":
        from mobius.integrations.gguf._qwen4_exp import reject_qwen4exp_payload

        reject_qwen4exp_payload()
    mismatched_architectures = {
        name: architecture
        for name, architecture in declared_architectures.items()
        if architecture != primary_architecture
    }
    if mismatched_architectures:
        raise ValueError(
            f"GGUF split siblings disagree with primary architecture "
            f"{primary_architecture!r}: {mismatched_architectures}. "
            "No payload was downloaded."
        )
    return [by_split_no[index] for index in range(selected_info.split_count)]


def _existing_disk_usage_path(path: Path) -> Path:
    """Return the nearest existing ancestor accepted by ``disk_usage``."""
    candidate = path.expanduser().absolute()
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate


def _preflight_hf_download_space(total_bytes: int, *, cache_path: Path) -> None:
    """Fail before download when the Hub cache cannot hold the complete set."""
    if total_bytes < 0:
        raise ValueError(f"GGUF download size cannot be negative, got {total_bytes} bytes.")
    if total_bytes == 0:
        return
    usage = shutil.disk_usage(_existing_disk_usage_path(cache_path))
    if usage.free < total_bytes:
        raise OSError(
            "Insufficient free space for the complete GGUF split set: "
            f"requires {total_bytes:,} bytes ({total_bytes / (1 << 30):.2f} GiB), "
            f"but only {usage.free:,} bytes ({usage.free / (1 << 30):.2f} GiB) "
            f"are available for the Hugging Face cache at {cache_path}. "
            "Free space or set HF_HOME/HF_HUB_CACHE to a larger volume; no shard "
            "download was started."
        )


def _download_hf_gguf_shards(
    api: HfApi,
    *,
    repo_id: str,
    selected_filename: str,
    shard_filenames: list[str],
    revision: str,
) -> _ResolvedGGUFPath:
    """Preflight and download one complete shard set at an immutable revision."""
    from huggingface_hub.constants import HF_HUB_CACHE

    paths_info = api.get_paths_info(
        repo_id,
        shard_filenames,
        revision=revision,
        expand=True,
    )
    info_by_path = {getattr(info, "path", None): info for info in paths_info}
    sizes: dict[str, int] = {}
    sha256_by_remote: dict[str, str] = {}
    required_bytes = 0
    for name in shard_filenames:
        info = info_by_path.get(name)
        size = int(getattr(info, "size", 0) or 0)
        if info is None or size <= 0:
            raise ValueError(
                f"Hugging Face metadata omitted a positive size for GGUF shard "
                f"{name!r} at {repo_id}@{revision}; no payload was downloaded."
            )
        lfs = getattr(info, "lfs", None)
        sha256 = (
            lfs.get("sha256") or lfs.get("oid")
            if isinstance(lfs, Mapping)
            else getattr(lfs, "sha256", None) or getattr(lfs, "oid", None)
        )
        if isinstance(sha256, str) and sha256.startswith("sha256:"):
            sha256 = sha256.removeprefix("sha256:")
        if not isinstance(sha256, str) or re.fullmatch(r"[0-9a-fA-F]{64}", sha256) is None:
            raise ValueError(
                f"Hugging Face metadata omitted a valid LFS SHA-256 for GGUF shard "
                f"{name!r} at {repo_id}@{revision}; no payload was downloaded."
            )
        sizes[name] = size
        sha256_by_remote[name] = sha256.lower()
        cached = try_to_load_from_cache(
            repo_id,
            name,
            cache_dir=HF_HUB_CACHE,
            revision=revision,
        )
        if not (
            isinstance(cached, str)
            and Path(cached).is_file()
            and Path(cached).stat().st_size == size
        ):
            required_bytes += size

    total_bytes = sum(sizes.values())
    _preflight_hf_download_space(required_bytes, cache_path=Path(HF_HUB_CACHE))
    logger.info(
        "Downloading complete GGUF split set: %d shards, %.3f GiB total, "
        "%.3f GiB not cached from %s@%s",
        len(shard_filenames),
        total_bytes / float(1 << 30),
        required_bytes / float(1 << 30),
        repo_id,
        revision,
    )

    downloaded: dict[str, str] = {}
    for name in shard_filenames:
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=name,
            revision=revision,
        )
        actual_size = Path(local_path).stat().st_size
        if actual_size != sizes[name]:
            raise OSError(
                f"Downloaded GGUF shard {name!r} has {actual_size:,} bytes, "
                f"expected {sizes[name]:,}; remove the corrupt cache entry and retry."
            )
        downloaded[name] = local_path

    expected_sizes: dict[str, int] = {}
    expected_sha256: dict[str, str] = {}
    for name in shard_filenames:
        basename = PurePosixPath(name).name
        if basename in expected_sizes:
            raise ValueError(
                f"GGUF shard manifest contains duplicate basename {basename!r}; "
                "the local reader cannot bind identities unambiguously."
            )
        expected_sizes[basename] = sizes[name]
        expected_sha256[basename] = sha256_by_remote[name]
    return _ResolvedGGUFPath(
        downloaded[selected_filename],
        expected_sha256=expected_sha256,
        expected_sizes=expected_sizes,
        shard_paths=[downloaded[name] for name in shard_filenames],
    )


def _resolve_gguf_path_impl(
    gguf_path: str | Path,
    *,
    allow_mmproj_companion: bool,
    keep_quantized: bool = True,
) -> str | _ResolvedGGUFPath:
    """Resolve a GGUF reference with an internal primary/companion context.

    Accepts:
    - An existing local filesystem path (returned unchanged).
    - A HuggingFace Hub reference ``"owner/repo"`` — the repo must contain
      exactly one ``*.gguf`` file, which is downloaded.
    - A HuggingFace Hub reference ``"owner/repo:filename.gguf"`` to pick a
      specific file from a multi-file repo.
    """
    raw = str(gguf_path)
    if Path(raw).exists():
        return raw

    # Split the optional ":filename" suffix before classifying so HF refs like
    # "owner/repo:weights.gguf" are not mistaken for a local path ending in .gguf.
    repo_revision, _, filename = raw.partition(":")
    repo_id, revision_separator, requested_revision = repo_revision.partition("@")
    revision = requested_revision if revision_separator else "main"
    if not revision:
        raise ValueError(f"HF GGUF reference {raw!r} has an empty revision")
    if not _looks_like_hf_repo_id(repo_id):
        # Looks like a local path that doesn't exist; let GGUFModel raise
        # FileNotFoundError with the original path.
        return raw

    api = HfApi()
    if filename and _GGUF_SHARD_FILENAME_RE.search(PurePosixPath(filename).name) is None:
        selected_files = [filename]
    else:
        repo_files = api.list_repo_files(repo_id, revision=revision)
        filename, selected_files = _select_complete_hf_gguf_set(repo_files, filename or None)
    primary_filename = selected_files[0]

    preflight_revision: str | _GGUFPreflightRevision | _GGUFPreflightFallbackRevision
    if allow_mmproj_companion:
        if len(selected_files) != 1:
            raise ValueError("Sharded mmproj companion GGUF files are not supported.")
        preflight_revision = _preflight_hf_mmproj_companion_file(
            repo_id,
            primary_filename,
            revision=revision,
        )
    else:
        preflight_revision = _preflight_hf_gguf_file(
            repo_id,
            primary_filename,
            revision=revision,
        )
    resolved_revision = str(preflight_revision)
    selected_header = getattr(preflight_revision, "header_info", None)
    if (
        len(selected_files) == 1
        and isinstance(preflight_revision, _GGUFPreflightFallbackRevision)
        and _GGUF_SHARD_FILENAME_RE.search(PurePosixPath(filename).name) is None
    ):
        pinned_files = api.list_repo_files(repo_id, revision=resolved_revision)
        selected_parent = PurePosixPath(filename).parent
        same_directory_ggufs = sorted(
            name
            for name in pinned_files
            if PurePosixPath(name).parent == selected_parent and name.lower().endswith(".gguf")
        )
        if same_directory_ggufs != [filename]:
            raise ValueError(
                f"Bounded header preflight did not establish whether {filename!r} "
                "is a standalone GGUF, and its directory contains other GGUF files "
                f"{same_directory_ggufs}. Refusing a potentially partial download."
            )
    if (
        len(selected_files) == 1
        and selected_header is not None
        and selected_header.split_count is not None
        and selected_header.split_count > 1
    ):
        assert isinstance(preflight_revision, _GGUFPreflightRevision)
        pinned_files = api.list_repo_files(repo_id, revision=resolved_revision)
        selected_files = _select_hf_gguf_set_from_split_headers(
            pinned_files,
            repo_id=repo_id,
            selected_filename=filename,
            revision=resolved_revision,
            selected_preflight=preflight_revision,
        )
    elif len(selected_files) > 1:
        pinned_files = api.list_repo_files(repo_id, revision=resolved_revision)
        filename, selected_files = _select_complete_hf_gguf_set(pinned_files, filename)

    if len(selected_files) > 1:
        return _download_hf_gguf_shards(
            api,
            repo_id=repo_id,
            selected_filename=filename,
            shard_filenames=selected_files,
            revision=resolved_revision,
        )

    logger.info("Downloading %s from %s", filename, repo_id)
    return hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        revision=resolved_revision,
    )


def _resolve_gguf_path(
    gguf_path: str | Path,
    keep_quantized: bool = True,
) -> str | _ResolvedGGUFPath:
    """Resolve a primary GGUF reference without allowing mmproj sidecars."""
    return _resolve_gguf_path_impl(
        gguf_path,
        allow_mmproj_companion=False,
        keep_quantized=keep_quantized,
    )


def _resolve_mmproj_companion_path(
    gguf_path: str | Path,
) -> str | _ResolvedGGUFPath:
    """Resolve an internal mmproj companion, allowing only ``clip`` Hub metadata."""
    return _resolve_gguf_path_impl(gguf_path, allow_mmproj_companion=True)


def _regular_file_identity_paths(paths: Collection[Path]) -> list[Path] | None:
    """Resolve opened shard paths to regular-file targets for identity hashing."""
    resolved_paths: list[Path] = []
    for path in paths:
        try:
            resolved = path.expanduser().resolve(strict=True)
        except OSError:
            return None
        if not resolved.is_file() or resolved.is_symlink():
            return None
        resolved_paths.append(resolved)
    return resolved_paths


def _logical_source_filename(reference: str | Path, resolved_path: str | Path) -> str:
    """Preserve an explicitly selected Hub-relative filename for evidence."""
    raw = str(reference)
    repo_revision, _, requested_filename = raw.partition(":")
    repo_id = repo_revision.partition("@")[0]
    if requested_filename and _looks_like_hf_repo_id(repo_id):
        return requested_filename
    resolved = Path(resolved_path)
    parts = resolved.parts
    if _looks_like_hf_repo_id(repo_id) and "snapshots" in parts:
        snapshot_index = parts.index("snapshots")
        if len(parts) > snapshot_index + 2:
            return Path(*parts[snapshot_index + 2 :]).as_posix()
    return resolved.name


_SPECIALIZED_ENCODER_FINGERPRINT_ARCHITECTURES = frozenset(
    {
        "eurobert",
        "neo-bert",
        "nomic-bert",
        "nomic-bert-moe",
        "jina-bert-v2",
        "jina-bert-v3",
        "gemma-embedding",
        "llama-embed",
    }
)
_SPECIALIZED_ENCODER_FINGERPRINT_FIELDS = (
    "encoder_use_token_type_embeddings",
    "encoder_q_bias",
    "encoder_k_bias",
    "encoder_v_bias",
    "encoder_ffn_up_bias",
    "encoder_ffn_down_bias",
    "encoder_qk_norm",
    "encoder_extra_attention_norm",
    "encoder_fused_geglu",
    "pooling_type",
    "embedding_dense_2_out",
    "embedding_dense_3_in",
    "encoder_fused_qkv",
)
_ARCHITECTURE_CONFIG_FINGERPRINT_FIELDS = {
    "attention_clamp": frozenset({"dbrx"}),
    "encoder_fused_qkv": frozenset({"jina-bert-v3"}),
    "moe_layer_frequency": frozenset({"ernie4_5-moe", "nomic-bert-moe"}),
    "routing_weight_normalization_floor": frozenset({"dots1", "ernie4_5-moe", "smallthinker"}),
}


def _graph_config_fields_for_fingerprint(config, gguf_arch: str) -> dict[str, object]:
    """Serialize only fields consumed by an architecture's imported graph."""
    fields = asdict(config)
    if gguf_arch not in _SPECIALIZED_ENCODER_FINGERPRINT_ARCHITECTURES:
        # Keep established route fingerprints byte-identical when encoder-only
        # graph fields are added to ArchitectureConfig.
        for field_name in _SPECIALIZED_ENCODER_FINGERPRINT_FIELDS:
            fields.pop(field_name, None)
    for field_name, consumers in _ARCHITECTURE_CONFIG_FINGERPRINT_FIELDS.items():
        if gguf_arch not in consumers:
            fields.pop(field_name, None)
    return fields


def _serialize_route_graph_config(config: Any, gguf_arch: str) -> str:
    """Serialize the architecture-isolated graph fields for route fingerprints."""
    import json

    return json.dumps(
        _graph_config_fields_for_fingerprint(config, gguf_arch),
        default=str,
        separators=(",", ":"),
        sort_keys=True,
    )


def build_from_gguf(
    gguf_path: str | Path,
    *,
    task: str | None = None,
    dtype: str | None = None,
    keep_quantized: bool = True,
    execution_provider: str = "default",
    mmproj: str | Path | None = None,
    image_token_id: int | None = None,
    static_cache: bool = False,
    max_seq_len: int | None = None,
    allow_dense_moe: bool | None = None,
    reuse_gguf_weights: bool = False,
    target_config: str | Path | Mapping[str, object] | None = None,
    _gguf_model: Any | None = None,
) -> ModelPackage:
    """Build an ONNX :class:`ModelPackage` from a GGUF file.

    1. Parse GGUF metadata → :class:`ArchitectureConfig`
    2. Look up the model class and task from the registry
    3. Build the ONNX graph (standard ``build_from_module`` pipeline)
    4. Map GGUF tensor names → HuggingFace names
    5. Replace native-block projection modules when present
    6. Apply architecture-specific tensor processors
    7. Run ``preprocess_weights()`` (HF → ONNX name mapping)
    8. Apply weights to the ONNX model

    By default, supported tensors use quantized target storage.
    For text-only builds, operator-native IQ/MXFP4 projection blocks
    are retained byte-for-byte for BlockQuantizedMatMul. Multimodal text
    backbones normalize quantized projections to their common affine target. GGUFs
    containing only F32, F16, or BF16 weights use the float path because there
    is no quantization to preserve.
    Quantized files with no supported preservation target raise an actionable
    error instead of silently falling back to a float model.

    Args:
        gguf_path: Path to a ``.gguf`` file or any member of a complete local
            split set, *or* a HuggingFace Hub
            reference of the form ``"owner/repo"`` (the repo must
            contain exactly one ``*.gguf`` file) or
            ``"owner/repo:filename.gguf"`` to pick a specific file. HF
            references are downloaded via ``huggingface_hub`` into the
            standard local cache. A selected Hub shard resolves every sibling
            at one immutable commit after a total-size/free-space preflight.
        task: Override the model task (e.g. ``"text-generation"``).
            When ``None``, the task is auto-detected from the
            model type.
        dtype: Override model dtype (e.g. ``"f16"``). When ``None``,
            defaults to float32.
        keep_quantized: Keep quantized target storage when supported. This is
            the default. Supported affine blocks are repacked,
            text-only runtime-supported native IQ/MXFP4 projection blocks
            retain their bytes, and incompatible source qtypes are explicitly
            classified as lossy dequantize/requantize conversions. This does not
            guarantee source-byte or source-value fidelity. Set to ``False`` to
            dequantize all weights to float.
        execution_provider: Target execution provider for EP-aware
            optimisations (e.g. ``"cpu"`` to apply the
            GroupQueryAttention rewrite). Defaults to ``"default"``
            (portable, no vendor fusions).
        mmproj: Optional path (or HF ref) to a companion ``clip``
            multimodal-projector GGUF. When set, this becomes the single
            entry point for a multimodal build: the text GGUF and the
            mmproj vision/audio encoder are fused into one multimodal
            :class:`ModelPackage` (delegates to
            :func:`build_gemma4_vlm_from_gguf`).
        image_token_id: Optional processor-owned image placeholder or sentinel
            ID for companion mmproj packages. It is forwarded unchanged; GGUF
            text vocabularies do not always serialize this processor contract.
        static_cache: When ``True``, build with a pre-allocated static KV
            cache (fixed-width buffers written in place via ``TensorScatter``)
            instead of the default dynamic concat-grow cache. Produces a
            fully static-shaped graph, which is required by fixed-shape
            runtimes such as the QNN HTP backend. Cannot be combined with an
            explicit *task* override.
        max_seq_len: Maximum sequence length for the static cache buffers.
            Only used when ``static_cache=True``. Defaults to the model's
            ``max_position_embeddings``.
        allow_dense_moe: Opt in to exporting routed MoE experts that have no
            sparse top-k fusion (they lower to per-expert native
            ``BlockQuantizedMatMul`` nodes, i.e. dense-all-expert compute with
            no performance guarantee). When ``None`` (default), the value of
            the ``allow_dense_moe_experts`` flag is used, which defaults to
            ``False`` — the build fails closed with a typed capability error
            rather than silently shipping a dense graph. This is a research /
            correctness knob and makes no throughput claim.
        reuse_gguf_weights: Reuse compatible tensor payloads directly from the
            original GGUF via ONNX external-data ranges. The GGUF must be a real
            file in the final flat output directory. Converted tensors are
            written once to ``model.onnx.data``.
        target_config: Exact target model directory, config path, or explicit
            config mapping for a ``dflash``/``eagle3`` speculative draft. A
            mapping must include the complete ``tokenizer_json`` object. Required for draft GGUFs
            and rejected for standalone architectures.

    Returns:
        A :class:`ModelPackage` containing the built model(s).

    Raises:
        ImportError: If the ``gguf`` package is not installed.
        FileNotFoundError: If the GGUF file does not exist.
        KeyError: If the GGUF architecture is not in the registry.
        ValueError: If *static_cache* is combined with an explicit *task*, or
            if a quantized input has no supported preservation target.
    """
    import dataclasses
    import hashlib
    import json

    # A companion mmproj GGUF turns this into a multimodal build: the text +
    # vision/audio encoders are assembled by the dedicated VLM builder. Keep
    # build_from_gguf as the single public entry point (text-only or multimodal).
    if mmproj is not None:
        if reuse_gguf_weights:
            raise ValueError(
                "reuse_gguf_weights=True does not yet support multimodal/mmproj packages."
            )
        if static_cache:
            raise ValueError("static_cache=True is not supported with a companion mmproj.")
        from mobius.integrations.gguf._mmproj import build_vlm_from_gguf

        parsed_source = {"_text_gguf_model": _gguf_model} if _gguf_model is not None else {}
        return build_vlm_from_gguf(
            gguf_path,
            mmproj,
            dtype=dtype,
            execution_provider=execution_provider,
            image_token_id=image_token_id,
            keep_quantized=keep_quantized,
            **parsed_source,
        )
    if image_token_id is not None:
        raise ValueError("image_token_id requires a companion mmproj package.")

    from mobius._builder import (
        build_from_module,
        resolve_dtype,
    )
    from mobius._registry import registry
    from mobius.integrations.gguf._arch_registry import get_arch_spec
    from mobius.integrations.gguf._config_mapping import (
        GGUF_ARCH_TO_MODEL_TYPE,
        gguf_to_config,
    )
    from mobius.integrations.gguf._shard_set import GgufShardSet, open_gguf_model
    from mobius.integrations.gguf._tensor_processors import (
        process_tensors,
    )
    from mobius.integrations.transformers import (
        _default_task_for_model,
    )

    if static_cache and task is not None:
        raise ValueError(
            "static_cache=True cannot be combined with an explicit task "
            "override; the static cache is wired through CausalLMTask."
        )

    from mobius._flags import flags as _mobius_flags

    if allow_dense_moe is None:
        allow_dense_moe = _mobius_flags.allow_dense_moe_experts

    # 1. Parse GGUF file (auto-download from HF Hub when given "owner/repo[:filename]").
    #    A ``-000i-of-000N.gguf`` split set is assembled directly from its shards
    #    (never merged into a second on-disk GGUF); a plain file opens as before.
    source_reference = str(gguf_path)
    resolved_gguf_path = _resolve_gguf_path(gguf_path, keep_quantized)
    expected_sha256 = getattr(resolved_gguf_path, "expected_sha256", None)
    expected_sizes = getattr(resolved_gguf_path, "expected_sizes", None)
    resolved_shard_paths = getattr(resolved_gguf_path, "shard_paths", None)
    gguf_path = str(resolved_gguf_path)
    logical_source_filename = _logical_source_filename(source_reference, gguf_path)
    source_path = Path(gguf_path)
    if source_path.is_symlink() and _GGUF_SHARD_FILENAME_RE.search(source_path.name) is None:
        from huggingface_hub.constants import HF_HUB_CACHE

        try:
            source_path.absolute().relative_to(Path(HF_HUB_CACHE).absolute())
        except ValueError:
            pass
        else:
            # Snapshot links point into the immutable content-addressed blob store.
            source_path = source_path.resolve(strict=True)
            gguf_path = str(source_path)
    if _gguf_model is not None:
        gguf_model = _gguf_model
    else:
        shard_open_kwargs: dict[str, Any] = {}
        if resolved_shard_paths is not None:
            shard_open_kwargs["shard_paths"] = resolved_shard_paths
        if expected_sha256 is not None:
            shard_open_kwargs["expected_sha256"] = expected_sha256
        if expected_sizes is not None:
            shard_open_kwargs["expected_sizes"] = expected_sizes
        gguf_model = open_gguf_model(gguf_path, **shard_open_kwargs)
    if isinstance(gguf_model, GgufShardSet):
        identity_paths = _regular_file_identity_paths(gguf_model.shard_paths)
        if identity_paths is not None:
            gguf_model._set_identity_paths(identity_paths)
    _validate_gguf_model(
        gguf_model,
        source=str(gguf_path),
        keep_quantized=keep_quantized,
    )
    if reuse_gguf_weights and isinstance(gguf_model, GgufShardSet):
        raise ValueError(
            "reuse_gguf_weights=True does not yet support multi-shard GGUF sets. "
            "Build without reuse to preserve current multi-shard import behavior."
        )
    if reuse_gguf_weights and not gguf_model.is_little_endian:
        raise ValueError(
            "reuse_gguf_weights=True requires a little-endian GGUF because ONNX "
            "external tensors interpret the referenced bytes as little-endian. "
            "Build without reuse to convert this file."
        )
    from mobius.integrations.gguf._tokenizer import inspect_gguf_tokenizer

    tokenizer_verdict = inspect_gguf_tokenizer(gguf_model.metadata, source=str(gguf_path))
    gguf_arch = gguf_model.architecture
    if gguf_arch == "plamo" and reuse_gguf_weights:
        raise ValueError(
            "reuse_gguf_weights=True is not supported for PLaMo because its Q/output "
            "projection shuffles require materialized value transforms"
        )
    if gguf_arch == "plamo" and static_cache:
        raise ValueError(
            "static_cache=True is not supported for PLaMo; the exact source contract "
            "stores cyclically expanded 40-head dynamic K/V state"
        )
    if static_cache and gguf_arch in {
        "gptneox",
        "jais",
        "mpt",
        "refact",
        "ernie4_5",
        "openelm",
    }:
        raise ValueError(
            f"static_cache=True is not supported for exact legacy {gguf_arch} GGUF models; "
            "their dedicated decoder layers currently implement dynamic KV cache only."
        )
    if static_cache and int(gguf_model.metadata.get(f"{gguf_arch}.nextn_predict_layers", 0)):
        raise ValueError(
            "static_cache=True cannot represent the GGUF MTP head's independent "
            "dynamic concat-grow KV cache; refusing to silently omit the sidecar"
        )
    logger.info("Loaded GGUF model: %s (arch=%s)", gguf_path, gguf_arch)
    preserve_quantization = keep_quantized and _has_quantized_weights(gguf_model, gguf_arch)
    arch_spec = try_get_arch_spec(gguf_arch)
    if (
        preserve_quantization
        and arch_spec is not None
        and arch_spec.quantized_import is not Support.SUPPORTED
    ):
        raise ValueError(
            f"GGUF architecture {gguf_arch!r} does not support keep_quantized=True: "
            f"{arch_spec.reason}"
        )
    if keep_quantized and not preserve_quantization:
        logger.info("GGUF contains no mapped quantized weights; using the float import path")
    # 2. Extract config from GGUF metadata
    config = gguf_to_config(gguf_model)
    spec = get_arch_spec(gguf_arch)
    from mobius.integrations.gguf._draft import (
        is_draft_architecture,
        validate_draft_pairing,
    )

    if target_config is not None and not is_draft_architecture(gguf_arch):
        raise ValueError(
            f"target_config is only valid for dflash/eagle3 draft GGUFs, got {gguf_arch!r}"
        )
    draft_manifest = (
        validate_draft_pairing(gguf_model, config, target_config)
        if is_draft_architecture(gguf_arch)
        else None
    )
    model_type = getattr(config, "_gguf_model_type", None)
    if model_type is None:
        model_type = GGUF_ARCH_TO_MODEL_TYPE.get(gguf_arch, gguf_arch)
    if gguf_arch in {"dream", "llada", "llada-moe", "rnd1"}:
        from mobius.tasks import MaskedDiffusionTask

        if static_cache:
            raise ValueError(
                f"static_cache=True is not valid for masked-diffusion {gguf_arch} GGUF; "
                "the model is bidirectional and has no KV cache"
            )
        if (
            task is not None
            and task != "masked-diffusion"
            and not isinstance(task, MaskedDiffusionTask)
        ):
            raise ValueError(
                f"{gguf_arch} GGUF only supports task='masked-diffusion', got {task!r}"
            )
    if static_cache and model_type in {"mamba", "mamba2"}:
        raise ValueError(
            f"static_cache=True is not supported for recurrent {model_type} GGUF models; "
            "they carry per-layer conv_state and ssm_state rather than a KV cache."
        )
    if static_cache and gguf_arch in {"kimi-linear", "kimi-k3"}:
        raise ValueError(
            f"{gguf_arch} does not support static cache: KDA layers carry three "
            "convolution histories and one recurrent matrix state"
        )
    if gguf_arch in {"kimi-linear", "kimi-k3"}:
        from mobius.tasks import KimiK3CausalLMTask, KimiLinearCausalLMTask

        expected_task, task_class = (
            ("kimi-linear-text-generation", KimiLinearCausalLMTask)
            if gguf_arch == "kimi-linear"
            else ("kimi-k3-text-generation", KimiK3CausalLMTask)
        )
        if task is not None and task != expected_task and not isinstance(task, task_class):
            raise ValueError(
                f"{gguf_arch} GGUF only supports the dedicated {expected_task!r} "
                "heterogeneous-state task"
            )
    if gguf_arch == "falcon-h1":
        from mobius.tasks import FalconH1CausalLMTask

        if static_cache:
            raise ValueError(
                "static_cache=True is not supported for falcon-h1 GGUF models; "
                "every layer requires a dynamic four-state K, V, convolution, and SSM ABI"
            )
        if preserve_quantization:
            raise ValueError(
                "keep_quantized=True is not supported for falcon-h1: recurrent and "
                "state-sensitive tensors must be dequantized while only exact "
                "attention/FFN MatMul roles may remain quantized"
            )
        if (
            task is not None
            and task != "falcon-h1-text-generation"
            and not isinstance(task, FalconH1CausalLMTask)
        ):
            raise ValueError(
                "falcon-h1 GGUF only supports the dedicated "
                "'falcon-h1-text-generation' four-state task"
            )
    if gguf_arch == "plamo2":
        from mobius.tasks import Plamo2CausalLMTask

        if static_cache:
            raise ValueError(
                "static_cache=True is not supported for plamo2 GGUF models; "
                "heterogeneous per-layer recurrent and KV states require the dynamic ABI"
            )
        if (
            task is not None
            and task != "plamo2-text-generation"
            and not isinstance(task, Plamo2CausalLMTask)
        ):
            raise ValueError(
                "plamo2 GGUF only supports the dedicated 'plamo2-text-generation' task"
            )
    if gguf_arch == "plamo":
        from mobius.tasks import PlamoCausalLMTask

        if (
            task is not None
            and task != "plamo-text-generation"
            and not isinstance(task, PlamoCausalLMTask)
        ):
            raise ValueError(
                "plamo GGUF only supports the dedicated 'plamo-text-generation' task"
            )
    if gguf_arch == "smallthinker":
        from mobius.tasks import SmallThinkerGGUFCausalLMTask

        if static_cache:
            raise ValueError(
                "static_cache=True is not supported for SmallThinker GGUF; "
                "the exact graph uses its dedicated dynamic concat-grow KV-cache task"
            )
        if (
            task is not None
            and task != "smallthinker-gguf-text-generation"
            and not isinstance(task, SmallThinkerGGUFCausalLMTask)
        ):
            raise ValueError(
                "smallthinker GGUF only supports the dedicated "
                "'smallthinker-gguf-text-generation' task"
            )
    if gguf_arch == "qwen4exp":
        from mobius.tasks import Qwen4ExpCausalLMTask

        if static_cache:
            raise ValueError(
                "static_cache=True is not supported for qwen4exp GGUF models; "
                "DeltaNet, PLE, QSA, and position histories require the dedicated "
                "heterogeneous dynamic-state ABI"
            )
        if (
            task is not None
            and task != "qwen4-exp-text-generation"
            and not isinstance(task, Qwen4ExpCausalLMTask)
        ):
            raise ValueError(
                "qwen4exp GGUF only supports the dedicated 'qwen4-exp-text-generation' task"
            )
    if gguf_arch in {
        "lfm2",
        "lfm2moe",
        "qwen35",
        "qwen35moe",
        "qwen3next",
        "jamba",
        "nemotron_h",
        "nemotron_h_moe",
        "granitehybrid",
    }:
        from mobius.tasks import HybridCausalLMTask

        if static_cache:
            raise ValueError(
                f"static_cache=True is not supported for hybrid {gguf_arch} GGUF models; "
                "attention layers carry KV while recurrent layers carry architecture-"
                "specific conv/recurrent state."
            )
        if (
            task is not None
            and task != "hybrid-text-generation"
            and not isinstance(task, HybridCausalLMTask)
        ):
            raise ValueError(
                f"{gguf_arch} GGUF only supports the mixed-state "
                f"'hybrid-text-generation' task, got {task!r}"
            )
    if model_type in {"bert", "modernbert", "t5encoder"} or gguf_arch in {
        "eurobert",
        "neo-bert",
        "nomic-bert",
        "nomic-bert-moe",
        "jina-bert-v2",
        "jina-bert-v3",
        "gemma-embedding",
        "llama-embed",
    }:
        if static_cache:
            raise ValueError("static_cache is not valid for encoder-only GGUF architectures")
        expected_task = (
            "t5-text-encoding" if model_type == "t5encoder" else "feature-extraction"
        )
        if task is not None and task != expected_task:
            raise ValueError(
                f"{gguf_arch} GGUF only supports task={expected_task!r}, got {task!r}"
            )
    if model_type == "t5":
        if static_cache:
            raise ValueError(
                "static_cache=True is not valid for T5 seq2seq GGUF; use the "
                "encoder/decoder cache contract"
            )
        if task is not None and task != "seq2seq":
            raise ValueError(f"t5 GGUF only supports task='seq2seq', got {task!r}")
    if is_draft_architecture(gguf_arch):
        if static_cache:
            raise ValueError(f"static_cache=True is not supported for {gguf_arch} drafts")
        expected_task = f"{gguf_arch}-draft"
        if task is not None and task != expected_task:
            raise ValueError(
                f"{gguf_arch} GGUF only supports task={expected_task!r}, got {task!r}"
            )

    # 2b. Architecture-resolution safety rail. When the GGUF architecture string
    # bridges to a specialised registry key, verify the metadata-derived config
    # actually describes that architecture before selecting its model class, so
    # a mislabelled or incomplete GGUF fails closed with precise reasons instead
    # of silently building the wrong graph.
    if model_type == "glm_moe_dsa":
        from mobius.integrations.gguf._config_mapping import assert_glm_moe_dsa_resolvable

        assert_glm_moe_dsa_resolvable(config, gguf_arch, source=str(gguf_path))

    # ``dataclasses.replace`` below (dtype / quantization) returns a fresh
    # instance that does NOT carry the private ``_gguf_*`` attributes set by
    # gguf_to_config (they are plain attributes, not dataclass fields). Capture
    # the MTP head metadata here — like ``model_type`` above — so it can be
    # re-attached onto the final config for auto-detection (step 4b).
    mtp_predict_layers = getattr(config, "_gguf_nextn_predict_layers", 0)
    mtp_block_indices = list(getattr(config, "_gguf_mtp_block_indices", []) or [])

    if dtype is not None:
        resolved = resolve_dtype(dtype)
        if resolved is not None:
            config = dataclasses.replace(config, dtype=resolved)

    # 3. Quantized path: detect dominant type and set config
    if preserve_quantization:
        from mobius._configs import QuantizationConfig
        from mobius._flags import flags
        from mobius.integrations.gguf._tencent_q1_0 import is_tencent_q1_0_layout

        bits, block_size, is_sym = _detect_quant_params(gguf_model, gguf_arch)
        # Float zero-point only when actually using Tencent's native 2-bit form.
        float_zp = is_tencent_q1_0_layout(gguf_model) and flags.tencent_q1_0_use_native_2bit
        quantize_embeddings = _can_quantize_embedding(
            gguf_model,
            gguf_arch,
            bits=bits,
            block_size=block_size,
        )
        quantize_lm_head = (
            quantize_embeddings
            if config.tie_word_embeddings
            else _can_quantize_lm_head(gguf_model, gguf_arch)
        )
        config = dataclasses.replace(
            config,
            quantization=QuantizationConfig(
                bits=bits,
                group_size=block_size,
                quant_method="gguf",
                sym=is_sym,
                float_zero_point=float_zp,
                quantize_embeddings=quantize_embeddings,
                quantize_lm_head=quantize_lm_head,
                tie_word_embeddings=quantize_lm_head and config.tie_word_embeddings,
            ),
        )
        logger.info(
            "Quantized mode: bits=%d, block_size=%d, symmetric=%s, "
            "float_zp=%s, embedding=%s, lm_head=%s",
            bits,
            block_size,
            is_sym,
            float_zp,
            quantize_embeddings,
            quantize_lm_head,
        )

    # 4. Look up module class and resolve task
    module_type = spec.module_type or model_type
    module_class = registry.get(module_type)
    resolved_task: str | ModelTask
    if model_type == "t5":
        from mobius.tasks import Seq2SeqTask

        resolved_task = Seq2SeqTask(
            use_cross_attention_cache=True,
            use_attention_masks=True,
        )
    elif static_cache:
        from mobius.tasks import CausalLMTask

        resolved_task = CausalLMTask(static_cache=True, max_seq_len=max_seq_len)
    elif gguf_arch in {
        "eurobert",
        "neo-bert",
        "nomic-bert",
        "nomic-bert-moe",
        "jina-bert-v2",
        "jina-bert-v3",
    }:
        from mobius.tasks import GGUFEncoderFeatureExtractionTask

        resolved_task = GGUFEncoderFeatureExtractionTask()
    elif gguf_arch in {"gemma-embedding", "llama-embed"}:
        from mobius.tasks import GGUFEmbeddingFeatureExtractionTask

        resolved_task = GGUFEmbeddingFeatureExtractionTask()
    elif task is None:
        resolved_task = _default_task_for_model(module_type)
    else:
        resolved_task = task

    # 4b. Auto-detect the Qwen3.5/3.8 MTP / "nextn" self-speculative head: if
    # the source GGUF ships the trailing nextn head block (surfaced by
    # ``has_mtp_head`` from ``<arch>.nextn_predict_layers`` > 0 + the
    # ``blk.<N>.nextn.*`` tensors), always emit the MTP sidecar — it is a purely
    # additive artifact that text-only consumers ignore. No opt-in flag: the
    # decision is driven entirely by presence in the source. When present, expose
    # the post-final-norm ``mtp_seed`` output consumed by the orchestrator. Keep
    # the final layer's ordinary ``hidden_states.N`` capture as the distinct
    # pre-final-norm ABI; neither output may stand in for the other. These fields
    # must be set before graph construction. Direct assignment preserves the
    # ``_gguf_*`` metadata attributes on the config. Static-cache requests fail
    # closed because the sidecar requires its own dynamic concat-grow cache.
    from mobius.integrations.gguf._mtp import build_mtp_head_from_gguf, has_mtp_head

    # Re-attach the GGUF metadata dropped by the dtype/quantization
    # ``dataclasses.replace`` calls above so auto-detection sees it on the final
    # config instance (and ``derive_mtp_config`` can read model_type/quant/dtype).
    # ``_gguf_arch`` matters most: it is the key ``process_tensors`` dispatches
    # on, so losing it here would silently demote every non-float32 and every
    # quantized import to the ``model_type`` fallback.
    config._gguf_arch = gguf_arch
    config._gguf_model_type = model_type
    config._gguf_nextn_predict_layers = mtp_predict_layers
    config._gguf_mtp_block_indices = mtp_block_indices
    emit_mtp_head = has_mtp_head(config)
    if has_mtp_head(config) and static_cache:
        raise ValueError(
            "static_cache=True cannot represent the GGUF MTP head's independent "
            "dynamic concat-grow KV cache; refusing to silently omit the sidecar"
        )

    if emit_mtp_head:
        config.output_final_hidden_state = True
        seed_index = int(config.num_hidden_layers) - 1
        existing = list(config.output_layer_indices or [])
        if seed_index not in existing:
            existing.append(seed_index)
        config.output_layer_indices = existing
        logger.info(
            "MTP head detected in source: exposing post-final-norm mtp_seed and "
            "pre-final-norm hidden_states.%d",
            seed_index,
        )

    # 5. Build ONNX graph
    module = module_class(config)
    float_linear_dequantization_types = _float_linear_dequantization_types(
        module,
        gguf_arch,
    )
    if preserve_quantization:
        _reject_unsupported_quantization_preservation(
            gguf_model,
            gguf_arch,
            preserve_quantization=True,
            dequantize_float_linear_types=float_linear_dequantization_types,
        )
        _replace_native_block_linears(module, gguf_model, gguf_arch)
        # The sparse-MoE honesty gate runs post-export on the final graph state
        # (see step 9b): routed native-block experts are first collapsed into a
        # sparse top-k pkg.nxrt::BlockQuantizedMoE by fuse_block_quantized_moe,
        # then the gate fails closed if any per-expert dense storm survives.
        # Enforcing here (pre-export, module level) would reject the very layers
        # the fusion can now collapse, so the authority moved to the graph.
    quantization_report = _preflight_quantization_report(
        gguf_model,
        gguf_arch,
        module,
        config,
        preserve_quantization=preserve_quantization,
        target_bits=(config.quantization.bits if preserve_quantization else None),
        target_block_size=(config.quantization.group_size if preserve_quantization else None),
        execution_provider=execution_provider,
        dequantize_float_linear_types=float_linear_dequantization_types,
        emit_warning=not emit_mtp_head,
        include_tensor=(
            (
                lambda name: (
                    not any(
                        name.startswith(f"blk.{block_index}.")
                        for block_index in mtp_block_indices
                    )
                )
            )
            if emit_mtp_head
            else None
        ),
    )
    mtp_pkg = None
    if emit_mtp_head:
        mtp_preflight_received = False

        def combine_mtp_report(mtp_report) -> None:
            nonlocal quantization_report, mtp_preflight_received
            from mobius.integrations.gguf._quantization_report import (
                GGUFQuantizationReport,
            )

            quantization_report = GGUFQuantizationReport.combine(
                quantization_report,
                mtp_report,
            )
            mtp_preflight_received = True
            warning = quantization_report.warning_message()
            if warning is not None:
                logger.warning("%s", warning)

        mtp_pkg = build_mtp_head_from_gguf(
            gguf_model,
            config,
            preserve_quantization=preserve_quantization,
            execution_provider=execution_provider,
            on_preflight=combine_mtp_report,
        )
        if not mtp_preflight_received:
            warning = quantization_report.warning_message()
            if warning is not None:
                logger.warning("%s", warning)
    pkg = build_from_module(
        module, config, resolved_task, execution_provider=execution_provider
    )
    pkg.gguf_quantization_report = quantization_report
    logger.info(
        "Built ONNX graph for %s (%d components)",
        model_type,
        len(pkg),
    )

    # 6. Load tensors from GGUF → state_dict
    reuse_candidates_by_id = {} if reuse_gguf_weights else None
    if preserve_quantization:
        state_dict = _load_quantized_state_dict(
            gguf_model,
            gguf_arch,
            module,
            config,
            reuse_candidates=reuse_candidates_by_id,
            dequantize_float_linear_types=float_linear_dequantization_types,
        )
    else:
        state_dict = _load_dequantized_state_dict(
            gguf_model,
            gguf_arch,
            reuse_candidates=reuse_candidates_by_id,
        )

    logger.info(
        "Mapped %d state_dict entries from GGUF tensors",
        len(state_dict),
    )

    # 7. Apply architecture-specific tensor processors.
    # For the quantized path, only float tensors go through
    # process_tensors; quantized Q/K tensors were permuted in
    # _load_quantized_state_dict already.
    if preserve_quantization:
        float_keys = {
            k
            for k in state_dict
            if not (
                k.endswith((".scales", ".zero_points"))
                or _is_quantized_weight(k, state_dict)
                or _is_native_block_weight(k, state_dict)
            )
        }
        float_dict = {k: state_dict[k] for k in float_keys}
        quant_dict = {k: state_dict[k] for k in state_dict if k not in float_keys}
        before_processing = dict(float_dict)
        float_dict = process_tensors(float_dict, config)
        if reuse_candidates_by_id is not None:
            _record_reuse_process_transforms(
                before_processing,
                float_dict,
                reuse_candidates_by_id,
                config,
            )
        state_dict = {**float_dict, **quant_dict}
    else:
        before_processing = dict(state_dict)
        state_dict = process_tensors(state_dict, config)
        if reuse_candidates_by_id is not None:
            _record_reuse_process_transforms(
                before_processing,
                state_dict,
                reuse_candidates_by_id,
                config,
            )

    # 7b. Normalize GGUF-specific weight shapes to match HF conventions.
    # This converts GGUF tensor quirks (stacked experts, 1D gates, 2D
    # conv weights, suffix artifacts) into the shapes that HF models
    # produce, so preprocess_weights only needs to handle HF→ONNX.
    state_dict = _normalize_gguf_weights(state_dict, gguf_arch, config)

    # 8. Run model-specific preprocess_weights (HF → ONNX names)
    if hasattr(module, "preprocess_weights"):
        state_dict = module.preprocess_weights(state_dict)
    if is_draft_architecture(gguf_arch):
        state_dict.pop("d2t", None)

    # 9. Apply weights to ONNX model
    prefix_map = getattr(module, "weight_prefix_map", None)
    pkg.apply_weights(
        state_dict,
        prefix_map=prefix_map,
        fold_constants=not reuse_gguf_weights,
    )
    if reuse_gguf_weights:
        from mobius.integrations.gguf._reuse import attach_reused_initializers

        if emit_mtp_head:
            raise ValueError(
                "reuse_gguf_weights=True does not yet support packages with an MTP sidecar."
            )
        final_candidates = {
            name: reuse_candidates_by_id[id(tensor)]
            for name, tensor in state_dict.items()
            if id(tensor) in reuse_candidates_by_id
        }
        attach_reused_initializers(pkg, gguf_path, final_candidates, gguf_model)

    # 9b. Sparse-MoE fusion + honesty gate (final graph state).
    # Now that every native block carries its real packed bytes, collapse the
    # routed per-expert BlockQuantizedMatMul storm into one sparse top-k
    # pkg.nxrt::BlockQuantizedMoE per layer (byte-for-byte, no requantization),
    # then fail closed if any dense-all-expert storm still survives.
    if preserve_quantization:
        _fuse_native_block_moe(pkg, allow_dense=allow_dense_moe)
        _assert_sparse_moe_graph(pkg, source=str(gguf_path), allow_dense=allow_dense_moe)

    # 10. Build the trailing MTP / "nextn" self-speculative head sidecar from
    # the GGUF's ``blk.<nextn>.*`` tensors (dropped by the backbone build) and
    # attach it to the package so the CLI can save it into a ``mtp/`` subdir.
    # Auto-detected from source-tensor presence (see step 4b).
    if mtp_pkg is not None:
        pkg.mtp_head = mtp_pkg

    if draft_manifest is not None:
        pkg.draft_manifest = draft_manifest
    canonical_source_path = (
        gguf_model.shard_paths[0] if isinstance(gguf_model, GgufShardSet) else Path(gguf_path)
    )
    if isinstance(gguf_model, GgufShardSet):
        logical_source_filename = (
            Path(logical_source_filename).with_name(canonical_source_path.name).as_posix()
        )
    pkg.gguf_source_path = str(canonical_source_path.resolve())
    pkg.gguf_source_filename = logical_source_filename
    pkg.gguf_architecture = spec.gguf_arch
    pkg.gguf_execution_provider = execution_provider
    if dataclasses.is_dataclass(resolved_task) and not isinstance(resolved_task, type):
        task_state: object = dataclasses.asdict(resolved_task)
    elif isinstance(resolved_task, str):
        task_state = resolved_task
    else:
        task_state = dict(sorted(vars(resolved_task).items()))
    graph_config = _serialize_route_graph_config(config, spec.gguf_arch)
    pkg.gguf_import_route = json.dumps(
        {
            "architecture": spec.gguf_arch,
            "config_sha256": hashlib.sha256(graph_config.encode()).hexdigest(),
            "execution_provider": execution_provider,
            "model_type": spec.model_type,
            "module_type": module_type,
            "preserve_quantization": preserve_quantization,
            "registry_import": {
                "config_key_map": spec.config_key_map,
                "config_postprocessor": spec.config_postprocessor,
                "llama_qk_permute": spec.llama_qk_permute,
                "offset_norm": spec.offset_norm,
                "required_metadata": spec.required_metadata,
                "rope_interleave": spec.rope_interleave,
                "tensor_processor": spec.tensor_processor,
                "v_head_reorder": spec.v_head_reorder,
                "vlm_builder": spec.vlm_builder,
            },
            "route_schema": 1,
            "static_cache": static_cache,
            "task": {
                "class": f"{type(resolved_task).__module__}.{type(resolved_task).__qualname__}",
                "state": task_state,
            },
            "tensor_map_recipe": spec.tensor_map_recipe,
        },
        default=str,
        separators=(",", ":"),
        sort_keys=True,
    )
    if spec.runtime is Support.SUPPORTED:
        if not gguf_model.source_matches_path():
            raise ValueError(
                "GGUF source changed after the reader opened it; refusing to bind the graph "
                "to a different artifact identity."
            )
        from mobius.integrations.gguf._runtime_evidence import gguf_artifact_identity

        pkg.gguf_artifact_identity = gguf_artifact_identity(
            Path(gguf_path),
            gguf_model,
            architecture=spec.gguf_arch,
            filename=logical_source_filename,
        )
        if not gguf_model.source_matches_path():
            raise ValueError(
                "GGUF source changed while its graph and artifact identity were being built."
            )
    pkg.gguf_tokenizer_verdict = tokenizer_verdict

    return pkg


def _record_reuse_process_transforms(
    before: dict,
    after: dict,
    candidates: dict,
    config,
) -> None:
    """Carry exact-byte candidates through known graph-expressible transforms."""
    import dataclasses

    from mobius.integrations.gguf._tensor_processors import needs_llama_qk_permute

    for name, transformed in after.items():
        original = before.get(name)
        if original is None or id(original) == id(transformed):
            continue
        candidate = candidates.get(id(original))
        if candidate is None:
            continue

        transform: str | None = None
        parameter: int | None = None
        if needs_llama_qk_permute(getattr(config, "model_type", None)) and (
            ".q_proj." in name or ".k_proj." in name
        ):
            transform = "llama_qk_permute"
            parameter = (
                config.num_attention_heads
                if ".q_proj." in name
                else config.num_key_value_heads
            )
        elif "A_log" in name:
            transform = "log_neg"
        elif "norm" in name and original.shape == transformed.shape:
            transform = "subtract_one"
        elif tuple(reversed(original.shape)) == tuple(transformed.shape):
            transform = "transpose"
        elif original.numel() == transformed.numel():
            transform = "reshape"

        if transform is not None:
            candidates[id(transformed)] = dataclasses.replace(
                candidate,
                transform=transform,
                transform_parameter=parameter,
            )


def _is_quantized_weight(key: str, state_dict: dict) -> bool:
    """Check if a .weight key has a matching .scales companion."""
    if not key.endswith(".weight"):
        return False
    stem = key[: -len(".weight")]
    return f"{stem}.scales" in state_dict


def _is_native_block_weight(key: str, state_dict: dict) -> bool:
    """Check for a packed runtime-native GGUF weight."""
    from mobius.integrations.gguf._repacker import NATIVE_BLOCK_BYTE_SIZES

    if not key.endswith(".weight"):
        return False
    value = state_dict[key]
    return (
        value.dtype.is_floating_point is False
        and value.dim() == 3
        and value.shape[-1] in NATIVE_BLOCK_BYTE_SIZES
    )


def _native_block_spec(qtype):
    """Return the runtime-native layout for a GGUF quantization enum."""
    from mobius.integrations.gguf._repacker import native_block_spec

    qtype_val = qtype.value if hasattr(qtype, "value") else qtype
    return native_block_spec(qtype_val)


def _native_block_format(qtype) -> str | None:
    """Return the runtime format string for supported native GGUF blocks."""
    spec = _native_block_spec(qtype)
    return spec.format if spec is not None else None


def _native_block_target_stems(
    hf_name: str,
    np_shape: tuple[int, ...],
    available_stems: set[str],
) -> list[str]:
    """Map a GGUF weight to one or more native-block module stems."""
    if not hf_name.endswith(".weight"):
        return []
    stem = hf_name[: -len(".weight")]
    if stem in available_stems:
        return [stem]

    if len(np_shape) == 3 and ".experts." in stem:
        prefix, projection = stem.rsplit(".experts.", 1)
        for container in (f"{prefix}.experts", f"{prefix}.moe.experts"):
            candidates = [f"{container}.{i}.{projection}" for i in range(np_shape[0])]
            if all(candidate in available_stems for candidate in candidates):
                return candidates
    return []


def _fused_projection_target_stems(hf_name: str, available_stems: set[str]) -> list[str]:
    """Return separate quantized targets for a fused GGUF projection."""
    if hf_name.endswith(".qkv_proj.weight"):
        stem = hf_name[: -len(".qkv_proj.weight")]
        candidates = [f"{stem}.{projection}_proj" for projection in ("q", "k", "v")]
    elif hf_name.endswith(".attn.Wqkv.weight"):
        stem = hf_name[: -len(".Wqkv.weight")]
        candidates = [f"{stem}.{projection}_proj" for projection in ("q", "k", "v")]
    elif hf_name.endswith(".attention.self.qkv.weight"):
        stem = hf_name[: -len(".qkv.weight")]
        candidates = [f"{stem}.{projection}" for projection in ("query", "key", "value")]
    elif hf_name.endswith(".mlp.Wi.weight"):
        stem = hf_name[: -len(".Wi.weight")]
        candidates = [f"{stem}.{projection}_proj" for projection in ("gate", "up")]
    else:
        return []
    return candidates if all(candidate in available_stems for candidate in candidates) else []


def _replace_child_module(root, path: str, replacement) -> None:
    """Replace a named ONNXScript child module while retaining its graph name."""
    parts = path.split(".")
    parent = root
    for part in parts[:-1]:
        try:
            parent = getattr(parent, part)
        except AttributeError as error:
            raise AttributeError(f"Module path {path!r} has no child {part!r}") from error
    child_name = parts[-1]
    try:
        old = getattr(parent, child_name)
    except AttributeError as error:
        raise AttributeError(f"Module path {path!r} has no child {child_name!r}") from error
    if hasattr(replacement, "_set_name") and hasattr(old, "name"):
        replacement._set_name(old.name)
    setattr(parent, child_name, replacement)


def _replace_native_block_linears(
    module,
    gguf_model,
    gguf_arch: str,
    *,
    name_mapper: Callable[[str, str], str | None] | None = None,
) -> None:
    """Swap MatMulNBits scaffolding for runtime-supported native linears."""
    from mobius.components import BlockQuantizedLinear, QuantizedLinear
    from mobius.integrations.gguf._tensor_mapping import map_gguf_to_hf_names

    if name_mapper is None:
        name_mapper = map_gguf_to_hf_names

    module_map = dict(module.named_modules())
    quantized_stems = {
        name for name, child in module_map.items() if isinstance(child, QuantizedLinear)
    }
    replacements: dict[str, str] = {}
    for gguf_name, _raw, qtype, np_shape in gguf_model.tensor_items_raw():
        format_name = _native_block_format(qtype)
        if format_name is None:
            continue
        hf_name = name_mapper(gguf_name, gguf_arch)
        if hf_name is None:
            continue
        if gguf_arch in {"kimi-linear", "kimi-k3"} and hf_name.endswith(
            (".k_b_proj.weight", ".v_b_proj.weight")
        ):
            continue
        for stem in _native_block_target_stems(hf_name, np_shape, quantized_stems):
            replacements[stem] = format_name

    for stem, format_name in replacements.items():
        old = module_map[stem]
        replacement = BlockQuantizedLinear(
            old._k,
            old._n,
            format=format_name,
            bias=old.bias is not None,
        )
        _replace_child_module(module, stem, replacement)

    if replacements:
        logger.info(
            "Preserving %d GGUF projection weights as runtime-native IQ/MXFP4 blocks",
            len(replacements),
        )


def _float_linear_dequantization_types(
    module,
    gguf_arch: str,
) -> Mapping[str, Collection[str]] | None:
    """Return explicitly float projection types for mixed quantized imports."""
    if gguf_arch != "jamba":
        return None

    from mobius.integrations.gguf._quant_registry import iter_quant_specs

    quantized_types = frozenset(
        spec.name
        for spec in iter_quant_specs()
        if spec.is_quantized_storage and spec.dequantize is Support.SUPPORTED
    )
    mamba_projection_suffixes = (
        ".mamba.in_proj",
        ".mamba.out_proj",
        ".mamba.ssm.x_proj",
        ".mamba.ssm.dt_proj",
    )
    return {
        name: quantized_types
        for name, _child in module.named_modules()
        if name.endswith(mamba_projection_suffixes)
    }


#: GGUF architectures whose transformer RMSNorms are zero-centered
#: (``output = norm(x) * (1 + weight)``, mobius :class:`OffsetRMSNorm`). Their
#: llama.cpp converter bakes the ``+1`` into every ``*norm.weight`` *except* the
#: Gated-DeltaNet internal ``linear_attn.norm`` (a plain gated RMSNorm), so the
#: GGUF path must undo it — see :func:`_normalize_gguf_weights`.
_OFFSET_NORM_GGUF_ARCHS: frozenset[str] = arch_names_with(lambda spec: spec.offset_norm)

#: GGUF architectures whose llama.cpp converter reorders Gated-DeltaNet V-heads
#: from HuggingFace *grouped* order (``head = group * v_per_k + j``) into ggml
#: *tiled* order (``head = j * num_k_heads + group``) whenever the linear layer
#: is grouped (``num_value_heads != num_key_heads``). mobius's ``GatedDeltaNet``
#: forward consumes the HF grouped order, so the GGUF path must undo the tiling —
#: see :func:`_reorder_deltanet_v_heads`.
_V_HEAD_REORDER_GGUF_ARCHS: frozenset[str] = arch_names_with(lambda spec: spec.v_head_reorder)


def _normalize_gguf_weights(
    state_dict: dict,
    gguf_arch: str | None = None,
    config=None,
) -> dict:
    """Normalize GGUF-specific weight shapes to match HF conventions.

    GGUF tensor mapping + dequantization produces weights that differ
    from HuggingFace in several ways. This function converts them so
    that ``preprocess_weights`` only needs to handle HF→ONNX mapping.

    Transforms applied:

    - **Stacked expert weights**: GGUF provides separate 3D tensors
      ``experts.{gate,up,down}_proj.weight`` with shape
      ``[num_experts, out, in]``. These are unpacked into per-expert
      ``experts.{i}.{proj}.weight`` tensors, matching the HF
      ``experts.down_proj`` format that ``preprocess_weights`` expects.
    - **1D shared_expert_gate**: GGUF stores as ``[hidden]``; HF/ONNX
      ``Linear(hidden, 1)`` expects ``[1, hidden]``.
    - **2D conv1d**: GGUF stores as ``[channels, kernel]``; depthwise
      ``Conv1d`` expects ``[channels, 1, kernel]``.
    - **dt_bias suffix**: GGUF ``ssm_dt.bias`` maps to
      ``dt_bias.bias`` after suffix splitting, but the model parameter
      is just ``dt_bias`` (an ``nn.Parameter``, not a module bias).
    - **DeltaNet A_log**: GGUF stores the SSM decay pre-transformed as
      ``ssm_a = -exp(A_log)``; mobius's ``GatedDeltaNet`` re-derives
      ``-exp(A_log)`` at runtime, so the raw log is recovered via
      ``A_log = log(-ssm_a)`` (scoped to ``linear_attn.A_log``).
    - **Zero-centered RMSNorm** (``gguf_arch`` in
      :data:`_OFFSET_NORM_GGUF_ARCHS`): the converter bakes ``+1`` into every
      ``*norm.weight`` except ``linear_attn.norm.weight``; mobius applies the
      ``1 +`` at runtime via :class:`OffsetRMSNorm`, so subtract ``1`` back out
      to avoid double-counting.
    - **Gated-DeltaNet V-head tiling** (``gguf_arch`` in
      :data:`_V_HEAD_REORDER_GGUF_ARCHS`, grouped linear attention): the
      converter reorders every V-indexed ``linear_attn`` tensor from HF grouped
      order into ggml tiled order; mobius consumes grouped order, so the tiling
      is undone via :func:`_reorder_deltanet_v_heads`.

    Args:
        state_dict: Dequantized GGUF weights keyed by HF tensor names.
        gguf_arch: The source GGUF architecture string (e.g. ``"qwen35"``),
            used to gate architecture-specific value transforms such as the
            zero-centered RMSNorm offset.
        config: The resolved :class:`ArchitectureConfig`; supplies the
            Gated-DeltaNet head counts / dims used to undo the V-head tiling.
    """
    import torch

    offset_norms = gguf_arch in _OFFSET_NORM_GGUF_ARCHS

    result: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if config is not None:
            _validate_moe_weight_shape(key, tuple(value.shape), config, gguf_arch=gguf_arch)
        if (
            gguf_arch == "jina-bert-v3"
            and key == "token_type_embeddings.weight"
            and value.dim() == 1
        ):
            # GGUF elides the unit token-type dimension; ONNX Gather needs [1, hidden].
            result[key] = value.unsqueeze(0)
            continue
        if gguf_arch in {"dream", "llada-moe", "rnd1"} and ".self_attn.qkv_proj." in key:
            suffix = key.rsplit(".", 1)[-1]
            q_width = int(config.num_attention_heads) * int(config.head_dim)
            kv_width = int(config.num_key_value_heads) * int(config.head_dim)
            expected_width = q_width + 2 * kv_width
            if value.shape[0] != expected_width:
                raise ValueError(
                    f"Invalid fused {gguf_arch} QKV {suffix} width: expected "
                    f"{expected_width}, got {value.shape[0]}"
                )
            query, key_projection, value_projection = value.split(
                [q_width, kv_width, kv_width], dim=0
            )
            stem = key.rsplit(".qkv_proj.", 1)[0]
            result[f"{stem}.q_proj.{suffix}"] = query
            result[f"{stem}.k_proj.{suffix}"] = key_projection
            result[f"{stem}.v_proj.{suffix}"] = value_projection
            continue
        if gguf_arch == "bert" and ".attention.self.qkv." in key:
            suffix = key.rsplit(".", 1)[-1]
            expected_width = 3 * int(config.hidden_size)
            if value.shape[0] != expected_width:
                raise ValueError(
                    f"Invalid fused BERT QKV {suffix} width: expected {expected_width}, "
                    f"got {value.shape[0]}"
                )
            query, key_value, value_projection = value.chunk(3, dim=0)
            stem = key.rsplit(".qkv.", 1)[0]
            result[f"{stem}.query.{suffix}"] = query
            result[f"{stem}.key.{suffix}"] = key_value
            result[f"{stem}.value.{suffix}"] = value_projection
            continue
        # Fused stacked gate/up experts [num_experts, 2*out, ...] are split
        # before ordinary stacked-expert unpacking. This handles float weights
        # and packed MatMulNBits companions without dropping either half.
        fused_marker = next(
            (
                marker
                for marker in (
                    ".mlp.experts.gate_up_proj.",
                    ".feed_forward.experts.gate_up_proj.",
                )
                if marker in key
            ),
            None,
        )
        if fused_marker is not None and value.dim() >= 3:
            prefix, suffix = key.rsplit(fused_marker, 1)
            container = fused_marker.removesuffix(".gate_up_proj.")
            if value.shape[1] % 2:
                raise ValueError(
                    f"Fused expert tensor {key!r} has odd gate/up width {value.shape[1]}"
                )
            gate, up = value.chunk(2, dim=1)
            for i in range(value.shape[0]):
                result[f"{prefix}{container}.{i}.gate_proj.{suffix}"] = gate[i]
                result[f"{prefix}{container}.{i}.up_proj.{suffix}"] = up[i]
            continue

        # Stacked expert weights [num_experts, out, ...] → per-expert.
        unpacked = False
        expert_containers = [
            ".mlp.experts",
            ".feed_forward.experts",
            ".block_sparse_moe.moe.experts",
        ]
        if gguf_arch == "arctic":
            expert_containers.append(".moe.experts")
        for proj in ("gate_proj", "up_proj", "down_proj"):
            for container in expert_containers:
                marker = f"{container}.{proj}."
                if marker in key and value.dim() >= 3:
                    prefix, suffix = key.rsplit(marker, 1)
                    for i in range(value.shape[0]):
                        result[f"{prefix}{container}.{i}.{proj}.{suffix}"] = value[i]
                    unpacked = True
                    break
            if unpacked:
                break
        if unpacked:
            continue

        # 1D shared_expert_gate → [1, hidden]
        if key.endswith(".mlp.shared_expert_gate.weight") and value.dim() == 1:
            result[key] = value.unsqueeze(0)
            continue

        # 2D conv1d → [channels, 1, kernel]
        if key.endswith(".conv1d.weight") and value.dim() == 2:
            result[key] = value.unsqueeze(1)
            continue
        if key.endswith(".conv.conv.weight") and value.dim() == 2:
            result[key] = value.unsqueeze(1)
            continue

        # dt_bias.bias → dt_bias (nn.Parameter, not module bias)
        if key.endswith(".dt_bias.bias"):
            result[key[: -len(".bias")]] = value
            continue

        # DeltaNet A_log: undo the converter's pre-transform. GGUF's converter
        # stores the SSM decay already transformed as ``ssm_a = -exp(A_log)``
        # (llama.cpp applies ``-torch.exp`` to every ``.A_log`` tensor and the
        # reference then uses it *directly* as the decay coefficient ``a`` in
        # ``a * softplus(dt)``). mobius's ``GatedDeltaNet`` parameter is the raw
        # ``A_log`` and recomputes ``g = -exp(A_log) * softplus(...)`` at
        # runtime, so feeding it the already-negated-exp value squashes every
        # head's decay to ``-exp(-exp(A_log)) ≈ -1`` and the linear-attention
        # recurrence emits garbage. Invert to recover the raw log parameter,
        # ``A_log = log(-ssm_a)``, so mobius's ``-exp(A_log)`` reproduces the
        # original ``ssm_a`` exactly. Scoped to the GatedDeltaNet ``linear_attn``
        # projection so Mamba/PLaMo SSM modules (which consume ``A = -exp(A_log)``
        # directly) are left untouched.
        if key.endswith(".linear_attn.A_log"):
            result[key] = torch.log(-value)
            continue

        # Zero-centered RMSNorm: undo the converter's baked-in ``+1`` so
        # mobius's OffsetRMSNorm (which adds it back at runtime) does not
        # double-count. The DeltaNet internal ``linear_attn.norm`` is a plain
        # gated RMSNorm (no offset) and is excluded — mirroring exactly which
        # tensors the llama.cpp converter transforms.
        if (
            offset_norms
            and key.endswith("norm.weight")
            and not key.endswith(".linear_attn.norm.weight")
        ):
            result[key] = value - 1.0
            continue

        # layer_scalar.weight → layer_scalar (Gemma4 per-layer output scale is an
        # nn.Parameter, not a module weight). GGUF stores it as
        # blk.{i}.layer_output_scale.weight, which the tensor mapping renames to
        # model.layers.{i}.layer_scalar.weight; strip the artefact .weight suffix.
        if key.endswith(".layer_scalar.weight"):
            result[key[: -len(".weight")]] = value
            continue

        result[key] = value

    # DeltaNet V-head tiling: undo the converter's grouped→tiled permutation of
    # every V-indexed linear_attn tensor so mobius's GatedDeltaNet (which expects
    # HF grouped order) reads consistent heads. Runs last so it operates on the
    # already-normalized keys/shapes (renamed dt_bias, unsqueezed conv1d, ...).
    if gguf_arch in _V_HEAD_REORDER_GGUF_ARCHS:
        result = _reorder_deltanet_v_heads(result, config)

    return result


def _reorder_deltanet_v_heads(state_dict: dict, config) -> dict:
    """Undo the GGUF converter's grouped→tiled V-head permutation.

    llama.cpp's ``_LinearAttentionVReorderBase`` reorders every V-indexed
    Gated-DeltaNet tensor from HuggingFace *grouped* order (V-head
    ``s = group * v_per_k + j``) into ggml *tiled* order
    (``t = j * num_k_heads + group``) whenever the linear layer is grouped
    (``num_value_heads != num_key_heads``). mobius's ``GatedDeltaNet`` forward
    reshapes the value stream as ``num_value_heads`` contiguous ``head_v_dim``
    blocks in the original grouped order, so the GGUF tensors must be permuted
    back: grouped position ``s`` is fetched from tiled slot
    ``perm[s] = (s % v_per_k) * num_key_heads + (s // v_per_k)``.

    The permutation is applied (all derived from ``config`` — no hardcoded head
    counts) to:

    - ``in_proj_qkv`` output rows — V rows only (after ``2 * key_dim``);
    - ``in_proj_z`` output rows — all rows;
    - ``in_proj_a`` / ``in_proj_b`` output rows — one row per V-head;
    - ``A_log`` / ``dt_bias`` — one element per V-head;
    - ``conv1d`` channels — V channels only (after ``2 * key_dim``);
    - ``out_proj`` input columns — all columns (a block-granular permutation of
      the quantized ``K`` axis).

    Quantized projections are stored as MatMulNBits triplets
    (``weight`` ``[N, K/block, block/2]``, ``scales`` ``[N, K/block]``,
    ``zero_points`` ``[N, K/block/2]``). Output-row permutations reindex axis 0
    of all three; the ``out_proj`` input permutation reindexes the block axis
    (axis 1), valid because ``head_v_dim`` is a whole number of quant blocks.
    """
    import torch

    num_k_heads = getattr(config, "linear_num_key_heads", None)
    num_v_heads = getattr(config, "linear_num_value_heads", None)
    head_k_dim = getattr(config, "linear_key_head_dim", None)
    head_v_dim = getattr(config, "linear_value_head_dim", None)
    # Nothing to do unless this is a grouped linear-attention model.
    if not (num_k_heads and num_v_heads and head_k_dim and head_v_dim):
        return state_dict
    if num_v_heads == num_k_heads or num_v_heads % num_k_heads != 0:
        return state_dict

    v_per_k = num_v_heads // num_k_heads
    key_dim = head_k_dim * num_k_heads
    v_offset = 2 * key_dim  # in_proj_qkv / conv1d layout is [Q | K | V]

    # perm[s] = tiled slot holding grouped V-head s.
    head_perm = torch.tensor(
        [(s % v_per_k) * num_k_heads + (s // v_per_k) for s in range(num_v_heads)],
        dtype=torch.long,
    )

    def _expand(perm: torch.Tensor, stride: int) -> torch.Tensor:
        # Expand a per-head permutation into a per-row/-channel index.
        base = (perm * stride).unsqueeze(1) + torch.arange(stride)
        return base.reshape(-1)

    v_rows = _expand(head_perm, head_v_dim)  # length value_dim

    def _index_dim0(t: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        return t.index_select(0, idx)

    def _apply_rows(stem: str, idx: torch.Tensor) -> None:
        # Permute axis 0 of a float weight or a quantized triplet in place.
        for suffix in (".weight", ".scales", ".zero_points"):
            key = stem + suffix
            if key in state_dict:
                state_dict[key] = _index_dim0(state_dict[key], idx)

    def _apply_bare(key: str, idx: torch.Tensor) -> None:
        if key in state_dict:
            state_dict[key] = _index_dim0(state_dict[key], idx)

    layer_stems = {k.rsplit(".", 1)[0] for k in state_dict if ".linear_attn." in k}
    for stem in layer_stems:
        name = stem.rsplit(".", 1)[-1]
        if name == "in_proj_z":
            _apply_rows(stem, v_rows)
        elif name in ("in_proj_a", "in_proj_b"):
            _apply_rows(stem, head_perm)
        elif name == "in_proj_qkv":
            n_rows = state_dict[stem + ".weight"].shape[0]
            full = torch.cat([torch.arange(v_offset), v_offset + v_rows])
            assert full.numel() == n_rows, (n_rows, full.numel())
            _apply_rows(stem, full)
        elif name == "out_proj":
            _reorder_out_proj_cols(state_dict, stem, head_perm, head_v_dim)

    # Bare (non-".weight") linear_attn parameters.
    for k in list(state_dict):
        if k.endswith((".linear_attn.A_log", ".linear_attn.dt_bias")):
            _apply_bare(k, head_perm)
        elif k.endswith(".linear_attn.conv1d.weight"):
            conv = state_dict[k]
            n_ch = conv.shape[0]
            full = torch.cat([torch.arange(v_offset), v_offset + v_rows])
            assert full.numel() == n_ch, (n_ch, full.numel())
            state_dict[k] = _index_dim0(conv, full)

    return state_dict


def _reorder_out_proj_cols(state_dict: dict, stem: str, head_perm, head_v_dim: int) -> None:
    """Permute the quantized ``out_proj`` input (K) axis by V-head.

    ``out_proj`` maps ``value_dim -> hidden``; its input columns are the V
    stream, so they carry the same head tiling. In MatMulNBits form the K axis is
    the block axis (axis 1) of ``weight``/``scales`` and the packed block axis of
    ``zero_points`` (two 4-bit blocks per byte). ``head_v_dim`` spans a whole
    number of blocks, so the permutation is block-granular and lossless.
    """
    import torch

    weight = state_dict.get(stem + ".weight")
    if weight is None or weight.dim() < 2:
        return
    n_blocks = weight.shape[1]
    if n_blocks % head_perm.numel() != 0:
        raise ValueError(
            f"{stem}: cannot map {n_blocks} quant blocks onto "
            f"{head_perm.numel()} V-heads for column reorder"
        )
    blocks_per_head = n_blocks // head_perm.numel()

    def _expand(perm, stride):
        base = (perm * stride).unsqueeze(1) + torch.arange(stride)
        return base.reshape(-1)

    blk_idx = _expand(head_perm, blocks_per_head)
    state_dict[stem + ".weight"] = weight.index_select(1, blk_idx)

    scales = state_dict.get(stem + ".scales")
    if scales is not None:
        state_dict[stem + ".scales"] = scales.index_select(1, blk_idx)

    zp = state_dict.get(stem + ".zero_points")
    if zp is not None and zp.dim() >= 2:
        # zero_points pack two 4-bit blocks per byte along the block axis.
        if blocks_per_head % 2 != 0:
            raise ValueError(
                f"{stem}: {blocks_per_head} blocks/head is not byte-aligned for "
                "packed zero_points reorder"
            )
        zp_bytes_per_head = blocks_per_head // 2
        zp_idx = _expand(head_perm, zp_bytes_per_head)
        state_dict[stem + ".zero_points"] = zp.index_select(1, zp_idx)


def _has_quantized_weights(gguf_model, gguf_arch: str) -> bool:
    """Return whether a GGUF has stored weights with a quantized tensor type."""
    from mobius.integrations.gguf._quant_registry import float_storage_type_ids

    float_type_ids = float_storage_type_ids()

    for name, _raw, qtype, _shape in gguf_model.tensor_items_raw():
        type_id = getattr(qtype, "value", qtype)
        if name.endswith(".weight") and type_id not in float_type_ids:
            return True
    return False


def _preflight_quantization_report(
    gguf_model,
    gguf_arch: str,
    module,
    config,
    *,
    preserve_quantization: bool,
    target_bits: int | None,
    target_block_size: int | None,
    execution_provider: str,
    name_mapper: Callable[[str, str], str | None] | None = None,
    target_name_mapper: Callable[[str], str] | None = None,
    dequantize_float_linear_types: Mapping[str, Collection[str]] | None = None,
    emit_warning: bool = True,
    include_tensor: Callable[[str], bool] | None = None,
):
    """Classify every mapped tensor from header metadata before payload conversion."""
    from mobius.components import (
        BlockQuantizedLinear,
        Embedding,
        Linear,
        QuantizedEmbedding,
        QuantizedLinear,
    )
    from mobius.integrations.gguf._quant_registry import (
        float_storage_type_ids,
        get_quant_spec,
        quant_import_decision,
    )
    from mobius.integrations.gguf._quantization_report import (
        GGUFQuantizationReport,
        QuantizationDisposition,
        QuantizationTensorRecord,
        disposition_for_import_route,
    )
    from mobius.integrations.gguf._spec import (
        QuantImportRoute,
        RepackExactness,
        Support,
        TensorRole,
    )
    from mobius.integrations.gguf._tencent_q1_0 import (
        is_tencent_q1_0_layout,
        tencent_q1_0_source_nbytes,
        tencent_q1_0_target_bits,
    )
    from mobius.integrations.gguf._tensor_mapping import map_gguf_to_hf_names

    if name_mapper is None:
        name_mapper = map_gguf_to_hf_names

    quantized_stems: set[str] = set()
    native_stems: set[str] = set()
    quantized_embedding_stems: set[str] = set()
    float_stems: set[str] = set()
    embedding_stems: set[str] = set()
    parameter_names = {name for name, _ in module.named_parameters()}
    for module_name, child in module.named_modules():
        if isinstance(child, QuantizedLinear) or getattr(
            child, "_gguf_quantized_linear", False
        ):
            quantized_stems.add(module_name)
        elif isinstance(child, BlockQuantizedLinear):
            native_stems.add(module_name)
        elif isinstance(child, QuantizedEmbedding):
            quantized_embedding_stems.add(module_name)
            embedding_stems.add(module_name)
        elif isinstance(child, Linear):
            float_stems.add(module_name)
        elif isinstance(child, Embedding):
            embedding_stems.add(module_name)

    float_type_ids = float_storage_type_ids()
    tencent_q1_0 = is_tencent_q1_0_layout(gguf_model)
    source_qtypes: list[tuple[str, int]] = []
    records: list[QuantizationTensorRecord] = []
    rejected: list[QuantizationTensorRecord] = []
    for tensor in gguf_model.reader_tensors():
        qtype = tensor.tensor_type
        qtype_id = getattr(qtype, "value", qtype)
        quant_spec = get_quant_spec(qtype)
        qtype_name = (
            quant_spec.name if quant_spec is not None else str(getattr(qtype, "name", qtype))
        )
        is_tencent_q1_0_tensor = tencent_q1_0 and qtype_name == "Q1_0"
        source_bytes = (
            tencent_q1_0_source_nbytes(tensor)
            if is_tencent_q1_0_tensor
            else int(tensor.n_bytes)
        )
        source_qtypes.append((qtype_name, source_bytes))
        if include_tensor is not None and not include_tensor(tensor.name):
            continue
        hf_name = name_mapper(tensor.name, gguf_arch)
        if hf_name is None:
            continue
        shape = tuple(int(dim) for dim in reversed(tensor.shape))
        module_hf_name = target_name_mapper(hf_name) if target_name_mapper else hf_name
        if gguf_arch == "bert" and module_hf_name.startswith("bert."):
            module_hf_name = module_hf_name[len("bert.") :]
        elif gguf_arch == "modern-bert" and module_hf_name.startswith("model."):
            module_hf_name = module_hf_name[len("model.") :]
        elif gguf_arch in {"t5", "t5encoder"}:
            from mobius.models.t5 import _rename_t5_weight

            renamed = _rename_t5_weight(
                module_hf_name,
                is_gated_act=bool(getattr(config, "is_gated_act", False)),
            )
            if renamed is not None:
                module_hf_name = renamed

        module_stem = module_hf_name.removesuffix(".weight")
        native_targets = _native_block_target_stems(hf_name, shape, native_stems)
        affine_targets = _native_block_target_stems(module_hf_name, shape, quantized_stems)
        fused_targets = _fused_projection_target_stems(module_hf_name, quantized_stems)
        is_kimi_fused_kv_projection = gguf_arch == "kimi-k3" and module_hf_name.endswith(
            ".kv_b_proj.weight"
        )
        is_quantized_embedding = module_stem in quantized_embedding_stems
        target_is_quantized = bool(
            native_targets
            or affine_targets
            or fused_targets
            or module_stem in quantized_stems
            or is_quantized_embedding
            or is_kimi_fused_kv_projection
        )
        if qtype_id in float_type_ids:
            disposition = (
                QuantizationDisposition.REJECTED
                if preserve_quantization and target_is_quantized
                else QuantizationDisposition.SOURCE_FLOAT
            )
            target = "rejected" if disposition is QuantizationDisposition.REJECTED else "float"
            reason = (
                "The selected quantized graph would quantize a source-float tensor."
                if disposition is QuantizationDisposition.REJECTED
                else "The GGUF tensor is already stored as float."
            )
        elif quant_spec is None:
            disposition = QuantizationDisposition.REJECTED
            target = "rejected"
            reason = "The qtype is outside the pinned llama.cpp census."
        elif not preserve_quantization:
            disposition = QuantizationDisposition.DEQUANTIZED_FLOAT
            target = "float"
            reason = "Explicit float import dequantizes every mapped quantized tensor."
        else:
            explicitly_dequantized = (
                dequantize_float_linear_types is not None
                and module_stem in dequantize_float_linear_types
                and quant_spec.name in dequantize_float_linear_types[module_stem]
            )
            known_float_route = explicitly_dequantized or _uses_explicit_float_route(
                gguf_arch, tensor.name
            )
            if known_float_route:
                role = TensorRole.NON_MATMUL
            elif is_quantized_embedding:
                role = TensorRole.EMBEDDING
            elif native_targets:
                role = TensorRole.PROJECTION
            elif target_is_quantized:
                role = TensorRole.AFFINE_PROJECTION
            elif (
                module_stem in float_stems
                or module_stem in embedding_stems
                or module_hf_name in parameter_names
            ):
                role = TensorRole.NON_MATMUL
            else:
                role = TensorRole.NON_MATMUL
                route = QuantImportRoute.REJECTED
                exactness = None
                reason = "The mapped tensor has no corresponding graph parameter target."
            if (
                module_hf_name in parameter_names
                or module_stem in float_stems
                or module_stem in embedding_stems
                or target_is_quantized
                or known_float_route
            ):
                if is_tencent_q1_0_tensor and role is not TensorRole.NON_MATMUL:
                    selected_bits = tencent_q1_0_target_bits()
                    if (target_bits, target_block_size) == (selected_bits, 128):
                        route = QuantImportRoute.AFFINE_REPACK
                        exactness = RepackExactness.EXACT
                        reason = (
                            "Tencent Q1_0 2-bit/512 blocks are exactly represented as "
                            f"INT{selected_bits} affine block-128."
                        )
                    else:
                        route = QuantImportRoute.REJECTED
                        exactness = None
                        reason = (
                            "Tencent Q1_0 requires the selected exact "
                            f"INT{selected_bits} affine block-128 target."
                        )
                else:
                    route, exactness, reason = quant_import_decision(
                        qtype,
                        role,
                        target_bits=target_bits,
                        target_block_size=target_block_size,
                    )
            is_kimi_reshaped_projection = gguf_arch in {
                "kimi-linear",
                "kimi-k3",
            } and module_hf_name.endswith((".k_b_proj.weight", ".v_b_proj.weight"))
            if is_kimi_reshaped_projection and route is not QuantImportRoute.REJECTED:
                if quant_spec.dequantize is not Support.SUPPORTED:
                    route = QuantImportRoute.REJECTED
                    exactness = None
                    reason = "The reshaped projection has no trusted dequantizer."
                else:
                    route = QuantImportRoute.DEQUANTIZE_REQUANTIZE
                    exactness = RepackExactness.LOSSY
                    reason = (
                        "The Kimi MLA layout transform changes affine block groups and "
                        "requires lossy dequantization/requantization."
                    )
            disposition = disposition_for_import_route(route, exactness)
            target = (
                "native GGUF block storage"
                if route is QuantImportRoute.NATIVE_BYTES
                else "float"
                if route is QuantImportRoute.DEQUANTIZE_FLOAT
                else "rejected"
                if route is QuantImportRoute.REJECTED
                else f"INT{target_bits} affine block-{target_block_size}"
            )
        record = QuantizationTensorRecord(
            name=tensor.name,
            qtype=qtype_name,
            source_bytes=source_bytes,
            disposition=disposition,
            target_storage=target,
            reason=reason,
        )
        records.append(record)
        if disposition is QuantizationDisposition.REJECTED:
            rejected.append(record)

    quantized_records = [
        record
        for record in records
        if record.disposition
        in {
            QuantizationDisposition.NATIVE_BYTES,
            QuantizationDisposition.LOSSLESS_REPACK,
            QuantizationDisposition.LOSSY_REQUANTIZE,
        }
    ]
    if quantized_records:
        targets = {record.target_storage for record in quantized_records}
        target_storage_format = " + ".join(sorted(targets))
    else:
        target_storage_format = "float"
    compute_mode = (
        "float operators"
        if not quantized_records
        else "runtime-dependent native custom op or inline standard-ONNX fallback"
    )
    compute_capability = (
        "Storage is float and executes through float operators."
        if not quantized_records
        else (
            "Packed storage may be consumed by native MatMulNBits/"
            "GatherBlockQuantized/BlockQuantizedMatMul implementations. "
            "MatMulNBits may instead be inlined as nibble BitShift/BitwiseAnd, "
            "DequantizeLinear, and float MatMul; this does not change packed storage "
            f"and makes no promise about the {execution_provider!r} runtime kernel."
        )
    )
    report = GGUFQuantizationReport.create(
        source_qtypes=source_qtypes,
        tensor_records=records,
        target_storage_format=target_storage_format,
        compute_mode=compute_mode,
        compute_capability=compute_capability,
    )
    if rejected:
        details = "; ".join(
            f"{record.name} ({record.qtype}): {record.reason}" for record in rejected[:5]
        )
        suffix = "" if len(rejected) <= 5 else f"; and {len(rejected) - 5} more"
        raise ValueError(
            "GGUF quantization preflight could not determine a safe disposition for "
            f"{len(rejected)} mapped tensor(s): {details}{suffix}"
        )
    warning = report.warning_message()
    if warning is not None and emit_warning:
        logger.warning("%s", warning)
    return report


def _uses_explicit_float_route(
    gguf_arch: str,
    tensor_name: str,
) -> bool:
    """Return whether a quantized source weight is intentionally loaded as float."""
    if gguf_arch in {"bert", "modern-bert"} and tensor_name in {
        "token_embd.weight",
        "token_embd_norm.weight",
        "token_types.weight",
        "position_embd.weight",
    }:
        return True
    if gguf_arch in {"t5", "t5encoder"} and tensor_name in {
        "token_embd.weight",
        "shared.weight",
    }:
        return True
    if tensor_name.endswith(("_norm.weight", ".norm.weight")):
        return True
    return gguf_arch == "jamba" and tensor_name.endswith(
        (
            "ssm_in.weight",
            "ssm_out.weight",
            "ssm_x.weight",
            "ssm_dt.weight",
            "ssm_conv1d.weight",
        )
    )


def _reject_unsupported_quantization_preservation(
    gguf_model,
    gguf_arch: str,
    *,
    preserve_quantization: bool,
    allow_native_blocks: bool = True,
    allow_quantized_embeddings: bool = True,
    allow_quantized_lm_head: bool = True,
    dequantize_float_linear_types: Mapping[str, Collection[str]] | None = None,
) -> None:
    """Reject architectures that cannot preserve all compatible quantized weights.

    The diffusion graph owns separate QuantizedLinear Q/K/V modules, while the
    fused GGUF family maps to a synthetic ``qkv_proj`` stem that is not a graph
    target. The quantized loader therefore cannot attach packed blocks to it.
    Dequantizing the fused tensor and splitting it later is also invalid because
    the graph still expects packed parameters. This applies even when the fused
    tensor itself is float: any other quantized mapped tensor selects the packed
    graph, so the split Q/K/V targets remain quantized.
    """
    if not preserve_quantization:
        return
    if gguf_arch in {"chatglm", "phi2"}:
        reason = {
            "chatglm": (
                "its fused QKV and gate/up tensors must be split into separate packed "
                "graph targets"
            ),
            "phi2": (
                "the Phi-2 attention, MLP, and output graph currently uses float-only "
                "linear modules"
            ),
        }[gguf_arch]
        raise ValueError(
            f"Quantization-preserving {gguf_arch} import is unsupported because {reason}. "
            "Use keep_quantized=False (or --dequantize) for a float import."
        )

    from mobius.integrations.gguf._quant_registry import (
        float_storage_type_ids,
        get_quant_spec,
    )
    from mobius.integrations.gguf._tensor_mapping import map_gguf_to_hf_names

    float_type_ids = float_storage_type_ids()
    metadata = getattr(gguf_model, "metadata", {})
    block_count = int(metadata.get(f"{gguf_arch}.block_count", 0))
    mtp_count = int(metadata.get(f"{gguf_arch}.nextn_predict_layers", 0))
    mtp_blocks = set(range(max(0, block_count - mtp_count), block_count))
    for tensor_name, _raw, qtype, _shape in gguf_model.tensor_items_raw():
        type_id = getattr(qtype, "value", qtype)
        type_name = getattr(qtype, "name", str(qtype))
        if not tensor_name.endswith(".weight") or type_id in float_type_ids:
            continue
        hf_name = map_gguf_to_hf_names(tensor_name, gguf_arch)
        module_stem = (
            hf_name[: -len(".weight")]
            if hf_name is not None and hf_name.endswith(".weight")
            else None
        )
        if gguf_arch == "bert" and module_stem is not None and module_stem.startswith("bert."):
            module_stem = module_stem[len("bert.") :]
        explicitly_dequantized = (
            dequantize_float_linear_types is not None
            and module_stem in dequantize_float_linear_types
            and type_name in dequantize_float_linear_types[module_stem]
        )
        if explicitly_dequantized or _uses_explicit_float_route(gguf_arch, tensor_name):
            continue
        if tensor_name.endswith(".ffn_gate_up_exps.weight"):
            raise ValueError(
                "Quantization-preserving GGUF import cannot split packed fused expert "
                f"tensor {tensor_name} ({type_name}) into separate gate/up graph "
                "targets without changing its stored representation. Use "
                "keep_quantized=False (API) or --dequantize (CLI) for explicit "
                "float import."
            )
        if not allow_quantized_embeddings and tensor_name in {
            "token_embd.weight",
            "shared.weight",
        }:
            raise ValueError(
                "Quantization-preserving GGUF import cannot retain packed embedding "
                f"{tensor_name} ({type_name}) in this graph. Use keep_quantized=False "
                "(API) or --dequantize (CLI) for explicit float import."
            )
        if not allow_quantized_lm_head and tensor_name == "output.weight":
            raise ValueError(
                "Quantization-preserving GGUF import cannot retain packed LM head "
                f"{tensor_name} ({type_name}) in this graph. Use keep_quantized=False "
                "(API) or --dequantize (CLI) for explicit float import."
            )
        spec = get_quant_spec(qtype)
        block_match = re.match(r"blk\.(\d+)\.", tensor_name)
        is_mtp_block = block_match is not None and int(block_match.group(1)) in mtp_blocks
        if (
            spec is not None
            and spec.native_preserve is not None
            and (not allow_native_blocks or ".nextn." in tensor_name or is_mtp_block)
            and spec.dequantize is not Support.SUPPORTED
        ):
            raise ValueError(
                "Quantization-preserving GGUF import cannot normalize native block "
                f"format {type_name} for {tensor_name}: no trusted dequantizer is available."
            )

    if gguf_arch not in {"dream", "llada-moe", "rnd1"}:
        return

    fused = sorted(
        name
        for name in gguf_model.tensor_names
        if re.fullmatch(r"blk\.\d+\.attn_qkv\.weight", name)
    )
    if fused:
        raise ValueError(
            f"Quantization-preserving import of fused QKV is not supported for "
            f"{gguf_arch} GGUF ({fused[0]}). The graph has separate packed Q/K/V "
            "targets, so splitting after weight loading would leave them uninitialized. "
            "Use keep_quantized=False (or --dequantize) for a float import."
        )


def _detect_quant_params(gguf_model, gguf_arch: str) -> tuple[int, int, bool]:
    """Detect the common MatMulNBits target for GGUF projection weights.

    Q4_K-containing mixed presets target 4-bit, block-32 asymmetric
    MatMulNBits. Other files use the most common directly repackable type.

    Returns:
        ``(bits, block_size, is_symmetric)`` tuple.

    Raises:
        ValueError: If no mapped or repackable weight tensors are found.
    """
    from gguf import GGMLQuantizationType

    from mobius.integrations.gguf._quant_registry import (
        float_storage_type_ids,
        get_quant_spec,
    )
    from mobius.integrations.gguf._repacker import can_repack, repack_quant_params
    from mobius.integrations.gguf._spec import QuantImportRoute
    from mobius.integrations.gguf._tencent_q1_0 import is_tencent_q1_0_layout
    from mobius.integrations.gguf._tensor_mapping import (
        map_gguf_to_hf_names,
    )

    counts: Counter = Counter()
    float_type_ids = float_storage_type_ids()
    for name, _raw, qtype, _shape in gguf_model.tensor_items_raw():
        hf_name = map_gguf_to_hf_names(name, gguf_arch)
        if hf_name is None or not hf_name.endswith(".weight"):
            continue
        if _uses_explicit_float_route(gguf_arch, name):
            continue
        type_id = getattr(qtype, "value", qtype)
        if type_id not in float_type_ids:
            counts[qtype] += 1

    if not counts:
        raise ValueError(
            "No mapped weight tensors found in GGUF file. "
            "Use keep_quantized=False for dequantized import."
        )

    quant_specs = {qtype: get_quant_spec(qtype) for qtype in counts}
    unknown = [qtype for qtype, spec in quant_specs.items() if spec is None]
    if unknown:
        names = ", ".join(sorted(getattr(qtype, "name", str(qtype)) for qtype in unknown))
        raise ValueError(
            f"GGUF contains qtypes outside the pinned llama.cpp census: {names}. "
            "Update the importer policy before loading this file."
        )
    rejected = [
        spec
        for spec in quant_specs.values()
        if spec is not None and spec.import_route is QuantImportRoute.REJECTED
    ]
    if rejected:
        details = "; ".join(
            f"{spec.name}: {spec.reason or 'no supported importer route'}" for spec in rejected
        )
        raise ValueError(
            "GGUF contains stored qtypes that cannot be imported safely: "
            f"{details} Re-quantize those tensors to a supported qtype."
        )

    native_counts = Counter(
        {qtype: count for qtype, count in counts.items() if _native_block_format(qtype)}
    )
    if native_counts:
        requires_normalization = any(
            qtype not in native_counts
            and spec is not None
            and spec.import_route is QuantImportRoute.DEQUANTIZE_REQUANTIZE
            for qtype, spec in quant_specs.items()
        )
        affine_specs = {
            spec.affine_repack
            for qtype in counts
            if qtype not in native_counts
            if (spec := get_quant_spec(qtype)) is not None and spec.affine_repack is not None
        }
        if requires_normalization or len(affine_specs) > 1:
            bits, block_size, can_omit_zero_points = 4, 32, False
        elif affine_specs:
            target = next(iter(affine_specs))
            bits, block_size = target.as_params()
            can_omit_zero_points = target.omit_zero_points
        else:
            bits, block_size, can_omit_zero_points = 4, 32, True
        logger.info(
            "Native GGUF quant types present; using %d-bit/block-%d module "
            "scaffolding for affine tensors",
            bits,
            block_size,
        )
        return bits, block_size, can_omit_zero_points

    affine_specs = {
        spec.affine_repack
        for spec in quant_specs.values()
        if spec is not None and spec.affine_repack is not None
    }
    if len(affine_specs) > 1 or any(
        spec is not None and spec.import_route is QuantImportRoute.DEQUANTIZE_REQUANTIZE
        for spec in quant_specs.values()
    ):
        logger.info(
            "GGUF contains qtypes requiring dequantize/requantize; using the "
            "supported 4-bit/block-32 affine target"
        )
        return 4, 32, False

    # Q4_K_M is deliberately a mixed preset. Depending on tensor dimensions
    # and importance it may contain mostly Q5_0 plus Q4_K, Q6_K, and Q8_0.
    # The presence of Q4_K identifies the desired 4-bit MatMulNBits target;
    # choosing only among already-repackable types would incorrectly select
    # Q8_0 for the official Qwen2.5-0.5B Q4_K_M file.
    if GGMLQuantizationType.Q4_K in counts:
        dominant = GGMLQuantizationType.Q4_K
    else:
        repackable_counts = Counter(
            {
                qtype: count
                for qtype, count in counts.items()
                if can_repack(qtype.value if hasattr(qtype, "value") else qtype)
            }
        )
        if not repackable_counts:
            source_types = ", ".join(
                sorted({getattr(qtype, "name", str(qtype)) for qtype in counts})
            )
            raise ValueError(
                "No supported quantized preservation target for GGUF weight "
                f"types: {source_types}. Use keep_quantized=False (API) or "
                "--dequantize (CLI) for explicit float import."
            )
        dominant = repackable_counts.most_common(1)[0][0]
    dominant_value = dominant.value if hasattr(dominant, "value") else dominant
    params = repack_quant_params(dominant_value)
    assert params is not None
    bits, block_size = params
    # Whether the repacked form may drop zero points is a property of the repack
    # *target*, not of the source format: Q4_K and Q6_K both requantize through
    # the asymmetric affine path, so both need zero points even though Q6_K's
    # source form is symmetric around 32. Q4_0/Q8_0 look symmetric too, but
    # their GGUF dequantization is still ``(q - 8) * scale`` / ``(q - 128) *
    # scale``, and GatherBlockQuantized has diverging CPU/CUDA defaults when the
    # input is omitted, which corrupts embeddings before the first decoder layer.
    dominant_spec = get_quant_spec(dominant)
    assert dominant_spec is not None and dominant_spec.affine_repack is not None
    is_sym = dominant_spec.affine_repack.omit_zero_points

    # Tencent Q1_0 files reuse the Q1_0 type id but ship a different
    # on-disk layout (2-bit SEQ, 512-element blocks, fp16 scale per block).
    # Override the mainline defaults so the resulting QuantizedLinear
    # matches what parse_tencent_q1_0_tensor produces (4-bit packed).
    if dominant == GGMLQuantizationType.Q1_0 and is_tencent_q1_0_layout(gguf_model):
        # See _tencent_q1_0.py — the bits/zp flavour depends on a flag:
        #   default (fast):  bits=4 packed-uint8 zp=3 (inflated codebook)
        #   opt-in (small):  bits=2 float zp=1.5 (native SEQ layout)
        from mobius._flags import flags

        if flags.tencent_q1_0_use_native_2bit:
            bits, block_size, is_sym = 2, 128, False
        else:
            bits, block_size, is_sym = 4, 128, False

    logger.info(
        "Dominant GGUF quant type: %s (%d tensors, bits=%d, block_size=%d)",
        dominant,
        counts[dominant],
        bits,
        block_size,
    )
    return bits, block_size, is_sym


def _can_quantize_embedding(
    gguf_model,
    gguf_arch: str,
    *,
    bits: int,
    block_size: int,
) -> bool:
    """Return whether the token embedding can use GatherBlockQuantized.

    The graph has one quantization configuration shared by its quantized
    modules. Preserve the GGUF embedding only when its repacked representation
    uses the same bit width and block size as the projection weights.
    """
    from mobius.integrations.gguf._quant_registry import quant_import_decision
    from mobius.integrations.gguf._repacker import repack_quant_params
    from mobius.integrations.gguf._spec import QuantImportRoute, TensorRole
    from mobius.integrations.gguf._tencent_q1_0 import is_tencent_q1_0_layout
    from mobius.integrations.gguf._tensor_mapping import map_gguf_to_hf_names

    # Encoder modules currently expose plain Embedding parameters. Keep their
    # token tables explicitly dequantized until they have a QuantizedEmbedding
    # factory and a validated GatherBlockQuantized ABI.
    if gguf_arch in {"bert", "modern-bert", "t5", "t5encoder"}:
        return False

    # Tencent files reuse the Q1_0 type id for a custom layout that gguf-py
    # cannot size correctly. Embeddings from those files must stay dequantized.
    if is_tencent_q1_0_layout(gguf_model):
        return False

    for tensor in gguf_model.reader_tensors():
        mapped_name = map_gguf_to_hf_names(tensor.name, gguf_arch)
        if mapped_name is None or not mapped_name.endswith(
            ("model.embed_tokens.weight", "shared.weight")
        ):
            continue
        shape = tuple(reversed(tensor.shape))
        if len(shape) != 2:
            return False
        qtype = tensor.tensor_type
        qtype_val = qtype.value if hasattr(qtype, "value") else qtype
        if repack_quant_params(qtype_val) == (bits, block_size):
            return True
        route, _, _ = quant_import_decision(
            qtype,
            TensorRole.EMBEDDING,
            target_bits=bits,
            target_block_size=block_size,
        )
        return route is QuantImportRoute.DEQUANTIZE_REQUANTIZE
    return False


def _can_quantize_lm_head(gguf_model, gguf_arch: str) -> bool:
    """Return whether an untied GGUF output head can be kept quantized."""
    from mobius.integrations.gguf._quant_registry import lm_head_preserve_type_names
    from mobius.integrations.gguf._tensor_mapping import map_gguf_to_hf_names

    supported_types = lm_head_preserve_type_names()
    for name, _raw, qtype, shape in gguf_model.tensor_items_raw():
        mapped = map_gguf_to_hf_names(name, gguf_arch)
        if mapped is None or not mapped.endswith("lm_head.weight"):
            continue
        return len(shape) == 2 and getattr(qtype, "name", None) in supported_types
    return False


def _require_supported_requantization(
    *,
    bits: int,
    block_size: int,
    tensor_name: str,
) -> None:
    if bits != 4 or block_size != 32:
        raise ValueError(
            "keep_quantized MatMulNBits requantization currently supports only "
            f"4-bit/block-32 targets; got bits={bits} block={block_size} "
            f"for tensor {tensor_name}. Use keep_quantized=False or a "
            "4-bit/block-32 target."
        )


def repack_gguf_weight_to_target(
    gguf_model,
    raw,
    qtype,
    np_shape,
    *,
    target_bits: int,
    target_block_size: int,
    target_symmetric: bool,
    tensor_name: str,
    tensor_role: TensorRole | None = None,
):
    """Repack one 2-D GGUF weight to the graph's common MatMulNBits target.

    Reuses the shared repacker machinery: a tensor whose native repacked layout
    already matches ``(target_bits, target_block_size)`` is repacked directly;
    otherwise it is dequantized and requantized to the target layout. This is
    the single-tensor building block reused by both the text-only
    (:func:`_load_quantized_state_dict`) and the multimodal quantized loaders.

    Args:
        gguf_model: The source :class:`GGUFModel`.
        raw: Raw tensor bytes (as returned by ``tensor_items_raw``).
        qtype: The GGUF quantization type of the tensor.
        np_shape: The tensor's logical ``(N, K)`` shape.
        target_bits: Target MatMulNBits bit width.
        target_block_size: Target MatMulNBits block size.
        target_symmetric: Whether the requantization path should omit
            zero-points (symmetric). Only used when requantizing.
        tensor_name: Name used for error messages.

    Returns:
        A ``RepackedTensor`` with the target ``(bits, block_size)`` layout.
    """
    import numpy as np

    from mobius.integrations.gguf._quant_registry import quant_import_decision
    from mobius.integrations.gguf._repacker import (
        can_repack,
        repack_dequantized_tensor,
        repack_gguf_tensor,
    )
    from mobius.integrations.gguf._spec import QuantImportRoute

    if tensor_role is None:
        tensor_role = TensorRole.PROJECTION
    route, _exactness, reason = quant_import_decision(
        qtype,
        tensor_role,
        target_bits=target_bits,
        target_block_size=target_block_size,
    )
    if route is QuantImportRoute.REJECTED:
        qtype_name = getattr(qtype, "name", str(qtype))
        raise ValueError(
            f"Cannot import GGUF tensor {tensor_name} ({qtype_name}, "
            f"role={tensor_role.value}): {reason}"
        )

    qtype_val = qtype.value if hasattr(qtype, "value") else qtype
    if route is QuantImportRoute.AFFINE_REPACK and can_repack(qtype_val):
        shape_2d = (int(np_shape[0]), int(np_shape[1]))
        repacked = repack_gguf_tensor(raw.ravel().view(np.uint8), qtype_val, shape_2d)
        if repacked.bits == target_bits and repacked.block_size == target_block_size:
            return repacked

    _require_supported_requantization(
        bits=target_bits,
        block_size=target_block_size,
        tensor_name=tensor_name,
    )
    values = gguf_model.dequantize_raw_tensor(raw, qtype, np_shape)
    return repack_dequantized_tensor(
        values,
        bits=target_bits,
        block_size=target_block_size,
        symmetric=target_symmetric,
    )


def _load_dequantized_state_dict(
    gguf_model,
    gguf_arch: str,
    name_mapper: Callable[[str, str], str | None] | None = None,
    warn_unmapped: bool = True,
    reuse_candidates: dict | None = None,
) -> dict:
    """Load all tensors dequantized to float (Phase 1 path).

    ``name_mapper`` maps a GGUF tensor name to its target state-dict key (or
    ``None`` to skip). Defaults to the main-model mapping; the MTP head builder
    injects a head-scoped mapper so the trailing ``blk.<mtp>.nextn.*`` /
    attention block is routed to the head module instead of being dropped.
    """
    import numpy as np
    import torch

    from mobius.integrations.gguf._tensor_mapping import (
        map_gguf_to_hf_names,
    )

    if name_mapper is None:
        name_mapper = map_gguf_to_hf_names

    if gguf_arch in {"dflash", "eagle3"}:
        # d2t is an integer orchestration table, not an ONNX initializer. The
        # gguf package intentionally has no I64 "dequantizer", so exclude it
        # before asking the reader to materialize neural tensors.
        tensors = (
            (name, gguf_model.dequantize_raw_tensor(raw, qtype, shape))
            for name, raw, qtype, shape in gguf_model.tensor_items_raw()
            if name != "d2t"
        )
    else:
        tensors = gguf_model.tensor_items()

    state_dict = {}
    for gguf_name, np_array in tqdm.tqdm(
        tensors,
        desc="Dequantizing tensors",
        total=gguf_model.num_tensors,
    ):
        hf_name = name_mapper(gguf_name, gguf_arch)
        if hf_name is not None:
            # F32/F16 tensors are mmap'd read-only views; make
            # writable so PyTorch can mutate if needed.
            if not np_array.flags.writeable:
                np_array = np.array(np_array)
            tensor = torch.from_numpy(np_array)
            state_dict[hf_name] = tensor
            if reuse_candidates is not None:
                qtype = gguf_model.get_tensor_type(gguf_name)
                if getattr(qtype, "name", None) in {"F32", "F16"}:
                    from mobius.integrations.gguf._reuse import GGUFReuseCandidate

                    offset, length, qtype_name = gguf_model.tensor_storage_range(gguf_name)
                    reuse_candidates[id(tensor)] = GGUFReuseCandidate(
                        gguf_name,
                        offset,
                        length,
                        qtype_name,
                        tuple(int(dim) for dim in np_array.shape),
                    )
        else:
            if warn_unmapped:
                logger.warning("Unmapped GGUF tensor: %s (skipped)", gguf_name)
    return state_dict


def _load_quantized_state_dict(
    gguf_model,
    gguf_arch: str,
    module,
    config,
    name_mapper: Callable[[str, str], str | None] | None = None,
    warn_unmapped: bool = True,
    reuse_candidates: dict | None = None,
    dequantize_float_linear_types: Mapping[str, Collection[str]] | None = None,
) -> dict:
    """Load tensors, preserving native blocks or normalizing to MatMulNBits.

    Projection weights (Q/K/V/O, MLP, and a quantized output head) are
    converted to the graph's common MatMulNBits format, and compatible token
    embeddings to GatherBlockQuantized format. Mixed or unsupported source
    types are dequantized and requantized when they do not match that target.
    Incompatible embeddings, norms, and other non-linear tensors remain
    dequantized.

    For llama-family models, quantized Q/K weights receive the
    row-level reverse-permutation that ``process_tensors`` would
    normally apply.
    """
    import numpy as np
    import torch
    from gguf import GGMLQuantizationType, dequantize

    from mobius.components import (
        BlockQuantizedLinear,
        Embedding,
        QuantizedEmbedding,
        QuantizedLinear,
    )
    from mobius.integrations.gguf._quant_registry import (
        float_storage_type_ids,
        get_quant_spec,
        quant_import_decision,
    )
    from mobius.integrations.gguf._repacker import (
        can_repack,
        preserve_native_blocks,
        repack_dequantized_tensor,
        repack_gguf_tensor,
    )
    from mobius.integrations.gguf._spec import QuantImportRoute, TensorRole
    from mobius.integrations.gguf._tencent_q1_0 import (
        is_tencent_q1_0_layout,
        parse_tencent_q1_0_tensor,
    )
    from mobius.integrations.gguf._tensor_mapping import (
        map_gguf_to_hf_names,
    )
    from mobius.integrations.gguf._tensor_processors import (
        _reverse_permute,
    )

    if name_mapper is None:
        name_mapper = map_gguf_to_hf_names

    # Collect module paths that use QuantizedLinear so we know
    # which .weight parameters should receive repacked data.
    quantized_stems = set()
    quantized_output_sizes: dict[str, int] = {}
    native_block_stems: dict[str, str] = {}
    quantized_embedding_stems = set()
    embedding_stems = set()
    for mod_name, mod in module.named_modules():
        if isinstance(mod, QuantizedLinear) or getattr(mod, "_gguf_quantized_linear", False):
            quantized_stems.add(mod_name)
            quantized_output_sizes[mod_name] = mod._n
        elif isinstance(mod, BlockQuantizedLinear):
            native_block_stems[mod_name] = mod._format
        elif isinstance(mod, QuantizedEmbedding):
            quantized_embedding_stems.add(mod_name)
            embedding_stems.add(mod_name)
        elif isinstance(mod, Embedding):
            embedding_stems.add(mod_name)

    num_heads = getattr(config, "num_attention_heads", None)
    num_kv_heads = getattr(config, "num_key_value_heads", None)
    model_type = getattr(config, "model_type", None)

    # Detect Tencent's non-mainline Q1_0 layout once per file. Reading
    # such tensors requires a custom parser keyed on the explicit
    # per-tensor file offset (mainline byte sizes are wrong).
    tencent_q1_0 = is_tencent_q1_0_layout(gguf_model)
    if tencent_q1_0:
        logger.info(
            "Detected Tencent Q1_0 layout (block_size=512, 2-bit SEQ); "
            "using custom per-tensor parser"
        )

    state_dict: dict[str, torch.Tensor] = {}
    n_repacked = 0
    n_requantized = 0
    target_bits = config.quantization.bits
    target_block_size = config.quantization.group_size
    target_symmetric = config.quantization.sym
    float_type_ids = float_storage_type_ids()

    for gguf_name, raw, qtype, np_shape in tqdm.tqdm(
        gguf_model.tensor_items_raw(),
        desc="Repacking tensors",
        total=gguf_model.num_tensors,
    ):
        if gguf_arch in {"dflash", "eagle3"} and gguf_name == "d2t":
            continue
        hf_name = name_mapper(gguf_name, gguf_arch)
        if hf_name is None:
            if warn_unmapped:
                logger.warning("Unmapped GGUF tensor: %s (skipped)", gguf_name)
            continue
        _validate_moe_weight_shape(
            hf_name,
            tuple(int(dim) for dim in np_shape),
            config,
            gguf_arch=gguf_arch,
        )
        module_hf_name = hf_name
        if gguf_arch == "bert" and module_hf_name.startswith("bert."):
            module_hf_name = module_hf_name[len("bert.") :]
        elif gguf_arch == "modern-bert" and module_hf_name.startswith("model."):
            module_hf_name = module_hf_name[len("model.") :]
        elif gguf_arch in {"t5", "t5encoder"}:
            from mobius.models.t5 import _rename_t5_weight

            renamed = _rename_t5_weight(
                module_hf_name,
                is_gated_act=bool(getattr(config, "is_gated_act", False)),
            )
            if renamed is not None:
                module_hf_name = renamed

        # Determine the int value of the quant type for can_repack
        qtype_val = qtype.value if hasattr(qtype, "value") else qtype

        if gguf_arch == "kimi-k3" and module_hf_name.endswith(".kv_b_proj.weight"):
            # K3 may serialize MLA K/V-B as one head-major matrix, while the
            # graph has distinct quantized projections. Preserve each source
            # row exactly while splitting and reordering the head-major blocks.
            route, _exactness, _reason = quant_import_decision(
                qtype,
                TensorRole.PROJECTION,
                target_bits=target_bits,
                target_block_size=target_block_size,
            )
            if route not in {
                QuantImportRoute.AFFINE_REPACK,
                QuantImportRoute.DEQUANTIZE_REQUANTIZE,
            }:
                quant_name = get_quant_spec(qtype)
                quant_name = quant_name.name if quant_name is not None else str(qtype)
                raise ValueError(
                    "Quantization-preserving GGUF import would change the dequantized "
                    f"values of {gguf_name} ({quant_name}). Use keep_quantized=False "
                    "(API) or --dequantize (CLI) for explicit float import."
                )
            repacked = repack_gguf_weight_to_target(
                gguf_model,
                raw,
                qtype,
                np_shape,
                target_bits=target_bits,
                target_block_size=target_block_size,
                target_symmetric=target_symmetric,
                tensor_name=hf_name,
                tensor_role=TensorRole.AFFINE_PROJECTION,
            )
            if repacked.bits != target_bits or repacked.block_size != target_block_size:
                raise ValueError(
                    f"Kimi-K3 fused KV-B tensor {gguf_name} does not match the "
                    f"{target_bits}-bit/block-{target_block_size} target"
                )
            heads = int(config.num_attention_heads)
            nope_dim = int(config.qk_nope_head_dim)
            value_dim = int(config.v_head_dim)
            prefix = module_hf_name.removesuffix("kv_b_proj.weight")
            split_ranges = {
                prefix + "k_b_proj": (0, nope_dim),
                prefix + "v_b_proj": (nope_dim, nope_dim + value_dim),
            }
            for target_stem, (start, end) in split_ranges.items():
                if target_stem not in quantized_stems:
                    raise ValueError(
                        f"Kimi-K3 fused KV-B target {target_stem!r} is not quantized"
                    )
                weight = repacked.weight.reshape(
                    heads, nope_dim + value_dim, *repacked.weight.shape[1:]
                )[:, start:end].reshape(-1, *repacked.weight.shape[1:])
                scales = repacked.scales.reshape(
                    heads, nope_dim + value_dim, *repacked.scales.shape[1:]
                )[:, start:end].reshape(-1, *repacked.scales.shape[1:])
                zero_points = (
                    repacked.zero_points.reshape(
                        heads,
                        nope_dim + value_dim,
                        *repacked.zero_points.shape[1:],
                    )[:, start:end].reshape(-1, *repacked.zero_points.shape[1:])
                    if repacked.zero_points is not None
                    else None
                )
                state_dict[f"{target_stem}.weight"] = torch.from_numpy(weight.copy())
                state_dict[f"{target_stem}.scales"] = torch.from_numpy(scales.copy())
                if zero_points is not None:
                    state_dict[f"{target_stem}.zero_points"] = torch.from_numpy(
                        zero_points.copy()
                    )
            n_repacked += 2
            continue

        # Repack every target QuantizedLinear weight. Mixed GGUF presets
        # otherwise leave unsupported source types as full float matrices,
        # which cannot fit the graph's packed MatMulNBits initializer shape.
        stem = hf_name[: -len(".weight")] if hf_name.endswith(".weight") else None
        module_stem = (
            module_hf_name[: -len(".weight")] if module_hf_name.endswith(".weight") else None
        )
        is_tencent_q1_0_tensor = tencent_q1_0 and qtype_val == GGMLQuantizationType.Q1_0.value
        is_quantized_embedding = (
            module_stem is not None and module_stem in quantized_embedding_stems
        )
        should_repack = module_stem is not None and (
            module_stem in quantized_stems or is_quantized_embedding
        )

        native_targets = _native_block_target_stems(
            hf_name,
            np_shape,
            set(native_block_stems),
        )
        affine_targets = _native_block_target_stems(
            module_hf_name,
            np_shape,
            quantized_stems,
        )
        if is_tencent_q1_0_tensor:
            # Tencent tensors are ordinary rank-2 projections. Route them
            # through the custom 130-byte-block parser below rather than the
            # generic target-splitting path, which assumes mainline Q1_0 bytes.
            affine_targets = []
        is_kimi_reshaped_projection = gguf_arch in {
            "kimi-linear",
            "kimi-k3",
        } and module_hf_name.endswith((".k_b_proj.weight", ".v_b_proj.weight"))
        if is_kimi_reshaped_projection:
            # These tensors are rank-3 in GGUF. They target one flattened
            # projection rather than an expert-major collection.
            affine_targets = []
            native_targets = []
        fused_projection_targets = _fused_projection_target_stems(
            module_hf_name, quantized_stems
        )
        native_spec = _native_block_spec(qtype)
        quant_spec = get_quant_spec(qtype)
        is_embedding_tensor = module_stem is not None and module_stem in embedding_stems
        is_encoder_embedding = is_embedding_tensor and gguf_arch in {
            "bert",
            "modern-bert",
        }
        is_float_embedding = is_embedding_tensor and not is_quantized_embedding
        output_targets_quantized = module_stem is not None and (
            module_stem in quantized_stems or module_stem in native_block_stems
        )
        tensor_role = (
            TensorRole.EMBEDDING
            if is_embedding_tensor
            else TensorRole.EXPERT
            if len(np_shape) == 3 and ".experts." in hf_name
            else TensorRole.OUTPUT
            if hf_name == "lm_head.weight" and output_targets_quantized
            else TensorRole.PROJECTION
            if fused_projection_targets
            or (
                module_stem is not None
                and (module_stem in quantized_stems or module_stem in native_block_stems)
            )
            else TensorRole.NON_MATMUL
        )
        route = QuantImportRoute.DEQUANTIZE_REQUANTIZE
        if quant_spec is not None and quant_spec.is_quantized_storage:
            route, _exactness, reason = quant_import_decision(
                qtype,
                tensor_role,
                target_bits=target_bits,
                target_block_size=target_block_size,
            )
            if is_tencent_q1_0_tensor and tensor_role is not TensorRole.NON_MATMUL:
                route = QuantImportRoute.AFFINE_REPACK
                reason = (
                    "Tencent Q1_0 uses the selected exact "
                    f"INT{target_bits} affine block-{target_block_size} representation."
                )
            explicitly_dequantized = (
                dequantize_float_linear_types is not None
                and module_stem in dequantize_float_linear_types
                and quant_spec.name in dequantize_float_linear_types[module_stem]
            )
            if explicitly_dequantized and quant_spec.dequantize is Support.SUPPORTED:
                route = QuantImportRoute.DEQUANTIZE_FLOAT
            if is_kimi_reshaped_projection:
                if quant_spec.dequantize is not Support.SUPPORTED:
                    raise ValueError(
                        f"Cannot reshape quantized {quant_spec.name} tensor {hf_name}: "
                        "the stored format has no supported dequantization route."
                    )
                route = QuantImportRoute.DEQUANTIZE_REQUANTIZE
            if (
                is_encoder_embedding or is_float_embedding
            ) and quant_spec.dequantize is Support.SUPPORTED:
                route = QuantImportRoute.DEQUANTIZE_FLOAT
            if route is QuantImportRoute.REJECTED:
                raise ValueError(
                    f"Cannot import GGUF tensor {gguf_name} mapped to {hf_name} "
                    f"({quant_spec.name}, role={tensor_role.value}): {reason}"
                )
            if route is QuantImportRoute.NATIVE_BYTES and not native_targets:
                raise ValueError(
                    f"Cannot preserve native {quant_spec.name} bytes for {hf_name}: "
                    "the mapped module does not expose a compatible "
                    "BlockQuantizedMatMul target."
                )
            if (
                tensor_role is TensorRole.EMBEDDING
                and route is not QuantImportRoute.DEQUANTIZE_FLOAT
                and not is_quantized_embedding
                and not is_encoder_embedding
            ):
                raise ValueError(
                    f"Cannot keep {quant_spec.name} embedding {hf_name} quantized: "
                    "the model graph does not expose GatherBlockQuantized. Use "
                    "keep_quantized=False for explicit float import."
                )
            if (
                tensor_role in {TensorRole.PROJECTION, TensorRole.OUTPUT}
                and route
                in {
                    QuantImportRoute.AFFINE_REPACK,
                    QuantImportRoute.DEQUANTIZE_REQUANTIZE,
                }
                and not should_repack
                and not fused_projection_targets
                and (
                    dequantize_float_linear_types is None
                    or module_stem not in dequantize_float_linear_types
                    or quant_spec.name not in dequantize_float_linear_types[module_stem]
                )
            ):
                raise ValueError(
                    f"Cannot keep {quant_spec.name} {tensor_role.value} {hf_name} "
                    "quantized: the model graph does not expose MatMulNBits. Use "
                    "keep_quantized=False for explicit float import."
                )
        if qtype_val in float_type_ids and (
            fused_projection_targets or affine_targets or should_repack
        ):
            raise ValueError(
                "Quantization-preserving GGUF import would quantize float projection "
                f"{gguf_name} ({getattr(qtype, 'name', qtype)}) to the graph's "
                f"{target_bits}-bit/block-{target_block_size} MatMulNBits contract. "
                "Use keep_quantized=False (API) or --dequantize (CLI) for an "
                "explicit float import."
            )
        if fused_projection_targets:
            if route not in {
                QuantImportRoute.AFFINE_REPACK,
                QuantImportRoute.DEQUANTIZE_REQUANTIZE,
            }:
                raise ValueError(
                    "Quantization-preserving GGUF import cannot split fused projection "
                    f"{gguf_name} ({getattr(qtype, 'name', qtype)}): no packed target "
                    "conversion route is available."
                )
            repacked = repack_gguf_weight_to_target(
                gguf_model,
                raw,
                qtype,
                np_shape,
                target_bits=target_bits,
                target_block_size=target_block_size,
                target_symmetric=target_symmetric,
                tensor_name=hf_name,
                tensor_role=TensorRole.AFFINE_PROJECTION,
            )
            offset = 0
            for target_stem in fused_projection_targets:
                n_out = quantized_output_sizes[target_stem]
                end = offset + n_out
                target_name = f"{target_stem}.weight"
                weight = torch.from_numpy(np.array(repacked.weight[offset:end], copy=True))
                scales = torch.from_numpy(np.array(repacked.scales[offset:end], copy=True))
                zero_points = (
                    torch.from_numpy(np.array(repacked.zero_points[offset:end], copy=True))
                    if repacked.zero_points is not None
                    else None
                )
                if _needs_qk_permute(
                    target_name,
                    num_heads,
                    num_kv_heads,
                    model_type,
                    gguf_arch,
                ):
                    n_head = num_heads if ".q_proj." in target_name else num_kv_heads
                    weight = _reverse_permute(weight, n_head)
                    scales = _reverse_permute(scales, n_head)
                    if zero_points is not None:
                        zero_points = _reverse_permute(zero_points, n_head)
                state_dict[target_name] = weight
                state_dict[f"{target_stem}.scales"] = scales
                if zero_points is not None:
                    state_dict[f"{target_stem}.zero_points"] = zero_points
                offset = end
            if offset != int(np_shape[0]):
                raise ValueError(
                    f"Fused projection tensor {hf_name!r} has {np_shape[0]} rows, "
                    f"but its split targets require {offset}"
                )
            n_repacked += len(fused_projection_targets)
        elif native_targets and native_spec is not None:
            n_out = int(np_shape[-2])
            k_in = int(np_shape[-1])
            packed = preserve_native_blocks(
                raw,
                qtype_val,
                (len(native_targets) * n_out, k_in),
            )
            packed = packed.reshape(
                len(native_targets),
                n_out,
                packed.shape[-2],
                native_spec.bytes,
            )
            for index, native_stem in enumerate(native_targets):
                w = torch.from_numpy(np.array(packed[index], copy=True))
                target_name = f"{native_stem}.weight"
                needs_permute = _needs_qk_permute(
                    target_name,
                    num_heads,
                    num_kv_heads,
                    model_type,
                    gguf_arch,
                )
                if needs_permute:
                    n_head = (
                        num_heads
                        if ".q_proj." in target_name or ".qkv_proj." in target_name
                        else num_kv_heads
                    )
                    w = _reverse_permute(w, n_head)
                state_dict[target_name] = w
                if (
                    reuse_candidates is not None
                    and len(native_targets) == 1
                    and not needs_permute
                ):
                    from mobius.integrations.gguf._reuse import GGUFReuseCandidate

                    offset, length, qtype_name = gguf_model.tensor_storage_range(gguf_name)
                    reuse_candidates[id(w)] = GGUFReuseCandidate(
                        gguf_name,
                        offset,
                        length,
                        qtype_name,
                        tuple(int(dim) for dim in w.shape),
                    )
            n_repacked += len(native_targets)
        elif affine_targets:
            # GGUF stores routed experts as one expert-major 3-D tensor
            # [E, N, K]. MatMulNBits parameters remain per expert, so repack
            # the contiguous [E*N, K] matrix once and split only after packing.
            num_experts = len(affine_targets)
            n_out = int(np_shape[-2])
            k_in = int(np_shape[-1])
            aggregate_shape = (num_experts * n_out, k_in)
            if can_repack(qtype_val):
                repacked = repack_gguf_tensor(
                    raw.ravel().view(np.uint8),
                    qtype_val,
                    aggregate_shape,
                )
                if repacked.bits != target_bits or repacked.block_size != target_block_size:
                    _require_supported_requantization(
                        bits=target_bits,
                        block_size=target_block_size,
                        tensor_name=hf_name,
                    )
                    values = gguf_model.dequantize_raw_tensor(raw, qtype, np_shape)
                    repacked = repack_dequantized_tensor(
                        values.reshape(aggregate_shape),
                        bits=target_bits,
                        block_size=target_block_size,
                        symmetric=target_symmetric,
                    )
                    n_requantized += 1
            else:
                _require_supported_requantization(
                    bits=target_bits,
                    block_size=target_block_size,
                    tensor_name=hf_name,
                )
                values = gguf_model.dequantize_raw_tensor(raw, qtype, np_shape)
                repacked = repack_dequantized_tensor(
                    values.reshape(aggregate_shape),
                    bits=target_bits,
                    block_size=target_block_size,
                    symmetric=target_symmetric,
                )
                n_requantized += 1

            packed = repacked.weight.reshape(num_experts, n_out, *repacked.weight.shape[1:])
            scales = repacked.scales.reshape(num_experts, n_out, *repacked.scales.shape[1:])
            zero_points = (
                repacked.zero_points.reshape(
                    num_experts,
                    n_out,
                    *repacked.zero_points.shape[1:],
                )
                if repacked.zero_points is not None
                else None
            )
            for index, target_stem in enumerate(affine_targets):
                target_name = f"{target_stem}.weight"
                weight = torch.from_numpy(np.array(packed[index], copy=True))
                scale = torch.from_numpy(np.array(scales[index], copy=True))
                zero_point = (
                    torch.from_numpy(np.array(zero_points[index], copy=True))
                    if zero_points is not None
                    else None
                )
                if _needs_qk_permute(
                    target_name,
                    num_heads,
                    num_kv_heads,
                    model_type,
                    gguf_arch,
                ):
                    n_head = num_heads if ".q_proj." in target_name else num_kv_heads
                    weight = _reverse_permute(weight, n_head)
                    scale = _reverse_permute(scale, n_head)
                    if zero_point is not None:
                        zero_point = _reverse_permute(zero_point, n_head)
                state_dict[target_name] = weight
                state_dict[f"{target_stem}.scales"] = scale
                if zero_points is not None:
                    assert zero_point is not None
                    state_dict[f"{target_stem}.zero_points"] = zero_point
            n_repacked += num_experts
        elif should_repack:
            if is_tencent_q1_0_tensor:
                read_source_range, data_section_offset, reader_tensor = (
                    gguf_model._tensor_source(gguf_name)
                )
                repacked = parse_tencent_q1_0_tensor(
                    read_source_range,
                    data_section_offset,
                    reader_tensor,
                )
                if (repacked.bits, repacked.block_size) != (
                    target_bits,
                    target_block_size,
                ):
                    raise ValueError(
                        f"Tencent Q1_0 parser produced INT{repacked.bits} "
                        f"block-{repacked.block_size} for {hf_name}, but the graph "
                        f"expects INT{target_bits} block-{target_block_size}."
                    )
            elif is_kimi_reshaped_projection:
                values = gguf_model.dequantize_raw_tensor(raw, qtype, np_shape)
                if hf_name.endswith(".k_b_proj.weight"):
                    values = values.transpose(0, 2, 1).reshape(
                        int(np_shape[0]) * int(np_shape[2]), int(np_shape[1])
                    )
                else:
                    values = values.reshape(
                        int(np_shape[0]) * int(np_shape[1]), int(np_shape[2])
                    )
                repacked = repack_dequantized_tensor(
                    values,
                    bits=target_bits,
                    block_size=target_block_size,
                    symmetric=target_symmetric,
                )
                n_requantized += 1
            elif route is QuantImportRoute.AFFINE_REPACK and can_repack(qtype_val):
                shape_2d = (int(np_shape[0]), int(np_shape[1]))
                repacked = repack_gguf_tensor(
                    raw.ravel().view(np.uint8),
                    qtype_val,
                    shape_2d,
                )
                if repacked.bits != target_bits or repacked.block_size != target_block_size:
                    _require_supported_requantization(
                        bits=target_bits,
                        block_size=target_block_size,
                        tensor_name=hf_name,
                    )
                    values = gguf_model.dequantize_raw_tensor(raw, qtype, np_shape)
                    repacked = repack_dequantized_tensor(
                        values,
                        bits=target_bits,
                        block_size=target_block_size,
                        symmetric=target_symmetric,
                    )
                    n_requantized += 1
            else:
                _require_supported_requantization(
                    bits=target_bits,
                    block_size=target_block_size,
                    tensor_name=hf_name,
                )
                values = gguf_model.dequantize_raw_tensor(raw, qtype, np_shape)
                repacked = repack_dequantized_tensor(
                    values,
                    bits=target_bits,
                    block_size=target_block_size,
                    symmetric=target_symmetric,
                )
                n_requantized += 1
            w = torch.from_numpy(repacked.weight)
            s = torch.from_numpy(repacked.scales)

            # Apply Q/K row permutation to quantized tensors
            # (same transform as _process_llama, on all arrays). Only
            # llama-family archs use the interleaved-rope permute; Qwen
            # and others must NOT be permuted.
            if _needs_qk_permute(hf_name, num_heads, num_kv_heads, model_type, gguf_arch):
                n_head = (
                    num_heads
                    if ".q_proj." in hf_name or ".qkv_proj." in hf_name
                    else num_kv_heads
                )
                w = _reverse_permute(w, n_head)
                s = _reverse_permute(s, n_head)

            if is_quantized_embedding:
                state_dict[f"{stem}.qweight"] = w.reshape(w.shape[0], -1)
            else:
                state_dict[hf_name] = w
            state_dict[f"{stem}.scales"] = s
            if repacked.zero_points is not None:
                zp = torch.from_numpy(repacked.zero_points)
                if _needs_qk_permute(hf_name, num_heads, num_kv_heads, model_type, gguf_arch):
                    zp = _reverse_permute(zp, n_head)
                state_dict[f"{stem}.zero_points"] = zp
            n_repacked += 1
        else:
            # Dequantize weights whose graph target is intentionally float.
            if qtype in (
                GGMLQuantizationType.F32,
                GGMLQuantizationType.F16,
            ):
                arr = gguf_model.get_tensor(gguf_name)
                # F32/F16 tensors are mmap'd read-only views
                if not arr.flags.writeable:
                    arr = np.array(arr)
            else:
                arr = dequantize(raw, qtype).reshape(np_shape)
            tensor = torch.from_numpy(arr)
            state_dict[hf_name] = tensor
            if reuse_candidates is not None and qtype in (
                GGMLQuantizationType.F32,
                GGMLQuantizationType.F16,
            ):
                from mobius.integrations.gguf._reuse import GGUFReuseCandidate

                offset, length, qtype_name = gguf_model.tensor_storage_range(gguf_name)
                reuse_candidates[id(tensor)] = GGUFReuseCandidate(
                    gguf_name,
                    offset,
                    length,
                    qtype_name,
                    tuple(int(dim) for dim in arr.shape),
                )

    logger.info(
        "Loaded %d state_dict entries (%d GGUF tensors repacked for quantized ops, "
        "%d requantized from mixed source types)",
        len(state_dict),
        n_repacked,
        n_requantized,
    )
    return state_dict


def _validate_moe_weight_shape(
    name: str,
    shape: tuple[int, ...],
    config,
    *,
    gguf_arch: str | None = None,
) -> None:
    """Reject router/expert tensors that could otherwise be partially routed."""
    num_experts = getattr(config, "num_local_experts", None)
    if num_experts is None:
        return
    expert_size = getattr(config, "moe_intermediate_size", None) or config.intermediate_size
    expert_markers = [".mlp.experts.", ".feed_forward.experts."]
    router_suffixes = [".mlp.gate.weight", ".feed_forward.gate.weight"]
    if gguf_arch == "arctic":
        expert_markers.append(".moe.experts.")
        router_suffixes.append(".moe.gate.weight")
    expert_marker = next(
        (marker for marker in expert_markers if marker in name),
        None,
    )
    if expert_marker is not None:
        projection = name.rsplit(expert_marker, 1)[1].split(".", 1)[0]
        if projection not in {"gate_proj", "up_proj", "down_proj"}:
            return
        expected = (
            (num_experts, config.hidden_size, expert_size)
            if projection == "down_proj"
            else (num_experts, expert_size, config.hidden_size)
        )
        if shape != expected:
            raise ValueError(
                f"Invalid stacked expert shape for {name}: expected {expected}, got {shape}"
            )
    elif name.endswith(tuple(router_suffixes)):
        expected = (num_experts, config.hidden_size)
        if shape != expected:
            raise ValueError(
                f"Invalid router shape for {name}: expected {expected}, got {shape}"
            )


def _needs_qk_permute(
    hf_name: str,
    num_heads: int | None,
    num_kv_heads: int | None,
    model_type: str | None = None,
    gguf_arch: str | None = None,
) -> bool:
    """Check if this tensor needs Q/K reverse-permutation.

    Two conditions must hold: the tensor must be a Q/K projection weight,
    AND the model architecture must actually use llama.cpp's
    interleaved-rope permute. Name-based gating alone is insufficient —
    Qwen2/Qwen3 use ``.q_proj.``/``.k_proj.`` names too but store Q/K in
    plain HF order (NEOX rope) and must NOT be permuted, or their
    attention heads get scrambled and the model emits garbage.
    """
    from mobius.integrations.gguf._arch_registry import try_get_arch_spec
    from mobius.integrations.gguf._tensor_processors import needs_llama_qk_permute

    if num_heads is None or num_kv_heads is None:
        return False
    spec = try_get_arch_spec(gguf_arch) if gguf_arch is not None else None
    permute = spec.llama_qk_permute if spec is not None else needs_llama_qk_permute(model_type)
    if not permute:
        return False
    return (
        ".q_proj." in hf_name or ".k_proj." in hf_name or ".qkv_proj." in hf_name
    ) and hf_name.endswith(".weight")
