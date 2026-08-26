# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Structured real-artifact evidence required for GGUF runtime support."""

from __future__ import annotations

__all__ = [
    "GGUFArtifactIdentity",
    "GGUFGraphPackageIdentity",
    "GGUFRuntimeEvidence",
    "gguf_artifact_identity",
    "gguf_graph_package_identity",
    "iter_runtime_evidence",
    "matching_runtime_evidence",
    "runtime_evidence",
    "validate_quant_runtime_evidence_ids",
    "validate_runtime_evidence_ids",
]

import dataclasses
import hashlib
import os
import stat
from collections import Counter
from pathlib import Path
from types import MappingProxyType
from typing import Any


@dataclasses.dataclass(frozen=True, slots=True)
class GGUFArtifactIdentity:
    """Immutable identity of the exact GGUF bytes used for graph construction."""

    architecture: str
    filename: str
    size: int
    sha256: str
    tensor_count: int
    tensor_qtypes: tuple[tuple[str, int], ...]


@dataclasses.dataclass(frozen=True, slots=True)
class GGUFGraphPackageIdentity:
    """Canonical identity of the exact serialized ONNX graph package."""

    files: tuple[str, ...]
    sha256: str


@dataclasses.dataclass(frozen=True, slots=True)
class GGUFRuntimeEvidence:
    """One immutable end-to-end runtime evidence record."""

    evidence_id: str
    architecture: str
    repository: str
    revision: str
    filename: str
    size: int
    lfs_sha256: str
    config_repository: str
    config_revision: str
    tokenizer_repository: str
    tokenizer_revision: str
    tokenizer_metadata_sha256: str
    tokenizer_assets: tuple[tuple[str, int, str], ...]
    tensor_count: int
    tensor_qtypes: tuple[tuple[str, int], ...]
    import_route: str
    graph_files: tuple[str, ...]
    graph_sha256: str
    runtime_package_files: tuple[str, ...]
    runtime_package_sha256: str
    parity_test: str
    parity_kind: str
    deterministic_test: str
    stateful_semantics: str
    execution_provider: str
    onnxruntime_version: str
    runtime: str
    runtime_version: str
    result: str = "passed"

    def __post_init__(self) -> None:
        text_fields = (
            self.evidence_id,
            self.architecture,
            self.repository,
            self.revision,
            self.filename,
            self.lfs_sha256,
            self.config_repository,
            self.config_revision,
            self.tokenizer_repository,
            self.tokenizer_revision,
            self.tokenizer_metadata_sha256,
            self.import_route,
            self.graph_sha256,
            self.runtime_package_sha256,
            self.parity_test,
            self.parity_kind,
            self.deterministic_test,
            self.stateful_semantics,
            self.execution_provider,
            self.onnxruntime_version,
            self.runtime,
            self.runtime_version,
            self.result,
        )
        if any(not value.strip() for value in text_fields):
            raise ValueError("GGUF runtime evidence fields must be non-empty")
        if self.size <= 0 or self.tensor_count <= 0 or not self.tensor_qtypes:
            raise ValueError("GGUF runtime evidence requires positive artifact/tensor census")
        revisions = (self.revision, self.config_revision, self.tokenizer_revision)
        if (
            any(len(value) != 40 for value in revisions)
            or any(
                len(value) != 64
                for value in (
                    self.lfs_sha256,
                    self.tokenizer_metadata_sha256,
                    self.graph_sha256,
                    self.runtime_package_sha256,
                )
            )
            or any(
                not _is_hex(value)
                for value in (
                    *revisions,
                    self.lfs_sha256,
                    self.tokenizer_metadata_sha256,
                    self.graph_sha256,
                    self.runtime_package_sha256,
                )
            )
        ):
            raise ValueError(
                "GGUF runtime evidence requires immutable 40-hex revisions and LFS SHA-256"
            )
        if self.parity_kind not in {"full-logit", "component"}:
            raise ValueError(
                "GGUF runtime evidence parity_kind must be full-logit or component"
            )
        if self.result != "passed":
            raise ValueError(
                "GGUF runtime evidence may only support a route after a passed result"
            )
        asset_names = tuple(asset[0] for asset in self.tokenizer_assets)
        if (
            not self.tokenizer_assets
            or "tokenizer.json" not in asset_names
            or asset_names != tuple(sorted(asset_names))
            or len(set(asset_names)) != len(asset_names)
            or any(
                filename != Path(filename).name
                or size <= 0
                or len(sha256) != 64
                or not _is_hex(sha256)
                for filename, size, sha256 in self.tokenizer_assets
            )
        ):
            raise ValueError(
                "GGUF runtime evidence tokenizer_assets must be sorted, unique, "
                "basename-only exact file identities including tokenizer.json"
            )
        if (
            not self.graph_files
            or tuple(sorted(self.graph_files)) != self.graph_files
            or len(set(self.graph_files)) != len(self.graph_files)
        ):
            raise ValueError("GGUF runtime evidence graph_files must be sorted and unique")
        if (
            not self.runtime_package_files
            or tuple(sorted(self.runtime_package_files)) != self.runtime_package_files
            or len(set(self.runtime_package_files)) != len(self.runtime_package_files)
        ):
            raise ValueError(
                "GGUF runtime evidence runtime_package_files must be sorted and unique"
            )


def _is_hex(value: str) -> bool:
    return all(character in "0123456789abcdefABCDEF" for character in value)


_SMOLLM_F16_ROUTE = (
    '{"architecture":"llama","config_sha256":'
    '"d3f3f2abf531abde55e04a52b5c892c93943b7e260160c796c490618a2e84886",'
    '"execution_provider":"cpu","model_type":"llama","module_type":"llama",'
    '"preserve_quantization":false,"registry_import":{"config_key_map":null,'
    '"config_postprocessor":null,"llama_qk_permute":true,"offset_norm":false,'
    '"required_metadata":[],"rope_interleave":false,"tensor_processor":"llama",'
    '"v_head_reorder":false,"vlm_builder":null},"route_schema":1,"static_cache":false,'
    '"task":{"class":"builtins.str","state":"text-generation"},'
    '"tensor_map_recipe":["llama"]}'
)

_SMOLLM_F16_ONNX_RUNTIME = GGUFRuntimeEvidence(
    evidence_id="smollm-135m-f16-onnxruntime-1.29.0",
    architecture="llama",
    repository="neopolita/smollm-135m-gguf",
    revision="22cca988936eafe92908e7558907c3964e10bba7",
    filename="ggml-model-f16.gguf",
    size=270_885_504,
    lfs_sha256="ec8c775c16944a7e4b5251f97b3f848500dcc3e701b0d492ce9055cea42138a2",
    config_repository="HuggingFaceTB/SmolLM-135M",
    config_revision="1d461723eec654e65efdc40cf49301c89c0c92f4",
    tokenizer_repository="HuggingFaceTB/SmolLM-135M",
    tokenizer_revision="1d461723eec654e65efdc40cf49301c89c0c92f4",
    tokenizer_metadata_sha256="46646ba36ecae43de6f9f649d217774b889e0fd405af92205319b882927493fc",
    tokenizer_assets=(
        (
            "special_tokens_map.json",
            831,
            "e786b595b9a23148bf1630df78d9037a048ea671e48bfd3549a1e3c233742bb3",
        ),
        (
            "tokenizer.json",
            2_104_556,
            "9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c",
        ),
        (
            "tokenizer_config.json",
            3_685,
            "238ad6b60d48e471624ea70bc79e92f2611844d5016471fee8c167854bcb98e8",
        ),
    ),
    tensor_count=272,
    tensor_qtypes=(("F16", 211), ("F32", 61)),
    import_route=_SMOLLM_F16_ROUTE,
    graph_files=("model.onnx", "model.onnx.data"),
    graph_sha256="3d242b09fcb5041d71e5914084cf00780867b3b0e32f669f8733369b19b6ea9b",
    runtime_package_files=(
        "gguf_tokenizer_manifest.json",
        "inference_metadata.yaml",
        "model.onnx",
        "model.onnx.data",
        "policies/cache_length_update.onnx",
        "policies/decoder_state_initializer.onnx",
        "policies/decoder_step_update.onnx",
        "policies/generated_length_update.onnx",
        "policies/last_token_logits.onnx",
        "policies/termination.onnx",
        "policies/termination_batch_initializer.onnx",
        "policies/token_sampler.onnx",
        "policies/token_state_update.onnx",
        "policies/token_to_slot.onnx",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ),
    runtime_package_sha256="5b6fdbdb1db7f7fb9423f2356820812712556abf783acb6bd63c572920031982",
    parity_test="test_small_f16_gguf_cli_full_logit_and_generation_parity[smollm-135m-f16]",
    parity_kind="full-logit",
    deterministic_test=(
        "test_small_f16_gguf_cli_full_logit_and_generation_parity[smollm-135m-f16]"
    ),
    stateful_semantics="dynamic KV cache prefill plus 20 cache-threaded decode steps",
    execution_provider="CPUExecutionProvider",
    onnxruntime_version="1.29.0",
    runtime="onnx-genai",
    runtime_version="1.29.0",
)

_SMOLLM_F16_ORT_GENAI = dataclasses.replace(
    _SMOLLM_F16_ONNX_RUNTIME,
    evidence_id="smollm-135m-f16-ort-genai-0.15.2",
    runtime_package_files=(
        "genai_config.json",
        "gguf_tokenizer_manifest.json",
        "model.onnx",
        "model.onnx.data",
        "runtime_compatibility.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ),
    runtime_package_sha256="43568320f669d259d5a570ee04bd6378316ab31ce2fcb6383e75b479b4f2b349",
    deterministic_test="test_smollm_generic_ort_genai_generation",
    stateful_semantics="ORT GenAI prefill plus 20 cache-threaded decode steps",
    execution_provider="CPUExecutionProvider",
    onnxruntime_version="1.29.0",
    runtime="ort-genai",
    runtime_version="0.15.2",
)

_QWEN25_Q8_ORT_GENAI = GGUFRuntimeEvidence(
    evidence_id="qwen2.5-0.5b-instruct-q8-ort-genai-0.15.2",
    architecture="qwen2",
    repository="Qwen/Qwen2.5-0.5B-Instruct-GGUF",
    revision="9217f5db79a29953eb74d5343926648285ec7e67",
    filename="qwen2.5-0.5b-instruct-q8_0.gguf",
    size=675_710_816,
    lfs_sha256="ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e",
    config_repository="Qwen/Qwen2.5-0.5B-Instruct",
    config_revision="7ae557604adf67be50417f59c2c2f167def9a775",
    tokenizer_repository="Qwen/Qwen2.5-0.5B-Instruct",
    tokenizer_revision="a338b55dd21219a5f4da42bc11a9313d1a27d4cc",
    tokenizer_metadata_sha256="8fc8ef848104e931f14ae03d9581699d54813a2ff952fb7caac0654e8aa27ee3",
    tokenizer_assets=(
        (
            "tokenizer.json",
            7_031_645,
            "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539",
        ),
        (
            "tokenizer_config.json",
            7_308,
            "5214600ee45ca2f887ce2eede8910378a0111ea99d657428bcbce94778e65a92",
        ),
    ),
    tensor_count=291,
    tensor_qtypes=(("F32", 121), ("Q8_0", 170)),
    import_route='{"architecture":"qwen2","config_sha256":"f7391f2aac9a7617c1c10e397e91b6f31b80bb3c5f338966b46e7d3935246500","execution_provider":"cpu","model_type":"qwen2","module_type":"qwen2","preserve_quantization":true,"registry_import":{"config_key_map":null,"config_postprocessor":null,"llama_qk_permute":false,"offset_norm":false,"required_metadata":[],"rope_interleave":false,"tensor_processor":null,"v_head_reorder":false,"vlm_builder":null},"route_schema":1,"static_cache":false,"task":{"class":"builtins.str","state":"text-generation"},"tensor_map_recipe":["llama"]}',
    graph_files=("model.onnx", "model.onnx.data"),
    graph_sha256="b0d1814ea69dddd405e30541e9009ba6fb73930f98538921d5c2eaa4a14f5d2c",
    runtime_package_files=(
        "genai_config.json",
        "gguf_tokenizer_manifest.json",
        "model.onnx",
        "model.onnx.data",
        "runtime_compatibility.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ),
    runtime_package_sha256="df2211e72f43d3ee0aa31e6ff7b2cc9817f7a76d6169d28f96e71b65fd553d69",
    parity_test="test_promoted_gguf_full_runtime_evidence[qwen2.5-0.5b-instruct-q8]",
    parity_kind="full-logit",
    deterministic_test="test_promoted_gguf_full_runtime_evidence[qwen2.5-0.5b-instruct-q8]",
    stateful_semantics="dynamic KV cache prefill, replay, rollback, reorder, and 20 decode steps",
    execution_provider="CPUExecutionProvider",
    onnxruntime_version="1.29.0",
    runtime="ort-genai",
    runtime_version="0.15.2",
)

_LFM2_350M_F16_ORT_GENAI = GGUFRuntimeEvidence(
    evidence_id="lfm2-350m-f16-ort-genai-0.15.2",
    architecture="lfm2",
    repository="LiquidAI/LFM2-350M-GGUF",
    revision="8fdc9d526b7ed346b19257551b05816c7912ecc2",
    filename="LFM2-350M-F16.gguf",
    size=711_482_304,
    lfs_sha256="379ffdcbf08147c0313f6f1ce7ff558a2bc935eda633f4b46c52347032419c42",
    config_repository="LiquidAI/LFM2-350M",
    config_revision="f37d3f5c8c5484bc01dad379a595cf4c68c4e70e",
    tokenizer_repository="LiquidAI/LFM2-350M",
    tokenizer_revision="73e3c253078a3b97c2e14b4c4665679f4d9b6d56",
    tokenizer_metadata_sha256="e5626d605bb50bc53fdb0fbfcf374fb33dfbaa0cc698d9746ba1e9b0b7e6d07d",
    tokenizer_assets=(
        (
            "chat_template.jinja",
            209,
            "a805e50fed68938a076b07e2e602639611b50b1ced0e50f11eb92f1ba25be4dc",
        ),
        (
            "special_tokens_map.json",
            434,
            "742aefe2b7dec496e8caffdba03a75d0c1a9925d53bd3f3e0d388c96b591b6f4",
        ),
        (
            "tokenizer.json",
            4_732_426,
            "98cff83b4f6d7e9d8929bebc62b07e92cf1b3f99c80d16bafe8b84a75448f40b",
        ),
        (
            "tokenizer_config.json",
            91_509,
            "36f511115e9d8952cbc9d15d9a20dfa7ce7d1444940e5c1dc42a762020c99bf5",
        ),
    ),
    tensor_count=148,
    tensor_qtypes=(("F16", 93), ("F32", 55)),
    import_route='{"architecture":"lfm2","config_sha256":"c961bba579ea33a2472a7c5d3f469c76f1f8c7aae8440a7eaa86bc6e878a42f4","execution_provider":"cpu","model_type":"lfm2","module_type":"lfm2","preserve_quantization":false,"registry_import":{"config_key_map":null,"config_postprocessor":null,"llama_qk_permute":false,"offset_norm":false,"required_metadata":["attention.head_count_kv","attention.layer_norm_rms_epsilon","shortconv.l_cache"],"rope_interleave":false,"tensor_processor":null,"v_head_reorder":false,"vlm_builder":null},"route_schema":1,"static_cache":false,"task":{"class":"builtins.str","state":"hybrid-text-generation"},"tensor_map_recipe":["lfm2"]}',
    graph_files=("model.onnx", "model.onnx.data"),
    graph_sha256="2a15694cd5ff9f5c9f798feeca91cc41174842065ada80a18b288478725b3342",
    runtime_package_files=(
        "chat_template.jinja",
        "genai_config.json",
        "gguf_tokenizer_manifest.json",
        "model.onnx",
        "model.onnx.data",
        "runtime_compatibility.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ),
    runtime_package_sha256="87b8cdd1edc8c5be948716df3efe551f5fc8f96762d7ea3f99b27688ca9f24af",
    parity_test="test_promoted_gguf_full_runtime_evidence[lfm2-350m-f16]",
    parity_kind="full-logit",
    deterministic_test="test_promoted_gguf_full_runtime_evidence[lfm2-350m-f16]",
    stateful_semantics=(
        "hybrid convolution and KV state prefill, replay, rollback, reorder, "
        "and 20 decode steps"
    ),
    execution_provider="CPUExecutionProvider",
    onnxruntime_version="1.29.0",
    runtime="ort-genai",
    runtime_version="0.15.2",
)

_RUNTIME_EVIDENCE: MappingProxyType[str, GGUFRuntimeEvidence] = MappingProxyType(
    {
        record.evidence_id: record
        for record in (
            _LFM2_350M_F16_ORT_GENAI,
            _QWEN25_Q8_ORT_GENAI,
            _SMOLLM_F16_ONNX_RUNTIME,
            _SMOLLM_F16_ORT_GENAI,
        )
    }
)


def runtime_evidence(evidence_id: str) -> GGUFRuntimeEvidence | None:
    """Return a structured evidence record by stable ID."""
    return _RUNTIME_EVIDENCE.get(evidence_id)


def iter_runtime_evidence() -> tuple[GGUFRuntimeEvidence, ...]:
    """Return every runtime evidence record ordered by stable evidence ID."""
    return tuple(_RUNTIME_EVIDENCE[key] for key in sorted(_RUNTIME_EVIDENCE))


def validate_quant_runtime_evidence_ids(qtype: str, evidence_ids: tuple[str, ...]) -> None:
    """Require complete preserved-route evidence for a stored quantization type."""
    if not evidence_ids:
        raise ValueError("quantized runtime=SUPPORTED requires structured evidence IDs")
    unknown = sorted(
        evidence_id
        for evidence_id in evidence_ids
        if not evidence_id or runtime_evidence(evidence_id) is None
    )
    if unknown:
        raise ValueError(f"Unknown GGUF quantized runtime evidence IDs: {unknown}")
    required_state_semantics = ("prefill", "replay", "rollback", "reorder", "decode")
    invalid = []
    for evidence_id in evidence_ids:
        evidence = _RUNTIME_EVIDENCE[evidence_id]
        qtypes = dict(evidence.tensor_qtypes)
        if (
            qtypes.get(qtype, 0) <= 0
            or evidence.parity_kind != "full-logit"
            or '"preserve_quantization":true' not in evidence.import_route
            or evidence.execution_provider != "CPUExecutionProvider"
            or any(
                term not in evidence.stateful_semantics for term in required_state_semantics
            )
        ):
            invalid.append(evidence_id)
    if invalid:
        raise ValueError(
            f"GGUF quantized runtime evidence does not prove preserved {qtype} full-logit "
            f"CPU prefill/decode/replay/rollback/reorder semantics: {sorted(invalid)}"
        )


def validate_runtime_evidence_ids(architecture: str, evidence_ids: tuple[str, ...]) -> None:
    """Require every runtime evidence ID to resolve to a complete record."""
    if not evidence_ids:
        raise ValueError("runtime=SUPPORTED requires structured evidence IDs")
    unknown = sorted(
        evidence_id
        for evidence_id in evidence_ids
        if not evidence_id or runtime_evidence(evidence_id) is None
    )
    if unknown:
        raise ValueError(f"Unknown GGUF runtime evidence IDs: {unknown}")
    mismatched = sorted(
        evidence_id
        for evidence_id in evidence_ids
        if _RUNTIME_EVIDENCE[evidence_id].architecture != architecture
    )
    if mismatched:
        raise ValueError(
            f"GGUF runtime evidence IDs do not belong to {architecture!r}: {mismatched}"
        )


def matching_runtime_evidence(
    evidence_ids: tuple[str, ...],
    *,
    architecture: str,
    runtime: str,
    source_path: Path,
    gguf_model: Any,
    built_identity: GGUFArtifactIdentity,
    import_route: str,
    runtime_version: str | None,
    tokenizer_repository: str,
    tokenizer_revision: str,
) -> GGUFRuntimeEvidence:
    """Return exact evidence for the package source, route, and requested runtime."""
    validate_runtime_evidence_ids(architecture, evidence_ids)
    current_identity = gguf_artifact_identity(
        source_path,
        gguf_model,
        architecture=architecture,
        filename=built_identity.filename,
    )
    if current_identity != built_identity:
        raise ValueError(
            "The GGUF source no longer matches the exact artifact identity captured during "
            f"graph construction: built={built_identity!r}, current={current_identity!r}."
        )
    if runtime_version is None:
        raise ValueError(
            "Runtime packaging requires the exact runtime version covered by evidence."
        )
    identity = built_identity
    candidates = [
        _RUNTIME_EVIDENCE[evidence_id]
        for evidence_id in evidence_ids
        if _RUNTIME_EVIDENCE[evidence_id].runtime == runtime
        and _RUNTIME_EVIDENCE[evidence_id].runtime_version == runtime_version
        and _RUNTIME_EVIDENCE[evidence_id].filename == identity.filename
        and _RUNTIME_EVIDENCE[evidence_id].size == identity.size
        and _RUNTIME_EVIDENCE[evidence_id].lfs_sha256 == identity.sha256
        and _RUNTIME_EVIDENCE[evidence_id].tensor_count == identity.tensor_count
        and _RUNTIME_EVIDENCE[evidence_id].tensor_qtypes == identity.tensor_qtypes
        and _RUNTIME_EVIDENCE[evidence_id].import_route == import_route
        and _RUNTIME_EVIDENCE[evidence_id].tokenizer_repository == tokenizer_repository
        and _RUNTIME_EVIDENCE[evidence_id].tokenizer_revision == tokenizer_revision
    ]
    if len(candidates) != 1:
        raise ValueError(
            f"No unique GGUF runtime evidence matches architecture={architecture!r}, "
            f"runtime={runtime!r} {runtime_version!r}, artifact={identity!r}, "
            f"import_route={import_route!r}."
        )
    return candidates[0]


def gguf_artifact_identity(
    source_path: Path,
    gguf_model: Any,
    *,
    architecture: str,
    filename: str | None = None,
) -> GGUFArtifactIdentity:
    """Fingerprint source bytes and parsed tensor census under a canonical architecture."""
    shard_paths = getattr(gguf_model, "shard_paths", None)
    if shard_paths is None:
        stat, sha256 = _hash_regular_file(source_path)
        size = stat.st_size
    else:
        paths = tuple(Path(path) for path in shard_paths)
        if not paths:
            raise ValueError("A GGUF shard set must contain at least one source file.")
        digest = hashlib.sha256()
        size = 0
        for path in paths:
            stat, file_sha256 = _hash_regular_file(path)
            encoded = path.name.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
            digest.update(stat.st_size.to_bytes(8, "big"))
            digest.update(bytes.fromhex(file_sha256))
            size += stat.st_size
        sha256 = digest.hexdigest()
    tensors = gguf_model.reader_tensors()
    qtypes = Counter(tensor.tensor_type.name for tensor in tensors)
    return GGUFArtifactIdentity(
        architecture=architecture,
        filename=filename or source_path.name,
        size=size,
        sha256=sha256,
        tensor_count=len(tensors),
        tensor_qtypes=tuple(sorted(qtypes.items())),
    )


def gguf_graph_package_identity(package_dir: Path) -> GGUFGraphPackageIdentity:
    """Hash every regular graph-package file with its relative path."""
    paths: list[Path] = []
    for root, directories, filenames in os.walk(package_dir, followlinks=False):
        root_path = Path(root)
        entries = [root_path / name for name in (*directories, *filenames)]
        if any(path.is_symlink() for path in entries):
            raise ValueError("GGUF graph package must not contain symlinks")
        paths.extend(root_path / name for name in filenames)
    paths.sort()
    if not paths:
        raise ValueError("GGUF graph package must contain regular files and no symlinks")
    digest = hashlib.sha256()
    names: list[str] = []
    for path in paths:
        name = path.relative_to(package_dir).as_posix()
        names.append(name)
        encoded = name.encode("utf-8")
        stat, file_sha256 = _hash_regular_file(path)
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(stat.st_size.to_bytes(8, "big"))
        digest.update(bytes.fromhex(file_sha256))
    return GGUFGraphPackageIdentity(files=tuple(names), sha256=digest.hexdigest())


def _hash_regular_file(path: Path) -> tuple[os.stat_result, str]:
    """Hash one non-symlink regular file through the descriptor being validated."""
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or path.is_symlink():
            raise ValueError(f"Expected a non-symlink regular file: {path}")
        digest = hashlib.sha256()
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
        after = os.fstat(descriptor)

        def identity(value: os.stat_result) -> tuple[int, int, int, int]:
            return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)

        if identity(before) != identity(after) or identity(after) != identity(path.stat()):
            raise ValueError(f"File changed while its immutable identity was computed: {path}")
        return after, digest.hexdigest()
    finally:
        os.close(descriptor)
