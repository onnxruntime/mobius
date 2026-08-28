# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Structured real-artifact evidence required for GGUF runtime support."""

from __future__ import annotations

__all__ = [
    "FINAL_RUNTIME_PACKAGE_SCHEMA",
    "GGUFArtifactIdentity",
    "GGUFGraphPackageIdentity",
    "GGUFRuntimeEvidence",
    "RuntimeEvidenceUnavailableError",
    "find_matching_runtime_evidence",
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
import json
import os
import stat
from collections import Counter
from pathlib import Path
from types import MappingProxyType
from typing import Any

from mobius.integrations.gguf._reader import _descriptor_identity

FINAL_RUNTIME_PACKAGE_SCHEMA = "mobius.gguf-runtime-package.v2"
LEGACY_RUNTIME_PACKAGE_SCHEMA = "legacy-without-component-export-report"


class RuntimeEvidenceUnavailableError(ValueError):
    """An otherwise-valid package route without exact downstream runtime evidence."""


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
    source_fidelity: bool
    storage_quantized: bool
    target_storage_format: str
    compute_mode: str
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
    limitations: str | None = None
    runtime_package_schema: str = LEGACY_RUNTIME_PACKAGE_SCHEMA

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
            self.target_storage_format,
            self.compute_mode,
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
            self.runtime_package_schema,
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
        if self.limitations is not None and not self.limitations.strip():
            raise ValueError("GGUF runtime evidence limitations must be non-empty")
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
    '"9f917f4a59c907325a069735b9a5d07177f3d665b5e812c10b626ecf1c94708e",'
    '"execution_provider":"cpu","model_type":"llama","module_type":"llama",'
    '"preserve_quantization":false,"registry_import":{"config_key_map":null,'
    '"config_postprocessor":null,"llama_qk_permute":true,"offset_norm":false,'
    '"required_metadata":[],"rope_interleave":false,"tensor_processor":"llama",'
    '"v_head_reorder":false,"vlm_builder":"generic_projector"},"route_schema":1,'
    '"static_cache":false,'
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
    source_fidelity=True,
    storage_quantized=False,
    target_storage_format="float",
    compute_mode="float operators",
    graph_files=("model.onnx", "model.onnx.data", "quantization_report.json"),
    graph_sha256="4b608b099fb17471f342c925c20173f297abd0f8456c9e96a11b1d044272d1ad",
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
        "quantization_report.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ),
    runtime_package_sha256="57038e28e83a2b4251e334b6098f59b4344c7e526746e29d9fa42eeaebfcbddc",
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
        "quantization_report.json",
        "runtime_compatibility.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ),
    runtime_package_sha256="15d0218a4b326648d514a98c4f073251c674312b1faa83bec9022cc91daa0a53",
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
    import_route='{"architecture":"qwen2","config_sha256":"d2b0c54fdd96c5122a344240557f6b35aafa65a0e8f8981158683c089559b29a","execution_provider":"cpu","model_type":"qwen2","module_type":"qwen2","preserve_quantization":true,"registry_import":{"config_key_map":null,"config_postprocessor":null,"llama_qk_permute":false,"offset_norm":false,"required_metadata":[],"rope_interleave":false,"tensor_processor":null,"v_head_reorder":false,"vlm_builder":null},"route_schema":1,"static_cache":false,"task":{"class":"builtins.str","state":"text-generation"},"tensor_map_recipe":["llama"]}',
    source_fidelity=True,
    storage_quantized=True,
    target_storage_format="INT8 affine block-32",
    compute_mode="runtime-dependent native custom op or inline standard-ONNX fallback",
    graph_files=("model.onnx", "model.onnx.data", "quantization_report.json"),
    graph_sha256="240e5e374803c94efdb17eee39c09b0d3e9aed10b6d8b4e1c92e39918ea2155e",
    runtime_package_files=(
        "genai_config.json",
        "gguf_tokenizer_manifest.json",
        "model.onnx",
        "model.onnx.data",
        "quantization_report.json",
        "runtime_compatibility.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ),
    runtime_package_sha256="5029bbfcdd8d1ae1d2b0ed9587cc288a68cb38bb7f92b9e10f9bf64a436b1762",
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
    import_route='{"architecture":"lfm2","config_sha256":"e7746a202f3679ac05311fbc7a25414e4c106008fc3bd257001e97a8d7cab575","execution_provider":"cpu","model_type":"lfm2","module_type":"lfm2","preserve_quantization":false,"registry_import":{"config_key_map":null,"config_postprocessor":null,"llama_qk_permute":false,"offset_norm":false,"required_metadata":["attention.head_count_kv","attention.layer_norm_rms_epsilon","shortconv.l_cache"],"rope_interleave":false,"tensor_processor":null,"v_head_reorder":false,"vlm_builder":null},"route_schema":1,"static_cache":false,"task":{"class":"builtins.str","state":"hybrid-text-generation"},"tensor_map_recipe":["lfm2"]}',
    source_fidelity=True,
    storage_quantized=False,
    target_storage_format="float",
    compute_mode="float operators",
    graph_files=("model.onnx", "model.onnx.data", "quantization_report.json"),
    graph_sha256="27e4ebb4c0c8b6c01ee57fa7825f34c5ddadc8ca5dc0c75d989e4507d4dcdfdb",
    runtime_package_files=(
        "chat_template.jinja",
        "genai_config.json",
        "gguf_tokenizer_manifest.json",
        "model.onnx",
        "model.onnx.data",
        "quantization_report.json",
        "runtime_compatibility.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ),
    runtime_package_sha256="ef3816d4f93c7061fd4653248629f0439d6ad1c623bdd0f27d31fe0349cb3505",
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

_QWEN35MOE_087B_Q2_K_ORT_GENAI = GGUFRuntimeEvidence(
    evidence_id="qwen3.5-moe-0.87b-q2-k-ort-genai-0.15.2",
    architecture="qwen35moe",
    repository="Flexan/kshitijthakkar-qwen3.5-moe-0.87B-d0.8B-GGUF",
    revision="a9b8adbec2cc87479c772dac1944f313b4036c26",
    filename="qwen3.5-moe-0.87B-d0.8B.Q2_K.gguf",
    size=626_599_552,
    lfs_sha256="e8a84df1a50ce65cf80c2b55bba8c6e80f913679fdf9e9439f2c3b52ef3145d5",
    config_repository="kshitijthakkar/qwen3.5-moe-0.87B-d0.8B",
    config_revision="e5b5b3d7c3cc5593196902fd3c23964e891a6ea6",
    tokenizer_repository="kshitijthakkar/qwen3.5-moe-0.87B-d0.8B",
    tokenizer_revision="e5b5b3d7c3cc5593196902fd3c23964e891a6ea6",
    tokenizer_metadata_sha256="45302b58b2086a666a874652d0e9e1d5b4b26e786ffbaf9362a4f902eba0b10d",
    tokenizer_assets=(
        (
            "chat_template.jinja",
            7_755,
            "273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80",
        ),
        (
            "tokenizer.json",
            12_807_982,
            "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42",
        ),
        (
            "tokenizer_config.json",
            16_709,
            "49e2b6e395f959f077f1e992b338919c0d4a9732fc6e613995e06557f843500c",
        ),
    ),
    tensor_count=441,
    tensor_qtypes=(
        ("F16", 48),
        ("F32", 181),
        ("Q2_K", 193),
        ("Q5_K", 6),
        ("Q6_K", 1),
        ("Q8_0", 12),
    ),
    import_route='{"architecture":"qwen35moe","config_sha256":"48019eea50654171acb6b87074d3c3a212bdf156aa6045c221067732bf25af91","execution_provider":"cpu","model_type":"qwen3_5_moe","module_type":"qwen3_5_moe","preserve_quantization":false,"registry_import":{"config_key_map":null,"config_postprocessor":null,"llama_qk_permute":false,"offset_norm":true,"required_metadata":["attention.layer_norm_rms_epsilon","expert_count","expert_used_count","rope.dimension_sections","ssm.conv_kernel","ssm.group_count","ssm.inner_size","ssm.state_size","ssm.time_step_rank"],"rope_interleave":false,"tensor_processor":null,"v_head_reorder":true,"vlm_builder":null},"route_schema":1,"static_cache":false,"task":{"class":"builtins.str","state":"hybrid-text-generation"},"tensor_map_recipe":["llama","moe_extras","qwen35_hybrid_extras"]}',
    source_fidelity=False,
    storage_quantized=False,
    target_storage_format="float",
    compute_mode="float operators",
    graph_files=("model.onnx", "model.onnx.data", "quantization_report.json"),
    graph_sha256="8c1aa1075cee03ffd5ce5bbd283ee88b2466e6c1d645f0a41e542230951d6f09",
    runtime_package_files=(
        "chat_template.jinja",
        "genai_config.json",
        "gguf_tokenizer_manifest.json",
        "model.onnx",
        "model.onnx.data",
        "quantization_report.json",
        "runtime_compatibility.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ),
    runtime_package_sha256="bd69cf72acffd393a706bb012c23990acfdaf3a43e25ad919ef1148afac42a15",
    parity_test="test_promoted_gguf_full_runtime_evidence[qwen3.5-moe-0.87b-q2-k]",
    parity_kind="full-logit",
    deterministic_test="test_promoted_gguf_full_runtime_evidence[qwen3.5-moe-0.87b-q2-k]",
    stateful_semantics=(
        "hybrid convolution, recurrent, and KV state prefill, replay, rollback, reorder, "
        "and 20 cache-threaded decode steps"
    ),
    execution_provider="CPUExecutionProvider",
    onnxruntime_version="1.29.0",
    runtime="ort-genai",
    runtime_version="0.15.2",
    limitations=(
        "Explicit-float correctness route only: source quantization is dequantized, dense "
        "MoE execution is opt-in, and the selected publisher marks this reduced checkpoint "
        "as low quality. Same-value full-logit comparison uses atol=0.35 because small "
        "backend differences can cross a routed-expert boundary; all greedy tokens match."
    ),
)

_LOW_COST_GRAPH_FILES = ("model.onnx", "model.onnx.data", "quantization_report.json")
_LOW_COST_RUNTIME_PACKAGE_FILES = (
    "genai_config.json",
    "gguf_tokenizer_manifest.json",
    "model.onnx",
    "model.onnx.data",
    "quantization_report.json",
    "runtime_compatibility.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
)
_LOW_COST_STATEFUL_SEMANTICS = (
    "dynamic KV cache prefill, full-sequence replay, rollback, reorder, and 20 decode steps"
)

_APERTUS_15B_BF16_ORT_GENAI = GGUFRuntimeEvidence(
    evidence_id="apertus-v1.1-1.5b-instruct-bf16-ort-genai-0.15.2",
    architecture="apertus",
    repository="MrMeOrYou/Apertus-v1.1-1.5B-Instruct-GGUF",
    revision="88c75ad49566d3c2157d03709bf772262c3241ed",
    filename="Apertus-v1.1-1.5B-Instruct-BF16.gguf",
    size=3_028_052_608,
    lfs_sha256="f9ec154d0ec29dad1f6465b458b7f27bd25ad7b9a3899233ae98ca6d358501c2",
    config_repository="swiss-ai/Apertus-v1.1-1.5B-Instruct",
    config_revision="9e9d01154446a645d30f04174cf1515a38058be7",
    tokenizer_repository="swiss-ai/Apertus-v1.1-1.5B-Instruct",
    tokenizer_revision="9e9d01154446a645d30f04174cf1515a38058be7",
    tokenizer_metadata_sha256=(
        "3097bd9f22efd32db9045c4705d978539dc8adeeae763324062a5e1a73fc24a5"
    ),
    tokenizer_assets=(
        (
            "chat_template.jinja",
            5_250,
            "4afab8361a4bd0c2994404e0b0851dbaad461a06e56bdfdb635aae3977473c19",
        ),
        (
            "special_tokens_map.json",
            560,
            "9f69883bd70fc5d8b55822799837a216d3ac4fb565e05256a0d4f9850404bbc5",
        ),
        (
            "tokenizer.json",
            17_078_368,
            "be12f4375d655cc740864e3a9041bcddd8477942f209d9e7f27f6c8767162638",
        ),
        (
            "tokenizer_config.json",
            177_274,
            "77b14a0664585c26065f07d7a4c852a4615c83348d9378e23def01957bbd3f57",
        ),
    ),
    tensor_count=163,
    tensor_qtypes=(("BF16", 98), ("F32", 65)),
    import_route='{"architecture":"apertus","config_sha256":"2a9660c48656c0980e76f1b374262bf8383256603e8b4221aa99bcfa11c510b0","execution_provider":"cpu","model_type":"apertus","module_type":"apertus","preserve_quantization":false,"registry_import":{"config_key_map":null,"config_postprocessor":"apertus","llama_qk_permute":false,"offset_norm":false,"required_metadata":["attention.layer_norm_rms_epsilon"],"rope_interleave":false,"tensor_processor":null,"v_head_reorder":false,"vlm_builder":null},"route_schema":1,"static_cache":false,"task":{"class":"builtins.str","state":"text-generation"},"tensor_map_recipe":["llama","apertus_extras"]}',
    source_fidelity=True,
    storage_quantized=False,
    target_storage_format="float",
    compute_mode="float operators",
    graph_files=_LOW_COST_GRAPH_FILES,
    graph_sha256="4bb91bade19d41559cb524e28692453851fe67274cc58109787ee968df3e0fe5",
    runtime_package_files=(
        "chat_template.jinja",
        "export_report.json",
        *_LOW_COST_RUNTIME_PACKAGE_FILES,
    ),
    runtime_package_sha256="37c8a587eaeda7dc31bd1a4a6b7611a8aae5e91b7d7bbeb1a106e3a7a2441f08",
    parity_test=("test_promoted_gguf_full_runtime_evidence[apertus-v1.1-1.5b-instruct-bf16]"),
    parity_kind="full-logit",
    deterministic_test=(
        "test_promoted_gguf_full_runtime_evidence[apertus-v1.1-1.5b-instruct-bf16]"
    ),
    stateful_semantics=_LOW_COST_STATEFUL_SEMANTICS,
    execution_provider="CPUExecutionProvider",
    onnxruntime_version="1.29.0",
    runtime="ort-genai",
    runtime_version="0.15.2",
    runtime_package_schema=FINAL_RUNTIME_PACKAGE_SCHEMA,
)

_GPT2_Q2_K_ORT_GENAI = GGUFRuntimeEvidence(
    evidence_id="gpt2-q2-k-ort-genai-0.15.2",
    architecture="gpt2",
    repository="tensorblock/gpt2-GGUF",
    revision="5b01870b15c4b2e43695d7f3f3bfb5b26106f23b",
    filename="gpt2-Q2_K.gguf",
    size=81_196_544,
    lfs_sha256="4234545f917ec1df10dab4d926796a83422b68e9010d85a4c111b8b541f32892",
    config_repository="openai-community/gpt2",
    config_revision="607a30d783dfa663caf39e06633721c8d4cfcd7e",
    tokenizer_repository="Xenova/gpt2",
    tokenizer_revision="bf2c7f02e0b826c60d03af341171bde20893da66",
    tokenizer_metadata_sha256="b2417176025f8500d864004b0bf93b1403dc3c52238f6628f82fb0e3c498977e",
    tokenizer_assets=(
        (
            "special_tokens_map.json",
            99,
            "6f50ab5a5a509a1c309d6171f339b196a900dc9c99ad0408ff23bb615fdae7ad",
        ),
        (
            "tokenizer.json",
            2_107_653,
            "cda20b8ca044949aa07ac4078420c80d1a57139d5f9f33700e46fb2d891e7c66",
        ),
        (
            "tokenizer_config.json",
            234,
            "551e26ec611d8d0c8edc3ef72e518a38418cb71f40de1347dd486a595e1557d7",
        ),
    ),
    tensor_count=149,
    tensor_qtypes=(("F32", 99), ("Q2_K", 25), ("Q3_K", 24), ("Q6_K", 1)),
    import_route='{"architecture":"gpt2","config_sha256":"1e0363dbc5f3427d873aeb46c27d5713ae0668f0e7cf695b5dde00bdd4d390a6","execution_provider":"cpu","model_type":"gpt2","module_type":"gpt2","preserve_quantization":false,"registry_import":{"config_key_map":null,"config_postprocessor":null,"llama_qk_permute":false,"offset_norm":false,"required_metadata":[],"rope_interleave":false,"tensor_processor":"gpt2","v_head_reorder":false,"vlm_builder":null},"route_schema":1,"static_cache":false,"task":{"class":"builtins.str","state":"text-generation"},"tensor_map_recipe":["gpt2"]}',
    source_fidelity=False,
    storage_quantized=False,
    target_storage_format="float",
    compute_mode="float operators",
    graph_files=_LOW_COST_GRAPH_FILES,
    graph_sha256="ae121ccf53782ffdc530cf96329e5eacabfc0baf5fcdbfc71925e6edc4ad1423",
    runtime_package_files=_LOW_COST_RUNTIME_PACKAGE_FILES,
    runtime_package_sha256="7bfca0bcb5d5aacf39738306c15d0c27db7f7c5810fd6259609fd0d2370becde",
    parity_test="test_promoted_gguf_full_runtime_evidence[gpt2-q2-k]",
    parity_kind="full-logit",
    deterministic_test="test_promoted_gguf_full_runtime_evidence[gpt2-q2-k]",
    stateful_semantics=_LOW_COST_STATEFUL_SEMANTICS,
    execution_provider="CPUExecutionProvider",
    onnxruntime_version="1.29.0",
    runtime="ort-genai",
    runtime_version="0.15.2",
    limitations="Explicit-float correctness route; source GGUF blocks are dequantized.",
)

_GPTNEOX_Q2_K_ORT_GENAI = GGUFRuntimeEvidence(
    evidence_id="pythia-70m-q2-k-ort-genai-0.15.2",
    architecture="gptneox",
    repository="mradermacher/pythia-70m-GGUF",
    revision="52d6f045404c9f93418df2a0144d20c9de34316b",
    filename="pythia-70m.Q2_K.gguf",
    size=38_508_192,
    lfs_sha256="8e331c8c8016bed8ff1863b78fafe51e86b2364f32c2d7f3e201687e081cf7f7",
    config_repository="EleutherAI/pythia-70m",
    config_revision="a39f36b100fe8a5377810d56c3f4789b9c53ac42",
    tokenizer_repository="EleutherAI/pythia-70m",
    tokenizer_revision="a39f36b100fe8a5377810d56c3f4789b9c53ac42",
    tokenizer_metadata_sha256="d5c722f646ff6462ac217da5e3514d1fa8b4a7b33aedced3daa8e7f00cc74f78",
    tokenizer_assets=(
        (
            "special_tokens_map.json",
            99,
            "6f50ab5a5a509a1c309d6171f339b196a900dc9c99ad0408ff23bb615fdae7ad",
        ),
        (
            "tokenizer.json",
            2_113_710,
            "c24618a1b3e6a38167beff1c72cffd126c3a66254347304b50547d12c5f25624",
        ),
        (
            "tokenizer_config.json",
            396,
            "70e38394e494931c6f773ba41e19460dd4436526b852207367f04341b4066d3f",
        ),
    ),
    tensor_count=76,
    tensor_qtypes=(("F32", 50), ("Q2_K", 13), ("Q3_K", 12), ("Q6_K", 1)),
    import_route='{"architecture":"gptneox","config_sha256":"19607e074d570fe92a572204c482e7fde81dd3f7d6d29758fd36dc4a89ba6153","execution_provider":"default","model_type":"gpt_neox","module_type":"gguf_legacy","preserve_quantization":false,"registry_import":{"config_key_map":null,"config_postprocessor":"exact_legacy_gguf","llama_qk_permute":false,"offset_norm":false,"required_metadata":["attention.layer_norm_epsilon","use_parallel_residual"],"rope_interleave":false,"tensor_processor":null,"v_head_reorder":false,"vlm_builder":null},"route_schema":1,"static_cache":false,"task":{"class":"builtins.str","state":"text-generation"},"tensor_map_recipe":["legacy_layernorm"]}',
    source_fidelity=False,
    storage_quantized=False,
    target_storage_format="float",
    compute_mode="float operators",
    graph_files=_LOW_COST_GRAPH_FILES,
    graph_sha256="115a5b47ae04263465e8063661381c183c0b77c870ce3c3dfc339387fbbf9c2f",
    runtime_package_files=_LOW_COST_RUNTIME_PACKAGE_FILES,
    runtime_package_sha256="798cc7d612214b032e6354d75c282d75119b3649d603a85d49643c815c24ca77",
    parity_test="test_promoted_gguf_full_runtime_evidence[pythia-70m-q2-k]",
    parity_kind="full-logit",
    deterministic_test="test_promoted_gguf_full_runtime_evidence[pythia-70m-q2-k]",
    stateful_semantics=_LOW_COST_STATEFUL_SEMANTICS,
    execution_provider="CPUExecutionProvider",
    onnxruntime_version="1.29.0",
    runtime="ort-genai",
    runtime_version="0.15.2",
    limitations=(
        "Explicit-float correctness route on the portable graph; source GGUF blocks are "
        "dequantized and the pinned tokenizer vocabulary is extended only with deterministic "
        "padding IDs present in the GGUF."
    ),
)

_MPT_Q2_K_ORT_GENAI = GGUFRuntimeEvidence(
    evidence_id="tiny-mpt-q2-k-ort-genai-0.15.2",
    architecture="mpt",
    repository="tensorblock/tiny-mpt-random-remote-code-GGUF",
    revision="c151eb3f349485ee8ae72841d7d90b9121d5baa2",
    filename="tiny-mpt-random-remote-code-Q2_K.gguf",
    size=8_734_304,
    lfs_sha256="5627dcb0ff18f6f7200f83c0aed2056a6a7c86b5f2d865833e1b5f00b00e4daa",
    config_repository="echarlaix/tiny-mpt-random-remote-code",
    config_revision="85e64794e74a6fb2e71e7055c7e0188ccdd32905",
    tokenizer_repository="echarlaix/tiny-mpt-random-remote-code",
    tokenizer_revision="85e64794e74a6fb2e71e7055c7e0188ccdd32905",
    tokenizer_metadata_sha256="f15522d34f33354bc96d36a73ad4619925240328342d84a156855251c20d43af",
    tokenizer_assets=(
        (
            "special_tokens_map.json",
            99,
            "6f50ab5a5a509a1c309d6171f339b196a900dc9c99ad0408ff23bb615fdae7ad",
        ),
        (
            "tokenizer.json",
            2_113_738,
            "3cf430678137c8491ca82fb7092ee49e44ad38857fffe1e4a4a5ed860139a5b8",
        ),
        (
            "tokenizer_config.json",
            237,
            "7671fbb5b3d610e6e11d4f5fc78d3a7716e8846112ac7e0f72124caedf887570",
        ),
    ),
    tensor_count=8,
    tensor_qtypes=(("F32", 3), ("IQ4_NL", 3), ("Q3_K", 1), ("Q8_0", 1)),
    import_route='{"architecture":"mpt","config_sha256":"7239b094edec93d3b29e2ffc53b08be450cb7c8536dd93120c4db618d47cd4f3","execution_provider":"default","model_type":"mpt","module_type":"gguf_legacy","preserve_quantization":false,"registry_import":{"config_key_map":null,"config_postprocessor":"exact_legacy_gguf","llama_qk_permute":false,"offset_norm":false,"required_metadata":["attention.layer_norm_epsilon","attention.max_alibi_bias"],"rope_interleave":false,"tensor_processor":null,"v_head_reorder":false,"vlm_builder":null},"route_schema":1,"static_cache":false,"task":{"class":"builtins.str","state":"text-generation"},"tensor_map_recipe":["legacy_layernorm"]}',
    source_fidelity=False,
    storage_quantized=False,
    target_storage_format="float",
    compute_mode="float operators",
    graph_files=_LOW_COST_GRAPH_FILES,
    graph_sha256="5daf516f886110f257105a4122cba9fbdb920fab92bedf057a4f67899e513c8a",
    runtime_package_files=_LOW_COST_RUNTIME_PACKAGE_FILES,
    runtime_package_sha256="20dd912c318d8115af5a5e6e92672624c00120455346ff5c9294e9dc1a1cc234",
    parity_test="test_promoted_gguf_full_runtime_evidence[tiny-mpt-q2-k]",
    parity_kind="full-logit",
    deterministic_test="test_promoted_gguf_full_runtime_evidence[tiny-mpt-q2-k]",
    stateful_semantics=_LOW_COST_STATEFUL_SEMANTICS,
    execution_provider="CPUExecutionProvider",
    onnxruntime_version="1.29.0",
    runtime="ort-genai",
    runtime_version="0.15.2",
    limitations="Explicit-float correctness route; source GGUF blocks are dequantized.",
)

_OLMO_Q2_K_ORT_GENAI = GGUFRuntimeEvidence(
    evidence_id="tiny-olmo-q2-k-ort-genai-0.15.2",
    architecture="olmo",
    repository="tensorblock/tiny-random-olmo-GGUF",
    revision="d0ee9498d082d6dc3e730b9765e6e87a9bb5d995",
    filename="tiny-random-olmo-Q2_K.gguf",
    size=33_860_576,
    lfs_sha256="be1c5a22ac0e75cd5874467ffd80bcd2c8500609d3bf7ccdb5b269373e4d6da4",
    config_repository="hyper-accel/tiny-random-olmo",
    config_revision="88675ef0caa5bd10ece810c0f2a79faa7724f536",
    tokenizer_repository="hyper-accel/tiny-random-olmo",
    tokenizer_revision="88675ef0caa5bd10ece810c0f2a79faa7724f536",
    tokenizer_metadata_sha256="25efe0090ffe5a6deb777743917fe572548a0f23357f74d8bd9e969cc911fd73",
    tokenizer_assets=(
        (
            "special_tokens_map.json",
            293,
            "a6188c1e366f8ed715e60ff39c46a8c500fc33508e2affeb23e8c547c5853193",
        ),
        (
            "tokenizer.json",
            2_115_417,
            "a094266ac6c4982efba277bc251349a5a6d6ad37efb39a2a90f53d8be2a40a40",
        ),
        (
            "tokenizer_config.json",
            5_372,
            "78a839c7851f14f9fb30e664c2b46166dc0628f2900679e5ec160656f702edff",
        ),
    ),
    tensor_count=16,
    tensor_qtypes=(("IQ4_NL", 2), ("Q2_K", 9), ("Q3_K", 4), ("Q6_K", 1)),
    import_route='{"architecture":"olmo","config_sha256":"c02542bb6b7c076f641ff0724409471deefb0782eab37dff076e80746a59652f","execution_provider":"cpu","model_type":"olmo","module_type":"olmo","preserve_quantization":false,"registry_import":{"config_key_map":null,"config_postprocessor":"olmo","llama_qk_permute":true,"offset_norm":false,"required_metadata":["attention.layer_norm_epsilon"],"rope_interleave":false,"tensor_processor":"llama","v_head_reorder":false,"vlm_builder":null},"route_schema":1,"static_cache":false,"task":{"class":"builtins.str","state":"text-generation"},"tensor_map_recipe":["olmo"]}',
    source_fidelity=False,
    storage_quantized=False,
    target_storage_format="float",
    compute_mode="float operators",
    graph_files=_LOW_COST_GRAPH_FILES,
    graph_sha256="eb67fb20ae4257af04db8b1554c56ae6b3f66340f2b6ff97dd55cbd14c7bce5b",
    runtime_package_files=_LOW_COST_RUNTIME_PACKAGE_FILES,
    runtime_package_sha256="4f2817a0af1b58922f3b82d455844664948dbc67f0ffe6af576baedf2eac3835",
    parity_test="test_promoted_gguf_full_runtime_evidence[tiny-olmo-q2-k]",
    parity_kind="full-logit",
    deterministic_test="test_promoted_gguf_full_runtime_evidence[tiny-olmo-q2-k]",
    stateful_semantics=_LOW_COST_STATEFUL_SEMANTICS,
    execution_provider="CPUExecutionProvider",
    onnxruntime_version="1.29.0",
    runtime="ort-genai",
    runtime_version="0.15.2",
    limitations="Explicit-float correctness route; source GGUF blocks are dequantized.",
)

_STARCODER_Q2_K_ORT_GENAI = GGUFRuntimeEvidence(
    evidence_id="tiny-starcoder-q2-k-ort-genai-0.15.2",
    architecture="starcoder",
    repository="RichardErkhov/bigcode_-_tiny_starcoder_py-gguf",
    revision="fa6f9fdbc134d86d78a3ff9ce08ab14ba33d4718",
    filename="tiny_starcoder_py.Q2_K.gguf",
    size=103_899_456,
    lfs_sha256="aa8c2170bb9172447baba14309916cfc0d901dbffaf10f1448f4f631e10c1f41",
    config_repository="bigcode/tiny_starcoder_py",
    config_revision="8547527bef0bc927268c1653cce6948c5c242dd1",
    tokenizer_repository="bigcode/tiny_starcoder_py",
    tokenizer_revision="8547527bef0bc927268c1653cce6948c5c242dd1",
    tokenizer_metadata_sha256="23379a715b3983ce0f1559645984431bc039c16eaf6e062b7e62bceac6fa64cd",
    tokenizer_assets=(
        (
            "special_tokens_map.json",
            532,
            "0823292e24ea07b89317e9ede9d08da2a1b6c014290c06908a7ad04f1efd6719",
        ),
        (
            "tokenizer.json",
            2_057_395,
            "42b5a37ba11199f024f2b8873e1ecba98da33166e16f700bf7cb2304b0a5583f",
        ),
        (
            "tokenizer_config.json",
            677,
            "95684c52ad9a970dbbb17576ee2237cb62902c1eff6804c7c91a4d6219a4a6d7",
        ),
    ),
    tensor_count=244,
    tensor_qtypes=(("F32", 163), ("Q2_K", 40), ("Q3_K", 40), ("Q6_K", 1)),
    import_route='{"architecture":"starcoder","config_sha256":"45f95ad88af3f5378bdfacd633f1717012e1f3699bbe880af4f6d29fd79f2d4f","execution_provider":"cpu","model_type":"gpt_bigcode","module_type":"gpt_bigcode","preserve_quantization":false,"registry_import":{"config_key_map":null,"config_postprocessor":"conventional_legacy","llama_qk_permute":false,"offset_norm":false,"required_metadata":["attention.layer_norm_epsilon"],"rope_interleave":false,"tensor_processor":null,"v_head_reorder":false,"vlm_builder":null},"route_schema":1,"static_cache":false,"task":{"class":"builtins.str","state":"text-generation"},"tensor_map_recipe":["starcoder"]}',
    source_fidelity=False,
    storage_quantized=False,
    target_storage_format="float",
    compute_mode="float operators",
    graph_files=_LOW_COST_GRAPH_FILES,
    graph_sha256="ab4edee4c001e3df8f4bce477447bea1c0360221163e01dd1cb03e89737107bd",
    runtime_package_files=_LOW_COST_RUNTIME_PACKAGE_FILES,
    runtime_package_sha256="ecbbae525e12d250c3eaa782b9eec14dba2c09fb9dbde29a8331e741e7676171",
    parity_test="test_promoted_gguf_full_runtime_evidence[tiny-starcoder-q2-k]",
    parity_kind="full-logit",
    deterministic_test="test_promoted_gguf_full_runtime_evidence[tiny-starcoder-q2-k]",
    stateful_semantics=_LOW_COST_STATEFUL_SEMANTICS,
    execution_provider="CPUExecutionProvider",
    onnxruntime_version="1.29.0",
    runtime="ort-genai",
    runtime_version="0.15.2",
    limitations="Explicit-float correctness route; source GGUF blocks are dequantized.",
)

_STARCODER2_Q2_K_ORT_GENAI = GGUFRuntimeEvidence(
    evidence_id="tiny-starcoder2-q2-k-ort-genai-0.15.2",
    architecture="starcoder2",
    repository="tensorblock/tiny-random-starcoder2-GGUF",
    revision="82c9eb61d1af6ea00dff834f1ff0620144b333e8",
    filename="tiny-random-starcoder2-Q2_K.gguf",
    size=68_039_904,
    lfs_sha256="ab0a4b4e79c906520808db065a00a317c8b097e2176638b34088975eada6e0ed",
    config_repository="hyper-accel/tiny-random-starcoder2",
    config_revision="193576733055d2108dc9906d0da6e0806ad9be57",
    tokenizer_repository="hyper-accel/tiny-random-starcoder2",
    tokenizer_revision="193576733055d2108dc9906d0da6e0806ad9be57",
    tokenizer_metadata_sha256="38608a4848dc8535113cff0312858853a0c8e29a5cdc145f258e4a9d84b35113",
    tokenizer_assets=(
        (
            "special_tokens_map.json",
            1_300,
            "0fc9ac706a35e6337b19d484bc8f866b6a0ee7ad2509b03f7305dc0838c82d2c",
        ),
        (
            "tokenizer.json",
            2_060_947,
            "17fa145258b20c18287f1e3bd804e074cc13333f11984a2f5a2f11c5110437aa",
        ),
        (
            "tokenizer_config.json",
            7_877,
            "8149d8e6b21275ad2cc346885ad92c0e9b5aa3e28a78bb5d39b7febc7e52545d",
        ),
    ),
    tensor_count=36,
    tensor_qtypes=(
        ("F32", 22),
        ("Q2_K", 7),
        ("Q3_K", 4),
        ("Q4_K", 2),
        ("Q6_K", 1),
    ),
    import_route='{"architecture":"starcoder2","config_sha256":"25eaa4f5baa4e4725930298aaa921804d1e6f202737a3b10ec32070c4acfcb71","execution_provider":"cpu","model_type":"starcoder2","module_type":"starcoder2","preserve_quantization":false,"registry_import":{"config_key_map":null,"config_postprocessor":"starcoder2","llama_qk_permute":false,"offset_norm":false,"required_metadata":[],"rope_interleave":false,"tensor_processor":null,"v_head_reorder":false,"vlm_builder":null},"route_schema":1,"static_cache":false,"task":{"class":"builtins.str","state":"text-generation"},"tensor_map_recipe":["llama"]}',
    source_fidelity=False,
    storage_quantized=False,
    target_storage_format="float",
    compute_mode="float operators",
    graph_files=_LOW_COST_GRAPH_FILES,
    graph_sha256="34c3e58ac9d95b5d1097393782d41a942ac9b6188e6d57adff21212030897a23",
    runtime_package_files=_LOW_COST_RUNTIME_PACKAGE_FILES,
    runtime_package_sha256="4a9ab2edf80d8cf778fccd56d306cb56dfda5bb190068faf04c78ac5d180f564",
    parity_test="test_promoted_gguf_full_runtime_evidence[tiny-starcoder2-q2-k]",
    parity_kind="full-logit",
    deterministic_test="test_promoted_gguf_full_runtime_evidence[tiny-starcoder2-q2-k]",
    stateful_semantics=_LOW_COST_STATEFUL_SEMANTICS,
    execution_provider="CPUExecutionProvider",
    onnxruntime_version="1.29.0",
    runtime="ort-genai",
    runtime_version="0.15.2",
    limitations=(
        "Explicit-float correctness route; source GGUF blocks are dequantized. The tiny "
        "source config declares BOS/EOS ID 50256 outside its 49152-token model vocabulary."
    ),
)

_RUNTIME_EVIDENCE: MappingProxyType[str, GGUFRuntimeEvidence] = MappingProxyType(
    {
        record.evidence_id: record
        for record in (
            _APERTUS_15B_BF16_ORT_GENAI,
            _LFM2_350M_F16_ORT_GENAI,
            _QWEN25_Q8_ORT_GENAI,
            _QWEN35MOE_087B_Q2_K_ORT_GENAI,
            _SMOLLM_F16_ONNX_RUNTIME,
            _SMOLLM_F16_ORT_GENAI,
            _GPT2_Q2_K_ORT_GENAI,
            _GPTNEOX_Q2_K_ORT_GENAI,
            _MPT_Q2_K_ORT_GENAI,
            _OLMO_Q2_K_ORT_GENAI,
            _STARCODER_Q2_K_ORT_GENAI,
            _STARCODER2_Q2_K_ORT_GENAI,
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
        try:
            import_route = _strict_json_object(evidence.import_route)
        except (TypeError, ValueError):
            invalid.append(evidence_id)
            continue
        if (
            qtypes.get(qtype, 0) <= 0
            or evidence.parity_kind != "full-logit"
            or import_route.get("preserve_quantization") is not True
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


def _strict_json_object(payload: str) -> dict[str, object]:
    """Parse a JSON object while rejecting duplicate keys at every depth."""

    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key {key!r}")
            result[key] = value
        return result

    parsed = json.loads(payload, object_pairs_hook=reject_duplicates)
    if not isinstance(parsed, dict):
        raise TypeError("GGUF import route must be a JSON object")
    return parsed


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


def find_matching_runtime_evidence(
    evidence_ids: tuple[str, ...],
    *,
    architecture: str,
    runtime: str,
    source_path: Path,
    gguf_model: Any,
    built_identity: GGUFArtifactIdentity,
    import_route: str,
    runtime_version: str | None,
    tokenizer_repository: str | None,
    tokenizer_revision: str | None,
) -> GGUFRuntimeEvidence | None:
    """Return exact runtime evidence, while treating an absent match as unvalidated."""
    if evidence_ids:
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
    if (
        not evidence_ids
        or runtime_version is None
        or tokenizer_repository is None
        or tokenizer_revision is None
    ):
        return None
    identity = built_identity
    candidates = [
        _RUNTIME_EVIDENCE[evidence_id]
        for evidence_id in evidence_ids
        if _RUNTIME_EVIDENCE[evidence_id].runtime_package_schema
        == FINAL_RUNTIME_PACKAGE_SCHEMA
        and _RUNTIME_EVIDENCE[evidence_id].runtime == runtime
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
    if len(candidates) > 1:
        raise RuntimeError(
            "GGUF runtime evidence contains duplicate package identities for "
            f"architecture={architecture!r}, runtime={runtime!r} {runtime_version!r}."
        )
    return candidates[0] if candidates else None


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
    tokenizer_repository: str | None,
    tokenizer_revision: str | None,
) -> GGUFRuntimeEvidence:
    """Return exact evidence for the package source, route, and requested runtime."""
    match = find_matching_runtime_evidence(
        evidence_ids,
        architecture=architecture,
        runtime=runtime,
        source_path=source_path,
        gguf_model=gguf_model,
        built_identity=built_identity,
        import_route=import_route,
        runtime_version=runtime_version,
        tokenizer_repository=tokenizer_repository,
        tokenizer_revision=tokenizer_revision,
    )
    if match is None:
        raise RuntimeEvidenceUnavailableError(
            f"No unique GGUF runtime evidence matches architecture={architecture!r}, "
            f"runtime={runtime!r} {runtime_version!r}, artifact={built_identity!r}, "
            f"import_route={import_route!r}."
        )
    return match


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
        open_descriptor = getattr(gguf_model, "open_source_descriptor", None)
        if callable(open_descriptor):
            with open_descriptor() as descriptor:
                stat, sha256 = _hash_regular_descriptor(
                    descriptor,
                    path=source_path,
                    expected_identity=gguf_model.source_identity,
                )
        else:
            stat, sha256 = _hash_regular_file(source_path)
        size = stat.st_size
    else:
        paths = tuple(Path(path) for path in shard_paths)
        identity_paths = tuple(
            Path(path) for path in getattr(gguf_model, "identity_paths", paths)
        )
        source_identities = getattr(gguf_model, "source_identities", None)
        if not paths:
            raise ValueError("A GGUF shard set must contain at least one source file.")
        if len(identity_paths) != len(paths):
            raise ValueError("GGUF shard identity paths must match the shard set length.")
        if source_identities is None or len(source_identities) != len(paths):
            raise ValueError(
                "GGUF shard source identities must be captured for the complete shard set."
            )
        digest = hashlib.sha256()
        size = 0
        open_descriptor = getattr(gguf_model, "open_source_descriptor", None)
        for index, (path, identity_path, source_identity) in enumerate(
            zip(paths, identity_paths, source_identities)
        ):
            if callable(open_descriptor):
                with open_descriptor(index) as descriptor:
                    stat, file_sha256 = _hash_regular_descriptor(
                        descriptor,
                        path=identity_path,
                        expected_identity=source_identity,
                    )
            else:
                stat, file_sha256 = _hash_regular_file(
                    identity_path,
                    expected_identity=source_identity,
                )
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


def gguf_graph_package_identity(
    package_dir: Path,
    *,
    files: tuple[str, ...] | None = None,
) -> GGUFGraphPackageIdentity:
    """Hash every regular graph-package file, or one exact relative file set."""
    if package_dir.is_symlink() or not package_dir.is_dir():
        raise ValueError("GGUF graph package root must be a real directory")
    paths: list[Path]
    if files is None:
        paths = []
        for root, directories, filenames in os.walk(package_dir, followlinks=False):
            root_path = Path(root)
            entries = [root_path / name for name in (*directories, *filenames)]
            if any(path.is_symlink() for path in entries):
                raise ValueError("GGUF graph package must not contain symlinks")
            paths.extend(root_path / name for name in filenames)
        paths.sort()
    else:
        if files != tuple(sorted(files)) or len(files) != len(set(files)):
            raise ValueError("GGUF graph package file selection must be sorted and unique")
        relative_paths = [Path(name) for name in files]
        if any(path.is_absolute() or ".." in path.parts for path in relative_paths):
            raise ValueError("GGUF graph package file selection must stay inside the package")
        paths = [package_dir / path for path in relative_paths]
        for path in paths:
            current = package_dir
            for part in path.relative_to(package_dir).parts:
                current /= part
                if current.is_symlink():
                    raise ValueError("GGUF graph package selection must not traverse symlinks")
        if any(path.is_symlink() or not path.is_file() for path in paths):
            raise ValueError("GGUF graph package selection must contain regular files")
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


def _hash_regular_file(
    path: Path,
    *,
    expected_identity: tuple[int, int, int, int, int] | None = None,
) -> tuple[os.stat_result, str]:
    """Hash one non-symlink regular file through the descriptor being validated."""
    flags = (
        os.O_RDONLY
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    descriptor = os.open(path, flags)
    try:
        stat_result, sha256 = _hash_regular_descriptor(
            descriptor,
            path=path,
            expected_identity=expected_identity,
        )
        if path.is_symlink():
            raise ValueError(f"Expected a non-symlink regular file: {path}")
        path_descriptor = os.open(path, flags)
        try:
            if _descriptor_identity(descriptor) != _descriptor_identity(path_descriptor):
                raise ValueError(f"File changed while its identity was computed: {path}")
        finally:
            os.close(path_descriptor)
        return stat_result, sha256
    finally:
        os.close(descriptor)


def _hash_regular_descriptor(
    descriptor: int,
    *,
    path: Path,
    expected_identity: tuple[int, int, int, int, int] | None,
) -> tuple[os.stat_result, str]:
    """Hash the exact descriptor retained by a GGUF reader."""
    os.lseek(descriptor, 0, os.SEEK_SET)
    before = os.fstat(descriptor)
    before_identity = _descriptor_identity(descriptor)
    if not stat.S_ISREG(before.st_mode):
        raise ValueError(f"Expected a regular GGUF source file: {path}")
    if expected_identity is not None and before_identity != expected_identity:
        raise ValueError(f"File no longer matches the opened GGUF source identity: {path}")
    digest = hashlib.sha256()
    while chunk := os.read(descriptor, 1024 * 1024):
        digest.update(chunk)
    after = os.fstat(descriptor)
    if before_identity != _descriptor_identity(descriptor):
        raise ValueError(f"File changed while its immutable identity was computed: {path}")
    return after, digest.hexdigest()
