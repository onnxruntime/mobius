# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Machine-readable GGUF stored-quantization capability and evidence matrix."""

from __future__ import annotations

__all__ = [
    "CAPABILITY_MATRIX_PATH",
    "check_quantization_capability_matrix",
    "quantization_capability_matrix",
    "render_quantization_capability_matrix",
]

import json
from pathlib import Path
from typing import cast

from mobius.integrations.gguf._quant_registry import iter_quant_specs, quant_import_decision
from mobius.integrations.gguf._runtime_evidence import iter_runtime_evidence
from mobius.integrations.gguf._spec import (
    QuantImportRoute,
    RepackExactness,
    TensorRole,
)
from mobius.integrations.gguf._upstream import UPSTREAM_COMMIT

CAPABILITY_MATRIX_PATH = (
    Path(__file__).resolve().parents[4]
    / "testdata"
    / "evidence"
    / "gguf_quantization_capabilities.json"
)
_ARTIFACT_BUDGET_BYTES = 16 * 2**30
_RUNTIME_EVIDENCED_ROLES: dict[str, frozenset[TensorRole]] = {
    "Q8_0": frozenset(
        {
            TensorRole.PROJECTION,
            TensorRole.OUTPUT,
            TensorRole.EMBEDDING,
        }
    ),
}
_REJECTED_ARTIFACTS: tuple[dict[str, object], ...] = (
    {
        "repository": "unsloth/SmolLM2-135M-Instruct-GGUF",
        "revision": "9e6855bc4be717fca1ef21360a1db4b29d5c559a",
        "filename": "SmolLM2-135M-Instruct-Q4_K_M.gguf",
        "size": 105_454_144,
        "lfs_sha256": "ed5fa30c487b282ec156c29062f1222e5c20875a944ac98289dbd242e947f747",
        "tensor_qtypes": {
            "F32": 61,
            "Q4_K": 16,
            "Q5_0": 166,
            "Q6_K": 14,
            "Q8_0": 15,
        },
        "disposition": (
            "keep-quantized rejected because the mixed source requires lossy "
            "dequantize/requantize; explicit float import passed same-artifact full-logit "
            "prefill/decode and deterministic generation"
        ),
        "test": (
            "tests/gguf_small_model_runtime_integration_test.py::"
            "test_smollm_q4_k_m_fails_closed_or_matches_same_artifact_when_dequantized"
        ),
    },
)

_TRANSFORM_EVIDENCE: dict[str, tuple[str, ...]] = {
    "codes-scales-zero-points-and-block-tails": (
        "src/mobius/integrations/gguf/_repacker_test.py"
        "::TestRepackQ40::test_nibble_ordering_reordered",
        "src/mobius/integrations/gguf/_repacker_test.py"
        "::TestRepackQ41::test_zp_clamped_to_15",
        "src/mobius/integrations/gguf/_repacker_test.py"
        "::TestRepackQ80::test_int8_to_uint8_conversion",
        "src/mobius/integrations/gguf/_repacker_test.py"
        "::TestRepackQ4K::test_requantized_values_stay_within_half_scale",
        "src/mobius/integrations/gguf/_repacker_test.py"
        "::TestRepackQ6K::test_dequantization_matches_gguf_reference_exactly",
        "src/mobius/integrations/gguf/_repacker_test.py"
        "::TestRepackQ10::test_round_trip_dequantize",
    ),
    "qkv-split-and-row-permutation": (
        "src/mobius/integrations/gguf/_builder_test.py"
        "::test_phimoe_fused_qkv_is_split_without_loss",
        "src/mobius/integrations/gguf/_minicpm_test.py"
        "::test_quantized_minicpm_loader_permutes_packed_rows_scales_and_zero_points",
        "src/mobius/integrations/gguf/_kimi_k3_test.py"
        "::test_fused_kv_b_float_values_and_quantized_import",
    ),
    "transpose-and-concat": (
        "src/mobius/integrations/gguf/_tensor_processors_test.py"
        "::test_attn_weights_transposed",
        "src/mobius/integrations/gguf/_tensor_processors_test.py"
        "::test_granitehybrid_expert_gate_up_fusion_preserves_expert_order",
    ),
    "embedding-and-output-aliases": (
        "src/mobius/integrations/gguf/_builder_test.py"
        "::test_tied_quantized_embedding_is_shared_with_output_head",
        "src/mobius/integrations/gguf/_builder_test.py"
        "::test_quantized_untied_output_head_is_preserved",
    ),
    "expert-stacking-and-3d-experts": (
        "src/mobius/integrations/gguf/_builder_test.py"
        "::test_fused_experts_are_split_without_tensor_loss",
        "src/mobius/integrations/gguf/_block_quantized_moe_builder_test.py"
        "::test_e2e_uniform_native_moe_fuses_through_builder",
        "src/mobius/integrations/_block_quant_test.py"
        "::test_stack_is_byte_exact_and_recoverable",
    ),
}


def _transform(route: QuantImportRoute) -> str | None:
    if route is QuantImportRoute.NATIVE_BYTES:
        return "mobius.integrations.gguf._repacker.preserve_native_blocks"
    if route is QuantImportRoute.AFFINE_REPACK:
        return "mobius.integrations.gguf._repacker.repack_gguf_tensor"
    if route is QuantImportRoute.DEQUANTIZE_REQUANTIZE:
        return (
            "mobius.integrations.gguf._reader.GGUFModel.dequantize_raw_tensor -> "
            "mobius.integrations.gguf._repacker.repack_dequantized_tensor"
        )
    if route is QuantImportRoute.DEQUANTIZE_FLOAT:
        return "mobius.integrations.gguf._reader.GGUFModel.dequantize_raw_tensor"
    return None


def _operator_abi(route: QuantImportRoute, role: TensorRole) -> str | None:
    if route is QuantImportRoute.NATIVE_BYTES:
        return "pkg.nxrt::BlockQuantizedMatMul/v1"
    if route in {QuantImportRoute.AFFINE_REPACK, QuantImportRoute.DEQUANTIZE_REQUANTIZE}:
        if role is TensorRole.EMBEDDING:
            return "com.microsoft::GatherBlockQuantized/v1"
        return "com.microsoft::MatMulNBits/v1"
    if route is QuantImportRoute.DEQUANTIZE_FLOAT:
        return "standard ONNX float initializer"
    return None


def _route_record(
    qtype_name: str,
    ggml_type_id: int,
    role: TensorRole,
    runtime_evidence_ids: tuple[str, ...],
) -> dict[str, object]:
    route, exactness, reason = quant_import_decision(ggml_type_id, role)
    preserves_source_values = route is QuantImportRoute.NATIVE_BYTES or (
        route is QuantImportRoute.AFFINE_REPACK and exactness is RepackExactness.EXACT
    )
    return {
        "route": route.value,
        "exactness": None if exactness is None else exactness.value,
        "preserves_source_values": preserves_source_values,
        "keep_quantized_supported": preserves_source_values,
        "transform": _transform(route),
        "operator_abi": _operator_abi(route, role),
        "runtime_support": (
            "supported"
            if role in _RUNTIME_EVIDENCED_ROLES.get(qtype_name, frozenset())
            else "deferred"
        ),
        "runtime_evidence_ids": (
            list(runtime_evidence_ids)
            if role in _RUNTIME_EVIDENCED_ROLES.get(qtype_name, frozenset())
            else []
        ),
        "reason": reason,
    }


def _artifact_records() -> tuple[list[dict[str, object]], int]:
    grouped: dict[tuple[str, str, str], dict[str, object]] = {}
    for evidence in iter_runtime_evidence():
        key = (evidence.repository, evidence.revision, evidence.filename)
        if key not in grouped:
            grouped[key] = {
                "repository": evidence.repository,
                "revision": evidence.revision,
                "filename": evidence.filename,
                "size": evidence.size,
                "lfs_sha256": evidence.lfs_sha256,
                "tensor_qtypes": dict(evidence.tensor_qtypes),
                "evidence_ids": [],
                "runtime_results": [],
            }
        evidence_ids = grouped[key]["evidence_ids"]
        assert isinstance(evidence_ids, list)
        evidence_ids.append(evidence.evidence_id)
        runtime_results = grouped[key]["runtime_results"]
        assert isinstance(runtime_results, list)
        runtime_results.append(
            {
                "evidence_id": evidence.evidence_id,
                "onnxruntime_version": evidence.onnxruntime_version,
                "execution_provider": evidence.execution_provider,
                "downstream_runtime": evidence.runtime,
                "downstream_runtime_version": evidence.runtime_version,
                "result": evidence.result,
                "parity_kind": evidence.parity_kind,
                "stateful_semantics": evidence.stateful_semantics,
                "parity_test": evidence.parity_test,
                "deterministic_test": evidence.deterministic_test,
                "graph_sha256": evidence.graph_sha256,
                "runtime_package_sha256": evidence.runtime_package_sha256,
            }
        )
    artifacts = [grouped[key] for key in sorted(grouped)]
    return artifacts, sum(cast(int, record["size"]) for record in artifacts)


def quantization_capability_matrix() -> dict[str, object]:
    """Return the complete JSON-serializable stored-qtype capability matrix."""
    artifacts, selected_bytes = _artifact_records()
    rejected_artifact_bytes = sum(
        cast(int, record["size"]) for record in _REJECTED_ARTIFACTS
    )
    selected_bytes += rejected_artifact_bytes
    if selected_bytes > _ARTIFACT_BUDGET_BYTES:
        raise ValueError(
            f"Pinned GGUF evidence totals {selected_bytes} bytes, exceeding "
            f"the {_ARTIFACT_BUDGET_BYTES}-byte policy"
        )

    qtypes: list[dict[str, object]] = []
    for spec in iter_quant_specs():
        if not spec.is_quantized_storage:
            continue
        native = (
            None
            if spec.native_preserve is None
            else {
                "format": spec.native_preserve.format,
                "block_elements": spec.native_preserve.elements,
                "block_bytes": spec.native_preserve.bytes,
                "operator_abi": "pkg.nxrt::BlockQuantizedMatMul/v1",
                "execution_evidenced": spec.runtime_evidence_ids != (),
            }
        )
        affine = (
            None
            if spec.affine_repack is None
            else {
                "bits": spec.affine_repack.bits,
                "block_size": spec.affine_repack.block_size,
                "zero_points": (
                    "omitted" if spec.affine_repack.omit_zero_points else "explicit"
                ),
                "exactness": (
                    None if spec.repack_exactness is None else spec.repack_exactness.value
                ),
            }
        )
        qtypes.append(
            {
                "name": spec.name,
                "ggml_type_id": spec.ggml_type_id,
                "storage_role": spec.role.value,
                "parse_support": "supported" if spec.readable else "rejected",
                "block_elements": spec.block_elements,
                "block_bytes": spec.block_bytes,
                "exact_dequantization": spec.dequantize.value,
                "native_block_abi": native,
                "affine_target": affine,
                "runtime_support": spec.runtime.value,
                "runtime_reason": spec.runtime_reason,
                "runtime_evidence_ids": list(spec.runtime_evidence_ids),
                "roles": {
                    role.value: _route_record(
                        spec.name,
                        spec.ggml_type_id,
                        role,
                        spec.runtime_evidence_ids,
                    )
                    for role in TensorRole
                },
            }
        )
    return {
        "schema_version": 1,
        "llama_cpp_commit": UPSTREAM_COMMIT,
        "policy": {
            "preserved_definition": (
                "Only byte-identical native blocks or exact affine repacks preserve "
                "source represented values. Dequantize/requantize is never preserved."
            ),
            "runtime_definition": (
                "Runtime support requires immutable same-artifact full-logit parity plus "
                "deterministic prefill/decode/replay/rollback/reorder evidence."
            ),
            "max_selected_artifact_bytes": _ARTIFACT_BUDGET_BYTES,
            "selected_artifact_bytes": selected_bytes,
        },
        "operator_abis": {
            "affine_projection": "com.microsoft::MatMulNBits/v1",
            "affine_embedding": "com.microsoft::GatherBlockQuantized/v1",
            "native_projection": "pkg.nxrt::BlockQuantizedMatMul/v1",
        },
        "transform_evidence": {
            key: list(value) for key, value in sorted(_TRANSFORM_EVIDENCE.items())
        },
        "selected_artifacts": artifacts,
        "rejected_artifacts": list(_REJECTED_ARTIFACTS),
        "qtypes": qtypes,
    }


def render_quantization_capability_matrix() -> str:
    """Render the capability matrix in canonical JSON form."""
    return json.dumps(quantization_capability_matrix(), indent=2, sort_keys=True) + "\n"


def check_quantization_capability_matrix(
    path: Path = CAPABILITY_MATRIX_PATH,
) -> bool:
    """Return whether the committed matrix exactly matches live registries."""
    return path.read_text(encoding="utf-8") == render_quantization_capability_matrix()
