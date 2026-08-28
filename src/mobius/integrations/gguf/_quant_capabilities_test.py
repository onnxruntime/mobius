# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Closure tests for the machine-readable GGUF quantization capability matrix."""

from __future__ import annotations

import json
from pathlib import Path

from mobius.integrations.gguf._quant_capabilities import (
    CAPABILITY_MATRIX_PATH,
    check_quantization_capability_matrix,
    quantization_capability_matrix,
)


def _qtypes(matrix: dict[str, object]) -> dict[str, dict[str, object]]:
    records = matrix["qtypes"]
    assert isinstance(records, list)
    return {str(record["name"]): record for record in records}


def test_committed_quantization_capability_matrix_is_current() -> None:
    assert check_quantization_capability_matrix()
    parsed = json.loads(CAPABILITY_MATRIX_PATH.read_text(encoding="utf-8"))
    assert parsed == quantization_capability_matrix()


def test_runtime_blocker_candidate_is_metadata_only_and_not_budgeted_as_support() -> None:
    matrix = quantization_capability_matrix()
    records = matrix["runtime_blocker_evidence"]
    assert isinstance(records, list)
    assert len(records) == 1
    record = records[0]
    policy = matrix["policy"]
    selected_artifacts = matrix["selected_artifacts"]
    assert isinstance(policy, dict)
    assert isinstance(selected_artifacts, list)
    assert record["evidence_id"] == "nemotron-h-moe-30b-iq2-xxs-runtime-blocker"
    assert record["result"] == "blocked"
    assert record["size"] > policy["max_selected_artifact_bytes"]
    assert record["size"] not in {artifact["size"] for artifact in selected_artifacts}
    assert record["tokenizer"]["revision"] == "bf77c3174f68ad409e1c2aa60daeb46e32d1c606"
    assert record["graph"]["pre_optimization_node_count"] == 40_167
    assert record["graph"]["node_count"] == 37_142
    assert record["graph"]["initializer_count"] == 6_255
    assert record["graph"]["matmul_count"] == 6_028
    assert record["graph"]["state_slots"]["mamba2.ssm_state"] == 23
    assert record["runtime_schema_issue"].endswith("/issues/605")
    assert record["onnxruntime_version"] == "1.29.0"
    assert record["execution_provider"] == "CPUExecutionProvider"
    assert "full-logit parity" in record["withheld_checks"]
    assert "separate router_probs/router_weights" in record["blockers"][1]
    assert (
        "derives recurrent_state names while this export uses ssm_state"
        in (record["blockers"][2])
    )


def test_mtp_runtime_status_is_machine_readable_and_never_selected() -> None:
    matrix = quantization_capability_matrix()
    records = matrix["mtp_runtime_evidence"]
    selected_artifacts = matrix["selected_artifacts"]
    assert isinstance(records, list)
    assert isinstance(selected_artifacts, list)
    assert {record["architecture"] for record in records} == {"hy_v3", "qwen35"}

    selected_hashes = {artifact["lfs_sha256"] for artifact in selected_artifacts}
    for record in records:
        assert record["result"] == "runtime_unvalidated"
        assert record["graph_sha256"] is None
        assert record["runtime_package_sha256"] is None
        assert record["source_fidelity"] is False
        assert record["storage_fidelity"] is False
        assert record["tokenizer"]["status"] == "separately-deferred"
        assert record["cache_topology"]["target_namespace"] == "target"
        assert record["cache_topology"]["mtp_namespace"] == "mtp"
        assert record["runtime"]["version"] == "0.15.2"
        assert record["runtime"]["onnxruntime_version"] == "1.29.0"
        assert record["runtime"]["execution_provider"] == "CPUExecutionProvider"
        assert "two_model_draft_target_binding" in record["runtime"]["missing_capabilities"]
        for layout in record["layouts"]:
            for artifact in layout["artifacts"]:
                assert artifact["lfs_sha256"] not in selected_hashes
    qwen = next(record for record in records if record["architecture"] == "qwen35")
    assert qwen["synthetic_coordinator"]["acceptance_statistics"] == {
        "accepted": 1,
        "proposal_steps": 51,
        "rejected": 50,
        "rollbacks": 50,
    }
    hy_v3 = next(record for record in records if record["architecture"] == "hy_v3")
    assert hy_v3["synthetic_coordinator"] is None


def test_every_stored_qtype_and_tensor_role_is_explicit() -> None:
    matrix = quantization_capability_matrix()
    qtypes = _qtypes(matrix)
    assert len(qtypes) == 25
    expected_roles = {
        "projection",
        "projection (affine-only graph)",
        "output",
        "embedding",
        "expert-major",
        "non-MatMul",
    }
    for record in qtypes.values():
        assert record["parse_support"] == "supported"
        roles = record["roles"]
        assert isinstance(roles, dict)
        assert set(roles) == expected_roles
        for route in roles.values():
            assert isinstance(route, dict)
            assert {
                "route",
                "exactness",
                "source_fidelity",
                "target_storage",
                "target_storage_supported",
                "keep_quantized_supported",
                "transform",
                "operator_abi",
                "runtime_support",
                "runtime_evidence_ids",
                "reason",
            } == set(route)


def test_lossy_affine_routes_keep_target_storage_without_claiming_fidelity() -> None:
    qtypes = _qtypes(quantization_capability_matrix())
    for name in ("Q4_K", "Q6_K"):
        record = qtypes[name]
        roles = record["roles"]
        assert isinstance(roles, dict)
        projection = roles["projection"]
        assert isinstance(projection, dict)
        assert projection["route"] == "affine repack"
        assert projection["exactness"] == "lossy"
        assert projection["source_fidelity"] is False
        assert projection["target_storage"] == "affine integer blocks"
        assert projection["target_storage_supported"] is True
        assert projection["keep_quantized_supported"] is True
        assert f"_repack_{name.lower()}" in str(projection["transform"])


def test_only_q8_has_qtype_level_runtime_execution_evidence() -> None:
    qtypes = _qtypes(quantization_capability_matrix())
    supported = {
        name: record["runtime_evidence_ids"]
        for name, record in qtypes.items()
        if record["runtime_support"] == "supported"
    }
    assert supported == {
        "Q8_0": ["qwen2.5-0.5b-instruct-q8-ort-genai-0.15.2"],
    }
    q8_roles = qtypes["Q8_0"]["roles"]
    assert isinstance(q8_roles, dict)
    assert {
        role
        for role, record in q8_roles.items()
        if isinstance(record, dict) and record["runtime_support"] == "supported"
    } == {"projection", "output", "embedding"}
    for name, record in qtypes.items():
        native = record["native_block_abi"]
        if native is not None:
            assert isinstance(native, dict)
            assert native["execution_evidenced"] is False, name
            assert record["runtime_support"] == "deferred"


def test_transform_evidence_names_real_network_free_tests() -> None:
    evidence = quantization_capability_matrix()["transform_evidence"]
    assert isinstance(evidence, dict)
    assert set(evidence) == {
        "codes-scales-zero-points-and-block-tails",
        "embedding-and-output-aliases",
        "expert-stacking-and-3d-experts",
        "qkv-split-and-row-permutation",
        "transpose-and-concat",
    }
    for references in evidence.values():
        assert isinstance(references, list)
        for reference in references:
            parts = str(reference).split("::")
            assert len(parts) in {2, 3}
            path_text, test_name = parts[0], parts[-1]
            assert test_name.startswith("test_")
            source = Path(path_text)
            assert source.is_file()
            text = source.read_text(encoding="utf-8")
            if len(parts) == 3:
                assert f"class {parts[1]}" in text
            assert f"def {test_name}(" in text


def test_selected_real_artifacts_stay_within_global_budget() -> None:
    matrix = quantization_capability_matrix()
    policy = matrix["policy"]
    artifacts = matrix["selected_artifacts"]
    lossy = matrix["lossy_target_artifacts"]
    assert isinstance(policy, dict)
    assert isinstance(artifacts, list)
    assert isinstance(lossy, list)
    assert len(artifacts) == 11
    assert len(lossy) == 1
    selected = sum(int(record["size"]) for record in [*artifacts, *lossy])
    assert selected == 5_752_423_904
    assert selected == policy["selected_artifact_bytes"]
    assert selected <= policy["max_selected_artifact_bytes"]
    assert lossy[0]["lfs_sha256"] == (
        "ed5fa30c487b282ec156c29062f1222e5c20875a944ac98289dbd242e947f747"
    )
    assert "source_fidelity=false" in lossy[0]["disposition"]
    assert "runtime support deferred" in lossy[0]["runtime_disposition"]

    qwen = next(record for record in artifacts if record["filename"].endswith("Q2_K.gguf"))
    [runtime] = qwen["runtime_results"]
    assert runtime["source_fidelity"] is False
    assert runtime["storage_quantized"] is False
    assert runtime["target_storage_format"] == "float"
    assert runtime["compute_mode"] == "float operators"
    assert '"preserve_quantization":false' in runtime["import_route"]
    assert "Explicit-float correctness route only" in runtime["limitations"]
