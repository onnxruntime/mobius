# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for exact GGUF runtime-evidence binding."""

from __future__ import annotations

import hashlib
import os
from collections import Counter
from dataclasses import replace
from types import MappingProxyType, SimpleNamespace

import onnx_ir as ir
import pytest

from mobius._builder import build_from_module
from mobius._configs import NemotronHConfig
from mobius.integrations.gguf import _runtime_evidence
from mobius.integrations.gguf._reader import _descriptor_identity
from mobius.integrations.gguf._runtime_blocker_evidence import (
    iter_runtime_blocker_evidence,
    runtime_blocker_evidence,
)
from mobius.integrations.gguf._runtime_evidence import (
    GGUFRuntimeEvidence,
    gguf_artifact_identity,
    gguf_graph_package_identity,
    matching_runtime_evidence,
    validate_quant_runtime_evidence_ids,
    validate_runtime_evidence_ids,
)
from mobius.models.nemotron_h import NemotronHCausalLMModel
from mobius.tasks import HybridCausalLMTask


def _file_identity(path) -> tuple[int, int, int, int, int]:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_BINARY", 0))
    try:
        return _descriptor_identity(descriptor)
    finally:
        os.close(descriptor)


def _pinned_nemotron_h_config() -> NemotronHConfig:
    pattern = "MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME"
    layer_types = {
        "M": "mamba2",
        "E": "moe",
        "*": "full_attention",
    }
    return NemotronHConfig(
        model_type="nemotron_h",
        vocab_size=131_072,
        hidden_size=2_688,
        intermediate_size=1_856,
        num_hidden_layers=len(pattern),
        num_attention_heads=32,
        num_key_value_heads=2,
        head_dim=128,
        rms_norm_eps=1e-5,
        layer_types=[layer_types[layer] for layer in pattern],
        hidden_act="relu2",
        mamba_n_heads=64,
        mamba_d_head=64,
        mamba_d_state=128,
        mamba_n_groups=8,
        mamba_d_conv=4,
        mamba_expand=2,
        mamba_conv_bias=True,
        mamba_proj_bias=False,
        mamba_time_step_min=0.001,
        num_local_experts=128,
        num_experts_per_tok=6,
        moe_intermediate_size=1_856,
        moe_latent_size=None,
        shared_expert_intermediate_size=3_712,
        norm_topk_prob=True,
        routed_scaling_factor=2.5,
        dtype=ir.DataType.BFLOAT16,
    )


def _record(payload: bytes) -> GGUFRuntimeEvidence:
    return GGUFRuntimeEvidence(
        evidence_id="llama-onnx-genai",
        architecture="llama",
        repository="owner/model",
        revision="a" * 40,
        filename="model.gguf",
        size=len(payload),
        lfs_sha256=hashlib.sha256(payload).hexdigest(),
        config_repository="owner/config",
        config_revision="b" * 40,
        tokenizer_repository="owner/tokenizer",
        tokenizer_revision="c" * 40,
        tokenizer_metadata_sha256="f" * 64,
        tokenizer_assets=(("tokenizer.json", 2, hashlib.sha256(b"{}").hexdigest()),),
        tensor_count=2,
        tensor_qtypes=(("F32", 1), ("Q4_K", 1)),
        import_route='{"route_schema":1}',
        source_fidelity=False,
        storage_quantized=False,
        target_storage_format="float",
        compute_mode="float operators",
        graph_files=("model.onnx",),
        graph_sha256="d" * 64,
        runtime_package_files=("model.onnx", "tokenizer.json"),
        runtime_package_sha256="e" * 64,
        parity_test="test_full_logit_parity",
        parity_kind="full-logit",
        deterministic_test="test_cached_generation",
        stateful_semantics="dynamic KV cache prefill, replay, rollback, reorder, and decode",
        execution_provider="CPUExecutionProvider",
        onnxruntime_version="1.29.0",
        runtime="onnx-genai",
        runtime_version="1.0.0",
    )


def _model():
    tensors = [
        SimpleNamespace(tensor_type=SimpleNamespace(name="Q4_K")),
        SimpleNamespace(tensor_type=SimpleNamespace(name="F32")),
    ]
    return SimpleNamespace(
        _reader=SimpleNamespace(tensors=tensors),
        reader_tensors=lambda: tensors,
    )


def test_runtime_evidence_rejects_non_hex_tokenizer_metadata_digest() -> None:
    with pytest.raises(ValueError, match="immutable 40-hex revisions and LFS SHA-256"):
        replace(_record(b"pinned-gguf"), tokenizer_metadata_sha256="g" * 64)


def test_nemotron_h_runtime_blocker_is_pinned_without_support_claim() -> None:
    records = iter_runtime_blocker_evidence()
    assert len(records) == 1
    record = records[0]
    assert runtime_blocker_evidence(record.evidence_id) is record
    assert record.architecture == "nemotron_h_moe"
    assert record.result == "blocked"
    assert record.size == 18_010_755_296
    assert record.size > 16 * 1024**3
    assert record.tensor_count == 401
    assert dict(record.tensor_qtypes)["IQ2_XXS"] == 23
    assert record.state_slots == (
        ("attention.key", 6),
        ("attention.value", 6),
        ("mamba2.conv_state", 23),
        ("mamba2.ssm_state", 23),
    )
    assert record.explicit_float16_bytes == record.logical_parameter_count * 2
    assert record.explicit_float32_bytes == record.logical_parameter_count * 4
    assert record.tokenizer_metadata_sha256 == (
        "6089bcaf08b3fe0d49379ca7e85bd3c93e8705bac6130636425c159212971225"
    )
    assert "tokenizer.json" in {name for name, _, _ in record.tokenizer_assets}
    assert record.runtime_version == "0.15.2"
    assert record.runtime_schema_issue.endswith("/issues/605")
    assert _runtime_evidence.runtime_evidence(record.evidence_id) is None
    assert "full-logit parity" in record.withheld_checks
    assert record.graph_node_count == 37_142
    assert record.pre_optimization_graph_node_count == 40_167
    assert "separate router_probs/router_weights" in record.blockers[1]
    assert "not fused-op blockers" in record.blockers[1]
    assert "has no latent projection" in record.blockers[1]
    assert "discovers sparse/nonconsecutive KV" in record.blockers[2]
    assert (
        "derives recurrent_state names while this export uses ssm_state" in record.blockers[2]
    )
    assert "does not beam-reorder recurrent state" in record.blockers[2]
    assert "rejects nonzero recurrent-state rewind" in record.blockers[2]
    assert all("cannot describe" not in blocker for blocker in record.blockers)


def test_nemotron_h_runtime_blocker_graph_census_matches_pinned_config() -> None:
    evidence = iter_runtime_blocker_evidence()[0]
    config = _pinned_nemotron_h_config()

    raw_graph = (
        HybridCausalLMTask()
        .build(
            NemotronHCausalLMModel(config),
            config,
        )["model"]
        .graph
    )
    assert len(raw_graph) == evidence.pre_optimization_graph_node_count

    production_graph = build_from_module(
        NemotronHCausalLMModel(config),
        config,
        task="hybrid-text-generation",
        execution_provider="cpu",
    )["model"].graph
    op_counts = Counter(node.op_type for node in production_graph)

    assert len(production_graph) == evidence.graph_node_count
    assert len(production_graph.initializers) == evidence.graph_initializer_count
    assert op_counts["MatMul"] == evidence.graph_matmul_count
    assert len(production_graph.inputs) == 60
    assert len(production_graph.outputs) == 59


def test_matching_evidence_binds_arch_runtime_source_qtypes_and_route(
    tmp_path, monkeypatch
) -> None:
    payload = b"pinned-gguf"
    source = tmp_path / "model.gguf"
    source.write_bytes(payload)
    record = _record(payload)
    monkeypatch.setattr(
        _runtime_evidence,
        "_RUNTIME_EVIDENCE",
        MappingProxyType({record.evidence_id: record}),
    )

    assert (
        matching_runtime_evidence(
            (record.evidence_id,),
            architecture="llama",
            runtime="onnx-genai",
            source_path=source,
            gguf_model=_model(),
            built_identity=gguf_artifact_identity(source, _model(), architecture="llama"),
            import_route=record.import_route,
            runtime_version="1.0.0",
            tokenizer_repository=record.tokenizer_repository,
            tokenizer_revision=record.tokenizer_revision,
        )
        is record
    )

    with pytest.raises(ValueError, match="No unique GGUF runtime evidence"):
        matching_runtime_evidence(
            (record.evidence_id,),
            architecture="llama",
            runtime="ort-genai",
            source_path=source,
            gguf_model=_model(),
            built_identity=gguf_artifact_identity(source, _model(), architecture="llama"),
            import_route=record.import_route,
            runtime_version="1.0.0",
            tokenizer_repository=record.tokenizer_repository,
            tokenizer_revision=record.tokenizer_revision,
        )
    with pytest.raises(ValueError, match="No unique GGUF runtime evidence"):
        matching_runtime_evidence(
            (record.evidence_id,),
            architecture="llama",
            runtime="onnx-genai",
            source_path=source,
            gguf_model=_model(),
            built_identity=gguf_artifact_identity(source, _model(), architecture="llama"),
            import_route=record.import_route,
            runtime_version="1.0.0",
            tokenizer_repository="attacker/replacement",
            tokenizer_revision=record.tokenizer_revision,
        )
    with pytest.raises(ValueError, match="No unique GGUF runtime evidence"):
        matching_runtime_evidence(
            (record.evidence_id,),
            architecture="llama",
            runtime="onnx-genai",
            source_path=source,
            gguf_model=_model(),
            built_identity=gguf_artifact_identity(source, _model(), architecture="llama"),
            import_route='{"route_schema":2}',
            runtime_version="1.0.0",
            tokenizer_repository=record.tokenizer_repository,
            tokenizer_revision=record.tokenizer_revision,
        )


def test_matching_evidence_rejects_source_replaced_after_build(tmp_path, monkeypatch) -> None:
    payload = b"pinned-gguf"
    source = tmp_path / "model.gguf"
    source.write_bytes(payload)
    record = _record(payload)
    built_identity = gguf_artifact_identity(source, _model(), architecture="llama")
    source.write_bytes(b"changed-gguf")
    monkeypatch.setattr(
        _runtime_evidence,
        "_RUNTIME_EVIDENCE",
        MappingProxyType({record.evidence_id: record}),
    )

    with pytest.raises(ValueError, match="no longer matches"):
        matching_runtime_evidence(
            (record.evidence_id,),
            architecture="llama",
            runtime="onnx-genai",
            source_path=source,
            gguf_model=_model(),
            built_identity=built_identity,
            import_route=record.import_route,
            runtime_version="1.0.0",
            tokenizer_repository=record.tokenizer_repository,
            tokenizer_revision=record.tokenizer_revision,
        )


def test_sharded_artifact_identity_frames_every_shard_and_tensor(tmp_path) -> None:
    first = tmp_path / "model-00001-of-00002.gguf"
    second = tmp_path / "model-00002-of-00002.gguf"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    tensors = [
        SimpleNamespace(tensor_type=SimpleNamespace(name="F32")),
        SimpleNamespace(tensor_type=SimpleNamespace(name="Q4_K")),
    ]
    model = SimpleNamespace(
        shard_paths=[first, second],
        source_identities=[
            _file_identity(first),
            _file_identity(second),
        ],
        reader_tensors=lambda: tensors,
    )

    identity = gguf_artifact_identity(
        second,
        model,
        architecture="llama",
        filename=first.name,
    )

    assert identity.filename == first.name
    assert identity.size == len(b"firstsecond")
    assert identity.tensor_count == 2
    assert identity.tensor_qtypes == (("F32", 1), ("Q4_K", 1))

    second.write_bytes(b"change")
    with pytest.raises(ValueError, match="no longer matches the opened GGUF source identity"):
        gguf_artifact_identity(
            first,
            model,
            architecture="llama",
            filename=first.name,
        )


def test_sharded_artifact_identity_hashes_regular_aliases_for_snapshot_links(
    tmp_path,
) -> None:
    blobs = tmp_path / "blobs"
    snapshot = tmp_path / "snapshots" / ("a" * 40)
    blobs.mkdir()
    snapshot.mkdir(parents=True)
    first_blob = blobs / "first-blob"
    second_blob = blobs / "second-blob"
    first_blob.write_bytes(b"first")
    second_blob.write_bytes(b"second")
    first = snapshot / "model-00001-of-00002.gguf"
    second = snapshot / "model-00002-of-00002.gguf"
    first.symlink_to(first_blob)
    second.symlink_to(second_blob)
    tensors = [SimpleNamespace(tensor_type=SimpleNamespace(name="F32"))]
    model = SimpleNamespace(
        shard_paths=[first, second],
        identity_paths=[first_blob, second_blob],
        source_identities=[
            _file_identity(first),
            _file_identity(second),
        ],
        reader_tensors=lambda: tensors,
    )

    identity = gguf_artifact_identity(
        second,
        model,
        architecture="llama",
        filename=first.name,
    )

    assert identity.filename == first.name
    assert identity.size == len(b"firstsecond")


def test_evidence_id_cannot_cross_architectures(monkeypatch) -> None:
    record = _record(b"pinned-gguf")
    monkeypatch.setattr(
        _runtime_evidence,
        "_RUNTIME_EVIDENCE",
        MappingProxyType({record.evidence_id: record}),
    )
    with pytest.raises(ValueError, match="do not belong to 'qwen2'"):
        validate_runtime_evidence_ids("qwen2", (record.evidence_id,))


def test_quantized_runtime_evidence_requires_preserved_full_stateful_route(
    monkeypatch,
) -> None:
    record = replace(
        _record(b"pinned-gguf"),
        import_route='{"preserve_quantization":true}',
    )
    monkeypatch.setattr(
        _runtime_evidence,
        "_RUNTIME_EVIDENCE",
        MappingProxyType({record.evidence_id: record}),
    )
    validate_quant_runtime_evidence_ids("Q4_K", (record.evidence_id,))

    lossy = replace(record, import_route='{"preserve_quantization":false}')
    monkeypatch.setattr(
        _runtime_evidence,
        "_RUNTIME_EVIDENCE",
        MappingProxyType({lossy.evidence_id: lossy}),
    )
    with pytest.raises(ValueError, match="does not prove preserved Q4_K"):
        validate_quant_runtime_evidence_ids("Q4_K", (lossy.evidence_id,))


@pytest.mark.parametrize(
    "import_route",
    [
        '{"preserve_quantization":"true"}',
        '{"preserve_quantization":1}',
        '{"preserve_quantization":true,"preserve_quantization":true}',
        '{"preserve_quantization":true',
        "[]",
    ],
)
def test_quantized_runtime_evidence_rejects_noncanonical_route_json(
    import_route: str,
    monkeypatch,
) -> None:
    record = replace(_record(b"pinned-gguf"), import_route=import_route)
    monkeypatch.setattr(
        _runtime_evidence,
        "_RUNTIME_EVIDENCE",
        MappingProxyType({record.evidence_id: record}),
    )
    with pytest.raises(ValueError, match="does not prove preserved Q4_K"):
        validate_quant_runtime_evidence_ids("Q4_K", (record.evidence_id,))


def test_graph_package_identity_frames_files_and_rejects_symlinks(tmp_path) -> None:
    package = tmp_path / "package"
    package.mkdir()
    (package / "a.onnx").write_bytes(b"ab")
    (package / "b.data").write_bytes(b"c")
    first = gguf_graph_package_identity(package)

    (package / "a.onnx").write_bytes(b"a")
    (package / "b.data").write_bytes(b"bc")
    second = gguf_graph_package_identity(package)
    assert first.files == second.files == ("a.onnx", "b.data")
    assert first.sha256 != second.sha256

    (package / "linked.data").symlink_to(package / "b.data")
    with pytest.raises(ValueError, match="must not contain symlinks"):
        gguf_graph_package_identity(package)
