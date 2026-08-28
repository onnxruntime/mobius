# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for atomic, component-aware GGUF runtime package emission."""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

import pytest

from mobius.integrations.gguf import _runtime_package, write_gguf_runtime_package
from mobius.integrations.gguf._component_export import attach_tokenizer_export_report
from mobius.integrations.gguf._runtime_evidence import RuntimeEvidenceUnavailableError
from mobius.integrations.gguf._spec import Support
from mobius.integrations.gguf._tokenizer import GGUFTokenizerVerdict

_TOKENIZER_REPOSITORY = "owner/tokenizer"
_TOKENIZER_REVISION = "c" * 40
_BUILT_IDENTITY = SimpleNamespace(
    architecture="llama",
    filename="model.gguf",
    sha256="a" * 64,
)


class _FakePackage:
    def __init__(self):
        self.config = object()
        self.gguf_architecture = "llama"
        self.gguf_execution_provider = "cpu"
        self.gguf_import_route = '{"route_schema":1}'
        self.gguf_artifact_identity = _BUILT_IDENTITY
        self.gguf_tokenizer_verdict = _materialized()
        self.export_report: Any = None
        self.saved_to: str | None = None

    def save(self, path, **kwargs):
        self.saved_to = path
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "model.onnx").write_bytes(b"stub")
        if self.export_report is not None:
            self.export_report.write_json(Path(path) / "export_report.json")

    def __iter__(self):
        return iter(("model",))


def _materialized():
    return SimpleNamespace(
        materialized=True,
        route_identifier="embedded",
        reason="exact embedded tokenizer",
        metadata_sha256="f" * 64,
    )


def _write_tokenizer(_source, output, **_kwargs):
    path = Path(output) / "tokenizer.json"
    path.write_text("{}", encoding="utf-8")
    return str(path)


def _write_config(_pkg, output, **_kwargs):
    path = Path(output) / "inference_metadata.yaml"
    path.write_text("model: {}", encoding="utf-8")
    return {"inference_metadata": str(path)}


def _write_draft(_manifest, output):
    path = Path(output) / "draft_manifest.json"
    path.write_text("{}", encoding="utf-8")
    return str(path)


@contextmanager
def _successful_runtime_dependencies(runtime: str = "onnx-genai"):
    config_target = (
        "mobius.integrations.ort_genai.write_ort_genai_config"
        if runtime == "ort-genai"
        else "mobius.integrations.onnx_genai.write_onnx_genai_config"
    )
    with (
        mock.patch(
            "mobius.integrations.gguf._runtime_package.open_gguf_model",
            return_value=SimpleNamespace(
                metadata={},
                architecture="llama",
                source_matches_path=lambda: True,
            ),
        ),
        mock.patch(
            "mobius.integrations.gguf._runtime_package.inspect_gguf_tokenizer",
            return_value=_materialized(),
        ),
        mock.patch(
            "mobius.integrations.gguf._runtime_package.materialize_gguf_tokenizer",
            side_effect=_write_tokenizer,
        ),
        mock.patch(
            "mobius.integrations.gguf._runtime_package.write_gguf_tokenizer_json",
            side_effect=_write_tokenizer,
        ),
        mock.patch(config_target, side_effect=_write_config),
        mock.patch(
            "mobius.integrations.gguf._draft.write_draft_manifest",
            side_effect=_write_draft,
        ),
        mock.patch(
            "mobius.integrations.onnx_genai.inference_metadata.write_mtp_speculator_metadata",
            return_value=None,
        ),
    ):
        yield


def _source_model():
    return SimpleNamespace(
        metadata={},
        architecture="llama",
        source_matches_path=lambda: True,
    )


def _write_runtime(pkg, source, output, **kwargs):
    return write_gguf_runtime_package(
        pkg,
        source,
        output,
        tokenizer_repository=_TOKENIZER_REPOSITORY,
        tokenizer_revision=_TOKENIZER_REVISION,
        **kwargs,
    )


@pytest.fixture(autouse=True)
def _runtime_supported():
    with (
        mock.patch(
            "mobius.integrations.gguf._runtime_package.get_arch_spec",
            return_value=SimpleNamespace(
                runtime=Support.SUPPORTED,
                reason=None,
                gguf_arch="llama",
                runtime_evidence_ids=("test-evidence",),
            ),
        ),
        mock.patch(
            "mobius.integrations.gguf._runtime_package.matching_runtime_evidence",
            return_value=SimpleNamespace(
                evidence_id="test-evidence",
                graph_files=("model.onnx",),
                graph_sha256="b" * 64,
                runtime_package_files=("recorded-package.onnx",),
                runtime_package_sha256="c" * 64,
                tokenizer_repository=_TOKENIZER_REPOSITORY,
                tokenizer_revision=_TOKENIZER_REVISION,
                tokenizer_metadata_sha256="f" * 64,
                tokenizer_assets=(("tokenizer.json", 2, "a" * 64),),
            ),
        ),
        mock.patch(
            "mobius.integrations.gguf._runtime_package.gguf_graph_package_identity",
            return_value=SimpleNamespace(files=("model.onnx",), sha256="b" * 64),
        ),
        mock.patch(
            "mobius.integrations.gguf._runtime_package.gguf_artifact_identity",
            return_value=_BUILT_IDENTITY,
        ),
    ):
        yield


class TestWriteGgufRuntimePackage:
    def test_atomic_publication_refuses_concurrent_destination(self, tmp_path):
        stage = tmp_path / "stage"
        stage.mkdir()
        (stage / "model.onnx").write_bytes(b"staged")
        output = tmp_path / "output"
        output.mkdir()

        with pytest.raises(FileExistsError):
            _runtime_package._publish_directory_no_replace(stage, output)

        assert (stage / "model.onnx").read_bytes() == b"staged"
        assert not list(output.iterdir())

    def test_mtp_runtime_status_hashes_payload_and_separates_cache_namespaces(self, tmp_path):
        import onnx_ir as ir

        from mobius._model_package import ModelPackage
        from mobius.integrations.gguf._runtime_evidence import (
            gguf_graph_package_identity,
        )

        def model(inputs, outputs):
            graph = ir.Graph(
                [ir.Value(name=name) for name in inputs],
                [ir.Value(name=name) for name in outputs],
                nodes=[],
                name="cache-model",
            )
            return ir.Model(graph, ir_version=10)

        package = ModelPackage(
            {
                "model": model(
                    ["input_ids", "past_key_values.0.key", "past_key_values.0.value"],
                    ["logits", "present.0.key", "present.0.value"],
                )
            }
        )
        package.mtp_head = ModelPackage(
            {
                "model": model(
                    [
                        "hidden_states",
                        "past_key_values.0.key",
                        "past_key_values.0.value",
                    ],
                    ["mtp_hidden", "present.0.key", "present.0.value"],
                )
            }
        )
        package.gguf_execution_provider = "cpu"
        stage = tmp_path / "stage"
        package.save(stage, progress_bar=False, check_weights=False)
        graph_identity = gguf_graph_package_identity(stage)
        (stage / "mtp_config.json").write_text('{"status":"runtime_unvalidated"}\n')
        runtime_identity = gguf_graph_package_identity(stage)
        identity = SimpleNamespace(
            architecture="qwen35",
            filename="model.gguf",
            size=123,
            sha256="a" * 64,
            tensor_count=7,
            tensor_qtypes=(("F32", 7),),
        )

        status_path = _runtime_package._write_mtp_runtime_status(
            stage,
            pkg=package,
            built_identity=identity,
            graph_identity=graph_identity,
            runtime_payload_identity=runtime_identity,
            runtime="ort-genai",
            runtime_version="0.15.2",
            tokenizer_repository="owner/tokenizer",
            tokenizer_revision="b" * 40,
            tokenizer_metadata_sha256="c" * 64,
        )
        status = json.loads(Path(status_path).read_text(encoding="utf-8"))

        assert status["status"] == "runtime_unvalidated"
        assert status["artifact"]["sha256"] == "a" * 64
        assert status["graph_package"]["sha256"] == graph_identity.sha256
        assert status["runtime_payload"]["sha256"] == runtime_identity.sha256
        assert status["config_sha256"]["mtp_config.json"]
        assert status["cache_namespaces"]["target"]["namespace"] == "target"
        assert status["cache_namespaces"]["mtp"]["namespace"] == "mtp"
        assert (
            status["cache_namespaces"]["target"]["ports"]
            == status["cache_namespaces"]["mtp"]["ports"]
        )
        assert status["validated_claims"] == {
            "artifact_identity": True,
            "graph_serialization": True,
            "cache_namespace_separation": True,
            "runtime_execution": False,
            "source_value_fidelity": False,
            "storage_fidelity": False,
            "target_only_output_equality": False,
        }

    def test_deferred_architecture_exports_unvalidated_metadata(self, tmp_path):
        pkg = _FakePackage()
        out = tmp_path / "out"
        with (
            mock.patch(
                "mobius.integrations.gguf._runtime_package.get_arch_spec",
                return_value=SimpleNamespace(
                    runtime=Support.DEFERRED,
                    reason="independent parity is missing",
                    gguf_arch="llama",
                    runtime_evidence_ids=(),
                ),
            ),
            mock.patch(
                "mobius.integrations.gguf._runtime_package.matching_runtime_evidence",
                side_effect=RuntimeEvidenceUnavailableError(
                    "No unique GGUF runtime evidence matches architecture='llama'"
                ),
            ),
            _successful_runtime_dependencies(),
        ):
            _write_runtime(pkg, tmp_path / "m.gguf", out)
        compatibility = json.loads((out / "runtime_compatibility.json").read_text())
        assert compatibility["runtime_validation_status"] == "unvalidated"
        assert "independent parity is missing" in compatibility["warnings"][0]
        assert pkg.export_report is not None
        runtime_component = pkg.export_report.component("runtime")
        assert runtime_component is not None
        assert runtime_component.output == "exported"
        assert runtime_component.support == "deferred"

    def test_atomically_emits_graph_tokenizer_and_runtime_config(self, tmp_path):
        pkg = _FakePackage()
        out = tmp_path / "out"
        with (
            mock.patch(
                "mobius.integrations.gguf._runtime_package.open_gguf_model",
                return_value=SimpleNamespace(
                    **vars(_source_model()),
                ),
            ),
            mock.patch(
                "mobius.integrations.gguf._runtime_package.inspect_gguf_tokenizer",
                return_value=_materialized(),
            ),
            mock.patch(
                "mobius.integrations.gguf._runtime_package.materialize_gguf_tokenizer",
                side_effect=_write_tokenizer,
            ),
            mock.patch(
                "mobius.integrations.onnx_genai.write_onnx_genai_config",
                side_effect=_write_config,
            ),
        ):
            artifacts = _write_runtime(pkg, tmp_path / "m.gguf", out)

        assert (out / "model.onnx").read_bytes() == b"stub"
        assert Path(artifacts["tokenizer"]) == out / "tokenizer.json"
        assert Path(artifacts["inference_metadata"]) == out / "inference_metadata.yaml"
        compatibility = json.loads((out / "runtime_compatibility.json").read_text())
        assert compatibility["runtime_validation_status"] == "unvalidated"
        assert "completed runtime package" in compatibility["warnings"][-1]
        assert pkg.export_report is not None
        assert pkg.export_report.component("runtime").output == "exported"
        assert pkg.export_report.component("tokenizer").output == "exported"
        assert (out / "export_report.json").is_file()
        assert not list(tmp_path.glob(".out.*.tmp"))

    def test_exact_runtime_evidence_marks_package_validated(self, tmp_path):
        pkg = _FakePackage()
        out = tmp_path / "out"
        evidence = SimpleNamespace(
            evidence_id="test-evidence",
            graph_files=("model.onnx",),
            graph_sha256="b" * 64,
            runtime_package_files=(
                "inference_metadata.yaml",
                "model.onnx",
                "tokenizer.json",
            ),
            runtime_package_sha256="c" * 64,
            tokenizer_repository=_TOKENIZER_REPOSITORY,
            tokenizer_revision=_TOKENIZER_REVISION,
            tokenizer_metadata_sha256="f" * 64,
            tokenizer_assets=(("tokenizer.json", 2, "a" * 64),),
        )
        with (
            _successful_runtime_dependencies(),
            mock.patch(
                "mobius.integrations.gguf._runtime_package.matching_runtime_evidence",
                return_value=evidence,
            ),
            mock.patch(
                "mobius.integrations.gguf._runtime_package.gguf_graph_package_identity",
                side_effect=(
                    SimpleNamespace(files=evidence.graph_files, sha256=evidence.graph_sha256),
                    SimpleNamespace(
                        files=evidence.runtime_package_files,
                        sha256=evidence.runtime_package_sha256,
                    ),
                ),
            ),
        ):
            _write_runtime(pkg, tmp_path / "m.gguf", out, runtime_version="0.15.2")

        assert pkg.export_report is not None
        assert pkg.export_report.export_status == "complete"
        assert pkg.export_report.runtime_validation_status == "validated"
        assert pkg.export_report.end_to_end_runnable is True
        runtime_component = pkg.export_report.component("runtime")
        assert runtime_component is not None
        assert runtime_component.support == "supported"
        assert runtime_component.runtime_validation_status == "validated"
        compatibility = json.loads((out / "runtime_compatibility.json").read_text())
        assert compatibility["runtime_validation_status"] == "validated"
        assert compatibility["runtime_evidence_id"] == "test-evidence"

    def test_missing_evidence_runtime_version_does_not_block_export(self, tmp_path):
        pkg = _FakePackage()
        out = tmp_path / "out"
        with (
            mock.patch(
                "mobius.integrations.gguf._runtime_package.matching_runtime_evidence",
                side_effect=RuntimeEvidenceUnavailableError(
                    "Runtime packaging requires the exact runtime version covered by evidence."
                ),
            ),
            _successful_runtime_dependencies(),
        ):
            artifacts = _write_runtime(pkg, tmp_path / "m.gguf", out)

        assert Path(artifacts["inference_metadata"]).is_file()
        compatibility = json.loads((out / "runtime_compatibility.json").read_text())
        assert compatibility["runtime_validation_status"] == "unvalidated"

    def test_portable_graph_targets_ort_genai_cpu(self, tmp_path):
        pkg = _FakePackage()
        pkg.gguf_execution_provider = "default"
        out = tmp_path / "out"
        with (
            mock.patch(
                "mobius.integrations.gguf._runtime_package.open_gguf_model",
                return_value=SimpleNamespace(
                    **vars(_source_model()),
                ),
            ),
            mock.patch(
                "mobius.integrations.gguf._runtime_package.inspect_gguf_tokenizer",
                return_value=_materialized(),
            ),
            mock.patch(
                "mobius.integrations.gguf._runtime_package.write_gguf_tokenizer_json",
                side_effect=_write_tokenizer,
            ),
            mock.patch(
                "mobius.integrations.gguf._runtime_package.materialize_gguf_tokenizer",
                side_effect=_write_tokenizer,
            ),
            mock.patch(
                "mobius.integrations.ort_genai.write_ort_genai_config",
                side_effect=_write_config,
            ) as write_config,
        ):
            _write_runtime(
                pkg,
                tmp_path / "m.gguf",
                out,
                runtime="ort-genai",
                runtime_version="0.15.2",
            )

        assert write_config.call_args.kwargs["ep"] == "cpu"

    def test_failure_after_graph_save_removes_staging_and_publishes_nothing(self, tmp_path):
        pkg = _FakePackage()
        out = tmp_path / "out"
        with (
            mock.patch(
                "mobius.integrations.gguf._runtime_package.open_gguf_model",
                return_value=SimpleNamespace(
                    **vars(_source_model()),
                ),
            ),
            mock.patch(
                "mobius.integrations.gguf._runtime_package.inspect_gguf_tokenizer",
                return_value=_materialized(),
            ),
            mock.patch(
                "mobius.integrations.gguf._runtime_package.materialize_gguf_tokenizer",
                side_effect=OSError("tokenizer asset write failed"),
            ),
            pytest.raises(OSError, match="tokenizer asset write failed"),
        ):
            _write_runtime(pkg, tmp_path / "m.gguf", out)

        assert pkg.saved_to is not None
        assert not out.exists()
        assert not list(tmp_path.glob(".out.*.tmp"))

    def test_tokenizer_source_options_must_be_paired(self, tmp_path):
        pkg = _FakePackage()
        out = tmp_path / "out"
        with (
            mock.patch(
                "mobius.integrations.gguf._runtime_package.open_gguf_model"
            ) as read_source,
            pytest.raises(ValueError, match="must be provided together"),
        ):
            write_gguf_runtime_package(
                pkg,
                tmp_path / "m.gguf",
                out,
                tokenizer_repository=_TOKENIZER_REPOSITORY,
            )
        read_source.assert_not_called()
        assert pkg.saved_to is None
        assert not out.exists()

    def test_mutable_tokenizer_revision_rejects_before_source_read(self, tmp_path):
        pkg = _FakePackage()
        pkg.mtp_head = SimpleNamespace(config=object())
        with (
            mock.patch(
                "mobius.integrations.gguf._runtime_package.open_gguf_model"
            ) as read_source,
            pytest.raises(ValueError, match="immutable 40-hex"),
        ):
            write_gguf_runtime_package(
                pkg,
                tmp_path / "m.gguf",
                tmp_path / "out",
                tokenizer_repository="owner/tokenizer",
                tokenizer_revision="main",
            )
        read_source.assert_not_called()

    def test_tokenizer_blocker_omits_only_tokenizer_assets(self, tmp_path, caplog):
        pkg = _FakePackage()
        blocker = GGUFTokenizerVerdict(
            route="deferred",
            model="gpt2",
            pre="blocked-pre",
            canonical_pre="blocked-pre",
            token_count=2,
            metadata_sha256="f" * 64,
            blocker_category="semantic-mismatch",
            audit_status="deferred-pinned-artifact-mismatch",
            reason="recorded tokenizer mismatch",
            evidence_id="blocker-evidence",
        )
        out = tmp_path / "out"
        with (
            caplog.at_level("WARNING"),
            _successful_runtime_dependencies(),
            mock.patch(
                "mobius.integrations.gguf._runtime_package.inspect_gguf_tokenizer",
                return_value=blocker,
            ),
        ):
            pkg.gguf_tokenizer_verdict = blocker
            attach_tokenizer_export_report(pkg, blocker, model_route="llama")
            artifacts = _write_runtime(pkg, tmp_path / "m.gguf", out)

        assert (out / "model.onnx").is_file()
        assert Path(artifacts["export_report"]).is_file()
        assert pkg.export_report.component("tokenizer").output == "omitted"
        runtime_component = pkg.export_report.component("runtime")
        assert runtime_component.output == "exported"
        assert runtime_component.support == "deferred"
        assert not (out / "tokenizer.json").exists()
        assert (out / "inference_metadata.yaml").is_file()
        assert caplog.text.count("GGUF PARTIAL EXPORT WARNING:") == 1

    def test_replaced_source_tokenizer_rejects_before_save_or_output(self, tmp_path):
        pkg = _FakePackage()
        out = tmp_path / "out"
        with (
            mock.patch(
                "mobius.integrations.gguf._runtime_package.open_gguf_model",
                return_value=SimpleNamespace(
                    metadata={},
                    architecture="llama",
                    source_matches_path=lambda: True,
                ),
            ),
            mock.patch(
                "mobius.integrations.gguf._runtime_package.inspect_gguf_tokenizer",
                return_value=SimpleNamespace(
                    materialized=True,
                    reason="replacement",
                    metadata_sha256="different-tokenizer-metadata",
                ),
            ),
            pytest.raises(ValueError, match="replaced tokenizer source"),
        ):
            _write_runtime(pkg, tmp_path / "m.gguf", out)
        assert pkg.saved_to is None
        assert not out.exists()

    def test_replaced_same_architecture_source_identity_always_rejects(self, tmp_path):
        pkg = _FakePackage()
        out = tmp_path / "out"
        current_identity = SimpleNamespace(
            architecture="llama",
            filename="model.gguf",
            sha256="d" * 64,
        )
        with (
            mock.patch(
                "mobius.integrations.gguf._runtime_package.open_gguf_model",
                return_value=SimpleNamespace(
                    metadata={},
                    architecture="llama",
                    source_matches_path=lambda: True,
                ),
            ),
            mock.patch(
                "mobius.integrations.gguf._runtime_package.gguf_artifact_identity",
                return_value=current_identity,
            ),
            mock.patch(
                "mobius.integrations.gguf._runtime_package.matching_runtime_evidence"
            ) as evidence_lookup,
            pytest.raises(ValueError, match="exact artifact identity"),
        ):
            write_gguf_runtime_package(pkg, tmp_path / "replacement.gguf", out)

        evidence_lookup.assert_not_called()
        assert pkg.saved_to is None
        assert not out.exists()

    def test_existing_output_is_never_replaced(self, tmp_path):
        pkg = _FakePackage()
        out = tmp_path / "out"
        out.mkdir()
        sentinel = out / "sentinel.bin"
        sentinel.write_bytes(b"unchanged")
        with (
            mock.patch(
                "mobius.integrations.gguf._runtime_package.open_gguf_model"
            ) as read_source,
            pytest.raises(FileExistsError, match="non-atomic directory replacement"),
        ):
            _write_runtime(pkg, tmp_path / "m.gguf", out)
        read_source.assert_not_called()
        assert {path.name: path.read_bytes() for path in out.iterdir()} == {
            "sentinel.bin": b"unchanged"
        }

    def test_ort_genai_exports_reused_weights_as_runtime_unvalidated(self, tmp_path, caplog):
        pkg = _FakePackage()
        pkg.gguf_reuse_plan = object()
        out = tmp_path / "out"
        with (
            _successful_runtime_dependencies("ort-genai"),
            caplog.at_level("WARNING"),
        ):
            artifacts = _write_runtime(
                pkg,
                tmp_path / "m.gguf",
                out,
                runtime="ort-genai",
            )
        assert (out / "model.onnx").is_file()
        assert Path(artifacts["export_report"]).is_file()
        runtime_component = pkg.export_report.component("runtime")
        assert runtime_component.output == "exported"
        assert "cannot disable constant folding" in runtime_component.reason
        assert "cannot disable constant folding" in caplog.text

    def test_unknown_execution_provider_is_preserved_as_advisory(self, tmp_path):
        pkg = _FakePackage()
        pkg.gguf_execution_provider = "future-accelerator"
        out = tmp_path / "out"

        with _successful_runtime_dependencies("ort-genai"):
            _write_runtime(pkg, tmp_path / "m.gguf", out, runtime="ort-genai")

        compatibility = json.loads((out / "runtime_compatibility.json").read_text())
        assert compatibility["execution_provider"] == "future-accelerator"
        assert compatibility["runtime_validation_status"] == "unvalidated"
        assert any("future-accelerator" in warning for warning in compatibility["warnings"])

    def test_target_coupled_draft_runtime_package_is_exported(self, tmp_path):
        pkg = _FakePackage()
        pkg.draft_manifest = {"architecture": "eagle3"}
        out = tmp_path / "out"

        with _successful_runtime_dependencies():
            _write_runtime(
                pkg,
                tmp_path / "eagle3.gguf",
                out,
            )
        assert (out / "draft_manifest.json").is_file()

    def test_ort_runtime_emits_unvalidated_mtp_without_artifact_allowlist(self, tmp_path):
        pkg = _FakePackage()
        pkg.mtp_head = SimpleNamespace(config=object())
        pkg.gguf_source_filename = "quantized/model.gguf"
        pkg.gguf_artifact_identity = SimpleNamespace(
            architecture="llama",
            filename=pkg.gguf_source_filename,
            sha256="a" * 64,
        )
        out = tmp_path / "out"

        def write_status(stage, **_kwargs):
            path = stage / "mtp_runtime_status.json"
            path.write_text('{"status":"runtime_unvalidated"}\n', encoding="utf-8")
            return str(path)

        with (
            mock.patch(
                "mobius.integrations.gguf._runtime_package.get_arch_spec",
                return_value=SimpleNamespace(
                    runtime=Support.DEFERRED,
                    reason="real MTP execution is unvalidated",
                    gguf_arch="llama",
                    runtime_evidence_ids=(),
                ),
            ),
            mock.patch(
                "mobius.integrations.gguf._runtime_package.open_gguf_model",
                return_value=SimpleNamespace(
                    metadata={},
                    architecture="llama",
                    source_matches_path=lambda: True,
                ),
            ),
            mock.patch(
                "mobius.integrations.gguf._runtime_package.gguf_artifact_identity",
                return_value=pkg.gguf_artifact_identity,
            ) as artifact_identity,
            mock.patch(
                "mobius.integrations.gguf._runtime_package.inspect_gguf_tokenizer",
                return_value=_materialized(),
            ),
            mock.patch(
                "mobius.integrations.gguf._runtime_package.write_gguf_tokenizer_json",
                side_effect=_write_tokenizer,
            ),
            mock.patch(
                "mobius.integrations.gguf._runtime_package.matching_runtime_evidence"
            ) as match_evidence,
            mock.patch(
                "mobius.integrations.ort_genai.write_ort_genai_config",
                side_effect=_write_config,
            ),
            mock.patch(
                "mobius.integrations.ort_genai.auto_export._copy_tokenizer_files",
                return_value=[],
            ) as copy_tokenizer,
            mock.patch(
                "mobius.integrations.gguf._runtime_package._write_mtp_runtime_status",
                side_effect=write_status,
            ),
        ):
            artifacts = write_gguf_runtime_package(
                pkg,
                tmp_path / "m.gguf",
                out,
                runtime="ort-genai",
                runtime_version="0.15.2",
            )
        match_evidence.assert_not_called()
        assert artifact_identity.call_args.kwargs["filename"] == "quantized/model.gguf"
        copy_tokenizer.assert_not_called()
        assert out.is_dir()
        assert (out / "mtp_runtime_status.json").is_file()
        assert artifacts["mtp_runtime_status"] == str(out / "mtp_runtime_status.json")

    def test_onnx_runtime_emits_unvalidated_mtp_without_artifact_allowlist(self, tmp_path):
        pkg = _FakePackage()
        pkg.mtp_head = SimpleNamespace(config=object())
        out = tmp_path / "out"

        def write_status(stage, **_kwargs):
            path = stage / "mtp_runtime_status.json"
            path.write_text('{"status":"runtime_unvalidated"}\n', encoding="utf-8")
            return str(path)

        with (
            mock.patch(
                "mobius.integrations.gguf._runtime_package.get_arch_spec",
                return_value=SimpleNamespace(
                    runtime=Support.DEFERRED,
                    reason="real MTP execution is unvalidated",
                    gguf_arch="llama",
                    runtime_evidence_ids=(),
                ),
            ),
            mock.patch(
                "mobius.integrations.gguf._runtime_package.open_gguf_model",
                return_value=SimpleNamespace(
                    metadata={},
                    architecture="llama",
                    source_matches_path=lambda: True,
                ),
            ),
            mock.patch(
                "mobius.integrations.gguf._runtime_package.gguf_artifact_identity",
                return_value=pkg.gguf_artifact_identity,
            ),
            mock.patch(
                "mobius.integrations.gguf._runtime_package.inspect_gguf_tokenizer",
                return_value=_materialized(),
            ),
            mock.patch(
                "mobius.integrations.gguf._runtime_package.write_gguf_tokenizer_json",
                side_effect=_write_tokenizer,
            ),
            mock.patch(
                "mobius.integrations.gguf._runtime_package.matching_runtime_evidence"
            ) as match_evidence,
            mock.patch(
                "mobius.integrations.onnx_genai.write_onnx_genai_config",
                side_effect=_write_config,
            ),
            mock.patch(
                "mobius._model_package._read_mtp_sidecar_name",
                return_value=".mobius-mtp",
            ),
            mock.patch(
                "mobius.integrations.onnx_genai.inference_metadata.write_mtp_speculator_metadata",
                side_effect=lambda directory, **_kwargs: (
                    Path(directory) / "inference_metadata.yaml"
                ),
            ),
            mock.patch(
                "mobius.integrations.gguf._runtime_package._write_mtp_runtime_status",
                side_effect=write_status,
            ),
        ):
            artifacts = _write_runtime(
                pkg,
                tmp_path / "m.gguf",
                out,
                runtime="onnx-genai",
                runtime_version="1.29.0",
            )
        match_evidence.assert_not_called()
        assert out.is_dir()
        assert artifacts["mtp_runtime_status"] == str(out / "mtp_runtime_status.json")

    def test_mtp_source_change_during_serialization_publishes_nothing(self, tmp_path):
        pkg = _FakePackage()
        pkg.mtp_head = SimpleNamespace(config=object())
        matches_source = mock.Mock(side_effect=[True, True, True, False])
        out = tmp_path / "out"
        with (
            mock.patch(
                "mobius.integrations.gguf._runtime_package.get_arch_spec",
                return_value=SimpleNamespace(
                    runtime=Support.DEFERRED,
                    reason="real MTP execution is unvalidated",
                    gguf_arch="llama",
                    runtime_evidence_ids=(),
                ),
            ),
            mock.patch(
                "mobius.integrations.gguf._runtime_package.open_gguf_model",
                return_value=SimpleNamespace(
                    metadata={},
                    architecture="llama",
                    source_matches_path=matches_source,
                ),
            ),
            mock.patch(
                "mobius.integrations.gguf._runtime_package.gguf_artifact_identity",
                return_value=pkg.gguf_artifact_identity,
            ),
            mock.patch(
                "mobius.integrations.gguf._runtime_package.inspect_gguf_tokenizer",
                return_value=_materialized(),
            ),
            pytest.raises(ValueError, match="while target/MTP graphs were being serialized"),
        ):
            _write_runtime(pkg, tmp_path / "m.gguf", out)
        assert not out.exists()
        assert not list(tmp_path.glob(".out.*.tmp"))

    def test_unknown_runtime_rejects_before_source_read(self, tmp_path):
        with pytest.raises(ValueError, match="Unknown runtime"):
            write_gguf_runtime_package(
                _FakePackage(), tmp_path / "m.gguf", tmp_path / "out", runtime="tflite"
            )

    def test_save_model_false_is_rejected_before_source_read(self, tmp_path):
        with (
            mock.patch(
                "mobius.integrations.gguf._runtime_package.open_gguf_model"
            ) as read_source,
            pytest.raises(ValueError, match="save_model=False is not supported"),
        ):
            write_gguf_runtime_package(
                _FakePackage(), tmp_path / "m.gguf", tmp_path / "output", save_model=False
            )
        read_source.assert_not_called()
