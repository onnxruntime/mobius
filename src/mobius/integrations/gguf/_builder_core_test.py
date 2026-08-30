# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for core GGUF weight import, reuse, and quantized builds."""

from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar
from unittest import mock

import numpy as np
import onnx_ir as ir
import onnxruntime as ort
import pytest

from mobius.integrations.gguf._builder_test_utils import (
    _run_gather_block_quantized,
    _symlink_or_skip,
    _write_moe_gguf,
    _write_quantized_gguf,
    _write_tencent_q1_0_gguf,
)
from mobius.integrations.gguf._builder_test_utils import (
    float_only_gguf as float_only_gguf,
)
from mobius.integrations.gguf._builder_test_utils import (
    iq4_nl_embedding_gguf as iq4_nl_embedding_gguf,
)
from mobius.integrations.gguf._builder_test_utils import (
    mixed_native_q5_q8_gguf as mixed_native_q5_q8_gguf,
)
from mobius.integrations.gguf._builder_test_utils import (
    native_block_gguf as native_block_gguf,
)
from mobius.integrations.gguf._builder_test_utils import (
    q4_0_embedding_gguf as q4_0_embedding_gguf,
)
from mobius.integrations.gguf._builder_test_utils import (
    q4_0_embedding_q8_head_gguf as q4_0_embedding_q8_head_gguf,
)
from mobius.integrations.gguf._builder_test_utils import (
    q4_0_gguf as q4_0_gguf,
)
from mobius.integrations.gguf._builder_test_utils import (
    q4_0_tied_embedding_gguf as q4_0_tied_embedding_gguf,
)
from mobius.integrations.gguf._builder_test_utils import (
    q5_1_gguf as q5_1_gguf,
)
from mobius.integrations.gguf._builder_test_utils import (
    q8_0_projection_q4_head_gguf as q8_0_projection_q4_head_gguf,
)


@pytest.mark.parametrize("architecture", ["llama", "qwen2", "lfm2", "qwen35moe"])
def test_existing_evidence_route_fingerprints_ignore_jina_fused_qkv(
    architecture: str,
) -> None:
    from mobius._testing import make_config
    from mobius.integrations.gguf._builder import _serialize_route_graph_config

    config = make_config()
    legacy = _serialize_route_graph_config(config, architecture)
    changed = _serialize_route_graph_config(
        dataclasses.replace(config, encoder_fused_qkv=True),
        architecture,
    )

    assert changed == legacy


def test_jina_v3_route_fingerprint_retains_fused_qkv_semantics() -> None:
    from mobius._testing import make_config
    from mobius.integrations.gguf._builder import _serialize_route_graph_config

    split = _serialize_route_graph_config(make_config(), "jina-bert-v3")
    fused = _serialize_route_graph_config(
        make_config(encoder_fused_qkv=True),
        "jina-bert-v3",
    )

    assert fused != split


class TestReuseGgufWeights:
    """Tests for mixed GGUF references plus converted ONNX sidecar weights."""

    def test_fsync_file_uses_writable_descriptor_on_windows(self, tmp_path: Path, monkeypatch):
        from mobius.integrations.gguf import _reuse

        path = tmp_path / "artifact.bin"
        path.write_bytes(b"artifact")
        opened_modes = []
        real_open = Path.open

        def tracking_open(self, *args, **kwargs):
            if self == path:
                opened_modes.append(args[0] if args else kwargs.get("mode", "r"))
            return real_open(self, *args, **kwargs)

        monkeypatch.setattr(_reuse.os, "name", "nt")
        monkeypatch.setattr(Path, "open", tracking_open)

        _reuse._fsync_file(path)

        assert opened_modes == ["r+b"]

    def test_mixed_save_preserves_ranges_and_runs(self, tmp_path: Path):
        from mobius._model_package import ModelPackage
        from mobius.integrations.gguf import (
            build_from_gguf,
            verify_gguf_reuse_manifest,
        )

        gguf_path = tmp_path / "model.gguf"
        _write_quantized_gguf(gguf_path, projection_quantization="f32")
        package = build_from_gguf(gguf_path, reuse_gguf_weights=True)
        from mobius.integrations.gguf._reader import GGUFModel

        with mock.patch.object(
            GGUFModel,
            "source_sha256",
            autospec=True,
            side_effect=GGUFModel.source_sha256,
        ) as sha256:
            package.save(str(tmp_path), progress_bar=False)
        assert sha256.call_count == 1

        verify_gguf_reuse_manifest(tmp_path)
        manifest = json.loads((tmp_path / "gguf-reuse.json").read_text())
        converted = manifest["converted_tensors"]
        assert len(converted) == len(set(converted))
        assert "model.layers.0.self_attn.q_proj.weight" not in converted
        assert "model.embed_tokens.weight" not in converted
        q_route = next(
            route
            for route in manifest["reused_tensors"]
            if route["initializer"] == "model.layers.0.self_attn.q_proj.weight"
        )
        assert q_route["transform"] == "llama_qk_permute"
        reloaded = ModelPackage.load(str(tmp_path))
        initializers = reloaded["model"].graph.initializers
        embedding = initializers["model.embed_tokens.weight"].const_value
        q_proj = initializers["model.layers.0.self_attn.q_proj.weight"].const_value
        assert isinstance(embedding, ir.ExternalTensor)
        assert embedding.location == "model.gguf"
        assert embedding.offset is not None
        assert embedding.length == 256 * 64 * 4
        # Llama Q weights keep their GGUF bytes; ONNX performs the row permutation.
        assert isinstance(q_proj, ir.ExternalTensor)
        assert q_proj.location == "model.gguf"
        assert any(
            node.name == "model.layers.0.self_attn.q_proj.weight.gguf_reuse.Transpose"
            for node in reloaded["model"].graph
        )

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        session = ort.InferenceSession(
            str(tmp_path / "model.onnx"),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        feeds = {
            "input_ids": np.zeros((1, 2), dtype=np.int64),
            "attention_mask": np.zeros((1, 2), dtype=np.int64),
            "position_ids": np.zeros((1, 2), dtype=np.int64),
            "past_key_values.0.key": np.zeros((1, 2, 0, 16), dtype=np.float32),
            "past_key_values.0.value": np.zeros((1, 2, 0, 16), dtype=np.float32),
        }
        outputs = session.run(None, feeds)
        assert outputs[0].shape == (1, 2, 256)
        assert np.isfinite(outputs[0]).all()

        reference_dir = tmp_path / "reference"
        build_from_gguf(gguf_path).save(str(reference_dir), progress_bar=False)
        reference_session = ort.InferenceSession(
            str(reference_dir / "model.onnx"),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        reference_outputs = reference_session.run(None, feeds)
        np.testing.assert_allclose(outputs[0], reference_outputs[0], rtol=1e-5, atol=1e-5)

    def test_save_rehashes_source_immediately_before_publish(
        self,
        tmp_path: Path,
        monkeypatch,
    ) -> None:
        from mobius.integrations.gguf import _reuse, build_from_gguf

        gguf_path = tmp_path / "model.gguf"
        _write_quantized_gguf(gguf_path, projection_quantization="f32")
        package = build_from_gguf(gguf_path, reuse_gguf_weights=True)
        real_write_json = _reuse._write_json_exclusive
        mutated = False

        def mutate_source_before_verification(path, payload):
            nonlocal mutated
            if path.name.startswith(".gguf-reuse.json.") and path.name.endswith(".tmp"):
                with gguf_path.open("r+b") as stream:
                    stream.seek(-1, os.SEEK_END)
                    value = stream.read(1)
                    stream.seek(-1, os.SEEK_END)
                    stream.write(bytes([value[0] ^ 0xFF]))
                    stream.flush()
                    os.fsync(stream.fileno())
                mutated = True
            return real_write_json(path, payload)

        monkeypatch.setattr(_reuse, "_write_json_exclusive", mutate_source_before_verification)
        with pytest.raises(ValueError, match="source identity mismatch"):
            package.save(str(tmp_path), progress_bar=False)

        assert mutated
        assert not (tmp_path / "model.onnx").exists()
        assert not (tmp_path / "gguf-reuse.json").exists()

    def test_reuse_plan_hashes_the_model_source_used_for_tensor_parsing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        from mobius.integrations.gguf import build_from_gguf
        from mobius.integrations.gguf._reader import GGUFModel

        gguf_path = tmp_path / "model.gguf"
        _write_quantized_gguf(gguf_path, projection_quantization="f32")
        source_sha256 = GGUFModel.source_sha256
        mutated = False

        def mutate_before_plan_hash(model, **kwargs):
            nonlocal mutated
            if not mutated and Path(model._path) == gguf_path:
                with gguf_path.open("r+b") as stream:
                    stream.seek(-1, os.SEEK_END)
                    value = stream.read(1)
                    stream.seek(-1, os.SEEK_END)
                    stream.write(bytes([value[0] ^ 0xFF]))
                mutated = True
            return source_sha256(model, **kwargs)

        monkeypatch.setattr(GGUFModel, "source_sha256", mutate_before_plan_hash)

        with pytest.raises(ValueError, match="source changed after its reader was opened"):
            build_from_gguf(gguf_path, reuse_gguf_weights=True)
        assert mutated

    def test_reuse_plan_with_relative_source_survives_cwd_change(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        from mobius.integrations.gguf import build_from_gguf, verify_gguf_reuse_manifest

        gguf_path = tmp_path / "model.gguf"
        _write_quantized_gguf(gguf_path, projection_quantization="f32")
        monkeypatch.chdir(tmp_path)
        package = build_from_gguf(Path("model.gguf"), reuse_gguf_weights=True)
        other = tmp_path / "other"
        other.mkdir()
        monkeypatch.chdir(other)

        package.save(str(tmp_path), progress_bar=False)
        verify_gguf_reuse_manifest(tmp_path)

    def test_reuse_verifier_detects_mutation_between_hash_and_tensor_validation(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        from mobius.integrations.gguf import build_from_gguf, verify_gguf_reuse_manifest
        from mobius.integrations.gguf._reader import GGUFModel

        gguf_path = tmp_path / "model.gguf"
        _write_quantized_gguf(gguf_path, projection_quantization="f32")
        build_from_gguf(gguf_path, reuse_gguf_weights=True).save(
            str(tmp_path), progress_bar=False
        )
        source_sha256 = GGUFModel.source_sha256
        mutated = False

        def mutate_after_verifier_hash(model, **kwargs):
            nonlocal mutated
            digest = source_sha256(model, **kwargs)
            if not mutated and Path(model._path) == gguf_path:
                with gguf_path.open("r+b") as stream:
                    stream.seek(-1, os.SEEK_END)
                    value = stream.read(1)
                    stream.seek(-1, os.SEEK_END)
                    stream.write(bytes([value[0] ^ 0xFF]))
                mutated = True
            return digest

        monkeypatch.setattr(GGUFModel, "source_sha256", mutate_after_verifier_hash)

        with pytest.raises(ValueError, match="reuse manifest was verified"):
            verify_gguf_reuse_manifest(tmp_path)
        assert mutated

    def test_reuse_verifier_rejects_source_replaced_by_symlink_before_open(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        from mobius.integrations.gguf import build_from_gguf, verify_gguf_reuse_manifest
        from mobius.integrations.gguf._reader import GGUFModel

        gguf_path = tmp_path / "model.gguf"
        _write_quantized_gguf(gguf_path, projection_quantization="f32")
        build_from_gguf(gguf_path, reuse_gguf_weights=True).save(
            str(tmp_path), progress_bar=False
        )
        moved_source = tmp_path / "moved-source.gguf"
        original_init = GGUFModel.__init__
        raced = False

        def replace_before_verifier_open(model, path, **kwargs):
            nonlocal raced
            if not raced and Path(path) == gguf_path:
                gguf_path.replace(moved_source)
                _symlink_or_skip(gguf_path, moved_source)
                raced = True
            return original_init(model, path, **kwargs)

        monkeypatch.setattr(GGUFModel, "__init__", replace_before_verifier_open)

        with pytest.raises(ValueError, match="source is missing or unsafe"):
            verify_gguf_reuse_manifest(tmp_path)
        assert raced

    def test_reuse_verifier_rechecks_symlink_after_final_handle_comparison(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        from mobius.integrations.gguf import build_from_gguf, verify_gguf_reuse_manifest
        from mobius.integrations.gguf._reader import GGUFModel

        gguf_path = tmp_path / "model.gguf"
        _write_quantized_gguf(gguf_path, projection_quantization="f32")
        build_from_gguf(gguf_path, reuse_gguf_weights=True).save(
            str(tmp_path), progress_bar=False
        )
        source_matches_path = GGUFModel.source_matches_path
        path_is_symlink = Path.is_symlink
        raced = False
        symlink_recheck_observed = False

        def replace_after_handle_comparison(model, path=None):
            nonlocal raced
            matches = source_matches_path(model, path)
            if not raced and Path(model._path) == gguf_path:
                assert matches
                raced = True
            return matches

        def report_post_comparison_symlink(path):
            nonlocal symlink_recheck_observed
            if path == gguf_path and raced:
                symlink_recheck_observed = True
                return True
            return path_is_symlink(path)

        monkeypatch.setattr(GGUFModel, "source_matches_path", replace_after_handle_comparison)
        monkeypatch.setattr(Path, "is_symlink", report_post_comparison_symlink)

        with pytest.raises(ValueError, match="reuse manifest was verified"):
            verify_gguf_reuse_manifest(tmp_path)
        assert raced
        assert symlink_recheck_observed

    def test_reuse_verifier_rechecks_source_after_onnx_validation(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        from mobius.integrations.gguf import _reuse, build_from_gguf

        gguf_path = tmp_path / "model.gguf"
        _write_quantized_gguf(gguf_path, projection_quantization="f32")
        build_from_gguf(gguf_path, reuse_gguf_weights=True).save(
            str(tmp_path), progress_bar=False
        )
        load_model = _reuse.ir.load
        mutated = False

        def mutate_after_onnx_load(path):
            nonlocal mutated
            model = load_model(path)
            if not mutated:
                with gguf_path.open("r+b") as stream:
                    stream.seek(-1, os.SEEK_END)
                    value = stream.read(1)
                    stream.seek(-1, os.SEEK_END)
                    stream.write(bytes([value[0] ^ 0xFF]))
                mutated = True
            return model

        monkeypatch.setattr(_reuse.ir, "load", mutate_after_onnx_load)

        with pytest.raises(ValueError, match="reuse manifest was verified"):
            _reuse.verify_gguf_reuse_manifest(tmp_path)
        assert mutated

    def test_reuse_save_rechecks_source_immediately_before_publication(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        from mobius.integrations.gguf import _reuse, build_from_gguf

        gguf_path = tmp_path / "model.gguf"
        _write_quantized_gguf(gguf_path, projection_quantization="f32")
        package = build_from_gguf(gguf_path, reuse_gguf_weights=True)
        verify_manifest = _reuse.verify_gguf_reuse_manifest
        mutated = False

        def mutate_after_verification(*args, **kwargs):
            nonlocal mutated
            result = verify_manifest(*args, **kwargs)
            with gguf_path.open("r+b") as stream:
                stream.seek(-1, os.SEEK_END)
                value = stream.read(1)
                stream.seek(-1, os.SEEK_END)
                stream.write(bytes([value[0] ^ 0xFF]))
            mutated = True
            return result

        monkeypatch.setattr(_reuse, "verify_gguf_reuse_manifest", mutate_after_verification)

        with pytest.raises(ValueError, match="package was being prepared"):
            package.save(str(tmp_path), progress_bar=False)
        assert mutated
        assert not (tmp_path / "model.onnx").exists()
        assert not (tmp_path / "gguf-reuse.json").exists()

    def test_native_projection_bytes_are_not_copied_to_sidecar(self, tmp_path: Path):
        from mobius.integrations.gguf import build_from_gguf
        from mobius.integrations.gguf._reader import GGUFModel

        gguf_path = tmp_path / "model.gguf"
        _write_quantized_gguf(
            gguf_path,
            architecture="qwen2",
            hidden_size=256,
            num_heads=4,
            num_kv_heads=4,
            intermediate_size=256,
            projection_quantization="iq4_nl",
        )
        package = build_from_gguf(gguf_path, reuse_gguf_weights=True)
        package.save(str(tmp_path), progress_bar=False)

        source = GGUFModel(gguf_path)
        offset, length, _ = source.tensor_storage_range("blk.0.attn_q.weight")
        direct_payload = gguf_path.read_bytes()[offset : offset + length]
        sidecar = (tmp_path / "model.onnx.data").read_bytes()
        assert direct_payload not in sidecar

        reloaded = ir.load(tmp_path / "model.onnx")
        q_proj = reloaded.graph.initializers[
            "model.layers.0.self_attn.q_proj.weight"
        ].const_value
        assert isinstance(q_proj, ir.ExternalTensor)
        assert (q_proj.location, q_proj.offset, q_proj.length) == (
            "model.gguf",
            offset,
            length,
        )

    def test_rejects_non_flat_source_and_detects_identity_change(self, tmp_path: Path):
        from mobius.integrations.gguf import (
            build_from_gguf,
            verify_gguf_reuse_manifest,
        )

        source_dir = tmp_path / "source"
        source_dir.mkdir()
        gguf_path = source_dir / "model.gguf"
        _write_quantized_gguf(gguf_path, projection_quantization="f32")
        package = build_from_gguf(gguf_path, reuse_gguf_weights=True)

        with pytest.raises(ValueError, match="flat same-directory packaging"):
            package.save(str(tmp_path / "output"), progress_bar=False)

        package.save(str(source_dir), progress_bar=False)
        with gguf_path.open("r+b") as stream:
            stream.seek(-1, 2)
            byte = stream.read(1)
            stream.seek(-1, 2)
            stream.write(bytes([byte[0] ^ 0xFF]))
        with pytest.raises(ValueError, match="identity mismatch"):
            verify_gguf_reuse_manifest(source_dir)

    def test_does_not_reuse_same_size_dtype_cast(self, tmp_path: Path):
        from mobius.integrations.gguf import build_from_gguf

        gguf_path = tmp_path / "model.gguf"
        _write_quantized_gguf(
            gguf_path,
            projection_quantization="f16",
            float_type="f16",
        )
        with pytest.raises(ValueError, match="no byte-compatible tensors"):
            build_from_gguf(
                gguf_path,
                dtype="bf16",
                reuse_gguf_weights=True,
            )

    @pytest.mark.parametrize("artifact_name", ["model.onnx.data", ".gguf-reuse.lock"])
    def test_rejects_generated_artifact_name_collision(
        self, tmp_path: Path, artifact_name: str
    ):
        from mobius.integrations.gguf import build_from_gguf

        gguf_path = tmp_path / artifact_name
        _write_quantized_gguf(gguf_path, projection_quantization="f32")
        package = build_from_gguf(gguf_path, reuse_gguf_weights=True)
        with pytest.raises(ValueError, match="collides"):
            package.save(str(tmp_path), progress_bar=False)
        # Validation happens before any generated artifact can truncate the source.
        assert gguf_path.stat().st_size == package.gguf_reuse_plan.size

    @pytest.mark.parametrize("artifact_name", ["model.onnx.data", ".gguf-reuse.lock"])
    def test_rejects_hardlink_to_generated_artifact(self, tmp_path: Path, artifact_name: str):
        from mobius.integrations.gguf import build_from_gguf

        gguf_path = tmp_path / "source.gguf"
        _write_quantized_gguf(gguf_path, projection_quantization="f32")
        package = build_from_gguf(gguf_path, reuse_gguf_weights=True)
        os.link(gguf_path, tmp_path / artifact_name)

        with pytest.raises(ValueError, match="hard-linked"):
            package.save(str(tmp_path), progress_bar=False)
        assert gguf_path.stat().st_size == package.gguf_reuse_plan.size

    def test_generated_looking_files_without_journal_are_preserved(self, tmp_path: Path):
        from mobius.integrations.gguf import build_from_gguf

        token = "0" * 32
        gguf_path = tmp_path / f".model.onnx.{token}.tmp"
        unrelated = tmp_path / f".gguf-reuse.json.{token}.tmp"
        _write_quantized_gguf(gguf_path, projection_quantization="f32")
        source_bytes = gguf_path.read_bytes()
        unrelated.write_bytes(b"user-owned temporary data")

        build_from_gguf(gguf_path, reuse_gguf_weights=True).save(
            str(tmp_path), progress_bar=False
        )

        assert gguf_path.read_bytes() == source_bytes
        assert unrelated.read_bytes() == b"user-owned temporary data"

    def test_ordinary_resave_removes_stale_reuse_manifest(self, tmp_path: Path):
        from mobius._model_package import ModelPackage
        from mobius.integrations.gguf import build_from_gguf

        gguf_path = tmp_path / "source.gguf"
        _write_quantized_gguf(gguf_path, projection_quantization="f32")
        build_from_gguf(gguf_path, reuse_gguf_weights=True).save(
            str(tmp_path), progress_bar=False
        )

        loaded = ModelPackage.load(str(tmp_path))
        ordinary_output = tmp_path / "ordinary-resave"
        loaded.save(str(ordinary_output), progress_bar=False)
        assert not (ordinary_output / "gguf-reuse.json").exists()

    @pytest.mark.parametrize("projection_quantization", ["f32", "q4_0"])
    def test_reuse_transaction_removes_stale_optional_package_metadata(
        self,
        tmp_path: Path,
        projection_quantization: str,
    ) -> None:
        from mobius.integrations.gguf import build_from_gguf

        gguf_path = tmp_path / "source.gguf"
        _write_quantized_gguf(
            gguf_path,
            projection_quantization=projection_quantization,
        )
        package = build_from_gguf(gguf_path, reuse_gguf_weights=True)
        package.draft_manifest = {"architecture": "eagle3"}
        package.save(str(tmp_path), progress_bar=False)
        assert (tmp_path / "draft_manifest.json").is_file()
        assert (tmp_path / "export_report.json").is_file()
        assert (tmp_path / "quantization_report.json").is_file()

        package.draft_manifest = None
        package.export_report = None
        package.gguf_quantization_report = None
        package.save(str(tmp_path), progress_bar=False)

        assert not (tmp_path / "draft_manifest.json").exists()
        assert not (tmp_path / "export_report.json").exists()
        assert not (tmp_path / "quantization_report.json").exists()

    def test_verifier_rejects_unmanifested_external_initializer(self, tmp_path: Path):
        from mobius.integrations.gguf import (
            build_from_gguf,
            verify_gguf_reuse_manifest,
        )

        gguf_path = tmp_path / "source.gguf"
        _write_quantized_gguf(gguf_path, projection_quantization="f32")
        build_from_gguf(gguf_path, reuse_gguf_weights=True).save(
            str(tmp_path), progress_bar=False
        )
        model_path = tmp_path / "model.onnx"
        model = ir.load(model_path)
        initializer = next(
            value
            for value in model.graph.initializers.values()
            if not isinstance(value.const_value, ir.ExternalTensor)
        )
        tensor = initializer.const_value
        assert tensor is not None
        initializer.const_value = ir.ExternalTensor(
            "model.onnx.data",
            0,
            tensor.nbytes,
            tensor.dtype,
            shape=tensor.shape,
            name=tensor.name or initializer.name,
            base_dir=tmp_path,
        )
        ir.save(model, model_path)

        with pytest.raises(ValueError, match="Unmanifested sidecar initializer"):
            verify_gguf_reuse_manifest(tmp_path)

    def test_verifier_rejects_wrong_external_dtype(self, tmp_path: Path):
        from mobius.integrations.gguf import (
            build_from_gguf,
            verify_gguf_reuse_manifest,
        )

        gguf_path = tmp_path / "source.gguf"
        _write_quantized_gguf(gguf_path, projection_quantization="f32")
        build_from_gguf(gguf_path, reuse_gguf_weights=True).save(
            str(tmp_path), progress_bar=False
        )
        model_path = tmp_path / "model.onnx"
        model = ir.load(model_path)
        initializer = model.graph.initializers["model.embed_tokens.weight"]
        external = initializer.const_value
        assert isinstance(external, ir.ExternalTensor)
        initializer.const_value = ir.ExternalTensor(
            external.location,
            external.offset,
            external.length,
            ir.DataType.UINT8,
            shape=ir.Shape([external.length]),
            name=external.name,
            base_dir=tmp_path,
        )
        initializer.dtype = ir.DataType.UINT8
        initializer.shape = ir.Shape([external.length])
        ir.save(model, model_path)

        with pytest.raises(ValueError, match="incompatible dtype"):
            verify_gguf_reuse_manifest(tmp_path)

    def test_verifier_rejects_wrong_manifest_qtype(self, tmp_path: Path):
        from mobius.integrations.gguf import (
            build_from_gguf,
            verify_gguf_reuse_manifest,
        )

        gguf_path = tmp_path / "source.gguf"
        _write_quantized_gguf(gguf_path, projection_quantization="f32")
        build_from_gguf(gguf_path, reuse_gguf_weights=True).save(
            str(tmp_path), progress_bar=False
        )
        manifest_path = tmp_path / "gguf-reuse.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["reused_tensors"][0]["qtype"] = "F16"
        manifest_path.write_text(json.dumps(manifest))

        with pytest.raises(ValueError, match="does not match source tensor"):
            verify_gguf_reuse_manifest(tmp_path)

    @pytest.mark.parametrize("length_delta", [-1, 1])
    def test_verifier_rejects_wrong_sidecar_byte_length(
        self, tmp_path: Path, length_delta: int
    ):
        from mobius.integrations.gguf import (
            build_from_gguf,
            verify_gguf_reuse_manifest,
        )

        gguf_path = tmp_path / "source.gguf"
        _write_quantized_gguf(gguf_path, projection_quantization="f32")
        build_from_gguf(gguf_path, reuse_gguf_weights=True).save(
            str(tmp_path), progress_bar=False
        )
        model_path = tmp_path / "model.onnx"
        model = ir.load(model_path)
        initializer = next(
            value
            for value in model.graph.initializers.values()
            if isinstance(value.const_value, ir.ExternalTensor)
            and value.const_value.location == "model.onnx.data"
        )
        external = initializer.const_value
        assert isinstance(external, ir.ExternalTensor)
        assert external.length is not None
        initializer.const_value = ir.ExternalTensor(
            external.location,
            external.offset,
            external.length + length_delta,
            external.dtype,
            shape=external.shape,
            name=external.name,
            base_dir=tmp_path,
        )
        ir.save(model, model_path)

        with pytest.raises(ValueError, match="byte length"):
            verify_gguf_reuse_manifest(tmp_path)

    def test_save_and_verifier_do_not_run_behind_active_writer(self, tmp_path: Path):
        from mobius.integrations.gguf import (
            _reuse,
            build_from_gguf,
            verify_gguf_reuse_manifest,
        )

        gguf_path = tmp_path / "source.gguf"
        _write_quantized_gguf(gguf_path, projection_quantization="f32")
        package = build_from_gguf(gguf_path, reuse_gguf_weights=True)
        package.save(str(tmp_path), progress_bar=False)

        with _reuse._package_lock(tmp_path):
            operations = (
                lambda: verify_gguf_reuse_manifest(tmp_path),
                lambda: package.save(str(tmp_path), progress_bar=False),
            )
            for operation in operations:
                with pytest.raises(ValueError, match="locked by active writer"):
                    operation()

    @pytest.mark.parametrize(
        "artifact_name", [".gguf-reuse.lock", ".gguf-reuse.transaction.json"]
    )
    def test_rejects_dangling_control_artifact_symlink(
        self, tmp_path: Path, artifact_name: str
    ):
        from mobius.integrations.gguf import (
            _reuse,
            build_from_gguf,
            verify_gguf_reuse_manifest,
        )

        gguf_path = tmp_path / "source.gguf"
        _write_quantized_gguf(gguf_path, projection_quantization="f32")
        build_from_gguf(gguf_path, reuse_gguf_weights=True).save(
            str(tmp_path), progress_bar=False
        )
        artifact = tmp_path / artifact_name
        if artifact_name == _reuse._LOCK_NAME:
            artifact.unlink()
        artifact.symlink_to(tmp_path / "missing-target")

        with pytest.raises(ValueError, match="Unsafe GGUF"):
            if artifact_name == _reuse._LOCK_NAME:
                verify_gguf_reuse_manifest(tmp_path)
            else:
                build_from_gguf(gguf_path, reuse_gguf_weights=True).save(
                    str(tmp_path), progress_bar=False
                )
        assert artifact.is_symlink()

    @pytest.mark.parametrize(
        "artifact_name", ["model.onnx", "model.onnx.data", "gguf-reuse.json"]
    )
    def test_verifier_rejects_symlinked_package_artifact(
        self, tmp_path: Path, artifact_name: str
    ):
        from mobius.integrations.gguf import (
            build_from_gguf,
            verify_gguf_reuse_manifest,
        )

        gguf_path = tmp_path / "source.gguf"
        _write_quantized_gguf(gguf_path, projection_quantization="f32")
        build_from_gguf(gguf_path, reuse_gguf_weights=True).save(
            str(tmp_path), progress_bar=False
        )
        artifact = tmp_path / artifact_name
        moved = tmp_path / f"{artifact_name}.moved"
        artifact.replace(moved)
        artifact.symlink_to(moved)

        with pytest.raises(ValueError, match="Unsafe GGUF"):
            verify_gguf_reuse_manifest(tmp_path)

    def test_failed_rerun_restores_existing_package(self, tmp_path: Path, monkeypatch):
        from mobius.integrations.gguf import _reuse, build_from_gguf

        gguf_path = tmp_path / "source.gguf"
        _write_quantized_gguf(gguf_path, projection_quantization="f32")
        initial = build_from_gguf(gguf_path, reuse_gguf_weights=True)
        initial.draft_manifest = {"architecture": "eagle3", "generation": 1}
        initial.save(str(tmp_path), progress_bar=False)
        artifact_names = (
            "model.onnx",
            "model.onnx.data",
            "gguf-reuse.json",
            "quantization_report.json",
            "export_report.json",
            "draft_manifest.json",
        )
        original = {name: (tmp_path / name).read_bytes() for name in artifact_names}

        real_replace = _reuse.os.replace
        injected = False

        def fail_manifest_install(source, destination):
            nonlocal injected
            source_path = Path(source)
            destination_path = Path(destination)
            if (
                not injected
                and source_path.name.startswith(".gguf-reuse.json.")
                and destination_path.name == "gguf-reuse.json"
            ):
                injected = True
                raise OSError("injected manifest install failure")
            return real_replace(source, destination)

        monkeypatch.setattr(_reuse.os, "replace", fail_manifest_install)
        rerun = build_from_gguf(gguf_path, reuse_gguf_weights=True)
        rerun.draft_manifest = {"architecture": "eagle3", "generation": 2}
        with pytest.raises(OSError, match="injected"):
            rerun.save(str(tmp_path), progress_bar=False)

        assert injected
        assert {name: (tmp_path / name).read_bytes() for name in artifact_names} == original
        assert not list(tmp_path.glob(".*.tmp"))
        assert not list(tmp_path.glob(".*.backup"))

    def test_interrupted_rerun_recovers_from_transaction_journal(
        self, tmp_path: Path, monkeypatch
    ):
        from mobius.integrations.gguf import (
            _reuse,
            build_from_gguf,
            verify_gguf_reuse_manifest,
        )

        gguf_path = tmp_path / "source.gguf"
        _write_quantized_gguf(gguf_path, projection_quantization="f32")
        build_from_gguf(gguf_path, reuse_gguf_weights=True).save(
            str(tmp_path), progress_bar=False
        )
        artifact_names = ("model.onnx", "model.onnx.data", "gguf-reuse.json")
        original = {name: (tmp_path / name).read_bytes() for name in artifact_names}

        real_replace = _reuse.os.replace
        interrupted = False

        def interrupt_manifest_install(source, destination):
            nonlocal interrupted
            source_path = Path(source)
            destination_path = Path(destination)
            if (
                not interrupted
                and source_path.name.startswith(".gguf-reuse.json.")
                and destination_path.name == "gguf-reuse.json"
            ):
                interrupted = True
                raise KeyboardInterrupt
            return real_replace(source, destination)

        monkeypatch.setattr(_reuse.os, "replace", interrupt_manifest_install)
        rerun = build_from_gguf(gguf_path, reuse_gguf_weights=True)
        with pytest.raises(KeyboardInterrupt):
            rerun.save(str(tmp_path), progress_bar=False)
        monkeypatch.setattr(_reuse.os, "replace", real_replace)

        verify_gguf_reuse_manifest(tmp_path)
        assert {name: (tmp_path / name).read_bytes() for name in artifact_names} == original
        assert not (tmp_path / ".gguf-reuse.transaction.json").exists()
        assert not list(tmp_path.glob(".*.backup"))

    def test_committed_transaction_recovery_keeps_new_artifacts(self, tmp_path: Path):
        from mobius.integrations.gguf import _reuse

        token = "0" * 32
        managed = []
        for name in ("model.onnx", "model.onnx.data", "gguf-reuse.json"):
            final = tmp_path / name
            final.write_bytes(f"new {name}".encode())
            backup_name = f".{name}.{token}.backup"
            (tmp_path / backup_name).write_bytes(f"old {name}".encode())
            managed.append({"final": name, "backup": backup_name, "had_existing": True})
        journal = {
            "phase": "committed",
            "managed": managed,
            "staged": [
                f".model.onnx.{token}.tmp",
                f".gguf-reuse.json.{token}.tmp",
            ],
        }
        (tmp_path / ".gguf-reuse.transaction.json").write_text(json.dumps(journal))

        _reuse._recover_transaction(tmp_path)

        for name in ("model.onnx", "model.onnx.data", "gguf-reuse.json"):
            assert (tmp_path / name).read_bytes() == f"new {name}".encode()
        assert not list(tmp_path.glob(".*.backup"))
        assert not (tmp_path / ".gguf-reuse.transaction.json").exists()

    def test_verifier_rejects_wrong_transform_parameter(self, tmp_path: Path):
        from mobius.integrations.gguf import (
            build_from_gguf,
            verify_gguf_reuse_manifest,
        )

        gguf_path = tmp_path / "source.gguf"
        _write_quantized_gguf(gguf_path, projection_quantization="f32")
        build_from_gguf(gguf_path, reuse_gguf_weights=True).save(
            str(tmp_path), progress_bar=False
        )
        manifest_path = tmp_path / "gguf-reuse.json"
        manifest = json.loads(manifest_path.read_text())
        q_route = next(
            route
            for route in manifest["reused_tensors"]
            if route["transform"] == "llama_qk_permute"
        )
        q_route["transform_parameter"] = 3
        manifest_path.write_text(json.dumps(manifest))

        with pytest.raises(ValueError, match="Invalid Q/K head count"):
            verify_gguf_reuse_manifest(tmp_path)

    def test_verifier_rejects_wrong_transform_permutation(self, tmp_path: Path):
        from mobius.integrations.gguf import (
            build_from_gguf,
            verify_gguf_reuse_manifest,
        )

        gguf_path = tmp_path / "source.gguf"
        _write_quantized_gguf(gguf_path, projection_quantization="f32")
        build_from_gguf(gguf_path, reuse_gguf_weights=True).save(
            str(tmp_path), progress_bar=False
        )
        model_path = tmp_path / "model.onnx"
        model = ir.load(model_path)
        transpose = next(
            node
            for node in model.graph
            if node.name == "model.layers.0.self_attn.q_proj.weight.gguf_reuse.Transpose"
        )
        transpose.attributes["perm"] = ir.AttrInt64s("perm", [0, 1, 2, 3])
        ir.save(model, model_path)

        with pytest.raises(ValueError, match="Q/K permutation shapes are wrong"):
            verify_gguf_reuse_manifest(tmp_path)

    def test_overwrite_rejects_filesystem_without_hardlinks(self, tmp_path: Path, monkeypatch):
        from mobius.integrations.gguf import _reuse, build_from_gguf

        gguf_path = tmp_path / "source.gguf"
        _write_quantized_gguf(gguf_path, projection_quantization="f32")
        package = build_from_gguf(gguf_path, reuse_gguf_weights=True)
        package.save(str(tmp_path), progress_bar=False)
        original = (tmp_path / "model.onnx").read_bytes()

        def reject_link(source, destination):
            raise OSError("hard links unavailable")

        monkeypatch.setattr(_reuse.os, "link", reject_link)
        with pytest.raises(ValueError, match="requires same-directory hard-link"):
            package.save(str(tmp_path), progress_bar=False)
        assert (tmp_path / "model.onnx").read_bytes() == original
        assert not (tmp_path / ".gguf-reuse.transaction.json").exists()

    def test_recovery_rejects_unsafe_journal_paths(self, tmp_path: Path):
        from mobius.integrations.gguf import _reuse

        victim = tmp_path.parent / "victim"
        victim.write_bytes(b"keep")
        journal = {
            "managed": [
                {
                    "final": "../victim",
                    "backup": ".victim.00000000000000000000000000000000.backup",
                    "had_existing": True,
                },
                {
                    "final": "model.onnx.data",
                    "backup": ".model.onnx.data.00000000000000000000000000000000.backup",
                    "had_existing": False,
                },
                {
                    "final": "gguf-reuse.json",
                    "backup": ".gguf-reuse.json.00000000000000000000000000000000.backup",
                    "had_existing": False,
                },
            ],
            "staged": [
                ".model.onnx.00000000000000000000000000000000.tmp",
                ".gguf-reuse.json.00000000000000000000000000000000.tmp",
            ],
        }
        (tmp_path / ".gguf-reuse.transaction.json").write_text(json.dumps(journal))

        with pytest.raises(ValueError, match="Unsafe GGUF transaction"):
            _reuse._recover_transaction(tmp_path)
        assert victim.read_bytes() == b"keep"

    def test_transaction_removes_obsolete_sidecar(self, tmp_path: Path):
        from mobius.integrations.gguf import _reuse

        final_model = tmp_path / "model.onnx"
        final_sidecar = tmp_path / "model.onnx.data"
        final_manifest = tmp_path / "gguf-reuse.json"
        for path in (final_model, final_sidecar, final_manifest):
            path.write_bytes(b"old")
        staged_model = tmp_path / f".model.onnx.{'0' * 32}.tmp"
        staged_manifest = tmp_path / f".gguf-reuse.json.{'1' * 32}.tmp"
        staged_model.write_bytes(b"new model")
        staged_manifest.write_bytes(b"new manifest")

        _reuse._replace_artifacts(
            {
                final_model: staged_model,
                final_manifest: staged_manifest,
            },
            (final_model, final_sidecar, final_manifest),
        )

        assert final_model.read_bytes() == b"new model"
        assert final_manifest.read_bytes() == b"new manifest"
        assert not final_sidecar.exists()


class TestBuildQuantizedGguf:
    """Tests for the default quantization-preserving GGUF build."""

    def test_produces_model_package(self, q4_0_gguf: Path):
        """Quantized build returns a valid ModelPackage."""
        from mobius.integrations.gguf import build_from_gguf

        pkg = build_from_gguf(q4_0_gguf)
        assert "model" in pkg
        assert pkg["model"].graph is not None

    def test_model_has_matmulnbits_ops(self, q4_0_gguf: Path):
        """The API default uses MatMulNBits instead of float MatMul weights."""
        from mobius.integrations.gguf import build_from_gguf

        pkg = build_from_gguf(q4_0_gguf)
        model = pkg["model"]

        op_types = {node.op_type for node in model.graph if node.op_type}
        assert "MatMulNBits" in op_types, (
            f"Expected MatMulNBits in ops, got: {sorted(op_types)}"
        )

    def test_sharded_quantized_build_preserves_matmulnbits(self, tmp_path: Path):
        """A metadata-only primary and split Q4 payloads use the normal packed path."""
        from mobius.integrations.gguf import build_from_gguf
        from mobius.integrations.gguf._reader import GGUFModel

        stem = tmp_path / "split-q4.gguf"
        _write_quantized_gguf(
            stem,
            split_max_tensors=4,
            small_first_shard=True,
        )
        shards = sorted(tmp_path.glob("split-q4-*.gguf"))
        assert len(shards) > 1
        assert GGUFModel(shards[0]).num_tensors == 0

        model = build_from_gguf(shards[-1], keep_quantized=True)["model"]
        assert any(node.op_type == "MatMulNBits" for node in model.graph)

    def test_default_quantized_package_save_reload(self, q4_0_gguf: Path, tmp_path: Path):
        """Default quantized ops and weights survive ModelPackage persistence."""
        from mobius._model_package import ModelPackage
        from mobius.integrations.gguf import build_from_gguf

        output_dir = tmp_path / "saved"
        build_from_gguf(q4_0_gguf).save(str(output_dir), progress_bar=False)
        reloaded = ModelPackage.load(str(output_dir))

        op_types = {node.op_type for node in reloaded["model"].graph}
        assert "MatMulNBits" in op_types
        assert (
            reloaded["model"]
            .graph.initializers["model.layers.0.self_attn.q_proj.weight"]
            .const_value
            is not None
        )

    def test_decoder_backed_qtype_warns_and_persists_lossy_report(
        self, q5_1_gguf: Path, tmp_path: Path, caplog
    ) -> None:
        """A Q5_1 source stays packed but is reported as lossy target quantization."""
        from mobius._model_package import ModelPackage
        from mobius.integrations.gguf import QuantizationDisposition, build_from_gguf

        output_dir = tmp_path / "saved_q5_1"
        with caplog.at_level("WARNING"):
            package = build_from_gguf(q5_1_gguf)
        assert caplog.text.count("GGUF QUANTIZATION FIDELITY WARNING") == 1
        assert "Q5_1: 7 tensor(s)" in caplog.text
        assert "Losslessly preserved/repacked qtypes in this artifact: none" in caplog.text
        report = package.gguf_quantization_report
        assert report.target_storage_format == "INT4 affine block-32"
        assert report.storage_quantized is True
        assert report.source_fidelity is False
        assert {
            record.disposition for record in report.tensor_records if record.qtype == "Q5_1"
        } == {QuantizationDisposition.LOSSY_REQUANTIZE}
        assert any(node.op_type == "MatMulNBits" for node in package["model"].graph)
        package.save(str(output_dir), progress_bar=False)
        reloaded = ModelPackage.load(str(output_dir))
        assert (output_dir / "quantization_report.json").is_file()
        assert reloaded.gguf_quantization_report == report
        assert any(node.op_type == "MatMulNBits" for node in reloaded["model"].graph)

    def test_dequantize_reports_float_without_quantized_claim(self, q5_1_gguf: Path) -> None:
        from mobius.integrations.gguf import QuantizationDisposition, build_from_gguf

        package = build_from_gguf(q5_1_gguf, keep_quantized=False)
        report = package.gguf_quantization_report
        assert report.target_storage_format == "float"
        assert report.storage_quantized is False
        assert report.source_fidelity is False
        assert {
            record.disposition for record in report.tensor_records if record.qtype == "Q5_1"
        } == {QuantizationDisposition.DEQUANTIZED_FLOAT}
        assert all(node.op_type != "MatMulNBits" for node in package["model"].graph)

    def test_mixed_k_quant_census_warns_once_and_remains_packed(
        self, tmp_path: Path, caplog
    ) -> None:
        from mobius.integrations.gguf import QuantizationDisposition, build_from_gguf

        path = tmp_path / "mixed.gguf"
        _write_quantized_gguf(
            path,
            hidden_size=256,
            intermediate_size=256,
            vocab_size=256,
            num_heads=8,
            num_kv_heads=2,
            projection_quantization="q4_k",
            value_projection_quantization="q6_k",
            embedding_quantization="q5_k",
            output_quantization="q4_0",
        )
        with caplog.at_level("WARNING"):
            package = build_from_gguf(path)
        report = package.gguf_quantization_report
        assert caplog.text.count("GGUF QUANTIZATION FIDELITY WARNING") == 1
        assert all(name in caplog.text for name in ("Q4_K:", "Q5_K:", "Q6_K:"))
        assert "Losslessly preserved/repacked qtypes in this artifact: Q4_0:" in caplog.text
        assert report.converted_from == "Q4_K_M-like mixed GGUF"
        assert {
            record.qtype
            for record in report.tensor_records
            if record.disposition is QuantizationDisposition.LOSSY_REQUANTIZE
        } == {"Q4_K", "Q5_K", "Q6_K"}
        assert report.storage_quantized is True
        packed = [
            initializer
            for model in package.values()
            for initializer in model.graph.initializers.values()
            if initializer.dtype == ir.DataType.UINT8
        ]
        assert packed

    def test_float_only_default_uses_float_path(self, float_only_gguf: Path):
        """F32/BF16-only GGUFs do not fail when preservation is the default."""
        from mobius.integrations.gguf import build_from_gguf

        model = build_from_gguf(float_only_gguf)["model"]
        op_types = {node.op_type for node in model.graph}
        assert "MatMulNBits" not in op_types
        assert "GatherBlockQuantized" not in op_types
        assert "BlockQuantizedMatMul" not in op_types

    @pytest.mark.parametrize(
        "architecture",
        ["olmo", "olmo2", "cohere2", "arcee", "smollm3", "exaone"],
    )
    @pytest.mark.parametrize("projection_quantization", ["f32", "q4_0"])
    def test_dense_cohort_builds_complete_graphs(
        self,
        architecture: str,
        projection_quantization: str,
        tmp_path: Path,
    ) -> None:
        """Exact per-architecture tensor sets satisfy the full graph."""
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / f"{architecture}-{projection_quantization}.gguf"
        _write_quantized_gguf(
            path,
            architecture=architecture,
            projection_quantization=projection_quantization,
        )

        model = build_from_gguf(path)["model"]
        op_types = {node.op_type for node in model.graph}
        if projection_quantization == "q4_0":
            assert "MatMulNBits" in op_types
        else:
            assert "MatMulNBits" not in op_types

        output_dir = tmp_path / f"{architecture}-{projection_quantization}-onnx"
        build_from_gguf(path).save(output_dir, progress_bar=False)
        session = ort.InferenceSession(
            str(output_dir / "model.onnx"),
            providers=["CPUExecutionProvider"],
        )
        feeds = {
            "input_ids": np.array([[1, 2]], dtype=np.int64),
            "attention_mask": np.ones((1, 2), dtype=np.int64),
            "position_ids": np.array([[0, 1]], dtype=np.int64),
            "past_key_values.0.key": np.empty((1, 2, 0, 16), dtype=np.float32),
            "past_key_values.0.value": np.empty((1, 2, 0, 16), dtype=np.float32),
        }
        first = session.run(["logits"], feeds)[0]
        second = session.run(["logits"], feeds)[0]
        assert first.shape == (1, 2, 256)
        assert np.isfinite(first).all()
        np.testing.assert_array_equal(first, second)

    @pytest.mark.parametrize("projection_quantization", ["f32", "q4_0"])
    def test_dream_masked_diffusion_float_and_quantized_forward(
        self, projection_quantization: str, tmp_path: Path
    ) -> None:
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / f"dream-{projection_quantization}.gguf"
        quantized = projection_quantization == "q4_0"
        _write_quantized_gguf(
            path,
            architecture="dream",
            projection_quantization=projection_quantization,
            quantize_embedding=quantized,
            output_quantization=projection_quantization,
        )

        preserved = build_from_gguf(path)
        model = preserved["model"]
        op_types = {node.op_type for node in model.graph}
        assert ("MatMulNBits" in op_types) is quantized
        assert [value.name for value in model.graph.inputs] == ["input_ids"]
        assert not any(
            token in value.name
            for value in (*model.graph.inputs, *model.graph.outputs)
            for token in ("past", "present", "cache")
        )

        output_dir = tmp_path / f"dream-{projection_quantization}-onnx"
        preserved.save(output_dir, progress_bar=False)
        session = ort.InferenceSession(
            str(output_dir / "model.onnx"),
            providers=["CPUExecutionProvider"],
        )
        logits, proposed = session.run(
            None, {"input_ids": np.array([[1, 2, 255]], dtype=np.int64)}
        )
        assert logits.shape == (1, 3, 256)
        assert proposed.shape == (1, 3)
        assert np.isfinite(logits).all()
        np.testing.assert_array_equal(proposed, np.argmax(logits, axis=-1))

        if quantized:
            dequantized_dir = tmp_path / "dream-dequantized-onnx"
            build_from_gguf(path, keep_quantized=False).save(
                dequantized_dir, progress_bar=False
            )
            float_session = ort.InferenceSession(
                str(dequantized_dir / "model.onnx"),
                providers=["CPUExecutionProvider"],
            )
            float_logits = float_session.run(
                ["logits"], {"input_ids": np.array([[1, 2, 255]], dtype=np.int64)}
            )[0]
            np.testing.assert_allclose(logits, float_logits, rtol=0, atol=2e-2)

    @pytest.mark.parametrize("architecture", ["dream", "llada-moe", "rnd1"])
    @pytest.mark.parametrize(
        ("source_kind", "writer_kwargs"),
        [
            ("affine", {"projection_quantization": "q4_0"}),
            (
                "native",
                {"projection_quantization": "mxfp4"},
            ),
            (
                "mixed_float_fused",
                {
                    "projection_quantization": "q4_0",
                    "fused_qkv_float": True,
                },
            ),
        ],
    )
    def test_quantized_diffusion_fused_qkv_rejects_before_graph_build(
        self,
        architecture: str,
        source_kind: str,
        writer_kwargs: dict,
        tmp_path: Path,
        monkeypatch,
    ) -> None:
        import mobius._builder
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / f"{architecture}-fused-{source_kind}.gguf"
        if architecture == "dream":
            _write_quantized_gguf(
                path,
                architecture=architecture,
                fused_qkv=True,
                **writer_kwargs,
            )
        else:
            _write_moe_gguf(
                path,
                architecture,
                writer_kwargs["projection_quantization"],
                diffusion_fused_qkv=True,
                fused_qkv_float=writer_kwargs.get("fused_qkv_float", False),
            )
        monkeypatch.setattr(
            mobius._builder,
            "build_from_module",
            lambda *_args, **_kwargs: pytest.fail("graph construction must not run"),
        )

        with pytest.raises(
            ValueError,
            match=rf"Quantization-preserving import of fused QKV.*{architecture}",
        ) as error:
            build_from_gguf(path)
        assert "keep_quantized=False" in str(error.value)

    @pytest.mark.parametrize("architecture", ["dream", "llada-moe", "rnd1"])
    @pytest.mark.parametrize(
        ("projection_quantization", "keep_quantized"),
        [("f32", True), ("q4_0", False)],
    )
    def test_float_diffusion_fused_qkv_still_builds_and_runs(
        self,
        architecture: str,
        projection_quantization: str,
        keep_quantized: bool,
        tmp_path: Path,
    ) -> None:
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / f"{architecture}-fused-{projection_quantization}.gguf"
        if architecture == "dream":
            _write_quantized_gguf(
                path,
                architecture=architecture,
                projection_quantization=projection_quantization,
                fused_qkv=True,
            )
        else:
            _write_moe_gguf(
                path,
                architecture,
                projection_quantization,
                diffusion_fused_qkv=True,
                fused_qkv_float=projection_quantization == "f32",
            )

        package = build_from_gguf(path, keep_quantized=keep_quantized)
        model = package["model"]
        assert "MatMulNBits" not in {node.op_type for node in model.graph}
        output_dir = tmp_path / f"{architecture}-fused-{projection_quantization}-onnx"
        package.save(output_dir, progress_bar=False)
        session = ort.InferenceSession(
            str(output_dir / "model.onnx"),
            providers=["CPUExecutionProvider"],
        )
        logits, proposed = session.run(
            None, {"input_ids": np.array([[1, 2, 255]], dtype=np.int64)}
        )
        assert logits.shape == (1, 3, 256)
        np.testing.assert_array_equal(proposed, np.argmax(logits, axis=-1))

    @pytest.mark.parametrize(
        "architecture", ["olmoe", "phimoe", "qwen2moe", "qwen3moe", "granitemoe"]
    )
    @pytest.mark.parametrize("projection_quantization", ["f32", "q4_0"])
    def test_moe_cohort_builds_complete_graphs(
        self,
        architecture: str,
        projection_quantization: str,
        tmp_path: Path,
    ) -> None:
        """All routed/shared expert tensors survive build, save, load, and ORT."""
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / f"{architecture}-{projection_quantization}.gguf"
        _write_moe_gguf(path, architecture, projection_quantization)
        package = build_from_gguf(path)
        model = package["model"]
        initializer_names = set(model.graph.initializers)

        def has_weight(stem: str) -> bool:
            return (
                f"{stem}.weight" in initializer_names
                or f"{stem}.weight_t" in initializer_names
            )

        for expert in range(4):
            for projection in ("gate_proj", "up_proj", "down_proj"):
                assert has_weight(f"model.layers.0.mlp.experts.{expert}.{projection}")
        assert has_weight("model.layers.0.mlp.gate")
        if architecture in {"qwen2moe", "granitemoe"}:
            assert has_weight("model.layers.0.mlp.shared_expert.gate_proj")
            assert has_weight("model.layers.0.mlp.shared_expert.up_proj")
            assert has_weight("model.layers.0.mlp.shared_expert.down_proj")

        op_types = {node.op_type for node in model.graph}
        if projection_quantization == "q4_0":
            assert "MatMulNBits" in op_types
            assert not any("fc1_experts" in name for name in initializer_names)
        else:
            assert "MatMulNBits" not in op_types

        output_dir = tmp_path / f"{architecture}-{projection_quantization}-onnx"
        package.save(output_dir, progress_bar=False)
        session = ort.InferenceSession(
            str(output_dir / "model.onnx"),
            providers=["CPUExecutionProvider"],
        )
        feeds = {
            "input_ids": np.array([[1, 2]], dtype=np.int64),
            "attention_mask": np.ones((1, 2), dtype=np.int64),
            "position_ids": np.array([[0, 1]], dtype=np.int64),
            "past_key_values.0.key": np.empty((1, 2, 0, 16), dtype=np.float32),
            "past_key_values.0.value": np.empty((1, 2, 0, 16), dtype=np.float32),
        }
        first = session.run(["logits"], feeds)[0]
        second = session.run(["logits"], feeds)[0]
        assert first.shape == (1, 2, 256)
        assert np.isfinite(first).all()
        np.testing.assert_array_equal(first, second)

    @pytest.mark.parametrize("architecture", ["bailingmoe", "deepseek", "dots1"])
    @pytest.mark.parametrize("projection_quantization", ["f32", "q4_0"])
    def test_promoted_shared_swiglu_moe_builds_and_runs(
        self,
        architecture: str,
        projection_quantization: str,
        tmp_path: Path,
    ) -> None:
        """The promoted expert banks, router, and shared branch survive import."""
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / f"{architecture}-{projection_quantization}.gguf"
        _write_moe_gguf(path, architecture, projection_quantization)
        package = build_from_gguf(path)
        model = package["model"]
        names = set(model.graph.initializers)

        expert_container = "mlp.experts" if architecture == "bailingmoe" else "mlp.moe.experts"
        shared_container = (
            "mlp.shared_expert" if architecture == "bailingmoe" else "mlp.shared_experts"
        )
        for expert in range(4):
            for projection in ("gate_proj", "up_proj", "down_proj"):
                stem = f"model.layers.0.{expert_container}.{expert}.{projection}"
                assert f"{stem}.weight" in names or f"{stem}.weight_t" in names
        for projection in ("gate_proj", "up_proj", "down_proj"):
            stem = f"model.layers.0.{shared_container}.{projection}"
            assert f"{stem}.weight" in names or f"{stem}.weight_t" in names

        output_dir = tmp_path / "onnx"
        package.save(output_dir, progress_bar=False)
        session = ort.InferenceSession(
            str(output_dir / "model.onnx"), providers=["CPUExecutionProvider"]
        )
        logits = session.run(
            ["logits"],
            {
                "input_ids": np.array([[1, 2]], dtype=np.int64),
                "attention_mask": np.ones((1, 2), dtype=np.int64),
                "position_ids": np.array([[0, 1]], dtype=np.int64),
                "past_key_values.0.key": np.empty(
                    (1, 4 if architecture == "dots1" else 2, 0, 16),
                    dtype=np.float32,
                ),
                "past_key_values.0.value": np.empty(
                    (1, 4 if architecture == "dots1" else 2, 0, 16),
                    dtype=np.float32,
                ),
            },
        )[0]
        assert logits.shape == (1, 2, 256)
        assert np.isfinite(logits).all()

    @pytest.mark.parametrize("architecture", ["bailingmoe", "deepseek", "dots1"])
    @pytest.mark.parametrize("projection_quantization", ["f32", "q4_0"])
    def test_promoted_moe_fused_biased_qkv_builds_and_runs(
        self,
        architecture: str,
        projection_quantization: str,
        tmp_path: Path,
    ) -> None:
        import torch

        from mobius.integrations.gguf import build_from_gguf
        from mobius.integrations.gguf._reader import GGUFModel
        from mobius.integrations.gguf._repacker import repack_gguf_tensor
        from mobius.integrations.gguf._tensor_processors import _reverse_permute

        path = tmp_path / f"{architecture}-{projection_quantization}-fused-qkv.gguf"
        _write_moe_gguf(
            path,
            architecture,
            projection_quantization,
            phi_fused_qkv=True,
        )
        package = build_from_gguf(path)
        model = package["model"]
        names = set(model.graph.initializers)
        source = GGUFModel(path)
        raw, qtype, shape = next(
            (raw, qtype, shape)
            for name, raw, qtype, shape in source.tensor_items_raw()
            if name == "blk.0.attn_qkv.weight"
        )
        if projection_quantization == "f32":
            fused_weight = torch.from_numpy(
                np.array(source.get_tensor("blk.0.attn_qkv.weight"))
            )
            fused_scales = fused_zero_points = None
        else:
            repacked = repack_gguf_tensor(raw, qtype.value, shape)
            fused_weight = torch.from_numpy(repacked.weight)
            fused_scales = torch.from_numpy(repacked.scales)
            fused_zero_points = torch.from_numpy(repacked.zero_points)

        offset = 0
        kv_rows = 64 if architecture == "dots1" else 32
        kv_heads = 4 if architecture == "dots1" else 2
        for projection, rows, heads in (
            ("q", 64, 4),
            ("k", kv_rows, kv_heads),
            ("v", kv_rows, None),
        ):
            stem = f"model.layers.0.self_attn.{projection}_proj"
            assert f"{stem}.weight" in names or f"{stem}.weight_t" in names
            assert f"{stem}.bias" in names
            end = offset + rows
            expected_weight = fused_weight[offset:end]
            if heads is not None and architecture != "dots1":
                expected_weight = _reverse_permute(expected_weight, heads)
            if projection_quantization == "f32":
                actual_weight = (
                    model.graph.initializers[f"{stem}.weight_t"].const_value.numpy().T
                )
            else:
                actual_weight = model.graph.initializers[f"{stem}.weight"].const_value.numpy()
            np.testing.assert_array_equal(actual_weight, expected_weight.numpy())

            if fused_scales is not None and fused_zero_points is not None:
                expected_scales = fused_scales[offset:end]
                expected_zero_points = fused_zero_points[offset:end]
                if heads is not None and architecture != "dots1":
                    expected_scales = _reverse_permute(expected_scales, heads)
                    expected_zero_points = _reverse_permute(expected_zero_points, heads)
                np.testing.assert_array_equal(
                    model.graph.initializers[f"{stem}.scales"].const_value.numpy(),
                    expected_scales.numpy(),
                )
                np.testing.assert_array_equal(
                    model.graph.initializers[f"{stem}.zero_points"].const_value.numpy(),
                    expected_zero_points.numpy(),
                )
            offset = end

    @pytest.mark.parametrize(
        ("quantize_embedding", "quantize_output"),
        [(True, True), (False, False)],
    )
    def test_deepseek_quantization_flags_match_serialized_tensors(
        self,
        quantize_embedding: bool,
        quantize_output: bool,
        tmp_path: Path,
    ) -> None:
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / "deepseek-mixed-q4.gguf"
        _write_moe_gguf(
            path,
            "deepseek",
            "q4_0",
            quantize_tied_embedding=quantize_embedding,
            output_quantization="q4_0" if quantize_output else "f32",
        )
        package = build_from_gguf(path)
        names = set(package["model"].graph.initializers)
        assert ("model.embed_tokens.scales" in names) is quantize_embedding
        assert ("lm_head.scales" in names) is quantize_output

    def test_deepseek_tied_quantized_head_reuses_embedding_storage(
        self, tmp_path: Path
    ) -> None:
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / "deepseek-tied-q4.gguf"
        _write_moe_gguf(
            path,
            "deepseek",
            "q4_0",
            quantize_tied_embedding=True,
            include_output=False,
        )
        package = build_from_gguf(path)
        model = package["model"]
        assert model.graph.initializers["model.embed_tokens.qweight"].const_value is not None
        assert not any(name.startswith("lm_head.") for name in model.graph.initializers)
        package.save(tmp_path / "onnx", progress_bar=False)

    @pytest.mark.parametrize("architecture", ["llada-moe", "rnd1"])
    @pytest.mark.parametrize("projection_quantization", ["f32", "q4_0"])
    def test_diffusion_moe_cohort_builds_and_runs_masked_forward(
        self,
        architecture: str,
        projection_quantization: str,
        tmp_path: Path,
    ) -> None:
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / f"{architecture}-{projection_quantization}.gguf"
        _write_moe_gguf(path, architecture, projection_quantization)
        package = build_from_gguf(path)
        model = package["model"]
        op_types = {node.op_type for node in model.graph}
        assert ("MatMulNBits" in op_types) is (projection_quantization == "q4_0")
        assert [value.name for value in model.graph.inputs] == ["input_ids"]
        assert not any(
            token in value.name
            for value in (*model.graph.inputs, *model.graph.outputs)
            for token in ("past", "present", "cache")
        )

        output_dir = tmp_path / f"{architecture}-{projection_quantization}-onnx"
        package.save(output_dir, progress_bar=False)
        session = ort.InferenceSession(
            str(output_dir / "model.onnx"),
            providers=["CPUExecutionProvider"],
        )
        feeds = {"input_ids": np.array([[1, 2, 255]], dtype=np.int64)}
        first_logits, first_proposal = session.run(None, feeds)
        second_logits, second_proposal = session.run(None, feeds)
        assert first_logits.shape == (1, 3, 256)
        assert first_proposal.shape == (1, 3)
        assert np.isfinite(first_logits).all()
        np.testing.assert_array_equal(first_logits, second_logits)
        np.testing.assert_array_equal(first_proposal, second_proposal)

    @pytest.mark.parametrize("projection_quantization", ["f32", "q4_0"])
    def test_granitemoe_qk_rows_are_reverse_permuted_by_value(
        self, projection_quantization: str, tmp_path: Path
    ) -> None:
        import torch

        from mobius.integrations.gguf import build_from_gguf
        from mobius.integrations.gguf._reader import GGUFModel
        from mobius.integrations.gguf._repacker import repack_gguf_tensor
        from mobius.integrations.gguf._tensor_processors import _reverse_permute

        path = tmp_path / f"granitemoe-qk-{projection_quantization}.gguf"
        _write_moe_gguf(path, "granitemoe", projection_quantization)
        source = GGUFModel(path)
        model = build_from_gguf(path)["model"]

        raw_tensors = {
            name: (raw, qtype, tuple(int(dim) for dim in shape))
            for name, raw, qtype, shape in source.tensor_items_raw()
        }
        for gguf_name, projection, heads in (
            ("blk.0.attn_q.weight", "q_proj", 4),
            ("blk.0.attn_k.weight", "k_proj", 2),
        ):
            raw, qtype, shape = raw_tensors[gguf_name]
            if projection_quantization == "f32":
                unpermuted = torch.from_numpy(np.array(source.get_tensor(gguf_name)))
            else:
                packed = repack_gguf_tensor(
                    raw.ravel().view(np.uint8),
                    qtype.value,
                    shape,
                )
                unpermuted = torch.from_numpy(packed.weight)
            expected = _reverse_permute(unpermuted, heads)
            stem = f"model.layers.0.self_attn.{projection}"
            if projection_quantization == "f32":
                actual = model.graph.initializers[f"{stem}.weight_t"].const_value.numpy().T
            else:
                actual = model.graph.initializers[f"{stem}.weight"].const_value.numpy()
            np.testing.assert_array_equal(actual, expected.numpy())
            assert not torch.equal(expected, unpermuted)

    def test_granitemoe_zero_experts_selects_dense_graph(self, tmp_path: Path) -> None:
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / "granitemoe-dense.gguf"
        _write_quantized_gguf(
            path,
            architecture="granitemoe",
            projection_quantization="q4_0",
            tie_embeddings=True,
            quantize_embedding=True,
        )
        model = build_from_gguf(path, keep_quantized=False)["model"]
        names = set(model.graph.initializers)
        assert any(".mlp.gate_proj.weight" in name for name in names)
        assert not any(".mlp.experts." in name for name in names)
        assert not any(".mlp.gate.weight" in name for name in names)
        assert "model.embed_tokens.weight" in names
        assert not any(name.startswith("lm_head.") for name in names)

    @pytest.mark.parametrize("projection_quantization", ["f32", "q4_0"])
    def test_phimoe_fused_qkv_is_split_without_loss(
        self, projection_quantization: str, tmp_path: Path
    ) -> None:
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / f"phimoe-fused-{projection_quantization}.gguf"
        _write_moe_gguf(
            path,
            "phimoe",
            projection_quantization,
            phi_fused_qkv=True,
        )
        model = build_from_gguf(path)["model"]
        names = set(model.graph.initializers)
        for projection in ("q_proj", "k_proj", "v_proj"):
            assert any(
                f"model.layers.0.self_attn.{projection}.weight" in name for name in names
            )
            assert any(f"model.layers.0.self_attn.{projection}.bias" in name for name in names)
        assert not any("qkv_proj" in name for name in names)
        if projection_quantization == "q4_0":
            from mobius.integrations.gguf._reader import GGUFModel
            from mobius.integrations.gguf._repacker import repack_gguf_tensor

            gguf_model = GGUFModel(path)
            raw, qtype, shape = next(
                (raw, qtype, shape)
                for name, raw, qtype, shape in gguf_model.tensor_items_raw()
                if name == "blk.0.attn_qkv.weight"
            )
            repacked = repack_gguf_tensor(raw, qtype.value, shape)
            offset = 0
            for projection, rows in (("q_proj", 64), ("k_proj", 32), ("v_proj", 32)):
                stem = f"model.layers.0.self_attn.{projection}"
                end = offset + rows
                np.testing.assert_array_equal(
                    model.graph.initializers[f"{stem}.weight"].const_value.numpy(),
                    repacked.weight[offset:end],
                )
                np.testing.assert_array_equal(
                    model.graph.initializers[f"{stem}.scales"].const_value.numpy(),
                    repacked.scales[offset:end],
                )
                np.testing.assert_array_equal(
                    model.graph.initializers[f"{stem}.zero_points"].const_value.numpy(),
                    repacked.zero_points[offset:end],
                )
                offset = end

    @pytest.mark.parametrize("output_quantization", ["f32", "q4_0"])
    def test_phimoe_output_head_honors_storage_and_values(
        self, output_quantization: str, tmp_path: Path
    ) -> None:
        from mobius.integrations.gguf import build_from_gguf
        from mobius.integrations.gguf._reader import GGUFModel
        from mobius.integrations.gguf._repacker import repack_gguf_tensor

        path = tmp_path / f"phimoe-output-{output_quantization}.gguf"
        _write_moe_gguf(
            path,
            "phimoe",
            "q4_0",
            output_quantization=output_quantization,
        )
        source = GGUFModel(path)
        package = build_from_gguf(path, keep_quantized=True)
        model = package["model"]
        raw, qtype, shape = next(
            (raw, qtype, shape)
            for name, raw, qtype, shape in source.tensor_items_raw()
            if name == "output.weight"
        )

        if output_quantization == "q4_0":
            assert package.config.quantization is not None
            assert package.config.quantization.quantize_lm_head
            repacked = repack_gguf_tensor(raw, qtype.value, shape)
            np.testing.assert_array_equal(
                model.graph.initializers["lm_head.weight"].const_value.numpy(),
                repacked.weight,
            )
            np.testing.assert_array_equal(
                model.graph.initializers["lm_head.scales"].const_value.numpy(),
                repacked.scales,
            )
        else:
            assert package.config.quantization is not None
            assert not package.config.quantization.quantize_lm_head
            assert "lm_head.scales" not in model.graph.initializers
            np.testing.assert_array_equal(
                model.graph.initializers["lm_head.weight_t"].const_value.numpy(),
                source.get_tensor("output.weight").T,
            )

    @pytest.mark.parametrize("architecture", ["qwen3moe", "granitemoe"])
    def test_tied_quantized_embedding_is_shared_with_output_head(
        self, architecture: str, tmp_path: Path
    ) -> None:
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / f"{architecture}-tied-q4.gguf"
        _write_moe_gguf(
            path,
            architecture,
            "q4_0",
            quantize_tied_embedding=True,
        )
        package = build_from_gguf(path)
        model = package["model"]
        op_types = [node.op_type for node in model.graph]
        assert op_types.count("GatherBlockQuantized") == 1
        assert "MatMulNBits" in op_types
        assert "model.embed_tokens.qweight" in model.graph.initializers
        assert not any(name.startswith("lm_head.") for name in model.graph.initializers)
        output_dir = tmp_path / f"{architecture}-tied-q4-onnx"
        package.save(output_dir, progress_bar=False)
        session = ort.InferenceSession(
            str(output_dir / "model.onnx"),
            providers=["CPUExecutionProvider"],
        )
        logits = session.run(
            ["logits"],
            {
                "input_ids": np.array([[1, 2]], dtype=np.int64),
                "attention_mask": np.ones((1, 2), dtype=np.int64),
                "position_ids": np.array([[0, 1]], dtype=np.int64),
                "past_key_values.0.key": np.empty((1, 2, 0, 16), dtype=np.float32),
                "past_key_values.0.value": np.empty((1, 2, 0, 16), dtype=np.float32),
            },
        )[0]
        assert logits.shape == (1, 2, 256)
        assert np.isfinite(logits).all()

    @pytest.mark.parametrize("suffix", ["scale", "input_scale"])
    def test_qwen3moe_auxiliary_expert_scales_are_rejected_before_build(
        self, suffix: str, tmp_path: Path, monkeypatch
    ) -> None:
        from mobius import _builder as core_builder
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / f"qwen3moe-{suffix}.gguf"
        _write_moe_gguf(path, "qwen3moe", "q4_0", expert_scale_suffix=suffix)

        graph_build_started = False

        def unexpected_graph_build(*args, **kwargs):
            nonlocal graph_build_started
            graph_build_started = True
            raise AssertionError("graph construction must not start")

        monkeypatch.setattr(core_builder, "build_from_module", unexpected_graph_build)
        with pytest.raises(ValueError, match="cannot represent GGUF scale/input_scale"):
            build_from_gguf(path)
        assert not graph_build_started

    def test_malformed_qwen3moe_expert_scale_is_rejected(self, tmp_path: Path) -> None:
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / "qwen3moe-malformed-scale.gguf"
        _write_moe_gguf(
            path,
            "qwen3moe",
            "q4_0",
            expert_scale_suffix="scale",
            malformed_expert_scale=True,
        )
        with pytest.raises(ValueError, match=r"expected shape \(4,\), got \(3,\)"):
            build_from_gguf(path)

    def test_qwen3moe_optional_expert_scales_may_be_absent(self, tmp_path: Path) -> None:
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / "qwen3moe-no-scale.gguf"
        _write_moe_gguf(path, "qwen3moe", "q4_0")
        assert build_from_gguf(path)["model"].graph.num_nodes() > 0

    def test_q4_0_matmulnbits_has_explicit_zero_points(self, q4_0_gguf: Path):
        """GGUF Q4_0 projections explicitly encode zp=8 instead of EP defaults."""
        from mobius.integrations.gguf import build_from_gguf

        model = build_from_gguf(q4_0_gguf, keep_quantized=True)["model"]
        nodes = [node for node in model.graph if node.op_type == "MatMulNBits"]
        assert nodes
        for node in nodes:
            assert len(node.inputs) == 4
            zero_point_name = node.inputs[3].name
            assert zero_point_name.endswith(".zero_points")
            zero_points = model.graph.initializers[zero_point_name]
            np.testing.assert_array_equal(zero_points.const_value.numpy(), 0x88)

    def test_native_blocks_emit_block_quantized_matmul_and_preserve_bytes(
        self,
        native_block_gguf: tuple[Path, str, int, int],
    ):
        """Runtime-native IQ/MXFP4 projections retain their exact GGUF bytes."""
        import onnx_ir as ir

        from mobius.integrations.gguf import build_from_gguf

        path, format_name, block_elements, block_bytes = native_block_gguf
        model = build_from_gguf(path, keep_quantized=True)["model"]
        nodes = [node for node in model.graph if node.op_type == "BlockQuantizedMatMul"]
        assert len(nodes) == 7
        assert all(node.domain == "pkg.nxrt" for node in nodes)
        for node in nodes:
            attrs = {attribute.name: attribute.value for attribute in node.attributes.values()}
            assert attrs["format"] == format_name
            assert attrs["block_layout_version"] == 1
            assert attrs["K"] == 256
            assert attrs["N"] == 256

        weight = model.graph.initializers["model.layers.0.self_attn.o_proj.weight"]
        assert weight.dtype == ir.DataType.UINT8
        n_blocks = (256 + block_elements - 1) // block_elements
        assert list(weight.shape) == [256, n_blocks, block_bytes]
        expected = np.arange(256 * n_blocks * block_bytes, dtype=np.uint8).reshape(
            256, n_blocks, block_bytes
        )
        np.testing.assert_array_equal(weight.const_value.numpy(), expected)
        assert "model.layers.0.self_attn.o_proj.scales" not in model.graph.initializers

        assert model.graph.opset_imports["pkg.nxrt"] == 1
        proto = ir.serde.serialize_model(model)
        imports = {opset.domain: opset.version for opset in proto.opset_import}
        assert imports["pkg.nxrt"] == 1

    def test_mixed_native_quantization_reports_lossy_affine_normalization(
        self, mixed_native_q5_q8_gguf: Path, caplog
    ):
        """Native IQ bytes coexist with explicitly reported Q5/Q8 normalization."""
        from mobius.integrations.gguf import QuantizationDisposition, build_from_gguf

        with caplog.at_level("WARNING"):
            package = build_from_gguf(mixed_native_q5_q8_gguf, keep_quantized=True)
        report = package.gguf_quantization_report
        assert caplog.text.count("GGUF QUANTIZATION FIDELITY WARNING") == 1
        assert {
            record.qtype
            for record in report.tensor_records
            if record.disposition is QuantizationDisposition.LOSSY_REQUANTIZE
        } == {"Q5_1", "Q8_0"}
        assert "native GGUF block storage" in report.target_storage_format

    def test_quantized_embedding_uses_gatherblockquantized(self, q4_0_embedding_gguf: Path):
        """A quantized GGUF embedding remains packed in the ONNX graph."""
        import onnx_ir as ir

        from mobius.integrations.gguf import build_from_gguf

        model = build_from_gguf(q4_0_embedding_gguf, keep_quantized=True)["model"]
        gather_nodes = [node for node in model.graph if node.op_type == "GatherBlockQuantized"]
        assert len(gather_nodes) == 1
        assert gather_nodes[0].domain == "com.microsoft"
        assert len(gather_nodes[0].inputs) == 4

        qweight = model.graph.initializers["model.embed_tokens.qweight"]
        assert qweight.dtype == ir.DataType.UINT8
        assert list(qweight.shape) == [256, 32]
        assert list(model.graph.initializers["model.embed_tokens.scales"].shape) == [
            256,
            2,
        ]
        zero_points = model.graph.initializers["model.embed_tokens.zero_points"]
        assert zero_points.dtype == ir.DataType.UINT8
        assert list(zero_points.shape) == [256, 1]
        np.testing.assert_array_equal(zero_points.const_value.numpy(), 0x88)
        assert "model.embed_tokens.weight" not in model.graph.initializers

    def test_gatherblockquantized_zero_point_dequantizes_q4_0(self, tmp_path: Path):
        """GatherBlockQuantized output must match GGUF Q4_0's ``(q - 8) * scale``."""
        actual = _run_gather_block_quantized(tmp_path, zero_point=0x08).astype(np.float32)
        expected = np.stack(
            [
                np.full(32, (10 - 8) * 0.5, dtype=np.float32),
                np.full(32, (10 - 8) * 0.25, dtype=np.float32),
            ]
        )
        np.testing.assert_allclose(actual, expected)
        wrong = _run_gather_block_quantized(tmp_path, zero_point=0x00).astype(np.float32)
        assert not np.allclose(wrong, expected)

    def test_native_projection_abi_embedding_is_lossily_normalized(
        self, iq4_nl_embedding_gguf: Path, caplog
    ) -> None:
        from mobius.integrations.gguf import QuantizationDisposition, build_from_gguf

        with caplog.at_level("WARNING"):
            package = build_from_gguf(iq4_nl_embedding_gguf)
        report = package.gguf_quantization_report
        assert "IQ4_NL:" in caplog.text
        assert any(
            record.qtype == "IQ4_NL"
            and record.disposition is QuantizationDisposition.LOSSY_REQUANTIZE
            for record in report.tensor_records
        )
        assert any(node.op_type == "GatherBlockQuantized" for node in package["model"].graph)

        model = build_from_gguf(iq4_nl_embedding_gguf, keep_quantized=False)["model"]
        assert all(node.op_type != "GatherBlockQuantized" for node in model.graph)
        assert "model.embed_tokens.weight" in model.graph.initializers

    def test_tied_quantized_embedding_drives_matmulnbits_head(
        self, q4_0_tied_embedding_gguf: Path
    ):
        """Tied embedding/head share one packed table across both contrib ops."""
        from mobius.integrations.gguf import build_from_gguf

        model = build_from_gguf(q4_0_tied_embedding_gguf, keep_quantized=True)["model"]
        op_types = [node.op_type for node in model.graph]
        assert op_types.count("GatherBlockQuantized") == 1
        assert "MatMulNBits" in op_types
        assert "model.embed_tokens.qweight" in model.graph.initializers
        assert "model.embed_tokens.zero_points" in model.graph.initializers
        assert not any(name.startswith("lm_head.") for name in model.graph.initializers)

    def test_untied_head_with_incompatible_affine_target_is_reported(
        self, q4_0_embedding_q8_head_gguf: Path, caplog
    ):
        """An untied Q8 head is visibly requantized to the graph's Q4 layout."""
        from mobius.integrations.gguf import QuantizationDisposition, build_from_gguf

        with caplog.at_level("WARNING"):
            package = build_from_gguf(q4_0_embedding_q8_head_gguf, keep_quantized=True)
        assert "Q8_0: 1 tensor(s)" in caplog.text
        assert any(
            record.name == "output.weight"
            and record.disposition is QuantizationDisposition.LOSSY_REQUANTIZE
            for record in package.gguf_quantization_report.tensor_records
        )

    def test_mixed_affine_targets_select_supported_int4_normalization(
        self, q8_0_projection_q4_head_gguf: Path, caplog
    ):
        """Mixed affine targets normalize to the supported INT4 target."""
        from mobius.integrations.gguf import build_from_gguf

        with caplog.at_level("WARNING"):
            package = build_from_gguf(q8_0_projection_q4_head_gguf, keep_quantized=True)
        assert package.gguf_quantization_report.target_storage_format == (
            "INT4 affine block-32"
        )
        assert "Q8_0:" in caplog.text

    def test_quantized_source_float_output_head_dequantizes_consistently(
        self, tmp_path: Path
    ) -> None:
        from mobius.integrations.gguf import QuantizationDisposition, build_from_gguf

        path = tmp_path / "olmo2-q6-output.gguf"
        _write_quantized_gguf(
            path,
            architecture="olmo2",
            hidden_size=256,
            intermediate_size=256,
            vocab_size=256,
            num_heads=8,
            num_kv_heads=2,
            projection_quantization="q4_k",
            output_quantization="q6_k",
        )

        package = build_from_gguf(path, keep_quantized=True)
        report_record = next(
            record
            for record in package.gguf_quantization_report.tensor_records
            if record.name == "output.weight"
        )
        assert report_record.disposition is QuantizationDisposition.DEQUANTIZED_FLOAT
        assert report_record.target_storage == "float"
        head_initializers = [
            value
            for name, value in package["model"].graph.initializers.items()
            if name.startswith("lm_head.weight")
        ]
        assert len(head_initializers) == 1
        assert head_initializers[0].dtype == ir.DataType.FLOAT

    @pytest.mark.parametrize(
        ("native_2bit", "expected_bits"),
        [(False, 4), (True, 2)],
    )
    def test_tencent_q1_0_preflight_uses_layout_and_exact_payload_bytes(
        self,
        tmp_path: Path,
        native_2bit: bool,
        expected_bits: int,
    ) -> None:
        from mobius._flags import override_flags
        from mobius.integrations.gguf import QuantizationDisposition, build_from_gguf

        path = tmp_path / f"tencent-q1-{expected_bits}.gguf"
        _write_tencent_q1_0_gguf(path)
        with override_flags(tencent_q1_0_use_native_2bit=native_2bit):
            package = build_from_gguf(path, keep_quantized=True)

        report = package.gguf_quantization_report
        q1_records = [record for record in report.tensor_records if record.qtype == "Q1_0"]
        assert len(q1_records) == 7
        assert sum(record.source_bytes for record in q1_records) == 2_816 * 130
        assert {record.disposition for record in q1_records} == {
            QuantizationDisposition.LOSSLESS_REPACK
        }
        assert {record.target_storage for record in q1_records} == {
            f"INT{expected_bits} affine block-128"
        }
        assert any(node.op_type == "MatMulNBits" for node in package["model"].graph)

    def test_tencent_q1_0_reads_pinned_bytes_during_symlink_aba(self, tmp_path: Path):
        from mobius.integrations.gguf._reader import GGUFModel
        from mobius.integrations.gguf._tencent_q1_0 import parse_tencent_q1_0_tensor

        source_a = tmp_path / "source-a.gguf"
        source_b = tmp_path / "source-b.gguf"
        _write_tencent_q1_0_gguf(source_a)
        _write_tencent_q1_0_gguf(source_b)
        inspect_b = GGUFModel(source_b)
        _read_b, data_offset_b, tensor_b = inspect_b._tensor_source("blk.0.attn_q.weight")
        tensor_offset_b = int(tensor_b.field.parts[tensor_b.field.data[-1]][0])
        inspect_b.close()
        with source_b.open("r+b") as stream:
            stream.seek(data_offset_b + tensor_offset_b + 2)
            value = stream.read(1)
            stream.seek(data_offset_b + tensor_offset_b + 2)
            stream.write(bytes([value[0] ^ 0xFF]))

        logical = tmp_path / "logical.gguf"
        _symlink_or_skip(logical, source_a)
        model_a = GGUFModel(logical)
        read_a, data_offset_a, tensor_a = model_a._tensor_source("blk.0.attn_q.weight")
        expected = parse_tencent_q1_0_tensor(read_a, data_offset_a, tensor_a)

        logical.unlink()
        _symlink_or_skip(logical, source_b)
        injected = parse_tencent_q1_0_tensor(read_a, data_offset_a, tensor_a)
        logical.unlink()
        _symlink_or_skip(logical, source_a)

        model_b = GGUFModel(source_b)
        read_b, replacement_offset, replacement_tensor = model_b._tensor_source(
            "blk.0.attn_q.weight"
        )
        replacement = parse_tencent_q1_0_tensor(
            read_b,
            replacement_offset,
            replacement_tensor,
        )

        np.testing.assert_array_equal(injected.weight, expected.weight)
        assert not np.array_equal(injected.weight, replacement.weight)
        assert model_a.source_matches_path()

    def test_preflight_records_mapped_bias_and_scale_parameters(self) -> None:
        from gguf import GGMLQuantizationType
        from onnxscript import nn

        from mobius.integrations.gguf import QuantizationDisposition
        from mobius.integrations.gguf._builder import _preflight_quantization_report

        module = nn.Module()
        module.bias = nn.Parameter([2])
        module.scale = nn.Parameter([1])
        tensors = [
            SimpleNamespace(
                name="source.bias",
                tensor_type=GGMLQuantizationType.F32,
                shape=(2,),
                n_bytes=8,
            ),
            SimpleNamespace(
                name="source.scale",
                tensor_type=GGMLQuantizationType.F32,
                shape=(1,),
                n_bytes=4,
            ),
        ]
        source = SimpleNamespace(
            reader_tensors=lambda: iter(tensors),
            _reader=None,
        )
        report = _preflight_quantization_report(
            source,
            "test",
            module,
            SimpleNamespace(),
            preserve_quantization=True,
            target_bits=4,
            target_block_size=32,
            execution_provider="default",
            name_mapper=lambda name, _architecture: name.removeprefix("source."),
        )

        assert {record.name for record in report.tensor_records} == {
            "source.bias",
            "source.scale",
        }
        assert {record.disposition for record in report.tensor_records} == {
            QuantizationDisposition.SOURCE_FLOAT
        }
        assert sum(record.source_bytes for record in report.tensor_records) == sum(
            stat.source_bytes for stat in report.source_qtype_census
        )

    def test_qwen35moe_mixed_float_experts_fail_closed_on_default_route(self) -> None:
        """The quantized Qwen3.5-MoE route must not requantize source-float experts."""
        from gguf import GGMLQuantizationType
        from onnxscript import nn

        from mobius.components import QuantizedLinear
        from mobius.integrations.gguf._builder import _preflight_quantization_report

        def make_expert() -> nn.Module:
            expert = nn.Module()
            expert.gate_proj = QuantizedLinear(64, 64)
            expert.up_proj = QuantizedLinear(64, 64)
            expert.down_proj = QuantizedLinear(64, 64)
            return expert

        mlp = nn.Module()
        mlp.experts = nn.ModuleList([make_expert(), make_expert()])
        layer = nn.Module()
        layer.mlp = mlp
        layer.self_attn = nn.Module()
        layer.self_attn.o_proj = QuantizedLinear(64, 64)
        module = nn.Module()
        module.model = nn.Module()
        module.model.layers = nn.ModuleList([layer])
        tensors = [
            SimpleNamespace(
                name="blk.0.attn_output.weight",
                tensor_type=GGMLQuantizationType.Q4_0,
                shape=(64, 64),
                n_bytes=2_304,
            ),
            SimpleNamespace(
                name="blk.0.ffn_gate_exps.weight",
                tensor_type=GGMLQuantizationType.F16,
                shape=(64, 64, 2),
                n_bytes=16_384,
            ),
        ]
        source = SimpleNamespace(reader_tensors=lambda: iter(tensors), _reader=None)

        with pytest.raises(
            ValueError,
            match="selected quantized graph would quantize a source-float tensor",
        ):
            _preflight_quantization_report(
                source,
                "qwen35moe",
                module,
                SimpleNamespace(),
                preserve_quantization=True,
                target_bits=4,
                target_block_size=32,
                execution_provider="cpu",
            )

    def test_norms_are_float(self, q4_0_gguf: Path):
        """Norm weights remain float, not quantized."""
        import onnx_ir as ir

        from mobius.integrations.gguf import build_from_gguf

        pkg = build_from_gguf(q4_0_gguf, keep_quantized=True)
        model = pkg["model"]

        for init in model.graph.initializers.values():
            name = init.name or ""
            if "norm" in name and "weight" in name:
                assert init.dtype != ir.DataType.UINT8, (
                    f"Norm {name} should be float, not uint8"
                )

    def test_dequantized_path_no_matmulnbits(self, q4_0_gguf: Path):
        """Explicit API dequantization emits no quantized projection ops."""
        from mobius.integrations.gguf import build_from_gguf

        pkg = build_from_gguf(q4_0_gguf, keep_quantized=False)
        model = pkg["model"]

        op_types = {node.op_type for node in model.graph if node.op_type}
        assert "MatMulNBits" not in op_types
        assert "BlockQuantizedMatMul" not in op_types

    def test_detect_quant_params(self, q4_0_gguf: Path):
        """_detect_quant_params finds Q4_0 as dominant type."""
        from mobius.integrations.gguf._builder import (
            _detect_quant_params,
        )
        from mobius.integrations.gguf._reader import GGUFModel

        gguf_model = GGUFModel(q4_0_gguf)
        bits, block_size, is_sym = _detect_quant_params(gguf_model, gguf_model.architecture)
        assert bits == 4
        assert block_size == 32
        assert is_sym is False

    def test_embedding_quantization_check_is_metadata_only(self, monkeypatch):
        """Embedding compatibility does not read or repack tensor data."""
        from types import SimpleNamespace

        from gguf import GGMLQuantizationType

        from mobius.integrations.gguf import _tencent_q1_0
        from mobius.integrations.gguf._builder import _can_quantize_embedding

        monkeypatch.setattr(_tencent_q1_0, "is_tencent_q1_0_layout", lambda _model: False)
        tensor = SimpleNamespace(
            name="token_embd.weight",
            tensor_type=GGMLQuantizationType.Q4_0,
            shape=(64, 256),
        )
        model = SimpleNamespace(
            _reader=SimpleNamespace(tensors=[tensor]),
            reader_tensors=lambda: [tensor],
        )

        assert _can_quantize_embedding(model, "llama", bits=4, block_size=32)

    def test_tencent_q1_0_embedding_is_not_quantized(self, monkeypatch):
        """Tencent Q1_0 detection short-circuits before inspecting tensors."""
        from types import SimpleNamespace

        from mobius.integrations.gguf import _tencent_q1_0
        from mobius.integrations.gguf._builder import _can_quantize_embedding

        monkeypatch.setattr(_tencent_q1_0, "is_tencent_q1_0_layout", lambda _model: True)
        model = SimpleNamespace()

        assert not _can_quantize_embedding(model, "llama", bits=4, block_size=128)

    def test_decoder_backed_embedding_uses_affine_gather_target(self, monkeypatch):
        from gguf import GGMLQuantizationType

        from mobius.integrations.gguf import _tencent_q1_0
        from mobius.integrations.gguf._builder import _can_quantize_embedding

        monkeypatch.setattr(_tencent_q1_0, "is_tencent_q1_0_layout", lambda _model: False)
        tensor = SimpleNamespace(
            name="token_embd.weight",
            tensor_type=GGMLQuantizationType.Q5_1,
            shape=(64, 256),
        )
        model = SimpleNamespace(
            _reader=SimpleNamespace(tensors=[tensor]),
            reader_tensors=lambda: [tensor],
        )

        assert _can_quantize_embedding(model, "llama", bits=4, block_size=32)

    def test_decoder_backed_output_head_is_not_claimed_as_preserved(self):
        from mobius.integrations.gguf._builder import _can_quantize_lm_head

        class _OutputModel:
            def tensor_items_raw(self):
                yield (
                    "output.weight",
                    np.empty(0, dtype=np.uint8),
                    SimpleNamespace(name="TQ1_0"),
                    (256, 64),
                )

        assert not _can_quantize_lm_head(_OutputModel(), "llama")

    def test_q4_k_m_mixed_profile_is_allowed_for_reported_normalization(self):
        """Mixed Q4_K_M-like projections may use the declared lossy INT4 route."""
        from gguf import GGMLQuantizationType

        from mobius.integrations.gguf._builder import (
            _reject_unsupported_quantization_preservation,
        )

        class _MixedModel:
            def tensor_items_raw(self):
                for i in range(5):
                    yield (
                        f"blk.{i}.attn_q.weight",
                        np.empty(0, dtype=np.uint8),
                        GGMLQuantizationType.Q5_0,
                        (64, 64),
                    )
                yield (
                    "blk.0.ffn_down.weight",
                    np.empty(0, dtype=np.uint8),
                    GGMLQuantizationType.Q4_K,
                    (64, 128),
                )

        _reject_unsupported_quantization_preservation(
            _MixedModel(), "llama", preserve_quantization=True
        )

    def test_decoder_backed_qtype_selects_explicit_requantization_target(self):
        """A decoder-backed qtype takes the declared 4-bit requantization route."""
        from gguf import GGMLQuantizationType

        from mobius.integrations.gguf._builder import _detect_quant_params

        class _UnsupportedModel:
            def tensor_items_raw(self):
                yield (
                    "blk.0.ffn_down.weight",
                    np.empty(0, dtype=np.uint8),
                    GGMLQuantizationType.Q5_K,
                    (64, 128),
                )

        assert _detect_quant_params(_UnsupportedModel(), "llama") == (4, 32, False)

    def test_qtype_without_decoder_or_kernel_is_rejected_actionably(self):
        import enum

        from mobius.integrations.gguf._builder import _detect_quant_params

        class _PinnedQType(enum.IntEnum):
            Q2_0 = 42

        class _Q20Model:
            def tensor_items_raw(self):
                yield (
                    "blk.0.ffn_down.weight",
                    np.empty(0, dtype=np.uint8),
                    _PinnedQType.Q2_0,
                    (64, 128),
                )

        with pytest.raises(
            ValueError,
            match=r"Q2_0: gguf-py ships no Python dequantizer.*Re-quantize",
        ):
            _detect_quant_params(_Q20Model(), "llama")

    def test_out_of_census_qtype_is_rejected_before_route_selection(self):
        import enum

        from mobius.integrations.gguf._builder import _detect_quant_params

        class _FutureQType(enum.IntEnum):
            FUTURE = 99

        class _FutureModel:
            def tensor_items_raw(self):
                yield (
                    "blk.0.ffn_down.weight",
                    np.empty(0, dtype=np.uint8),
                    _FutureQType.FUTURE,
                    (64, 128),
                )

        with pytest.raises(ValueError, match=r"outside the pinned llama\.cpp census"):
            _detect_quant_params(_FutureModel(), "llama")

    def test_architecture_without_quantized_modules_rejects_preservation(
        self, tmp_path: Path
    ) -> None:
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / "internlm2_q4_0.gguf"
        _write_quantized_gguf(path, architecture="internlm2")

        with pytest.raises(
            ValueError,
            match=r"does not support keep_quantized=True.*floating Linear modules",
        ):
            build_from_gguf(path)

    def test_q6_k_projection_is_allowed_for_reported_requantization(self):
        """A stacked 6-bit expert projection may use the declared lossy route."""
        from gguf import GGMLQuantizationType

        from mobius.integrations.gguf._builder import (
            _reject_unsupported_quantization_preservation,
        )

        class _Q6KModel:
            def tensor_items_raw(self):
                yield (
                    "blk.0.ffn_down.weight",
                    np.empty(0, dtype=np.uint8),
                    GGMLQuantizationType.Q6_K,
                    (4, 64, 128),
                )

        _reject_unsupported_quantization_preservation(
            _Q6KModel(), "llama", preserve_quantization=True
        )

    def test_native_blocks_may_normalize_when_builder_cannot_install_native_modules(self):
        """A builder without native modules may use a declared lossy affine route."""
        from gguf import GGMLQuantizationType

        from mobius.integrations.gguf._builder import (
            _reject_unsupported_quantization_preservation,
        )

        class _NativeModel:
            def tensor_items_raw(self):
                yield (
                    "blk.0.attn_q.weight",
                    np.empty(0, dtype=np.uint8),
                    GGMLQuantizationType.MXFP4,
                    (64, 64),
                )

        _reject_unsupported_quantization_preservation(
            _NativeModel(),
            "gemma4",
            preserve_quantization=True,
            allow_native_blocks=False,
        )

    def test_native_mtp_blocks_may_use_affine_requantization(self):
        """Native MTP blocks with a decoder may use a declared lossy affine route."""
        from gguf import GGMLQuantizationType

        from mobius.integrations.gguf._builder import (
            _reject_unsupported_quantization_preservation,
        )

        class _NativeMtpModel:
            metadata: ClassVar[dict[str, int]] = {
                "qwen35.block_count": 25,
                "qwen35.nextn_predict_layers": 1,
            }

            def tensor_items_raw(self):
                yield (
                    "blk.24.attn_q.weight",
                    np.empty(0, dtype=np.uint8),
                    GGMLQuantizationType.MXFP4,
                    (64, 128),
                )

        _reject_unsupported_quantization_preservation(
            _NativeMtpModel(), "qwen35", preserve_quantization=True
        )

    @pytest.mark.parametrize(
        ("tensor_name", "options", "message"),
        [
            (
                "token_embd.weight",
                {"allow_quantized_embeddings": False},
                "packed embedding",
            ),
            (
                "output.weight",
                {"allow_quantized_lm_head": False},
                "packed LM head",
            ),
        ],
    )
    def test_builder_specific_float_roles_fail_closed(
        self, tensor_name: str, options: dict[str, bool], message: str
    ):
        """A multimodal builder cannot silently dequantize packed tables."""
        from gguf import GGMLQuantizationType

        from mobius.integrations.gguf._builder import (
            _reject_unsupported_quantization_preservation,
        )

        class _PackedTableModel:
            def tensor_items_raw(self):
                yield (
                    tensor_name,
                    np.empty(0, dtype=np.uint8),
                    GGMLQuantizationType.Q4_0,
                    (256, 64),
                )

        with pytest.raises(ValueError, match=message):
            _reject_unsupported_quantization_preservation(
                _PackedTableModel(),
                "muse_glimmer",
                preserve_quantization=True,
                **options,
            )

    def test_quantized_fused_expert_gate_up_fails_closed(self):
        """Packed fused expert tensors cannot be split through the float normalizer."""
        from gguf import GGMLQuantizationType

        from mobius.integrations.gguf._builder import (
            _reject_unsupported_quantization_preservation,
        )

        class _FusedExpertModel:
            def tensor_items_raw(self):
                yield (
                    "blk.0.ffn_gate_up_exps.weight",
                    np.empty(0, dtype=np.uint8),
                    GGMLQuantizationType.Q4_0,
                    (4, 128, 64),
                )

        with pytest.raises(ValueError, match=r"cannot split packed fused expert tensor"):
            _reject_unsupported_quantization_preservation(
                _FusedExpertModel(), "qwen35moe", preserve_quantization=True
            )

    def test_mtp_projection_is_allowed_for_reported_requantization(self):
        """Sidecar-specific mappings may use the declared lossy conversion route."""
        from gguf import GGMLQuantizationType

        from mobius.integrations.gguf._builder import (
            _has_quantized_weights,
            _reject_unsupported_quantization_preservation,
        )

        class _MTPModel:
            def tensor_items_raw(self):
                yield (
                    "blk.24.nextn.ffn_down.weight",
                    np.empty(0, dtype=np.uint8),
                    GGMLQuantizationType.Q4_K,
                    (64, 128),
                )

        model = _MTPModel()
        assert _has_quantized_weights(model, "qwen3")
        _reject_unsupported_quantization_preservation(
            model, "qwen3", preserve_quantization=True
        )

    def test_lossy_tied_embedding_source_is_allowed_for_reported_conversion(self):
        """A tied Q4_K table may use the declared affine conversion route."""
        from gguf import GGMLQuantizationType

        from mobius.integrations.gguf._builder import (
            _reject_unsupported_quantization_preservation,
        )

        class _EmbeddingModel:
            def tensor_items_raw(self):
                yield (
                    "token_embd.weight",
                    np.empty(0, dtype=np.uint8),
                    GGMLQuantizationType.Q4_K,
                    (256, 128),
                )

        _reject_unsupported_quantization_preservation(
            _EmbeddingModel(), "llama", preserve_quantization=True
        )

    def test_mixed_lossless_affine_targets_may_normalize_to_one_contract(self):
        """Q4_0 and Q8_0 may share one contract through reported normalization."""
        from gguf import GGMLQuantizationType

        from mobius.integrations.gguf._builder import (
            _reject_unsupported_quantization_preservation,
        )

        class _MixedExactModel:
            def tensor_items_raw(self):
                yield (
                    "blk.0.attn_q.weight",
                    np.empty(0, dtype=np.uint8),
                    GGMLQuantizationType.Q4_0,
                    (64, 64),
                )
                yield (
                    "blk.0.attn_v.weight",
                    np.empty(0, dtype=np.uint8),
                    GGMLQuantizationType.Q8_0,
                    (64, 64),
                )

        _reject_unsupported_quantization_preservation(
            _MixedExactModel(), "llama", preserve_quantization=True
        )

    def test_encoder_float_embedding_does_not_constrain_projection_target(self):
        """An encoder embedding's float route is independent of packed projections."""
        from gguf import GGMLQuantizationType

        from mobius.integrations.gguf._builder import (
            _detect_quant_params,
            _reject_unsupported_quantization_preservation,
        )

        class _EncoderModel:
            def tensor_items_raw(self):
                yield (
                    "position_embd.weight",
                    np.empty(0, dtype=np.uint8),
                    GGMLQuantizationType.Q8_0,
                    (256, 64),
                )
                yield (
                    "blk.0.attn_norm.weight",
                    np.empty(0, dtype=np.uint8),
                    GGMLQuantizationType.Q8_0,
                    (64,),
                )
                yield (
                    "blk.0.attn_q.weight",
                    np.empty(0, dtype=np.uint8),
                    GGMLQuantizationType.Q4_0,
                    (64, 64),
                )

        _reject_unsupported_quantization_preservation(
            _EncoderModel(), "bert", preserve_quantization=True
        )
        assert _detect_quant_params(_EncoderModel(), "bert") == (4, 32, False)

    def test_explicit_float_linear_does_not_constrain_projection_target(self):
        """A hybrid float-only linear may use a different source qtype."""
        from gguf import GGMLQuantizationType

        from mobius.integrations.gguf._builder import (
            _detect_quant_params,
            _reject_unsupported_quantization_preservation,
        )

        class _HybridModel:
            def tensor_items_raw(self):
                yield (
                    "blk.0.ssm_in.weight",
                    np.empty(0, dtype=np.uint8),
                    GGMLQuantizationType.Q8_0,
                    (64, 64),
                )
                yield (
                    "blk.1.attn_q.weight",
                    np.empty(0, dtype=np.uint8),
                    GGMLQuantizationType.Q4_0,
                    (64, 64),
                )

        _reject_unsupported_quantization_preservation(
            _HybridModel(),
            "jamba",
            preserve_quantization=True,
            dequantize_float_linear_types={
                "model.layers.0.mamba.in_proj": {"Q8_0"},
            },
        )
        assert _detect_quant_params(_HybridModel(), "jamba") == (4, 32, False)

    def test_native_blocks_use_the_exact_affine_companion_target(self):
        """Native projections do not force exact Q8_0 companions through INT4."""
        from gguf import GGMLQuantizationType

        from mobius.integrations.gguf._builder import _detect_quant_params

        class _NativeAndQ8Model:
            def tensor_items_raw(self):
                yield (
                    "blk.0.attn_q.weight",
                    np.empty(0, dtype=np.uint8),
                    GGMLQuantizationType.IQ4_XS,
                    (64, 64),
                )
                yield (
                    "blk.0.attn_v.weight",
                    np.empty(0, dtype=np.uint8),
                    GGMLQuantizationType.Q8_0,
                    (64, 64),
                )

        assert _detect_quant_params(_NativeAndQ8Model(), "llama") == (8, 32, False)

    def test_runtime_unsupported_format_does_not_select_native_op(self):
        """A GGUF type outside the runtime contract remains on the fallback."""
        from gguf import GGMLQuantizationType

        from mobius.integrations.gguf._builder import _native_block_format

        assert _native_block_format(GGMLQuantizationType.Q5_0) is None

    @pytest.mark.parametrize("moe_container", ["experts", "moe.experts"])
    def test_native_moe_tensor_maps_to_each_expert(self, moe_container: str):
        """Stacked GGUF MoE blocks route to standard and DeepSeek expert paths."""
        from mobius.integrations.gguf._builder import _native_block_target_stems

        available = {f"model.layers.0.mlp.{moe_container}.{i}.gate_proj" for i in range(3)}
        assert _native_block_target_stems(
            "model.layers.0.mlp.experts.gate_proj.weight",
            (3, 64, 64),
            available,
        ) == sorted(available, key=lambda name: int(name.split(".")[-2]))

    def test_float_moe_tensor_preserves_expert_order_and_rejects_bad_shape(self):
        import torch

        from mobius._configs import ArchitectureConfig
        from mobius.integrations.gguf._builder import _normalize_gguf_weights

        config = ArchitectureConfig(
            vocab_size=32,
            hidden_size=4,
            intermediate_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            num_local_experts=3,
            num_experts_per_tok=2,
            moe_intermediate_size=6,
        )
        key = "model.layers.0.mlp.experts.gate_proj.weight"
        stacked = torch.arange(3 * 6 * 4, dtype=torch.float32).reshape(3, 6, 4)
        normalized = _normalize_gguf_weights({key: stacked}, config=config)

        assert key not in normalized
        for expert in range(3):
            target = f"model.layers.0.mlp.experts.{expert}.gate_proj.weight"
            assert torch.equal(normalized[target], stacked[expert])

        with pytest.raises(ValueError, match="Invalid stacked expert shape"):
            _normalize_gguf_weights({key: stacked[:, :-1]}, config=config)
        with pytest.raises(ValueError, match="Invalid stacked expert shape"):
            _normalize_gguf_weights({key: stacked.reshape(18, 4)}, config=config)

        router_key = "model.layers.0.mlp.gate.weight"
        with pytest.raises(ValueError, match="Invalid router shape"):
            _normalize_gguf_weights(
                {router_key: torch.empty(2, 4)},
                config=config,
            )

    @pytest.mark.parametrize("architecture", ["dream", "llada-moe", "rnd1"])
    def test_float_diffusion_fused_qkv_splits_by_gqa_width(self, architecture: str):
        import torch

        from mobius._configs import ArchitectureConfig
        from mobius.integrations.gguf._builder import _normalize_gguf_weights

        config = ArchitectureConfig(
            hidden_size=8,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=4,
        )
        stem = "model.layers.0.self_attn"
        fused_weight = torch.arange(16 * 8, dtype=torch.float32).reshape(16, 8)
        fused_bias = torch.arange(16, dtype=torch.float32)
        normalized = _normalize_gguf_weights(
            {
                f"{stem}.qkv_proj.weight": fused_weight,
                f"{stem}.qkv_proj.bias": fused_bias,
            },
            gguf_arch=architecture,
            config=config,
        )

        expected_widths = {"q_proj": (0, 8), "k_proj": (8, 12), "v_proj": (12, 16)}
        for projection, (start, end) in expected_widths.items():
            assert torch.equal(
                normalized[f"{stem}.{projection}.weight"],
                fused_weight[start:end],
            )
            assert torch.equal(
                normalized[f"{stem}.{projection}.bias"],
                fused_bias[start:end],
            )

        with pytest.raises(ValueError, match=r"Invalid fused .* QKV weight width"):
            _normalize_gguf_weights(
                {f"{stem}.qkv_proj.weight": fused_weight[:-1]},
                gguf_arch=architecture,
                config=config,
            )
