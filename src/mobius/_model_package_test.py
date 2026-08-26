# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for ModelPackage."""

from __future__ import annotations

import logging
import threading
import types

import onnx_ir as ir
import pytest
import torch

from mobius._builder import build_from_module
from mobius._configs import VisionConfig
from mobius._model_package import (
    ModelPackage,
    _make_progress_callback,
    _namespaced_symbolic_dimensions,
)
from mobius._testing import make_config
from mobius.generation import build_greedy_sampler
from mobius.models.base import CausalLMModel
from mobius.models.gemma3 import Gemma3MultiModalModel
from mobius.tasks import CausalLMTask, VisionLanguageTask


def _make_simple_model(name: str = "test") -> ir.Model:
    """Create a minimal ir.Model for testing."""
    graph = ir.Graph([], [], nodes=[], name=name)
    return ir.Model(graph, ir_version=10)


class TestModelPackageDict:
    def test_getitem(self):
        m = _make_simple_model()
        pkg = ModelPackage({"a": m})
        assert pkg["a"] is m

    def test_setitem(self):
        pkg = ModelPackage()
        m = _make_simple_model()
        pkg["new"] = m
        assert pkg["new"] is m

    def test_rejects_component_without_graph(self):
        with pytest.raises(TypeError, match=r"must be an ir\.Model with a graph"):
            ModelPackage({"invalid": object()})  # type: ignore[arg-type]

        pkg = ModelPackage()
        with pytest.raises(TypeError, match=r"must be an ir\.Model with a graph"):
            pkg["invalid"] = object()  # type: ignore[assignment]

    def test_delitem(self):
        pkg = ModelPackage({"a": _make_simple_model()})
        del pkg["a"]
        assert "a" not in pkg

    def test_contains(self):
        pkg = ModelPackage({"a": _make_simple_model()})
        assert "a" in pkg
        assert "b" not in pkg

    def test_len(self):
        pkg = ModelPackage({"a": _make_simple_model(), "b": _make_simple_model()})
        assert len(pkg) == 2

    def test_iter(self):
        pkg = ModelPackage({"x": _make_simple_model(), "y": _make_simple_model()})
        assert sorted(pkg) == ["x", "y"]

    def test_keys_values_items(self):
        m1 = _make_simple_model("m1")
        m2 = _make_simple_model("m2")
        pkg = ModelPackage({"a": m1, "b": m2})
        assert list(pkg.keys()) == ["a", "b"]
        assert list(pkg.values()) == [m1, m2]
        assert list(pkg.items()) == [("a", m1), ("b", m2)]

    def test_repr(self):
        pkg = ModelPackage({"text_decoder": _make_simple_model()})
        assert "text_decoder" in repr(pkg)

    def test_empty(self):
        pkg = ModelPackage()
        assert len(pkg) == 0

    def test_config_stored(self):
        config = make_config()
        pkg = ModelPackage({"m": _make_simple_model()}, config=config)
        assert pkg.config is config


class TestWeightLoadingReport:
    def test_roundtrips_with_package(self, tmp_path):
        pkg = ModelPackage({"model": _make_simple_model()})
        pkg.weight_loading_report = {
            "format": "mobius.weight-loading-report.v1",
            "native_fp8": False,
            "output_weight_format": "dense",
        }

        pkg.save(str(tmp_path), progress_bar=False)
        loaded = ModelPackage.load(str(tmp_path))

        assert loaded.weight_loading_report == pkg.weight_loading_report
        assert loaded.weight_loading_report["external_data_shard_limit_bytes"] == 1 << 30
        assert loaded.weight_loading_report["largest_dense_tensor_bytes"] == 0
        assert loaded.weight_loading_report["serializer_max_workers"] == 1
        assert (tmp_path / "weight-loading-report.json").is_file()

    def test_dense_streaming_forces_serial_external_data_save(self, tmp_path, monkeypatch):
        workers = []

        def save(
            model,
            path,
            *,
            external_data,
            max_shard_size_bytes,
            callback,
            max_workers,
        ):
            workers.append(max_workers)

        monkeypatch.setattr(ir, "save", save)
        pkg = ModelPackage({"model": _make_simple_model()})
        pkg.weight_loading_report = {
            "format": "mobius.weight-loading-report.v1",
            "native_fp8": False,
            "output_weight_format": "dense",
        }

        pkg.save(
            str(tmp_path),
            max_workers=8,
            progress_bar=False,
            check_weights=False,
        )

        assert workers == [1]
        assert pkg.weight_loading_report["serializer_max_workers"] == 1
        assert (
            "forced to one worker" in pkg.weight_loading_report["serialization_memory_bound"]
        )

    def test_rejects_unbounded_dense_streaming_shard(self, tmp_path):
        pkg = ModelPackage({"model": _make_simple_model()})
        pkg.weight_loading_report = {
            "format": "mobius.weight-loading-report.v1",
            "native_fp8": False,
            "output_weight_format": "dense",
        }

        with pytest.raises(ValueError, match="serializer buffers one output shard"):
            pkg.save(
                str(tmp_path),
                max_shard_size_bytes=5_000_000_001,
                progress_bar=False,
            )

    def test_ordinary_resave_removes_stale_report(self, tmp_path):
        pkg = ModelPackage({"model": _make_simple_model()})
        pkg.weight_loading_report = {
            "format": "mobius.weight-loading-report.v1",
            "native_fp8": False,
        }
        pkg.save(str(tmp_path), progress_bar=False)

        pkg.weight_loading_report = None
        pkg.save(str(tmp_path), progress_bar=False)

        assert not (tmp_path / "weight-loading-report.json").exists()

    def test_load_rejects_invalid_report(self, tmp_path):
        pkg = ModelPackage({"model": _make_simple_model()})
        pkg.save(str(tmp_path), progress_bar=False)
        (tmp_path / "weight-loading-report.json").write_text('{"format": "unknown"}')

        with pytest.raises(ValueError, match="Invalid weight-loading report"):
            ModelPackage.load(str(tmp_path))


class TestOnnxShardedSave:
    def test_onnx_external_data_is_sharded(self, tmp_path):
        graph = ir.Graph([], [], nodes=[], name="m")
        for index in range(6):
            name = f"weight_{index}"
            graph.register_initializer(
                ir.Value(
                    name=name,
                    const_value=ir.Tensor(
                        torch.full((1024,), index, dtype=torch.float32),
                        name=name,
                        dtype=ir.DataType.FLOAT,
                    ),
                )
            )
        pkg = ModelPackage({"m": ir.Model(graph, ir_version=10)})

        pkg.save(
            str(tmp_path),
            external_data="onnx",
            max_shard_size_bytes=8192,
            progress_bar=False,
        )

        shards = sorted(
            [
                *tmp_path.glob("model-*-of-*.onnx.data"),
                # onnx_ir 1.0.0 did not yet recognize .onnx.data as a
                # compound suffix.
                *tmp_path.glob("model.onnx-*-of-*.data"),
            ]
        )
        assert len(shards) == 3
        assert all(shard.stat().st_size <= 8192 for shard in shards)
        loaded = ModelPackage.load(str(tmp_path))
        assert set(loaded.data) == {"model"}

    def test_defaults_to_eight_workers_when_supported(self, tmp_path, monkeypatch):
        calls = []

        def save(
            model,
            path,
            *,
            external_data,
            max_shard_size_bytes,
            callback,
            max_workers,
        ):
            calls.append(max_workers)

        monkeypatch.setattr(ir, "save", save)

        ModelPackage({"m": _make_simple_model()}).save(
            str(tmp_path), progress_bar=False, check_weights=False
        )

        assert calls == [8]

    def test_forwards_serial_worker_override(self, tmp_path, monkeypatch):
        calls = []

        def save(
            model,
            path,
            *,
            external_data,
            max_shard_size_bytes,
            callback,
            max_workers,
        ):
            calls.append(max_workers)

        monkeypatch.setattr(ir, "save", save)

        ModelPackage({"m": _make_simple_model()}).save(
            str(tmp_path),
            max_workers=1,
            progress_bar=False,
            check_weights=False,
        )

        assert calls == [1]

    def test_omits_max_workers_for_onnx_ir_1_0(self, tmp_path, monkeypatch):
        calls = []

        def save(model, path, *, external_data, max_shard_size_bytes, callback):
            calls.append((model, path))

        monkeypatch.setattr(ir, "save", save)

        ModelPackage({"m": _make_simple_model()}).save(
            str(tmp_path), progress_bar=False, check_weights=False
        )

        assert len(calls) == 1

    @pytest.mark.parametrize("max_workers", [0, -1])
    def test_rejects_non_positive_max_workers(self, tmp_path, max_workers):
        with pytest.raises(ValueError, match="max_workers must be positive"):
            ModelPackage({"m": _make_simple_model()}).save(
                str(tmp_path),
                max_workers=max_workers,
                progress_bar=False,
                check_weights=False,
            )


class TestProgressCallback:
    class _Tensor:
        name = "w"
        shape = (2, 2)

        class dtype:  # noqa: N801
            @staticmethod
            def short_name():
                return "f32"

    class _Bar:
        def __init__(self, *, total, desc, position, leave):
            self.total = total
            self.desc = desc
            self.position = position
            self.leave = leave
            self.n = 0
            self.postfix = ""
            self.closed = False

        def update(self):
            self.n += 1

        def set_postfix_str(self, value):
            self.postfix = value

        def close(self):
            self.closed = True

    def test_orders_progress_bars_by_shard_number(self, monkeypatch):
        bars = []

        def make_bar(**kwargs):
            bar = self._Bar(**kwargs)
            bars.append(bar)
            return bar

        monkeypatch.setattr("mobius._model_package.tqdm.tqdm", make_bar)
        callback = _make_progress_callback()
        for filename in (
            "model-00002-of-00002.onnx.data",
            "model-00001-of-00002.onnx.data",
        ):
            for shard_index in reversed(range(2)):
                callback(
                    self._Tensor(),
                    types.SimpleNamespace(
                        total=4,
                        index=shard_index,
                        offset=0,
                        filename=filename,
                        shard_total=2,
                        shard_index=shard_index,
                    ),
                )

        assert len(bars) == 2
        assert [bar.position for bar in bars] == [1, 0]
        assert all(bar.total == 2 and bar.n == 2 and bar.closed for bar in bars)
        assert "model-00002-of-00002.onnx.data" in bars[0].desc
        assert "model-00001-of-00002.onnx.data" in bars[1].desc

    def test_falls_back_to_one_bar_with_onnx_ir_1_0(self, monkeypatch):
        bars = []

        def make_bar(**kwargs):
            bar = self._Bar(**kwargs)
            bars.append(bar)
            return bar

        monkeypatch.setattr("mobius._model_package.tqdm.tqdm", make_bar)
        callback = _make_progress_callback()
        for index in range(4):
            callback(
                self._Tensor(),
                types.SimpleNamespace(
                    total=4,
                    index=index,
                    offset=0,
                    filename=f"model-{index}.data",
                ),
            )

        assert len(bars) == 1
        assert bars[0].total == 4
        assert bars[0].n == 4
        assert bars[0].position == 0
        assert bars[0].closed

    def test_is_thread_safe(self, monkeypatch):
        bars = []

        def make_bar(**kwargs):
            bar = self._Bar(**kwargs)
            bars.append(bar)
            return bar

        monkeypatch.setattr("mobius._model_package.tqdm.tqdm", make_bar)
        callback = _make_progress_callback()
        total = 200

        def invoke(index):
            callback(
                self._Tensor(),
                types.SimpleNamespace(
                    total=total,
                    index=index,
                    offset=0,
                    filename="model.onnx.data",
                    shard_total=total,
                    shard_index=index,
                ),
            )

        threads = [threading.Thread(target=invoke, args=(index,)) for index in range(total)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert len(bars) == 1
        assert bars[0].total == total
        assert bars[0].n == total
        assert bars[0].closed


class TestModelPackageSaveLoad:
    def test_save_creates_files(self, tmp_path):
        pkg = ModelPackage(
            {
                "text_decoder": _make_simple_model("decoder"),
                "vision_encoder": _make_simple_model("encoder"),
            }
        )
        pkg.save(str(tmp_path))
        assert (tmp_path / "text_decoder" / "model.onnx").exists()
        assert (tmp_path / "vision_encoder" / "model.onnx").exists()

    def test_roundtrip(self, tmp_path):
        config = make_config()
        module = CausalLMModel(config)
        pkg = build_from_module(module, config)

        pkg.save(str(tmp_path), check_weights=False)
        loaded = ModelPackage.load(str(tmp_path))

        assert len(loaded) == 1
        assert "model" in loaded
        assert pkg["model"].graph is not None
        assert loaded["model"].graph.num_nodes() == pkg["model"].graph.num_nodes()

    def test_save_without_report_rejects_stale_report(self, tmp_path):
        report_path = tmp_path / "quantization_report.json"
        report_path.write_text('{"user": "owned"}\n', encoding="utf-8")
        pkg = ModelPackage({"model": _make_simple_model("model")})

        with pytest.raises(ValueError, match=r"stale quantization_report.json"):
            pkg.save(str(tmp_path))

        assert report_path.read_text(encoding="utf-8") == '{"user": "owned"}\n'
        assert not (tmp_path / "model.onnx").exists()

    def test_load_multiple(self, tmp_path):
        pkg = ModelPackage(
            {
                "a": _make_simple_model("a"),
                "b": _make_simple_model("b"),
            }
        )
        pkg.save(str(tmp_path))
        loaded = ModelPackage.load(str(tmp_path))
        assert sorted(loaded) == ["a", "b"]

    def test_mtp_head_roundtrip_with_flat_target(self, tmp_path):
        pkg = ModelPackage({"model": _make_simple_model("target")})
        pkg.mtp_head = ModelPackage({"model": _make_simple_model("mtp")})

        pkg.save(str(tmp_path))
        loaded = ModelPackage.load(str(tmp_path))

        assert sorted(loaded) == ["model"]
        assert loaded.mtp_head is not None
        assert sorted(loaded.mtp_head) == ["model"]
        assert (tmp_path / ".mobius-mtp" / "model.onnx").exists()

    def test_mtp_head_roundtrip_with_multicomponent_target(self, tmp_path):
        pkg = ModelPackage(
            {
                "decoder": _make_simple_model("decoder"),
                "embedding": _make_simple_model("embedding"),
            }
        )
        pkg.mtp_head = ModelPackage({"model": _make_simple_model("mtp")})

        pkg.save(str(tmp_path))
        loaded = ModelPackage.load(str(tmp_path))

        assert sorted(loaded) == ["decoder", "embedding"]
        assert loaded.mtp_head is not None
        assert sorted(loaded.mtp_head) == ["model"]

    def test_legitimate_mtp_component_roundtrip(self, tmp_path):
        pkg = ModelPackage(
            {
                "model": _make_simple_model("target"),
                "mtp": _make_simple_model("mtp"),
            }
        )

        pkg.save(str(tmp_path))
        loaded = ModelPackage.load(str(tmp_path))

        assert sorted(loaded) == ["model", "mtp"]
        assert loaded.mtp_head is None

    def test_mtp_sidecar_directory_collision_is_encoded(self, tmp_path):
        pkg = ModelPackage(
            {
                "model": _make_simple_model("target"),
                ".mobius-mtp": _make_simple_model("legitimate-component"),
                "mtp": _make_simple_model("mtp-component"),
            }
        )
        pkg.mtp_head = ModelPackage(
            {
                "draft": _make_simple_model("draft"),
                "auxiliary": _make_simple_model("auxiliary"),
            }
        )

        pkg.save(str(tmp_path))
        loaded = ModelPackage.load(str(tmp_path))

        assert sorted(loaded) == [".mobius-mtp", "model", "mtp"]
        assert loaded.mtp_head is not None
        assert sorted(loaded.mtp_head) == ["auxiliary", "draft"]
        assert (tmp_path / ".mobius-mtp-1" / "draft" / "model.onnx").is_file()

    def test_mtp_sidecar_case_insensitive_collision_is_encoded(self, tmp_path):
        pkg = ModelPackage(
            {
                "model": _make_simple_model("target"),
                ".MOBIUS-MTP": _make_simple_model("legitimate-component"),
            }
        )
        pkg.mtp_head = ModelPackage({"model": _make_simple_model("draft")})

        pkg.save(str(tmp_path))
        loaded = ModelPackage.load(str(tmp_path))

        assert sorted(loaded) == [".MOBIUS-MTP", "model"]
        assert loaded.mtp_head is not None
        assert (tmp_path / ".mobius-mtp-1" / "model.onnx").is_file()

    def test_nested_mtp_sidecars_roundtrip(self, tmp_path):
        pkg = ModelPackage({"model": _make_simple_model("target")})
        pkg.mtp_head = ModelPackage(
            {
                "draft": _make_simple_model("draft"),
                "auxiliary": _make_simple_model("auxiliary"),
            }
        )
        pkg.mtp_head.mtp_head = ModelPackage(
            {
                "next": _make_simple_model("next"),
                "next_auxiliary": _make_simple_model("next-auxiliary"),
            }
        )

        pkg.save(str(tmp_path))
        loaded = ModelPackage.load(str(tmp_path))

        assert loaded.mtp_head is not None
        assert sorted(loaded.mtp_head) == ["auxiliary", "draft"]
        assert loaded.mtp_head.mtp_head is not None
        assert sorted(loaded.mtp_head.mtp_head) == ["next", "next_auxiliary"]

    def test_self_referential_mtp_head_fails_before_writing(self, tmp_path):
        pkg = ModelPackage({"model": _make_simple_model("target")})
        pkg.mtp_head = pkg
        output = tmp_path / "output"

        with pytest.raises(ValueError, match="must not contain a cycle"):
            pkg.save(str(output))

        assert not output.exists()

    def test_cyclic_nested_mtp_heads_fail_before_writing(self, tmp_path):
        pkg = ModelPackage({"model": _make_simple_model("target")})
        sidecar = ModelPackage({"model": _make_simple_model("draft")})
        pkg.mtp_head = sidecar
        sidecar.mtp_head = pkg
        output = tmp_path / "output"

        with pytest.raises(ValueError, match="must not contain a cycle"):
            pkg.save(str(output))

        assert not output.exists()

    def test_reserved_manifest_component_fails_before_writing(self, tmp_path):
        pkg = ModelPackage(
            {
                "model": _make_simple_model("target"),
                ".mobius-package.json": _make_simple_model("collision"),
            }
        )
        output = tmp_path / "output"

        with pytest.raises(ValueError, match="reserved for package metadata"):
            pkg.save(str(output))

        assert not output.exists()

    @pytest.mark.parametrize("name", ["", ".", "..", "../outside", "nested/model"])
    def test_unsafe_component_name_fails_before_writing(self, tmp_path, name):
        pkg = ModelPackage(
            {
                "model": _make_simple_model("target"),
                name: _make_simple_model("unsafe"),
            }
        )
        output = tmp_path / "output"

        with pytest.raises(ValueError, match="must be a non-empty path segment"):
            pkg.save(str(output))

        assert not output.exists()

    def test_removing_mtp_head_removes_stale_sidecar(self, tmp_path):
        pkg = ModelPackage({"model": _make_simple_model("target")})
        pkg.mtp_head = ModelPackage({"model": _make_simple_model("draft")})
        pkg.save(str(tmp_path))

        pkg.mtp_head = None
        pkg.save(str(tmp_path))
        loaded = ModelPackage.load(str(tmp_path))

        assert sorted(loaded) == ["model"]
        assert loaded.mtp_head is None
        assert not (tmp_path / ".mobius-mtp").exists()

    def test_manifest_artifact_collision_fails_before_writing(self, tmp_path):
        output = tmp_path / "output"
        output.mkdir()
        (output / ".mobius-package.json").write_text(
            '{"format": "mobius.model-package.v1", "mtp_head": "policies"}\n'
        )
        pkg = ModelPackage({"model": _make_simple_model("target")})
        pkg.add_policy_component("sample", build_greedy_sampler())

        with pytest.raises(ValueError, match="collides with an active artifact namespace"):
            pkg.save(str(output))

        assert not (output / "model.onnx").exists()

    def test_resaving_mtp_head_replaces_stale_components(self, tmp_path):
        pkg = ModelPackage({"model": _make_simple_model("target")})
        pkg.mtp_head = ModelPackage(
            {
                "draft": _make_simple_model("draft"),
                "stale": _make_simple_model("stale"),
            }
        )
        pkg.save(str(tmp_path))

        pkg.mtp_head = ModelPackage({"model": _make_simple_model("replacement")})
        pkg.save(str(tmp_path))
        loaded = ModelPackage.load(str(tmp_path))

        assert loaded.mtp_head is not None
        assert sorted(loaded.mtp_head) == ["model"]
        assert not (tmp_path / ".mobius-mtp" / "stale").exists()

    def test_sidecar_symlink_fails_before_writing(self, tmp_path):
        output = tmp_path / "output"
        output.mkdir()
        external = tmp_path / "external"
        external.mkdir()
        (output / ".mobius-mtp").symlink_to(external, target_is_directory=True)
        pkg = ModelPackage({"model": _make_simple_model("target")})
        pkg.mtp_head = ModelPackage({"model": _make_simple_model("draft")})

        with pytest.raises(ValueError, match="must be a real directory"):
            pkg.save(str(output))

        assert not (output / "model.onnx").exists()
        assert not list(external.iterdir())

    def test_manifest_symlink_fails_before_writing(self, tmp_path):
        output = tmp_path / "output"
        output.mkdir()
        external = tmp_path / "external.json"
        external.write_text("unchanged")
        (output / ".mobius-package.json").symlink_to(external)
        pkg = ModelPackage({"model": _make_simple_model("target")})

        with pytest.raises(ValueError, match="manifest must not be a symlink"):
            pkg.save(str(output))

        assert not (output / "model.onnx").exists()
        assert external.read_text() == "unchanged"

    def test_component_directory_symlink_fails_before_writing(self, tmp_path):
        output = tmp_path / "output"
        output.mkdir()
        external = tmp_path / "external"
        external.mkdir()
        (output / "decoder").symlink_to(external, target_is_directory=True)
        pkg = ModelPackage(
            {
                "decoder": _make_simple_model("decoder"),
                "embedding": _make_simple_model("embedding"),
            }
        )

        with pytest.raises(ValueError, match=r"component 'decoder'.*real directory"):
            pkg.save(str(output))

        assert not (output / "embedding").exists()
        assert not list(external.iterdir())

    def test_flat_model_symlink_fails_before_writing(self, tmp_path):
        output = tmp_path / "output"
        output.mkdir()
        external = tmp_path / "external.onnx"
        external.write_text("unchanged")
        (output / "model.onnx").symlink_to(external)
        pkg = ModelPackage({"model": _make_simple_model("target")})

        with pytest.raises(ValueError, match="model output must not be a symlink"):
            pkg.save(str(output))

        assert external.read_text() == "unchanged"

    @pytest.mark.parametrize("name", ["policies", "adapters"])
    def test_artifact_namespace_component_fails_before_writing(self, tmp_path, name):
        pkg = ModelPackage(
            {
                "model": _make_simple_model("target"),
                name: _make_simple_model(name),
            }
        )
        output = tmp_path / "output"

        with pytest.raises(ValueError, match="reserved for package artifacts"):
            pkg.save(str(output))

        assert not output.exists()

    def test_case_insensitive_component_collision_fails_before_writing(self, tmp_path):
        pkg = ModelPackage(
            {
                "model": _make_simple_model("lower"),
                "MODEL": _make_simple_model("upper"),
            }
        )
        output = tmp_path / "output"

        with pytest.raises(ValueError, match="distinct when compared case-insensitively"):
            pkg.save(str(output))

        assert not output.exists()

    def test_flat_safetensors_symlink_fails_before_writing(self, tmp_path):
        output = tmp_path / "output"
        output.mkdir()
        external = tmp_path / "external.safetensors"
        external.write_text("unchanged")
        (output / "model.safetensors").symlink_to(external)
        pkg = ModelPackage({"model": _make_simple_model("target")})

        with pytest.raises(ValueError, match="model output must not be a symlink"):
            pkg.save(str(output), external_data="safetensors")

        assert external.read_text() == "unchanged"

    @pytest.mark.parametrize("sidecar_name", [".", ".."])
    def test_load_rejects_escaping_sidecar_name(self, tmp_path, sidecar_name):
        (tmp_path / ".mobius-package.json").write_text(
            f'{{"format": "mobius.model-package.v1", "mtp_head": "{sidecar_name}"}}\n'
        )

        with pytest.raises(ValueError, match="Invalid ModelPackage manifest"):
            ModelPackage.load(str(tmp_path))

    def test_save_creates_directory(self, tmp_path):
        outdir = tmp_path / "nested" / "dir"
        pkg = ModelPackage({"m": _make_simple_model()})
        pkg.save(str(outdir))
        assert (outdir / "model.onnx").exists()

    def test_policy_components_roundtrip(self, tmp_path):
        pkg = ModelPackage({"model": _make_simple_model()})
        pkg.add_policy_component("sample", build_greedy_sampler())
        pkg.save(str(tmp_path))

        assert (tmp_path / "policies" / "sample.onnx").exists()
        loaded = ModelPackage.load(str(tmp_path))
        assert loaded.policy_components["sample"].contract_id == "onnx-genai.token-sampler@1"

    def test_save_namespaces_component_symbols_without_mutating_package(self, tmp_path):
        input_value = ir.Value(
            name="input",
            type=ir.TensorType(ir.DataType.FLOAT),
            shape=ir.Shape(["batch", "sequence"]),
        )
        input_value.shape.set_denotation(0, "DATA_BATCH")
        intermediate = ir.Value(
            name="intermediate",
            type=ir.TensorType(ir.DataType.FLOAT),
            shape=ir.Shape(["batch", "sequence"]),
        )
        output_value = ir.Value(
            name="output",
            type=ir.TensorType(ir.DataType.FLOAT),
            shape=ir.Shape(["batch", "sequence"]),
        )
        model = ir.Model(
            ir.Graph(
                [input_value],
                [output_value],
                nodes=[
                    ir.Node("", "Identity", [input_value], outputs=[intermediate]),
                    ir.Node("", "Identity", [intermediate], outputs=[output_value]),
                ],
                name="symbolic",
            ),
            ir_version=10,
        )
        pkg = ModelPackage({"decoder": model})

        pkg.save(str(tmp_path))

        saved = ir.load(tmp_path / "model.onnx")
        assert [str(dimension) for dimension in saved.graph.inputs[0].shape] == [
            "component.decoder.batch",
            "component.decoder.sequence",
        ]
        assert saved.graph.inputs[0].shape.get_denotation(0) == "DATA_BATCH"
        assert all(
            dimension.value is None for dimension in next(iter(saved.graph)).outputs[0].shape
        )
        assert [str(dimension) for dimension in model.graph.inputs[0].shape] == [
            "batch",
            "sequence",
        ]

    def test_save_anonymizes_nested_graph_intermediate_symbols(self):
        condition = ir.Value(
            name="condition",
            type=ir.TensorType(ir.DataType.BOOL),
            shape=ir.Shape([]),
        )
        data = ir.Value(
            name="data",
            type=ir.TensorType(ir.DataType.FLOAT),
            shape=ir.Shape(["batch", "sequence"]),
        )
        branch_output = ir.Value(
            name="branch_output",
            type=ir.TensorType(ir.DataType.FLOAT),
            shape=ir.Shape(["batch", "branch_sequence"]),
        )
        branch = ir.Graph(
            [],
            [branch_output],
            nodes=[ir.Node("", "Identity", [data], outputs=[branch_output])],
        )
        output = ir.Value(
            name="output",
            type=ir.TensorType(ir.DataType.FLOAT),
            shape=ir.Shape(["batch", "sequence"]),
        )
        model = ir.Model(
            ir.Graph(
                [condition, data],
                [output],
                nodes=[
                    ir.Node(
                        "",
                        "If",
                        [condition],
                        attributes={
                            "then_branch": ir.AttrGraph("then_branch", branch),
                            "else_branch": ir.AttrGraph("else_branch", branch),
                        },
                        outputs=[output],
                    )
                ],
            ),
            ir_version=10,
        )

        with _namespaced_symbolic_dimensions(model, "component.decoder"):
            assert all(dimension.value is None for dimension in branch_output.shape)

        assert str(branch_output.shape[1]) == "branch_sequence"

    def test_save_preserves_public_policy_symbols(self, tmp_path):
        sampler = build_greedy_sampler()
        pkg = ModelPackage({"model": _make_simple_model()})
        pkg.add_policy_component("sample", sampler)

        pkg.save(str(tmp_path))

        saved = ir.load(tmp_path / "policies" / "sample.onnx")
        assert str(saved.graph.inputs[0].shape[0]) == "batch"
        assert str(saved.graph.inputs[0].shape[1]) == "vocabulary"
        assert str(sampler.model.graph.inputs[0].shape[0]) == "batch"


class TestModelPackageApplyWeights:
    def test_single_component(self):
        config = make_config()
        module = CausalLMModel(config)
        pkg = build_from_module(module, config)
        model = pkg["model"]

        init_names = list(model.graph.initializers.keys())
        name = init_names[0]
        shape = list(model.graph.initializers[name].shape)
        weight = torch.ones(shape)

        pkg.apply_weights({name: weight})
        assert model.graph.initializers[name].const_value is not None

    def test_multi_component_with_prefix_map(self):
        config = make_config()
        m1 = CausalLMModel(config)
        m2 = CausalLMModel(config)
        pkg1 = build_from_module(m1, config)
        pkg2 = build_from_module(m2, config)
        model1 = pkg1["model"]
        model2 = pkg2["model"]

        pkg = ModelPackage({"text": model1, "vision_encoder": model2})

        # Get a weight name from model1
        init_name = next(iter(model1.graph.initializers.keys()))
        shape = list(model1.graph.initializers[init_name].shape)
        weight = torch.ones(shape)

        # Route via prefix
        pkg.apply_weights(
            {f"text.{init_name}": weight},
            prefix_map={"text.": "text", "vision_encoder.": "vision_encoder"},
        )
        assert model1.graph.initializers[init_name].const_value is not None


class TestBuildPackageFromModule:
    def test_returns_model_package(self):
        config = make_config()
        module = CausalLMModel(config)
        pkg = build_from_module(module, config)
        assert isinstance(pkg, ModelPackage)
        assert len(pkg) == 1
        assert "model" in pkg

    def test_package_model_has_correct_outputs(self):
        config = make_config()
        module = CausalLMModel(config)
        pkg = build_from_module(module, config)
        model = pkg["model"]
        output_names = [v.name for v in model.graph.outputs]
        assert "logits" in output_names

    def test_with_task_string(self):
        config = make_config()
        module = CausalLMModel(config)
        pkg = build_from_module(module, config, task="text-generation")
        assert isinstance(pkg, ModelPackage)

    def test_with_task_instance(self):
        config = make_config()
        module = CausalLMModel(config)
        pkg = build_from_module(module, config, task=CausalLMTask())
        assert isinstance(pkg, ModelPackage)

    def test_config_stored(self):
        config = make_config()
        module = CausalLMModel(config)
        pkg = build_from_module(module, config)
        assert pkg.config is config

    def test_package_save_load_roundtrip(self, tmp_path):
        config = make_config()
        module = CausalLMModel(config)
        pkg = build_from_module(module, config)

        pkg.save(str(tmp_path), check_weights=False)
        loaded = ModelPackage.load(str(tmp_path))

        assert len(loaded) == 1
        assert loaded["model"].graph.num_nodes() == pkg["model"].graph.num_nodes()


class TestMultiModalPackageIntegration:
    """Integration tests for multimodal model packages."""

    def _make_multimodal_config(self):
        return make_config(
            sliding_window=8,
            layer_types=["full_attention", "sliding_attention"],
            attn_qk_norm=True,
            rope_local_base_freq=10_000.0,
            vision=VisionConfig(
                hidden_size=64,
                intermediate_size=128,
                num_hidden_layers=2,
                num_attention_heads=4,
                image_size=32,
                patch_size=8,
                norm_eps=1e-6,
                image_token_id=999,
            ),
            image_token_id=999,
        )

    def test_build_multimodal_package(self):
        config = self._make_multimodal_config()
        module = Gemma3MultiModalModel(config)
        task = VisionLanguageTask()
        pkg = task.build(module, config)
        assert isinstance(pkg, ModelPackage)
        assert set(pkg.keys()) == {"decoder", "vision_encoder", "embedding"}
        assert pkg["decoder"].graph.num_nodes() > 0

    def test_multimodal_package_save_load(self, tmp_path):
        config = self._make_multimodal_config()
        module = Gemma3MultiModalModel(config)
        task = VisionLanguageTask()
        pkg = task.build(module, config)

        pkg.save(str(tmp_path), check_weights=False)
        loaded = ModelPackage.load(str(tmp_path))
        assert len(loaded) == len(pkg)
        for name in pkg:
            assert loaded[name].graph.num_nodes() == pkg[name].graph.num_nodes()

    def test_multimodal_model_has_vision_params(self):
        config = self._make_multimodal_config()
        module = Gemma3MultiModalModel(config)
        task = VisionLanguageTask()
        pkg = task.build(module, config)

        # Vision model has vision_tower and projector params
        vision_inits = list(pkg["vision_encoder"].graph.initializers.keys())
        assert any("vision_tower" in n for n in vision_inits)
        assert any("multi_modal_projector" in n for n in vision_inits)

        # Decoder has language model params
        decoder_inits = list(pkg["decoder"].graph.initializers.keys())
        assert any("lm_head" in n for n in decoder_inits)

        # Embedding has embed_tokens
        embed_inits = list(pkg["embedding"].graph.initializers.keys())
        assert any("embed_tokens" in n for n in embed_inits)


class TestApplyWeightsLogging:
    """Tests for unmapped-weight warnings and DEBUG mapping logs."""

    def test_unmapped_weights_logged_as_info(self, caplog):
        """Weights not matching any ONNX initializer produce an INFO message."""
        config = make_config()
        module = CausalLMModel(config)
        pkg = build_from_module(module, config)
        model = pkg["model"]

        init_name = next(iter(model.graph.initializers.keys()))
        shape = list(model.graph.initializers[init_name].shape)

        state_dict = {
            init_name: torch.ones(shape),
            "unmapped.weight": torch.zeros(4, 4),
        }

        with caplog.at_level(logging.INFO, logger="mobius"):
            pkg.apply_weights(state_dict)

        assert "unmapped.weight" in caplog.text
        assert "(4, 4)" in caplog.text
        assert "1 weight(s) not applied" in caplog.text

    def test_all_weights_mapped_no_info_message(self, caplog):
        """When all weights are mapped, no unmapped INFO message is emitted."""
        config = make_config()
        module = CausalLMModel(config)
        pkg = build_from_module(module, config)
        model = pkg["model"]

        init_name = next(iter(model.graph.initializers.keys()))
        shape = list(model.graph.initializers[init_name].shape)

        state_dict = {init_name: torch.ones(shape)}

        with caplog.at_level(logging.INFO, logger="mobius"):
            pkg.apply_weights(state_dict)

        assert "not applied" not in caplog.text

    def test_debug_log_includes_applied_weights(self, caplog):
        """At DEBUG level, applied weights are logged."""
        config = make_config()
        module = CausalLMModel(config)
        pkg = build_from_module(module, config)
        model = pkg["model"]

        init_name = next(iter(model.graph.initializers.keys()))
        shape = list(model.graph.initializers[init_name].shape)

        state_dict = {init_name: torch.ones(shape)}

        with caplog.at_level(logging.DEBUG, logger="mobius"):
            pkg.apply_weights(state_dict)

        assert "Applied 1 of 1 weight(s)" in caplog.text
        assert init_name in caplog.text

    def test_empty_state_dict_no_info_message(self, caplog):
        """An empty state dict produces no unmapped INFO message."""
        config = make_config()
        module = CausalLMModel(config)
        pkg = build_from_module(module, config)

        with caplog.at_level(logging.INFO, logger="mobius"):
            pkg.apply_weights({})

        assert "not applied" not in caplog.text
