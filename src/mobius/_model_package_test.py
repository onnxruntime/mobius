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
from mobius._model_package import ModelPackage, _make_progress_callback
from mobius._testing import make_config
from mobius.generation import PolicyRole, build_greedy_sampler
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
        assert loaded.policy_components["sample"].role is PolicyRole.TOKEN_SAMPLER


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
