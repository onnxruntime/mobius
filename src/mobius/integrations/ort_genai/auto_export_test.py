# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for the ORT-GenAI auto-export pipeline."""

from __future__ import annotations

import json
import os
from unittest import mock

import numpy as np
import pytest

from mobius.integrations.ort_genai.auto_export import (
    _copy_tokenizer_files,
    _copy_tokenizer_files_from_local,
    _graph_input_names,
    _resolve_ort_genai_model_type,
    _write_genai_config,
    _write_processor_config,
    write_ort_genai_config,
)


class TestResolveOrtGenaiModelType:
    def test_known_model_type(self):
        assert _resolve_ort_genai_model_type("qwen3") == "qwen2"
        assert _resolve_ort_genai_model_type("gemma2") == "gemma"
        assert _resolve_ort_genai_model_type("llama") == "llama"

    def test_unknown_model_type_passthrough(self):
        assert _resolve_ort_genai_model_type("my_custom") == "my_custom"

    def test_phi4mm_model_types(self):
        assert _resolve_ort_genai_model_type("phi4mm") == "phi4mm"
        assert _resolve_ort_genai_model_type("phi4_multimodal") == "phi4mm"
        assert _resolve_ort_genai_model_type("phi") == "phi"


class TestWriteProcessorConfig:
    def test_no_vision_returns_none(self, tmp_path):
        config = mock.MagicMock(spec=[])
        del config.vision  # ensure no vision attribute
        assert _write_processor_config(config, str(tmp_path)) is None

    def test_writes_vision_config(self, tmp_path):
        vision = mock.MagicMock()
        vision.image_size = 224
        vision.patch_size = 16
        config = mock.MagicMock()
        config.vision = vision

        path = _write_processor_config(config, str(tmp_path))
        assert path is not None
        with open(path) as f:
            data = json.load(f)
        assert data["image_size"] == 224
        assert data["patch_size"] == 16


class TestCopyTokenizerFiles:
    def test_copies_available_files(self, tmp_path):
        # Create a fake tokenizer file to "download"
        fake_src = tmp_path / "src"
        fake_src.mkdir()
        (fake_src / "tokenizer.json").write_text('{"test": true}')
        (fake_src / "chat_template.jinja").write_text("{{ messages }}")

        with mock.patch("huggingface_hub.hf_hub_download") as mock_dl:
            mock_dl.side_effect = lambda model_id, filename: (
                str(fake_src / filename)
                if (fake_src / filename).exists()
                else (_ for _ in ()).throw(OSError("not found"))
            )

            dst = tmp_path / "output"
            dst.mkdir()
            copied = _copy_tokenizer_files("fake/model", str(dst))

        assert "tokenizer.json" in copied
        assert (dst / "tokenizer.json").exists()
        assert "chat_template.jinja" in copied
        assert (dst / "chat_template.jinja").exists()


class TestCopyTokenizerFilesFromLocal:
    """Tests for _copy_tokenizer_files_from_local."""

    def test_copies_present_files(self, tmp_path):
        """Copies tokenizer files that exist in the source directory."""
        src = tmp_path / "model"
        src.mkdir()
        (src / "tokenizer.json").write_text('{"test": true}')
        (src / "tokenizer_config.json").write_text('{"model_type": "llama"}')
        (src / "chat_template.jinja").write_text("{{ messages }}")

        dst = tmp_path / "output"
        dst.mkdir()
        copied = _copy_tokenizer_files_from_local(str(src), str(dst))

        assert set(copied) == {
            "tokenizer.json",
            "tokenizer_config.json",
            "chat_template.jinja",
        }
        assert (dst / "tokenizer.json").read_text() == '{"test": true}'
        assert (dst / "chat_template.jinja").read_text() == "{{ messages }}"

    def test_skips_absent_files(self, tmp_path):
        """Files not present in the source directory are silently skipped."""
        src = tmp_path / "model"
        src.mkdir()
        # Only tokenizer.json present — tokenizer.model (SentencePiece), etc. absent

        (src / "tokenizer.json").write_text("{}")

        dst = tmp_path / "output"
        dst.mkdir()
        copied = _copy_tokenizer_files_from_local(str(src), str(dst))

        assert copied == ["tokenizer.json"]
        assert not (dst / "tokenizer.model").exists()

    def test_empty_source_returns_empty_list(self, tmp_path):
        """No tokenizer files in source returns an empty list."""
        src = tmp_path / "model"
        src.mkdir()
        dst = tmp_path / "output"
        dst.mkdir()
        copied = _copy_tokenizer_files_from_local(str(src), str(dst))
        assert copied == []

    def test_missing_source_dir_warns_and_returns_empty(self, tmp_path, caplog):
        """Non-existent source_dir emits a warning and returns empty list."""
        import logging

        dst = tmp_path / "output"
        dst.mkdir()
        with caplog.at_level(
            logging.WARNING, logger="mobius.integrations.ort_genai.auto_export"
        ):
            copied = _copy_tokenizer_files_from_local(str(tmp_path / "nonexistent"), str(dst))
        assert copied == []
        assert "does not exist" in caplog.text


class TestWriteOrtGenaiConfigLocalDir:
    """Tests for write_ort_genai_config with local_config_dir."""

    @staticmethod
    def _make_pkg():
        import dataclasses

        from mobius._model_package import ModelPackage

        @dataclasses.dataclass
        class FakeConfig:
            model_type: str = "llama"
            vocab_size: int = 256
            hidden_size: int = 64
            num_hidden_layers: int = 2
            num_attention_heads: int = 4
            num_key_value_heads: int = 2
            head_dim: int = 16
            max_position_embeddings: int = 128

        return ModelPackage({"model": mock.MagicMock()}, config=FakeConfig())

    def test_local_config_dir_copies_tokenizer_files(self, tmp_path):
        """When local_config_dir is set, tokenizer files are copied from it."""
        src = tmp_path / "local_model"
        src.mkdir()
        (src / "tokenizer.json").write_text('{"local": true}')
        (src / "tokenizer_config.json").write_text('{"type": "llama"}')

        out = tmp_path / "output"
        out.mkdir()
        pkg = self._make_pkg()

        result = write_ort_genai_config(
            pkg,
            str(out),
            local_config_dir=str(src),
        )

        assert "tokenizer.json" in result
        assert (out / "tokenizer.json").read_text() == '{"local": true}'
        assert "tokenizer_config.json" in result

    def test_hf_model_id_takes_precedence_over_local_dir(self, tmp_path):
        """When both hf_model_id and local_config_dir are set, HF takes precedence."""
        src = tmp_path / "local_model"
        src.mkdir()
        (src / "tokenizer.json").write_text('{"local": true}')

        out = tmp_path / "output"
        out.mkdir()
        pkg = self._make_pkg()

        with (
            mock.patch(
                "mobius.integrations.ort_genai.auto_export._copy_tokenizer_files",
                return_value=["tokenizer.json"],
            ) as mock_hf,
            mock.patch(
                "mobius.integrations.ort_genai.auto_export._copy_tokenizer_files_from_local",
                return_value=[],
            ) as mock_local,
            mock.patch(
                "transformers.AutoConfig.from_pretrained",
                return_value=mock.MagicMock(
                    model_type="llama",
                    bos_token_id=1,
                    eos_token_id=2,
                    pad_token_id=0,
                ),
            ),
        ):
            (out / "tokenizer.json").write_text("{}")  # pretend HF copy happened
            write_ort_genai_config(
                pkg,
                str(out),
                hf_model_id="meta-llama/Llama-3-8B",
                local_config_dir=str(src),
            )

        mock_hf.assert_called_once()
        mock_local.assert_not_called()


class TestExportForOrtGenai:
    """Unit tests for write_ort_genai_config()."""

    @staticmethod
    def _make_pkg():
        """Build a minimal LLM-only ModelPackage with a fake config."""
        import dataclasses

        from mobius._model_package import ModelPackage

        @dataclasses.dataclass
        class FakeConfig:
            model_type: str = "qwen2"
            vocab_size: int = 256
            hidden_size: int = 64
            num_hidden_layers: int = 2
            num_attention_heads: int = 4
            num_key_value_heads: int = 2
            head_dim: int = 16
            max_position_embeddings: int = 128

        pkg = ModelPackage({"model": mock.MagicMock()}, config=FakeConfig())
        return pkg

    def test_genai_config_json_is_written(self, tmp_path):
        """genai_config.json is always written to the output directory."""
        pkg = self._make_pkg()
        result = write_ort_genai_config(pkg, str(tmp_path))

        assert "genai_config" in result
        assert os.path.isfile(result["genai_config"])
        with open(result["genai_config"]) as f:
            data = json.load(f)
        assert "model" in data
        assert data["model"]["type"] == "qwen2"

    def test_processor_config_written_with_vision(self, tmp_path):
        """processor_config.json is written when pkg.config.vision is set."""
        import dataclasses

        from mobius._model_package import ModelPackage
        from mobius.integrations.ort_genai.auto_export import write_ort_genai_config

        @dataclasses.dataclass
        class FakeVision:
            image_size: int = 448
            patch_size: int = 14

        @dataclasses.dataclass
        class FakeConfig:
            model_type: str = "qwen2"
            vocab_size: int = 256
            hidden_size: int = 64
            num_hidden_layers: int = 2
            num_attention_heads: int = 4
            num_key_value_heads: int = 2
            head_dim: int = 16
            vision: FakeVision = dataclasses.field(default_factory=FakeVision)

        pkg = ModelPackage(
            {
                "model": mock.MagicMock(),
                "vision": mock.MagicMock(),
                "embedding": mock.MagicMock(),
            },
            config=FakeConfig(),
        )
        result = write_ort_genai_config(pkg, str(tmp_path))

        assert "processor_config" in result
        assert os.path.isfile(result["processor_config"])
        with open(result["processor_config"]) as f:
            data = json.load(f)
        assert data["image_size"] == 448

    def test_processor_config_not_written_without_vision(self, tmp_path):
        """processor_config.json is NOT written when pkg.config has no vision attr."""
        from mobius.integrations.ort_genai.auto_export import write_ort_genai_config

        pkg = self._make_pkg()
        result = write_ort_genai_config(pkg, str(tmp_path))

        assert "processor_config" not in result
        assert not os.path.exists(os.path.join(str(tmp_path), "processor_config.json"))

    def test_gemma4_image_processor_json_written(self, tmp_path):
        """Gemma4 writes image_processor.json with onnxruntime-extensions transforms pipeline."""
        import dataclasses

        from mobius._model_package import ModelPackage
        from mobius.integrations.ort_genai.auto_export import write_ort_genai_config

        @dataclasses.dataclass
        class FakeVision:
            image_size: int = 448
            patch_size: int = 16
            mm_tokens_per_image: int = 260
            pooling_kernel_size: int = 3

        @dataclasses.dataclass
        class FakeConfig:
            model_type: str = "gemma4"
            vocab_size: int = 262144
            hidden_size: int = 1536
            num_hidden_layers: int = 35
            num_attention_heads: int = 8
            num_key_value_heads: int = 1
            head_dim: int = 256
            vision: FakeVision = dataclasses.field(default_factory=FakeVision)

        pkg = ModelPackage(
            {
                "model": mock.MagicMock(),
                "vision": mock.MagicMock(),
                "embedding": mock.MagicMock(),
            },
            config=FakeConfig(),
        )
        result = write_ort_genai_config(pkg, str(tmp_path))

        # Should write image_processor.json, not processor_config.json
        assert "processor_config" in result
        proc_path = result["processor_config"]
        assert proc_path.endswith("image_processor.json")
        assert os.path.isfile(proc_path)
        assert not os.path.exists(os.path.join(str(tmp_path), "processor_config.json"))

        with open(proc_path) as f:
            data = json.load(f)

        # Verify onnxruntime-extensions transforms pipeline structure
        assert "processor" in data
        assert "transforms" in data["processor"]
        transforms = data["processor"]["transforms"]
        assert len(transforms) == 2

        # First op: DecodeImage
        op0 = transforms[0]["operation"]
        assert op0["type"] == "DecodeImage"
        assert op0["attrs"]["color_space"] == "RGB"

        # Second op: Gemma4ImageTransform with correct attrs from config
        op1 = transforms[1]["operation"]
        assert op1["type"] == "Gemma4ImageTransform"
        assert op1["attrs"]["patch_size"] == 16
        assert op1["attrs"]["max_soft_tokens"] == 260
        assert op1["attrs"]["pooling_kernel_size"] == 3

    def test_tokenizer_not_copied_without_model_id(self, tmp_path):
        """No tokenizer files copied when hf_model_id=None."""
        from mobius.integrations.ort_genai.auto_export import write_ort_genai_config

        pkg = self._make_pkg()
        with mock.patch(
            "mobius.integrations.ort_genai.auto_export._copy_tokenizer_files"
        ) as mock_copy:
            write_ort_genai_config(pkg, str(tmp_path), hf_model_id=None)
        mock_copy.assert_not_called()

    def test_tokenizer_copied_when_model_id_provided(self, tmp_path):
        """Tokenizer files are copied when hf_model_id is provided."""
        from mobius.integrations.ort_genai.auto_export import write_ort_genai_config

        pkg = self._make_pkg()
        with (
            mock.patch(
                "mobius.integrations.ort_genai.auto_export._copy_tokenizer_files",
                return_value=["tokenizer.json"],
            ) as mock_copy,
            mock.patch("transformers.AutoConfig.from_pretrained") as mock_hf,
        ):
            mock_hf.return_value = mock.MagicMock(
                model_type="qwen2", bos_token_id=1, eos_token_id=2, pad_token_id=0
            )
            result = write_ort_genai_config(pkg, str(tmp_path), hf_model_id="fake/model")

        mock_copy.assert_called_once_with("fake/model", str(tmp_path))
        assert "tokenizer.json" in result

    def test_ep_default_normalizes_to_cpu(self, tmp_path):
        """ep='default' is normalized to cpu (provider_options=[])."""
        from mobius.integrations.ort_genai.auto_export import write_ort_genai_config

        pkg = self._make_pkg()
        result = write_ort_genai_config(pkg, str(tmp_path), ep="default")

        with open(result["genai_config"]) as f:
            data = json.load(f)
        assert data["model"]["decoder"]["session_options"]["provider_options"] == []

    def test_ep_onnx_standard_normalizes_to_cpu(self, tmp_path):
        """ep='onnx-standard' is normalized to cpu (provider_options=[])."""
        from mobius.integrations.ort_genai.auto_export import write_ort_genai_config

        pkg = self._make_pkg()
        result = write_ort_genai_config(pkg, str(tmp_path), ep="onnx-standard")

        with open(result["genai_config"]) as f:
            data = json.load(f)
        assert data["model"]["decoder"]["session_options"]["provider_options"] == []

    def test_ep_cuda_passes_through(self, tmp_path):
        """ep='cuda' passes through to session_options with CUDA provider."""
        from mobius.integrations.ort_genai.auto_export import write_ort_genai_config

        pkg = self._make_pkg()
        result = write_ort_genai_config(pkg, str(tmp_path), ep="cuda")

        with open(result["genai_config"]) as f:
            data = json.load(f)
        provider_opts = data["model"]["decoder"]["session_options"]["provider_options"]
        assert len(provider_opts) == 1
        assert "CUDAExecutionProvider" in provider_opts[0]

    def test_raises_when_pkg_config_is_none(self, tmp_path):
        """ValueError is raised when pkg.config is None."""
        from mobius._model_package import ModelPackage
        from mobius.integrations.ort_genai.auto_export import write_ort_genai_config

        pkg = ModelPackage({"model": mock.MagicMock()}, config=None)
        with pytest.raises(ValueError, match="config"):
            write_ort_genai_config(pkg, str(tmp_path))

    def test_does_not_require_onnxruntime_genai(self, tmp_path):
        """write_ort_genai_config works without onnxruntime-genai installed."""
        import sys

        from mobius.integrations.ort_genai.auto_export import write_ort_genai_config

        pkg = self._make_pkg()
        # Remove onnxruntime_genai from sys.modules if present, then restore
        saved = sys.modules.pop("onnxruntime_genai", None)
        try:
            # Should not raise ImportError — ort-genai runtime is not needed
            result = write_ort_genai_config(pkg, str(tmp_path))
            assert "genai_config" in result
        finally:
            if saved is not None:
                sys.modules["onnxruntime_genai"] = saved

    def test_config_mode_model_type_propagated(self, tmp_path):
        """model.type in genai_config.json is correct when hf_model_id=None."""
        import dataclasses

        from mobius._model_package import ModelPackage
        from mobius.integrations.ort_genai.auto_export import write_ort_genai_config

        @dataclasses.dataclass
        class FakeConfig:
            # model_type now stored on ArchitectureConfig for --config mode
            model_type: str = "gemma2"
            vocab_size: int = 256
            hidden_size: int = 64
            num_hidden_layers: int = 2
            num_attention_heads: int = 4
            num_key_value_heads: int = 2
            head_dim: int = 16
            max_position_embeddings: int = 128

        pkg = ModelPackage({"model": mock.MagicMock()}, config=FakeConfig())
        result = write_ort_genai_config(pkg, str(tmp_path), hf_model_id=None)

        with open(result["genai_config"]) as f:
            data = json.load(f)
        # "gemma2" maps to "gemma" in _ORT_GENAI_MODEL_TYPE
        assert data["model"]["type"] == "gemma"

    def test_config_mode_token_ids_propagated(self, tmp_path):
        """bos/eos token IDs in genai_config.json come from config fields in --config mode."""
        import dataclasses

        from mobius._model_package import ModelPackage
        from mobius.integrations.ort_genai.auto_export import write_ort_genai_config

        @dataclasses.dataclass
        class FakeConfig:
            model_type: str = "llama"
            vocab_size: int = 256
            hidden_size: int = 64
            num_hidden_layers: int = 2
            num_attention_heads: int = 4
            num_key_value_heads: int = 2
            head_dim: int = 16
            max_position_embeddings: int = 128
            bos_token_id: int = 1
            eos_token_id: int = 2
            pad_token_id: int = 0

        pkg = ModelPackage({"model": mock.MagicMock()}, config=FakeConfig())
        result = write_ort_genai_config(pkg, str(tmp_path), hf_model_id=None)

        with open(result["genai_config"]) as f:
            data = json.load(f)
        assert data["model"]["bos_token_id"] == 1
        assert data["model"]["eos_token_id"] == 2
        assert data["model"]["pad_token_id"] == 0

    def test_config_mode_eos_token_id_as_list(self, tmp_path):
        """eos_token_id can be a list[int] (e.g. Gemma multi-stop tokens)."""
        import dataclasses

        from mobius._model_package import ModelPackage
        from mobius.integrations.ort_genai.auto_export import write_ort_genai_config

        @dataclasses.dataclass
        class FakeConfig:
            model_type: str = "gemma2"
            vocab_size: int = 256
            hidden_size: int = 64
            num_hidden_layers: int = 2
            num_attention_heads: int = 4
            num_key_value_heads: int = 2
            head_dim: int = 16
            max_position_embeddings: int = 128
            bos_token_id: int = 2
            eos_token_id: list = dataclasses.field(default_factory=lambda: [1, 106])
            pad_token_id: int = 0

        pkg = ModelPackage({"model": mock.MagicMock()}, config=FakeConfig())
        result = write_ort_genai_config(pkg, str(tmp_path), hf_model_id=None)

        with open(result["genai_config"]) as f:
            data = json.load(f)
        assert data["model"]["eos_token_id"] == [1, 106]


class TestGemma4GenaiConfig:
    """Tests for Gemma4-specific genai_config generation via graph introspection."""

    @staticmethod
    def _make_gemma4_pkg():
        """Build a mock Gemma4 VLM package with graph inputs."""
        import dataclasses

        from mobius._model_package import ModelPackage

        @dataclasses.dataclass
        class FakeVision:
            image_size: int = 448
            patch_size: int = 16
            mm_tokens_per_image: int = 256

        @dataclasses.dataclass
        class FakeConfig:
            model_type: str = "gemma4"
            vocab_size: int = 262144
            hidden_size: int = 2048
            num_hidden_layers: int = 26
            num_attention_heads: int = 8
            num_key_value_heads: int = 4
            head_dim: int = 256
            max_position_embeddings: int = 8192
            image_token_id: int = 255999
            vision: FakeVision = dataclasses.field(default_factory=FakeVision)

        # Mock graph inputs for each sub-model
        def _mock_model_with_inputs(names):
            inputs = []
            for n in names:
                inp = mock.MagicMock()
                inp.name = n
                inputs.append(inp)
            m = mock.MagicMock()
            m.graph.inputs = inputs
            return m

        decoder = _mock_model_with_inputs(
            [
                "inputs_embeds",
                "input_ids",
                "attention_mask",
                "position_ids",
                "past_key_values.0.key",
                "past_key_values.0.value",
            ]
        )
        vision = _mock_model_with_inputs(
            [
                "pixel_values",
                "pixel_position_ids",
            ]
        )
        embedding = _mock_model_with_inputs(
            [
                "input_ids",
                "image_features",
            ]
        )

        return ModelPackage(
            {
                "decoder": decoder,
                "vision": vision,
                "embedding": embedding,
            },
            config=FakeConfig(),
        )

    def test_gemma4_vision_inputs(self, tmp_path):
        """Gemma4 vision uses pixel_values + pixel_position_ids."""
        pkg = self._make_gemma4_pkg()
        path = _write_genai_config(
            pkg.config,
            str(tmp_path),
            pkg=pkg,
            ort_model_type="gemma4",
            ep="cpu",
            context_length=4096,
            bos_token_id=2,
            eos_token_id=1,
            pad_token_id=0,
            is_vlm=True,
            has_speech=False,
        )
        with open(path) as f:
            data = json.load(f)
        vision_inputs = data["model"]["vision"]["inputs"]
        assert "pixel_values" in vision_inputs
        assert "pixel_position_ids" in vision_inputs
        assert "image_grid_thw" not in vision_inputs
        assert data["model"]["vision"]["spatial_merge_size"] == 2

    def test_gemma4_decoder_has_input_ids_and_inputs_embeds(self, tmp_path):
        """Gemma4 decoder has both inputs_embeds and input_ids."""
        pkg = self._make_gemma4_pkg()
        path = _write_genai_config(
            pkg.config,
            str(tmp_path),
            pkg=pkg,
            ort_model_type="gemma4",
            ep="cpu",
            context_length=4096,
            bos_token_id=2,
            eos_token_id=1,
            pad_token_id=0,
            is_vlm=True,
            has_speech=False,
        )
        with open(path) as f:
            data = json.load(f)
        decoder_inputs = data["model"]["decoder"]["inputs"]
        assert "inputs_embeds" in decoder_inputs
        assert "input_ids" in decoder_inputs
        # KV cache templates are present
        assert decoder_inputs["past_key_names"] == "past_key_values.%d.key"

    def test_gemma4_embedding_inputs(self, tmp_path):
        """Gemma4 embedding inputs discovered from graph."""
        pkg = self._make_gemma4_pkg()
        path = _write_genai_config(
            pkg.config,
            str(tmp_path),
            pkg=pkg,
            ort_model_type="gemma4",
            ep="cpu",
            context_length=4096,
            bos_token_id=2,
            eos_token_id=1,
            pad_token_id=0,
            is_vlm=True,
            has_speech=False,
        )
        with open(path) as f:
            data = json.load(f)
        emb_inputs = data["model"]["embedding"]["inputs"]
        assert "input_ids" in emb_inputs
        assert "image_features" in emb_inputs


class TestGraphInputNames:
    """Tests for _graph_input_names() helper."""

    @staticmethod
    def _mock_model(names):
        inputs = []
        for n in names:
            inp = mock.MagicMock()
            inp.name = n
            inputs.append(inp)
        m = mock.MagicMock()
        m.graph.inputs = inputs
        return m

    def test_filters_kv_cache_inputs(self):
        """KV cache inputs (past_key_values.*) are filtered out."""
        model = self._mock_model(
            [
                "input_ids",
                "attention_mask",
                "past_key_values.0.key",
                "past_key_values.0.value",
                "past_key_values.1.key",
                "past_key_values.1.value",
            ]
        )
        result = _graph_input_names(model)
        assert result == ["input_ids", "attention_mask"]

    def test_filters_past_prefix(self):
        """Inputs starting with 'past_' are also filtered out."""
        model = self._mock_model(
            [
                "input_ids",
                "past_something",
            ]
        )
        result = _graph_input_names(model)
        assert result == ["input_ids"]

    def test_skips_none_names(self):
        """Inputs with name=None are skipped."""
        inp_good = mock.MagicMock()
        inp_good.name = "input_ids"
        inp_none = mock.MagicMock()
        inp_none.name = None
        m = mock.MagicMock()
        m.graph.inputs = [inp_good, inp_none]
        result = _graph_input_names(m)
        assert result == ["input_ids"]

    def test_returns_all_semantic_inputs(self):
        """All non-KV-cache inputs are returned in order."""
        model = self._mock_model(
            [
                "inputs_embeds",
                "input_ids",
                "attention_mask",
                "position_ids",
            ]
        )
        result = _graph_input_names(model)
        assert result == [
            "inputs_embeds",
            "input_ids",
            "attention_mask",
            "position_ids",
        ]


class TestGemma4RealModel:
    """Build a real tiny Gemma4 model and verify genai config inputs."""

    def test_gemma4_genai_config_from_real_model(self, tmp_path):
        """Build tiny Gemma4 VLM, generate genai config, verify inputs."""
        from mobius._builder import build_from_module
        from mobius._config_resolver import _default_task_for_model
        from mobius._configs import Gemma4Config, VisionConfig
        from mobius._registry import registry
        from mobius.tasks import get_task

        config = Gemma4Config(
            model_type="gemma4",
            num_hidden_layers=2,
            hidden_size=64,
            intermediate_size=128,
            num_attention_heads=4,
            num_key_value_heads=1,
            head_dim=16,
            vocab_size=256,
            rms_norm_eps=1e-6,
            hidden_act="silu",
            attn_qk_norm=True,
            layer_types=["sliding_attention", "full_attention"],
            sliding_window=8,
            global_head_dim=16,
            global_rope_theta=10_000.0,
            global_partial_rotary_factor=0.25,
            final_logit_softcapping=0.0,
            hidden_size_per_layer_input=0,
            image_token_id=255999,
            pad_token_id=0,
            tie_word_embeddings=True,
            vision=VisionConfig(
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=1,
                num_attention_heads=2,
                patch_size=16,
                norm_eps=1e-6,
            ),
        )
        model_cls = registry.get("gemma4")
        module = model_cls(config)
        task_name = _default_task_for_model("gemma4")
        task = get_task(task_name)
        pkg = build_from_module(module, config, task=task)
        pkg.config = config

        result = write_ort_genai_config(pkg, str(tmp_path))
        with open(result["genai_config"]) as f:
            data = json.load(f)

        # Decoder inputs introspected from graph
        decoder_inputs = data["model"]["decoder"]["inputs"]
        assert "inputs_embeds" in decoder_inputs
        assert "input_ids" in decoder_inputs
        assert "attention_mask" in decoder_inputs
        assert "position_ids" in decoder_inputs
        assert decoder_inputs["past_key_names"] == ("past_key_values.%d.key")

        # Vision inputs introspected from graph
        vision_inputs = data["model"]["vision"]["inputs"]
        assert "pixel_values" in vision_inputs
        assert "pixel_position_ids" in vision_inputs
        assert "image_grid_thw" not in vision_inputs

        # Embedding inputs introspected from graph
        emb_inputs = data["model"]["embedding"]["inputs"]
        assert "input_ids" in emb_inputs
        assert "image_features" in emb_inputs

        # Config-level properties are still present
        assert data["model"]["image_token_id"] == 255999
        assert data["model"]["vision"]["spatial_merge_size"] == 2
        assert data["model"]["vision"]["config_filename"] == "image_processor.json"

    def test_auto_export_produces_genai_config(self, tmp_path):
        """Mock build() to return a tiny package, verify genai_config."""
        import onnx_ir as ir

        from mobius._builder import build_from_module
        from mobius._configs import ArchitectureConfig
        from mobius._registry import registry
        from mobius.integrations.ort_genai.genai_config import (
            GenaiConfigGenerator,
        )

        # Build a tiny model
        config = ArchitectureConfig(
            hidden_size=64,
            intermediate_size=128,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            num_hidden_layers=2,
            vocab_size=256,
            max_position_embeddings=128,
            hidden_act="silu",
            rms_norm_eps=1e-6,
            rope_type="default",
            rope_theta=10000.0,
            pad_token_id=0,
        )
        module = registry.get("qwen2")(config)
        pkg = build_from_module(module, config)

        # Fill with random weights
        rng = np.random.default_rng(42)
        for model in pkg.values():
            for init in model.graph.initializers.values():
                if init.const_value is None:
                    shape = list(init.shape)
                    init.const_value = ir.Tensor(rng.standard_normal(shape).astype(np.float32))

        # Generate genai_config from the config
        gen = GenaiConfigGenerator.from_config(config, "qwen2")
        genai_config = gen.generate()

        assert "model" in genai_config
        assert genai_config["model"]["type"] == "qwen2"
        assert genai_config["model"]["vocab_size"] == 256
        assert genai_config["model"]["decoder"]["num_hidden_layers"] == 2

        # Save models and config
        output_dir = str(tmp_path / "export")
        os.makedirs(output_dir)
        pkg.save(output_dir, progress_bar=False)
        gen.write(output_dir)

        assert os.path.exists(os.path.join(output_dir, "model.onnx"))
        assert os.path.exists(os.path.join(output_dir, "genai_config.json"))

        with open(os.path.join(output_dir, "genai_config.json")) as f:
            saved = json.load(f)
        assert saved["model"]["type"] == "qwen2"

    def test_phi4mm_detection_and_config(self, tmp_path):
        """Simulate phi4mm auto-export: verify detection and config."""
        import onnx_ir as ir

        from mobius._builder import build_from_module
        from mobius._configs import ArchitectureConfig, AudioConfig, VisionConfig
        from mobius.integrations.ort_genai.genai_config import (
            GenaiConfigGenerator,
        )
        from mobius.models.phi import Phi4MMMultiModalModel
        from mobius.tasks import Phi4MMMultiModalTask

        # Build a tiny phi4mm model
        # LongRoPE requires rope_scaling with long/short factors
        # inv_freq has int(head_dim * partial_rotary_factor) // 2 elements
        rope_dim = int(16 * 0.75) // 2  # head_dim=16, partial_rotary_factor=0.75
        config = ArchitectureConfig(
            hidden_size=64,
            intermediate_size=128,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            num_hidden_layers=1,
            vocab_size=256,
            max_position_embeddings=128,
            hidden_act="silu",
            rms_norm_eps=1e-6,
            rope_type="longrope",
            rope_theta=10000.0,
            partial_rotary_factor=0.75,
            original_max_position_embeddings=128,
            rope_scaling={
                "long_factor": [1.0] * rope_dim,
                "short_factor": [1.0] * rope_dim,
            },
            pad_token_id=0,
            image_token_id=200010,
            vision=VisionConfig(
                hidden_size=64,
                intermediate_size=128,
                num_hidden_layers=1,
                num_attention_heads=4,
                image_size=28,
                patch_size=14,
                lora={"r": 4, "lora_alpha": 8},
            ),
            audio=AudioConfig(
                attention_dim=64,
                attention_heads=4,
                num_blocks=1,
                linear_units=128,
                kernel_size=3,
                input_size=80,
                token_id=200011,
                lora={"r": 4, "lora_alpha": 8},
            ),
        )
        module = Phi4MMMultiModalModel(config)
        pkg = build_from_module(module, config, task=Phi4MMMultiModalTask())

        # Verify 4-model split
        assert "vision" in pkg
        assert "speech" in pkg
        assert "embedding" in pkg
        assert "model" in pkg

        # Simulate auto_export detection logic
        is_vlm = "vision" in pkg and "embedding" in pkg
        has_speech = "speech" in pkg
        ort_model_type = "phi"  # HF model_type for phi4mm
        if ort_model_type == "phi" and has_speech:
            ort_model_type = "phi4mm"

        assert ort_model_type == "phi4mm"
        assert is_vlm
        assert has_speech

        # Build genai_config using the same logic as auto_export
        generator = GenaiConfigGenerator.from_config(config, ort_model_type)
        vision_kwargs = {
            "spatial_merge_size": None,
            "config_filename": "vision_processor.json",
            "input_names": {
                "pixel_values": "pixel_values",
                "image_sizes": "image_sizes",
            },
        }
        generator.with_vision(image_token_id=config.image_token_id, **vision_kwargs)
        generator.with_speech(audio_token_id=config.audio.token_id)

        genai_config = generator.generate()

        # Verify config structure
        model = genai_config["model"]
        assert model["type"] == "phi4mm"
        assert model["image_token_id"] == 200010
        assert model["audio_token_id"] == 200011

        # All 4 model sections present
        assert "decoder" in model
        assert "vision" in model
        assert "speech" in model
        assert "embedding" in model

        # Vision uses phi4mm-specific inputs
        assert model["vision"]["inputs"]["pixel_values"] == "pixel_values"
        assert model["vision"]["inputs"]["image_sizes"] == "image_sizes"
        assert "image_grid_thw" not in model["vision"]["inputs"]
        assert "spatial_merge_size" not in model["vision"]
        assert model["vision"]["config_filename"] == "vision_processor.json"

        # Speech section
        assert model["speech"]["inputs"]["audio_embeds"] == "audio_embeds"
        assert model["speech"]["inputs"]["audio_sizes"] == "audio_sizes"
        assert model["speech"]["inputs"]["audio_projection_mode"] == "audio_projection_mode"

        # Embedding includes audio_features
        assert model["embedding"]["inputs"]["audio_features"] == "audio_features"

        # Decoder uses inputs_embeds (multimodal)
        assert "inputs_embeds" in model["decoder"]["inputs"]

        # Save and verify files
        output_dir = str(tmp_path / "phi4mm_export")
        os.makedirs(output_dir)

        # Fill with random weights so save() doesn't complain
        rng = np.random.default_rng(42)
        for model in pkg.values():
            for init in model.graph.initializers.values():
                if init.const_value is None:
                    shape = list(init.shape)
                    init.const_value = ir.Tensor(rng.standard_normal(shape).astype(np.float32))

        pkg.save(output_dir, progress_bar=False)
        generator.write(output_dir)

        assert os.path.exists(os.path.join(output_dir, "genai_config.json"))
        # 4-model split produces subdirectories
        assert os.path.exists(os.path.join(output_dir, "vision"))
        assert os.path.exists(os.path.join(output_dir, "speech"))
        assert os.path.exists(os.path.join(output_dir, "embedding"))

        with open(os.path.join(output_dir, "genai_config.json")) as f:
            saved = json.load(f)
        assert saved["model"]["type"] == "phi4mm"
        assert "speech" in saved["model"]
