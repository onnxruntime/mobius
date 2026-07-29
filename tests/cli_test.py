# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Unit tests for the CLI (``__main__.py``).

These tests invoke ``main()`` directly with argv lists, so they do not
require network access. All build tests use ``--no-weights``.
"""

from __future__ import annotations

import os
import tempfile
from types import SimpleNamespace
from unittest import mock

import onnx
import pytest

from mobius.__main__ import _save_package, main


class TestCLIList:
    """Test the ``list`` subcommand."""

    def test_list_models(self, capsys):
        main(["list", "models"])
        out = capsys.readouterr().out
        assert "Supported model architectures" in out
        assert "llama" in out

    def test_list_tasks(self, capsys):
        main(["list", "tasks"])
        out = capsys.readouterr().out
        assert "Available tasks" in out
        assert "text-generation" in out

    def test_list_dtypes(self, capsys):
        main(["list", "dtypes"])
        out = capsys.readouterr().out
        assert "Available dtypes" in out
        assert "f32" in out


class TestCLIBuild:
    """Test the ``build`` subcommand with ``--no-weights``."""

    def test_build_no_weights_creates_model_onnx(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            main(["build", "--model", "Qwen/Qwen2.5-0.5B", tmpdir, "--no-weights"])
            assert os.path.isfile(os.path.join(tmpdir, "model.onnx"))

    def test_build_with_dtype(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            main(
                [
                    "build",
                    "--model",
                    "Qwen/Qwen2.5-0.5B",
                    tmpdir,
                    "--no-weights",
                    "--dtype",
                    "f16",
                ]
            )
            assert os.path.isfile(os.path.join(tmpdir, "model.onnx"))

    def test_build_encoder_decoder_produces_separate_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            main(["build", "--model", "facebook/bart-base", tmpdir, "--no-weights"])
            assert os.path.isfile(os.path.join(tmpdir, "encoder", "model.onnx"))
            assert os.path.isfile(os.path.join(tmpdir, "decoder", "model.onnx"))

    def test_build_missing_model_errors(self):
        with tempfile.TemporaryDirectory() as tmpdir, pytest.raises(SystemExit):
            main(["build", tmpdir])  # no --model or --config

    def test_text_only_with_config_errors(self):
        """--text-only is rejected on the --config (local dir) path."""
        with tempfile.TemporaryDirectory() as tmpdir, pytest.raises(SystemExit):
            main(["build", "--config", tmpdir, tmpdir, "--text-only", "--no-weights"])

    def test_text_only_with_component_errors(self):
        """--text-only is rejected when combined with --component."""
        with tempfile.TemporaryDirectory() as tmpdir, pytest.raises(SystemExit):
            main(
                [
                    "build",
                    "--model",
                    "google/gemma-4-12B",
                    tmpdir,
                    "--text-only",
                    "--component",
                    "vision_encoder",
                    "--no-weights",
                ]
            )

    def test_text_only_skips_diffusers_autodetect(self):
        """--text-only bypasses diffusers autodetect so build() validates it.

        Without this, a diffusers repo + --text-only would hit the autodetect
        branch and silently export a diffusion pipeline, ignoring the flag.
        """
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            mock.patch(
                "mobius._diffusers_builder._load_diffusers_pipeline_index"
            ) as mock_diffusers,
            mock.patch("mobius.__main__.build", return_value=mock.MagicMock()) as mock_build,
            mock.patch("mobius.__main__._save_package"),
        ):
            main(
                [
                    "build",
                    "--model",
                    "some/diffusion-repo",
                    tmpdir,
                    "--text-only",
                    "--no-weights",
                ]
            )

        mock_diffusers.assert_not_called()
        mock_build.assert_called_once()
        assert mock_build.call_args.kwargs.get("text_only") is True

    def test_glm_full_attention_passes_config_overrides(self):
        """--glm-full-attention disables DSA and MTP for registry builds."""
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            mock.patch("mobius.__main__.build", return_value=mock.MagicMock()) as mock_build,
            mock.patch("mobius.__main__._save_package"),
        ):
            main(
                [
                    "build",
                    "--model",
                    "THUDM/GLM-5.2-Air",
                    tmpdir,
                    "--no-weights",
                    "--glm-full-attention",
                ]
            )

        mock_build.assert_called_once()
        assert mock_build.call_args.kwargs.get("config_overrides") == {
            "use_dsa": False,
            "num_nextn_predict_layers": 0,
        }

    def test_build_static_cache(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            main(
                [
                    "build",
                    "--model",
                    "Qwen/Qwen2.5-0.5B",
                    tmpdir,
                    "--no-weights",
                    "--static-cache",
                ]
            )
            assert os.path.isfile(os.path.join(tmpdir, "model.onnx"))

    def test_max_seq_len_without_static_cache_errors(self):
        with tempfile.TemporaryDirectory() as tmpdir, pytest.raises(SystemExit):
            main(
                [
                    "build",
                    "--model",
                    "Qwen/Qwen2.5-0.5B",
                    tmpdir,
                    "--no-weights",
                    "--max-seq-len",
                    "512",
                ]
            )

    def test_static_cache_with_task_errors(self):
        """--static-cache cannot be combined with any --task."""
        with tempfile.TemporaryDirectory() as tmpdir, pytest.raises(SystemExit):
            main(
                [
                    "build",
                    "--model",
                    "Qwen/Qwen2.5-0.5B",
                    tmpdir,
                    "--no-weights",
                    "--static-cache",
                    "--task",
                    "text-generation",
                ]
            )

    def test_non_positive_max_seq_len_errors(self):
        with tempfile.TemporaryDirectory() as tmpdir, pytest.raises(SystemExit):
            main(
                [
                    "build",
                    "--model",
                    "Qwen/Qwen2.5-0.5B",
                    tmpdir,
                    "--no-weights",
                    "--static-cache",
                    "--max-seq-len",
                    "0",
                ]
            )

    def test_static_cache_with_max_seq_len(self):
        """--max-seq-len is passed through and sets cache dimensions."""
        max_seq_len = 256
        with tempfile.TemporaryDirectory() as tmpdir:
            main(
                [
                    "build",
                    "--model",
                    "Qwen/Qwen2.5-0.5B",
                    tmpdir,
                    "--no-weights",
                    "--static-cache",
                    "--max-seq-len",
                    str(max_seq_len),
                ]
            )
            model_path = os.path.join(tmpdir, "model.onnx")
            assert os.path.isfile(model_path)

            # Verify the cache input has the expected max_seq_len
            # dimension. Static cache shape: [batch, max_seq_len, kv_hidden]
            model = onnx.load(model_path)
            cache_inputs = [
                inp for inp in model.graph.input if inp.name.startswith("key_cache.")
            ]
            assert len(cache_inputs) > 0, "No key_cache inputs found"
            seq_dim = cache_inputs[0].type.tensor_type.shape.dim[1].dim_value
            assert seq_dim == max_seq_len, (
                f"key_cache.0 seq dimension is {seq_dim}, expected {max_seq_len}"
            )


class TestCLIInfo:
    """Test the ``info`` subcommand."""

    def test_info_known_model(self, capsys):
        main(["info", "Qwen/Qwen2.5-0.5B"])
        out = capsys.readouterr().out
        assert "qwen2" in out
        assert "Supported" in out


class TestCLIBuildRuntime:
    """Test the ``--runtime`` flag on the ``build`` subcommand."""

    def test_runtime_ort_genai_calls_write_ort_genai_config(self):
        """--runtime ort-genai calls write_ort_genai_config() after building."""
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            mock.patch(
                "mobius.integrations.ort_genai.write_ort_genai_config",
                return_value={},
            ) as mock_export,
        ):
            main(
                [
                    "build",
                    "--model",
                    "Qwen/Qwen2.5-0.5B",
                    tmpdir,
                    "--no-weights",
                    "--runtime",
                    "ort-genai",
                ]
            )

        mock_export.assert_called_once()
        call_kwargs = mock_export.call_args
        assert call_kwargs.kwargs.get("hf_model_id") == "Qwen/Qwen2.5-0.5B"

    def test_runtime_onnx_genai_calls_write_onnx_genai_config(self, capsys):
        """--runtime onnx-genai calls the unified config writer."""
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            mock.patch(
                "mobius.integrations.onnx_genai.write_onnx_genai_config",
                return_value={
                    "inference_metadata": "inference_metadata.yaml",
                    "mtp_config": "mtp_config.json",
                },
            ) as mock_export,
            mock.patch(
                "mobius.integrations.ort_genai.write_ort_genai_config",
                return_value={},
            ) as mock_ort,
            mock.patch("mobius._model_package.ModelPackage.save"),
        ):
            main(
                [
                    "build",
                    "--model",
                    "Qwen/Qwen2.5-0.5B",
                    tmpdir,
                    "--no-weights",
                    "--runtime",
                    "onnx-genai",
                ]
            )

        mock_export.assert_called_once()
        mock_ort.assert_not_called()
        assert "mtp_config: mtp_config.json" in capsys.readouterr().out

    def test_runtime_onnx_genai_uses_native_vlm_emitter(self):
        pkg = mock.MagicMock()
        pkg.items.return_value = []
        pkg.__iter__.return_value = iter(())
        pkg.config = object()
        args = SimpleNamespace(
            max_shard_size=None,
            external_data="onnx",
            execution_provider="cpu",
            no_weights=True,
            runtime="onnx-genai",
            config="/models/vlm",
            model=None,
        )
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            mock.patch(
                "mobius.integrations.onnx_genai.inference_metadata.is_native_vlm_package",
                return_value=True,
            ),
            mock.patch(
                "mobius.integrations.onnx_genai.inference_metadata."
                "write_native_vlm_package_metadata",
                return_value={},
            ) as native_writer,
            mock.patch(
                "mobius.integrations.onnx_genai.write_onnx_genai_config"
            ) as generic_writer,
        ):
            _save_package(pkg, tmpdir, args, None, None)

        native_writer.assert_called_once_with(
            pkg,
            tmpdir,
            config=pkg.config,
            source="/models/vlm",
        )
        generic_writer.assert_not_called()

    def test_runtime_onnx_genai_does_not_fallback_for_unsupported_vlm(self):
        pkg = mock.MagicMock()
        pkg.items.return_value = []
        pkg.__iter__.return_value = iter(("vision_encoder", "embedding", "decoder"))
        pkg.config = object()
        args = SimpleNamespace(
            max_shard_size=None,
            external_data="onnx",
            execution_provider="cpu",
            no_weights=True,
            runtime="onnx-genai",
            config="/models/unsupported-vlm",
            model=None,
        )
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            mock.patch(
                "mobius.integrations.onnx_genai.inference_metadata.is_native_vlm_package",
                return_value=True,
            ),
            mock.patch(
                "mobius.integrations.onnx_genai.inference_metadata."
                "write_native_vlm_package_metadata",
                side_effect=ValueError(
                    "unsupported VLM signature; regenerate processor assets or register it"
                ),
            ) as native_writer,
            mock.patch(
                "mobius.integrations.onnx_genai.write_onnx_genai_config"
            ) as generic_writer,
            pytest.raises(SystemExit, match=r"regenerate.*register"),
        ):
            _save_package(pkg, tmpdir, args, None, None)

        native_writer.assert_called_once()
        generic_writer.assert_not_called()


    def test_no_runtime_does_not_call_write_ort_genai_config(self):
        """Omitting --runtime does NOT call write_ort_genai_config()."""
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            mock.patch("mobius.integrations.ort_genai.write_ort_genai_config") as mock_export,
        ):
            main(["build", "--model", "Qwen/Qwen2.5-0.5B", tmpdir, "--no-weights"])

        mock_export.assert_not_called()

    def test_build_dot_nemo_model_routes_to_nemo(self):
        """A ``.nemo`` --model argument is auto-detected and routed to NeMo."""
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            mock.patch(
                "mobius.integrations.nemo.build_from_nemo",
                return_value=mock.MagicMock(),
            ) as mock_build_nemo,
            mock.patch("mobius.__main__._save_package") as mock_save,
        ):
            main(["build", "--model", "/some/model.nemo", tmpdir])

        mock_build_nemo.assert_called_once()
        assert mock_build_nemo.call_args.args[0] == "/some/model.nemo"
        mock_save.assert_called_once()

    def test_invalid_runtime_value_errors(self):
        """An unrecognised --runtime value causes argparse to exit with an error."""
        with tempfile.TemporaryDirectory() as tmpdir, pytest.raises(SystemExit):
            main(
                [
                    "build",
                    "--model",
                    "Qwen/Qwen2.5-0.5B",
                    tmpdir,
                    "--no-weights",
                    "--runtime",
                    "tensorrt",  # not a supported value
                ]
            )
