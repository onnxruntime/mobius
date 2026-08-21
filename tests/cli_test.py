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

    def test_max_workers_defaults_to_eight(self):
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            mock.patch(
                "mobius.integrations.diffusers._builder._load_diffusers_pipeline_index",
                return_value=None,
            ),
            mock.patch("mobius.__main__.build", return_value=mock.MagicMock()),
            mock.patch("mobius.__main__._save_package") as save_package,
        ):
            main(["build", "--model", "Qwen/Qwen2.5-0.5B", tmpdir, "--no-weights"])

        assert save_package.call_args.args[2].max_workers == 8

    def test_max_workers_override_reaches_model_package_save(self):
        pkg = mock.MagicMock()
        pkg.items.return_value = []
        pkg.__iter__.return_value = iter(())
        args = SimpleNamespace(
            max_shard_size=None,
            max_workers=1,
            external_data="onnx",
            execution_provider="cpu",
            no_weights=True,
            runtime=None,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            _save_package(pkg, tmpdir, args, None, None)

        assert pkg.save.call_args.kwargs["max_workers"] == 1

    @pytest.mark.parametrize("max_workers", [0, -1])
    def test_non_positive_max_workers_errors(self, max_workers):
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            pytest.raises(SystemExit, match=r"--max-workers must be a positive integer"),
        ):
            main(
                [
                    "build",
                    "--model",
                    "Qwen/Qwen2.5-0.5B",
                    tmpdir,
                    "--no-weights",
                    "--max-workers",
                    str(max_workers),
                ]
            )

    def test_build_encoder_decoder_produces_separate_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            main(["build", "--model", "facebook/bart-base", tmpdir, "--no-weights"])
            assert os.path.isfile(os.path.join(tmpdir, "encoder", "model.onnx"))
            assert os.path.isfile(os.path.join(tmpdir, "decoder", "model.onnx"))

    def test_build_missing_model_errors(self):
        with tempfile.TemporaryDirectory() as tmpdir, pytest.raises(SystemExit):
            main(["build", tmpdir])  # no --model or --config

    def test_text_only_with_config_errors(self):
        """The text-only feature is rejected on the --config (local dir) path."""
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            pytest.raises(SystemExit, match=r"--features text-only.*--config"),
        ):
            main(
                [
                    "build",
                    "--config",
                    tmpdir,
                    tmpdir,
                    "--features",
                    "text-only",
                    "--no-weights",
                ]
            )

    def test_text_only_with_component_errors(self):
        """The text-only feature is rejected when combined with --component."""
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            pytest.raises(SystemExit, match=r"--features text-only.*--component"),
        ):
            main(
                [
                    "build",
                    "--model",
                    "google/gemma-4-12B",
                    tmpdir,
                    "--features",
                    "text-only",
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
                "mobius.integrations.diffusers._builder._load_diffusers_pipeline_index"
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
                    "--features",
                    "text-only",
                    "--no-weights",
                ]
            )

        mock_diffusers.assert_not_called()
        mock_build.assert_called_once()
        assert mock_build.call_args.kwargs.get("text_only") is True

    def test_revision_is_forwarded_to_detection_and_build(self):
        revision = "61ba4e0b3309b6656edea3e93e419f7bd5c61957"
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            mock.patch(
                "mobius.integrations.diffusers._builder._load_diffusers_pipeline_index",
                return_value=None,
            ) as mock_diffusers,
            mock.patch("mobius.__main__.build", return_value=mock.MagicMock()) as mock_build,
            mock.patch("mobius.__main__._save_package"),
        ):
            main(
                [
                    "build",
                    "--model",
                    "zai-org/GLM-ASR-Nano-2512",
                    tmpdir,
                    "--revision",
                    revision,
                    "--no-weights",
                ]
            )

        mock_diffusers.assert_called_once_with(
            "zai-org/GLM-ASR-Nano-2512",
            revision=revision,
        )
        assert mock_build.call_args.kwargs["revision"] == revision

    def test_build_static_cache(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            main(
                [
                    "build",
                    "--model",
                    "Qwen/Qwen2.5-0.5B",
                    tmpdir,
                    "--no-weights",
                    "--features",
                    "static-cache",
                ]
            )
            assert os.path.isfile(os.path.join(tmpdir, "model.onnx"))

    def test_features_static_cache_equivalent(self):
        """--features static-cache builds the same static-cache model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            main(
                [
                    "build",
                    "--model",
                    "Qwen/Qwen2.5-0.5B",
                    tmpdir,
                    "--no-weights",
                    "--features",
                    "static-cache",
                ]
            )
            model = onnx.load(os.path.join(tmpdir, "model.onnx"))
            cache_inputs = [
                inp for inp in model.graph.input if inp.name.startswith("key_cache.")
            ]
            assert len(cache_inputs) > 0, "static cache not applied via --features"

    def test_features_max_seq_len_pairs_with_static_cache(self):
        """--max-seq-len works when static-cache is enabled via --features."""
        with tempfile.TemporaryDirectory() as tmpdir:
            main(
                [
                    "build",
                    "--model",
                    "Qwen/Qwen2.5-0.5B",
                    tmpdir,
                    "--no-weights",
                    "--features",
                    "static-cache",
                    "--max-seq-len",
                    "128",
                ]
            )
            assert os.path.isfile(os.path.join(tmpdir, "model.onnx"))

    def test_features_text_only_passed_through(self):
        """--features text-only sets text_only on the build() call."""
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            mock.patch(
                "mobius.integrations.diffusers._builder._load_diffusers_pipeline_index",
                return_value=None,
            ),
            mock.patch("mobius.__main__.build", return_value=mock.MagicMock()) as mock_build,
            mock.patch("mobius.__main__._save_package"),
        ):
            main(
                [
                    "build",
                    "--model",
                    "some/model",
                    tmpdir,
                    "--no-weights",
                    "--features",
                    "text-only",
                ]
            )
        assert mock_build.call_args.kwargs.get("text_only") is True

    def test_features_fp8_kv_cache_passed_through(self):
        """--features fp8-kv-cache sets fp8_kv_cache on the build() call."""
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            mock.patch(
                "mobius.integrations.diffusers._builder._load_diffusers_pipeline_index",
                return_value=None,
            ),
            mock.patch("mobius.__main__.build", return_value=mock.MagicMock()) as mock_build,
            mock.patch("mobius.__main__._save_package"),
        ):
            main(
                [
                    "build",
                    "--model",
                    "some/model",
                    tmpdir,
                    "--no-weights",
                    "--features",
                    "fp8-kv-cache",
                ]
            )
        assert mock_build.call_args.kwargs.get("fp8_kv_cache") is True

    def test_features_prune_prefill_prefix_passed_through(self):
        """--features prune-prefill-prefix sets the build option."""
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            mock.patch(
                "mobius.integrations.diffusers._builder._load_diffusers_pipeline_index",
                return_value=None,
            ),
            mock.patch("mobius.__main__.build", return_value=mock.MagicMock()) as mock_build,
            mock.patch("mobius.__main__._save_package"),
        ):
            main(
                [
                    "build",
                    "--model",
                    "some/model",
                    tmpdir,
                    "--no-weights",
                    "--features",
                    "prune-prefill-prefix",
                ]
            )
        assert mock_build.call_args.kwargs.get("prune_prefill_prefix") is True

    def test_features_comma_separated_multiple(self):
        """A single --features accepts a comma-separated list."""
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            mock.patch(
                "mobius.integrations.diffusers._builder._load_diffusers_pipeline_index",
                return_value=None,
            ),
            mock.patch("mobius.__main__.build", return_value=mock.MagicMock()) as mock_build,
            mock.patch("mobius.__main__._save_package"),
        ):
            main(
                [
                    "build",
                    "--model",
                    "some/model",
                    tmpdir,
                    "--no-weights",
                    "--features",
                    "text-only,fp8-kv-cache,prune-prefill-prefix",
                ]
            )
        kwargs = mock_build.call_args.kwargs
        assert kwargs.get("text_only") is True
        assert kwargs.get("fp8_kv_cache") is True
        assert kwargs.get("prune_prefill_prefix") is True

    def test_features_unknown_errors(self):
        """An unrecognised feature name is rejected with a clear error."""
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            pytest.raises(SystemExit, match=r"unknown feature 'bogus'"),
        ):
            main(
                [
                    "build",
                    "--model",
                    "some/model",
                    tmpdir,
                    "--no-weights",
                    "--features",
                    "bogus",
                ]
            )

    def test_max_seq_len_without_static_cache_errors(self):
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            pytest.raises(SystemExit, match=r"--features static-cache"),
        ):
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
        """The static-cache feature cannot be combined with any --task."""
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            pytest.raises(SystemExit, match=r"--features static-cache.*--task"),
        ):
            main(
                [
                    "build",
                    "--model",
                    "Qwen/Qwen2.5-0.5B",
                    tmpdir,
                    "--no-weights",
                    "--features",
                    "static-cache",
                    "--task",
                    "text-generation",
                ]
            )

    def test_kv_cache_scale_file_without_fp8_feature_errors(self):
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            pytest.raises(SystemExit, match=r"--features fp8-kv-cache"),
        ):
            main(
                [
                    "build",
                    "--model",
                    "Qwen/Qwen2.5-0.5B",
                    tmpdir,
                    "--no-weights",
                    "--kv-cache-scale-file",
                    "scales.json",
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
                    "--features",
                    "static-cache",
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
                    "--features",
                    "static-cache",
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
        assert call_kwargs.kwargs.get("trust_remote_code") is False

    def test_runtime_ort_genai_propagates_trust_remote_code(self):
        """--trust-remote-code also applies to runtime config generation."""
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
                    "--trust-remote-code",
                    "--runtime",
                    "ort-genai",
                ]
            )

        assert mock_export.call_args.kwargs["trust_remote_code"] is True

    def test_build_propagates_revision(self):
        revision = "5a414ead75d45db003906d06fb62bd5b6846cec0"
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            mock.patch(
                "mobius.integrations.diffusers._builder._load_diffusers_pipeline_index",
                return_value=None,
            ) as detect_diffusers,
            mock.patch("mobius.__main__.build", return_value=mock.MagicMock()) as build_model,
            mock.patch("mobius.__main__._save_package"),
        ):
            main(
                [
                    "build",
                    "--model",
                    "LiquidAI/LFM2.5-VL-3B",
                    "--revision",
                    revision,
                    tmpdir,
                    "--no-weights",
                ]
            )

        detect_diffusers.assert_called_once_with("LiquidAI/LFM2.5-VL-3B", revision=revision)
        assert build_model.call_args.kwargs["revision"] == revision

    def test_runtime_ort_genai_rejects_mage_vl_before_saving(self):
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            mock.patch("mobius._model_package.ModelPackage.save") as save,
            mock.patch(
                "mobius.integrations.ort_genai.write_ort_genai_config"
            ) as config_writer,
            pytest.raises(
                SystemExit,
                match=r"Mage-VL.*patch_positions.*1D decoder position_ids",
            ),
        ):
            main(
                [
                    "build",
                    "--model",
                    "microsoft/Mage-VL",
                    tmpdir,
                    "--no-weights",
                    "--trust-remote-code",
                    "--runtime",
                    "ort-genai",
                ]
            )

        save.assert_not_called()
        config_writer.assert_not_called()

    def test_runtime_onnx_genai_uses_native_vlm_emitter(self):
        pkg = mock.MagicMock()
        pkg.items.return_value = []
        pkg.__iter__.return_value = iter(())
        pkg.config = object()
        args = SimpleNamespace(
            max_shard_size=None,
            max_workers=8,
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
            max_workers=8,
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
