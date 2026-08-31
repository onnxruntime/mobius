# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Unit tests for the CLI (``__main__.py``).

These tests invoke ``main()`` directly with argv lists, so they do not
require network access. All build tests use ``--no-weights``.
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import onnx
import onnx_ir as ir
import pytest

from mobius.__main__ import _save_package, build_parser, main


def _write_gated_gguf(path: Path, *, architecture: str, quantized: bool) -> None:
    """Write just enough of a GGUF to exercise the header architecture gate."""
    from gguf import GGMLQuantizationType, GGUFWriter

    writer = GGUFWriter(str(path), architecture)
    if quantized:
        writer.add_tensor(
            "token_embd.weight",
            np.zeros((1, 18), dtype=np.uint8),
            raw_dtype=GGMLQuantizationType.Q4_0,
        )
    else:
        writer.add_tensor("token_embd.weight", np.ones((1, 1), dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=False)
    writer.close()


class TestCLIList:
    """Test the ``list`` subcommand."""

    @pytest.mark.parametrize("command", ["reuse", "mimi", "moshi", "personaplex"])
    def test_native_audio_models_are_not_subcommands(self, command):
        parser = build_parser()
        subparsers = next(
            action
            for action in parser._actions
            if isinstance(action, argparse._SubParsersAction)
        )

        assert command not in subparsers.choices

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

    def test_standard_build_dispatches_reuse_through_public_build(self):
        from mobius.models.reuse import REUSE_REVISION

        with (
            tempfile.TemporaryDirectory() as tmpdir,
            mock.patch(
                "mobius.integrations.diffusers._builder._load_diffusers_pipeline_index",
                return_value=None,
            ) as pipeline_probe,
            mock.patch("mobius.__main__.build", return_value=mock.MagicMock()) as build_model,
            mock.patch("mobius.__main__._save_package"),
        ):
            main(["build", "--model", "nvidia/RE-USE", tmpdir, "--no-weights"])

        assert pipeline_probe.call_args.kwargs["revision"] == REUSE_REVISION
        assert build_model.call_args.kwargs["revision"] == REUSE_REVISION

    def test_standard_build_dispatches_personaplex_through_public_build(self):
        from mobius.integrations._moshi import _PERSONAPLEX_REVISION

        with (
            tempfile.TemporaryDirectory() as tmpdir,
            mock.patch(
                "mobius.integrations.diffusers._builder._load_diffusers_pipeline_index"
            ) as pipeline_probe,
            mock.patch("mobius.__main__.build", return_value=mock.MagicMock()) as build_model,
            mock.patch("mobius.__main__._save_package") as save_package,
        ):
            main(
                [
                    "build",
                    "--model",
                    "nvidia/personaplex-7b-v1",
                    tmpdir,
                    "--no-weights",
                    "--dtype",
                    "f32",
                    "--execution-provider",
                    "cuda",
                ]
            )

        pipeline_probe.assert_not_called()
        assert build_model.call_args.kwargs["revision"] == _PERSONAPLEX_REVISION
        assert build_model.call_args.kwargs["load_weights"] is False
        assert build_model.call_args.kwargs["execution_provider"] == "cuda"
        assert build_model.call_args.kwargs["dtype"] == ir.DataType.FLOAT
        save_package.assert_called_once()

    def test_local_personaplex_config_bypasses_transformers(self):
        with (
            tempfile.TemporaryDirectory() as checkpoint,
            tempfile.TemporaryDirectory() as output,
            mock.patch(
                "mobius.integrations._moshi._is_personaplex_checkpoint",
                return_value=True,
            ),
            mock.patch(
                "mobius.integrations.diffusers._builder._load_diffusers_pipeline_index"
            ) as diffusers_probe,
            mock.patch("mobius.__main__.build", return_value=mock.MagicMock()) as build_model,
            mock.patch("transformers.AutoConfig.from_pretrained") as transformers_probe,
            mock.patch("mobius.__main__._save_package") as save_package,
        ):
            main(["build", "--config", checkpoint, output, "--no-weights"])

        diffusers_probe.assert_not_called()
        transformers_probe.assert_not_called()
        assert build_model.call_args.args == (checkpoint,)
        assert build_model.call_args.kwargs["revision"] is None
        assert build_model.call_args.kwargs["load_weights"] is False
        save_package.assert_called_once()

    @pytest.mark.parametrize("option", ["--input-sample-rate", "--bwe-sample-rate"])
    def test_reuse_rate_options_are_rejected_for_diffusers(self, option):
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            mock.patch(
                "mobius.integrations.diffusers._builder._load_diffusers_pipeline_index",
                return_value={"_class_name": "ExamplePipeline"},
            ),
            pytest.raises(SystemExit, match="only supported for RE-USE"),
        ):
            main(["build", "--model", "example/diffusers", tmpdir, option, "16000"])

    @pytest.mark.parametrize(
        ("extra_args", "expected"),
        [([], True), (["--dequantize"], False)],
    )
    def test_transformers_quantization_choice_reaches_build(
        self,
        extra_args,
        expected,
    ):
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            mock.patch(
                "mobius.integrations.diffusers._builder._load_diffusers_pipeline_index",
                return_value=None,
            ),
            mock.patch(
                "mobius.__main__.build",
                return_value=mock.MagicMock(),
            ) as mock_build,
            mock.patch("mobius.__main__._save_package"),
        ):
            main(
                [
                    "build",
                    "--model",
                    "unsloth/Qwen3.8-Flash-Next-FP8",
                    tmpdir,
                    "--no-weights",
                    *extra_args,
                ]
            )

        assert mock_build.call_args.kwargs["keep_quantized"] is expected

    def test_revision_propagates_to_diffusers_detection_and_build(self):
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            mock.patch(
                "mobius.integrations.diffusers._builder._load_diffusers_pipeline_index",
                return_value={"_class_name": "TestPipeline"},
            ) as mock_detect,
            mock.patch(
                "mobius.integrations.diffusers._builder.build_diffusers_pipeline",
                return_value=mock.MagicMock(),
            ) as mock_build,
            mock.patch("mobius.__main__._save_package"),
        ):
            main(
                [
                    "build",
                    "--model",
                    "test/diffusion-model",
                    "--revision",
                    "pinned-revision",
                    tmpdir,
                ]
            )

        mock_detect.assert_called_once_with(
            "test/diffusion-model",
            revision="pinned-revision",
        )
        assert mock_build.call_args.kwargs["revision"] == "pinned-revision"

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

    def test_local_qwen4_composite_requires_remote_text_only_route(self, tmp_path):
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        (config_dir / "config.json").write_text(
            """{
  "model_type": "qwen4_exp",
  "architectures": ["Qwen4ExpForConditionalGeneration"],
  "text_config": {"model_type": "qwen4_exp_text"}
}"""
        )
        with pytest.raises(SystemExit, match="cannot be silently exported as text-only"):
            main(
                [
                    "build",
                    "--config",
                    str(config_dir),
                    str(tmp_path / "output"),
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

    def test_static_cache_with_onnx_genai_runtime_emits_scatter_abi(self):
        """A static-cache export is describable, so the CLI must describe it.

        The two control ports are rank-1 integer vectors and are therefore
        shape-indistinguishable from one another, which is exactly why the ABI
        is *declared* rather than inferred. It is declared once, in the
        workflow: the state group that scatters into the buffers names the port
        carrying the write cursor and the port carrying the non-pad length, and
        the component those ports belong to declares both.
        """
        import onnx_ir as ir
        import yaml

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
                    "--runtime",
                    "onnx-genai",
                ]
            )
            with open(
                os.path.join(tmpdir, "inference_metadata.yaml"), encoding="utf-8"
            ) as handle:
                metadata = yaml.safe_load(handle)
            # The exported graph is the authority on which ports exist and what
            # rank they have, so read it here rather than trusting a copy of it
            # in the metadata — a copy is what this contract exists to avoid.
            artifact = metadata["pipeline"]["workflow"]["components"]["model"][
                "implementation"
            ]["artifact"]
            exported = {
                str(value.name): len(value.shape)
                for value in ir.load(os.path.join(tmpdir, artifact)).graph.inputs
            }

        # One canonical description: no second copy of the port ABI outside it.
        assert "io" not in metadata.get("model", {})

        workflow = metadata["pipeline"]["workflow"]
        assert workflow["inputs"]["package.cache_capacity"]["default"] == 128
        groups = workflow["serving"]["state_service"]["groups"]
        update = next(group["update"] for group in groups.values() if "update" in group)
        assert update["kind"] == "indexed_scatter"
        assert update["capacity"] == "package.cache_capacity"
        # The write cursor and the logical length are the same quantity.
        assert update["write_indices"] == "cache_lengths"
        assert update["write_indices_ports"] == {"model": "write_indices"}
        assert update["kv_length_ports"] == {"model": "nonpad_kv_seqlen"}

        # The component transcribes none of this: it declares the roles a graph
        # cannot state, and the scatter's control ports are named by the state
        # group. Both names have to resolve in the artifact.
        assert not (workflow["components"]["model"]["ports"].get("inputs"))
        assert workflow["components"]["model"]["ports"]["roles"]["input_ids"] == "token_ids"
        assert exported["write_indices"] == 1
        assert exported["nonpad_kv_seqlen"] == 1

        group = next(group for group in groups.values() if "update" in group)
        pairs = group["ports"]["model"]
        assert {alias["output"] for alias in pairs.values()} == {
            f"updated_{alias['input']}" for alias in pairs.values()
        }
        assert pairs["cache_0"] == {
            "input": "key_cache.0",
            "output": "updated_key_cache.0",
            "role": "key",
            "layer": 0,
        }

    def test_static_cache_task_follows_text_only_substitution(self):
        """``text-only`` + ``static-cache`` must resolve the *text* task.

        ``build()`` swaps a multimodal ``model_type`` for its text-only
        registry sibling, so the deferred static-cache task has to be resolved
        against the substituted type. Resolving against the raw checkpoint type
        pairs the text-only module with the multimodal task, which then fails
        looking for sub-modules a text-only module does not have.
        """
        from mobius.tasks._gemma4 import Gemma4TextCausalLMTask

        hf_config = mock.MagicMock()
        hf_config.model_type = "gemma4"
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            mock.patch("transformers.AutoConfig.from_pretrained", return_value=hf_config),
            mock.patch("mobius.__main__.build", return_value=mock.MagicMock()) as mock_build,
            mock.patch("mobius.__main__._save_package"),
        ):
            main(
                [
                    "build",
                    "--model",
                    "google/gemma-4-E2B-it",
                    tmpdir,
                    "--no-weights",
                    "--features",
                    "text-only,static-cache",
                    "--max-seq-len",
                    "128",
                ]
            )

        task = mock_build.call_args.kwargs["task"]
        assert isinstance(task, Gemma4TextCausalLMTask)

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

    def test_features_paged_attention_passed_through(self):
        """--features paged-attention threads the flag + paged task into build."""
        from mobius.tasks import CausalLMTask

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
                    "paged-attention",
                ]
            )
        kwargs = mock_build.call_args.kwargs
        assert kwargs.get("export_paged_attention") is True
        task = kwargs.get("task")
        assert isinstance(task, CausalLMTask)
        assert task._paged_cache is True

    def test_paged_attention_with_task_errors(self):
        """--features paged-attention owns the task; it cannot combine with --task."""
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            pytest.raises(SystemExit, match=r"paged-attention.*--task"),
        ):
            main(
                [
                    "build",
                    "--model",
                    "some/model",
                    tmpdir,
                    "--no-weights",
                    "--features",
                    "paged-attention",
                    "--task",
                    "text-generation",
                ]
            )

    def test_paged_attention_with_static_cache_errors(self):
        """PagedAttention and static cache are distinct, exclusive cache modes."""
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            pytest.raises(SystemExit, match=r"paged-attention.*static-cache"),
        ):
            main(
                [
                    "build",
                    "--model",
                    "some/model",
                    tmpdir,
                    "--no-weights",
                    "--features",
                    "paged-attention,static-cache",
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


class TestCLIConvertComfyUI:
    def test_revision_is_forwarded_to_conversion(self, tmp_path):
        workflow_path = tmp_path / "workflow.json"
        workflow_path.write_text("{}", encoding="utf-8")
        result = SimpleNamespace(
            output_dir=str(tmp_path / "output"),
            metadata_path=str(tmp_path / "output" / "inference_metadata.yaml"),
            run_params_path=str(tmp_path / "output" / "run.json"),
            workflow=SimpleNamespace(
                steps=20,
                cfg=7.5,
                sampler_name="euler",
                scheduler_kind="euler",
                width=512,
                height=512,
                loras=[],
                prompt="a cat",
                negative_prompt="",
                seed=42,
            ),
        )
        with mock.patch(
            "mobius.integrations.onnx_genai.convert_comfyui_workflow",
            return_value=result,
        ) as convert:
            main(
                [
                    "convert-comfyui",
                    str(workflow_path),
                    "--checkpoint",
                    "nota-ai/bk-sdm-small",
                    "--revision",
                    "pinned-revision",
                    "--output",
                    str(tmp_path / "output"),
                ]
            )

        convert.assert_called_once_with(
            {},
            "nota-ai/bk-sdm-small",
            str(tmp_path / "output"),
            sdxl=False,
            revision="pinned-revision",
        )


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

    def test_qwen4_ort_genai_runtime_gap_does_not_block_package_save(self, tmp_path):
        pkg = mock.MagicMock()
        pkg.config = SimpleNamespace(model_type="qwen4_exp_text")
        args = SimpleNamespace(
            runtime="ort-genai",
            external_data="onnx",
            execution_provider="cpu",
            max_shard_size=None,
            max_workers=1,
            no_weights=False,
            release=False,
        )

        with mock.patch(
            "mobius.integrations.ort_genai.write_ort_genai_config"
        ) as config_writer:
            _save_package(pkg, str(tmp_path), args, None, None)

        pkg.save.assert_called_once()
        config_writer.assert_called_once()

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

    def test_runtime_ort_genai_mage_vl_gap_does_not_block_saving(self):
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            mock.patch("mobius._model_package.ModelPackage.save") as save,
            mock.patch(
                "mobius.integrations.ort_genai.write_ort_genai_config"
            ) as config_writer,
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

        save.assert_called()
        config_writer.assert_called_once()

    def test_runtime_onnx_genai_routes_vlm_through_workflow_emitter(self):
        """A VLM package emits the workflow IR, not a legacy composite pipeline."""
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
            config="/models/vlm",
            model=None,
            revision="pinned-revision",
        )
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            mock.patch(
                "mobius.integrations.onnx_genai.write_onnx_genai_config",
                return_value={},
            ) as writer,
        ):
            _save_package(pkg, tmpdir, args, None, None)

        writer.assert_called_once_with(
            pkg,
            tmpdir,
            config=pkg.config,
            source="/models/vlm",
            revision="pinned-revision",
            guidance_scale=None,
        )

    def test_runtime_onnx_genai_forwards_guidance_scale(self):
        pkg = mock.MagicMock()
        pkg.items.return_value = []
        pkg.__iter__.return_value = iter(("transformer", "text_encoder", "vae_decoder"))
        pkg.config = object()
        args = SimpleNamespace(
            max_shard_size=None,
            max_workers=8,
            external_data="onnx",
            execution_provider="cpu",
            no_weights=True,
            runtime="onnx-genai",
            config="/models/video",
            model=None,
            guidance_scale=6.0,
            revision="pinned-revision",
        )
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            mock.patch(
                "mobius.integrations.onnx_genai.write_onnx_genai_config",
                return_value={},
            ) as writer,
        ):
            _save_package(pkg, tmpdir, args, None, None)

        assert writer.call_args.kwargs["guidance_scale"] == pytest.approx(6.0)
        assert writer.call_args.kwargs["revision"] == "pinned-revision"

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
                "mobius.integrations.onnx_genai.write_onnx_genai_config",
                side_effect=ValueError(
                    "unsupported VLM signature; regenerate processor assets or register it"
                ),
            ) as writer,
            pytest.raises(SystemExit, match=r"regenerate.*register"),
        ):
            _save_package(pkg, tmpdir, args, None, None)

        writer.assert_called_once()

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
            main(
                [
                    "build",
                    "--model",
                    "/some/model.nemo",
                    "--revision",
                    "pinned-revision",
                    tmpdir,
                ]
            )

        mock_build_nemo.assert_called_once()
        assert mock_build_nemo.call_args.args[0] == "/some/model.nemo"
        assert mock_build_nemo.call_args.kwargs["revision"] == "pinned-revision"
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


class TestCLIBuildGGUF:
    """The CLI must preserve the public GGUF architecture gate for every mode."""

    def test_image_token_id_accepts_negative_processor_sentinel(self) -> None:
        args = build_parser().parse_args(
            [
                "build-gguf",
                "text.gguf",
                "--output",
                "output",
                "--mmproj",
                "mmproj.gguf",
                "--image-token-id",
                "-200",
            ]
        )

        assert args.image_token_id == -200

    def test_runtime_allows_missing_downstream_tokenizer_processor(
        self, tmp_path: Path
    ) -> None:
        gguf_path = tmp_path / "llama.gguf"
        output_dir = tmp_path / "must-not-exist"
        _write_gated_gguf(gguf_path, architecture="llama", quantized=False)

        package = mock.MagicMock()
        package.__iter__.return_value = iter(("model",))
        with (
            mock.patch(
                "mobius.integrations.gguf._builder._resolve_gguf_path",
                return_value=gguf_path,
            ),
            mock.patch(
                "mobius.integrations.gguf._shard_set.open_gguf_model",
                return_value=mock.MagicMock(),
            ),
            mock.patch("mobius.integrations.gguf._builder._validate_gguf_model"),
            mock.patch(
                "mobius.integrations.gguf.build_from_gguf",
                return_value=package,
            ),
            mock.patch(
                "mobius.integrations.gguf.write_gguf_runtime_package",
                return_value={},
            ) as writer,
            mock.patch("mobius.__main__._print_saved_gguf_models"),
        ):
            main(
                [
                    "build-gguf",
                    str(gguf_path),
                    "--output",
                    str(output_dir),
                    "--runtime",
                    "onnx-genai",
                    "--dequantize",
                ]
            )

        assert writer.call_args.kwargs["tokenizer_repository"] is None
        assert writer.call_args.kwargs["tokenizer_revision"] is None

    def test_runtime_rejects_mutable_tokenizer_revision_before_build(
        self, tmp_path: Path
    ) -> None:
        gguf_path = tmp_path / "llama.gguf"
        output_dir = tmp_path / "must-not-exist"
        _write_gated_gguf(gguf_path, architecture="llama", quantized=False)

        with pytest.raises(SystemExit, match="immutable 40-hex"):
            main(
                [
                    "build-gguf",
                    str(gguf_path),
                    "--output",
                    str(output_dir),
                    "--runtime",
                    "onnx-genai",
                    "--runtime-version",
                    "1.29.0",
                    "--tokenizer-repository",
                    "owner/tokenizer",
                    "--tokenizer-revision",
                    "main",
                    "--dequantize",
                ]
            )

        assert not output_dir.exists()

    @pytest.mark.parametrize(
        ("quantized", "options"),
        [
            pytest.param(False, ["--dequantize"], id="float"),
            pytest.param(True, [], id="quantized"),
            pytest.param(True, ["--dtype", "f16"], id="dtype"),
            pytest.param(True, ["--static-cache"], id="static-cache"),
            pytest.param(True, ["--runtime", "onnx-genai"], id="onnx-genai-runtime"),
            pytest.param(True, ["--runtime", "ort-genai"], id="ort-genai-runtime"),
            pytest.param(True, ["--release"], id="release"),
        ],
    )
    def test_deferred_architecture_fails_before_output_creation(
        self, quantized: bool, options: list[str], tmp_path: Path
    ) -> None:
        from mobius.integrations.gguf._errors import UnsupportedGGUFArchitectureError

        gguf_path = tmp_path / "pockettts.gguf"
        output_dir = tmp_path / "must-not-exist"
        _write_gated_gguf(gguf_path, architecture="pockettts", quantized=quantized)

        with pytest.raises(
            UnsupportedGGUFArchitectureError,
            match=r"pockettts.*before config extraction",
        ):
            main(
                [
                    "build-gguf",
                    str(gguf_path),
                    "--output",
                    str(output_dir),
                    *options,
                ]
            )

        assert not output_dir.exists()

    def test_standalone_clip_rejects_before_output_creation(self, tmp_path: Path) -> None:
        from mobius.integrations.gguf._errors import DisabledGGUFArchitectureError

        gguf_path = tmp_path / "mmproj.gguf"
        output_dir = tmp_path / "must-not-exist"
        _write_gated_gguf(gguf_path, architecture="clip", quantized=True)

        with pytest.raises(
            DisabledGGUFArchitectureError,
            match=r"clip.*intentionally disabled",
        ):
            main(
                [
                    "build-gguf",
                    str(gguf_path),
                    "--output",
                    str(output_dir),
                ]
            )

        assert not output_dir.exists()
