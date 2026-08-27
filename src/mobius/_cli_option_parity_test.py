# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for the unified ``build`` / ``build-gguf`` argument surface.

``build`` and ``build-gguf`` describe the same operation from two sources, so an
option that exists on both should be spelled and behave the same way. These tests
pin the parts of that which were previously inconsistent.
"""

from __future__ import annotations

from unittest import mock

import onnx_ir as ir
import pytest

from mobius.__main__ import build_parser
from mobius._model_package import ModelPackage
from mobius._testing import make_config

_SOURCE_ARGS = {
    "build": ["--model", "some/model"],
    "build-gguf": ["some.gguf"],
}


class TestOutputDirForms:
    """Both build commands expose one canonical output option."""

    @pytest.mark.parametrize("command", ["build", "build-gguf"])
    @pytest.mark.parametrize("flag", ["--output", "-o"])
    def test_output_option_is_accepted(self, command, flag):
        args = build_parser().parse_args([command, *_SOURCE_ARGS[command], flag, "out_dir"])
        assert args.output_dir == "out_dir"

    @pytest.mark.parametrize("command", ["build", "build-gguf"])
    def test_legacy_positional_is_still_accepted(self, command):
        args = build_parser().parse_args([command, *_SOURCE_ARGS[command], "out_dir"])
        assert args.output_dir == "out_dir"

    @pytest.mark.parametrize("command", ["build", "build-gguf"])
    def test_missing_output_is_rejected(self, command):
        with pytest.raises(SystemExit):
            build_parser().parse_args([command, *_SOURCE_ARGS[command]])

    @pytest.mark.parametrize("command", ["build", "build-gguf"])
    @pytest.mark.parametrize("positional", ["out_dir", "different_out_dir"])
    def test_option_and_positional_together_are_rejected(self, command, positional):
        with pytest.raises(SystemExit):
            build_parser().parse_args(
                [
                    command,
                    *_SOURCE_ARGS[command],
                    positional,
                    "--output",
                    "out_dir",
                ]
            )

    @pytest.mark.parametrize("command", ["build", "build-gguf"])
    @pytest.mark.parametrize("second_flag", ["--output", "-o"])
    def test_repeated_output_option_is_rejected(self, command, second_flag):
        with pytest.raises(SystemExit):
            build_parser().parse_args(
                [
                    command,
                    *_SOURCE_ARGS[command],
                    "--output",
                    "out_dir",
                    second_flag,
                    "other_dir",
                ]
            )

    @pytest.mark.parametrize(
        ("command", "handler_name"),
        [("build", "_cmd_build"), ("build-gguf", "_cmd_build_gguf")],
    )
    def test_cli_dispatch_receives_normalized_output(self, monkeypatch, command, handler_name):
        from mobius import __main__ as cli

        received = []
        monkeypatch.setattr(
            cli,
            handler_name,
            lambda args: received.append(args.output_dir),
        )

        cli.main([command, *_SOURCE_ARGS[command], "--output", "out_dir"])

        assert received == ["out_dir"]

    @pytest.mark.parametrize("command", ["build", "build-gguf"])
    def test_help_advertises_only_output_option(self, command, capsys):
        with pytest.raises(SystemExit):
            build_parser().parse_args([command, "--help"])
        help_text = capsys.readouterr().out
        assert "--output OUTPUT_DIR" in help_text
        assert "legacy positional" not in help_text
        assert "[_legacy_output_dir]" not in help_text


class TestKeepQuantizedRemoved:
    def test_flag_is_gone(self):
        """It was documented as deprecated and never read.

        ``_cmd_build_gguf`` computed ``keep_quantized = not args.dequantize`` and
        never looked at ``args.keep_quantized``, so passing it did nothing beyond
        occupying the mutually exclusive group.
        """
        with pytest.raises(SystemExit):
            build_parser().parse_args(
                ["build-gguf", "some.gguf", "--output", "out", "--keep-quantized"]
            )

    def test_dequantize_still_works(self):
        """Removing the dead alias must not disturb the flag that does the work."""
        args = build_parser().parse_args(
            ["build-gguf", "some.gguf", "--output", "out", "--dequantize"]
        )
        assert args.dequantize is True

        default = build_parser().parse_args(["build-gguf", "some.gguf", "--output", "out"])
        assert default.dequantize is False

    def test_transformers_build_uses_the_same_explicit_dequantize_spelling(self):
        args = build_parser().parse_args(
            ["build", "--model", "some/model", "--output", "out", "--dequantize"]
        )
        assert args.dequantize is True


class TestLocalCompressedCheckpointRouting:
    @staticmethod
    def _run(monkeypatch, tmp_path, *extra_args):
        from mobius import __main__ as cli
        from mobius.integrations import compressed_tensors

        class DummyModule:
            def __init__(self, config):
                self.config = config

            @staticmethod
            def preprocess_weights(state):
                return state

        hf_config = type("HFConfig", (), {"model_type": "qwen2"})()
        compressed_config = object()
        config = make_config(model_type="qwen2")
        package = ModelPackage({})
        built_configs = []
        stream = mock.Mock()

        import transformers

        monkeypatch.setattr(
            transformers.AutoConfig,
            "from_pretrained",
            lambda *args, **kwargs: hf_config,
        )
        monkeypatch.setattr(cli, "_config_from_hf", lambda *args, **kwargs: config)
        monkeypatch.setattr(cli.registry, "get", lambda _model_type: DummyModule)

        def fake_build_from_module(_module, built_config, *args, **kwargs):
            built_configs.append(built_config)
            return package

        monkeypatch.setattr(cli, "build_from_module", fake_build_from_module)
        monkeypatch.setattr(cli, "_save_package", mock.Mock())
        monkeypatch.setattr(
            compressed_tensors.CompressedTensorsConfig,
            "from_hf_config",
            lambda _config: compressed_config,
        )
        monkeypatch.setattr(
            compressed_tensors,
            "stream_compressed_tensors_to_package",
            stream,
        )
        args = build_parser().parse_args(
            [
                "build",
                "--config",
                str(tmp_path),
                "--output",
                str(tmp_path / "output"),
                *extra_args,
            ]
        )
        cli._resolve_build_features(args)

        cli._cmd_build(args)

        return built_configs, stream, compressed_config

    def test_default_routes_local_checkpoint_to_native_streaming_loader(
        self, monkeypatch, tmp_path
    ):
        built_configs, stream, compressed_config = self._run(
            monkeypatch,
            tmp_path,
        )

        assert built_configs[0].dtype == ir.DataType.FLOAT16
        stream.assert_called_once()
        assert stream.call_args.args[2] is compressed_config
        assert stream.call_args.kwargs["keep_quantized"] is True

    def test_dequantize_routes_local_checkpoint_to_dense_streaming_fallback(
        self, monkeypatch, tmp_path
    ):
        built_configs, stream, _compressed_config = self._run(
            monkeypatch,
            tmp_path,
            "--dequantize",
        )

        assert built_configs[0].dtype == ir.DataType.FLOAT
        stream.assert_called_once()
        assert stream.call_args.kwargs["keep_quantized"] is False

    def test_no_weights_does_not_open_local_compressed_shards(self, monkeypatch, tmp_path):
        built_configs, stream, _compressed_config = self._run(
            monkeypatch,
            tmp_path,
            "--no-weights",
        )

        assert built_configs[0].dtype == ir.DataType.FLOAT16
        stream.assert_not_called()


class TestSharedSaveOptions:
    """Options that control how the package is written exist on both commands."""

    @pytest.mark.parametrize("command", ["build", "build-gguf"])
    @pytest.mark.parametrize(
        "flag,value,dest",
        [
            ("--max-shard-size", "5GB", "max_shard_size"),
            ("--max-workers", "2", "max_workers"),
            ("--external-data", "safetensors", "external_data"),
            ("--dtype", "f16", "dtype"),
        ],
    )
    def test_option_is_accepted_by_both(self, command, flag, value, dest):
        args = build_parser().parse_args(
            [
                command,
                *_SOURCE_ARGS[command],
                "--output",
                "out_dir",
                flag,
                value,
            ]
        )
        parsed = getattr(args, dest)
        assert str(parsed) == value or parsed == int(value)

    @pytest.mark.parametrize(
        "flag", ["--max-shard-size", "--max-workers", "--external-data", "--dtype", "--ep"]
    )
    def test_help_text_matches_across_commands(self, flag):
        """Same flag, same wording — these had already drifted once."""
        parser = build_parser()
        subparsers = next(
            action
            for action in parser._actions
            if isinstance(action, __import__("argparse")._SubParsersAction)
        )
        helps = {}
        for command in ("build", "build-gguf"):
            sub = subparsers.choices[command]
            action = next(a for a in sub._actions if flag in a.option_strings)
            helps[command] = action.help
        assert helps["build"] == helps["build-gguf"], (
            f"{flag} is documented differently on build vs build-gguf: {helps}"
        )

    def test_max_shard_size_reaches_the_gguf_save(self):
        """Parsing it is not enough — it has to be threaded into ``pkg.save``.

        An option that parses but is never forwarded is worse than a missing one:
        the user gets no error and no sharding.
        """
        import inspect

        from mobius import __main__ as cli

        source = inspect.getsource(cli._cmd_build_gguf)
        assert "max_shard_size_bytes" in source, (
            "build-gguf accepts --max-shard-size but never passes it to save()"
        )
