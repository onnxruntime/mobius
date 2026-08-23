# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for the unified ``build`` / ``build-gguf`` argument surface.

``build`` and ``build-gguf`` describe the same operation from two sources, so an
option that exists on both should be spelled and behave the same way. These tests
pin the parts of that which were previously inconsistent.
"""

from __future__ import annotations

import pytest

from mobius.__main__ import build_parser

_MINIMAL = {
    "build": ["--model", "some/model", "out_dir"],
    "build-gguf": ["some.gguf", "out_dir"],
}


class TestOutputDirIsPositionalEverywhere:
    """Both commands take the output directory the same way.

    ``build-gguf`` used to take ``--output/-o`` with a default derived from the
    GGUF filename, so the same concept was positional-and-required on one command
    and optional-with-a-default on the other.
    """

    @pytest.mark.parametrize("command", ["build", "build-gguf"])
    def test_output_dir_is_a_required_positional(self, command):
        args = build_parser().parse_args([command, *_MINIMAL[command]])
        assert args.output_dir == "out_dir"

        with pytest.raises(SystemExit):
            # Dropping the trailing output_dir must be an error, not a default.
            build_parser().parse_args([command, *_MINIMAL[command][:-1]])

    def test_gguf_no_longer_accepts_the_output_flag(self):
        """The old spelling is gone rather than silently kept as an alias.

        Left in place it would be a second way to say the same thing, and the two
        would drift.
        """
        with pytest.raises(SystemExit):
            build_parser().parse_args(["build-gguf", "some.gguf", "-o", "out_dir"])
        with pytest.raises(SystemExit):
            build_parser().parse_args(["build-gguf", "some.gguf", "--output", "out_dir"])


class TestKeepQuantizedRemoved:
    def test_flag_is_gone(self):
        """It was documented as deprecated and never read.

        ``_cmd_build_gguf`` computed ``keep_quantized = not args.dequantize`` and
        never looked at ``args.keep_quantized``, so passing it did nothing beyond
        occupying the mutually exclusive group.
        """
        with pytest.raises(SystemExit):
            build_parser().parse_args(["build-gguf", "some.gguf", "out", "--keep-quantized"])

    def test_dequantize_still_works(self):
        """Removing the dead alias must not disturb the flag that does the work."""
        args = build_parser().parse_args(["build-gguf", "some.gguf", "out", "--dequantize"])
        assert args.dequantize is True

        default = build_parser().parse_args(["build-gguf", "some.gguf", "out"])
        assert default.dequantize is False


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
        args = build_parser().parse_args([command, *_MINIMAL[command], flag, value])
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
