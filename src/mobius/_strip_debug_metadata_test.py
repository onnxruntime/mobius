# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for :class:`~mobius._optimizations.StripDebugMetadataPass`."""

from __future__ import annotations

from typing import ClassVar

import onnx_ir as ir
import pytest

from mobius import build_from_module
from mobius._configs import ArchitectureConfig
from mobius._optimizations import (
    DEBUG_METADATA_KEYS,
    DEBUG_METADATA_PREFIXES,
    FUNCTIONAL_METADATA_PREFIX,
    StripDebugMetadataPass,
    strip_debug_metadata,
)
from mobius._registry import registry
from mobius.tasks import get_task


def _tiny_llama():
    """A two-layer Llama, built inline rather than from ``tests/_test_configs``.

    Co-located unit tests should not reach back into the top-level test package;
    this also keeps the fixture stable if the shared tiny configs are retuned.
    """
    config = ArchitectureConfig(
        model_type="llama",
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        vocab_size=256,
        max_position_embeddings=128,
        hidden_act="silu",
        rms_norm_eps=1e-6,
    )
    module = registry.get("llama")(config)
    return build_from_module(module, config, task=get_task("text-generation"))["model"]


def _all_metadata(model: ir.Model) -> dict[str, int]:
    """Every metadata key present anywhere in the model, with its occurrence count."""
    counts: dict[str, int] = {}

    def add(props) -> None:
        for key in props:
            counts[key] = counts.get(key, 0) + 1

    add(model.metadata_props)
    add(model.graph.metadata_props)
    seen: set[int] = set()
    for node in model.graph.all_nodes():
        add(node.metadata_props)
        for value in (*node.inputs, *node.outputs):
            if value is not None and id(value) not in seen:
                seen.add(id(value))
                add(value.metadata_props)
    return counts


class TestStripDebugMetadata:
    def test_removes_every_known_debug_key(self):
        model = _tiny_llama()
        before = _all_metadata(model)
        debug_before = {
            key
            for key in before
            if key.startswith(DEBUG_METADATA_PREFIXES) or key in DEBUG_METADATA_KEYS
        }
        assert debug_before, "the fixture carries no debug metadata; the test proves nothing"

        strip_debug_metadata(model)

        after = _all_metadata(model)
        assert not (debug_before & set(after)), (
            f"debug metadata survived the pass: {sorted(debug_before & set(after))}"
        )

    def test_preserves_functional_metadata(self):
        """``mobius.``-prefixed metadata is read back by runtimes and must survive.

        Injected rather than relying on a task that happens to emit some, so the
        guarantee is tested directly and does not quietly stop being exercised if
        that task changes.
        """
        model = _tiny_llama()
        model.graph.metadata_props[f"{FUNCTIONAL_METADATA_PREFIX}pipeline.when_present"] = "x"
        node = next(iter(model.graph))
        node.metadata_props[f"{FUNCTIONAL_METADATA_PREFIX}generation.policy_effects"] = "y"

        strip_debug_metadata(model)

        assert (
            model.graph.metadata_props[f"{FUNCTIONAL_METADATA_PREFIX}pipeline.when_present"]
            == "x"
        )
        assert (
            node.metadata_props[f"{FUNCTIONAL_METADATA_PREFIX}generation.policy_effects"]
            == "y"
        )

    def test_preserves_unrecognised_metadata(self):
        """An unknown key costs bytes; it must not be silently dropped.

        The pass is an allowlist for exactly this reason: if a toolchain upgrade
        starts writing a new provenance key, the failure mode should be a graph
        that is bigger than necessary, not a runtime missing something it read.
        """
        model = _tiny_llama()
        node = next(iter(model.graph))
        node.metadata_props["some.future.toolchain.key"] = "keep me"

        strip_debug_metadata(model)

        assert node.metadata_props["some.future.toolchain.key"] == "keep me"

    def test_reports_modified_only_when_it_removed_something(self):
        model = _tiny_llama()
        assert StripDebugMetadataPass()(model).modified is True
        # Second run has nothing left to do.
        assert StripDebugMetadataPass()(model).modified is False

    def test_shrinks_the_serialized_graph(self):
        """The size win is the whole reason the flag exists, so it is asserted.

        A loose bound (>10%) rather than the ~36% measured today: this should fail
        if stripping stops working, not every time the graph builder changes.
        """
        model = _tiny_llama()
        before = len(ir.to_proto(model).SerializeToString())
        strip_debug_metadata(model)
        after = len(ir.to_proto(model).SerializeToString())
        assert after < before * 0.9, f"expected a real size drop, got {before} -> {after}"

    def test_graph_is_otherwise_unchanged(self):
        """Metadata only — no node, initializer or port may move."""
        model = _tiny_llama()

        def shape_of(m):
            return (
                [(n.op_type, n.name) for n in m.graph.all_nodes()],
                [v.name for v in m.graph.inputs],
                [v.name for v in m.graph.outputs],
                sorted(m.graph.initializers),
            )

        before = shape_of(model)
        strip_debug_metadata(model)
        assert shape_of(model) == before

    def test_every_key_in_a_built_graph_is_classified(self):
        """No metadata key may be unclassified — it is either debug or functional.

        This is the guard that ages well. A toolchain upgrade that introduces a
        new provenance key would otherwise be shipped in every model silently;
        here it fails, and whoever sees the failure decides which bucket it
        belongs in rather than discovering it from a model-size regression.
        """
        model = _tiny_llama()
        unclassified = sorted(
            key
            for key in _all_metadata(model)
            if not (
                key.startswith((*DEBUG_METADATA_PREFIXES, FUNCTIONAL_METADATA_PREFIX))
                or key in DEBUG_METADATA_KEYS
            )
        )
        assert not unclassified, (
            f"unclassified metadata in a built graph: {unclassified}. Add each to "
            "DEBUG_METADATA_PREFIXES/DEBUG_METADATA_KEYS if it is build-time "
            "provenance, or give it the 'mobius.' prefix if a runtime reads it."
        )


class TestReleaseFlag:
    #: Smallest argument set each build command accepts, so the test exercises the
    #: real parser rather than a stand-in.
    _MINIMAL_ARGS: ClassVar[dict[str, list[str]]] = {
        "build": ["--model", "some/model", "--output", "out_dir"],
        "build-gguf": ["some.gguf", "--output", "out_dir"],
    }

    def test_build_and_gguf_describe_release_identically(self):
        """One flag, one meaning. Both subcommands take it from the same helper."""
        import argparse

        from mobius.__main__ import _add_release_argument

        helps = []
        for _ in range(2):
            parser = argparse.ArgumentParser()
            _add_release_argument(parser)
            action = next(a for a in parser._actions if a.dest == "release")
            helps.append(action.help)
            assert action.default is False
        assert helps[0] == helps[1]

    def test_help_renders_a_single_percent_sign(self):
        """``%%`` in the help string is correct: argparse %-formats help text.

        Asserted because it looks like a typo and has already been reported as
        one; a plain ``%`` is what would actually be wrong, raising at ``--help``.
        """
        import contextlib
        import io

        from mobius.__main__ import main

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer), contextlib.suppress(SystemExit):
            main(["build", "--help"])
        text = buffer.getvalue()
        assert "35-40%" in text
        assert "35-40%%" not in text

    @pytest.mark.parametrize("command", ["build", "build-gguf"])
    def test_release_defaults_to_off(self, command):
        """Debug metadata stays unless asked for — the flag opts *out* of it.

        Asserted on the parsed value, not on ``--help`` exiting. The first version
        of this test only checked that ``--help`` raised ``SystemExit``, which is
        true whatever the default is, so it asserted nothing about the flag.
        """
        from mobius.__main__ import build_parser

        parser = build_parser()
        without = parser.parse_args([command, *self._MINIMAL_ARGS[command]])
        assert without.release is False

        with_flag = parser.parse_args([command, *self._MINIMAL_ARGS[command], "--release"])
        assert with_flag.release is True
