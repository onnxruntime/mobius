# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Keep the bounded opset-29 survey distinct from architecture/runtime closure."""

from __future__ import annotations

import json
from pathlib import Path

import onnx
import pytest

from mobius import build_from_gguf
from mobius._flags import override_flags
from mobius.integrations.gguf._arch_registry import iter_arch_specs
from mobius.integrations.gguf._builder_test_utils import _write_quantized_gguf


@pytest.mark.opset29
def test_packed_construction_does_not_require_installed_schemas(tmp_path, monkeypatch) -> None:
    source = tmp_path / "weights.gguf"
    _write_quantized_gguf(source)
    monkeypatch.setattr(onnx.defs, "has", lambda *_args: False)
    with override_flags(experimental_opset29=True):
        package = build_from_gguf(source, execution_provider="onnx-standard")
    assert any(node.op_type == "DequantizePacked" for node in package["model"].graph)


def test_lossless_survey_covers_census_without_promoting_runtime() -> None:
    path = (
        Path(__file__).resolve().parents[4]
        / "testdata"
        / "evidence"
        / "gguf_opset29_survey.json"
    )
    evidence = json.loads(path.read_text(encoding="utf-8"))
    assert {row["architecture"] for row in evidence["census_architectures"]} == {
        spec.gguf_arch for spec in iter_arch_specs()
    }
    results = evidence["architecture_probe"]["results"]
    assert len({row["architecture"] for row in results}) == len(results)
    for row in results:
        assert row["numerical_parity"] == "not-tested-by-this-probe"
        assert row["full_checkpoint_runtime"] == "not-tested"
        assert (row["result"] == "passed") == any(
            attempt["result"] == "graph-and-checker-passed" for attempt in row["attempts"]
        )
    for row in evidence["format_tests"]:
        assert row["full_checkpoint_runtime"] == "not-tested"
