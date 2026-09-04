# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import io
from types import SimpleNamespace

import pytest
import yaml

from mobius.integrations.onnx_genai._metadata_io import _dump_yaml, _published_metadata
from mobius.integrations.onnx_genai._workflow_contract import _Port, _shape_metadata


def test_published_metadata_retires_tensor_rank_without_touching_adapter_rank():
    metadata = {
        "pipeline": {
            "workflow": {
                "manifest": {"capabilities": ["workflow_ssa"]},
            }
        },
        "scalar": {"dtype": "float32", "rank": 0, "shape": []},
        "dynamic": {"dtype": "float32", "rank": 2, "shape": ["batch", "Any"]},
        "adapter": {"dtype": "float32", "rank": 8, "alpha": 16.0},
    }

    published = _published_metadata(metadata)

    assert published["scalar"] == {"dtype": "float32", "shape": []}
    assert published["dynamic"] == {
        "dtype": "float32",
        "shape": ["batch", "Any"],
    }
    assert published["adapter"]["rank"] == 8
    assert published["pipeline"]["workflow"]["manifest"] == {}
    assert metadata["scalar"]["rank"] == 0


def test_published_metadata_rejects_rank_shape_disagreement():
    with pytest.raises(ValueError, match="declares rank 2.*has rank 1"):
        _published_metadata(
            {"contract": {"dtype": "int64", "rank": 2, "shape": ["batch"]}}
        )


def test_dump_yaml_uses_published_tensor_contract():
    output = io.StringIO()
    _dump_yaml(
        {"contract": {"dtype": "float32", "rank": 1, "shape": ["Any"]}},
        output,
    )

    assert yaml.safe_load(output.getvalue()) == {
        "contract": {"dtype": "float32", "shape": ["Any"]}
    }


def test_shape_metadata_preserves_scalar_symbolic_and_independent_dynamic_semantics():
    assert _shape_metadata(_Port(None, "scalar", "fp32", 0, ())) == []
    assert _shape_metadata(
        _Port(
            None,
            "dynamic",
            "fp32",
            3,
            (SimpleNamespace(value="batch"), object(), object()),
        )
    ) == ["batch", "Any", "Any"]


def test_shape_metadata_rejects_unknown_rank_instead_of_inventing_a_scalar():
    with pytest.raises(ValueError, match="unknown rank"):
        _shape_metadata(_Port(None, "unknown", "fp32", None, ()))
