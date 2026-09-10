# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for Mobius-owned graph finalization."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import numpy as np
import onnx_ir as ir
import pytest
from _test_configs import _base_config

from mobius._builder import build_from_module
from mobius._optimizations import _count_ops, fold_initializers_after_weights
from mobius._registry import registry


def _make_llama_pkg(ep: str, dtype: ir.DataType = ir.DataType.FLOAT):
    config = _base_config(dtype=dtype)
    return build_from_module(
        registry.get("llama")(config),
        config,
        execution_provider=ep,
    )


def test_finalization_does_not_apply_post_export_attention_rewrites():
    pkg = _make_llama_pkg("dml", ir.DataType.FLOAT16)
    model = pkg["model"]

    assert _count_ops(model, "GroupQueryAttention") == 0
    assert _count_ops(model, "Attention") > 0


def test_unknown_ep_raises():
    config = _base_config()
    with pytest.raises(ValueError, match="Unknown execution provider"):
        build_from_module(
            registry.get("llama")(config),
            config,
            execution_provider="nonexistent-ep",
        )


def test_onnx_standard_ep_has_no_custom_domain_nodes():
    model = _make_llama_pkg("onnx-standard")["model"]
    non_standard = [
        (node.domain, node.op_type)
        for node in model.graph.all_nodes()
        if node.domain not in {"", "ai.onnx"}
    ]

    assert non_standard == []
    assert _count_ops(model, "FusedMatMul") == 0
    assert _count_ops(model, "MatMul") > 0


def test_fold_initializers_after_weights_eliminates_weight_transposes():
    model = _make_llama_pkg("default")["model"]
    rng = np.random.default_rng(0)
    for initializer in model.graph.initializers.values():
        if initializer.const_value is None:
            shape = [int(dim) for dim in initializer.shape]
            initializer.const_value = ir.Tensor(rng.standard_normal(shape, dtype=np.float32))

    fold_initializers_after_weights(model)

    remaining = [
        node
        for node in model.graph.all_nodes()
        if node.op_type == "Transpose"
        and node.inputs[0] is not None
        and node.inputs[0].name in model.graph.initializers
        and list(node.attributes["perm"].value) == [1, 0]
    ]
    assert remaining == []
