# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Native bank streaming must be bounded, byte-exact and fail closed."""

from __future__ import annotations

import dataclasses
import math
import weakref

import numpy as np
import onnx_ir as ir
import pytest
import safetensors.torch
import torch

from mobius.integrations import _weight_loading as loading


def _fixture(tmp_path):
    shape = (2, 1, 66)
    state = {
        f"expert.{index}": (torch.arange(math.prod(shape)) + index)
        .to(torch.uint8)
        .reshape(shape)
        for index in range(3)
    }
    safetensors.torch.save_file(state, str(tmp_path / "model.safetensors"))
    bank = ir.Value(
        name="bank", type=ir.TensorType(ir.DataType.UINT8), shape=ir.Shape([3, *shape])
    )
    graph = ir.Graph([], [bank], nodes=[], opset_imports={"": 24})
    graph.initializers["bank"] = bank
    source = loading.StreamingPackedExpertBankSource(tuple(state), shape)
    return ir.Model(graph, ir_version=11), source, state


def test_packed_bank_loads_one_expert_at_a_time_without_dense_materialization(
    tmp_path, monkeypatch
):
    model, source, state = _fixture(tmp_path)
    original = loading._load_indexed_tensor
    references = []
    loaded = []

    def tracked_load(name, index):
        assert all(ref() is None for ref in references)
        value = original(name, index)
        references.append(weakref.ref(value))
        loaded.append(name)
        return value

    def reject_dense(*_args):
        raise AssertionError("Native banks must not reconstruct dense weights")

    monkeypatch.setattr(loading, "_load_indexed_tensor", tracked_load)
    monkeypatch.setattr(loading, "_materialize_preprocessed_source", reject_dense)
    source = dataclasses.replace(source, validation_scratch_bytes=8)
    report = loading.stream_preprocessed_safetensors_to_model(
        model,
        str(tmp_path),
        lambda *_args: loading.StreamingWeightPlan(targets={"bank": source}),
    )
    assert loaded == list(state)
    expert_bytes = math.prod(source.expected_source_shape)
    assert report["largest_source_tensor_bytes"] == expert_bytes
    assert report["largest_reconstruction_working_set_bytes"] == 4 * expert_bytes + 8
    assert report["ignored_tensors"] == 0
    for _iteration in range(2):
        actual = model.graph.initializers["bank"].const_value.numpy()
        np.testing.assert_array_equal(actual, torch.stack(list(state.values())).numpy())
    # cache=False: each materialization revalidates/reloads individual source bytes.
    assert loaded == list(state) * 3


@pytest.mark.parametrize("failure", ["duplicate", "shape", "target_dtype", "scratch"])
def test_invalid_packed_bank_contract_refuses_before_binding(tmp_path, failure):
    model, source, _ = _fixture(tmp_path)
    if failure == "duplicate":
        source = dataclasses.replace(source, source_names=("expert.0",) * 3)
    elif failure == "shape":
        source = dataclasses.replace(source, expected_source_shape=(2, 2, 66))
    elif failure == "target_dtype":
        model.graph.initializers["bank"].dtype = ir.DataType.FLOAT
    else:
        source = dataclasses.replace(source, validation_scratch_bytes=-1)
    with pytest.raises(ValueError):
        loading.stream_preprocessed_safetensors_to_model(
            model,
            str(tmp_path),
            lambda *_args: loading.StreamingWeightPlan(targets={"bank": source}),
        )
    assert model.graph.initializers["bank"].const_value is None


def test_qdq_refuses_native_expert_bank_without_reinterpreting_bytes(tmp_path):
    model, source, _ = _fixture(tmp_path)
    with pytest.raises(TypeError, match="use the native streaming loader"):
        loading.stream_qdq_safetensors_to_model(
            model,
            str(tmp_path),
            lambda *_args: loading.StreamingWeightPlan(targets={"bank": source}),
        )
    assert model.graph.initializers["bank"].const_value is None
