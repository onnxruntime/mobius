# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for the streaming (bounded-RAM) safetensors weight loader.

These cover the pass-through streaming path added to avoid materializing a whole
checkpoint in host RAM: correctness (byte round-trip), the multi-shard index,
deterministic external-data checksums, and the refusal semantics that keep the
loader honest (shape mismatch, and a graph that needs preprocessing).
"""

from __future__ import annotations

import json
import pathlib

import onnx_ir as ir
import pytest
import safetensors.torch
import torch
from onnx_ir import tensor_adapters

from mobius._builder import build_from_module
from mobius._model_package import ModelPackage
from mobius._testing import make_config
from mobius.integrations._weight_loading import (
    StreamingWeightPlan,
    StreamingWeightSource,
    _shard_key_index,
    external_data_checksums,
    stream_preprocessed_safetensors_to_package,
    stream_safetensors_to_model,
)
from mobius.models.base import CausalLMModel


def _weight_initializers(model: ir.Model) -> dict[str, ir.Value]:
    """Return initializers that still need weights (const_value is None)."""
    return {
        name: init
        for name, init in model.graph.initializers.items()
        if init.const_value is None
    }


def _make_checkpoint_state(model: ir.Model) -> dict[str, torch.Tensor]:
    """Build a deterministic state dict matching the model's empty initializers."""
    torch.manual_seed(0)
    state: dict[str, torch.Tensor] = {}
    for name, init in _weight_initializers(model).items():
        torch_dtype = tensor_adapters.to_torch_dtype(init.dtype)
        shape = tuple(int(d) for d in init.shape)
        if torch_dtype.is_floating_point:
            state[name] = (torch.randn(shape) * 0.02).to(torch_dtype)
        else:
            state[name] = torch.zeros(shape, dtype=torch_dtype)
    return state


def _fresh_model() -> ir.Model:
    config = make_config()
    module = CausalLMModel(config)
    return build_from_module(module, config)["model"]


def _save_single(state: dict[str, torch.Tensor], directory: pathlib.Path) -> None:
    safetensors.torch.save_file(state, str(directory / "model.safetensors"))


def _save_sharded(
    state: dict[str, torch.Tensor], directory: pathlib.Path, n_shards: int = 3
) -> None:
    names = list(state)
    weight_map: dict[str, str] = {}
    for i in range(n_shards):
        shard_names = names[i::n_shards]
        if not shard_names:
            continue
        fn = f"model-{i + 1:05d}-of-{n_shards:05d}.safetensors"
        safetensors.torch.save_file({n: state[n] for n in shard_names}, str(directory / fn))
        for n in shard_names:
            weight_map[n] = fn
    (directory / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map})
    )


def _roundtrip_and_compare(model: ir.Model, state, tmp_path: pathlib.Path):
    out = tmp_path / "out"
    out.mkdir()
    ir.save(model, str(out / "model.onnx"), external_data="model.onnx.data")
    reloaded = ir.load(str(out / "model.onnx"))
    for name, tensor in state.items():
        if name not in reloaded.graph.initializers:
            # Folded away (e.g. tied / packed) — skip; covered by other cases.
            continue
        got = torch.from_numpy(reloaded.graph.initializers[name].const_value.numpy().copy())
        expected = tensor.to(got.dtype)
        assert torch.equal(got, expected), f"weight {name} did not round-trip"
    return out


class TestStreamingCorrectness:
    def test_single_file_streams_and_roundtrips(self, tmp_path, monkeypatch):
        def _no_hub(*_a, **_k):
            raise AssertionError("streaming a local dir must not call the Hub")

        monkeypatch.setattr("mobius.integrations._weight_loading.hf_hub_download", _no_hub)
        model = _fresh_model()
        state = _make_checkpoint_state(model)
        _save_single(state, tmp_path)

        assigned = stream_safetensors_to_model(model, str(tmp_path))

        assert assigned == set(state)
        _roundtrip_and_compare(model, state, tmp_path)

    def test_sharded_index_streams_and_roundtrips(self, tmp_path):
        model = _fresh_model()
        state = _make_checkpoint_state(model)
        _save_sharded(state, tmp_path, n_shards=3)

        assigned = stream_safetensors_to_model(model, str(tmp_path))

        assert assigned == set(state)
        _roundtrip_and_compare(model, state, tmp_path)

    def test_assignment_is_deferred_not_materialized(self, tmp_path):
        model = _fresh_model()
        state = _make_checkpoint_state(model)
        _save_single(state, tmp_path)

        stream_safetensors_to_model(model, str(tmp_path))

        # Every streamed weight is a LazyTensor, i.e. nothing is materialized in
        # RAM at assignment time. That is the whole point of the streaming path.
        lazy = [
            init
            for name, init in model.graph.initializers.items()
            if name in state and isinstance(init.const_value, ir.LazyTensor)
        ]
        assert lazy, "expected streamed weights to be deferred LazyTensors"

    def test_preprocessed_package_streams_and_records_every_component(
        self, tmp_path, monkeypatch
    ):
        def _no_hub(*_a, **_k):
            raise AssertionError("streaming a local dir must not call the Hub")

        monkeypatch.setattr("mobius.integrations._weight_loading.hf_hub_download", _no_hub)
        model = _fresh_model()
        state = _make_checkpoint_state(model)
        _save_single(state, tmp_path)
        package = ModelPackage({"decoder": model})

        def plan(sources, initializers):
            return StreamingWeightPlan(
                targets={
                    name: StreamingWeightSource(name, expected_dtype="F32")
                    for name, initializer in initializers.items()
                    if initializer.const_value is None
                }
            )

        report = stream_preprocessed_safetensors_to_package(package, str(tmp_path), plan)

        assert report["assigned_tensors"] == len(state)
        assert model.metadata_props["mobius.weight_loading"]
        _roundtrip_and_compare(model, state, tmp_path)


class TestStreamingRefusals:
    def test_duplicate_tensor_across_shards_is_rejected(self, tmp_path):
        shard_a = tmp_path / "a.safetensors"
        shard_b = tmp_path / "b.safetensors"
        safetensors.torch.save_file({"duplicate.weight": torch.ones(2)}, str(shard_a))
        safetensors.torch.save_file({"duplicate.weight": torch.zeros(2)}, str(shard_b))

        with pytest.raises(ValueError, match=r"Duplicate tensor key 'duplicate\.weight'"):
            _shard_key_index([str(shard_a), str(shard_b)])

    def test_shape_mismatch_raises(self, tmp_path):
        model = _fresh_model()
        state = _make_checkpoint_state(model)
        # Corrupt one tensor's shape.
        bad_name = next(iter(state))
        state[bad_name] = torch.zeros(7, 11)
        _save_single(state, tmp_path)

        with pytest.raises(ValueError, match="shape mismatch"):
            stream_safetensors_to_model(model, str(tmp_path))

    def test_missing_weight_refuses_in_passthrough_mode(self, tmp_path):
        model = _fresh_model()
        state = _make_checkpoint_state(model)
        # Drop a weight the graph needs — the signature of a checkpoint that
        # requires preprocessing rather than a pass-through map.
        dropped = next(iter(state))
        del state[dropped]
        _save_single(state, tmp_path)

        with pytest.raises(ValueError, match="no matching checkpoint tensor"):
            stream_safetensors_to_model(model, str(tmp_path))

    def test_missing_weight_tolerated_when_not_passthrough(self, tmp_path):
        model = _fresh_model()
        state = _make_checkpoint_state(model)
        dropped = next(iter(state))
        del state[dropped]
        _save_single(state, tmp_path)

        assigned = stream_safetensors_to_model(model, str(tmp_path), require_passthrough=False)
        assert dropped not in assigned

    def test_fp8_checkpoint_refuses_rather_than_dropping_scale(self, tmp_path):
        model = _fresh_model()
        state = _make_checkpoint_state(model)
        # Store one weight as raw fp8 (the way DeepSeek/GLM fp8 checkpoints ship)
        # plus a companion scale. A pass-through cast fp8 -> bf16 would silently
        # drop the scale and emit wrong weights, so the loader must refuse.
        target = next(iter(state))
        state[target] = (state[target].float() * 0.1).to(torch.float8_e4m3fn)
        state[f"{target}_scale_inv"] = torch.ones(1, dtype=torch.float32)
        _save_single(state, tmp_path)

        with pytest.raises(ValueError, match="quantized"):
            stream_safetensors_to_model(model, str(tmp_path))

    def test_scale_key_alone_signals_quantized_source(self, tmp_path):
        model = _fresh_model()
        state = _make_checkpoint_state(model)
        # Even with every real weight present, a leftover *_scale_inv tensor is
        # the signature of a quantized checkpoint the pass-through loader can't
        # honor; it must refuse instead of silently ignoring the scale.
        state["extra.weight_scale_inv"] = torch.ones(4, dtype=torch.float32)
        _save_single(state, tmp_path)

        with pytest.raises(ValueError, match="quantized"):
            stream_safetensors_to_model(model, str(tmp_path))


class TestExternalDataChecksums:
    def test_checksums_are_deterministic_across_exports(self, tmp_path):
        model_a = _fresh_model()
        state = _make_checkpoint_state(model_a)
        _save_single(state, tmp_path)

        stream_safetensors_to_model(model_a, str(tmp_path))
        out_a = tmp_path / "a"
        out_a.mkdir()
        ir.save(model_a, str(out_a / "model.onnx"), external_data="model.onnx.data")
        manifest_a = external_data_checksums(out_a)

        # Re-stream from the same checkpoint into a fresh graph.
        model_b = _fresh_model()
        stream_safetensors_to_model(model_b, str(tmp_path))
        out_b = tmp_path / "b"
        out_b.mkdir()
        ir.save(model_b, str(out_b / "model.onnx"), external_data="model.onnx.data")
        manifest_b = external_data_checksums(out_b)

        assert manifest_a  # non-empty (external data was actually written)
        assert manifest_a == manifest_b, "external data must be byte-reproducible"

    def test_manifest_reports_size_and_sha256(self, tmp_path):
        (tmp_path / "model.onnx.data").write_bytes(b"hello world")
        manifest = external_data_checksums(tmp_path)
        assert manifest["model.onnx.data"]["size"] == 11
        assert (
            manifest["model.onnx.data"]["sha256"]
            == "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
        )
