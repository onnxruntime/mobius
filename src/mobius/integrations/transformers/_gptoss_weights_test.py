# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for bounded native GPT-OSS MXFP4 safetensors loading."""

from __future__ import annotations

import gc
import json
import weakref
from pathlib import Path

import onnx
import onnx_ir as ir
import pytest
import safetensors.torch
import torch
from onnx_ir import tensor_adapters
from onnx_ir.serde import SerdeError

from mobius._builder import build_from_module
from mobius._configs import QuantizationConfig, QuantizedWeightFormat
from mobius._model_package import ModelPackage
from mobius._testing import make_config
from mobius.integrations.transformers import _builder as transformers_builder
from mobius.integrations.transformers import _config_resolver, _gptoss_weights
from mobius.models.gptoss import GPTOSSCausalLMModel, repack_gptoss_mxfp4_blocks

_E = 2
_H = 64
_I = 32
_ROOT = "model.layers.0.mlp"


def _config(**overrides):
    options = dict(
        model_type="gpt_oss",
        dtype=ir.DataType.FLOAT16,
        num_hidden_layers=1,
        hidden_size=_H,
        intermediate_size=_I,
        num_local_experts=_E,
        num_experts_per_tok=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        vocab_size=32,
        layer_types=["sliding_attention"],
        sliding_window=32,
        partial_rotary_factor=1.0,
        rope_interleave=False,
        attn_qkv_bias=True,
        attn_o_bias=True,
        quantization=QuantizationConfig(
            bits=4,
            group_size=32,
            quant_method="mxfp4",
            weight_format=QuantizedWeightFormat.MXFP4,
        ),
    )
    options.update(overrides)
    return make_config(**options)


def _package_and_state():
    config = _config()
    module = GPTOSSCausalLMModel(config)
    package = build_from_module(module, config, execution_provider="cuda")
    model = package["model"]
    state: dict[str, torch.Tensor] = {}
    special_targets = {
        f"{_ROOT}.fc1_experts_weights",
        f"{_ROOT}.fc1_scales",
        f"{_ROOT}.fc1_experts_bias",
        f"{_ROOT}.fc1_global_scales",
        f"{_ROOT}.fc2_experts_weights",
        f"{_ROOT}.fc2_scales",
        f"{_ROOT}.fc2_experts_bias",
        f"{_ROOT}.fc2_global_scales",
        f"{_ROOT}.gate.weight",
        f"{_ROOT}.gate.bias",
    }
    for name, initializer in model.graph.initializers.items():
        if initializer.const_value is not None or name in special_targets:
            continue
        dtype = tensor_adapters.to_torch_dtype(initializer.dtype)
        shape = tuple(int(dim) for dim in initializer.shape)
        state[name] = torch.zeros(shape, dtype=dtype)

    state.update(
        {
            f"{_ROOT}.experts.gate_up_proj_blocks": torch.randint(
                0, 256, (_E, 2 * _I, _H // 32, 16), dtype=torch.uint8
            ),
            f"{_ROOT}.experts.gate_up_proj_scales": torch.randint(
                0, 255, (_E, 2 * _I, _H // 32), dtype=torch.uint8
            ),
            f"{_ROOT}.experts.down_proj_blocks": torch.randint(
                0, 256, (_E, _H, _I // 32, 16), dtype=torch.uint8
            ),
            f"{_ROOT}.experts.down_proj_scales": torch.randint(
                0, 255, (_E, _H, _I // 32), dtype=torch.uint8
            ),
            f"{_ROOT}.experts.gate_up_proj_bias": torch.randn(_E, 2 * _I, dtype=torch.float16),
            f"{_ROOT}.experts.down_proj_bias": torch.randn(_E, _H, dtype=torch.float16),
            f"{_ROOT}.router.weight": torch.randn(_E, _H, dtype=torch.float16),
            f"{_ROOT}.router.bias": torch.randn(_E, dtype=torch.float16),
        }
    )
    return config, package, state


def _save_cross_sharded(state, directory, *, shard_prefix=""):
    names = sorted(state)
    shard_a_names = [
        name for index, name in enumerate(names) if name.endswith("_blocks") or index % 2 == 0
    ]
    shard_b_names = [
        name
        for index, name in enumerate(names)
        if name.endswith("_scales") or (index % 2 == 1 and not name.endswith("_blocks"))
    ]
    # Explicitly put every blocks/scales pair in different files.
    shard_a_names = [name for name in shard_a_names if not name.endswith("_scales")]
    shard_b_names = [name for name in shard_b_names if not name.endswith("_blocks")]
    shards = {
        "model-00001-of-00002.safetensors": {name: state[name] for name in shard_a_names},
        "model-00002-of-00002.safetensors": {name: state[name] for name in shard_b_names},
    }
    weight_map = {}
    for filename, tensors in shards.items():
        relative_filename = f"{shard_prefix}/{filename}" if shard_prefix else filename
        shard_path = directory / relative_filename
        shard_path.parent.mkdir(parents=True, exist_ok=True)
        safetensors.torch.save_file(tensors, str(shard_path))
        weight_map.update(dict.fromkeys(tensors, relative_filename))
    (directory / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map})
    )


def _save_single_file(state, directory):
    safetensors.torch.save_file(state, str(directory / "model.safetensors"))


def test_lazy_source_parent_aliases_include_snapshot_and_blob_directories(tmp_path):
    snapshot = tmp_path / "snapshots" / "revision"
    blobs = tmp_path / "blobs"
    snapshot.mkdir(parents=True)
    blobs.mkdir()
    blob = blobs / "hash"
    blob.write_bytes(b"shard")
    shard = snapshot / "model.safetensors"
    try:
        shard.symlink_to(blob)
    except OSError as error:
        pytest.skip(f"file symlinks are unavailable: {error}")

    aliases = _gptoss_weights._lazy_safetensors_source_parent_aliases([str(shard)])

    assert aliases == frozenset({snapshot.resolve(), blobs.resolve()})


def _directory_snapshot(directory: Path) -> tuple[tuple[str, ...], dict[str, bytes]]:
    entries = tuple(
        sorted(
            f"{path.relative_to(directory)}{'/' if path.is_dir() else ''}"
            for path in directory.rglob("*")
        )
    )
    contents = {
        str(path.relative_to(directory)): path.read_bytes()
        for path in directory.rglob("*")
        if path.is_file()
    }
    return entries, contents


def _replace_source_shard(state, directory, source_name):
    index = json.loads((directory / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    filename = weight_map[source_name]
    shard_names = [name for name, shard in weight_map.items() if shard == filename]
    safetensors.torch.save_file(
        {name: state[name] for name in shard_names},
        str(directory / filename),
    )


def test_cross_shard_pairs_transform_and_bind_final_initializers(tmp_path, monkeypatch):
    config, package, state = _package_and_state()
    _save_cross_sharded(state, tmp_path)
    source_refs: list[weakref.ReferenceType[torch.Tensor]] = []

    def tracked_repack(tensor, _source_name):
        source_refs.append(weakref.ref(tensor))
        return repack_gptoss_mxfp4_blocks(tensor)

    monkeypatch.setattr(_gptoss_weights, "_repack_blocks", tracked_repack)

    report = _gptoss_weights.stream_gptoss_mxfp4_safetensors_to_package(
        package, str(tmp_path), config
    )

    model = package["model"]
    fc1 = model.graph.initializers[f"{_ROOT}.fc1_experts_weights"]
    fc2 = model.graph.initializers[f"{_ROOT}.fc2_experts_weights"]
    assert isinstance(fc1.const_value, ir.LazyTensor)
    assert isinstance(fc2.const_value, ir.LazyTensor)
    assert report["output_weight_format"] == "mxfp4"
    assert report["streaming_unit"] == "one_moe_projection"

    actual_fc1 = torch.from_numpy(fc1.const_value.numpy().copy())
    torch.testing.assert_close(
        actual_fc1,
        repack_gptoss_mxfp4_blocks(state[f"{_ROOT}.experts.gate_up_proj_blocks"]),
    )
    del actual_fc1
    gc.collect()
    assert source_refs[0]() is None

    actual_fc2 = torch.from_numpy(fc2.const_value.numpy().copy())
    torch.testing.assert_close(
        actual_fc2,
        repack_gptoss_mxfp4_blocks(state[f"{_ROOT}.experts.down_proj_blocks"]),
    )
    del actual_fc2
    gc.collect()
    assert all(reference() is None for reference in source_refs)

    scales = model.graph.initializers[f"{_ROOT}.fc1_scales"].const_value
    assert scales is not None
    actual_scale_bytes = torch.from_numpy(scales.numpy().view("uint8").copy())
    torch.testing.assert_close(
        actual_scale_bytes,
        state[f"{_ROOT}.experts.gate_up_proj_scales"],
    )
    global_scales = model.graph.initializers[f"{_ROOT}.fc1_global_scales"].const_value
    assert global_scales is not None
    assert global_scales.dtype == ir.DataType.FLOAT


def test_mxfp4_package_save_uses_bounded_serial_serializer(tmp_path, monkeypatch):
    config, package, state = _package_and_state()
    _save_cross_sharded(state, tmp_path)
    report = _gptoss_weights.stream_gptoss_mxfp4_safetensors_to_package(
        package, str(tmp_path), config
    )
    calls = []

    def save(
        model,
        path,
        *,
        external_data,
        max_shard_size_bytes,
        callback,
        max_workers,
    ):
        calls.append((external_data, max_shard_size_bytes, max_workers))

    monkeypatch.setattr(ir, "save", save)

    package.save(
        str(tmp_path / "output"),
        max_workers=8,
        progress_bar=False,
        check_weights=False,
    )

    assert report["streaming_external_data"] is True
    assert calls == [("model.onnx.data", 1 << 30, 1)]
    assert package.weight_loading_report is not None
    assert package.weight_loading_report["external_data_shard_limit_bytes"] == 1 << 30
    assert package.weight_loading_report["serializer_max_workers"] == 1
    assert (
        "forced to one worker" in package.weight_loading_report["serialization_memory_bound"]
    )


@pytest.mark.parametrize(
    ("layout", "alias"),
    [
        ("sharded", "direct"),
        ("single", "relative"),
        ("sharded", "symlink"),
    ],
)
def test_safetensors_save_rejects_source_directory_alias_before_mutation(
    tmp_path, monkeypatch, layout, alias
):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    config, package, state = _package_and_state()
    if layout == "sharded":
        _save_cross_sharded(state, checkpoint)
    else:
        _save_single_file(state, checkpoint)
    _gptoss_weights.stream_gptoss_mxfp4_safetensors_to_package(
        package, str(checkpoint), config
    )
    assert package._native_streaming_source_directories == frozenset({checkpoint.resolve()})

    if alias == "direct":
        destination = checkpoint
    elif alias == "relative":
        monkeypatch.chdir(tmp_path)
        destination = Path("checkpoint")
    else:
        destination = tmp_path / "checkpoint-alias"
        try:
            destination.symlink_to(checkpoint, target_is_directory=True)
        except OSError as error:
            pytest.skip(f"directory symlinks are unavailable: {error}")

    source_before = _directory_snapshot(checkpoint)
    parent_listing_before = tuple(sorted(path.name for path in tmp_path.iterdir()))
    with pytest.raises(ValueError, match="separate output directory"):
        package.save(
            str(destination),
            external_data="safetensors",
            progress_bar=False,
        )

    assert _directory_snapshot(checkpoint) == source_before
    assert tuple(sorted(path.name for path in tmp_path.iterdir())) == parent_listing_before


def test_safetensors_save_rejects_hard_link_to_lazy_source_before_mutation(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    output = tmp_path / "output"
    checkpoint.mkdir()
    output.mkdir()
    config, package, state = _package_and_state()
    _save_single_file(state, checkpoint)
    _gptoss_weights.stream_gptoss_mxfp4_safetensors_to_package(
        package, str(checkpoint), config
    )
    source = checkpoint / "model.safetensors"
    hard_link = output / "model.safetensors"
    try:
        hard_link.hardlink_to(source)
    except OSError as error:
        pytest.skip(f"hard links are unavailable: {error}")

    source_before = _directory_snapshot(checkpoint)
    output_before = _directory_snapshot(output)
    with pytest.raises(ValueError, match="aliases lazy source file"):
        package.save(
            str(output),
            external_data="safetensors",
            progress_bar=False,
        )

    assert _directory_snapshot(checkpoint) == source_before
    assert _directory_snapshot(output) == output_before


def test_nested_shard_checkpoint_root_is_registered_and_rejected_for_safetensors(
    tmp_path,
):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    config, package, state = _package_and_state()
    _save_cross_sharded(state, checkpoint, shard_prefix="weights")
    _gptoss_weights.stream_gptoss_mxfp4_safetensors_to_package(
        package, str(checkpoint), config
    )

    assert package._native_streaming_source_directories == frozenset(
        {checkpoint.resolve(), (checkpoint / "weights").resolve()}
    )
    assert (checkpoint / "model.safetensors.index.json").resolve() in (
        package._native_streaming_source_files
    )
    source_before = _directory_snapshot(checkpoint)

    with pytest.raises(ValueError, match="source checkpoint directory"):
        package.save(
            str(checkpoint),
            external_data="safetensors",
            progress_bar=False,
        )

    assert _directory_snapshot(checkpoint) == source_before


@pytest.mark.parametrize(
    ("external_data", "output_name", "link_kind"),
    [
        ("onnx", "model.onnx", "symlink"),
        ("onnx", "model.onnx.data", "hardlink"),
        ("onnx", "weight-loading-report.json", "hardlink"),
        ("safetensors", "model.safetensors.index.json", "hardlink"),
    ],
)
def test_streaming_save_rejects_exact_output_file_alias_for_every_format(
    tmp_path,
    external_data,
    output_name,
    link_kind,
):
    checkpoint = tmp_path / "checkpoint"
    output = tmp_path / "output"
    checkpoint.mkdir()
    output.mkdir()
    config, package, state = _package_and_state()
    _save_single_file(state, checkpoint)
    _gptoss_weights.stream_gptoss_mxfp4_safetensors_to_package(
        package, str(checkpoint), config
    )
    source = checkpoint / "model.safetensors"
    alias = output / output_name
    try:
        if link_kind == "symlink":
            alias.symlink_to(source)
        else:
            alias.hardlink_to(source)
    except OSError as error:
        pytest.skip(f"{link_kind}s are unavailable: {error}")

    source_before = source.read_bytes()
    output_before = _directory_snapshot(output)
    with pytest.raises(ValueError, match="aliases lazy source file"):
        package.save(
            str(output),
            external_data=external_data,
            progress_bar=False,
        )

    assert source.read_bytes() == source_before
    assert _directory_snapshot(output) == output_before


def test_added_component_cannot_redirect_output_into_lazy_source_before_any_write(
    tmp_path,
):
    parent = tmp_path / "parent"
    checkpoint = parent / "model"
    checkpoint.mkdir(parents=True)
    config, package, state = _package_and_state()
    _save_cross_sharded(state, checkpoint)
    _gptoss_weights.stream_gptoss_mxfp4_safetensors_to_package(
        package, str(checkpoint), config
    )
    # Streaming started with one flat "model" component. Adding another
    # component changes "model" to a subdirectory output at the source path.
    package["auxiliary"] = ir.Model(
        ir.Graph([], [], nodes=[], name="auxiliary"),
        ir_version=11,
    )
    source_before = _directory_snapshot(checkpoint)
    parent_listing_before = tuple(sorted(path.name for path in parent.iterdir()))

    with pytest.raises(ValueError, match="still read lazily"):
        package.save(
            str(parent),
            external_data="safetensors",
            progress_bar=False,
        )

    assert _directory_snapshot(checkpoint) == source_before
    assert tuple(sorted(path.name for path in parent.iterdir())) == parent_listing_before


def test_stateful_component_filter_is_frozen_before_streaming_collision_validation(
    tmp_path,
):
    parent = tmp_path / "parent"
    checkpoint = parent / "model"
    checkpoint.mkdir(parents=True)
    config, package, state = _package_and_state()
    _save_cross_sharded(state, checkpoint)
    _gptoss_weights.stream_gptoss_mxfp4_safetensors_to_package(
        package, str(checkpoint), config
    )
    package["auxiliary"] = ir.Model(
        ir.Graph([], [], nodes=[], name="auxiliary"),
        ir_version=11,
    )
    calls: list[str] = []

    def stateful_filter(name: str) -> bool:
        calls.append(name)
        # A later evaluation would switch from a two-component/subdirectory
        # layout to a one-component/flat layout.
        return len(calls) <= len(package) or name == "model"

    source_before = _directory_snapshot(checkpoint)
    parent_listing_before = tuple(sorted(path.name for path in parent.iterdir()))

    with pytest.raises(ValueError, match="still read lazily"):
        package.save(
            str(parent),
            external_data="safetensors",
            components=stateful_filter,
            progress_bar=False,
        )

    assert calls == list(package)
    assert _directory_snapshot(checkpoint) == source_before
    assert tuple(sorted(path.name for path in parent.iterdir())) == parent_listing_before


def test_component_filter_does_not_replace_existing_streaming_destination(tmp_path):
    parent = tmp_path / "parent"
    checkpoint = parent / "model"
    checkpoint.mkdir(parents=True)
    config, package, state = _package_and_state()
    _save_cross_sharded(state, checkpoint)
    _gptoss_weights.stream_gptoss_mxfp4_safetensors_to_package(
        package, str(checkpoint), config
    )
    package["auxiliary"] = ir.Model(
        ir.Graph([], [], nodes=[], name="auxiliary"),
        ir_version=11,
    )
    source_before = _directory_snapshot(checkpoint)

    with pytest.raises(FileExistsError, match="destination already exists"):
        package.save(
            str(parent),
            external_data="safetensors",
            components=lambda name: name == "auxiliary",
            progress_bar=False,
        )

    assert _directory_snapshot(checkpoint) == source_before
    assert not (parent / "model.onnx").exists()


def test_safetensors_save_to_separate_directory_roundtrips_without_source_path(
    tmp_path,
):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    config, package, state = _package_and_state()
    _save_cross_sharded(state, checkpoint)
    _gptoss_weights.stream_gptoss_mxfp4_safetensors_to_package(
        package, str(checkpoint), config
    )

    output = tmp_path / "output"
    package.save(
        str(output),
        external_data="safetensors",
        progress_bar=False,
    )
    loaded = ModelPackage.load(str(output))

    source_path = str(checkpoint.resolve())
    report_bytes = (output / "weight-loading-report.json").read_bytes()
    metadata = loaded["model"].metadata_props["mobius.weight_loading"]
    assert source_path.encode() not in report_bytes
    assert source_path not in metadata
    assert loaded.weight_loading_report is not None
    assert loaded.weight_loading_report["source"] == "local-safetensors-checkpoint"
    assert loaded["model"].ir_version == 12
    onnx.checker.check_model(str(output / "model.onnx"))
    scale = loaded["model"].graph.initializers[f"{_ROOT}.fc1_scales"].const_value
    assert scale is not None
    torch.testing.assert_close(
        torch.from_numpy(scale.numpy().view("uint8").copy()),
        state[f"{_ROOT}.experts.gate_up_proj_scales"],
    )


def test_transformers_builder_keeps_local_checkpoint_path_out_of_portable_fields(
    tmp_path,
    monkeypatch,
):
    checkpoint = tmp_path / "absolute-local-checkpoint"
    checkpoint.mkdir()
    config, package, state = _package_and_state()
    _save_cross_sharded(state, checkpoint)
    hf_config = type("HFConfig", (), {"model_type": "gpt_oss"})()
    monkeypatch.setattr(
        transformers_builder,
        "_load_transformers_config",
        lambda *args, **kwargs: (hf_config, False),
    )
    monkeypatch.setattr(
        transformers_builder,
        "_select_primary_config",
        lambda value: (value, value, "gpt_oss"),
    )
    monkeypatch.setattr(
        transformers_builder,
        "_resolve_module_class",
        lambda *args, **kwargs: (
            GPTOSSCausalLMModel,
            "text-generation",
            "gpt_oss",
        ),
    )
    monkeypatch.setattr(
        _config_resolver,
        "_config_from_hf",
        lambda *args, **kwargs: config,
    )
    monkeypatch.setattr(
        transformers_builder,
        "build_from_module",
        lambda *args, **kwargs: package,
    )

    result = transformers_builder.build_transformers_model(
        str(checkpoint.resolve()),
        execution_provider="cuda",
    )

    source_path = str(checkpoint.resolve())
    model = result["model"]
    assert model.graph.name == "gpt_oss/model"
    assert source_path not in model.graph.name
    assert source_path not in json.dumps(result.weight_loading_report, sort_keys=True)
    assert source_path not in json.dumps(dict(model.metadata_props), sort_keys=True)


def test_non_streaming_package_can_save_safetensors_in_existing_directory(tmp_path):
    package = ModelPackage(
        {"model": ir.Model(ir.Graph([], [], nodes=[], name="model"), ir_version=11)}
    )

    package.save(
        str(tmp_path),
        external_data="safetensors",
        progress_bar=False,
    )

    assert (tmp_path / "model.onnx").is_file()


def test_native_streaming_rejects_onnx_output_in_existing_source_directory(tmp_path):
    config, package, state = _package_and_state()
    _save_cross_sharded(state, tmp_path)
    _gptoss_weights.stream_gptoss_mxfp4_safetensors_to_package(package, str(tmp_path), config)
    source_contents = _directory_snapshot(tmp_path)[1]

    with pytest.raises(FileExistsError, match="destination already exists"):
        package.save(str(tmp_path), external_data="onnx", progress_bar=False)

    assert _directory_snapshot(tmp_path)[1] == source_contents
    assert not (tmp_path / "model.onnx").exists()


@pytest.mark.parametrize("replacement_byte", [0x00, 0xFE])
def test_lazy_scale_materialization_accepts_finite_boundaries_byte_exactly(
    tmp_path, replacement_byte
):
    config, package, state = _package_and_state()
    scale_name = f"{_ROOT}.experts.gate_up_proj_scales"
    state[scale_name].fill_(0x01)
    _save_cross_sharded(state, tmp_path)
    _gptoss_weights.stream_gptoss_mxfp4_safetensors_to_package(package, str(tmp_path), config)

    state[scale_name].fill_(replacement_byte)
    _replace_source_shard(state, tmp_path, scale_name)
    output = tmp_path / "output"
    package.save(str(output), progress_bar=False)

    loaded = ir.load(str(output / "model.onnx"))
    scales = loaded.graph.initializers[f"{_ROOT}.fc1_scales"].const_value
    assert scales is not None
    actual = torch.from_numpy(scales.numpy().view("uint8").copy())
    torch.testing.assert_close(actual, state[scale_name])


def test_lazy_scale_materialization_revalidates_replaced_source_shard(tmp_path):
    config, package, state = _package_and_state()
    scale_name = f"{_ROOT}.experts.gate_up_proj_scales"
    state[scale_name].fill_(0x01)
    _save_cross_sharded(state, tmp_path)
    _gptoss_weights.stream_gptoss_mxfp4_safetensors_to_package(package, str(tmp_path), config)

    state[scale_name].fill_(0xFF)
    _replace_source_shard(state, tmp_path, scale_name)

    with pytest.raises(SerdeError) as exc_info:
        package.save(str(tmp_path / "output"), progress_bar=False)
    cause: BaseException = exc_info.value
    while cause.__cause__ is not None:
        cause = cause.__cause__
    assert isinstance(cause, ValueError)
    assert "0xff (NaN)" in str(cause)


def test_pass_through_u8_tensor_is_rejected_during_header_preflight(tmp_path):
    config, package, state = _package_and_state()
    direct_name = next(name for name in state if ".mlp.experts." not in name)
    state[direct_name] = torch.zeros(state[direct_name].shape, dtype=torch.uint8)
    _save_cross_sharded(state, tmp_path)

    with pytest.raises(ValueError, match=r"pass-through.*dtype.*U8"):
        _gptoss_weights.stream_gptoss_mxfp4_safetensors_to_package(
            package, str(tmp_path), config
        )


def test_lazy_direct_materialization_rechecks_indexed_dtype(tmp_path):
    config, package, state = _package_and_state()
    direct_name = next(name for name in state if ".mlp.experts." not in name)
    _save_single_file(state, tmp_path)
    _gptoss_weights.stream_gptoss_mxfp4_safetensors_to_package(package, str(tmp_path), config)

    state[direct_name] = torch.zeros(state[direct_name].shape, dtype=torch.uint8)
    _save_single_file(state, tmp_path)
    direct = package["model"].graph.initializers[direct_name].const_value
    assert direct is not None

    with pytest.raises(ValueError, match=r"changed after indexing.*U8"):
        direct.numpy()


@pytest.mark.parametrize("mismatch", ["shape", "dtype"])
def test_transformed_target_metadata_mismatch_fails_before_transform(
    tmp_path, monkeypatch, mismatch
):
    config, package, state = _package_and_state()
    _save_cross_sharded(state, tmp_path)
    target = package["model"].graph.initializers[f"{_ROOT}.fc1_experts_weights"]
    if mismatch == "shape":
        target.shape = ir.Shape([_E, _H, _I + 1])
        error = ValueError
    else:
        target.type = ir.TensorType(ir.DataType.FLOAT16)
        error = TypeError
    transform = pytest.fail
    monkeypatch.setattr(_gptoss_weights, "_repack_blocks", transform)

    with pytest.raises(error, match=rf"declares target {mismatch}"):
        _gptoss_weights.stream_gptoss_mxfp4_safetensors_to_package(
            package, str(tmp_path), config
        )


def test_unexpected_expert_diagnostics_are_counted_and_bounded(tmp_path):
    config, package, state = _package_and_state()
    for index in range(20):
        state[f"{_ROOT}.experts.unexpected_{index:02d}"] = torch.zeros(1)
    _save_single_file(state, tmp_path)

    with pytest.raises(ValueError) as exc_info:
        _gptoss_weights.stream_gptoss_mxfp4_safetensors_to_package(
            package, str(tmp_path), config
        )

    message = str(exc_info.value)
    assert "unexpected 20" in message
    assert "unexpected_00" in message
    assert "unexpected_19" not in message


@pytest.mark.parametrize("malformation", ["missing_pair", "invalid_scale"])
def test_incomplete_or_invalid_native_set_fails_before_assignment(tmp_path, malformation):
    config, package, state = _package_and_state()
    if malformation == "missing_pair":
        del state[f"{_ROOT}.experts.down_proj_scales"]
    else:
        state[f"{_ROOT}.experts.down_proj_scales"].reshape(-1)[-1] = 0xFF
    _save_cross_sharded(state, tmp_path)
    target = package["model"].graph.initializers[f"{_ROOT}.fc1_experts_weights"]

    with pytest.raises(ValueError, match=r"Malformed|0xff"):
        _gptoss_weights.stream_gptoss_mxfp4_safetensors_to_package(
            package, str(tmp_path), config
        )

    assert target.const_value is None


def test_native_legacy_checkpoint_fails_closed_without_eager_fallback(tmp_path):
    config, package, _state = _package_and_state()
    torch.save({"not": torch.ones(1)}, tmp_path / "pytorch_model.bin")

    with pytest.raises(ValueError, match="requires a safetensors checkpoint"):
        _gptoss_weights.stream_gptoss_mxfp4_safetensors_to_package(
            package, str(tmp_path), config
        )


@pytest.mark.parametrize(
    ("layers", "experts", "profile"),
    [(24, 32, "gpt-oss-20b"), (36, 128, "gpt-oss-120b")],
)
def test_official_profile_geometry_is_header_only(layers, experts, profile):
    config = _config(
        num_hidden_layers=layers,
        num_local_experts=experts,
        hidden_size=2880,
        intermediate_size=2880,
    )

    specs = _gptoss_weights._native_mxfp4_projection_specs(config)

    assert len(specs) == layers, profile
    assert specs["model.layers.0.mlp"]["gate_up_proj"][0] == (
        experts,
        5760,
        90,
        16,
    )
    assert specs[f"model.layers.{layers - 1}.mlp"]["down_proj"][0] == (
        experts,
        2880,
        90,
        16,
    )
