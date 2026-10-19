# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import onnx_ir as ir
import pytest
import safetensors.torch
import torch

from mobius._builder import build_from_module
from mobius.integrations.transformers import _qwen4_exp_weights as qwen4_weights
from mobius.integrations.transformers._qwen4_exp_weights import (
    _source_name,
    stream_qwen4_exp_safetensors_to_model,
    stream_qwen4_exp_safetensors_to_package,
)
from mobius.models.qwen4_exp import (
    Qwen4ExpCausalLMModel,
    Qwen4ExpForConditionalGeneration,
    _qwen4_exp_ple_buffer_values,
)
from mobius.models.qwen4_exp_test import _config, _vl_config
from mobius.tasks._qwen4_exp import Qwen4ExpCausalLMTask, Qwen4ExpVisionLanguageTask


def _official_state(module: Qwen4ExpCausalLMModel) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    for target_name, parameter in module.named_parameters():
        value = (
            torch.from_numpy(parameter._const_value.numpy().copy())
            if parameter._const_value is not None
            else torch.randn(tuple(int(dim) for dim in parameter.shape))
        )
        source_name = (
            f"model.language_model.{target_name[len('model.') :]}"
            if target_name.startswith("model.")
            else target_name
        )
        if target_name.endswith(".ple.ple_embedding.ngram_embedding.weight"):
            rows = value.shape[0]
            for shard_index, shard in enumerate(
                torch.tensor_split(value, module.config.split_ngram_parts, dim=0)
            ):
                state[f"{source_name[: -len('.weight')]}.shard_{shard_index}.weight"] = (
                    shard.contiguous()
                )
            assert (
                sum(
                    tensor.shape[0]
                    for name, tensor in state.items()
                    if name.startswith(source_name[: -len(".weight")])
                )
                == rows
            )
            continue
        state[source_name] = value
    return state


def test_streaming_loader_maps_packed_experts_and_ple_without_eager_checkpoint(
    tmp_path,
):
    config = _config(split_ngram_parts=2)
    module = Qwen4ExpCausalLMModel(config)
    package = build_from_module(
        module,
        config,
        task="qwen4-exp-text-generation",
    )
    source = _official_state(module)
    safetensors.torch.save_file(source, tmp_path / "model.safetensors")

    stream_qwen4_exp_safetensors_to_model(
        package["model"],
        str(tmp_path),
        config,
    )

    initializers = package["model"].graph.initializers
    assert all(value.const_value is not None for value in initializers.values())
    assert (
        initializers["model.layers.0.mlp.experts.gate_up_proj"].const_value.dtype
        == ir.DataType.FLOAT
    )
    ple_initializers = [
        initializer
        for initializer in initializers.values()
        if ".ple.ple_embedding.ngram_embedding.shard_" in initializer.name
        and initializer.name.endswith(".weight")
    ]
    assert len(ple_initializers) == config.split_ngram_parts
    assert all(
        isinstance(initializer.const_value, ir.LazyTensor) for initializer in ple_initializers
    )


def _official_package_state(package, config) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    for model in package.values():
        for target_name, initializer in model.graph.initializers.items():
            if initializer.const_value is not None:
                if not target_name.endswith(
                    (
                        ".ple_embedding.layer_multipliers",
                        ".ple_embedding.ngram_heads_vocab_sizes",
                        ".ple_embedding.ngram_heads_offsets",
                    )
                ):
                    continue
                value = torch.from_numpy(initializer.const_value.numpy().copy())
            else:
                value = torch.randn(tuple(int(dim) for dim in initializer.shape))
            source_name = _source_name(target_name)
            if target_name.endswith(".ple.ple_embedding.ngram_embedding.weight"):
                for shard_index, shard in enumerate(
                    torch.tensor_split(value, config.split_ngram_parts, dim=0)
                ):
                    state[f"{source_name[: -len('.weight')]}.shard_{shard_index}.weight"] = (
                        shard.contiguous()
                    )
            else:
                existing = state.setdefault(source_name, value)
                assert existing.shape == value.shape
    for ple_layer_index, layer_id in enumerate(config.ple_layer_ids or []):
        buffers, _padded_vocab_size = _qwen4_exp_ple_buffer_values(
            config,
            ple_layer_index,
        )
        prefix = f"model.language_model.layers.{layer_id - 1}.ple.ple_embedding."
        for name, value in buffers.items():
            state[f"{prefix}{name}"] = torch.from_numpy(value.copy())
    return state


def test_multimodal_streaming_is_transactional_and_retains_no_source_tensors(
    tmp_path,
    monkeypatch,
):
    config = _vl_config(split_ngram_parts=2)
    package = build_from_module(
        Qwen4ExpForConditionalGeneration(config),
        config,
        task="qwen4-exp-vision-language",
    )
    source = _official_package_state(package, config)
    safetensors.torch.save_file(source, tmp_path / "model.safetensors")

    report = stream_qwen4_exp_safetensors_to_package(
        package,
        str(tmp_path),
        config,
    )

    assert report.model_count == 3
    assert report.lazy_initializer_count > 0
    assert report.initializer_count >= report.lazy_initializer_count
    assert 0 < report.eagerly_validated_bytes < 4096
    assert report.retained_source_tensor_count == 0
    for model in package.values():
        assert all(
            initializer.const_value is not None
            for initializer in model.graph.initializers.values()
        )
    ple_initializers = [
        initializer
        for initializer in package["decoder"].graph.initializers.values()
        if ".ple.ple_embedding.ngram_embedding.shard_" in initializer.name
        and initializer.name.endswith(".weight")
    ]
    assert len(ple_initializers) == config.split_ngram_parts
    assert all(
        isinstance(initializer.const_value, ir.LazyTensor) for initializer in ple_initializers
    )

    failed_package = build_from_module(
        Qwen4ExpForConditionalGeneration(config),
        config,
        task="qwen4-exp-vision-language",
    )
    missing_source = dict(source)
    missing_name = next(
        name for name in missing_source if name.startswith("model.visual.patch_embed")
    )
    missing_source.pop(missing_name)
    failed_dir = tmp_path / "missing"
    failed_dir.mkdir()
    safetensors.torch.save_file(
        missing_source,
        failed_dir / "model.safetensors",
    )
    unset_before = {
        (model_name, initializer.name)
        for model_name, model in failed_package.items()
        for initializer in model.graph.initializers.values()
        if initializer.const_value is None
    }
    with pytest.raises(ValueError, match="checkpoint has no source tensor"):
        stream_qwen4_exp_safetensors_to_package(
            failed_package,
            str(failed_dir),
            config,
        )
    unset_after = {
        (model_name, initializer.name)
        for model_name, model in failed_package.items()
        for initializer in model.graph.initializers.values()
        if initializer.const_value is None
    }
    assert unset_after == unset_before

    corrupt_package = build_from_module(
        Qwen4ExpForConditionalGeneration(config),
        config,
        task="qwen4-exp-vision-language",
    )
    corrupt_source = dict(source)
    multiplier_name = next(
        name for name in corrupt_source if name.endswith("layer_multipliers")
    )
    corrupt_source[multiplier_name] = torch.full_like(
        corrupt_source[multiplier_name],
        999,
    )
    corrupt_dir = tmp_path / "corrupt"
    corrupt_dir.mkdir()
    safetensors.torch.save_file(
        corrupt_source,
        corrupt_dir / "model.safetensors",
    )
    corrupt_unset_before = {
        (model_name, initializer.name)
        for model_name, model in corrupt_package.items()
        for initializer in model.graph.initializers.values()
        if initializer.const_value is None
    }
    with pytest.raises(ValueError, match="does not match the pinned hash"):
        stream_qwen4_exp_safetensors_to_package(
            corrupt_package,
            str(corrupt_dir),
            config,
        )
    corrupt_unset_after = {
        (model_name, initializer.name)
        for model_name, model in corrupt_package.items()
        for initializer in model.graph.initializers.values()
        if initializer.const_value is None
    }
    assert corrupt_unset_after == corrupt_unset_before

    fold_failed_package = build_from_module(
        Qwen4ExpForConditionalGeneration(config),
        config,
        task="qwen4-exp-vision-language",
    )
    original_models = dict(fold_failed_package)
    unset_before_fold = {
        (model_name, initializer.name)
        for model_name, model in fold_failed_package.items()
        for initializer in model.graph.initializers.values()
        if initializer.const_value is None
    }
    real_fold = qwen4_weights.fold_initializers_after_weights
    fold_calls = 0

    def fail_second_fold(model):
        nonlocal fold_calls
        fold_calls += 1
        if fold_calls == 2:
            raise RuntimeError("synthetic fold failure")
        real_fold(model)

    monkeypatch.setattr(
        qwen4_weights,
        "fold_initializers_after_weights",
        fail_second_fold,
    )
    with pytest.raises(RuntimeError, match="synthetic fold failure"):
        stream_qwen4_exp_safetensors_to_package(
            fold_failed_package,
            str(tmp_path),
            config,
        )
    assert all(fold_failed_package[name] is model for name, model in original_models.items())
    unset_after_fold = {
        (model_name, initializer.name)
        for model_name, model in fold_failed_package.items()
        for initializer in model.graph.initializers.values()
        if initializer.const_value is None
    }
    assert unset_after_fold == unset_before_fold


@pytest.mark.parametrize(
    "vlm,selected",
    [
        (False, {"main"}),
        (False, {"ngram"}),
        (False, {"main", "ngram"}),
        (True, {"main"}),
        (True, {"ngram"}),
        (True, {"embedding"}),
        (True, {"vision_encoder"}),
        (True, {"main", "ngram", "embedding", "vision_encoder"}),
    ],
)
def test_split_selected_payloads_and_source_values(tmp_path, monkeypatch, vlm, selected):
    config = (_vl_config if vlm else _config)()
    task = (Qwen4ExpVisionLanguageTask if vlm else Qwen4ExpCausalLMTask)(
        split_ngram=True, requested_components=selected
    )
    module = Qwen4ExpForConditionalGeneration(config) if vlm else Qwen4ExpCausalLMModel(config)
    package = build_from_module(module, config, task, execution_provider="onnx-standard")
    manifest = task.component_manifest(hf_config=config)
    source = _official_package_state(package, config)
    # Equal shapes must not conceal a swapped source. Each parameter has unique
    # element values, not merely a different count or dtype.
    for index, (name, value) in enumerate(source.items()):
        if value.is_floating_point():
            source[name] = (
                torch.arange(value.numel(), dtype=torch.float32).reshape(value.shape) * 0.00001
                + index * 0.01
            )
    needed = {
        name: value
        for name, value in source.items()
        if any(manifest[key].owns_source(name) for key in selected)
    }
    # Truly absent unselected dependencies must not prevent a partial build.
    checkpoint = tmp_path / "model.safetensors"
    # Header indexing may inspect these, but no payload may be requested.
    unrelated = {
        "unselected.component.sentinel": torch.tensor([991.0]),
    }
    unrelated.update(
        {
            name: torch.zeros_like(value)
            for name, value in source.items()
            if name not in needed and not value.is_floating_point()
        }
    )
    safetensors.torch.save_file(needed | unrelated, checkpoint)
    original_bytes = checkpoint.read_bytes()
    accessed = set()
    real_open = qwen4_weights.safe_open

    class SelectedReader:
        def __init__(self, *args, **kwargs):
            self.handle = real_open(*args, **kwargs)

        def __enter__(self):
            self.handle.__enter__()
            return self

        def __exit__(self, *args):
            return self.handle.__exit__(*args)

        def get_tensor(self, name):
            assert name in needed, f"Unselected payload {name}"
            accessed.add(name)
            return self.handle.get_tensor(name)

        def get_slice(self, name):
            assert name in needed, f"Unselected payload {name}"
            accessed.add(name)
            return self.handle.get_slice(name)

    monkeypatch.setattr(qwen4_weights, "safe_open", SelectedReader)
    original_index = qwen4_weights._shard_key_index
    indexed = []

    def count_index(paths):
        indexed.append(paths)
        return original_index(paths)

    monkeypatch.setattr(qwen4_weights, "_shard_key_index", count_index)
    # Test the plan's exact source values before folding rewrites names/shapes.
    real_fold = qwen4_weights.fold_initializers_after_weights
    checked = set()

    def check_and_fold(model):
        for name, value in model.graph.initializers.items():
            if isinstance(value.const_value, ir.LazyTensor):
                expected = needed[_source_name(name)]
                torch.testing.assert_close(
                    torch.from_numpy(value.const_value.numpy().copy()),
                    expected,
                    rtol=0,
                    atol=0,
                )
                checked.add(_source_name(name))
        real_fold(model)

    monkeypatch.setattr(qwen4_weights, "fold_initializers_after_weights", check_and_fold)
    report = stream_qwen4_exp_safetensors_to_package(
        package, str(tmp_path), config, manifest=manifest
    )
    assert len(indexed) == 1
    assert report.retained_source_tensor_count == 0
    assert report.model_count == len(selected)
    assert checked
    assert accessed <= needed.keys()
    assert checkpoint.read_bytes() == original_bytes
    assert all(
        value.const_value is not None
        for model in package.values()
        for value in model.graph.initializers.values()
    )
    from mobius._model_package import ModelPackage

    package.save(str(tmp_path / "roundtrip"), progress_bar=False)
    restored = ModelPackage.load(str(tmp_path / "roundtrip"))
    assert restored.keys() == package.keys()
    if len(package) > 1:
        singleton = next(iter(package))
        package.save(
            str(tmp_path / "filtered"),
            components=lambda key: key == singleton,
            progress_bar=False,
        )
        assert set(ModelPackage.load(str(tmp_path / "filtered"))) == {singleton}
    assert checkpoint.read_bytes() == original_bytes


@pytest.mark.parametrize(
    "failure",
    ["missing", "missing_buffer", "shape", "buffer", "buffer_dtype", "fold", "standard"],
)
def test_split_binding_rolls_back_all_model_and_source_identities(
    tmp_path, monkeypatch, failure
):
    config = _vl_config()
    task = Qwen4ExpVisionLanguageTask(split_ngram=True)
    package = build_from_module(
        Qwen4ExpForConditionalGeneration(config),
        config,
        task,
        execution_provider="onnx-standard",
    )
    source = _official_package_state(package, config)
    target = "model.visual.patch_embed.proj.weight"
    if failure == "missing":
        del source[target]
    elif failure == "missing_buffer":
        del source[next(name for name in source if name.endswith("layer_multipliers"))]
    elif failure == "shape":
        source[target] = torch.zeros(1)
    elif failure == "buffer":
        name = next(name for name in source if name.endswith("layer_multipliers"))
        source[name] = torch.zeros_like(source[name])
    elif failure == "buffer_dtype":
        name = next(name for name in source if name.endswith("layer_multipliers"))
        source[name] = source[name].to(torch.int32)
    path = tmp_path / "model.safetensors"
    safetensors.torch.save_file(source, path)
    before_bytes = path.read_bytes()
    before_models = dict(package)
    before_initializers = {
        (key, name): (value, value.const_value)
        for key, model in package.items()
        for name, value in model.graph.initializers.items()
    }
    if failure in {"fold", "standard"}:
        method = (
            "fold_initializers_after_weights"
            if failure == "fold"
            else "validate_standard_onnx"
        )
        original = getattr(qwen4_weights, method)
        calls = []

        def fail_later(*args, **kwargs):
            calls.append(1)
            if len(calls) == 2:
                raise ValueError("synthetic later component failure")
            return original(*args, **kwargs)

        monkeypatch.setattr(qwen4_weights, method, fail_later)
    with pytest.raises(ValueError):
        stream_qwen4_exp_safetensors_to_package(
            package, str(tmp_path), config, manifest=task.component_manifest(hf_config=config)
        )
    assert all(package[key] is model for key, model in before_models.items())
    for (key, name), (value, tensor) in before_initializers.items():
        assert package[key].graph.initializers[name] is value
        assert value.const_value is tensor
    assert path.read_bytes() == before_bytes


def test_split_shared_canonical_source_can_belong_to_omitted_component(tmp_path):
    config = _vl_config(tie_word_embeddings=True)
    task = Qwen4ExpVisionLanguageTask(split_ngram=True, requested_components={"main"})
    package = build_from_module(
        Qwen4ExpForConditionalGeneration(config),
        config,
        task,
        execution_provider="onnx-standard",
    )
    manifest = task.component_manifest(hf_config=config)
    assert len(manifest.shared_weights) == 1
    assert manifest.shared_weights[0].canonical.component == "embedding"
    assert all(
        endpoint.component != "ngram" for endpoint in manifest.shared_weights[0].endpoints
    )
    source = _official_package_state(package, config)
    source = {
        name: value for name, value in source.items() if manifest["main"].owns_source(name)
    }
    canonical = "model.language_model.embed_tokens.weight"
    source[canonical] = source.pop("lm_head.weight")
    safetensors.torch.save_file(source, tmp_path / "model.safetensors")
    stream_qwen4_exp_safetensors_to_package(package, str(tmp_path), config, manifest=manifest)
    head = next(
        value
        for name, value in package["main"].graph.initializers.items()
        if "lm_head" in name
    )
    actual = torch.from_numpy(head.const_value.numpy().copy())
    expected = source[canonical]
    if actual.shape != expected.shape:
        expected = expected.T
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_split_text_proven_tie_resolves_missing_head_from_main_token_table(tmp_path):
    config = _config(tie_word_embeddings=True)
    task = Qwen4ExpCausalLMTask(split_ngram=True, requested_components={"main"})
    package = build_from_module(
        Qwen4ExpCausalLMModel(config), config, task, execution_provider="onnx-standard"
    )
    manifest = task.component_manifest(hf_config=config)
    assert manifest.shared_weights[0].canonical.component == "main"
    source = _official_package_state(package, config)
    del source["lm_head.weight"]
    safetensors.torch.save_file(source, tmp_path / "model.safetensors")
    stream_qwen4_exp_safetensors_to_package(package, str(tmp_path), config, manifest=manifest)
    head = next(
        value
        for name, value in package["main"].graph.initializers.items()
        if "lm_head" in name
    )
    torch.testing.assert_close(
        torch.from_numpy(head.const_value.numpy().copy()),
        source["model.language_model.embed_tokens.weight"].T,
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("selected", [{"main"}, {"ngram"}])
def test_split_bfloat16_sources_roundtrip_without_changing_bits(tmp_path, selected):
    import numpy as np

    config = _config(dtype=ir.DataType.BFLOAT16)
    task = Qwen4ExpCausalLMTask(split_ngram=True, requested_components=selected)
    package = build_from_module(
        Qwen4ExpCausalLMModel(config), config, task, execution_provider="onnx-standard"
    )
    source = _official_package_state(package, config)
    for name, value in source.items():
        if value.is_floating_point():
            source[name] = value.to(torch.bfloat16)
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    safetensors.torch.save_file(source, source_dir / "model.safetensors")
    stream_qwen4_exp_safetensors_to_package(
        package, str(source_dir), config, manifest=task.component_manifest(hf_config=config)
    )
    package.save(str(tmp_path / "saved"), progress_bar=False)
    from mobius._model_package import ModelPackage

    loaded = ModelPackage.load(str(tmp_path / "saved"))
    key = next(iter(selected))
    name = (
        "model.embed_tokens.weight"
        if key == "main"
        else "model.layers.0.ple.ple_embedding.ngram_embedding.shard_0.weight"
    )
    value = loaded[key].graph.initializers[name].const_value
    assert value.dtype == ir.DataType.BFLOAT16
    expected = source[_source_name(name)].view(torch.uint16).numpy()
    np.testing.assert_array_equal(value.numpy().view(np.uint16), expected, strict=True)


@pytest.mark.parametrize("quantized", [False, True])
def test_split_rejects_unsupported_checkpoint_payload_layout(tmp_path, quantized):
    config = _config()
    task = Qwen4ExpCausalLMTask(split_ngram=True, requested_components={"main"})
    package = build_from_module(
        Qwen4ExpCausalLMModel(config), config, task, execution_provider="onnx-standard"
    )
    source = _official_package_state(package, config)
    if quantized:
        source["model.language_model.layers.0.weight_scale"] = torch.ones(1)
        message = "unquantized"
    else:
        source["model.language_model.embed_tokens.weight"] = source[
            "model.language_model.embed_tokens.weight"
        ].to(torch.int8)
        message = "Unsupported split source dtype"
    safetensors.torch.save_file(source, tmp_path / "model.safetensors")
    original = dict(package)
    with pytest.raises(ValueError, match=message):
        stream_qwen4_exp_safetensors_to_package(
            package, str(tmp_path), config, manifest=task.component_manifest(hf_config=config)
        )
    assert all(package[key] is model for key, model in original.items())
