# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for the Transformers integration builder."""

from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import onnx_ir as ir
import pytest
from onnxscript import nn

from mobius._configs import QuantizationConfig, QuantizationOverride
from mobius._model_package import ModelPackage
from mobius._testing import make_config
from mobius.integrations._block_quant import BlockQuantScheme
from mobius.integrations.diffusers import _builder as diffusers_builder
from mobius.integrations.transformers import _builder as transformers_builder
from mobius.integrations.transformers import _config_resolver


class _DummyModule(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config


class _DummyFp8Module(_DummyModule):
    def build_fp8_streaming_plan(self, *_args):
        raise AssertionError("mock streaming function should own planning")


@pytest.mark.parametrize(
    ("keep_quantized", "expected_loader"),
    [(True, "qdq"), (False, "dense")],
)
def test_qwen38_fp8_selects_storage_or_dense_loader(
    monkeypatch,
    keep_quantized,
    expected_loader,
) -> None:
    text_config = SimpleNamespace(model_type="qwen4_exp_text")
    expected_parent = SimpleNamespace(
        model_type="qwen4_exp",
        architectures=["Qwen4ExpForConditionalGeneration"],
        text_config=text_config,
        quantization_config={
            "quant_method": "fp8",
            "weight_block_size": [128, 128],
            "activation_scheme": "dynamic",
        },
        vision_config=object(),
        image_token_id=248056,
        video_token_id=248057,
        vision_start_token_id=248053,
        vision_end_token_id=248054,
    )
    model = ir.Model(ir.Graph([], [], nodes=[], name="model"), ir_version=11)
    package = ModelPackage({"model": model})
    config_calls = []
    loader_calls = []
    built_configs = []

    def load_config(model_id, **kwargs):
        config_calls.append((model_id, kwargs))
        return expected_parent, False

    monkeypatch.setattr(
        transformers_builder,
        "_load_transformers_config",
        load_config,
    )
    monkeypatch.setattr(
        transformers_builder,
        "_resolve_module_class",
        lambda *args, **kwargs: (
            _DummyFp8Module,
            "qwen4-exp-text-generation",
            "qwen4_exp_text",
        ),
    )

    def resolve_config(primary, *, parent_config, module_class):
        assert primary is text_config
        assert parent_config is expected_parent
        assert not hasattr(primary, "quantization_config")
        scheme = BlockQuantScheme.from_quantization_config(parent_config.quantization_config)
        assert scheme is not None
        return make_config(
            model_type="qwen4_exp",
            block_quant_scheme=scheme,
            vision=object(),
            image_token_id=parent_config.image_token_id,
            video_token_id=parent_config.video_token_id,
            vision_start_token_id=parent_config.vision_start_token_id,
            vision_end_token_id=parent_config.vision_end_token_id,
            deepstack_visual_indexes=[],
        )

    monkeypatch.setattr(_config_resolver, "_config_from_hf", resolve_config)

    def build_module(_module, config, *args, **kwargs):
        built_configs.append(config)
        package.config = config
        return package

    monkeypatch.setattr(transformers_builder, "build_from_module", build_module)

    def qdq(*args, **kwargs):
        loader_calls.append(("qdq", args, kwargs))
        return {"format": "mobius.weight-loading-report.v1"}

    def dense(*args, **kwargs):
        loader_calls.append(("dense", args, kwargs))
        return {"format": "mobius.weight-loading-report.v1"}

    monkeypatch.setattr(transformers_builder, "stream_qdq_safetensors_to_model", qdq)
    monkeypatch.setattr(
        transformers_builder,
        "stream_preprocessed_safetensors_to_model",
        dense,
    )

    result = transformers_builder.build_transformers_model(
        "unsloth/Qwen3.8-Flash-Next-FP8",
        revision="feature/revision",
        keep_quantized=keep_quantized,
        text_only=True,
    )

    assert result is package
    assert config_calls == [
        (
            "unsloth/Qwen3.8-Flash-Next-FP8",
            {"revision": "feature/revision", "trust_remote_code": False},
        )
    ]
    assert len(loader_calls) == 1
    loader_name, loader_args, loader_kwargs = loader_calls[0]
    assert loader_name == expected_loader
    assert loader_args[1] == "unsloth/Qwen3.8-Flash-Next-FP8"
    assert loader_kwargs["revision"] == "feature/revision"
    assert built_configs[0].block_quant_scheme is not None
    assert built_configs[0].model_type == "qwen4_exp_text"
    assert built_configs[0].vision is None
    assert built_configs[0].image_token_id is None
    assert built_configs[0].video_token_id is None
    assert built_configs[0].vision_start_token_id is None
    assert built_configs[0].vision_end_token_id is None
    assert getattr(built_configs[0], "unsupported_video_token_id", None) is None
    assert built_configs[0].deepstack_visual_indexes is None


def test_qwen38_multimodal_config_keeps_parent_fields(monkeypatch) -> None:
    text_config = SimpleNamespace(model_type="qwen4_exp_text")
    vision = object()
    parent = SimpleNamespace(
        model_type="qwen4_exp",
        architectures=["Qwen4ExpForConditionalGeneration"],
        text_config=text_config,
        quantization_config={
            "quant_method": "fp8",
            "weight_block_size": [128, 128],
        },
        vision_config=vision,
        image_token_id=248056,
    )
    built_configs = []
    package = ModelPackage(
        {"decoder": ir.Model(ir.Graph([], [], nodes=[], name="decoder"), ir_version=11)}
    )
    monkeypatch.setattr(
        transformers_builder,
        "_load_transformers_config",
        lambda *args, **kwargs: (parent, False),
    )
    monkeypatch.setattr(
        transformers_builder,
        "_resolve_module_class",
        lambda *args, **kwargs: (_DummyModule, "qwen4-exp-vision-language", "qwen4_exp"),
    )

    def resolve_config(primary, *, parent_config, module_class):
        assert primary is text_config
        assert parent_config is parent
        return make_config(
            model_type="qwen4_exp",
            vision=vision,
            image_token_id=parent.image_token_id,
        )

    monkeypatch.setattr(_config_resolver, "_config_from_hf", resolve_config)

    def build_module(_module, config, *args, **kwargs):
        built_configs.append(config)
        return package

    monkeypatch.setattr(transformers_builder, "build_from_module", build_module)

    transformers_builder.build_transformers_model(
        "fake/qwen4-exp",
        load_weights=False,
        text_only=False,
    )

    assert built_configs[0].model_type == "qwen4_exp"
    assert built_configs[0].vision is vision
    assert built_configs[0].image_token_id == 248056


def test_qwen38_fp8_none_revision_uses_hugging_face_default(monkeypatch) -> None:
    calls = []

    def stop_after_config(model_id, **kwargs):
        calls.append((model_id, kwargs))
        raise RuntimeError("stop after revision assertion")

    monkeypatch.setattr(
        transformers_builder,
        "_load_transformers_config",
        stop_after_config,
    )

    with pytest.raises(RuntimeError, match="stop after revision assertion"):
        transformers_builder.build_transformers_model(
            "unsloth/Qwen3.8-Flash-Next-FP8",
            load_weights=False,
        )

    assert calls == [
        (
            "unsloth/Qwen3.8-Flash-Next-FP8",
            {
                "revision": None,
                "trust_remote_code": False,
            },
        )
    ]


def test_vibevoice_none_revision_pins_first_config_probe(monkeypatch) -> None:
    from mobius.models.vibevoice import VIBEVOICE_MODEL_ID, VIBEVOICE_REVISION

    calls = []

    def stop_after_config(model_id, **kwargs):
        calls.append((model_id, kwargs))
        raise RuntimeError("stop after revision assertion")

    monkeypatch.setattr(
        transformers_builder,
        "_load_transformers_config",
        stop_after_config,
    )

    with pytest.raises(RuntimeError, match="stop after revision assertion"):
        transformers_builder.build_transformers_model(
            VIBEVOICE_MODEL_ID,
            load_weights=False,
        )

    assert calls == [
        (
            VIBEVOICE_MODEL_ID,
            {
                "revision": VIBEVOICE_REVISION,
                "trust_remote_code": False,
            },
        )
    ]


def test_vibevoice_streaming_none_revision_pins_first_config_probe(monkeypatch) -> None:
    from mobius.models.vibevoice_streaming import (
        VIBEVOICE_STREAMING_MODEL_ID,
        VIBEVOICE_STREAMING_REVISION,
    )

    calls = []

    def stop_after_config(model_id, **kwargs):
        calls.append((model_id, kwargs))
        raise RuntimeError("stop after revision assertion")

    monkeypatch.setattr(
        transformers_builder,
        "_load_transformers_config",
        stop_after_config,
    )

    with pytest.raises(RuntimeError, match="stop after revision assertion"):
        transformers_builder.build_transformers_model(
            VIBEVOICE_STREAMING_MODEL_ID,
            load_weights=False,
        )

    assert calls == [
        (
            VIBEVOICE_STREAMING_MODEL_ID,
            {
                "revision": VIBEVOICE_STREAMING_REVISION,
                "trust_remote_code": False,
            },
        )
    ]


@pytest.mark.parametrize("revision", [None, "feature/revision"])
def test_transformers_config_forwards_only_explicit_revision(monkeypatch, revision) -> None:
    import transformers

    calls = []
    config = SimpleNamespace(model_type="qwen2")

    def from_pretrained(model_id, **kwargs):
        calls.append((model_id, kwargs))
        return config

    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", from_pretrained)

    result = transformers_builder._load_transformers_config(
        "unsloth/Qwen3.8-Flash-Next-FP8",
        revision=revision,
        trust_remote_code=False,
    )

    assert result == (config, False)
    expected_kwargs = {"trust_remote_code": False}
    if revision is not None:
        expected_kwargs["revision"] = revision
    assert calls == [("unsloth/Qwen3.8-Flash-Next-FP8", expected_kwargs)]


def test_strip_to_text_only_drops_component_quantization() -> None:
    decoder = QuantizationConfig(
        bits=4,
        group_size=16,
        quant_method="olive",
    )
    config = make_config(
        component_quantization={
            "decoder": decoder,
            "vision_encoder": decoder,
        }
    )

    stripped = transformers_builder._strip_to_text_only(config, "qwen2")

    assert stripped.component_quantization is None
    assert stripped.quantization is decoder


def test_strip_to_text_only_resolves_decoder_module_plan() -> None:
    decoder = QuantizationConfig(
        bits=4,
        group_size=16,
        quant_method="olive",
        overrides={"model.language_model": QuantizationOverride(bits=8, group_size=32)},
    )
    config = make_config(
        quantization=decoder,
        component_quantization={"decoder": decoder},
    )

    stripped = transformers_builder._strip_to_text_only(
        config,
        "qwen2",
        decoder_source_paths=(
            "model.language_model.layers",
            "model.language_model.norm",
        ),
    )

    assert stripped.component_quantization is None
    assert stripped.quantization is not None
    assert (stripped.quantization.bits, stripped.quantization.group_size) == (8, 32)
    assert stripped.quantization.overrides == {}


def test_transformers_build_uses_canonical_weight_loader(monkeypatch) -> None:
    hf_config = type("HFConfig", (), {"model_type": "qwen2"})()
    config = make_config(model_type="qwen2")
    model = ir.Model(ir.Graph([], [], nodes=[], name="model"), ir_version=11)
    package = ModelPackage({"model": model}, config=config)
    download = mock.Mock(return_value={})

    monkeypatch.setattr(
        transformers_builder,
        "_load_transformers_config",
        lambda *args, **kwargs: (hf_config, False),
    )
    monkeypatch.setattr(
        transformers_builder,
        "_select_primary_config",
        lambda value: (value, value, "qwen2"),
    )
    monkeypatch.setattr(
        transformers_builder,
        "_resolve_module_class",
        lambda *args, **kwargs: (_DummyModule, "text-generation", "qwen2"),
    )
    monkeypatch.setattr(_config_resolver, "_config_from_hf", lambda *args, **kwargs: config)
    monkeypatch.setattr(
        transformers_builder, "build_from_module", lambda *args, **kwargs: package
    )
    monkeypatch.setattr(transformers_builder, "_download_weights", download)

    result = transformers_builder.build_transformers_model("fake/model")

    assert result is package
    download.assert_called_once_with("fake/model", revision=None)


def test_text_only_resolution_ignores_multimodal_parent_architecture() -> None:
    from mobius.models import Qwen4ExpCausalLMModel

    parent = type(
        "Qwen4ExpParent",
        (),
        {"architectures": ["Qwen4ExpForConditionalGeneration"]},
    )()
    module_class, task, model_type = transformers_builder._resolve_module_class(
        "qwen4_exp_text",
        parent,
        None,
        None,
        allow_parent_architecture_override=False,
    )
    assert module_class is Qwen4ExpCausalLMModel
    assert task is None
    assert model_type == "qwen4_exp_text"


def test_qwen4_text_only_build_inherits_parent_metadata_then_strips_multimodal(
    monkeypatch,
) -> None:
    text_config = type("TextConfig", (), {"model_type": "qwen4_exp_text"})()
    parent_config = type(
        "Qwen4ExpParent",
        (),
        {
            "model_type": "qwen4_exp",
            "architectures": ["Qwen4ExpForConditionalGeneration"],
            "text_config": text_config,
            "vision_config": object(),
            "quantization_config": {
                "quant_method": "fp8",
                "weight_block_size": [128, 128],
            },
        },
    )()
    config = make_config(
        model_type="qwen4_exp",
        vision=object(),
        image_token_id=248056,
        block_quant_scheme=BlockQuantScheme.from_quantization_config(
            parent_config.quantization_config
        ),
    )
    model = ir.Model(ir.Graph([], [], nodes=[], name="model"), ir_version=11)

    monkeypatch.setattr(
        transformers_builder,
        "_load_transformers_config",
        lambda *args, **kwargs: (parent_config, False),
    )
    monkeypatch.setattr(
        transformers_builder,
        "_select_primary_config",
        lambda value: (text_config, parent_config, "qwen4_exp"),
    )
    monkeypatch.setattr(
        transformers_builder,
        "_resolve_module_class",
        lambda *args, **kwargs: (_DummyModule, "text-generation", "qwen4_exp_text"),
    )

    def config_from_hf(_config, *, parent_config, module_class):
        assert parent_config is globals_parent
        assert module_class is _DummyModule
        return config

    globals_parent = parent_config
    monkeypatch.setattr(_config_resolver, "_config_from_hf", config_from_hf)
    monkeypatch.setattr(
        transformers_builder,
        "build_from_module",
        lambda _module, built_config, *args, **kwargs: ModelPackage(
            {"model": model},
            config=built_config,
        ),
    )

    package = transformers_builder.build_transformers_model(
        "Qwen/Qwen3.8-Flash-Next",
        text_only=True,
        load_weights=False,
    )
    assert package.config.model_type == "qwen4_exp_text"
    assert package.config.block_quant_scheme is not None
    assert package.config.vision is None
    assert package.config.image_token_id is None


def test_qwen4_multimodal_build_streams_entire_package_without_eager_loader(
    monkeypatch,
) -> None:
    text_config = type("TextConfig", (), {"model_type": "qwen4_exp_text"})()
    parent_config = type(
        "Qwen4ExpParent",
        (),
        {
            "model_type": "qwen4_exp",
            "architectures": ["Qwen4ExpForConditionalGeneration"],
            "text_config": text_config,
            "vision_config": object(),
            "quantization_config": None,
        },
    )()
    config = make_config(model_type="qwen4_exp")
    package = ModelPackage(
        {
            name: ir.Model(ir.Graph([], [], nodes=[], name=name), ir_version=11)
            for name in ("decoder", "vision_encoder", "embedding")
        },
        config=config,
    )
    monkeypatch.setattr(
        transformers_builder,
        "_load_transformers_config",
        lambda *args, **kwargs: (parent_config, False),
    )
    monkeypatch.setattr(
        transformers_builder,
        "_select_primary_config",
        lambda value: (text_config, parent_config, "qwen4_exp"),
    )
    monkeypatch.setattr(
        transformers_builder,
        "_resolve_module_class",
        lambda *args, **kwargs: (
            _DummyModule,
            "qwen4-exp-vision-language",
            "qwen4_exp",
        ),
    )
    monkeypatch.setattr(_config_resolver, "_config_from_hf", lambda *args, **kwargs: config)
    monkeypatch.setattr(
        transformers_builder,
        "build_from_module",
        lambda *args, **kwargs: package,
    )
    eager = mock.Mock(side_effect=AssertionError("must not eagerly download"))
    monkeypatch.setattr(transformers_builder, "_download_weights", eager)

    with mock.patch(
        "mobius.integrations.transformers._qwen4_exp_weights."
        "stream_qwen4_exp_safetensors_to_package"
    ) as stream:
        result = transformers_builder.build_transformers_model(
            "Qwen/Qwen3.8-Flash-Next",
            revision="immutable",
        )

    assert result is package
    eager.assert_not_called()
    stream.assert_called_once_with(
        package,
        "Qwen/Qwen3.8-Flash-Next",
        config,
        revision="immutable",
    )
    assert {model.metadata_props["mobius.source_revision"] for model in package.values()} == {
        "immutable"
    }


def test_qwen4_affine_component_plan_fails_before_weight_loading(
    monkeypatch,
) -> None:
    text_config = type("TextConfig", (), {"model_type": "qwen4_exp_text"})()
    parent_config = type(
        "Qwen4ExpParent",
        (),
        {
            "model_type": "qwen4_exp",
            "architectures": ["Qwen4ExpForConditionalGeneration"],
            "text_config": text_config,
            "vision_config": object(),
            "quantization_config": None,
        },
    )()
    decoder_quantization = QuantizationConfig(
        bits=4,
        group_size=32,
        quant_method="olive",
    )
    config = make_config(
        model_type="qwen4_exp",
        quantization=decoder_quantization,
        component_quantization={"decoder": decoder_quantization},
    )
    package = ModelPackage(
        {
            name: ir.Model(ir.Graph([], [], nodes=[], name=name), ir_version=11)
            for name in ("decoder", "vision_encoder", "embedding")
        },
        config=config,
    )
    monkeypatch.setattr(
        transformers_builder,
        "_load_transformers_config",
        lambda *args, **kwargs: (parent_config, False),
    )
    monkeypatch.setattr(
        transformers_builder,
        "_select_primary_config",
        lambda value: (text_config, parent_config, "qwen4_exp"),
    )
    monkeypatch.setattr(
        transformers_builder,
        "_resolve_module_class",
        lambda *args, **kwargs: (
            _DummyModule,
            "qwen4-exp-vision-language",
            "qwen4_exp",
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
    download = mock.Mock()
    monkeypatch.setattr(transformers_builder, "_download_weights", download)

    with (
        mock.patch(
            "mobius.integrations.transformers._qwen4_exp_weights."
            "stream_qwen4_exp_safetensors_to_package"
        ) as stream,
        pytest.raises(NotImplementedError, match="packed expert"),
    ):
        transformers_builder.build_transformers_model(
            "Qwen/Qwen3.8-Flash-Next",
        )

    stream.assert_not_called()
    download.assert_not_called()


def test_transformers_build_routes_compressed_tensors_to_streaming_loader(
    monkeypatch,
) -> None:
    quantization_config = {
        "quant_method": "compressed-tensors",
        "version": "0.17.2",
        "format": "mixed-precision",
        "quantization_status": "compressed",
        "config_groups": {
            "group_0": {
                "format": "float-quantized",
                "targets": ["fp8"],
                "weights": {
                    "num_bits": 8,
                    "type": "float",
                    "strategy": "channel",
                    "symmetric": True,
                    "dynamic": False,
                    "group_size": None,
                    "scale_dtype": None,
                },
                "input_activations": {
                    "num_bits": 8,
                    "type": "float",
                    "strategy": "token",
                    "symmetric": True,
                    "dynamic": True,
                    "group_size": None,
                    "scale_dtype": None,
                },
            },
            "group_1": {
                "format": "nvfp4-pack-quantized",
                "targets": ["nvfp4"],
                "weights": {
                    "num_bits": 4,
                    "type": "float",
                    "strategy": "tensor_group",
                    "symmetric": True,
                    "dynamic": False,
                    "group_size": 16,
                    "scale_dtype": "torch.float8_e4m3fn",
                },
                "input_activations": {
                    "num_bits": 4,
                    "type": "float",
                    "strategy": "tensor_group",
                    "symmetric": True,
                    "dynamic": "local",
                    "group_size": 16,
                    "scale_dtype": "torch.float8_e4m3fn",
                },
            },
        },
        "ignore": [],
    }
    hf_config = type(
        "HFConfig",
        (),
        {"model_type": "qwen2", "quantization_config": quantization_config},
    )()
    config = make_config(model_type="qwen2")
    model = ir.Model(ir.Graph([], [], nodes=[], name="model"), ir_version=11)
    package = ModelPackage({"model": model}, config=config)
    stream = mock.Mock()
    built_configs = []

    monkeypatch.setattr(
        transformers_builder,
        "_load_transformers_config",
        lambda *args, **kwargs: (hf_config, False),
    )
    monkeypatch.setattr(
        transformers_builder,
        "_select_primary_config",
        lambda value: (value, value, "qwen2"),
    )
    monkeypatch.setattr(
        transformers_builder,
        "_resolve_module_class",
        lambda *args, **kwargs: (_DummyModule, "text-generation", "qwen2"),
    )
    monkeypatch.setattr(_config_resolver, "_config_from_hf", lambda *args, **kwargs: config)

    def fake_build_from_module(_module, built_config, *args, **kwargs):
        built_configs.append(built_config)
        return package

    monkeypatch.setattr(
        transformers_builder,
        "build_from_module",
        fake_build_from_module,
    )
    monkeypatch.setattr(transformers_builder, "stream_compressed_tensors_to_package", stream)
    download = mock.Mock(side_effect=AssertionError("must not eagerly download"))
    monkeypatch.setattr(transformers_builder, "_download_weights", download)

    result = transformers_builder.build_transformers_model("fake/model", revision="immutable")

    assert result is package
    download.assert_not_called()
    stream.assert_called_once()
    assert stream.call_args.kwargs["revision"] == "immutable"
    assert stream.call_args.kwargs["keep_quantized"] is True
    assert built_configs[0].dtype == ir.DataType.FLOAT16


def test_transformers_build_can_explicitly_dequantize_compressed_tensors(
    monkeypatch,
) -> None:
    quantization_config = {
        "quant_method": "compressed-tensors",
        "version": "0.17.2",
        "format": "mixed-precision",
        "quantization_status": "compressed",
        "config_groups": {
            "fp8": {
                "format": "float-quantized",
                "targets": ["fp8"],
                "weights": {
                    "num_bits": 8,
                    "type": "float",
                    "strategy": "channel",
                    "symmetric": True,
                    "dynamic": False,
                },
                "input_activations": {
                    "num_bits": 8,
                    "type": "float",
                    "strategy": "token",
                    "symmetric": True,
                    "dynamic": True,
                },
            },
            "nvfp4": {
                "format": "nvfp4-pack-quantized",
                "targets": ["nvfp4"],
                "weights": {
                    "num_bits": 4,
                    "type": "float",
                    "strategy": "tensor_group",
                    "symmetric": True,
                    "dynamic": False,
                    "group_size": 16,
                    "scale_dtype": "torch.float8_e4m3fn",
                },
                "input_activations": {
                    "num_bits": 4,
                    "type": "float",
                    "strategy": "tensor_group",
                    "symmetric": True,
                    "dynamic": "local",
                    "group_size": 16,
                    "scale_dtype": "torch.float8_e4m3fn",
                },
            },
        },
        "ignore": [],
    }
    hf_config = type(
        "HFConfig",
        (),
        {"model_type": "qwen2", "quantization_config": quantization_config},
    )()
    config = make_config(model_type="qwen2")
    package = ModelPackage(
        {"model": ir.Model(ir.Graph([], [], nodes=[], name="model"), ir_version=11)},
        config=config,
    )
    stream = mock.Mock()
    monkeypatch.setattr(
        transformers_builder,
        "_load_transformers_config",
        lambda *args, **kwargs: (hf_config, False),
    )
    monkeypatch.setattr(
        transformers_builder,
        "_select_primary_config",
        lambda value: (value, value, "qwen2"),
    )
    monkeypatch.setattr(
        transformers_builder,
        "_resolve_module_class",
        lambda *args, **kwargs: (_DummyModule, "text-generation", "qwen2"),
    )
    monkeypatch.setattr(_config_resolver, "_config_from_hf", lambda *args, **kwargs: config)
    monkeypatch.setattr(
        transformers_builder, "build_from_module", lambda *args, **kwargs: package
    )
    monkeypatch.setattr(transformers_builder, "stream_compressed_tensors_to_package", stream)

    transformers_builder.build_transformers_model(
        "fake/model",
        keep_quantized=False,
    )

    assert stream.call_args.kwargs["keep_quantized"] is False


def test_compressed_checkpoint_fp8_kv_cache_requires_checkpoint_scales(
    monkeypatch,
) -> None:
    hf_config = type("HFConfig", (), {"model_type": "qwen2"})()
    compressed = type("Compressed", (), {"kv_cache_scheme": object()})()
    config = make_config(
        model_type="qwen2",
        num_hidden_layers=2,
        layer_types=["linear_attention", "full_attention"],
    )
    monkeypatch.setattr(
        transformers_builder,
        "_load_transformers_config",
        lambda *args, **kwargs: (hf_config, False),
    )
    monkeypatch.setattr(
        transformers_builder,
        "_select_primary_config",
        lambda value: (value, value, "qwen2"),
    )
    monkeypatch.setattr(
        transformers_builder.CompressedTensorsConfig,
        "from_hf_config",
        lambda value: compressed,
    )
    monkeypatch.setattr(
        transformers_builder,
        "_resolve_module_class",
        lambda *args, **kwargs: (_DummyModule, "text-generation", "qwen2"),
    )
    monkeypatch.setattr(_config_resolver, "_config_from_hf", lambda *args, **kwargs: config)

    with pytest.raises(ValueError, match=r"complete per-layer.*Missing layers: \[1\]"):
        transformers_builder.build_transformers_model(
            "fake/model",
            fp8_kv_cache=True,
        )


def test_compressed_checkpoint_fp8_kv_cache_rejects_partial_scale_map(
    monkeypatch,
) -> None:
    hf_config = type("HFConfig", (), {"model_type": "qwen2"})()
    compressed = type("Compressed", (), {"kv_cache_scheme": object()})()
    config = make_config(
        model_type="qwen2",
        num_hidden_layers=3,
        layer_types=["full_attention", "linear_attention", "full_attention"],
    )
    monkeypatch.setattr(
        transformers_builder,
        "_load_transformers_config",
        lambda *args, **kwargs: (hf_config, False),
    )
    monkeypatch.setattr(
        transformers_builder,
        "_select_primary_config",
        lambda value: (value, value, "qwen2"),
    )
    monkeypatch.setattr(
        transformers_builder.CompressedTensorsConfig,
        "from_hf_config",
        lambda value: compressed,
    )
    monkeypatch.setattr(
        transformers_builder,
        "_resolve_module_class",
        lambda *args, **kwargs: (_DummyModule, "text-generation", "qwen2"),
    )
    monkeypatch.setattr(_config_resolver, "_config_from_hf", lambda *args, **kwargs: config)

    with pytest.raises(ValueError, match=r"Missing layers: \[2\]"):
        transformers_builder.build_transformers_model(
            "fake/model",
            fp8_kv_cache=True,
            kv_cache_scales={0: (1.0, 1.0)},
        )


def test_glm_full_attention_overrides_use_dsa_for_glm_moe_dsa(monkeypatch) -> None:
    """``--glm-full-attention`` forces ``config.use_dsa=False`` for GLM-5.2."""
    hf_config = type("HFConfig", (), {"model_type": "glm_moe_dsa"})()
    config = make_config(model_type="glm_moe_dsa", use_dsa=True)
    model = ir.Model(ir.Graph([], [], nodes=[], name="model"), ir_version=11)
    package = ModelPackage({"model": model}, config=config)
    captured_configs: list = []

    monkeypatch.setattr(
        transformers_builder,
        "_load_transformers_config",
        lambda *args, **kwargs: (hf_config, False),
    )
    monkeypatch.setattr(
        transformers_builder,
        "_select_primary_config",
        lambda value: (value, value, "glm_moe_dsa"),
    )
    monkeypatch.setattr(
        transformers_builder,
        "_resolve_module_class",
        lambda *args, **kwargs: (_DummyModule, "text-generation", "glm_moe_dsa"),
    )
    monkeypatch.setattr(_config_resolver, "_config_from_hf", lambda *args, **kwargs: config)

    def fake_build_from_module(_module, built_config, *args, **kwargs):
        captured_configs.append(built_config)
        return package

    monkeypatch.setattr(transformers_builder, "build_from_module", fake_build_from_module)
    monkeypatch.setattr(transformers_builder, "_download_weights", mock.Mock(return_value={}))

    result = transformers_builder.build_transformers_model(
        "zai-org/GLM-5.2", glm_full_attention=True
    )

    assert result is package
    assert captured_configs[0].use_dsa is False


def test_glm_full_attention_rejects_non_glm_model_type(monkeypatch) -> None:
    """``--glm-full-attention`` is only meaningful for ``glm_moe_dsa``."""
    hf_config = type("HFConfig", (), {"model_type": "qwen2"})()
    config = make_config(model_type="qwen2")

    monkeypatch.setattr(
        transformers_builder,
        "_load_transformers_config",
        lambda *args, **kwargs: (hf_config, False),
    )
    monkeypatch.setattr(
        transformers_builder,
        "_select_primary_config",
        lambda value: (value, value, "qwen2"),
    )
    monkeypatch.setattr(
        transformers_builder,
        "_resolve_module_class",
        lambda *args, **kwargs: (_DummyModule, "text-generation", "qwen2"),
    )
    monkeypatch.setattr(_config_resolver, "_config_from_hf", lambda *args, **kwargs: config)

    with pytest.raises(ValueError, match="glm_full_attention=True is not supported"):
        transformers_builder.build_transformers_model("fake/model", glm_full_attention=True)


def test_build_threads_revision_to_diffusers_fallback(monkeypatch) -> None:
    import transformers

    monkeypatch.setattr(
        transformers.AutoConfig,
        "from_pretrained",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("not transformers")),
    )
    monkeypatch.setattr(
        _config_resolver, "_try_load_config_json", lambda *args, **kwargs: None
    )
    expected = ModelPackage({})
    calls: list[tuple[tuple, dict]] = []

    def fake_build_diffusers(*args, **kwargs):
        calls.append((args, kwargs))
        return expected

    monkeypatch.setattr(
        diffusers_builder,
        "build_diffusers_pipeline",
        fake_build_diffusers,
    )

    result = transformers_builder.build_transformers_model(
        "fake/diffusers",
        revision="pinned-revision",
        load_weights=False,
    )

    assert result is expected
    assert calls == [
        (
            ("fake/diffusers",),
            {
                "revision": "pinned-revision",
                "dtype": None,
                "load_weights": False,
                "execution_provider": "default",
            },
        )
    ]


def test_glm_full_attention_rejects_diffusers_dispatch(monkeypatch) -> None:
    """``--glm-full-attention`` must raise on the diffusers-dispatch branch.

    Mirrors the existing ``text_only`` guard on the same early-return path:
    a repo that doesn't resolve to a registered ``model_type`` (and so falls
    through to the Diffusers integration) can never be GLM-5.2, so silently
    ignoring the flag there would swallow a real user error.
    """
    import transformers

    monkeypatch.setattr(
        transformers.AutoConfig,
        "from_pretrained",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("not transformers")),
    )
    monkeypatch.setattr(
        _config_resolver, "_try_load_config_json", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        diffusers_builder,
        "build_diffusers_pipeline",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("should not reach build_diffusers_pipeline")
        ),
    )

    with pytest.raises(ValueError, match="glm_full_attention=True is not supported"):
        transformers_builder.build_transformers_model(
            "fake/diffusers", glm_full_attention=True
        )
