# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import dataclasses
from unittest.mock import patch

import onnx_ir as ir
import pytest
import torch

from mobius._configs.fish_codec import (
    DRAMA3_CODEC_REVISION,
    FishCodecConfig,
)
from mobius.models.fish_codec import FishCodecDecoder
from mobius.tasks.fish_codec import build_fish_codec_decoder, load_drama3_codec_decoder


def _tiny():
    return FishCodecConfig(
        latent_dim=16,
        transformer_layers=1,
        transformer_heads=2,
        transformer_intermediate_size=32,
        max_frames=16,
        decoder_dim=32,
        decoder_rates=(2, 2),
    )


def test_native_codec_contract():
    config = FishCodecConfig()
    assert config.num_codebooks == 10
    assert config.samples_per_frame == 2048
    assert config.sample_rate == 44100
    model = build_fish_codec_decoder(_tiny())["waveform_decoder"]
    assert list(model.graph.inputs[0].shape) == ["batch", 10, "frames"]
    assert model.graph.inputs[0].dtype == ir.DataType.INT64
    assert model.graph.outputs[0].name == "waveform"
    assert model.metadata_props["codec_revision"] == DRAMA3_CODEC_REVISION
    assert any(node.op_type == "Attention" for node in model.graph)
    assert not any(node.domain == "com.microsoft" for node in model.graph)


def test_missing_and_mismatched_weights_fail_closed():
    module = FishCodecDecoder(_tiny())
    with pytest.raises(ValueError, match="Missing codec weight"):
        module.preprocess_weights({})
    name = next(iter(dict(module.named_parameters())))
    with pytest.raises(ValueError, match=r"expected .*got"):
        module.preprocess_weights({name: torch.zeros(1)})


def test_preprocess_both_weight_norm_formats():
    module = FishCodecDecoder(_tiny())
    state = {name: torch.ones(tuple(param.shape)) for name, param in module.named_parameters()}
    for index, name in enumerate(
        [
            "quantizer.semantic_quantizer.quantizers.0.out_proj.weight",
            "decoder.model.0.conv.weight",
        ]
    ):
        v = state.pop(name)
        prefix = name.removesuffix(".weight")
        g = torch.full((v.shape[0], 1, 1), 0.5)
        if index == 0:
            state[prefix + ".weight_g"] = g
            state[prefix + ".weight_v"] = v
        else:
            state[prefix + ".parametrizations.weight.original0"] = g
            state[prefix + ".parametrizations.weight.original1"] = v
    converted = module.preprocess_weights(state)
    assert converted.keys() == dict(module.named_parameters()).keys()
    for name in (
        "quantizer.semantic_quantizer.quantizers.0.out_proj.weight",
        "decoder.model.0.conv.weight",
    ):
        tensor = converted[name]
        torch.testing.assert_close(
            torch.linalg.vector_norm(tensor, dim=(1, 2)), torch.full((tensor.shape[0],), 0.5)
        )


def test_loader_pins_revision_and_safe_checkpoint_options():
    with (
        patch("mobius.tasks.fish_codec.hf_hub_download", return_value="codec.pth") as download,
        patch("mobius.tasks.fish_codec.torch.load", return_value={}) as load,
        patch("mobius.tasks.fish_codec.build_fish_codec_decoder") as build,
    ):
        load_drama3_codec_decoder(cache_dir="cache")
        assert download.call_args.kwargs["revision"] == DRAMA3_CODEC_REVISION
        load.assert_called_once_with(
            "codec.pth", map_location="cpu", mmap=True, weights_only=True
        )
        build.assert_called_once_with(state_dict={})


@pytest.mark.parametrize("value", [None, [], {"weight": "not a tensor"}])
def test_loader_rejects_non_tensor_checkpoints(value):
    with (
        patch("mobius.tasks.fish_codec.torch.load", return_value=value),
        pytest.raises(ValueError, match="tensor state dictionary"),
    ):
        load_drama3_codec_decoder("caller-owned.pth")


@pytest.mark.parametrize(
    "overrides",
    [
        {"latent_dim": 15},
        {"transformer_heads": 0},
        {"decoder_dim": 15},
        {"transformer_window": 0},
        {"decoder_rates": ()},
    ],
)
def test_invalid_config(overrides):
    with pytest.raises(ValueError):
        dataclasses.replace(_tiny(), **overrides)
