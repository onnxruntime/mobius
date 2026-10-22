# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Fail-closed architecture and component contracts for the Omni split."""

from __future__ import annotations

import dataclasses

import onnx_ir as ir
import pytest
from onnxscript import nn

from mobius import build_from_module
from mobius._configs import NemotronHOmniConfig
from mobius._configs.per_model._nemotron_h_omni_test import _omni_config
from mobius._flags import override_flags
from mobius.integrations.transformers._nemotron_h_omni_weights import _fixed_shape
from mobius.integrations.transformers._nemotron_h_omni_weights_test import _fixture
from mobius.models.nemotron_h import NemotronHMoEBlock
from mobius.models.nemotron_h_omni import NemotronHOmniModel, _OmniVisionEncoder
from mobius.tasks._nemotron_h_omni import NemotronHOmniTask


@pytest.mark.parametrize(
    "field",
    [
        "patch_size",
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
    ],
)
def test_omni_vision_rejects_missing_architecture_dimension(field):
    config = NemotronHOmniConfig.from_transformers(_omni_config())
    config = dataclasses.replace(
        config, vision=dataclasses.replace(config.vision, **{field: None})
    )
    with pytest.raises(ValueError, match="explicit RADIO"):
        _OmniVisionEncoder(config)


@pytest.mark.parametrize("component", ["decoder", "vision_encoder", "embedding"])
def test_omni_task_rejects_non_module_component(component):
    config = NemotronHOmniConfig.from_transformers(_omni_config())
    module = nn.Module()
    for name in ("decoder", "vision_encoder", "embedding"):
        setattr(module, name, nn.Module() if name != component else object())
    with pytest.raises(TypeError, match="must be modules"):
        NemotronHOmniTask().build(module, config)


@pytest.mark.parametrize("shape", [[], [2, 3], [0, 3]])
def test_native_omni_initializer_dimensions_are_fixed(shape):
    assert _fixed_shape(ir.Shape(shape)) == tuple(shape)


def test_native_omni_initializer_rejects_symbolic_dimensions():
    with pytest.raises(TypeError, match="fixed dimensions"):
        _fixed_shape(ir.Shape([2, "packed_k"]))


def test_sparse_iq2_emits_per_layer_without_building_dense_expert_dispatch(monkeypatch):
    def reject_dense(*_args):
        raise AssertionError("The dense expert dispatch must never be built")

    monkeypatch.setattr(NemotronHMoEBlock, "_dispatch_experts", reject_dense)
    config, package, _ = _fixture(sparse=True)
    decoder = package["decoder"]
    nodes = [node for node in decoder.graph if node.op_type == "QMoE"]
    assert len(nodes) == 2
    assert [node.attributes["k"].as_int() for node in nodes] == [1, 2]
    assert [list(node.inputs[2].shape) for node in nodes] == [[2, 256, 1, 66], [3, 512, 1, 66]]
    assert [list(node.inputs[5].shape) for node in nodes] == [[2, 256, 1, 66], [3, 256, 2, 66]]
    assert not any(".moe.experts." in name for name in decoder.graph.initializers)
    assert "decoder.model.layers.1.moe.gate.weight" in decoder.graph.initializers
    assert "decoder.model.layers.1.moe.shared_experts.up_proj.weight" in (
        decoder.graph.initializers
    )
    assert "decoder.model.layers.1.moe.fc1_latent_proj.weight" in (decoder.graph.initializers)
    assert config.num_hidden_layers == 4
    inputs = {value.name for value in decoder.graph.inputs}
    assert {
        "inputs_embeds",
        "past_key_values.0.conv_state",
        "past_key_values.0.ssm_state",
        "past_key_values.2.key",
        "past_key_values.2.value",
    } <= inputs
    assert set(package) == {"decoder", "vision_encoder", "embedding"}


def test_sparse_iq2_is_opt_in_and_dense_diagnostic_is_preserved():
    _, package, _ = _fixture()
    assert not any(node.op_type == "QMoE" for node in package["decoder"].graph)
    assert _fixed_shape(
        package["decoder"]
        .graph.initializers["decoder.model.layers.1.moe.experts.0.up_proj.weight"]
        .shape
    ) == (8, 1, 66)


@pytest.mark.parametrize("invalid", ["missing", "mixed", "bias", "activation"])
def test_sparse_iq2_does_not_silently_fall_back_for_ineligible_experts(invalid):
    config, _, _ = _fixture(sparse=True)
    projections = dict(config.quantized_projections)
    root = "language_model.backbone.layers.1.mixer.experts.0.down_proj"
    if invalid == "missing":
        del projections[root]
    elif invalid == "mixed":
        projections[root] = dataclasses.replace(
            projections[root], quant_algo="Q8_0", group_size=32, block_payload_bytes=34
        )
    config = dataclasses.replace(
        config,
        quantized_projections=projections,
        mlp_bias=invalid == "bias",
        hidden_act="silu" if invalid == "activation" else "relu2",
    )
    with (
        override_flags(sparse_iq2_moe=True),
        pytest.raises((ValueError, TypeError), match="Sparse IQ2 MoE"),
    ):
        NemotronHOmniModel(config)


@pytest.mark.parametrize("provider", ["default", "cpu", "dml"])
def test_sparse_omni_build_rejects_unsupported_target_provider(provider):
    config, _, _ = _fixture(sparse=True)
    with (
        override_flags(sparse_iq2_moe=True),
        pytest.raises(ValueError, match="companion CUDA QMoE"),
    ):
        build_from_module(
            NemotronHOmniModel(config),
            config,
            task="nemotron-h-omni",
            execution_provider=provider,
        )
