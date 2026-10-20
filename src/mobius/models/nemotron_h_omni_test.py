# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Fail-closed architecture and component contracts for the Omni split."""

from __future__ import annotations

import dataclasses

import onnx_ir as ir
import pytest
from onnxscript import nn

from mobius._configs import NemotronHOmniConfig
from mobius._configs.per_model._nemotron_h_omni_test import _omni_config
from mobius.integrations.transformers._nemotron_h_omni_weights import _fixed_shape
from mobius.models.nemotron_h_omni import _OmniVisionEncoder
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
