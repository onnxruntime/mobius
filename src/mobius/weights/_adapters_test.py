# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for the model weight adapter boundary."""

from __future__ import annotations

import torch
from onnxscript import nn

from mobius._component_manifest import ComponentDescriptor, ComponentManifest
from mobius._configs import ArchitectureConfig
from mobius.weights import WeightAdapterContext, adapt_model_weights


def _manifest() -> ComponentManifest:
    return ComponentManifest(
        (
            ComponentDescriptor(
                name="model",
                module_path="",
                role="decoder",
            ),
        )
    )


class _Module(nn.Module):
    def preprocess_weights(self, state_dict):
        return {f"legacy.{name}": value for name, value in state_dict.items()}


def test_legacy_preprocess_hook_remains_supported():
    tensor = torch.ones(2)

    result = adapt_model_weights(
        _Module(),
        {"weight": tensor},
        config=ArchitectureConfig(),
        manifest=_manifest(),
    )

    assert result["legacy.weight"] is tensor


def test_explicit_adapter_takes_precedence():
    class _Adapter:
        def adapt(self, module, state_dict, context: WeightAdapterContext):
            assert isinstance(module, _Module)
            assert context.manifest.names == ("model",)
            return {f"adapter.{name}": value for name, value in state_dict.items()}

    module = _Module()
    module.weight_adapter = _Adapter()
    tensor = torch.ones(2)

    result = adapt_model_weights(
        module,
        {"weight": tensor},
        config=ArchitectureConfig(),
        manifest=_manifest(),
    )

    assert result["adapter.weight"] is tensor
    assert "legacy.weight" not in result
