# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Unit tests for the Cosmos3-Omni reasoner weight translation."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

# tests/_test_configs.py provides the tiny cosmos3_omni ArchitectureConfig.
_TESTS_DIR = Path(__file__).resolve().parents[3] / "tests"
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

from _test_configs import VL_CONFIGS, _base_config  # noqa: E402

from mobius.models.cosmos3_omni import Cosmos3OmniReasonerModel  # noqa: E402


def _tiny_reasoner() -> Cosmos3OmniReasonerModel:
    overrides = next(o for mt, o, _ in VL_CONFIGS if mt == "cosmos3_omni")
    return Cosmos3OmniReasonerModel(_base_config(**overrides))


def test_self_attn_renames_collapse_to_hf_names():
    """Native ``to_{q,k,v,out}`` / ``norm_{q,k}`` map to HF Qwen3-VL names."""
    module = _tiny_reasoner()
    sd = {
        "layers.0.self_attn.to_q.weight": torch.zeros(1),
        "layers.0.self_attn.to_k.weight": torch.zeros(1),
        "layers.0.self_attn.to_v.weight": torch.zeros(1),
        "layers.0.self_attn.to_out.weight": torch.zeros(1),
        "layers.0.self_attn.norm_q.weight": torch.zeros(1),
        "layers.0.self_attn.norm_k.weight": torch.zeros(1),
    }
    out = set(module.preprocess_weights(sd))
    # Native diffusers-style names must not survive the translation.
    assert not any("to_out" in k or "to_q" in k or "norm_q" in k for k in out)
    assert any(k.endswith("self_attn.q_proj.weight") for k in out)
    assert any(k.endswith("self_attn.o_proj.weight") for k in out)
    assert any(k.endswith("self_attn.q_norm.weight") for k in out)


def test_diffusers_style_to_out_sequential_maps_to_single_o_proj():
    """A diffusers-style ``to_out.0.weight`` collapses to ``o_proj.weight``.

    diffusers wraps the attention output projection in an ``nn.Sequential``
    ([Linear, Dropout]), producing ``...to_out.0.weight``.  The rename must
    yield the single HF ``o_proj.weight`` — never ``o_proj.0.weight``.
    """
    module = _tiny_reasoner()
    out = set(
        module.preprocess_weights({"layers.0.self_attn.to_out.0.weight": torch.zeros(1)})
    )
    assert any(k.endswith("self_attn.o_proj.weight") for k in out)
    assert not any("o_proj.0" in k or "to_out" in k for k in out)
