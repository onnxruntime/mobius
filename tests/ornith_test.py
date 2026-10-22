# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Pinned Ornith-1.5 compatibility tests using the existing Qwen3.5 models.

Only configs are downloaded: even the smallest checkpoint has 9B parameters.
These tests establish configuration/graph compatibility, not real-weight parity.
"""

from __future__ import annotations

import re
from pathlib import Path

import onnx_ir as ir
import pytest

from mobius import build
from mobius._registry import registry
from mobius._testing.golden import load_test_case
from mobius.integrations.transformers._builder import (
    _load_transformers_config,
    _select_primary_config,
)
from mobius.integrations.transformers._config_resolver import (
    _config_from_hf,
    _try_load_config_json,
)

_CASES = Path(__file__).resolve().parent.parent / "testdata" / "cases" / "vision-language"
_VARIANTS = [
    pytest.param("9b", 4096, 32, 16, 4, 32, 0, 0, 0, id="ornith-1.5-9b"),
    pytest.param("35b-a3b", 2048, 40, 16, 2, 32, 256, 8, 512, id="ornith-1.5-35b-a3b"),
    pytest.param("122b", 3072, 48, 32, 2, 64, 256, 8, 1024, id="ornith-1.5-122b"),
    pytest.param("397b", 4096, 60, 32, 2, 64, 512, 10, 1024, id="ornith-1.5-397b"),
]


@pytest.mark.parametrize(
    "variant", ["9b", "35b-a3b", "122b", "397b"], ids=lambda v: f"ornith-1.5-{v}"
)
def test_ornith_case_is_pinned(variant: str):
    case = load_test_case(_CASES / f"ornith-1_5-{variant}.yaml")
    assert re.fullmatch(r"[0-9a-f]{40}", case.revision)
    assert case.model_id == f"ornith-ai/Ornith-1.5-{variant.upper()}"
    assert case.model_type == ("qwen3_5" if variant == "9b" else "qwen3_5_moe_vl")
    assert case.reference_loader == "multimodal"
    assert not case.trust_remote_code
    assert case.generation_params["eos_token_id"] == 248046


@pytest.mark.arch_validation
@pytest.mark.parametrize(
    "variant", ["9b", "35b-a3b", "122b", "397b"], ids=lambda v: f"ornith-1.5-{v}"
)
def test_ornith_raw_config_matches_transformers(variant: str):
    """Raw JSON fallback must preserve the same nested decoder/vision semantics."""
    case = load_test_case(_CASES / f"ornith-1_5-{variant}.yaml")
    typed, raw_fallback = _load_transformers_config(
        case.model_id, revision=case.revision, trust_remote_code=False
    )
    assert typed is not None
    assert not raw_fallback
    raw = _try_load_config_json(case.model_id, revision=case.revision)
    assert raw is not None
    configs = []
    for hf_config in (typed, raw):
        primary, parent, model_type = _select_primary_config(hf_config)
        assert model_type == case.model_type
        configs.append(
            _config_from_hf(
                primary, parent_config=parent, module_class=registry.get(model_type)
            )
        )
    assert configs[0] == configs[1]


@pytest.mark.arch_validation
@pytest.mark.parametrize("text_only", [False, True], ids=["vision-language", "text-only"])
@pytest.mark.parametrize(
    "variant,hidden,layers,q_heads,kv_heads,value_heads,experts,top_k,expert_size",
    _VARIANTS,
)
def test_ornith_pinned_graph(
    variant: str,
    hidden: int,
    layers: int,
    q_heads: int,
    kv_heads: int,
    value_heads: int,
    experts: int,
    top_k: int,
    expert_size: int,
    text_only: bool,
):
    """Exercise public dispatch, nested config extraction, and hybrid cache wiring."""
    case = load_test_case(_CASES / f"ornith-1_5-{variant}.yaml")
    package = build(
        case.model_id,
        revision=case.revision,
        dtype="bf16",
        load_weights=False,
        text_only=text_only,
    )
    config = package.config
    assert config.hidden_size == hidden
    assert config.num_hidden_layers == layers
    assert config.num_attention_heads == q_heads
    assert config.num_key_value_heads == kv_heads
    assert config.head_dim == 256
    assert config.vocab_size == 248320
    assert config.max_position_embeddings == 262144
    assert config.dtype == ir.DataType.BFLOAT16
    assert not config.tie_word_embeddings
    assert config.rope_theta == 10_000_000
    assert config.partial_rotary_factor == pytest.approx(0.25)
    assert config.mrope_section == [11, 11, 10]
    assert config.linear_num_key_heads == 16
    assert config.linear_num_value_heads == value_heads
    assert config.linear_key_head_dim == 128
    assert config.linear_value_head_dim == 128
    assert config.linear_conv_kernel_dim == 4
    assert config.layer_types == [
        "full_attention" if i % 4 == 3 else "linear_attention" for i in range(layers)
    ]
    if experts:
        assert config.num_local_experts == experts
        assert config.num_experts_per_tok == top_k
        assert config.moe_intermediate_size == expert_size
        assert config.shared_expert_intermediate_size == expert_size
    else:
        assert config.intermediate_size == 12288

    if text_only:
        assert set(package) == {"model"}
        assert config.model_type == ("qwen3_5_text" if variant == "9b" else "qwen3_5_moe_text")
        assert config.vision is None
        decoder = package["model"]
        assert "input_ids" in {value.name for value in decoder.graph.inputs}
    else:
        assert set(package) == {"decoder", "vision_encoder", "embedding"}
        assert config.vision is not None
        assert config.vision.hidden_size == 1152
        assert config.vision.num_hidden_layers == 27
        assert config.vision.num_attention_heads == 16
        assert config.vision.out_hidden_size == hidden
        assert config.vision.deepstack_visual_indexes == []
        assert config.image_token_id == 248056
        assert config.video_token_id == 248057
        decoder = package["decoder"]
        assert "inputs_embeds" in {value.name for value in decoder.graph.inputs}

    inputs = {value.name for value in decoder.graph.inputs}
    outputs = {value.name for value in decoder.graph.outputs}
    assert "logits" in outputs
    for i in range(layers):
        state_names = ("key", "value") if i % 4 == 3 else ("conv_state", "recurrent_state")
        for state in state_names:
            assert f"past_key_values.{i}.{state}" in inputs
            assert f"present.{i}.{state}" in outputs
    for name, model in package.items():
        assert model.graph.name == f"{case.model_id}/{name}"
        assert model.graph.initializers
        assert len(model.graph) > 0
