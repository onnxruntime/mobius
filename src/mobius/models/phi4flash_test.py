# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Unit coverage for the pinned Phi-4 Flash SambaY configuration and routing."""

from __future__ import annotations

from types import SimpleNamespace

import onnx_ir as ir
import pytest
import torch

from mobius._builder import build_from_module
from mobius._configs import Phi4FlashConfig
from mobius._testing import count_op_type
from mobius.models.phi4flash import Phi4FlashCausalLMModel
from mobius.tasks import Phi4FlashCausalLMTask

_PINNED_REVISION = "1dff8163d28ec880ca2411c474ddc0a927792810"


def _config(**overrides: object) -> Phi4FlashConfig:
    values: dict[str, object] = {
        "model_type": "phi4flash",
        "vocab_size": 64,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "hidden_act": "silu",
        "max_position_embeddings": 128,
        "pad_token_id": 0,
        "tie_word_embeddings": True,
        "mamba_d_state": 4,
        "mamba_d_conv": 4,
        "mamba_expand": 2,
        "mamba_dt_rank": 2,
        "local_attention_window": 8,
        "layer_norm_eps": 1e-5,
        "rope_type": None,
        "rope_theta": None,
    }
    values.update(overrides)
    return Phi4FlashConfig(**values)


def _build(config: Phi4FlashConfig):
    module = Phi4FlashCausalLMModel(config)
    package = Phi4FlashCausalLMTask().build(module, config)
    return module, package


def _parameter_names(package) -> set[str]:
    return {
        name
        for name, initializer in package["model"].graph.initializers.items()
        if initializer.const_value is None
    }


def test_pinned_remote_config_extracts_exact_sambay_contract() -> None:
    """The remote ``config.json`` values map to the model's fixed topology."""
    # This is the complete architecture-relevant subset of
    # microsoft/Phi-4-mini-flash-reasoning at _PINNED_REVISION. The remote
    # class supplies omitted Mamba values from its checked-in defaults.
    remote = SimpleNamespace(
        model_type="phi4flash",
        architectures=["Phi4FlashForCausalLM"],
        vocab_size=200_064,
        hidden_size=2_560,
        intermediate_size=10_240,
        num_hidden_layers=32,
        num_attention_heads=40,
        num_key_value_heads=20,
        max_position_embeddings=262_144,
        sliding_window=512,
        hidden_act="silu",
        layer_norm_eps=1e-5,
        tie_word_embeddings=True,
        torch_dtype="bfloat16",
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )

    config = Phi4FlashConfig.from_transformers(remote)

    assert _PINNED_REVISION == "1dff8163d28ec880ca2411c474ddc0a927792810"
    assert config.dtype == ir.DataType.BFLOAT16
    assert (
        config.vocab_size,
        config.hidden_size,
        config.intermediate_size,
        config.num_attention_heads,
        config.num_key_value_heads,
        config.head_dim,
        config.max_position_embeddings,
    ) == (200_064, 2_560, 10_240, 40, 20, 64, 262_144)
    assert config.local_attention_window == 512
    assert (
        config.mamba_d_state,
        config.mamba_d_conv,
        config.mamba_expand,
        config.mamba_dt_rank,
    ) == (16, 4, 2, 160)
    assert config.cache_slot_count == 18
    assert config.rope_type is None
    assert config.rope_theta is None
    assert config.layer_types[:18] == [
        "mamba",
        "local_differential_attention",
    ] * 8 + ["shared_memory_mamba", "global_differential_attention"]
    assert config.layer_types[18:] == [
        "cross_mamba",
        "cross_differential_attention",
    ] * 7


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"num_hidden_layers": 6}, "divisible by four"),
        ({"mb_per_layer": 1}, "mb_per_layer=2"),
        ({"local_attention_window": 0}, "must be positive"),
        ({"mamba_d_conv": 1}, "convolution width"),
        ({"mamba_dt_rank": 0}, "must be positive"),
        ({"num_attention_heads": 1, "num_key_value_heads": 1, "head_dim": 32}, "even Q"),
        ({"export_paged_attention": True}, "cannot use paged attention"),
    ],
)
def test_sambay_config_rejects_non_source_topologies(
    overrides: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _config(**overrides)


def test_all_checkpoint_parameter_families_route_without_value_loss() -> None:
    """Every tiny-model parameter family uses its remote checkpoint name directly."""
    module, package = _build(_config())
    expected = _parameter_names(package)
    embedding = torch.full((64, 32), 0.25)
    source: dict[str, torch.Tensor] = {"lm_head.weight": embedding}
    for index, name in enumerate(sorted(expected - {"model.embed_tokens.weight"}), start=1):
        initializer = package["model"].graph.initializers[name]
        source[name] = torch.full(tuple(initializer.shape), float(index))

    routed = module.preprocess_weights(source)

    assert set(routed) == expected
    assert routed["model.embed_tokens.weight"] is embedding
    assert torch.equal(routed["model.layers.0.attn.A_log"], source["model.layers.0.attn.A_log"])
    assert torch.equal(
        routed["model.layers.1.attn.inner_cross_attn.lambda_q1"],
        source["model.layers.1.attn.inner_cross_attn.lambda_q1"],
    )
    assert torch.equal(
        routed["model.layers.2.attn.conv1d.weight"],
        source["model.layers.2.attn.conv1d.weight"],
    )
    assert torch.equal(routed["model.layers.3.attn.Wqkv.bias"], source["model.layers.3.attn.Wqkv.bias"])
    assert torch.equal(
        routed["model.final_layernorm.bias"], source["model.final_layernorm.bias"]
    )


def test_tied_checkpoint_duplicates_must_match() -> None:
    module = Phi4FlashCausalLMModel(_config())
    with pytest.raises(ValueError, match="different tied"):
        module.preprocess_weights(
            {
                "model.embed_tokens.weight": torch.zeros(64, 32),
                "lm_head.weight": torch.ones(64, 32),
            }
        )


def test_bf16_build_keeps_source_float_spectrum_and_lambda_parameters() -> None:
    config = _config(dtype=ir.DataType.BFLOAT16)
    module = Phi4FlashCausalLMModel(config)
    package = build_from_module(module, config, task=Phi4FlashCausalLMTask())
    initializers = package["model"].graph.initializers

    assert initializers["model.layers.0.attn.A_log"].dtype == ir.DataType.FLOAT
    assert initializers["model.layers.0.attn.D"].dtype == ir.DataType.FLOAT
    assert (
        initializers["model.layers.1.attn.inner_cross_attn.lambda_q1"].dtype
        == ir.DataType.FLOAT
    )
    assert initializers["model.layers.1.attn.Wqkv.weight"].dtype == ir.DataType.BFLOAT16


@pytest.mark.parametrize("dtype", [ir.DataType.FLOAT, ir.DataType.FLOAT16, ir.DataType.BFLOAT16])
def test_all_export_weight_dtypes_build_the_dynamic_state_contract(dtype: ir.DataType) -> None:
    """Weight storage varies independently from the source-mandated BF16 attention read."""
    _, package = _build(_config(dtype=dtype))
    model = package["model"]

    assert model.graph.outputs[0].type == ir.TensorType(dtype)
    assert model.graph.inputs[2].type == ir.TensorType(dtype)


def test_requested_layer_states_and_graph_derived_metadata_are_exposed() -> None:
    _, package = _build(_config(output_layer_indices=[0, 2]))
    model = package["model"]
    outputs = {value.name for value in model.graph.outputs}

    assert {"hidden_states.0", "hidden_states.2"} <= outputs
    assert "B,64,4" in model.metadata_props["mobius.cache_abi"]
    assert "<=8,8" in model.metadata_props["mobius.cache_abi"]
    assert "CUDA: supported" in model.metadata_props["mobius.execution_provider_feasibility"]
    assert "selective_scan" in model.metadata_props["mobius.reference_kernel_audit"]
    assert "unvalidated" in model.metadata_props["mobius.quantization_assessment"]


def test_attention_uses_compact_native_and_windowed_masks_for_long_prefill() -> None:
    """No causal Q-by-K bias is materialized for the 64K+ source context contract."""
    _, package = _build(_config(num_hidden_layers=8, output_layer_indices=list(range(8))))
    graph = package["model"].graph

    # Each differential read selects compact GQA/causal Attention for unpadded
    # inputs or a source-faithful standard Attention mask for padded batches.
    assert count_op_type(graph, "If") == 16
    assert sum(node.op_type == "GroupQueryAttention" for node in graph.all_nodes()) == 8
    assert sum(node.op_type == "Attention" for node in graph.all_nodes()) == 24
    assert count_op_type(graph, "Where") == 0


def test_ort_genai_config_rejects_heterogeneous_sambay_state(tmp_path) -> None:
    """Do not write an OGA config that advertises an ABI its runtime cannot load."""
    from mobius.integrations.ort_genai import write_ort_genai_config

    _, package = _build(_config())
    with pytest.raises(ValueError, match="heterogeneous dynamic SambaY cache ABI"):
        write_ort_genai_config(package, str(tmp_path))
