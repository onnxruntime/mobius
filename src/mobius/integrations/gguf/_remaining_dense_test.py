# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Executable coverage for the remaining MiniMax-M2, Mistral4, and GLM-DSA routes."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import onnx_ir as ir
import pytest
import torch
from onnxscript import GraphBuilder

from mobius._builder import build_from_module
from mobius._constants import OPSET_VERSION
from mobius._registry import registry
from mobius._testing import make_config
from mobius._testing.ort_inference import OnnxModelSession
from mobius.integrations.gguf import build_from_gguf
from mobius.integrations.gguf._arch_registry import get_arch_spec
from mobius.integrations.gguf._builder import _serialize_route_graph_config
from mobius.integrations.gguf._config_mapping import gguf_to_config
from mobius.integrations.gguf._remaining_dense import (
    validate_remaining_dense_tensor_contract,
)
from mobius.integrations.gguf._spec import Support
from mobius.integrations.gguf._tensor_mapping import map_gguf_to_hf_names
from mobius.integrations.gguf._upstream import upstream_architectures
from mobius.models.deepseek import DeepSeekV3CausalLMModel
from mobius.models.gguf_minimax_m2 import (
    MiniMaxM2Gate,
    MiniMaxM2GGUFCausalLMModel,
)
from mobius.models.gguf_mistral4 import (
    Mistral4GGUFCausalLMModel,
    Mistral4LatentAttention,
)
from mobius.tasks import CausalLMTask


@dataclass
class _FakeGGUF:
    architecture: str
    metadata: dict[str, object]
    tensors: dict[str, np.ndarray]

    def __post_init__(self) -> None:
        self.tensor_names = list(self.tensors)
        self.qtypes = {
            name: SimpleNamespace(value=0, name="F32") for name in self.tensor_names
        }

    def get_metadata(self, key: str, default=None):
        return self.metadata.get(key, default)

    def tensor_items_raw(self):
        for name, tensor in self.tensors.items():
            yield name, None, self.qtypes[name], tensor.shape


def _values(
    rng: np.random.Generator,
    shape: tuple[int, ...],
    *,
    norm: bool = False,
) -> np.ndarray:
    if norm:
        return rng.uniform(0.7, 1.3, shape).astype(np.float32)
    return (rng.standard_normal(shape) * 0.08).astype(np.float32)


def _m2_fixture() -> _FakeGGUF:
    arch = "minimax-m2"
    hidden, layers, heads, kv_heads, head_dim = 8, 1, 2, 1, 8
    intermediate, experts, top_k, vocab = 6, 3, 2, 16
    metadata: dict[str, object] = {
        f"{arch}.context_length": 16,
        f"{arch}.embedding_length": hidden,
        f"{arch}.feed_forward_length": intermediate,
        f"{arch}.block_count": layers,
        f"{arch}.attention.head_count": heads,
        f"{arch}.attention.head_count_kv": kv_heads,
        f"{arch}.attention.key_length": head_dim,
        f"{arch}.attention.value_length": head_dim,
        f"{arch}.attention.layer_norm_rms_epsilon": 1e-6,
        f"{arch}.rope.freq_base": 5_000_000.0,
        f"{arch}.rope.dimension_count": 4,
        f"{arch}.vocab_size": vocab,
        f"{arch}.expert_count": experts,
        f"{arch}.expert_used_count": top_k,
        f"{arch}.expert_feed_forward_length": intermediate,
        f"{arch}.expert_gating_func": 2,
    }
    rng = np.random.default_rng(11)
    q_width = heads * head_dim
    kv_width = kv_heads * head_dim
    tensors = {
        "token_embd.weight": _values(rng, (vocab, hidden)),
        "output_norm.weight": _values(rng, (hidden,), norm=True),
        "output.weight": _values(rng, (vocab, hidden)),
        "blk.0.attn_norm.weight": _values(rng, (hidden,), norm=True),
        "blk.0.attn_q.weight": _values(rng, (q_width, hidden)),
        "blk.0.attn_k.weight": _values(rng, (kv_width, hidden)),
        "blk.0.attn_v.weight": _values(rng, (kv_width, hidden)),
        "blk.0.attn_q_norm.weight": np.linspace(0.3, 1.7, q_width, dtype=np.float32),
        "blk.0.attn_k_norm.weight": np.linspace(0.6, 1.4, kv_width, dtype=np.float32),
        "blk.0.attn_output.weight": _values(rng, (hidden, q_width)),
        "blk.0.ffn_norm.weight": _values(rng, (hidden,), norm=True),
        "blk.0.ffn_gate_inp.weight": _values(rng, (experts, hidden)),
        "blk.0.ffn_gate_exps.weight": _values(rng, (experts, intermediate, hidden)),
        "blk.0.ffn_up_exps.weight": _values(rng, (experts, intermediate, hidden)),
        "blk.0.ffn_down_exps.weight": _values(rng, (experts, hidden, intermediate)),
        "blk.0.exp_probs_b.bias": np.array([0.8, -0.5, 0.1], dtype=np.float32),
    }
    return _FakeGGUF(arch, metadata, tensors)


def _m4_fixture() -> _FakeGGUF:
    arch = "mistral4"
    hidden, layers, heads = 8, 2, 2
    dense_width, q_lora, kv_lora = 6, 4, 4
    nope, rope, value_dim = 2, 2, 2
    experts, top_k, expert_width, vocab = 2, 1, 3, 16
    metadata: dict[str, object] = {
        f"{arch}.context_length": 16,
        f"{arch}.embedding_length": hidden,
        f"{arch}.feed_forward_length": dense_width,
        f"{arch}.block_count": layers,
        f"{arch}.attention.head_count": heads,
        f"{arch}.attention.head_count_kv": 1,
        f"{arch}.attention.key_length": kv_lora + rope,
        f"{arch}.attention.value_length": kv_lora,
        f"{arch}.attention.key_length_mla": nope + rope,
        f"{arch}.attention.value_length_mla": value_dim,
        f"{arch}.attention.q_lora_rank": q_lora,
        f"{arch}.attention.kv_lora_rank": kv_lora,
        f"{arch}.attention.layer_norm_rms_epsilon": 1e-5,
        f"{arch}.rope.freq_base": 10_000.0,
        f"{arch}.rope.dimension_count": rope,
        f"{arch}.vocab_size": vocab,
        f"{arch}.expert_count": experts,
        f"{arch}.expert_used_count": top_k,
        f"{arch}.expert_feed_forward_length": expert_width,
        f"{arch}.expert_shared_count": 1,
        f"{arch}.expert_weights_norm": True,
        f"{arch}.leading_dense_block_count": 1,
    }
    rng = np.random.default_rng(17)
    tensors: dict[str, np.ndarray] = {
        "token_embd.weight": _values(rng, (vocab, hidden)),
        "output_norm.weight": _values(rng, (hidden,), norm=True),
        "output.weight": _values(rng, (vocab, hidden)),
    }
    for layer in range(layers):
        prefix = f"blk.{layer}."
        tensors.update(
            {
                prefix + "attn_norm.weight": _values(rng, (hidden,), norm=True),
                prefix + "attn_q_a_norm.weight": _values(rng, (q_lora,), norm=True),
                prefix + "attn_kv_a_norm.weight": _values(rng, (kv_lora,), norm=True),
                prefix + "attn_q_a.weight": _values(rng, (q_lora, hidden)),
                prefix + "attn_q_b.weight": _values(rng, (heads * (nope + rope), q_lora)),
                prefix + "attn_kv_a_mqa.weight": _values(rng, (kv_lora + rope, hidden)),
                # GGUF stores K-B as [heads, kv_lora, nope].
                prefix + "attn_k_b.weight": _values(rng, (heads, kv_lora, nope)),
                prefix + "attn_v_b.weight": _values(rng, (heads, value_dim, kv_lora)),
                prefix + "attn_output.weight": _values(rng, (hidden, heads * value_dim)),
                prefix + "ffn_norm.weight": _values(rng, (hidden,), norm=True),
            }
        )
    tensors.update(
        {
            "blk.0.ffn_gate.weight": _values(rng, (dense_width, hidden)),
            "blk.0.ffn_up.weight": _values(rng, (dense_width, hidden)),
            "blk.0.ffn_down.weight": _values(rng, (hidden, dense_width)),
            "blk.1.ffn_gate_inp.weight": _values(rng, (experts, hidden)),
            "blk.1.ffn_gate_up_exps.weight": _values(rng, (experts, 2 * expert_width, hidden)),
            "blk.1.ffn_down_exps.weight": _values(rng, (experts, hidden, expert_width)),
            "blk.1.ffn_gate_shexp.weight": _values(rng, (expert_width, hidden)),
            "blk.1.ffn_up_shexp.weight": _values(rng, (expert_width, hidden)),
            "blk.1.ffn_down_shexp.weight": _values(rng, (hidden, expert_width)),
        }
    )
    return _FakeGGUF(arch, metadata, tensors)


def _glm_fixture() -> _FakeGGUF:
    source = _m4_fixture()
    arch = "glm-dsa"
    source.architecture = arch
    source.metadata = {
        key.replace("mistral4.", f"{arch}."): value for key, value in source.metadata.items()
    }
    source.metadata.update(
        {
            f"{arch}.attention.indexer.head_count": 2,
            f"{arch}.attention.indexer.key_length": 4,
            f"{arch}.attention.indexer.top_k": 4,
            f"{arch}.attention.indexer.types": [True, False],
            f"{arch}.expert_gating_func": 2,
            f"{arch}.expert_weights_scale": 2.5,
        }
    )
    rng = np.random.default_rng(23)
    hidden = int(source.metadata[f"{arch}.embedding_length"])
    q_lora = int(source.metadata[f"{arch}.attention.q_lora_rank"])
    experts = int(source.metadata[f"{arch}.expert_count"])
    expert_width = int(source.metadata[f"{arch}.expert_feed_forward_length"])
    source.tensors.update(
        {
            "blk.0.indexer.k_norm.weight": _values(rng, (4,), norm=True),
            "blk.0.indexer.k_norm.bias": _values(rng, (4,)),
            "blk.0.indexer.proj.weight": _values(rng, (2, hidden)),
            "blk.0.indexer.attn_k.weight": _values(rng, (4, hidden)),
            "blk.0.indexer.attn_q_b.weight": _values(rng, (8, q_lora)),
            "blk.1.exp_probs_b.bias": _values(rng, (experts,)),
            "blk.1.ffn_gate_exps.weight": _values(rng, (experts, expert_width, hidden)),
            "blk.1.ffn_up_exps.weight": _values(rng, (experts, expert_width, hidden)),
        }
    )
    source.tensors.pop("blk.1.ffn_gate_up_exps.weight")
    source.tensor_names = list(source.tensors)
    source.qtypes = {
        name: SimpleNamespace(value=0, name="F32") for name in source.tensor_names
    }
    return source


def _write_gguf(path: Path, source: _FakeGGUF) -> None:
    from gguf import GGUFWriter

    writer = GGUFWriter(str(path), source.architecture)
    for key, value in source.metadata.items():
        if isinstance(value, bool):
            writer.add_bool(key, value)
        elif isinstance(value, (int, np.integer)):
            writer.add_uint32(key, int(value))
        elif isinstance(value, (float, np.floating)):
            writer.add_float32(key, float(value))
        else:
            writer.add_array(key, value)
    for name, tensor in source.tensors.items():
        writer.add_tensor(name, tensor)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def _run_component(
    inputs: list[ir.Value],
    output,
    feeds: dict[str, np.ndarray],
) -> list[np.ndarray]:
    graph = ir.Graph(
        inputs=inputs,
        outputs=[],
        nodes=[],
        name="remaining_dense_component",
        opset_imports={"": OPSET_VERSION},
    )
    builder = GraphBuilder(graph)
    realized = output(builder.op)
    outputs = realized if isinstance(realized, tuple) else (realized,)
    for index, value in enumerate(outputs):
        value.name = f"output_{index}"
        graph.outputs.append(value)
    return list(OnnxModelSession(ir.Model(graph, ir_version=11)).run(feeds).values())


def test_specs_promote_only_explicit_float_graph_routes() -> None:
    expected = {
        "minimax-m2": ("minimax_m2_gguf", MiniMaxM2GGUFCausalLMModel),
        "mistral4": ("mistral4_gguf", Mistral4GGUFCausalLMModel),
        "glm-dsa": ("glm_moe_dsa", registry.get("glm_moe_dsa")),
    }
    for architecture, (model_type, model_class) in expected.items():
        spec = get_arch_spec(architecture)
        assert spec.is_importable
        assert spec.runtime is Support.DEFERRED
        assert spec.quantized_import is Support.REJECTED
        assert spec.model_type == model_type
        assert registry.get(spec.module_type or spec.model_type) is model_class


def test_routing_floor_fingerprint_is_isolated_to_consuming_routes() -> None:
    config = make_config(routing_weight_normalization_floor=6.103515625e-5)
    changed = dataclasses.replace(
        config,
        routing_weight_normalization_floor=0.25,
    )
    for architecture in ("glm-dsa", "minimax-m2", "mistral4"):
        assert _serialize_route_graph_config(
            config,
            architecture,
        ) != _serialize_route_graph_config(changed, architecture)
    assert _serialize_route_graph_config(
        config,
        "llama",
    ) == _serialize_route_graph_config(changed, "llama")


def test_pinned_mistral4_inventory_names_the_deepseek2_loader() -> None:
    record = upstream_architectures()["mistral4"]
    assert record.loader_source == "src/models/deepseek2.cpp"
    assert record.loader_helpers == ()
    assert "blk.{bid}.attn_kv_a_mqa.weight" in record.tensor_names
    assert "blk.{bid}.ffn_gate_up_exps.weight" in record.tensor_names
    assert "blk.{bid}.attn_qkv.weight" not in record.tensor_names
    assert "rope_factors_long.weight" not in record.tensor_names


@pytest.mark.parametrize(
    ("architecture", "gguf_name", "hf_name"),
    [
        (
            "minimax-m2",
            "blk.4.exp_probs_b.bias",
            "model.layers.4.mlp.gate.e_score_correction_bias",
        ),
        (
            "minimax-m2",
            "blk.4.ffn_gate_exps.weight",
            "model.layers.4.mlp.experts.gate_proj.weight",
        ),
        (
            "mistral4",
            "blk.4.attn_k_b.weight",
            "model.layers.4.self_attn.k_b_proj.weight",
        ),
        (
            "mistral4",
            "blk.4.ffn_gate_up_exps.weight",
            "model.layers.4.mlp.experts.gate_up_proj.weight",
        ),
        (
            "glm-dsa",
            "blk.4.indexer.proj.weight",
            "model.layers.4.self_attn.indexer.weights_proj.weight",
        ),
    ],
)
def test_suffix_exact_tensor_mapping(
    architecture: str,
    gguf_name: str,
    hf_name: str,
) -> None:
    assert map_gguf_to_hf_names(gguf_name, architecture) == hf_name


def test_config_extraction_restores_architecture_discriminators() -> None:
    m2 = gguf_to_config(_m2_fixture())
    assert m2.model_type == "minimax_m2_gguf"
    assert m2.head_dim == 8
    assert m2.partial_rotary_factor == pytest.approx(0.5)
    assert m2.attn_qk_norm and m2.attn_qk_norm_full
    assert m2.scoring_func == "sigmoid"
    assert m2.routing_weight_normalization_floor == pytest.approx(6.103515625e-5)

    m4 = gguf_to_config(_m4_fixture())
    assert m4.model_type == "mistral4_gguf"
    assert m4.num_key_value_heads == 1
    assert (m4.qk_nope_head_dim, m4.qk_rope_head_dim, m4.v_head_dim) == (2, 2, 2)
    assert m4.first_k_dense_replace == 1
    assert m4.scoring_func == "softmax"

    glm = gguf_to_config(_glm_fixture())
    assert glm.model_type == "glm_moe_dsa"
    assert glm.indexer_types == ["full", "shared"]
    assert glm.use_expert_bias
    assert glm.scoring_func == "sigmoid"


def test_mistral4_restores_pinned_yarn_log_multiplier() -> None:
    source = _m4_fixture()
    source.metadata.update(
        {
            "mistral4.rope.scaling.type": "yarn",
            "mistral4.rope.scaling.factor": 4.0,
            "mistral4.rope.scaling.original_context_length": 8,
            "mistral4.rope.scaling.yarn_beta_fast": 32.0,
            "mistral4.rope.scaling.yarn_beta_slow": 1.0,
            "mistral4.rope.scaling.yarn_log_multiplier": 0.0707,
        }
    )
    config = gguf_to_config(source)
    assert config.rope_type == "yarn"
    assert config.rope_scaling is not None
    assert config.rope_scaling["mscale"] == pytest.approx(1.0)
    assert config.rope_scaling["mscale_all_dim"] == pytest.approx(0.707)
    expected_scale = (1.0 + 0.1 * np.log(4.0)) ** 2 / np.sqrt(4.0)
    assert Mistral4LatentAttention(config).scaling == pytest.approx(expected_scale)


def test_glm_dsa_indexer_schedule_preserves_explicit_and_legacy_defaults() -> None:
    source = _glm_fixture()
    assert gguf_to_config(source).indexer_types == ["full", "shared"]

    source.metadata.pop("glm-dsa.attention.indexer.types")
    assert gguf_to_config(source).indexer_types == ["full", "full"]

    source.metadata["glm-dsa.context_length"] = 1_048_576
    assert gguf_to_config(source).indexer_types == ["full", "full"]

    source.metadata["glm-dsa.attention.indexer.types"] = [True]
    with pytest.raises(ValueError, match=r"expected 2, got 1"):
        gguf_to_config(source)


def test_glm_dsa_rejects_unowned_rope_scaling() -> None:
    source = _glm_fixture()
    source.metadata["glm-dsa.rope.scaling.type"] = "yarn"
    with pytest.raises(ValueError, match="unsupported RoPE scaling"):
        gguf_to_config(source)


@pytest.mark.parametrize("factory", [_m2_fixture, _m4_fixture, _glm_fixture])
def test_exact_tensor_closures_accept_synthetic_routes(factory) -> None:
    validate_remaining_dense_tensor_contract(factory())


@pytest.mark.parametrize(
    ("factory", "mutation", "message"),
    [
        (_m2_fixture, "bias", "ignored"),
        (_m2_fixture, "fused", "cannot execute"),
        (_m2_fixture, "missing", "missing="),
        (_m4_fixture, "legacy_mla", "unexpected="),
        (_m4_fixture, "partial_experts", "exactly one fused or split"),
        (_glm_fixture, "shared_indexer", "unexpected="),
        (_glm_fixture, "fused_experts", "unexpected="),
    ],
)
def test_exact_tensor_closures_fail_closed(
    factory,
    mutation: str,
    message: str,
) -> None:
    source = factory()
    if mutation == "bias":
        source.tensors["blk.0.attn_q.bias"] = np.zeros(16, dtype=np.float32)
    elif mutation == "fused":
        source.tensors["blk.0.attn_qkv.weight"] = np.zeros((32, 8), dtype=np.float32)
        for projection in ("q", "k", "v"):
            source.tensors.pop(f"blk.0.attn_{projection}.weight")
    elif mutation == "missing":
        source.tensors.pop("blk.0.ffn_down_exps.weight")
    elif mutation == "legacy_mla":
        source.tensors["blk.0.attn_kv_b.weight"] = np.zeros((8, 4), dtype=np.float32)
    elif mutation == "partial_experts":
        source.tensors.pop("blk.1.ffn_gate_up_exps.weight")
        source.tensors["blk.1.ffn_gate_exps.weight"] = np.zeros((2, 3, 8), dtype=np.float32)
    elif mutation == "shared_indexer":
        source.tensors["blk.1.indexer.k_norm.weight"] = np.ones(4, dtype=np.float32)
    else:
        source.tensors["blk.1.ffn_gate_up_exps.weight"] = np.zeros((2, 6, 8), dtype=np.float32)
    source.tensor_names = list(source.tensors)
    source.qtypes.update(
        {
            name: SimpleNamespace(value=0, name="F32")
            for name in source.tensor_names
            if name not in source.qtypes
        }
    )
    with pytest.raises(ValueError, match=message):
        validate_remaining_dense_tensor_contract(source)


@pytest.mark.parametrize(
    ("factory", "expected_model_type"),
    [
        (_m2_fixture, "minimax_m2_gguf"),
        (_m4_fixture, "mistral4_gguf"),
        (_glm_fixture, "glm_moe_dsa"),
    ],
)
def test_tiny_gguf_closes_graph_and_weight_loading(
    factory,
    expected_model_type: str,
    tmp_path: Path,
) -> None:
    source = factory()
    path = tmp_path / f"{source.architecture}.gguf"
    _write_gguf(path, source)
    package = build_from_gguf(path, keep_quantized=False)

    assert package.config.model_type == expected_model_type
    assert all(
        initializer.const_value is not None
        for initializer in package["model"].graph.initializers.values()
    )
    if source.architecture == "glm-dsa":
        assert (
            "shared-indexer-selection" in package["model"].metadata_props["mobius.cache_abi"]
        )


def test_minimax_m2_prefill_matches_cached_decode(tmp_path: Path) -> None:
    source = _m2_fixture()
    path = tmp_path / "minimax-m2.gguf"
    _write_gguf(path, source)
    model = build_from_gguf(path, keep_quantized=False)["model"]
    session = OnnxModelSession(model)
    tokens = np.array([[2, 5, 7]], dtype=np.int64)
    empty = np.empty((1, 1, 0, 8), dtype=np.float32)

    def run(
        input_ids: np.ndarray,
        past_key: np.ndarray,
        past_value: np.ndarray,
    ):
        total = past_key.shape[2] + input_ids.shape[1]
        return session.run(
            {
                "input_ids": input_ids,
                "attention_mask": np.ones((1, total), dtype=np.int64),
                "position_ids": np.arange(
                    past_key.shape[2],
                    total,
                    dtype=np.int64,
                ).reshape(1, -1),
                "past_key_values.0.key": past_key,
                "past_key_values.0.value": past_value,
            }
        )

    full = run(tokens, empty, empty)
    prefix = run(tokens[:, :2], empty, empty)
    decoded = run(
        tokens[:, 2:],
        prefix["present.0.key"],
        prefix["present.0.value"],
    )
    np.testing.assert_allclose(
        decoded["logits"],
        full["logits"][:, -1:],
        rtol=2e-4,
        atol=2e-5,
    )
    assert full["present.0.key"].shape == (1, 1, 3, 8)
    q_norm = model.graph.initializers["model.layers.0.self_attn.q_norm.weight"]
    assert tuple(q_norm.shape) == (16,)
    q_norm_node = next(
        node
        for node in model.graph
        if node.op_type == "RMSNormalization"
        and node.inputs[1].name == "model.layers.0.self_attn.q_norm.weight"
    )
    assert q_norm_node.inputs[0].producer().op_type == "MatMul"
    rotary_nodes = [node for node in model.graph if node.op_type == "RotaryEmbedding"]
    assert rotary_nodes
    assert {
        int(node.attributes["rotary_embedding_dim"].as_int()) for node in rotary_nodes
    } == {4}


def test_minimax_m2_router_matches_selection_only_bias_and_floor() -> None:
    config = make_config(
        hidden_size=2,
        num_local_experts=3,
        num_experts_per_tok=2,
        scoring_func="sigmoid",
        norm_topk_prob=True,
        routed_scaling_factor=1.0,
        routing_weight_normalization_floor=6.103515625e-5,
    )
    gate = MiniMaxM2Gate(config)
    gate.weight.const_value = ir.tensor(
        np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]], dtype=np.float32)
    )
    gate.e_score_correction_bias.const_value = ir.tensor(
        np.array([-2.0, 1.0, 2.0], dtype=np.float32)
    )
    hidden = ir.Value(
        name="hidden",
        shape=ir.Shape([1, 1, 2]),
        type=ir.TensorType(ir.DataType.FLOAT),
    )
    hidden_values = np.array([[[2.0, 1.0]]], dtype=np.float32)
    actual_weights, actual_experts = _run_component(
        [hidden],
        lambda op: gate(op, hidden),
        {"hidden": hidden_values},
    )

    logits = hidden_values @ gate.weight.const_value.numpy().T
    probabilities = 1.0 / (1.0 + np.exp(-logits))
    bias = gate.e_score_correction_bias.const_value.numpy()
    expected_experts = np.argsort(-(probabilities + bias), axis=-1)[..., :2]
    assert set(actual_experts.reshape(-1)) == set(expected_experts.reshape(-1))
    expected_weights = np.take_along_axis(probabilities, actual_experts, axis=-1)
    expected_weights /= np.maximum(
        expected_weights.sum(axis=-1, keepdims=True),
        6.103515625e-5,
    )
    np.testing.assert_allclose(actual_weights, expected_weights, rtol=1e-6)


def test_minimax_m2_static_cache_uses_standard_kv_contract(tmp_path: Path) -> None:
    source = _m2_fixture()
    path = tmp_path / "minimax-m2-static.gguf"
    _write_gguf(path, source)
    graph = build_from_gguf(
        path,
        keep_quantized=False,
        static_cache=True,
        max_seq_len=8,
    )["model"].graph
    inputs = {value.name: tuple(value.shape) for value in graph.inputs}
    assert inputs["key_cache.0"] == ("batch", 8, 8)
    assert inputs["value_cache.0"] == ("batch", 8, 8)
    assert any(node.op_type == "TensorScatter" for node in graph)


def test_mistral4_latent_cache_prefill_matches_decode(tmp_path: Path) -> None:
    source = _m4_fixture()
    path = tmp_path / "mistral4.gguf"
    _write_gguf(path, source)
    model = build_from_gguf(path, keep_quantized=False)["model"]
    session = OnnxModelSession(model)
    tokens = np.array([[1, 4, 9]], dtype=np.int64)
    empty = np.empty((1, 1, 0, 6), dtype=np.float32)

    def run(input_ids: np.ndarray, past: list[np.ndarray]):
        total = past[0].shape[2] + input_ids.shape[1]
        feeds = {
            "input_ids": input_ids,
            "attention_mask": np.ones((1, total), dtype=np.int64),
            "position_ids": np.arange(
                past[0].shape[2],
                total,
                dtype=np.int64,
            ).reshape(1, -1),
        }
        feeds.update(
            {f"past_key_values.{layer}.key": value for layer, value in enumerate(past)}
        )
        return session.run(feeds)

    full = run(tokens, [empty, empty])
    prefix = run(tokens[:, :2], [empty, empty])
    decoded = run(
        tokens[:, 2:],
        [prefix["present.0.key"], prefix["present.1.key"]],
    )
    np.testing.assert_allclose(
        decoded["logits"],
        full["logits"][:, -1:],
        rtol=2e-4,
        atol=2e-5,
    )
    assert full["present.0.key"].shape == (1, 1, 3, 6)
    assert "present.0.value" not in full
    assert "no-value-cache" in model.metadata_props["mobius.cache_abi"]


def test_mistral4_latent_cache_matches_expanded_mla_reference(
    tmp_path: Path,
) -> None:
    source = _m4_fixture()
    path = tmp_path / "mistral4-reference.gguf"
    _write_gguf(path, source)
    package = build_from_gguf(path, keep_quantized=False)
    latent_model = package["model"]
    config = dataclasses.replace(
        package.config,
        partial_rotary_factor=1.0,
        num_key_value_heads=package.config.num_attention_heads,
    )
    expanded_model = build_from_module(
        DeepSeekV3CausalLMModel(config),
        config,
        CausalLMTask(),
    )["model"]

    latent_initializers = latent_model.graph.initializers
    for name, initializer in expanded_model.graph.initializers.items():
        if initializer.const_value is not None:
            continue
        if name.endswith(".self_attn.kv_b_proj.weight"):
            prefix = name[: -len("kv_b_proj.weight")]
            key = latent_initializers[prefix + "k_b_proj.weight_t"].const_value
            value = latent_initializers[prefix + "v_b_proj.weight_t"].const_value
            assert key is not None and value is not None
            key_array = key.numpy().T.reshape(
                config.num_attention_heads,
                config.qk_nope_head_dim,
                config.kv_lora_rank,
            )
            value_array = value.numpy().T.reshape(
                config.num_attention_heads,
                config.v_head_dim,
                config.kv_lora_rank,
            )
            fused = np.concatenate((key_array, value_array), axis=1).reshape(
                -1,
                config.kv_lora_rank,
            )
            initializer.const_value = ir.tensor(fused, name=name)
            continue
        source_initializer = latent_initializers.get(name)
        transpose = False
        if source_initializer is None and name.endswith(".weight"):
            source_initializer = latent_initializers.get(name + "_t")
            transpose = source_initializer is not None
        assert source_initializer is not None, name
        assert source_initializer.const_value is not None, name
        value = source_initializer.const_value.numpy()
        if transpose:
            value = value.T
        initializer.const_value = ir.tensor(
            value,
            name=name,
        )

    tokens = np.array([[1, 4, 9]], dtype=np.int64)
    attention_mask = np.ones_like(tokens)
    position_ids = np.arange(tokens.shape[1], dtype=np.int64).reshape(1, -1)
    latent = OnnxModelSession(latent_model).run(
        {
            "input_ids": tokens,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "past_key_values.0.key": np.empty((1, 1, 0, 6), dtype=np.float32),
            "past_key_values.1.key": np.empty((1, 1, 0, 6), dtype=np.float32),
        }
    )
    expanded = OnnxModelSession(expanded_model).run(
        {
            "input_ids": tokens,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "past_key_values.0.key": np.empty((1, 2, 0, 4), dtype=np.float32),
            "past_key_values.0.value": np.empty((1, 2, 0, 2), dtype=np.float32),
            "past_key_values.1.key": np.empty((1, 2, 0, 4), dtype=np.float32),
            "past_key_values.1.value": np.empty((1, 2, 0, 2), dtype=np.float32),
        }
    )
    np.testing.assert_allclose(
        latent["logits"],
        expanded["logits"],
        rtol=2e-4,
        atol=2e-5,
    )


def test_glm_dsa_explicit_indexer_schedule_and_dense_kv_reassembly() -> None:
    from mobius.integrations.gguf._tensor_processors import process_tensors

    source = _glm_fixture()
    config = gguf_to_config(source)
    assert config.indexer_types == ["full", "shared"]

    dense_config = SimpleNamespace(
        model_type="glm_moe_dsa",
        _gguf_arch="glm-dsa",
        use_dsa=False,
    )
    state = {
        "model.layers.0.self_attn.k_b_proj.weight": torch.from_numpy(
            source.tensors["blk.0.attn_k_b.weight"]
        ),
        "model.layers.0.self_attn.v_b_proj.weight": torch.from_numpy(
            source.tensors["blk.0.attn_v_b.weight"]
        ),
    }
    processed = process_tensors(state, dense_config)
    assert set(processed) == {"model.layers.0.self_attn.kv_b_proj.weight"}
    assert processed["model.layers.0.self_attn.kv_b_proj.weight"].shape == (8, 4)


@pytest.mark.parametrize("task_name", ["glm-moe-dsa", "mistral4-gguf-text-generation"])
def test_specialized_cache_tasks_reject_static_cache(task_name: str) -> None:
    from mobius.tasks import get_task

    task_class = type(get_task(task_name))
    with pytest.raises(ValueError, match="static cache"):
        task_class(static_cache=True)


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (_glm_fixture, "dsa_kv_cache_specs"),
        (_m4_fixture, "latent K-only cache"),
    ],
)
def test_specialized_gguf_routes_reject_static_cache_before_graph_build(
    factory,
    message: str,
    tmp_path: Path,
) -> None:
    source = factory()
    path = tmp_path / f"{source.architecture}-static.gguf"
    _write_gguf(path, source)
    with pytest.raises(ValueError, match=message):
        build_from_gguf(
            path,
            keep_quantized=False,
            static_cache=True,
            max_seq_len=8,
        )
