# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import onnx_ir as ir
import onnxruntime as ort
import pytest
from onnxscript import GraphBuilder

from mobius._constants import OPSET_VERSION
from mobius._testing import make_config
from mobius.integrations.gguf._arch_registry import get_arch_spec
from mobius.integrations.gguf._builder import (
    _raise_for_invalid_smallthinker_tensor_contract,
)
from mobius.integrations.gguf._config_mapping import gguf_to_config
from mobius.integrations.gguf._spec import Support
from mobius.integrations.gguf._tensor_mapping import map_gguf_to_hf_names
from mobius.models.smallthinker import SmallThinkerGate, SmallThinkerMoELayer
from mobius.tasks import SmallThinkerGGUFCausalLMTask


class _FakeGGUF:
    def __init__(
        self,
        *,
        gating: int | float = 1,
        sliding: bool = False,
        fused_qkv: bool = False,
        expert_weights_scale: float | None = None,
    ):
        hidden, expert_width, layers, vocab = 8, 6, 2, 24
        heads, kv_heads, head_dim, experts = 2, 1, 4, 4
        self.architecture = "smallthinker"
        self.metadata: dict[str, object] = {
            "smallthinker.context_length": 32,
            "smallthinker.embedding_length": hidden,
            "smallthinker.feed_forward_length": expert_width,
            "smallthinker.block_count": layers,
            "smallthinker.attention.head_count": heads,
            "smallthinker.attention.head_count_kv": kv_heads,
            "smallthinker.attention.layer_norm_rms_epsilon": 1e-5,
            "smallthinker.rope.dimension_count": head_dim,
            "smallthinker.rope.freq_base": 10_000.0,
            "smallthinker.vocab_size": vocab,
            "smallthinker.expert_count": experts,
            "smallthinker.expert_used_count": 2,
            "smallthinker.expert_feed_forward_length": expert_width,
            "smallthinker.expert_gating_func": gating,
        }
        if expert_weights_scale is not None:
            self.metadata["smallthinker.expert_weights_scale"] = expert_weights_scale
        if sliding:
            self.metadata.update(
                {
                    "smallthinker.attention.sliding_window": 1024,
                    "smallthinker.attention.sliding_window_pattern": 2,
                    "smallthinker.rope.freq_base_swa": 20_000.0,
                }
            )
        self._tensors: dict[str, tuple[int, ...]] = {
            "token_embd.weight": (vocab, hidden),
            "output_norm.weight": (hidden,),
        }
        q_width, kv_width = heads * head_dim, kv_heads * head_dim
        for layer in range(layers):
            prefix = f"blk.{layer}."
            self._tensors.update(
                {
                    prefix + "attn_norm.weight": (hidden,),
                    prefix + "attn_output.weight": (hidden, q_width),
                    prefix + "ffn_norm.weight": (hidden,),
                    prefix + "ffn_gate_inp.weight": (experts, hidden),
                    prefix + "ffn_gate_exps.weight": (experts, expert_width, hidden),
                    prefix + "ffn_up_exps.weight": (experts, expert_width, hidden),
                    prefix + "ffn_down_exps.weight": (experts, hidden, expert_width),
                }
            )
            if fused_qkv:
                self._tensors[prefix + "attn_qkv.weight"] = (
                    q_width + 2 * kv_width,
                    hidden,
                )
            else:
                self._tensors.update(
                    {
                        prefix + "attn_q.weight": (q_width, hidden),
                        prefix + "attn_k.weight": (kv_width, hidden),
                        prefix + "attn_v.weight": (kv_width, hidden),
                    }
                )
        self.tensor_names = list(self._tensors)
        self.qtypes = {name: SimpleNamespace(value=0, name="F32") for name in self._tensors}

    def get_metadata(self, key: str, default=None):
        return self.metadata.get(key, default)

    def tensor_items_raw(self):
        for name, shape in self._tensors.items():
            yield name, None, self.qtypes[name], shape


@pytest.mark.parametrize("fused_qkv", [False, True])
def test_smallthinker_tensor_closure_and_mapping_are_exact(fused_qkv: bool) -> None:
    model = _FakeGGUF(fused_qkv=fused_qkv)
    _raise_for_invalid_smallthinker_tensor_contract(model)

    assert (
        map_gguf_to_hf_names("blk.0.ffn_gate_inp.weight", "smallthinker")
        == "model.layers.0.mlp.gate.weight"
    )
    assert (
        map_gguf_to_hf_names("blk.0.ffn_gate_exps.weight", "smallthinker")
        == "model.layers.0.mlp.experts.gate_proj.weight"
    )
    assert map_gguf_to_hf_names("blk.0.ffn_gate.weight", "smallthinker") is None


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("missing", "missing="),
        ("partial_qkv", "complete QKV"),
        ("malformed", "malformed="),
        ("unexpected", "unexpected="),
        ("out_of_range", "out_of_range="),
        ("unsupported_qtype", "qtypes without"),
        ("quantized_norm", "normalization tensors"),
    ],
)
def test_smallthinker_tensor_contract_fails_closed(mutation: str, match: str) -> None:
    model = _FakeGGUF()
    if mutation == "missing":
        del model._tensors["blk.0.ffn_up_exps.weight"]
    elif mutation == "partial_qkv":
        del model._tensors["blk.0.attn_v.weight"]
    elif mutation == "malformed":
        model._tensors["blk.0.ffn_down_exps.weight"] = (4, 7, 6)
    elif mutation == "unexpected":
        model._tensors["blk.0.ffn_gate.weight"] = (6, 8)
        model.qtypes["blk.0.ffn_gate.weight"] = SimpleNamespace(value=0, name="F32")
    elif mutation == "out_of_range":
        model._tensors["blk.2.ffn_gate_inp.weight"] = (4, 8)
        model.qtypes["blk.2.ffn_gate_inp.weight"] = SimpleNamespace(value=0, name="F32")
    elif mutation == "unsupported_qtype":
        model.qtypes["blk.0.attn_q.weight"] = SimpleNamespace(value=9, name="Q8_1")
    elif mutation == "quantized_norm":
        model.qtypes["blk.0.attn_norm.weight"] = SimpleNamespace(value=2, name="Q4_0")
    model.tensor_names = list(model._tensors)

    with pytest.raises(ValueError, match=match):
        _raise_for_invalid_smallthinker_tensor_contract(model)


@pytest.mark.parametrize(("gating", "scoring"), [(1, "softmax"), (2, "sigmoid")])
def test_smallthinker_config_restores_routing_and_swa_schedule(
    gating: int, scoring: str
) -> None:
    config = gguf_to_config(_FakeGGUF(gating=gating, sliding=True))

    assert config.model_type == "smallthinker_gguf"
    assert config.hidden_act == "relu"
    assert config.scoring_func == scoring
    assert config.routing_weight_normalization_floor == pytest.approx(6.103515625e-5)
    assert config.sliding_window == 4096
    assert config.layer_types == ["full_attention", "sliding_attention"]
    assert config.no_rope_layers == [0, 1]
    assert config.rope_theta == pytest.approx(10_000.0)
    assert config.rope_local_base_freq == pytest.approx(20_000.0)


@pytest.mark.parametrize(
    ("model", "match"),
    [
        (_FakeGGUF(gating=1.5), "expert_gating_func must be an integer"),
        (
            _FakeGGUF(expert_weights_scale=1.25),
            "expert_weights_scale must be absent or the zero sentinel",
        ),
    ],
)
def test_smallthinker_rejects_unproven_routing_metadata(model: _FakeGGUF, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        gguf_to_config(model)


def test_smallthinker_accepts_loader_zero_scale_sentinel() -> None:
    config = gguf_to_config(_FakeGGUF(expert_weights_scale=0.0))

    assert config.routed_scaling_factor == pytest.approx(1.0)


def test_smallthinker_capabilities_fail_closed_for_quantization_and_runtime() -> None:
    spec = get_arch_spec("smallthinker")

    assert spec.quantized_import is Support.REJECTED
    assert spec.runtime is Support.DEFERRED
    assert spec.module_type == "smallthinker_gguf"
    assert isinstance(SmallThinkerGGUFCausalLMTask(), SmallThinkerGGUFCausalLMTask)


def _run_module(inputs: list[ir.Value], output, feeds, tmp_path):
    graph = ir.Graph(
        inputs=inputs,
        outputs=[],
        nodes=[],
        name="smallthinker_synthetic",
        opset_imports={"": OPSET_VERSION},
    )
    builder = GraphBuilder(graph)
    realized = output(builder.op)
    outputs = realized if isinstance(realized, tuple) else (realized,)
    for index, value in enumerate(outputs):
        value.name = f"output_{index}"
        graph.outputs.append(value)
    model_path = tmp_path / "smallthinker.onnx"
    ir.save(ir.Model(graph, ir_version=11), model_path)
    return ort.InferenceSession(
        model_path,
        providers=["CPUExecutionProvider"],
    ).run(None, feeds)


@pytest.mark.parametrize("scoring_func", ["softmax", "sigmoid"])
def test_smallthinker_gate_matches_both_proven_scoring_modes(
    scoring_func: str, tmp_path
) -> None:
    config = make_config(
        hidden_size=2,
        num_local_experts=3,
        num_experts_per_tok=2,
        scoring_func=scoring_func,
        norm_topk_prob=True,
        routed_scaling_factor=1.25,
        routing_weight_normalization_floor=6.103515625e-5,
    )
    gate = SmallThinkerGate(config)
    weights = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]], dtype=np.float32)
    gate.weight.const_value = ir.tensor(weights)
    hidden = ir.Value(
        name="hidden",
        shape=ir.Shape([1, 1, 2]),
        type=ir.TensorType(ir.DataType.FLOAT),
    )
    hidden_values = np.array([[[2.0, 1.0]]], dtype=np.float32)
    actual_weights, actual_indices = _run_module(
        [hidden],
        lambda op: gate(op, hidden),
        {"hidden": hidden_values},
        tmp_path,
    )

    logits = hidden_values @ weights.T
    scores = (
        1.0 / (1.0 + np.exp(-logits))
        if scoring_func == "sigmoid"
        else np.exp(logits) / np.exp(logits).sum(axis=-1, keepdims=True)
    )
    indices = np.argsort(scores, axis=-1)[..., -2:][..., ::-1]
    selected = np.take_along_axis(scores, indices, axis=-1)
    expected_weights = selected / np.maximum(
        selected.sum(axis=-1, keepdims=True), 6.103515625e-5
    )
    np.testing.assert_array_equal(actual_indices, indices)
    np.testing.assert_allclose(actual_weights, expected_weights * 1.25, rtol=1e-6)


def test_smallthinker_moe_routes_from_unnormalized_input(tmp_path) -> None:
    config = make_config(
        hidden_size=2,
        intermediate_size=1,
        moe_intermediate_size=1,
        num_local_experts=2,
        num_experts_per_tok=1,
        hidden_act="relu",
        scoring_func="softmax",
        norm_topk_prob=True,
        routing_weight_normalization_floor=6.103515625e-5,
        mlp_bias=False,
    )
    moe = SmallThinkerMoELayer(config)
    values = {
        "gate.weight": np.eye(2, dtype=np.float32),
        "experts.0.gate_proj.weight": np.array([[1.0, 0.0]], dtype=np.float32),
        "experts.0.up_proj.weight": np.array([[1.0, 0.0]], dtype=np.float32),
        "experts.0.down_proj.weight": np.array([[1.0], [0.0]], dtype=np.float32),
        "experts.1.gate_proj.weight": np.array([[1.0, 0.0]], dtype=np.float32),
        "experts.1.up_proj.weight": np.array([[1.0, 0.0]], dtype=np.float32),
        "experts.1.down_proj.weight": np.array([[0.0], [2.0]], dtype=np.float32),
    }
    for name, parameter in moe.named_parameters():
        parameter.const_value = ir.tensor(values[name])

    expert_input = ir.Value(
        name="expert_input",
        shape=ir.Shape([2, 1, 2]),
        type=ir.TensorType(ir.DataType.FLOAT),
    )
    router_input = ir.Value(
        name="router_input",
        shape=ir.Shape([2, 1, 2]),
        type=ir.TensorType(ir.DataType.FLOAT),
    )
    actual = _run_module(
        [expert_input, router_input],
        lambda op: moe(op, expert_input, router_input),
        {
            "expert_input": np.ones((2, 1, 2), dtype=np.float32),
            "router_input": np.array(
                [[[10.0, 0.0]], [[0.0, 10.0]]],
                dtype=np.float32,
            ),
        },
        tmp_path,
    )[0]

    np.testing.assert_allclose(actual, [[[1.0, 0.0]], [[0.0, 2.0]]], atol=1e-6)
