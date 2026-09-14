# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Unit tests for the GraniteSWA architecture (``granite_swa``).

Covers the three ways GraniteSWA departs from Granite — mixed
full/sliding attention spans, learnable per-head attention sinks, and a
per-layer RoPE base with ``0`` meaning NoPE — plus the config extraction
that feeds them.
"""

from __future__ import annotations

import dataclasses
import types

import numpy as np
import onnx_ir as ir
import pytest
import torch

from mobius._configs import ArchitectureConfig, GraniteSwaConfig, QuantizationConfig
from mobius._registry import registry
from mobius._testing import count_op_type, create_test_builder, create_test_input
from mobius._testing.ort_inference import OnnxModelSession
from mobius.components import (
    Float32SinkAttention,
    Float32SlidingWindowSinkAttention,
    QuantizedEmbedding,
    SinkAttention,
    TiedQuantizedLMHead,
)
from mobius.components._attention import GQAContext
from mobius.models.granite_swa import (
    GraniteSwaCausalLMModel,
    GraniteSwaTextModel,
    resolve_layer_rope_theta,
)
from mobius.tasks import CausalLMTask, get_task

# The pinned ``ibm-granite/granite-swash-2b`` config.json contents at revision
# af1e3227100b61088eead48389ab5409b5d0e39c, inlined so the extraction test
# needs no network access.
PINNED_REVISION = "af1e3227100b61088eead48389ab5409b5d0e39c"
_PINNED_CONFIG_JSON: dict = {
    "model_type": "granite_swa",
    "architectures": ["GraniteSWAForCausalLM"],
    "vocab_size": 100352,
    "hidden_size": 2560,
    "intermediate_size": 8192,
    "num_hidden_layers": 24,
    "num_attention_heads": 20,
    "num_key_value_heads": 4,
    "hidden_act": "silu",
    "max_position_embeddings": 8192,
    "rms_norm_eps": 1e-05,
    "attention_bias": False,
    "mlp_bias": False,
    "tie_word_embeddings": True,
    "initializer_range": 0.1,
    "bos_token_id": 100257,
    "eos_token_id": 100257,
    "pad_token_id": 100256,
    "sliding_window": 128,
    "layer_types": ["full_attention"]
    + ["sliding_attention", "sliding_attention", "sliding_attention", "full_attention"] * 5
    + ["sliding_attention", "sliding_attention", "sliding_attention"],
    "embedding_multiplier": 12,
    "residual_multiplier": 0.28,
    "logits_scaling": 10,
    "attention_multiplier": 0.0078125,
    "rope_theta": 10000,
}


def _tiny_config(_config_cls: type[GraniteSwaConfig] = GraniteSwaConfig, **overrides):
    """A 4-layer GraniteSWA config exercising every per-layer dispatch."""
    fields = dict(
        vocab_size=64,
        max_position_embeddings=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        hidden_act="silu",
        pad_token_id=0,
        rope_type="default",
        rope_theta=10_000.0,
        sliding_window=4,
        layer_types=[
            "full_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention",
        ],
        layer_rope_theta=[10_000.0, 10_000.0, 0, 500_000.0],
        embedding_multiplier=12.0,
        attention_multiplier=0.0078125,
        logits_scaling=10.0,
        residual_multiplier=0.28,
        dtype=ir.DataType.FLOAT,
    )
    fields.update(overrides)
    return _config_cls(**fields)


class TestConfigExtraction:
    """``GraniteSwaConfig.from_transformers`` must match HF ``__post_init__``."""

    def test_pinned_checkpoint_config(self):
        hf_config = types.SimpleNamespace(**_PINNED_CONFIG_JSON)
        config = GraniteSwaConfig.from_transformers(hf_config)

        assert config.num_hidden_layers == 24
        assert config.hidden_size == 2560
        assert config.num_attention_heads == 20
        assert config.num_key_value_heads == 4
        # head_dim is derived: 2560 / 20, matching HF LlamaAttention's default.
        assert config.head_dim == 128
        assert config.sliding_window == 128
        assert config.tie_word_embeddings is True
        assert config.rms_norm_eps == pytest.approx(1e-5)
        # Granite scaling multipliers survive extraction verbatim.
        assert config.embedding_multiplier == pytest.approx(12.0)
        assert config.residual_multiplier == pytest.approx(0.28)
        assert config.logits_scaling == pytest.approx(10.0)
        assert config.attention_multiplier == pytest.approx(0.0078125)
        # Every fourth layer is full attention in this checkpoint.
        assert config.layer_types is not None
        assert len(config.layer_types) == 24
        assert [i for i, t in enumerate(config.layer_types) if t == "full_attention"] == [
            0,
            4,
            8,
            12,
            16,
            20,
        ]
        # No layer_rope_theta in the checkpoint → global theta on every layer.
        assert config.layer_rope_theta == [10_000.0] * 24
        assert config.no_rope_layers == []

    def test_layer_types_default_when_absent(self):
        """HF defaults full attention on every fourth layer."""
        raw = dict(_PINNED_CONFIG_JSON)
        raw.pop("layer_types")
        raw["num_hidden_layers"] = 6
        config = GraniteSwaConfig.from_transformers(types.SimpleNamespace(**raw))

        assert config.layer_types == [
            "full_attention",
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention",
            "sliding_attention",
        ]

    def test_zero_theta_marks_a_nope_layer(self):
        raw = dict(_PINNED_CONFIG_JSON)
        raw["num_hidden_layers"] = 4
        raw["layer_types"] = ["full_attention"] * 4
        raw["layer_rope_theta"] = [10_000.0, 0, 500_000.0, 0]
        config = GraniteSwaConfig.from_transformers(types.SimpleNamespace(**raw))

        assert config.layer_rope_theta == [10_000.0, 0, 500_000.0, 0]
        assert config.no_rope_layers == [1, 3]

    def test_registry_wiring(self):
        assert registry.get("granite_swa") is GraniteSwaCausalLMModel
        assert registry.get_config_class("granite_swa") is GraniteSwaConfig
        # The registry entry and the model class must agree; inheriting
        # CausalLMConfig would silently drop ``layer_rope_theta``.
        assert GraniteSwaCausalLMModel.config_class is GraniteSwaConfig
        registration = registry.get_registration("granite_swa")
        assert registration.test_model_id == "ibm-granite/granite-swash-2b"


class TestResolveLayerRopeTheta:
    def test_explicit_list_wins(self):
        config = _tiny_config()
        assert resolve_layer_rope_theta(config) == [10_000.0, 10_000.0, 0, 500_000.0]

    def test_falls_back_to_no_rope_layers(self):
        """A plain ArchitectureConfig has no ``layer_rope_theta`` field at all."""
        config = ArchitectureConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            rope_type="default",
            rope_theta=10_000.0,
            no_rope_layers=[2],
        )
        assert resolve_layer_rope_theta(config) == [10_000.0, 10_000.0, 0, 10_000.0]

    def test_rotates_every_layer_by_default(self):
        config = ArchitectureConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=3,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            rope_type="default",
            rope_theta=10_000.0,
        )
        assert resolve_layer_rope_theta(config) == [10_000.0] * 3


class TestModuleStructure:
    def test_one_rotary_module_per_distinct_nonzero_theta(self):
        model = GraniteSwaTextModel(_tiny_config())
        # thetas {10_000, 500_000}; the NoPE layer contributes none.
        assert len(model.rotary_embs) == 2
        assert model._rope_thetas == [10_000.0, 500_000.0]
        assert model.rotary_emb is model.rotary_embs[0]

    def test_sink_parameter_names_match_huggingface(self):
        config = _tiny_config()
        module = GraniteSwaCausalLMModel(config)
        package = get_task("text-generation").build(module, config)
        names = set(package["model"].graph.initializers)

        for layer in range(config.num_hidden_layers):
            name = f"model.layers.{layer}.self_attn.sinks"
            assert name in names
            assert list(package["model"].graph.initializers[name].shape) == [
                config.num_attention_heads
            ]

    def test_tied_embeddings_share_one_initializer(self):
        config = _tiny_config(tie_word_embeddings=True)
        module = GraniteSwaCausalLMModel(config)
        package = get_task("text-generation").build(module, config)
        names = set(package["model"].graph.initializers)

        assert "model.embed_tokens.weight" in names
        assert "lm_head.weight" not in names

    def test_attention_uses_the_granite_multiplier_as_scale(self):
        config = _tiny_config()
        module = GraniteSwaCausalLMModel(config)
        for layer in module.model.layers:
            # Float32SinkAttention, not the base SinkAttention: GraniteSWA's
            # eager kernel forces the sink scaling and softmax to float32,
            # whereas GPT-OSS deliberately stays in the compute dtype.
            assert isinstance(layer.self_attn, Float32SinkAttention)
            assert layer.self_attn.upcast_sink_softmax is True
            assert layer.self_attn.scaling == pytest.approx(config.attention_multiplier)

    def test_sliding_layers_retain_only_the_next_step_window(self):
        config = _tiny_config()
        module = GraniteSwaCausalLMModel(config)

        for layer_type, layer in zip(config.layer_types, module.model.layers):
            if layer_type == "sliding_attention":
                assert isinstance(layer.self_attn, Float32SlidingWindowSinkAttention)
                assert layer.self_attn._cache_length == config.sliding_window - 1
            else:
                assert isinstance(layer.self_attn, Float32SinkAttention)
                assert not isinstance(layer.self_attn, Float32SlidingWindowSinkAttention)

    def test_quantized_backbone_uses_quantized_projections_and_tied_table(self):
        config = _tiny_config(
            quantization=QuantizationConfig(
                bits=4,
                group_size=16,
                quant_method="olive",
                quantize_embeddings=True,
                quantize_lm_head=True,
                tie_word_embeddings=True,
            )
        )
        module = GraniteSwaCausalLMModel(config)

        assert isinstance(module.model.embed_tokens, QuantizedEmbedding)
        assert isinstance(module.lm_head, TiedQuantizedLMHead)
        graph = get_task("text-generation").build(module, config)["model"].graph
        assert count_op_type(graph, "MatMulNBits") > 0

    def test_output_layer_indices_emit_requested_hidden_states_in_order(self):
        config = _tiny_config(output_layer_indices=[2, 0])
        module = GraniteSwaCausalLMModel(config)
        graph = get_task("text-generation").build(module, config)["model"].graph

        hidden_outputs = [
            output.name for output in graph.outputs if output.name.startswith("hidden_states.")
        ]
        assert hidden_outputs == ["hidden_states.2", "hidden_states.0"]

    def test_static_cache_is_rejected_before_graph_construction(self):
        config = _tiny_config()
        module = GraniteSwaCausalLMModel(config)
        with pytest.raises(
            ValueError, match="GraniteSwaCausalLMModel does not support static cache"
        ):
            CausalLMTask(static_cache=True).build(module, config)

    def test_nope_layer_skips_rotary_embedding(self):
        """Three RoPE layers, two RotaryEmbedding nodes each (q, k) = 6, not 8."""
        config = _tiny_config()
        module = GraniteSwaCausalLMModel(config)
        package = get_task("text-generation").build(module, config)
        assert count_op_type(package["model"].graph, "RotaryEmbedding") == 6

    def test_all_nope_config_builds_without_rotary_modules(self):
        config = _tiny_config(layer_rope_theta=[0, 0, 0, 0])
        module = GraniteSwaCausalLMModel(config)
        assert len(module.model.rotary_embs) == 0
        assert module.model.rotary_emb is None
        package = get_task("text-generation").build(module, config)
        assert count_op_type(package["model"].graph, "RotaryEmbedding") == 0

    def test_sliding_layers_use_a_distinct_bias_from_full_layers(self):
        config = _tiny_config()
        module = GraniteSwaCausalLMModel(config)
        package = get_task("text-generation").build(module, config)
        graph = package["model"].graph
        # ``create_attention_bias`` only emits a ``Less`` comparison for the
        # window term, so exactly one of the two biases is windowed.
        assert count_op_type(graph, "Less") == 1

    def test_uniform_full_attention_builds_a_single_bias(self):
        config = _tiny_config(
            layer_types=["full_attention"] * 4,
        )
        module = GraniteSwaCausalLMModel(config)
        package = get_task("text-generation").build(module, config)
        assert count_op_type(package["model"].graph, "Less") == 0


class TestSinkAttention:
    """The sink must behave exactly like HF's ``sigmoid(logsumexp - sink)``."""

    def test_rejects_gqa_context(self):
        config = _tiny_config()
        attn = SinkAttention(config)
        builder, op, _ = create_test_builder()
        hidden = create_test_input(builder, "hidden", [1, 4, config.hidden_size])
        ctx = GQAContext(
            seqlens_k=create_test_input(builder, "seqlens_k", [1], ir.DataType.INT32),
            total_seq_len=create_test_input(builder, "tsl", [], ir.DataType.INT32),
            cos_cache=create_test_input(builder, "cos", [32, 4]),
            sin_cache=create_test_input(builder, "sin", [32, 4]),
        )
        with pytest.raises(TypeError, match="cannot emit GroupQueryAttention"):
            attn(op, hidden, attention_bias=ctx)

    def test_rejects_missing_float_attention_bias(self):
        config = _tiny_config()
        attn = SinkAttention(config)
        builder, op, _ = create_test_builder()
        hidden = create_test_input(builder, "hidden", [1, 4, config.hidden_size])
        with pytest.raises(ValueError, match="requires a float additive attention bias"):
            attn(op, hidden, attention_bias=None)

    def test_rejects_softcapping(self):
        @dataclasses.dataclass
        class _SoftcappedConfig(GraniteSwaConfig):
            attn_logit_softcapping: float = 30.0

        config = _tiny_config(_config_cls=_SoftcappedConfig)
        assert config.attn_logit_softcapping == pytest.approx(30.0)
        with pytest.raises(ValueError, match="softcapping"):
            SinkAttention(config)

    def test_emits_no_fused_attention_op(self):
        """The sink lives inside the softmax, so no fused op may appear."""
        config = _tiny_config()
        module = GraniteSwaCausalLMModel(config)
        package = get_task("text-generation").build(module, config)
        graph = package["model"].graph
        assert count_op_type(graph, "Attention") == 0
        assert count_op_type(graph, "GroupQueryAttention") == 0
        assert count_op_type(graph, "Softmax") == config.num_hidden_layers

    @pytest.mark.parametrize(
        "dtype", [ir.DataType.FLOAT16, ir.DataType.BFLOAT16], ids=["f16", "bf16"]
    )
    def test_reduced_precision_upcasts_the_sink_softmax(self, dtype):
        """Mirror upstream's forced-fp32 sink softmax for f16/bf16 builds.

        HuggingFace computes logsumexp in the compute dtype, then upcasts the
        sink scale and token softmax independently. Keeping these paths
        separate preserves its f16/bf16 rounding behavior.
        """
        config = _tiny_config(dtype=dtype)
        module = GraniteSwaCausalLMModel(config)
        graph = get_task("text-generation").build(module, config)["model"].graph

        softmax_nodes = [node for node in graph if node.op_type == "Softmax"]
        assert len(softmax_nodes) == config.num_hidden_layers
        assert count_op_type(graph, "ReduceLogSumExp") == config.num_hidden_layers
        assert count_op_type(graph, "Sigmoid") == config.num_hidden_layers
        for softmax in softmax_nodes:
            # Token scores are directly upcast to float32 for Softmax.
            upcast = softmax.inputs[0].producer()
            assert upcast is not None and upcast.op_type == "Cast"
            assert upcast.attributes["to"].as_int() == ir.DataType.FLOAT
            # The fp32 probabilities are downcast before the value MatMul.
            (downcast, _), *_ = softmax.outputs[0].uses()
            assert downcast.op_type == "Cast"
            assert downcast.attributes["to"].as_int() == dtype

    def test_float32_build_has_no_sink_softmax_casts(self):
        """float32 builds must stay Cast-free around the sink softmax."""
        config = _tiny_config()
        module = GraniteSwaCausalLMModel(config)
        graph = get_task("text-generation").build(module, config)["model"].graph

        softmax_nodes = [node for node in graph if node.op_type == "Softmax"]
        assert len(softmax_nodes) == config.num_hidden_layers
        assert count_op_type(graph, "ReduceLogSumExp") == config.num_hidden_layers
        for softmax in softmax_nodes:
            assert softmax.inputs[0].producer().op_type != "Cast"
            assert all(use[0].op_type != "Cast" for use in softmax.outputs[0].uses())

    def test_matches_the_logsumexp_sigmoid_reference(self):
        """Run one SinkAttention layer in ORT against the upstream formula.

        HuggingFace ``granite_swa.eager_attention_forward`` computes an
        ordinary softmax and then rescales the output by
        ``sigmoid(logsumexp(scores) - sink)``.  This implementation instead
        keeps the sink as an extra softmax column.  Both must agree.
        """
        rng = np.random.default_rng(0)
        config = _tiny_config(num_hidden_layers=1)
        num_heads = config.num_attention_heads
        num_kv_heads = config.num_key_value_heads
        head_dim = config.head_dim
        hidden_size = config.hidden_size
        batch, seq_len = 1, 5

        attn = Float32SinkAttention(config, scale=config.attention_multiplier)
        builder, op, graph = create_test_builder()
        hidden = create_test_input(builder, "hidden", [batch, seq_len, hidden_size])
        bias = create_test_input(builder, "bias", [batch, 1, seq_len, seq_len])
        output, _ = attn(op, hidden, attention_bias=bias, position_embeddings=None)
        builder._adapt_outputs([output], "")
        output.name = "attn_out"
        graph.outputs.append(output)

        weights = {}
        for name, out_features in (
            ("q_proj", num_heads * head_dim),
            ("k_proj", num_kv_heads * head_dim),
            ("v_proj", num_kv_heads * head_dim),
            ("o_proj", hidden_size),
        ):
            in_features = num_heads * head_dim if name == "o_proj" else hidden_size
            weights[name] = rng.standard_normal((out_features, in_features)).astype(np.float32)
        weights["sinks"] = rng.standard_normal(num_heads).astype(np.float32)

        for value in graph.initializers.values():
            # Skip the folded scalar/shape constants; only the four projection
            # weights and the sink vector are unset parameters.
            if value.const_value is not None:
                continue
            key = "sinks" if value.name.endswith("sinks") else value.name.split(".")[0]
            value.const_value = ir.Tensor(weights[key])

        model = ir.Model(graph, ir_version=10)
        hidden_states = rng.standard_normal((batch, seq_len, hidden_size)).astype(np.float32)
        # Causal float additive bias, matching create_attention_bias's output.
        mask = np.triu(np.full((seq_len, seq_len), np.float32(-3.4e38)), k=1)
        attention_bias = mask.reshape(1, 1, seq_len, seq_len)

        session = OnnxModelSession(model)
        try:
            actual = session.run({"hidden": hidden_states, "bias": attention_bias})
        finally:
            session.close()
        onnx_out = next(iter(actual.values()))

        # --- Upstream reference (torch, sigmoid-LSE form) ---
        hs = torch.from_numpy(hidden_states)
        query = (
            (hs @ torch.from_numpy(weights["q_proj"]).T)
            .view(batch, seq_len, num_heads, head_dim)
            .transpose(1, 2)
        )
        key = (
            (hs @ torch.from_numpy(weights["k_proj"]).T)
            .view(batch, seq_len, num_kv_heads, head_dim)
            .transpose(1, 2)
        )
        value = (
            (hs @ torch.from_numpy(weights["v_proj"]).T)
            .view(batch, seq_len, num_kv_heads, head_dim)
            .transpose(1, 2)
        )
        groups = num_heads // num_kv_heads
        key = key.repeat_interleave(groups, dim=1)
        value = value.repeat_interleave(groups, dim=1)

        scores = (query @ key.transpose(2, 3)) * config.attention_multiplier
        scores = scores + torch.from_numpy(attention_bias)
        lse = torch.logsumexp(scores, dim=-1)  # (B, H, S)
        sink_scale = (
            (lse - torch.from_numpy(weights["sinks"]).view(1, -1, 1))
            .to(torch.float32)
            .sigmoid()
        )
        probs = torch.softmax(scores, dim=-1, dtype=torch.float32).to(scores.dtype)
        expected = probs @ value
        expected = expected * sink_scale.unsqueeze(-1)
        expected = expected.transpose(1, 2).reshape(batch, seq_len, num_heads * head_dim)
        expected = expected @ torch.from_numpy(weights["o_proj"]).T

        np.testing.assert_allclose(onnx_out, expected.numpy(), rtol=1e-4, atol=1e-4)
