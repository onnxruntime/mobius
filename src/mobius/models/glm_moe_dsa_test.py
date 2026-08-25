# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for GLM-5.2 (``glm_moe_dsa``) enablement.

Covers: registry wiring, the ``indexer_types`` full/shared schedule formula
(matching HF ``GlmMoeDsaConfig.__post_init__`` exactly, including the real
``zai-org/GLM-5.2`` ``offset=3, freq=4`` schedule), HF config extraction
(including the ``scoring_func``/``topk_method`` model-type-defaulting fix
for fields that are not real ``GlmMoeDsaConfig`` dataclass fields), the DSA
(``IndexShare``) vs ``--glm-full-attention`` dense-fallback graph-build
structural contract, MTP/indexer weight dropping, and a numeric parity
check of the indexer against the real installed
``transformers.models.glm_moe_dsa`` reference implementation.
"""

from __future__ import annotations

import logging

import numpy as np
import onnx_ir as ir
import onnxruntime as ort
import pytest
import torch

from mobius._builder import build_from_module
from mobius._configs import ArchitectureConfig, QuantizationConfig
from mobius._registry import registry
from mobius._testing import count_op_type, create_test_builder, create_test_input, make_config
from mobius.components._rotary_embedding import initialize_rope
from mobius.models.glm_moe_dsa import (
    GlmMoeDsaCausalLMModel,
    GlmMoeDsaIndexer,
    _indexer_types,
)

transformers = pytest.importorskip("transformers")
from transformers.models.glm_moe_dsa.configuration_glm_moe_dsa import (  # noqa: E402
    GlmMoeDsaConfig,
)
from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import (  # noqa: E402
    GlmMoeDsaIndexer as HFGlmMoeDsaIndexer,
)
from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import (  # noqa: E402
    GlmMoeDsaRotaryEmbedding,
    apply_rotary_pos_emb_interleave,
)


def _glm_config(**overrides) -> ArchitectureConfig:
    """Tiny real-config-derived GLM-5.2 fixture (4 layers, full/shared mix).

    Dimensions are shape-faithful to the pinned ``zai-org/GLM-5.2`` identity
    (MLA + DSA, sigmoid/noaux_tc grouped-topk routing, one shared expert)
    but shrunk to a handful of experts/heads so tests build/run in
    milliseconds. ``indexer_types`` is explicit (``["full", "shared",
    "full", "shared"]``) so structural assertions don't depend on the
    default-schedule formula (covered separately by
    ``test_indexer_types_*``).
    """
    defaults = dict(
        vocab_size=48,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=16,
        max_position_embeddings=32,
        q_lora_rank=24,
        kv_lora_rank=16,
        qk_nope_head_dim=8,
        qk_rope_head_dim=8,
        v_head_dim=16,
        first_k_dense_replace=0,
        num_local_experts=4,
        num_experts_per_tok=2,
        n_group=2,
        topk_group=1,
        n_shared_experts=1,
        moe_intermediate_size=16,
        scoring_func="sigmoid",
        topk_method="noaux_tc",
        norm_topk_prob=True,
        routed_scaling_factor=2.5,
        index_n_heads=2,
        index_head_dim=16,
        index_topk=3,
        indexer_rope_interleave=True,
        indexer_types=["full", "shared", "full", "shared"],
        use_dsa=True,
    )
    defaults.update(overrides)
    return make_config(**defaults)


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------


def test_registry_maps_glm_moe_dsa_to_causal_lm_model():
    assert registry.get("glm_moe_dsa") is GlmMoeDsaCausalLMModel


# --------------------------------------------------------------------------
# _indexer_types() -- full/shared DSA schedule
# --------------------------------------------------------------------------


class TestIndexerTypes:
    def test_explicit_list_is_used_verbatim(self):
        config = _glm_config(num_hidden_layers=3, indexer_types=["full", "full", "shared"])
        assert _indexer_types(config) == ["full", "full", "shared"]

    def test_explicit_list_length_mismatch_raises(self):
        config = _glm_config(num_hidden_layers=4, indexer_types=["full", "shared"])
        with pytest.raises(ValueError, match="one entry per hidden layer"):
            _indexer_types(config)

    def test_default_with_no_freq_offset_is_all_full(self):
        """freq=1 (default) makes every layer 'full', matching HF's default."""
        config = _glm_config(
            num_hidden_layers=4,
            indexer_types=None,
            index_topk_freq=None,
            index_skip_topk_offset=None,
        )
        assert _indexer_types(config) == ["full"] * 4

    def test_default_matches_real_glm_5_2_offset_and_freq(self):
        """Cross-check the schedule formula against a hand-derived expectation.

        offset=3, freq=4 (the real zai-org/GLM-5.2 config) -> full at
        layers 0, 1, 2, 6 for an 8-layer slice, matching
        ``GlmMoeDsaConfig.__post_init__``'s own formula exactly.
        """
        config = _glm_config(
            num_hidden_layers=8,
            indexer_types=None,
            index_topk_freq=4,
            index_skip_topk_offset=3,
        )
        assert _indexer_types(config) == [
            "full",
            "full",
            "full",
            "shared",
            "shared",
            "shared",
            "full",
            "shared",
        ]

    def test_default_matches_hf_post_init_formula_directly(self):
        """Cross-check against the real HF config's own derivation.

        Not just a hand-verified expectation.
        """
        hf_config = GlmMoeDsaConfig(
            num_hidden_layers=8, index_topk_freq=4, index_skip_topk_offset=3
        )
        config = _glm_config(
            num_hidden_layers=8,
            indexer_types=None,
            index_topk_freq=4,
            index_skip_topk_offset=3,
        )
        assert _indexer_types(config) == hf_config.indexer_types


# --------------------------------------------------------------------------
# HF config extraction: scoring_func/topk_method model-type defaulting fix
# --------------------------------------------------------------------------


class TestHFConfigExtraction:
    def test_glm_moe_dsa_defaults_to_sigmoid_noaux_tc_when_omitted(self):
        """Regression test for scoring_func/topk_method model-type defaulting.

        GlmMoeDsaConfig doesn't declare scoring_func/topk_method as real
        dataclass fields, so a config that omits them (unlike the real
        zai-org/GLM-5.2 config.json, which sets them explicitly) must still
        extract GLM's architecturally-fixed sigmoid + noaux_tc routing --
        not silently fall back to DeepSeek's softmax/greedy defaults.
        """
        hf_config = GlmMoeDsaConfig(hidden_size=32, num_hidden_layers=2)
        assert not hasattr(hf_config, "scoring_func")
        assert not hasattr(hf_config, "topk_method")

        config = ArchitectureConfig.from_transformers(hf_config)
        assert config.scoring_func == "sigmoid"
        assert config.topk_method == "noaux_tc"

    def test_explicit_scoring_func_in_config_is_respected(self):
        hf_config = GlmMoeDsaConfig(
            hidden_size=32, num_hidden_layers=2, scoring_func="softmax", topk_method="greedy"
        )
        config = ArchitectureConfig.from_transformers(hf_config)
        assert config.scoring_func == "softmax"
        assert config.topk_method == "greedy"

    def test_non_glm_model_type_keeps_deepseek_defaults(self):
        """The model-type-conditional default must not leak into DeepSeek."""
        from transformers.models.deepseek_v3.configuration_deepseek_v3 import (
            DeepseekV3Config,
        )

        hf_config = DeepseekV3Config(hidden_size=32, num_hidden_layers=2)
        assert not hasattr(hf_config, "scoring_func")
        config = ArchitectureConfig.from_transformers(hf_config)
        assert config.scoring_func == "softmax"
        assert config.topk_method == "greedy"

    def test_extracts_dsa_indexer_fields_and_num_local_experts_alias(self):
        hf_config = GlmMoeDsaConfig(
            hidden_size=32,
            num_hidden_layers=4,
            n_routed_experts=4,
            index_n_heads=2,
            index_head_dim=16,
            index_topk=3,
            index_topk_freq=4,
            index_skip_topk_offset=3,
        )
        config = ArchitectureConfig.from_transformers(hf_config)
        assert config.num_local_experts == 4
        assert config.index_n_heads == 2
        assert config.index_head_dim == 16
        assert config.index_topk == 3
        assert config.index_topk_freq == 4
        assert config.index_skip_topk_offset == 3
        assert config.use_dsa is True  # no HF equivalent -- always defaults True.


# --------------------------------------------------------------------------
# Graph build: DSA (IndexShare) path vs --glm-full-attention dense fallback
# --------------------------------------------------------------------------


class TestGlmMoeDsaGraphBuild:
    def _build(self, **overrides):
        config = _glm_config(**overrides)
        model = GlmMoeDsaCausalLMModel(config)
        package = build_from_module(model, config, task="glm-moe-dsa")
        return config, package["model"]

    def test_dsa_path_emits_one_index_share_per_hidden_layer_and_no_attention(self):
        config, onnx_model = self._build()
        graph = onnx_model.graph
        assert count_op_type(graph, "IndexShare") == config.num_hidden_layers
        assert count_op_type(graph, "Attention") == 0
        assert count_op_type(graph, "GroupQueryAttention") == 0
        assert count_op_type(graph, "MultiHeadAttention") == 0

    def test_dsa_path_uses_packed_single_head_cache_io(self):
        config, onnx_model = self._build()
        cache_inputs = {
            value.name: list(value.shape)
            for value in onnx_model.graph.inputs
            if value.name.startswith("past_key_values.")
        }
        cache_outputs = {
            value.name: list(value.shape)
            for value in onnx_model.graph.outputs
            if value.name.startswith("present.")
        }
        for i, indexer_type in enumerate(config.indexer_types):
            key_width = config.num_attention_heads * (
                config.qk_nope_head_dim + config.qk_rope_head_dim
            ) + (config.index_head_dim if indexer_type == "full" else 0)
            value_width = config.num_attention_heads * config.v_head_dim
            past_key_shape = cache_inputs[f"past_key_values.{i}.key"]
            past_value_shape = cache_inputs[f"past_key_values.{i}.value"]
            present_key_shape = cache_outputs[f"present.{i}.key"]
            present_value_shape = cache_outputs[f"present.{i}.value"]
            assert [past_key_shape[1], past_key_shape[3]] == [1, key_width]
            assert [past_value_shape[1], past_value_shape[3]] == [1, value_width]
            assert [present_key_shape[1], present_key_shape[3]] == [1, key_width]
            assert [present_value_shape[1], present_value_shape[3]] == [1, value_width]

    def test_index_share_inputs_are_float32_under_a_non_float_dtype(self):
        """Regression test for the frozen ``IndexShare`` f32-only schema.

        The op's query/key/value/output are pinned to f32 by
        ``docs/architecture/INDEXSHARE_DESIGN.md`` regardless of the model's
        compute dtype, and the real ``zai-org/GLM-5.2`` checkpoint is
        bfloat16. Build under a non-float dtype and assert every IndexShare
        node's first three inputs resolve to FLOAT so a missing cast (which
        would otherwise only surface as a runtime dtype mismatch under a
        non-default dtype) is caught at graph-build time.
        """
        config, onnx_model = self._build(dtype=ir.DataType.FLOAT16)
        graph = onnx_model.graph
        index_share_nodes = [n for n in graph if n.op_type == "IndexShare"]
        assert len(index_share_nodes) == config.num_hidden_layers
        for node in index_share_nodes:
            query, key, value = node.inputs[0], node.inputs[1], node.inputs[2]
            assert query.dtype == ir.DataType.FLOAT
            assert key.dtype == ir.DataType.FLOAT
            assert value.dtype == ir.DataType.FLOAT

    def test_dsa_path_full_layers_get_indexer_weights_shared_layers_dont(self):
        _config, onnx_model = self._build()
        initializer_names = {i.name for i in onnx_model.graph.initializers.values()}
        for i, indexer_type in enumerate(["full", "shared", "full", "shared"]):
            has_indexer_weight = any(
                f"layers.{i}.self_attn.indexer." in name for name in initializer_names
            )
            assert has_indexer_weight == (indexer_type == "full"), (
                i,
                indexer_type,
                has_indexer_weight,
            )

    def test_full_attention_fallback_emits_dense_attention_and_no_index_share(self):
        config, onnx_model = self._build(use_dsa=False)
        graph = onnx_model.graph
        assert count_op_type(graph, "IndexShare") == 0
        dense_attn_ops = (
            count_op_type(graph, "Attention")
            + count_op_type(graph, "GroupQueryAttention")
            + count_op_type(graph, "MultiHeadAttention")
        )
        assert dense_attn_ops > 0
        initializer_names = {i.name for i in onnx_model.graph.initializers.values()}
        assert not any(".self_attn.indexer." in name for name in initializer_names)
        del config

    def test_moe_layers_use_dense_expert_loop_without_quantization(self):
        """Verify dense per-expert MoE path is used without quantization.

        No quantization configured -> the (already-tested, unchanged)
        DeepSeek dense per-expert MoE path, not QMoE.
        """
        _config, onnx_model = self._build()
        assert count_op_type(onnx_model.graph, "QMoE") == 0

    def test_moe_layers_fuse_to_qmoe_when_quantized(self):
        """Verify quantized MoE layers fuse to a single QMoE node.

        Same QMoE fusion path proven for DeepSeek-V4 (#550) wires
        through unchanged for GLM-5.2's config shape -- one QMoE node per
        MoE layer, no per-expert MatMulNBits loop left over for the routed
        experts (other quantized linears -- attention projections, shared
        expert -- legitimately still lower to MatMulNBits; only the
        *routed-expert* dense loop must disappear).
        """
        config, onnx_model = self._build(
            first_k_dense_replace=0,
            quantization=QuantizationConfig(
                bits=4, group_size=16, quant_method="gptq", sym=True
            ),
        )
        graph = onnx_model.graph
        assert count_op_type(graph, "QMoE") == config.num_hidden_layers
        expert_matmulnbits = [
            node
            for node in graph
            if node.op_type == "MatMulNBits"
            and any(
                ".mlp.moe.experts." in (v.name or "") for v in node.inputs if v is not None
            )
        ]
        assert expert_matmulnbits == []


# --------------------------------------------------------------------------
# preprocess_weights: MTP drop + indexer drop for the dense fallback
# --------------------------------------------------------------------------


class TestPreprocessWeights:
    def _state_dict(self, config: ArchitectureConfig) -> dict[str, torch.Tensor]:
        model = GlmMoeDsaCausalLMModel(config)
        state = {
            name: torch.zeros(tuple(int(d) for d in p.shape))
            for name, p in model.named_parameters()
        }
        # Convert Mobius's own attribute-path names into the HF-style
        # ``model.layers.N.*`` names preprocess_weights expects on input.
        renamed = {
            f"model.{name}" if not name.startswith("lm_head") else name: v
            for name, v in state.items()
        }
        return renamed

    def test_drops_mtp_layer_weights_with_warning(self, caplog):
        config = _glm_config(num_hidden_layers=4)
        model = GlmMoeDsaCausalLMModel(config)
        state = self._state_dict(config)
        # Inject a fake MTP layer (index == num_hidden_layers) as the real
        # transformers checkpoint would ship one.
        state["model.layers.4.input_layernorm.weight"] = torch.zeros(config.hidden_size)
        state["model.layers.4.mlp.gate_proj.weight"] = torch.zeros(
            config.intermediate_size, config.hidden_size
        )

        with caplog.at_level(logging.WARNING):
            out = model.preprocess_weights(state)

        assert not any(k.startswith("layers.4.") for k in out)
        assert any("multi-token-prediction" in r.message for r in caplog.records)

    def test_dense_fallback_drops_indexer_weights(self):
        config = _glm_config(use_dsa=False)
        model = GlmMoeDsaCausalLMModel(config)
        state = self._state_dict(config)
        state["model.layers.0.self_attn.indexer.wk.weight"] = torch.zeros(
            config.index_head_dim, config.hidden_size
        )
        out = model.preprocess_weights(state)
        assert not any(".self_attn.indexer." in k for k in out)

    def test_dsa_path_keeps_indexer_weights(self):
        config = _glm_config(use_dsa=True)
        model = GlmMoeDsaCausalLMModel(config)
        state = self._state_dict(config)
        out = model.preprocess_weights(state)
        assert any(".self_attn.indexer." in k for k in out)


# --------------------------------------------------------------------------
# Numeric parity: GlmMoeDsaIndexer vs the real transformers reference
# --------------------------------------------------------------------------


class TestIndexerNumericParity:
    """Validates the two bugs fixed vs the stale sibling branch.

    The indexer's RoPE split (rope only ``qk_rope_head_dim`` of the wider
    ``index_head_dim``) and the 4D-query x 3D-key MatMul broadcast (which
    silently pairs the wrong axes whenever batch_size != seq_len).

    Raw (post-mask, pre-top-k) index scores are compared exactly against a
    manual PyTorch recompute using the *real* installed
    ``transformers.models.glm_moe_dsa`` RoPE/rotation helpers -- this is
    the strongest available ground truth and is immune to the top-k
    tie-breaking differences between ``torch.topk`` and ONNX ``TopK`` that
    make direct selected-index comparison unreliable on tied/degenerate
    scores (verified empirically: raw scores match to float32 precision
    while ties at the exact top-k boundary can pick different -- but
    equally valid -- indices).
    """

    def _build_indexer_pair(
        self,
        rng,
        batch,
        seq_len,
        hidden_size,
        q_lora_rank,
        index_head_dim,
        index_n_heads,
        qk_rope_head_dim,
        index_topk,
    ):
        hf_config = GlmMoeDsaConfig(
            hidden_size=hidden_size,
            q_lora_rank=q_lora_rank,
            index_head_dim=index_head_dim,
            index_n_heads=index_n_heads,
            qk_rope_head_dim=qk_rope_head_dim,
            index_topk=index_topk,
            num_hidden_layers=1,
            max_position_embeddings=32,
        )
        hf_indexer = HFGlmMoeDsaIndexer(hf_config, layer_idx=0)
        hf_indexer.eval()
        with torch.no_grad():
            for p in hf_indexer.parameters():
                p.copy_(
                    torch.from_numpy(rng.standard_normal(p.shape).astype(np.float32)) * 2.0
                )

        config = make_config(
            hidden_size=hidden_size,
            q_lora_rank=q_lora_rank,
            index_head_dim=index_head_dim,
            index_n_heads=index_n_heads,
            qk_rope_head_dim=qk_rope_head_dim,
            index_topk=index_topk,
            indexer_rope_interleave=True,
            rope_type="default",
            head_dim=qk_rope_head_dim,
            max_position_embeddings=32,
        )
        mobius_indexer = GlmMoeDsaIndexer(config)
        state = dict(hf_indexer.state_dict())
        for name, param in mobius_indexer.named_parameters():
            param.const_value = ir.tensor(state[name].detach().numpy().astype(np.float32))
        return hf_config, hf_indexer, config, mobius_indexer

    def _run_mobius_indexer(
        self, config, mobius_indexer, hidden_np, q_resid_np, position_ids_np, bias_np
    ):
        builder, op, graph = create_test_builder()
        batch, seq_len, hidden_size = hidden_np.shape
        q_lora_rank = q_resid_np.shape[-1]
        hidden_states = create_test_input(
            builder, "hidden_states", [batch, seq_len, hidden_size]
        )
        q_resid = create_test_input(builder, "q_resid", [batch, seq_len, q_lora_rank])
        position_ids = ir.Value(
            name="position_ids",
            shape=ir.Shape([batch, seq_len]),
            type=ir.TensorType(ir.DataType.INT64),
        )
        graph.inputs.append(position_ids)
        attention_bias = create_test_input(
            builder, "attention_bias", [batch, 1, seq_len, seq_len]
        )

        rotary_emb = initialize_rope(config)
        position_embeddings = rotary_emb(op, position_ids)
        _all_keys, indices = mobius_indexer(
            op, hidden_states, q_resid, position_embeddings, None, attention_bias
        )
        indices.name = "indices"
        graph.outputs.append(indices)

        model = ir.Model(graph, ir_version=11)
        session = ort.InferenceSession(
            ir.to_proto(model).SerializeToString(), providers=["CPUExecutionProvider"]
        )
        (indices_out,) = session.run(
            None,
            {
                "hidden_states": hidden_np,
                "q_resid": q_resid_np,
                "position_ids": position_ids_np,
                "attention_bias": bias_np,
            },
        )
        return indices_out

    def _hf_raw_scores(
        self,
        hf_config,
        hf_indexer,
        hidden_np,
        q_resid_np,
        position_ids_np,
        bias_np,
        qk_rope_head_dim,
        index_head_dim,
        index_n_heads,
    ):
        rope = GlmMoeDsaRotaryEmbedding(hf_config)
        with torch.no_grad():
            cos, sin = rope(torch.zeros(1), torch.tensor(position_ids_np))
            hs_t = torch.tensor(hidden_np)
            qr_t = torch.tensor(q_resid_np)
            batch, seq_len, _ = hidden_np.shape
            q = hf_indexer.wq_b(qr_t).view(batch, seq_len, index_n_heads, index_head_dim)
            q_rot, q_pass = torch.split(
                q, [qk_rope_head_dim, index_head_dim - qk_rope_head_dim], dim=-1
            )
            k = hf_indexer.k_norm(hf_indexer.wk(hs_t)).unsqueeze(2)
            k_rot, k_pass = torch.split(
                k, [qk_rope_head_dim, index_head_dim - qk_rope_head_dim], dim=-1
            )
            q_rot, k_rot = apply_rotary_pos_emb_interleave(
                q_rot, k_rot, cos, sin, unsqueeze_dim=2
            )
            q_full = torch.cat([q_rot, q_pass], dim=-1)
            k_full = torch.cat([k_rot, k_pass], dim=-1).squeeze(2)
            scores = (
                torch.matmul(q_full.float(), k_full.transpose(-1, -2).float().unsqueeze(1))
                * hf_indexer.softmax_scale
            )
            scores = torch.relu(scores)
            weights = hf_indexer.weights_proj(hs_t).float() * (index_n_heads**-0.5)
            index_scores = torch.matmul(weights.unsqueeze(-2), scores).squeeze(-2)
            index_scores = index_scores + torch.tensor(bias_np).squeeze(1)
        return index_scores.numpy()

    def test_raw_scores_match_real_hf_reference_batch_ne_seq_len(self):
        """Verify raw indexer scores match HF when batch_size != seq_len.

        batch_size != seq_len on purpose: this is exactly the shape the
        stale sibling branch's un-``unsqueeze``d MatMul broadcast bug
        silently mishandled.
        """
        rng = np.random.default_rng(42)
        batch, seq_len = 3, 6
        hidden_size, q_lora_rank = 8, 6
        index_head_dim, index_n_heads, qk_rope_head_dim, index_topk = 8, 2, 4, 3

        hf_config, hf_indexer, config, mobius_indexer = self._build_indexer_pair(
            rng,
            batch,
            seq_len,
            hidden_size,
            q_lora_rank,
            index_head_dim,
            index_n_heads,
            qk_rope_head_dim,
            index_topk,
        )

        hidden_np = (rng.standard_normal((batch, seq_len, hidden_size)) * 3).astype(np.float32)
        q_resid_np = (rng.standard_normal((batch, seq_len, q_lora_rank)) * 3).astype(
            np.float32
        )
        position_ids_np = np.tile(np.arange(seq_len), (batch, 1)).astype(np.int64)
        bias_np = np.zeros((batch, 1, seq_len, seq_len), dtype=np.float32)
        causal = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)
        bias_np[:, :, causal] = -3.4e38

        expected_scores = self._hf_raw_scores(
            hf_config,
            hf_indexer,
            hidden_np,
            q_resid_np,
            position_ids_np,
            bias_np,
            qk_rope_head_dim,
            index_head_dim,
            index_n_heads,
        )

        # Recompute Mobius's own raw scores (pre-topk) the same way the
        # module does internally, via a patched select() that also emits
        # the score tensor -- mirrors GlmMoeDsaIndexer.select() exactly.
        builder, op, graph = create_test_builder()
        hidden_states = create_test_input(
            builder, "hidden_states", [batch, seq_len, hidden_size]
        )
        q_resid = create_test_input(builder, "q_resid", [batch, seq_len, q_lora_rank])
        position_ids = ir.Value(
            name="position_ids",
            shape=ir.Shape([batch, seq_len]),
            type=ir.TensorType(ir.DataType.INT64),
        )
        graph.inputs.append(position_ids)
        attention_bias = create_test_input(
            builder, "attention_bias", [batch, 1, seq_len, seq_len]
        )
        rotary_emb = initialize_rope(config)
        position_embeddings = rotary_emb(op, position_ids)

        query = mobius_indexer.wq_b(op, q_resid)
        query = op.Reshape(query, [0, 0, mobius_indexer.n_heads, mobius_indexer.head_dim])
        query = mobius_indexer._rope_split(
            op, query, mobius_indexer.n_heads, position_embeddings
        )
        query = op.Cast(query, to=ir.DataType.FLOAT)
        key = mobius_indexer.project_key(op, hidden_states, position_embeddings)
        keys_t = op.Unsqueeze(
            op.Transpose(op.Cast(key, to=ir.DataType.FLOAT), perm=[0, 2, 1]), [1]
        )
        raw = op.Mul(op.MatMul(query, keys_t), mobius_indexer.softmax_scale)
        raw = op.Relu(raw)
        weights = op.Mul(
            op.Cast(mobius_indexer.weights_proj(op, hidden_states), to=ir.DataType.FLOAT),
            mobius_indexer.n_heads**-0.5,
        )
        scores = op.ReduceSum(op.Mul(raw, op.Unsqueeze(weights, [3])), [2], keepdims=False)
        scores = op.Add(scores, op.Cast(op.Squeeze(attention_bias, [1]), to=ir.DataType.FLOAT))
        scores.name = "scores"
        graph.outputs.append(scores)
        model = ir.Model(graph, ir_version=11)
        session = ort.InferenceSession(
            ir.to_proto(model).SerializeToString(), providers=["CPUExecutionProvider"]
        )
        (actual_scores,) = session.run(
            None,
            {
                "hidden_states": hidden_np,
                "q_resid": q_resid_np,
                "position_ids": position_ids_np,
                "attention_bias": bias_np,
            },
        )

        np.testing.assert_allclose(actual_scores, expected_scores, rtol=1e-3, atol=1e-2)

    def test_select_returns_true_top_k_strictly_increasing_indices(self):
        """Verify selected indices form a true top-k, tie-breaking aside.

        Independent of any cross-framework tie-breaking: the selected
        indices must (a) be a valid top-k of Mobius's own scores (every
        selected score >= every non-selected causally-valid score) and
        (b) be strictly increasing, satisfying ``pkg.nxrt::IndexShare``'s
        ascending-index requirement.
        """
        rng = np.random.default_rng(7)
        batch, seq_len = 2, 7
        hidden_size, q_lora_rank = 8, 6
        index_head_dim, index_n_heads, qk_rope_head_dim, index_topk = 8, 2, 4, 3

        _hf_config, _hf_indexer, config, mobius_indexer = self._build_indexer_pair(
            rng,
            batch,
            seq_len,
            hidden_size,
            q_lora_rank,
            index_head_dim,
            index_n_heads,
            qk_rope_head_dim,
            index_topk,
        )
        hidden_np = (rng.standard_normal((batch, seq_len, hidden_size)) * 3).astype(np.float32)
        q_resid_np = (rng.standard_normal((batch, seq_len, q_lora_rank)) * 3).astype(
            np.float32
        )
        position_ids_np = np.tile(np.arange(seq_len), (batch, 1)).astype(np.int64)
        bias_np = np.zeros((batch, 1, seq_len, seq_len), dtype=np.float32)
        causal = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)
        bias_np[:, :, causal] = -3.4e38

        indices_out = self._run_mobius_indexer(
            config, mobius_indexer, hidden_np, q_resid_np, position_ids_np, bias_np
        )

        # Strictly increasing (IndexShare ABI contract).
        assert np.all(np.diff(indices_out, axis=-1) > 0)

        expected_scores = self._hf_raw_scores(
            _hf_config,
            _hf_indexer,
            hidden_np,
            q_resid_np,
            position_ids_np,
            bias_np,
            qk_rope_head_dim,
            index_head_dim,
            index_n_heads,
        )
        for b in range(batch):
            for s in range(seq_len):
                valid = expected_scores[b, s, : s + 1]
                selected = indices_out[b, s]
                selected_scores = expected_scores[b, s, selected]
                non_selected = np.setdiff1d(np.arange(s + 1), selected)
                if non_selected.size == 0:
                    continue
                # Every selected (causally valid) score must be >= every
                # non-selected causally valid score -- true top-k property,
                # tolerant of ties at the boundary.
                assert (
                    selected_scores.min() >= expected_scores[b, s, non_selected].max() - 1e-4
                )
                del valid
