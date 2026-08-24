# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Model-level export tests for opt-in ``--paged-attention`` (LATENT dense MLA).

Covers the full DeepSeek-V3 / GLM-5.2 ``--glm-full-attention`` export routed
through ``CausalLMTask(paged_cache=True)`` with ``export_paged_attention=True``:

* feature-off is byte-identical to the current dense-MLA export (the flag is
  default off and every new code path is gated behind it);
* feature-on emits ``com.microsoft::PagedAttention`` v1 in ``LATENT`` mode, one
  node per hidden layer, with the exact op attrs, caller-owned page inputs and
  in-place-aliased latent cache IO;
* feature-on with an inexpressible geometry or an *active* query-dependent
  sparse selection (GLM DSA, DeepSeek-V4 CSA/HCA, MTP, quantized cache,
  head_sink, qk-norm, sliding window) is a typed error at model construction /
  build, never a silent dense fallback.

There is no CPU ``PagedAttention`` kernel and the base onnx checker does not
know the contrib op, so the assertions here are structural (op attrs / inputs /
outputs / model IO). Numerical parity against the decomposed oracle lives in
``mobius/components/_paged_mla_test.py``.
"""

from __future__ import annotations

import dataclasses

import onnx_ir as ir
import pytest
import torch

from mobius._builder import build_from_module
from mobius._testing import count_op_type, make_config
from mobius.models.deepseek import DeepSeekV3CausalLMModel
from mobius.models.glm_moe_dsa import GlmMoeDsaCausalLMModel
from mobius.tasks import CausalLMTask

# Paged-eligible tiny MLA geometry: head_size = kv_lora + rope = 32 (% 16 == 0),
# kv_lora_rank % 16 == 0, qk_rope_head_dim % 16 == 0. Shared by every model here.
_PAGED_GEOMETRY = dict(
    kv_lora_rank=16,
    qk_nope_head_dim=8,
    qk_rope_head_dim=16,
    v_head_dim=16,
    dtype=ir.DataType.FLOAT16,
)


def _glm_full_attention_config(**overrides):
    """Tiny GLM-5.2 ``--glm-full-attention`` config (dense MLA, DSA off)."""
    defaults = dict(
        model_type="glm_moe_dsa",
        vocab_size=48,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=16,
        max_position_embeddings=32,
        q_lora_rank=24,
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
        indexer_types=["full", "shared"],
        use_dsa=False,
    )
    defaults.update(_PAGED_GEOMETRY)
    defaults.update(overrides)
    return make_config(**defaults)


def _deepseek_config(**overrides):
    """Tiny DeepSeek-V3 dense-MLA config (proves eligibility is not name-gated).

    ``use_dsa`` is intentionally left at its ``True`` default: it is vestigial
    for plain DeepSeek-V3 (the dense text model never reads it, and no indexer
    is configured), so the paged export must still qualify — guarding the
    "use_dsa alone must not reject" regression.
    """
    defaults = dict(
        model_type="deepseek_v3",
        vocab_size=48,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=16,
        max_position_embeddings=32,
        q_lora_rank=24,
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
    )
    defaults.update(_PAGED_GEOMETRY)
    defaults.update(overrides)
    return make_config(**defaults)


def _build(model, config):
    return build_from_module(model, config, task=CausalLMTask(paged_cache=True))


def _paged_nodes(graph):
    return [n for n in graph if n.op_type == "PagedAttention"]


# --------------------------------------------------------------------------
# Feature-off: byte-identical dense export
# --------------------------------------------------------------------------


class TestFeatureOff:
    def test_no_paged_nodes_and_dense_attention(self):
        """Feature-off GLM full-attention stays the current dense-MLA graph."""
        config = _glm_full_attention_config(export_paged_attention=False)
        model = GlmMoeDsaCausalLMModel(config)
        graph = build_from_module(model, config, task="glm-moe-dsa")["model"].graph
        assert count_op_type(graph, "PagedAttention") == 0
        assert count_op_type(graph, "IndexShare") == 0
        dense = (
            count_op_type(graph, "Attention")
            + count_op_type(graph, "GroupQueryAttention")
            + count_op_type(graph, "MultiHeadAttention")
        )
        assert dense > 0

    def test_flag_off_is_byte_identical_to_flag_absent(self):
        """Presence of the (off) flag must not perturb the serialized graph."""
        base = _glm_full_attention_config()  # export_paged_attention defaults False
        assert base.export_paged_attention is False
        explicit_off = dataclasses.replace(base, export_paged_attention=False)

        def _bytes(config):
            model = GlmMoeDsaCausalLMModel(config)
            pkg = build_from_module(model, config, task="glm-moe-dsa")
            return ir.to_proto(pkg["model"]).SerializeToString()

        assert _bytes(base) == _bytes(explicit_off)


# --------------------------------------------------------------------------
# Feature-on: LATENT PagedAttention structural contract
# --------------------------------------------------------------------------


class TestFeatureOnStructure:
    def _graph(self, config, model_cls=GlmMoeDsaCausalLMModel):
        model = model_cls(config)
        return _build(model, config)["model"].graph

    def test_glm_emits_one_paged_node_per_layer(self):
        config = _glm_full_attention_config(export_paged_attention=True)
        graph = self._graph(config)
        assert len(_paged_nodes(graph)) == config.num_hidden_layers
        assert count_op_type(graph, "Attention") == 0
        assert count_op_type(graph, "GroupQueryAttention") == 0
        assert count_op_type(graph, "IndexShare") == 0

    def test_deepseek_v3_emits_paged_not_name_gated(self):
        config = _deepseek_config(export_paged_attention=True)
        # Regression: use_dsa defaults to True on every config, but plain
        # DeepSeek-V3 has no indexer configured, so DSA is not active and the
        # paged export must still qualify (eligibility is property-based, not
        # gated on the vestigial use_dsa flag or the model name).
        assert getattr(config, "use_dsa", True) is True
        graph = self._graph(config, model_cls=DeepSeekV3CausalLMModel)
        assert len(_paged_nodes(graph)) == config.num_hidden_layers

    def test_op_attrs_are_latent(self):
        config = _glm_full_attention_config(export_paged_attention=True)
        graph = self._graph(config)
        node = _paged_nodes(graph)[0]
        assert node.domain == "com.microsoft"
        attrs = {a.name: a.value for a in node.attributes.values()}
        assert attrs["kv_cache_layout"] == "LATENT"
        assert attrs["num_heads"] == config.num_attention_heads
        assert attrs["kv_num_heads"] == 1
        assert attrs["v_head_size"] == config.kv_lora_rank
        assert attrs["rotary_offset"] == config.kv_lora_rank
        assert attrs["do_rotary"] == 1
        assert "scale" in attrs
        # rotary_dim is DERIVED from cos_cache last dim, never an attribute.
        assert "rotary_dim" not in attrs

    def test_node_input_latent_contract(self):
        config = _glm_full_attention_config(export_paged_attention=True)
        graph = self._graph(config)
        node = _paged_nodes(graph)[0]
        names = [v.name if v is not None else None for v in node.inputs]
        # value (2) and value_cache (4) are absent in LATENT; key_cache is bound.
        assert names[2] is None
        assert names[3] is not None and names[3].startswith("key_cache.")
        assert names[4] is None
        # key_cache_out (output 1) must alias input 3 in place at runtime.
        assert len(node.outputs) == 3

    def test_model_io_contract(self):
        config = _glm_full_attention_config(export_paged_attention=True)
        graph = self._graph(config)
        n = config.num_hidden_layers

        inputs = {v.name: v for v in graph.inputs}
        expected_inputs = {
            "input_ids",
            "block_table",
            "slot_mapping",
            "cumulative_sequence_length",
            "past_seqlens",
            *(f"key_cache.{i}" for i in range(n)),
        }
        assert set(inputs) == expected_inputs
        # No position_ids: LATENT derives positions from the length tensors.
        assert "position_ids" not in inputs

        assert inputs["block_table"].dtype == ir.DataType.INT32
        assert inputs["slot_mapping"].dtype == ir.DataType.INT32
        assert inputs["cumulative_sequence_length"].dtype == ir.DataType.INT32
        assert inputs["past_seqlens"].dtype == ir.DataType.INT32
        for i in range(n):
            kc = inputs[f"key_cache.{i}"]
            assert kc.dtype == ir.DataType.FLOAT16
            # [num_blocks, block_size, kv_num_heads=1, head_size=l+r]
            assert len(kc.shape) == 4
            assert kc.shape[2] == 1
            assert kc.shape[3] == config.kv_lora_rank + config.qk_rope_head_dim

        outputs = {v.name for v in graph.outputs}
        expected_outputs = {"logits", *(f"updated_key_cache.{i}" for i in range(n))}
        assert outputs == expected_outputs
        # LATENT has a single cache component: no updated_value_cache.
        assert not any(o.startswith("updated_value_cache") for o in outputs)


# --------------------------------------------------------------------------
# Feature-on incompatible: typed error, never silent fallback
# --------------------------------------------------------------------------


class TestFeatureOnRejects:
    def test_dsa_active_rejected_at_construction(self):
        """GLM with DSA on + paged flag must error, never build the DSA graph."""
        config = _glm_full_attention_config(export_paged_attention=True, use_dsa=True)
        with pytest.raises(ValueError, match=r"DSA|IndexShare|dense MLA"):
            GlmMoeDsaCausalLMModel(config)

    def test_ineligible_geometry_rejected_at_construction(self):
        # qk_rope_head_dim=8 violates rotary_dim % 16 == 0.
        config = _deepseek_config(export_paged_attention=True, qk_rope_head_dim=8)
        with pytest.raises(ValueError):
            DeepSeekV3CausalLMModel(config)

    def test_csa_hca_rejected_at_construction(self):
        for override in (
            dict(compress_ratios=[4, 8]),
            dict(o_lora_rank=64),
            dict(o_groups=2),
            dict(hc_mult=2),
        ):
            config = _deepseek_config(export_paged_attention=True, **override)
            with pytest.raises(ValueError):
                DeepSeekV3CausalLMModel(config)

    def test_optional_modes_rejected_at_construction(self):
        """MTP / sliding-window are typed-rejected, not silently miscomputed."""
        with pytest.raises(ValueError, match=r"Multi-Token|num_nextn"):
            DeepSeekV3CausalLMModel(
                _deepseek_config(export_paged_attention=True, num_nextn_predict_layers=1)
            )
        with pytest.raises(ValueError, match="window"):
            DeepSeekV3CausalLMModel(
                _deepseek_config(export_paged_attention=True, sliding_window=64)
            )


# --------------------------------------------------------------------------
# Feature-on weight absorption (torch state dict wrapper)
# --------------------------------------------------------------------------


class TestWeightAbsorption:
    def test_kv_b_proj_absorbed_and_dropped(self):
        """``_absorb_paged_mla_weights`` folds kv_b_proj and rewrites shapes."""
        config = _deepseek_config(export_paged_attention=True, num_hidden_layers=1)
        model = DeepSeekV3CausalLMModel(config)

        nh = config.num_attention_heads
        d = config.qk_nope_head_dim
        r = config.qk_rope_head_dim
        dv = config.v_head_dim
        lat = config.kv_lora_rank
        q_lora = config.q_lora_rank
        hidden = config.hidden_size
        p = "model.layers.0.self_attn"

        torch.manual_seed(0)
        state = {
            f"{p}.q_b_proj.weight": torch.randn(nh * (d + r), q_lora, dtype=torch.float16),
            f"{p}.kv_b_proj.weight": torch.randn(nh * (d + dv), lat, dtype=torch.float16),
            f"{p}.o_proj.weight": torch.randn(hidden, nh * dv, dtype=torch.float16),
        }

        absorbed = model._absorb_paged_mla_weights(state)

        # kv_b_proj is fully folded away.
        assert f"{p}.kv_b_proj.weight" not in absorbed
        # q_b_proj rows grow from nh*(d+r) to nh*(latent+r).
        assert absorbed[f"{p}.q_b_proj.weight"].shape == (nh * (lat + r), q_lora)
        # o_proj columns become nh*latent.
        assert absorbed[f"{p}.o_proj.weight"].shape == (hidden, nh * lat)
        # Results stay torch tensors in the source dtype.
        assert absorbed[f"{p}.q_b_proj.weight"].dtype == torch.float16
        assert absorbed[f"{p}.o_proj.weight"].dtype == torch.float16
