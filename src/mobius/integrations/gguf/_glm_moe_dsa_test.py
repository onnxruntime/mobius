# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""GLM-5.2 architecture-bridge and sparse-MoE honesty-gate tests.

The ``glm-dsa`` → ``glm_moe_dsa`` bridge plus the fail-closed sparse-MoE gate.
These cover the two safety rails added for the direct GLM-5.2 GGUF import:

* :func:`resolve_model_type` / :func:`assert_glm_moe_dsa_resolvable` — the
  explicit, metadata-driven format bridge (no filename/model-name heuristics)
  that verifies the head/layer/expert/MLA/DSA properties before the builder
  selects ``GlmMoeDsaCausalLMModel``.
* :func:`_assert_sparse_moe_capability` — the fail-closed gate that refuses to
  export routed IQ-block experts that would lower to per-expert
  ``BlockQuantizedMatMul`` nodes (dense-all-expert compute) with no sparse
  fusion.
"""

from __future__ import annotations

import dataclasses

import pytest


class _FakeGlmDsaModel:
    architecture = "glm-dsa"

    def __init__(self, md: dict) -> None:
        self.metadata = md

    def get_metadata(self, key, default=None):
        return self.metadata.get(key, default)

    @property
    def tensor_names(self) -> list[str]:
        return [
            "output.weight",
            "blk.0.attn_q.weight",
            *(f"blk.{layer}.exp_probs_b.bias" for layer in range(3, 78)),
        ]


def _valid_glm_dsa_metadata() -> dict:
    return {
        "glm-dsa.embedding_length": 6144,
        "glm-dsa.context_length": 1048576,
        "glm-dsa.block_count": 79,
        "glm-dsa.nextn_predict_layers": 1,
        "glm-dsa.attention.head_count": 64,
        "glm-dsa.attention.head_count_kv": 1,
        "glm-dsa.attention.layer_norm_rms_epsilon": 1e-5,
        "glm-dsa.feed_forward_length": 12288,
        "glm-dsa.vocab_size": 154880,
        "glm-dsa.expert_count": 256,
        "glm-dsa.expert_used_count": 8,
        "glm-dsa.expert_feed_forward_length": 2048,
        "glm-dsa.expert_shared_count": 1,
        "glm-dsa.expert_gating_func": 2,
        "glm-dsa.expert_group_count": 1,
        "glm-dsa.expert_group_used_count": 1,
        "glm-dsa.expert_weights_norm": True,
        "glm-dsa.expert_weights_scale": 2.5,
        "glm-dsa.leading_dense_block_count": 3,
        "glm-dsa.attention.q_lora_rank": 2048,
        "glm-dsa.attention.kv_lora_rank": 512,
        "glm-dsa.attention.key_length": 576,
        "glm-dsa.attention.key_length_mla": 256,
        "glm-dsa.attention.value_length": 512,
        "glm-dsa.attention.value_length_mla": 256,
        "glm-dsa.rope.dimension_count": 64,
        "glm-dsa.rope.freq_base": 8_000_000.0,
        "glm-dsa.attention.indexer.head_count": 32,
        "glm-dsa.attention.indexer.key_length": 128,
        "glm-dsa.attention.indexer.top_k": 2048,
    }


# --------------------------------------------------------------------------- #
# resolve_model_type — the format bridge
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "arch,expected",
    [
        ("glm-dsa", "glm_moe_dsa"),
        ("glm_dsa", "glm_moe_dsa"),
        ("llama", "llama"),  # unknown/passthrough unchanged
        ("qwen2", "qwen2"),
    ],
)
def test_resolve_model_type(arch, expected):
    from mobius.integrations.gguf._config_mapping import resolve_model_type

    assert resolve_model_type(arch) == expected


def test_glm_dsa_config_resolves_to_glm_moe_dsa():
    from mobius.integrations.gguf._config_mapping import gguf_to_config, resolve_model_type

    config = gguf_to_config(_FakeGlmDsaModel(_valid_glm_dsa_metadata()))
    model_type = getattr(config, "_gguf_model_type", None) or resolve_model_type("glm-dsa")
    assert model_type == "glm_moe_dsa"


def test_glm_dsa_config_matches_official_checkpoint_geometry():
    from mobius.integrations.gguf._config_mapping import gguf_to_config

    config = gguf_to_config(_FakeGlmDsaModel(_valid_glm_dsa_metadata()))

    assert config.num_hidden_layers == 78
    assert config.num_attention_heads == 64
    assert config.num_key_value_heads == 64
    assert config.first_k_dense_replace == 3
    assert config.q_lora_rank == 2048
    assert config.kv_lora_rank == 512
    assert config.qk_nope_head_dim == 192
    assert config.qk_rope_head_dim == 64
    assert config.v_head_dim == 256
    assert config.scoring_func == "sigmoid"
    assert config.topk_method == "noaux_tc"
    assert config.use_expert_bias is True
    assert config.index_topk_freq == 4
    assert config.index_skip_topk_offset == 3
    assert len(config.indexer_types) == 78
    assert config.indexer_types[:10] == [
        "full",
        "full",
        "full",
        "shared",
        "shared",
        "shared",
        "full",
        "shared",
        "shared",
        "shared",
    ]


@pytest.mark.parametrize(
    ("gguf_name", "hf_name"),
    [
        ("blk.4.attn_k_b.weight", "model.layers.4.self_attn.k_b_proj.weight"),
        ("blk.4.attn_v_b.weight", "model.layers.4.self_attn.v_b_proj.weight"),
        ("blk.4.indexer.attn_k.weight", "model.layers.4.self_attn.indexer.wk.weight"),
        ("blk.4.indexer.attn_q_b.weight", "model.layers.4.self_attn.indexer.wq_b.weight"),
        (
            "blk.4.indexer.proj.weight",
            "model.layers.4.self_attn.indexer.weights_proj.weight",
        ),
        (
            "blk.4.ffn_gate_exps.weight",
            "model.layers.4.mlp.experts.gate_proj.weight",
        ),
        (
            "blk.4.ffn_down_shexp.weight",
            "model.layers.4.mlp.shared_experts.down_proj.weight",
        ),
        (
            "blk.4.exp_probs_b.bias",
            "model.layers.4.mlp.gate.e_score_correction_bias",
        ),
    ],
)
def test_glm_dsa_tensor_mapping(gguf_name, hf_name):
    from mobius.integrations.gguf._tensor_mapping import map_gguf_to_hf_names

    assert map_gguf_to_hf_names(gguf_name, "glm-dsa") == hf_name


# --------------------------------------------------------------------------- #
# assert_glm_moe_dsa_resolvable — valid / invalid
# --------------------------------------------------------------------------- #


def test_valid_glm_dsa_config_passes():
    from mobius.integrations.gguf._config_mapping import (
        assert_glm_moe_dsa_resolvable,
        gguf_to_config,
    )

    config = gguf_to_config(_FakeGlmDsaModel(_valid_glm_dsa_metadata()))
    assert_glm_moe_dsa_resolvable(config, "glm-dsa", source="valid.gguf")  # no raise


def test_missing_expert_count_rejected():
    from mobius.integrations.gguf._config_mapping import gguf_to_config

    md = _valid_glm_dsa_metadata()
    del md["glm-dsa.expert_count"]
    with pytest.raises(ValueError, match=r"(?i)expert"):
        gguf_to_config(_FakeGlmDsaModel(md))


def test_missing_sigmoid_gate_uses_pinned_loader_default():
    from mobius.integrations.gguf._config_mapping import gguf_to_config

    md = _valid_glm_dsa_metadata()
    del md["glm-dsa.expert_gating_func"]
    config = gguf_to_config(_FakeGlmDsaModel(md))
    assert config.scoring_func == "sigmoid"


def test_missing_mla_rank_rejected():
    from mobius.integrations.gguf._config_mapping import gguf_to_config

    md = _valid_glm_dsa_metadata()
    del md["glm-dsa.attention.q_lora_rank"]
    del md["glm-dsa.attention.kv_lora_rank"]
    with pytest.raises(ValueError, match=r"(?i)q_lora|kv_lora"):
        gguf_to_config(_FakeGlmDsaModel(md))


def test_missing_dsa_indexer_rejected():
    from mobius.integrations.gguf._config_mapping import gguf_to_config

    md = _valid_glm_dsa_metadata()
    del md["glm-dsa.attention.indexer.head_count"]
    del md["glm-dsa.attention.indexer.key_length"]
    del md["glm-dsa.attention.indexer.top_k"]
    with pytest.raises(ValueError, match=r"(?i)indexer"):
        gguf_to_config(_FakeGlmDsaModel(md))


def test_rejection_lists_all_reasons():
    # A bare decoder mislabelled 'glm-dsa' should report every missing property,
    # not just the first (precise rejection reasons).
    from mobius.integrations.gguf._config_mapping import (
        GgufArchResolutionError,
        assert_glm_moe_dsa_resolvable,
        gguf_to_config,
    )

    config = gguf_to_config(_FakeGlmDsaModel(_valid_glm_dsa_metadata()))
    config = dataclasses.replace(
        config,
        num_local_experts=None,
        q_lora_rank=None,
        kv_lora_rank=None,
        index_n_heads=None,
        index_head_dim=None,
        index_topk=None,
    )
    with pytest.raises(GgufArchResolutionError) as excinfo:
        assert_glm_moe_dsa_resolvable(config, "glm-dsa", source="bare_decoder.gguf")
    message = str(excinfo.value)
    assert "expert" in message.lower()
    assert "mla" in message.lower() or "latent" in message.lower()
    assert "indexer" in message.lower()


# --------------------------------------------------------------------------- #
# Sparse-MoE honesty gate
# --------------------------------------------------------------------------- #


def _moe_module_with_block_experts(num_experts: int = 3):
    from onnxscript import nn

    from mobius.components import BlockQuantizedLinear

    class _Experts(nn.Module):
        def __init__(self, n):
            super().__init__()
            self.experts = nn.ModuleList(
                [BlockQuantizedLinear(256, 8, format="iq1_s") for _ in range(n)]
            )

    class _MLP(nn.Module):
        def __init__(self, n):
            super().__init__()
            self.moe = _Experts(n)
            self.shared_experts = BlockQuantizedLinear(256, 8, format="iq1_s")

    class _Root(nn.Module):
        def __init__(self, n):
            super().__init__()
            self.layers = nn.ModuleList([_MLP(n)])

    return _Root(num_experts)


class _Cfg:
    def __init__(self, num_local_experts):
        self.num_local_experts = num_local_experts


def test_routed_dense_block_expert_paths_finds_experts():
    from mobius.integrations.gguf._builder import _routed_dense_block_expert_paths

    module = _moe_module_with_block_experts(num_experts=3)
    paths = _routed_dense_block_expert_paths(module)
    assert len(paths) == 3
    assert all(".experts." in p for p in paths)
    # Shared experts are explicitly excluded (they are not routed).
    assert all("shared_expert" not in p for p in paths)


def test_honesty_gate_blocks_iq1_moe_by_default():
    from mobius.integrations.gguf._builder import (
        SparseMoEExportError,
        _assert_sparse_moe_capability,
    )

    module = _moe_module_with_block_experts(num_experts=4)
    with pytest.raises(SparseMoEExportError, match=r"(?i)sparse|dense-all-expert"):
        _assert_sparse_moe_capability(module, _Cfg(4), source="glm52.gguf", allow_dense=False)


def test_honesty_gate_error_points_to_next_slice():
    from mobius.integrations.gguf._builder import (
        SparseMoEExportError,
        _assert_sparse_moe_capability,
    )

    module = _moe_module_with_block_experts(num_experts=2)
    with pytest.raises(SparseMoEExportError) as excinfo:
        _assert_sparse_moe_capability(module, _Cfg(2), source="glm52.gguf", allow_dense=False)
    assert "BlockQuantizedMoE" in str(excinfo.value)


def test_honesty_gate_allows_when_opted_in():
    from mobius.integrations.gguf._builder import _assert_sparse_moe_capability

    module = _moe_module_with_block_experts(num_experts=3)
    # allow_dense=True proceeds (research/correctness path) without raising.
    _assert_sparse_moe_capability(module, _Cfg(3), source="glm52.gguf", allow_dense=True)


def test_honesty_gate_noop_for_non_moe():
    from mobius.integrations.gguf._builder import _assert_sparse_moe_capability

    module = _moe_module_with_block_experts(num_experts=3)
    # num_local_experts=0 => not treated as MoE, gate is a no-op.
    _assert_sparse_moe_capability(module, _Cfg(0), source="x.gguf", allow_dense=False)


def test_allow_dense_moe_flag_defaults_false():
    from mobius._flags import flags

    assert flags.allow_dense_moe_experts is False
