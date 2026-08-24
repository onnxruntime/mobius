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

import pytest


class _FakeGlmDsaModel:
    architecture = "glm-dsa"

    def __init__(self, md: dict) -> None:
        self.metadata = md

    def get_metadata(self, key, default=None):
        return self.metadata.get(key, default)

    @property
    def tensor_names(self) -> list[str]:
        return ["output.weight", "blk.0.attn_q.weight"]


def _valid_glm_dsa_metadata() -> dict:
    return {
        "glm-dsa.embedding_length": 5120,
        "glm-dsa.block_count": 92,
        "glm-dsa.attention.head_count": 96,
        "glm-dsa.attention.head_count_kv": 96,
        "glm-dsa.feed_forward_length": 12288,
        "glm-dsa.vocab_size": 151552,
        "glm-dsa.expert_count": 160,
        "glm-dsa.expert_used_count": 8,
        "glm-dsa.expert_feed_forward_length": 1536,
        "glm-dsa.expert_shared_count": 1,
        "glm-dsa.attention.q_lora_rank": 1536,
        "glm-dsa.attention.kv_lora_rank": 512,
        "glm-dsa.attention.key_length": 128,
        "glm-dsa.rope.dimension_count": 64,
        "glm-dsa.attention.indexer.head_count": 64,
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
    from mobius.integrations.gguf._config_mapping import (
        GgufArchResolutionError,
        assert_glm_moe_dsa_resolvable,
        gguf_to_config,
    )

    md = _valid_glm_dsa_metadata()
    del md["glm-dsa.expert_count"]
    config = gguf_to_config(_FakeGlmDsaModel(md))
    with pytest.raises(GgufArchResolutionError, match=r"(?i)expert"):
        assert_glm_moe_dsa_resolvable(config, "glm-dsa", source="no_experts.gguf")


def test_missing_mla_rank_rejected():
    from mobius.integrations.gguf._config_mapping import (
        GgufArchResolutionError,
        assert_glm_moe_dsa_resolvable,
        gguf_to_config,
    )

    md = _valid_glm_dsa_metadata()
    del md["glm-dsa.attention.q_lora_rank"]
    del md["glm-dsa.attention.kv_lora_rank"]
    config = gguf_to_config(_FakeGlmDsaModel(md))
    with pytest.raises(GgufArchResolutionError, match=r"(?i)MLA|latent|lora"):
        assert_glm_moe_dsa_resolvable(config, "glm-dsa", source="no_mla.gguf")


def test_missing_dsa_indexer_rejected():
    from mobius.integrations.gguf._config_mapping import (
        GgufArchResolutionError,
        assert_glm_moe_dsa_resolvable,
        gguf_to_config,
    )

    md = _valid_glm_dsa_metadata()
    del md["glm-dsa.attention.indexer.head_count"]
    del md["glm-dsa.attention.indexer.key_length"]
    del md["glm-dsa.attention.indexer.top_k"]
    config = gguf_to_config(_FakeGlmDsaModel(md))
    with pytest.raises(GgufArchResolutionError, match=r"(?i)DSA|indexer"):
        assert_glm_moe_dsa_resolvable(config, "glm-dsa", source="no_dsa.gguf")


def test_rejection_lists_all_reasons():
    # A bare decoder mislabelled 'glm-dsa' should report every missing property,
    # not just the first (precise rejection reasons).
    from mobius.integrations.gguf._config_mapping import (
        GgufArchResolutionError,
        assert_glm_moe_dsa_resolvable,
        gguf_to_config,
    )

    md = {
        "glm-dsa.embedding_length": 4096,
        "glm-dsa.block_count": 32,
        "glm-dsa.attention.head_count": 32,
        "glm-dsa.attention.head_count_kv": 8,
        "glm-dsa.feed_forward_length": 11008,
        "glm-dsa.vocab_size": 128000,
    }
    config = gguf_to_config(_FakeGlmDsaModel(md))
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
