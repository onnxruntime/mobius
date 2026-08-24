# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for paged, absorbed MLA (``com.microsoft::PagedAttention`` LATENT).

The numerical proof is GPU-independent: there is no CPU ``PagedAttention``
kernel, so parity is asserted against a numpy LATENT reference that mirrors the
onnx-genai oracle (``crates/onnx-genai-paged-attention/tests/equivalence.rs``),
comparing the absorbed LATENT path against a decomposed dense-MLA reference over
the full prefill + decode token progression.
"""

from __future__ import annotations

import numpy as np
import onnx_ir as ir
import pytest

from mobius._testing import (
    count_op_type,
    create_test_builder,
    create_test_input,
    make_config,
)
from mobius.components._paged_mla import (
    PagedCacheState,
    PagedLatentMLA,
    absorb_mla_weights,
    mla_paged_geometry,
    paged_attention_eligible,
    paged_attention_rejection,
)

# Tiny geometry that satisfies the LATENT constraints (head_size % 16 == 0,
# kv_lora_rank % 16 == 0, qk_rope_head_dim % 16 == 0).
_TINY = dict(
    hidden_size=64,
    num_attention_heads=2,
    num_key_value_heads=2,
    q_lora_rank=24,
    kv_lora_rank=16,
    qk_nope_head_dim=16,
    qk_rope_head_dim=16,
    v_head_dim=16,
    rope_interleave=False,
    rms_norm_eps=1e-6,
    dtype=ir.DataType.FLOAT16,
    use_dsa=False,
)

# Real GLM-5.2 / DeepSeek-V3 head dims (few heads for test speed).
_GLM = dict(
    hidden_size=256,
    num_attention_heads=4,
    num_key_value_heads=4,
    q_lora_rank=64,
    kv_lora_rank=512,
    qk_nope_head_dim=128,
    qk_rope_head_dim=64,
    v_head_dim=128,
    rope_interleave=True,
    rms_norm_eps=1e-6,
    dtype=ir.DataType.FLOAT16,
    use_dsa=False,
)


def _cfg(**overrides):
    return make_config(**{**_TINY, **overrides})


# ---------------------------------------------------------------------------
# numpy references (mirror equivalence.rs at the Mobius weight level)
# ---------------------------------------------------------------------------


def _rmsnorm(x: np.ndarray, weight: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    var = np.mean(x.astype(np.float64) ** 2, axis=-1, keepdims=True)
    return (x / np.sqrt(var + eps)) * weight


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)


def _build_cos_sin(max_pos: int, rope_dim: int, theta: float = 10000.0):
    half = rope_dim // 2
    inv_freq = 1.0 / (theta ** (np.arange(0, rope_dim, 2, dtype=np.float64) / rope_dim))
    pos = np.arange(max_pos, dtype=np.float64)
    angles = np.outer(pos, inv_freq)  # [max_pos, half]
    assert angles.shape[1] == half
    return np.cos(angles), np.sin(angles)


def _rope1(x: np.ndarray, cos: np.ndarray, sin: np.ndarray, interleaved: bool) -> np.ndarray:
    """Apply RoPE to a single vector ``x`` (dim ``rope_dim``).

    Mirrors ONNX opset-24 ``RotaryEmbedding``; ``cos``/``sin`` are ``rope_dim/2``.
    """
    x = x.astype(np.float64)
    if interleaved:
        x1 = x[0::2]
        x2 = x[1::2]
        o1 = x1 * cos - x2 * sin
        o2 = x1 * sin + x2 * cos
        out = np.empty_like(x)
        out[0::2] = o1
        out[1::2] = o2
        return out
    half = x.shape[-1] // 2
    x1 = x[:half]
    x2 = x[half:]
    return np.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos])


def _decomposed_reference(hidden, w, config, cos, sin):
    """Decomposed dense-MLA oracle (numpy) with full causal attention."""
    geom = mla_paged_geometry(config)
    nh, d, r, dv, l = (  # noqa: E741  (matches the oracle's l/d/r/dv naming)
        geom.num_heads,
        geom.qk_nope_head_dim,
        geom.qk_rope_head_dim,
        geom.v_head_dim,
        geom.kv_lora_rank,
    )
    qh = d + r
    interleaved = geom.rope_interleaved
    scale = geom.scale
    t = hidden.shape[0]

    q_c = _rmsnorm(hidden @ w["q_a"].T, w["q_a_norm"], config.rms_norm_eps)
    q = (q_c @ w["q_b"].T).reshape(t, nh, qh)
    comp = hidden @ w["kv_a"].T
    k_pass = _rmsnorm(comp[:, :l], w["kv_a_norm"], config.rms_norm_eps)
    k_rope = comp[:, l:]
    kv = (k_pass @ w["kv_b"].T).reshape(t, nh, d + dv)
    k_nope = kv[:, :, :d]
    v = kv[:, :, d:]

    out = np.zeros((t, w["o"].shape[0]), dtype=np.float64)
    for tok in range(t):
        concat = np.zeros(nh * dv)
        for h in range(nh):
            q_nope_h = q[tok, h, :d].astype(np.float64)
            q_rope_h = _rope1(q[tok, h, d:], cos[tok], sin[tok], interleaved)
            scores = []
            for j in range(tok + 1):
                kr = _rope1(k_rope[j], cos[j], sin[j], interleaved)
                s = (q_nope_h @ k_nope[j, h] + q_rope_h @ kr) * scale
                scores.append(s)
            p = _softmax(np.array(scores))
            ctx = np.zeros(dv)
            for j in range(tok + 1):
                ctx += p[j] * v[j, h]
            concat[h * dv : (h + 1) * dv] = ctx
        out[tok] = w["o"] @ concat
    return out


def _paged_latent_reference(hidden, w, config, cos, sin):
    """Absorbed LATENT reference (numpy): folds ``kv_b_proj`` via absorption."""
    geom = mla_paged_geometry(config)
    nh, r, l = geom.num_heads, geom.qk_rope_head_dim, geom.kv_lora_rank  # noqa: E741
    hs = l + r
    interleaved = geom.rope_interleaved
    scale = geom.scale
    t = hidden.shape[0]

    absorbed = absorb_mla_weights(
        {"q_b": w["q_b"], "kv_b": w["kv_b"], "o": w["o"]},
        config,
        q_key="q_b",
        kv_b_key="kv_b",
        o_key="o",
    )
    q_b_abs = absorbed["q_b"]
    o_abs = absorbed["o"]

    q_c = _rmsnorm(hidden @ w["q_a"].T, w["q_a_norm"], config.rms_norm_eps)
    q = (q_c @ q_b_abs.T).reshape(t, nh, hs)
    comp = hidden @ w["kv_a"].T
    k_pass = _rmsnorm(comp[:, :l], w["kv_a_norm"], config.rms_norm_eps)
    key = np.concatenate([k_pass, comp[:, l:]], axis=-1)  # [t, hs]

    out = np.zeros((t, o_abs.shape[0]), dtype=np.float64)
    for tok in range(t):
        ctx_latent = np.zeros(nh * l)
        for h in range(nh):
            q_prefix = q[tok, h, :l].astype(np.float64)
            q_suffix = _rope1(q[tok, h, l:], cos[tok], sin[tok], interleaved)
            scores = []
            for j in range(tok + 1):
                k_suffix = _rope1(key[j, l:], cos[j], sin[j], interleaved)
                s = (q_prefix @ key[j, :l] + q_suffix @ k_suffix) * scale
                scores.append(s)
            p = _softmax(np.array(scores))
            ctx = np.zeros(l)
            for j in range(tok + 1):
                ctx += p[j] * key[j, :l]
            ctx_latent[h * l : (h + 1) * l] = ctx
        out[tok] = o_abs @ ctx_latent
    return out


def _random_weights(config, seed=0):
    rng = np.random.default_rng(seed)
    nh = config.num_attention_heads
    d = config.qk_nope_head_dim
    r = config.qk_rope_head_dim
    dv = config.v_head_dim
    l = config.kv_lora_rank  # noqa: E741
    ql = config.q_lora_rank
    h = config.hidden_size

    def rn(*shape, s=1.0):
        return rng.standard_normal(shape) * s / np.sqrt(shape[-1])

    return {
        "q_a": rn(ql, h),
        "q_a_norm": np.ones(ql) + 0.05 * rng.standard_normal(ql),
        "q_b": rn(nh * (d + r), ql),
        "kv_a": rn(l + r, h),
        "kv_a_norm": np.ones(l) + 0.05 * rng.standard_normal(l),
        "kv_b": rn(nh * (d + dv), l),
        "o": rn(h, nh * dv),
    }


# ---------------------------------------------------------------------------
# Eligibility / typed rejections
# ---------------------------------------------------------------------------


class TestEligibility:
    def test_dense_mla_eligible(self):
        assert paged_attention_rejection(_cfg()) is None
        assert paged_attention_eligible(_cfg())
        assert paged_attention_eligible(make_config(**_GLM))

    def test_dsa_indexshare_rejected(self):
        # DSA is active only when use_dsa is set AND an indexer is configured.
        reason = paged_attention_rejection(
            _cfg(use_dsa=True, index_n_heads=2, index_head_dim=16, index_topk=8)
        )
        assert reason is not None
        assert "DSA" in reason or "IndexShare" in reason

    def test_deepseek_v3_default_use_dsa_is_eligible(self):
        # use_dsa defaults True on every config and is vestigial for plain
        # DeepSeek-V2/V3 (no indexer configured); it must NOT trigger a DSA
        # rejection, or --features paged-attention would be unusable for
        # DeepSeek-V3 (its headline target).
        cfg = _cfg(use_dsa=True)  # no index_* fields => not DSA-active
        assert paged_attention_rejection(cfg) is None
        assert paged_attention_eligible(cfg)

    def test_vestigial_indexer_config_is_eligible_when_dsa_off(self):
        # --glm-full-attention drops the indexer weights and builds plain dense
        # MLA, so GLM's still-present indexer config fields (index_topk /
        # indexer_types / index_n_heads / index_head_dim) must NOT reject on
        # their own -- only an *active* DSA (use_dsa=True) is inexpressible.
        assert paged_attention_rejection(_cfg(use_dsa=False, index_topk=64)) is None
        assert paged_attention_rejection(_cfg(use_dsa=False, indexer_types=["full"])) is None
        assert (
            paged_attention_rejection(_cfg(use_dsa=False, index_n_heads=4, index_head_dim=32))
            is None
        )

    def test_deepseek_v4_csa_hca_rejected(self):
        assert paged_attention_rejection(_cfg(compress_ratios=[4, 8])) is not None
        assert paged_attention_rejection(_cfg(hc_mult=2)) is not None
        assert paged_attention_rejection(_cfg(o_lora_rank=64)) is not None
        assert paged_attention_rejection(_cfg(o_groups=2)) is not None

    def test_mtp_rejected(self):
        assert paged_attention_rejection(_cfg(num_nextn_predict_layers=1)) is not None

    def test_quant_and_dtype_rejected(self):
        assert paged_attention_rejection(_cfg(dtype=ir.DataType.FLOAT)) is not None
        assert paged_attention_rejection(_cfg(dtype=ir.DataType.INT8)) is not None
        # bf16 accepted
        assert paged_attention_rejection(_cfg(dtype=ir.DataType.BFLOAT16)) is None

    def test_non_mla_rejected(self):
        cfg = make_config(hidden_size=64, dtype=ir.DataType.FLOAT16)
        assert paged_attention_rejection(cfg) is not None

    def test_geometry_constraints_rejected(self):
        # head_size = kv_lora + rope not divisible by 16.
        assert paged_attention_rejection(_cfg(kv_lora_rank=16, qk_rope_head_dim=4)) is not None
        # rope not divisible by 16.
        assert paged_attention_rejection(_cfg(qk_rope_head_dim=8)) is not None
        # kv_lora not divisible by 16 (8-aligned but not 16-aligned).
        assert paged_attention_rejection(_cfg(kv_lora_rank=8, qk_rope_head_dim=16)) is not None

    def test_mla_geometry_raises_on_ineligible(self):
        with pytest.raises(ValueError, match="DSA"):
            mla_paged_geometry(
                _cfg(use_dsa=True, index_n_heads=2, index_head_dim=16, index_topk=8)
            )


class TestGeometry:
    def test_glm_dims(self):
        g = mla_paged_geometry(make_config(**_GLM))
        assert g.head_size == 576
        assert g.v_head_size == 512
        assert g.rotary_offset == 512
        assert g.rotary_dim == 64
        assert g.qk_head_dim == 192
        assert g.scale == pytest.approx(192**-0.5)

    def test_tiny_dims(self):
        g = mla_paged_geometry(_cfg())
        assert g.head_size == 32
        assert g.v_head_size == 16
        assert g.rotary_offset == 16
        assert g.rotary_dim == 16
        assert g.qk_head_dim == 32


# ---------------------------------------------------------------------------
# Weight absorption
# ---------------------------------------------------------------------------


class TestAbsorption:
    def test_absorbed_shapes(self):
        config = _cfg()
        w = _random_weights(config)
        out = absorb_mla_weights(
            {"q_b": w["q_b"], "kv_b": w["kv_b"], "o": w["o"]},
            config,
            q_key="q_b",
            kv_b_key="kv_b",
            o_key="o",
        )
        nh, l, r = (  # noqa: E741
            config.num_attention_heads,
            config.kv_lora_rank,
            config.qk_rope_head_dim,
        )
        assert out["q_b"].shape == (nh * (l + r), config.q_lora_rank)
        assert out["o"].shape == (config.hidden_size, nh * l)
        assert "kv_b" not in out  # fully absorbed / dropped

    def test_absorption_rejects_bad_shapes(self):
        config = _cfg()
        w = _random_weights(config)
        bad = dict(w)
        bad["kv_b"] = w["kv_b"][:-1]  # wrong rows
        with pytest.raises(ValueError, match="kv_b"):
            absorb_mla_weights(
                {"q_b": w["q_b"], "kv_b": bad["kv_b"], "o": w["o"]},
                config,
                q_key="q_b",
                kv_b_key="kv_b",
                o_key="o",
            )


# ---------------------------------------------------------------------------
# Numerical parity: absorbed LATENT == decomposed dense MLA
# ---------------------------------------------------------------------------


class TestNumericalParity:
    @pytest.mark.parametrize("interleaved", [False, True])
    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_tiny_parity_full_sequence(self, interleaved, seed):
        config = _cfg(rope_interleave=interleaved)
        w = _random_weights(config, seed=seed)
        t = 5
        hidden = np.random.default_rng(100 + seed).standard_normal((t, config.hidden_size))
        cos, sin = _build_cos_sin(32, config.qk_rope_head_dim)
        dec = _decomposed_reference(hidden, w, config, cos, sin)
        pag = _paged_latent_reference(hidden, w, config, cos, sin)
        rel = np.max(np.abs(dec - pag)) / (np.max(np.abs(dec)) + 1e-9)
        assert rel < 1e-6, f"rel diff {rel}"

    def test_glm_dims_parity(self):
        config = make_config(**_GLM)
        w = _random_weights(config, seed=7)
        t = 4
        hidden = np.random.default_rng(7).standard_normal((t, config.hidden_size))
        cos, sin = _build_cos_sin(32, config.qk_rope_head_dim)
        dec = _decomposed_reference(hidden, w, config, cos, sin)
        pag = _paged_latent_reference(hidden, w, config, cos, sin)
        rel = np.max(np.abs(dec - pag)) / (np.max(np.abs(dec)) + 1e-9)
        assert rel < 1e-6, f"rel diff {rel}"

    def test_prefill_then_decode_progression(self):
        """Per-token equivalence == prefill + incremental decode equivalence."""
        config = _cfg()
        w = _random_weights(config, seed=3)
        t = 6
        hidden = np.random.default_rng(3).standard_normal((t, config.hidden_size))
        cos, sin = _build_cos_sin(32, config.qk_rope_head_dim)
        dec = _decomposed_reference(hidden, w, config, cos, sin)
        pag = _paged_latent_reference(hidden, w, config, cos, sin)
        # Prefill (all but last) and each decode token must match row-wise.
        for tok in range(t):
            rel = np.max(np.abs(dec[tok] - pag[tok])) / (np.max(np.abs(dec[tok])) + 1e-9)
            assert rel < 1e-6, f"token {tok} rel diff {rel}"


# ---------------------------------------------------------------------------
# Structural op emission
# ---------------------------------------------------------------------------


def _build_paged_graph(config):
    mla = PagedLatentMLA(config)
    builder, op, graph = create_test_builder()
    geom = mla.geom
    dt = config.dtype
    hidden = create_test_input(builder, "hidden", [1, 4, config.hidden_size], dt)
    num_blocks, block_size = 2, 16
    cache = PagedCacheState(
        key_cache=create_test_input(
            builder, "key_cache", [num_blocks, block_size, 1, geom.head_size], dt
        ),
        block_table=create_test_input(builder, "block_table", [1, 2], ir.DataType.INT32),
        slot_mapping=create_test_input(builder, "slot_mapping", [4], ir.DataType.INT32),
        cumulative_sequence_length=create_test_input(
            builder, "cu_seqlens", [2], ir.DataType.INT32
        ),
        past_seqlens=create_test_input(builder, "past_seqlens", [1], ir.DataType.INT32),
        cos_cache=create_test_input(builder, "cos_cache", [32, geom.rotary_dim // 2], dt),
        sin_cache=create_test_input(builder, "sin_cache", [32, geom.rotary_dim // 2], dt),
    )
    output, (key_cache_out,) = mla(op, hidden, cache)
    builder._adapt_outputs([output, key_cache_out], "")
    return graph, geom


def _find_paged_node(graph):
    for node in graph:
        if node.op_type == "PagedAttention":
            return node
    return None


class TestStructuralEmission:
    def test_emits_single_paged_attention_node(self):
        graph, _ = _build_paged_graph(_cfg())
        assert count_op_type(graph, "PagedAttention") == 1
        assert count_op_type(graph, "Attention") == 0

    def test_node_domain_and_attrs(self):
        graph, geom = _build_paged_graph(_cfg())
        node = _find_paged_node(graph)
        assert node is not None
        assert node.domain == "com.microsoft"
        attrs = {a.name: a.value for a in node.attributes.values()}
        assert attrs["kv_cache_layout"] == "LATENT"
        assert attrs["num_heads"] == geom.num_heads
        assert attrs["kv_num_heads"] == 1
        assert attrs["v_head_size"] == geom.v_head_size
        assert attrs["rotary_offset"] == geom.rotary_offset
        assert attrs["do_rotary"] == 1
        assert pytest.approx(attrs["scale"]) == geom.scale
        # rotary_dim is DERIVED from cos_cache, never emitted as an attribute.
        assert "rotary_dim" not in attrs

    def test_input_positions(self):
        graph, _ = _build_paged_graph(_cfg())
        node = _find_paged_node(graph)
        names = [v.name if v is not None else None for v in node.inputs]
        # Positions 0,1,3,5,6,7,8,9,10 present; 2 and 4 absent (LATENT).
        assert names[0] is not None and "Reshape" in names[0]  # query (2D)
        assert names[1] is not None and "Reshape" in names[1]  # latent key (2D)
        assert names[2] is None  # value absent
        assert names[3] == "key_cache"
        assert names[4] is None  # value_cache absent
        assert names[5] == "cu_seqlens"
        assert names[6] == "past_seqlens"
        assert names[7] == "block_table"
        assert names[8] == "cos_cache"
        assert names[9] == "sin_cache"
        assert names[10] == "slot_mapping"

    def test_key_cache_aliases_output(self):
        graph, _ = _build_paged_graph(_cfg())
        node = _find_paged_node(graph)
        # Output 1 is key_cache_out; the runtime must alias input 3 in place.
        assert len(node.outputs) == 3
        assert node.inputs[3].name == "key_cache"

    def test_glm_dims_emits(self):
        graph, _ = _build_paged_graph(make_config(**_GLM))
        node = _find_paged_node(graph)
        attrs = {a.name: a.value for a in node.attributes.values()}
        assert attrs["v_head_size"] == 512
        assert attrs["rotary_offset"] == 512
        assert attrs["num_heads"] == 4
