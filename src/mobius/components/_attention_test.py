# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for Attention components."""

from __future__ import annotations

import onnx_ir as ir
import pytest

from mobius._testing import (
    count_op_type,
    create_test_builder,
    create_test_input,
    make_config,
)
from mobius.components._attention import Attention, Qwen35Attention


class TestAttention:
    """Tests for the standard multi-head Attention module."""

    def test_projection_weight_shapes(self):
        config = make_config()
        attn = Attention(config)
        # q_proj: (num_heads * head_dim, hidden_size) = (4*16, 64)
        assert list(attn.q_proj.weight.shape) == [64, 64]
        # k_proj: (num_kv_heads * head_dim, hidden_size) = (2*16, 64)
        assert list(attn.k_proj.weight.shape) == [32, 64]
        # v_proj same as k_proj
        assert list(attn.v_proj.weight.shape) == [32, 64]
        # o_proj: (hidden_size, num_heads * head_dim)
        assert list(attn.o_proj.weight.shape) == [64, 64]

    def test_no_qk_norm_by_default(self):
        config = make_config()
        attn = Attention(config)
        assert attn.q_norm is None
        assert attn.k_norm is None

    def test_qk_norm_enabled(self):
        config = make_config(attn_qk_norm=True)
        attn = Attention(config)
        assert attn.q_norm is not None
        assert attn.k_norm is not None

    def test_qk_norm_full_enabled(self):
        config = make_config(attn_qk_norm=True, attn_qk_norm_full=True)
        attn = Attention(config)
        assert attn.q_norm is not None
        # Full norm: weight shape = (num_heads * head_dim,)
        assert list(attn.q_norm.weight.shape) == [64]
        assert list(attn.k_norm.weight.shape) == [32]

    def test_qk_norm_per_head(self):
        config = make_config(attn_qk_norm=True, attn_qk_norm_full=False)
        attn = Attention(config)
        # Per-head norm: weight shape = (head_dim,)
        assert list(attn.q_norm.weight.shape) == [16]
        assert list(attn.k_norm.weight.shape) == [16]

    def test_custom_scale(self):
        config = make_config()
        attn = Attention(config, scale=0.5)
        assert attn.scaling == pytest.approx(0.5)

    def test_default_scale(self):
        config = make_config()
        attn = Attention(config)
        assert attn.scaling == pytest.approx(16**-0.5)

    def test_forward_builds_graph(self):
        config = make_config()
        attn = Attention(config)
        builder, op, graph = create_test_builder()
        hidden = create_test_input(builder, "hidden", [1, 8, 64])
        bias = create_test_input(builder, "bias", [1, 4, 8, 8])

        output, (present_key, present_value) = attn(op, hidden, attention_bias=bias)
        builder._adapt_outputs([output, present_key, present_value], "")
        assert graph.num_nodes() > 0
        assert count_op_type(graph, "Attention") >= 1

    def test_forward_with_past_kv(self):
        config = make_config()
        attn = Attention(config)
        builder, op, graph = create_test_builder()
        hidden = create_test_input(builder, "hidden", [1, 1, 64])
        bias = create_test_input(builder, "bias", [1, 4, 1, 9])
        past_key = create_test_input(builder, "pk", [1, 8, 2, 16])
        past_value = create_test_input(builder, "pv", [1, 8, 2, 16])

        output, (pk, pv) = attn(
            op,
            hidden,
            attention_bias=bias,
            past_key_value=(past_key, past_value),
        )
        builder._adapt_outputs([output, pk, pv], "")
        assert count_op_type(graph, "Attention") >= 1

    def test_forward_with_rope(self):
        config = make_config()
        attn = Attention(config)
        builder, op, graph = create_test_builder()
        hidden = create_test_input(builder, "hidden", [1, 8, 64])
        bias = create_test_input(builder, "bias", [1, 4, 8, 8])
        cos = create_test_input(builder, "cos", [1, 8, 16])
        sin = create_test_input(builder, "sin", [1, 8, 16])

        output, _ = attn(
            op,
            hidden,
            attention_bias=bias,
            position_embeddings=(cos, sin),
        )
        builder._adapt_outputs([output], "")
        assert graph.num_nodes() > 0

    def test_forward_with_qk_norm_builds_graph(self):
        config = make_config(attn_qk_norm=True)
        attn = Attention(config)
        builder, op, graph = create_test_builder()
        hidden = create_test_input(builder, "hidden", [1, 8, 64])
        bias = create_test_input(builder, "bias", [1, 4, 8, 8])

        output, _ = attn(op, hidden, attention_bias=bias)
        builder._adapt_outputs([output], "")
        assert count_op_type(graph, "RMSNormalization") >= 2

    def test_gqa_head_counts(self):
        config = make_config(num_attention_heads=8, num_key_value_heads=2, head_dim=16)
        attn = Attention(config)
        assert attn.num_attention_heads == 8
        assert attn.num_key_value_heads == 2

    def test_parameter_count(self):
        config = make_config()
        attn = Attention(config)
        params = list(attn.parameters())
        # q_proj.weight, k_proj.weight, v_proj.weight, o_proj.weight = 4
        assert len(params) == 4

    def test_parameter_count_with_bias(self):
        config = make_config(attn_qkv_bias=True, attn_o_bias=True)
        attn = Attention(config)
        params = list(attn.parameters())
        # 4 weights + 4 biases = 8
        assert len(params) == 8


class TestQwen35Attention:
    """Tests for Qwen3.5 gated attention."""

    def test_q_proj_doubled(self):
        config = make_config(
            partial_rotary_factor=0.5,
        )
        attn = Qwen35Attention(config)
        # Q proj is doubled: 2 * (num_heads * head_dim)
        assert list(attn.q_proj.weight.shape) == [128, 64]

    def test_has_offset_rms_norm(self):
        from mobius.components._rms_norm import OffsetRMSNorm

        config = make_config(partial_rotary_factor=0.5)
        attn = Qwen35Attention(config)
        assert isinstance(attn.q_norm, OffsetRMSNorm)
        assert isinstance(attn.k_norm, OffsetRMSNorm)

    def test_forward_builds_graph(self):
        config = make_config(partial_rotary_factor=0.5)
        attn = Qwen35Attention(config)
        builder, op, graph = create_test_builder()
        hidden = create_test_input(builder, "hidden", [1, 8, 64])
        bias = create_test_input(builder, "bias", [1, 4, 8, 8])
        cos = create_test_input(builder, "cos", [1, 8, 16])
        sin = create_test_input(builder, "sin", [1, 8, 16])

        output, (pk, pv) = attn(
            op,
            hidden,
            attention_bias=bias,
            position_embeddings=(cos, sin),
        )
        builder._adapt_outputs([output, pk, pv], "")
        # Should have Attention op + Sigmoid (for gate) + Mul (output gating)
        assert count_op_type(graph, "Attention") >= 1
        assert count_op_type(graph, "Sigmoid") >= 1


class TestGQAContextDispatch:
    """Tests for the GQAContext direct GroupQueryAttention emission path."""

    def test_gqa_context_emits_group_query_attention(self):
        """When attention_bias is a GQAContext, Attention emits GroupQueryAttention directly."""
        from mobius.components._attention import GQAContext

        config = make_config()
        attn = Attention(config)
        builder, op, graph = create_test_builder()

        hidden = create_test_input(builder, "hidden", [1, 8, 64])
        past_key = create_test_input(builder, "past_key", [1, 2, 4, 16])
        past_value = create_test_input(builder, "past_value", [1, 2, 4, 16])
        seqlens_k = create_test_input(builder, "seqlens_k", [1], dtype=ir.DataType.INT32)
        total_seq_len = create_test_input(
            builder, "total_seq_len", [], dtype=ir.DataType.INT32
        )
        # rotary_dim = head_dim / 2 = 16 / 2 = 8 (inv_freq has half the head_dim entries)
        cos_cache = create_test_input(builder, "cos_cache", [32, 8])
        sin_cache = create_test_input(builder, "sin_cache", [32, 8])

        gqa_ctx = GQAContext(
            seqlens_k=seqlens_k,
            total_seq_len=total_seq_len,
            cos_cache=cos_cache,
            sin_cache=sin_cache,
        )

        output, (pk, pv) = attn(
            op, hidden, attention_bias=gqa_ctx, past_key_value=(past_key, past_value)
        )
        builder._adapt_outputs([output, pk, pv], "")

        # Direct path: GroupQueryAttention instead of ONNX Attention
        assert count_op_type(graph, "GroupQueryAttention") >= 1
        assert count_op_type(graph, "Attention") == 0

    def test_gqa_context_respects_rotary_interleaved(self):
        """rotary_interleaved attribute is set from config.rope_interleave."""
        from mobius.components._attention import GQAContext

        config = make_config(rope_interleave=True)
        attn = Attention(config)
        builder, op, graph = create_test_builder()

        hidden = create_test_input(builder, "hidden", [1, 8, 64])
        past_key = create_test_input(builder, "past_key", [1, 2, 4, 16])
        past_value = create_test_input(builder, "past_value", [1, 2, 4, 16])
        seqlens_k = create_test_input(builder, "seqlens_k", [1], dtype=ir.DataType.INT32)
        total_seq_len = create_test_input(
            builder, "total_seq_len", [], dtype=ir.DataType.INT32
        )
        cos_cache = create_test_input(builder, "cos_cache", [32, 8])
        sin_cache = create_test_input(builder, "sin_cache", [32, 8])

        gqa_ctx = GQAContext(seqlens_k, total_seq_len, cos_cache, sin_cache)

        output, _ = attn(
            op, hidden, attention_bias=gqa_ctx, past_key_value=(past_key, past_value)
        )
        builder._adapt_outputs([output], "")

        gqa_node = next(n for n in graph if n.op_type == "GroupQueryAttention")
        assert gqa_node.attributes["rotary_interleaved"].value == 1

    def test_gqa_context_local_window_size(self):
        """local_window_size attribute is set on GQA node when > 0."""
        from mobius.components._attention import GQAContext

        config = make_config()
        attn = Attention(config)
        builder, op, graph = create_test_builder()

        hidden = create_test_input(builder, "hidden", [1, 8, 64])
        past_key = create_test_input(builder, "past_key", [1, 2, 4, 16])
        past_value = create_test_input(builder, "past_value", [1, 2, 4, 16])
        seqlens_k = create_test_input(builder, "seqlens_k", [1], dtype=ir.DataType.INT32)
        total_seq_len = create_test_input(
            builder, "total_seq_len", [], dtype=ir.DataType.INT32
        )
        cos_cache = create_test_input(builder, "cos_cache", [32, 8])
        sin_cache = create_test_input(builder, "sin_cache", [32, 8])

        gqa_ctx = GQAContext(
            seqlens_k, total_seq_len, cos_cache, sin_cache, local_window_size=512
        )

        output, _ = attn(
            op, hidden, attention_bias=gqa_ctx, past_key_value=(past_key, past_value)
        )
        builder._adapt_outputs([output], "")

        gqa_node = next(n for n in graph if n.op_type == "GroupQueryAttention")
        assert gqa_node.attributes["local_window_size"].value == 512

    def test_gqa_context_no_local_window_size_when_default(self):
        """local_window_size attribute is absent when default (-1)."""
        from mobius.components._attention import GQAContext

        config = make_config()
        attn = Attention(config)
        builder, op, graph = create_test_builder()

        hidden = create_test_input(builder, "hidden", [1, 8, 64])
        past_key = create_test_input(builder, "past_key", [1, 2, 4, 16])
        past_value = create_test_input(builder, "past_value", [1, 2, 4, 16])
        seqlens_k = create_test_input(builder, "seqlens_k", [1], dtype=ir.DataType.INT32)
        total_seq_len = create_test_input(
            builder, "total_seq_len", [], dtype=ir.DataType.INT32
        )
        cos_cache = create_test_input(builder, "cos_cache", [32, 8])
        sin_cache = create_test_input(builder, "sin_cache", [32, 8])

        gqa_ctx = GQAContext(seqlens_k, total_seq_len, cos_cache, sin_cache)

        output, _ = attn(
            op, hidden, attention_bias=gqa_ctx, past_key_value=(past_key, past_value)
        )
        builder._adapt_outputs([output], "")

        gqa_node = next(n for n in graph if n.op_type == "GroupQueryAttention")
        assert "local_window_size" not in gqa_node.attributes

    def test_standard_attention_when_no_gqa_context(self):
        """Without GQAContext, standard ONNX Attention is emitted."""
        config = make_config()
        attn = Attention(config)
        builder, op, graph = create_test_builder()

        hidden = create_test_input(builder, "hidden", [1, 8, 64])
        bias = create_test_input(builder, "bias", [1, 4, 8, 8])

        output, _ = attn(op, hidden, attention_bias=bias)
        builder._adapt_outputs([output], "")

        assert count_op_type(graph, "Attention") >= 1
        assert count_op_type(graph, "GroupQueryAttention") == 0

    def test_build_with_cuda_ep_emits_gqa_directly(self):
        """build_from_module with CUDA EP and float16 config emits GroupQueryAttention directly."""
        from mobius._builder import build_from_module
        from mobius._registry import registry
        from mobius.rewrite_rules._testing_utils import count_ops

        config = make_config(
            dtype=ir.DataType.FLOAT16,
            max_position_embeddings=128,
            rope_type="default",
            rope_theta=10000.0,
        )
        pkg = build_from_module(
            registry.get("llama")(config),
            config,
            execution_provider="cuda",
        )
        ops = count_ops(pkg["model"])
        # Direct generation: each layer should have a GroupQueryAttention node
        assert ops.get("GroupQueryAttention", 0) == config.num_hidden_layers
        # Standard ONNX Attention should not appear
        assert ops.get("Attention", 0) == 0

    def test_build_with_default_ep_uses_standard_attention(self):
        """build_from_module with default EP keeps standard ONNX Attention (no GQA)."""
        from mobius._builder import build_from_module
        from mobius._registry import registry
        from mobius.rewrite_rules._testing_utils import count_ops

        config = make_config(
            max_position_embeddings=128,
            rope_type="default",
            rope_theta=10000.0,
        )
        pkg = build_from_module(
            registry.get("llama")(config),
            config,
            execution_provider="default",
        )
        ops = count_ops(pkg["model"])
        assert ops.get("GroupQueryAttention", 0) == 0
        assert ops.get("Attention", 0) == config.num_hidden_layers

    def test_mrope_model_does_not_use_direct_gqa(self):
        """Models with MRoPE (Qwen2.5-VL, Qwen3-VL) must NOT emit GQA directly.

        GQA do_rotary=1 only supports 1D RoPE. _MRopeBase subclasses
        (ChunkedMRope, InterleavedMRope) use 3D position_ids for temporal/
        height/width axes. Silently emitting GQA would produce wrong outputs.
        CUDA+f16 EP is used to trigger the GQA path for 1D-RoPE models;
        the MRoPE model must fall through to the rewrite-rule path instead.
        """
        from mobius._builder import build_from_module
        from mobius._configs import ArchitectureConfig
        from mobius._registry import registry
        from mobius.rewrite_rules._testing_utils import count_ops

        # Minimal Qwen2.5-VL-style config with mrope_section (activates ChunkedMRope).
        mrope_config = ArchitectureConfig(
            hidden_size=64,
            intermediate_size=128,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            num_hidden_layers=2,
            vocab_size=256,
            max_position_embeddings=128,
            hidden_act="silu",
            rms_norm_eps=1e-6,
            rope_type="default",
            rope_theta=10000.0,
            pad_token_id=0,
            dtype=ir.DataType.FLOAT16,
            mrope_section=[4, 6, 6],  # activates ChunkedMRope
        )
        # Use "llama" model class (TextModel backbone) so use_gqa condition is evaluated.
        # qwen2_5_vl uses a different model class, but the relevant guard is in TextModel.
        pkg = build_from_module(
            registry.get("llama")(mrope_config),
            mrope_config,
            execution_provider="cuda",
        )
        ops = count_ops(pkg["model"])
        # MRoPE model must NOT use direct GQA (do_rotary=1 is 1D only).
        # The rewrite rule path still applies GroupQueryAttention after graph construction.
        assert ops.get("GroupQueryAttention", 0) == mrope_config.num_hidden_layers

    def test_direct_gqa_and_rewrite_rule_produce_same_structure(self):
        """Direct GQA and rewrite-rule paths both produce GroupQueryAttention per layer.

        Verifies that for a standard 1D-RoPE model:
        - CPU EP (direct path): num_layers GQA nodes
        - Default EP + manual rewrite rule: same count
        """
        from onnxscript.rewriter import rewrite

        from mobius._builder import build_from_module
        from mobius._registry import registry
        from mobius.rewrite_rules import group_query_attention_rules
        from mobius.rewrite_rules._testing_utils import count_ops

        config = make_config(
            max_position_embeddings=128,
            rope_type="default",
            rope_theta=10000.0,
        )
        num_layers = config.num_hidden_layers

        # Direct path: CPU EP uses GQA directly (FLOAT is in cpu.gqa_dtypes)
        pkg_direct = build_from_module(
            registry.get("llama")(config),
            config,
            execution_provider="cpu",
        )

        # Rewrite-rule path: default EP keeps Attention + RoPE, then rewrite fires
        pkg_default = build_from_module(
            registry.get("llama")(config),
            config,
            execution_provider="default",
        )
        rewrite(pkg_default["model"], group_query_attention_rules())

        ops_direct = count_ops(pkg_direct["model"])
        ops_rewrite = count_ops(pkg_default["model"])

        # Both paths must produce the same number of GroupQueryAttention nodes
        assert ops_direct.get("GroupQueryAttention", 0) == num_layers
        assert ops_rewrite.get("GroupQueryAttention", 0) == num_layers
        # Neither path should leave any standard Attention nodes
        assert ops_direct.get("Attention", 0) == 0
        assert ops_rewrite.get("Attention", 0) == 0
