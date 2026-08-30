# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Static-cache, KV-sharing, GQA, sliding-window, and RoPE L1 tests.

Run the complete L1 suite with ``pytest tests/build_graph_*_test.py``.
"""

from __future__ import annotations

import numpy as np
import onnx_ir as ir
import pytest
from _build_graph_support import _assert_outputs_have_shapes_and_dtypes
from _test_configs import (
    TINY_HEAD_DIM,
    TINY_HEADS,
    TINY_HIDDEN,
    TINY_INTERMEDIATE,
    TINY_KV_HEADS,
    TINY_LAYERS,
    TINY_VOCAB,
    _base_config,
)

from mobius._builder import build_from_module
from mobius._configs import ArchitectureConfig
from mobius._registry import registry
from mobius.integrations.transformers._config_resolver import _default_task_for_model
from mobius.tasks import get_task


class TestBuildStaticCacheGraph:
    """Verify CausalLMTask(static_cache=True) builds a valid graph."""

    MAX_SEQ_LEN = 128

    def _build_static_cache_model(self, model_type: str = "qwen2", **config_overrides):
        """Build a model with CausalLMTask(static_cache=True) and return (model, config)."""
        from mobius.tasks import CausalLMTask

        config = _base_config(**config_overrides)
        model_cls = registry.get(model_type)
        module = model_cls(config)
        task = CausalLMTask(static_cache=True, max_seq_len=self.MAX_SEQ_LEN)
        pkg = task.build(module, config)
        return pkg["model"], config

    def test_static_cache_graph_builds(self):
        """Build a Qwen2 model with static cache."""
        model, _ = self._build_static_cache_model()

        assert model.graph is not None
        assert len(model.graph.inputs) > 0
        assert len(model.graph.outputs) > 0

    def test_static_cache_graph_inputs(self):
        """Verify expected inputs: standard + per-layer caches + shared."""
        model, config = self._build_static_cache_model()
        input_names = {inp.name for inp in model.graph.inputs}
        num_layers = config.num_hidden_layers

        # Standard inputs
        assert "input_ids" in input_names
        assert "position_ids" in input_names

        # No attention_mask in static cache mode — causal masking is
        # handled by is_causal=1 on the Attention op.
        assert "attention_mask" not in input_names

        # Per-layer static cache inputs
        for i in range(num_layers):
            assert f"key_cache.{i}" in input_names, f"Missing key_cache.{i}"
            assert f"value_cache.{i}" in input_names, f"Missing value_cache.{i}"

        # Shared cache management inputs
        assert "write_indices" in input_names
        assert "nonpad_kv_seqlen" in input_names

        # Exact count: 2 standard + 2*num_layers caches + 2 shared
        expected_count = 2 + 2 * num_layers + 2
        assert len(model.graph.inputs) == expected_count, (
            f"Expected {expected_count} inputs, got {len(model.graph.inputs)}"
        )

    def test_static_cache_graph_outputs(self):
        """Verify outputs: logits + updated caches per layer."""
        model, config = self._build_static_cache_model()
        output_names = {out.name for out in model.graph.outputs}
        num_layers = config.num_hidden_layers

        assert "logits" in output_names

        # Updated caches per layer (not present.{i}.key/value)
        for i in range(num_layers):
            assert f"updated_key_cache.{i}" in output_names, f"Missing updated_key_cache.{i}"
            assert f"updated_value_cache.{i}" in output_names, (
                f"Missing updated_value_cache.{i}"
            )

        # Should NOT have dynamic cache outputs
        assert not any(n.startswith("present.") for n in output_names), (
            "Static cache graph should not have present.* outputs"
        )

        # Exact count: 1 logits + 2*num_layers updated caches
        expected_count = 1 + 2 * num_layers
        assert len(model.graph.outputs) == expected_count, (
            f"Expected {expected_count} outputs, got {len(model.graph.outputs)}"
        )

    def test_static_cache_has_tensorscatter_and_attention(self):
        """Verify graph contains TensorScatter and Attention ops."""
        model, _ = self._build_static_cache_model()

        op_types = {n.op_type for n in model.graph}
        assert "TensorScatter" in op_types, "Static cache graph should use TensorScatter"
        assert "Attention" in op_types, "Static cache graph should use Attention"

    def test_static_cache_has_initializers(self):
        """Verify the graph has model parameters."""
        model, _ = self._build_static_cache_model()

        init_names = list(model.graph.initializers)
        assert len(init_names) > 0
        assert any("embed_tokens" in n for n in init_names)
        assert any("self_attn" in n for n in init_names)
        assert any("mlp" in n for n in init_names)

    def test_static_cache_graph_validates(self):
        """Verify the graph survives a serialization round-trip."""
        model, _config = self._build_static_cache_model()
        proto = ir.serde.serialize_model(model)
        assert len(proto.SerializeToString()) > 0

    def test_static_cache_attention_uses_maskless_causal_alignment(self):
        """Verify maskless external-cache Attention uses built-in causality."""
        model, config = self._build_static_cache_model()

        attention_nodes = [n for n in model.graph if n.op_type == "Attention"]
        assert len(attention_nodes) == config.num_hidden_layers

        for node in attention_nodes:
            is_causal = node.attributes.get("is_causal")
            assert is_causal is not None, (
                f"Attention node {node.name} missing is_causal attribute"
            )
            assert is_causal.as_int() == 1, (
                f"Attention node {node.name} should have is_causal=1"
            )

    def test_static_cache_attention_no_attn_mask_input(self):
        """Verify Attention ops do NOT receive attn_mask in static cache mode."""
        model, config = self._build_static_cache_model()

        attention_nodes = [n for n in model.graph if n.op_type == "Attention"]
        assert len(attention_nodes) == config.num_hidden_layers

        for node in attention_nodes:
            # Input 3 (0-indexed) is attn_mask — should be empty/None
            attn_mask_input = node.inputs[3]
            assert attn_mask_input is None or attn_mask_input.name == "", (
                f"Attention node {node.name} should not have attn_mask "
                f"connected, but got input: {attn_mask_input}"
            )

    def test_static_cache_moe_graph_builds(self):
        """Build a MoE model (qwen2_moe) with static cache."""
        model, _config = self._build_static_cache_model(
            model_type="qwen2_moe",
            num_local_experts=4,
            num_experts_per_tok=2,
            attn_qkv_bias=True,
            shared_expert_intermediate_size=64,
        )

        assert model.graph is not None
        assert len(model.graph.inputs) > 0
        assert len(model.graph.outputs) > 0

        input_names = {inp.name for inp in model.graph.inputs}
        assert "input_ids" in input_names
        assert "position_ids" in input_names
        assert "attention_mask" not in input_names

        # Verify TensorScatter and Attention ops are present
        op_types = {n.op_type for n in model.graph}
        assert "TensorScatter" in op_types
        assert "Attention" in op_types

    def test_outputs_have_shapes_and_dtypes(self):
        """Verify shape inference populates all output shapes and dtypes."""
        model, _ = self._build_static_cache_model()
        _assert_outputs_have_shapes_and_dtypes({"model": model}, "qwen2-static")

    def _build_gemma4_static(self):
        """Build gemma4_text with static cache: KV-sharing + dual head_dim."""
        from mobius._configs import Gemma4Config
        from mobius.tasks import CausalLMTask

        config = _base_config(
            _config_cls=Gemma4Config,
            num_hidden_layers=6,
            layer_types=[
                "sliding_attention",
                "sliding_attention",
                "full_attention",
                "sliding_attention",
                "sliding_attention",
                "full_attention",
            ],
            num_kv_shared_layers=2,  # first_kv_shared = 4 -> layers 0..3 own a cache
            sliding_window=8,
            global_head_dim=2 * TINY_HEAD_DIM,
            global_rope_theta=10_000.0,
            rope_local_base_freq=10_000.0,
            attn_qk_norm=True,
            hidden_size_per_layer_input=0,
        )
        module = registry.get("gemma4_text")(config)
        pkg = CausalLMTask(static_cache=True, max_seq_len=self.MAX_SEQ_LEN).build(
            module, config
        )
        return pkg["model"], config

    def test_gemma4_static_cache_only_for_cache_owning_layers(self):
        """Gemma4 KV-shared layers own no cache: 4 cache buffers, not 6."""
        model, _ = self._build_gemma4_static()
        input_names = {inp.name for inp in model.graph.inputs}
        # first_kv_shared_layer_idx = 6 - 2 = 4 -> cache-owning layers 0..3
        for i in range(4):
            assert f"key_cache.{i}" in input_names
            assert f"value_cache.{i}" in input_names
        # No cache for the KV-shared layers (indices 4, 5).
        assert "key_cache.4" not in input_names
        assert "key_cache.5" not in input_names
        output_names = {out.name for out in model.graph.outputs}
        for i in range(4):
            assert f"updated_key_cache.{i}" in output_names
        assert "updated_key_cache.4" not in output_names

    def test_gemma4_static_cache_dual_head_dim(self):
        """Sliding (head_dim) and full (global_head_dim) cache buffers differ."""
        model, _ = self._build_gemma4_static()
        by_name = {inp.name: inp for inp in model.graph.inputs}
        # num_kv_heads=TINY_KV_HEADS; sliding head_dim=TINY_HEAD_DIM,
        # full head_dim=2*TINY_HEAD_DIM. Cache-owning layer 2 is full_attention.
        sliding_kv = int(by_name["key_cache.0"].shape[2])  # layer 0 sliding
        full_kv = int(by_name["key_cache.2"].shape[2])  # layer 2 full
        assert sliding_kv == TINY_KV_HEADS * TINY_HEAD_DIM
        assert full_kv == TINY_KV_HEADS * (2 * TINY_HEAD_DIM)

    def test_gemma4_static_cache_has_tensorscatter(self):
        """Gemma4 static cache uses TensorScatter for in-place KV writes."""
        model, _ = self._build_gemma4_static()
        op_types = {n.op_type for n in model.graph}
        assert "TensorScatter" in op_types

    def test_gemma4_static_qnn_lowering_is_htp_friendly(self):
        """The qnn build lowers all ops the QNN HTP backend cannot run.

        RotaryEmbedding -> rotate-half, TensorScatter -> ScatterND, Tile ->
        Expand, Range -> Constant, Attention -> SDPA are all HTP-unsupported and
        must be gone; their HTP-friendly replacements must appear.
        """
        from mobius._builder import build_from_module
        from mobius._configs import Gemma4Config
        from mobius.tasks import CausalLMTask

        config = _base_config(
            _config_cls=Gemma4Config,
            num_hidden_layers=3,
            layer_types=["sliding_attention", "full_attention", "sliding_attention"],
            num_kv_shared_layers=1,
            sliding_window=8,
            global_head_dim=2 * TINY_HEAD_DIM,
            global_rope_theta=10_000.0,
            rope_local_base_freq=10_000.0,
            attn_qk_norm=True,
            hidden_size_per_layer_input=0,
        )
        module = registry.get("gemma4_text")(config)
        model = build_from_module(
            module,
            config,
            CausalLMTask(static_cache=True, max_seq_len=self.MAX_SEQ_LEN),
            execution_provider="qnn",
        )["model"]
        op_types = {n.op_type for n in model.graph}
        for forbidden in ("RotaryEmbedding", "TensorScatter", "Tile", "Range", "Attention"):
            assert forbidden not in op_types, f"{forbidden} should be lowered for qnn"
        assert "ScatterND" in op_types  # TensorScatter replacement
        assert "Expand" in op_types


class TestBuildGemma3nKvSharing:
    """Gemma 3n's trailing layers borrow K,V instead of projecting their own.

    Layers at or after ``num_hidden_layers - num_kv_shared_layers`` reuse the
    K,V of the last preceding layer of the *same* attention type, so they own
    no KV cache entry and carry no ``k_proj``/``v_proj``/``k_norm`` weights.
    """

    @staticmethod
    def _build(num_kv_shared_layers, layer_types):
        from mobius._configs import Gemma3nConfig

        config = Gemma3nConfig(
            num_hidden_layers=len(layer_types),
            hidden_size=64,
            intermediate_size=128,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            vocab_size=256,
            max_position_embeddings=128,
            rms_norm_eps=1e-6,
            hidden_act="gelu_pytorch_tanh",
            rope_type="default",
            rope_theta=10_000.0,
            rope_local_base_freq=10_000.0,
            attn_qk_norm=True,
            layer_types=layer_types,
            sliding_window=8,
            altup_num_inputs=2,
            altup_active_idx=0,
            altup_correct_scale=True,
            laurel_rank=16,
            hidden_size_per_layer_input=32,
            vocab_size_per_layer_input=256,
            num_kv_shared_layers=num_kv_shared_layers,
            pad_token_id=0,
        )
        module = registry.get("gemma3n_text")(config)
        pkg = get_task(_default_task_for_model("gemma3n_text")).build(module, config)
        return module, pkg["model"], config

    def test_cache_io_excludes_shared_layers(self):
        """Only non-shared layers get past/present KV entries."""
        module, model, _config = self._build(2, ["full_attention"] * 4)

        assert module.kv_cache_layer_count() == 2
        input_names = {i.name for i in model.graph.inputs}
        output_names = {o.name for o in model.graph.outputs}
        for i in range(2):
            assert f"past_key_values.{i}.key" in input_names
            assert f"past_key_values.{i}.value" in input_names
            assert f"present.{i}.key" in output_names
            assert f"present.{i}.value" in output_names
        for i in (2, 3):
            assert f"past_key_values.{i}.key" not in input_names
            assert f"present.{i}.key" not in output_names

    def test_shared_layers_have_no_kv_weights(self):
        """KV-shared layers must not request k_proj/v_proj/k_norm initializers.

        The checkpoint ships these tensors for every layer, but HF only builds
        them for the non-shared layers — emitting them here would create
        initializers with no consumer.
        """
        _module, model, _config = self._build(2, ["full_attention"] * 4)

        names = set(model.graph.initializers)
        for i in (0, 1):
            assert f"model.layers.{i}.self_attn.k_proj.weight" in names
            assert f"model.layers.{i}.self_attn.v_proj.weight" in names
        for i in (2, 3):
            assert f"model.layers.{i}.self_attn.k_proj.weight" not in names
            assert f"model.layers.{i}.self_attn.v_proj.weight" not in names
            assert f"model.layers.{i}.self_attn.k_norm.weight" not in names

    def test_source_layer_matches_attention_type(self):
        """Sliding and full layers borrow from different source layers.

        HF indexes the *pre-cutoff* slice of ``layer_types`` for the matching
        type, so a shared sliding layer never borrows a full layer's K,V.
        """
        layer_types = [
            "sliding_attention",
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "full_attention",
        ]
        module, _model, _config = self._build(2, layer_types)

        attns = [layer.self_attn for layer in module.model.layers]
        assert [a.is_kv_shared_layer for a in attns] == [False, False, False, True, True]
        # Last non-shared layer of each type publishes its K,V for reuse.
        assert [a.provides_shared_kv for a in attns] == [False, True, True, False, False]
        # Shared sliding layer 3 -> layer 1; shared full layer 4 -> layer 2.
        assert attns[3].kv_shared_layer_index == 1
        assert attns[4].kv_shared_layer_index == 2

    def test_drops_shared_layer_weights_from_state_dict(self):
        """preprocess_weights discards the K/V tensors HF never constructs."""
        import torch

        module, _model, config = self._build(2, ["full_attention"] * 4)
        state_dict = {
            f"model.layers.{i}.self_attn.{name}.weight": torch.zeros(1)
            for i in range(config.num_hidden_layers)
            for name in ("q_proj", "k_proj", "v_proj", "k_norm")
        }

        result = module.preprocess_weights(state_dict)

        for i in range(config.num_hidden_layers):
            assert f"model.layers.{i}.self_attn.q_proj.weight" in result
        for i in (0, 1):
            assert f"model.layers.{i}.self_attn.k_proj.weight" in result
        for i in (2, 3):
            assert f"model.layers.{i}.self_attn.k_proj.weight" not in result
            assert f"model.layers.{i}.self_attn.v_proj.weight" not in result
            assert f"model.layers.{i}.self_attn.k_norm.weight" not in result

    def test_rejects_sharing_every_layer(self):
        """A layer cannot borrow K,V when no earlier layer computes any."""
        with pytest.raises(ValueError, match="num_kv_shared_layers"):
            self._build(4, ["full_attention"] * 4)


class TestBuildGemma4StaticCacheGraph:
    """Verify Gemma4TextCausalLMTask(static_cache=True) builds a valid graph."""

    MAX_SEQ_LEN = 128

    @staticmethod
    def _gemma4_config(**overrides):
        from mobius._configs import Gemma4Config

        defaults = dict(
            num_hidden_layers=6,
            hidden_size=64,
            intermediate_size=128,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            global_head_dim=32,
            vocab_size=256,
            rms_norm_eps=1e-6,
            hidden_act="gelu_pytorch_tanh",
            # Mixed: 5 sliding + 1 full (Gemma4-like hybrid pattern)
            layer_types=[
                "sliding_attention",
                "sliding_attention",
                "sliding_attention",
                "sliding_attention",
                "sliding_attention",
                "full_attention",
            ],
            sliding_window=64,
            rope_theta=10000.0,
            global_rope_theta=1000000.0,
            partial_rotary_factor=0.5,
            max_position_embeddings=256,
            hidden_size_per_layer_input=0,
            num_kv_shared_layers=0,
        )
        defaults.update(overrides)
        return Gemma4Config(**defaults)

    def _build(self, **config_overrides):
        from mobius.models.gemma4 import Gemma4CausalLMModel
        from mobius.tasks._gemma4 import Gemma4TextCausalLMTask

        config = self._gemma4_config(**config_overrides)
        module = Gemma4CausalLMModel(config)
        task = Gemma4TextCausalLMTask(static_cache=True, max_seq_len=self.MAX_SEQ_LEN)
        pkg = task.build(module, config)
        return pkg["model"], config

    def test_gemma4_static_cache_builds(self):
        """Build Gemma4 with static cache and verify basic graph structure."""
        model, _config = self._build()

        assert model.graph is not None
        input_names = {inp.name for inp in model.graph.inputs}
        assert "input_ids" in input_names
        assert "position_ids" in input_names
        # Hybrid mode: attention_mask for sliding GQA + write_indices for static
        assert "attention_mask" in input_names
        assert "write_indices" in input_names
        assert "nonpad_kv_seqlen" in input_names

    def test_gemma4_static_cache_hybrid_inputs(self):
        """Verify full-attention gets static cache, sliding gets dynamic."""
        model, _config = self._build()

        input_map = {inp.name: inp for inp in model.graph.inputs}

        # Layer 0-4: sliding → dynamic cache (past_key_values.N.key)
        assert "past_key_values.0.key" in input_map

        # Layer 5: full_attention → static cache (key_cache.5)
        assert "key_cache.5" in input_map
        kv_hidden_full = _config.num_key_value_heads * _config.global_head_dim
        k5 = input_map["key_cache.5"]
        assert k5.shape[2] == kv_hidden_full

    def test_gemma4_static_cache_has_tensorscatter(self):
        """Verify TensorScatter for full-attention layers in hybrid mode."""
        model, _config = self._build()

        op_counts = {}
        for n in model.graph:
            op_counts[n.op_type] = op_counts.get(n.op_type, 0) + 1

        layer_types = _config.layer_types or []
        num_full = sum(1 for lt in layer_types if lt == "full_attention")

        # TensorScatter: 2 per full-attention layer (key + value)
        assert op_counts.get("TensorScatter", 0) == 2 * num_full
        # Sliding layers use either GQA (CUDA EP) or Attention (default EP)
        # In unit tests without EP context, all use standard Attention.
        total_attn = op_counts.get("Attention", 0) + op_counts.get("GroupQueryAttention", 0)
        assert total_attn == _config.num_hidden_layers

    def test_gemma4_static_cache_kv_shared(self):
        """Verify KV-shared layers are excluded from cache I/O."""
        model, config = self._build(
            num_hidden_layers=8,
            num_kv_shared_layers=2,
            layer_types=[
                "sliding_attention",
                "sliding_attention",
                "sliding_attention",
                "sliding_attention",
                "sliding_attention",
                "full_attention",
                "sliding_attention",
                "full_attention",
            ],
        )

        input_names = {inp.name for inp in model.graph.inputs}
        num_kv_layers = config.num_hidden_layers - config.num_kv_shared_layers

        # Non-shared layers 0-5 should have cache entries (type depends on layer)
        for i in range(num_kv_layers):
            lt = config.layer_types[i]
            if lt == "full_attention":
                assert f"key_cache.{i}" in input_names
            else:
                assert f"past_key_values.{i}.key" in input_names

        # Shared layers (6, 7) should NOT have any cache entries
        assert f"key_cache.{num_kv_layers}" not in input_names
        assert f"past_key_values.{num_kv_layers}.key" not in input_names

    def test_gemma4_static_cache_input_ordering(self):
        """Verify write_indices/nonpad_kv_seqlen come after cache inputs."""
        model, _config = self._build()

        input_names = [inp.name for inp in model.graph.inputs]
        # write_indices and nonpad_kv_seqlen should come after all cache inputs
        last_cache_idx = max(i for i, n in enumerate(input_names) if "cache" in n)
        write_idx = input_names.index("write_indices")
        nonpad_idx = input_names.index("nonpad_kv_seqlen")
        assert write_idx > last_cache_idx, "write_indices should come after all cache inputs"
        assert nonpad_idx > last_cache_idx, (
            "nonpad_kv_seqlen should come after all cache inputs"
        )


class TestGQASlidingWindow:
    """Wire ``config.sliding_window`` into GQA's ``local_window_size``.

    On the direct GQA path (``TextModel.forward``), GQA
    ``local_window_size=W`` masks each query to the most recent ``W`` keys
    (positions ``[i-W+1, i]``), matching HuggingFace ``sliding_window=W``.
    Because the global GQAContext is shared by every layer, the window is only
    emitted for models with a *uniform* sliding window across all layers.
    """

    @staticmethod
    def _build_gqa_model(**overrides):
        from mobius.models.base import CausalLMModel

        overrides.setdefault("num_hidden_layers", 2)
        config = ArchitectureConfig(
            hidden_size=64,
            intermediate_size=128,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            vocab_size=256,
            max_position_embeddings=128,
            hidden_act="silu",
            rms_norm_eps=1e-6,
            rope_type="default",
            rope_theta=10000.0,
            pad_token_id=0,
            dtype=ir.DataType.FLOAT16,
            **overrides,
        )
        module = CausalLMModel(config)
        # execution_provider="cuda" + fp16 activates the direct GQA path.
        return build_from_module(module, config, execution_provider="cuda")["model"]

    @classmethod
    def _build_gqa_decoder(cls, **overrides):
        model = cls._build_gqa_model(**overrides)
        gqa_nodes = [n for n in model.graph if n.op_type == "GroupQueryAttention"]
        assert gqa_nodes, "expected GroupQueryAttention nodes on the cuda/fp16 path"
        return gqa_nodes

    @staticmethod
    def _fill_fp16_weights(model, seed):
        """Deterministically fill uninitialised params, honouring fp16 dtype."""
        rng = np.random.default_rng(seed)
        npdt = {ir.DataType.FLOAT16: np.float16, ir.DataType.FLOAT: np.float32}
        for init in model.graph.initializers.values():
            if init.const_value is None:
                shape = [int(d) for d in init.shape]
                arr = (rng.standard_normal(shape) * 0.1).astype(
                    npdt.get(init.dtype, np.float32)
                )
                init.const_value = ir.Tensor(arr)

    def test_uniform_sliding_window_sets_local_window_size(self):
        """A uniform sliding window is forwarded to every GQA node verbatim."""
        gqa_nodes = self._build_gqa_decoder(sliding_window=3000)
        for node in gqa_nodes:
            assert node.attributes["local_window_size"].value == 3000

    def test_no_sliding_window_omits_local_window_size(self):
        """Full-attention models must not carry a local_window_size attribute."""
        gqa_nodes = self._build_gqa_decoder(sliding_window=None)
        for node in gqa_nodes:
            assert "local_window_size" not in node.attributes

    def test_mixed_layer_types_omits_local_window_size(self):
        """A per-layer schedule cannot be expressed by one global window."""
        gqa_nodes = self._build_gqa_decoder(
            sliding_window=8,
            layer_types=["sliding_attention", "full_attention"],
        )
        for node in gqa_nodes:
            assert "local_window_size" not in node.attributes

    def test_all_sliding_layer_types_sets_local_window_size(self):
        """An explicit all-sliding schedule is uniform, so window is emitted."""
        gqa_nodes = self._build_gqa_decoder(
            sliding_window=8,
            layer_types=["sliding_attention", "sliding_attention"],
        )
        for node in gqa_nodes:
            assert node.attributes["local_window_size"].value == 8

    def test_window_confines_receptive_field(self):
        """The sliding window must actually bound each query's receptive field.

        Property test (no golden, weight-agnostic): with window ``W`` over
        ``L`` layers, the last position can only be influenced by inputs within
        ``L*(W-1)`` steps. Perturbing an out-of-window token must leave the
        windowed model's last-position logits unchanged, while the same
        perturbation *does* change the full-attention twin's logits — proving
        the window is genuinely applied (and exercising seq > window, unlike
        the short-sequence Moshi parity golden). Runs the fp16 GQA op on CPU.
        """
        from mobius._testing.ort_inference import OnnxModelSession

        window, seq, layers = 4, 24, 2
        # Pos 0 is well outside the last position's receptive field
        # (layers*(window-1) = 6 << seq-1 = 23).
        windowed = self._build_gqa_model(sliding_window=window, num_hidden_layers=layers)
        full = self._build_gqa_model(sliding_window=None, num_hidden_layers=layers)
        # Same seed + identical structure (only the GQA attr differs) => same weights.
        self._fill_fp16_weights(windowed, seed=1234)
        self._fill_fp16_weights(full, seed=1234)

        def feeds(first_token):
            ids = np.arange(1, seq + 1, dtype=np.int64).reshape(1, seq)
            ids[0, 0] = first_token
            f = {"input_ids": ids, "attention_mask": np.ones((1, seq), np.int64)}
            for i in range(layers):
                f[f"past_key_values.{i}.key"] = np.zeros((1, 2, 0, 16), np.float16)
                f[f"past_key_values.{i}.value"] = np.zeros((1, 2, 0, 16), np.float16)
            return f

        def last_logits(model, first_token):
            sess = OnnxModelSession(model)
            out = sess.run(feeds(first_token))
            return out["logits"][0, -1].astype(np.float32)

        # Windowed: perturbing the out-of-window first token leaves the last
        # position's logits unchanged.
        w_a = last_logits(windowed, first_token=5)
        w_b = last_logits(windowed, first_token=200)
        np.testing.assert_allclose(w_a, w_b, atol=1e-3)

        # Full attention: the same perturbation DOES reach the last position.
        f_a = last_logits(full, first_token=5)
        f_b = last_logits(full, first_token=200)
        assert np.abs(f_a - f_b).max() > 1e-2, (
            "full-attention twin should be sensitive to the first token"
        )

    def test_empty_layer_types_omits_local_window_size(self):
        """An empty ``layer_types`` is not a valid uniform schedule.

        ``all(... for t in [])`` is vacuously True, so an unhardened guard
        would wrongly treat ``[]`` as uniform-sliding. The length check against
        ``num_hidden_layers`` rejects it, leaving the window disabled.
        """
        gqa_nodes = self._build_gqa_decoder(
            sliding_window=8,
            layer_types=[],
            num_hidden_layers=2,
        )
        for node in gqa_nodes:
            assert "local_window_size" not in node.attributes

    def test_non_gqa_path_warns_on_uniform_window(self, caplog):
        """Non-GQA (static-cache) export of a uniform-sliding model warns.

        That path cannot represent the window, so the warning flags the
        divergence from HuggingFace for sequences longer than the window.
        """
        import logging

        from mobius.models.base import CausalLMModel

        config = ArchitectureConfig(
            hidden_size=64,
            intermediate_size=128,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            vocab_size=256,
            max_position_embeddings=128,
            hidden_act="silu",
            rms_norm_eps=1e-6,
            rope_type="default",
            rope_theta=10000.0,
            pad_token_id=0,
            num_hidden_layers=2,
            sliding_window=8,
            dtype=ir.DataType.FLOAT,
        )
        module = CausalLMModel(config)
        # Static cache feeds attention_mask=None, taking the non-GQA else
        # branch that cannot express the window.
        from mobius.tasks import CausalLMTask

        task = CausalLMTask(static_cache=True, max_seq_len=128)
        with caplog.at_level(logging.WARNING, logger="mobius.models.base"):
            build_from_module(module, config, task=task, execution_provider="cpu")
        assert "sliding window" in caplog.text

    def test_partial_layer_types_omits_local_window_size(self):
        """A ``layer_types`` shorter than the layer count is not uniform."""
        gqa_nodes = self._build_gqa_decoder(
            sliding_window=8,
            layer_types=["sliding_attention"],
            num_hidden_layers=2,
        )
        for node in gqa_nodes:
            assert "local_window_size" not in node.attributes


class TestResolveSlidingWindow:
    """``from_transformers`` must honor HF's ``use_sliding_window`` gate.

    Qwen2/Qwen3 keep a non-null ``sliding_window`` in the config even when the
    window is disabled and signal activation via ``use_sliding_window``. A raw
    ``config.json`` fallback bypasses HF's ``__post_init__`` (which would null
    the field), so the gate is re-applied in ``_resolve_sliding_window``.
    """

    @staticmethod
    def _cfg(**attrs):
        from types import SimpleNamespace

        defaults = dict(
            model_type="qwen2",
            hidden_size=64,
            intermediate_size=128,
            num_attention_heads=4,
            num_key_value_heads=2,
            num_hidden_layers=2,
            vocab_size=256,
            max_position_embeddings=128,
            hidden_act="silu",
            rms_norm_eps=1e-6,
            rope_theta=10000.0,
        )
        defaults.update(attrs)
        return SimpleNamespace(**defaults)

    def test_disabled_window_is_nulled(self):
        """``use_sliding_window=False`` drops a non-null ``sliding_window``."""
        cfg = ArchitectureConfig.from_transformers(
            self._cfg(sliding_window=4096, use_sliding_window=False), "qwen2"
        )
        assert cfg.sliding_window is None

    def test_enabled_window_is_kept(self):
        """``use_sliding_window=True`` keeps the window."""
        cfg = ArchitectureConfig.from_transformers(
            self._cfg(sliding_window=4096, use_sliding_window=True), "qwen2"
        )
        assert cfg.sliding_window == 4096

    def test_window_without_flag_is_kept(self):
        """Models without ``use_sliding_window`` (e.g. Mistral) are unaffected."""
        cfg = ArchitectureConfig.from_transformers(
            self._cfg(model_type="mistral", sliding_window=4096), "mistral"
        )
        assert cfg.sliding_window == 4096


class TestLongRopeAliasExtraction:
    """``rope_type`` alias handling for Phi LongRoPE.

    Phi-3/Phi-3.5 checkpoints label LongRoPE as ``"su"`` (short/long-factor
    scaled rotary embeddings); newer HuggingFace configs spell the identical
    algorithm ``"longrope"``. ``_extract_rope_config`` must canonicalize the
    legacy ``"su"`` alias to ``"longrope"`` so both configs resolve to the
    same ``LongRope`` code path.
    """

    _SHORT_FACTOR = [1.0] * (TINY_HEAD_DIM // 2)
    _LONG_FACTOR = [2.0] * (TINY_HEAD_DIM // 2)

    @staticmethod
    def _cfg(rope_scaling, **attrs):
        from types import SimpleNamespace

        defaults = dict(
            model_type="phi3",
            hidden_size=TINY_HIDDEN,
            intermediate_size=TINY_INTERMEDIATE,
            num_attention_heads=TINY_HEADS,
            num_key_value_heads=TINY_KV_HEADS,
            head_dim=TINY_HEAD_DIM,
            num_hidden_layers=TINY_LAYERS,
            vocab_size=TINY_VOCAB,
            max_position_embeddings=1024,
            original_max_position_embeddings=128,
            hidden_act="silu",
            rms_norm_eps=1e-6,
            rope_theta=10000.0,
            rope_scaling=rope_scaling,
        )
        defaults.update(attrs)
        return SimpleNamespace(**defaults)

    def _scaling(self, rope_type_key, rope_type_value):
        return {
            rope_type_key: rope_type_value,
            "short_factor": self._SHORT_FACTOR,
            "long_factor": self._LONG_FACTOR,
        }

    def test_su_and_longrope_produce_identical_rope_config(self):
        """``type="su"`` and ``rope_type="longrope"`` extract identically."""
        from mobius._configs._base import _extract_rope_config

        su_config = _extract_rope_config(self._cfg(self._scaling("type", "su")))
        longrope_config = _extract_rope_config(
            self._cfg(self._scaling("rope_type", "longrope"))
        )

        assert su_config is not None
        assert su_config.rope_type == "longrope"
        assert longrope_config.rope_type == "longrope"
        assert su_config.original_max_position_embeddings == 128
        assert su_config.rope_type == longrope_config.rope_type
        assert (
            su_config.rope_scaling["short_factor"]
            == longrope_config.rope_scaling["short_factor"]
        )
        assert (
            su_config.rope_scaling["long_factor"]
            == longrope_config.rope_scaling["long_factor"]
        )
        assert (
            su_config.original_max_position_embeddings
            == longrope_config.original_max_position_embeddings
        )

    def test_su_alias_dispatches_to_longrope_module(self):
        """A ``"su"`` config resolves to the ``LongRope`` runtime module."""
        from mobius.components._rotary_embedding import LongRope, initialize_rope

        config = ArchitectureConfig.from_transformers(self._cfg(self._scaling("type", "su")))
        assert config.rope_type == "longrope"
        rope = initialize_rope(config)
        assert isinstance(rope, LongRope)

    def test_su_graph_builds_end_to_end(self):
        """A phi3 ``"su"`` config builds a valid ONNX graph without weights."""
        config = ArchitectureConfig.from_transformers(self._cfg(self._scaling("type", "su")))
        module = registry.get("phi3")(config)
        task = get_task(_default_task_for_model("phi3"))
        pkg = task.build(module, config)
        model = pkg["model"]

        assert model.graph is not None
        input_names = {inp.name for inp in model.graph.inputs}
        assert "input_ids" in input_names
        assert "position_ids" in input_names
        output_names = {out.name for out in model.graph.outputs}
        assert "logits" in output_names

    def test_missing_original_max_position_embeddings_still_maps_to_longrope(self):
        """A ``"su"`` config without ``original_max_position_embeddings``.

        The alias must still resolve to ``longrope`` and ``LongRope`` falls
        back to ``max_position_embeddings`` for the short cache length rather
        than crashing.
        """
        from mobius._configs._base import _extract_rope_config
        from mobius.components._rotary_embedding import LongRope, initialize_rope

        config_source = self._cfg(
            self._scaling("type", "su"), original_max_position_embeddings=None
        )
        rope_config = _extract_rope_config(config_source)
        assert rope_config.rope_type == "longrope"
        assert rope_config.original_max_position_embeddings is None

        arch_config = ArchitectureConfig.from_transformers(config_source)
        rope = initialize_rope(arch_config)
        assert isinstance(rope, LongRope)

    def test_factor_length_mismatch_is_rejected(self):
        """Short/long factor arrays must match the rotary dimension.

        A factor list whose length does not equal ``head_dim / 2`` cannot be
        broadcast against the inverse-frequency vector, so ``LongRope``
        construction raises rather than silently producing a wrong cache.
        """
        from mobius.components._rotary_embedding import initialize_rope

        bad_scaling = {
            "type": "su",
            "short_factor": [1.0] * (TINY_HEAD_DIM // 2 + 1),
            "long_factor": [2.0] * (TINY_HEAD_DIM // 2 + 1),
        }
        config = ArchitectureConfig.from_transformers(self._cfg(bad_scaling))
        assert config.rope_type == "longrope"
        with pytest.raises(ValueError, match="broadcast"):
            initialize_rope(config)

    def test_non_alias_rope_types_are_unchanged(self):
        """Canonicalization only rewrites known aliases, not other types."""
        from mobius._configs._base import _canonical_rope_type

        assert _canonical_rope_type("su") == "longrope"
        assert _canonical_rope_type("longrope") == "longrope"
        assert _canonical_rope_type("yarn") == "yarn"
        assert _canonical_rope_type("default") == "default"
        assert _canonical_rope_type(None) is None
