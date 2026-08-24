# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for GGUF tensor processors."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from mobius.integrations.gguf._tensor_processors import (
    process_tensors,
)


class TestProcessTensorsLlama:
    """Tests for Llama/Mistral Q/K reverse permutation."""

    def _make_config(
        self,
        model_type: str = "llama",
        num_heads: int = 8,
        num_kv_heads: int = 8,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            model_type=model_type,
            num_attention_heads=num_heads,
            num_key_value_heads=num_kv_heads,
        )

    def test_qk_roundtrip(self) -> None:
        """Verify permute → reverse_permute is identity."""
        config = self._make_config(num_heads=4, num_kv_heads=4)
        # Create a known weight matrix
        original_q = torch.randn(64, 128)
        original_k = torch.randn(64, 128)

        # Simulate what llama.cpp does: permute
        q_perm = self._forward_permute(original_q, 4)
        k_perm = self._forward_permute(original_k, 4)

        state_dict = {
            "model.layers.0.self_attn.q_proj.weight": q_perm,
            "model.layers.0.self_attn.k_proj.weight": k_perm,
            "model.layers.0.self_attn.v_proj.weight": (torch.randn(64, 128)),
        }
        result = process_tensors(state_dict, config)

        # After reverse permute, should match original
        torch.testing.assert_close(
            result["model.layers.0.self_attn.q_proj.weight"],
            original_q,
        )
        torch.testing.assert_close(
            result["model.layers.0.self_attn.k_proj.weight"],
            original_k,
        )

    def test_gqa_different_head_counts(self) -> None:
        """Test GQA with num_kv_heads < num_attention_heads."""
        config = self._make_config(num_heads=8, num_kv_heads=2)
        original_k = torch.randn(32, 128)
        k_perm = self._forward_permute(original_k, 2)

        state_dict = {
            "model.layers.0.self_attn.k_proj.weight": k_perm,
        }
        result = process_tensors(state_dict, config)
        torch.testing.assert_close(
            result["model.layers.0.self_attn.k_proj.weight"],
            original_k,
        )

    def test_v_proj_untouched(self) -> None:
        """V projection should NOT be permuted."""
        config = self._make_config()
        v_weight = torch.randn(64, 128)
        state_dict = {
            "model.layers.0.self_attn.v_proj.weight": (v_weight.clone()),
        }
        result = process_tensors(state_dict, config)
        torch.testing.assert_close(
            result["model.layers.0.self_attn.v_proj.weight"],
            v_weight,
        )

    def test_mistral_uses_llama_processor(self) -> None:
        """Mistral should use the same processor as Llama."""
        config = self._make_config(model_type="mistral", num_heads=4, num_kv_heads=4)
        original = torch.randn(64, 128)
        perm = self._forward_permute(original, 4)
        state_dict = {
            "model.layers.0.self_attn.q_proj.weight": perm,
        }
        result = process_tensors(state_dict, config)
        torch.testing.assert_close(
            result["model.layers.0.self_attn.q_proj.weight"],
            original,
        )

    def test_internlm2_reverses_the_pinned_converter_permutation(self) -> None:
        """InternLM2's converter calls LlamaModel.permute for both Q and K."""
        config = self._make_config(
            model_type="internlm2",
            num_heads=4,
            num_kv_heads=2,
        )
        config._gguf_arch = "internlm2"
        original_q = torch.arange(64 * 8, dtype=torch.float32).reshape(64, 8)
        original_k = torch.arange(32 * 8, dtype=torch.float32).reshape(32, 8)
        state_dict = {
            "model.layers.0.self_attn.q_proj.weight": self._forward_permute(original_q, 4),
            "model.layers.0.self_attn.k_proj.weight": self._forward_permute(original_k, 2),
        }

        result = process_tensors(state_dict, config)

        torch.testing.assert_close(
            result["model.layers.0.self_attn.q_proj.weight"], original_q
        )
        torch.testing.assert_close(
            result["model.layers.0.self_attn.k_proj.weight"], original_k
        )

    def test_dense_cohort_converter_permutation_contract(self) -> None:
        from mobius.integrations.gguf._tensor_processors import (
            needs_llama_qk_permute,
        )

        for model_type in (
            "olmo",
            "arcee",
            "smollm3",
            "internlm2",
            "granitemoe",
            "llada",
        ):
            assert needs_llama_qk_permute(model_type)
        for model_type in ("olmo2", "cohere2", "exaone", "dream"):
            assert not needs_llama_qk_permute(model_type)

    def test_reverse_matches_hf_reference_head_dim_64(self) -> None:
        """Reverse permute must match HF's reference for real head dims.

        Regression test for the Q4/Q-K garbage-output bug: the old code
        reshaped as ``(n_head, 2, dim)`` which only inverts llama.cpp's
        permute when ``dim == 2`` (head_dim == 4). For head_dim == 64 it
        scrambled Q/K rows. This asserts the exact inverse of llama.cpp's
        forward permute is recovered, and cross-checks HF's reference
        ``_reverse_permute_weights`` formula.
        """
        config = self._make_config(num_heads=14, num_kv_heads=14)
        # Qwen2.5-0.5B geometry: hidden=896, 14 heads, head_dim=64.
        original_q = torch.randn(896, 896)
        q_perm = self._forward_permute(original_q, 14)

        state_dict = {"model.layers.0.self_attn.q_proj.weight": q_perm}
        result = process_tensors(state_dict, config)
        torch.testing.assert_close(
            result["model.layers.0.self_attn.q_proj.weight"],
            original_q,
        )

        # Cross-check against HF's exact reference formula.
        dim = q_perm.shape[0] // 14 // 2
        hf_ref = (
            q_perm.reshape(14, dim, 2, *q_perm.shape[1:]).swapaxes(2, 1).reshape(q_perm.shape)
        )
        torch.testing.assert_close(
            result["model.layers.0.self_attn.q_proj.weight"],
            hf_ref,
        )

    @staticmethod
    def _forward_permute(weights: torch.Tensor, n_head: int) -> torch.Tensor:
        """Simulate llama.cpp's forward permutation (HF -> GGUF).

        Reference: ``convert_hf_to_gguf.py`` ``LlamaModel.permute``::

            weights.reshape(n_head, 2, dim, ...).swapaxes(1, 2).reshape(orig)

        The reverse permute in production must exactly invert this for any
        head dim, not only ``dim == 2``.
        """
        dim = weights.shape[0] // n_head // 2
        w = weights.reshape(n_head, 2, dim, *weights.shape[1:])
        return w.swapaxes(1, 2).reshape(weights.shape)


class TestQwenNotPermuted:
    """Regression: Qwen2/Qwen3 Q/K must NOT be reverse-permuted.

    Qwen uses NEOX-style rope and stores Q/K in plain HF order. Applying
    the llama interleaved-rope permute scrambles attention heads and
    produces garbage output (root cause of the invalid Q4 benchmark).
    """

    def test_needs_llama_qk_permute_helper(self) -> None:
        from mobius.integrations.gguf._tensor_processors import (
            needs_llama_qk_permute,
        )

        assert needs_llama_qk_permute("llama") is True
        assert needs_llama_qk_permute("mistral") is True
        assert needs_llama_qk_permute("qwen2") is False
        assert needs_llama_qk_permute("qwen3") is False
        assert needs_llama_qk_permute("gemma2") is False
        assert needs_llama_qk_permute(None) is False

    def test_qwen2_qk_weights_unchanged(self) -> None:
        config = SimpleNamespace(
            model_type="qwen2",
            num_attention_heads=14,
            num_key_value_heads=2,
        )
        q = torch.randn(896, 896)
        k = torch.randn(128, 896)
        state_dict = {
            "model.layers.0.self_attn.q_proj.weight": q.clone(),
            "model.layers.0.self_attn.k_proj.weight": k.clone(),
        }
        result = process_tensors(state_dict, config)
        # Qwen has no registered processor → weights pass through untouched.
        torch.testing.assert_close(result["model.layers.0.self_attn.q_proj.weight"], q)
        torch.testing.assert_close(result["model.layers.0.self_attn.k_proj.weight"], k)


class TestBuilderNeedsQkPermute:
    """Regression: the quantized inline permute gate must honor model_type."""

    def test_qwen2_quantized_not_permuted(self) -> None:
        from mobius.integrations.gguf._builder import _needs_qk_permute

        name = "model.layers.0.self_attn.q_proj.weight"
        assert _needs_qk_permute(name, 14, 2, "qwen2") is False
        assert _needs_qk_permute(name, 14, 2, "qwen3") is False

    def test_llama_quantized_permuted(self) -> None:
        from mobius.integrations.gguf._builder import _needs_qk_permute

        name = "model.layers.0.self_attn.q_proj.weight"
        assert _needs_qk_permute(name, 32, 32, "llama") is True
        assert _needs_qk_permute(name, 32, 32, "mistral") is True
        assert _needs_qk_permute(name, 32, 8, "granitemoe") is True
        assert _needs_qk_permute(name, 32, 32, "llada", "llada") is True

    def test_diffusion_moe_quantized_qk_is_not_permuted(self) -> None:
        from mobius.integrations.gguf._builder import _needs_qk_permute

        name = "model.layers.0.self_attn.q_proj.weight"
        assert _needs_qk_permute(name, 32, 8, "llada", "llada-moe") is False
        assert _needs_qk_permute(name, 32, 8, "llada", "rnd1") is False

    def test_non_qk_tensor_never_permuted(self) -> None:
        from mobius.integrations.gguf._builder import _needs_qk_permute

        name = "model.layers.0.self_attn.v_proj.weight"
        assert _needs_qk_permute(name, 32, 32, "llama") is False


class TestProcessTensorsGemma:
    """Tests for Gemma norm weight offset."""

    def test_norm_weights_restored(self) -> None:
        config = SimpleNamespace(model_type="gemma2")
        state_dict = {
            "model.layers.0.input_layernorm.weight": (torch.tensor([0.0, 1.0, -1.0])),
            "model.norm.weight": torch.tensor([0.5]),
            "model.layers.0.self_attn.q_proj.weight": (torch.tensor([1.0, 2.0])),
        }
        result = process_tensors(state_dict, config)

        # Norm weights: undo llama.cpp's baked (w_hf + 1) offset → subtract 1.
        torch.testing.assert_close(
            result["model.layers.0.input_layernorm.weight"],
            torch.tensor([-1.0, 0.0, -2.0]),
        )
        torch.testing.assert_close(
            result["model.norm.weight"],
            torch.tensor([-0.5]),
        )
        # Non-norm weights: unchanged
        torch.testing.assert_close(
            result["model.layers.0.self_attn.q_proj.weight"],
            torch.tensor([1.0, 2.0]),
        )


class TestProcessTensorsGPT2:
    """Tests for GPT-2 weight transpose."""

    def test_attn_weights_transposed(self) -> None:
        config = SimpleNamespace(model_type="gpt2")
        w = torch.randn(3, 5)
        state_dict = {
            "transformer.h.0.attn.c_attn.weight": w.clone(),
            "transformer.h.0.attn.c_attn.bias": (torch.randn(5)),
        }
        result = process_tensors(state_dict, config)

        # Weight should be transposed
        torch.testing.assert_close(
            result["transformer.h.0.attn.c_attn.weight"],
            w.T,
        )
        # Bias should be unchanged
        assert result["transformer.h.0.attn.c_attn.bias"].shape == (5,)

    def test_ffn_weights_transposed(self) -> None:
        config = SimpleNamespace(model_type="gpt2")
        w = torch.randn(3, 5)
        state_dict = {
            "transformer.h.0.mlp.c_fc.weight": w.clone(),
            "transformer.h.0.mlp.c_proj.weight": w.clone(),
        }
        result = process_tensors(state_dict, config)
        torch.testing.assert_close(result["transformer.h.0.mlp.c_fc.weight"], w.T)
        torch.testing.assert_close(result["transformer.h.0.mlp.c_proj.weight"], w.T)


class TestProcessTensorsMamba:
    """Tests for Mamba tensor fixes."""

    def test_conv1d_unsqueeze(self) -> None:
        config = SimpleNamespace(model_type="mamba")
        w = torch.randn(16, 4)
        state_dict = {
            "backbone.layers.0.mixer.conv1d.weight": (w.clone()),
        }
        result = process_tensors(state_dict, config)
        assert result["backbone.layers.0.mixer.conv1d.weight"].shape == (16, 1, 4)

    def test_decay_transform_recovers_original_a_log_values(self) -> None:
        config = SimpleNamespace(model_type="mamba")
        a_log = torch.tensor([-2.0, 0.0, 1.5], dtype=torch.float32)
        gguf_a = -torch.exp(a_log)
        state_dict = {"model.layers.0.mixer.A_log": gguf_a}

        result = process_tensors(state_dict, config)

        torch.testing.assert_close(result["model.layers.0.mixer.A_log"], a_log)
        torch.testing.assert_close(
            -torch.exp(result["model.layers.0.mixer.A_log"]),
            gguf_a,
        )

    def test_decay_transform_rejects_wrong_direction_input(self) -> None:
        config = SimpleNamespace(model_type="mamba2")
        state_dict = {"backbone.layers.0.mixer.A_log": torch.tensor([-1.0, 0.25])}

        with pytest.raises(ValueError, match="only negative"):
            process_tensors(state_dict, config)

    def test_mamba2_squeezes_cpp_head_parameter_layout(self) -> None:
        config = SimpleNamespace(model_type="mamba2")
        state_dict = {
            "backbone.layers.0.mixer.A_log": -torch.exp(torch.tensor([[0.0], [1.0]])),
            "backbone.layers.0.mixer.D": torch.tensor([[1.0], [2.0]]),
            "backbone.layers.0.mixer.norm.weight": torch.tensor([[3.0, 4.0]]),
        }

        result = process_tensors(state_dict, config)

        torch.testing.assert_close(
            result["backbone.layers.0.mixer.A_log"],
            torch.tensor([0.0, 1.0]),
        )
        torch.testing.assert_close(
            result["backbone.layers.0.mixer.D"],
            torch.tensor([1.0, 2.0]),
        )
        torch.testing.assert_close(
            result["backbone.layers.0.mixer.norm.weight"],
            torch.tensor([3.0, 4.0]),
        )


class TestProcessTensorsNoop:
    """Test that unknown architectures pass through."""

    def test_unknown_model_type_noop(self) -> None:
        config = SimpleNamespace(model_type="some_unknown_model")
        original = {"a.weight": torch.tensor([1.0])}
        result = process_tensors(dict(original), config)
        torch.testing.assert_close(result["a.weight"], original["a.weight"])

    def test_no_model_type_noop(self) -> None:
        config = SimpleNamespace()
        original = {"a.weight": torch.tensor([1.0])}
        result = process_tensors(dict(original), config)
        torch.testing.assert_close(result["a.weight"], original["a.weight"])


class TestProcessMuseGlimmer:
    """Muse Glimmer stores centered block norms as ``w_hf + 1``."""

    def test_block_norms_lose_the_offset(self) -> None:
        config = SimpleNamespace(
            model_type="muse_glimmer_text",
            num_attention_heads=1,
            num_key_value_heads=1,
        )
        state_dict = {
            "model.layers.0.input_layernorm.weight": torch.tensor([1.25, 0.0]),
            "model.layers.0.post_attention_layernorm.weight": (torch.tensor([2.0, 1.0])),
            "model.layers.0.pre_feedforward_layernorm.weight": (torch.tensor([1.5])),
            "model.layers.0.post_feedforward_layernorm.weight": (torch.tensor([0.5])),
        }
        result = process_tensors(state_dict, config)
        torch.testing.assert_close(
            result["model.layers.0.input_layernorm.weight"],
            torch.tensor([0.25, -1.0]),
        )
        torch.testing.assert_close(
            result["model.layers.0.post_attention_layernorm.weight"],
            torch.tensor([1.0, 0.0]),
        )
        torch.testing.assert_close(
            result["model.layers.0.pre_feedforward_layernorm.weight"],
            torch.tensor([0.5]),
        )
        torch.testing.assert_close(
            result["model.layers.0.post_feedforward_layernorm.weight"],
            torch.tensor([-0.5]),
        )

    def test_final_norm_is_untouched(self) -> None:
        # model.norm is a plain RMSNorm in this architecture.
        config = SimpleNamespace(
            model_type="muse_glimmer_text",
            num_attention_heads=1,
            num_key_value_heads=1,
        )
        state_dict = {"model.norm.weight": torch.tensor([0.25, -3.0])}
        result = process_tensors(state_dict, config)
        torch.testing.assert_close(result["model.norm.weight"], torch.tensor([0.25, -3.0]))

    def test_qk_weights_are_reverse_permuted(self) -> None:
        # llama.cpp's muse-glimmer converter permutes attn_q/attn_k for
        # interleaved rope, on every layer including the NoPE ones.
        config = SimpleNamespace(
            model_type="muse_glimmer_text",
            num_attention_heads=1,
            num_key_value_heads=1,
        )
        original = torch.arange(8.0).reshape(4, 2)
        state_dict = {
            "model.layers.0.self_attn.q_proj.weight": original.clone(),
            "model.layers.3.self_attn.k_proj.weight": original.clone(),
            "model.layers.0.self_attn.v_proj.weight": original.clone(),
        }
        result = process_tensors(state_dict, config)
        expected = torch.tensor([[0.0, 1.0], [4.0, 5.0], [2.0, 3.0], [6.0, 7.0]])
        torch.testing.assert_close(result["model.layers.0.self_attn.q_proj.weight"], expected)
        torch.testing.assert_close(result["model.layers.3.self_attn.k_proj.weight"], expected)
        torch.testing.assert_close(result["model.layers.0.self_attn.v_proj.weight"], original)

    def test_quantized_path_permutes_qk(self) -> None:
        from mobius.integrations.gguf._builder import _needs_qk_permute

        name = "model.layers.0.self_attn.q_proj.weight"
        assert _needs_qk_permute(name, 32, 2, "muse_glimmer_text") is True
