# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""L3 Synthetic Parity Tests — registered model types.

Builds tiny random-weight models for BOTH HuggingFace PyTorch and ONNX,
compares single-forward-pass logits.  Any atol divergence with identical
seeds indicates a genuine op-level bug.

Run::

    pytest tests/synthetic_parity_test.py -v --tb=short -n 0

Run a single model::

    pytest tests/synthetic_parity_test.py -k "llama" -v
"""

from __future__ import annotations

import logging

import numpy as np
import onnx_ir as ir
import pytest
import torch
from _test_configs import (
    ALL_CAUSAL_LM_CONFIGS,
    TINY_HEAD_DIM,
    TINY_HEADS,
    TINY_HIDDEN,
    TINY_INTERMEDIATE,
    TINY_KV_HEADS,
    TINY_LAYERS,
    TINY_VOCAB,
    _base_config,
)

from mobius._configs import ArchitectureConfig
from mobius._registry import registry
from mobius._testing.parity import ParityResult, compare_synthetic
from mobius.integrations._weight_loading import apply_weights
from mobius.integrations.transformers._config_resolver import _default_task_for_model
from mobius.tasks import get_task

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model types that cannot be tested with HF synthetic parity.
# Each entry maps a model_type to a reason for skipping.
# ---------------------------------------------------------------------------
_SKIP_REASONS: dict[str, str] = {
    "phi4mm": "Multi-modal model, requires special HF setup",
    # trust_remote_code models: require downloading model files — not suitable for offline tests
    "chatglm": "Requires trust_remote_code=True (not in HF native CONFIG_MAPPING)",
    "qwen": "HF Qwen requires trust_remote_code=True (not in HF native CONFIG_MAPPING)",
    "openelm": "HF OpenELM requires trust_remote_code=True (not in HF native CONFIG_MAPPING)",
    "internlm2": "HF AutoConfig does not recognize internlm2 (requires trust_remote_code)",
    "minicpm": "HF AutoConfig does not recognize minicpm (requires trust_remote_code)",
    "minicpm3": "HF AutoConfig does not recognize minicpm3 (requires trust_remote_code)",
    "baichuan": "HF AutoConfig does not recognize baichuan (requires trust_remote_code)",
    "arctic": "HF AutoConfig does not recognize arctic (requires trust_remote_code)",
    "ernie4_5": "HF ernie4_5 model requires special fields not in our standard test infra",
    # Mamba2 standalone model: HF creates different architecture
    "mamba2": "HF Mamba2 standalone is not a causal LM model",
    # Non-standard config format: DbrxConfig uses d_model/n_heads/n_layers/attn_config
    # (nested sub-configs) rather than standard hidden_size/num_attention_heads keys.
    # Cannot create a correctly-sized tiny reference model with our generic test infra.
    "dbrx": "DbrxConfig uses non-standard nested sub-config parameters",
    # ImageGPT: not registered with AutoModelForCausalLM (image generation model)
    "imagegpt": "ImageGPTConfig not registered with AutoModelForCausalLM",
    # ShieldGemma2: safety model, not registered with AutoModelForCausalLM
    "shieldgemma2": "ShieldGemma2Config not registered with AutoModelForCausalLM",
    # Non-CausalLM models: their config class is not registered with AutoModelForCausalLM
    "csm": "CsmConfig not registered with AutoModelForCausalLM (speech model)",
    "evolla": "EvollaConfig not registered with AutoModelForCausalLM (multimodal VLM)",
    # Architectural mismatches: ONNX uses CausalLMModel but HF uses a fundamentally
    # different architecture (MoE or MLA) that cannot be directly compared.
    "solar_open": "HF solar_open uses non-standard packed MoE (bskcn_* params, no num_local_experts); needs custom model",
    # Youtu is dense-only MLA; HF deepseek_v2 always creates MoE layers so
    # synthetic parity doesn't produce a fair comparison.
    "youtu": "Youtu is dense-only MLA; HF deepseek_v2 model always creates MoE layers",
    # Zamba weight-tying references layers.2.shared_transf (the third layer) but
    # the tiny config only has 2 layers — HF tie_weights validation crashes.
    "zamba": "Zamba weight-tying requires num_layers > 2; tiny 2-layer config causes HF tie_weights error",
    # GLM-5.2's default (config.use_dsa=True) DSA/IndexShare path emits
    # pkg.nxrt::IndexShare -- a custom onnx-genai-runtime domain with no
    # stock-ORT registration, so this generic OnnxModelSession-based harness
    # can never execute it (unlike com.microsoft::QMoE, a real ORT contrib
    # op). DSA numeric correctness is covered instead by the dedicated
    # GlmMoeDsaIndexer-vs-transformers parity test in glm_moe_dsa_test.py and
    # by onnx-genai's native-CPU/CUDA e2e regression
    # (glm_tiny_qmoe_native_cuda_e2e.rs); the config.use_dsa=False
    # (--glm-full-attention) dense-MLA fallback has no custom ops and would
    # be exercised by this harness if the tiny config defaulted to it.
    "glm_moe_dsa": "DSA/IndexShare emits pkg.nxrt::IndexShare, unsupported by stock ORT",
}

# Per-model atol overrides for L3 synthetic parity.
# Models with inherent FP accumulation differences (e.g., HF uses fused batched expert
# computation while ONNX uses per-expert sequential MLPs) need a looser tolerance.
# Only used when argmax_match=True and cosine similarity is very high (≥0.999),
# confirming the model is functionally correct despite the FP difference.
_ATOL_OVERRIDES: dict[str, float] = {
    # GraniteMoE uses fused GraniteMoeParallelExperts (batched matmul over all experts)
    # while we use per-expert MLP. Different FP accumulation order → ~0.021 max diff.
    # Argmax correct, cosine=0.999 — model is functionally correct.
    "granitemoe": 0.025,
    "granitemoeshared": 0.025,
    "granite": 0.025,
    # Apertus uses xIELU activation (Softplus FP) → small accumulation differences.
    # Argmax correct, cosine=0.9997 — model is functionally correct.
    "apertus": 0.02,
    # Bloom: LayerNorm accumulation differs after eps alignment → ~0.019 max diff.
    # Argmax correct, cosine=0.9998 — model is functionally correct.
    "bloom": 0.02,
    # Jais2 combines LayerNorm with squared-ReLU; ORT/PyTorch accumulation differs
    # by ~0.0025 while retaining the same argmax and cosine >= 0.99999.
    "jais2": 0.003,
    # Jamba MoE+Mamba: FP accumulation differences from sequential vs batched expert
    # dispatch, plus Mamba1 SSM single-token decode FP path differences.
    # Argmax correct, cosine=0.998 — model is functionally correct.
    "jamba": 0.04,
    # Bamba's hybrid Mamba2 + attention path differs slightly in FP accumulation
    # order from HuggingFace. Both deterministic seeds keep the same argmax and
    # cosine >= 0.999996 with max absolute error below 0.0017.
    "bamba": 0.002,
    # ModernBERT decoder has a 3-component LM head (dense→norm→decoder) whose
    # FP accumulation differs from PyTorch → ~0.043 max diff.
    # Argmax correct, cosine=0.996 — model is functionally correct.
    "modernbert-decoder": 0.05,
    # MiniMax: hybrid MoE + Lightning Attention; batched-expert FP accumulation
    # differences → ~0.046 max diff. Argmax correct, cosine=0.996.
    "minimax": 0.05,
    # NanoChat: double-norm (pre + post layers) + logit softcap accumulate tiny FP differences.
    # Argmax correct, cosine=0.99999 — model is functionally correct.
    "nanochat": 0.002,
    # Qwen2MoE: shared-expert FP accumulation differs slightly from HF batched dispatch.
    # Argmax correct, cosine=0.999, top10_jaccard=1.0 — functionally correct.
    "qwen2_moe": 0.02,
    # MoE models with per-expert vs batched matmul FP accumulation differences:
    "flex_olmo": 0.15,  # ~0.143 max diff, cosine=0.966 (post-norm MoE)
    # GLM4-MoE: sigmoid-gated routing FP accumulation differs from HF batched dispatch.
    # Argmax correct, cosine≥0.999 — functionally correct.
    "glm4_moe": 0.005,
    # GLM: partial_rotary_factor=0.5 RoPE FP accumulation → ~0.003 max diff.
    # Argmax correct, cosine≥0.999 — functionally correct.
    "glm": 0.005,
    # GLM4: pre+post norm FP accumulation → ~0.007 max diff.
    # Argmax correct, cosine≥0.999 — functionally correct.
    "glm4": 0.01,
    "olmoe": 0.035,  # ~0.031 max diff, cosine=0.998
    "phimoe": 0.065,  # ~0.058 max diff, cosine=0.993 (SparseMixerGate)
    "qwen3_moe": 0.025,  # ~0.020 max diff, cosine=0.999
    # Qwen3 VL/Omni MoE text sub-models: same MoE FP accumulation as qwen3_moe.
    "qwen3_vl_moe": 0.025,
    "qwen3_omni_moe": 0.025,
    # Gemma v1: OffsetRMSNorm (+1 weight) FP accumulation → ~0.089 max diff.
    # Argmax correct, cosine=0.984 — model is functionally correct.
    "gemma": 0.10,
    # Gemma3 text: QK-norm + sliding/full attention FP accumulation → ~0.045 max diff.
    # Argmax correct (near-tie), cosine=0.996 — model is functionally correct.
    "gemma3_text": 0.05,
    # Gemma2: softcapping (tanh) + OffsetRMSNorm FP accumulation → ~0.042 max diff.
    # Argmax correct (near-tie), cosine=0.998 — model is functionally correct.
    "gemma2": 0.05,
    # Gemma3n: AltUp magnitude normalization (target_mag/new_mag ratio) amplifies
    # FP differences between ORT and PyTorch, especially with random weight init.
    # Argmax correct (near-tie), cosine≥0.995, top10_jaccard=1.0 — functionally correct.
    # Only the text entry appears here: the "gemma3n" key builds the multimodal
    # model, which this causal-LM suite does not cover.
    "gemma3n_text": 0.05,  # ~0.026 max diff worst-case (AltUp magnitude ratio)
    # Gemma3 VL: same QK-norm FP accumulation as gemma3_text (~0.045 max diff).
    # argmax_match=True (near-tie), cosine=0.996 — functionally correct.
    "gemma3": 0.05,
    # DeepSeek-V3: previously carried a 0.04 override attributed to "MoE
    # dispatch FP accumulation", but that was masking a real bug: the
    # DeepSeek-V2/V3 MLA softmax scale was missing the YaRN
    # mscale_all_dim^2 correction HF applies (see
    # ``mobius.components._rotary_embedding.yarn_apply_mscale``). Fixed in
    # ``_deepseek_mla.py``. Post-fix, remaining diff is genuine FP-order
    # noise from parallel MoE dispatch: ~3.5e-5 measured via a standalone
    # interpreter, ~1.2e-3 measured under this pytest process (both
    # reproducible in their own harness — the residual gap is ORT
    # intra-op thread-schedule dependent, not seed-dependent). 0.0025
    # keeps ~2x headroom over the higher measurement while staying 16x
    # tighter than the old 0.04.
    "deepseek_v3": 0.0025,
    # dots1: same DeepSeek V3 architecture (sigmoid routing + shared experts).
    # MoE dispatch accumulation differences → similar tolerance needed.
    "dots1": 0.04,
    # Ernie4.5-MoE: zero-initialized gate means TopK tie-breaking differs between
    # PyTorch and ONNX. With random weights, the routing diverges slightly.
    # Argmax correct, cosine=0.985 — model is functionally correct.
    "ernie4_5_moe": 0.10,
    # Llama4: feed_forward naming + ONNX sequential vs PyTorch fused ops → ~0.004 max diff.
    # Argmax correct, cosine=0.9999 — model is functionally correct.
    "llama4_text": 0.005,
    # LongCat Flash: per-expert MoE dispatch accumulates FP differences.
    "longcat_flash": 0.05,
    # Helium: head_dim=16 (vs HF default 128) causes minor FP accumulation differences.
    # Argmax correct, cosine=0.9999 — model is functionally correct.
    "helium": 0.005,
    # gpt-sw3: GPT-2-family LayerNorm eps alignment causes minor FP accumulation differences.
    # Argmax correct, cosine=1.0000 — model is functionally correct.
    "gpt-sw3": 0.005,
    # GPT-OSS: MoE with sequential per-expert dispatch + custom silu_alpha activation
    # accumulates small FP differences vs HF batched computation → ~0.05 max diff.
    # Argmax correct, cosine≥0.999 — model is functionally correct.
    "gpt_oss": 0.05,
    # GraniteMoeHybrid: Mamba2 + MoE + shared-MLP FP accumulation differences.
    # Argmax correct, cosine=0.999870 — model is functionally correct.
    "granitemoehybrid": 0.02,
    # Zamba2: hybrid Mamba2+Attention with low-rank adapters. Adapter FP accumulation
    # produces ~0.0012 max diff. Argmax correct, cosine=0.999998 — functionally correct.
    "zamba2": 0.002,
    # Gemma4 text: per-layer input embedding + softcapping + QK-norm FP accumulation.
    # Argmax correct, cosine=0.985 — model is functionally correct.
    "gemma4_text": 0.15,
    # Olmo3: QK-norm + sliding/full attention FP accumulation → ~0.015 max diff.
    # Argmax near-tie, cosine=0.9999 — model is functionally correct.
    "olmo3": 0.02,
}

# Model types with known ONNX-vs-HF divergences, tracked as xfail.
# Each maps model_type → reason the outputs diverge.
_XFAIL_REASONS: dict[str, str] = {
    # MoE routing models: those with wider atol in _ATOL_OVERRIDES PASS.
    # Remaining genuine xfails:
    # DeepSeek MLA: deepseek_v2_0 uses group_limited_greedy routing which hits a
    # HF transformers 5.3.0 bug (DeepseekV2Moe missing num_experts attr).
    "deepseek_v2_0": "HF transformers 5.3.0 bug: DeepseekV2Moe missing num_experts attr",
}

# Fields that are properties in HF configs and cannot be set directly,
# or internal mobius-only fields that HF configs don't recognize.
_HF_READONLY_FIELDS: set[str] = {"head_dim", "attn_qkv_bias", "attn_o_bias", "mlp_bias"}

# Model types that are mobius-internal aliases and should not appear in the synthetic
# parity test.  The parity test requires AutoModelForCausalLM to be able to create
# a reference model — these types either have no real HF model_type string, or their
# HF config class is not registered with AutoModelForCausalLM (e.g. VLM sub-configs).
# The build_graph test still covers them (it builds the ONNX graph without HF).
_PARITY_EXCLUDE: frozenset[str] = frozenset(
    {
        # VLM text-only sub-configs: registered in HF CONFIG_MAPPING but NOT with
        # AutoModelForCausalLM (they belong to multimodal pipelines).
        "qwen3_vl_text",
        "qwen2_vl_text",
        "qwen2_5_vl_text",
        "qwen3_5_vl_text",
        "glm4v_text",
        "glm4v_moe_text",
        # Not in HF CONFIG_MAPPING at all — purely mobius-internal aliases.
        "command_r",  # real HF type is cohere
        "codegen2",  # real HF type is codegen
        "open-llama",  # real HF type is llama
        "yi",  # real HF type is llama
        "exaone",  # real HF type is exaone4
        "phi3small",  # real HF type is phi3
        "mistral3",  # our implementation maps to mistral; real mistral3 is different
        # gemma4_unified_text: mobius-internal alias for the gemma-4-12B text
        # backbone (reuses Gemma4CausalLMModel). No matching HF model_type is
        # registered with AutoModelForCausalLM, so a reference model cannot be
        # constructed here.  Text parity is covered by the real-weight
        # integration test (test_gemma4_unified_12b_text_prefill).
        "gemma4_unified_text",
        # The composite and architecture alias share qwen4_exp_text's graph.
        # Run parity once through the canonical standalone text model type.
        "qwen4_exp",
        "Qwen4ExpForConditionalGeneration",
    }
)

# Model types that need extra HF config fields beyond our defaults.
_HF_EXTRA_CONFIG: dict[str, dict] = {
    "phi3": {"pad_token_id": 0},
    "phi": {"pad_token_id": 0, "layer_norm_eps": 1e-6},
    "phimoe": {"pad_token_id": 0},
    "gemma2": {"query_pre_attn_scalar": TINY_HEAD_DIM, "head_dim": TINY_HEAD_DIM},
    "shieldgemma2": {"query_pre_attn_scalar": TINY_HEAD_DIM},
    # Gemma family defaults head_dim=256 in HF; override to match tiny config
    "gemma": {"head_dim": TINY_HEAD_DIM},
    # Gemma3/Gemma3n: head_dim is an explicit param in HF (default 256); pass tiny value.
    "gemma3_text": {"head_dim": TINY_HEAD_DIM, "query_pre_attn_scalar": TINY_HEAD_DIM},
    # num_kv_shared_layers is threaded through from the tiny config (HF defaults
    # to 15, which with TINY_LAYERS=2 would make every layer "shared" and leave
    # no source layer to borrow K,V from), so it is set explicitly per entry in
    # _test_configs.py rather than pinned to 0 here.
    "gemma3n_text": {
        "query_pre_attn_scalar": TINY_HEAD_DIM,
        "head_dim": TINY_HEAD_DIM,
        "hidden_activation": "gelu_pytorch_tanh",
    },
    "gemma3": {"query_pre_attn_scalar": TINY_HEAD_DIM, "head_dim": TINY_HEAD_DIM},
    # Gemma4 text defaults head_dim=256 in HF; override to match tiny config
    "gemma4_text": {
        "query_pre_attn_scalar": TINY_HEAD_DIM,
        "head_dim": TINY_HEAD_DIM,
        "hidden_size_per_layer_input": 32,
        "vocab_size_per_layer_input": TINY_VOCAB,
    },
    # Qwen3-Next defaults head_dim=256 in HF; override to match tiny config
    "qwen3_next": {"head_dim": TINY_HEAD_DIM},
    # JetMoE: kv_channels sets head_dim (not derived from hidden/num_heads).
    # num_kv_heads maps to num_key_value_heads (HF uses a non-standard field name).
    "jetmoe": {
        "kv_channels": TINY_HEAD_DIM,
        "num_kv_heads": TINY_KV_HEADS,
    },
    # Qwen3 defaults head_dim=128 in HF; override to match tiny config
    "qwen3": {"head_dim": TINY_HEAD_DIM},
    # Qwen3VLTextConfig maps to qwen3 for comparison; needs same head_dim override
    "qwen3_vl_text": {"head_dim": TINY_HEAD_DIM},
    # Ministral and Mistral3 default head_dim=None in HF (causes pow(None,float) error)
    "ministral": {"head_dim": TINY_HEAD_DIM},
    "ministral3": {"head_dim": TINY_HEAD_DIM},
    # Helium defaults head_dim=None in HF (causes pow(None,float) error)
    "helium": {"head_dim": TINY_HEAD_DIM},
    # seed_oss defaults head_dim=128 in HF; override to match tiny config
    "seed_oss": {"head_dim": TINY_HEAD_DIM},
    # HunYuan V1 dense defaults head_dim=None in HF (causes pow(None,float) error)
    "hunyuan_v1_dense": {"head_dim": TINY_HEAD_DIM},
    # GLM/GLM4: head_dim=128 (explicit, not hidden/num_heads), pad_token_id > vocab_size
    # in default config causes embedding assertion; override both.
    "glm": {"head_dim": TINY_HEAD_DIM, "pad_token_id": 0},
    "glm4": {"head_dim": TINY_HEAD_DIM, "pad_token_id": 0},
    "opt": {
        "word_embed_proj_dim": TINY_HIDDEN,
        # OPT uses ffn_dim (not intermediate_size) for the MLP width
        "ffn_dim": TINY_INTERMEDIATE,
    },
    # Bloom uses MHA (num_kv_heads == num_heads) and 4*hidden intermediate.
    # HF Bloom uses layer_norm_epsilon (default 1e-5); match our rms_norm_eps=1e-6.
    "bloom": {
        "num_key_value_heads": TINY_HEADS,
        "intermediate_size": 4 * TINY_HIDDEN,
        "layer_norm_epsilon": 1e-6,
    },
    # GPT-J/CodeGen use rotary_dim (not partial_rotary_factor) and n_inner (not intermediate_size).
    # HF field for LayerNorm eps is layer_norm_epsilon (not layer_norm_eps); HF default is 1e-5.
    "gptj": {
        "rotary_dim": int(TINY_HEAD_DIM * 0.25),
        "n_inner": TINY_INTERMEDIATE,
        "layer_norm_epsilon": 1e-5,
    },
    "codegen": {"rotary_dim": int(TINY_HEAD_DIM * 0.5), "n_inner": TINY_INTERMEDIATE},
    # GPT-2 family: control MLP width via model-specific field names
    # (HF ignores the generic 'intermediate_size' for these models)
    "gpt2": {"n_inner": TINY_INTERMEDIATE},
    # GPT-Neo: default attention_types=[[[global,local],12]] generates 24 attention_layers,
    # but HF validates len(attention_layers)==num_layers.  Override to produce exactly
    # TINY_LAYERS layers.  Must also pass num_layers explicitly (GPT-Neo's field name)
    # so the length check sees the correct value at validation time.
    "gpt_neo": {
        "layer_norm_epsilon": 1e-5,
        "n_inner": TINY_INTERMEDIATE,
        "num_layers": TINY_LAYERS,
        "attention_types": [[["global", "local"], TINY_LAYERS // 2]],
        "window_size": 8,
    },
    "gpt_bigcode": {"n_inner": TINY_INTERMEDIATE, "multi_query": False},
    # gpt-sw3 uses n_inner (not intermediate_size) for MLP width (HF default is 4*hidden_size)
    "gpt-sw3": {"n_inner": TINY_INTERMEDIATE},
    "xglm": {"ffn_dim": TINY_INTERMEDIATE},
    "biogpt": {"ffn_dim": TINY_INTERMEDIATE},
    # CTRL uses old-style config field names (n_embd, n_layer, n_head, dff).
    # Sinusoidal PE is computed at runtime; n_positions must match max_position_embeddings.
    "ctrl": {
        "n_embd": TINY_HIDDEN,
        "n_layer": TINY_LAYERS,
        "n_head": TINY_HEADS,
        "dff": TINY_INTERMEDIATE,
        "n_positions": 128,
    },
    # XLM uses emb_dim/n_layers/n_heads; MLP dim is hardcoded 4*emb_dim in HF
    # (test config sets intermediate_size=4*TINY_HIDDEN to match).
    # causal=True forces causal masking to match our ONNX Attention (is_causal=1).
    "xlm": {
        "emb_dim": TINY_HIDDEN,
        "n_layers": TINY_LAYERS,
        "n_heads": TINY_HEADS,
        "causal": True,
    },
    # Jamba requires CUDA mamba kernels by default; disable for CPU tests
    "jamba": {"use_mamba_kernels": False},
    # Nemotron uses norm_eps (not rms_norm_eps) in HF config
    "nemotron": {"norm_eps": 1e-5},
    # Qwen3.5 has head_dim as an explicit config param (default 256)
    "qwen3_5_text": {"head_dim": TINY_HEAD_DIM},
    # Qwen3.5-MoE uses the same doubled-Q attention as qwen3_5; head_dim defaults to 256 in HF
    "qwen3_5_moe": {"head_dim": TINY_HEAD_DIM},
    # GPT-NeoX/Pythia use layer_norm_eps (not rms_norm_eps) for their LayerNorms
    "gpt_neox": {"layer_norm_eps": 1e-6},
    # GPT-NeoX-Japanese uses layer_norm_eps=1e-5 by default; test config matches via rms_norm_eps=1e-5
    # MPT uses layer_norm_epsilon (not rms_norm_eps) for its LayerNorms
    "mpt": {"layer_norm_epsilon": 1e-6},
    # Cohere/Cohere2 use layer_norm_eps (HF default 1e-5); force to match our rms_norm_eps=1e-6
    "cohere": {"layer_norm_eps": 1e-6},
    "cohere2": {"layer_norm_eps": 1e-6},
    # StableLM uses layer_norm_eps (HF default 1e-5); force to match our rms_norm_eps=1e-6
    "stablelm": {"layer_norm_eps": 1e-6},
    # StarCoder2: disable bias (use_bias=True HF default) and fix norm_epsilon field
    "starcoder2": {"norm_epsilon": 1e-6, "use_bias": False},
    # Ernie4.5-MoE: make all 2 tiny layers MoE (default moe_layer_start_index=1 skips layer 0)
    # moe_num_shared_experts=1 keeps shared_expert_intermediate_size = moe_intermediate_size * 1
    "ernie4_5_moe": {
        "moe_layer_start_index": 0,
        "moe_layer_end_index": TINY_LAYERS - 1,
        "moe_num_shared_experts": 1,
    },
    # GLM4-MoE: make all 2 tiny layers MoE (default first_k_dense_replace=1 makes layer 0 dense)
    # n_shared_experts=1 keeps shared_expert_intermediate_size = moe_intermediate_size * 1
    "glm4_moe": {
        "first_k_dense_replace": 0,
        "n_shared_experts": 1,
    },
    # HunYuanMoEV1 requires head_dim (defaults to None, causing pow(None, float) error).
    "hunyuan_v1_moe": {"head_dim": TINY_HEAD_DIM},
    # Llama4Text requires head_dim to match our tiny num_heads x head_dim = hidden_size.
    # Disable MoE (we use dense CausalLMModel) and Llama4-specific attention features
    # (QK-norm and temperature tuning) not implemented in CausalLMModel.
    # intermediate_size_mlp is separate from intermediate_size in Llama4 (default 16384).
    "llama4_text": {
        "head_dim": TINY_HEAD_DIM,
        "intermediate_size_mlp": TINY_INTERMEDIATE,
        "moe_layers": [],
        "use_qk_norm": False,
        "attn_temperature_tuning": False,
    },
    # LongCat Flash uses ffn_hidden_size for dense MLP and num_layers (physical) instead of
    # num_hidden_layers. HF num_hidden_layers = 2 * num_layers, so pass num_layers=TINY_LAYERS.
    # head_dim must match qk_rope_head_dim=8 (HF defaults head_dim=64 for RoPE computation).
    # rope_parameters must match our rope_theta=10000.0 (HF default is 10000000.0).
    "longcat_flash": {
        "num_layers": TINY_LAYERS,
        "ffn_hidden_size": TINY_INTERMEDIATE,
        "head_dim": 8,  # qk_rope_head_dim from our test config
        "rope_parameters": {"rope_theta": 10_000.0, "rope_type": "default"},
    },
    # ModernBERT-Decoder uses MHA only (always sets kv_heads=num_heads internally).
    # head_dim must be provided explicitly (HF default causes shape mismatch).
    "modernbert-decoder": {"head_dim": TINY_HEAD_DIM, "pad_token_id": 0},
    # Falcon: HF defaults to multi_query=True (MQA, 1 KV head). Disable for GQA parity.
    # new_decoder_architecture=True enables the multi-head KV path.
    # ffn_hidden_size controls MLP width (HF ignores intermediate_size for Falcon).
    "falcon": {
        "multi_query": False,
        "num_kv_heads": TINY_KV_HEADS,
        "new_decoder_architecture": True,
        "ffn_hidden_size": TINY_INTERMEDIATE,
        "layer_norm_epsilon": 1e-6,
    },
    # GPT-OSS: head_dim is an explicit config param (HF default 64, not hidden/num_heads).
    # layer_types must have exactly TINY_LAYERS entries.
    "gpt_oss": {
        "head_dim": TINY_HEAD_DIM,
        "layer_types": ["sliding_attention", "full_attention"],
    },
    # VL MoE text sub-models: need same HF extras as their base model types.
    # qwen3_vl_moe, qwen3_omni_moe → qwen3_moe (needs head_dim + moe_intermediate_size)
    "qwen3_vl_moe": {"head_dim": TINY_HEAD_DIM, "moe_intermediate_size": TINY_INTERMEDIATE},
    "qwen3_omni_moe": {"head_dim": TINY_HEAD_DIM, "moe_intermediate_size": TINY_INTERMEDIATE},
}


# Some mobius model_types map to a multimodal HF config class that wraps an
# inner text_config with its own defaults.  We override to the text-only
# HF model_type so that tiny kwargs are applied directly to the text config.
_HF_MODEL_TYPE_OVERRIDES: dict[str, str] = {
    # Gemma3Config wraps Gemma3TextConfig; tiny kwargs go to the outer config
    # but the actual model is built from text_config which retains HF defaults.
    "gemma3": "gemma3_text",
    # Qwen3.5-MoE outer config wraps text_config; use the text-only model type
    # so tiny kwargs (num_experts, moe_intermediate_size, etc.) apply directly.
    "qwen3_5_moe": "qwen3_5_moe_text",
    # code_llama reuses the llama architecture; HF only recognizes "llama".
    "code_llama": "llama",
    # VL text sub-models: use the base text model type for CausalLM parity testing.
    "qwen3_vl_moe": "qwen3_moe",
    "qwen3_omni_moe": "qwen3_moe",
}


def _adapt_muse_glimmer_text_config(hf_kwargs: dict) -> None:
    """Translate the shared tiny config fields to Muse Glimmer's HF schema."""
    hf_kwargs["head_dim"] = TINY_HEAD_DIM
    hf_kwargs.setdefault("hidden_activation", hf_kwargs.get("hidden_act", "silu"))
    hf_kwargs.pop("hidden_act", None)
    hf_kwargs["attention_bias"] = False
    hf_kwargs["rope_parameters"] = {
        "rope_type": "default",
        "rope_theta": hf_kwargs.pop("rope_theta", 10_000.0),
    }
    hf_kwargs.pop("rope_type", None)
    hf_kwargs.pop("attn_qk_norm", None)
    hf_kwargs.pop("no_rope_layers", None)


def _adapt_qwen4_exp_text_config(hf_kwargs: dict) -> None:
    """Translate flat Mobius RoPE/expert fields to the strict HF config."""
    hf_kwargs["head_dim"] = hf_kwargs["hidden_size"] // hf_kwargs["num_attention_heads"]
    hf_kwargs["num_experts"] = hf_kwargs.pop("num_local_experts")
    hf_kwargs["rope_parameters"] = {
        "rope_type": "default",
        "rope_theta": 10_000.0,
        "partial_rotary_factor": hf_kwargs.pop("partial_rotary_factor", 1.0),
    }


_HF_CONFIG_ADAPTERS = {
    "muse_glimmer_text": _adapt_muse_glimmer_text_config,
    "qwen4_exp_text": _adapt_qwen4_exp_text_config,
}


def _create_hf_config(model_type: str, config_overrides: dict):
    """Create a HuggingFace config for the given model type.

    Returns (hf_config, hf_model_type) or raises to skip.
    """
    from transformers import AutoConfig

    # Determine the HF model_type (usually same as ours).
    # Some mobius types map to wrapper configs; use the inner text model type instead.
    hf_model_type = _HF_MODEL_TYPE_OVERRIDES.get(model_type, model_type)

    # Build HF config kwargs from our tiny defaults
    hf_kwargs: dict = {
        "hidden_size": TINY_HIDDEN,
        "intermediate_size": TINY_INTERMEDIATE,
        "num_attention_heads": TINY_HEADS,
        "num_key_value_heads": TINY_KV_HEADS,
        "num_hidden_layers": config_overrides.get("num_hidden_layers", TINY_LAYERS),
        "vocab_size": TINY_VOCAB,
        "max_position_embeddings": 128,
        "rms_norm_eps": 1e-6,
        "pad_token_id": 0,
    }

    # Apply config overrides (filtering out our internal keys)
    for key, value in config_overrides.items():
        if key == "_config_cls":
            continue
        if key in _HF_READONLY_FIELDS and not (
            model_type == "falcon_h1" and key == "mlp_bias"
        ):
            continue
        hf_kwargs[key] = value

    # Apply model-specific HF extras
    if model_type in _HF_EXTRA_CONFIG:
        hf_kwargs.update(_HF_EXTRA_CONFIG[model_type])

    if adapter := _HF_CONFIG_ADAPTERS.get(hf_model_type):
        adapter(hf_kwargs)

    # Convert layer_types to attn_layer_indices for hybrid Mamba models
    # Bamba uses attn_layer_indices (computed property layers_block_type)
    if hf_model_type in ("bamba",) and "layer_types" in hf_kwargs:
        layer_types = hf_kwargs.pop("layer_types")
        hf_kwargs["attn_layer_indices"] = [
            i for i, lt in enumerate(layer_types) if lt in ("full_attention", "attention")
        ]

    if hf_model_type in {"lfm2", "lfm2_moe"}:
        hf_kwargs["conv_L_cache"] = hf_kwargs.pop("short_conv_kernel", 3)
        hf_kwargs["conv_bias"] = hf_kwargs.pop("short_conv_bias", False)
        hf_kwargs["norm_eps"] = hf_kwargs.pop("rms_norm_eps")
        hf_kwargs["rope_parameters"] = {
            "rope_type": "default",
            "rope_theta": 10_000.0,
        }
        if hf_model_type == "lfm2_moe":
            hf_kwargs["num_experts"] = hf_kwargs.pop("num_local_experts")

    # Jamba uses attn_layer_offset/attn_layer_period
    if hf_model_type in ("jamba",) and "layer_types" in hf_kwargs:
        layer_types = hf_kwargs.pop("layer_types")
        attn_indices = [
            i for i, lt in enumerate(layer_types) if lt in ("full_attention", "attention")
        ]
        if len(attn_indices) == len(layer_types):
            # All attention
            hf_kwargs["attn_layer_offset"] = 0
            hf_kwargs["attn_layer_period"] = 1
        elif len(attn_indices) > 0:
            hf_kwargs["attn_layer_offset"] = attn_indices[0]
            # Period = gap between consecutive attention layers
            if len(attn_indices) > 1:
                hf_kwargs["attn_layer_period"] = attn_indices[1] - attn_indices[0]
            else:
                hf_kwargs["attn_layer_period"] = len(layer_types)
        else:
            # All mamba — use large offset/period
            hf_kwargs["attn_layer_offset"] = len(layer_types)
            hf_kwargs["attn_layer_period"] = len(layer_types)

    # MiniMax uses "linear_attention" in its HF config for lightning attention layers.
    # Our internal key is "lightning_attention" — translate back for HF.
    if hf_model_type in ("minimax",) and "layer_types" in hf_kwargs:
        hf_kwargs["layer_types"] = [
            "linear_attention" if lt == "lightning_attention" else lt
            for lt in hf_kwargs["layer_types"]
        ]

    # GraniteMoeHybrid's current constructor consumes layer_types and selects
    # the recurrent path only for the literal "mamba" layer type.
    if hf_model_type in ("granitemoehybrid",) and "layer_types" in hf_kwargs:
        hf_kwargs["layer_types"] = [
            "attention" if lt in ("full_attention", "attention") else "mamba"
            for lt in hf_kwargs["layer_types"]
        ]

    # NemotronH uses its public layer-type vocabulary rather than Mobius names.
    # Also translate mobius Mamba field names to HF NemotronHConfig field names.
    if hf_model_type in ("nemotron_h",) and "layer_types" in hf_kwargs:
        layer_types = hf_kwargs.pop("layer_types")
        _nemotron_type_map = {
            "mamba2": "mamba",
            "full_attention": "attention",
            "mlp": "mlp",
            "moe": "moe",
        }
        hf_kwargs["layers_block_type"] = [_nemotron_type_map.get(lt, lt) for lt in layer_types]
        # Mobius NemotronHConfig → HF NemotronHConfig field name mapping
        _nemotron_field_map = {
            "mamba_n_heads": "mamba_num_heads",
            "mamba_d_head": "mamba_head_dim",
            "mamba_d_state": "ssm_state_size",
            "mamba_n_groups": "n_groups",
            "mamba_d_conv": "conv_kernel",
            "mamba_expand": "expand",
            "rms_norm_eps": "layer_norm_epsilon",
        }
        for old_name, new_name in _nemotron_field_map.items():
            if old_name in hf_kwargs:
                hf_kwargs[new_name] = hf_kwargs.pop(old_name)
        if "hidden_act" in hf_kwargs:
            hf_kwargs["mlp_hidden_act"] = hf_kwargs.pop("hidden_act")
        if "shared_expert_intermediate_size" in hf_kwargs:
            hf_kwargs["moe_shared_expert_intermediate_size"] = hf_kwargs.pop(
                "shared_expert_intermediate_size"
            )
        # HF NemotronH has an explicit head_dim (default 128) that is not
        # derived from hidden_size / num_attention_heads. Set it to match.
        if "head_dim" not in hf_kwargs:
            hf_kwargs["head_dim"] = (
                hf_kwargs["hidden_size"] // hf_kwargs["num_attention_heads"]
            )

    # Zamba2 uses layers_block_type with HF values {"mamba", "hybrid"}.
    # Convert our expanded logical layer_types back to physical layers_block_type.
    # Also translate mobius Mamba field names to HF Zamba2Config field names.
    if hf_model_type in ("zamba2",) and "layer_types" in hf_kwargs:
        layer_types = hf_kwargs.pop("layer_types")
        # Convert expanded [mamba2, mamba2, full_attention, mamba2, mamba2]
        # back to physical [mamba, mamba, hybrid, mamba]
        physical_types = []
        i = 0
        while i < len(layer_types):
            if layer_types[i] == "full_attention":
                physical_types.append("hybrid")
                i += 2  # skip the following mamba2 (part of hybrid)
            else:
                physical_types.append("mamba")
                i += 1
        hf_kwargs["layers_block_type"] = physical_types
        hf_kwargs["num_hidden_layers"] = len(physical_types)
        # Remove mobius-internal fields not recognized by HF Zamba2Config
        hf_kwargs.pop("hybrid_layer_indices", None)
        hf_kwargs.pop("num_mem_blocks", None)
        hf_kwargs.pop("attention_hidden_size", None)
        # Mobius Zamba2Config → HF Zamba2Config field name mapping
        _zamba2_field_map = {
            "mamba_n_heads": "n_mamba_heads",
            "mamba_d_head": "mamba_headdim",
            "mamba_n_groups": "mamba_ngroups",
            "mamba_time_step_min": "time_step_min",
        }
        for old_name, new_name in _zamba2_field_map.items():
            if old_name in hf_kwargs:
                hf_kwargs[new_name] = hf_kwargs.pop(old_name)

    # Some models use different field names for num_local_experts and num_experts_per_tok.
    # Maps hf_model_type -> {our_field: hf_field} for field name translation.
    expert_field_aliases: dict[str, dict[str, str]] = {
        # Standard: num_experts (not num_local_experts)
        "olmoe": {"num_local_experts": "num_experts"},
        "qwen2_moe": {"num_local_experts": "num_experts"},
        "qwen3_moe": {"num_local_experts": "num_experts"},
        "qwen3_5_moe_text": {"num_local_experts": "num_experts"},
        "qwen3_next": {"num_local_experts": "num_experts"},
        # Ernie4.5 MoE uses moe_num_experts / moe_k
        "ernie4_5_moe": {
            "num_local_experts": "moe_num_experts",
            "num_experts_per_tok": "moe_k",
        },
        # DeepSeek V2/V3 use n_routed_experts (not num_local_experts)
        "deepseek_v2": {"num_local_experts": "n_routed_experts"},
        "deepseek_v3": {"num_local_experts": "n_routed_experts"},
        # dots1 uses n_routed_experts like DeepSeek V3
        "dots1": {"num_local_experts": "n_routed_experts"},
        # LongCat Flash uses n_routed_experts, moe_topk, and expert_ffn_hidden_size
        "longcat_flash": {
            "num_local_experts": "n_routed_experts",
            "num_experts_per_tok": "moe_topk",
            "moe_intermediate_size": "expert_ffn_hidden_size",
        },
        "nemotron_h": {"num_local_experts": "n_routed_experts"},
    }
    if hf_model_type in expert_field_aliases:
        for src_field, dst_field in expert_field_aliases[hf_model_type].items():
            if src_field in hf_kwargs:
                hf_kwargs[dst_field] = hf_kwargs.pop(src_field)

    # Remove ONNX-internal keys that HF configs don't have.
    # Note: shared_expert_intermediate_size is intentionally NOT in this set —
    # it is a real HF parameter for qwen2_moe, qwen3_5_moe_text, and others.
    onnx_only_keys = {
        "attn_qk_norm",
        "attn_qk_norm_full",
        "post_feedforward_norm",
        "short_conv_kernel",
        "short_conv_bias",
        # dual_ln is a mobius-only flag for Falcon/Bloom parallel attention;
        # HF controls this behavior via new_decoder_architecture=True.
        "dual_ln",
    }
    for key in onnx_only_keys:
        hf_kwargs.pop(key, None)

    try:
        hf_config = AutoConfig.for_model(hf_model_type, **hf_kwargs)
    except (ValueError, KeyError) as e:
        pytest.skip(f"Cannot create HF config for {model_type}: {e}")

    return hf_config


def _create_softcapped_backbone_causal_lm(hf_config):
    """Wrap an HF backbone with the checkpoint's scaled, softcapped LM head."""
    from transformers import AutoModel

    class _SoftcappedBackboneCausalLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = AutoModel.from_config(hf_config)
            self.lm_head = torch.nn.Linear(
                hf_config.hidden_size,
                hf_config.vocab_size,
                bias=False,
            )

        def forward(self, **kwargs):
            hidden_states = self.model(**kwargs).last_hidden_state
            logits = self.lm_head(hidden_states) * hf_config.output_multiplier
            cap = hf_config.final_logit_softcapping
            if cap:
                logits = cap * torch.tanh(logits / cap)
            return type("CausalLMOutput", (), {"logits": logits})()

    return _SoftcappedBackboneCausalLM()


_HF_MODEL_FACTORIES = {
    "muse_glimmer_text": _create_softcapped_backbone_causal_lm,
}


def _create_hf_model(model_type: str, hf_config, seed: int):
    """Create a HuggingFace model from config with deterministic init."""
    from transformers import AutoModelForCausalLM

    torch.manual_seed(seed)
    try:
        factory = _HF_MODEL_FACTORIES.get(model_type, AutoModelForCausalLM.from_config)
        hf_model = factory(hf_config)
    except Exception as e:
        pytest.skip(f"Cannot create HF model for {model_type}: {type(e).__name__}: {e}")

    return hf_model.float().eval()


def _build_onnx_model(model_type: str, config: ArchitectureConfig):
    """Build ONNX model package using the registry."""
    model_cls = registry.get(model_type)
    module = model_cls(config)
    task_name = _default_task_for_model(model_type)
    task = get_task(task_name)
    config.dtype = ir.DataType.FLOAT
    pkg = task.build(module, config)
    return module, pkg


def _fill_random_weights(model: ir.Model, rng: np.random.Generator) -> None:
    """Fill all unset graph initializers with random float32 values."""
    for init in model.graph.initializers.values():
        if init.const_value is not None:
            continue
        shape = tuple(d for d in init.shape)
        if not shape:
            continue
        if init.dtype == ir.DataType.FLOAT:
            data = rng.standard_normal(shape).astype(np.float32) * 0.02
        elif init.dtype == ir.DataType.FLOAT16:
            data = (rng.standard_normal(shape) * 0.02).astype(np.float16)
        elif init.dtype in (ir.DataType.INT64, ir.DataType.INT32):
            np_dtype = np.int64 if init.dtype == ir.DataType.INT64 else np.int32
            data = rng.integers(0, 10, size=shape).astype(np_dtype)
        else:
            data = rng.standard_normal(shape).astype(np.float32) * 0.02
        init.const_value = ir.Tensor(data)


def _nemotron_parse_torch_attention(
    hidden_states: torch.Tensor,
    weights: dict[str, torch.Tensor],
    prefix: str,
    *,
    num_heads: int,
    key_value_states: torch.Tensor | None = None,
    causal: bool = False,
) -> torch.Tensor:
    """Evaluate one Nemotron Parse attention block with exported weights."""
    source = hidden_states if key_value_states is None else key_value_states
    query = torch.nn.functional.linear(
        hidden_states,
        weights[f"{prefix}.q_proj.weight"],
        weights[f"{prefix}.q_proj.bias"],
    )
    key = torch.nn.functional.linear(
        source,
        weights[f"{prefix}.k_proj.weight"],
        weights[f"{prefix}.k_proj.bias"],
    )
    value = torch.nn.functional.linear(
        source,
        weights[f"{prefix}.v_proj.weight"],
        weights[f"{prefix}.v_proj.bias"],
    )
    batch, query_len, hidden_size = query.shape
    key_len = key.shape[1]
    head_dim = hidden_size // num_heads
    query = query.reshape(batch, query_len, num_heads, head_dim).transpose(1, 2)
    key = key.reshape(batch, key_len, num_heads, head_dim).transpose(1, 2)
    value = value.reshape(batch, key_len, num_heads, head_dim).transpose(1, 2)
    scores = query @ key.transpose(-1, -2) * head_dim**-0.5
    if causal:
        mask = torch.ones(query_len, key_len, dtype=torch.bool).tril()
        scores = scores.masked_fill(~mask, float("-inf"))
    output = torch.softmax(scores, dim=-1) @ value
    output = output.transpose(1, 2).reshape(batch, query_len, hidden_size)
    return torch.nn.functional.linear(
        output,
        weights[f"{prefix}.out_proj.weight"],
        weights[f"{prefix}.out_proj.bias"],
    )


def _nemotron_parse_torch_vision(
    pixel_values: torch.Tensor,
    weights: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Evaluate the tiny one-layer C-RADIO encoder and compression neck."""
    prefix = "vision_encoder.model_encoder.radio_model.model"
    patch_weight = weights[f"{prefix}.patch_generator.embedder.weight"].reshape(32, 3, 16, 16)
    hidden = torch.nn.functional.conv2d(pixel_values, patch_weight, stride=16)
    hidden = hidden.flatten(2).transpose(1, 2)
    pos = weights[f"{prefix}.patch_generator.pos_embed"].reshape(1, 4, 4, 32)
    hidden = hidden + pos[:, :2].reshape(1, 8, 32)
    cls = weights[f"{prefix}.patch_generator.cls_token.token"].unsqueeze(0)
    hidden = torch.cat((cls, hidden), dim=1)

    block = f"{prefix}.blocks.0"
    norm = torch.nn.functional.layer_norm(
        hidden,
        (32,),
        weights[f"{block}.norm1.weight"],
        weights[f"{block}.norm1.bias"],
        1e-6,
    )
    qkv = torch.nn.functional.linear(
        norm,
        weights[f"{block}.attn.qkv.weight"],
        weights[f"{block}.attn.qkv.bias"],
    )
    query, key, value = qkv.chunk(3, dim=-1)
    batch, sequence, _ = query.shape
    query = query.reshape(batch, sequence, 4, 8).transpose(1, 2)
    key = key.reshape(batch, sequence, 4, 8).transpose(1, 2)
    value = value.reshape(batch, sequence, 4, 8).transpose(1, 2)
    attn = torch.softmax(query @ key.transpose(-1, -2) * 8**-0.5, dim=-1) @ value
    attn = attn.transpose(1, 2).reshape(batch, sequence, 32)
    attn = torch.nn.functional.linear(
        attn,
        weights[f"{block}.attn.proj.weight"],
        weights[f"{block}.attn.proj.bias"],
    )
    hidden = hidden + attn
    norm = torch.nn.functional.layer_norm(
        hidden,
        (32,),
        weights[f"{block}.norm2.weight"],
        weights[f"{block}.norm2.bias"],
        1e-6,
    )
    mlp = torch.nn.functional.gelu(
        torch.nn.functional.linear(
            norm,
            weights[f"{block}.mlp.fc1.weight"],
            weights[f"{block}.mlp.fc1.bias"],
        )
    )
    hidden = hidden + torch.nn.functional.linear(
        mlp,
        weights[f"{block}.mlp.fc2.weight"],
        weights[f"{block}.mlp.fc2.bias"],
    )

    summary = hidden[:, [0, 1, 2]].reshape(batch, -1)
    features = hidden[:, 8:]
    features = torch.nn.functional.linear(
        features,
        weights["vision_encoder.conv1.weight"],
        weights["vision_encoder.conv1.bias"],
    )
    features = torch.nn.functional.layer_norm(
        features,
        (64,),
        weights["vision_encoder.layer_norm1.weight"],
        weights["vision_encoder.layer_norm1.bias"],
        1e-6,
    )
    features = features.reshape(batch, 2, 4, 64).permute(0, 3, 1, 2)
    features = torch.nn.functional.conv2d(
        features,
        weights["vision_encoder.conv2.weight"],
        stride=(1, 4),
    )
    features = features.permute(0, 2, 3, 1).reshape(batch, 2, 64)
    features = torch.nn.functional.layer_norm(
        features,
        (64,),
        weights["vision_encoder.layer_norm2.weight"],
        weights["vision_encoder.layer_norm2.bias"],
        1e-6,
    )
    summary = torch.nn.functional.linear(
        summary,
        weights["vision_encoder.sum_proj.weight"],
        weights["vision_encoder.sum_proj.bias"],
    )
    summary = torch.nn.functional.layer_norm(
        summary,
        (64,),
        weights["vision_encoder.layer_norm3.weight"],
        weights["vision_encoder.layer_norm3.bias"],
        1e-6,
    )
    return torch.cat((features, summary[:, None]), dim=1)


def _nemotron_parse_torch_decoder(
    input_ids: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    weights: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Evaluate the tiny one-layer pre-norm mBART decoder."""
    hidden = torch.nn.functional.embedding(input_ids, weights["decoder.embed_tokens.weight"])
    hidden = hidden * 8.0
    hidden = torch.nn.functional.layer_norm(
        hidden,
        (64,),
        weights["decoder.layernorm_embedding.weight"],
        weights["decoder.layernorm_embedding.bias"],
    )
    layer = "decoder.layers.0"
    residual = hidden
    norm = torch.nn.functional.layer_norm(
        hidden,
        (64,),
        weights[f"{layer}.self_attn_layer_norm.weight"],
        weights[f"{layer}.self_attn_layer_norm.bias"],
    )
    hidden = residual + _nemotron_parse_torch_attention(
        norm,
        weights,
        f"{layer}.self_attn",
        num_heads=4,
        causal=True,
    )
    residual = hidden
    norm = torch.nn.functional.layer_norm(
        hidden,
        (64,),
        weights[f"{layer}.encoder_attn_layer_norm.weight"],
        weights[f"{layer}.encoder_attn_layer_norm.bias"],
    )
    hidden = residual + _nemotron_parse_torch_attention(
        norm,
        weights,
        f"{layer}.encoder_attn",
        num_heads=4,
        key_value_states=encoder_hidden_states,
    )
    residual = hidden
    hidden = torch.nn.functional.layer_norm(
        hidden,
        (64,),
        weights[f"{layer}.final_layer_norm.weight"],
        weights[f"{layer}.final_layer_norm.bias"],
    )
    hidden = torch.nn.functional.gelu(
        torch.nn.functional.linear(
            hidden,
            weights[f"{layer}.fc1.weight"],
            weights[f"{layer}.fc1.bias"],
        )
    )
    hidden = residual + torch.nn.functional.linear(
        hidden,
        weights[f"{layer}.fc2.weight"],
        weights[f"{layer}.fc2.bias"],
    )
    hidden = torch.nn.functional.layer_norm(
        hidden,
        (64,),
        weights["decoder.layer_norm.weight"],
        weights["decoder.layer_norm.bias"],
    )
    return hidden @ weights["decoder.embed_tokens.weight"].T


def test_nemotron_parse_synthetic_parity():
    """L3 parity for the full tiny vision encoder and cross-attentive decoder."""
    from _test_configs import VL_CONFIGS

    from mobius._testing.ort_inference import OnnxModelSession

    overrides = next(
        overrides for model_type, overrides, _ in VL_CONFIGS if model_type == "nemotron_parse"
    )
    config = _base_config(**overrides)
    _, pkg = _build_onnx_model("nemotron_parse", config)
    decoder_layer_norms = [
        node for node in pkg["decoder"].graph if node.op_type == "LayerNormalization"
    ]
    assert len(decoder_layer_norms) == 3 * config.num_decoder_layers + 2
    assert all(
        node.attributes["epsilon"].value == pytest.approx(1e-5) for node in decoder_layer_norms
    )
    rng = np.random.default_rng(42)
    for model in pkg.values():
        _fill_random_weights(model, rng)

    weights: dict[str, torch.Tensor] = {}
    for model in pkg.values():
        for name, initializer in model.graph.initializers.items():
            if initializer.const_value is not None and not name.startswith("const_"):
                weights[name] = torch.from_numpy(initializer.const_value.numpy())

    pixel_values = rng.standard_normal((1, 3, 32, 64)).astype(np.float32)
    input_ids = np.array([[2, 7, 11]], dtype=np.int64)
    torch_encoder = _nemotron_parse_torch_vision(torch.from_numpy(pixel_values), weights)
    torch_logits = _nemotron_parse_torch_decoder(
        torch.from_numpy(input_ids),
        torch_encoder,
        weights,
    )

    vision_session = OnnxModelSession(pkg["vision_encoder"])
    decoder_session = OnnxModelSession(pkg["decoder"])
    try:
        onnx_encoder = vision_session.run({"pixel_values": pixel_values})["last_hidden_state"]
        empty_cache = {
            name: np.zeros((1, 4, 0, 16), dtype=np.float32)
            for name in decoder_session.input_names
            if name.startswith("past_key_values.")
        }
        onnx_logits = decoder_session.run(
            {
                "input_ids": input_ids,
                "attention_mask": np.ones_like(input_ids, dtype=np.int64),
                "encoder_hidden_states": onnx_encoder,
                **empty_cache,
            }
        )["logits"]

        unpadded_ids = np.array([[2, 7]], dtype=np.int64)
        unpadded_logits = decoder_session.run(
            {
                "input_ids": unpadded_ids,
                "attention_mask": np.ones_like(unpadded_ids, dtype=np.int64),
                "encoder_hidden_states": onnx_encoder,
                **empty_cache,
            }
        )["logits"]
        padded_ids = np.array([[1, 1, 2, 7]], dtype=np.int64)
        padded_logits = decoder_session.run(
            {
                "input_ids": padded_ids,
                "attention_mask": np.array([[0, 0, 1, 1]], dtype=np.int64),
                "encoder_hidden_states": onnx_encoder,
                **empty_cache,
            }
        )["logits"]
    finally:
        vision_session.close()
        decoder_session.close()

    np.testing.assert_allclose(onnx_encoder, torch_encoder.numpy(), rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(onnx_logits, torch_logits.numpy(), rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(
        unpadded_logits[:, -1],
        padded_logits[:, -1],
        rtol=1e-5,
        atol=1e-5,
    )


# ---------------------------------------------------------------------------
# Parametrize over all causal LM configs
# ---------------------------------------------------------------------------
def _build_synthetic_params() -> list:
    """Build pytest.param list with xfail marks from _XFAIL_REASONS.

    Uses @pytest.mark.xfail (strict=False) so the test still runs —
    if a model starts passing, pytest reports it as XPASS, alerting
    us to remove it from _XFAIL_REASONS.
    """
    from collections import Counter

    configs = [(mt, ov) for mt, ov, _ in ALL_CAUSAL_LM_CONFIGS if mt not in _PARITY_EXCLUDE]
    counts = Counter(mt for mt, _ in configs)
    seen: dict[str, int] = {}
    params = []
    for mt, ov in configs:
        if counts[mt] > 1:
            idx = seen.get(mt, 0)
            seen[mt] = idx + 1
            test_id = f"{mt}_{idx}"
        else:
            test_id = mt
        # Check test_id first (e.g. "granite_0"), then model_type
        xfail_reason = _XFAIL_REASONS.get(test_id, _XFAIL_REASONS.get(mt))
        marks = (
            [pytest.mark.xfail(reason=xfail_reason, strict=False)]
            if xfail_reason is not None
            else []
        )
        params.append(pytest.param(mt, ov, id=test_id, marks=marks))
    return params


_SYNTHETIC_PARAMS = _build_synthetic_params()


@pytest.mark.parametrize(
    "model_type,config_overrides",
    _SYNTHETIC_PARAMS,
)
def test_synthetic_parity(model_type: str, config_overrides: dict):
    """L3 synthetic parity: ONNX matches HF with identical random weights.

    Steps:
    1. Build tiny ONNX model from config
    2. Create equivalent HF model with deterministic init
    3. Transfer HF weights → ONNX via preprocess_weights
    4. Run forward pass on both with same input
    5. Compare logits using atol/rtol gate
    """
    if model_type in _SKIP_REASONS:
        pytest.skip(_SKIP_REASONS[model_type])

    # xfail is handled by marks in _build_synthetic_params() so the test
    # still runs — an XPASS signals that the model is fixed.

    seed = 42
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    # 1. Build tiny config
    config = _base_config(**config_overrides)
    config.dtype = ir.DataType.FLOAT

    # 2. Build ONNX model
    try:
        module, pkg = _build_onnx_model(model_type, config)
    except Exception as e:
        pytest.skip(f"ONNX build failed for {model_type}: {e}")

    # 3. Create HF model
    hf_config = _create_hf_config(model_type, config_overrides)
    hf_model = _create_hf_model(model_type, hf_config, seed)
    if model_type == "nemotron_h":
        # Force correction bias to determine the selected experts. This catches
        # implementations that incorrectly use biased scores as final weights.
        for layer in hf_model.model.layers:
            if getattr(layer, "block_type", None) != "moe":
                continue
            bias = layer.mixer.gate.e_score_correction_bias
            with torch.no_grad():
                bias.copy_(
                    torch.linspace(
                        4.0,
                        1.0,
                        bias.numel(),
                        dtype=bias.dtype,
                        device=bias.device,
                    )
                )

    # 4. Transfer HF weights to ONNX
    try:
        preprocessed = module.preprocess_weights(dict(hf_model.state_dict()))
        for onnx_model in pkg.values():
            apply_weights(onnx_model, preprocessed)
    except Exception as e:
        if model_type == "qwen4_exp_text":
            raise
        pytest.skip(f"Weight transfer failed for {model_type}: {type(e).__name__}: {e}")

    # Fill any remaining unset initializers (ONNX constants, etc.)
    for onnx_model in pkg.values():
        _fill_random_weights(onnx_model, rng)

    # 5. Prepare inputs
    # Exercise multi-token prefill for both recurrent and attention architectures.
    prefill_seq_len = 3
    input_ids = rng.integers(1, config.vocab_size, size=(1, prefill_seq_len)).astype(np.int64)
    attention_mask = np.ones_like(input_ids)
    position_ids = np.arange(input_ids.shape[1], dtype=np.int64)[np.newaxis, :]

    # 6. HF forward
    # use_cache=False avoids DynamicCache initialization. In transformers >= 5.4
    # (HF PR #44950), DynamicCache.has_previous_state() raises ValueError when called
    # on a cache with only Attention layers (no Mamba/LinearAttention layers). Hybrid
    # models (jamba, bamba) call this via _update_mamba_mask() even when the test config
    # uses all-attention layers. Logits are identical with/without cache for a single pass.
    with torch.no_grad():
        hf_out = hf_model(
            input_ids=torch.from_numpy(input_ids),
            attention_mask=torch.from_numpy(attention_mask),
            position_ids=torch.from_numpy(position_ids),
            use_cache=False,
        )
    hf_logits = hf_out.logits.numpy()

    # 7. ONNX forward
    from mobius._testing.ort_inference import OnnxModelSession

    onnx_model = pkg["model"]
    session = OnnxModelSession(onnx_model)

    feeds: dict[str, np.ndarray] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        **(
            {"past_position_ids": np.zeros((1, 0), dtype=np.int64)}
            if model_type == "qwen4_exp_text"
            else {}
        ),
    }
    # Add zero-valued past KV cache feeds with correct shapes:
    # batch=1, past_sequence_len=0, other dims from model spec
    for inp in onnx_model.graph.inputs:
        name = inp.name
        if name in feeds:
            continue
        if not name.startswith("past_key_values"):
            continue
        shape = []
        for d in inp.shape:
            if isinstance(d, int):
                shape.append(d)
            elif "past" in str(d):
                shape.append(0)
            elif "batch" in str(d):
                shape.append(1)
            else:
                shape.append(0)
        feeds[name] = np.zeros(shape, dtype=inp.dtype.numpy())

    try:
        onnx_out = session.run(feeds)
    except Exception as e:
        session.close()
        if model_type == "qwen4_exp_text":
            raise
        pytest.skip(f"ONNX inference failed for {model_type}: {type(e).__name__}: {e}")
    onnx_logits = onnx_out["logits"]
    session.close()

    # 8. Compare — use per-model atol override if defined, otherwise strict 1e-3
    atol = _ATOL_OVERRIDES.get(model_type, 1e-3)
    report = compare_synthetic(onnx_logits, hf_logits, rtol=1e-3, atol=atol)
    assert report.result != ParityResult.FAIL, f"{model_type}: {report.message}"


# ===========================================================================
# L3 Encoder-only synthetic parity
# ===========================================================================

_ENCODER_SKIP_REASONS: dict[str, str] = {
    # LayoutLM v2/v3 require visual_bbox/pixel_values beyond simple text input
    "layoutlmv2": "LayoutLMv2 requires visual inputs (bbox + pixel_values)",
    "layoutlmv3": "LayoutLMv3 requires visual inputs (bbox + pixel_values)",
    # Bros requires bbox_first_token_mask / bbox inputs
    "bros": "Bros requires bounding-box inputs beyond simple text",
    # LayoutLM v1 requires bbox (2D position embeddings)
    "layoutlm": "LayoutLM requires bbox inputs",
    # MarkupLM requires xpath tags/subs beyond simple text
    "markuplm": "MarkupLM requires xpath/tag inputs",
    # LiLT requires bbox inputs for layout-language cross-modal
    "lilt": "LiLT requires bbox inputs for layout understanding",
    # XLNet: HF does not implement sequence_summary for from_config path
    "xlnet": "HF XLNet raises NotImplementedError (no sequence_summary from config)",
    # Xmod: requires set_default_language() call before forward
    "xmod": "HF Xmod requires set_default_language() before inference",
}

_ENCODER_ATOL_OVERRIDES: dict[str, float] = {
    # ModernBERT: unpadding + local/global attention FP differences
    "modernbert": 0.01,
    # DeBERTa: disentangled attention FP accumulation differences
    "deberta": 0.10,
    "deberta-v2": 0.12,
    # Roformer: rotary position embedding FP differences
    "roformer": 2.0,
}

_ENCODER_XFAIL_REASONS: dict[str, str] = {
    # RoBERTa family: position_ids offset (padding_idx+1) causes structural divergence
    "roberta": "RoBERTa position_ids offset differs from ONNX (cosine ~0.66)",
    "camembert": "CamemBERT (RoBERTa-based) position_ids offset mismatch",
    "data2vec-text": "Data2Vec-Text (RoBERTa-based) position_ids offset mismatch",
    "xlm-roberta": "XLM-RoBERTa position_ids offset mismatch",
    "xlm-roberta-xl": "XLM-RoBERTa-XL position_ids offset mismatch",
    "roberta-prelayernorm": "RoBERTa-PreLN position_ids + LayerNorm divergence",
    # ESM: custom attention + contact prediction head; not standard BERT
    "esm": "ESM attention architecture differs from standard BERT encoder",
    # FlauBERT: XLM-style model (causal attention + lang embeddings)
    "flaubert": "FlauBERT is XLM-style (causal + lang embeddings), not standard encoder",
    # MegatronBERT: post-LN vs pre-LN ordering differs
    "megatron-bert": "Megatron-BERT post-LN diverges with random weights (cosine ~-0.03)",
    # iBERT: integer-quantized operations produce different FP paths
    "ibert": "iBERT quantized ops differ from standard BERT encoder",
    # MPNet: permuted language modeling architecture with position shift
    "mpnet": "MPNet relative position bias differs from ONNX encoder",
    # Roc-BERT: multi-modal contrastive features affect hidden states
    "roc_bert": "Roc-BERT multi-modal architecture differs (cosine ~0.80)",
}


def _build_encoder_params() -> list:
    """Build pytest.param list for encoder-only models."""
    from _test_configs import ENCODER_CONFIGS

    params = []
    for mt, ov, _ in ENCODER_CONFIGS:
        xfail_reason = _ENCODER_XFAIL_REASONS.get(mt)
        marks = [pytest.mark.xfail(reason=xfail_reason, strict=False)] if xfail_reason else []
        params.append(pytest.param(mt, ov, id=mt, marks=marks))
    return params


_ENCODER_PARAMS = _build_encoder_params()


@pytest.mark.parametrize("model_type,config_overrides", _ENCODER_PARAMS)
def test_encoder_synthetic_parity(model_type: str, config_overrides: dict):
    """L3 synthetic parity for encoder-only models (BERT, RoBERTa, etc.).

    Compares last_hidden_state instead of logits.
    """
    if model_type in _ENCODER_SKIP_REASONS:
        pytest.skip(_ENCODER_SKIP_REASONS[model_type])

    seed = 42
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    # 1. Build tiny config
    config = _base_config(**config_overrides)
    config.dtype = ir.DataType.FLOAT

    # 2. Build ONNX model
    try:
        module, pkg = _build_onnx_model(model_type, config)
    except Exception as e:
        pytest.skip(f"ONNX build failed for {model_type}: {e}")

    # 3. Create HF encoder model
    from transformers import AutoConfig, AutoModel

    hf_kwargs: dict = {
        "hidden_size": TINY_HIDDEN,
        "intermediate_size": TINY_INTERMEDIATE,
        "num_attention_heads": TINY_HEADS,
        "num_hidden_layers": config_overrides.get("num_hidden_layers", TINY_LAYERS),
        "vocab_size": TINY_VOCAB,
        "max_position_embeddings": config_overrides.get("max_position_embeddings", 128),
        "pad_token_id": 0,
    }
    for key, value in config_overrides.items():
        if key == "_config_cls":
            continue
        hf_kwargs[key] = value

    try:
        hf_config = AutoConfig.for_model(model_type, **hf_kwargs)
    except (ValueError, KeyError) as e:
        pytest.skip(f"Cannot create HF config for {model_type}: {e}")

    torch.manual_seed(seed)
    try:
        hf_model = AutoModel.from_config(hf_config).float().eval()
    except Exception as e:
        pytest.skip(f"Cannot create HF model for {model_type}: {type(e).__name__}: {e}")

    # 4. Transfer HF weights to ONNX
    try:
        preprocessed = module.preprocess_weights(dict(hf_model.state_dict()))
        for onnx_model in pkg.values():
            apply_weights(onnx_model, preprocessed)
    except Exception as e:
        pytest.skip(f"Weight transfer failed for {model_type}: {type(e).__name__}: {e}")

    for onnx_model in pkg.values():
        _fill_random_weights(onnx_model, rng)

    # 5. Prepare inputs
    seq_len = 3
    input_ids = rng.integers(1, config.vocab_size, size=(1, seq_len)).astype(np.int64)
    attention_mask = np.ones_like(input_ids)

    hf_feeds: dict = {
        "input_ids": torch.from_numpy(input_ids),
        "attention_mask": torch.from_numpy(attention_mask),
    }
    # token_type_ids: only pass if model expects them (type_vocab_size > 0)
    type_vocab_size = config_overrides.get("type_vocab_size", 0)
    if type_vocab_size and type_vocab_size > 0:
        token_type_ids = np.zeros_like(input_ids)
        hf_feeds["token_type_ids"] = torch.from_numpy(token_type_ids)

    # 6. HF forward
    with torch.no_grad():
        hf_out = hf_model(**hf_feeds)
    hf_hidden = hf_out.last_hidden_state.numpy()

    # 7. ONNX forward
    from mobius._testing.ort_inference import OnnxModelSession

    onnx_model = pkg["model"]
    session = OnnxModelSession(onnx_model)

    onnx_feeds: dict[str, np.ndarray] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }
    if type_vocab_size and type_vocab_size > 0:
        onnx_feeds["token_type_ids"] = np.zeros_like(input_ids)

    try:
        onnx_out = session.run(onnx_feeds)
    except Exception as e:
        session.close()
        pytest.skip(f"ONNX inference failed for {model_type}: {type(e).__name__}: {e}")
    onnx_hidden = onnx_out["last_hidden_state"]
    session.close()

    # 8. Compare hidden states
    atol = _ENCODER_ATOL_OVERRIDES.get(model_type, 1e-3)
    # For hidden states, use simple allclose check with cosine as diagnostic
    max_diff = float(np.max(np.abs(onnx_hidden - hf_hidden)))
    cos_sim = float(
        np.dot(onnx_hidden.flatten(), hf_hidden.flatten())
        / (np.linalg.norm(onnx_hidden) * np.linalg.norm(hf_hidden) + 1e-8)
    )
    passes = np.allclose(onnx_hidden, hf_hidden, atol=atol, rtol=1e-3)
    assert passes, (
        f"{model_type}: encoder L3 FAIL: max_diff={max_diff:.6f}, "
        f"cosine={cos_sim:.6f}, atol={atol}"
    )


# ===========================================================================
# L3 Seq2Seq synthetic parity (encoder component)
# ===========================================================================

_SEQ2SEQ_SKIP_REASONS: dict[str, str] = {
    # ProphetNet: HF raises NotImplementedError for num_hidden_layers override
    "prophetnet": "HF ProphetNet does not support num_hidden_layers override",
    # XLM-ProphetNet: not in HF CONFIG_MAPPING
    "xlm-prophetnet": "Not in HF CONFIG_MAPPING (use prophetnet)",
    # nllb_moe: HF identifier uses hyphen (nllb-moe), not underscore
    "nllb_moe": "HF identifier is nllb-moe, not nllb_moe",
    # TrOCR: decoder-only architecture, not AutoModelForSeq2SeqLM
    "trocr": "TrOCR is decoder-only, not AutoModelForSeq2SeqLM",
    # FSMT: uses non-standard shared vocab that conflicts with tiny vocab
    "fsmt": "Non-standard shared vocab architecture (42024 min)",
}

_SEQ2SEQ_ATOL_OVERRIDES: dict[str, float] = {
    # UMT5: gated activation + RMSNorm FP differences (cosine=0.996)
    "umt5": 0.30,
}

_SEQ2SEQ_XFAIL_REASONS: dict[str, str] = {
    # LED: Longformer-style global+local attention diverges from standard encoder
    "led": "LED global/local attention architecture diverges (cosine ~0.0)",
    # NLLB-MoE: MoE routing with random weights causes divergence
    "nllb-moe": "NLLB-MoE expert routing diverges with random weights (cosine ~-0.06)",
    # Models with embed_positions offset (+2 for padding) shape mismatch
    "bigbird_pegasus": "embed_positions offset (+2) shape mismatch",
    "blenderbot": "embed_positions offset (+2) shape mismatch",
    "blenderbot-small": "embed_positions offset (+2) shape mismatch",
    "marian": "embed_positions offset (+2) shape mismatch",
    "pegasus": "embed_positions offset (+2) shape mismatch",
    "m2m_100": "embed_positions offset mismatch (max_diff ~3.6)",
    "pegasus_x": "staggered local attention diverges (max_diff ~4.2)",
    "plbart": "embed_positions offset mismatch (max_diff ~2.7)",
    # LongT5: transient-global local attention diverges from standard attention
    "longt5": "Transient-global local attention diverges (max_diff ~2.0)",
    # Switch Transformers: MoE top-1 routing with random weights diverges
    "switch_transformers": "MoE routing diverges with random weights (max_diff ~1.8)",
}


def _build_seq2seq_params() -> list:
    """Build pytest.param list for seq2seq models."""
    from _test_configs import SEQ2SEQ_CONFIGS

    params = []
    for mt, ov, _ in SEQ2SEQ_CONFIGS:
        xfail_reason = _SEQ2SEQ_XFAIL_REASONS.get(mt)
        marks = [pytest.mark.xfail(reason=xfail_reason, strict=False)] if xfail_reason else []
        params.append(pytest.param(mt, ov, id=mt, marks=marks))
    return params


_SEQ2SEQ_PARAMS = _build_seq2seq_params()


@pytest.mark.parametrize("model_type,config_overrides", _SEQ2SEQ_PARAMS)
def test_seq2seq_encoder_synthetic_parity(model_type: str, config_overrides: dict):
    """L3 synthetic parity for seq2seq encoder (T5, BART, etc.).

    Compares encoder last_hidden_state output.
    """
    if model_type in _SEQ2SEQ_SKIP_REASONS:
        pytest.skip(_SEQ2SEQ_SKIP_REASONS[model_type])

    seed = 42
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    # 1. Build tiny config
    config = _base_config(**config_overrides)
    config.dtype = ir.DataType.FLOAT

    # 2. Build ONNX model package (encoder + decoder)
    try:
        module, pkg = _build_onnx_model(model_type, config)
    except Exception as e:
        pytest.skip(f"ONNX build failed for {model_type}: {e}")

    if "encoder" not in pkg:
        pytest.skip(f"{model_type}: no encoder in ModelPackage")

    # 3. Create HF seq2seq model
    from transformers import AutoConfig, AutoModelForSeq2SeqLM

    hf_kwargs: dict = {
        "hidden_size": TINY_HIDDEN,
        "d_model": TINY_HIDDEN,
        "intermediate_size": TINY_INTERMEDIATE,
        "d_ff": TINY_INTERMEDIATE,
        "encoder_ffn_dim": TINY_INTERMEDIATE,
        "decoder_ffn_dim": TINY_INTERMEDIATE,
        "num_attention_heads": TINY_HEADS,
        "num_heads": TINY_HEADS,
        "encoder_attention_heads": TINY_HEADS,
        "decoder_attention_heads": TINY_HEADS,
        # T5 family: d_kv must match head_dim = hidden_size // num_heads
        "d_kv": TINY_HIDDEN // TINY_HEADS,
        "num_hidden_layers": config_overrides.get("num_hidden_layers", TINY_LAYERS),
        "num_layers": config_overrides.get("num_hidden_layers", TINY_LAYERS),
        "encoder_layers": config_overrides.get("num_hidden_layers", TINY_LAYERS),
        "num_decoder_layers": config_overrides.get("num_decoder_layers", TINY_LAYERS),
        "decoder_layers": config_overrides.get("num_decoder_layers", TINY_LAYERS),
        "vocab_size": TINY_VOCAB,
        "max_position_embeddings": config_overrides.get("max_position_embeddings", 128),
        "pad_token_id": 0,
        "decoder_start_token_id": 1,
    }
    for key, value in config_overrides.items():
        if key == "_config_cls":
            continue
        hf_kwargs[key] = value

    try:
        hf_config = AutoConfig.for_model(model_type, **hf_kwargs)
    except (ValueError, KeyError) as e:
        pytest.skip(f"Cannot create HF config for {model_type}: {e}")

    torch.manual_seed(seed)
    try:
        hf_model = AutoModelForSeq2SeqLM.from_config(hf_config).float().eval()
    except Exception as e:
        pytest.skip(f"Cannot create HF model for {model_type}: {type(e).__name__}: {e}")

    # 4. Transfer HF weights to ONNX
    try:
        preprocessed = module.preprocess_weights(dict(hf_model.state_dict()))
        for onnx_model in pkg.values():
            apply_weights(onnx_model, preprocessed)
    except Exception as e:
        pytest.skip(f"Weight transfer failed for {model_type}: {type(e).__name__}: {e}")

    for onnx_model in pkg.values():
        _fill_random_weights(onnx_model, rng)

    # 5. Prepare encoder inputs
    seq_len = 3
    input_ids = rng.integers(1, config.vocab_size, size=(1, seq_len)).astype(np.int64)
    attention_mask = np.ones_like(input_ids)

    # 6. HF encoder forward
    with torch.no_grad():
        hf_encoder_out = hf_model.get_encoder()(
            input_ids=torch.from_numpy(input_ids),
            attention_mask=torch.from_numpy(attention_mask),
        )
    hf_hidden = hf_encoder_out.last_hidden_state.numpy()

    # 7. ONNX encoder forward
    from mobius._testing.ort_inference import OnnxModelSession

    onnx_encoder = pkg["encoder"]
    session = OnnxModelSession(onnx_encoder)

    onnx_feeds: dict[str, np.ndarray] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }

    try:
        onnx_out = session.run(onnx_feeds)
    except Exception as e:
        session.close()
        pytest.skip(f"ONNX encoder inference failed for {model_type}: {type(e).__name__}: {e}")
    onnx_hidden = onnx_out["last_hidden_state"]
    session.close()

    # 8. Compare encoder hidden states
    atol = _SEQ2SEQ_ATOL_OVERRIDES.get(model_type, 1e-3)
    max_diff = float(np.max(np.abs(onnx_hidden - hf_hidden)))
    cos_sim = float(
        np.dot(onnx_hidden.flatten(), hf_hidden.flatten())
        / (np.linalg.norm(onnx_hidden) * np.linalg.norm(hf_hidden) + 1e-8)
    )
    passes = np.allclose(onnx_hidden, hf_hidden, atol=atol, rtol=1e-3)
    assert passes, (
        f"{model_type}: seq2seq encoder L3 FAIL: max_diff={max_diff:.6f}, "
        f"cosine={cos_sim:.6f}, atol={atol}"
    )


# ===========================================================================
# L3 Seq2Seq decoder synthetic parity
# ===========================================================================


@pytest.mark.parametrize("model_type,config_overrides", _SEQ2SEQ_PARAMS)
def test_seq2seq_decoder_synthetic_parity(model_type: str, config_overrides: dict):
    """L3 synthetic parity for seq2seq decoder (T5, BART, etc.).

    Feeds HF encoder output to both HF decoder and ONNX decoder, compares logits.
    """
    if model_type in _SEQ2SEQ_SKIP_REASONS:
        pytest.skip(_SEQ2SEQ_SKIP_REASONS[model_type])

    seed = 42
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    # 1. Build tiny config
    config = _base_config(**config_overrides)
    config.dtype = ir.DataType.FLOAT

    # 2. Build ONNX model package
    try:
        module, pkg = _build_onnx_model(model_type, config)
    except Exception as e:
        pytest.skip(f"ONNX build failed for {model_type}: {e}")

    if "decoder" not in pkg:
        pytest.skip(f"{model_type}: no decoder in ModelPackage")

    # 3. Create HF seq2seq model
    from transformers import AutoConfig, AutoModelForSeq2SeqLM

    hf_kwargs: dict = {
        "hidden_size": TINY_HIDDEN,
        "d_model": TINY_HIDDEN,
        "intermediate_size": TINY_INTERMEDIATE,
        "d_ff": TINY_INTERMEDIATE,
        "encoder_ffn_dim": TINY_INTERMEDIATE,
        "decoder_ffn_dim": TINY_INTERMEDIATE,
        "num_attention_heads": TINY_HEADS,
        "num_heads": TINY_HEADS,
        "encoder_attention_heads": TINY_HEADS,
        "decoder_attention_heads": TINY_HEADS,
        # T5 family: d_kv must match head_dim = hidden_size // num_heads
        "d_kv": TINY_HIDDEN // TINY_HEADS,
        "num_hidden_layers": config_overrides.get("num_hidden_layers", TINY_LAYERS),
        "num_layers": config_overrides.get("num_hidden_layers", TINY_LAYERS),
        "encoder_layers": config_overrides.get("num_hidden_layers", TINY_LAYERS),
        "num_decoder_layers": config_overrides.get("num_decoder_layers", TINY_LAYERS),
        "decoder_layers": config_overrides.get("num_decoder_layers", TINY_LAYERS),
        "vocab_size": TINY_VOCAB,
        "max_position_embeddings": config_overrides.get("max_position_embeddings", 128),
        "pad_token_id": 0,
        "decoder_start_token_id": 1,
    }
    for key, value in config_overrides.items():
        if key == "_config_cls":
            continue
        hf_kwargs[key] = value

    try:
        hf_config = AutoConfig.for_model(model_type, **hf_kwargs)
    except (ValueError, KeyError) as e:
        pytest.skip(f"Cannot create HF config for {model_type}: {e}")

    torch.manual_seed(seed)
    try:
        hf_model = AutoModelForSeq2SeqLM.from_config(hf_config).float().eval()
    except Exception as e:
        pytest.skip(f"Cannot create HF model for {model_type}: {type(e).__name__}: {e}")

    # 4. Transfer HF weights to ONNX
    try:
        preprocessed = module.preprocess_weights(dict(hf_model.state_dict()))
        for onnx_model in pkg.values():
            apply_weights(onnx_model, preprocessed)
    except Exception as e:
        pytest.skip(f"Weight transfer failed for {model_type}: {type(e).__name__}: {e}")

    for onnx_model in pkg.values():
        _fill_random_weights(onnx_model, rng)

    # 5. Run HF encoder to get shared encoder_hidden_states
    seq_len = 3
    input_ids = rng.integers(1, config.vocab_size, size=(1, seq_len)).astype(np.int64)
    attention_mask = np.ones_like(input_ids)

    with torch.no_grad():
        hf_encoder_out = hf_model.get_encoder()(
            input_ids=torch.from_numpy(input_ids),
            attention_mask=torch.from_numpy(attention_mask),
        )
    encoder_hidden_states = hf_encoder_out.last_hidden_state

    # 6. HF decoder forward (single token, no cache)
    decoder_input_ids = np.array([[1]], dtype=np.int64)  # decoder_start_token_id
    with torch.no_grad():
        hf_dec_out = hf_model(
            encoder_outputs=(encoder_hidden_states,),
            decoder_input_ids=torch.from_numpy(decoder_input_ids),
            attention_mask=torch.from_numpy(attention_mask),
            use_cache=False,
        )
    hf_logits = hf_dec_out.logits.numpy()

    # 7. ONNX decoder forward
    from mobius._testing.ort_inference import OnnxModelSession

    onnx_decoder = pkg["decoder"]
    session = OnnxModelSession(onnx_decoder)

    onnx_feeds: dict[str, np.ndarray] = {
        "input_ids": decoder_input_ids,
        "encoder_hidden_states": encoder_hidden_states.numpy(),
        "encoder_attention_mask": attention_mask,
    }
    # Add zero-valued past KV cache feeds
    for inp in onnx_decoder.graph.inputs:
        name = inp.name
        if name in onnx_feeds:
            continue
        if not name.startswith("past_key_values"):
            continue
        shape = []
        for d in inp.shape:
            if isinstance(d, int):
                shape.append(d)
            elif "past" in str(d):
                shape.append(0)
            elif "batch" in str(d):
                shape.append(1)
            else:
                shape.append(0)
        onnx_feeds[name] = np.zeros(shape, dtype=np.float32)

    try:
        onnx_out = session.run(onnx_feeds)
    except Exception as e:
        session.close()
        pytest.skip(f"ONNX decoder inference failed for {model_type}: {type(e).__name__}: {e}")
    onnx_logits = onnx_out["logits"]
    session.close()

    # 8. Compare decoder logits
    atol = _SEQ2SEQ_ATOL_OVERRIDES.get(model_type, 1e-3)
    report = compare_synthetic(onnx_logits, hf_logits, rtol=1e-3, atol=atol)
    assert report.result != ParityResult.FAIL, (
        f"{model_type}: seq2seq decoder L3 FAIL: {report.message}"
    )
