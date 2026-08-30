# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Shared helpers and stable coverage inventory for L1 graph-construction tests."""

from __future__ import annotations

import numpy as np
import onnx_ir as ir
import pytest
from onnx_ir.passes.common import CheckerPass

from mobius._optimizations import SymbolicShapeInferencePass

_onnx_checker = CheckerPass()
_shape_inference = SymbolicShapeInferencePass()

_CHECKER_SKIP_MODELS: set[str] = {
    "minimax",
    "qwen3_5_text",
    "qwen3_5_moe",
    "qwen3_5_moe_text",
    "qwen3_next",
    "qwen4_exp",
    "qwen4_exp_text",
    "Qwen4ExpForConditionalGeneration",
    "bamba",
    "granitemoehybrid",
    "mamba2",
    "nemotron_h",
    "zamba2",
    "qwen2_vl",
    "qwen2_5_vl",
    "qwen3_vl",
    "qwen3_vl_single",
    "qwen3_5_vl",
    "qwen3_5_moe_vl",
    "qwen3_tts_tokenizer_12hz",
}


def _fill_dummy_weights(model: ir.Model) -> None:
    """Fill initializers that have no const_value with zero tensors."""
    for initializer in model.graph.initializers.values():
        if initializer.const_value is not None:
            continue
        shape = initializer.shape
        dims = [d if isinstance(d, int) else 1 for d in shape] if shape else [1]
        dtype = initializer.dtype or ir.DataType.FLOAT
        initializer.const_value = ir.Tensor(np.zeros(dims, dtype=dtype.numpy()))


def _run_onnx_checker(pkg: dict[str, ir.Model], model_type: str) -> None:
    """Run ONNX CheckerPass on all models in a package."""
    if model_type in _CHECKER_SKIP_MODELS:
        pytest.skip(
            f"ONNX checker skipped for {model_type}: "
            "upstream onnx-ir value_info missing type field for custom ops"
        )
    for model in pkg.values():
        _shape_inference(model)
        _fill_dummy_weights(model)
        _onnx_checker(model)


def _assert_outputs_have_shapes_and_dtypes(
    pkg: dict[str, ir.Model],
    model_type: str,
) -> None:
    """Assert every graph output has a non-None shape and dtype."""
    if model_type in _CHECKER_SKIP_MODELS:
        pytest.skip(
            f"Shape assertion skipped for {model_type}: "
            "custom ops prevent full shape propagation"
        )
    for sub_name, model in pkg.items():
        _shape_inference(model)
        for output in model.graph.outputs:
            assert output.shape is not None, (
                f"{model_type}/{sub_name}: output '{output.name}' "
                f"has no shape after shape inference"
            )
            assert output.type is not None, (
                f"{model_type}/{sub_name}: output '{output.name}' "
                f"has no dtype after shape inference"
            )


_SEMANTIC_IDS: dict[tuple[str, int], str] = {
    ("deepseek_v2", 0): "deepseek_v2_mla",
    ("deepseek_v2", 1): "deepseek_v2_no_mla",
    ("deepseek_v2", 2): "deepseek_v2_mla_dense",
    ("qwen3_5_text", 0): "qwen3_5_text_default",
    ("qwen3_5_text", 1): "qwen3_5_text_linear_attn",
    ("qwen3_next", 0): "qwen3_next_hybrid",
    ("qwen3_next", 1): "qwen3_next_all_full_attn",
    ("qwen3_next", 2): "qwen3_next_all_linear_attn",
    ("jamba", 0): "jamba_hybrid_moe",
    ("jamba", 1): "jamba_all_attention",
    ("bamba", 0): "bamba_hybrid",
    ("bamba", 1): "bamba_all_attention",
    ("gemma3n_text", 0): "gemma3n_text_sliding",
    ("gemma3n_text", 1): "gemma3n_text_full_attn",
    ("granite", 0): "granite_default",
    ("granite", 1): "granite_scaling",
    ("phi3small", 0): "phi3small_default",
    ("phi3small", 1): "phi3small_rotary_025",
}


def _make_params(configs: list[tuple[str, dict, bool]]) -> list:
    """Create pytest.param entries with stable unique IDs."""
    from collections import Counter

    stripped = [(mt, ov) for mt, ov, _ in configs]
    counts = Counter(mt for mt, _ in stripped)
    seen: dict[str, int] = {}
    params = []
    for model_type, overrides in stripped:
        if counts[model_type] > 1:
            idx = seen.get(model_type, 0)
            seen[model_type] = idx + 1
            test_id = _SEMANTIC_IDS.get((model_type, idx), f"{model_type}_{idx}")
        else:
            test_id = model_type
        params.append(pytest.param(model_type, overrides, id=test_id))
    return params


_SPECIALIZED_TEST_MODEL_TYPES: set[str] = {
    # Internal GGUF-only graph covered by _exact_legacy_decoder_test.py.
    "gguf_legacy",
    # Mistral4 owns a K-only latent cache, covered by _remaining_dense_test.py.
    "mistral4_gguf",
    # T5 encoder-only hidden-state contract (co-located models/t5_test.py).
    "t5encoder",
    # LLaDA masked-diffusion LM (co-located src/mobius/models/llada_test.py):
    # bidirectional Llama backbone with a masked-diffusion task, so it has no
    # attention_mask / KV cache and does not fit the generic causal-LM harness.
    "llada",
    # VLM alias tests (test_llava_aliases_build)
    "aya_vision",
    "chameleon",
    "cohere2_vision",
    "deepseek_vl",
    "deepseek_vl_hybrid",
    # FastConformer-RNNT (co-located src/mobius/models/nemo_rnnt_test.py)
    "fastconformer_rnnt",
    "florence2",
    "fuyu",
    "glm4v",
    "glm4v_moe",
    "got_ocr2",
    "idefics2",
    "idefics3",
    "instructblip",
    "instructblipvideo",
    "internvl",
    "internvl_chat",
    "internvl2",
    "janus",
    "llava_next",
    "llava_next_video",
    "llava_onevision",
    "mistral3",
    "molmo",
    # SenseNova-U1.5 NEO-unify (co-located src/mobius/models/sensenova_u1_test.py):
    # a unified any-to-any package whose five components include an
    # image-generation branch, so it does not fit the generic VLM harness.
    "neo_chat",
    "ovis2",
    "paligemma",
    "pixtral",
    "smolvlm",
    "video_llava",
    "vipllava",
    # VLM dedicated tests
    "blip-2",
    "deepseek_vl_v2",
    "gemma3",
    "gemma4",
    "gemma4_unified",
    "gemma4_unified_text",
    "llava",
    "mllama",
    "phi3_v",
    "phi4-siglip",
    "phi4_multimodal",
    "phi4mm",
    "qwen2_5_vl",
    "qwen2_5_vl_text",
    "qwen2_vl",
    "qwen2_vl_text",
    "qwen3_5",
    "qwen3_5_vl",
    "qwen3_vl",
    "qwen3_vl_single",
    # Audio alias tests (test_audio_aliases_build)
    "data2vec-audio",
    "hubert",
    "mctct",
    "musicgen",
    "seamless_m4t",
    "seamless_m4t_v2",
    "sew",
    "sew-d",
    "sortformer",
    "speecht5",
    "unispeech",
    "unispeech-sat",
    "voxtral_encoder",
    "wav2vec2",
    "wav2vec2-bert",
    "wav2vec2-conformer",
    "wavlm",
    # Audio/TTS dedicated tests
    "fun_asr",
    "mms",
    "qwen3_asr",
    "qwen3_forced_aligner",
    "qwen3_tts",
    "qwen3_tts_tokenizer_12hz",
    "whisper",
    # SSM dedicated tests
    "falcon_mamba",
    "mamba",
    "mamba2",
    # Speech enhancement: "reuse" is driven by SPEECH_CONFIGS; "semamba" is
    # a bare alias of the same class, so it has no config of its own.
    "semamba",
    # Hybrid SSM+Attention dedicated tests
    "bamba",
    "jamba",
    # Speculative-decoding draft models with bespoke IO contracts
    # (DFlash drafter takes noise_embedding + target_hidden instead of
    # input_ids; the generic ALL_CAUSAL_LM_CONFIGS matrix can't drive it).
    # Covered by src/mobius/models/_dflash_test.py.
    "DFlashDraftModel",
    # Gemma4-Assistant: bespoke IO contract (consumes inputs_embeds +
    # the target's shared KV instead of input_ids), so the generic
    # ALL_CAUSAL_LM_CONFIGS matrix can't drive it. Covered by
    # src/mobius/models/_gemma4_assistant_test.py.
    "gemma4_assistant",
    "Gemma4AssistantForCausalLM",
    "gemma4_unified_assistant",
    "Gemma4UnifiedAssistantForCausalLM",
    # Qwen3.6 MTP self-speculative head: bespoke IO contract (consumes
    # inputs_embeds + the target's hidden_states instead of input_ids;
    # borrows the target's embed/lm_head), so the generic
    # ALL_CAUSAL_LM_CONFIGS matrix can't drive it. Covered by
    # src/mobius/models/_qwen35_mtp_test.py and the registered dense-qwen35
    # GGUF capability in src/mobius/integrations/gguf/_mtp_test.py.
    "Qwen35MtpModel",
    # HyV3 MTP head: bespoke IO contract with target hidden states and a
    # one-layer independent cache. Covered by models/hy_v3_test.py and the
    # combined/target-only GGUF tests in integrations/gguf/_hy_v3_test.py.
    "HyV3MtpModel",
    # EAGLE-3 drafter: bespoke IO contract (inputs_embeds, fused_hidden,
    # recycled_hidden and draft-vocab logits). Covered by _eagle3_test.py.
    "Eagle3LlamaForCausalLM",
    "LlamaForCausalLMEagle3",
    "Eagle3Speculator",
    "Eagle3DraftModel",
}


_KNOWN_UNTESTED_MODEL_TYPES: set[str] = {
    "deepseek_v2_moe",  # Alias for deepseek_v2 — tested via deepseek_v2
    "qwen3_5_vl_text",  # VL text decoder — tested via parent VL model
}
