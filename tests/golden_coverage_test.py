# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Golden data coverage test: every registered model must have L4/L5 test data.

Full L4/L5 coverage requires **both**:

1. A **YAML test case** in ``testdata/cases/<task>/<name>.yaml`` (the spec)
2. A **JSON golden output** in ``testdata/golden/<task>/<name>.json`` (reference data)

This test iterates over ALL model types in the registry and verifies that
each one with a ``test_model_id`` has a corresponding YAML file.  For
models that have YAML, it additionally checks that a JSON golden file
exists (unless the YAML has a ``skip_reason`` explaining why golden
generation isn't feasible).

Models that genuinely cannot have golden data at all are listed in
``_GOLDEN_SKIP`` with an explicit reason.

Adding a new model?  Either:
- Add a YAML case + generate the golden JSON, or
- Add a YAML case with ``skip_reason`` (for models too large for CI), or
- Add an entry to ``_GOLDEN_SKIP`` explaining why no YAML is possible.
"""

from __future__ import annotations

import pytest

from mobius._registry import _TEST_MODEL_IDS, registry
from mobius._testing.golden import (
    discover_test_cases,
    golden_path_for_case,
    has_golden,
)

# ── Models that cannot have golden YAML data (with reasons) ──────────────
_GOLDEN_SKIP: dict[str, str] = {
    # --- Internal / duplicate aliases ---
    "code_llama": "Alias for llama — covered by llama YAML",
    "command_r": "Alias for cohere — covered by cohere YAML",
    "gpt_oss": "Internal model — no public HF checkpoint",
    "gptoss": "Internal model — no public HF checkpoint",
    "deepseek_v2_moe": "Internal alias for deepseek_v2",
    "helium": "Alias for mistral — covered by mistral YAML",
    "open-llama": "Alias for llama — covered by llama YAML",
    "seed_oss": "Internal model — no public HF checkpoint",
    "shieldgemma2": "Alias for gemma2 — covered by gemma2 YAML",
    "yi": "Alias for llama — covered by llama YAML",
    # --- VL text-decoder submodels (tested via their parent VL model) ---
    "glm4v_text": "VL text decoder — tested via glm4v VL YAML",
    "glm4v_moe_text": "VL text decoder — tested via glm4v_moe VL YAML",
    "qwen2_5_vl_text": "VL text decoder — tested via qwen2_5_vl VL YAML",
    "qwen2_vl_text": "VL text decoder — tested via qwen2_vl VL YAML",
    "qwen3_5_vl_text": "VL text decoder — tested via qwen3_5_vl VL YAML",
    "qwen3_vl_text": "VL text decoder — tested via qwen3_vl VL YAML",
    "qwen3_vl_moe": "VL MoE text decoder — tested via qwen3_vl VL YAML",
    "qwen3_omni_moe": "VL MoE text decoder — tested via qwen3_omni VL YAML",
    # --- Multimodal models (golden data requires image/video inputs) ---
    "blip": "Multimodal — golden data requires image inputs",
    "blip-2": "Multimodal — golden data requires image inputs",
    "florence2": "Multimodal — golden data requires image inputs",
    "idefics2": "Multimodal — golden data requires image inputs",
    "idefics3": "Multimodal — golden data requires image inputs",
    "instructblip": "Multimodal — golden data requires image inputs",
    "llava_next": "Multimodal — golden data requires image inputs",
    "llava_onevision": "Multimodal — golden data requires image inputs",
    "molmo": "Multimodal — golden data requires image inputs",
    "phi4_multimodal": "Multimodal (14B) — needs GPU for golden",
    "phi4mm": "Multimodal (14B) — needs GPU for golden",
    "qwen2_vl": "VL model — golden data requires image inputs",
    # --- Audio models (golden data requires audio inputs) ---
    "data2vec-audio": "Audio model — golden data requires audio inputs",
    "musicgen": "Audio model — golden data requires audio inputs",
    "seamless_m4t": "Audio model — golden data requires audio inputs",
    "seamless_m4t_v2": "Audio model — golden data requires audio inputs",
    "sew": "Audio model — golden data requires audio inputs",
    "sew-d": "Audio model — golden data requires audio inputs",
    "speecht5": "Audio model — golden data requires audio inputs",
    "unispeech": "Audio model — golden data requires audio inputs",
    "unispeech-sat": "Audio model — golden data requires audio inputs",
    "wav2vec2-bert": "Audio model — golden data requires audio inputs",
    "wav2vec2-conformer": "Audio model — golden data requires audio inputs",
    "wavlm": "Audio model — golden data requires audio inputs",
    "qwen3_asr": "Audio model — golden data requires audio inputs",
    # --- Models requiring trust_remote_code or no public weights ---
    "chatglm": "Requires trust_remote_code (custom HF modeling code)",
    "dots1": "Requires trust_remote_code (custom HF modeling code)",
    # --- Very large models without small public checkpoints ---
    "arctic": "Very large MoE (480B) — no small public checkpoint",
    "dbrx": "Large MoE (132B) — no small public checkpoint",
    "deepseek_v3": "Very large MoE (671B) — no small public checkpoint",
    "llama4_text": "Very large MoE (109B) — no small public checkpoint",
    "switch_transformers": "Large MoE — needs many experts for golden",
    "qwen3_5_moe": "Large MoE (22B) — no small public checkpoint",
    # --- Vision models (YAML not yet created) ---
    "beit": "Vision model — YAML not yet created",
    "cvt": "Vision model — YAML not yet created",
    "data2vec-vision": "Vision model — YAML not yet created",
    "deit": "Vision model — YAML not yet created",
    "depth_anything": "Vision model — YAML not yet created",
    "dinov2": "Vision model — YAML not yet created",
    "dinov2_with_registers": "Vision model — YAML not yet created",
    "hiera": "Vision model — YAML not yet created",
    "imagegpt": "Vision model — YAML not yet created",
    "mobilevit": "Vision model — YAML not yet created",
    "mobilevitv2": "Vision model — YAML not yet created",
    "pvt": "Vision model — YAML not yet created",
    "pvt_v2": "Vision model — YAML not yet created",
    "segformer": "Vision model — YAML not yet created",
    "siglip2_vision_model": "Vision model — YAML not yet created",
    "siglip_vision_model": "Vision model — YAML not yet created",
    "swin": "Vision model — YAML not yet created",
    "swin2sr": "Vision model — YAML not yet created",
    "swinv2": "Vision model — YAML not yet created",
    "vit_mae": "Vision model — YAML not yet created",
    "vit_msn": "Vision model — YAML not yet created",
    "yolos": "Vision model — YAML not yet created",
    # --- Encoder models (YAML not yet created) ---
    "bros": "Encoder model — YAML not yet created",
    "camembert": "Encoder model — YAML not yet created",
    "data2vec-text": "Encoder model — YAML not yet created",
    "deberta-v2": "Encoder model — YAML not yet created",
    "electra": "Encoder model — YAML not yet created",
    "ernie": "Encoder model — YAML not yet created",
    "ernie4_5": "Encoder model — YAML not yet created",
    "ernie_m": "Encoder model — YAML not yet created",
    "esm": "Encoder model — YAML not yet created",
    "flaubert": "Encoder model — YAML not yet created",
    "ibert": "Encoder model — YAML not yet created",
    "layoutlm": "Encoder model — YAML not yet created",
    "layoutlmv2": "Encoder model — YAML not yet created",
    "layoutlmv3": "Encoder model — YAML not yet created",
    "lilt": "Encoder model — YAML not yet created",
    "markuplm": "Encoder model — YAML not yet created",
    "mega": "Encoder model — YAML not yet created",
    "mobilebert": "Encoder model — YAML not yet created",
    "mpnet": "Encoder model — YAML not yet created",
    "mra": "Encoder model — YAML not yet created",
    "nezha": "Encoder model — YAML not yet created",
    "nystromformer": "Encoder model — YAML not yet created",
    "rembert": "Encoder model — YAML not yet created",
    "roberta-prelayernorm": "Encoder model — YAML not yet created",
    "roc_bert": "Encoder model — YAML not yet created",
    "roformer": "Encoder model — YAML not yet created",
    "splinter": "Encoder model — YAML not yet created",
    "squeezebert": "Encoder model — YAML not yet created",
    "xlm-prophetnet": "Encoder/seq2seq — YAML not yet created",
    "xlm-roberta-xl": "Encoder model — YAML not yet created",
    "xlnet": "Encoder model — YAML not yet created",
    "xmod": "Encoder model — YAML not yet created",
    "yoso": "Encoder model — YAML not yet created",
    # --- Seq2seq models (YAML not yet created) ---
    "bigbird_pegasus": "Seq2seq model — YAML not yet created",
    "blenderbot": "Seq2seq model — YAML not yet created",
    "blenderbot-small": "Seq2seq model — YAML not yet created",
    "fsmt": "Seq2seq model — YAML not yet created",
    "led": "Seq2seq model — YAML not yet created",
    "longt5": "Seq2seq model — YAML not yet created",
    "m2m_100": "Seq2seq model — YAML not yet created",
    "mbart": "Seq2seq model — YAML not yet created",
    "mvp": "Seq2seq model — YAML not yet created",
    "pegasus": "Seq2seq model — YAML not yet created",
    "pegasus_x": "Seq2seq model — YAML not yet created",
    "plbart": "Seq2seq model — YAML not yet created",
    "prophetnet": "Seq2seq model — YAML not yet created",
    "trocr": "Seq2seq model — YAML not yet created",
    "umt5": "Seq2seq model — YAML not yet created",
    # --- CausalLM models (YAML not yet created) ---
    "baichuan": "CausalLM — YAML not yet created",
    "doge": "CausalLM — YAML not yet created",
    "exaone": "CausalLM — YAML not yet created",
    "falcon_h1": "CausalLM — YAML not yet created",
    "falcon_mamba": "CausalLM — YAML not yet created",
    "internlm2": "CausalLM — YAML not yet created",
    "minicpm": "CausalLM — YAML not yet created",
    "minicpm3": "CausalLM — YAML not yet created",
    "ministral": "CausalLM — YAML not yet created",
    "ministral3": "CausalLM — YAML not yet created",
    "mistral3": "CausalLM — YAML not yet created",
    "openelm": "CausalLM — YAML not yet created",
    "qwen": "CausalLM — YAML not yet created",
    "youtu": "CausalLM — YAML not yet created",
    "zamba": "CausalLM — YAML not yet created",
    "zamba2": "CausalLM — YAML not yet created",
}


def _yaml_model_ids() -> set[str]:
    """Collect all model_ids from YAML test case files."""
    return {case.model_id for case in discover_test_cases()}


def _yaml_cases_by_model_id() -> dict[str, object]:
    """Map model_id → GoldenTestCase for JSON golden file lookups."""
    cases = {}
    for case in discover_test_cases():
        cases[case.model_id] = case
    return cases


def _all_registered_with_test_id() -> dict[str, str]:
    """Return {model_type: test_model_id} for all registered models
    that have a test_model_id.
    """
    return {
        arch: model_id for arch, model_id in _TEST_MODEL_IDS.items() if arch in registry._map
    }


class TestGoldenDataCoverage:
    """Ensure every registered model has L4/L5 golden test data."""

    def test_all_models_have_yaml_or_skip(self):
        """Each model with a test_model_id must have YAML or be in _GOLDEN_SKIP."""
        yaml_ids = _yaml_model_ids()
        models = _all_registered_with_test_id()

        missing = []
        for arch, model_id in sorted(models.items()):
            if arch in _GOLDEN_SKIP:
                continue
            if model_id in yaml_ids:
                continue
            missing.append(f"  {arch}: {model_id}")

        if missing:
            msg = (
                f"{len(missing)} registered model(s) have no YAML test case "
                f"and are not in _GOLDEN_SKIP:\n"
                + "\n".join(missing)
                + "\n\nFix: add a YAML file in testdata/cases/ or add to "
                "_GOLDEN_SKIP with a reason."
            )
            pytest.fail(msg)

    def test_all_yaml_cases_have_golden_json_or_skip_reason(self):
        """Each YAML case must have a golden JSON file or a skip_reason.

        A model with a YAML test spec but no JSON golden output is
        incomplete — it should either have the golden data generated
        or its YAML should include a ``skip_reason`` explaining why.
        """
        cases = discover_test_cases()
        incomplete = []
        for case in cases:
            if has_golden(case):
                continue
            if case.skip_reason:
                continue
            golden = golden_path_for_case(case)
            incomplete.append(
                f"  {case.model_id}: YAML exists but golden JSON missing at {golden}"
            )

        if incomplete:
            msg = (
                f"{len(incomplete)} YAML test case(s) have no golden "
                f"JSON output and no skip_reason:\n"
                + "\n".join(incomplete)
                + "\n\nFix: generate the golden JSON, or add "
                "skip_reason to the YAML."
            )
            pytest.fail(msg)

    def test_skip_entries_are_still_registered(self):
        """Entries in _GOLDEN_SKIP should still be registered models.

        If a model is removed from the registry, it should also be
        removed from _GOLDEN_SKIP to avoid stale entries.
        """
        registered = set(registry._map.keys())
        stale = set(_GOLDEN_SKIP.keys()) - registered
        if stale:
            pytest.fail(
                f"Stale entries in _GOLDEN_SKIP (no longer registered): {sorted(stale)}"
            )

    def test_skip_entries_have_reasons(self):
        """Every _GOLDEN_SKIP entry must have a non-empty reason string."""
        empty = [arch for arch, reason in _GOLDEN_SKIP.items() if not reason.strip()]
        if empty:
            pytest.fail(f"_GOLDEN_SKIP entries with empty reasons: {sorted(empty)}")

    @pytest.mark.parametrize(
        "arch,model_id",
        [
            pytest.param(arch, mid, id=arch)
            for arch, mid in sorted(_all_registered_with_test_id().items())
        ],
    )
    def test_model_has_golden_data(self, arch: str, model_id: str):
        """Per-model parametrized check for YAML + JSON coverage."""
        if arch in _GOLDEN_SKIP:
            pytest.skip(_GOLDEN_SKIP[arch])

        yaml_ids = _yaml_model_ids()
        if model_id not in yaml_ids:
            pytest.fail(
                f"Model '{arch}' (test_model_id='{model_id}') has no "
                f"YAML test case in testdata/cases/. "
                f"Add a YAML file or add '{arch}' to _GOLDEN_SKIP."
            )

        # YAML exists — also check for golden JSON output
        cases_by_id = _yaml_cases_by_model_id()
        case = cases_by_id.get(model_id)
        if case is not None and not has_golden(case):
            if case.skip_reason:
                pytest.skip(f"YAML has skip_reason (no golden JSON): {case.skip_reason}")
            golden = golden_path_for_case(case)
            pytest.fail(
                f"Model '{arch}' has YAML but no golden JSON at "
                f"{golden}. Generate golden data or add skip_reason "
                f"to the YAML."
            )
