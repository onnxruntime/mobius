# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Model test-coverage audit: every registered architecture must have L1-L5 data.

Replaces ``scripts/check_new_model_coverage.py`` and the CI workflow
``.github/workflows/model-coverage.yml`` which only checked *newly added*
models via ``git diff``.  This pytest checks **all** registered models on
every run, so coverage gaps cannot accumulate silently.

Coverage levels
~~~~~~~~~~~~~~~

=====  ========  =====================================================
Level  Artefact  What it proves
=====  ========  =====================================================
L1     ``tests/_test_configs.py`` entry       Graph builds without error
L2     ``test_model_id`` in ``_registry.py``  HF config can be fetched
L3     (same as L1)                           Synthetic-parity forward pass
L4     YAML in ``testdata/cases/``            End-to-end test spec exists
L5     JSON in ``testdata/golden/``           Reference outputs available
=====  ========  =====================================================

Adding a new model?
~~~~~~~~~~~~~~~~~~~

1. Add a config to ``tests/_test_configs.py``  →  L1 + L3
2. Set ``test_model_id`` in ``_registry.py``   →  L2
3. Add YAML in ``testdata/cases/``             →  L4
4. Generate golden JSON (or add ``skip_reason`` to YAML)  →  L5

See ``.github/skills/writing-tests/SKILL.md`` for the full guide.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from mobius._registry import _TEST_MODEL_IDS, registry
from mobius._testing.golden import (
    discover_test_cases,
    golden_path_for_case,
    has_golden,
)

# ---------------------------------------------------------------------------
# L1/L3 config discovery (test configs from _test_configs.py)
# ---------------------------------------------------------------------------
_TESTS_DIR = Path(__file__).resolve().parent

if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

from _test_configs import (  # noqa: E402
    ALL_CAUSAL_LM_CONFIGS,
    ENCODER_CONFIGS,
    SEQ2SEQ_CONFIGS,
    VISION_CONFIGS,
)


def _l1_l3_model_types() -> set[str]:
    """Return model_types that have a test config in _test_configs.py."""
    types: set[str] = set()
    for mt, _, _ in ALL_CAUSAL_LM_CONFIGS + ENCODER_CONFIGS + SEQ2SEQ_CONFIGS + VISION_CONFIGS:
        types.add(mt)
    return types


# ---------------------------------------------------------------------------
# L4/L5 helpers
# ---------------------------------------------------------------------------


def _yaml_model_ids() -> set[str]:
    """Collect all model_ids present in YAML test case files."""
    return {case.model_id for case in discover_test_cases()}


def _yaml_cases_by_model_id() -> dict[str, object]:
    """Map model_id → GoldenTestCase for JSON golden file lookups."""
    return {case.model_id: case for case in discover_test_cases()}


# ---------------------------------------------------------------------------
# Registry helpers
# ---------------------------------------------------------------------------


def _all_registered() -> list[str]:
    """Return all model_types from the registry, sorted."""
    return sorted(registry.architectures())


def _all_registered_with_test_id() -> dict[str, str]:
    """Return {model_type: test_model_id} for registered models with one."""
    return {
        arch: model_id for arch, model_id in _TEST_MODEL_IDS.items() if arch in registry._map
    }


# ── Skip list: models that cannot have full coverage (with reasons) ──────
#
# Each entry maps model_type → reason.  The reason is displayed when the
# parametrized test is skipped, so keep it informative.
#
# Categories (each section has a comment header):
#   - Internal / duplicate aliases
#   - VL text-decoder submodels
#   - Multimodal (needs image/video inputs)
#   - Audio (needs audio inputs)
#   - trust_remote_code / no public weights
#   - Very large models
#   - Vision / Encoder / Seq2seq / CausalLM not yet created
#
_COVERAGE_SKIP: dict[str, str] = {
    # --- Internal / duplicate aliases ---
    "code_llama": "Alias for llama — covered by llama",
    "command_r": "Alias for cohere — covered by cohere",
    "deepseek_v2_moe": "Internal alias for deepseek_v2",
    "gpt_oss": "Internal model — no public HF checkpoint",
    "gptoss": "Internal model — no public HF checkpoint",
    "helium": "Alias for mistral — covered by mistral",
    "open-llama": "Alias for llama — covered by llama",
    "seed_oss": "Internal model — no public HF checkpoint",
    "shieldgemma2": "Alias for gemma2 — covered by gemma2",
    "yi": "Alias for llama — covered by llama",
    # --- VL text-decoder submodels (tested via their parent VL model) ---
    "glm4v_text": "VL text decoder — tested via glm4v",
    "glm4v_moe_text": "VL text decoder — tested via glm4v_moe",
    "qwen2_5_vl_text": "VL text decoder — tested via qwen2_5_vl",
    "qwen2_vl_text": "VL text decoder — tested via qwen2_vl",
    "qwen3_5_vl_text": "VL text decoder — tested via qwen3_5_vl",
    "qwen3_vl_text": "VL text decoder — tested via qwen3_vl",
    "qwen3_vl_moe": "VL MoE text decoder — tested via qwen3_vl",
    "qwen3_omni_moe": "VL MoE text decoder — tested via qwen3_omni",
    # --- Vision-language models (test config requires image inputs) ---
    "aya_vision": "VL model — test config requires image inputs",
    "blip": "VL model — test config requires image inputs",
    "blip-2": "VL model — test config requires image inputs",
    "chameleon": "VL model — test config requires image inputs",
    "cohere2_vision": "VL model — test config requires image inputs",
    "deepseek_vl": "VL model — test config requires image inputs",
    "deepseek_vl_hybrid": "VL model — test config requires image inputs",
    "deepseek_vl_v2": "VL model — test config requires image inputs",
    "florence2": "VL model — test config requires image inputs",
    "fuyu": "VL model — test config requires image inputs",
    "gemma3_multimodal": "VL model — test config requires image inputs",
    "glm4v": "VL model — test config requires image inputs",
    "glm4v_moe": "VL model — test config requires image inputs",
    "got_ocr2": "VL model — test config requires image inputs",
    "idefics2": "VL model — test config requires image inputs",
    "idefics3": "VL model — test config requires image inputs",
    "instructblip": "VL model — test config requires image inputs",
    "instructblipvideo": "VL model — test config requires video inputs",
    "internvl": "VL model — test config requires image inputs",
    "internvl2": "VL model — test config requires image inputs",
    "internvl_chat": "VL model — test config requires image inputs",
    "janus": "VL model — test config requires image inputs",
    "llava": "VL model — test config requires image inputs",
    "llava_next": "VL model — test config requires image inputs",
    "llava_next_video": "VL model — test config requires video inputs",
    "llava_onevision": "VL model — test config requires image inputs",
    "mllama": "VL model — test config requires image inputs",
    "molmo": "VL model — test config requires image inputs",
    "ovis2": "VL model — test config requires image inputs",
    "paligemma": "VL model — test config requires image inputs",
    "phi4_multimodal": "VL model (14B) — needs GPU for golden",
    "phi4mm": "VL model (14B) — needs GPU for golden",
    "pixtral": "VL model — test config requires image inputs",
    "qwen2_5_vl": "VL model — test config requires image inputs",
    "qwen2_vl": "VL model — test config requires image inputs",
    "qwen3_5": "VL model — hybrid VL, requires image inputs",
    "qwen3_5_vl": "VL model — test config requires image inputs",
    "qwen3_vl": "VL model — test config requires image inputs",
    "qwen3_vl_single": "VL model — test config requires image inputs",
    "smolvlm": "VL model — test config requires image inputs",
    "video_llava": "VL model — test config requires video inputs",
    "vipllava": "VL model — test config requires image inputs",
    # --- Audio / speech models (test config requires audio inputs) ---
    "data2vec-audio": "Audio model — requires audio inputs",
    "hubert": "Audio model — requires audio inputs",
    "mctct": "Audio model — requires audio inputs",
    "musicgen": "Audio model — requires audio inputs",
    "qwen3_asr": "Audio model — requires audio inputs",
    "qwen3_forced_aligner": "Speech model — requires audio inputs",
    "qwen3_tts": "TTS model — requires specialised pipeline",
    "qwen3_tts_tokenizer_12hz": "Codec model — requires audio inputs",
    "seamless_m4t": "Audio model — requires audio inputs",
    "seamless_m4t_v2": "Audio model — requires audio inputs",
    "sew": "Audio model — requires audio inputs",
    "sew-d": "Audio model — requires audio inputs",
    "speecht5": "Audio model — requires audio inputs",
    "unispeech": "Audio model — requires audio inputs",
    "unispeech-sat": "Audio model — requires audio inputs",
    "voxtral_encoder": "Audio encoder — requires audio inputs",
    "wav2vec2": "Audio model — requires audio inputs",
    "wav2vec2-bert": "Audio model — requires audio inputs",
    "wav2vec2-conformer": "Audio model — requires audio inputs",
    "wavlm": "Audio model — requires audio inputs",
    "whisper": "Speech-to-text — requires audio inputs",
    # --- Models requiring trust_remote_code or no public weights ---
    "chatglm": "Requires trust_remote_code (custom HF modeling code)",
    "dots1": "Requires trust_remote_code (custom HF modeling code)",
    # --- SSM / state-space models (specialised config not yet added) ---
    "mamba": "SSM model — specialised config not yet added",
    "mamba2": "SSM model — specialised config not yet added",
    # --- Very large models without small public checkpoints ---
    "arctic": "Very large MoE (480B) — no small public checkpoint",
    "dbrx": "Large MoE (132B) — no small public checkpoint",
    "deepseek_v3": "Very large MoE (671B) — no small public checkpoint",
    "llama4_text": "Very large MoE (109B) — no small public checkpoint",
    "switch_transformers": "Large MoE — needs many experts for golden",
    "qwen3_5_moe": "Large MoE (22B) — no small public checkpoint",
    # --- Models without test_model_id (no public HF checkpoint found) ---
    "codegen2": "No test_model_id — no suitable public checkpoint",
    "csm": "No test_model_id — no suitable public checkpoint",
    "dinov3_vit": "Vision model — no test_model_id yet",
    "evolla": "No test_model_id — no suitable public checkpoint",
    "ijepa": "Vision model — no test_model_id yet",
    "megatron-bert": "Encoder — no test_model_id yet",
    "modernbert-decoder": "Decoder variant — no test_model_id yet",
    "nemotron_h": "No test_model_id — no suitable public checkpoint",
    "nllb-moe": "Seq2seq MoE — no test_model_id yet",
    "nllb_moe": "Seq2seq MoE — no test_model_id yet",
    "persimmon": "No test_model_id — no suitable public checkpoint",
    "qdqbert": "Quantised BERT — no test_model_id yet",
    "sam2": "Vision model — no test_model_id yet",
    "solar_open": "No test_model_id — no suitable public checkpoint",
    "vit_hybrid": "Vision model — no test_model_id yet",
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


# ── Tests ─────────────────────────────────────────────────────────────────


class TestSkipListIntegrity:
    """Guard-rail tests for the skip list itself."""

    def test_skip_entries_are_still_registered(self):
        """Entries in _COVERAGE_SKIP must still be in the registry.

        If a model is removed from the registry, clean it from
        _COVERAGE_SKIP too.
        """
        registered = set(registry._map.keys())
        stale = set(_COVERAGE_SKIP.keys()) - registered
        if stale:
            pytest.fail(
                f"Stale entries in _COVERAGE_SKIP (no longer registered): {sorted(stale)}"
            )

    def test_skip_entries_have_reasons(self):
        """Every _COVERAGE_SKIP entry must have a non-empty reason."""
        empty = [arch for arch, reason in _COVERAGE_SKIP.items() if not reason.strip()]
        if empty:
            pytest.fail(f"_COVERAGE_SKIP entries with empty reasons: {sorted(empty)}")


class TestL1L3GraphBuildCoverage:
    """L1 + L3: every model needs a test config in _test_configs.py.

    The config enables ``build_graph_test.py`` and
    ``synthetic_parity_test.py`` to exercise the model.
    """

    def test_all_models_have_test_config_or_skip(self):
        """Aggregate check: every registered model needs an L1/L3 config."""
        l13 = _l1_l3_model_types()
        all_reg = _all_registered()

        missing = [mt for mt in all_reg if mt not in l13 and mt not in _COVERAGE_SKIP]
        if missing:
            pytest.fail(
                f"{len(missing)} registered model(s) have no test "
                f"config in _test_configs.py and are not in "
                f"_COVERAGE_SKIP:\n"
                + "\n".join(f"  {mt}" for mt in missing)
                + "\n\nFix: add a config entry to "
                "tests/_test_configs.py or add to _COVERAGE_SKIP."
            )

    @pytest.mark.parametrize("arch", _all_registered())
    def test_model_has_l1_l3_config(self, arch: str):
        """Per-model check for L1/L3 test config."""
        if arch in _COVERAGE_SKIP:
            pytest.skip(_COVERAGE_SKIP[arch])
        l13 = _l1_l3_model_types()
        if arch not in l13:
            pytest.fail(
                f"Model '{arch}' has no test config in "
                f"tests/_test_configs.py. Add one for L1/L3 coverage."
            )


class TestL2ConfigValidation:
    """L2: every model needs a ``test_model_id`` in ``_TEST_MODEL_IDS``.

    This allows ``arch_validation_test.py`` to fetch and validate its
    HuggingFace config.
    """

    def test_all_models_have_test_model_id_or_skip(self):
        """Aggregate check: every registered model needs a test_model_id."""
        all_reg = _all_registered()
        missing = [
            mt for mt in all_reg if mt not in _TEST_MODEL_IDS and mt not in _COVERAGE_SKIP
        ]
        if missing:
            pytest.fail(
                f"{len(missing)} registered model(s) have no "
                f"test_model_id in _registry.py and are not in "
                f"_COVERAGE_SKIP:\n"
                + "\n".join(f"  {mt}" for mt in missing)
                + "\n\nFix: add test_model_id to _TEST_MODEL_IDS "
                "in src/mobius/_registry.py."
            )

    @pytest.mark.parametrize("arch", _all_registered())
    def test_model_has_test_model_id(self, arch: str):
        """Per-model check for test_model_id (L2)."""
        if arch in _COVERAGE_SKIP:
            pytest.skip(_COVERAGE_SKIP[arch])
        if arch not in _TEST_MODEL_IDS:
            pytest.fail(
                f"Model '{arch}' has no test_model_id in "
                f"_TEST_MODEL_IDS. Add one for L2 config validation."
            )


class TestL4L5GoldenDataCoverage:
    """L4 + L5: every model needs a YAML test case and JSON golden output.

    L4 = YAML in ``testdata/cases/``, L5 = JSON in ``testdata/golden/``.
    """

    def test_all_models_have_yaml_or_skip(self):
        """Aggregate: each model with test_model_id needs YAML or skip."""
        yaml_ids = _yaml_model_ids()
        models = _all_registered_with_test_id()

        missing = []
        for arch, model_id in sorted(models.items()):
            if arch in _COVERAGE_SKIP:
                continue
            if model_id in yaml_ids:
                continue
            missing.append(f"  {arch}: {model_id}")

        if missing:
            pytest.fail(
                f"{len(missing)} registered model(s) have no YAML "
                f"test case and are not in _COVERAGE_SKIP:\n"
                + "\n".join(missing)
                + "\n\nFix: add a YAML file in testdata/cases/ or "
                "add to _COVERAGE_SKIP with a reason."
            )

    def test_all_yaml_cases_have_golden_json_or_skip_reason(self):
        """Aggregate: YAML without JSON must have a skip_reason."""
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
            pytest.fail(
                f"{len(incomplete)} YAML test case(s) have no golden "
                f"JSON output and no skip_reason:\n"
                + "\n".join(incomplete)
                + "\n\nFix: generate the golden JSON, or add "
                "skip_reason to the YAML."
            )

    @pytest.mark.parametrize(
        "arch,model_id",
        [
            pytest.param(arch, mid, id=arch)
            for arch, mid in sorted(_all_registered_with_test_id().items())
        ],
    )
    def test_model_has_golden_data(self, arch: str, model_id: str):
        """Per-model check for YAML (L4) + JSON golden output (L5)."""
        if arch in _COVERAGE_SKIP:
            pytest.skip(_COVERAGE_SKIP[arch])

        yaml_ids = _yaml_model_ids()
        if model_id not in yaml_ids:
            pytest.fail(
                f"Model '{arch}' (test_model_id='{model_id}') has "
                f"no YAML test case in testdata/cases/. "
                f"Add a YAML file or add '{arch}' to _COVERAGE_SKIP."
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
