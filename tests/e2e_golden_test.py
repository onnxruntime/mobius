# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""L4 (Checkpoint Verified) and L5 (Generation E2E) golden tests.

Data-driven: each YAML file in ``testdata/cases/`` is a test case.
Adding coverage = adding a YAML + ``.json`` file.  No code changes needed.

Run::

    pytest tests/e2e_golden_test.py -v                   # all
    pytest tests/e2e_golden_test.py -k "qwen2_5-0_5b"    # by model
    pytest tests/e2e_golden_test.py -m golden              # L4 only
    pytest tests/e2e_golden_test.py -m generation          # L5 only
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np
import pytest

from mobius import build
from mobius._model_package import ModelPackage
from mobius._testing.generation import OnnxGenerator
from mobius._testing.golden import (
    GoldenRef,
    GoldenTestCase,
    discover_test_cases,
    golden_path_for_case,
    has_golden,
    load_golden_ref,
    load_tolerances,
)
from mobius._testing.ort_inference import OnnxModelSession
from mobius._testing.parity import ParityResult, compare_golden

# Root of test data (images, audio, etc.)
_TESTDATA_DIR = Path(__file__).resolve().parent.parent / "testdata"


@pytest.fixture(autouse=True)
def _use_temp_hf_cache(tmp_path):
    """Redirect HuggingFace downloads to a per-test temp dir.

    Each test gets a fresh cache that is deleted when the test finishes,
    so only one model's weights are on disk at a time.  This prevents
    unbounded disk growth across the full test suite.

    Each pytest-xdist worker gets its own ``tmp_path``, so parallel
    workers don't collide.
    """
    cache_dir = str(tmp_path / "hf_cache")
    old = os.environ.get("HF_HOME")
    os.environ["HF_HOME"] = cache_dir
    yield
    if old is None:
        os.environ.pop("HF_HOME", None)
    else:
        os.environ["HF_HOME"] = old


# ---------------------------------------------------------------------------
# Test case discovery (runs at collection time)
# ---------------------------------------------------------------------------

# Known failures that should be xfailed rather than treated as regressions.
# Key: "{task_type}/{case_id}" matching the pytest test ID.
_XFAIL_REASONS: dict[str, str] = {
    # Weight loading bugs: preprocess_weights doesn't map all HF names
    "text-generation/gpt2": "GPT-2 preprocess_weights incomplete (197/203 weights unmapped)",
    "text-generation/pythia-70m": "GPT-NeoX preprocess_weights incomplete",
    "text-generation/phi-1_5": "Phi-1.5 preprocess_weights incomplete",
    "text-generation/starcoder2-3b": "StarCoder2 preprocess_weights incomplete",
    "text-generation/mamba-130m": "Mamba conv_state requires rank-3 tensors (not standard KV cache)",
    "text-generation/olmoe-1b-7b": "OLMoE MoE weight mapping incomplete",
    "text-generation/qwen1_5-moe": "Qwen1.5-MoE weight mapping incomplete",
    "text-generation/bamba-9b": "Bamba conv_state requires rank-3 tensors (hybrid Mamba)",
    "feature-extraction/albert-base-v2": "ALBERT shared-parameter weight loading incomplete",
    "feature-extraction/modernbert-base": "ModernBERT preprocess_weights incomplete",
    # Real parity failures: weights load but argmax doesn't match
    "text-generation/granitemoe-1b": "GraniteMoE parity failure (argmax mismatch, 0% Jaccard)",
    "text-generation/stablelm-2-1_6b": "StableLM parity failure (argmax mismatch, 0% Jaccard)",
    "text-generation/gemma-2-2b": "Gemma-2 L5 generation diverges (10% token match ratio)",
    # VL multi-model inference: test infra needs model-specific position_ids
    "image-text-to-text/qwen3-vl-2b": "VL prefill needs 3D position_ids (model-specific RoPE)",
    "image-text-to-text/qwen2_5-vl-3b": "VL prefill needs 3D position_ids (model-specific RoPE)",
    "image-text-to-text/llava-1_5-7b": "VL multi-model prefill pipeline not yet implemented for LLaVA",
}


def _discover_cases(level: str) -> list[pytest.ParameterSet]:
    """Discover YAML test cases and wrap as ``pytest.param`` entries.

    Missing golden files or explicit ``skip_reason`` fields produce
    ``pytest.mark.skip`` so pytest shows "SKIPPED" at collection time
    rather than failing at run time.
    """
    cases = discover_test_cases(level=level)
    params: list[pytest.ParameterSet] = []
    for case in cases:
        marks: list[pytest.MarkDecorator] = []
        test_id = f"{case.task_type}/{case.case_id}"

        if case.skip_reason:
            marks.append(pytest.mark.skip(reason=case.skip_reason))
        elif not has_golden(case):
            marks.append(
                pytest.mark.skip(reason=(f"Golden file missing: {golden_path_for_case(case)}"))
            )
        elif test_id in _XFAIL_REASONS:
            marks.append(pytest.mark.xfail(reason=_XFAIL_REASONS[test_id], strict=False))

        params.append(
            pytest.param(
                case,
                id=test_id,
                marks=marks,
            )
        )
    return params


_L4_CASES = _discover_cases("L4")
_L5_CASES = _discover_cases("L5")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_model_package(case: GoldenTestCase) -> ModelPackage:
    """Build an ONNX ModelPackage with real weights from HuggingFace."""
    return build(
        case.model_id,
        dtype=case.dtype,
        load_weights=True,
        trust_remote_code=case.trust_remote_code,
    )


def _open_decoder_session(pkg: ModelPackage) -> OnnxModelSession:
    """Open an ORT session for the decoder / primary model.

    Single-model packages (causal-lm, encoder): uses the sole model.
    Multi-model packages (vision-language): uses the ``"model"`` key,
    which is the decoder component that produces logits.
    Seq2seq packages: uses the ``"decoder"`` key.
    """
    if len(pkg) == 1:
        return OnnxModelSession(pkg)
    if "model" in pkg:
        return OnnxModelSession(pkg["model"])
    if "decoder" in pkg:
        return OnnxModelSession(pkg["decoder"])
    raise KeyError(f"Cannot find decoder model in package. Keys: {sorted(pkg.keys())}")


def _run_seq2seq_prefill(
    pkg: ModelPackage,
    golden: GoldenRef,
    config: object,
) -> dict[str, np.ndarray]:
    """Run encoder → decoder for seq2seq models and return decoder outputs.

    Seq2seq requires a two-step inference: first run the encoder on the
    source input_ids, then feed encoder_hidden_states plus a decoder
    start token to the decoder.
    """
    input_ids = np.array(golden.input_ids, dtype=np.int64).reshape(1, -1)
    seq_len = input_ids.shape[1]

    # Step 1: Run encoder
    enc_session = OnnxModelSession(pkg["encoder"])
    try:
        enc_feeds = {
            "input_ids": input_ids,
            "attention_mask": np.ones_like(input_ids),
        }
        enc_outputs = enc_session.run(enc_feeds)
    finally:
        enc_session.close()

    # Extract encoder_hidden_states (key may vary)
    enc_hidden = None
    for key in ("encoder_hidden_states", "last_hidden_state"):
        if key in enc_outputs:
            enc_hidden = enc_outputs[key]
            break
    if enc_hidden is None:
        raise KeyError(
            f"Encoder output missing hidden states. Keys: {sorted(enc_outputs.keys())}"
        )

    # Step 2: Run decoder with encoder output + decoder start token
    dec_session = OnnxModelSession(pkg["decoder"])
    try:
        decoder_start_id = getattr(config, "decoder_start_token_id", 0) or 0
        dec_input_ids = np.array([[decoder_start_id]], dtype=np.int64)

        dec_feeds: dict[str, np.ndarray] = {
            "input_ids": dec_input_ids,
            "encoder_hidden_states": enc_hidden,
            "attention_mask": np.ones((1, seq_len), dtype=np.int64),
        }

        # Fill KV cache inputs for decoder
        num_kv_heads = getattr(config, "num_key_value_heads", None)
        head_dim = getattr(config, "head_dim", None)
        if num_kv_heads is not None and head_dim is not None:
            for name in dec_session.input_names:
                if name.startswith("past_key_values."):
                    if ".cross." in name:
                        # Cross-attention cache: enc_seq_len=0 initially
                        dec_feeds[name] = np.zeros(
                            (1, num_kv_heads, 0, head_dim),
                            dtype=np.float32,
                        )
                    else:
                        # Self-attention cache: past_seq_len=0
                        dec_feeds[name] = np.zeros(
                            (1, num_kv_heads, 0, head_dim),
                            dtype=np.float32,
                        )

        outputs = dec_session.run(dec_feeds)
    finally:
        dec_session.close()

    return outputs


def _prepare_prefill_feeds(
    golden: GoldenRef,
    config: object,
    session: OnnxModelSession,
) -> dict[str, np.ndarray]:
    """Prepare input feeds for a prefill forward pass.

    Uses the tokenized ``input_ids`` from the golden file to guarantee
    the same tokenization that produced the reference.  Initialises
    empty KV cache for all ``past_key_values`` inputs.

    Args:
        golden: Golden reference data (provides tokenized input_ids).
        config: ArchitectureConfig with ``num_key_value_heads`` and
            ``head_dim`` attributes.
        session: Open ORT session (provides input name list).
    """
    # Golden input_ids are stored as a flat int list; reshape to (1, seq_len)
    input_ids = np.array(golden.input_ids, dtype=np.int64).reshape(1, -1)
    seq_len = input_ids.shape[1]

    feeds: dict[str, np.ndarray] = {
        "input_ids": input_ids,
        "attention_mask": np.ones_like(input_ids),
        "position_ids": np.arange(seq_len, dtype=np.int64).reshape(1, -1),
    }

    # Provide token_type_ids for models that need it (BERT, ALBERT, etc.)
    if "token_type_ids" in session.input_names:
        feeds["token_type_ids"] = np.zeros_like(input_ids)

    # Fill KV cache inputs with zero-length tensors.
    # Shape: (batch=1, num_kv_heads, past_seq_len=0, head_dim)
    has_kv_inputs = any(n.startswith("past_key_values.") for n in session.input_names)
    if has_kv_inputs:
        assert hasattr(config, "num_key_value_heads"), (
            f"Config {type(config).__name__} missing "
            f"'num_key_value_heads' — cannot build KV cache feeds"
        )
        assert hasattr(config, "head_dim"), (
            f"Config {type(config).__name__} missing 'head_dim' — cannot build KV cache feeds"
        )
        num_kv_heads = config.num_key_value_heads  # type: ignore[union-attr]
        head_dim = config.head_dim  # type: ignore[union-attr]
        for name in session.input_names:
            if name.startswith("past_key_values."):
                feeds[name] = np.zeros(
                    (1, num_kv_heads, 0, head_dim),
                    dtype=np.float32,
                )

    return feeds


# Task types that produce ``last_hidden_state`` instead of ``logits``.
_HIDDEN_STATE_TASKS: frozenset[str] = frozenset(
    {
        "feature-extraction",
        "image-classification",
        "audio-feature-extraction",
    }
)


def _extract_logits(
    outputs: dict[str, np.ndarray],
    task_type: str,
) -> np.ndarray:
    """Extract the logit tensor from model outputs.

    For text-generation and seq2seq tasks, returns ``outputs["logits"]``.
    For feature-extraction, image-classification, and audio-feature-extraction
    tasks, falls back to ``outputs["last_hidden_state"]``.
    """
    if "logits" in outputs:
        return outputs["logits"]
    if task_type in _HIDDEN_STATE_TASKS and "last_hidden_state" in outputs:
        return outputs["last_hidden_state"]
    raise KeyError(
        f"No logits found in outputs for task_type={task_type!r}. "
        f"Available keys: {sorted(outputs.keys())}"
    )


def _token_match_ratio(
    actual: np.ndarray,
    expected: np.ndarray,
) -> float:
    """Compute the fraction of matching tokens between two sequences.

    When lengths differ, only the overlapping prefix is compared.
    The denominator is ``len(expected)`` so shorter actual sequences
    are penalized.  Callers should emit a diagnostic warning when
    ``len(actual) != len(expected)`` to distinguish length mismatches
    from token-value mismatches.
    """
    min_len = min(len(actual), len(expected))
    if min_len == 0:
        return 0.0
    matches = sum(1 for a, e in zip(actual[:min_len], expected[:min_len]) if a == e)
    return matches / len(expected)


def _prepare_vision_feeds(
    case: GoldenTestCase,
) -> dict[str, np.ndarray]:
    """Prepare input feeds for an image-classification forward pass.

    Loads the test image and preprocesses it with the HuggingFace
    image processor to produce ``pixel_values``.
    """
    import transformers
    from PIL import Image

    processor = transformers.AutoImageProcessor.from_pretrained(
        case.model_id, trust_remote_code=case.trust_remote_code
    )
    image = Image.open(_TESTDATA_DIR / case.images[0])
    processed = processor(images=image, return_tensors="np")
    feeds: dict[str, np.ndarray] = {
        "pixel_values": processed["pixel_values"].astype(np.float32),
    }
    return feeds


def _prepare_audio_feeds(
    case: GoldenTestCase,
) -> dict[str, np.ndarray]:
    """Prepare input feeds for an audio-feature-extraction forward pass.

    Loads the test audio and preprocesses it with the HuggingFace
    feature extractor to produce ``input_values``.
    """
    import librosa
    import transformers

    # Fall back to AutoFeatureExtractor for models without a tokenizer
    try:
        processor = transformers.AutoProcessor.from_pretrained(
            case.model_id, trust_remote_code=case.trust_remote_code
        )
    except (TypeError, OSError):
        processor = transformers.AutoFeatureExtractor.from_pretrained(
            case.model_id, trust_remote_code=case.trust_remote_code
        )
    audio_path = _TESTDATA_DIR / case.audio[0]
    audio_array, _sr = librosa.load(str(audio_path), sr=16000)
    processed = processor(audio_array, sampling_rate=16000, return_tensors="np")
    feeds: dict[str, np.ndarray] = {
        # Assumes the ONNX model's input is named "input_values" — the
        # standard key for Wav2Vec2-family audio encoder models.
        "input_values": processed["input_values"].astype(np.float32),
    }
    return feeds


def _run_vision_language_prefill(
    pkg: ModelPackage,
    case: GoldenTestCase,
    config: object,
) -> dict[str, np.ndarray]:
    """Run vision → embedding → decoder for vision-language models.

    The VL pipeline has 3 ONNX models:
    - ``vision``: pixel_values → image hidden states
    - ``embedding``: input_ids + image hidden states → inputs_embeds
    - ``model``: inputs_embeds → logits

    This replicates the full HuggingFace forward pass used during
    golden generation.
    """
    import transformers
    from PIL import Image

    # --- Step 0: Preprocess image with HF processor ---
    processor = transformers.AutoProcessor.from_pretrained(
        case.model_id, trust_remote_code=case.trust_remote_code
    )
    image = Image.open(_TESTDATA_DIR / case.images[0])

    # Build chat template for models that need it
    prompt_text = case.prompts[0]
    if hasattr(processor, "apply_chat_template"):
        content: list[dict[str, str]] = []
        for img_path in case.images:
            content.append({"type": "image", "image": str(_TESTDATA_DIR / img_path)})
        content.append({"type": "text", "text": prompt_text})
        messages = [{"role": "user", "content": content}]
        prompt_text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    # Use PyTorch tensors then convert — some processors don't support np
    processed_pt = processor(text=prompt_text, images=[image], return_tensors="pt")
    processed: dict[str, np.ndarray] = {
        k: v.numpy() if hasattr(v, "numpy") else np.array(v) for k, v in processed_pt.items()
    }

    # --- Step 1: Run vision encoder ---
    vis_session = OnnxModelSession(pkg["vision"])
    try:
        vis_feeds: dict[str, np.ndarray] = {}
        for name in vis_session.input_names:
            if name in processed:
                val = processed[name]
                vis_feeds[name] = val if isinstance(val, np.ndarray) else np.array(val)
        vis_out = vis_session.run(vis_feeds)
    finally:
        vis_session.close()

    # Extract the image hidden states (first output)
    vis_hidden_key = next(iter(vis_out))
    vis_hidden = vis_out[vis_hidden_key]

    # --- Step 2: Run embedding model ---
    emb_session = OnnxModelSession(pkg["embedding"])
    try:
        emb_feeds: dict[str, np.ndarray] = {
            "input_ids": processed["input_ids"].astype(np.int64),
        }
        # Pass vision hidden states
        for name in emb_session.input_names:
            if name not in emb_feeds and name in vis_out:
                emb_feeds[name] = vis_out[name]
            elif name == "image_features":
                emb_feeds[name] = vis_hidden
        emb_out = emb_session.run(emb_feeds)
    finally:
        emb_session.close()

    # Extract inputs_embeds
    emb_key = next(iter(emb_out))
    inputs_embeds = emb_out[emb_key]

    # --- Step 3: Run decoder ---
    # VL packages may use "model" or "decoder" for the text decoder.
    dec_key = "model" if "model" in pkg else "decoder"
    dec_session = OnnxModelSession(pkg[dec_key])
    try:
        seq_len = inputs_embeds.shape[1]
        dec_feeds: dict[str, np.ndarray] = {
            "inputs_embeds": inputs_embeds,
        }
        # Pass through processor outputs that match decoder inputs
        # (e.g., attention_mask, position_ids with model-specific shapes)
        for name in dec_session.input_names:
            if name in dec_feeds:
                continue
            if name in processed:
                dec_feeds[name] = processed[name]
            elif name == "attention_mask":
                dec_feeds[name] = np.ones((1, seq_len), dtype=np.int64)
            elif name == "position_ids":
                dec_feeds[name] = np.arange(seq_len, dtype=np.int64).reshape(1, -1)
            elif name.startswith("past_key_values."):
                num_kv_heads = getattr(config, "num_key_value_heads", 1)
                head_dim = getattr(config, "head_dim", 64)
                dec_feeds[name] = np.zeros(
                    (1, num_kv_heads, 0, head_dim),
                    dtype=np.float32,
                )
        outputs = dec_session.run(dec_feeds)
    finally:
        dec_session.close()

    return outputs


# ---------------------------------------------------------------------------
# L4 Tests: Checkpoint Verified
# ---------------------------------------------------------------------------


@pytest.mark.golden
@pytest.mark.integration
class TestL4CheckpointVerified:
    """L4: Single forward pass, compare argmax against golden reference.

    Gate: argmax match (with near-tie AMBIGUOUS tolerance).
    Diagnostic: top-10 Jaccard warning on low overlap.
    """

    @pytest.mark.parametrize("case", _L4_CASES)
    def test_prefill_argmax_matches_golden(self, case: GoldenTestCase) -> None:
        golden_path = golden_path_for_case(case)
        golden = load_golden_ref(golden_path)
        if golden is None:
            pytest.skip(f"Golden file missing: {golden_path}")

        tolerances = load_tolerances("L4", case.dtype)
        pkg = _build_model_package(case)
        config = pkg.config
        assert config is not None, (
            f"ModelPackage for {case.model_id} has no config; "
            "cannot determine KV cache dimensions"
        )

        # Seq2seq models require running encoder → decoder
        if case.task_type == "seq2seq":
            outputs = _run_seq2seq_prefill(pkg, golden, config)
        elif case.task_type == "image-text-to-text":
            outputs = _run_vision_language_prefill(pkg, case, config)
        elif case.task_type == "image-classification":
            session = _open_decoder_session(pkg)
            try:
                feeds = _prepare_vision_feeds(case)
                outputs = session.run(feeds)
            finally:
                session.close()
        elif case.task_type == "audio-feature-extraction":
            session = _open_decoder_session(pkg)
            try:
                feeds = _prepare_audio_feeds(case)
                outputs = session.run(feeds)
            finally:
                session.close()
        else:
            session = _open_decoder_session(pkg)
            try:
                feeds = _prepare_prefill_feeds(golden, config, session)
                outputs = session.run(feeds)
            finally:
                session.close()

        logits = _extract_logits(outputs, case.task_type)

        report = compare_golden(
            onnx_logits=logits,
            golden_top1_id=golden.top1_id,
            golden_top2_id=golden.top2_id,
            golden_top10_ids=golden.top10_ids,
            dtype=case.dtype,
        )

        if report.top10_jaccard < tolerances.top10_jaccard_warn:
            warnings.warn(
                f"Low top-10 Jaccard for {case.case_id}: "
                f"{report.top10_jaccard:.2f} "
                f"< {tolerances.top10_jaccard_warn}",
                stacklevel=1,
            )

        assert report.result != ParityResult.FAIL, report.message


# ---------------------------------------------------------------------------
# L5 Helpers
# ---------------------------------------------------------------------------

# Task types that support autoregressive generation via OnnxGenerator.
# Other task types (seq2seq, speech-to-text, vision-language) require
# specialised generation loops that are not yet implemented.
_GENERATION_SUPPORTED_TASKS = frozenset(
    {
        "text-generation",
    }
)


def _validate_greedy(case: GoldenTestCase) -> None:
    """Ensure the test case uses deterministic (greedy) decoding.

    Golden tests must be reproducible.  Sampling introduces
    platform-dependent randomness and is not supported.
    """
    params = case.generation_params
    if params.get("do_sample", False):
        pytest.skip(
            f"Sampling (do_sample=true) is not supported for "
            f"golden tests ({case.case_id}). "
            f"Golden tests require greedy decoding."
        )
    if params.get("temperature", 0) not in (0, 1, 1.0):
        # temperature != 0 or 1 implies soft sampling
        pytest.skip(
            f"Non-default temperature={params['temperature']} "
            f"not supported for golden tests ({case.case_id})"
        )


def _run_causal_lm_generation(
    pkg: ModelPackage,
    case: GoldenTestCase,
    golden: GoldenRef,
) -> np.ndarray:
    """Run greedy generation for a causal-lm model.

    Returns only the newly generated token IDs (prompt stripped).
    """
    config = pkg.config
    session = _open_decoder_session(pkg)
    try:
        generator = OnnxGenerator(session, config)

        # Golden input_ids are stored as a flat int list
        input_ids = np.array(golden.input_ids, dtype=np.int64).reshape(1, -1)

        max_new_tokens = case.generation_params.get("max_new_tokens", 20)
        eos_token_id = case.generation_params.get("eos_token_id", None)

        all_ids = generator.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
        )
    finally:
        session.close()

    # Strip prompt — generator returns [prompt + generated]
    prompt_len = input_ids.shape[1]
    return all_ids[0, prompt_len:]


# ---------------------------------------------------------------------------
# L5 Tests: Generation E2E
# ---------------------------------------------------------------------------


@pytest.mark.generation
@pytest.mark.integration
class TestL5GenerationE2E:
    """L5: Full autoregressive generation, compare token sequences.

    Gate: token match ratio >= ``min_token_match_ratio`` from tolerances.
    Only causal-lm models are supported; other task types (seq2seq,
    vision-language, speech-to-text) are skipped until specialised
    generation loops are implemented.
    """

    @pytest.mark.parametrize("case", _L5_CASES)
    def test_generation_matches_golden(self, case: GoldenTestCase) -> None:
        # --- Guard: skip unsupported task types ---
        if case.task_type not in _GENERATION_SUPPORTED_TASKS:
            pytest.skip(
                f"L5 generation not yet supported for "
                f"task_type={case.task_type!r} ({case.case_id}). "
                f"Supported: {sorted(_GENERATION_SUPPORTED_TASKS)}"
            )

        # --- Guard: sampling not supported ---
        _validate_greedy(case)

        # --- Load golden data ---
        golden_path = golden_path_for_case(case)
        golden = load_golden_ref(golden_path)
        if golden is None:
            pytest.skip(f"Golden file missing: {golden_path}")

        tolerances = load_tolerances("L5", case.dtype)

        assert golden.generated_ids is not None, (
            f"Golden file for {case.case_id} includes L5 in its level "
            f"but contains no generated_ids"
        )

        # --- Build and generate ---
        pkg = _build_model_package(case)
        config = pkg.config
        assert config is not None, (
            f"ModelPackage for {case.model_id} has no config; "
            "cannot determine KV cache dimensions for generation"
        )

        new_tokens = _run_causal_lm_generation(pkg, case, golden)

        # --- Diagnostics ---
        expected_tokens = np.array(golden.generated_ids, dtype=np.int64)
        expected_len = len(expected_tokens)
        actual_len = len(new_tokens)
        if actual_len != expected_len:
            warnings.warn(
                f"Length mismatch for {case.case_id}: "
                f"expected {expected_len} tokens, "
                f"got {actual_len}",
                stacklevel=1,
            )

        # --- Compare ---
        match_ratio = _token_match_ratio(new_tokens, expected_tokens)

        assert match_ratio >= tolerances.min_token_match_ratio, (
            f"L5 FAIL: token match ratio {match_ratio:.2f} "
            f"< {tolerances.min_token_match_ratio:.2f}\n"
            f"  Expected ({expected_len} tokens): "
            f"{expected_tokens.tolist()}\n"
            f"  Got      ({actual_len} tokens): "
            f"{new_tokens.tolist()}"
        )
