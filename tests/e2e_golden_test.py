# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""L4 (Checkpoint Verified) and L5 (Generation E2E) golden tests.

Data-driven: each YAML file in ``testdata/cases/`` is a test case.
Adding coverage = adding a YAML + ``.json`` file.  No code changes needed.

Run::

    pytest tests/e2e_golden_test.py -v                   # all
    pytest tests/e2e_golden_test.py -k "qwen2_5-0_5b"    # by model
    pytest tests/e2e_golden_test.py -m golden              # L4 only
    pytest tests/e2e_golden_test.py -m generation          # L5 only

    # Run on CUDA GPU:
    MOBIUS_TEST_DEVICE=cuda pytest tests/e2e_golden_test.py -v
"""

from __future__ import annotations

import dataclasses
import os
import warnings
from pathlib import Path

import numpy as np
import pytest

from mobius import build
from mobius._model_package import ModelPackage
from mobius._testing.generation import OnnxGenerator, OnnxSeq2SeqGenerator
from mobius._testing.golden import (
    GoldenRef,
    GoldenTestCase,
    discover_test_cases,
    generation_json_path_for_case,
    golden_path_for_case,
    has_golden,
    load_generation_golden,
    load_golden_ref,
    load_tolerances,
)
from mobius._testing.ort_inference import OnnxModelSession
from mobius._testing.parity import ParityResult, compare_golden

# Root of test data (images, audio, etc.)
_TESTDATA_DIR = Path(__file__).resolve().parent.parent / "testdata"


def _get_test_device_kwargs() -> dict[str, str]:
    """Return OnnxModelSession kwargs from environment variables.

    Set ``MOBIUS_TEST_DEVICE`` to ``cuda`` to run on GPU.
    Set ``MOBIUS_TEST_EP`` to override the execution provider
    (e.g. ``CUDAExecutionProvider``).
    """
    kwargs: dict[str, str] = {}
    device = os.environ.get("MOBIUS_TEST_DEVICE", "").lower()
    if device:
        kwargs["device"] = device
    ep = os.environ.get("MOBIUS_TEST_EP", "")
    if ep:
        kwargs["providers"] = [ep]
    return kwargs


_IN_CI = os.environ.get("GITHUB_ACTIONS") == "true"


def _load_suppress_token_ids(model_id: str, trust_remote_code: bool = False) -> list[int]:
    """Return ``generation_config.suppress_tokens`` for a model (empty if none).

    Mirrors HuggingFace ``generate()``: tokens in ``suppress_tokens`` are forced
    to ``-inf`` at every decode step. Needed for base checkpoints (e.g.
    ``google/gemma-4-12B``) whose generation_config suppresses the structural
    ``<end_of_image>`` / ``<end_of_audio>`` tokens — without it, greedy decode
    degenerates (repeating those tokens) and diverges from the golden reference
    produced by ``model.generate``. For models with no suppress_tokens this is a
    no-op, so it is safe to apply unconditionally.
    """
    import transformers

    try:
        gen_config = transformers.GenerationConfig.from_pretrained(
            model_id, trust_remote_code=trust_remote_code
        )
    except Exception:
        return []
    return [int(t) for t in (gen_config.suppress_tokens or [])]


def _suppress_logits(logits: np.ndarray, suppress_ids: list[int]) -> np.ndarray:
    """Force ``suppress_ids`` columns of the last-position logits to ``-inf``."""
    if suppress_ids:
        logits[..., suppress_ids] = -np.inf
    return logits


def _build_mm_prompt(
    processor: object,
    base_prompt: str,
    media_paths: list[str],
    media_kind: str,
) -> str:
    """Format a multimodal prompt, falling back when there is no chat template.

    Instruction-tuned processors expose a chat template that injects the right
    media placeholder tokens. Base checkpoints (e.g. ``google/gemma-4-12B``)
    ship none, so manually prepend one placeholder token per media item — the
    processor then expands each into the correct number of soft tokens.

    ``media_kind`` is ``"image"`` or ``"audio"``.
    """
    if getattr(processor, "chat_template", None):
        content: list[dict[str, str]] = []
        for path in media_paths:
            content.append({"type": media_kind, media_kind: str(_TESTDATA_DIR / path)})
        content.append({"type": "text", "text": base_prompt})
        messages = [{"role": "user", "content": content}]
        return processor.apply_chat_template(  # type: ignore[attr-defined]
            messages, tokenize=False, add_generation_prompt=True
        )
    placeholder = getattr(processor, f"{media_kind}_token", None)
    if placeholder:
        return placeholder * len(media_paths) + base_prompt
    return base_prompt


def _make_empty_kv_cache(
    session: OnnxModelSession,
    config: object,
) -> dict[str, np.ndarray]:
    """Create empty KV cache feeds using the ORT session's declared shapes.

    Uses the model's own shape declarations so that per-layer dimension
    variations (e.g. KV sharing in Gemma4) are handled correctly.
    The sequence/time dimension is set to 0.
    """
    feeds: dict[str, np.ndarray] = {}
    # Fallback values from config
    default_kv_heads = getattr(config, "num_key_value_heads", 1)
    default_head_dim = getattr(config, "head_dim", 64)
    layer_types = getattr(config, "layer_types", None) or []

    for name in session.input_names:
        if not name.startswith("past_key_values."):
            continue
        shape = session.get_input_shape(name)
        if shape is not None and len(shape) >= 2:
            # Use declared shape; set dynamic/zero dims appropriately
            parts = name.split(".")
            layer_idx = int(parts[1]) if len(parts) >= 3 and parts[1].isdigit() else 0
            ltype = (
                layer_types[layer_idx] if layer_idx < len(layer_types) else "full_attention"
            )
            if ltype in ("linear_attention", "mamba", "mamba2"):
                # Fixed-size recurrent state: replace symbolic dims with 1
                static = [d if isinstance(d, int) and d > 0 else 1 for d in shape]
            else:
                # KV cache: use declared dims but seq=0
                static = []
                for i, d in enumerate(shape):
                    if isinstance(d, int) and d > 0:
                        static.append(d)
                    elif i == 0:
                        static.append(1)  # batch
                    elif i == 2:
                        static.append(0)  # seq dim
                    else:
                        static.append(default_kv_heads if i == 1 else default_head_dim)
            feeds[name] = np.zeros(static, dtype=session.get_input_dtype(name) or np.float32)
        else:
            # Fallback: standard shape
            feeds[name] = np.zeros(
                (1, default_kv_heads, 0, default_head_dim),
                dtype=session.get_input_dtype(name) or np.float32,
            )
    return feeds


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
    # Unresolved decoder-side parity gap exposed by a near-tie argmax flip.
    # NOT a GPU regression and NOT an encoder/fusion bug:
    #   * vision/audio encoders + projector + InputMixer fusion match HF at
    #     cos ~1.0 (feeding HF's exact inputs_embeds into the mobius decoder
    #     still flips), so the gap is decoder-side;
    #   * mobius produces IDENTICAL logits on CPU and CUDA (not an EP issue);
    #   * decoder components (LongRoPE short-path freqs, rotary_dim=96,
    #     attention_scaling=1.19, LoRA gates, GQA 24q/8kv) verified vs HF.
    # The decoder's final-position logit cosine vs HF is ~0.983 over ~3619
    # tokens. mobius ranks golden top1 (38229) as its own top2 — a clean
    # top1<->top2 swap (mobius gap 0.88 logits) of a golden 2.15-logit
    # near-tie. The passing phi4mm-multi-image case (len 3503, no audio) shows
    # the same ~0.991 cosine but survives because its golden top1/top2 gap is
    # 3.5 logits. top10_jaccard=0.33 (tail tokens at near-equal logits
    # reshuffle) misses compare_golden's AMBIGUOUS guard.
    "phi4mm-multimodal/phi4mm-multi-image-audio": (
        "Decoder-side parity gap (cos~0.983) flips a 2.15-logit near-tie "
        "(golden top1 38229 vs top2 976; mobius ranks 38229 as its top2). "
        "Identical on CPU+CUDA, encoders/fusion match HF at cos~1.0. Not a "
        "GPU regression or structural encoder bug."
    ),
}

# Failures that only apply to L5 (generation loop), not L4 (single forward).
_L5_ONLY_XFAIL_REASONS: dict[str, str] = {
    # Generation loop divergence (L4 prefill passes, but decode loop drifts)
    "text-generation/helium-1-2b": "Helium decode loop diverges from HF after first token",
    "text-generation/nanochat-d20": "NanoChat decode loop diverges from HF after first token",
    "text-generation/ernie4_5-0_3b": "ERNIE 4.5 decode loop diverges from HF after first token",
    # Nemotron-H hybrid Mamba2 SSM decode loop diverges from HF after the first
    # token. L4 prefill passes (argmax matches); identical CPU+CUDA. The golden
    # is a degenerate greedy repetition, so the token-match ratio is near zero.
    "text-generation/nemotron-h-nano-4b": (
        "Nemotron-H hybrid Mamba2 SSM decode loop diverges from HF after first "
        "token (L4 prefill passes; identical CPU+CUDA; golden is degenerate)"
    ),
    # MLA compressed KV cache dimensions not yet handled by OnnxGenerator
    "text-generation/youtu-2b": "Youtu MLA KV cache dims differ from standard attention (v_head_dim != head_dim)",
}


def _discover_cases(
    level: str,
    xfails: dict[str, str] | None = None,
) -> list[pytest.ParameterSet]:
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
        elif case.ci_skip_reason and _IN_CI:
            marks.append(pytest.mark.skip(reason=f"[CI] {case.ci_skip_reason}"))
        elif not has_golden(case):
            marks.append(
                pytest.mark.skip(reason=(f"Golden file missing: {golden_path_for_case(case)}"))
            )
        elif xfails and test_id in xfails:
            marks.append(pytest.mark.xfail(reason=xfails[test_id], strict=False))

        params.append(
            pytest.param(
                case,
                id=test_id,
                marks=marks,
            )
        )
    return params


_L4_CASES = _discover_cases("L4", xfails=_XFAIL_REASONS)
_L5_CASES = _discover_cases("L5", xfails=_L5_ONLY_XFAIL_REASONS)


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
    device_kwargs = _get_test_device_kwargs()
    if len(pkg) == 1:
        return OnnxModelSession(pkg, **device_kwargs)
    if "model" in pkg:
        return OnnxModelSession(pkg["model"], **device_kwargs)
    if "decoder" in pkg:
        return OnnxModelSession(pkg["decoder"], **device_kwargs)
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
    enc_session = OnnxModelSession(pkg["encoder"], **_get_test_device_kwargs())
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
    dec_session = OnnxModelSession(pkg["decoder"], **_get_test_device_kwargs())
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
        feeds.update(_make_empty_kv_cache(session, config))

    return feeds


# Task types that produce ``last_hidden_state`` instead of ``logits``.
_HIDDEN_STATE_TASKS: frozenset[str] = frozenset(
    {
        "feature-extraction",
        "image-classification",
        "audio-feature-extraction",
        # CTC ASR's model output is named "logits", but the saved golden's
        # top-K is over the *last frame's* vocab vector. The
        # ``_extract_logits`` path picks ``outputs["logits"]``; the test
        # comparator below slices to the last frame so the shape matches
        # the saved golden's per-token vector.
        "ctc-asr",
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
    forced_size: dict | None = None,
) -> dict[str, np.ndarray]:
    """Prepare input feeds for an image-classification forward pass.

    Loads the test image and preprocesses it with the HuggingFace
    image processor to produce ``pixel_values``.  When ``forced_size`` is
    given (object-detection), the processor is forced to that exact
    resolution to match the model's fixed-size export.
    """
    import transformers
    from PIL import Image

    processor = transformers.AutoImageProcessor.from_pretrained(
        case.model_id, trust_remote_code=case.trust_remote_code
    )
    image = Image.open(_TESTDATA_DIR / case.images[0])
    proc_kwargs: dict = {"images": image, "return_tensors": "np"}
    if forced_size is not None:
        proc_kwargs["size"] = forced_size
    processed = processor(**proc_kwargs)
    feeds: dict[str, np.ndarray] = {
        "pixel_values": processed["pixel_values"].astype(np.float32),
    }
    return feeds


def _detection_forced_size(case: GoldenTestCase) -> dict | None:
    """Return the fixed ``{height, width}`` export size for object detection.

    mobius exports object-detection models at a fixed resolution from
    ``config.image_size`` (no position-embedding interpolation), so the
    image processor must emit exactly that size.
    """
    import transformers

    config = transformers.AutoConfig.from_pretrained(
        case.model_id, trust_remote_code=case.trust_remote_code
    )
    image_size = getattr(config, "image_size", None)
    if isinstance(image_size, (list, tuple)) and len(image_size) == 2:
        return {"height": int(image_size[0]), "width": int(image_size[1])}
    if isinstance(image_size, int):
        return {"height": image_size, "width": image_size}
    return None


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


def _compute_mrope_position_ids(
    input_ids: np.ndarray,
    image_grid_thw: np.ndarray,
    spatial_merge_size: int,
    mm_token_type_ids: np.ndarray | None = None,
) -> np.ndarray:
    """Compute 3D MRoPE position IDs for Qwen VL models.

    Ports the HuggingFace ``get_rope_index`` / ``get_vision_position_ids``
    logic to numpy so that no HF model weights need to be loaded.

    For each token in the sequence:
    - Text tokens: all three dims (T, H, W) equal to the sequential position.
    - Image tokens: T=start_pos (flat), H=row within image grid, W=column.

    Args:
        input_ids: ``[batch, seq_len]`` int64 — token IDs.
        image_grid_thw: ``[num_images, 3]`` int64 — (T, H, W) grid per image
            *after* vision backbone (before spatial merge).
        spatial_merge_size: Factor by which H and W are reduced in the backbone.
        mm_token_type_ids: ``[batch, seq_len]`` int32 — 0=text, 1=image,
            2=video.  If ``None``, falls back to sequential 1D position IDs
            broadcast to shape ``[3, batch, seq_len]``.

    Returns:
        position_ids ``[3, batch, seq_len]`` int64.
    """
    batch_size, seq_len = input_ids.shape

    if mm_token_type_ids is None:
        # Fallback: plain sequential IDs replicated across all 3 dims.
        ids_1d = np.arange(seq_len, dtype=np.int64).reshape(1, 1, seq_len)
        return np.broadcast_to(ids_1d, (3, batch_size, seq_len)).copy()

    import itertools

    position_ids = np.zeros((3, batch_size, seq_len), dtype=np.int64)

    for batch_idx in range(batch_size):
        token_types = mm_token_type_ids[batch_idx]  # (seq_len,)
        image_iter = iter(image_grid_thw)
        current_pos = 0
        out_pos = np.zeros((3, seq_len), dtype=np.int64)

        # Group consecutive tokens by modality type.
        for tok_type, group in itertools.groupby(
            enumerate(token_types.tolist()), key=lambda x: x[1]
        ):
            indices = [i for i, _ in group]
            span_len = len(indices)

            if tok_type == 0:
                # Text: sequential 1D position IDs for all three dims.
                positions = np.arange(current_pos, current_pos + span_len, dtype=np.int64)
                out_pos[:, indices[0] : indices[-1] + 1] = positions[np.newaxis, :]
                current_pos += span_len

            else:
                # Image (1) or video (2): 3D vision positions.
                grid_thw = next(image_iter)
                t, h, w = int(grid_thw[0]), int(grid_thw[1]), int(grid_thw[2])
                llm_h = h // spatial_merge_size
                llm_w = w // spatial_merge_size
                llm_t = t  # temporal merge size = 1 for images

                pos_t = np.full(llm_h * llm_w * llm_t, current_pos, dtype=np.int64)
                pos_h = np.repeat(
                    np.arange(current_pos, current_pos + llm_h, dtype=np.int64),
                    llm_w * llm_t,
                )
                pos_w = np.tile(
                    np.arange(current_pos, current_pos + llm_w, dtype=np.int64),
                    llm_h * llm_t,
                )
                vision_pos = np.stack([pos_t, pos_h, pos_w], axis=0)  # (3, tokens)
                out_pos[:, indices[0] : indices[-1] + 1] = vision_pos
                current_pos += max(llm_h, llm_w)

        position_ids[:, batch_idx, :] = out_pos

    return position_ids


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

    # Build the prompt (chat template when available, else manual placeholder).
    prompt_text = _build_mm_prompt(processor, case.prompts[0], case.images, "image")

    # Use PyTorch tensors then convert — some processors don't support np
    processed_pt = processor(text=prompt_text, images=[image], return_tensors="pt")
    processed: dict[str, np.ndarray] = {
        k: v.numpy() if hasattr(v, "numpy") else np.array(v) for k, v in processed_pt.items()
    }

    # --- Step 1: Run vision encoder ---
    vis_session = OnnxModelSession(pkg["vision_encoder"], **_get_test_device_kwargs())
    try:
        vis_feeds: dict[str, np.ndarray] = {}
        for name in vis_session.input_names:
            if name in processed:
                val = processed[name]
                vis_feeds[name] = val if isinstance(val, np.ndarray) else np.array(val)
            else:
                # Handle HF↔ONNX name mismatches (e.g. HF "image_position_ids"
                # vs ONNX "pixel_position_ids").
                for hf_key, val in processed.items():
                    if hf_key.replace("image_", "pixel_") == name:
                        vis_feeds[name] = val if isinstance(val, np.ndarray) else np.array(val)
                        break
        vis_out = vis_session.run(vis_feeds)
    finally:
        vis_session.close()

    # Extract the image hidden states (first output)
    vis_hidden_key = next(iter(vis_out))
    vis_hidden = vis_out[vis_hidden_key]

    # --- Step 2: Run embedding model ---
    emb_session = OnnxModelSession(pkg["embedding"], **_get_test_device_kwargs())
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
            elif name not in emb_feeds:
                # Provide empty tensor for unused modalities (e.g. audio_features)
                shape = emb_session.get_input_shape(name) or []
                static_shape = [d if isinstance(d, int) and d > 0 else 0 for d in shape]
                emb_feeds[name] = np.zeros(static_shape, dtype=np.float32)
        emb_out = emb_session.run(emb_feeds)
    finally:
        emb_session.close()

    # Extract inputs_embeds
    emb_key = next(iter(emb_out))
    inputs_embeds = emb_out[emb_key]

    # --- Step 3: Run decoder ---
    # VL packages may use "model" or "decoder" for the text decoder.
    dec_key = "model" if "model" in pkg else "decoder"
    dec_session = OnnxModelSession(pkg[dec_key], **_get_test_device_kwargs())
    try:
        seq_len = inputs_embeds.shape[1]
        kv_cache = _make_empty_kv_cache(dec_session, config)
        embeds_dtype = dec_session.get_input_dtype("inputs_embeds")
        if embeds_dtype is not None and inputs_embeds.dtype != embeds_dtype:
            inputs_embeds = inputs_embeds.astype(embeds_dtype)
        dec_feeds: dict[str, np.ndarray] = {
            "inputs_embeds": inputs_embeds,
            **kv_cache,
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
                # Check if the decoder expects 3D MRoPE position_ids [3, batch, seq_len].
                pos_shape = dec_session.get_input_shape(name)
                if pos_shape is not None and len(pos_shape) == 3:
                    spatial_merge = getattr(config, "spatial_merge_size", 2)
                    dec_feeds[name] = _compute_mrope_position_ids(
                        processed["input_ids"].astype(np.int64),
                        processed.get("image_grid_thw"),
                        spatial_merge_size=spatial_merge,
                        mm_token_type_ids=processed.get("mm_token_type_ids"),
                    )
                else:
                    dec_feeds[name] = np.arange(seq_len, dtype=np.int64).reshape(1, -1)
            elif name in emb_out:
                # Gemma4 (when hidden_size_per_layer_input > 0) emits a second
                # embedding output ``per_layer_inputs`` that the decoder needs.
                dec_feeds[name] = emb_out[name]
        outputs = dec_session.run(dec_feeds)
    finally:
        dec_session.close()

    return outputs


def _make_vl_decoder_cache_feeds(
    dec_session: OnnxModelSession,
    config: object,
) -> dict[str, np.ndarray]:
    """Build empty past KV / recurrent state feeds for a VL decoder.

    Handles full-attention (key/value), linear-attention (conv_state/
    recurrent_state), and mamba/mamba2 (conv_state/ssm_state) layer types.
    """
    feeds: dict[str, np.ndarray] = {}
    layer_types = getattr(config, "layer_types", None) or []
    num_kv_heads = getattr(config, "num_key_value_heads", 1)
    head_dim = getattr(config, "head_dim", 64)

    for name in dec_session.input_names:
        if not name.startswith("past_key_values."):
            continue
        parts = name.split(".")
        layer_idx = int(parts[1]) if len(parts) >= 3 and parts[1].isdigit() else 0
        ltype = layer_types[layer_idx] if layer_idx < len(layer_types) else "full_attention"
        shape = dec_session.get_input_shape(name) or []

        if ltype in ("linear_attention", "mamba", "mamba2"):
            # Fixed-size recurrent state: replace symbolic/zero dims with 1
            feeds[name] = np.zeros(
                [d if isinstance(d, int) and d > 0 else 1 for d in shape], dtype=np.float32
            )
        else:
            # Standard KV cache: seq dim starts at 0 (empty)
            feeds[name] = np.zeros((1, num_kv_heads, 0, head_dim), dtype=np.float32)

    return feeds


def _update_vl_cache(
    past_cache: dict[str, np.ndarray],
    outputs: dict[str, np.ndarray],
    config: object,
) -> None:
    """Update past KV / recurrent state entries with present step outputs."""
    layer_types = getattr(config, "layer_types", None) or []
    num_hidden_layers = getattr(config, "num_hidden_layers", 0)
    for i in range(num_hidden_layers):
        ltype = layer_types[i] if i < len(layer_types) else "full_attention"
        if ltype == "linear_attention":
            suffixes = ("conv_state", "recurrent_state")
        elif ltype in ("mamba", "mamba2"):
            suffixes = ("conv_state", "ssm_state")
        else:
            suffixes = ("key", "value")
        for suffix in suffixes:
            src = f"present.{i}.{suffix}"
            dst = f"past_key_values.{i}.{suffix}"
            if src in outputs and dst in past_cache:
                past_cache[dst] = outputs[src]


def _run_vl_generation(
    pkg: ModelPackage,
    case: GoldenTestCase,
    config: object,
    max_new_tokens: int = 30,
    eos_token_id: int | None = None,
) -> np.ndarray:
    """Run greedy generation for a VL model.

    The VL decoder only accepts ``inputs_embeds`` (not raw ``input_ids``).
    Each decode step therefore re-runs the embedding model with the next
    token and empty image features to get a single-token embedding.

    Returns newly generated token IDs (prompt excluded).
    """
    import transformers
    from PIL import Image

    # --- Step 0: prepare multimodal inputs ---
    processor = transformers.AutoProcessor.from_pretrained(
        case.model_id, trust_remote_code=case.trust_remote_code
    )
    image = Image.open(_TESTDATA_DIR / case.images[0])

    prompt_text = _build_mm_prompt(processor, case.prompts[0], case.images, "image")
    suppress_ids = _load_suppress_token_ids(case.model_id, case.trust_remote_code)

    processed_pt = processor(text=prompt_text, images=[image], return_tensors="pt")
    processed: dict[str, np.ndarray] = {
        k: v.numpy() if hasattr(v, "numpy") else np.array(v) for k, v in processed_pt.items()
    }

    # --- Step 1: vision encoder ---
    vis_session = OnnxModelSession(pkg["vision_encoder"], **_get_test_device_kwargs())
    try:
        vis_feeds: dict[str, np.ndarray] = {}
        for name in vis_session.input_names:
            if name in processed:
                vis_feeds[name] = processed[name]
            else:
                # HF↔ONNX name mismatch (e.g. image_position_ids → pixel_position_ids)
                for hf_key, val in processed.items():
                    if hf_key.replace("image_", "pixel_") == name:
                        vis_feeds[name] = val if isinstance(val, np.ndarray) else np.array(val)
                        break
        vis_out = vis_session.run(vis_feeds)
    finally:
        vis_session.close()

    vis_hidden = vis_out[next(iter(vis_out))]  # image feature tensor

    # --- Step 2: embedding (prefill) ---
    # VL packages use "decoder" as the decoder key
    dec_key = "decoder" if "decoder" in pkg else "model"
    dec_session = OnnxModelSession(pkg[dec_key], **_get_test_device_kwargs())
    emb_session = OnnxModelSession(pkg["embedding"], **_get_test_device_kwargs())

    # Find the image features input name on the embedding model
    image_feat_input = next(
        (n for n in emb_session.input_names if "image" in n),
        None,
    )

    try:
        emb_feeds: dict[str, np.ndarray] = {
            "input_ids": processed["input_ids"].astype(np.int64),
        }
        if image_feat_input is not None:
            emb_feeds[image_feat_input] = vis_hidden
        # Provide empty tensors for unused modalities (e.g. audio_features)
        for name in emb_session.input_names:
            if name not in emb_feeds:
                shape = emb_session.get_input_shape(name) or []
                static_shape = [d if isinstance(d, int) and d > 0 else 0 for d in shape]
                emb_feeds[name] = np.zeros(static_shape, dtype=np.float32)
        emb_out = emb_session.run(emb_feeds)
        inputs_embeds = emb_out[next(iter(emb_out))]  # [1, seq_len, hidden_size]

        batch_size = 1
        prompt_seq_len = inputs_embeds.shape[1]
        hidden_size = inputs_embeds.shape[2]

        # --- Step 3: determine position_ids style ---
        pos_shape = dec_session.get_input_shape("position_ids")
        uses_mrope = pos_shape is not None and len(pos_shape) == 3
        spatial_merge = getattr(config, "spatial_merge_size", 2)

        # --- Step 4: prefill decoder ---
        past_cache = _make_empty_kv_cache(dec_session, config)
        embeds_dtype = dec_session.get_input_dtype("inputs_embeds")
        if embeds_dtype is not None and inputs_embeds.dtype != embeds_dtype:
            inputs_embeds = inputs_embeds.astype(embeds_dtype)
        dec_feeds: dict[str, np.ndarray] = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": np.ones((batch_size, prompt_seq_len), dtype=np.int64),
            **past_cache,
        }
        # Gemma4 decoder requires input_ids alongside inputs_embeds
        if "input_ids" in dec_session.input_names:
            dec_feeds["input_ids"] = processed["input_ids"].astype(np.int64)
        # Track the next decode position (may differ from token count for MRoPE
        # because image tokens consume fewer positions than tokens: image group
        # advances current_pos by max(H, W), not by num_image_tokens).
        next_decode_pos: int
        if "position_ids" in dec_session.input_names:
            if uses_mrope:
                prefill_pos_ids = _compute_mrope_position_ids(
                    processed["input_ids"].astype(np.int64),
                    processed.get("image_grid_thw"),
                    spatial_merge_size=spatial_merge,
                    mm_token_type_ids=processed.get("mm_token_type_ids"),
                )
                dec_feeds["position_ids"] = prefill_pos_ids
                # Next decode position = last token's position + 1.
                # For text tokens all three dims are equal; use dim 0.
                next_decode_pos = int(prefill_pos_ids[0, 0, -1]) + 1
            else:
                dec_feeds["position_ids"] = np.arange(prompt_seq_len, dtype=np.int64).reshape(
                    1, -1
                )
                next_decode_pos = prompt_seq_len
        else:
            next_decode_pos = prompt_seq_len

        # Wire extra embedding outputs the decoder expects by name
        # (e.g. Gemma4 ``per_layer_inputs``).
        for name in dec_session.input_names:
            if name not in dec_feeds and name in emb_out:
                dec_feeds[name] = emb_out[name]

        prefill_out = dec_session.run(dec_feeds)
        logits = _suppress_logits(prefill_out["logits"], suppress_ids)
        next_token = np.argmax(logits[:, -1, :], axis=-1, keepdims=True).astype(np.int64)
        _update_vl_cache(past_cache, prefill_out, config)

        generated = [next_token]
        # past_seq_len tracks total token count (for attention_mask length).
        # next_decode_pos tracks the MRoPE position for the next new token.
        past_seq_len = prompt_seq_len

        # --- Step 5: decode loop ---
        # Embed each new token through the embedding model with no image features.
        image_feat_dtype = (
            emb_session.get_input_dtype(image_feat_input)
            if image_feat_input is not None
            else None
        ) or np.float32
        empty_image = np.zeros((0, hidden_size), dtype=image_feat_dtype)
        for _ in range(max_new_tokens - 1):
            if eos_token_id is not None and np.all(next_token == eos_token_id):
                break

            # Embed single token (no new vision features during decode)
            step_emb_feeds: dict[str, np.ndarray] = {"input_ids": next_token}
            if image_feat_input is not None:
                step_emb_feeds[image_feat_input] = empty_image
            # Provide empty tensors for other modalities
            for name in emb_session.input_names:
                if name not in step_emb_feeds:
                    shape = emb_session.get_input_shape(name) or []
                    static_shape = [d if isinstance(d, int) and d > 0 else 0 for d in shape]
                    step_emb_feeds[name] = np.zeros(static_shape, dtype=np.float32)
            step_emb_out = emb_session.run(step_emb_feeds)
            step_embeds = step_emb_out[next(iter(step_emb_out))]  # [1, 1, hidden_size]
            if embeds_dtype is not None and step_embeds.dtype != embeds_dtype:
                step_embeds = step_embeds.astype(embeds_dtype)

            total_len = past_seq_len + 1
            step_feeds: dict[str, np.ndarray] = {
                "inputs_embeds": step_embeds,
                "attention_mask": np.ones((batch_size, total_len), dtype=np.int64),
                **past_cache,
            }
            # Gemma4 decoder requires input_ids alongside inputs_embeds
            if "input_ids" in dec_session.input_names:
                step_feeds["input_ids"] = next_token
            if "position_ids" in dec_session.input_names:
                if uses_mrope:
                    # Use the true MRoPE position (not the token count), since
                    # image tokens occupy fewer positions than tokens.
                    step_feeds["position_ids"] = np.full(
                        (3, batch_size, 1), next_decode_pos, dtype=np.int64
                    )
                else:
                    step_feeds["position_ids"] = np.array([[next_decode_pos]], dtype=np.int64)

            # Wire extra embedding outputs the decoder expects by name
            # (e.g. Gemma4 ``per_layer_inputs``).
            for name in dec_session.input_names:
                if name not in step_feeds and name in step_emb_out:
                    step_feeds[name] = step_emb_out[name]

            step_out = dec_session.run(step_feeds)
            logits = _suppress_logits(step_out["logits"], suppress_ids)
            next_token = np.argmax(logits[:, -1, :], axis=-1, keepdims=True).astype(np.int64)
            generated.append(next_token)
            _update_vl_cache(past_cache, step_out, config)
            past_seq_len = total_len
            next_decode_pos += 1

    finally:
        dec_session.close()
        emb_session.close()

    return np.concatenate(generated, axis=1)[0]  # [generated_len]


# ---------------------------------------------------------------------------
# Multimodal prefill helpers (speech-to-text, speech-language, text-only VL)
# ---------------------------------------------------------------------------


def _run_speech_to_text_prefill(
    pkg: ModelPackage,
    case: GoldenTestCase,
    golden: GoldenRef,
    config: object,
) -> dict[str, np.ndarray]:
    """Run encoder → decoder for speech-to-text models (e.g. Whisper).

    Unlike seq2seq text models, the encoder takes ``input_features``
    (mel spectrogram) rather than ``input_ids``.
    """
    import librosa
    import transformers

    # Load audio and extract features
    processor = transformers.AutoProcessor.from_pretrained(
        case.model_id, trust_remote_code=case.trust_remote_code
    )
    audio_path = _TESTDATA_DIR / case.audio[0]
    audio_array, _sr = librosa.load(str(audio_path), sr=16000)
    processed = processor(audio_array, sampling_rate=16000, return_tensors="np")

    # Step 1: Run encoder
    enc_session = OnnxModelSession(pkg["encoder"], **_get_test_device_kwargs())
    try:
        enc_feeds: dict[str, np.ndarray] = {}
        for name in enc_session.input_names:
            if name in processed:
                enc_feeds[name] = processed[name].astype(np.float32)
        enc_outputs = enc_session.run(enc_feeds)
    finally:
        enc_session.close()

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
    dec_session = OnnxModelSession(pkg["decoder"], **_get_test_device_kwargs())
    try:
        decoder_start_id = getattr(config, "decoder_start_token_id", 0) or 0
        dec_input_ids = np.array([[decoder_start_id]], dtype=np.int64)

        dec_feeds: dict[str, np.ndarray] = {
            "encoder_hidden_states": enc_hidden,
        }

        # Map decoder inputs by name — whisper uses "decoder_input_ids"
        for name in dec_session.input_names:
            if name in dec_feeds:
                continue
            if name in ("input_ids", "decoder_input_ids"):
                dec_feeds[name] = dec_input_ids
            elif name == "encoder_attention_mask":
                enc_seq_len = enc_hidden.shape[1]
                dec_feeds[name] = np.ones((1, enc_seq_len), dtype=np.int64)
            elif name == "position_ids":
                dec_feeds[name] = np.zeros((1, 1), dtype=np.int64)
            elif name.startswith("past_key_values."):
                num_kv_heads = getattr(config, "num_key_value_heads", None) or getattr(
                    config, "num_attention_heads", 1
                )
                head_dim = getattr(config, "head_dim", None) or (
                    getattr(config, "d_model", 256)
                    // getattr(config, "decoder_attention_heads", 1)
                )
                dec_feeds[name] = np.zeros(
                    (1, num_kv_heads, 0, head_dim),
                    dtype=np.float32,
                )
        outputs = dec_session.run(dec_feeds)
    finally:
        dec_session.close()

    return outputs


def _run_text_only_multimodel_prefill(
    pkg: ModelPackage,
    golden: GoldenRef,
    config: object,
) -> dict[str, np.ndarray]:
    """Run embedding → decoder for text-only input on a multi-model package.

    Multi-model packages (e.g. Gemma4 VL) require text to go through the
    embedding model first since the decoder only accepts ``inputs_embeds``.
    """
    device_kwargs = _get_test_device_kwargs()
    input_ids = np.array(golden.input_ids, dtype=np.int64).reshape(1, -1)

    # Step 1: Run embedding with text-only input (no image/audio features)
    emb_session = OnnxModelSession(pkg["embedding"], **device_kwargs)
    try:
        emb_feeds: dict[str, np.ndarray] = {"input_ids": input_ids}
        # Provide empty features for any non-text inputs
        for name in emb_session.input_names:
            if name not in emb_feeds:
                shape = emb_session.get_input_shape(name) or []
                static_shape = [d if isinstance(d, int) and d > 0 else 0 for d in shape]
                emb_feeds[name] = np.zeros(static_shape, dtype=np.float32)
        emb_out = emb_session.run(emb_feeds)
    finally:
        emb_session.close()

    inputs_embeds = emb_out[next(iter(emb_out))]

    # Step 2: Run decoder
    dec_key = "model" if "model" in pkg else "decoder"
    dec_session = OnnxModelSession(pkg[dec_key], **device_kwargs)
    try:
        seq_len = inputs_embeds.shape[1]
        kv_cache = _make_empty_kv_cache(dec_session, config)
        dec_feeds: dict[str, np.ndarray] = {
            "inputs_embeds": inputs_embeds,
            **kv_cache,
        }
        for name in dec_session.input_names:
            if name in dec_feeds:
                continue
            if name == "input_ids":
                dec_feeds[name] = input_ids
            elif name == "attention_mask":
                dec_feeds[name] = np.ones((1, seq_len), dtype=np.int64)
            elif name == "position_ids":
                dec_feeds[name] = np.arange(seq_len, dtype=np.int64).reshape(1, -1)
            elif name in emb_out:
                # Gemma4 (when hidden_size_per_layer_input > 0) emits a second
                # embedding output ``per_layer_inputs`` that the decoder needs.
                # Wire any extra embedding outputs through by name.
                dec_feeds[name] = emb_out[name]
        outputs = dec_session.run(dec_feeds)
    finally:
        dec_session.close()

    return outputs


def _run_phi4mm_multimodal_prefill(
    pkg: ModelPackage,
    case: GoldenTestCase,
    golden: GoldenRef,
    config: object,
) -> dict[str, np.ndarray]:
    """Run the Phi4MM 4-model pipeline: vision → speech → embedding → decoder.

    Phi4MM is a multimodal model with separate ONNX models for vision
    (SigLIP), speech (Conformer), embedding (InputMixer), and decoder.
    The pipeline chains: pixel_values → vision → image_features,
    audio → speech → audio_features, then input_ids + features → embedding
    → inputs_embeds → decoder → logits.
    """
    import transformers

    device_kwargs = _get_test_device_kwargs()
    hidden_size = getattr(config, "hidden_size", 3072)
    input_ids = np.array(golden.input_ids, dtype=np.int64).reshape(1, -1)

    # Step 1: Vision encoder (if images are provided)
    if case.images:
        from PIL import Image

        processor = transformers.AutoProcessor.from_pretrained(
            case.model_id, trust_remote_code=True
        )
        images = [Image.open(_TESTDATA_DIR / img_path) for img_path in case.images]
        img_inputs = processor.image_processor(images=images, return_tensors="np")
        # Phi4MM image_processor returns 'input_image_embeds' as pixel tensor
        pixel_values = img_inputs["input_image_embeds"].astype(np.float32)
        image_sizes = img_inputs["image_sizes"].astype(np.int64)
        # Per-crop validity mask (HD transform padding crop). Cast to the
        # model dtype so it matches the vision graph's declared input type.
        image_attention_mask = img_inputs["image_attention_mask"].astype(np.float32)

        # The vision model processes one image at a time (image_sizes is
        # [1, 2] per call).  For multi-image, loop and concatenate.
        vision_session = OnnxModelSession(pkg["vision_encoder"], **device_kwargs)
        try:
            all_features = []
            num_images = pixel_values.shape[0]
            for img_idx in range(num_images):
                # Per-image crops: [crops, C, H, W]
                per_img_pv = pixel_values[img_idx].astype(np.float32)
                per_img_sizes = image_sizes[img_idx : img_idx + 1]  # [1, 2]
                per_img_mask = image_attention_mask[img_idx]  # [crops, 32, 32]
                vision_out = vision_session.run(
                    {
                        "pixel_values": per_img_pv,
                        "image_sizes": per_img_sizes,
                        "image_attention_mask": per_img_mask,
                    }
                )
                feat = vision_out["image_features"]
                if feat.ndim == 3:
                    feat = feat[0]
                all_features.append(feat)
            image_features = np.concatenate(all_features, axis=0)
        finally:
            vision_session.close()
    else:
        image_features = np.zeros((0, hidden_size), dtype=np.float32)

    # Step 2: Speech encoder (if audio is provided)
    if case.audio:
        import librosa

        processor = transformers.AutoProcessor.from_pretrained(
            case.model_id, trust_remote_code=True
        )
        audios = []
        for audio_path in case.audio:
            audio_array, _sr = librosa.load(str(_TESTDATA_DIR / audio_path), sr=16000)
            audios.append((audio_array, 16000))

        # Process audio through the HF feature extractor
        # Phi4MM expects list of (audio_array, sample_rate) tuples
        audio_inputs = processor.audio_processor(audios, return_tensors="np")
        audio_embeds = audio_inputs["input_audio_embeds"].astype(np.float32)
        # audio_embeds: [num_clips, seq, features]
        # audio_embed_sizes: per-clip output token counts from the processor
        audio_sizes = audio_inputs["audio_embed_sizes"].astype(np.int64)
        if audio_sizes.ndim > 1:
            audio_sizes = audio_sizes.flatten()
        # audio_projection_mode: 0=speech-only, 1=combined with vision
        # When images are also present, HF uses the "vision" audio projection
        audio_projection_mode = np.array(1 if case.images else 0, dtype=np.int64)

        speech_session = OnnxModelSession(pkg["audio_encoder"], **device_kwargs)
        try:
            speech_out = speech_session.run(
                {
                    "audio_embeds": audio_embeds,
                    "audio_sizes": audio_sizes,
                    "audio_projection_mode": audio_projection_mode,
                }
            )
        finally:
            speech_session.close()
        audio_features = speech_out["audio_features"]
        # Flatten batch of clips: [num_clips, tokens, H] → [total_tokens, H]
        if audio_features.ndim == 3:
            audio_features = audio_features.reshape(-1, audio_features.shape[-1])
    else:
        audio_features = np.zeros((0, hidden_size), dtype=np.float32)

    # Step 3: Embedding (fuse text + vision + speech)
    emb_session = OnnxModelSession(pkg["embedding"], **device_kwargs)
    try:
        emb_out = emb_session.run(
            {
                "input_ids": input_ids,
                "image_features": image_features,
                "audio_features": audio_features,
            }
        )
    finally:
        emb_session.close()
    inputs_embeds = emb_out["inputs_embeds"]

    # Step 4: Decoder
    dec_session = OnnxModelSession(pkg["decoder"], **device_kwargs)
    try:
        seq_len = inputs_embeds.shape[1]
        kv_cache = _make_empty_kv_cache(dec_session, config)
        dec_feeds: dict[str, np.ndarray] = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": np.ones((1, seq_len), dtype=np.int64),
            "position_ids": np.arange(seq_len, dtype=np.int64).reshape(1, -1),
            **kv_cache,
        }
        # Wire any extra embedding outputs the decoder expects by name
        # (e.g. Gemma4 ``per_layer_inputs``).
        for name in dec_session.input_names:
            if name not in dec_feeds and name in emb_out:
                dec_feeds[name] = emb_out[name]
        outputs = dec_session.run(dec_feeds)
    finally:
        dec_session.close()

    return outputs


def _run_speech_language_prefill(
    pkg: ModelPackage,
    case: GoldenTestCase,
    golden: GoldenRef,
    config: object,
) -> dict[str, np.ndarray]:
    """Run audio encoder → embedding → decoder for speech-language models.

    The audio encoder produces audio features which are fed to the
    embedding model along with input_ids, then the decoder runs on
    inputs_embeds.
    """
    import librosa
    import transformers

    device_kwargs = _get_test_device_kwargs()

    # Load audio and extract features
    processor = transformers.AutoProcessor.from_pretrained(
        case.model_id, trust_remote_code=case.trust_remote_code
    )
    audio_path = _TESTDATA_DIR / case.audio[0]
    audio_array, _sr = librosa.load(str(audio_path), sr=16000)

    # Use the feature extractor component for audio.
    # For Qwen3-ASR, AutoProcessor returns a tokenizer (not the
    # full Qwen3ASRProcessor) because the HF repo lacks auto_map.
    # Fall back to WhisperFeatureExtractor which is what Qwen3-ASR
    # actually uses under the hood.
    fe = getattr(processor, "feature_extractor", None)
    if fe is None or not hasattr(fe, "sampling_rate"):
        fe = transformers.WhisperFeatureExtractor.from_pretrained(case.model_id)
    audio_processed = fe(
        [audio_array],
        sampling_rate=16000,
        return_tensors="np",
        padding=False,
    )

    # Step 1: Run audio encoder
    audio_session = OnnxModelSession(pkg["audio_encoder"], **device_kwargs)
    try:
        audio_feeds: dict[str, np.ndarray] = {}
        for name in audio_session.input_names:
            if name in audio_processed:
                # Cast to the session's declared dtype (e.g. input_features_mask
                # is BOOL on Gemma4 audio encoder; the HF feature extractor may
                # emit it as float or int).
                target_dtype = audio_session.get_input_dtype(name) or np.float32
                audio_feeds[name] = audio_processed[name].astype(target_dtype)
            elif name == "input_features" and "input_features" in audio_processed:
                audio_feeds[name] = audio_processed["input_features"].astype(np.float32)
        # Provide all-True mask for single-clip inference when the model
        # expects input_features_mask but the feature extractor didn't
        # produce one.
        if (
            "input_features_mask" in audio_session.input_names
            and "input_features_mask" not in audio_feeds
            and "input_features" in audio_feeds
        ):
            feats = audio_feeds["input_features"]
            mask_dtype = audio_session.get_input_dtype("input_features_mask") or np.bool_
            audio_feeds["input_features_mask"] = np.ones(feats.shape[:2], dtype=mask_dtype)
        # Qwen3-ASR requires ``feature_attention_mask`` of shape
        # ``(batch, mel_seq)`` where ``input_features`` is
        # ``(batch, n_mels, mel_seq)``.  The feature extractor is called with
        # ``padding=False`` so every frame is real -> all-ones is correct.
        if (
            "feature_attention_mask" in audio_session.input_names
            and "feature_attention_mask" not in audio_feeds
            and "input_features" in audio_feeds
        ):
            feats = audio_feeds["input_features"]
            real_len = feats.shape[2]
            # Whisper-style feature extractors pad mel frames to 3000 (30s);
            # the audio tower reshapes mel_seq into chunks of 100, so it must
            # be a multiple of 100.  Pad ``input_features`` with zeros to the
            # padded length and mark the real frames in the attention mask.
            target_len = max(3000, ((real_len + 99) // 100) * 100)
            if target_len != real_len:
                feats = np.pad(feats, ((0, 0), (0, 0), (0, target_len - real_len)))
                audio_feeds["input_features"] = feats
            mask_dtype = audio_session.get_input_dtype("feature_attention_mask") or np.int64
            mask = np.zeros((feats.shape[0], target_len), dtype=mask_dtype)
            mask[:, :real_len] = 1
            audio_feeds["feature_attention_mask"] = mask
        audio_out = audio_session.run(audio_feeds)
    finally:
        audio_session.close()

    audio_hidden = audio_out[next(iter(audio_out))]
    # Audio encoder output is [batch, seq, hidden]; embedding expects
    # [num_tokens, hidden] (no batch dim).
    if audio_hidden.ndim == 3:
        audio_hidden = audio_hidden[0]  # squeeze batch

    # Build input_ids from golden reference
    input_ids = np.array(golden.input_ids, dtype=np.int64).reshape(1, -1)

    # Adjust audio placeholder count to match encoder output.
    # The HF processor may generate a different number of audio tokens
    # than the ONNX encoder actually produces.  Re-build input_ids so
    # the placeholder count matches the encoder output exactly.
    num_encoder_tokens = audio_hidden.shape[0]
    audio_token_id = getattr(config, "audio_token_id", None)
    if audio_token_id is None:
        thinker_cfg = getattr(config, "thinker_config", None)
        if thinker_cfg is not None:
            audio_token_id = getattr(thinker_cfg, "audio_token_id", None)
    if audio_token_id is not None:
        flat = input_ids[0].tolist()
        num_placeholders = flat.count(audio_token_id)
        if num_placeholders != num_encoder_tokens:
            # Replace existing placeholders with correct count
            new_ids: list[int] = []
            replaced = False
            for tok in flat:
                if tok == audio_token_id:
                    if not replaced:
                        new_ids.extend([audio_token_id] * num_encoder_tokens)
                        replaced = True
                    # Skip remaining old placeholders
                else:
                    new_ids.append(tok)
            input_ids = np.array(new_ids, dtype=np.int64).reshape(1, -1)

    # Step 2: Run embedding with input_ids + audio features
    emb_session = OnnxModelSession(pkg["embedding"], **device_kwargs)
    try:
        emb_feeds: dict[str, np.ndarray] = {"input_ids": input_ids}
        for name in emb_session.input_names:
            if name in emb_feeds:
                continue
            if name == "audio_features":
                emb_feeds[name] = audio_hidden
            elif name in audio_out:
                emb_feeds[name] = audio_out[name]
            else:
                # Empty features for unused modalities (e.g. image)
                shape = emb_session.get_input_shape(name) or []
                static_shape = [d if isinstance(d, int) and d > 0 else 0 for d in shape]
                emb_feeds[name] = np.zeros(static_shape, dtype=np.float32)
        emb_out = emb_session.run(emb_feeds)
    finally:
        emb_session.close()

    inputs_embeds = emb_out[next(iter(emb_out))]

    # Step 3: Run decoder
    dec_key = "model" if "model" in pkg else "decoder"
    dec_session = OnnxModelSession(pkg[dec_key], **device_kwargs)
    try:
        seq_len = inputs_embeds.shape[1]
        kv_cache = _make_empty_kv_cache(dec_session, config)
        dec_feeds: dict[str, np.ndarray] = {
            "inputs_embeds": inputs_embeds,
            **kv_cache,
        }
        for name in dec_session.input_names:
            if name in dec_feeds:
                continue
            if name == "input_ids":
                dec_feeds[name] = input_ids
            elif name == "attention_mask":
                dec_feeds[name] = np.ones((1, seq_len), dtype=np.int64)
            elif name == "position_ids":
                pos = np.arange(seq_len, dtype=np.int64).reshape(1, -1)
                # MRoPE models expect 3D position_ids: (dims, batch, seq)
                pos_shape = dec_session.get_input_shape(name)
                if pos_shape and len(pos_shape) == 3:
                    ndims = pos_shape[0] if isinstance(pos_shape[0], int) else 3
                    pos = np.tile(pos, (ndims, 1, 1))
                dec_feeds[name] = pos
            elif name in emb_out:
                # Extra embedding outputs the decoder expects by name
                # (e.g. Gemma4 ``per_layer_inputs``).
                dec_feeds[name] = emb_out[name]
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
        elif case.task_type == "speech-to-text":
            outputs = _run_speech_to_text_prefill(pkg, case, golden, config)
        elif case.task_type == "speech-language":
            outputs = _run_speech_language_prefill(pkg, case, golden, config)
        elif case.task_type == "image-text-to-text":
            outputs = _run_vision_language_prefill(pkg, case, config)
        elif case.task_type == "phi4mm-multimodal":
            outputs = _run_phi4mm_multimodal_prefill(pkg, case, golden, config)
        elif case.task_type == "image-classification":
            session = _open_decoder_session(pkg)
            try:
                feeds = _prepare_vision_feeds(case)
                outputs = session.run(feeds)
            finally:
                session.close()
        elif case.task_type == "object-detection":
            # Object detection (e.g. YOLOS): pixel_values → class logits +
            # predicted boxes. The golden top-K is over the last query's
            # class ``logits``. The processor is forced to the model's fixed
            # export resolution (no position-embedding interpolation).
            session = _open_decoder_session(pkg)
            try:
                feeds = _prepare_vision_feeds(case, forced_size=_detection_forced_size(case))
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
        elif case.task_type == "ctc-asr":
            # CTC ASR: input_values + attention_mask → logits per frame.
            # The ONNX graph requires an explicit attention_mask (an
            # all-ones mask when the input has no padding); the existing
            # _prepare_audio_feeds only emits input_values, so we add the
            # mask here from the same audio length used to build feeds.
            session = _open_decoder_session(pkg)
            try:
                feeds = _prepare_audio_feeds(case)
                feeds["attention_mask"] = np.ones_like(feeds["input_values"], dtype=np.int64)
                outputs = session.run(feeds)
            finally:
                session.close()
        elif len(pkg) > 1 and "embedding" in pkg:
            # Multi-model text-generation (e.g. Gemma4 VL text-only)
            outputs = _run_text_only_multimodel_prefill(pkg, golden, config)
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

# Task types that support autoregressive generation.
_GENERATION_SUPPORTED_TASKS = frozenset(
    {
        "text-generation",
        "image-text-to-text",
        "seq2seq",
        "speech-to-text",
        "speech-language",
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


def _run_multimodel_text_generation(
    pkg: ModelPackage,
    case: GoldenTestCase,
    config: object,
    golden: GoldenRef,
    max_new_tokens: int = 20,
    eos_token_id: int | None = None,
) -> np.ndarray:
    """Greedy generation for multi-model text-generation packages.

    Some text-only models (e.g. Gemma4 "any-to-any") split into separate
    ``embedding`` and ``decoder`` ONNX models even on the text path: the
    embedding model maps ``input_ids`` to ``inputs_embeds`` (plus extra
    decoder inputs such as Gemma4 ``per_layer_inputs``) and the decoder
    consumes embeddings rather than raw ``input_ids``.

    This mirrors :func:`_run_vl_generation` without any vision/audio
    encoder, using 1D position IDs.  Returns newly generated token IDs
    (prompt excluded).
    """
    suppress_ids = _load_suppress_token_ids(case.model_id, case.trust_remote_code)
    device_kwargs = _get_test_device_kwargs()

    dec_key = "decoder" if "decoder" in pkg else "model"
    dec_session = OnnxModelSession(pkg[dec_key], **device_kwargs)
    emb_session = OnnxModelSession(pkg["embedding"], **device_kwargs)

    input_ids = np.array(golden.input_ids, dtype=np.int64).reshape(1, -1)
    batch_size = 1

    def _embed(ids: np.ndarray) -> dict[str, np.ndarray]:
        """Run the embedding model for ``ids``, zero-filling extra inputs."""
        feeds: dict[str, np.ndarray] = {"input_ids": ids}
        for name in emb_session.input_names:
            if name in feeds:
                continue
            shape = emb_session.get_input_shape(name) or []
            static_shape = [d if isinstance(d, int) and d > 0 else 0 for d in shape]
            feeds[name] = np.zeros(
                static_shape, dtype=emb_session.get_input_dtype(name) or np.float32
            )
        return emb_session.run(feeds)

    try:
        embeds_dtype = dec_session.get_input_dtype("inputs_embeds")

        def _decoder_feeds(
            emb_out: dict[str, np.ndarray],
            ids: np.ndarray,
            total_len: int,
            position_ids: np.ndarray,
            past_cache: dict[str, np.ndarray],
        ) -> dict[str, np.ndarray]:
            inputs_embeds = emb_out[next(iter(emb_out))]
            if embeds_dtype is not None and inputs_embeds.dtype != embeds_dtype:
                inputs_embeds = inputs_embeds.astype(embeds_dtype)
            feeds: dict[str, np.ndarray] = {
                "inputs_embeds": inputs_embeds,
                "attention_mask": np.ones((batch_size, total_len), dtype=np.int64),
                **past_cache,
            }
            if "input_ids" in dec_session.input_names:
                feeds["input_ids"] = ids
            if "position_ids" in dec_session.input_names:
                feeds["position_ids"] = position_ids
            # Wire extra embedding outputs the decoder expects by name
            # (e.g. Gemma4 ``per_layer_inputs``).
            for name in dec_session.input_names:
                if name not in feeds and name in emb_out:
                    feeds[name] = emb_out[name]
            return feeds

        # --- Prefill ---
        prompt_seq_len = input_ids.shape[1]
        past_cache = _make_empty_kv_cache(dec_session, config)
        prefill_emb = _embed(input_ids)
        prefill_pos = np.arange(prompt_seq_len, dtype=np.int64).reshape(1, -1)
        prefill_out = dec_session.run(
            _decoder_feeds(prefill_emb, input_ids, prompt_seq_len, prefill_pos, past_cache)
        )
        logits = _suppress_logits(prefill_out["logits"], suppress_ids)
        next_token = np.argmax(logits[:, -1, :], axis=-1, keepdims=True).astype(np.int64)
        _update_vl_cache(past_cache, prefill_out, config)

        generated = [next_token]
        past_seq_len = prompt_seq_len
        next_pos = prompt_seq_len

        # --- Decode loop ---
        for _ in range(max_new_tokens - 1):
            if eos_token_id is not None and np.all(next_token == eos_token_id):
                break
            step_emb = _embed(next_token)
            total_len = past_seq_len + 1
            step_pos = np.array([[next_pos]], dtype=np.int64)
            step_out = dec_session.run(
                _decoder_feeds(step_emb, next_token, total_len, step_pos, past_cache)
            )
            logits = _suppress_logits(step_out["logits"], suppress_ids)
            next_token = np.argmax(logits[:, -1, :], axis=-1, keepdims=True).astype(np.int64)
            generated.append(next_token)
            _update_vl_cache(past_cache, step_out, config)
            past_seq_len = total_len
            next_pos += 1
    finally:
        dec_session.close()
        emb_session.close()

    return np.concatenate(generated, axis=1)[0]


def _run_seq2seq_generation(
    pkg: ModelPackage,
    case: GoldenTestCase,
    golden: GoldenRef,
    expected_token_ids: list[int] | None = None,
) -> np.ndarray:
    """Run greedy generation for a seq2seq (encoder-decoder) model.

    Returns the full decoder output including decoder_start_token,
    matching HuggingFace model.generate() output format.
    """
    config = pkg.config
    device_kwargs = _get_test_device_kwargs()
    enc_session = OnnxModelSession(pkg["encoder"], **device_kwargs)
    dec_session = OnnxModelSession(pkg["decoder"], **device_kwargs)
    try:
        generator = OnnxSeq2SeqGenerator(enc_session, dec_session, config)
        input_ids = np.array(golden.input_ids, dtype=np.int64).reshape(1, -1)
        max_new_tokens = case.generation_params.get("max_new_tokens", 20)
        eos_token_id = case.generation_params.get("eos_token_id", None)

        # Extract decoder_start_token_id: prefer config, fall back to
        # the first token of the golden generation sequence.
        decoder_start_id = getattr(config, "decoder_start_token_id", None)
        if decoder_start_id is None and expected_token_ids:
            decoder_start_id = expected_token_ids[0]
        if decoder_start_id is None:
            decoder_start_id = 0

        all_ids = generator.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
            decoder_start_token_id=decoder_start_id,
        )
    finally:
        enc_session.close()
        dec_session.close()

    # Return the full decoder output including decoder_start_token,
    # matching HuggingFace model.generate() output format.
    return all_ids[0]


def _run_speech_to_text_generation(
    pkg: ModelPackage,
    case: GoldenTestCase,
    golden: GoldenRef,
) -> np.ndarray:
    """Run greedy generation for a speech-to-text (Whisper) model.

    Returns only the "real" generated tokens (after the forced decoder
    prefix), matching HuggingFace ``model.generate()`` output which
    strips the forced prefix tokens.
    """
    import librosa
    import transformers

    from mobius._testing.generation import OnnxSpeechToTextGenerator

    config = pkg.config
    device_kwargs = _get_test_device_kwargs()

    # Load audio and extract features (same as L4 prefill)
    processor = transformers.AutoProcessor.from_pretrained(
        case.model_id,
        trust_remote_code=case.trust_remote_code,
    )
    audio_path = _TESTDATA_DIR / case.audio[0]
    audio_array, _sr = librosa.load(str(audio_path), sr=16000)
    processed = processor(
        audio_array,
        sampling_rate=16000,
        return_tensors="np",
    )

    # Step 1: Run encoder
    enc_session = OnnxModelSession(pkg["encoder"], **device_kwargs)
    try:
        enc_feeds: dict[str, np.ndarray] = {}
        for name in enc_session.input_names:
            if name in processed:
                enc_feeds[name] = processed[name].astype(np.float32)
        enc_outputs = enc_session.run(enc_feeds)
    finally:
        enc_session.close()

    enc_hidden = None
    for key in ("encoder_hidden_states", "last_hidden_state"):
        if key in enc_outputs:
            enc_hidden = enc_outputs[key]
            break
    if enc_hidden is None:
        raise KeyError(
            f"Encoder output missing hidden states. Keys: {sorted(enc_outputs.keys())}"
        )

    # Step 2: Run decoder generation
    dec_session = OnnxModelSession(pkg["decoder"], **device_kwargs)
    try:
        decoder_start_id = getattr(config, "decoder_start_token_id", 0) or 0
        max_new_tokens = case.generation_params.get("max_new_tokens", 50)
        eos_token_id = case.generation_params.get("eos_token_id", None)

        generator = OnnxSpeechToTextGenerator(dec_session, config)
        all_ids = generator.generate(
            enc_hidden,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
            decoder_start_token_id=decoder_start_id,
        )
    finally:
        dec_session.close()

    # Strip forced decoder prefix.  HF model.generate() returns only
    # the "real" generated tokens — it internally handles forced decoder
    # IDs (language, task, notimestamps) and strips them from output.
    # We align by finding where the expected content starts in our
    # greedy output, using the golden's first token as anchor.
    output = all_ids[0]  # drop batch dim

    # Load the expected tokens to find the prefix boundary.
    expected = load_generation_golden(case)
    if expected and len(output) > 0:
        first_expected = expected[0]
        # Find the first occurrence of the expected first token
        prefix_len = 0
        for i, tok in enumerate(output):
            if tok == first_expected:
                prefix_len = i
                break

        output = output[prefix_len:]

    # Strip trailing EOS tokens that HF generation suppresses.
    eos_id = case.generation_params.get("eos_token_id", None)
    if eos_id is not None:
        while len(output) > 0 and output[-1] == eos_id:
            output = output[:-1]

    return output


def _run_speech_language_generation(
    pkg: ModelPackage,
    case: GoldenTestCase,
    config: object,
    golden: GoldenRef,
    max_new_tokens: int = 50,
) -> np.ndarray:
    """Run greedy generation for a speech-language (3-model) pipeline.

    Pipeline: audio_encoder → embedding → decoder (autoregressive).
    Uses the same embed→decode loop as VL generation but with audio
    features instead of image features.

    Returns newly generated token IDs (prompt excluded).
    """
    import librosa
    import transformers

    device_kwargs = _get_test_device_kwargs()

    # --- Load audio and extract features ---
    processor = transformers.AutoProcessor.from_pretrained(
        case.model_id, trust_remote_code=case.trust_remote_code
    )
    audio_path = _TESTDATA_DIR / case.audio[0]
    audio_array, _sr = librosa.load(str(audio_path), sr=16000)

    fe = getattr(processor, "feature_extractor", None)
    if fe is None or not hasattr(fe, "sampling_rate"):
        fe = transformers.WhisperFeatureExtractor.from_pretrained(case.model_id)
    audio_processed = fe(
        [audio_array],
        sampling_rate=16000,
        return_tensors="np",
        padding=False,
    )

    # --- Step 1: audio encoder ---
    suppress_ids = _load_suppress_token_ids(case.model_id, case.trust_remote_code)
    audio_session = OnnxModelSession(pkg["audio_encoder"], **device_kwargs)
    try:
        audio_feeds: dict[str, np.ndarray] = {}
        for name in audio_session.input_names:
            if name in audio_processed:
                target_dtype = audio_session.get_input_dtype(name) or np.float32
                audio_feeds[name] = audio_processed[name].astype(target_dtype)
            elif name == "input_features" and "input_features" in audio_processed:
                audio_feeds[name] = audio_processed["input_features"].astype(np.float32)
        # Provide all-True mask for single-clip inference when the model
        # expects input_features_mask but the feature extractor didn't
        # produce one.
        if (
            "input_features_mask" in audio_session.input_names
            and "input_features_mask" not in audio_feeds
            and "input_features" in audio_feeds
        ):
            feats = audio_feeds["input_features"]
            mask_dtype = audio_session.get_input_dtype("input_features_mask") or np.bool_
            audio_feeds["input_features_mask"] = np.ones(feats.shape[:2], dtype=mask_dtype)
        # Qwen3-ASR requires ``feature_attention_mask`` of shape
        # ``(batch, mel_seq)`` where ``input_features`` is
        # ``(batch, n_mels, mel_seq)``.  The feature extractor is called with
        # ``padding=False`` so every frame is real -> all-ones is correct.
        if (
            "feature_attention_mask" in audio_session.input_names
            and "feature_attention_mask" not in audio_feeds
            and "input_features" in audio_feeds
        ):
            feats = audio_feeds["input_features"]
            real_len = feats.shape[2]
            # Whisper-style feature extractors pad mel frames to 3000 (30s);
            # the audio tower reshapes mel_seq into chunks of 100, so it must
            # be a multiple of 100.  Pad ``input_features`` with zeros to the
            # padded length and mark the real frames in the attention mask.
            target_len = max(3000, ((real_len + 99) // 100) * 100)
            if target_len != real_len:
                feats = np.pad(feats, ((0, 0), (0, 0), (0, target_len - real_len)))
                audio_feeds["input_features"] = feats
            mask_dtype = audio_session.get_input_dtype("feature_attention_mask") or np.int64
            mask = np.zeros((feats.shape[0], target_len), dtype=mask_dtype)
            mask[:, :real_len] = 1
            audio_feeds["feature_attention_mask"] = mask
        audio_out = audio_session.run(audio_feeds)
    finally:
        audio_session.close()

    audio_hidden = audio_out[next(iter(audio_out))]
    if audio_hidden.ndim == 3:
        audio_hidden = audio_hidden[0]  # squeeze batch → [seq, hidden]

    # --- Build input_ids from golden reference ---
    input_ids = np.array(golden.input_ids, dtype=np.int64).reshape(1, -1)

    # Adjust audio placeholder count to match encoder output
    num_encoder_tokens = audio_hidden.shape[0]
    audio_token_id = getattr(config, "audio_token_id", None)
    if audio_token_id is None:
        thinker_cfg = getattr(config, "thinker_config", None)
        if thinker_cfg is not None:
            audio_token_id = getattr(thinker_cfg, "audio_token_id", None)
    if audio_token_id is not None:
        flat = input_ids[0].tolist()
        num_placeholders = flat.count(audio_token_id)
        if num_placeholders != num_encoder_tokens:
            new_ids: list[int] = []
            replaced = False
            for tok in flat:
                if tok == audio_token_id:
                    if not replaced:
                        new_ids.extend([audio_token_id] * num_encoder_tokens)
                        replaced = True
                else:
                    new_ids.append(tok)
            input_ids = np.array(new_ids, dtype=np.int64).reshape(1, -1)

    # --- Step 2: embedding (prefill) ---
    dec_key = "model" if "model" in pkg else "decoder"
    dec_session = OnnxModelSession(pkg[dec_key], **device_kwargs)
    emb_session = OnnxModelSession(pkg["embedding"], **device_kwargs)

    # Find the audio features input name on the embedding model
    audio_feat_input = next(
        (n for n in emb_session.input_names if "audio" in n),
        None,
    )

    try:
        emb_feeds: dict[str, np.ndarray] = {
            "input_ids": input_ids,
        }
        if audio_feat_input is not None:
            emb_feeds[audio_feat_input] = audio_hidden
        for name in emb_session.input_names:
            if name not in emb_feeds:
                shape = emb_session.get_input_shape(name) or []
                static_shape = [d if isinstance(d, int) and d > 0 else 0 for d in shape]
                emb_feeds[name] = np.zeros(static_shape, dtype=np.float32)
        emb_out = emb_session.run(emb_feeds)
        inputs_embeds = emb_out[next(iter(emb_out))]

        batch_size = 1
        prompt_seq_len = inputs_embeds.shape[1]
        hidden_size = inputs_embeds.shape[2]

        # --- Step 3: prefill decoder ---
        past_cache = _make_empty_kv_cache(dec_session, config)
        dec_feeds: dict[str, np.ndarray] = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": np.ones((batch_size, prompt_seq_len), dtype=np.int64),
            **past_cache,
        }
        if "input_ids" in dec_session.input_names:
            dec_feeds["input_ids"] = input_ids

        # Detect 3D position_ids (e.g. Qwen3-ASR uses MRoPE-style)
        uses_3d_pos = False
        ndims_pos = 3
        if "position_ids" in dec_session.input_names:
            pos = np.arange(prompt_seq_len, dtype=np.int64).reshape(1, -1)
            pos_shape = dec_session.get_input_shape("position_ids")
            if pos_shape and len(pos_shape) == 3:
                uses_3d_pos = True
                ndims_pos = pos_shape[0] if isinstance(pos_shape[0], int) else 3
                pos = np.tile(pos, (ndims_pos, 1, 1))
            dec_feeds["position_ids"] = pos

        # Wire extra embedding outputs the decoder expects by name
        # (e.g. Gemma4 ``per_layer_inputs``).
        for name in dec_session.input_names:
            if name not in dec_feeds and name in emb_out:
                dec_feeds[name] = emb_out[name]

        prefill_out = dec_session.run(dec_feeds)
        logits = _suppress_logits(prefill_out["logits"], suppress_ids)
        next_token = np.argmax(logits[:, -1, :], axis=-1, keepdims=True).astype(np.int64)
        _update_vl_cache(past_cache, prefill_out, config)

        generated = [next_token]
        past_seq_len = prompt_seq_len

        # --- Step 4: decode loop ---
        empty_audio = np.zeros((0, hidden_size), dtype=np.float32)
        eos_token_id = case.generation_params.get("eos_token_id")
        for _ in range(max_new_tokens - 1):
            if eos_token_id is not None and np.all(next_token == eos_token_id):
                break

            # Embed single token (no new audio features during decode)
            step_emb_feeds: dict[str, np.ndarray] = {"input_ids": next_token}
            if audio_feat_input is not None:
                step_emb_feeds[audio_feat_input] = empty_audio
            for name in emb_session.input_names:
                if name not in step_emb_feeds:
                    shape = emb_session.get_input_shape(name) or []
                    static_shape = [d if isinstance(d, int) and d > 0 else 0 for d in shape]
                    step_emb_feeds[name] = np.zeros(static_shape, dtype=np.float32)
            step_emb_out = emb_session.run(step_emb_feeds)
            step_embeds = step_emb_out[next(iter(step_emb_out))]

            total_len = past_seq_len + 1
            step_feeds: dict[str, np.ndarray] = {
                "inputs_embeds": step_embeds,
                "attention_mask": np.ones((batch_size, total_len), dtype=np.int64),
                **past_cache,
            }
            if "input_ids" in dec_session.input_names:
                step_feeds["input_ids"] = next_token
            if "position_ids" in dec_session.input_names:
                if uses_3d_pos:
                    step_feeds["position_ids"] = np.full(
                        (ndims_pos, batch_size, 1), past_seq_len, dtype=np.int64
                    )
                else:
                    step_feeds["position_ids"] = np.array([[past_seq_len]], dtype=np.int64)

            # Wire extra embedding outputs the decoder expects by name
            # (e.g. Gemma4 ``per_layer_inputs``).
            for name in dec_session.input_names:
                if name not in step_feeds and name in step_emb_out:
                    step_feeds[name] = step_emb_out[name]

            step_out = dec_session.run(step_feeds)
            logits = _suppress_logits(step_out["logits"], suppress_ids)
            next_token = np.argmax(logits[:, -1, :], axis=-1, keepdims=True).astype(np.int64)
            generated.append(next_token)
            _update_vl_cache(past_cache, step_out, config)
            past_seq_len = total_len

    finally:
        dec_session.close()
        emb_session.close()

    return np.concatenate(generated, axis=1)[0]  # [generated_len]


# ---------------------------------------------------------------------------
# L5 Tests: Generation E2E
# ---------------------------------------------------------------------------


@pytest.mark.generation
@pytest.mark.integration
class TestL5GenerationE2E:
    """L5: Full autoregressive generation, compare token sequences.

    Gate: token match ratio >= ``min_token_match_ratio`` from tolerances.
    Supported task types: ``text-generation`` (causal-LM via OnnxGenerator),
    ``image-text-to-text`` (VL three-model pipeline), ``seq2seq`` and
    ``speech-to-text`` (encoder-decoder via OnnxSeq2SeqGenerator).
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
        # For causal-lm: L4 golden (main JSON) provides input_ids.
        # For VL: image is re-processed from the YAML case at generation time.
        golden_path = golden_path_for_case(case)
        golden = load_golden_ref(golden_path)
        if golden is None:
            pytest.skip(f"Golden file missing: {golden_path}")

        # L5 generation golden is stored in the separate *_generation.json file.
        gen_path = generation_json_path_for_case(case)
        expected_token_ids = load_generation_golden(case)
        if expected_token_ids is None:
            pytest.skip(f"Generation golden file missing: {gen_path}")

        tolerances = load_tolerances("L5", case.dtype)
        # Per-case tolerance override (e.g. VL multi-model pipeline has known
        # float32 precision divergence vs HF after several decode steps).
        if case.min_token_match_ratio is not None:
            tolerances = dataclasses.replace(
                tolerances, min_token_match_ratio=case.min_token_match_ratio
            )

        # --- Build and generate ---
        pkg = _build_model_package(case)
        config = pkg.config
        assert config is not None, (
            f"ModelPackage for {case.model_id} has no config; "
            "cannot determine KV cache dimensions for generation"
        )

        if case.task_type == "image-text-to-text":
            new_tokens = _run_vl_generation(
                pkg,
                case,
                config,
                max_new_tokens=case.generation_params.get("max_new_tokens", 30),
                eos_token_id=case.generation_params.get("eos_token_id"),
            )
        elif case.task_type == "seq2seq":
            new_tokens = _run_seq2seq_generation(pkg, case, golden, expected_token_ids)
        elif case.task_type == "speech-to-text":
            new_tokens = _run_speech_to_text_generation(pkg, case, golden)
        elif case.task_type == "speech-language":
            new_tokens = _run_speech_language_generation(
                pkg,
                case,
                config,
                golden,
                max_new_tokens=case.generation_params.get("max_new_tokens", 50),
            )
        elif len(pkg) > 1 and "embedding" in pkg:
            # Multi-model text-generation (e.g. Gemma4 any-to-any text path):
            # embedding model maps input_ids -> inputs_embeds (+ extra decoder
            # inputs), and the decoder consumes embeddings.
            new_tokens = _run_multimodel_text_generation(
                pkg,
                case,
                config,
                golden,
                max_new_tokens=case.generation_params.get("max_new_tokens", 20),
                eos_token_id=case.generation_params.get("eos_token_id"),
            )
        else:
            new_tokens = _run_causal_lm_generation(pkg, case, golden)

        # --- Diagnostics ---
        expected_tokens = np.array(expected_token_ids, dtype=np.int64)
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
