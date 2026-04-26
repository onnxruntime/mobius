#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

r"""Generate golden reference files for L4/L5 testing.

Reads YAML test case definitions from ``testdata/cases/`` and generates
``.json`` golden reference files in ``testdata/golden/`` by running
HuggingFace inference.

Requires: ``pip install transformers torch accelerate``
GPU recommended for models > 1B parameters.

Usage::

    # Generate golden files for ALL test cases
    python scripts/generate_golden.py

    # Generate for a specific task type
    python scripts/generate_golden.py --task-type causal-lm

    # Generate for a single test case
    python scripts/generate_golden.py --case testdata/cases/causal-lm/gpt2.yaml

    # Regenerate all (overwrite existing)
    python scripts/generate_golden.py --force

    # Use GPU for large models
    python scripts/generate_golden.py --device cuda

    # Dry run (show what would be generated)
    python scripts/generate_golden.py --dry-run

    # Filter by glob pattern on case_id
    python scripts/generate_golden.py --filter "qwen*"
"""

from __future__ import annotations

import argparse
import fnmatch
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mobius._testing.golden import GoldenTestCase as TestCase

import numpy as np


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=("Generate golden reference .json files for L4/L5 testing"),
    )
    parser.add_argument(
        "--case",
        type=Path,
        default=None,
        help="Path to a single YAML test case. If omitted, processes all.",
    )
    parser.add_argument(
        "--task-type",
        type=str,
        default=None,
        help=(
            "Only generate for this task type subdirectory "
            "(causal-lm, encoder, seq2seq, vision-language, audio, "
            "diffusion)."
        ),
    )
    parser.add_argument(
        "--level",
        type=str,
        default=None,
        choices=["L4", "L5"],
        help="Only generate cases that include this level.",
    )
    parser.add_argument(
        "--filter",
        type=str,
        default=None,
        help="Glob pattern to filter case_ids (e.g. 'qwen*', '*7b*').",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device: 'cpu', 'cuda', 'cuda:0', etc.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing golden files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be generated without running inference.",
    )
    parser.add_argument(
        "--cases-dir",
        type=Path,
        default=Path("testdata/cases"),
        help="Root directory for YAML test case files.",
    )
    parser.add_argument(
        "--golden-dir",
        type=Path,
        default=Path("testdata/golden"),
        help="Root directory for golden .json output files.",
    )
    return parser.parse_args()


# ---- Logit extraction helpers ----
# Shared by all generators that produce logit-based golden data.


def _extract_logits_golden(
    last_logits: np.ndarray,
) -> dict[str, np.ndarray]:
    """Extract top-k IDs, logits, and summary from a logit vector.

    Args:
        last_logits: 1-D float array of shape ``(vocab_size,)``.

    Returns:
        Dict with keys ready for ``save_golden_ref()``.
    """
    last_logits_f64 = last_logits.astype(np.float64)
    sorted_indices = np.argsort(last_logits_f64)[::-1]
    top10_ids = sorted_indices[:10].tolist()
    top10_logits = last_logits_f64[sorted_indices[:10]].tolist()
    logits_summary = np.array(
        [
            float(np.max(last_logits_f64)),
            float(np.min(last_logits_f64)),
            float(np.mean(last_logits_f64)),
            float(np.std(last_logits_f64)),
        ],
        dtype=np.float64,
    )
    return {
        "top1_id": top10_ids[0],
        "top2_id": top10_ids[1] if len(top10_ids) > 1 else top10_ids[0],
        "top10_ids": top10_ids,
        "top10_logits": top10_logits,
        "logits_summary": logits_summary,
    }


# ---- Compat patches ----


def _apply_nemotron_h_generate_patch(model: object) -> None:
    """Patch NemotronH prepare_inputs_for_generation for transformers 5.x.

    The HF remote code accesses ``cache_position[-1]`` without checking
    for ``None``, which crashes under transformers >=5.x where
    ``cache_position`` is no longer passed on the first prefill call.
    We wrap the method to supply a default ``cache_position`` when missing.
    """
    cls_name = type(model).__name__
    if "NemotronH" not in cls_name:
        return

    import torch

    original_prepare = model.prepare_inputs_for_generation

    def _patched_prepare(input_ids, **kwargs):
        if kwargs.get("past_key_values") is not None and kwargs.get("cache_position") is None:
            seq_len = input_ids.shape[-1]
            kwargs["cache_position"] = torch.arange(seq_len, device=input_ids.device)
        return original_prepare(input_ids, **kwargs)

    model.prepare_inputs_for_generation = _patched_prepare


# ---- Task-specific generators ----
# Each generator loads a HF model, runs inference, and calls
# save_golden_ref() from golden.py.  Heavy imports (torch,
# transformers) are deferred to avoid import cost when --dry-run.


def _generate_causal_lm(case: TestCase, json_path: Path, device: str) -> None:
    """Generate golden data for a causal-lm (text-generation) model."""
    from mobius._testing.golden import save_generation_json, save_golden_ref
    from mobius._testing.torch_reference import (
        load_torch_model,
        torch_forward,
    )

    model, tokenizer = load_torch_model(case.model_id, device=device)

    encoded = tokenizer(case.prompts[0], return_tensors="np", padding=False)
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    seq_len = input_ids.shape[1]
    position_ids = np.arange(seq_len).reshape(1, -1)

    # L4: single forward pass → last-token logits
    logits, _ = torch_forward(model, input_ids, attention_mask, position_ids)
    last_logits = logits[0, -1, :]  # (vocab_size,)
    golden = _extract_logits_golden(last_logits)

    # L5: greedy generation
    generated_ids = None
    if "L5" in case.level:
        import torch

        _apply_nemotron_h_generate_patch(model)

        with torch.no_grad():
            gen_output = model.generate(
                torch.from_numpy(input_ids).to(device),
                max_new_tokens=case.generation_params.get("max_new_tokens", 20),
                do_sample=False,
            )
        generated_ids = gen_output[0, seq_len:].cpu().numpy()

    save_golden_ref(
        json_path,
        top1_id=golden["top1_id"],
        top2_id=golden["top2_id"],
        top10_ids=golden["top10_ids"],
        top10_logits=golden["top10_logits"],
        logits_summary=golden["logits_summary"],
        input_ids=input_ids,
    )

    # Save a separate *_generation.json marker for L5 dashboard detection.
    if generated_ids is not None:
        generated_text = tokenizer.decode(generated_ids.tolist(), skip_special_tokens=True)
        gen_path = json_path.with_name(json_path.stem + "_generation.json")
        save_generation_json(
            gen_path,
            model_id=case.model_id,
            prompt=case.prompts[0],
            generated_tokens=generated_ids.tolist(),
            generated_text=generated_text,
        )


def _generate_encoder(case: TestCase, json_path: Path, device: str) -> None:
    """Generate golden data for an encoder-only (feature-extraction) model.

    Encoder models produce ``last_hidden_state`` instead of logits.
    We treat the last token's hidden state as the "logit" vector for
    top-k extraction — this gives us a meaningful argmax gate for L4.
    """
    from mobius._testing.golden import save_golden_ref
    from mobius._testing.torch_reference import (
        load_torch_encoder_model,
        torch_encoder_forward,
    )

    model, tokenizer = load_torch_encoder_model(case.model_id, device=device)

    # CLIP-like multimodal models wrap a text sub-model that can be
    # called with text-only inputs (pixel_values not required).
    if hasattr(model, "text_model"):
        model = model.text_model

    # X-MOD requires setting a default language before inference.
    if hasattr(model, "set_default_language"):
        model.set_default_language("en_XX")

    encoded = tokenizer(case.prompts[0], return_tensors="np", padding=False)
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    token_type_ids = encoded.get("token_type_ids")

    # Forward pass → last_hidden_state
    hidden_states = torch_encoder_forward(model, input_ids, attention_mask, token_type_ids)
    # Use the last token's hidden state as the "logit" vector
    last_hidden = hidden_states[0, -1, :]  # (hidden_size,)
    golden = _extract_logits_golden(last_hidden)

    save_golden_ref(
        json_path,
        top1_id=golden["top1_id"],
        top2_id=golden["top2_id"],
        top10_ids=golden["top10_ids"],
        top10_logits=golden["top10_logits"],
        logits_summary=golden["logits_summary"],
        input_ids=input_ids,
    )


def _generate_seq2seq(case: TestCase, json_path: Path, device: str) -> None:
    """Generate golden data for a seq2seq (encoder-decoder) model."""
    import torch

    from mobius._testing.golden import save_generation_json, save_golden_ref
    from mobius._testing.torch_reference import (
        load_torch_seq2seq_model,
    )

    model, tokenizer = load_torch_seq2seq_model(case.model_id, device=device)

    encoded = tokenizer(case.prompts[0], return_tensors="np", padding=False)
    input_ids = encoded["input_ids"]

    # Prepare decoder input (pad token for autoregressive start)
    decoder_start_id = getattr(model.config, "decoder_start_token_id", None)
    if decoder_start_id is None:
        generation_config = getattr(model, "generation_config", None)
        decoder_start_id = getattr(generation_config, "decoder_start_token_id", None) or 0
    decoder_start = np.array([[decoder_start_id]], dtype=np.int64)

    # L4: single forward pass through full model
    torch_device = next(model.parameters()).device
    with torch.no_grad():
        outputs = model(
            input_ids=torch.from_numpy(input_ids).to(torch_device),
            decoder_input_ids=torch.from_numpy(decoder_start).to(torch_device),
        )
    last_logits = outputs.logits[0, -1, :].cpu().numpy()
    golden = _extract_logits_golden(last_logits)

    # L5: greedy generation
    generated_ids = None
    if "L5" in case.level:
        with torch.no_grad():
            gen_output = model.generate(
                torch.from_numpy(input_ids).to(torch_device),
                max_new_tokens=case.generation_params.get("max_new_tokens", 20),
                do_sample=False,
            )
        generated_ids = gen_output[0].cpu().numpy()

    save_golden_ref(
        json_path,
        top1_id=golden["top1_id"],
        top2_id=golden["top2_id"],
        top10_ids=golden["top10_ids"],
        top10_logits=golden["top10_logits"],
        logits_summary=golden["logits_summary"],
        input_ids=input_ids,
    )

    if generated_ids is not None:
        generated_text = tokenizer.decode(generated_ids.tolist(), skip_special_tokens=True)
        gen_path = json_path.with_name(json_path.stem + "_generation.json")
        save_generation_json(
            gen_path,
            model_id=case.model_id,
            prompt=case.prompts[0],
            generated_tokens=generated_ids.tolist(),
            generated_text=generated_text,
        )


def _generate_vision_language(case: TestCase, json_path: Path, device: str) -> None:
    """Generate golden data for a vision-language (image-text-to-text) model.

    Multi-model task: stores decoder golden data with dotted key prefix
    and component norms/shapes for vision + embedding diagnostics.
    """
    import torch
    from PIL import Image

    from mobius._testing.golden import save_generation_json, save_golden_ref
    from mobius._testing.torch_reference import (
        load_torch_multimodal_model,
    )

    model, _tokenizer, processor = load_torch_multimodal_model(case.model_id, device=device)

    # Load images from testdata/
    images = [Image.open(Path("testdata") / img_path) for img_path in case.images]

    # Build chat-formatted prompt with image placeholders if the
    # processor supports apply_chat_template (Qwen-VL, Gemma-3, etc.)
    prompt_text = case.prompts[0]
    if hasattr(processor, "apply_chat_template"):
        content: list[dict[str, str]] = []
        for img_path in case.images:
            content.append({"type": "image", "image": str(Path("testdata") / img_path)})
        content.append({"type": "text", "text": prompt_text})
        messages = [{"role": "user", "content": content}]
        prompt_text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    # Process multimodal inputs through the HF processor
    processed = processor(
        text=prompt_text,
        images=images if images else None,
        return_tensors="pt",
    )

    if device == "auto":
        first_device = next(model.parameters()).device
        processed = processed.to(first_device)
    else:
        processed = processed.to(device)

    # L4: single forward pass
    with torch.no_grad():
        outputs = model(**processed)

    last_logits = outputs.logits[0, -1, :].cpu().numpy()
    golden = _extract_logits_golden(last_logits)
    input_ids_np = processed["input_ids"].cpu().numpy()

    # L5: greedy generation
    generated_ids = None
    if "L5" in case.level:
        with torch.no_grad():
            gen = model.generate(
                **processed,
                max_new_tokens=case.generation_params.get("max_new_tokens", 30),
                do_sample=False,
            )
        input_len = processed["input_ids"].shape[1]
        generated_ids = gen[0, input_len:].cpu().numpy()

    save_golden_ref(
        json_path,
        top1_id=golden["top1_id"],
        top2_id=golden["top2_id"],
        top10_ids=golden["top10_ids"],
        top10_logits=golden["top10_logits"],
        logits_summary=golden["logits_summary"],
        input_ids=input_ids_np,
    )

    if generated_ids is not None:
        tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else None
        generated_text = (
            tokenizer.decode(generated_ids.tolist(), skip_special_tokens=True)
            if tokenizer is not None
            else None
        )
        gen_path = json_path.with_name(json_path.stem + "_generation.json")
        save_generation_json(
            gen_path,
            model_id=case.model_id,
            prompt=case.prompts[0],
            generated_tokens=generated_ids.tolist(),
            generated_text=generated_text,
        )


def _generate_speech_to_text(case: TestCase, json_path: Path, device: str) -> None:
    """Generate golden data for a speech-to-text (Whisper) model."""
    import librosa
    import torch

    from mobius._testing.golden import save_generation_json, save_golden_ref
    from mobius._testing.torch_reference import (
        load_torch_whisper_model,
    )

    model, processor = load_torch_whisper_model(case.model_id, device=device)

    # Load and preprocess audio
    audio_path = Path("testdata") / case.audio[0]
    audio_array, sample_rate = librosa.load(str(audio_path), sr=16000)
    processed = processor(audio_array, sampling_rate=sample_rate, return_tensors="pt").to(
        device
    )
    input_features = processed["input_features"]

    # L4: single decoder step with forced decoder start token
    decoder_start_id = (
        model.config.decoder_start_token_id or model.generation_config.decoder_start_token_id
    )
    decoder_input_ids = torch.tensor([[decoder_start_id]], dtype=torch.long, device=device)
    with torch.no_grad():
        outputs = model(
            input_features=input_features,
            decoder_input_ids=decoder_input_ids,
        )
    last_logits = outputs.logits[0, -1, :].cpu().numpy()
    golden = _extract_logits_golden(last_logits)
    input_ids_np = decoder_input_ids.cpu().numpy()

    # L5: greedy generation
    generated_ids = None
    if "L5" in case.level:
        with torch.no_grad():
            gen = model.generate(
                input_features=input_features,
                max_new_tokens=case.generation_params.get("max_new_tokens", 50),
                do_sample=False,
            )
        generated_ids = gen[0].cpu().numpy()

    save_golden_ref(
        json_path,
        top1_id=golden["top1_id"],
        top2_id=golden["top2_id"],
        top10_ids=golden["top10_ids"],
        top10_logits=golden["top10_logits"],
        logits_summary=golden["logits_summary"],
        input_ids=input_ids_np,
    )

    if generated_ids is not None:
        generated_text = processor.decode(generated_ids.tolist(), skip_special_tokens=True)
        gen_path = json_path.with_name(json_path.stem + "_generation.json")
        save_generation_json(
            gen_path,
            model_id=case.model_id,
            prompt=case.audio[0],
            generated_tokens=generated_ids.tolist(),
            generated_text=generated_text,
        )


def _try_register_qwen3_asr() -> None:
    """Register Qwen3-ASR config with transformers if available.

    The ``qwen_asr`` pip package provides the config and model classes
    but does not auto-register with transformers' ``AutoConfig``.  We
    do that here so ``AutoConfig.from_pretrained`` can load the config
    from HuggingFace without ``auto_map`` in the repo.
    """
    try:
        from qwen_asr.core.transformers_backend.configuration_qwen3_asr import (
            Qwen3ASRConfig,
        )
        from transformers import AutoConfig

        # register() is a no-op if already registered.
        AutoConfig.register("qwen3_asr", Qwen3ASRConfig)
    except ImportError:
        # Optional dependency is not installed; skip registration and
        # let speech-language generation proceed via other supported paths.
        pass


def _generate_speech_language(case: TestCase, json_path: Path, device: str) -> None:
    """Generate golden data for a speech-language model.

    Supports two architectures:

    * **Gemma4-style**: Uses ``AutoModelForImageTextToText`` with a
      multimodal processor that combines text prompts and audio.
    * **Qwen3-ASR-style**: Uses the ``qwen_asr`` package with its own
      processor and chat template (``trust_remote_code`` required).

    The model type is auto-detected from the HuggingFace config.
    """
    import librosa
    import torch

    from mobius._testing.golden import save_generation_json, save_golden_ref

    audio_path = Path("testdata") / case.audio[0]
    audio_array, _sample_rate = librosa.load(str(audio_path), sr=16000)

    model, processor, forward_model = _load_speech_language_model(case, device)

    processed, prompt_for_golden = _prepare_speech_language_inputs(
        case, model, processor, audio_array, audio_path, device
    )

    # L4: single forward pass
    with torch.no_grad():
        outputs = forward_model(**processed)

    last_logits = outputs.logits[0, -1, :].cpu().numpy()
    golden = _extract_logits_golden(last_logits)
    input_ids_np = processed["input_ids"].cpu().numpy()

    # L5: greedy generation
    generated_ids = None
    if "L5" in case.level:
        with torch.no_grad():
            gen = model.generate(
                **processed,
                max_new_tokens=case.generation_params.get("max_new_tokens", 50),
                do_sample=False,
            )
        input_len = processed["input_ids"].shape[1]
        gen_seq = gen.sequences if hasattr(gen, "sequences") else gen
        generated_ids = gen_seq[0, input_len:].cpu().numpy()

    save_golden_ref(
        json_path,
        top1_id=golden["top1_id"],
        top2_id=golden["top2_id"],
        top10_ids=golden["top10_ids"],
        top10_logits=golden["top10_logits"],
        logits_summary=golden["logits_summary"],
        input_ids=input_ids_np,
    )

    if generated_ids is not None:
        tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
        generated_text = tokenizer.decode(generated_ids.tolist(), skip_special_tokens=True)
        gen_path = json_path.with_name(json_path.stem + "_generation.json")
        save_generation_json(
            gen_path,
            model_id=case.model_id,
            prompt=prompt_for_golden,
            generated_tokens=generated_ids.tolist(),
            generated_text=generated_text,
        )


def _load_speech_language_model(case: TestCase, device: str) -> tuple:
    """Load a speech-language model and processor.

    Returns ``(model, processor, forward_model)`` where
    *forward_model* is the module whose ``forward()`` produces logits
    (may differ from *model* for nested architectures like Qwen3-ASR).
    """
    import torch
    import transformers

    # Try to register qwen3_asr classes before loading config,
    # since the HF repo lacks auto_map and trust_remote_code alone
    # won't resolve it.
    _try_register_qwen3_asr()

    config = transformers.AutoConfig.from_pretrained(
        case.model_id, trust_remote_code=case.trust_remote_code
    )
    model_type = getattr(config, "model_type", "")

    if model_type == "qwen3_asr":
        from qwen_asr.core.transformers_backend.modeling_qwen3_asr import (
            Qwen3ASRForConditionalGeneration,
        )

        model = Qwen3ASRForConditionalGeneration.from_pretrained(
            case.model_id, torch_dtype=torch.float32
        )
        model = model.to(device).eval()
        processor = transformers.AutoProcessor.from_pretrained(
            case.model_id, trust_remote_code=True
        )
        # Qwen3-ASR wraps a thinker; the thinker produces logits.
        forward_model = model.thinker
    else:
        # Gemma4-style: AutoModelForImageTextToText
        from mobius._testing.torch_reference import (
            load_torch_multimodal_model,
        )

        model, _tokenizer, processor = load_torch_multimodal_model(
            case.model_id, device=device
        )
        forward_model = model

    return model, processor, forward_model


def _prepare_speech_language_inputs(
    case: TestCase,
    model: object,
    processor: object,
    audio_array: np.ndarray,
    audio_path: Path,
    device: str,
) -> tuple:
    """Build model inputs and a prompt string for the golden file.

    Returns ``(processed, prompt_for_golden)`` where *processed* is a
    dict/BatchEncoding ready for ``model(**processed)`` and
    *prompt_for_golden* is the string saved in the generation JSON.
    """
    # Detect Qwen3-ASR by processor class name (avoids redundant
    # config download).
    is_qwen3_asr = "Qwen3ASR" in type(processor).__name__

    if is_qwen3_asr:
        # Qwen3-ASR prompt: system + user with audio placeholder
        messages = [
            {"role": "system", "content": ""},
            {
                "role": "user",
                "content": [{"type": "audio", "audio": ""}],
            },
        ]
        text_prompt = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        # If prompts are provided, append as forced decoder prefix
        # (e.g. "language English<asr_text>" to skip language detection).
        force_prefix = ""
        if case.prompts:
            force_prefix = case.prompts[0]
            text_prompt = text_prompt + force_prefix
        processed = processor(
            text=text_prompt,
            audio=[audio_array],
            return_tensors="pt",
        ).to(device)
        prompt_for_golden = force_prefix or str(audio_path)
    else:
        # Gemma4-style: text prompt + audio
        prompt_text = case.prompts[0]
        if hasattr(processor, "apply_chat_template"):
            content: list[dict[str, str]] = [
                {"type": "audio", "audio": str(audio_path)},
                {"type": "text", "text": prompt_text},
            ]
            messages = [{"role": "user", "content": content}]
            prompt_text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        processed = processor(
            text=prompt_text,
            audio=[audio_array],
            return_tensors="pt",
        ).to(device)
        prompt_for_golden = case.prompts[0]

    return processed, prompt_for_golden


def _generate_audio_feature_extraction(case: TestCase, json_path: Path, device: str) -> None:
    """Generate golden data for audio feature extraction (Wav2Vec2 etc.).

    Similar to encoder-only: last hidden state is used as the
    "logit" vector for top-k extraction.
    """
    import librosa

    from mobius._testing.golden import save_golden_ref
    from mobius._testing.torch_reference import (
        load_torch_audio_model,
        torch_audio_forward,
    )

    model, processor = load_torch_audio_model(
        case.model_id, device=device, trust_remote_code=case.trust_remote_code
    )

    # Load and preprocess audio
    audio_path = Path("testdata") / case.audio[0]
    audio_array, sample_rate = librosa.load(str(audio_path), sr=16000)
    processed = processor(
        audio_array,
        sampling_rate=sample_rate,
        return_tensors="np",
    )
    input_values = processed["input_values"]

    # Forward pass → last_hidden_state
    hidden_states = torch_audio_forward(model, input_values)
    # Use the last frame's hidden state for top-k extraction
    last_hidden = hidden_states[0, -1, :]  # (hidden_size,)
    golden = _extract_logits_golden(last_hidden)

    # Audio feature extraction is L4-only (no generation)
    save_golden_ref(
        json_path,
        top1_id=golden["top1_id"],
        top2_id=golden["top2_id"],
        top10_ids=golden["top10_ids"],
        top10_logits=golden["top10_logits"],
        logits_summary=golden["logits_summary"],
        input_ids=np.array([[0]], dtype=np.int64),  # placeholder
    )


def _generate_image_classification(case: TestCase, json_path: Path, device: str) -> None:
    """Generate golden data for image classification (ViT, CLIP, etc.).

    Similar to encoder-only: last hidden state is used as the
    "logit" vector for top-k extraction.
    """
    from PIL import Image

    from mobius._testing.golden import save_golden_ref
    from mobius._testing.torch_reference import (
        load_torch_vision_model,
        torch_vision_forward,
    )

    model, processor = load_torch_vision_model(
        case.model_id, device=device, trust_remote_code=case.trust_remote_code
    )

    # Load and preprocess image
    image = Image.open(Path("testdata") / case.images[0])
    # Use PyTorch tensors then convert — some processors don't support np
    processed = processor(images=image, return_tensors="pt")
    pixel_values = processed["pixel_values"].numpy()

    # Forward pass → last_hidden_state
    hidden_states = torch_vision_forward(model, pixel_values)
    # Vision models return different output shapes:
    # - ViT-like: [B, seq_len, hidden] → select first token (CLS)
    # - CNN-like (CvT, MobileViT, PVT): [B, C, H, W] → flatten feature map
    # - Classification head: [B, num_classes] → 1-D logits
    batch_hidden = hidden_states[0]  # drop batch dim
    if batch_hidden.ndim == 2:
        # (seq_len, hidden) — take CLS token
        last_hidden = batch_hidden[0]
    elif batch_hidden.ndim >= 3:
        # (C, H, W) feature map — flatten
        last_hidden = batch_hidden.reshape(-1)
    else:
        # 1-D logits or already flat
        last_hidden = batch_hidden
    golden = _extract_logits_golden(last_hidden)

    # Image classification is L4-only (no generation)
    save_golden_ref(
        json_path,
        top1_id=golden["top1_id"],
        top2_id=golden["top2_id"],
        top10_ids=golden["top10_ids"],
        top10_logits=golden["top10_logits"],
        logits_summary=golden["logits_summary"],
        input_ids=np.array([[0]], dtype=np.int64),  # placeholder
    )


# ---- Phi4MM multimodal generator ----


def _apply_phi4mm_compat_patches():
    """Apply compatibility patches for Phi4MM on transformers 5.x.

    Phi4MM's remote code was written for transformers 4.x and has several
    incompatibilities with transformers 5.x:

    1. NemoConvSubsampling.__init__() calls int() on a meta tensor
       (calc_length result) — we intercept __int__ to compute on CPU.
    2. _tied_weights_keys is a list but transformers 5.x expects a dict.
    3. PeftModelForCausalLM expects prepare_inputs_for_generation.
    4. DynamicCache.get_usable_length was renamed to get_seq_length.

    Returns a cleanup function that restores original behavior.
    """
    import math

    import torch

    patches = {}

    # Patch 1: meta tensor int() for NemoConvSubsampling calc_length
    patches["tensor_int"] = torch.Tensor.__int__
    _orig_int = patches["tensor_int"]

    def _meta_safe_int(self):
        if self.is_meta:
            import inspect

            frame = inspect.currentframe().f_back
            lv = frame.f_locals
            if "out_length" in lv and "conv_channels" in lv:
                s = lv["self"]
                val = float(lv.get("feat_in", 80))
                add_pad = float(s._left_padding + s._right_padding - s._kernel_size)
                for _ in range(s._sampling_num):
                    val = (val + add_pad) / s._stride + 1.0
                    val = math.ceil(val) if s._ceil_mode else math.floor(val)
                return int(val)
            return 0
        return _orig_int(self)

    torch.Tensor.__int__ = _meta_safe_int

    # Patch 2: tied_weights_keys list→dict compat
    # Phi4MM remote code sets _tied_weights_keys as a list (transformers 4.x
    # format) but transformers 5.x expects a dict {tied_param: source_param}.
    # We convert the list format before the original methods see it.
    import transformers.modeling_utils as mu

    def _fix_tied_weights_keys(model):
        """Convert _tied_weights_keys from list to dict if needed."""
        twk = getattr(model, "_tied_weights_keys", None)
        if isinstance(twk, list):
            # Build dict: for each tied key, find the source via
            # _get_tied_params (the embedding weight it's tied to)
            tied_dict = {}
            for key in twk:
                # Standard pattern: lm_head.weight -> model.embed_tokens.weight
                if key == "lm_head.weight":
                    tied_dict[key] = "model.embed_tokens.weight"
                else:
                    tied_dict[key] = key  # fallback: self-reference
            model._tied_weights_keys = tied_dict

    patches["get_expanded"] = mu.PreTrainedModel.get_expanded_tied_weights_keys
    _orig_get_expanded = patches["get_expanded"]

    def _patched_get_expanded(self, all_submodels=False):
        _fix_tied_weights_keys(self)
        return _orig_get_expanded(self, all_submodels=all_submodels)

    mu.PreTrainedModel.get_expanded_tied_weights_keys = _patched_get_expanded

    patches["post_init"] = mu.PreTrainedModel.post_init
    _orig_post_init = patches["post_init"]

    def _patched_post_init(self):
        _fix_tied_weights_keys(self)
        _orig_post_init(self)

    mu.PreTrainedModel.post_init = _patched_post_init

    patches["total_bytes"] = mu.get_total_byte_count
    _orig_total_bytes = patches["total_bytes"]

    def _patched_total_bytes(model, dm, q):
        _fix_tied_weights_keys(model)
        return _orig_total_bytes(model, dm, q)

    mu.get_total_byte_count = _patched_total_bytes

    # Patch 3: peft prepare_inputs_for_generation
    import peft.peft_model as pm

    patches["peft_init"] = pm.PeftModelForCausalLM.__init__
    _orig_peft_init = patches["peft_init"]

    def _patched_peft_init(self, model, peft_config=None, adapter_name="default", **kwargs):
        if not hasattr(model, "prepare_inputs_for_generation"):
            model.prepare_inputs_for_generation = lambda *a, **kw: {}
        _orig_peft_init(
            self,
            model,
            peft_config=peft_config,
            adapter_name=adapter_name,
            **kwargs,
        )

    pm.PeftModelForCausalLM.__init__ = _patched_peft_init

    # Patch 4: DynamicCache compat (methods removed in transformers 5.x)
    from transformers import DynamicCache

    if not hasattr(DynamicCache, "get_usable_length"):
        DynamicCache.get_usable_length = lambda self, *a, **kw: self.get_seq_length()
        patches["dc_get_usable_length"] = True

    if not hasattr(DynamicCache, "to_legacy_cache"):

        def _to_legacy_cache(self):
            return tuple((layer.keys, layer.values) for layer in self.layers)

        DynamicCache.to_legacy_cache = _to_legacy_cache
        patches["dc_to_legacy_cache"] = True

    if not hasattr(DynamicCache, "from_legacy_cache"):

        @classmethod
        def _from_legacy_cache(cls, past_kv):
            cache = cls()
            if past_kv is not None:
                for layer_idx, (k, v) in enumerate(past_kv):
                    cache.update(k, v, layer_idx)
            return cache

        DynamicCache.from_legacy_cache = _from_legacy_cache
        patches["dc_from_legacy_cache"] = True

    def cleanup():
        torch.Tensor.__int__ = patches["tensor_int"]
        mu.PreTrainedModel.get_expanded_tied_weights_keys = patches["get_expanded"]
        mu.PreTrainedModel.post_init = patches["post_init"]
        mu.get_total_byte_count = patches["total_bytes"]
        pm.PeftModelForCausalLM.__init__ = patches["peft_init"]
        if patches.get("dc_get_usable_length"):
            del DynamicCache.get_usable_length
        if patches.get("dc_to_legacy_cache"):
            del DynamicCache.to_legacy_cache
        if patches.get("dc_from_legacy_cache"):
            del DynamicCache.from_legacy_cache

    return cleanup


def _generate_phi4mm_multimodal(case: TestCase, json_path: Path, device: str) -> None:
    """Generate golden data for Phi4MM (text + vision + audio).

    Phi4MM is a 4-model-split multimodal model that accepts text, images,
    and audio in any combination. Uses trust_remote_code for the custom
    HF model and processor classes.
    """
    import torch
    import transformers
    from PIL import Image

    from mobius._testing.golden import save_generation_json, save_golden_ref

    # Apply transformers 5.x compatibility patches for Phi4MM remote code
    cleanup = _apply_phi4mm_compat_patches()

    try:
        # Load model and processor
        processor = transformers.AutoProcessor.from_pretrained(
            case.model_id, trust_remote_code=True
        )
        # Load in bfloat16 to reduce memory (14B model)
        model = transformers.AutoModelForCausalLM.from_pretrained(
            case.model_id,
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True,
            _attn_implementation="eager",
        )
        model.eval()
    except Exception:
        cleanup()
        raise

    # Build processor inputs based on available modalities
    call_kwargs: dict = {}

    # Text prompt
    prompt_text = case.prompts[0] if case.prompts else ""

    # Images
    images = None
    if case.images:
        images = [Image.open(Path("testdata") / img_path) for img_path in case.images]

    # Audio (processor expects list of (audio_array, sample_rate) tuples)
    audios = None
    if case.audio:
        import librosa

        audios = []
        for audio_path in case.audio:
            audio_array, _sr = librosa.load(str(Path("testdata") / audio_path), sr=16000)
            audios.append((audio_array, 16000))

    # Build prompt with special tokens for each modality
    # Phi4MM uses <|image_N|> and <|audio_N|> placeholders (1-indexed)
    user_content = ""
    if images:
        for i in range(len(images)):
            user_content += f"<|image_{i + 1}|>\n"
    if audios:
        for i in range(len(audios)):
            user_content += f"<|audio_{i + 1}|>\n"
    user_content += prompt_text

    messages = [{"role": "user", "content": user_content}]
    prompt_formatted = processor.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    call_kwargs["text"] = prompt_formatted
    if images:
        call_kwargs["images"] = images
    if audios:
        call_kwargs["audios"] = audios
    call_kwargs["return_tensors"] = "pt"

    processed = processor(**call_kwargs).to(device)

    # L4: single forward pass for prefill logits
    with torch.no_grad():
        outputs = model(**processed)
    last_logits = outputs.logits[0, -1, :].float().cpu().numpy()
    golden = _extract_logits_golden(last_logits)
    input_ids_np = processed["input_ids"].cpu().numpy()

    # L5: manual greedy decode (model.generate() has compatibility issues
    # with phi4mm's custom modeling code on transformers 4.x)
    generated_ids = None
    if "L5" in case.level:
        max_new_tokens = case.generation_params.get("max_new_tokens", 20)
        gen_ids = []
        past_kv = None
        cur_input_ids = processed["input_ids"]
        input_mode = processed.get("input_mode")
        # Build initial kwargs (first step uses full processed inputs)
        fwd_kwargs = dict(processed)
        fwd_kwargs.pop("input_ids", None)
        for step in range(max_new_tokens):
            with torch.no_grad():
                if step == 0:
                    out = model(input_ids=cur_input_ids, **fwd_kwargs)
                else:
                    # Get sequence length from past KV cache
                    # (works with both tuple-style and DynamicCache)
                    if hasattr(past_kv, "get_seq_length"):
                        kv_len = past_kv.get_seq_length()
                    else:
                        kv_len = past_kv[0][0].shape[2]
                    out = model(
                        input_ids=cur_input_ids,
                        past_key_values=past_kv,
                        input_mode=input_mode,
                        attention_mask=torch.ones(
                            1,
                            kv_len + 1,
                            dtype=torch.long,
                            device=device,
                        ),
                    )
            next_id = int(out.logits[0, -1, :].argmax())
            past_kv = out.past_key_values
            gen_ids.append(next_id)
            cur_input_ids = torch.tensor([[next_id]], dtype=torch.long, device=device)
            # Stop on EOS
            eos = getattr(model.config, "eos_token_id", None)
            if eos is not None and next_id == eos:
                break
        if gen_ids:
            generated_ids = np.array(gen_ids)

    save_golden_ref(
        json_path,
        top1_id=golden["top1_id"],
        top2_id=golden["top2_id"],
        top10_ids=golden["top10_ids"],
        top10_logits=golden["top10_logits"],
        logits_summary=golden["logits_summary"],
        input_ids=input_ids_np,
    )

    if generated_ids is not None:
        generated_text = processor.tokenizer.decode(
            generated_ids.tolist(), skip_special_tokens=True
        )
        gen_path = json_path.with_name(json_path.stem + "_generation.json")
        save_generation_json(
            gen_path,
            model_id=case.model_id,
            prompt=case.prompts[0] if case.prompts else "",
            generated_tokens=generated_ids.tolist(),
            generated_text=generated_text,
        )

    cleanup()


# ---- Dispatcher ----

# Map task_type strings to generator functions.
_GENERATORS = {
    "text-generation": _generate_causal_lm,
    "feature-extraction": _generate_encoder,
    "seq2seq": _generate_seq2seq,
    "image-text-to-text": _generate_vision_language,
    "image-classification": _generate_image_classification,
    "speech-to-text": _generate_speech_to_text,
    "speech-language": _generate_speech_language,
    "audio-feature-extraction": _generate_audio_feature_extraction,
    # Vision tasks that produce last_hidden_state — reuse image classification.
    "depth-estimation": _generate_image_classification,
    "image-segmentation": _generate_image_classification,
    "image-to-image": _generate_image_classification,
    "object-detection": _generate_image_classification,
    "phi4mm-multimodal": _generate_phi4mm_multimodal,
}


def generate_golden_for_case(case: TestCase, json_path: Path, device: str) -> bool:
    """Generate golden reference data for one test case.

    Returns True on success, False on failure (logged to stderr).
    """
    generator = _GENERATORS.get(case.task_type)
    if generator is None:
        print(
            f"  SKIP: unsupported task_type={case.task_type!r}",
            file=sys.stderr,
        )
        return False

    try:
        generator(case, json_path, device)
    except Exception as exc:
        print(
            f"  ERROR: {case.case_id}: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return False
    else:
        return True


# ---- Main ----


def main() -> int:
    """Entry point.  Returns 0 on success, 1 if any cases failed."""
    args = parse_args()

    if args.device.startswith("cuda"):
        import torch

        # Disable cuDNN to avoid CUDNN_STATUS_NOT_INITIALIZED on
        # systems where the cuDNN library version doesn't match the
        # CUDA toolkit bundled with PyTorch.
        torch.backends.cudnn.enabled = False

    from mobius._testing.golden import (
        discover_test_cases,
        golden_path_for_case,
        has_golden,
        load_test_case,
    )

    golden_dir: Path = args.golden_dir

    # Collect test cases.
    if args.case is not None:
        cases = [load_test_case(args.case)]
    else:
        cases = discover_test_cases(
            task_type=args.task_type,
            level=args.level,
            root=args.cases_dir,
        )

    # Apply glob filter on case_id.
    if args.filter:
        cases = [c for c in cases if fnmatch.fnmatch(c.case_id, args.filter)]

    if not cases:
        print("No test cases found matching the given filters.")
        return 0

    print(f"Found {len(cases)} test case(s).")

    succeeded = 0
    skipped = 0
    failed = 0
    failed_ids: list[str] = []

    for case in cases:
        json_path = golden_path_for_case(case, golden_dir=golden_dir)
        label = f"{case.yaml_path.parent.name}/{case.case_id}"

        if case.skip_reason:
            print(f"  SKIP: {label} — {case.skip_reason}")
            skipped += 1
            continue

        if has_golden(case, golden_dir=golden_dir) and not args.force:
            print(f"  EXISTS: {label} (use --force to overwrite)")
            skipped += 1
            continue

        if args.dry_run:
            print(f"  DRY-RUN: {label} → {json_path}")
            skipped += 1
            continue

        print(f"  GENERATING: {label} ...")
        start = time.time()
        ok = generate_golden_for_case(case, json_path, args.device)
        elapsed = time.time() - start

        if ok:
            print(f"  SAVED: {json_path} ({elapsed:.1f}s)")
            succeeded += 1
        else:
            failed += 1
            failed_ids.append(label)

    # Summary
    print(f"\nDone: {succeeded} saved, {skipped} skipped, {failed} failed.")
    if failed_ids:
        print("Failed cases:")
        for fid in failed_ids:
            print(f"  - {fid}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
