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
import contextlib
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


# ---- Device helpers ----


def _get_model_device(model: object, device: str):
    """Return the concrete ``torch.device`` where model inputs should go.

    ``device_map="auto"`` is valid for ``from_pretrained()`` (it
    distributes layers across available GPUs), but ``.to("auto")`` is
    not valid for tensors or BatchEncoding objects.  When *device* is
    ``"auto"``, we inspect the model's first parameter to find the
    actual device that inputs should be placed on.
    """
    import torch

    if device == "auto":
        return next(model.parameters()).device
    return torch.device(device)


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

    model, tokenizer = load_torch_model(
        case.model_id, device=device, trust_remote_code=case.trust_remote_code
    )

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

        model_device = _get_model_device(model, device)
        gen_ids = torch.from_numpy(input_ids).to(model_device)
        max_new = case.generation_params.get("max_new_tokens", 20)
        with torch.no_grad():
            try:
                gen_output = model.generate(gen_ids, max_new_tokens=max_new, do_sample=False)
            except ValueError as e:
                # All-attention GraniteMoeHybrid variants (e.g. granite-4.0-1b)
                # trip transformers' hybrid Mamba/attention generation cache,
                # which assumes at least one linear-attention (Mamba) layer:
                # "`has_previous_state` can only be called on LinearAttention
                # layers". Greedy output is cache-independent, so fall back to
                # the (slower) cache-free path.
                if "has_previous_state" not in str(e):
                    raise
                gen_output = model.generate(
                    gen_ids, max_new_tokens=max_new, do_sample=False, use_cache=False
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

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    torch_dtype = dtype_map.get(case.dtype, torch.float32)
    model, _tokenizer, processor = load_torch_multimodal_model(
        case.model_id, dtype=torch_dtype, device=device
    )

    # Load images from testdata/
    images = [Image.open(Path("testdata") / img_path) for img_path in case.images]

    # Build a chat-formatted prompt when a usable template is available.
    # Phi-3 Vision exposes its template on the underlying tokenizer rather
    # than on the processor.
    prompt_text = case.prompts[0]
    template_applied = False
    if getattr(processor, "chat_template", None):
        content: list[dict[str, str]] = []
        for img_path in case.images:
            content.append({"type": "image", "image": str(Path("testdata") / img_path)})
        content.append({"type": "text", "text": prompt_text})
        messages = [{"role": "user", "content": content}]
        try:
            prompt_text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            template_applied = True
        except (AttributeError, ValueError):
            pass
    if not template_applied:
        tokenizer_inner = getattr(processor, "tokenizer", None)
        img_tokens = getattr(processor, "img_tokens", None)
        if (
            tokenizer_inner is not None
            and img_tokens is not None
            and getattr(tokenizer_inner, "chat_template", None)
        ):
            img_prefix = "".join(f"{img_tokens[i]}\n" for i in range(len(images)))
            messages = [{"role": "user", "content": img_prefix + prompt_text}]
            prompt_text = tokenizer_inner.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        elif getattr(processor, "image_token", None):
            prompt_text = processor.image_token * len(case.images) + prompt_text

    # Process multimodal inputs through the HF processor
    processed = processor(
        text=prompt_text,
        images=images if images else None,
        return_tensors="pt",
    )

    # Normalize the CLI/device-map selection to a concrete runtime device
    # before moving any tensors. `device="auto"` is handled by Transformers/
    # Accelerate during model loading, but `.to("auto")` is not valid for
    # BatchEncoding or Tensor objects.
    model_device = _get_model_device(model, device)
    processed = processed.to(model_device)

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
    model_device = _get_model_device(model, device)
    processed = processor(audio_array, sampling_rate=sample_rate, return_tensors="pt").to(
        model_device
    )
    input_features = processed["input_features"]

    # L4: single decoder step with forced decoder start token
    decoder_start_id = (
        model.config.decoder_start_token_id or model.generation_config.decoder_start_token_id
    )
    decoder_input_ids = torch.tensor(
        [[decoder_start_id]], dtype=torch.long, device=model_device
    )
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

        if device == "auto":
            model = Qwen3ASRForConditionalGeneration.from_pretrained(
                case.model_id, torch_dtype=torch.float32, device_map=device
            )
        else:
            model = Qwen3ASRForConditionalGeneration.from_pretrained(
                case.model_id, torch_dtype=torch.float32
            )
            model = model.to(device)
        model.eval()
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
        model_device = _get_model_device(model, device)
        processed = processor(
            text=text_prompt,
            audio=[audio_array],
            return_tensors="pt",
        ).to(model_device)
        prompt_for_golden = force_prefix or str(audio_path)
    else:
        # Gemma4-style: text prompt + audio
        prompt_text = case.prompts[0]
        if getattr(processor, "chat_template", None):
            content: list[dict[str, str]] = [
                {"type": "audio", "audio": str(audio_path)},
                {"type": "text", "text": prompt_text},
            ]
            messages = [{"role": "user", "content": content}]
            prompt_text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        elif getattr(processor, "audio_token", None):
            # Base checkpoint (no chat template): manually prepend the audio
            # placeholder; the processor expands it to the right token count.
            prompt_text = processor.audio_token + prompt_text
        model_device = _get_model_device(model, device)
        processed = processor(
            text=prompt_text,
            audio=[audio_array],
            return_tensors="pt",
        ).to(model_device)
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


def _generate_ctc_asr(case: TestCase, json_path: Path, device: str) -> None:
    """Generate golden data for CTC-based ASR (Wav2Vec2ForCTC / MMS).

    The model output is per-frame logits over a vocabulary; we save the
    top-K over the final frame's logit vector (matching the existing
    audio-feature-extraction pattern), and when L5 is requested we also
    save the CTC-greedy-decoded transcript as a token-id sequence so the
    end-to-end test can compare against the runtime's greedy decode.

    MMS specifically requires picking a target language adapter via
    ``processor.tokenizer.set_target_lang(lang)`` and
    ``model.load_adapter(lang)`` before the forward pass. The language is
    read from ``case.generation_params['lang']`` (default ``"eng"``).
    """
    import librosa
    import torch
    import transformers

    from mobius._testing.golden import save_generation_json, save_golden_ref

    lang = case.generation_params.get("lang", "eng")

    processor = transformers.AutoProcessor.from_pretrained(
        case.model_id, trust_remote_code=case.trust_remote_code, target_lang=lang
    )
    model = transformers.Wav2Vec2ForCTC.from_pretrained(
        case.model_id,
        torch_dtype=torch.float32,
        device_map=device,
        trust_remote_code=case.trust_remote_code,
        target_lang=lang,
        ignore_mismatched_sizes=True,  # MMS lm_head shape changes per language
    )
    # For MMS, switching languages also requires loading the per-language adapter.
    # Non-MMS Wav2Vec2ForCTC checkpoints don't have language adapters;
    # the missing-adapter case is expected and harmless there.
    if hasattr(model, "load_adapter"):
        with contextlib.suppress(ValueError, KeyError, OSError):
            model.load_adapter(lang)
    model.eval()

    audio_path = Path("testdata") / case.audio[0]
    audio_array, sample_rate = librosa.load(str(audio_path), sr=16000)
    processed = processor(audio_array, sampling_rate=sample_rate, return_tensors="pt").to(
        next(model.parameters()).device
    )

    with torch.no_grad():
        outputs = model(**processed)

    # CTC logits: (batch, num_frames, vocab_size). Use last frame for top-K.
    logits = outputs.logits[0]  # (num_frames, vocab_size)
    last_logits = logits[-1].cpu().numpy()
    golden = _extract_logits_golden(last_logits)

    save_golden_ref(
        json_path,
        top1_id=golden["top1_id"],
        top2_id=golden["top2_id"],
        top10_ids=golden["top10_ids"],
        top10_logits=golden["top10_logits"],
        logits_summary=golden["logits_summary"],
        input_ids=np.array([[0]], dtype=np.int64),  # placeholder
    )

    if "L5" in case.level:
        # CTC greedy decode: argmax over vocab per frame, then collapse
        # repeats and remove blanks. Save the post-collapse token IDs (and
        # the decoded text for human inspection) into the standard
        # ``*_generation.json`` sidecar.
        predicted_ids = torch.argmax(logits, dim=-1).cpu().numpy()
        transcript = processor.batch_decode(predicted_ids[np.newaxis, :])[0]
        gen_path = json_path.with_name(json_path.stem + "_generation.json")
        save_generation_json(
            gen_path,
            model_id=case.model_id,
            prompt=str(audio_path),
            generated_tokens=predicted_ids.tolist(),
            generated_text=transcript,
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


def _detection_forced_size(model_id: str, trust_remote_code: bool) -> dict | None:
    """Return a fixed ``{height, width}`` size for object-detection export.

    mobius exports object-detection models (e.g. YOLOS) at a *fixed* input
    resolution taken from ``config.image_size`` (position embeddings are not
    interpolated for arbitrary sizes).  To keep the golden reference and the
    ONNX forward pass on the same footing, the HF image processor must be
    forced to emit exactly that resolution instead of its default
    aspect-preserving resize.
    """
    import transformers

    config = transformers.AutoConfig.from_pretrained(
        model_id, trust_remote_code=trust_remote_code
    )
    image_size = getattr(config, "image_size", None)
    if isinstance(image_size, (list, tuple)) and len(image_size) == 2:
        return {"height": int(image_size[0]), "width": int(image_size[1])}
    if isinstance(image_size, int):
        return {"height": image_size, "width": image_size}
    return None


def _generate_object_detection(case: TestCase, json_path: Path, device: str) -> None:
    """Generate golden data for object detection (e.g. YOLOS).

    The model emits per-query class ``logits`` of shape
    ``[batch, num_queries, num_labels + 1]``.  The golden top-K is taken over
    the *last* query's class-logit vector to match ``compare_golden``, which
    slices ``logits[:, -1, :]``.  The image processor is forced to the model's
    fixed export resolution so the golden and ONNX forward pass agree.
    """
    import torch
    import transformers
    from PIL import Image

    from mobius._testing.golden import save_golden_ref

    processor = transformers.AutoImageProcessor.from_pretrained(
        case.model_id, trust_remote_code=case.trust_remote_code
    )
    model = transformers.AutoModelForObjectDetection.from_pretrained(
        case.model_id,
        torch_dtype=torch.float32,
        device_map=device,
        trust_remote_code=case.trust_remote_code,
    ).eval()

    image = Image.open(Path("testdata") / case.images[0])
    forced_size = _detection_forced_size(case.model_id, case.trust_remote_code)
    proc_kwargs = {"images": image, "return_tensors": "pt"}
    if forced_size is not None:
        proc_kwargs["size"] = forced_size
    processed = processor(**proc_kwargs)
    pixel_values = processed["pixel_values"].to(device)

    with torch.no_grad():
        outputs = model(pixel_values=pixel_values)
    # logits: [batch, num_queries, num_labels + 1] -> last query's class vector
    last_logits = outputs.logits[0, -1, :].cpu().numpy()
    golden = _extract_logits_golden(last_logits)

    save_golden_ref(
        json_path,
        top1_id=golden["top1_id"],
        top2_id=golden["top2_id"],
        top10_ids=golden["top10_ids"],
        top10_logits=golden["top10_logits"],
        logits_summary=golden["logits_summary"],
        input_ids=np.array([[0]], dtype=np.int64),  # placeholder (no text input)
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
        # Load in float32 to match the f32 runtime used by the L4/L5 tests.
        # bf16 goldens produced flat/tied logit distributions that caused
        # argmax instability against the f32 model output (see phi4mm L4
        # false-failures: exact top-2 ties, top-10 spans <3 logits).
        model = transformers.AutoModelForCausalLM.from_pretrained(
            case.model_id,
            torch_dtype=torch.float32,
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

    model_device = _get_model_device(model, device)
    processed = processor(**call_kwargs).to(model_device)

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
                            device=model_device,
                        ),
                    )
            next_id = int(out.logits[0, -1, :].argmax())
            past_kv = out.past_key_values
            gen_ids.append(next_id)
            cur_input_ids = torch.tensor([[next_id]], dtype=torch.long, device=model_device)
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


def _generate_gemma4_assistant(case: TestCase, json_path: Path, device: str) -> None:
    """Generate golden data for the Gemma4-Assistant MTP drafter.

    The drafter does not consume a tokenized prompt directly; it consumes
    ``inputs_embeds`` (target token embedding concatenated with the target's
    shared hidden state) plus the target's ``shared_kv`` and a fixed
    ``position_id``.  We reproduce exactly what HuggingFace assisted
    generation feeds the assistant by hooking the assistant during a real
    ``target.generate(assistant_model=...)`` run, capturing the first draft
    round (all steps share one position with a fixed shared KV — this is the
    ``SinglePositionMultiTokenCandidateGenerator``).

    Artefacts:
      - ``<name>.json``            L4: top-k of the first draft step's logits.
      - ``<name>_generation.json`` L5: the drafted token sequence.
      - ``<name>_inputs.npz``      replay tensors for the assistant ONNX graph.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from mobius._testing.golden import (
        drafter_inputs_path_for_case,
        save_drafter_inputs,
        save_generation_json,
        save_golden_ref,
    )

    # The drafter ships as ``<target>-assistant``; derive the target it pairs
    # with (the target supplies the shared KV + hidden state).
    assistant_id = case.model_id
    target_id = assistant_id.removesuffix("-assistant")
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    torch_dtype = dtype_map.get(case.dtype, torch.float32)

    tok = AutoTokenizer.from_pretrained(target_id)
    target = (
        AutoModelForCausalLM.from_pretrained(target_id, dtype=torch_dtype).to(device).eval()
    )
    assistant = (
        AutoModelForCausalLM.from_pretrained(assistant_id, dtype=torch_dtype).to(device).eval()
    )

    captured: list[dict] = []

    def _hook(_module, _args, kwargs, output):
        skv = kwargs.get("shared_kv_states") or {}
        captured.append(
            {
                "inputs_embeds": kwargs["inputs_embeds"].detach().float().cpu().numpy(),
                "position_ids": kwargs["position_ids"].detach().cpu().numpy().astype(np.int64),
                "attention_mask": kwargs["attention_mask"]
                .detach()
                .cpu()
                .numpy()
                .astype(np.int64),
                "shared_kv": {
                    lt: (
                        kv[0].detach().float().cpu().numpy(),
                        kv[1].detach().float().cpu().numpy(),
                    )
                    for lt, kv in skv.items()
                },
                "logits": output.logits[0, -1, :].detach().float().cpu().numpy(),
                "projected_state": output.last_hidden_state.detach().float().cpu().numpy(),
            }
        )

    handle = assistant.register_forward_hook(_hook, with_kwargs=True)
    enc = tok.apply_chat_template(
        [{"role": "user", "content": case.prompts[0]}],
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    input_ids = enc["input_ids"].to(device)
    max_new = case.generation_params.get("max_new_tokens", 16)
    with torch.inference_mode():
        target.generate(
            input_ids,
            max_new_tokens=max_new,
            do_sample=False,
            assistant_model=assistant,
            pad_token_id=tok.eos_token_id,
        )
    handle.remove()

    if not captured:
        raise RuntimeError("Assistant was never invoked during assisted generation.")

    # First draft round = consecutive calls sharing the first position id.
    first_pos = int(captured[0]["position_ids"].reshape(-1)[0])
    round1 = [c for c in captured if int(c["position_ids"].reshape(-1)[0]) == first_pos]

    drafted_tokens = [int(np.asarray(c["logits"]).argmax()) for c in round1]

    # L4 golden: top-k of the first draft step's logits.
    golden = _extract_logits_golden(round1[0]["logits"].astype(np.float64))
    proj0 = round1[0]["projected_state"]
    save_golden_ref(
        json_path,
        top1_id=golden["top1_id"],
        top2_id=golden["top2_id"],
        top10_ids=golden["top10_ids"],
        top10_logits=golden["top10_logits"],
        logits_summary=golden["logits_summary"],
        input_ids=input_ids.cpu().numpy(),
        component_norms={"projected_state": float(np.linalg.norm(proj0))},
        component_shapes={
            "projected_state": tuple(int(x) for x in proj0.shape),
            "logits": (1, 1, int(round1[0]["logits"].shape[-1])),
        },
    )

    # L5 golden: the drafted token sequence.
    if "L5" in case.level:
        gen_path = json_path.with_name(json_path.stem + "_generation.json")
        save_generation_json(
            gen_path,
            model_id=case.model_id,
            prompt=case.prompts[0],
            generated_tokens=drafted_tokens,
            generated_text=tok.decode(drafted_tokens, skip_special_tokens=True),
        )

    # Replay tensors: all round-1 steps share one position + shared KV; only
    # inputs_embeds varies (autoregressive feedback). Store every step's
    # inputs_embeds so the test can teacher-force the draft sequence.
    arrays: dict[str, np.ndarray] = {
        "inputs_embeds": np.concatenate([c["inputs_embeds"] for c in round1], axis=0),
        "position_ids": round1[0]["position_ids"],
        "attention_mask": round1[0]["attention_mask"],
        "layer_types": np.array(list(round1[0]["shared_kv"].keys())),
    }
    for lt, (key, value) in round1[0]["shared_kv"].items():
        arrays[f"skv_key_{lt}"] = key
        arrays[f"skv_val_{lt}"] = value
    save_drafter_inputs(drafter_inputs_path_for_case(case), arrays)


def _generate_dflash_draft(case: TestCase, json_path: Path, device: str) -> None:
    """Generate golden data for a DFlash speculative-decoding drafter.

    The drafter consumes ``noise_embedding`` + multi-layer ``target_hidden`` + a
    KV cache (not ``input_ids``) and outputs ``draft_hidden`` (decoded through the
    target ``lm_head`` to get draft logits). We capture the first draft block of
    the reference ``spec_generate`` loop by hooking the drafter, recording its
    inputs and the reference ``draft_hidden``.

    L4 is a hidden-state parity check: ``draft_hidden``'s last-token vector is
    treated as the logit vector for the argmax + cosine gate (mirroring the
    encoder / feature-extraction golden path). No L5 — the drafted token
    sequence would require the target ``lm_head`` and the block-diffusion loop.

    Artefacts:
      - ``<name>.json``        L4: top-k of the first block's last-token hidden.
      - ``<name>_inputs.npz``  replay tensors for the drafter ONNX graph.
    """
    import torch
    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

    from mobius._testing.golden import (
        drafter_inputs_path_for_case,
        save_drafter_inputs,
        save_golden_ref,
    )

    drafter_id = case.model_id
    target_id = case.generation_params.get("target_model_id")
    if not target_id:
        raise ValueError(
            "dflash-draft golden requires generation.target_model_id (the paired "
            "target that supplies target_hidden)."
        )
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    torch_dtype = dtype_map.get(case.dtype, torch.float32)

    tok = AutoTokenizer.from_pretrained(target_id)
    target = (
        AutoModelForCausalLM.from_pretrained(
            target_id, torch_dtype=torch_dtype, attn_implementation="eager"
        )
        .to(device)
        .eval()
    )
    drafter = (
        AutoModel.from_pretrained(
            drafter_id,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            attn_implementation="eager",
        )
        .to(device)
        .eval()
    )

    captured: list[dict] = []

    def _hook(_module, _args, kwargs, output):
        hidden = (
            output.last_hidden_state
            if hasattr(output, "last_hidden_state")
            else (output[0] if isinstance(output, tuple) else output)
        )
        captured.append(
            {
                "noise_embedding": kwargs["noise_embedding"].detach().float().cpu().numpy(),
                "target_hidden": kwargs["target_hidden"].detach().float().cpu().numpy(),
                "position_ids": kwargs["position_ids"].detach().cpu().numpy().astype(np.int64),
                "draft_hidden": hidden.detach().float().cpu().numpy(),
            }
        )

    handle = drafter.register_forward_hook(_hook, with_kwargs=True)
    input_ids = tok(case.prompts[0], return_tensors="pt").input_ids.to(device)
    stop = [tok.eos_token_id] if tok.eos_token_id is not None else None
    with torch.inference_mode():
        drafter.spec_generate(
            target=target,
            input_ids=input_ids,
            max_new_tokens=case.generation_params.get("max_new_tokens", 16),
            stop_token_ids=stop,
            temperature=0.0,
        )
    handle.remove()

    if not captured:
        raise RuntimeError("Drafter was never invoked during spec_generate.")

    first = captured[0]
    draft_hidden = first["draft_hidden"]
    last_hidden = draft_hidden[0, -1, :].astype(np.float64)
    golden = _extract_logits_golden(last_hidden)
    save_golden_ref(
        json_path,
        top1_id=golden["top1_id"],
        top2_id=golden["top2_id"],
        top10_ids=golden["top10_ids"],
        top10_logits=golden["top10_logits"],
        logits_summary=golden["logits_summary"],
        input_ids=input_ids.cpu().numpy(),
        component_norms={"draft_hidden": float(np.linalg.norm(draft_hidden))},
        component_shapes={"draft_hidden": tuple(int(x) for x in draft_hidden.shape)},
    )

    # Replay tensors: the drafter ONNX splits the reference's single position_ids
    # into position_ids ([ctx + q]) and q_position_ids (last q). The first block
    # call starts from an empty draft KV cache (the test supplies zero-length past
    # tensors sized from the config).
    pos = first["position_ids"]
    q_len = first["noise_embedding"].shape[1]
    arrays = {
        "noise_embedding": first["noise_embedding"],
        "target_hidden": first["target_hidden"],
        "position_ids": pos,
        "q_position_ids": pos[:, -q_len:],
    }
    save_drafter_inputs(drafter_inputs_path_for_case(case), arrays)


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
    "ctc-asr": _generate_ctc_asr,
    # Vision tasks that produce last_hidden_state — reuse image classification.
    "depth-estimation": _generate_image_classification,
    "image-segmentation": _generate_image_classification,
    "image-to-image": _generate_image_classification,
    "object-detection": _generate_object_detection,
    "phi4mm-multimodal": _generate_phi4mm_multimodal,
    "gemma4-assistant": _generate_gemma4_assistant,
    "dflash-draft": _generate_dflash_draft,
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
