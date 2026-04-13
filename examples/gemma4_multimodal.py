#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

r"""Gemma 4 Any-to-Any multimodal generation — text, vision, audio, and combined.

Demonstrates building ONNX models from ``google/gemma-4-E2B-it`` using
``mobius.build`` and running autoregressive generation across all supported
input modalities:

    - **Text-only**: Standard causal LM generation
    - **Vision**: Image understanding (describe, answer questions about images)
    - **Audio**: Speech transcription / audio question answering
    - **Vision + Audio**: Combined multimodal reasoning over image + speech

Gemma 4 uses a 4-model split (same pattern as Phi-4 Multimodal):

    - **Vision**:    ``pixel_values`` → ``image_features``
      (SigLIP ViT-like encoder + projector; 280 soft tokens/image)
    - **Audio**:     ``audio_features`` → ``audio_features``
      (Conformer encoder, 12 layers, hidden 1024; subsampling 4x)
    - **Embedding**: ``input_ids`` + ``image_features`` + ``audio_features``
      → ``inputs_embeds``
      (token embedding + multimodal feature fusion)
    - **Decoder**:   ``inputs_embeds`` + ``attention_mask`` +
      ``position_ids`` + KV cache → ``logits`` + present KV
      (Gemma4 text decoder with dual RoPE, GQA, optional MoE)

During prefill all four sessions run.  During decode only embedding + decoder
are used (vision and audio encoders run once per generation).

.. note::
    This script documents the Gemma 4 Any-to-Any inference API.  Run it
    against a real ``google/gemma-4-E2B-it`` checkpoint once the model
    weights are available locally.

Prerequisites::

    pip install mobius-ai[transformers]
    pip install torchaudio pillow  # for audio and image loading

Usage::

    # Run all modality demos (uses bundled test assets):
    python examples/gemma4_multimodal.py

    # Text-only generation:
    python examples/gemma4_multimodal.py --mode text --prompt "Explain gravity."

    # Vision (image + text):
    python examples/gemma4_multimodal.py --mode vision \
        --image testdata/pipeline-cat-chonk.jpeg \
        --prompt "Describe what you see."

    # Audio (speech + text):
    python examples/gemma4_multimodal.py --mode audio \
        --audio testdata/652-129742-0006.flac \
        --prompt "Transcribe the following audio."

    # Combined vision + audio:
    python examples/gemma4_multimodal.py --mode vision-audio \
        --image testdata/pipeline-cat-chonk.jpeg \
        --audio testdata/652-129742-0006.flac \
        --prompt "Describe the image and transcribe the audio."

    # Save ONNX models to disk without running inference:
    python examples/gemma4_multimodal.py --save-to output/gemma4/

    # Build graph skeleton only (no weight download):
    python examples/gemma4_multimodal.py --save-to output/gemma4/ --no-weights
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import transformers

from mobius import build
from mobius._testing.ort_inference import OnnxModelSession

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_ID = "google/gemma-4-E2B-it"
MAX_NEW_TOKENS = 128

# Gemma 4 special token IDs
# These match the token ids in the Gemma 4 tokenizer vocabulary.
# IMAGE_TOKEN_ID: placeholder inserted once per image in input_ids
IMAGE_TOKEN_ID = 255999  # <image_soft_token> (same as Gemma 3)
# AUDIO_TOKEN_ID: placeholder inserted once per audio chunk in input_ids
AUDIO_TOKEN_ID = 258881  # <audio_soft_token> — confirmed from google/gemma-4-E2B-it HF config
EOS_TOKEN_IDS = {1, 106}  # <eos> (1) and <turn|> (106, end-of-turn marker)

# Gemma 4 SigLIP vision encoder: 280 soft tokens per image after projection.
# This is set in the HF config as mm_tokens_per_image=280.
NUM_IMAGE_TOKENS = 280

# Gemma 4 Conformer audio encoder subsampling: two 2D conv layers with
# stride 2 each → total time reduction factor of 4.
AUDIO_SUBSAMPLING_FACTOR = 4

# Mel spectrogram parameters for the Gemma 4 audio encoder input
AUDIO_SAMPLE_RATE = 16_000
AUDIO_N_MELS = 128  # Gemma 4 uses 128-dim mel (vs Whisper's 80)


# ---------------------------------------------------------------------------
# Input preprocessing — one function per ONNX session
# ---------------------------------------------------------------------------


def prepare_vision_feeds(
    processor,
    image_path: str,
) -> dict[str, np.ndarray]:
    """Prepare feeds for the **vision** session.

    Loads the image and runs the HuggingFace processor to obtain
    pre-patchified pixel values and position IDs.

    Args:
        processor: ``AutoProcessor`` loaded for the Gemma 4 model.
        image_path: Path to a local image file (JPEG, PNG, etc.).

    Returns:
        ``{"pixel_values": float32[B, N, 3*P^2], "pixel_position_ids": int64[B, N, 2]}``
    """
    from PIL import Image

    img = Image.open(image_path).convert("RGB")
    # AutoProcessor for Gemma 4 returns pre-patchified pixel_values
    # [B, N, 3*P^2] and pixel_position_ids [B, N, 2] directly.
    processed = processor(images=img, return_tensors="np")
    pixel_values = processed["pixel_values"].astype(np.float32)
    pixel_position_ids = processed["pixel_position_ids"].astype(np.int64)
    return {"pixel_values": pixel_values, "pixel_position_ids": pixel_position_ids}


def prepare_audio_feeds(
    audio_path: str,
    sample_rate: int = AUDIO_SAMPLE_RATE,
    n_mels: int = AUDIO_N_MELS,
) -> dict[str, np.ndarray]:
    """Prepare feeds for the **audio** session.

    Loads the audio file, resamples to 16 kHz if needed, computes a
    128-dim log-mel spectrogram, and transposes to ``(1, time, n_mels)``
    layout expected by the Conformer encoder.

    Args:
        audio_path: Path to an audio file (WAV, FLAC, MP3, etc.).
        sample_rate: Target sample rate (16 000 Hz for Gemma 4).
        n_mels: Number of mel filterbank bins (128 for Gemma 4).

    Returns:
        ``{"audio_features": float32[1, T, n_mels]}``
    """
    import torchaudio

    waveform, sr = torchaudio.load(audio_path)
    if sr != sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, sample_rate)

    # Average to mono, convert to numpy
    audio_np = waveform.mean(dim=0).numpy().astype(np.float32)

    # Compute log-mel spectrogram using the HuggingFace feature extractor.
    # Gemma 4 uses the same interface as WhisperFeatureExtractor.
    feature_extractor = transformers.WhisperFeatureExtractor(
        feature_size=n_mels,
        sampling_rate=sample_rate,
        # Use a wider window than Whisper to match the Conformer receptive field
        hop_length=160,
        chunk_length=30,
    )
    out = feature_extractor(
        audio_np,
        sampling_rate=sample_rate,
        return_tensors="np",
        padding=False,
    )
    # out["input_features"]: [1, n_mels, time_frames]
    # Transpose to [1, time_frames, n_mels] for the Conformer encoder
    audio_features = out["input_features"].astype(np.float32).transpose(0, 2, 1)
    return {"audio_features": audio_features}  # [1, T, n_mels]


def prepare_embedding_feeds(
    input_ids: np.ndarray,
    image_features: np.ndarray,
    audio_features: np.ndarray,
) -> dict[str, np.ndarray]:
    """Prepare feeds for the **embedding** session.

    The embedding model fuses token embeddings with image and audio
    feature vectors by replacing the corresponding placeholder tokens.

    Args:
        input_ids: ``int64[1, seq_len]`` — token ids with image/audio
            placeholder tokens already inserted at the correct positions.
        image_features: ``float32[num_image_tokens, hidden_size]``, or
            ``float32[0, hidden_size]`` when no image is provided.
        audio_features: ``float32[num_audio_tokens, hidden_size]``, or
            ``float32[0, hidden_size]`` when no audio is provided.

    Returns:
        Feeds dict for the embedding ONNX model.
    """
    return {
        "input_ids": input_ids,
        "image_features": image_features,
        "audio_features": audio_features,
    }


def prepare_decoder_feeds(
    inputs_embeds: np.ndarray,
    past_seq_len: int,
    past_kv: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Prepare feeds for the **decoder** session.

    Constructs the attention mask and position ids from the current
    and accumulated past sequence lengths, then merges them with the
    existing KV cache.

    Args:
        inputs_embeds: ``float32[1, cur_seq_len, hidden_size]``.
        past_seq_len: Number of tokens already stored in the KV cache.
        past_kv: Mapping of ``"past_key_values.{i}.key/value"`` arrays
            from the previous decoder step.

    Returns:
        Complete feeds dict for the decoder ONNX model.
    """
    batch_size, cur_seq_len, _ = inputs_embeds.shape
    total_seq_len = past_seq_len + cur_seq_len

    return {
        "inputs_embeds": inputs_embeds,
        # Attend to all tokens (past + current)
        "attention_mask": np.ones((batch_size, total_seq_len), dtype=np.int64),
        "position_ids": np.arange(past_seq_len, total_seq_len, dtype=np.int64)[np.newaxis, :],
        **past_kv,
    }


# ---------------------------------------------------------------------------
# Token construction helpers
# ---------------------------------------------------------------------------


def _tokenize(tokenizer, prompt: str) -> np.ndarray:
    """Tokenize *prompt* and return ``int64[1, seq_len]``."""
    return tokenizer(prompt, return_tensors="np")["input_ids"].astype(np.int64)


def build_input_ids(
    tokenizer,
    prompt: str,
    *,
    num_image_tokens: int = 0,
    num_audio_tokens: int = 0,
) -> np.ndarray:
    r"""Tokenize a prompt using the chat template and insert modality placeholder tokens.

    Applies the Gemma 4 instruction chat template so the instruction-tuned
    model receives properly formatted input.  Modality placeholder tokens
    are inserted at the start of the user message content (before the text),
    matching the HuggingFace processor layout::

        <bos><|turn>user\\n[image*N][audio*M]text<turn|>\\n<|turn>model\\n

    Args:
        tokenizer: HuggingFace tokenizer with ``apply_chat_template`` support.
        prompt: The user's text prompt.
        num_image_tokens: Number of image soft tokens to insert (0 = no image).
        num_audio_tokens: Number of audio soft tokens to insert (0 = no audio).

    Returns:
        ``int64[1, seq_len]`` token ids.
    """
    # Apply instruction chat template: <bos><|turn>user\\n{prompt}<turn|>\\n<|turn>model\\n
    messages = [{"role": "user", "content": prompt}]
    template_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    template_ids = tokenizer(template_text, return_tensors="np")["input_ids"].astype(np.int64)

    if num_image_tokens == 0 and num_audio_tokens == 0:
        return template_ids

    # Build modality placeholder block
    modality_parts: list[np.ndarray] = []
    if num_image_tokens > 0:
        modality_parts.append(np.full((1, num_image_tokens), IMAGE_TOKEN_ID, dtype=np.int64))
    if num_audio_tokens > 0:
        modality_parts.append(np.full((1, num_audio_tokens), AUDIO_TOKEN_ID, dtype=np.int64))
    modality_ids = np.concatenate(modality_parts, axis=1)

    # Find insertion point: right after the user header "<|turn>user\n"
    # That sequence is token 105 (<|turn>), 2364 (user), 107 (\n)
    flat = template_ids[0].tolist()
    insert_pos = 1  # fallback: right after BOS
    for i in range(len(flat) - 2):
        if flat[i] == 105 and flat[i + 1] == 2364 and flat[i + 2] == 107:
            insert_pos = i + 3
            break

    prefix = template_ids[:, :insert_pos]
    suffix = template_ids[:, insert_pos:]
    return np.concatenate([prefix, modality_ids, suffix], axis=1)  # [1, seq_len]


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _empty_features(hidden_size: int) -> np.ndarray:
    """Return a ``float32[0, hidden_size]`` zero tensor (no modality tokens)."""
    return np.zeros((0, hidden_size), dtype=np.float32)


def _init_kv_cache(config) -> dict[str, np.ndarray]:
    """Create an empty KV cache for all independent decoder layers.

    Gemma 4 uses:
    - ``head_dim`` for local (sliding_attention) layers
    - ``global_head_dim`` for global (full_attention) layers
    - ``num_kv_shared_layers`` trailing layers that share KV from earlier layers
      (these have NO independent cache entries)
    """
    num_kv_shared = getattr(config, "num_kv_shared_layers", 0) or 0
    num_kv_layers = config.num_hidden_layers - num_kv_shared
    layer_types = (
        getattr(config, "layer_types", None)
        or ["sliding_attention"] * config.num_hidden_layers
    )
    local_hd = config.head_dim
    global_hd = getattr(config, "global_head_dim", None) or local_hd

    past_kv: dict[str, np.ndarray] = {}
    for i in range(num_kv_layers):
        layer_type = layer_types[i] if i < len(layer_types) else "sliding_attention"
        hd = global_hd if layer_type == "full_attention" else local_hd
        past_kv[f"past_key_values.{i}.key"] = np.zeros(
            (1, config.num_key_value_heads, 0, hd),
            dtype=np.float32,
        )
        past_kv[f"past_key_values.{i}.value"] = np.zeros(
            (1, config.num_key_value_heads, 0, hd),
            dtype=np.float32,
        )
    return past_kv


def _update_kv_cache(
    past_kv: dict[str, np.ndarray],
    outputs: dict[str, np.ndarray],
    num_layers: int,
) -> None:
    """Copy ``present.{i}.key/value`` from decoder outputs into *past_kv*."""
    for i in range(num_layers):
        past_kv[f"past_key_values.{i}.key"] = outputs[f"present.{i}.key"]
        past_kv[f"past_key_values.{i}.value"] = outputs[f"present.{i}.value"]


# ---------------------------------------------------------------------------
# Generation loop
# ---------------------------------------------------------------------------


def generate(
    vision_session: OnnxModelSession | None,
    audio_session: OnnxModelSession | None,
    embedding_session: OnnxModelSession,
    decoder_session: OnnxModelSession,
    tokenizer,
    input_ids: np.ndarray,
    image_features: np.ndarray,
    audio_features: np.ndarray,
    config,
    *,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> str:
    """Greedy autoregressive generation with KV cache and token streaming.

    Chains up to four ONNX sessions in two phases:

    **Prefill** (first step — processes the entire prompt):

    1. *vision*    → ``image_features`` (if image present)
    2. *audio*     → ``audio_features`` (if audio present)
    3. *embedding* → ``inputs_embeds`` (fuses text + modality features)
    4. *decoder*   → ``logits`` + initial KV cache

    **Decode** (subsequent steps — generates one token at a time):

    1. *embedding* → ``inputs_embeds`` (text token only; no modality features)
    2. *decoder*   → ``logits`` + updated KV cache

    Args:
        vision_session:   ONNX session for the SigLIP vision encoder.
            Pass ``None`` if no image is provided.
        audio_session:    ONNX session for the Conformer audio encoder.
            Pass ``None`` if no audio is provided.
        embedding_session: ONNX session for the embedding + fusion model.
        decoder_session:  ONNX session for the Gemma4 text decoder.
        tokenizer: HuggingFace tokenizer for decoding generated ids.
        input_ids: ``int64[1, seq_len]`` — prompt tokens with placeholders.
        image_features: ``float32[num_image_tokens, hidden]`` or
            ``float32[0, hidden]`` when no image.
        audio_features: ``float32[num_audio_tokens, hidden]`` or
            ``float32[0, hidden]`` when no audio.
        config: :class:`~mobius.Gemma4Config` for model dimensions.
        max_new_tokens: Maximum number of new tokens to generate.

    Returns:
        The generated text, decoded from token ids (prompt excluded).
    """
    num_kv_shared = getattr(config, "num_kv_shared_layers", 0) or 0
    num_kv_layers = config.num_hidden_layers - num_kv_shared
    hidden_size = config.hidden_size

    # Zero-length feature tensors used during decode steps (no modality input)
    zero_image = _empty_features(hidden_size)
    zero_audio = _empty_features(hidden_size)

    past_kv = _init_kv_cache(config)
    cur_ids = input_ids
    past_seq_len = 0
    generated_ids: list[int] = []

    for step in range(max_new_tokens):
        is_prefill = step == 0

        # ---- Embedding session ----
        # On the first step, pass real multimodal features so the embedding
        # model can splice image/audio tokens into the hidden states.
        # On subsequent decode steps pass zero-length tensors (no modality).
        embed_out = embedding_session.run(
            prepare_embedding_feeds(
                cur_ids,
                image_features if is_prefill else zero_image,
                audio_features if is_prefill else zero_audio,
            )
        )
        inputs_embeds: np.ndarray = embed_out["inputs_embeds"]

        # ---- Decoder session ----
        decoder_out = decoder_session.run(
            prepare_decoder_feeds(inputs_embeds, past_seq_len, past_kv)
        )

        # Greedy: pick the token with the highest logit at the last position
        next_token = int(np.argmax(decoder_out["logits"][:, -1, :]))
        generated_ids.append(next_token)

        # Stream the decoded piece to stdout
        piece = tokenizer.decode([next_token], skip_special_tokens=True)
        print(piece, end="", flush=True)

        if next_token in EOS_TOKEN_IDS:
            break

        # Advance KV cache and prepare single-token input for next step
        _update_kv_cache(past_kv, decoder_out, num_kv_layers)
        past_seq_len += cur_ids.shape[1]
        cur_ids = np.array([[next_token]], dtype=np.int64)

    print()  # newline after streamed output
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Modality demo functions
# ---------------------------------------------------------------------------


def demo_text(
    embedding_session: OnnxModelSession,
    decoder_session: OnnxModelSession,
    tokenizer,
    config,
    prompt: str = "Explain the theory of general relativity in simple terms.",
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> str:
    """Text-only generation demo."""
    print("\n" + "=" * 64)
    print("📝  TEXT-ONLY GENERATION")
    print("=" * 64)
    print(f"Prompt: {prompt}")
    print("-" * 64)

    input_ids = build_input_ids(tokenizer, prompt)
    zero = _empty_features(config.hidden_size)

    return generate(
        vision_session=None,
        audio_session=None,
        embedding_session=embedding_session,
        decoder_session=decoder_session,
        tokenizer=tokenizer,
        input_ids=input_ids,
        image_features=zero,
        audio_features=zero,
        config=config,
        max_new_tokens=max_new_tokens,
    )


def demo_vision(
    vision_session: OnnxModelSession,
    embedding_session: OnnxModelSession,
    decoder_session: OnnxModelSession,
    tokenizer,
    processor,
    config,
    image_path: str,
    prompt: str = "Describe what you see in this image in detail.",
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> str:
    """Vision (image + text) generation demo."""
    print("\n" + "=" * 64)
    print("🖼️   VISION GENERATION")
    print("=" * 64)
    print(f"Image:  {image_path}")
    print(f"Prompt: {prompt}")
    print("-" * 64)

    # Step 1: Encode the image through the SigLIP vision encoder.
    # Output: image_features [1, num_image_tokens, hidden_size]
    vision_out = vision_session.run(prepare_vision_feeds(processor, image_path))
    image_features: np.ndarray = vision_out["image_features"]
    # Squeeze the batch dimension: [1, T, H] → [T, H]
    if image_features.ndim == 3:
        image_features = image_features[0]

    # Step 2: Build input_ids with IMAGE_TOKEN_ID placeholders.
    # One placeholder per soft token produced by the vision encoder.
    num_image_tokens = image_features.shape[0]
    input_ids = build_input_ids(tokenizer, prompt, num_image_tokens=num_image_tokens)

    return generate(
        vision_session=vision_session,
        audio_session=None,
        embedding_session=embedding_session,
        decoder_session=decoder_session,
        tokenizer=tokenizer,
        input_ids=input_ids,
        image_features=image_features,
        audio_features=_empty_features(config.hidden_size),
        config=config,
        max_new_tokens=max_new_tokens,
    )


def demo_audio(
    audio_session: OnnxModelSession,
    embedding_session: OnnxModelSession,
    decoder_session: OnnxModelSession,
    tokenizer,
    config,
    audio_path: str,
    prompt: str = "Transcribe the following audio.",
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> str:
    """Audio (speech + text) generation demo."""
    print("\n" + "=" * 64)
    print("🔊  AUDIO GENERATION")
    print("=" * 64)
    print(f"Audio:  {audio_path}")
    print(f"Prompt: {prompt}")
    print("-" * 64)

    # Step 1: Encode audio through the Conformer encoder.
    # Input:  audio_features [1, T, n_mels]  (mel spectrogram)
    # Output: audio_features [1, T', hidden_size]  (T' = T / subsampling_factor)
    audio_out = audio_session.run(prepare_audio_feeds(audio_path))
    audio_features: np.ndarray = audio_out["audio_features"]
    if audio_features.ndim == 3:
        audio_features = audio_features[0]  # [T', hidden_size]

    # Step 2: Build input_ids with AUDIO_TOKEN_ID placeholders.
    num_audio_tokens = audio_features.shape[0]
    input_ids = build_input_ids(tokenizer, prompt, num_audio_tokens=num_audio_tokens)

    return generate(
        vision_session=None,
        audio_session=audio_session,
        embedding_session=embedding_session,
        decoder_session=decoder_session,
        tokenizer=tokenizer,
        input_ids=input_ids,
        image_features=_empty_features(config.hidden_size),
        audio_features=audio_features,
        config=config,
        max_new_tokens=max_new_tokens,
    )


def demo_vision_audio(
    vision_session: OnnxModelSession,
    audio_session: OnnxModelSession,
    embedding_session: OnnxModelSession,
    decoder_session: OnnxModelSession,
    tokenizer,
    processor,
    config,
    image_path: str,
    audio_path: str,
    prompt: str = "Describe the image and transcribe the audio.",
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> str:
    """Combined vision + audio generation demo."""
    print("\n" + "=" * 64)
    print("🖼️ + 🔊  VISION + AUDIO GENERATION")
    print("=" * 64)
    print(f"Image:  {image_path}")
    print(f"Audio:  {audio_path}")
    print(f"Prompt: {prompt}")
    print("-" * 64)

    # Step 1: Encode image
    vision_out = vision_session.run(prepare_vision_feeds(processor, image_path))
    image_features: np.ndarray = vision_out["image_features"]
    if image_features.ndim == 3:
        image_features = image_features[0]

    # Step 2: Encode audio
    audio_out = audio_session.run(prepare_audio_feeds(audio_path))
    audio_features: np.ndarray = audio_out["audio_features"]
    if audio_features.ndim == 3:
        audio_features = audio_features[0]

    # Step 3: Build input_ids — image placeholders first, then audio
    input_ids = build_input_ids(
        tokenizer,
        prompt,
        num_image_tokens=image_features.shape[0],
        num_audio_tokens=audio_features.shape[0],
    )

    return generate(
        vision_session=vision_session,
        audio_session=audio_session,
        embedding_session=embedding_session,
        decoder_session=decoder_session,
        tokenizer=tokenizer,
        input_ids=input_ids,
        image_features=image_features,
        audio_features=audio_features,
        config=config,
        max_new_tokens=max_new_tokens,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Gemma 4 Any-to-Any multimodal generation with ONNX Runtime "
            "— text, vision, audio, and combined."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model-id",
        default=MODEL_ID,
        help="HuggingFace model ID (default: %(default)s).",
    )
    parser.add_argument(
        "--mode",
        choices=["text", "vision", "audio", "vision-audio", "all"],
        default="all",
        help=(
            "Which modality demo to run (default: %(default)s). "
            "'all' runs all demos that have the required assets."
        ),
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Text prompt (default depends on --mode).",
    )
    parser.add_argument(
        "--image",
        default="testdata/pipeline-cat-chonk.jpeg",
        metavar="PATH",
        help="Path to image file used for vision demos (default: %(default)s).",
    )
    parser.add_argument(
        "--audio",
        default="testdata/652-129742-0006.flac",
        metavar="PATH",
        help="Path to audio file used for audio demos (default: %(default)s).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=MAX_NEW_TOKENS,
        help="Maximum tokens to generate per demo (default: %(default)s).",
    )
    parser.add_argument(
        "--save-to",
        metavar="DIR",
        default=None,
        help="Save all ONNX models to DIR and exit (skips inference).",
    )
    parser.add_argument(
        "--no-weights",
        action="store_true",
        help="Build ONNX graph skeletons only — do not download model weights.",
    )
    parser.add_argument(
        "--dtype",
        choices=["f32", "f16"],
        default="f32",
        help="Weight/activation dtype to use (default: %(default)s).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # ------------------------------------------------------------------
    # Step 1: Build the ONNX model package from HuggingFace weights.
    #
    # mobius.build() downloads the model config and (optionally) weights,
    # constructs the ONNX graphs using the registered Gemma4MultiModalModel,
    # and returns a ModelPackage containing all sub-models.
    # ------------------------------------------------------------------
    load_weights = not args.no_weights
    print(f"Building ONNX models from {args.model_id!r} (dtype={args.dtype}) ...")
    pkg = build(args.model_id, dtype=args.dtype, load_weights=load_weights)
    config = pkg.config
    print(f"Package components: {list(pkg.keys())}")
    print(
        f"Model dimensions: hidden={config.hidden_size}, "
        f"layers={config.num_hidden_layers}, "
        f"kv_heads={config.num_key_value_heads}"
    )

    if args.save_to:
        # Persist ONNX models to disk so they can be loaded by other runtimes
        # (e.g. onnxruntime-genai) without rebuilding.
        pkg.save(args.save_to, check_weights=load_weights)
        print(f"Saved ONNX models to {args.save_to!r}")
        return 0

    # ------------------------------------------------------------------
    # Step 2: Create ONNX Runtime inference sessions for each sub-model.
    #
    # Gemma 4 Any-to-Any models produce 4 ONNX graphs (decoder, vision, audio,
    # embedding). Vision-language-only models produce 3 (no audio). Both cases
    # are handled — audio_session is None when the model has no audio component.
    # ------------------------------------------------------------------
    print("\nCreating ONNX Runtime sessions ...")
    vision_session = OnnxModelSession(pkg["vision"])
    audio_session = OnnxModelSession(pkg["audio"]) if "audio" in pkg else None
    embedding_session = OnnxModelSession(pkg["embedding"])
    decoder_session = OnnxModelSession(pkg["decoder"])

    # ------------------------------------------------------------------
    # Step 3: Load the HuggingFace processor.
    #
    # AutoProcessor provides:
    #   - image_processor: resizes and normalises images for SigLIP
    #   - tokenizer:       converts text to/from token ids
    # ------------------------------------------------------------------
    print(f"Loading processor from {args.model_id!r} ...")
    processor = transformers.AutoProcessor.from_pretrained(
        args.model_id, trust_remote_code=True
    )
    tokenizer = processor.tokenizer

    # ------------------------------------------------------------------
    # Step 4: Check which assets are available and run the requested demos.
    #
    # Vision and audio demos are silently skipped if the required file
    # does not exist (allows running in text-only environments).
    # ------------------------------------------------------------------
    import os

    has_image = os.path.exists(args.image)
    has_audio = os.path.exists(args.audio) and audio_session is not None

    if args.mode in ("vision", "vision-audio", "all") and not has_image:
        print(
            f"⚠️  Image file not found: {args.image!r} — "
            "vision demos will be skipped.  Pass --image PATH to provide one."
        )
    if args.mode in ("audio", "vision-audio", "all") and audio_session is None:
        print("⚠️  Model has no audio component — audio demos will be skipped.")
    elif args.mode in ("audio", "vision-audio", "all") and not has_audio:
        print(
            f"⚠️  Audio file not found: {args.audio!r} — "
            "audio demos will be skipped.  Pass --audio PATH to provide one."
        )

    max_tokens = args.max_new_tokens
    modes = ["text", "vision", "audio", "vision-audio"] if args.mode == "all" else [args.mode]

    for mode in modes:
        if mode == "text":
            demo_text(
                embedding_session=embedding_session,
                decoder_session=decoder_session,
                tokenizer=tokenizer,
                config=config,
                prompt=args.prompt
                or ("Explain the theory of general relativity in simple terms."),
                max_new_tokens=max_tokens,
            )

        elif mode == "vision":
            if not has_image:
                continue
            demo_vision(
                vision_session=vision_session,
                embedding_session=embedding_session,
                decoder_session=decoder_session,
                tokenizer=tokenizer,
                processor=processor,
                config=config,
                image_path=args.image,
                prompt=args.prompt or "Describe what you see in this image in detail.",
                max_new_tokens=max_tokens,
            )

        elif mode == "audio":
            if not has_audio:
                continue
            demo_audio(
                audio_session=audio_session,
                embedding_session=embedding_session,
                decoder_session=decoder_session,
                tokenizer=tokenizer,
                config=config,
                audio_path=args.audio,
                prompt=args.prompt or "Transcribe the following audio.",
                max_new_tokens=max_tokens,
            )

        elif mode == "vision-audio":
            if not has_image or not has_audio:
                continue
            demo_vision_audio(
                vision_session=vision_session,
                audio_session=audio_session,
                embedding_session=embedding_session,
                decoder_session=decoder_session,
                tokenizer=tokenizer,
                processor=processor,
                config=config,
                image_path=args.image,
                audio_path=args.audio,
                prompt=args.prompt or "Describe the image and transcribe the audio.",
                max_new_tokens=max_tokens,
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
