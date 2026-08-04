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
      (SigLIP ViT-like encoder + projector; ~256 soft tokens/image)
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

    pip install mobius-onnx[transformers]
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

    # Compare ONNX Runtime vs HuggingFace PyTorch output side-by-side:
    python examples/gemma4_multimodal.py --mode text --compare-hf
    python examples/gemma4_multimodal.py --mode vision --compare-hf
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
# IMAGE_TOKEN_ID: placeholder inserted N times (once per soft token) in input_ids
IMAGE_TOKEN_ID = 258880  # <|image|>  — Gemma 4 image soft token (config.image_token_id)
IMAGE_OPEN_TOKEN_ID = 255999  # <|image>   — opening boundary marker before image tokens
IMAGE_CLOSE_TOKEN_ID = 258882  # <image|>   — closing boundary marker after image tokens
# AUDIO_TOKEN_ID: placeholder inserted N times (once per audio frame) in input_ids
AUDIO_TOKEN_ID = 258881  # <|audio|>  — confirmed from google/gemma-4-E2B-it HF config
AUDIO_OPEN_TOKEN_ID = 256000  # <|audio>   — opening boundary marker before audio tokens
AUDIO_CLOSE_TOKEN_ID = 258883  # <audio|>   — closing boundary marker after audio tokens
EOS_TOKEN_IDS = {1, 106}  # <eos> (1) and <turn|> (106, end-of-turn marker)

# Gemma 4 SigLIP vision encoder: default output length from vc.default_output_length.
# The actual number of soft tokens depends on the input image resolution and
# pooling kernel size; computed dynamically from image_features.shape[0] at runtime.
NUM_IMAGE_TOKENS = 256  # approximate; actual count is image-size dependent

# Gemma 4 Conformer audio encoder subsampling: two 2D conv layers with
# stride 2 each → total time reduction factor of 4.
AUDIO_SUBSAMPLING_FACTOR = 4


# ---------------------------------------------------------------------------
# Input preprocessing — one function per ONNX session
# ---------------------------------------------------------------------------


def prepare_vision_feeds(
    processor,
    image_path: str,
    np_dtype=np.float32,
) -> dict[str, np.ndarray]:
    """Prepare feeds for the **vision** session.

    Loads the image and runs the HuggingFace processor to obtain
    pre-patchified pixel values and position IDs.

    Args:
        processor: ``AutoProcessor`` loaded for the Gemma 4 model.
        image_path: Path to a local image file (JPEG, PNG, etc.).
        np_dtype: Numpy dtype for pixel values (default float32).

    Returns:
        ``{"pixel_values": [B, N, 3*P^2], "pixel_position_ids": int64[B, N, 2]}``
    """
    from PIL import Image

    img = Image.open(image_path).convert("RGB")
    # Gemma 4 processor requires a text argument alongside images.
    # Use a placeholder image token so the processor computes correct dims.
    processed = processor(images=img, text="<image>", return_tensors="np")
    pixel_values = processed["pixel_values"].astype(np_dtype)
    # The HF processor returns "image_position_ids"; our ONNX vision model input is "pixel_position_ids"
    pixel_position_ids = processed["image_position_ids"].astype(np.int64)
    return {"pixel_values": pixel_values, "pixel_position_ids": pixel_position_ids}


def prepare_audio_feeds(
    processor,
    audio_path: str,
    np_dtype=np.float32,
) -> dict[str, np.ndarray]:
    """Prepare feeds for the **audio** session.

    Loads the audio file and uses the Gemma 4 processor's built-in
    ``Gemma4AudioFeatureExtractor`` to compute the 128-dim log-mel
    spectrogram in ``(1, time, n_mels)`` layout expected by the
    Conformer encoder.

    Args:
        processor: ``AutoProcessor`` loaded for the Gemma 4 model.
        audio_path: Path to an audio file (WAV, FLAC, MP3, etc.).
        np_dtype: Numpy dtype for audio features (default float32).

    Returns:
        ``{"input_features": [1, T, n_mels], "input_features_mask": [1, T]}``
    """
    import soundfile as sf

    raw, sr = sf.read(audio_path, always_2d=True)  # [frames, channels]
    # Average channels to mono
    audio_np = raw.mean(axis=1).astype(np.float32)

    # Use the processor's Gemma4AudioFeatureExtractor for correct mel computation.
    # The feature extractor handles resampling internally.
    fe = processor.feature_extractor
    out = fe(
        [audio_np],
        sampling_rate=sr,
        return_tensors="np",
        padding=False,
    )
    # out["input_features"]: [1, T, n_mels]  (already in correct layout)
    audio_features = out["input_features"].astype(np_dtype)
    # All-True mask for single-clip inference (no padding to mask out).
    input_features_mask = np.ones(audio_features.shape[:2], dtype=np.bool_)  # [1, T]
    return {
        "input_features": audio_features,
        "input_features_mask": input_features_mask,
    }


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
    input_ids: np.ndarray,
    per_layer_inputs: np.ndarray | None = None,
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
        input_ids: ``int64[1, cur_seq_len]`` — original token ids (legacy
            decoder input, retained for backward compat). On builds where
            the embedding model emits ``per_layer_inputs`` the decoder
            does not consume ``input_ids``.
        per_layer_inputs: Optional ``[1, cur_seq_len, num_layers *
            per_layer_dim]`` tensor with the same dtype as
            ``inputs_embeds`` (that is, the model/config dtype), emitted
            by the embedding model when
            ``hidden_size_per_layer_input > 0``. The decoder requires
            this input on every step (prefill + decode).

    Returns:
        Complete feeds dict for the decoder ONNX model.
    """
    batch_size, cur_seq_len, _ = inputs_embeds.shape
    total_seq_len = past_seq_len + cur_seq_len

    feeds: dict[str, np.ndarray] = {
        "inputs_embeds": inputs_embeds,
        # Attend to all tokens (past + current)
        "attention_mask": np.ones((batch_size, total_seq_len), dtype=np.int64),
        "position_ids": np.arange(past_seq_len, total_seq_len, dtype=np.int64)[np.newaxis, :],
        "input_ids": input_ids,
        **past_kv,
    }
    if per_layer_inputs is not None:
        feeds["per_layer_inputs"] = per_layer_inputs
    return feeds


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

        <bos><|turn>user\\n<|image>[imagexN]<image|>[audioxM]text<turn|>\\n<|turn>model\\n

    Image tokens are wrapped in boundary markers ``<|image>`` (255999) and
    ``<image|>`` (258882) — the HuggingFace processor always inserts these, and
    the model has been trained to expect them around the soft image tokens.

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
        # Wrap image soft tokens with boundary markers, matching HF processor layout:
        # <|image>(255999) + Nx<|image|>(258880) + <image|>(258882)
        open_marker = np.array([[IMAGE_OPEN_TOKEN_ID]], dtype=np.int64)
        soft_tokens = np.full((1, num_image_tokens), IMAGE_TOKEN_ID, dtype=np.int64)
        close_marker = np.array([[IMAGE_CLOSE_TOKEN_ID]], dtype=np.int64)
        modality_parts.extend([open_marker, soft_tokens, close_marker])
    if num_audio_tokens > 0:
        # Wrap audio soft tokens with boundary markers, matching HF processor layout:
        # <|audio>(256000) + Nx<|audio|>(258881) + <audio|>(258883)
        audio_open = np.array([[AUDIO_OPEN_TOKEN_ID]], dtype=np.int64)
        audio_soft = np.full((1, num_audio_tokens), AUDIO_TOKEN_ID, dtype=np.int64)
        audio_close = np.array([[AUDIO_CLOSE_TOKEN_ID]], dtype=np.int64)
        modality_parts.extend([audio_open, audio_soft, audio_close])
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


def _empty_features(hidden_size: int, dtype=np.float32) -> np.ndarray:
    """Return a zero-length feature tensor ``[0, hidden_size]``."""
    return np.zeros((0, hidden_size), dtype=dtype)


def _init_kv_cache(config, dtype=np.float32) -> dict[str, np.ndarray]:
    """Create an empty KV cache for all independent decoder layers.

    Gemma 4 uses:
    - ``head_dim`` for local (sliding_attention) layers
    - ``global_head_dim`` for global (full_attention) layers
    - ``num_kv_shared_layers`` trailing layers that share KV from earlier layers
      (these have NO independent cache entries)

    All layers use the original ``num_key_value_heads`` (no expansion).
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
        kv_heads = config.num_key_value_heads
        past_kv[f"past_key_values.{i}.key"] = np.zeros(
            (1, kv_heads, 0, hd),
            dtype=dtype,
        )
        past_kv[f"past_key_values.{i}.value"] = np.zeros(
            (1, kv_heads, 0, hd),
            dtype=dtype,
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
    feat_dtype = image_features.dtype
    zero_image = _empty_features(hidden_size, dtype=feat_dtype)
    zero_audio = _empty_features(hidden_size, dtype=feat_dtype)

    cur_ids = input_ids
    past_seq_len = 0
    generated_ids: list[int] = []

    past_kv = _init_kv_cache(config, dtype=image_features.dtype)

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
        # Gemma4 builds with hidden_size_per_layer_input > 0 also emit a
        # second embedding output (``per_layer_inputs``) that the decoder
        # consumes on every step. Pass it through transparently when present.
        per_layer_inputs = embed_out.get("per_layer_inputs")

        # ---- Decoder session ----
        decoder_out = decoder_session.run(
            prepare_decoder_feeds(
                inputs_embeds, past_seq_len, past_kv, cur_ids, per_layer_inputs
            )
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
    model_np_dtype=np.float32,
) -> str:
    """Text-only generation demo."""
    print("\n" + "=" * 64)
    print("📝  TEXT-ONLY GENERATION")
    print("=" * 64)
    print(f"Prompt: {prompt}")
    print("-" * 64)

    input_ids = build_input_ids(tokenizer, prompt)
    zero = _empty_features(config.hidden_size, dtype=model_np_dtype)

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
    model_np_dtype=np.float32,
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
    vision_out = vision_session.run(
        prepare_vision_feeds(processor, image_path, np_dtype=model_np_dtype)
    )
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
        audio_features=_empty_features(config.hidden_size, dtype=image_features.dtype),
        config=config,
        max_new_tokens=max_new_tokens,
    )


def demo_audio(
    audio_session: OnnxModelSession,
    embedding_session: OnnxModelSession,
    decoder_session: OnnxModelSession,
    tokenizer,
    processor,
    config,
    audio_path: str,
    prompt: str = "Transcribe the following audio.",
    max_new_tokens: int = MAX_NEW_TOKENS,
    model_np_dtype=np.float32,
) -> str:
    """Audio (speech + text) generation demo."""
    print("\n" + "=" * 64)
    print("🔊  AUDIO GENERATION")
    print("=" * 64)
    print(f"Audio:  {audio_path}")
    print(f"Prompt: {prompt}")
    print("-" * 64)

    # Step 1: Encode audio through the Conformer encoder.
    # Input:  input_features [1, T, n_mels], input_features_mask [1, T]
    # Output: audio_features [1, T', hidden_size], audio_features_mask [1, T']
    #         where T' = T / 4 (two conv layers with stride 2)
    audio_out = audio_session.run(
        prepare_audio_feeds(processor, audio_path, np_dtype=model_np_dtype)
    )
    audio_features: np.ndarray = audio_out["audio_features"]
    audio_mask: np.ndarray = audio_out["audio_features_mask"]
    if audio_features.ndim == 3:
        # Strip padding using the downsampled mask, then flatten batch dim.
        valid = audio_mask[0].astype(bool)  # [T']
        audio_features = audio_features[0][valid]  # [num_valid, hidden_size]

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
        image_features=_empty_features(config.hidden_size, dtype=audio_features.dtype),
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
    model_np_dtype=np.float32,
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
    vision_out = vision_session.run(
        prepare_vision_feeds(processor, image_path, np_dtype=model_np_dtype)
    )
    image_features: np.ndarray = vision_out["image_features"]
    if image_features.ndim == 3:
        image_features = image_features[0]

    # Step 2: Encode audio
    audio_out = audio_session.run(
        prepare_audio_feeds(processor, audio_path, np_dtype=model_np_dtype)
    )
    audio_features: np.ndarray = audio_out["audio_features"]
    audio_mask: np.ndarray = audio_out["audio_features_mask"]
    if audio_features.ndim == 3:
        valid = audio_mask[0].astype(bool)
        audio_features = audio_features[0][valid]

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
# HuggingFace comparison
# ---------------------------------------------------------------------------


def _hf_generate_text(model_id: str, prompt: str, max_new_tokens: int) -> str:
    """Run text-only generation with HuggingFace PyTorch and return the output."""
    import torch
    from transformers import AutoProcessor, Gemma4ForConditionalGeneration

    processor = AutoProcessor.from_pretrained(model_id)
    model = Gemma4ForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float32)
    model.eval()

    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=text, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    prompt_len = inputs["input_ids"].shape[1]
    return processor.decode(out[0][prompt_len:], skip_special_tokens=True)


def _hf_generate_vision(
    model_id: str, image_path: str, prompt: str, max_new_tokens: int
) -> str:
    """Run vision generation with HuggingFace PyTorch and return the output."""
    import torch
    from PIL import Image
    from transformers import AutoProcessor, Gemma4ForConditionalGeneration

    processor = AutoProcessor.from_pretrained(model_id)
    model = Gemma4ForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float32)
    model.eval()

    image = Image.open(image_path).convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": prompt}],
        }
    ]
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=text, images=image, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    prompt_len = inputs["input_ids"].shape[1]
    return processor.decode(out[0][prompt_len:], skip_special_tokens=True)


def _hf_generate_audio(
    model_id: str, audio_path: str, prompt: str, max_new_tokens: int
) -> str:
    """Run audio generation with HuggingFace PyTorch and return the output."""
    import soundfile as sf
    import torch
    from transformers import AutoProcessor, Gemma4ForConditionalGeneration

    processor = AutoProcessor.from_pretrained(model_id)
    model = Gemma4ForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float32)
    model.eval()

    raw, sr = sf.read(audio_path, always_2d=True)
    audio_np = raw.mean(axis=1).astype(np.float32)

    messages = [
        {
            "role": "user",
            "content": [{"type": "audio"}, {"type": "text", "text": prompt}],
        }
    ]
    text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=text, audio=audio_np, sampling_rate=sr, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    prompt_len = inputs["input_ids"].shape[1]
    return processor.decode(out[0][prompt_len:], skip_special_tokens=True)


def _hf_generate_vision_audio(
    model_id: str,
    image_path: str,
    audio_path: str,
    prompt: str,
    max_new_tokens: int,
) -> str:
    """Run combined vision + audio generation with HuggingFace PyTorch."""
    import soundfile as sf
    import torch
    from PIL import Image
    from transformers import AutoProcessor, Gemma4ForConditionalGeneration

    processor = AutoProcessor.from_pretrained(model_id)
    model = Gemma4ForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float32)
    model.eval()

    image = Image.open(image_path).convert("RGB")
    raw, sr = sf.read(audio_path, always_2d=True)
    audio_np = raw.mean(axis=1).astype(np.float32)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "audio"},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = processor(
        text=text,
        images=image,
        audio=audio_np,
        sampling_rate=sr,
        return_tensors="pt",
    )
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    prompt_len = inputs["input_ids"].shape[1]
    return processor.decode(out[0][prompt_len:], skip_special_tokens=True)


def _print_side_by_side(label: str, onnx_out: str, hf_out: str) -> None:
    """Print ONNX and HuggingFace outputs side by side for easy comparison."""
    divider = "─" * 64
    print(f"\n{'=' * 64}")
    print(f"  COMPARISON: {label}")
    print(f"{'=' * 64}")
    print(f"\n[ONNX Runtime]\n{divider}")
    print(onnx_out.strip())
    print(f"\n[HuggingFace PyTorch]\n{divider}")
    print(hf_out.strip())
    print()


def run_compare_hf(
    model_id: str,
    onnx_outputs: dict[str, str],
    has_image: bool,
    has_audio: bool,
    image_path: str,
    audio_path: str,
    max_new_tokens: int,
    text_prompt: str,
    vision_prompt: str,
    audio_prompt: str,
    vision_audio_prompt: str,
) -> None:
    """Run HF PyTorch inference for each completed ONNX demo and compare outputs.

    Args:
        model_id: HuggingFace model ID.
        onnx_outputs: Mapping of mode name (``"text"``, ``"vision"``,
            ``"audio"``, ``"vision-audio"``) to ONNX text output.
        has_image: Whether the image asset is available.
        has_audio: Whether audio is available (file exists and model has audio).
        image_path: Path to the image file.
        audio_path: Path to the audio file.
        max_new_tokens: Max tokens for generation.
        text_prompt: Prompt used for text demo.
        vision_prompt: Prompt used for vision demo.
        audio_prompt: Prompt used for audio demo.
        vision_audio_prompt: Prompt used for vision-audio demo.
    """
    print("\n" + "=" * 64)
    print("🔍  --compare-hf: loading HuggingFace model for comparison ...")
    print("=" * 64)

    if "text" in onnx_outputs:
        print("Running HF text generation ...")
        hf_text = _hf_generate_text(model_id, text_prompt, max_new_tokens)
        _print_side_by_side("TEXT", onnx_outputs["text"], hf_text)

    if "vision" in onnx_outputs and has_image:
        print("Running HF vision generation ...")
        hf_vision = _hf_generate_vision(model_id, image_path, vision_prompt, max_new_tokens)
        _print_side_by_side("VISION", onnx_outputs["vision"], hf_vision)

    if "audio" in onnx_outputs and has_audio:
        print("Running HF audio generation ...")
        hf_audio = _hf_generate_audio(model_id, audio_path, audio_prompt, max_new_tokens)
        _print_side_by_side("AUDIO", onnx_outputs["audio"], hf_audio)

    if "vision-audio" in onnx_outputs and has_image and has_audio:
        print("Running HF vision + audio generation ...")
        hf_va = _hf_generate_vision_audio(
            model_id, image_path, audio_path, vision_audio_prompt, max_new_tokens
        )
        _print_side_by_side("VISION + AUDIO", onnx_outputs["vision-audio"], hf_va)


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
        choices=["f32", "f16", "bf16"],
        default="f32",
        help="Weight/activation dtype to use (default: %(default)s).",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "webgpu"],
        default="cpu",
        help="Device for inference (default: %(default)s).",
    )
    parser.add_argument(
        "--ep",
        choices=["default", "cpu", "cuda", "webgpu", "onnx-standard", "trt-rtx"],
        default="default",
        help="Execution provider for ONNX model build.",
    )
    parser.add_argument(
        "--compare-hf",
        action="store_true",
        help=(
            "Run the same prompts through HuggingFace PyTorch and show a side-by-side "
            "comparison with the ONNX Runtime output.  Requires transformers + torch."
        ),
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
    ep = args.ep

    # Auto-select f16 for CUDA EP: GQA and Flash Attention require fp16/bf16.
    dtype = args.dtype
    if dtype == "f32" and ep in ("cuda", "trt-rtx"):
        dtype = "f16"
        print(f"Note: auto-selecting dtype=f16 for {ep} EP (GQA/Flash require fp16/bf16)")

    np_dtype = {"f32": np.float32, "f16": np.float16, "bf16": np.float32}[dtype]

    print(f"Building ONNX models from {args.model_id!r} (dtype={dtype}, ep={ep}) ...")
    pkg = build(
        args.model_id,
        dtype=dtype,
        load_weights=load_weights,
        execution_provider=ep,
    )
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
    vision_session = OnnxModelSession(pkg["vision_encoder"], device=args.device)
    audio_session = (
        OnnxModelSession(pkg["audio_encoder"], device=args.device)
        if "audio_encoder" in pkg
        else None
    )
    embedding_session = OnnxModelSession(pkg["embedding"], device=args.device)
    # Use 'basic' graph optimization for the decoder on CUDA to prevent an
    # ORT crash in the extended graph optimization pass when Attention nodes
    # have mixed head configurations (GQA + expanded-KV MHA). This is an
    # ORT bug; remove once it's fixed upstream.
    decoder_opt = {"graph_optimization_level": "basic"} if ep in ("cuda", "trt-rtx") else {}
    decoder_session = OnnxModelSession(pkg["decoder"], device=args.device, **decoder_opt)

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

    text_prompt = args.prompt or "Explain the theory of general relativity in simple terms."
    vision_prompt = args.prompt or "Describe what you see in this image in detail."
    audio_prompt = args.prompt or "Transcribe the following audio."
    vision_audio_prompt = args.prompt or "Describe the image and transcribe the audio."

    # Collect ONNX outputs for optional --compare-hf side-by-side display
    onnx_outputs: dict[str, str] = {}

    for mode in modes:
        if mode == "text":
            result = demo_text(
                embedding_session=embedding_session,
                decoder_session=decoder_session,
                tokenizer=tokenizer,
                config=config,
                prompt=text_prompt,
                max_new_tokens=max_tokens,
                model_np_dtype=np_dtype,
            )
            onnx_outputs["text"] = result

        elif mode == "vision":
            if not has_image:
                continue
            result = demo_vision(
                vision_session=vision_session,
                embedding_session=embedding_session,
                decoder_session=decoder_session,
                tokenizer=tokenizer,
                processor=processor,
                config=config,
                image_path=args.image,
                prompt=vision_prompt,
                max_new_tokens=max_tokens,
                model_np_dtype=np_dtype,
            )
            onnx_outputs["vision"] = result

        elif mode == "audio":
            if not has_audio:
                continue
            result = demo_audio(
                audio_session=audio_session,
                embedding_session=embedding_session,
                decoder_session=decoder_session,
                tokenizer=tokenizer,
                processor=processor,
                config=config,
                audio_path=args.audio,
                prompt=audio_prompt,
                max_new_tokens=max_tokens,
                model_np_dtype=np_dtype,
            )
            onnx_outputs["audio"] = result

        elif mode == "vision-audio":
            if not has_image or not has_audio:
                continue
            result = demo_vision_audio(
                vision_session=vision_session,
                audio_session=audio_session,
                embedding_session=embedding_session,
                decoder_session=decoder_session,
                tokenizer=tokenizer,
                processor=processor,
                config=config,
                image_path=args.image,
                audio_path=args.audio,
                prompt=vision_audio_prompt,
                max_new_tokens=max_tokens,
                model_np_dtype=np_dtype,
            )
            onnx_outputs["vision-audio"] = result

    if args.compare_hf and onnx_outputs:
        run_compare_hf(
            model_id=args.model_id,
            onnx_outputs=onnx_outputs,
            has_image=has_image,
            has_audio=has_audio,
            image_path=args.image,
            audio_path=args.audio,
            max_new_tokens=max_tokens,
            text_prompt=text_prompt,
            vision_prompt=vision_prompt,
            audio_prompt=audio_prompt,
            vision_audio_prompt=vision_audio_prompt,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
