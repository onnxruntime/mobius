#!/usr/bin/env python
# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""AudioFlamingo-3 audio understanding with ONNX models.

AudioFlamingo-3 combines a Whisper-Large audio encoder with a Qwen2-7B language
decoder.  Three ONNX models are built by mobius and chained for inference:

    audio → mel spectrogram → audio_encoder → embedding → decoder → text

Prerequisites::

    pip install mobius-ai[transformers] librosa

Usage::

    # Answer a question about an audio file
    python examples/audioflamingo3.py --audio speech.wav

    # Custom prompt
    python examples/audioflamingo3.py --audio meeting.wav --prompt "Summarise the meeting."

    # Use a different model size or variant
    python examples/audioflamingo3.py --model nvidia/audio-flamingo-3-hf --audio clip.wav

    # Save ONNX models without running inference
    python examples/audioflamingo3.py --audio dummy.wav --save-to output/audioflamingo3/
"""

from __future__ import annotations

import argparse

import numpy as np
import transformers

from mobius import build
from mobius._testing.ort_inference import OnnxModelSession

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_ID = "nvidia/audio-flamingo-3-hf"
SAMPLE_RATE = 16000

# Whisper-Large audio encoder parameters
N_MELS = 128
N_FFT = 400
HOP_LENGTH = 160
MAX_AUDIO_FRAMES = 3000  # Receptive field: 30s at 16 kHz with hop_length=160

# Audio encoder output length (2x Conv1d downsampling of MAX_AUDIO_FRAMES)
NUM_AUDIO_TOKENS = 1500

# Qwen2 tokenizer special token IDs
AUDIO_TOKEN_ID = 151669  # <|audio_pad|>  — audio feature placeholder
IM_START = 151644  # <|im_start|>
IM_END = 151645  # <|im_end|>
EOS_IDS = {151643, 151645}  # <|endoftext|>, <|im_end|>

# Text segment token IDs used in the Qwen2 chat template
_SYSTEM_ID = 8948  # "system"
_USER_ID = 872  # "user"
_ASSISTANT_ID = 77091  # "assistant"
_NEWLINE_ID = 198  # "\n"

DEFAULT_PROMPT = "Please describe what you hear in this audio."
MAX_NEW_TOKENS = 128


# ---------------------------------------------------------------------------
# Mel spectrogram
# ---------------------------------------------------------------------------


def compute_mel_spectrogram(audio: np.ndarray) -> np.ndarray:
    """Compute Whisper-style log-mel spectrogram.

    Pads or truncates ``audio`` to exactly ``MAX_AUDIO_FRAMES`` time frames
    using :class:`~transformers.WhisperFeatureExtractor`.

    Args:
        audio: 1-D float32 array of raw audio samples at ``SAMPLE_RATE`` Hz.

    Returns:
        Float32 array of shape ``(1, N_MELS, MAX_AUDIO_FRAMES)`` — ready to
        pass into the ``audio_encoder`` ONNX model as ``input_features``.
    """
    from transformers import WhisperFeatureExtractor

    fe = WhisperFeatureExtractor(
        feature_size=N_MELS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        sampling_rate=SAMPLE_RATE,
    )
    out = fe(
        audio,
        sampling_rate=SAMPLE_RATE,
        return_tensors="np",
        padding=True,
    )
    return out["input_features"].astype(np.float32)  # (1, N_MELS, MAX_AUDIO_FRAMES)


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def build_prompt_ids(tokenizer, prompt: str) -> np.ndarray:
    """Construct Qwen2 chat-format input_ids with audio token placeholders.

    The audio section is represented by ``NUM_AUDIO_TOKENS`` consecutive
    ``AUDIO_TOKEN_ID`` placeholders.  The embedding model replaces these with
    the actual audio features at inference time.

    Format::

        <|im_start|>system
        You are a helpful audio AI assistant.<|im_end|>
        <|im_start|>user
        <NUM_AUDIO_TOKENS x AUDIO_TOKEN_ID><prompt text><|im_end|>
        <|im_start|>assistant

    Args:
        tokenizer: HuggingFace tokenizer loaded from the model.
        prompt: User's text question about the audio.

    Returns:
        Int64 array of shape ``(1, seq_len)``.
    """
    system_text_ids = tokenizer.encode(
        "You are a helpful audio AI assistant.", add_special_tokens=False
    )
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    audio_placeholders = [AUDIO_TOKEN_ID] * NUM_AUDIO_TOKENS

    ids = [
        IM_START,
        _SYSTEM_ID,
        _NEWLINE_ID,
        *system_text_ids,
        IM_END,
        _NEWLINE_ID,
        IM_START,
        _USER_ID,
        _NEWLINE_ID,
        *audio_placeholders,
        *prompt_ids,
        IM_END,
        _NEWLINE_ID,
        IM_START,
        _ASSISTANT_ID,
        _NEWLINE_ID,
    ]
    return np.array([ids], dtype=np.int64)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def transcribe(
    sessions: dict[str, OnnxModelSession],
    tokenizer,
    audio: np.ndarray,
    config,
    *,
    prompt: str = DEFAULT_PROMPT,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> str:
    """Run the full AudioFlamingo-3 ONNX inference pipeline.

    Chains the three ONNX models produced by mobius:

    1. ``audio_encoder`` — mel (1, 128, 3000) → audio_features (1, 1500, hidden)
    2. ``embedding`` — input_ids + audio_features → inputs_embeds
    3. ``decoder`` — inputs_embeds + KV cache → logits (autoregressive loop)

    Args:
        sessions: ``{model_name: OnnxModelSession}`` for the three models.
        tokenizer: HuggingFace Qwen2 tokenizer.
        audio: Raw 1-D float32 audio at 16 kHz.
        config: :class:`~mobius.ArchitectureConfig` from the built package.
        prompt: Text question about the audio content.
        max_new_tokens: Maximum number of tokens to generate.

    Returns:
        Decoded text response.
    """
    batch_size = 1

    # --- Step 1: mel spectrogram (1, 128, 3000) --------------------------------
    mel = compute_mel_spectrogram(audio)

    # --- Step 2: audio encoder → audio_features (1, 1500, hidden) -------------
    encoder_out = sessions["audio_encoder"].run({"input_features": mel})
    audio_features = encoder_out["audio_features"]  # (1, 1500, hidden)
    # Embedding model expects 2D input: (num_audio_tokens, hidden)
    audio_features_2d = audio_features[0]  # (1500, hidden)

    # --- Step 3: prompt input_ids with audio placeholders ----------------------
    input_ids = build_prompt_ids(tokenizer, prompt)  # (1, seq_len)

    # --- Step 4: embedding model — fuse text + audio embeddings ----------------
    embed_out = sessions["embedding"].run(
        {"input_ids": input_ids, "audio_features": audio_features_2d}
    )
    inputs_embeds = embed_out["inputs_embeds"]  # (1, seq_len, hidden)

    # --- Step 5: autoregressive decoding with the text decoder -----------------
    num_layers = config.num_hidden_layers
    num_kv_heads = config.num_key_value_heads
    head_dim = config.head_dim

    # Initialise empty KV cache
    past_kv: dict[str, np.ndarray] = {}
    for i in range(num_layers):
        past_kv[f"past_key_values.{i}.key"] = np.zeros(
            (batch_size, num_kv_heads, 0, head_dim), dtype=np.float32
        )
        past_kv[f"past_key_values.{i}.value"] = np.zeros(
            (batch_size, num_kv_heads, 0, head_dim), dtype=np.float32
        )

    # Prefill: standard 1D position_ids (Qwen2 uses 1D RoPE, not MRoPE)
    prefill_len = inputs_embeds.shape[1]
    position_ids = np.arange(prefill_len, dtype=np.int64)[np.newaxis, :]  # (1, seq_len)

    out = sessions["decoder"].run(
        {
            "inputs_embeds": inputs_embeds,
            "attention_mask": np.ones((batch_size, prefill_len), dtype=np.int64),
            "position_ids": position_ids,
            **past_kv,
        }
    )

    # First generated token
    next_token = int(np.argmax(out["logits"][:, -1, :]))
    generated_ids = [next_token]
    print(tokenizer.decode([next_token], skip_special_tokens=True), end="", flush=True)

    for i in range(num_layers):
        past_kv[f"past_key_values.{i}.key"] = out[f"present.{i}.key"]
        past_kv[f"past_key_values.{i}.value"] = out[f"present.{i}.value"]

    past_seq_len = prefill_len

    # Decode loop: single-token steps (no audio features in subsequent steps)
    for _ in range(max_new_tokens - 1):
        if next_token in EOS_IDS:
            break

        cur_ids = np.array([[next_token]], dtype=np.int64)
        # Pass empty audio_features — no audio tokens in decode steps
        dummy_audio = np.zeros((0, audio_features_2d.shape[-1]), dtype=np.float32)
        embed_out = sessions["embedding"].run(
            {"input_ids": cur_ids, "audio_features": dummy_audio}
        )
        cur_embeds = embed_out["inputs_embeds"]

        total_seq_len = past_seq_len + 1
        pos = np.array([[past_seq_len]], dtype=np.int64)  # (1, 1)

        out = sessions["decoder"].run(
            {
                "inputs_embeds": cur_embeds,
                "attention_mask": np.ones((batch_size, total_seq_len), dtype=np.int64),
                "position_ids": pos,
                **past_kv,
            }
        )

        next_token = int(np.argmax(out["logits"][:, -1, :]))
        generated_ids.append(next_token)
        print(tokenizer.decode([next_token], skip_special_tokens=True), end="", flush=True)

        for i in range(num_layers):
            past_kv[f"past_key_values.{i}.key"] = out[f"present.{i}.key"]
            past_kv[f"past_key_values.{i}.value"] = out[f"present.{i}.value"]

        past_seq_len = total_seq_len

    print()
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def load_audio_file(path: str) -> np.ndarray:
    """Load and resample an audio file to 16 kHz mono float32."""
    import librosa

    audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    return audio.astype(np.float32)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AudioFlamingo-3 audio understanding with ONNX models.",
    )
    parser.add_argument(
        "--model",
        default=MODEL_ID,
        help="HuggingFace model ID (default: %(default)s).",
    )
    parser.add_argument(
        "--audio",
        required=True,
        metavar="FILE",
        help="Audio file to process (WAV, FLAC, MP3, …).",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Text question about the audio (default: %(default)r).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=MAX_NEW_TOKENS,
        help="Maximum tokens to generate (default: %(default)s).",
    )
    parser.add_argument(
        "--save-to",
        metavar="DIR",
        default=None,
        help="Save ONNX models to DIR and exit without running inference.",
    )
    args = parser.parse_args()

    print(f"Building ONNX models from {args.model!r} ...")
    pkg = build(args.model, dtype="f32", load_weights=not args.save_to)
    config = pkg.config

    if args.save_to:
        pkg.save(args.save_to, check_weights=False)
        print(f"Saved to {args.save_to}")
        return

    print("Creating inference sessions ...")
    sessions = {name: OnnxModelSession(model) for name, model in pkg.items()}

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    print(f"Loading audio: {args.audio}")
    audio = load_audio_file(args.audio)
    duration = len(audio) / SAMPLE_RATE
    print(f"Duration: {duration:.1f}s — prompt: {args.prompt!r}\n")
    print("Response: ", end="", flush=True)

    result = transcribe(
        sessions,
        tokenizer,
        audio,
        config,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
    )
    print(f"\n📝 Result: {result}")


if __name__ == "__main__":
    main()
