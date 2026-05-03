#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Qwen3-ASR speech recognition with ONNX models.

Builds three ONNX models from ``mobius`` (audio encoder,
embedding, decoder) and runs the full ASR pipeline:

    audio → mel spectrogram → audio encoder → embedding fusion → decoder → text

Supports real-time microphone input and audio file input.

Prerequisites::

    pip install mobius-ai[transformers] sounddevice

Usage::

    # Record from microphone (press Enter to stop)
    python examples/qwen3_asr.py

    # Transcribe an audio file
    python examples/qwen3_asr.py --audio speech.wav

    # Continuous mic mode (Ctrl+C to exit)
    python examples/qwen3_asr.py --continuous

    # Force a specific language/dialect and skip auto-detection
    python examples/qwen3_asr.py --language zh          # Mandarin
    python examples/qwen3_asr.py --language yue          # Cantonese
    python examples/qwen3_asr.py --language en           # English
    python examples/qwen3_asr.py --language ja           # Japanese
    python examples/qwen3_asr.py --language dongbei      # Dongbei dialect
    python examples/qwen3_asr.py --language 四川话        # Sichuan dialect (Chinese aliases work)

    # Use a different model size
    python examples/qwen3_asr.py --model Qwen/Qwen3-ASR-1.7B

    # GPU inference with half precision
    python examples/qwen3_asr.py --device cuda --dtype f16

    # Disable streaming output
    python examples/qwen3_asr.py --no-stream

    # Save ONNX models without running inference
    python examples/qwen3_asr.py --save-to output/qwen3-asr/
"""

from __future__ import annotations

import argparse
import sys
import threading

import numpy as np
import transformers

from mobius import build
from mobius._testing.ort_inference import OnnxModelSession

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_ID = "Qwen/Qwen3-ASR-0.6B"
SAMPLE_RATE = 16000
MAX_RECORD_SECONDS = 60
MAX_NEW_TOKENS = 256

# Qwen3-ASR special tokens
AUDIO_START_TOKEN_ID = 151669
AUDIO_TOKEN_ID = 151676  # <|audio_pad|>
AUDIO_END_TOKEN_ID = 151670
ASR_TEXT_TOKEN = 151704  # <asr_text>

# Chat template token IDs (Qwen-style)
IM_START = 151644  # <|im_start|>
IM_END = 151645  # <|im_end|>
SYSTEM_ID = 8948  # "system"
USER_ID = 872  # "user"
ASSISTANT_ID = 77091  # "assistant"
NEWLINE_ID = 198  # "\n"

# Language / dialect mapping.
# Keys are CLI aliases (including Chinese character aliases);
# values are the language names used in the model's
# "language <NAME><asr_text>" generation prefix.
#
# Supports 30 languages + 22 Chinese dialects.
LANGUAGE_MAP: dict[str, str] = {
    # Auto-detect (default: model decides the language)
    "auto": "",
    # --- Languages (30) ---
    # Chinese / Mandarin
    "zh": "Chinese",
    "chinese": "Chinese",
    "mandarin": "Chinese",
    "普通话": "Chinese",
    "中文": "Chinese",
    # English
    "en": "English",
    "english": "English",
    # Japanese
    "ja": "Japanese",
    "japanese": "Japanese",
    # Korean
    "ko": "Korean",
    "korean": "Korean",
    # Arabic
    "ar": "Arabic",
    "arabic": "Arabic",
    # German
    "de": "German",
    "german": "German",
    # French
    "fr": "French",
    "french": "French",
    # Spanish
    "es": "Spanish",
    "spanish": "Spanish",
    # Portuguese
    "pt": "Portuguese",
    "portuguese": "Portuguese",
    # Indonesian
    "id": "Indonesian",
    "indonesian": "Indonesian",
    # Italian
    "it": "Italian",
    "italian": "Italian",
    # Russian
    "ru": "Russian",
    "russian": "Russian",
    # Thai
    "th": "Thai",
    "thai": "Thai",
    # Vietnamese
    "vi": "Vietnamese",
    "vietnamese": "Vietnamese",
    # Turkish
    "tr": "Turkish",
    "turkish": "Turkish",
    # Hindi
    "hi": "Hindi",
    "hindi": "Hindi",
    # Malay
    "ms": "Malay",
    "malay": "Malay",
    # Dutch
    "nl": "Dutch",
    "dutch": "Dutch",
    # Swedish
    "sv": "Swedish",
    "swedish": "Swedish",
    # Danish
    "da": "Danish",
    "danish": "Danish",
    # Finnish
    "fi": "Finnish",
    "finnish": "Finnish",
    # Polish
    "pl": "Polish",
    "polish": "Polish",
    # Czech
    "cs": "Czech",
    "czech": "Czech",
    # Filipino
    "fil": "Filipino",
    "filipino": "Filipino",
    # Persian
    "fa": "Persian",
    "persian": "Persian",
    # Greek
    "el": "Greek",
    "greek": "Greek",
    # Hungarian
    "hu": "Hungarian",
    "hungarian": "Hungarian",
    # Macedonian
    "mk": "Macedonian",
    "macedonian": "Macedonian",
    # Romanian
    "ro": "Romanian",
    "romanian": "Romanian",
    # --- Chinese Dialects (22) ---
    # Cantonese
    "yue": "Cantonese",
    "cantonese": "Cantonese",
    "粤语": "Cantonese",
    # Cantonese (Hong Kong)
    "cantonese-hk": "Cantonese-HK",
    "香港粤语": "Cantonese-HK",
    # Cantonese (Guangdong)
    "cantonese-gd": "Cantonese-GD",
    "广东粤语": "Cantonese-GD",
    # Wu / Shanghainese
    "wuu": "Shanghainese",
    "wu": "Shanghainese",
    "shanghainese": "Shanghainese",
    "吴语": "Shanghainese",
    "上海话": "Shanghainese",
    # Min Nan / Hokkien
    "minnan": "Minnan",
    "hokkien": "Minnan",
    "闽南语": "Minnan",
    # Hakka
    "hakka": "Hakka",
    "客家话": "Hakka",
    # Sichuan
    "sichuan": "Sichuan",
    "四川话": "Sichuan",
    # Anhui
    "anhui": "Anhui",
    "安徽话": "Anhui",
    # Dongbei (Northeastern)
    "dongbei": "Dongbei",
    "东北话": "Dongbei",
    # Fujian
    "fujian": "Fujian",
    "福建话": "Fujian",
    # Gansu
    "gansu": "Gansu",
    "甘肃话": "Gansu",
    # Guizhou
    "guizhou": "Guizhou",
    "贵州话": "Guizhou",
    # Hebei
    "hebei": "Hebei",
    "河北话": "Hebei",
    # Henan
    "henan": "Henan",
    "河南话": "Henan",
    # Hubei
    "hubei": "Hubei",
    "湖北话": "Hubei",
    # Hunan
    "hunan": "Hunan",
    "湖南话": "Hunan",
    # Jiangxi
    "jiangxi": "Jiangxi",
    "江西话": "Jiangxi",
    # Ningxia
    "ningxia": "Ningxia",
    "宁夏话": "Ningxia",
    # Shandong
    "shandong": "Shandong",
    "山东话": "Shandong",
    # Shaanxi
    "shaanxi": "Shaanxi",
    "陕西话": "Shaanxi",
    # Shanxi
    "shanxi": "Shanxi",
    "山西话": "Shanxi",
    # Tianjin
    "tianjin": "Tianjin",
    "天津话": "Tianjin",
    # Yunnan
    "yunnan": "Yunnan",
    "云南话": "Yunnan",
    # Zhejiang
    "zhejiang": "Zhejiang",
    "浙江话": "Zhejiang",
}


# ---------------------------------------------------------------------------
# Mel spectrogram
# ---------------------------------------------------------------------------


def compute_mel_spectrogram(
    audio: np.ndarray,
    *,
    sample_rate: int = SAMPLE_RATE,
    n_mels: int = 128,
    n_fft: int = 400,
    hop_length: int = 160,
) -> np.ndarray:
    """Compute log-mel spectrogram using WhisperFeatureExtractor.

    Returns array of shape ``(1, n_mels, time_frames)``.
    """
    from transformers import WhisperFeatureExtractor

    fe = WhisperFeatureExtractor(
        feature_size=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        sampling_rate=sample_rate,
    )
    out = fe(
        audio,
        sampling_rate=sample_rate,
        return_tensors="np",
        padding=False,
    )
    return out["input_features"].astype(np.float32)  # Always float32; caller casts to model dtype


# ---------------------------------------------------------------------------
# Microphone recording
# ---------------------------------------------------------------------------


def record_until_enter(
    sample_rate: int = SAMPLE_RATE,
    max_seconds: int = MAX_RECORD_SECONDS,
) -> np.ndarray:
    """Record audio from the default mic until Enter is pressed."""
    import sounddevice as sd

    chunks: list[np.ndarray] = []
    stop_event = threading.Event()

    def callback(indata, frames, time, status):
        if status:
            print(f"  [mic] {status}", file=sys.stderr)
        chunks.append(indata.copy())

    stream = sd.InputStream(
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
        callback=callback,
        blocksize=int(sample_rate * 0.1),
    )

    print("🎤 Recording... Press Enter to stop.")
    stream.start()

    input_thread = threading.Thread(target=lambda: (input(), stop_event.set()))
    input_thread.daemon = True
    input_thread.start()
    input_thread.join(timeout=max_seconds)
    stop_event.set()

    stream.stop()
    stream.close()

    if not chunks:
        return np.array([], dtype=np.float32)

    audio = np.concatenate(chunks, axis=0).flatten()
    duration = len(audio) / sample_rate
    print(f"  Recorded {duration:.1f}s of audio.")
    return audio


def load_audio_file(path: str, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Load an audio file and resample to target sample rate."""
    import torchaudio

    waveform, sr = torchaudio.load(path)
    if sr != sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
    # Mono, float32
    return waveform.mean(dim=0).numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# ASR inference pipeline
# ---------------------------------------------------------------------------


def transcribe(
    sessions: dict[str, OnnxModelSession],
    tokenizer,
    audio: np.ndarray,
    config,
    *,
    max_new_tokens: int = MAX_NEW_TOKENS,
    language: str = "",
    stream: bool = True,
    model_dtype: np.dtype = np.float32,
) -> str:
    """Full ASR pipeline: audio → text.

    Runs three ONNX models in sequence:
    1. Audio encoder: mel spectrogram → audio features
    2. Embedding: fuse text tokens with audio features
    3. Decoder: autoregressive text generation with KV cache

    Args:
        language: If non-empty, force language by prepending
            ``language <NAME><asr_text>`` as the assistant prefix.
        stream: If True, print tokens as they are generated.
        model_dtype: Numpy dtype matching the model precision.
    """
    batch_size = 1

    # Step 1: Compute mel spectrogram
    mel = compute_mel_spectrogram(audio).astype(model_dtype)  # (1, n_mels, time)

    # Step 2: Run audio encoder
    audio_out = sessions["audio_encoder"].run({"input_features": mel})
    audio_features = audio_out["audio_features"]  # (1, audio_seq, dim)
    num_audio_tokens = audio_features.shape[1]

    # Flatten to (num_audio_tokens, output_dim) for the embedding model
    audio_features_2d = audio_features.reshape(-1, audio_features.shape[-1])

    # Step 3: Build chat-template prompt with audio token placeholders
    # Format: <|im_start|>system\n<|im_end|>\n
    #         <|im_start|>user\n<|audio_start|><|audio_pad|>*N<|audio_end|>
    #         <|im_end|>\n<|im_start|>assistant\n
    prompt_ids = (
        [
            IM_START,
            SYSTEM_ID,
            NEWLINE_ID,
            IM_END,
            NEWLINE_ID,
            IM_START,
            USER_ID,
            NEWLINE_ID,
            AUDIO_START_TOKEN_ID,
        ]
        + [AUDIO_TOKEN_ID] * num_audio_tokens
        + [AUDIO_END_TOKEN_ID, IM_END, NEWLINE_ID, IM_START, ASSISTANT_ID, NEWLINE_ID]
    )

    # When language is forced, append "language <NAME><asr_text>" tokens
    # as the assistant's initial response to skip language detection.
    if language:
        prefix_text = f"language {language}<asr_text>"
        prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
        prompt_ids.extend(prefix_ids)

    input_ids = np.array([prompt_ids], dtype=np.int64)

    # Step 4: Run embedding model (fuse text + audio)
    embed_out = sessions["embedding"].run(
        {"input_ids": input_ids, "audio_features": audio_features_2d}
    )
    inputs_embeds = embed_out["inputs_embeds"]  # (1, seq_len, hidden)

    # Step 5: Autoregressive decoding with the decoder model
    num_layers = config.num_hidden_layers
    num_kv_heads = config.num_key_value_heads
    head_dim = config.head_dim

    # Initialize empty KV cache
    past_kv = {}
    for i in range(num_layers):
        past_kv[f"past_key_values.{i}.key"] = np.zeros(
            (batch_size, num_kv_heads, 0, head_dim), dtype=model_dtype
        )
        past_kv[f"past_key_values.{i}.value"] = np.zeros(
            (batch_size, num_kv_heads, 0, head_dim), dtype=model_dtype
        )

    # Prefill pass with fused embeddings
    prefill_len = inputs_embeds.shape[1]
    pos = np.arange(prefill_len, dtype=np.int64)[np.newaxis, :]
    # MRoPE: all 3 dims get same positions for text-only generation
    position_ids = np.stack([pos, pos, pos])  # (3, 1, seq_len)

    decoder_feeds = {
        "inputs_embeds": inputs_embeds,
        "attention_mask": np.ones((batch_size, prefill_len), dtype=np.int64),
        "position_ids": position_ids,
        **past_kv,
    }
    out = sessions["decoder"].run(decoder_feeds)

    # Get first generated token
    logits = out["logits"]
    next_token = int(np.argmax(logits[:, -1, :]))
    generated_ids = [next_token]

    # Update KV cache
    for i in range(num_layers):
        past_kv[f"past_key_values.{i}.key"] = out[f"present.{i}.key"]
        past_kv[f"past_key_values.{i}.value"] = out[f"present.{i}.value"]

    past_seq_len = prefill_len

    # Decode loop: feed each new token back through embedding + decoder
    eos_ids = {151643, 151645}  # <|endoftext|>, <|im_end|>
    # When language is forced, <asr_text> is already in the prefill prompt,
    # so the streaming gate should be open immediately.
    streaming = bool(language)
    # Track how many characters have been printed for delta streaming.
    # Decoding the full token sequence avoids incomplete UTF-8 byte issues
    # that occur when individual BPE tokens split multi-byte characters.
    printed_len = 0
    for _ in range(max_new_tokens - 1):
        if next_token in eos_ids:
            break

        # For decode steps, use embedding with single token
        # (no audio features — zeros since there are no audio tokens)
        cur_ids = np.array([[next_token]], dtype=np.int64)
        dummy_audio = np.zeros((0, audio_features_2d.shape[-1]), dtype=model_dtype)
        embed_out = sessions["embedding"].run(
            {"input_ids": cur_ids, "audio_features": dummy_audio}
        )
        cur_embeds = embed_out["inputs_embeds"]

        total_seq_len = past_seq_len + 1
        pos = np.array([[past_seq_len]], dtype=np.int64)
        position_ids = np.stack([pos, pos, pos])  # (3, 1, 1)

        decoder_feeds = {
            "inputs_embeds": cur_embeds,
            "attention_mask": np.ones((batch_size, total_seq_len), dtype=np.int64),
            "position_ids": position_ids,
            **past_kv,
        }
        out = sessions["decoder"].run(decoder_feeds)

        logits = out["logits"]
        next_token = int(np.argmax(logits[:, -1, :]))
        generated_ids.append(next_token)

        # Stream output: decode full sequence and print only new characters.
        # This avoids broken UTF-8 from per-token decode of multi-byte chars.
        if next_token == ASR_TEXT_TOKEN:
            streaming = True
        elif streaming and stream:
            full_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            if len(full_text) > printed_len:
                print(full_text[printed_len:], end="", flush=True)
                printed_len = len(full_text)

        for i in range(num_layers):
            past_kv[f"past_key_values.{i}.key"] = out[f"present.{i}.key"]
            past_kv[f"past_key_values.{i}.value"] = out[f"present.{i}.value"]
        past_seq_len = total_seq_len

    print()
    raw = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return parse_asr_output(raw)


def parse_asr_output(raw: str) -> str:
    """Strip ``language X<asr_text>`` prefix from raw ASR output."""
    import re

    m = re.match(r"language\s+\w+<asr_text>", raw)
    if m:
        return raw[m.end() :]
    return raw


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description=("Qwen3-ASR speech recognition with ONNX models."),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        default=MODEL_ID,
        help="HuggingFace model ID (default: %(default)s).",
    )
    parser.add_argument(
        "--audio",
        default=None,
        help="Path to an audio file. If omitted, records from mic.",
    )
    parser.add_argument(
        "--language",
        default="auto",
        help=(
            "Force language/dialect. "
            "Languages: auto, zh, en, ja, ko, ar, de, fr, es, pt, id, it, "
            "ru, th, vi, tr, hi, ms, nl, sv, da, fi, pl, cs, fil, fa, el, "
            "hu, mk, ro. "
            "Dialects: yue, wuu, sichuan, minnan, hakka, anhui, dongbei, "
            "fujian, gansu, guizhou, hebei, henan, hubei, hunan, jiangxi, "
            "ningxia, shandong, shaanxi, shanxi, tianjin, yunnan, zhejiang. "
            "Chinese aliases also accepted (普通话, 粤语, 四川话, etc). "
            "Default: auto (model auto-detects)."
        ),
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda", "webgpu"],
        help="Execution provider (default: cpu).",
    )
    parser.add_argument(
        "--dtype",
        default="f32",
        choices=["f32", "f16", "bf16"],
        help="Model precision (default: f32).",
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Disable streaming output (print all at once).",
    )
    parser.add_argument(
        "--continuous",
        action="store_true",
        help="Continuously record and transcribe (loop until Ctrl+C).",
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
        help="Save ONNX models to DIR and exit (no inference).",
    )
    parser.add_argument(
        "--ci",
        action="store_true",
        help="Exit with non-zero code on failure (for CI pipelines).",
    )
    args = parser.parse_args()

    # Resolve language
    lang_key = args.language.lower().strip()
    if lang_key not in LANGUAGE_MAP:
        supported = ", ".join(sorted(set(LANGUAGE_MAP.keys()) - {"auto"}))
        parser.error(f"Unknown language {args.language!r}. Supported: auto, {supported}")
    forced_language = LANGUAGE_MAP[lang_key]  # Empty string for "auto"

    # Build the 3 ONNX models (auto-detected from model_type)
    print(f"Building ONNX models from {args.model!r} (dtype={args.dtype}) ...")
    pkg = build(args.model, dtype=args.dtype, load_weights=not args.save_to)
    config = pkg.config

    if args.save_to:
        pkg.save(args.save_to, check_weights=False)
        print(f"Saved to {args.save_to}")
        return

    # Create ORT sessions for each model.
    device = args.device
    print(f"Creating inference sessions (device={device}) ...")
    sessions = {name: OnnxModelSession(model, device=device) for name, model in pkg.items()}

    # Load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    print("Ready.\n")
    if forced_language:
        print(f"Language: {forced_language} (forced)")
    else:
        print("Language: auto-detect")

    stream = not args.no_stream

    # Map mobius dtype string to numpy dtype for inference arrays
    np_dtype_map = {"f32": np.float32, "f16": np.float16, "bf16": np.float32}
    # Note: bf16 uses float32 for numpy arrays since numpy lacks bfloat16;
    # ORT handles the actual bf16 computation internally.
    np_dtype = np_dtype_map[args.dtype]

    def do_transcribe(audio_data):
        return transcribe(
            sessions,
            tokenizer,
            audio_data,
            config,
            max_new_tokens=args.max_new_tokens,
            language=forced_language,
            stream=stream,
            model_dtype=np_dtype,
        )

    if args.audio:
        print(f"Loading audio: {args.audio}")
        audio = load_audio_file(args.audio)
        print(f"Audio: {len(audio) / SAMPLE_RATE:.1f}s\n")
        text = do_transcribe(audio)
        print(f"\n📝 Result: {text}")
    elif args.continuous:
        print("=== Continuous ASR Mode (Ctrl+C to exit) ===\n")
        try:
            while True:
                audio = record_until_enter()
                if len(audio) < SAMPLE_RATE * 0.5:
                    print("  (too short, skipping)\n")
                    continue
                text = do_transcribe(audio)
                print(f"📝 {text}\n")
        except KeyboardInterrupt:
            print("\nDone.")
    else:
        audio = record_until_enter()
        if len(audio) < SAMPLE_RATE * 0.3:
            print("No audio recorded.")
            return
        text = do_transcribe(audio)
        print(f"\n📝 Result: {text}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        if "--ci" in sys.argv:
            print(f"FAILED: {e}", file=sys.stderr)
            sys.exit(1)
        raise
