#!/usr/bin/env python
# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""MMS (Massively Multilingual Speech) ASR with ONNX models.

Builds a single ONNX model from ``mobius`` and runs CTC speech-to-text
decoding.  MMS supports 1,100+ languages — select the language with
``--lang`` and the model uses the matching adapter.

    audio → raw waveform → wav2vec2 encoder → CTC head → text

Architecture overview
---------------------
- ``facebook/mms-300m``   — 300 M params, 1,107 languages
- ``facebook/mms-1b-all`` — 1 B params, 1,162 languages (best quality)

The language-specific adapter (``Wav2Vec2Adapter``) is loaded from the
checkpoint and baked into the ONNX model.  Switching languages requires
rebuilding the model with a different ``--lang``.

Prerequisites::

    pip install mobius-ai[transformers] torchaudio

Usage::

    # Transcribe an audio file (English, default)
    python examples/mms.py --audio speech.wav

    # Spanish transcription with the 1B model
    python examples/mms.py --audio habla.wav --lang spa --model facebook/mms-1b-all

    # Save ONNX model to disk without running inference
    python examples/mms.py --save-to output/mms/ --lang fra

    # List some supported languages
    python examples/mms.py --list-langs

    # Record from microphone (requires sounddevice)
    python examples/mms.py --mic --lang deu
"""

from __future__ import annotations

import argparse
import sys
import threading

import numpy as np

from mobius import build
from mobius._testing.ort_inference import OnnxModelSession

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "facebook/mms-300m"
SAMPLE_RATE = 16_000

# A selection of ISO 639-3 codes and display names for the --list-langs helper
_SAMPLE_LANGS = {
    "eng": "English",
    "spa": "Spanish",
    "fra": "French",
    "deu": "German",
    "zho": "Chinese (Mandarin)",
    "ara": "Arabic",
    "por": "Portuguese",
    "hin": "Hindi",
    "swh": "Swahili",
    "jpn": "Japanese",
    "kor": "Korean",
    "tur": "Turkish",
    "vie": "Vietnamese",
    "pol": "Polish",
    "nld": "Dutch",
}


# ---------------------------------------------------------------------------
# Audio utilities
# ---------------------------------------------------------------------------


def load_audio_file(path: str, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Load an audio file and resample to 16 kHz mono float32."""
    import torchaudio

    waveform, sr = torchaudio.load(path)
    if sr != sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
    return waveform.mean(dim=0).numpy().astype(np.float32)


def record_from_mic(
    sample_rate: int = SAMPLE_RATE, max_seconds: int = 60
) -> np.ndarray:
    """Record audio from the default microphone until Enter is pressed."""
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

    print("🎤 Recording… Press Enter to stop.")
    stream.start()
    t = threading.Thread(target=lambda: (input(), stop_event.set()))
    t.daemon = True
    t.start()
    t.join(timeout=max_seconds)
    stop_event.set()
    stream.stop()
    stream.close()

    if not chunks:
        return np.array([], dtype=np.float32)
    audio = np.concatenate(chunks, axis=0).flatten()
    print(f"  Recorded {len(audio) / sample_rate:.1f}s of audio.")
    return audio


# ---------------------------------------------------------------------------
# CTC decoding
# ---------------------------------------------------------------------------


def ctc_greedy_decode(logits: np.ndarray, tokenizer) -> str:
    """Greedy CTC decode: argmax → collapse repeats → remove blank.

    Args:
        logits: ``(batch, time, vocab)`` CTC logit scores (unnormalised)
        tokenizer: HuggingFace ``Wav2Vec2CTCTokenizer`` for id→char mapping

    Returns:
        Decoded transcript string.
    """
    # (batch=1, time, vocab) → (time,)
    ids = np.argmax(logits[0], axis=-1).tolist()

    # Collapse consecutive repeated tokens
    collapsed: list[int] = []
    prev = -1
    for tok in ids:
        if tok != prev:
            collapsed.append(tok)
            prev = tok

    # Remove blank token (id 0 in wav2vec2 CTC convention)
    non_blank = [t for t in collapsed if t != tokenizer.pad_token_id]

    return tokenizer.decode(non_blank)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def transcribe(
    session: OnnxModelSession,
    tokenizer,
    audio: np.ndarray,
    *,
    chunk_seconds: float = 30.0,
) -> str:
    """Run MMS ASR on raw waveform audio.

    For long recordings the audio is chunked into ``chunk_seconds``-length
    pieces; each chunk is decoded independently and joined with spaces.

    Args:
        session: ORT session wrapping the ONNX MMS model.
        tokenizer: HuggingFace ``Wav2Vec2CTCTokenizer``.
        audio: Raw waveform at 16 kHz, shape ``(num_samples,)``.
        chunk_seconds: Maximum chunk length in seconds.

    Returns:
        Decoded transcript string.
    """
    chunk_len = int(chunk_seconds * SAMPLE_RATE)
    chunks = [audio[i : i + chunk_len] for i in range(0, len(audio), chunk_len)]
    transcripts: list[str] = []

    for i, chunk in enumerate(chunks):
        if len(chunks) > 1:
            print(f"  Chunk {i + 1}/{len(chunks)} ({len(chunk) / SAMPLE_RATE:.1f}s) …")

        # MMS expects ``(batch, num_samples)`` — no mel spectrogram needed
        input_values = chunk[np.newaxis, :].astype(np.float32)
        attention_mask = np.ones((1, len(chunk)), dtype=np.int64)

        out = session.run(
            {"input_values": input_values, "attention_mask": attention_mask}
        )
        logits = out["logits"]  # (1, time_steps, vocab_size)
        transcripts.append(ctc_greedy_decode(logits, tokenizer))

    return " ".join(t for t in transcripts if t.strip())


# ---------------------------------------------------------------------------
# Language loading helper
# ---------------------------------------------------------------------------


def load_mms_model_and_tokenizer(model_id: str, lang: str):
    """Load HuggingFace MMS processor and set the language adapter.

    Returns ``(hf_model, tokenizer)`` with the correct adapter loaded.
    Callers pass the ``hf_model`` config to ``build()`` via ``model_id``
    but we also return it so the adapter weights are baked in.
    """
    from transformers import AutoProcessor, Wav2Vec2ForCTC

    print(f"  Loading HuggingFace checkpoint for language '{lang}' …")
    processor = AutoProcessor.from_pretrained(model_id)
    hf_model = Wav2Vec2ForCTC.from_pretrained(model_id, ignore_mismatched_sizes=True)

    processor.tokenizer.set_target_lang(lang)
    hf_model.load_adapter(lang)

    return hf_model, processor.tokenizer


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MMS multilingual ASR with ONNX (mobius).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="HuggingFace MMS model ID.",
    )
    parser.add_argument(
        "--lang",
        default="eng",
        metavar="ISO-639-3",
        help="Language code (ISO 639-3), e.g. 'eng', 'spa', 'fra'.",
    )
    parser.add_argument(
        "--audio",
        default=None,
        metavar="FILE",
        help="Path to an audio file to transcribe.",
    )
    parser.add_argument(
        "--mic",
        action="store_true",
        help="Record from the default microphone.",
    )
    parser.add_argument(
        "--chunk",
        type=float,
        default=30.0,
        metavar="SECONDS",
        help="Audio chunk size for long recordings.",
    )
    parser.add_argument(
        "--save-to",
        metavar="DIR",
        default=None,
        help="Save the ONNX model to DIR and exit.",
    )
    parser.add_argument(
        "--list-langs",
        action="store_true",
        help="Print a selection of supported language codes and exit.",
    )
    args = parser.parse_args()

    if args.list_langs:
        print("Sample MMS language codes (ISO 639-3):")
        for code, name in sorted(_SAMPLE_LANGS.items()):
            print(f"  {code}  {name}")
        print(
            "\nFull list: https://huggingface.co/facebook/mms-1b-all/blob/main/README.md"
        )
        return

    # ------------------------------------------------------------------
    # Load HF model with the target language adapter baked in
    # ------------------------------------------------------------------
    print(f"Language: {args.lang}  |  Model: {args.model}")
    hf_model, tokenizer = load_mms_model_and_tokenizer(args.model, args.lang)

    # ------------------------------------------------------------------
    # Build ONNX model from the weight-loaded HF model
    # ------------------------------------------------------------------
    print("Building ONNX model …")
    pkg = build(hf_model, load_weights=not args.save_to)

    if args.save_to:
        pkg.save(args.save_to)
        print(f"Saved ONNX model to {args.save_to}")
        return

    # ------------------------------------------------------------------
    # Create OnnxRuntime session
    # ------------------------------------------------------------------
    print("Creating ORT session …")
    session = OnnxModelSession(pkg["model"])
    print("Ready.\n")

    # ------------------------------------------------------------------
    # Get audio
    # ------------------------------------------------------------------
    if args.audio:
        print(f"Loading audio: {args.audio}")
        audio = load_audio_file(args.audio)
        print(f"Audio: {len(audio) / SAMPLE_RATE:.1f}s at 16 kHz\n")
    elif args.mic:
        audio = record_from_mic()
        if len(audio) < SAMPLE_RATE * 0.3:
            print("No audio captured.")
            return
    else:
        # Demonstrate with a synthetic 1-second sine-wave "utterance"
        print("(No --audio or --mic provided — using synthetic test tone.)")
        t = np.linspace(0, 1.0, SAMPLE_RATE, dtype=np.float32)
        audio = (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    # ------------------------------------------------------------------
    # Transcribe
    # ------------------------------------------------------------------
    print("Transcribing …")
    text = transcribe(session, tokenizer, audio, chunk_seconds=args.chunk)
    print(f"\n📝 Transcript [{args.lang}]: {text!r}")


if __name__ == "__main__":
    main()
