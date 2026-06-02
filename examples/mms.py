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

Modes
-----
**Batch** (default): Transcribes the full file in chunks, prints the
complete transcript when done.

**Streaming** (``--stream``): Processes overlapping windows as they
arrive and prints partial results to the terminal in real-time, updating
each line in place.  Works on both file input and microphone input.

**Live microphone** (``--live``): Captures from the default microphone
and continuously transcribes in the foreground, printing each decoded
segment as it is ready.  Press Ctrl-C to stop.

Prerequisites::

    pip install mobius-ai[transformers] torchaudio
    pip install sounddevice   # for --mic, --live, or audio playback

Usage::

    # Batch transcription of an audio file (English, default)
    python examples/mms.py --audio speech.wav

    # Streaming display while processing (print partial results)
    python examples/mms.py --audio speech.wav --stream

    # Play audio while transcribing
    python examples/mms.py --audio speech.wav --play

    # Spanish transcription with the 1B model
    python examples/mms.py --audio habla.wav --lang spa --model facebook/mms-1b-all

    # Save ONNX model to disk without running inference
    python examples/mms.py --save-to output/mms/ --lang fra

    # List some supported languages
    python examples/mms.py --list-langs

    # Record from microphone (batch, requires sounddevice)
    python examples/mms.py --mic --lang deu

    # Live real-time transcription from microphone
    python examples/mms.py --live --lang eng
"""

from __future__ import annotations

import argparse
import queue
import sys
import threading
import time

import numpy as np

from mobius import build_from_module
from mobius._configs import MMSConfig
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


def record_from_mic(sample_rate: int = SAMPLE_RATE, max_seconds: int = 60) -> np.ndarray:
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


def play_audio(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> None:
    """Play audio in a background thread using sounddevice.

    Silently skips if ``sounddevice`` is not installed.
    """
    try:
        import sounddevice as sd
    except ImportError:
        return

    def _play() -> None:
        sd.play(audio, samplerate=sample_rate)
        sd.wait()

    t = threading.Thread(target=_play, daemon=True)
    t.start()


# ---------------------------------------------------------------------------
# Streaming / progressive display helpers
# ---------------------------------------------------------------------------


def _clear_line() -> None:
    """Erase the current terminal line and return cursor to column 0."""
    sys.stdout.write("\r\033[K")
    sys.stdout.flush()


def transcribe_streaming(
    session,
    tokenizer,
    audio: np.ndarray,
    *,
    chunk_seconds: float = 1.0,
    overlap_seconds: float = 0.0,
    lang: str = "eng",
) -> str:
    """Stream-decode audio in short windows and print partial results live.

    Processes the audio in non-overlapping ``chunk_seconds`` windows by
    default and prints each decoded segment to stdout as it is ready,
    overwriting the previous partial line.

    Args:
        session: ORT session wrapping the ONNX MMS model.
        tokenizer: HuggingFace ``Wav2Vec2CTCTokenizer``.
        audio: Raw waveform at 16 kHz, shape ``(num_samples,)``.
        chunk_seconds: Window length in seconds (shorter = more responsive).
        overlap_seconds: Extra audio appended to each window to give the
            CTC decoder more context across boundaries. **Defaults to 0**
            because the current implementation appends each segment's
            decoded text verbatim, so any overlap region is decoded
            twice and the duplicated tokens show up in the final
            transcript. Set this to a small positive value only if you
            accept that trade-off (better acoustic boundaries, possible
            duplicate tokens).
        lang: ISO-639-3 language code used in the progress prefix.

    Returns:
        Final full transcript string.
    """
    chunk_len = int(chunk_seconds * SAMPLE_RATE)
    overlap_len = int(overlap_seconds * SAMPLE_RATE)
    total_samples = len(audio)
    total_duration = total_samples / SAMPLE_RATE

    segments: list[str] = []
    print(f"  [stream] {total_duration:.1f}s audio, {chunk_seconds:.1f}s windows\n")

    pos = 0
    chunk_idx = 0
    while pos < total_samples:
        end = min(pos + chunk_len + overlap_len, total_samples)
        chunk = audio[pos:end]

        input_values = chunk[np.newaxis, :].astype(np.float32)
        attention_mask = np.ones((1, len(chunk)), dtype=np.int64)

        out = session.run({"input_values": input_values, "attention_mask": attention_mask})
        segment_text = ctc_greedy_decode(out["logits"], tokenizer)
        segments.append(segment_text)

        # Progress bar + rolling transcript
        progress = min(end / total_samples, 1.0)
        bar_width = 20
        filled = int(bar_width * progress)
        bar = "█" * filled + "░" * (bar_width - filled)
        elapsed_s = end / SAMPLE_RATE
        partial = " ".join(s for s in segments if s.strip())

        _clear_line()
        sys.stdout.write(f"[{bar}] {elapsed_s:.1f}/{total_duration:.1f}s  📝 {partial!r}")
        sys.stdout.flush()

        pos += chunk_len
        chunk_idx += 1

    # Final newline after streaming output
    sys.stdout.write("\n")
    sys.stdout.flush()

    return " ".join(s for s in segments if s.strip())


def live_microphone_transcription(
    session,
    tokenizer,
    *,
    window_seconds: float = 2.0,
    lang: str = "eng",
    sample_rate: int = SAMPLE_RATE,
) -> None:
    """Transcribe live microphone audio in real time until Ctrl-C.

    Captures audio in ``window_seconds``-length blocks and decodes each
    block as soon as it is filled, printing the result immediately.

    Args:
        session: ORT session wrapping the ONNX MMS model.
        tokenizer: HuggingFace ``Wav2Vec2CTCTokenizer``.
        window_seconds: How many seconds of audio to buffer before decoding.
        lang: ISO-639-3 code shown in the output prefix.
        sample_rate: Input sample rate (must match model, default 16 kHz).
    """
    try:
        import sounddevice as sd
    except ImportError:
        print(
            "ERROR: sounddevice is required for --live.  "
            "Install with: pip install sounddevice",
            file=sys.stderr,
        )
        return

    window_samples = int(window_seconds * sample_rate)
    audio_queue: queue.Queue[np.ndarray | None] = queue.Queue()
    buffer: list[np.ndarray] = []
    buffer_len = 0

    def _callback(indata: np.ndarray, frames: int, time_info, status) -> None:
        if status:
            print(f"\n  [live] {status}", file=sys.stderr)
        chunk = indata[:, 0].copy()  # mono
        audio_queue.put(chunk)

    stream = sd.InputStream(
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
        callback=_callback,
        blocksize=int(sample_rate * 0.05),  # 50 ms callback blocks
    )

    print(f"🎙  Live transcription  [{lang}]  —  Ctrl-C to stop\n")
    stream.start()
    t0 = time.time()

    try:
        while True:
            try:
                chunk = audio_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if chunk is None:
                break

            buffer.append(chunk)
            buffer_len += len(chunk)

            if buffer_len >= window_samples:
                audio_window = np.concatenate(buffer).astype(np.float32)
                buffer.clear()
                buffer_len = 0

                input_values = audio_window[np.newaxis, :]
                attention_mask = np.ones((1, len(audio_window)), dtype=np.int64)
                out = session.run(
                    {"input_values": input_values, "attention_mask": attention_mask}
                )
                text = ctc_greedy_decode(out["logits"], tokenizer)
                elapsed = time.time() - t0
                if text.strip():
                    print(f"[{elapsed:6.1f}s] {text}")
    except KeyboardInterrupt:
        # Expected exit: user pressed Ctrl-C to stop the live capture.
        # Fall through to the ``finally`` block to release the mic stream.
        pass
    finally:
        stream.stop()
        stream.close()
        print("\n  [live] stopped.")


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

        out = session.run({"input_values": input_values, "attention_mask": attention_mask})
        logits = out["logits"]  # (1, time_steps, vocab_size)
        transcripts.append(ctc_greedy_decode(logits, tokenizer))

    return " ".join(t for t in transcripts if t.strip())


# ---------------------------------------------------------------------------
# Language loading helper
# ---------------------------------------------------------------------------


def load_mms_model_and_tokenizer(model_id: str, lang: str):
    """Load HuggingFace MMS processor and set the language adapter.

    Returns ``(hf_model, tokenizer)`` with the correct language adapter
    already loaded into the model's state.  The caller should pass
    ``hf_model.state_dict()`` to :func:`build_mms_package` so the adapter
    weights are baked into the ONNX graph.
    """
    from transformers import AutoProcessor, Wav2Vec2ForCTC

    print(f"  Loading HuggingFace checkpoint for language '{lang}' …")
    processor = AutoProcessor.from_pretrained(model_id)
    hf_model = Wav2Vec2ForCTC.from_pretrained(model_id, ignore_mismatched_sizes=True)

    processor.tokenizer.set_target_lang(lang)
    hf_model.load_adapter(lang)

    return hf_model, processor.tokenizer


def build_mms_package(hf_model, model_id: str):
    """Build an ONNX :class:`ModelPackage` from a weight-loaded HF MMS model.

    This is the correct way to build MMS with a specific language adapter.
    Because adapter weights are loaded into ``hf_model`` via
    ``hf_model.load_adapter(lang)``, we must use :func:`build_from_module`
    with the model's current state dict rather than :func:`build` (which
    re-downloads weights from HuggingFace Hub and would reset the adapter).

    Args:
        hf_model: A ``Wav2Vec2ForCTC`` instance with adapter already loaded.
        model_id: The HuggingFace model ID, used only for graph naming.

    Returns:
        A :class:`ModelPackage` with the single ``"model"`` ONNX graph.
    """
    from mobius.models.wav2vec2_ctc import Wav2Vec2ForCTCModel

    config = MMSConfig.from_transformers(hf_model.config)
    module = Wav2Vec2ForCTCModel(config)
    pkg = build_from_module(module, config, task="ctc-asr")

    # Name the graph after the model ID for easier debugging
    pkg["model"].graph.name = f"{model_id}/model"

    # Apply the HF model weights (including the language adapter that was
    # loaded via hf_model.load_adapter(lang)).
    state_dict = hf_model.state_dict()
    state_dict = module.preprocess_weights(state_dict)
    pkg.apply_weights(state_dict)

    return pkg


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
        help="Record from the default microphone (batch mode).",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help=(
            "Capture from the microphone and transcribe in real time "
            "(requires sounddevice).  Press Ctrl-C to stop."
        ),
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help=(
            "Stream-decode audio with progressive terminal display. "
            "Shows partial transcript as each short window is decoded."
        ),
    )
    parser.add_argument(
        "--play",
        action="store_true",
        help=(
            "Play audio through the default output device while transcribing "
            "(requires sounddevice)."
        ),
    )
    parser.add_argument(
        "--window",
        type=float,
        default=1.0,
        metavar="SECONDS",
        help="Window size for --stream and --live modes.",
    )
    parser.add_argument(
        "--chunk",
        type=float,
        default=30.0,
        metavar="SECONDS",
        help="Audio chunk size for batch transcription of long files.",
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
        print("\nFull list: https://huggingface.co/facebook/mms-1b-all/blob/main/README.md")
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
    pkg = build_mms_package(hf_model, model_id=args.model)

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
    # Live real-time microphone mode (--live)
    # ------------------------------------------------------------------
    if args.live:
        live_microphone_transcription(
            session,
            tokenizer,
            window_seconds=args.window,
            lang=args.lang,
        )
        return

    # ------------------------------------------------------------------
    # Get audio (file / mic / synthetic fallback)
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
    # Optional audio playback (--play)
    # ------------------------------------------------------------------
    if args.play:
        print("▶ Playing audio …")
        play_audio(audio)

    # ------------------------------------------------------------------
    # Transcribe: streaming (--stream) or batch (default)
    # ------------------------------------------------------------------
    if args.stream:
        print("Streaming transcription …")
        text = transcribe_streaming(
            session,
            tokenizer,
            audio,
            chunk_seconds=args.window,
            lang=args.lang,
        )
    else:
        print("Transcribing …")
        text = transcribe(session, tokenizer, audio, chunk_seconds=args.chunk)

    print(f"\n📝 Transcript [{args.lang}]: {text!r}")


if __name__ == "__main__":
    main()
