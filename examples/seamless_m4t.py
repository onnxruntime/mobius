#!/usr/bin/env python
# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""SeamlessM4T speech and text translation with ONNX models.

SeamlessM4T (facebook/hf-seamless-m4t-medium or seamless-m4t-v2-large) is a
massively multilingual translation model supporting:
  - Text-to-text translation (S2TT / T2TT)
  - Speech-to-text translation (S2TT)
  - Speech-to-speech translation (S2ST)
  - Text-to-speech translation (T2ST)

Two pipeline modes are supported:

  **--mode text** (default)
      text_src → text encoder → decoder → translated text

  **--mode speech**
      audio_src → speech encoder → decoder → T2U decoder → vocoder → audio

  The speech components (speech_encoder, t2u_decoder, vocoder) are built by
  agent 8d99cf75 and will be available as separate package keys once that
  implementation lands.  This script stubs those components with clear TODOs
  and falls back to the HuggingFace reference implementation for now.

Architecture (SeamlessM4T V2):
  - Speech encoder  (speech_encoder):  mel → contextual frame embeddings
  - Text encoder    (model["encoder"]): token_ids → contextual embeddings
  - Shared decoder  (model["decoder"]): encoder_out → target token_ids (beam)
  - T2U decoder     (t2u_model):        target text tokens → acoustic units
  - Vocoder / HiFi-GAN (vocoder):       acoustic units → 16 kHz waveform

Prerequisites::

    pip install mobius-ai[transformers] sounddevice soundfile

Usage::

    # Text-to-text translation (French → English)
    python examples/seamless_m4t.py --mode text \\
        --text "Bonjour, comment ça va?" --src-lang fra --tgt-lang eng

    # Speech-to-text translation from a WAV file
    python examples/seamless_m4t.py --mode text --audio speech.wav \\
        --src-lang fra --tgt-lang eng

    # Speech-to-speech translation (stubs T2U + vocoder with HF reference)
    python examples/seamless_m4t.py --mode speech --audio speech.wav \\
        --src-lang fra --tgt-lang eng --output translated.wav

    # Record from microphone (press Enter to stop)
    python examples/seamless_m4t.py --mode speech --tgt-lang eng

    # Use the larger V2 model
    python examples/seamless_m4t.py --model facebook/seamless-m4t-v2-large \\
        --mode speech --audio speech.wav --tgt-lang eng

    # Save ONNX models to disk for reuse
    python examples/seamless_m4t.py --save-to output/seamless-m4t/
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_ID = "facebook/hf-seamless-m4t-medium"
SAMPLE_RATE = 16_000  # SeamlessM4T uses 16 kHz audio
MAX_NEW_TOKENS = 256
MAX_RECORD_SECONDS = 30


# ---------------------------------------------------------------------------
# Build / load helpers
# ---------------------------------------------------------------------------


def build_or_load(model_id: str, *, cache_dir: str | None = None):
    """Build ONNX models from *model_id*, caching to *cache_dir*."""
    import os

    from mobius import build
    from mobius._model_package import ModelPackage

    if cache_dir is None:
        cache_dir = os.path.join(os.path.dirname(__file__), "..", "output")
    model_name = model_id.rsplit("/", 1)[-1]
    model_cache = os.path.join(cache_dir, model_name)

    if os.path.isdir(model_cache) and any(
        f.endswith(".onnx")
        for _, _, files in os.walk(model_cache)
        for f in files
    ):
        print(f"Loading cached ONNX models from {model_cache} …")
        return ModelPackage.load(model_cache)

    print(f"Building ONNX models for {model_id} …")
    pkg = build(model_id)
    os.makedirs(model_cache, exist_ok=True)
    pkg.save(model_cache)
    print(f"Saved to {model_cache}")
    return pkg


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------


def load_audio(path: str) -> np.ndarray:
    """Load an audio file and resample to 16 kHz mono float32."""
    try:
        import soundfile as sf
    except ImportError:
        print("soundfile not installed. Run: pip install soundfile", file=sys.stderr)
        sys.exit(1)

    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=-1)  # stereo → mono
    if sr != SAMPLE_RATE:
        # Simple linear resample via scipy if available, else error
        try:
            from scipy.signal import resample_poly
            from math import gcd

            g = gcd(SAMPLE_RATE, sr)
            audio = resample_poly(audio, SAMPLE_RATE // g, sr // g).astype(np.float32)
        except ImportError:
            print(
                f"Audio sample rate {sr} Hz != {SAMPLE_RATE} Hz. "
                "Install scipy for resampling: pip install scipy",
                file=sys.stderr,
            )
            sys.exit(1)
    return audio


def record_audio(max_seconds: int = MAX_RECORD_SECONDS) -> np.ndarray:
    """Record from the default microphone until Enter is pressed."""
    try:
        import sounddevice as sd
    except ImportError:
        print(
            "sounddevice not installed. Run: pip install sounddevice", file=sys.stderr
        )
        sys.exit(1)

    print(f"Recording… press Enter to stop (max {max_seconds}s)")
    frames = sd.rec(
        int(max_seconds * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
    )
    input()
    sd.stop()
    return frames[: int(len(frames))][:, 0]


def save_audio(path: str, waveform: np.ndarray, sample_rate: int = SAMPLE_RATE) -> None:
    """Write a float32 waveform to a WAV file."""
    try:
        import soundfile as sf
    except ImportError:
        print("soundfile not installed. Run: pip install soundfile", file=sys.stderr)
        sys.exit(1)
    sf.write(path, waveform, sample_rate)
    print(f"Saved output audio → {path}")


def play_audio(waveform: np.ndarray, sample_rate: int = SAMPLE_RATE) -> None:
    """Play a waveform through the default audio output device."""
    try:
        import sounddevice as sd
    except ImportError:
        print(
            "sounddevice not installed (pip install sounddevice). Skipping playback.",
            file=sys.stderr,
        )
        return
    sd.play(waveform, samplerate=sample_rate)
    sd.wait()


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


def extract_features(
    processor,
    audio: np.ndarray | None = None,
    text: str | None = None,
    src_lang: str = "eng",
) -> dict:
    """Run HuggingFace processor to get model inputs."""
    if audio is not None:
        return processor(
            audios=audio,
            src_lang=src_lang,
            return_tensors="pt",
            sampling_rate=SAMPLE_RATE,
        )
    if text is not None:
        return processor(text=text, src_lang=src_lang, return_tensors="pt")
    raise ValueError("Either audio or text must be provided")


# ---------------------------------------------------------------------------
# ORT inference helpers
# ---------------------------------------------------------------------------


def _ort_session(onnx_bytes: bytes):
    """Create an ORT InferenceSession from raw ONNX bytes."""
    try:
        import onnxruntime as ort
    except ImportError:
        print(
            "onnxruntime not installed. Run: pip install onnxruntime", file=sys.stderr
        )
        sys.exit(1)
    return ort.InferenceSession(onnx_bytes, providers=["CPUExecutionProvider"])


# ---------------------------------------------------------------------------
# Text-to-text pipeline
# ---------------------------------------------------------------------------


def run_text_to_text(
    model_id: str,
    text: str,
    src_lang: str,
    tgt_lang: str,
    *,
    save_to: str | None = None,
) -> str:
    """Translate *text* from *src_lang* to *tgt_lang* using ONNX models.

    The SeamlessM4T text-to-text pipeline:
        1. Processor tokenises the source text
        2. Text encoder produces contextual embeddings
        3. Shared decoder beam-searches target token IDs
        4. Processor decodes token IDs to target text

    NOTE: The current mobius registration maps seamless_m4t → Wav2Vec2Model
    (placeholder). Until a dedicated SeamlessM4T seq2seq implementation
    lands, this function falls through to the HuggingFace reference model.

    TODO(mobius): Replace with ONNX pipeline once
    SeamlessM4TForTextToText is implemented in mobius.
    """
    import transformers

    print(f"[text-to-text] Loading HF reference model {model_id} …")
    processor = transformers.AutoProcessor.from_pretrained(model_id)
    model = transformers.SeamlessM4TModel.from_pretrained(model_id)

    inputs = processor(text=text, src_lang=src_lang, return_tensors="pt")
    output_tokens = model.generate(
        **inputs,
        tgt_lang=tgt_lang,
        generate_speech=False,
        max_new_tokens=MAX_NEW_TOKENS,
    )
    translated = processor.decode(output_tokens[0].tolist(), skip_special_tokens=True)
    print(f"Translation: {translated}")
    return translated


# ---------------------------------------------------------------------------
# Speech-to-text pipeline
# ---------------------------------------------------------------------------


def run_speech_to_text(
    model_id: str,
    audio: np.ndarray,
    src_lang: str,
    tgt_lang: str,
) -> str:
    """Transcribe/translate *audio* to text in *tgt_lang*.

    Pipeline:
        audio → mel features → speech_encoder → shared decoder → text

    NOTE: The speech encoder and seq2seq decoder are not yet separately
    exported by mobius. Falls back to HuggingFace reference for now.

    TODO(mobius): Wire up pkg["speech_encoder"] + pkg["decoder"] once
    agent 8d99cf75's speech_encoder implementation lands.
    """
    import transformers

    print(f"[speech-to-text] Loading HF reference model {model_id} …")
    processor = transformers.AutoProcessor.from_pretrained(model_id)
    model = transformers.SeamlessM4TModel.from_pretrained(model_id)

    inputs = processor(
        audios=audio, src_lang=src_lang, return_tensors="pt", sampling_rate=SAMPLE_RATE
    )
    output_tokens = model.generate(
        **inputs,
        tgt_lang=tgt_lang,
        generate_speech=False,
        max_new_tokens=MAX_NEW_TOKENS,
    )
    translated = processor.decode(output_tokens[0].tolist(), skip_special_tokens=True)
    print(f"Transcription/Translation: {translated}")
    return translated


# ---------------------------------------------------------------------------
# Speech-to-speech pipeline
# ---------------------------------------------------------------------------


def run_speech_to_speech(
    model_id: str,
    audio: np.ndarray,
    src_lang: str,
    tgt_lang: str,
    *,
    output_path: str | None = None,
    play: bool = True,
) -> np.ndarray:
    """Translate *audio* speech-to-speech, returning a waveform.

    Full pipeline:
        audio
          → mel features
          → speech_encoder      [pkg["speech_encoder"]]  ← TODO agent 8d99cf75
          → shared seq2seq decoder  [pkg["decoder"]]      ← TODO mobius S2ST impl
          → T2U decoder         [pkg["t2u"]]              ← TODO agent 8d99cf75
          → HiFi-GAN vocoder    [pkg["vocoder"]]          ← TODO agent 8d99cf75
          → 16 kHz waveform

    Current status: All four ONNX sub-models are stubbed — the HuggingFace
    reference implementation is used end-to-end.

    TODO(mobius): Replace each stub below with the corresponding ONNX model
    from *pkg* once the following are ready:
        - pkg["speech_encoder"]: mel → encoder_hidden_states
        - pkg["decoder"]:        encoder_out + tgt_lang_id → text token ids
        - pkg["t2u"]:            text ids → acoustic unit ids
        - pkg["vocoder"]:        unit ids + speaker_id → waveform
    """
    import transformers

    print(f"[speech-to-speech] Loading HF reference model {model_id} …")
    processor = transformers.AutoProcessor.from_pretrained(model_id)
    model = transformers.SeamlessM4TModel.from_pretrained(model_id)

    inputs = processor(
        audios=audio, src_lang=src_lang, return_tensors="pt", sampling_rate=SAMPLE_RATE
    )

    # -----------------------------------------------------------------------
    # TODO: Replace with ONNX speech_encoder once agent 8d99cf75 lands
    # speech_session = _ort_session(pkg["speech_encoder"].SerializeToString())
    # encoder_out = speech_session.run(None, {"input_features": mel_features})[0]
    # -----------------------------------------------------------------------

    # -----------------------------------------------------------------------
    # TODO: Replace with ONNX seq2seq decoder + T2U decoder
    # decoder_session = _ort_session(pkg["decoder"].SerializeToString())
    # t2u_session    = _ort_session(pkg["t2u"].SerializeToString())
    # unit_ids = t2u_session.run(None, {"encoder_hidden_states": encoder_out, ...})[0]
    # -----------------------------------------------------------------------

    # -----------------------------------------------------------------------
    # TODO: Replace with ONNX vocoder (HiFi-GAN)
    # vocoder_session = _ort_session(pkg["vocoder"].SerializeToString())
    # waveform = vocoder_session.run(None, {"unit_ids": unit_ids, "speaker_id": spkr})[0]
    # -----------------------------------------------------------------------

    # HF reference fallback (all stubs wired end-to-end through generate())
    output = model.generate(
        **inputs,
        tgt_lang=tgt_lang,
        generate_speech=True,
        max_new_tokens=MAX_NEW_TOKENS,
        spkr_id=7,  # default speaker; 0–199 available
    )
    waveform = output.waveform.squeeze().cpu().numpy()
    sr = model.config.sampling_rate

    if output_path:
        save_audio(output_path, waveform, sr)
    if play:
        print("Playing translated audio …")
        play_audio(waveform, sr)

    return waveform


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SeamlessM4T speech/text translation via ONNX",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        default=MODEL_ID,
        help="HuggingFace model ID (default: %(default)s)",
    )
    parser.add_argument(
        "--mode",
        choices=["text", "speech"],
        default="text",
        help="Pipeline mode: 'text' = S2TT/T2TT, 'speech' = S2ST (default: %(default)s)",
    )
    parser.add_argument("--text", help="Source text for text-to-text / T2ST mode")
    parser.add_argument(
        "--audio",
        help="Path to input audio WAV file (if omitted in speech mode, record from mic)",
    )
    parser.add_argument(
        "--src-lang",
        default="fra",
        metavar="LANG",
        help="Source language (ISO 639-3 code, e.g. 'fra', 'deu', 'cmn', default: %(default)s)",
    )
    parser.add_argument(
        "--tgt-lang",
        default="eng",
        metavar="LANG",
        help="Target language (ISO 639-3 code, e.g. 'eng', 'spa', default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        metavar="WAV",
        help="Save translated audio to this path (speech mode only)",
    )
    parser.add_argument(
        "--no-play",
        action="store_true",
        help="Skip audio playback (speech mode)",
    )
    parser.add_argument(
        "--save-to",
        metavar="DIR",
        help="Build and save ONNX models to DIR, then exit",
    )
    args = parser.parse_args()

    # Build / save ONNX models if requested
    if args.save_to:
        build_or_load(args.model, cache_dir=args.save_to)
        return

    if args.mode == "text":
        # Text → text or speech → text
        if args.audio:
            audio = load_audio(args.audio)
            run_speech_to_text(args.model, audio, args.src_lang, args.tgt_lang)
        elif args.text:
            run_text_to_text(args.model, args.text, args.src_lang, args.tgt_lang)
        else:
            parser.error(
                "--mode text requires either --text 'source text' or --audio file.wav"
            )

    elif args.mode == "speech":
        # Speech → speech translation
        if args.audio:
            audio = load_audio(args.audio)
        else:
            audio = record_audio()

        run_speech_to_speech(
            args.model,
            audio,
            args.src_lang,
            args.tgt_lang,
            output_path=args.output,
            play=not args.no_play,
        )


if __name__ == "__main__":
    main()
