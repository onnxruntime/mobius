#!/usr/bin/env python
# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""SeamlessM4T v2 example — multilingual text-to-text translation and speech-to-speech.

SeamlessM4T v2 (facebook/seamless-m4t-v2-large) supports 100+ languages for
text translation (T2TT) and, once speech components are available, full
speech-to-speech (S2ST) pipelines.

This script demonstrates:
  - Text-to-text translation with autoregressive decoding
  - Real-time streaming output (token-by-token display)
  - Interactive mode for live translation from stdin
  - Speech-to-speech mode (full ONNX pipeline: speech encoder + decoder + T2U + vocoder)

Architecture (text-to-text path):
    encoder: input_ids → last_hidden_state
    decoder: (last_hidden_state, past_key_values) → logits → next_token

Speech-to-speech pipeline (once ONNX components land):
    speech_encoder:  mel_features → encoder_hidden_states
    decoder:         (encoder_hidden_states, tgt_lang_bos) → text_token_ids
    t2u:             text_token_ids → acoustic_unit_ids
    vocoder:         acoustic_unit_ids → waveform

Usage::

    # Text-to-text translation (English → French)
    python examples/seamless_m4t.py --mode text --text "Hello, how are you?" \\
        --src-lang eng --tgt-lang fra

    # Text-to-text (Spanish → English)
    python examples/seamless_m4t.py --mode text --text "¿Cómo estás?" \\
        --src-lang spa --tgt-lang eng

    # Interactive streaming mode (type to translate in real time)
    python examples/seamless_m4t.py --mode text --interactive \\
        --src-lang eng --tgt-lang fra

    # Speech-to-speech (requires audio file; full ONNX pipeline)
    python examples/seamless_m4t.py --mode speech \\
        --audio input.wav --src-lang eng --tgt-lang fra --output output.wav

Language code examples:
    eng (English), fra (French), spa (Spanish), deu (German),
    cmn (Mandarin), jpn (Japanese), arb (Arabic), por (Portuguese),
    hin (Hindi), ita (Italian)

Notes:
    - Building the ONNX model from the large checkpoint takes ~2 minutes and
      requires ~8 GB RAM.  Pass --save-to to cache the built ONNX files.
    - KV cache is fully supported for efficient autoregressive decoding.
    - Cross-attention KV cache is computed once from the encoder output and
      reused across all decode steps.
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from mobius import build
from mobius._testing.ort_inference import OnnxModelSession

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_ID = "facebook/seamless-m4t-v2-large"
EOS_TOKEN_ID = 3  # </s> token
MAX_NEW_TOKENS = 200


# ---------------------------------------------------------------------------
# Tokenizer helpers
# ---------------------------------------------------------------------------


def load_processor(model_id: str = MODEL_ID):
    """Load the SeamlessM4T processor (tokenizer + feature extractor)."""
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(model_id)


def get_lang_token_id(processor, lang_code: str) -> int:
    """Return the vocabulary ID for a language token like ``__fra__``."""
    token = f"__{lang_code}__"
    token_id = processor.tokenizer.convert_tokens_to_ids(token)
    if token_id == processor.tokenizer.unk_token_id:
        raise ValueError(
            f"Unknown language code: {lang_code!r}. "
            f"Use BCP-47 codes like 'eng', 'fra', 'spa', 'deu'."
        )
    return token_id


def encode_text(processor, text: str, src_lang: str) -> tuple[np.ndarray, np.ndarray]:
    """Tokenize *text* with source-language prefix.

    Returns (input_ids, attention_mask) as int64 numpy arrays shaped (1, seq).
    The tokenizer automatically prepends the ``__src_lang__`` language token.
    """
    encoded = processor.tokenizer(
        text,
        return_tensors="np",
        src_lang=src_lang,
    )
    return (
        encoded["input_ids"].astype(np.int64),
        encoded["attention_mask"].astype(np.int64),
    )


# ---------------------------------------------------------------------------
# ONNX inference helpers — encoder + decoder with KV cache
# ---------------------------------------------------------------------------


def run_encoder(
    enc_session: OnnxModelSession,
    input_ids: np.ndarray,
    attention_mask: np.ndarray,
) -> np.ndarray:
    """Run the encoder ONNX model.

    Returns last_hidden_state shaped (1, enc_seq_len, hidden_size).
    """
    outputs = enc_session.run({"input_ids": input_ids, "attention_mask": attention_mask})
    return outputs["last_hidden_state"]


def _make_empty_kv_cache(
    num_decoder_layers: int,
    num_heads: int,
    head_dim: int,
) -> dict[str, np.ndarray]:
    """Return a dict of empty KV-cache arrays for the decoder.

    The decoder expects two kinds of past KV:
      - Self-attention:  ``past_key_values.{i}.self.key/value``  shape (1, heads, 0, head_dim)
      - Cross-attention: ``past_key_values.{i}.cross.key/value`` shape (1, heads, 0, head_dim)
        (starts empty; filled after first decode step)
    """
    kv: dict[str, np.ndarray] = {}
    for i in range(num_decoder_layers):
        for kind in ("self", "cross"):
            for tensor in ("key", "value"):
                kv[f"past_key_values.{i}.{kind}.{tensor}"] = np.zeros(
                    (1, num_heads, 0, head_dim), dtype=np.float32
                )
    return kv


def _update_kv_cache(
    kv: dict[str, np.ndarray],
    outputs: dict[str, np.ndarray],
    num_decoder_layers: int,
) -> None:
    """Consume ``present.*`` tensors from *outputs* and update *kv* in-place."""
    for i in range(num_decoder_layers):
        for kind in ("self", "cross"):
            for tensor in ("key", "value"):
                present_name = f"present.{i}.{kind}.{tensor}"
                past_name = f"past_key_values.{i}.{kind}.{tensor}"
                if present_name in outputs:
                    kv[past_name] = outputs[present_name]


def translate_onnx(
    enc_session: OnnxModelSession,
    dec_session: OnnxModelSession,
    input_ids: np.ndarray,
    attention_mask: np.ndarray,
    tgt_lang_bos_id: int,
    num_decoder_layers: int,
    num_heads: int,
    head_dim: int,
    max_new_tokens: int = MAX_NEW_TOKENS,
    stream: bool = False,
) -> list[int]:
    """Greedy autoregressive decode using the encoder and decoder ONNX sessions.

    Args:
        enc_session: OnnxModelSession wrapping the encoder IR model.
        dec_session: OnnxModelSession wrapping the decoder IR model.
        input_ids: Source token IDs, shape (1, src_seq_len).
        attention_mask: Attention mask, shape (1, src_seq_len).
        tgt_lang_bos_id: Target language BOS token ID (e.g. 256026 for French).
        num_decoder_layers: Number of decoder layers (24 for large).
        num_heads: Number of attention heads (16 for large).
        head_dim: Per-head dimension (64 for large).
        max_new_tokens: Maximum tokens to generate.
        stream: If True, print each token to stdout as it is generated.

    Returns:
        List of generated token IDs (excluding BOS).
    """
    # 1. Encode source
    encoder_hidden_states = run_encoder(enc_session, input_ids, attention_mask)

    # 2. Initialise decode state
    # Decoder attention mask covers the full encoder sequence
    enc_seq_len = input_ids.shape[1]
    dec_attention_mask = np.ones((1, enc_seq_len), dtype=np.int64)

    kv = _make_empty_kv_cache(num_decoder_layers, num_heads, head_dim)
    cur_token = np.array([[tgt_lang_bos_id]], dtype=np.int64)
    generated: list[int] = []
    past_dec_len = 0

    for step in range(max_new_tokens):
        # Extend attention mask by one for the current decode step
        dec_attn = np.ones((1, past_dec_len + 1), dtype=np.int64)

        feeds: dict[str, np.ndarray] = {
            "input_ids": cur_token,
            "encoder_hidden_states": encoder_hidden_states,
            "attention_mask": dec_attn,
            **kv,
        }

        t0 = time.perf_counter()
        outputs = dec_session.run(feeds)
        dt = time.perf_counter() - t0

        # Greedy selection over the last token position
        logits = outputs["logits"]  # (1, 1, vocab_size)
        next_token_id = int(np.argmax(logits[0, -1]))

        if next_token_id == EOS_TOKEN_ID:
            break

        generated.append(next_token_id)
        _update_kv_cache(kv, outputs, num_decoder_layers)
        cur_token = np.array([[next_token_id]], dtype=np.int64)
        past_dec_len += 1

        if stream:
            # Will be decoded outside; yield timing info to stderr
            print(
                f"  step {step + 1:3d}: token={next_token_id} ({dt * 1000:.1f} ms)",
                file=sys.stderr,
            )

    return generated


# ---------------------------------------------------------------------------
# Speech encoder helper
# ---------------------------------------------------------------------------


def run_speech_encoder(
    speech_enc_session: OnnxModelSession,
    audio_array: np.ndarray,
    sample_rate: int,
    processor,
    src_lang: str,
) -> np.ndarray:
    """Extract mel-filterbank features and run the ONNX speech encoder.

    Args:
        speech_enc_session: OnnxModelSession wrapping the speech_encoder IR model.
        audio_array: Raw waveform samples (float32, mono).
        sample_rate: Sample rate of *audio_array* (typically 16 000 Hz).
        processor: HuggingFace AutoProcessor for SeamlessM4T.
        src_lang: BCP-47 source language code (e.g. ``"eng"``).

    Returns:
        encoder_hidden_states shaped (1, enc_seq_len, hidden_size) as float32.
    """
    features = processor(
        audios=audio_array,
        sampling_rate=sample_rate,
        return_tensors="np",
        src_lang=src_lang,
    )
    # Feature extractor produces (1, time_frames, num_mel_bins) float32
    input_features = features["input_features"].astype(np.float32)
    outputs = speech_enc_session.run({"input_features": input_features})
    return outputs["last_hidden_state"]  # (1, enc_seq_len, hidden_size)


# ---------------------------------------------------------------------------
# T2U (text-to-unit) helper
# ---------------------------------------------------------------------------


def run_t2u(
    t2u_session: OnnxModelSession,
    text_token_ids: list[int],
) -> np.ndarray:
    """Run the non-autoregressive T2U model.

    Takes all text token IDs in a single forward pass and returns the
    most likely acoustic unit at each position.

    Args:
        t2u_session: OnnxModelSession wrapping the t2u IR model.
        text_token_ids: Token IDs generated by the text decoder (no BOS/EOS).

    Returns:
        unit_ids shaped (1, T) as int64.
    """
    input_ids = np.array([text_token_ids], dtype=np.int64)  # (1, T)
    outputs = t2u_session.run({"input_ids": input_ids})
    logits = outputs["logits"]  # (1, T, t2u_vocab_size)
    return np.argmax(logits, axis=-1).astype(np.int64)  # (1, T)


# ---------------------------------------------------------------------------
# Vocoder helpers
# ---------------------------------------------------------------------------


def _load_vocoder_lang_map(model_id: str = MODEL_ID) -> dict[str, int]:
    """Return BCP-47 code → vocoder language index from generation config.

    The vocoder has 36 languages indexed 0–35.  The mapping is stored in
    ``generation_config.vocoder_lang_code_to_id``.
    """
    from transformers import GenerationConfig

    gc = GenerationConfig.from_pretrained(model_id)
    return dict(gc.vocoder_lang_code_to_id)


def run_vocoder(
    dur_session: OnnxModelSession,
    hifigan_session: OnnxModelSession,
    unit_ids: np.ndarray,
    lang_id: int = 0,
    speaker_id: int = 0,
) -> np.ndarray:
    """Predict durations, expand units, and synthesize a waveform.

    Pipeline:
        1. ``vocoder_dur``:  unit_ids → log_dur (per-unit log duration)
        2. Round ``exp(log_dur)`` to integer repeat counts (min 1).
        3. Repeat-interleave unit_ids by their counts → expanded_unit_ids.
        4. ``vocoder_hifigan``: expanded_unit_ids + lang_id + speaker_id → waveform.

    Args:
        dur_session: OnnxModelSession wrapping the vocoder_dur IR model.
        hifigan_session: OnnxModelSession wrapping the vocoder_hifigan IR model.
        unit_ids: Acoustic unit IDs shaped (1, T) as int64.
        lang_id: Vocoder language index (from :func:`_load_vocoder_lang_map`).
        speaker_id: Speaker embedding index (0 = default).

    Returns:
        Waveform as a 1-D float32 numpy array at 16 kHz.
    """
    # 1. Predict log durations: (1, T) float32
    log_dur = dur_session.run({"unit_ids": unit_ids})["log_dur"]
    # 2. Integer durations, at least 1 frame each
    durations = np.maximum(1, np.round(np.exp(log_dur[0])).astype(np.int32))  # (T,)
    # 3. Expand unit sequence via repeat_interleave
    expanded = np.repeat(unit_ids[0], durations)[np.newaxis, :]  # (1, T_expanded)
    # 4. Synthesize waveform
    lang_arr = np.array([[lang_id]], dtype=np.int64)
    spkr_arr = np.array([[speaker_id]], dtype=np.int64)
    out = hifigan_session.run(
        {"unit_ids": expanded, "speaker_id": spkr_arr, "lang_id": lang_arr}
    )
    waveform = out["waveform"]
    return waveform.reshape(-1).astype(np.float32)  # flatten to (samples,)


# ---------------------------------------------------------------------------
# Text-to-text translation
# ---------------------------------------------------------------------------


def text_to_text(
    args: argparse.Namespace,
    pkg,
    processor,
) -> str:
    """Translate a single sentence using the ONNX encoder/decoder."""
    from mobius._configs import SeamlessM4Tv2Config

    cfg: SeamlessM4Tv2Config = pkg.config  # type: ignore[assignment]
    num_decoder_layers = cfg.num_decoder_layers
    num_heads = cfg.num_attention_heads
    head_dim = cfg.head_dim

    enc_session = OnnxModelSession(pkg["encoder"])
    dec_session = OnnxModelSession(pkg["decoder"])

    input_ids, attention_mask = encode_text(processor, args.text, args.src_lang)
    tgt_bos = get_lang_token_id(processor, args.tgt_lang)

    print(f"Translating ({args.src_lang} → {args.tgt_lang}): {args.text!r}", flush=True)
    t_start = time.perf_counter()

    token_ids = translate_onnx(
        enc_session,
        dec_session,
        input_ids,
        attention_mask,
        tgt_lang_bos_id=tgt_bos,
        num_decoder_layers=num_decoder_layers,
        num_heads=num_heads,
        head_dim=head_dim,
        max_new_tokens=MAX_NEW_TOKENS,
        stream=args.verbose,
    )

    elapsed = time.perf_counter() - t_start
    translation = processor.tokenizer.decode(token_ids, skip_special_tokens=True)
    print(f"Translation: {translation}")
    print(f"({len(token_ids)} tokens in {elapsed:.2f}s, {len(token_ids) / elapsed:.1f} tok/s)")
    return translation


def interactive_text_to_text(
    args: argparse.Namespace,
    pkg,
    processor,
) -> None:
    """Interactive loop: type a sentence, get streaming translation."""
    from mobius._configs import SeamlessM4Tv2Config

    cfg: SeamlessM4Tv2Config = pkg.config  # type: ignore[assignment]
    num_decoder_layers = cfg.num_decoder_layers
    num_heads = cfg.num_attention_heads
    head_dim = cfg.head_dim

    enc_session = OnnxModelSession(pkg["encoder"])
    dec_session = OnnxModelSession(pkg["decoder"])

    print(
        f"Interactive mode ({args.src_lang} → {args.tgt_lang}). Ctrl+C or blank line to exit."
    )
    while True:
        try:
            text = input("\nSource: ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            break
        if not text:
            break

        input_ids, attention_mask = encode_text(processor, text, args.src_lang)
        tgt_bos = get_lang_token_id(processor, args.tgt_lang)

        print("Translation: ", end="", flush=True)
        t_start = time.perf_counter()
        partial_ids: list[int] = []

        # Encoder pass
        encoder_hidden_states = run_encoder(enc_session, input_ids, attention_mask)
        kv = _make_empty_kv_cache(num_decoder_layers, num_heads, head_dim)
        cur_token = np.array([[tgt_bos]], dtype=np.int64)
        past_dec_len = 0

        for _step in range(MAX_NEW_TOKENS):
            dec_attn = np.ones((1, past_dec_len + 1), dtype=np.int64)
            feeds: dict[str, np.ndarray] = {
                "input_ids": cur_token,
                "encoder_hidden_states": encoder_hidden_states,
                "attention_mask": dec_attn,
                **kv,
            }
            outputs = dec_session.run(feeds)
            logits = outputs["logits"]
            next_token_id = int(np.argmax(logits[0, -1]))

            if next_token_id == EOS_TOKEN_ID:
                break

            partial_ids.append(next_token_id)
            _update_kv_cache(kv, outputs, num_decoder_layers)
            cur_token = np.array([[next_token_id]], dtype=np.int64)
            past_dec_len += 1

            # Stream decoded text character-by-character
            word = processor.tokenizer.decode(partial_ids, skip_special_tokens=True)
            # Overwrite line to show growing translation
            print(f"\rTranslation: {word}", end="", flush=True)

        elapsed = time.perf_counter() - t_start
        final = processor.tokenizer.decode(partial_ids, skip_special_tokens=True)
        print(f"\rTranslation: {final}")
        print(
            f"  ({len(partial_ids)} tokens, {elapsed:.2f}s, {len(partial_ids) / max(elapsed, 1e-6):.1f} tok/s)"
        )


# ---------------------------------------------------------------------------
# Speech-to-speech pipeline
# ---------------------------------------------------------------------------


def speech_to_speech(
    args: argparse.Namespace,
    pkg,
    processor,
) -> None:
    """Speech-to-speech translation using the 5-model ONNX pipeline.

    Pipeline:
        audio → speech_encoder → encoder_hidden_states
             → decoder         → text_token_ids   (autoregressive)
             → t2u             → acoustic_unit_ids (non-autoregressive)
             → vocoder_dur     → log_dur
             → vocoder_hifigan → waveform
    """
    import soundfile as sf

    from mobius._configs import SeamlessM4Tv2Config

    cfg: SeamlessM4Tv2Config = pkg.config  # type: ignore[assignment]

    # Load audio
    audio_array, sample_rate = sf.read(args.audio)
    if audio_array.ndim > 1:
        audio_array = audio_array[:, 0]  # take first channel (mono)
    audio_array = audio_array.astype(np.float32)

    print(
        f"Loaded audio: {args.audio} ({len(audio_array) / sample_rate:.1f}s @ {sample_rate} Hz)"
    )
    print(f"Translating speech: {args.src_lang} → {args.tgt_lang}")

    # Create ONNX sessions for all five pipeline models
    speech_enc_session = OnnxModelSession(pkg["speech_encoder"])
    dec_session = OnnxModelSession(pkg["decoder"])
    t2u_session = OnnxModelSession(pkg["t2u"])
    dur_session = OnnxModelSession(pkg["vocoder_dur"])
    hifigan_session = OnnxModelSession(pkg["vocoder_hifigan"])

    # ── Step 1: Speech encoder ── audio → encoder hidden states ──────────
    print("  1/4 Speech encoder …", flush=True)
    t0 = time.perf_counter()
    encoder_hidden_states = run_speech_encoder(
        speech_enc_session, audio_array, sample_rate, processor, args.src_lang
    )
    print(f"       shape={encoder_hidden_states.shape}  ({time.perf_counter() - t0:.1f}s)")

    # ── Step 2: Text decoder ── encoder states → text token IDs ──────────
    # Greedy autoregressive decoding with self + cross KV cache.
    print("  2/4 Decoding text …", flush=True)
    tgt_bos = get_lang_token_id(processor, args.tgt_lang)
    kv = _make_empty_kv_cache(cfg.num_decoder_layers, cfg.num_attention_heads, cfg.head_dim)
    cur_token = np.array([[tgt_bos]], dtype=np.int64)
    text_ids: list[int] = []
    past_dec_len = 0
    t0 = time.perf_counter()

    for _ in range(MAX_NEW_TOKENS):
        # Attention mask grows by 1 each step (covers full decoder self-attn history)
        dec_attn = np.ones((1, past_dec_len + 1), dtype=np.int64)
        feeds: dict[str, np.ndarray] = {
            "input_ids": cur_token,
            "encoder_hidden_states": encoder_hidden_states,
            "attention_mask": dec_attn,
            **kv,
        }
        outputs = dec_session.run(feeds)
        next_token_id = int(np.argmax(outputs["logits"][0, -1]))
        if next_token_id == EOS_TOKEN_ID:
            break
        text_ids.append(next_token_id)
        _update_kv_cache(kv, outputs, cfg.num_decoder_layers)
        cur_token = np.array([[next_token_id]], dtype=np.int64)
        past_dec_len += 1

    print(f"       {len(text_ids)} tokens  ({time.perf_counter() - t0:.1f}s)")
    if args.verbose or not text_ids:
        text = processor.tokenizer.decode(text_ids, skip_special_tokens=True)
        print(f"  Intermediate text: {text!r}")

    # ── Step 3: T2U ── text token IDs → acoustic unit IDs ────────────────
    # Non-autoregressive: a single forward pass over all text tokens.
    print("  3/4 Text-to-unit …", flush=True)
    t0 = time.perf_counter()
    unit_ids = run_t2u(t2u_session, text_ids)
    print(f"       {unit_ids.shape[1]} units  ({time.perf_counter() - t0:.2f}s)")

    # ── Step 4: Vocoder ── acoustic units → waveform ──────────────────────
    print("  4/4 Vocoder …", flush=True)
    lang_map = _load_vocoder_lang_map()
    lang_id = lang_map.get(args.tgt_lang, 0)
    t0 = time.perf_counter()
    waveform = run_vocoder(dur_session, hifigan_session, unit_ids, lang_id=lang_id)
    output_sr = 16000
    print(
        f"       {len(waveform) / output_sr:.1f}s waveform  ({time.perf_counter() - t0:.1f}s)"
    )

    if args.output:
        sf.write(args.output, waveform, output_sr)
        print(f"Saved to: {args.output}")
    else:
        try:
            import sounddevice as sd

            print("Playing output audio …")
            sd.play(waveform, output_sr)
            sd.wait()
        except ImportError:
            print("Install sounddevice to play audio: pip install sounddevice")
            print(
                f"Waveform: {len(waveform)} samples, {len(waveform) / output_sr:.1f}s @ {output_sr} Hz"
            )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SeamlessM4T v2 translation example",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--mode",
        choices=["text", "speech"],
        default="text",
        help="text: text-to-text translation; speech: speech-to-speech (default: text)",
    )
    p.add_argument(
        "--text",
        type=str,
        default="Hello, how are you today?",
        help="Source text to translate (text mode only)",
    )
    p.add_argument(
        "--audio",
        type=str,
        default=None,
        help="Path to source audio WAV file (speech mode)",
    )
    p.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save translated audio WAV (speech mode)",
    )
    p.add_argument("--src-lang", default="eng", help="Source language code (default: eng)")
    p.add_argument("--tgt-lang", default="fra", help="Target language code (default: fra)")
    p.add_argument(
        "--interactive",
        action="store_true",
        help="Interactive mode: type text and get streaming translation",
    )
    p.add_argument(
        "--save-to",
        type=str,
        default=None,
        help="Directory to save the built ONNX models for reuse",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-step timing to stderr",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.mode == "speech" and args.audio is None:
        # Allow running without audio for quick syntax/import check
        print("Speech mode requires --audio <file.wav>")
        sys.exit(1)

    # Build ONNX model package — different model class per mode:
    #   text:   SeamlessM4Tv2Model            → pkg["encoder"], pkg["decoder"]
    #   speech: SeamlessM4Tv2SpeechToSpeechModel → pkg["speech_encoder"],
    #           pkg["decoder"], pkg["t2u"], pkg["vocoder_dur"], pkg["vocoder_hifigan"]
    print(f"Building ONNX model from {MODEL_ID} …", flush=True)
    t0 = time.perf_counter()
    if args.mode == "speech":
        from mobius.models.seamless_m4t_v2 import SeamlessM4Tv2SpeechToSpeechModel

        pkg = build(MODEL_ID, module_class=SeamlessM4Tv2SpeechToSpeechModel)
    else:
        pkg = build(MODEL_ID)
    print(f"Build complete ({time.perf_counter() - t0:.1f}s)")

    if args.save_to:
        from pathlib import Path

        import onnx_ir as ir

        save_dir = Path(args.save_to)
        save_dir.mkdir(parents=True, exist_ok=True)
        for name, model in pkg.items():
            path = save_dir / f"{name}.onnx"
            ir.save(model, str(path))
            print(f"Saved {name} → {path}")

    processor = load_processor(MODEL_ID)

    if args.mode == "text":
        if args.interactive:
            interactive_text_to_text(args, pkg, processor)
        else:
            text_to_text(args, pkg, processor)
    else:
        speech_to_speech(args, pkg, processor)


if __name__ == "__main__":
    main()
