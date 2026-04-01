#!/usr/bin/env python
# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Moshi/PersonaPlex real-time full-duplex speech conversation.

Moshi is a speech-to-speech model that simultaneously listens and speaks.
It encodes incoming audio as RVQ codec tokens, processes them through a
causal transformer backbone, and generates both text tokens (inner
monologue) and audio tokens (spoken response) autoregressively.

This example demonstrates the full streaming inference loop:

    microphone → RVQ encode → embedding → decoder → audio_decoder
                → RVQ decode → speakers

The pipeline runs in a dual-stream configuration:
- **Input stream**: mic audio → codec encoder → audio_codes input to the model
- **Output stream**: model generates audio_codes → codec decoder → speaker

Prerequisites::

    pip install mobius-ai[transformers] sounddevice numpy onnxruntime moshi

Usage::

    # Interactive conversation with PersonaPlex-7B
    python examples/moshi_realtime.py

    # Use base Moshi model
    python examples/moshi_realtime.py --model kyutai/moshiko-pytorch-bf16

    # Export ONNX models only (no inference)
    python examples/moshi_realtime.py --export-only --save-to output/moshi/

    # Use saved ONNX models
    python examples/moshi_realtime.py --onnx-dir output/moshi/

Notes:
    - Requires a HuggingFace account with accepted nvidia/personaplex-7b-v1
      license to download PersonaPlex weights.
    - The model runs at 12.5 Hz (80ms per frame) — one transformer step
      produces 1 text token + 16 audio codec tokens per step.
    - Audio sample rate: 24 kHz (Moshi's EnCodec codec).
    - This example uses a simplified single-stream mode for clarity.
      Production use would run listener and speaker in parallel threads.
"""

from __future__ import annotations

import argparse
import signal
import sys
import threading
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "nvidia/personaplex-7b-v1"

# Moshi audio parameters
SAMPLE_RATE = 24_000  # EnCodec sample rate (Hz)
FRAME_SAMPLES = 1920  # 80ms at 24kHz (one model step)
NUM_CODEBOOKS = 16  # RVQ codebook count
AUDIO_VOCAB_SIZE = 2048  # per-codebook vocabulary size
TEXT_VOCAB_SIZE = 32_000  # text token vocabulary

# Model step rate
STEPS_PER_SECOND = SAMPLE_RATE // FRAME_SAMPLES  # 12.5 Hz


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _try_import(name: str, pip_name: str | None = None) -> object:
    """Import optional dependency with a helpful error message."""
    import importlib

    try:
        return importlib.import_module(name)
    except ImportError:
        pkg = pip_name or name
        print(f"Missing dependency: {name}. Install with: pip install {pkg}", file=sys.stderr)
        sys.exit(1)


def _make_zero_kv_cache(
    num_layers: int,
    num_heads: int,
    head_dim: int,
    max_seq: int = 0,
    dtype: np.dtype = np.float32,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create an empty KV cache (zeros) for ``num_layers`` transformer layers."""
    shape = (1, num_heads, max_seq, head_dim)
    return [
        (np.zeros(shape, dtype=dtype), np.zeros(shape, dtype=dtype)) for _ in range(num_layers)
    ]


# ---------------------------------------------------------------------------
# ONNX model session helpers
# ---------------------------------------------------------------------------


class MoshiOnnxPipeline:
    """Wrapper around the three Moshi ONNX sub-models.

    Sub-models:
    - ``embedding``: (input_ids, audio_codes) → inputs_embeds
    - ``decoder``: inputs_embeds → logits + KV cache
    - ``audio_decoder``: backbone_hidden → codebook_logits (per step)

    Maintains KV cache state across inference steps for streaming.
    """

    def __init__(
        self,
        embedding_path: str,
        decoder_path: str,
        audio_decoder_path: str,
        *,
        hidden_size: int = 4096,
        num_decoder_layers: int = 32,
        num_decoder_heads: int = 32,
        head_dim: int = 128,
        depformer_dim: int = 1024,
        depformer_layers: int = 6,
        depformer_heads: int = 16,
        num_codebooks: int = NUM_CODEBOOKS,
        dtype: np.dtype = np.float32,
    ):
        ort = _try_import("onnxruntime")

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self._embedding = ort.InferenceSession(embedding_path, opts)
        self._decoder = ort.InferenceSession(decoder_path, opts)
        self._audio_decoder = ort.InferenceSession(audio_decoder_path, opts)

        self._hidden_size = hidden_size
        self._depformer_dim = depformer_dim
        self._num_codebooks = num_codebooks
        self._dtype = dtype

        # Decoder KV cache: num_heads * head_dim per layer
        self._decoder_kv = _make_zero_kv_cache(
            num_decoder_layers, num_decoder_heads, head_dim, dtype=dtype
        )
        # Depformer KV cache: depformer_heads x head_dim=depformer_dim per layer
        self._depformer_kv = _make_zero_kv_cache(
            depformer_layers, depformer_heads, depformer_dim, dtype=dtype
        )
        self._step = 0

    def reset(self) -> None:
        """Reset KV caches (start of new conversation)."""
        self._decoder_kv = [(np.zeros_like(k), np.zeros_like(v)) for k, v in self._decoder_kv]
        self._depformer_kv = [
            (np.zeros_like(k), np.zeros_like(v)) for k, v in self._depformer_kv
        ]
        self._step = 0

    def _build_decoder_feeds(
        self,
        inputs_embeds: np.ndarray,
        position_id: int,
    ) -> dict:
        """Build feed dict for the decoder with current KV cache."""
        feeds: dict = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": np.ones((1, position_id + 1), dtype=np.int64),
            "position_ids": np.array([[position_id]], dtype=np.int64),
        }
        for i, (k, v) in enumerate(self._decoder_kv):
            feeds[f"past_key_values.{i}.key"] = k
            feeds[f"past_key_values.{i}.value"] = v
        return feeds

    def _update_decoder_kv(self, outputs: list) -> None:
        """Extract present KV cache from decoder outputs."""
        # outputs[1:] are present key/value pairs interleaved
        kv_outputs = outputs[1:]
        for i in range(len(self._decoder_kv)):
            self._decoder_kv[i] = (kv_outputs[2 * i], kv_outputs[2 * i + 1])

    def _build_depformer_feeds(
        self,
        backbone_hidden: np.ndarray,
        prev_embedding: np.ndarray,
        codebook_idx: int,
    ) -> dict:
        """Build feed dict for the audio_decoder with current depformer KV cache."""
        feeds: dict = {
            "backbone_hidden": backbone_hidden,
            "prev_embedding": prev_embedding,
            "codebook_idx": np.array(codebook_idx, dtype=np.int64),
        }
        for i, (k, v) in enumerate(self._depformer_kv):
            feeds[f"past_key_values.{i}.key"] = k
            feeds[f"past_key_values.{i}.value"] = v
        return feeds

    def _update_depformer_kv(self, outputs: list) -> None:
        """Extract present KV cache from audio_decoder outputs."""
        kv_outputs = outputs[1:]
        for i in range(len(self._depformer_kv)):
            self._depformer_kv[i] = (kv_outputs[2 * i], kv_outputs[2 * i + 1])

    def step(
        self,
        text_token: int,
        audio_codes: np.ndarray,
    ) -> tuple[int, np.ndarray]:
        """Run one inference step.

        Args:
            text_token:  int — last generated text token (0 = pad/start)
            audio_codes: (num_codebooks,) int64 — incoming audio codec codes

        Returns:
            (next_text_token, output_audio_codes)
            - next_text_token: int — model's predicted next text token
            - output_audio_codes: (num_codebooks,) int64 — audio to synthesize
        """
        # 1. Embedding: (1, 1, hidden_size)
        emb_feeds = {
            "input_ids": np.array([[text_token]], dtype=np.int64),
            "audio_codes": audio_codes.reshape(1, 1, self._num_codebooks).astype(np.int64),
        }
        inputs_embeds = self._embedding.run(None, emb_feeds)[0]  # (1, 1, hidden_size)

        # 2. Decoder: logits + updated KV cache
        dec_feeds = self._build_decoder_feeds(inputs_embeds, self._step)
        dec_outputs = self._decoder.run(None, dec_feeds)
        logits = dec_outputs[0]  # (1, 1, vocab_size)
        self._update_decoder_kv(dec_outputs)

        # 3. Sample next text token (greedy)
        next_text_token = int(np.argmax(logits[0, 0]))

        # 4. Audio decoder: generate num_codebooks audio tokens autoregressively
        # backbone_hidden = decoder's last hidden state (approximated from logits for demo).
        # In production, expose the pre-projection hidden state from the decoder.
        backbone_hidden = inputs_embeds  # shape: (1, 1, hidden_size) — simplified

        output_codes = np.zeros(self._num_codebooks, dtype=np.int64)
        prev_embedding = np.zeros((1, 1, self._depformer_dim), dtype=self._dtype)

        # Reset depformer KV cache at the start of each backbone step
        depformer_kv_snapshot = [(k.copy(), v.copy()) for k, v in self._depformer_kv]
        self._depformer_kv = _make_zero_kv_cache(
            len(self._depformer_kv),
            self._depformer_kv[0][0].shape[1],
            self._depformer_kv[0][0].shape[3],
            dtype=self._dtype,
        )

        for codebook_idx in range(self._num_codebooks):
            dep_feeds = self._build_depformer_feeds(
                backbone_hidden, prev_embedding, codebook_idx
            )
            dep_outputs = self._audio_decoder.run(None, dep_feeds)
            codebook_logits = dep_outputs[0]  # (1, 1, audio_logits_size)
            self._update_depformer_kv(dep_outputs)

            # Greedy sample from codebook logits
            output_codes[codebook_idx] = int(np.argmax(codebook_logits[0, 0]))

            # Use sampled code as prev_embedding for next codebook (simplified: zero vec)
            prev_embedding = np.zeros((1, 1, self._depformer_dim), dtype=self._dtype)

        # Restore depformer snapshot (we don't accumulate cross-step depformer state)
        self._depformer_kv = depformer_kv_snapshot

        self._step += 1
        return next_text_token, output_codes


# ---------------------------------------------------------------------------
# Audio codec (EnCodec) placeholder
# ---------------------------------------------------------------------------


def encode_audio_frame(audio_frame: np.ndarray, codec) -> np.ndarray:
    """Encode a PCM audio frame to RVQ codec codes.

    Args:
        audio_frame: (frame_samples,) float32 PCM at SAMPLE_RATE Hz
        codec: EnCodec model (from moshi or encodec package)

    Returns:
        codes: (num_codebooks,) int64
    """
    import torch

    with torch.no_grad():
        # EnCodec expects (batch, channels, samples)
        wav = torch.from_numpy(audio_frame).float().unsqueeze(0).unsqueeze(0)
        encoded = codec.encode(wav)  # returns EncodedFrame list
        # Extract first frame, first batch: shape (num_codebooks, time)
        codes = encoded[0][0].squeeze(0).numpy()  # (num_codebooks, 1) or (num_codebooks,)
        if codes.ndim == 2:
            codes = codes[:, 0]
    return codes.astype(np.int64)


def decode_audio_codes(codes: np.ndarray, codec) -> np.ndarray:
    """Decode RVQ codec codes back to PCM waveform.

    Args:
        codes: (num_codebooks,) int64
        codec: EnCodec model

    Returns:
        audio_frame: (frame_samples,) float32 PCM
    """
    import torch

    with torch.no_grad():
        # codes: (num_codebooks,) → (1, num_codebooks, 1) for batch/time dims
        codes_t = torch.from_numpy(codes).long().unsqueeze(0).unsqueeze(-1)
        wav = codec.decode([(codes_t, None)])  # (1, 1, samples)
        return wav.squeeze().numpy()


# ---------------------------------------------------------------------------
# Real-time streaming loop
# ---------------------------------------------------------------------------


class MoshiStreamer:
    """Real-time duplex audio streamer for Moshi.

    Runs two parallel threads:
    - **Input thread**: records mic audio → encodes → queues input codes
    - **Inference + output thread**: consumes codes → runs model → plays audio
    """

    def __init__(
        self,
        pipeline: MoshiOnnxPipeline,
        codec,
        *,
        device: str | None = None,
    ):
        self._pipeline = pipeline
        self._codec = codec
        self._device = device

        self._input_queue: list[np.ndarray] = []
        self._output_queue: list[np.ndarray] = []
        self._lock = threading.Lock()
        self._running = False
        self._text_token = 0  # start with pad token

    def _record_callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        """Sounddevice input callback — encodes incoming mic audio."""
        if status:
            print(f"[audio] Input status: {status}", file=sys.stderr)

        audio_frame = indata[:, 0].astype(np.float32)  # take first channel
        try:
            codes = encode_audio_frame(audio_frame, self._codec)
            with self._lock:
                self._input_queue.append(codes)
        except Exception as e:
            print(f"[audio] Encode error: {e}", file=sys.stderr)

    def _play_callback(self, outdata: np.ndarray, frames: int, time_info, status) -> None:
        """Sounddevice output callback — decodes and plays generated audio."""
        if status:
            print(f"[audio] Output status: {status}", file=sys.stderr)

        with self._lock:
            if self._output_queue:
                audio_frame = self._output_queue.pop(0)
            else:
                audio_frame = np.zeros(frames, dtype=np.float32)

        outdata[:, 0] = audio_frame[:frames]

    def _inference_loop(self) -> None:
        """Main inference loop: consume input codes, run model, enqueue output."""
        print("[moshi] Inference loop started — speak into the microphone.")

        while self._running:
            with self._lock:
                if not self._input_queue:
                    codes_in = None
                else:
                    codes_in = self._input_queue.pop(0)

            if codes_in is None:
                # No input yet; feed silence (all-zero codes)
                codes_in = np.zeros(NUM_CODEBOOKS, dtype=np.int64)

            try:
                next_token, output_codes = self._pipeline.step(self._text_token, codes_in)
                self._text_token = next_token

                # Decode output codes to waveform
                audio_out = decode_audio_codes(output_codes, self._codec)

                with self._lock:
                    self._output_queue.append(audio_out)

            except Exception as e:
                print(f"[moshi] Inference error: {e}", file=sys.stderr)

            # Pace the inference loop to match model step rate (12.5 Hz)
            time.sleep(1.0 / STEPS_PER_SECOND)

    def run(self) -> None:
        """Start real-time streaming. Blocks until Ctrl+C."""
        sd = _try_import("sounddevice")

        self._running = True

        # Set up Ctrl+C handler for clean shutdown
        original_sigint = signal.getsignal(signal.SIGINT)

        def _stop(sig, frame):
            print("\n[moshi] Stopping...")
            self._running = False
            signal.signal(signal.SIGINT, original_sigint)

        signal.signal(signal.SIGINT, _stop)

        # Start inference thread
        inference_thread = threading.Thread(target=self._inference_loop, daemon=True)
        inference_thread.start()

        # Open audio streams
        input_stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=FRAME_SAMPLES,
            callback=self._record_callback,
            device=self._device,
        )
        output_stream = sd.OutputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=FRAME_SAMPLES,
            callback=self._play_callback,
            device=self._device,
        )

        print(f"[moshi] Streaming at {SAMPLE_RATE} Hz, {FRAME_SAMPLES} samples/frame")
        print("[moshi] Ctrl+C to stop")

        with input_stream, output_stream:
            while self._running:
                time.sleep(0.1)

        inference_thread.join(timeout=2.0)
        print("[moshi] Stopped.")


# ---------------------------------------------------------------------------
# Model building and export
# ---------------------------------------------------------------------------


def build_moshi_models(model_id: str, save_dir: Path | None = None) -> dict:
    """Build Moshi ONNX models via mobius and optionally save to disk.

    Returns a dict of ``{name: onnx_model}`` for embedding, decoder,
    and audio_decoder.
    """
    from mobius import build

    print(f"[moshi] Building ONNX models from {model_id} ...")
    pkg = build(model_id)

    if save_dir is not None:
        import onnx

        save_dir.mkdir(parents=True, exist_ok=True)
        for name, model in pkg.items():
            out_path = save_dir / f"{name}.onnx"
            onnx.save(model, str(out_path))
            print(f"[moshi]   Saved {name} → {out_path}")

    return pkg


def load_saved_models(onnx_dir: Path) -> dict[str, str]:
    """Return paths to saved ONNX models in onnx_dir."""
    paths = {}
    for name in ("embedding", "decoder", "audio_decoder"):
        path = onnx_dir / f"{name}.onnx"
        if not path.exists():
            print(f"[moshi] Missing model file: {path}", file=sys.stderr)
            sys.exit(1)
        paths[name] = str(path)
    return paths


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Moshi/PersonaPlex real-time speech conversation via ONNX"
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"HuggingFace model ID (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="Build and export ONNX models without running inference",
    )
    parser.add_argument(
        "--save-to",
        type=Path,
        default=None,
        metavar="DIR",
        help="Save exported ONNX models to this directory",
    )
    parser.add_argument(
        "--onnx-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="Load pre-exported ONNX models from this directory",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="sounddevice device name or index (default: system default)",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Step 1: Obtain ONNX model paths
    # ------------------------------------------------------------------
    if args.onnx_dir is not None:
        print(f"[moshi] Loading ONNX models from {args.onnx_dir}")
        model_paths = load_saved_models(args.onnx_dir)
    else:
        import tempfile

        onnx_models = build_moshi_models(args.model, save_dir=args.save_to)

        if args.export_only:
            if args.save_to is None:
                print("[moshi] --export-only requires --save-to <dir>")
                sys.exit(1)
            print("[moshi] Export complete.")
            return

        # Save to a temp dir so we can pass file paths to ORT
        _tmp = tempfile.mkdtemp(prefix="moshi_onnx_")
        tmp_dir = Path(_tmp)
        import onnx

        model_paths = {}
        for name, model in onnx_models.items():
            out = tmp_dir / f"{name}.onnx"
            onnx.save(model, str(out))
            model_paths[name] = str(out)
        print(f"[moshi] Temporary ONNX models saved to {tmp_dir}")

    # ------------------------------------------------------------------
    # Step 2: Load EnCodec / Moshi codec for audio encode/decode
    # ------------------------------------------------------------------
    try:
        import moshi.models  # type: ignore[import]

        print("[moshi] Loading EnCodec codec ...")
        codec = moshi.models.get_encodec(sample_rate=SAMPLE_RATE)
        codec.eval()
    except ImportError:
        print(
            "[moshi] 'moshi' package not found. Install with: pip install moshi\n"
            "        Falling back to silence-only demo (no audio encode/decode).",
            file=sys.stderr,
        )
        codec = None

    # ------------------------------------------------------------------
    # Step 3: Build inference pipeline
    # ------------------------------------------------------------------
    pipeline = MoshiOnnxPipeline(
        embedding_path=model_paths["embedding"],
        decoder_path=model_paths["decoder"],
        audio_decoder_path=model_paths["audio_decoder"],
    )

    if codec is None:
        # Dry-run demo: simulate one step without real audio
        print("[moshi] Running dry-run step (no codec, no audio device) ...")
        dummy_codes = np.zeros(NUM_CODEBOOKS, dtype=np.int64)
        text_token, out_codes = pipeline.step(0, dummy_codes)
        print(
            f"[moshi] Dry-run OK — text_token={text_token}, out_codes shape={out_codes.shape}"
        )
        return

    # ------------------------------------------------------------------
    # Step 4: Start real-time streaming
    # ------------------------------------------------------------------
    _try_import("sounddevice")  # ensure sounddevice is available before starting

    streamer = MoshiStreamer(pipeline, codec, device=args.device)
    streamer.run()


if __name__ == "__main__":
    main()
