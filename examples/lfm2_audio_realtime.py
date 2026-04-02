#!/usr/bin/env python
# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""LFM2-Audio real-time speech-to-speech conversation.

LFM2-Audio is a speech-to-speech model with a hybrid ShortConv+Attention
backbone (LFM2) that processes audio and generates spoken responses.

Architecture (4-model ONNX split):

    mic audio → mel spectrogram → [audio_encoder] → audio_features
    audio_features + text_ids → [embedding] → inputs_embeds
    inputs_embeds → [decoder] → logits + hybrid cache
    decoder_hidden → [audio_decoder] → codebook_logits → codec decode → audio

The decoder uses a **hybrid cache**: ShortConv layers carry a conv_state,
attention layers carry standard KV pairs (key + value).

Prerequisites::

    pip install mobius-ai[transformers] sounddevice numpy onnxruntime

Usage::

    # Interactive conversation with LFM2-Audio-1.5B
    python examples/lfm2_audio_realtime.py

    # Export ONNX models only (no inference)
    python examples/lfm2_audio_realtime.py --export-only --save-to output/lfm2/

    # Use saved ONNX models
    python examples/lfm2_audio_realtime.py --onnx-dir output/lfm2/

    # Dry run (no audio device required)
    python examples/lfm2_audio_realtime.py --dry-run

Notes:
    - Model: LiquidAI/LFM2-Audio-1.5B (gated — requires accepting the license)
    - The model runs at 12.5 Hz (80ms per frame), matching EnCodec's frame rate.
    - Audio sample rate: 24 kHz (EnCodec codec).
    - Hybrid cache: ShortConv layers use a conv_state; attention layers use KV pairs.
    - The depthformer generates 8 codebooks per backbone step autoregressively.
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
# LFM2-Audio configuration constants
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "LiquidAI/LFM2-Audio-1.5B"

# Audio parameters (EnCodec codec, 8 codebooks)
SAMPLE_RATE = 24_000  # Hz
FRAME_SAMPLES = 1920  # 80ms at 24kHz (one model step = one EnCodec frame)
NUM_CODEBOOKS = 8  # RVQ codebook depth
AUDIO_VOCAB_SIZE = 2048  # per-codebook vocabulary entries (excl. padding)
N_MELS = 128  # mel filter bank bins for Conformer audio encoder

# Model step rate
STEPS_PER_SECOND = SAMPLE_RATE // FRAME_SAMPLES  # 12.5 Hz

# LFM2-Audio-1.5B architecture defaults (used for cache initialization)
# These match the released LiquidAI/LFM2-Audio-1.5B checkpoint.
_DEFAULT_HIDDEN_SIZE = 2048
_DEFAULT_NUM_LAYERS = 24
_DEFAULT_NUM_KV_HEADS = 8
_DEFAULT_HEAD_DIM = 128
_DEFAULT_SHORT_CONV_KERNEL = 3  # conv_state width = kernel - 1 = 2

# Depthformer (audio decoder) defaults
_DEFAULT_DEPTHFORMER_DIM = 1024
_DEFAULT_DEPTHFORMER_LAYERS = 6
_DEFAULT_DEPTHFORMER_HEADS = 16
_DEFAULT_DEPTHFORMER_HEAD_DIM = _DEFAULT_DEPTHFORMER_DIM // _DEFAULT_DEPTHFORMER_HEADS  # 64


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


def _audio_to_mel(audio_frame: np.ndarray, n_mels: int = N_MELS) -> np.ndarray:
    """Convert a PCM audio frame to a log-mel spectrogram.

    Args:
        audio_frame: (frame_samples,) float32 PCM at SAMPLE_RATE Hz
        n_mels: number of mel filter bins

    Returns:
        mel: (1, n_mels, time_frames) float32 log-mel spectrogram
    """
    # Short-time Fourier transform (simple rectangular window)
    hop_length = 160  # 10ms at 16kHz, ~6.7ms at 24kHz
    fft_size = 512
    frames = []
    for start in range(0, len(audio_frame) - fft_size, hop_length):
        window = audio_frame[start : start + fft_size]
        spectrum = np.abs(np.fft.rfft(window * np.hanning(fft_size))) ** 2
        frames.append(spectrum)

    if not frames:
        # If the frame is too short, return a single zero frame
        frames = [np.zeros(fft_size // 2 + 1)]

    stft = np.array(frames).T  # (fft_bins, time_frames)

    # Mel filterbank (triangular filters in mel space)
    freq_bins = stft.shape[0]
    mel_low, mel_high = 0.0, 2595.0 * np.log10(1 + (SAMPLE_RATE / 2) / 700)
    mel_points = np.linspace(mel_low, mel_high, n_mels + 2)
    hz_points = 700 * (10 ** (mel_points / 2595) - 1)
    bin_points = np.floor((fft_size + 1) * hz_points / SAMPLE_RATE).astype(int)
    bin_points = np.clip(bin_points, 0, freq_bins - 1)

    mel_fb = np.zeros((n_mels, freq_bins))
    for m in range(1, n_mels + 1):
        lo, ctr, hi = bin_points[m - 1], bin_points[m], bin_points[m + 1]
        for k in range(lo, ctr):
            mel_fb[m - 1, k] = (k - lo) / max(ctr - lo, 1)
        for k in range(ctr, hi):
            mel_fb[m - 1, k] = (hi - k) / max(hi - ctr, 1)

    mel_spec = mel_fb @ stft  # (n_mels, time_frames)
    log_mel = np.log(mel_spec + 1e-9)  # log-mel
    return log_mel[np.newaxis, :, :].astype(np.float32)  # (1, n_mels, T)


# ---------------------------------------------------------------------------
# Hybrid cache helpers
# ---------------------------------------------------------------------------


def _init_hybrid_cache(session, batch: int = 1) -> dict[str, np.ndarray]:
    """Initialize hybrid cache state from ONNX session input metadata.

    Inspects the decoder session's input names to discover which layers are
    conv (conv_state) vs attention (key/value pairs), then allocates zero
    tensors of the correct shape.

    For attention layers the initial past_seq_len is 0.  Conv states have a
    fixed width of ``short_conv_kernel - 1`` which is inferred from the shape
    metadata if available, otherwise defaults to 2 (kernel=3).

    Returns:
        A dict mapping input name → zero numpy array.
    """
    cache: dict[str, np.ndarray] = {}

    for inp in session.get_inputs():
        name = inp.name
        if "past_key_values" not in name:
            continue

        shape = inp.shape  # may contain symbolic strings for dynamic dims
        np_shape = []
        for dim in shape:
            if isinstance(dim, int) and dim >= 0:
                np_shape.append(dim)
            elif isinstance(dim, str) and dim == "batch":
                np_shape.append(batch)
            else:
                # Symbolic sequence dimension → start with 0 (empty)
                np_shape.append(0)

        # conv_state has no variable sequence dimension — replace any 0-dims
        # with a concrete default (conv_kernel - 1 for the last dim).
        if name.endswith(".conv_state"):
            np_shape = [dim if dim != 0 else _DEFAULT_HIDDEN_SIZE for dim in np_shape]

        cache[name] = np.zeros(np_shape, dtype=np.float32)

    return cache


def _update_hybrid_cache(
    cache: dict[str, np.ndarray],
    outputs: list[np.ndarray],
    session,
) -> None:
    """Update hybrid cache in-place from decoder output tensors.

    The decoder outputs present_* states; we map them back to the
    corresponding past_key_values.* inputs for the next step.

    Mapping: ``present.{i}.X`` → ``past_key_values.{i}.X``
    """
    output_names = [out.name for out in session.get_outputs()]
    for out_name, out_val in zip(output_names[1:], outputs[1:]):
        # Convert present.N.X → past_key_values.N.X
        if out_name.startswith(("present.", "present_key_values.")):
            # Strip the "present" prefix and replace with "past_key_values"
            suffix = out_name.split(".", 1)[1]  # e.g. "4.conv_state" or "4.key"
            past_name = f"past_key_values.{suffix}"
            if past_name in cache:
                cache[past_name] = out_val


def _make_zero_kv_cache(
    num_layers: int,
    num_heads: int,
    head_dim: int,
    dtype: np.dtype = np.float32,
) -> dict[str, np.ndarray]:
    """Create a zero KV cache dict for ``num_layers`` attention layers.

    Returns:
        Dict with keys ``past_key_values.{i}.key`` / ``past_key_values.{i}.value``.
    """
    cache: dict[str, np.ndarray] = {}
    for i in range(num_layers):
        shape = (1, num_heads, 0, head_dim)
        cache[f"past_key_values.{i}.key"] = np.zeros(shape, dtype=dtype)
        cache[f"past_key_values.{i}.value"] = np.zeros(shape, dtype=dtype)
    return cache


# ---------------------------------------------------------------------------
# ONNX inference pipeline
# ---------------------------------------------------------------------------


class Lfm2AudioOnnxPipeline:
    """Wrapper around the four LFM2-Audio ONNX sub-models.

    Sub-models:
    - ``audio_encoder``: mel (1, n_mels, T) → audio_features (1, T', hidden)
    - ``embedding``: input_ids + audio_features → inputs_embeds (1, S, hidden)
    - ``decoder``: inputs_embeds → logits + hybrid KV/conv cache
    - ``audio_decoder``: backbone_hidden → codebook_logits (one codebook at a time)

    Maintains hybrid cache state (ShortConv conv_state + attention KV pairs)
    across inference steps for streaming.
    """

    def __init__(
        self,
        audio_encoder_path: str,
        embedding_path: str,
        decoder_path: str,
        audio_decoder_path: str,
        *,
        depthformer_dim: int = _DEFAULT_DEPTHFORMER_DIM,
        depthformer_layers: int = _DEFAULT_DEPTHFORMER_LAYERS,
        depthformer_heads: int = _DEFAULT_DEPTHFORMER_HEADS,
        num_codebooks: int = NUM_CODEBOOKS,
        dtype: np.dtype = np.float32,
    ):
        ort = _try_import("onnxruntime")

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self._audio_encoder = ort.InferenceSession(audio_encoder_path, opts)
        self._embedding = ort.InferenceSession(embedding_path, opts)
        self._decoder = ort.InferenceSession(decoder_path, opts)
        self._audio_decoder = ort.InferenceSession(audio_decoder_path, opts)

        self._depthformer_dim = depthformer_dim
        self._num_codebooks = num_codebooks
        self._dtype = dtype

        # Hybrid decoder cache: initialized by inspecting decoder session inputs
        self._decoder_cache = _init_hybrid_cache(self._decoder, batch=1)

        # Depthformer KV cache (pure attention, no conv layers)
        depthformer_head_dim = depthformer_dim // depthformer_heads
        self._depthformer_cache = _make_zero_kv_cache(
            depthformer_layers, depthformer_heads, depthformer_head_dim, dtype=dtype
        )

        self._step = 0

    def reset(self) -> None:
        """Reset all cache state (start of new conversation)."""
        self._decoder_cache = {k: np.zeros_like(v) for k, v in self._decoder_cache.items()}
        self._depthformer_cache = {
            k: np.zeros_like(v) for k, v in self._depthformer_cache.items()
        }
        self._step = 0

    def _build_depthformer_feeds(
        self,
        backbone_hidden: np.ndarray,
        prev_embedding: np.ndarray,
        codebook_idx: int,
    ) -> dict:
        feeds: dict = {
            "backbone_hidden": backbone_hidden,
            "prev_embedding": prev_embedding,
            "codebook_idx": np.array(codebook_idx, dtype=np.int64),
        }
        feeds.update(self._depthformer_cache)
        return feeds

    def _update_depthformer_cache(self, outputs: list) -> None:
        """Extract present KV from audio_decoder outputs."""
        output_names = [out.name for out in self._audio_decoder.get_outputs()]
        for out_name, out_val in zip(output_names[1:], outputs[1:]):
            suffix = out_name.split(".", 1)[1]
            past_name = f"past_key_values.{suffix}"
            if past_name in self._depthformer_cache:
                self._depthformer_cache[past_name] = out_val

    def step(
        self,
        text_token: int,
        audio_frame: np.ndarray,
    ) -> tuple[int, np.ndarray]:
        """Run one inference step.

        Args:
            text_token:  last generated text token (0 = pad/start)
            audio_frame: (frame_samples,) float32 PCM at SAMPLE_RATE Hz

        Returns:
            (next_text_token, output_audio_codes)
            - next_text_token: model's predicted next text token
            - output_audio_codes: (num_codebooks,) int64 — audio to synthesize
        """
        # 1. Audio encoder: mel (1, n_mels, T) → audio_features (1, T', hidden)
        mel = _audio_to_mel(audio_frame, n_mels=N_MELS)
        audio_features = self._audio_encoder.run(None, {"input_features": mel})[0]

        # 2. Embedding: text + audio → inputs_embeds (1, S, hidden)
        emb_feeds = {
            "input_ids": np.array([[text_token]], dtype=np.int64),
            "audio_features": audio_features,  # (1, T', hidden)
        }
        inputs_embeds = self._embedding.run(None, emb_feeds)[0]  # (1, S, hidden)

        # 3. Decoder: inputs_embeds → logits + updated hybrid cache
        dec_feeds: dict = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": np.ones(
                (1, self._step + inputs_embeds.shape[1]), dtype=np.int64
            ),
            "position_ids": np.arange(
                self._step, self._step + inputs_embeds.shape[1], dtype=np.int64
            ).reshape(1, -1),
        }
        dec_feeds.update(self._decoder_cache)
        dec_outputs = self._decoder.run(None, dec_feeds)
        logits = dec_outputs[0]  # (1, S, vocab_size)
        _update_hybrid_cache(self._decoder_cache, dec_outputs, self._decoder)

        # 4. Sample next text token (greedy from last position)
        next_text_token = int(np.argmax(logits[0, -1]))

        # 5. Audio decoder: generate num_codebooks audio tokens autoregressively.
        # Use the last inputs_embeds position as a proxy for backbone_hidden.
        # (In production, expose the pre-lm-head hidden state from the decoder.)
        backbone_hidden = inputs_embeds[:, -1:, :]  # (1, 1, hidden)
        backbone_hidden = backbone_hidden.astype(self._dtype)

        output_codes = np.zeros(self._num_codebooks, dtype=np.int64)
        prev_embedding = np.zeros((1, 1, self._depthformer_dim), dtype=self._dtype)

        # Reset depthformer KV: the depthformer recurs over codebooks within
        # a single backbone step, not across backbone steps.
        self._depthformer_cache = {
            k: np.zeros_like(v) for k, v in self._depthformer_cache.items()
        }

        for codebook_idx in range(self._num_codebooks):
            dep_feeds = self._build_depthformer_feeds(
                backbone_hidden, prev_embedding, codebook_idx
            )
            dep_outputs = self._audio_decoder.run(None, dep_feeds)
            codebook_logits = dep_outputs[0]  # (1, 1, audio_vocab_size)
            self._update_depthformer_cache(dep_outputs)

            output_codes[codebook_idx] = int(np.argmax(codebook_logits[0, 0]))
            # prev_embedding carries the sampled code embedding into the next codebook
            # (simplified: use zeros — production would look up the depth embedding)
            prev_embedding = np.zeros((1, 1, self._depthformer_dim), dtype=self._dtype)

        self._step += inputs_embeds.shape[1]
        return next_text_token, output_codes


# ---------------------------------------------------------------------------
# Audio codec helpers
# ---------------------------------------------------------------------------


def encode_audio_frame(audio_frame: np.ndarray, codec) -> np.ndarray:
    """Encode a PCM audio frame to RVQ codec codes via EnCodec.

    Args:
        audio_frame: (frame_samples,) float32 PCM at SAMPLE_RATE Hz
        codec: EnCodec model (from the ``encodec`` or ``liquid_audio`` package)

    Returns:
        codes: (num_codebooks,) int64
    """
    import torch

    with torch.no_grad():
        wav = torch.from_numpy(audio_frame).float().unsqueeze(0).unsqueeze(0)
        encoded = codec.encode(wav)
        # EnCodec returns a list of EncodedFrame; take first, first batch
        codes = encoded[0][0].squeeze(0).numpy()  # (num_codebooks, time)
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
        codes_t = torch.from_numpy(codes).long().unsqueeze(0).unsqueeze(-1)
        wav = codec.decode([(codes_t, None)])  # (1, 1, samples)
        return wav.squeeze().numpy()


# ---------------------------------------------------------------------------
# Real-time streaming loop
# ---------------------------------------------------------------------------


class Lfm2Streamer:
    """Real-time duplex audio streamer for LFM2-Audio.

    Runs two parallel threads:
    - **Input thread**: records mic audio → encodes via EnCodec + mel → queues frames
    - **Inference + output thread**: consumes frames → runs LFM2 → plays audio

    The ``codec`` argument may be ``None`` for a no-audio dry run.
    """

    def __init__(
        self,
        pipeline: Lfm2AudioOnnxPipeline,
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
        self._text_token = 0  # start with pad/bos token

    def _record_callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        """Sounddevice input callback — queues incoming mic frames."""
        if status:
            print(f"[audio] Input status: {status}", file=sys.stderr)
        audio_frame = indata[:, 0].astype(np.float32)
        with self._lock:
            self._input_queue.append(audio_frame)

    def _play_callback(self, outdata: np.ndarray, frames: int, time_info, status) -> None:
        """Sounddevice output callback — plays generated audio frames."""
        if status:
            print(f"[audio] Output status: {status}", file=sys.stderr)
        with self._lock:
            if self._output_queue:
                audio_frame = self._output_queue.pop(0)
            else:
                audio_frame = np.zeros(frames, dtype=np.float32)
        outdata[:, 0] = audio_frame[:frames]

    def _inference_loop(self) -> None:
        """Main inference loop: consume mic frames, run LFM2, enqueue output audio."""
        print("[lfm2] Inference loop started — speak into the microphone.")

        while self._running:
            with self._lock:
                audio_in = self._input_queue.pop(0) if self._input_queue else None

            if audio_in is None:
                # No input yet; feed a silence frame
                audio_in = np.zeros(FRAME_SAMPLES, dtype=np.float32)

            try:
                next_token, output_codes = self._pipeline.step(self._text_token, audio_in)
                self._text_token = next_token

                if self._codec is not None:
                    audio_out = decode_audio_codes(output_codes, self._codec)
                else:
                    audio_out = np.zeros(FRAME_SAMPLES, dtype=np.float32)

                with self._lock:
                    self._output_queue.append(audio_out)

            except Exception as e:
                print(f"[lfm2] Inference error: {e}", file=sys.stderr)

            # Pace to match the model's 12.5 Hz step rate
            time.sleep(1.0 / STEPS_PER_SECOND)

    def run(self) -> None:
        """Start real-time streaming. Blocks until Ctrl+C."""
        sd = _try_import("sounddevice")

        self._running = True

        original_sigint = signal.getsignal(signal.SIGINT)

        def _stop(sig, frame):
            print("\n[lfm2] Stopping...")
            self._running = False
            signal.signal(signal.SIGINT, original_sigint)

        signal.signal(signal.SIGINT, _stop)

        inference_thread = threading.Thread(target=self._inference_loop, daemon=True)
        inference_thread.start()

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

        print(f"[lfm2] Streaming at {SAMPLE_RATE} Hz, {FRAME_SAMPLES} samples/frame")
        print("[lfm2] Ctrl+C to stop")

        with input_stream, output_stream:
            while self._running:
                time.sleep(0.1)

        inference_thread.join(timeout=2.0)
        print("[lfm2] Stopped.")


# ---------------------------------------------------------------------------
# Model building and export
# ---------------------------------------------------------------------------


def build_lfm2_models(model_id: str, save_dir: Path | None = None) -> dict:
    """Build LFM2-Audio ONNX models via mobius and optionally save to disk.

    Returns a ``ModelPackage`` dict with keys:
    ``audio_encoder``, ``embedding``, ``decoder``, ``audio_decoder``.
    """
    from mobius import build

    print(f"[lfm2] Building ONNX models from {model_id} ...")
    pkg = build(model_id)

    if save_dir is not None:
        import onnx

        save_dir.mkdir(parents=True, exist_ok=True)
        for name, model in pkg.items():
            out_path = save_dir / f"{name}.onnx"
            onnx.save(model, str(out_path))
            print(f"[lfm2]   Saved {name} → {out_path}")

    return pkg


def load_saved_models(onnx_dir: Path) -> dict[str, str]:
    """Return paths to saved ONNX models in onnx_dir."""
    paths = {}
    for name in ("audio_encoder", "embedding", "decoder", "audio_decoder"):
        path = onnx_dir / f"{name}.onnx"
        if not path.exists():
            print(f"[lfm2] Missing model file: {path}", file=sys.stderr)
            sys.exit(1)
        paths[name] = str(path)
    return paths


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LFM2-Audio real-time speech-to-speech conversation via ONNX"
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
        "--dry-run",
        action="store_true",
        help="Run one inference step without audio I/O (requires no sounddevice)",
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
        print(f"[lfm2] Loading ONNX models from {args.onnx_dir}")
        model_paths = load_saved_models(args.onnx_dir)
    else:
        import tempfile

        onnx_models = build_lfm2_models(args.model, save_dir=args.save_to)

        if args.export_only:
            if args.save_to is None:
                print("[lfm2] --export-only requires --save-to <dir>")
                sys.exit(1)
            print("[lfm2] Export complete.")
            return

        # Save to a temp dir so ORT can load from file paths
        _tmp = tempfile.mkdtemp(prefix="lfm2_onnx_")
        tmp_dir = Path(_tmp)
        import onnx

        model_paths = {}
        for name, model in onnx_models.items():
            out = tmp_dir / f"{name}.onnx"
            onnx.save(model, str(out))
            model_paths[name] = str(out)
        print(f"[lfm2] Temporary ONNX models saved to {tmp_dir}")

    # ------------------------------------------------------------------
    # Step 2: Optionally load EnCodec codec for audio encode/decode
    # ------------------------------------------------------------------
    codec = None
    if not args.dry_run:
        try:
            from encodec.model import EncodecModel  # type: ignore[import]

            print("[lfm2] Loading EnCodec codec ...")
            codec = EncodecModel.encodec_model_24khz()
            codec.set_target_bandwidth(6.0)  # 8 codebooks x 6kbps
            codec.eval()
            print("[lfm2] EnCodec loaded.")
        except ImportError:
            print(
                "[lfm2] 'encodec' package not found. Install with: pip install encodec\n"
                "       Falling back to silent dry-run mode.",
                file=sys.stderr,
            )

    # ------------------------------------------------------------------
    # Step 3: Build inference pipeline
    # ------------------------------------------------------------------
    pipeline = Lfm2AudioOnnxPipeline(
        audio_encoder_path=model_paths["audio_encoder"],
        embedding_path=model_paths["embedding"],
        decoder_path=model_paths["decoder"],
        audio_decoder_path=model_paths["audio_decoder"],
    )

    if args.dry_run or codec is None:
        # Dry-run: simulate one step with a silence frame
        print("[lfm2] Running dry-run step (no audio device) ...")
        silence = np.zeros(FRAME_SAMPLES, dtype=np.float32)
        text_token, out_codes = pipeline.step(0, silence)
        print(
            f"[lfm2] Dry-run OK — text_token={text_token}, out_codes shape={out_codes.shape}"
        )
        return

    # ------------------------------------------------------------------
    # Step 4: Start real-time streaming
    # ------------------------------------------------------------------
    _try_import("sounddevice")  # confirm sounddevice is available

    streamer = Lfm2Streamer(pipeline, codec, device=args.device)
    streamer.run()


if __name__ == "__main__":
    main()
