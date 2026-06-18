#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""NVIDIA NeMo FastConformer-RNNT streaming speech recognition with ONNX.

Builds the four ONNX sub-models that ``mobius`` exports from a NeMo
``.nemo`` archive and runs RNN-T greedy decoding two different ways:

    audio --> log-mel --> encoder --> [joint <-> decoder] greedy loop --> text

* **Offline (from file):** the whole utterance is encoded in one shot with
  the ``encoder`` model, then greedily decoded into a full transcript.
* **Real-time (streaming):** the audio is split into chunks that are fed to
  the cache-aware ``encoder_streaming`` model one at a time, carrying the
  attention/conv caches forward, and partial text is printed as each chunk
  is decoded — emulating live transcription. The same loop drives a real
  microphone when ``--continuous`` is used.

The example is self-contained: the log-mel frontend (matching NeMo's
``AudioToMelSpectrogramPreprocessor``) and the SentencePiece BPE tokenizer
are reconstructed directly from the ``.nemo`` archive, so ``nemo_toolkit``
is *not* required at runtime.

Prerequisites::

    pip install mobius-ai onnxruntime onnxruntime-easy librosa soundfile sentencepiece
    pip install sounddevice          # only for microphone input

The example runs ONNX models through ``mobius._testing.ort_inference``,
which depends on ``onnxruntime`` and ``onnxruntime-easy`` (these are not
pulled in by the ``mobius-ai`` core package). ``torch`` is a core mobius
dependency and is installed automatically.

Tested on Linux/x86-64 (CPU and CUDA). On Apple Silicon / other platforms
the CPU path is expected to work (all dependencies ship arm64 wheels) but
has not been verified; ``--device cuda`` requires an NVIDIA GPU.

Usage::

    # Transcribe an audio file offline (full-context encoder)
    python examples/nemotron_fastconformer_rnnt.py --audio speech.wav

    # Simulate real-time streaming from a file (chunked encoder)
    python examples/nemotron_fastconformer_rnnt.py --audio speech.wav --stream

    # Live microphone transcription (Ctrl+C to stop)
    python examples/nemotron_fastconformer_rnnt.py --continuous

    # GPU inference
    python examples/nemotron_fastconformer_rnnt.py --audio speech.wav --device cuda

    # Just export the ONNX models, no inference
    python examples/nemotron_fastconformer_rnnt.py --save-to output/nemotron-rnnt/
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time

import numpy as np

from mobius._testing.ort_inference import OnnxModelSession
from mobius.integrations.nemo import build_from_nemo
from mobius.integrations.nemo._reader import NeMoArchive

# ---------------------------------------------------------------------------
# Model / decoding configuration
# ---------------------------------------------------------------------------

MODEL_ID = "nvidia/nemotron-speech-streaming-en-0.6b"
# Pin the HF revision so the exported graph matches the validated golden.
REVISION = "7a9b763e6c5fb103da690219c049fac917aa50b1"
SAMPLE_RATE = 16000

# RNN-T vocabulary bookkeeping (see tests/nemo_rnnt_integration_test.py):
#   * ``BLANK_ID`` is the reserved blank label that ends a frame's emissions.
#   * ``SOS_ID`` is the zero start-of-sequence embedding that primes the
#     prediction network (NeMo ``add_sos=True``).
BLANK_ID = 1024
SOS_ID = 1025
PRED_HIDDEN = 640  # prediction-network (LSTM) hidden size
PRED_LAYERS = 2  # prediction-network LSTM layers
MAX_SYMBOLS = 5  # max non-blank tokens emitted per encoder frame

# NeMo AudioToMelSpectrogramPreprocessor parameters (from model_config.yaml).
N_MELS = 128
N_FFT = 512
WIN_LENGTH = 400  # 0.025 s * 16 kHz
HOP_LENGTH = 160  # 0.010 s * 16 kHz
PREEMPH = 0.97
LOG_ZERO_GUARD = 2.0**-24
FEATURE_FRAME_SEC = HOP_LENGTH / SAMPLE_RATE  # 10 ms per feature frame


# ---------------------------------------------------------------------------
# Log-mel feature frontend (NeMo-compatible)
# ---------------------------------------------------------------------------


class MelFrontend:
    """Reconstructs NeMo's log-mel features without ``nemo_toolkit``.

    Mirrors ``AudioToMelSpectrogramPreprocessor`` with ``normalize="NA"``:
    pre-emphasis -> Hann-windowed STFT power spectrum -> Slaney mel filter
    bank -> natural log. Dithering is training-only and is skipped here.
    """

    def __init__(self) -> None:
        import librosa
        import torch

        self._torch = torch
        self._window = torch.hann_window(WIN_LENGTH, periodic=False)
        # (n_mels, n_fft//2 + 1) Slaney-normalized mel filter bank.
        fb = librosa.filters.mel(
            sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS, fmin=0.0, fmax=SAMPLE_RATE / 2
        )
        self._fb = torch.from_numpy(fb).float()

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        """audio: 1-D float waveform -> (1, N_MELS, T) log-mel features."""
        torch = self._torch
        sig = torch.from_numpy(np.ascontiguousarray(audio)).float()
        # Pre-emphasis: y[t] -= 0.97 * y[t-1] (first sample unchanged).
        sig = torch.cat([sig[:1], sig[1:] - PREEMPH * sig[:-1]])
        spec = torch.stft(
            sig,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            win_length=WIN_LENGTH,
            window=self._window,
            center=True,
            pad_mode="reflect",
            return_complex=True,
        )
        power = spec.real.pow(2) + spec.imag.pow(2)  # (n_fft//2+1, T)
        mel = torch.matmul(self._fb, power)  # (N_MELS, T)
        log_mel = torch.log(mel + LOG_ZERO_GUARD)
        return log_mel.unsqueeze(0).numpy().astype(np.float32)  # (1, N_MELS, T)


# ---------------------------------------------------------------------------
# RNN-T greedy decoder (shared by offline and streaming paths)
# ---------------------------------------------------------------------------


class RnntGreedyDecoder:
    """Stateful RNN-T greedy decoder over the decoder + joint ONNX models.

    Holds the prediction-network LSTM state and the last prediction output so
    that ``decode_frames`` can be called repeatedly across streaming chunks
    while preserving label-conditioning context.
    """

    def __init__(self, decoder: OnnxModelSession, joint: OnnxModelSession, np_dtype):
        self._decoder = decoder
        self._joint = joint
        self._np_dtype = np_dtype
        self.reset()

    def reset(self) -> None:
        """Reset LSTM state and prime with the start-of-sequence embedding.

        The prediction network is advanced once on the SOS label; both its
        output and the resulting LSTM state are kept so the first real token
        is conditioned on the post-SOS state (matching NeMo ``add_sos=True``).
        """
        self._h = np.zeros((PRED_LAYERS, 1, PRED_HIDDEN), dtype=self._np_dtype)
        self._c = np.zeros((PRED_LAYERS, 1, PRED_HIDDEN), dtype=self._np_dtype)
        self._g, self._h, self._c = self._predict(SOS_ID)

    def _predict(self, token: int):
        out = self._decoder.run(
            {
                "targets": np.array([[token]], dtype=np.int64),
                "state_h": self._h,
                "state_c": self._c,
            }
        )
        return out["decoder_output"], out["state_h_out"], out["state_c_out"]

    def decode_frames(self, encoder_output: np.ndarray) -> list[int]:
        """Greedily decode a time-major ``(1, T, d)`` encoder chunk into token ids.

        The encoder frames and the prediction-network output are fed to the
        joiner in the ONNX Runtime GenAI time-major single-frame layout
        (``(B, 1, d)`` / ``(B, 1, d_pred)``).
        """
        tokens: list[int] = []
        n_frames = encoder_output.shape[1]
        for t in range(n_frames):
            enc_t = encoder_output[:, t : t + 1, :]  # (B, 1, d)
            emitted = 0
            while emitted < MAX_SYMBOLS:
                logits = self._joint.run(
                    {
                        "encoder_outputs": enc_t,
                        "decoder_outputs": np.ascontiguousarray(self._g.transpose(0, 2, 1)),
                    }
                )["logits"]
                k = int(np.argmax(logits.reshape(-1)))
                if k == BLANK_ID:
                    break
                tokens.append(k)
                emitted += 1
                # Advance the prediction network on the emitted label.
                self._g, self._h, self._c = self._predict(k)
        return tokens


# ---------------------------------------------------------------------------
# Pipeline: ONNX models + tokenizer + frontend
# ---------------------------------------------------------------------------


class FastConformerRnntPipeline:
    def __init__(
        self,
        model_id: str = MODEL_ID,
        revision: str = REVISION,
        device: str = "cpu",
        dtype: str | None = None,
    ) -> None:
        print(f"Building ONNX models from {model_id} ...", file=sys.stderr)
        pkg = build_from_nemo(model_id, revision=revision, dtype=dtype)
        self._package = pkg
        self._encoder = OnnxModelSession(pkg["encoder"], device=device)
        self._encoder_streaming = OnnxModelSession(pkg["encoder_streaming"], device=device)
        self._decoder = OnnxModelSession(pkg["decoder"], device=device)
        self._joint = OnnxModelSession(pkg["joint"], device=device)

        # Audio-signal compute dtype (f32 / f16 / bf16) declared by the graph.
        self._signal_dtype = self._encoder.get_input_dtype("audio_signal") or np.dtype(
            np.float32
        )
        # State/cache tensors share the model compute dtype.
        self._compute_dtype = self._signal_dtype

        self._frontend = MelFrontend()
        # Hold the tokenizer temp dir on the instance so it lives as long as
        # the pipeline (and is cleaned up when the pipeline is collected).
        self._tok_tmpdir = tempfile.TemporaryDirectory(prefix="nemo_tok_")
        self._tokenizer = _load_tokenizer(model_id, revision, self._tok_tmpdir.name)
        # drop_extra_pre_encoded (subsampled frames) governs the left-context
        # overlap a streaming caller must prepend to each chunk.
        self._drop_extra = int(getattr(pkg.config, "fastconformer_streaming_drop_extra", 2))
        self._subsampling = int(getattr(pkg.config, "fastconformer_subsampling_factor", 8))

    # -- helpers ----------------------------------------------------------

    def save(self, dest: str) -> None:
        self._package.save(dest)
        print(f"Saved ONNX models to {dest}", file=sys.stderr)

    def _features(self, audio: np.ndarray) -> np.ndarray:
        return self._frontend(audio).astype(self._signal_dtype)

    def _decode_text(self, tokens: list[int]) -> str:
        return self._tokenizer.decode(tokens) if tokens else ""

    def _init_caches(self):
        """Zero initial streaming caches, shapes from the graph's inputs."""
        sess = self._encoder_streaming

        def concrete(name: str) -> list[int]:
            shape = sess.get_input_shape(name) or []
            return [1 if not isinstance(d, int) else d for d in shape]

        ch = np.zeros(concrete("cache_last_channel"), dtype=self._compute_dtype)
        ct = np.zeros(concrete("cache_last_time"), dtype=self._compute_dtype)
        cl = np.zeros((1,), dtype=np.int64)
        return ch, ct, cl

    # -- offline (file) mode ----------------------------------------------

    def transcribe_offline(self, audio: np.ndarray) -> str:
        """Full-context transcription of a complete utterance."""
        feats = self._features(audio)  # (1, N_MELS, T_total) feature-major
        length = np.array([feats.shape[2]], dtype=np.int64)
        # The encoder graph is natively time-major (B, T, mel) -> (B, T, d).
        audio_tm = np.ascontiguousarray(feats.transpose(0, 2, 1))
        enc = self._encoder.run({"audio_signal": audio_tm, "length": length})
        decoder = RnntGreedyDecoder(self._decoder, self._joint, self._compute_dtype)
        tokens = decoder.decode_frames(enc["encoder_output"])
        return self._decode_text(tokens)

    # -- streaming (real-time) mode ---------------------------------------

    def stream(self, audio: np.ndarray, chunk_seconds: float = 1.12):
        """Yield ``(partial_tokens, partial_text)`` per chunk in real time.

        The audio is split into fixed feature-frame chunks fed to the
        cache-aware ``encoder_streaming`` model. Each chunk overlaps the
        previous one by ``drop_extra_pre_encoded`` subsampled frames worth of
        left context (the graph drops that many output frames to splice with
        the carried caches), so emitted frames stay contiguous across chunks.

        The chunk length is snapped to a multiple of the encoder's subsampling
        factor (8) so the subsampling stem aligns cleanly with the streaming
        caches — matching the model's native chunked attention (the largest
        trained chunk is ``att_context_size`` ``[70, 13]`` -> 14 subsampled
        frames -> 1.12 s). Word boundaries that straddle a chunk are an
        inherent property of fixed-chunk streaming.
        """
        feats = self._features(audio)  # (1, N_MELS, T_total)
        total = feats.shape[2]
        raw_frames = max(1, round(chunk_seconds / FEATURE_FRAME_SEC))
        # Snap to a multiple of the subsampling factor for clean alignment.
        chunk_frames = max(
            self._subsampling, round(raw_frames / self._subsampling) * self._subsampling
        )
        overlap = self._drop_extra * self._subsampling  # feature frames

        ch, ct, cl = self._init_caches()
        decoder = RnntGreedyDecoder(self._decoder, self._joint, self._compute_dtype)

        start = 0
        while start < total:
            end = min(start + chunk_frames, total)
            # Prepend left-context overlap (zero-padded for the first chunk).
            lo = max(0, start - overlap)
            chunk = feats[:, :, lo:end]
            length = np.array([chunk.shape[2]], dtype=np.int64)
            # GenAI streaming encoder consumes time-major audio (B, T, mel).
            audio = np.ascontiguousarray(chunk.transpose(0, 2, 1))
            out = self._encoder_streaming.run(
                {
                    "audio_signal": audio,
                    "length": length,
                    "cache_last_channel": ch,
                    "cache_last_time": ct,
                    "cache_last_channel_len": cl,
                }
            )
            ch = out["cache_last_channel_next"]
            ct = out["cache_last_time_next"]
            cl = out["cache_last_channel_len_next"]
            tokens = decoder.decode_frames(out["encoder_output"])
            yield tokens, self._decode_text(tokens)
            start = end


# ---------------------------------------------------------------------------
# Tokenizer / audio IO helpers
# ---------------------------------------------------------------------------


def _load_tokenizer(model_id: str, revision: str, dest_dir: str):
    import sentencepiece as spm

    archive = NeMoArchive(model_id, revision=revision)
    written = archive.extract_tokenizer(dest_dir)
    model_path = written.get("model_path")
    if model_path is None:
        raise RuntimeError("No SentencePiece tokenizer found in the .nemo archive")
    sp = spm.SentencePieceProcessor()
    sp.Load(model_path)
    return sp


def _load_audio_file(path: str) -> np.ndarray:
    """Load an audio file as a mono 16 kHz float32 waveform."""
    import librosa

    audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    return audio.astype(np.float32)


# ---------------------------------------------------------------------------
# Run modes
# ---------------------------------------------------------------------------


def _run_offline(pipeline: FastConformerRnntPipeline, audio: np.ndarray) -> str:
    print("Transcribing (offline)...", file=sys.stderr)
    text = pipeline.transcribe_offline(audio)
    print("\nTranscript:\n" + text + "\n")
    return text


def _run_stream_from_file(
    pipeline: FastConformerRnntPipeline,
    audio: np.ndarray,
    chunk_seconds: float,
    realtime: bool,
) -> str:
    print("Transcribing (streaming, real-time simulation)...\n", file=sys.stderr)
    full_tokens: list[int] = []
    for tokens, _partial in pipeline.stream(audio, chunk_seconds=chunk_seconds):
        full_tokens.extend(tokens)
        # Print the running transcript, updated as chunks arrive.
        running = pipeline._decode_text(full_tokens)
        sys.stdout.write("\r" + running)
        sys.stdout.flush()
        if realtime:
            # Pace the loop to roughly the chunk's wall-clock duration.
            time.sleep(chunk_seconds)
    print()
    return pipeline._decode_text(full_tokens)


def _run_microphone(
    pipeline: FastConformerRnntPipeline, chunk_seconds: float
) -> None:  # pragma: no cover - requires hardware
    try:
        import sounddevice as sd
    except ImportError:
        print(
            "Microphone input requires 'sounddevice' (pip install sounddevice).",
            file=sys.stderr,
        )
        sys.exit(1)

    chunk_samples = int(chunk_seconds * SAMPLE_RATE)
    overlap = pipeline._drop_extra * pipeline._subsampling
    overlap_samples = overlap * HOP_LENGTH

    ch, ct, cl = pipeline._init_caches()
    decoder = RnntGreedyDecoder(pipeline._decoder, pipeline._joint, pipeline._compute_dtype)
    # Start with no left context (like the first file-streaming chunk); the
    # overlap is filled from the previous block on subsequent iterations.
    prev_tail = np.zeros(0, dtype=np.float32)
    full_tokens: list[int] = []

    print("Listening... (Ctrl+C to stop)\n", file=sys.stderr)
    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32") as stream:
            while True:
                block, _ = stream.read(chunk_samples)
                samples = block[:, 0]
                audio = np.concatenate([prev_tail, samples])
                # Carry the last ``overlap_samples`` of the running audio as the
                # next chunk's left context (robust to short blocks).
                prev_tail = audio[-overlap_samples:] if overlap_samples else prev_tail
                feats = pipeline._features(audio)
                length = np.array([feats.shape[2]], dtype=np.int64)
                # GenAI streaming encoder consumes time-major audio (B, T, mel).
                signal = np.ascontiguousarray(feats.transpose(0, 2, 1))
                out = pipeline._encoder_streaming.run(
                    {
                        "audio_signal": signal,
                        "length": length,
                        "cache_last_channel": ch,
                        "cache_last_time": ct,
                        "cache_last_channel_len": cl,
                    }
                )
                ch = out["cache_last_channel_next"]
                ct = out["cache_last_time_next"]
                cl = out["cache_last_channel_len_next"]
                tokens = decoder.decode_frames(out["encoder_output"])
                full_tokens.extend(tokens)
                sys.stdout.write("\r" + pipeline._decode_text(full_tokens))
                sys.stdout.flush()
    except KeyboardInterrupt:
        print("\n\nFinal transcript:\n" + pipeline._decode_text(full_tokens) + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", default=MODEL_ID, help="HuggingFace .nemo model id")
    parser.add_argument("--revision", default=REVISION, help="Pinned HF revision")
    parser.add_argument("--audio", help="Path to an audio file to transcribe")
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream the file chunk-by-chunk (real-time simulation)",
    )
    parser.add_argument(
        "--continuous",
        action="store_true",
        help="Transcribe live from the microphone (Ctrl+C to stop)",
    )
    parser.add_argument(
        "--chunk-seconds",
        type=float,
        default=1.12,
        help="Streaming chunk length in seconds (default: 1.12, the model's "
        "largest native chunk; snapped to an 8-frame multiple)",
    )
    parser.add_argument(
        "--no-realtime-pace",
        action="store_true",
        help="Do not sleep between chunks when streaming from a file",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda", "webgpu"],
        help=(
            "ORT device. 'cuda' needs an NVIDIA GPU. 'webgpu' is EXPERIMENTAL "
            "and unverified for this model: the encoder uses opset-24 Swish + "
            "Attention and the decoder uses LSTM, which may lack WebGPU kernels "
            "(falling back to CPU or erroring). Requires an onnxruntime build "
            "with the WebGPU EP."
        ),
    )
    parser.add_argument(
        "--dtype",
        default=None,
        choices=["f32", "f16", "bf16"],
        help="Model compute dtype (default: model default / f32)",
    )
    parser.add_argument("--save-to", help="Export ONNX models to this dir and exit")
    args = parser.parse_args(argv)

    if args.chunk_seconds <= 0:
        parser.error("--chunk-seconds must be positive")

    # Export-only fast path: build + save without loading audio/tokenizer.
    if args.save_to and not (args.audio or args.continuous):
        pkg = build_from_nemo(args.model, revision=args.revision, dtype=args.dtype)
        pkg.save(args.save_to)
        print(f"Saved ONNX models to {args.save_to}", file=sys.stderr)
        return

    pipeline = FastConformerRnntPipeline(
        model_id=args.model, revision=args.revision, device=args.device, dtype=args.dtype
    )
    if args.save_to:
        pipeline.save(args.save_to)

    if args.continuous:
        _run_microphone(pipeline, args.chunk_seconds)
        return

    if not args.audio:
        parser.error("provide --audio FILE, or --continuous for microphone input")

    audio = _load_audio_file(args.audio)
    if args.stream:
        _run_stream_from_file(
            pipeline, audio, args.chunk_seconds, realtime=not args.no_realtime_pace
        )
    else:
        _run_offline(pipeline, audio)


if __name__ == "__main__":
    main()
