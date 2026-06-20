#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""NVIDIA / Kyutai Moshi full-duplex speech-to-speech with ONNX Runtime.

Builds the four ONNX sub-models that ``mobius`` exports for
``nvidia/personaplex-7b-v1`` and runs the full-duplex generation loop end to
end with ``onnxruntime``::

    user audio --> Mimi encoder --> [Moshi temporal + depformer] --> Mimi
                                     decoder --> Moshi (assistant) audio

The four models (all built from the native Kyutai ``safetensors`` checkpoints
via :func:`mobius.integrations.moshi.build_mimi` /
:func:`~mobius.integrations.moshi.build_moshi_lm`):

* **Mimi encoder** ``waveform (B,1,T) -> codes (B,8,Tf)``
* **Moshi temporal** ``frame (B,17,S) -> hidden + text_logits + KV``
* **Moshi depformer** ``hidden + prev_token + substep_index + KV -> logits``
  (stepped 16 times per frame, once per audio codebook)
* **Mimi decoder** ``codes (B,8,Tf) -> waveform (B,1,T)``

The generation loop is a faithful NumPy port of Kyutai ``LMGen.step``: a ring
cache with per-codebook delays, the temporal step, greedy text sampling, the
16-substep autoregressive depformer, and delayed output collection. The user
audio stream is fed as the 8 "other" codebooks (``k = 9..16``); the assistant
(Moshi) audio is generated as codebooks ``k = 1..8`` and decoded by Mimi.

Prerequisites::

    pip install mobius-ai onnxruntime numpy soundfile

CUDA note: on H200 / Ampere+ GPUs ORT defaults to TF32 for fp32 matmul, which
can flip greedy sampling. ``--device cuda`` sets ``use_tf32=0`` for fp32
parity; pass ``--allow-tf32`` to keep the (faster, lower-precision) default.

Usage::

    # Smoke test: generate a few frames from silence, save assistant audio
    python examples/personaplex_moshi.py --frames 25 --save-to out/personaplex

    # Use a real input wav as the user stream
    python examples/personaplex_moshi.py --audio user.wav --save-to out/personaplex

    # Reuse already-exported ONNX models (skip the build step)
    python examples/personaplex_moshi.py --model-dir out/personaplex/onnx
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np

_MODEL_ID = "nvidia/personaplex-7b-v1"

# --- Architecture constants (personaplex-7b-v1). ---
SAMPLE_RATE = 24000
FRAME_RATE = 12.5
FRAME_SIZE = int(SAMPLE_RATE / FRAME_RATE)  # 1920 samples per 12.5 Hz frame
NUM_CODEBOOKS = 17  # 1 text + 16 audio
DEP_Q = 16  # audio codebooks predicted by the depformer
MIMI_CB = 8  # codebooks Mimi actually decodes (assistant voice)
AUDIO_OFFSET = 1
TEXT_CARD = 32000
AUDIO_CARD = 2048
ZERO_TEXT_CODE = 3
INITIAL_AUDIO_TOKEN = AUDIO_CARD  # card
INITIAL_TEXT_TOKEN = TEXT_CARD  # text_card
UNGENERATED = -2
DELAYS = [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1]
MAX_DELAY = max(DELAYS)
AUDIO_TOKENS_PER_STREAM = 8
# Mimi tokens that decode to a near-silent frame (from the Kyutai reference).
SILENCE_TOKENS = np.array([948, 243, 1178, 546, 1736, 1030, 1978, 2008], np.int64)

# Temporal / depformer head geometry.
T_LAYERS, T_HEADS, T_HEAD_DIM = 32, 32, 128
D_LAYERS, D_HEADS, D_HEAD_DIM = 6, 16, 64


def _provider(device: str, allow_tf32: bool):
    import onnxruntime as ort

    if device == "cuda":
        opts = {} if allow_tf32 else {"use_tf32": 0}
        if "CUDAExecutionProvider" in ort.get_available_providers():
            return [("CUDAExecutionProvider", opts), "CPUExecutionProvider"]
        print("[warn] CUDAExecutionProvider unavailable; falling back to CPU.")
    return ["CPUExecutionProvider"]


def _greedy(logits: np.ndarray) -> int:
    return int(np.argmax(logits))


class MoshiORT:
    """Full-duplex Moshi generation driven by four ONNX Runtime sessions."""

    def __init__(self, model_dir: str, device: str, allow_tf32: bool):
        import onnxruntime as ort

        providers = _provider(device, allow_tf32)

        def _load(name: str):
            path = os.path.join(model_dir, name, "model.onnx")
            if not os.path.isfile(path):
                path = os.path.join(model_dir, f"{name}.onnx")
            return ort.InferenceSession(path, providers=providers)

        self.enc = _load("mimi_encoder")
        self.dec = _load("mimi_decoder")
        self.temporal = _load("temporal")
        self.depformer = _load("depformer")
        self._reset_lm_state()

    # --- Mimi codec ------------------------------------------------------
    def encode(self, waveform: np.ndarray) -> np.ndarray:
        """Waveform (1,1,T) float32 -> codes (1, 8, Tf) int64."""
        return self.enc.run(["codes"], {"waveform": waveform})[0]

    def decode(self, codes: np.ndarray) -> np.ndarray:
        """Codes (1, 8, Tf) int64 -> waveform (1,1,T) float32."""
        return self.dec.run(["waveform"], {"codes": codes.astype(np.int64)})[0]

    # --- LM state (ring cache + delays), mirrors LMGen ------------------
    def _reset_lm_state(self):
        ct = MAX_DELAY + 3
        self.ct = ct
        self.offset = 0
        self.cache = np.full((1, NUM_CODEBOOKS, ct), UNGENERATED, np.int64)
        self.provided = np.zeros((1, NUM_CODEBOOKS, ct), bool)
        # initial token per codebook: text_card for k=0, card for k>=1.
        self.initial = np.empty((1, NUM_CODEBOOKS, 1), np.int64)
        self.initial[0, 0, 0] = INITIAL_TEXT_TOKEN
        self.initial[0, 1:, 0] = INITIAL_AUDIO_TOKEN
        # persistent temporal KV cache (grows one frame per step)
        self._tkv = [
            (
                np.zeros((1, T_HEADS, 0, T_HEAD_DIM), np.float32),
                np.zeros((1, T_HEADS, 0, T_HEAD_DIM), np.float32),
            )
            for _ in range(T_LAYERS)
        ]
        self._tpos = 0

    # --- Temporal transformer step --------------------------------------
    def _temporal_step(self, frame: np.ndarray):
        """Frame (1,17,1) int64 -> (hidden (1,1,4096), text_logits (1,32000))."""
        s = frame.shape[2]
        feeds = {
            "input_frame": frame,
            "attention_mask": np.ones((1, self._tpos + s), np.int64),
            "position_ids": np.arange(self._tpos, self._tpos + s, dtype=np.int64)[None],
        }
        for i in range(T_LAYERS):
            feeds[f"past_key_values.{i}.key"] = self._tkv[i][0]
            feeds[f"past_key_values.{i}.value"] = self._tkv[i][1]
        names = ["hidden", "text_logits"] + [
            f"present.{i}.{kv}" for i in range(T_LAYERS) for kv in ("key", "value")
        ]
        outs = self.temporal.run(names, feeds)
        hidden, text_logits = outs[0], outs[1]
        present = outs[2:]
        self._tkv = [(present[2 * i], present[2 * i + 1]) for i in range(T_LAYERS)]
        self._tpos += s
        return hidden[:, -1:], text_logits[0, -1]

    # --- Depformer: 16 autoregressive substeps --------------------------
    def _depformer_step(self, text_token, hidden, audio_target, audio_provided):
        """Generate the 16 audio codebooks for one frame.

        ``audio_target`` / ``audio_provided``: (16,) teacher-forcing of the
        16 audio codebooks (used for the user stream at k=9..16 and any forced
        Moshi tokens). Returns sampled (16,) int64.
        """
        past = [
            (
                np.zeros((1, D_HEADS, 0, D_HEAD_DIM), np.float32),
                np.zeros((1, D_HEADS, 0, D_HEAD_DIM), np.float32),
            )
            for _ in range(D_LAYERS)
        ]
        names = ["logits"] + [
            f"present.{i}.{kv}" for i in range(D_LAYERS) for kv in ("key", "value")
        ]
        prev = int(text_token)
        sampled = np.empty(DEP_Q, np.int64)
        for cb in range(DEP_Q):
            feeds = {
                "hidden": hidden,
                "prev_token": np.array([[prev]], np.int64),
                "substep_index": np.array(cb, np.int64),
            }
            for i in range(D_LAYERS):
                feeds[f"past_key_values.{i}.key"] = past[i][0]
                feeds[f"past_key_values.{i}.value"] = past[i][1]
            outs = self.depformer.run(names, feeds)
            logits = outs[0][0, 0]
            present = outs[1:]
            past = [(present[2 * i], present[2 * i + 1]) for i in range(D_LAYERS)]
            tok = _greedy(logits)
            # Teacher-force where provided (e.g. the echoed user stream).
            if audio_provided[cb]:
                prev = int(audio_target[cb])
            else:
                prev = tok
            sampled[cb] = tok
        return sampled

    # --- One full-duplex step (port of LMGen.step) ----------------------
    def step(self, user_codes_frame, text_token=None):
        """Advance one 12.5 Hz frame.

        ``user_codes_frame``: (8,) int64 user-stream Mimi codes, or None for
        silence. Returns the assistant audio codes (8,) for this frame once
        the pipeline has filled (``None`` during the initial warm-up frames).
        """
        ct = self.ct

        # Fill cache with provided user tokens at (offset + delay) % CT.
        if user_codes_frame is not None:
            for q in range(AUDIO_TOKENS_PER_STREAM):
                k = AUDIO_TOKENS_PER_STREAM + 1 + q  # k = 9..16
                wp = (self.offset + DELAYS[k]) % ct
                self.cache[0, k, wp] = user_codes_frame[q]
                self.provided[0, k, wp] = True
        # Force the text stream (zero_text_code) like the reference S2S loop.
        if text_token is None:
            text_token = ZERO_TEXT_CODE
        wp = (self.offset + DELAYS[0]) % ct
        self.cache[0, 0, wp] = text_token
        self.provided[0, 0, wp] = True

        # Seed delayed codebooks with the initial token at the very start.
        for k, delay in enumerate(DELAYS):
            if self.offset <= delay:
                self.cache[0, k, self.offset % ct] = self.initial[0, k, 0]
                self.provided[0, k, self.offset % ct] = True

        if self.offset == 0:
            self.cache[0, :, 0] = self.initial[0, :, 0]
            self.offset += 1
            return None

        mip = (self.offset - 1) % ct
        tp = self.offset % ct
        input_ = self.cache[:, :, mip : mip + 1]  # (1,17,1)
        target_ = self.cache[:, :, tp]  # (1,17)
        provided_ = self.provided[:, :, tp]  # (1,17)

        hidden, text_logits = self._temporal_step(input_)
        sampled_text = _greedy(text_logits)
        next_text = target_[0, 0] if provided_[0, 0] else sampled_text

        sampled_audio = self._depformer_step(
            next_text,
            hidden,
            target_[0, AUDIO_OFFSET:],
            provided_[0, AUDIO_OFFSET:],
        )

        # Write generated tokens into the cache where not provided.
        self.provided[0, :, mip] = False
        if not self.provided[0, 0, tp]:
            self.cache[0, 0, tp] = sampled_text
        for k in range(1, DEP_Q + 1):
            if not self.provided[0, k, tp]:
                self.cache[0, k, tp] = sampled_audio[k - 1]

        if self.offset <= MAX_DELAY:
            self.offset += 1
            return None

        # Collect delayed outputs: cache[k, (offset - max_delay + delay) % CT].
        out = np.empty(DEP_Q + 1, np.int64)
        for k in range(DEP_Q + 1):
            idx = (self.offset - MAX_DELAY + DELAYS[k]) % ct
            out[k] = self.cache[0, k, idx]
        self.offset += 1
        # Assistant audio = first 8 generated audio codebooks (k = 1..8).
        return out[1 : 1 + MIMI_CB]


def _build_models(model_dir: str, device: str):
    """Export the four ONNX models from the native checkpoints (once)."""
    from mobius.integrations.moshi import build_mimi, build_moshi_lm

    os.makedirs(model_dir, exist_ok=True)
    ep = "cuda" if device == "cuda" else "default"
    print(f"[build] Mimi codec from {_MODEL_ID} ...")
    mimi = build_mimi(_MODEL_ID, execution_provider=ep)
    mimi.save(os.path.join(model_dir, "mimi"))
    # Mimi saves encoder/ and decoder/ subdirs; flatten the names we load.
    for role in ("encoder", "decoder"):
        src = os.path.join(model_dir, "mimi", role)
        dst = os.path.join(model_dir, f"mimi_{role}")
        if os.path.isdir(src) and not os.path.isdir(dst):
            os.rename(src, dst)

    print(f"[build] Moshi LM (temporal + depformer) from {_MODEL_ID} ...")
    lm = build_moshi_lm(_MODEL_ID, execution_provider=ep)
    lm["temporal"].save(os.path.join(model_dir, "temporal"))
    lm["depformer"].save(os.path.join(model_dir, "depformer"))
    print(f"[build] saved ONNX models under {model_dir}")


def _load_user_codes(moshi: MoshiORT, audio_path: str | None, frames: int):
    """Return a list of (8,) user-stream code frames (silence if no audio)."""
    if audio_path is None:
        silence = SILENCE_TOKENS
        return [silence.copy() for _ in range(frames)]
    import soundfile as sf

    wav, sr = sf.read(audio_path, dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != SAMPLE_RATE:
        raise SystemExit(
            f"Input audio must be {SAMPLE_RATE} Hz mono (got {sr} Hz). "
            "Resample it first (e.g. with librosa/ffmpeg)."
        )
    n = (len(wav) // FRAME_SIZE) * FRAME_SIZE
    wav = wav[:n].reshape(1, 1, n).astype(np.float32)
    codes = moshi.encode(wav)[0]  # (8, Tf)
    return [codes[:, t] for t in range(codes.shape[1])]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default="output/personaplex/onnx")
    parser.add_argument("--audio", default=None, help="24kHz mono user-stream wav")
    parser.add_argument("--frames", type=int, default=25, help="frames if no --audio")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--allow-tf32", action="store_true")
    parser.add_argument("--save-to", default=None, help="dir to write assistant.wav")
    parser.add_argument("--skip-build", action="store_true", help="reuse existing --model-dir")
    args = parser.parse_args()

    if not args.skip_build and not os.path.isdir(os.path.join(args.model_dir, "temporal")):
        _build_models(args.model_dir, args.device)

    moshi = MoshiORT(args.model_dir, args.device, args.allow_tf32)
    user_frames = _load_user_codes(moshi, args.audio, args.frames)
    print(f"[run] {len(user_frames)} input frames on {args.device}")

    assistant_codes: list[np.ndarray] = []
    t0 = time.time()
    for uf in user_frames:
        out = moshi.step(uf)
        if out is not None:
            assistant_codes.append(out)
    dt = time.time() - t0
    n = len(user_frames)
    print(
        f"[run] generated {len(assistant_codes)} assistant frames in {dt:.2f}s "
        f"({n / dt:.1f} frames/s, {dt / max(n, 1) * 1000:.1f} ms/frame)"
    )

    if assistant_codes:
        codes = np.stack(assistant_codes, axis=1)[None]  # (1, 8, Tf)
        wav = moshi.decode(codes)[0, 0]  # (T,)
        dur = len(wav) / SAMPLE_RATE
        print(f"[run] decoded assistant audio: {len(wav)} samples ({dur:.2f}s)")
        if args.save_to:
            import soundfile as sf

            os.makedirs(args.save_to, exist_ok=True)
            out_path = os.path.join(args.save_to, "assistant.wav")
            sf.write(out_path, wav, SAMPLE_RATE)
            print(f"[run] wrote {out_path}")
    else:
        print("[run] no assistant frames produced (increase --frames).")


if __name__ == "__main__":
    main()
