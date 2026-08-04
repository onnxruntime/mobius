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

    pip install mobius-onnx onnxruntime numpy soundfile

CUDA note: on H200 / Ampere+ GPUs ORT defaults to TF32 for fp32 matmul, which
can flip greedy sampling. ``--device cuda`` sets ``use_tf32=0`` for fp32
parity; pass ``--allow-tf32`` to keep the (faster, lower-precision) default.

Usage::

    # Smoke test: generate a few frames from silence, save assistant audio
    python examples/personaplex/moshi_ort.py --frames 25 --save-to out/personaplex

    # Use a real input wav as the user stream
    python examples/personaplex/moshi_ort.py --audio user.wav --save-to out/personaplex

    # Reuse already-exported ONNX models (skip the build step)
    python examples/personaplex/moshi_ort.py --model-dir out/personaplex/onnx

    # Simulated real-time stream (reports RTF / per-frame budget). Build the
    # models with an fp16 LM on CUDA first for real-time speed:
    python examples/personaplex/moshi_ort.py --device cuda --lm-dtype f16 \
        --stream --audio user.wav --save-to out/personaplex

    # Live full-duplex mic -> speaker (needs sounddevice + audio hardware)
    python examples/personaplex/moshi_ort.py --skip-build --device cuda --mic

Real-time note: each 12.5 Hz frame must finish within 80 ms. On an fp16 LM +
CUDA the Moshi LM is ~27 ms/frame (~3x headroom); CPU fp32 (~1.8 s/frame) is
far too slow for ``--stream``/``--mic``. The Mimi codec stays fp32 (its fp16
export currently hits a Conv dtype mismatch).
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
# Mimi tokens for the reference sine fed on the user stream during prompting
# (PersonaPlex feeds a sine on the user channel while priming voice/persona).
SINE_TOKENS = np.array([430, 1268, 381, 1611, 1095, 1495, 56, 472], np.int64)

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


def _sample_token(logits: np.ndarray, temp: float, top_k: int, rng) -> int:
    """Temperature + top-k sampling, mirroring Kyutai ``sample_token``.

    ``temp <= 0`` falls back to greedy (argmax). Top-k is applied on the logits
    (equivalent to top-k on the softmax probs) and the survivors are
    renormalised before sampling.
    """
    if temp <= 0.0:
        return int(np.argmax(logits))
    logits = logits.astype(np.float64)
    if 0 < top_k < logits.shape[-1]:
        keep = np.argpartition(logits, -top_k)[-top_k:]
    else:
        keep = np.arange(logits.shape[-1])
    sub = logits[keep]
    sub = sub - sub.max()
    probs = np.exp(sub / temp)
    probs /= probs.sum()
    return int(keep[rng.choice(len(keep), p=probs)])


# --- Persona text tokenizer (SentencePiece) ----------------------------------
_MODEL_ID = "nvidia/personaplex-7b-v1"
TEXT_TOKENIZER_NAME = "tokenizer_spm_32k_3.model"
DEFAULT_PERSONA = (
    "You are a wise and friendly teacher. Answer questions or provide advice "
    "in a clear and engaging way."
)


def load_persona_tokenizer(path: str | None = None):
    """Load the PersonaPlex SentencePiece tokenizer for the persona text stream.

    ``path`` points at ``tokenizer_spm_32k_3.model``; when None it is downloaded
    from the ``nvidia/personaplex-7b-v1`` HuggingFace repo (cached).
    """
    import sentencepiece

    if path is None:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(_MODEL_ID, TEXT_TOKENIZER_NAME)
    return sentencepiece.SentencePieceProcessor(model_file=path)


def encode_persona(tokenizer, persona_text: str) -> list[int]:
    """Wrap the persona in ``<system> ... <system>`` tags and tokenize it.

    Mirrors PersonaPlex ``wrap_with_system_tags`` (both ends use ``<system>``).
    """
    text = persona_text.strip()
    if not (text.startswith("<system>") and text.endswith("<system>")):
        text = f"<system> {text} <system>"
    return list(tokenizer.encode(text))


class MoshiORT:
    """Full-duplex Moshi generation driven by four ONNX Runtime sessions."""

    def __init__(
        self,
        model_dir: str,
        device: str,
        allow_tf32: bool,
        *,
        temp_text: float = 0.7,
        top_k_text: int = 25,
        temp_audio: float = 0.8,
        top_k_audio: int = 250,
        seed: int | None = None,
    ):
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

        # Sampling controls (Kyutai defaults). Greedy text/audio makes the model
        # stay silent (it keeps emitting the text pad token), so the real-time
        # loop samples with temperature like the reference LMGen.
        self.temp_text = temp_text
        self.top_k_text = top_k_text
        self.temp_audio = temp_audio
        self.top_k_audio = top_k_audio
        self._rng = np.random.default_rng(seed)
        self._ort = ort

        # The graph optimizer may prune unused inputs (e.g. position_ids when
        # RoPE derives its offset from the KV-cache length), and the KV cache
        # dtype follows the exported model dtype (fp32 / fp16). Read both from
        # the session so this loop works for any build.
        self._t_inputs = {i.name for i in self.temporal.get_inputs()}
        _ort_to_np = {"tensor(float)": np.float32, "tensor(float16)": np.float16}
        self._kv_dtype = _ort_to_np.get(
            next(i.type for i in self.temporal.get_inputs() if i.name.endswith(".key")),
            np.float32,
        )
        # Device that the temporal KV cache lives on for IO binding.
        self._kv_device = "cuda" if device == "cuda" else "cpu"
        self._reset_lm_state()

    # --- Mimi codec ------------------------------------------------------
    def encode(self, waveform: np.ndarray) -> np.ndarray:
        """Waveform (1,1,T) float32 -> codes (1, 8, Tf) int64."""
        return self.enc.run(["codes"], {"waveform": waveform})[0]

    def decode(self, codes: np.ndarray) -> np.ndarray:
        """Codes (1, 8, Tf) int64 -> waveform (1,1,T) float32."""
        return self.dec.run(["waveform"], {"codes": codes.astype(np.int64)})[0]

    def warmup(self, frames: int = 3) -> None:
        """Run a few full frames to trigger CUDA/codec autotune, then reset.

        Avoids a multi-100ms stall (audio glitch) on the first real frame.
        """
        enc = StreamingMimiEncoder(self.enc)
        dec = StreamingMimiDecoder(self.dec)
        silence = np.zeros(FRAME_SIZE, np.float32)
        for _ in range(frames):
            out = self.step(enc.push(silence))
            if out is not None:
                dec.push(out)
        self._reset_lm_state()

    # --- Streaming front-door (shared sessions, per-conversation state) --
    def reset_stream(self) -> None:
        """Reset conversation state for a fresh stream (e.g. new client)."""
        self._senc = StreamingMimiEncoder(self.enc)
        self._sdec = StreamingMimiDecoder(self.dec)
        self._reset_lm_state()

    def set_seed(self, seed: int | None) -> None:
        """Reseed the sampling RNG so a conversation's voice is reproducible.

        The assistant's voice/accent is determined entirely by the temperature
        + top-k sampling trajectory (the ONNX models are deterministic), so a
        fixed ``seed`` yields the same voice every session; ``None`` reseeds
        from entropy for a fresh random voice.
        """
        self._rng = np.random.default_rng(seed)

    def process_frame(self, frame: np.ndarray) -> np.ndarray | None:
        """Process one 12.5Hz user frame, return the assistant frame.

        ``frame`` is (FRAME_SIZE,) float32; returns (FRAME_SIZE,) float32, or
        None during the initial warm-up frames.
        """
        if not hasattr(self, "_senc"):
            self.reset_stream()
        codes = self._senc.push(frame.astype(np.float32, copy=False))
        out = self.step(codes)
        return None if out is None else self._sdec.push(out)

    # --- System-prompt priming (voice + persona) -----------------------
    def prime(
        self,
        voice_pcm: np.ndarray | None = None,
        text_tokens: list[int] | None = None,
        silence_frames: int = 6,
    ) -> None:
        """Prime the model with a voice prompt and a persona text prompt.

        Mirrors PersonaPlex ``LMGen.step_system_prompts``: before the
        conversation, the assistant audio stream is force-fed a reference voice
        (so the model continues in that voice) and the text stream is force-fed
        the persona tokens (role / scenario). A sine is fed on the user stream
        throughout, with ~0.5 s silence spacers between phases.

        Call this right after :meth:`reset_stream`. ``voice_pcm`` is mono float32
        at 24 kHz (any length; encoded with Mimi). ``text_tokens`` is the
        SentencePiece-encoded persona wrapped in ``<system> ... <system>``.
        ``silence_frames`` defaults to 0.5 s at 12.5 Hz.
        """
        # Phase 1: voice prompt -> force assistant audio = reference voice codes.
        if voice_pcm is not None and voice_pcm.size >= FRAME_SIZE:
            n = (len(voice_pcm) // FRAME_SIZE) * FRAME_SIZE
            wav = voice_pcm[:n].reshape(1, 1, n).astype(np.float32)
            codes = self.encode(wav)[0]  # (8, Tf)
            for t in range(codes.shape[1]):
                self.step(SINE_TOKENS, text_token=ZERO_TEXT_CODE, moshi_tokens=codes[:, t])
            # Phase 2: silence spacer after the voice prompt.
            for _ in range(silence_frames):
                self.step(SINE_TOKENS, text_token=ZERO_TEXT_CODE, moshi_tokens=SILENCE_TOKENS)

        # Phase 3: text prompt -> force the text stream with the persona tokens.
        if text_tokens:
            for tok in text_tokens:
                self.step(SINE_TOKENS, text_token=int(tok), moshi_tokens=SILENCE_TOKENS)
            # Phase 4: silence spacer after the text prompt.
            for _ in range(silence_frames):
                self.step(SINE_TOKENS, text_token=ZERO_TEXT_CODE, moshi_tokens=SILENCE_TOKENS)

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
        # Persistent temporal KV cache, kept resident on the inference device
        # via ORT IO binding so it is never copied through host memory between
        # frames. With a plain numpy feed, ORT copies the whole (growing) cache
        # host->device every frame, an O(N) cost that dominates per-frame time
        # as the conversation lengthens (60ms -> 200ms+). Keeping it on-device
        # makes per-frame cost flat (~6ms temporal regardless of length). The
        # present.* outputs are bound to device and reused as next frame's
        # past.* inputs (double-buffering). ``_tkv_ov`` holds the (key, value)
        # OrtValue pair per layer; length-0 at the start of a conversation.
        self._tkv_ov = [
            (
                self._ort.OrtValue.ortvalue_from_numpy(
                    np.zeros((1, T_HEADS, 0, T_HEAD_DIM), self._kv_dtype), self._kv_device, 0
                ),
                self._ort.OrtValue.ortvalue_from_numpy(
                    np.zeros((1, T_HEADS, 0, T_HEAD_DIM), self._kv_dtype), self._kv_device, 0
                ),
            )
            for _ in range(T_LAYERS)
        ]
        self._tpos = 0

    # --- Temporal transformer step --------------------------------------
    def _temporal_step(self, frame: np.ndarray):
        """Frame (1,17,1) int64 -> (hidden (1,1,4096), text_logits (1,32000)).

        Runs with ORT IO binding so the KV cache stays resident on the
        inference device across frames: present.* outputs are bound to device
        and fed back as next frame's past.* inputs. This removes the per-frame
        host<->device copy of the (growing) cache, keeping per-frame temporal
        cost flat (~6ms) regardless of conversation length.
        """
        s = frame.shape[2]
        kv_len = self._tkv_ov[0][0].shape()[2]  # current on-device cache length
        io = self.temporal.io_binding()
        io.bind_cpu_input("input_frame", frame)
        io.bind_cpu_input("attention_mask", np.ones((1, kv_len + s), np.int64))
        if "position_ids" in self._t_inputs:
            io.bind_cpu_input(
                "position_ids",
                np.arange(self._tpos, self._tpos + s, dtype=np.int64)[None],
            )
        for i in range(T_LAYERS):
            io.bind_ortvalue_input(f"past_key_values.{i}.key", self._tkv_ov[i][0])
            io.bind_ortvalue_input(f"past_key_values.{i}.value", self._tkv_ov[i][1])
        # hidden / text_logits come back to host (needed for depformer + sampling);
        # present.* stay on device to become next frame's past.*.
        io.bind_output("hidden", "cpu")
        io.bind_output("text_logits", "cpu")
        for i in range(T_LAYERS):
            io.bind_output(f"present.{i}.key", self._kv_device, 0)
            io.bind_output(f"present.{i}.value", self._kv_device, 0)
        self.temporal.run_with_iobinding(io)
        outs = io.get_outputs()
        hidden = outs[0].numpy()
        text_logits = outs[1].numpy()
        self._tkv_ov = [(outs[2 + 2 * i], outs[2 + 2 * i + 1]) for i in range(T_LAYERS)]
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
                np.zeros((1, D_HEADS, 0, D_HEAD_DIM), self._kv_dtype),
                np.zeros((1, D_HEADS, 0, D_HEAD_DIM), self._kv_dtype),
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
            tok = _sample_token(logits, self.temp_audio, self.top_k_audio, self._rng)
            # Teacher-force where provided (e.g. the echoed user stream).
            if audio_provided[cb]:
                prev = int(audio_target[cb])
            else:
                prev = tok
            sampled[cb] = tok
        return sampled

    # --- One full-duplex step (port of LMGen.step) ----------------------
    def step(self, user_codes_frame, text_token=None, moshi_tokens=None):
        """Advance one 12.5 Hz frame.

        ``user_codes_frame``: (8,) int64 user-stream Mimi codes, or None for
        silence. Returns the assistant audio codes (8,) for this frame once
        the pipeline has filled (``None`` during the initial warm-up frames).

        ``text_token`` / ``moshi_tokens`` teacher-force the text codebook (k=0)
        and the assistant audio codebooks (k=1..8) respectively. These are used
        during system-prompt priming (voice + persona), where the model is fed
        a fixed voice and persona instead of generating them.
        """
        ct = self.ct

        # Fill cache with provided user tokens at (offset + delay) % CT.
        if user_codes_frame is not None:
            for q in range(AUDIO_TOKENS_PER_STREAM):
                k = AUDIO_TOKENS_PER_STREAM + 1 + q  # k = 9..16
                wp = (self.offset + DELAYS[k]) % ct
                self.cache[0, k, wp] = user_codes_frame[q]
                self.provided[0, k, wp] = True
        # The text stream is GENERATED by the model (it drives the assistant's
        # speech). The reference S2S loop samples it every step; only honour an
        # explicitly supplied ``text_token`` (e.g. to force-feed a transcript).
        if text_token is not None:
            wp = (self.offset + DELAYS[0]) % ct
            self.cache[0, 0, wp] = text_token
            self.provided[0, 0, wp] = True
        # Teacher-force the assistant audio stream (k = 1..8) during priming.
        if moshi_tokens is not None:
            for q in range(AUDIO_TOKENS_PER_STREAM):
                k = 1 + q  # k = 1..8
                wp = (self.offset + DELAYS[k]) % ct
                self.cache[0, k, wp] = moshi_tokens[q]
                self.provided[0, k, wp] = True

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
        sampled_text = _sample_token(text_logits, self.temp_text, self.top_k_text, self._rng)
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


def _build_models(model_dir: str, device: str, lm_dtype: str = "f32"):
    """Export the four ONNX models from the native checkpoints (once).

    The Mimi codec is always built in float32 (its fp16 export currently hits a
    Conv dtype mismatch); the Moshi LM honours ``lm_dtype`` (use ``"f16"`` on
    CUDA for real-time streaming).
    """
    from mobius.integrations.moshi import build_mimi, build_moshi_lm

    os.makedirs(model_dir, exist_ok=True)
    ep = "cuda" if device == "cuda" else "default"
    print(f"[build] Mimi codec (f32) from {_MODEL_ID} ...")
    mimi = build_mimi(_MODEL_ID, execution_provider=ep)
    mimi.save(os.path.join(model_dir, "mimi"))
    # Mimi saves encoder/ and decoder/ subdirs; flatten the names we load.
    for role in ("encoder", "decoder"):
        src = os.path.join(model_dir, "mimi", role)
        dst = os.path.join(model_dir, f"mimi_{role}")
        if os.path.isdir(src) and not os.path.isdir(dst):
            os.rename(src, dst)

    print(f"[build] Moshi LM ({lm_dtype}, temporal + depformer) from {_MODEL_ID} ...")
    dtype = None if lm_dtype == "f32" else lm_dtype
    lm = build_moshi_lm(_MODEL_ID, dtype=dtype, execution_provider=ep)
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


# --- Streaming (rolling-window) Mimi codec --------------------------------
#
# The exported Mimi encoder/decoder are whole-utterance graphs, but their
# convolutions are causal. To run frame-by-frame in real time we keep a short
# ring buffer of the most recent ``window`` frames, (re)run the codec on that
# window each step, and keep only the newest frame's output. ``window`` must
# cover the codec receptive field (~0.6 s is ample at 12.5 Hz).


class StreamingMimiEncoder:
    """Encode one 12.5 Hz frame at a time with rolling left context."""

    def __init__(self, session, window: int = 8):
        self.s = session
        self.win = window
        self.buf = np.zeros((1, 1, window * FRAME_SIZE), np.float32)

    def push(self, frame: np.ndarray) -> np.ndarray:
        """Frame (FRAME_SIZE,) float32 -> (8,) int64 codes for that frame."""
        self.buf = np.roll(self.buf, -FRAME_SIZE, axis=2)
        self.buf[0, 0, -FRAME_SIZE:] = frame
        codes = self.s.run(["codes"], {"waveform": self.buf})[0]  # (1, 8, win)
        return codes[0, :, -1]


class StreamingMimiDecoder:
    """Decode one frame at a time (8 codebooks) with rolling left context."""

    def __init__(self, session, window: int = 8):
        self.s = session
        self.win = window
        self.buf = np.zeros((1, MIMI_CB, window), np.int64)

    def push(self, codes: np.ndarray) -> np.ndarray:
        """Codes (8,) int64 -> (FRAME_SIZE,) float32 waveform for that frame."""
        self.buf = np.roll(self.buf, -1, axis=2)
        self.buf[0, :, -1] = codes
        wav = self.s.run(["waveform"], {"codes": self.buf})[0]  # (1, 1, win*1920)
        return wav[0, 0, -FRAME_SIZE:]


def _stream_frames(audio_path: str | None, max_frames: int):
    """Yield (FRAME_SIZE,) float32 user-stream frames from a wav (or silence)."""
    if audio_path is None:
        for _ in range(max_frames):
            yield np.zeros(FRAME_SIZE, np.float32)
        return
    import soundfile as sf

    wav, sr = sf.read(audio_path, dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != SAMPLE_RATE:
        raise SystemExit(f"Input audio must be {SAMPLE_RATE} Hz mono (got {sr} Hz).")
    n = (len(wav) // FRAME_SIZE) * FRAME_SIZE
    for t in range(0, min(n, max_frames * FRAME_SIZE), FRAME_SIZE):
        yield wav[t : t + FRAME_SIZE]


def run_stream_file(moshi, args) -> None:
    """Simulated real-time stream from a wav (or silence): measure RTF + save."""
    out_chunks: list[np.ndarray] = []
    per_frame_ms: list[float] = []
    pace = 1.0 / FRAME_RATE  # 80 ms wall-clock budget per frame

    print(f"[stream] simulated real-time ({FRAME_RATE} Hz, {pace * 1000:.0f} ms/frame)")
    moshi.warmup()
    moshi.reset_stream()
    wall0 = time.perf_counter()
    for frame in _stream_frames(args.audio, args.frames):
        t0 = time.perf_counter()
        out = moshi.process_frame(frame)
        if out is not None:
            out_chunks.append(out)
        dt = time.perf_counter() - t0
        per_frame_ms.append(dt * 1000)
        if args.pace:
            # Sleep to emulate a live 12.5 Hz source (skip if we overran).
            slack = pace - (time.perf_counter() - t0)
            if slack > 0:
                time.sleep(slack)
    n = len(per_frame_ms)
    wall = time.perf_counter() - wall0
    audio_s = n / FRAME_RATE
    import statistics as st

    mean_ms = st.mean(per_frame_ms)
    p90 = sorted(per_frame_ms)[int(0.9 * (n - 1))]
    over = sum(ms > pace * 1000 for ms in per_frame_ms)
    print(
        f"[stream] {n} frames | compute mean={mean_ms:.1f}ms p90={p90:.1f}ms "
        f"max={max(per_frame_ms):.1f}ms | budget={pace * 1000:.0f}ms"
    )
    rtf = (mean_ms / 1000) / pace
    verdict = "REAL-TIME OK" if rtf < 1.0 else "OVER BUDGET"
    print(
        f"[stream] compute RTF={rtf:.2f} ({verdict}); {over}/{n} frames over budget; "
        f"wall={wall:.2f}s for {audio_s:.2f}s audio"
    )
    if out_chunks and args.save_to:
        import soundfile as sf

        os.makedirs(args.save_to, exist_ok=True)
        wav = np.concatenate(out_chunks)
        out_path = os.path.join(args.save_to, "assistant_stream.wav")
        sf.write(out_path, wav, SAMPLE_RATE)
        print(f"[stream] wrote {out_path} ({len(wav) / SAMPLE_RATE:.2f}s)")


def run_stream_mic(moshi, args) -> None:
    """Live full-duplex from microphone to speaker (requires audio hardware)."""
    import queue

    try:
        import sounddevice as sd
    except Exception as exc:  # pragma: no cover - hardware/dep dependent
        raise SystemExit(
            f"--mic needs the 'sounddevice' package and audio hardware (import failed: {exc})."
        ) from exc

    in_q: queue.Queue = queue.Queue()
    out_q: queue.Queue = queue.Queue()

    def in_cb(indata, frames, t, status):  # pragma: no cover - hardware
        in_q.put(indata[:, 0].copy())

    def out_cb(outdata, frames, t, status):  # pragma: no cover - hardware
        try:
            outdata[:, 0] = out_q.get_nowait()
        except queue.Empty:
            outdata[:, 0] = 0.0

    print("[mic] live full-duplex; press Ctrl-C to stop.")
    moshi.warmup()
    moshi.reset_stream()
    with (
        sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, blocksize=FRAME_SIZE, callback=in_cb
        ),
        sd.OutputStream(
            samplerate=SAMPLE_RATE, channels=1, blocksize=FRAME_SIZE, callback=out_cb
        ),
    ):
        try:
            while True:  # pragma: no cover - hardware
                frame = in_q.get()
                out = moshi.process_frame(frame)
                if out is not None:
                    out_q.put(out.astype(np.float32))
        except KeyboardInterrupt:
            print("\n[mic] stopped.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default="output/personaplex/onnx")
    parser.add_argument("--audio", default=None, help="24kHz mono user-stream wav")
    parser.add_argument("--frames", type=int, default=25, help="frames if no --audio")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument(
        "--lm-dtype",
        choices=["f32", "f16"],
        default="f32",
        help="Moshi LM dtype (use f16 on cuda for real-time)",
    )
    parser.add_argument("--allow-tf32", action="store_true")
    parser.add_argument("--save-to", default=None, help="dir to write assistant.wav")
    parser.add_argument("--skip-build", action="store_true", help="reuse existing --model-dir")
    parser.add_argument(
        "--stream", action="store_true", help="simulated real-time stream (measures RTF)"
    )
    parser.add_argument(
        "--mic", action="store_true", help="live full-duplex mic->speaker (needs sounddevice)"
    )
    parser.add_argument(
        "--no-pace",
        dest="pace",
        action="store_false",
        help="don't sleep to 12.5Hz in --stream (max-throughput RTF)",
    )
    parser.set_defaults(pace=True)
    args = parser.parse_args()

    if not args.skip_build and not os.path.isdir(os.path.join(args.model_dir, "temporal")):
        _build_models(args.model_dir, args.device, args.lm_dtype)

    moshi = MoshiORT(args.model_dir, args.device, args.allow_tf32)

    if args.mic:
        run_stream_mic(moshi, args)
        return
    if args.stream:
        run_stream_file(moshi, args)
        return

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
