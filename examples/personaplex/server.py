#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Browser-based real-time voice chat server for Moshi (personaplex-7b-v1).

A tiny ``aiohttp`` server that serves a static web page (``static/index.html``)
and a WebSocket endpoint that streams full-duplex audio to/from the ONNX Moshi
models loaded via :class:`moshi_ort.MoshiORT`.

Wire protocol (binary WebSocket frames, little-endian ``float32``):

* **client -> server**: one 1920-sample (80 ms @ 24 kHz) mono PCM frame.
* **server -> client**: one 1920-sample mono PCM assistant frame (or nothing
  during the initial warm-up frames).

The browser must capture/playback at 24 kHz mono. ``static/index.html`` forces
``new AudioContext({sampleRate: 24000})`` so no resampling is needed.

This module deliberately does **not** import ``mobius`` — it only loads
pre-built ONNX models, so it can run in a lightweight ``onnxruntime-gpu``
environment. Build the models first with ``moshi_ort.py`` (in an environment
that has ``mobius`` installed)::

    python examples/personaplex/moshi_ort.py --device cuda --lm-dtype f16 \
        --frames 1 --model-dir output/personaplex/onnx

Then run this server (CUDA recommended for real-time speed)::

    pip install aiohttp sentencepiece huggingface_hub
    python examples/personaplex/server.py \
        --model-dir output/personaplex/onnx --device cuda \
        --host 127.0.0.1 --port 7681

Open ``http://localhost:7681`` (or port-forward the remote port over SSH:
``ssh -L 7681:localhost:7681 <host>``), set a persona / optional voice sample,
click **Start session**, then **Start**.

On connect the server replies ``config``; the browser sends a JSON
``{"persona": ..., "hasVoice": ...}`` line plus an optional binary float32
24 kHz PCM voice blob. The server tokenizes the persona, Mimi-encodes the voice,
runs the 4-phase PersonaPlex system-prompt priming, replies ``ready``, then
streams live frames.

Single-user demo: the model keeps one conversation state, so only one browser
tab should be connected at a time. Each new connection resets the state.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time

import numpy as np

# Import the ORT runtime helper that lives next to this file.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from moshi_ort import (
    DEFAULT_PERSONA,
    FRAME_SIZE,
    MoshiORT,
    encode_persona,
    load_persona_tokenizer,
)

try:
    from aiohttp import WSMsgType, web
except ImportError as exc:  # pragma: no cover - dependency hint
    raise SystemExit(
        "server.py needs aiohttp. Install it in your runtime env: pip install aiohttp"
    ) from exc

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
VOICES_DIR = os.path.join(STATIC_DIR, "voices")
_VOICE_EXTS = (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".opus")


async def index(request: web.Request) -> web.StreamResponse:  # noqa: RUF029
    return web.FileResponse(os.path.join(STATIC_DIR, "index.html"))


async def voices(request: web.Request) -> web.StreamResponse:  # noqa: RUF029
    """List preset voice clips under ``static/voices/`` for the browser."""
    names: list[str] = []
    if os.path.isdir(VOICES_DIR):
        names = sorted(f for f in os.listdir(VOICES_DIR) if f.lower().endswith(_VOICE_EXTS))
    return web.json_response(names)


async def websocket_handler(request: web.Request) -> web.WebSocketResponse:
    ws = web.WebSocketResponse(max_msg_size=0)
    await ws.prepare(request)

    moshi: MoshiORT = request.app["moshi"]
    lock: asyncio.Lock = request.app["lock"]

    # Single-user demo: serialize connections so the shared conversation state
    # is never interleaved between two tabs.
    if lock.locked():
        await ws.send_str("busy: another client is connected")
        await ws.close()
        return ws

    async with lock:
        print(f"[ws] client connected: {request.remote}")

        # --- Handshake: voice + persona system prompt --------------------
        # Tell the client we're ready for config immediately so the UI unlocks
        # without waiting for warmup; warm up + reset in the background while the
        # user picks a persona/voice, then join before priming.
        await ws.send_str("config")
        warm_task = asyncio.create_task(
            asyncio.to_thread(lambda: (moshi.warmup(), moshi.reset_stream()))
        )
        persona = DEFAULT_PERSONA
        expect_voice = False
        try:
            cfg = await asyncio.wait_for(ws.receive(), timeout=300.0)
        except asyncio.TimeoutError:
            warm_task.cancel()
            await ws.close()
            return ws
        if cfg.type == WSMsgType.TEXT:
            try:
                d = json.loads(cfg.data)
                persona = (d.get("persona") or "").strip() or DEFAULT_PERSONA
                expect_voice = bool(d.get("hasVoice"))
            except (ValueError, TypeError):
                pass

        voice_pcm = None
        if expect_voice:
            vmsg = await asyncio.wait_for(ws.receive(), timeout=120.0)
            if vmsg.type == WSMsgType.BINARY:
                voice_pcm = np.frombuffer(vmsg.data, dtype=np.float32).copy()

        tokenizer = request.app.get("tokenizer")
        text_tokens = None
        if tokenizer is not None and persona:
            text_tokens = encode_persona(tokenizer, persona)
        n_voice = 0 if voice_pcm is None else voice_pcm.size // FRAME_SIZE
        print(f"[ws] priming: persona={len(text_tokens or [])} toks, voice={n_voice} frames")
        # Make sure warmup + reset finished before we prime / generate.
        await warm_task
        t0 = time.perf_counter()
        await asyncio.to_thread(moshi.prime, voice_pcm, text_tokens)
        print(f"[ws] primed in {time.perf_counter() - t0:.1f}s")
        await ws.send_str("ready")

        n_frames = 0
        budget_ms = 1000.0 / 12.5  # 80 ms per frame
        over_budget = 0
        in_rms_acc = 0.0
        out_rms_acc = 0.0
        out_count = 0
        dt_acc = 0.0
        try:
            async for msg in ws:
                if msg.type == WSMsgType.BINARY:
                    frame = np.frombuffer(msg.data, dtype=np.float32)
                    if frame.size != FRAME_SIZE:
                        # Pad/trim defensively to one frame.
                        frame = np.resize(frame, FRAME_SIZE).astype(np.float32)
                    in_rms_acc += float(np.sqrt(np.mean(frame.astype(np.float64) ** 2)))
                    t0 = time.perf_counter()
                    out = await asyncio.to_thread(moshi.process_frame, frame)
                    dt_ms = (time.perf_counter() - t0) * 1000.0
                    dt_acc += dt_ms
                    n_frames += 1
                    if dt_ms > budget_ms:
                        over_budget += 1
                    if out is not None:
                        out_rms_acc += float(np.sqrt(np.mean(out.astype(np.float64) ** 2)))
                        out_count += 1
                        await ws.send_bytes(
                            np.ascontiguousarray(out, dtype=np.float32).tobytes()
                        )
                    # Report ~once per second so we can see whether the model
                    # is receiving non-silent mic audio and producing audio out.
                    if n_frames % 12 == 0:
                        in_rms = in_rms_acc / 12
                        out_rms = out_rms_acc / max(out_count, 1)
                        frame_ms = dt_acc / 12
                        print(
                            f"[ws] {n_frames} frames | in_rms={in_rms:.4f} "
                            f"out_rms={out_rms:.4f} | last_frame={dt_ms:.0f}ms"
                        )
                        # Push live perf stats to the browser for the stats panel.
                        await ws.send_str(
                            "stats "
                            + json.dumps(
                                {
                                    "frame_ms": round(frame_ms, 1),
                                    "rtf": round(frame_ms / budget_ms, 2),
                                    "in_rms": round(in_rms, 4),
                                    "out_rms": round(out_rms, 4),
                                    "frames": n_frames,
                                    "over": over_budget,
                                }
                            )
                        )
                        in_rms_acc = out_rms_acc = dt_acc = 0.0
                        out_count = 0
                elif msg.type == WSMsgType.TEXT:
                    if msg.data == "reset":
                        await asyncio.to_thread(moshi.reset_stream)
                elif msg.type == WSMsgType.ERROR:
                    print(f"[ws] error: {ws.exception()}")
        finally:
            rtf = "n/a"
            if n_frames:
                rtf = f"{over_budget}/{n_frames} frames over {budget_ms:.0f}ms"
            print(f"[ws] client disconnected after {n_frames} frames ({rtf})")
    return ws


def build_app(
    model_dir: str, device: str, allow_tf32: bool, tokenizer_path: str | None = None
) -> web.Application:
    print(f"[server] loading Moshi ONNX models from {model_dir} on {device}...")
    moshi = MoshiORT(model_dir, device, allow_tf32)
    print("[server] models loaded.")

    tokenizer = None
    try:
        tokenizer = load_persona_tokenizer(tokenizer_path)
        print("[server] persona tokenizer loaded.")
    except Exception as exc:
        print(f"[server] persona tokenizer unavailable ({exc}); text prompts disabled.")

    app = web.Application()
    app["moshi"] = moshi
    app["tokenizer"] = tokenizer
    app["lock"] = asyncio.Lock()
    app.router.add_get("/", index)
    app.router.add_get("/voices", voices)
    app.router.add_get("/ws", websocket_handler)
    app.router.add_static("/static/", STATIC_DIR)
    return app


def main() -> None:
    # Line-buffer stdout so connect/disconnect logs appear live over SSH.
    sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default="output/personaplex/onnx")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument(
        "--allow-tf32",
        action="store_true",
        help="keep CUDA TF32 (faster, lower precision) for fp32 matmuls",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7681)
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="path to tokenizer_spm_32k_3.model (default: download from HF)",
    )
    args = parser.parse_args()

    if not os.path.isdir(os.path.join(args.model_dir, "temporal")):
        raise SystemExit(
            f"No models under {args.model_dir!r}. Build them first with moshi_ort.py "
            "(see this file's module docstring)."
        )

    app = build_app(args.model_dir, args.device, args.allow_tf32, args.tokenizer)
    print(f"[server] listening on http://{args.host}:{args.port}  (open / in a browser)")
    web.run_app(app, host=args.host, port=args.port, print=None)


if __name__ == "__main__":
    main()
