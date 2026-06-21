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

    pip install aiohttp
    python examples/personaplex/server.py \
        --model-dir output/personaplex/onnx --device cuda \
        --host 0.0.0.0 --port 8080

Open ``http://localhost:8080`` (or port-forward the remote port over SSH:
``ssh -L 8080:localhost:8080 <host>``) and click **Start**.

Single-user demo: the model keeps one conversation state, so only one browser
tab should be connected at a time. Each new connection resets the state.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time

import numpy as np

# Import the ORT runtime helper that lives next to this file.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from moshi_ort import FRAME_SIZE, MoshiORT

try:
    from aiohttp import WSMsgType, web
except ImportError as exc:  # pragma: no cover - dependency hint
    raise SystemExit(
        "server.py needs aiohttp. Install it in your runtime env: pip install aiohttp"
    ) from exc

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")


async def index(request: web.Request) -> web.StreamResponse:  # noqa: RUF029
    return web.FileResponse(os.path.join(STATIC_DIR, "index.html"))


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
        # warm up + reset on a worker thread to keep the event loop responsive.
        await asyncio.to_thread(moshi.warmup)
        await asyncio.to_thread(moshi.reset_stream)
        await ws.send_str("ready")

        n_frames = 0
        budget_ms = 1000.0 / 12.5  # 80 ms per frame
        over_budget = 0
        try:
            async for msg in ws:
                if msg.type == WSMsgType.BINARY:
                    frame = np.frombuffer(msg.data, dtype=np.float32)
                    if frame.size != FRAME_SIZE:
                        # Pad/trim defensively to one frame.
                        frame = np.resize(frame, FRAME_SIZE).astype(np.float32)
                    t0 = time.perf_counter()
                    out = await asyncio.to_thread(moshi.process_frame, frame)
                    dt_ms = (time.perf_counter() - t0) * 1000.0
                    n_frames += 1
                    if dt_ms > budget_ms:
                        over_budget += 1
                    if out is not None:
                        await ws.send_bytes(
                            np.ascontiguousarray(out, dtype=np.float32).tobytes()
                        )
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


def build_app(model_dir: str, device: str, allow_tf32: bool) -> web.Application:
    print(f"[server] loading Moshi ONNX models from {model_dir} on {device}...")
    moshi = MoshiORT(model_dir, device, allow_tf32)
    print("[server] models loaded.")

    app = web.Application()
    app["moshi"] = moshi
    app["lock"] = asyncio.Lock()
    app.router.add_get("/", index)
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
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7681)
    args = parser.parse_args()

    if not os.path.isdir(os.path.join(args.model_dir, "temporal")):
        raise SystemExit(
            f"No models under {args.model_dir!r}. Build them first with moshi_ort.py "
            "(see this file's module docstring)."
        )

    app = build_app(args.model_dir, args.device, args.allow_tf32)
    print(f"[server] listening on http://{args.host}:{args.port}  (open / in a browser)")
    web.run_app(app, host=args.host, port=args.port, print=None)


if __name__ == "__main__":
    main()
