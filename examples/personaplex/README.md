# PersonaPlex / Moshi — full-duplex speech-to-speech with ONNX Runtime

Real-time, full-duplex voice chat with
[`nvidia/personaplex-7b-v1`](https://huggingface.co/nvidia/personaplex-7b-v1)
(Kyutai **Moshi** architecture) exported to ONNX by `mobius` and run with
`onnxruntime`.

The pipeline is four ONNX sub-models:

```
user audio --> Mimi encoder --> [Moshi temporal + depformer] --> Mimi decoder --> assistant audio
```

* **Mimi encoder** `waveform (B,1,T) -> codes (B,8,Tf)`
* **Moshi temporal** (7B) `frame (B,17,S) -> hidden + text_logits + KV`
* **Moshi depformer** `hidden + prev_token + substep_index + KV -> logits`
  (stepped 16× per frame, once per audio codebook)
* **Mimi decoder** `codes (B,8,Tf) -> waveform (B,1,T)`

Each 12.5 Hz frame (1920 samples @ 24 kHz = 80 ms of audio) must be processed
within 80 ms for real time. On an fp16 LM + CUDA GPU the Moshi LM is
~27 ms/frame (~3× headroom); CPU fp32 (~1.8 s/frame) is far too slow for the
streaming/server modes.

## Files

| File | Purpose |
|------|---------|
| `moshi_ort.py` | Builds the ONNX models and runs the generation loop (offline / `--stream` / `--mic`). |
| `server.py` | `aiohttp` WebSocket server for browser-based real-time chat. Loads pre-built models; does **not** import `mobius`. |
| `static/index.html` | Browser client: mic capture + speaker playback at 24 kHz. |

## 1. Build the ONNX models (needs `mobius`)

Build the models once in an environment that has `mobius` installed. For
real-time speed export the Moshi LM in **fp16** (the Mimi codec stays fp32):

```bash
python examples/personaplex/moshi_ort.py \
    --device cuda --lm-dtype f16 --frames 1 \
    --model-dir output/personaplex/onnx
```

This writes `encoder/`, `decoder/`, `temporal/`, `depformer/` under
`--model-dir`. (Graph construction runs on CPU even for `--device cuda`, so no
GPU is needed for the build step.)

## 2. Run the browser server (needs a CUDA GPU for real time)

The server only loads ONNX models, so it can run in a lightweight
`onnxruntime-gpu` environment without `mobius`:

```bash
pip install aiohttp onnxruntime-gpu numpy sentencepiece huggingface_hub
python examples/personaplex/server.py \
    --model-dir output/personaplex/onnx --device cuda \
    --host 127.0.0.1 --port 7681
```

On Ampere+/H200 GPUs ORT defaults to TF32 for fp32 matmuls, which can flip
greedy sampling; the server uses `use_tf32=0` for fp32 parity (pass
`--allow-tf32` to keep the faster default). TF32 does not affect the fp16 LM.

### Open it in your browser (SSH port-forward)

If the GPU box is remote, forward the port to your laptop and open the page
locally (browsers only grant microphone access on `localhost`/HTTPS):

```bash
ssh -L 7681:localhost:7681 <user>@<gpu-host>
# then open http://localhost:7681 and click "Start"
```

Click **Start session**, then **Start**, allow microphone access, and talk —
you should hear Moshi respond in real time. Only one tab can connect at a time
(single-user demo); a second connection receives a `busy` message.

### Persona + voice customization

Before talking, the page lets you condition the assistant (PersonaPlex
"system prompt" priming, ported from `LMGen.step_system_prompts`):

* **Persona (system prompt):** a text box (defaults to a friendly-teacher
  persona). The text is wrapped as `<system> … <system>`, tokenized with the
  model's SentencePiece tokenizer (`tokenizer_spm_32k_3.model`, downloaded from
  the HF repo on first run), and force-fed on the text stream.
* **Voice sample (optional):** record ~6 s of speech to clone the assistant's
  voice. The recording is Mimi-encoded and force-fed on the assistant audio
  stream so the model continues in that voice.

Click **Start session** to prime (a couple of seconds), wait for *ready*, then
**Start** to converse. **Restart session** re-primes with new settings.

The server pass-through is:

1. browser connects → server replies `config`
2. browser sends `{"persona": "...", "hasVoice": true|false}` (+ a binary
   float32 24 kHz PCM blob if `hasVoice`)
3. server tokenizes the persona, Mimi-encodes the voice, runs the 4-phase
   priming, replies `ready`
4. live 1920-sample float32 frames stream both ways

Persona priming is optional: if the tokenizer can't be loaded the server logs a
warning and disables text prompts (voice still works). Point `--tokenizer` at a
local `tokenizer_spm_32k_3.model` to avoid the HF download.

## 3. Offline / terminal modes (no browser)

`moshi_ort.py` also runs standalone:

```bash
# Offline: generate a few frames from silence, save assistant audio
python examples/personaplex/moshi_ort.py --frames 25 --save-to out/personaplex

# Drive with a real input wav as the user stream
python examples/personaplex/moshi_ort.py --audio user.wav --save-to out/personaplex

# Simulated real-time stream from a wav (reports RTF / per-frame budget)
python examples/personaplex/moshi_ort.py --skip-build --device cuda \
    --lm-dtype f16 --stream --audio user.wav --model-dir output/personaplex/onnx

# Live full-duplex mic -> speaker (needs `sounddevice` + audio hardware)
python examples/personaplex/moshi_ort.py --skip-build --device cuda --mic \
    --model-dir output/personaplex/onnx
```

`--skip-build` reuses an already-exported `--model-dir`.

## Performance reference (H200, fp16 LM + fp32 Mimi, `use_tf32=0`)

`--stream` over 60 frames: RTF ≈ 0.63, per-frame mean 51 / p90 57 / max 65 ms,
0/60 over the 80 ms budget. The WebSocket server measures a similar RTF
(≈ 0.77 including network/scheduling) and reports over-budget frame counts on
client disconnect.

## Notes

* Mobius detects the native Kyutai checkpoint and builds the flat four-model
  package through `mobius build --model nvidia/personaplex-7b-v1 OUTPUT`.
* The Mimi codec is built in fp32 by default; pairing an fp16 LM with an fp32
  Mimi gives exact codec codes with full real-time headroom.
* `MoshiORT.warmup()` runs a few frames at connect time to absorb the
  first-frame CUDA autotune stall (otherwise the first real frame glitches).
