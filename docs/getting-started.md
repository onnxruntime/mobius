# Getting Started

**mobius** generates ONNX models for common GenAI architectures
(LLMs, MoE, multimodal, speech-to-text, encoder-only, encoder-decoder, vision,
audio, and diffusion) built directly as ONNX graphs using
`onnxscript.nn.Module`. It supports building from HuggingFace model IDs with
automatic weight downloading and conversion, including bfloat16 models via
`ir.LazyTensor`-based dtype casting.

📦 [GitHub Repository](https://github.com/onnxruntime/mobius)

## Installation

```bash
pip install mobius-onnx
```

For development and testing:

```bash
pip install -e ".[testing]"
```

### Dependencies

- **Required**: `onnxscript`, `onnx_ir`, `numpy`, `torch`, `huggingface_hub`, `tqdm`
- **Optional (transformers)**: `transformers`, `safetensors` — for building from HuggingFace model IDs
- **Optional (gguf)**: `gguf` — for building from GGUF files
- **Optional (testing)**: `onnxruntime-easy`, `transformers`, `safetensors` — for running tests

## Quick Start

### Build from a HuggingFace model ID

```python
from mobius import build

pkg = build("meta-llama/Llama-3.2-1B")
pkg.save("output/llama-3.2-1b/")
```

### Build from a custom module

```python
from mobius import build_from_module
from mobius._configs import ArchitectureConfig
from mobius.models.base import CausalLMModel

config = ArchitectureConfig(
    vocab_size=32000,
    max_position_embeddings=4096,
    hidden_size=4096,
    intermediate_size=11008,
    num_hidden_layers=32,
    num_attention_heads=32,
    num_key_value_heads=32,
    hidden_act="silu",
    head_dim=128,
    pad_token_id=0,
)

model = build_from_module(CausalLMModel(config), config)
model["model"]  # ir.Model
```

### Build a multi-component model

Models with multiple components (e.g. encoder + decoder, or vision + text)
can be exported as a package:

```python
from mobius import build

pkg = build("openai/whisper-tiny")
pkg.save("/output/whisper/")
# Produces encoder.onnx and decoder.onnx
```

### Build different model types

```python
from mobius import build

# Encoder-only (BERT)
pkg = build("bert-base-uncased")

# Encoder-decoder (T5)
pkg = build("google-t5/t5-small")

# Multimodal (LLaVA)
pkg = build("llava-hf/llava-1.5-7b-hf")

# Audio encoder (Wav2Vec2)
pkg = build("facebook/wav2vec2-base")
```

### Export Drama 3 token generation

Drama 3 exports three graphs, not a standalone text-to-waveform service:

```python
from mobius import build

pkg = build(
    "fishaudio/drama-3",
    revision="b2b5ee998034b2975720906c815e7fb0e9f007bd",
    dtype="f32",
    keep_quantized=False,
)
pkg.save("output/drama-3", max_workers=1)
```

Equivalent CLI:

```bash
mobius build --model fishaudio/drama-3 --revision b2b5ee998034b2975720906c815e7fb0e9f007bd --dtype f32 --dequantize --max-workers 1 --output output/drama-3
```

The official model ID defaults to this pinned revision; explicit revisions and
local checkpoint directories are honored. The 35B checkpoint stores BF16
weights and block-scaled FP8 experts. `--dequantize` is required: conversion
streams parameters lazily into dense FP32 external data; `max_workers=1` bounds
serialization to one parameter at a time. Native
`bsgemm-moe` execution is **not** claimed. Plan for roughly 144 GB of dense FP32
weights, plus the source checkpoint and runtime allocations. FP16 and BF16
production exports remain **explicitly disabled**. Test-only qualification
prototypes pass synthetic full-logit parity at `1e-2` and 20 exact cached greedy
frames, but stricter pinned real-weight probes exposed remaining production-size
rounding differences. Full-checkpoint reduced-precision validation is pending.
The FP16 qualification
prototype implements FP32 residual streams, native FP16-input/FP32-output
`GemmFloat8` tower down projections, and logged FP32 saturation before backbone
weight casts. Overflow and synthetic-source tests pass, but a pinned real-weight,
production-dimension reduced fixture still exceeds the unchanged `1e-2` logit
gate (FP16: 3 of 12288 prefill-tower values, maximum violating error `0.01171875`;
BF16: 373 values, maximum violating error `0.0625`).
That is a failed qualification, not permission to widen tolerances or advertise
reduced-precision support. Checkpoint storage is never modified; other adapters retain their
conversion behavior unless they explicitly opt into FP16 saturation.
Recurrent state, nonlinear opmath and RoPE remain FP32. Masks are preserved
when padding is present; redundant causal masks are omitted on unpadded
reduced-precision attention to enable the source-compatible fused CUDA path.
ORT 1.28's native BF16 `CausalConvWithState` fallback also does not initialize
on CUDA; qualification prototypes use standard-ONNX expansion for this dtype.

| Graph | Inputs | Outputs |
|---|---|---|
| `embedding` | `frames`: INT64 `[B,T,11]`, each row `[main_token, code0, ..., code9]` | `inputs_embeds`, `shared_input`: `[B,T,H]` |
| `decoder` | `inputs_embeds`, `attention_mask`, `position_ids`, per-layer KV/conv/recurrent state | `logits`, `hidden_states`, updated states |
| `audio_decoder` | `hidden_states`, `codes`: INT64 `[B,T,10]`, `semantic_mask`: BOOL `[B,T]`, `shared_input`, mask, positions, tower KV caches | `codebook_logits`: `[B,T,9,4096]`, updated tower KV caches |

Use the checkpoint tokenizer and Fish Speech's `reference-bracket-v1` prompt
format. Initialize ordinary KV caches with sequence length zero; initialize
DeltaNet conv/recurrent state with zeros and the **actual batch size**.
Recurrent state is always FP32. Masks are INT64 `[B,past+T]` and positions are
INT64 `[B,T]`. The audio tower's cache persists across **time**, not just within
one frame. Serialized symbolic dimensions are component-qualified; initialize
KV caches using their sequence axis (axis 2), not a bare dimension-name match.
Padding must be removed before running the recurrent backbone;
padded rows would otherwise update its state.

For teacher-forced prompt ingestion, the backbone sees frame `p`, while the
tower's `codes` and `semantic_mask` label frame `p+1`; `shared_input` comes
from frame `p`. `semantic_mask` tests the next main token against the inclusive
semantic band `[248099,252194]`. Code zero is that main token minus 248099
(clipped to `[0,4095]`); EOS is 248046.

For generation, sample the main token from decoder logits restricted to EOS
and the semantic band. Fill code zero, then run `audio_decoder` once per
remaining codebook, sampling code `i+1` from logit slice `i`. Each invocation
uses the **same previous-frame caches**, not the preceding invocation's
outputs; retain only the last invocation's present caches. Later code inputs
may initially be zero because logit slice `i` depends only on codes `0..i`.
This teacher-forced ABI favors a compact export over efficient fused sampling.

For the faster generation ABI, explicitly select `task="fish-omni-generation"`.
It replaces `audio_decoder` with `audio_generator`, **not** a second copy of its
weights:

```python
pkg = build(
    "fishaudio/drama-3",
    dtype="f32",
    execution_provider="cuda",
    task="fish-omni-generation",
    keep_quantized=False,
)
pkg.save("output/drama-3-fast", max_workers=1)
```

The FP32 CUDA route uses packed expert banks and `com.microsoft::MoE`, executing
selected experts rather than evaluating all 256 experts per layer. Its raw-logit
routing and physical expert-byte layout match ORT's floating MoE ABI. It still
stores dense FP32 weights: this is **not native FP8**, and it does not remove the
large model-memory requirement. Other floating routes gather and evaluate only
selected token/expert pairs, preserving their activation and routing rounding
boundaries. Prefill is processed in bounded 16-token chunks; short prompts and
single-token decode bypass padding/Scan. This avoids evaluating inactive experts
whose FP16 overflows could otherwise poison output through `Inf * 0`.

`audio_generator` accepts the original tower inputs plus scalar BOOL `generate`
and `do_sample`, and per-row `temperature`, `top_k`, `top_p`, `seed`, `counter`
controls matching the sampling policies below. Set `generate=false` for
teacher-forced prompt prefill; no RNG/sorting runs in this mode. For generation,
use `T=1`, put the main semantic code in `codes[...,0]`, and set
`generate=true`. `do_sample=false` is exact greedy selection; `true` performs
the source sampling rule. The graph generates each residual code before the
next layer consumes it, traversing the tower **once per frame**, instead of
nine complete teacher-forced replays. With the released configuration this
reduces tower-block evaluations from 117 to 13 per frame; this is a work-count
reduction, **not a measured 9x latency claim**.

Outputs include `frame_codes` INT64 `[B,1,10]`, `valid` BOOL `[B]`,
`next_counter` INT64 `[B]`, logits and the complete committed KV state.
Nonfinite active sampling/neural inputs or multi-frame generation produce
`valid=false`, `frame_codes=-1`, and an unchanged counter. Do not commit cache
outputs on an invalid result. Greedy generation and prefill do not advance RNG
state. The test-only reduced-precision prototypes exercise this one-pass task
and selected-only expert execution, without opening production admission or
claiming the FP32 CUDA route's fused-expert kernel. Source-correct FP32 RoPE/mask dispatch can retain standard
Attention rather than a GQA rewrite. No reduced-precision throughput claim is
made without measurement.

Sampling math is exported in `policies/main_sampler.onnx` and
`policies/codebook_sampler.onnx` and restored in `pkg.policy_components` on load.
Both take last-step `logits` `[B,V]`, `temperature`/`top_p` FLOAT `[B]`, `top_k`,
`seed`, and `counter` INT64 `[B]`. The main sampler also takes the last 16 main
tokens as `history` INT64 `[B,16]`, restricts candidates to EOS and the semantic
band, and rerolls only repeated semantic tokens. It returns `semantic_code`
alongside `token`, `valid`, and `next_counter`. The codebook sampler returns the
latter three fields and is invoked after each conditioned codebook projection.
Check every `valid` result: invalid/nonfinite inputs produce `token=-1` and
leave the counter unchanged.

These policies implement Fish's drop-crossing-token nucleus rule, temperature
floor `1e-5`, and exponential-race sampling. They use explicit, reproducible
counter RNG state, **not Torch/Philox seed-bit identity**. Main sampling always
consumes both first/retry draws; counters advance by twice the compact candidate
count for main tokens and by the codebook vocabulary size for residual tokens,
modulo `2**31-1`. The caller updates history and orchestrates the temporal loop;
no huge neural weights are duplicated in the sampling policies.

Reference audio encoding and ORT GenAI generation integration remain external.
The waveform decoder can be
exported as a separate FP32 ONNX graph:

```python
from mobius.tasks.fish_codec import load_drama3_codec_decoder

codec = load_drama3_codec_decoder()  # pinned codec.pth, or checkpoint_path=local_path
codec.save("output/drama-3-codec")
# For generated NumPy frames [B,T,11], feed:
codes = frames[..., 1:].transpose(0, 2, 1)  # INT64 [B,10,T]
```

`waveform_decoder` produces FP32 `[B,1,T*2048]` at 44100 Hz. It is a full-sequence
decoder, not a streaming-cache codec. Keep it FP32 independently of the text/tower
dtype. The pinned real codec matches all 40960 samples from the 20-frame native
token golden on CPU: maximum absolute error `1.848e-6`, RMSE `1.184e-7`,
SNR `117.79 dB`. This verifies decoding, not reference encoding or speech quality.
Its residual-codebook upper clamping matches upstream, including tower IDs above
the codec's nine residual vocabularies of 1024.

`--runtime onnx-genai` writes graph-derived
advisory component contracts, not an executable speech pipeline;
`--runtime ort-genai` is explicitly rejected. Fish Audio's checkpoint/source
license is separate from Mobius's MIT license.

The complete pinned checkpoint was validated on H200 with ORT 1.28 CUDA and
TF32 disabled: a 30-row prefill and 20 greedy frames match the independent
upstream FP32 pipeline. Maximum observed decoder/tower logit differences were
`3.624e-5` / `3.910e-5`. This is token-level evidence, not waveform or speech-quality
validation. The native reference uses an argmax sampler for determinism.

To reproduce on a sufficiently large CUDA host, select free devices and run:

```bash
python scripts/validate_drama3.py download --work-dir drama3-validation
python scripts/validate_drama3.py export --work-dir drama3-validation
python scripts/validate_drama3.py reference --work-dir drama3-validation
python scripts/validate_drama3.py compare --work-dir drama3-validation
```

By default, reference loading spreads the backbone across GPUs 0-3 and places
the tower on GPU 4; comparison places embedding/audio/decoder on GPUs 0/1/2.
Override these with the script's device flags on a shared host. The script
retains full reference vectors and verifies the committed token-generation
golden, while the generic waveform golden runner remains out of scope.

For performance measurement, `scripts/benchmark_drama3.py` runs actual cached
requests with I/O binding: recurrent/KV state and neural intermediates remain on
the selected device, while only small policy outputs and the final waveform
return to the host. It supports both the original replay task and the one-pass
task, so their timings can be compared using the same ORT build and options.
Supply the pinned HF `config.json` and the independent reference stage's
`state-00.npz` prompt:

```bash
python scripts/benchmark_drama3.py --package output/drama-3-fast --config drama3-config.json --prompt drama3-validation/reference/state-00.npz --codec output/drama-3-codec --device cuda --device-id 0 --sample --frames 100 --output drama3-perf.json
```

The report separates initialization, prefill, first code frame, steady
frames/second, p50/p95 frame latency and waveform decoding. End-to-end RTF includes
prefill, generation and waveform decoding relative to actual audio duration.
The codec is full-sequence: **first-code-frame latency is not streaming
time-to-audio**. Omit `--codec` only for a token-pipeline diagnostic; such a run
does not attest full speech performance.

Use an idle, sufficiently large GPU. Busy devices are rejected by default;
`--allow-contended` and `--profile` are explicitly unqualified diagnostics.
Performance sign-off requires at least 50 complete frames, no competing GPU
processes observed around the run, unprofiled timing and an independently
established quality gate. The optional `--max-rtf` target defaults to `1.0`;
qualified end-to-end runs exceeding it fail after writing their measurements.
The current optimizations have source-parity and execution-path coverage, but
**no fresh uncontended full-model H200 latency/RTF result is claimed**.

### Build from a GGUF file

Convert a GGUF model (e.g. from llama.cpp) to ONNX. Quantized target storage is
used by default where the selected graph supports it:

```python
from mobius import build_from_gguf

pkg = build_from_gguf("path/to/model.gguf")
pkg.save("output/model/")
```

Or via CLI:

```bash
mobius build-gguf path/to/model.gguf --output output/model/
```

Pass `--dequantize` on the CLI, or `keep_quantized=False` to
`build_from_gguf()`, when a fully float ONNX model is required. F32-, F16-, and
BF16-only GGUFs automatically use the float path.

`keep_quantized=True` does **not** promise source byte or numerical fidelity.
Some qtypes are byte-preserved or losslessly repacked, while others are
dequantized and lossily requantized to a target such as INT4 affine block-32.
Mobius emits one aggregate warning for lossy conversion and saves the complete
classification as `quantization_report.json`; do not label normalized output as
the source GGUF preset (for example, Q4_K_M). The report lists source qtype
counts/bytes, explicit float tensors, storage fidelity, and compute capability.

Packed storage and runtime compute are separate. A runtime may execute a native
`MatMulNBits` custom op, or Mobius may inline its portable standard-ONNX
fallback (nibble `BitShift`/`BitwiseAnd`, `DequantizeLinear`, then float
`MatMul`). Inlining the fallback does not replace packed weight initializers
with dense float initializers and does not promise a particular ORT kernel.

> **Note**: GGUF support requires the optional `gguf` package:
> `pip install mobius-onnx[gguf]`

### Build quantized models (GPTQ/AWQ)

Quantized HuggingFace models are handled automatically:

```python
from mobius import build

# GPTQ quantized
pkg = build("TheBloke/Llama-2-7B-GPTQ")

# AWQ quantized
pkg = build("TheBloke/Llama-2-7B-AWQ")
```

### Choose a target dtype

```python
from mobius import build

# Build with float16 weights
pkg = build("meta-llama/Llama-3.2-1B", dtype="f16")

# Build with bfloat16 weights
pkg = build("meta-llama/Llama-3.2-1B", dtype="bf16")
```

### Extensibility

Register custom architectures or tasks:

```python
from mobius import registry
from mobius.tasks import TASK_REGISTRY

registry.register("my_arch", MyCustomModel)
TASK_REGISTRY["my-task"] = MyCustomTask
```

### Load config from a local directory

```python
from mobius import ArchitectureConfig

config = ArchitectureConfig.from_file("/path/to/model/")
config.validate()  # Check field consistency
```

### Apply graph optimizations

```python
from onnxscript.rewriter import rewrite
from mobius import build
from mobius.rewrite_rules import group_query_attention_rules, skip_norm_rules

pkg = build("Qwen/Qwen2.5-0.5B")
model = pkg["model"]
rewrite(model, pattern_rewrite_rules=group_query_attention_rules())
rewrite(model, pattern_rewrite_rules=skip_norm_rules())
```

Or via CLI:

```bash
mobius build --model Qwen/Qwen2.5-0.5B --output output/ --ep cuda --dtype f16
```

## CLI Quick Start

```bash
# Basic build
mobius build --model meta-llama/Llama-3.2-1B --output output/

# Build for CUDA with f16
mobius build --model meta-llama/Llama-3.2-1B --output output/ --ep cuda --dtype f16

# Build for ORT GenAI runtime
mobius build --model meta-llama/Llama-3.2-1B --output output/ --ep cuda --dtype f16 --runtime ort-genai
```

The `mobius` CLI has these subcommands:

- **`build`** — Build an ONNX model from a HuggingFace model ID or local config.
- **`build-gguf`** — Convert a GGUF file to ONNX.
- **`list`** — List supported models, tasks, dtypes, or execution providers.
- **`info`** — Inspect a model without building it.

For the full CLI reference including all flags, execution providers, and
optimization options, see [CLI Reference](cli_reference.md).

### Adding a new model

See the [adding-a-new-model skill](../.agents/skills/adding-a-new-model/SKILL.md)
for copy-paste examples and step-by-step instructions.

## Output Format

### ModelPackage

`build()` and `build_from_module()` return a `ModelPackage` — a dict-like
collection of named `ir.Model` objects.

- **Single-model tasks** (e.g. CausalLM): `pkg["model"]`
- **Multi-model tasks** (e.g. VLM): `pkg["model"]`, `pkg["vision"]`, `pkg["embedding"]`
- **Encoder-decoder**: `pkg["encoder"]`, `pkg["decoder"]`

### Saving to disk

```python
pkg.save("output/")
# Single model: output/model.onnx + output/model.onnx.data
# Multi model: output/model/model.onnx, output/vision/model.onnx, ...
```

Safetensors format:
```python
pkg.save("output/", external_data="safetensors")
```

### Discover supported models

```bash
# List all supported architectures
mobius list models

# Inspect a specific model
mobius info meta-llama/Llama-3.2-1B
```

## Development

### Running tests

```bash
# Unit tests (fast, no network needed)
pytest tests/build_graph -v

# Run a single model type
pytest tests/build_graph -k "phi4mm"

# Integration tests (downloads models, requires more time/memory)
pytest tests/integration -m integration -v

# Run a single integration test model
pytest tests/integration/text_test.py -m integration -k "qwen2.5-0.5b"
```

### Linting

```bash
lintrunner f --all-files
```
