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
serialization to one parameter at a time. Plan for roughly 144 GB of dense FP32
weights, plus the source checkpoint and runtime allocations. FP16 and BF16
production exports remain disabled while real-weight qualification continues.
Test-only prototypes pass synthetic full-logit parity at `1e-2` and 20 exact
cached greedy frames. The FP16
prototype implements FP32 residual streams, native FP16-input/FP32-output
`GemmFloat8` tower down projections, and logged FP32 saturation before backbone
weight casts. Checkpoint storage is unchanged; other adapters retain their
conversion behavior unless they opt into FP16 saturation.

The pinned Fish Speech source declares `torch==2.8.0`. Run
`scripts/validate_drama3.py reference` in a separate environment with that release;
the validator checks the Torch version before loading the model. New references
record Torch, CUDA and Triton versions, and comparisons check that metadata when
both goldens include it. Reference collection checks embeddings, hidden states
and logits for finite values before writing a golden.
Earlier Torch 2.10 diagnostics are retained separately:
CUDA reduction order differs between these releases. The committed token golden
has been regenerated with Torch 2.8.0+cu126 / Triton 3.4.0; all 20 frame IDs are
unchanged.

Fresh Torch 2.8 production-dimension, one-layer real-weight probes still exceed
the `atol=rtol=1e-2` gate: FP16 prefill-tower output has 5 violations among 12288
values (maximum violating error `0.01248169`); BF16 cached tower output has
200 violations among 4096 values (maximum `0.125`). With identical source
hidden-state inputs, the BF16 tower matches bit-for-bit through prefill and three
cached calls. The remaining pipeline difference starts in the backbone and
propagates into the tower.

Identical real BF16 expert-matmul operands isolate one cached discrepancy to
the CUDA library version: cuBLAS 12.6.4 matches the Torch 2.8+cu126 reference
exactly, while cuBLAS 13.6.0 changes three output values. Algorithm and workspace
controls reproduce the same distinction. A separately built CUDA 12.6 runtime
now reproduces all three unchanged ORT matmul forms and four direct cuBLAS
controls exactly. Its source is `dceae081a48ac45b32af785b057c0a30b8aa6856`, wheel
SHA256 `9baea5c2f5e300c39b903fb99364917a861aef291c90583360003d04c82589b5`.
QMoE, XOR, Pow, full-head Flash and Log1p operator gates pass on that runtime;
whole-model reduced-precision gates still fail. Matching the ORT version alone
does not match the loaded cuBLAS implementation.

Recurrent state, nonlinear opmath and RoPE remain FP32. Masks are preserved
when padding is present. The unpadded reduced-precision prototype omits redundant
causal masks to enable Flash. Profiling the actual pinned backbone instead shows
CUTLASS memory-efficient attention with its explicit BOOL causal mask. A separate
native MEA prototype retains that mask and matches the actual source attention
fixture bit-for-bit; a forced-Flash operator comparison is not source-backend
qualification.
ORT 1.28's native BF16 `CausalConvWithState` fallback also does not initialize
on CUDA; qualification prototypes use standard-ONNX expansion for this dtype.

Native W8A8 execution is implemented in
[microsoft/onnxruntime#33644](https://github.com/microsoft/onnxruntime/pull/33644).
The exporter prepares typed E4M3FN expert banks and unchanged multiplicative
FP32 grids for `QMoE(activation_quantization="dynamic_fp8")`, without a
dequantize-to-half proxy. Byte-preserving save/load and opt-in FP32/FP16/BF16
expert-only parity against pinned Triton pass on the custom runtime
(`MOBIUS_TEST_DRAMA3_NATIVE_FP8=1`). Production native-FP8 admission remains closed.
The earlier complete native-FP8/FP32 probe, using Torch 2.10, failed the prefill
hidden-state gate
(43436 of 61440 values, maximum absolute error `0.68518`), although all 40
real expert layers pass when supplied identical source inputs and logits.
The first expert input differs by only `1.192e-7`, but changes one FP8 code;
those discontinuities amplify across subsequent layers. Qualification against
the declared Torch 2.8 environment is in progress. Complete native-FP8 source runs
in BF16 and FP16 each produce 20 finite frames. A private source-ordered BF16
candidate on the CUDA 13 runtime fails the full prefill hidden-state gate:
22817 of 61440 values violate `atol=rtol=1e-2`, with maximum absolute error
`0.640625`. All 40 native experts execute on CUDA. Its first expert input differs
in nine BF16 values (maximum `0.000244140625`); later layers amplify the difference.
An independent source replay reproduces all 20 saved states bit-for-bit.
The candidate is limited to batch-one short prefill and explicitly bypasses a
failing symbolic-shape pass, so it is diagnostic, not a production export.
Normal graph construction now succeeds with the existing constraint-propagation
optimization in
[onnx/onnx-shape-inference#135](https://github.com/onnx/onnx-shape-inference/pull/135).
A normally built candidate reuses the unchanged external weights only after
checking computation graphs, generated constants and initializer contracts;
all 120 expert-boundary tensors reproduce the original prefill exactly.
FP16 tower inference additionally uses
[corrected `GemmFloat8` geometry and output typing](https://github.com/onnx/onnx-shape-inference/commit/d915e3dc6490abf07081e873ff43d4baef81de97);
the normal FP16 build completes without skipped shape passes and preserves all
120 original expert-boundary tensors.
On the CUDA 13 runtime, the complete native-FP8/FP16 candidate fails the same
prefill gate in 16862 of 61440 values, with maximum error `1.158203125`.
Its source replay reproduces all 20 reference states exactly. CUDA-library-aligned
original/normal/Log1p variants also fail their unchanged prefill gates; matching
the numerical libraries does not by itself qualify native models.
Source-ordered reductions
and [a CUDA reciprocal-square-root fast path](https://github.com/microsoft/onnxruntime/pull/33662)
give bit-exact normalization on real first-layer inputs; source-layout `Gemm`
also matches all five isolated DeltaNet projections exactly. An artifact-only
cached recurrence matches source output and state bitwise for zero/nonzero
cache inputs. Native unit-lower triangular solves now remove the first prefix
discrepancy: the first expert inputs and logits are bit-exact for both dtypes.
With source-output controls for the experts, runtime CUDA RoPE, native triangular
solves and source MEA reproduce all 80 prefill expert-input/logit vectors and the
final hidden tensor exactly. This is diagnostic isolation, never whole-model
qualification. The remaining earliest difference is inside QMoE: matching the
source sigmoid sequence corrects 301 activation-scale discrepancies. The captured
Triton down kernel uses register-local FP8-to-FP16 conversion and FP16 MMA;
matching its fragment order reproduces all 491520 first-case weighted-down
FP32 values exactly. Matching the source's four-accumulator reduction also makes
that case's BF16 output exact. This source-compatible mode retains FP8 storage
and activation quantization but uses FP16 MMA, not native hardware FP8 MMA.

With that mode, native triangular solves, source MEA and runtime source-style
RoPE in both decoder and tower, source-faithful top-k routing now gives a complete
saved/reloaded BF16 pipeline pass: 30-row prefill, every cached vector and all
20 complete generated frames. A separate raw-bit audit checks all 101 compared
tensors: **14,469,504 FP32-transport bit patterns, zero differences**. No source
expert outputs replace runtime computation.

The routing correction follows the pinned Torch CUDA softmax, threshold gather
and unstable bitonic top-k order. Previously, swapping two equal-logit expert
slots at cached state 17 changed summation order at a BF16 rounding midpoint:
one layer-12 output differed despite exact per-expert arithmetic and recurrent
caches. An arbitrary stable tie-index rule would not reproduce that source
contract.

This validates a private batch-one/short-prefill configuration, not production
reduced-precision admission or performance. The source-compatible compute is
register-local FP16 MMA; native hardware FP8 MMA remains separately unqualified.
An independent generalized chunk-core probe also matches source output/state
bits in 32 synthetic cases (batches 1/2, lengths 1/2/30/64/65/128/129/257,
zero/nonzero state), including the source's row-count-dependent scan tiling.
Full-model generalization and production wiring remain pending; public admission
guards stay closed.

Stable `Log(Add(x, 1))` fusion is implemented for CUDA and native WebGPU in
[this independent ORT change](https://github.com/microsoft/onnxruntime/commit/3c0638c4c402900e42d1a9a0f6ee8d9ea7015460).
It keeps the exported graph in standard ONNX and activates at
`ORT_ENABLE_EXTENDED` or `ORT_ENABLE_ALL`, after provider assignment.
Use `scripts/validate_drama3.py compare --optimization-level extended` (with the
usual work directory and reference arguments) to qualify this path. Comparison
defaults to BASIC; the selected level applies to every component and is recorded
in session-start events and the validation report.
The complete FP32 pipeline also passes EXTENDED on the combined QMoE/XOR/Pow/Flash
runtime: 30-row prefill, all cached vectors, 20 exact frames and 40 actual CUDA
MoE executions, with maximum decoder/tower/hidden errors
`5.149841e-5` / `5.054474e-5` / `2.384186e-5`.
The portable FP32 source-softplus subgraph matches all 960 real reference values
exactly on CUDA while preserving the preceding `Exp` rounding.
Native WebGPU FP32 and feature-enabled FP16 were exercised through lavapipe
(software Vulkan); physical-WebGPU and browser/WASM execution remain untested.
Deprecated JSEP is excluded by ORT policy.

The custom CUDA build needs `onnxruntime_USE_FP8_QMOE=ON`, SM89+, and dynamic
activation quantization. The runtime rejects more than
65,535 token/expert pairs per call (8,191 tokens at top-8); the graph records
that limit and never silently falls back to dense expert conversion.

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
routing and physical expert-byte layout match ORT's floating MoE ABI. Gate/up
rows are concatenated offline, with half-split SwiGLU fusion and no FC3 input:
the separate-FC3 runtime path would copy a complete 2 GiB expert bank on every
layer invocation. The vocabulary projection also uses output-major `Gemm`,
avoiding a 2 GiB runtime weight transpose above ORT's constant-folding limit. It still
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

The original complete pinned FP32 export was validated on H200 with ORT 1.28 CUDA and
TF32 disabled: a 30-row prefill and 20 greedy frames match the independent
upstream FP32 pipeline. Maximum observed decoder/tower logit differences were
`3.624e-5` / `3.910e-5`. This is token-level evidence, not waveform or speech-quality
validation. The native reference uses an argmax sampler for determinism.

The current pre-fused CUDA MoE plus single-pass tower route has also been
requalified against the complete independent Torch 2.8.0+cu126 reference:
all vectors for the 30-row prefill and 20 exact complete frames pass, using CPU
embedding and CUDA decoder/tower on one H200. Both prefill profiles contain
40 actual CUDA MoE executions.

| Runtime | Maximum decoder-logit error | Maximum tower-logit error | Maximum hidden error |
|---|---:|---:|---:|
| Stock ORT 1.28 | `5.5313e-5` | `5.0306e-5` | `2.2888e-5` |
| Custom ORT 1.32, QMoE + XOR + Pow | `5.1498e-5` | `5.0545e-5` | `2.3842e-5` |

The second correctness run uses the separately preserved combined-Pow wheel
(`da934e6b1791e3225c7f2e2a61a35a5ebd907d2946a8f0c0e02c7e455dd6395d`).
The performance measurements below retain their original two-patch runtime.

Recorded single-H200 measurements of this dense-FP32 graph use ORT 1.28
(`0368187f84`), TF32 disabled, two full-request warmups, the same 30-row prompt,
CPU embeddings, CUDA decoder/tower/codec, and no competing GPU processes:

| Tower / policy | Complete frames | Steady frames/s | p50 / p95 frame ms | End-to-end RTF |
|---|---:|---:|---:|---:|
| Replay / greedy | 100 | 14.92 | 64.67 / 66.22 | 1.549 |
| Single pass / greedy | 100 | 32.79 | 28.58 / 28.71 | **0.770** |
| Single pass / sampled | 53 (native EOS) | 26.31 | 34.12 / 34.50 | 1.058 |

The two greedy runs produce identical 100-frame hashes; the optimized
resident-state 20-frame prefix also matches the independent source golden.
Single-pass greedy is 2.20x faster in steady frame rate and 2.01x faster
end-to-end than replay in this controlled comparison, not 9x. Its prefill is
334 ms, first code frame 341 ms, and full waveform delivery 3.58 seconds for
4.64 seconds of audio. Model initialization is separate, approximately
101-105 seconds; GPU memory remains very large (about 142,161 MiB in the
single-pass greedy run). These are warmed single-request observations, not
universal throughput or streaming-audio claims. **The sampled run still misses
the RTF 1.0 target** and is not a performance pass.

A separately qualified custom ORT 1.32 build combining
[microsoft/onnxruntime#33644](https://github.com/microsoft/onnxruntime/pull/33644)
and [microsoft/onnxruntime#33645](https://github.com/microsoft/onnxruntime/pull/33645),
with the codec's NC1D layout enabled, reaches **RTF 0.901** on the same sampled
53-frame case (29.28 steady frames/s, 30.28 / 30.57 ms p50 / p95, 348 ms first
code frame). It produces the identical sampled-frame hash and passes the
independent 20-frame/full-vector dense-FP32 reference on that runtime.
This run still uses **dense FP32 experts**, not the unqualified native-FP8
pipeline. The combined build includes QMoE commit `e3181530` and XOR commit
`e90d897c`, CUDA 13.3 / SM90, FP8 QMoE enabled, and Flash/MEA attention disabled.
Its wheel SHA256 is
`f7e2f4cb9255e0556ca9cbc03c40189219880039eb76cdea23a000c0f2ae97dc`.
Initialization is separate (88.61 seconds), and waveform delivery is
full-sequence, not streaming. These cross-runtime observations do not isolate
the XOR kernel's speedup; separate kernel profiles confirm its CUDA placement.

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
To qualify the accelerated route instead, add
`--task fish-omni-generation --execution-provider cuda` to the export command
in a fresh output directory. Comparison automatically selects `audio_generator`
when present, uses one greedy conditioned tower call per frame, and checks its
full logits and frame IDs against the same independent reference.
`--audio-component audio_decoder` explicitly selects the replay graph when a
package contains both tower variants. Invalid generation or changed greedy RNG
state is an error, never a successful comparison.
Experimental native-FP8 references use `--native-fp8` and a separate work
directory; comparison requires its independent `--golden` until qualification.
The validator records and checks expert execution mode, so a native reference
cannot silently be compared as dense FP32. These flags do not bypass the
production native-export gate.

For performance measurement, `scripts/benchmark_drama3.py` runs actual cached
requests with I/O binding: recurrent/KV state and neural intermediates remain on
the selected device, while only small policy outputs and the final waveform
return to the host. It supports both the original replay task and the one-pass
task, so their timings can be compared using the same ORT build and options.
For a memory-constrained single H200, `--embedding-device cpu` keeps the
roughly 2 GiB vocabulary lookup on the host while leaving decoder/tower caches
on CUDA. Only the small embedding activations cross devices; this reserves GPU
headroom for `--codec-device cuda`. Component placement is recorded in the
report. `--codec-device cpu` is also supported, with its actual decode time
included in end-to-end RTF.
Supply the pinned HF `config.json` and the independent reference stage's
`state-00.npz` prompt:

```bash
python scripts/benchmark_drama3.py --package output/drama-3-fast --config drama3-config.json --prompt drama3-validation/reference/state-00.npz --codec output/drama-3-codec --device cuda --device-id 0 --embedding-device cpu --codec-device cuda --codec-conv1d-pad-to-nc1d --frames 100 --output drama3-perf.json
```

Add `--sample` for the ONNX stochastic policy. The recorded real-time sampled
case requires the custom runtime above; stock ORT 1.28 falls back to CPU for
integer XOR in the RNG. The codec layout option is explicit and should be
measured on the deployment GPU rather than assumed universally faster.

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
