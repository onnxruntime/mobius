# mobius

[![PyPI](https://img.shields.io/pypi/v/mobius-onnx)](https://pypi.org/project/mobius-onnx/)
[![CI](https://github.com/onnxruntime/mobius/actions/workflows/main.yml/badge.svg)](https://github.com/onnxruntime/mobius/actions/workflows/main.yml)
[![L4: Golden Checkpoint Parity (GPU)](https://github.com/onnxruntime/mobius/actions/workflows/gpu_l4_golden_parity.yml/badge.svg)](https://github.com/onnxruntime/mobius/actions/workflows/gpu_l4_golden_parity.yml)
[![L5: End-to-End Generation (GPU)](https://github.com/onnxruntime/mobius/actions/workflows/gpu_l5_generation_e2e.yml/badge.svg)](https://github.com/onnxruntime/mobius/actions/workflows/gpu_l5_generation_e2e.yml)
[![Nightly L2 Architecture Validation](https://github.com/onnxruntime/mobius/actions/workflows/nightly_l2.yml/badge.svg)](https://github.com/onnxruntime/mobius/actions/workflows/nightly_l2.yml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

ONNX model definitions for GenAI using the `onnxscript.nn` API.

## Overview

This package provides model definitions for generative AI architectures — LLMs,
MoE, multimodal, encoder-only, encoder-decoder, vision, audio, and diffusion
models — built directly as ONNX graphs using `onnxscript.nn.Module`. Rather than
tracing or exporting PyTorch models, it **constructs** the ONNX graph
declaratively, then applies pretrained HuggingFace weights.

Supports building ONNX models from HuggingFace model IDs with automatic weight
downloading, dtype casting (including bfloat16 via `ir.LazyTensor`), and
multi-component export for pipelines.

📖 **[Documentation](https://onnxruntime.github.io/mobius/)** · 📦 **[Supported Models](https://onnxruntime.github.io/mobius/models/index.html)**

## Highlighted Models

| Category | Examples |
|---|---|
| **Text Generation** | Llama 2/3/4, Mistral, Qwen 2/2.5/3/3.5/3.6, Phi-3/3.5, Gemma 1/2/3/4, Granite, GPT-2, OPT, OLMo, SmolLM3, and many more |
| **Mixture of Experts** | PhiMoE, GPTOSS, Mixtral, OLMoE, DeepSeek-V2/V3, Qwen2-MoE, Qwen3-MoE, Qwen3-Next, GLM-4-MoE, Arctic, DBRX, Jamba |
| **Multimodal** | Gemma 3/4, Phi-4MM (vision + audio + LoRA), Nemotron Parse, LLaVA, InternVL2, Mage-VL (image + streaming video), MiniCPM-V 4.6, Qwen2.5-VL, Qwen3-VL, Qwen3.5/3.6-VL, Pixtral |
| **Encoder-only** | BERT, RoBERTa, ALBERT, DeBERTa, DistilBERT, ELECTRA, XLNet |
| **Encoder-Decoder** | BART, T5/mT5, Marian, M2M-100, Pegasus, BigBird-Pegasus |
| **Speech-to-Text** | Whisper, Moonshine, Moonshine Streaming, FastConformer-RNNT, FunASR, GLM-ASR, Qwen3-ASR, SenseVoice |
| **Audio** | Wav2Vec2, HuBERT, WavLM, SpeechT5 |
| **Vision** | ViT, BEiT, DeiT, DINOv2, Swin, CLIP, SigLIP |
| **Diffusion** | Stable Diffusion (UNet + VAE + ControlNet), Flux, SD3, DiT, QwenImage / Qwen-Image-Edit-2509, HunyuanDiT, CogVideoX |
| **Adapters** | T2I-Adapter, IP-Adapter |

Mage-VL supports direct three-model ONNX export. ORT GenAI export is currently
rejected because the runtime cannot supply its required `patch_positions` input
or Mage-VL's 1D decoder positions.

Supports **290+ Transformers model types** and **10 Diffusers component types**
across **40+ task types** and **100+ reusable components**.

See the [model documentation](https://onnxruntime.github.io/mobius/models/index.html) for the complete list.

## Projection-specific QMoE export

Olive's integer-affine, fused, K-last expert checkpoints can independently
use INT2, INT3, INT4, or INT8 for FC1 and FC2, with a supported model-wide
INT4 fallback for non-expert projections. All 16 bit-width pairs are accepted
when their layouts satisfy the checks below; no experimental flag is required. INT3
uses the proposed raw QMoE contract from `microsoft/onnxruntime#32657`.

```python
from mobius import build

package = build("./local-olive-qwen3-moe", dtype="f16")
package.save("./qmoe-export")
```

Use an Olive model-wide INT4 configuration with exact source-name overrides for
`model.layers.<i>.mlp.experts.gate_up_proj` and
`model.layers.<i>.mlp.experts.down_proj`. The generic MoE adapter, including Qwen3-MoE,
validates fused checkpoint dimensions and binds packed bytes without
requantization. INT3 is symmetric unsigned offset-4, row-local LSB-first
`uint8`; FC1 gate/up rows and scales are interleaved together, while FC2
retains its K-last down-projection layout. Explicit all-4 INT3 zero points are
validated and omitted in the graph.

Both projections require a shared power-of-two block size of at least 16,
dividing hidden and intermediate sizes. Pairs containing INT3 require block
size 32/64/128, symmetric INT3, and FP16/BF16 activations. For these pairs both
FC scale dtypes must match the activations. Every projection's scales must be
finite and nonnegative. When the adapter supplies the activation dtype,
scales must remain finite after conversion. Non-INT3 layouts retain the
standard weight loader's dtype conversion.
The legacy uniform INT4 path receives the same numeric scale checks.
INT2, INT4, and INT8 projections retain their existing zero-point
semantics; asymmetric zero points must be packed `uint8` sidecars.
Dense INT3 `MatMulNBits`, embeddings, other source layouts, and schema-unapproved
tails are not enabled. Keep non-expert projections at supported precisions.

`build()` defers INT3-layout FC1 interleaving to initializer serialization,
using uncached lazy tensors instead of retaining a second complete FC1 payload.
The original checkpoint is still loaded eagerly; this bounds additional
interleaving storage, not total checkpoint memory. Direct `preprocess_weights()`
calls retain eager behavior unless an output `weight_transforms` mapping is
provided. Pass that mapping to `ModelPackage.apply_weights()` with the returned
weights to preserve the deferred transformations.

**Export support is not a runtime qualification matrix.** INT3 QMoE nodes
carry a draft-format/runtime-warning marker; they do not invent an
approved schema version. Execution of every pair depends on the target ORT
version and execution provider. Tests cover bytes, interleaving, binding, and external
data save/reload, not INT3 kernel execution, model quality, or speed. Final
schema/version review, a complete quantization provenance manifest, and
checkpoint-to-ORT parity remain necessary before production use. This is not
completion of the runtime plan's P5 acceptance gate.

## Installation

```bash
pip install -e .
```

For running tests:

```bash
pip install -e ".[testing]"
```

## Quick Start

### Python API

```python
from mobius import build

# Build a model package with weights
pkg = build("meta-llama/Llama-3.2-1B")
pkg.save("output/llama-3.2-1b/")
```

**Static cache** (opt-in) pre-allocates fixed-size KV cache buffers, which is
useful when you know the maximum sequence length up front:

```python
from mobius import build, CausalLMTask

task = CausalLMTask(static_cache=True, max_seq_len=2048)
pkg = build("meta-llama/Llama-3.2-1B", task=task)
pkg.save("output/llama-3.2-1b-static/")
```

**EP-aware optimization** generates graphs tuned for a specific runtime execution
provider. Pass `execution_provider` to target CUDA, DirectML, WebGPU, and more —
each with the right set of fused kernels and lowering passes applied automatically:

```python
from mobius import build

# CUDA: GQA fusion, SkipLayerNorm, PackQKV
pkg = build("meta-llama/Llama-3.2-1B",
            execution_provider="cuda", dtype="f16")

# WebGPU: GQA fusion, Shape ops replaced with portable alternatives
pkg = build("meta-llama/Llama-3.2-1B",
            execution_provider="webgpu", dtype="f16")
```

See the [EP quickstart](docs/ep_quickstart.md) and
[full EP reference](docs/execution_providers.md) for all supported EPs and options.

### CLI

```sh
mobius build --model Qwen/Qwen2.5-0.5B --output output_dir/

# Build for CUDA with f16
mobius build --model meta-llama/Llama-3.2-1B --output output_dir/ --ep cuda --dtype f16

# Build a diffusers pipeline (all components)
mobius build --model Qwen/Qwen-Image-2512 --output output_dir/

# Build encoder-decoder model (produces encoder/model.onnx + decoder/model.onnx)
mobius build --model openai/whisper-tiny --output output_dir/
```

Build-mode toggles use the cargo-style `--features` option. Available features
are `static-cache`, `fp8-kv-cache`, `prune-prefill-prefix`, and `text-only`. Pass them
as a comma-separated list or repeat the option:

```sh
mobius build --model meta-llama/Llama-3.2-1B --output output_dir/ \
      --features static-cache,prune-prefill-prefix --max-seq-len 2048
```

Use `--release` with either `build` or `build-gguf` to potentially reduce saved model size
by stripping build-time debug and provenance metadata. Functional metadata with keys prefixed by
`mobius.` is preserved:

```sh
mobius build --model meta-llama/Llama-3.2-1B --output output_dir/ --release
mobius build-gguf model.gguf --output output_dir/ --release
```

See the [CLI Reference](https://onnxruntime.github.io/mobius/cli_reference.html) for all subcommands and flags.

### Examples

- [`examples/build_and_save.py`](examples/build_and_save.py) — Build and save ONNX models (simplest workflow)
- [`examples/text_generation.py`](examples/text_generation.py) — Greedy text generation with a causal LM
- [`examples/static_cache_generation.py`](examples/static_cache_generation.py) — Text generation with static KV cache
- [`examples/multimodal_generation.py`](examples/multimodal_generation.py) — Image captioning with a multimodal model
- [`examples/ornith_export.py`](examples/ornith_export.py) — Experimental Ornith-1.5-122B
  INT2 decoder, FP8 LM head/MTP/KV, and FP16 vision export. **Not GenAI Engine-ready.**

#### Experimental Ornith export

Download `ornith-ai/Ornith-1.5-122B` at revision
`7681ef608624e7532d8164128bcffdbdbaf1b1af` using Hugging Face Hub's
`snapshot_download(..., local_dir=...)`. Retain its local download metadata:
the exporter verifies the config, checkpoint index, and every weight shard
against the pinned revision before quantization. Install Mobius, Olive,
Transformers' image processing dependencies (Pillow and torchvision), and
CUDA ONNX Runtime in the same environment.

```sh
python examples/ornith_export.py \
  --checkpoint /path/to/Ornith-1.5-122B-hf \
  --output /path/to/Ornith-1.5-122B-int2-fp8
python examples/ornith_smoke.py \
  --checkpoint /path/to/Ornith-1.5-122B-hf \
  --model /path/to/Ornith-1.5-122B-int2-fp8 \
  --report /path/to/component-smoke.json
```

The exporter handles fused decoder expert banks and individual MTP expert
projections. Decoder expert and dense projections use symmetric INT2 RTN
with 128-element groups. The LM head and MTP projections use E4M3FN weights
with floating-point activations; MTP routed expert scales use 128x128 blocks.
Norms, routing, shared-expert gates, embeddings, convolution caches, and
recurrent state retain their appropriate floating-point precision.
FP8 KV caches currently use unit scales, not calibrated scales.

CUDA ONNX Runtime must be built with `onnxruntime_USE_FP8_QMOE=ON`
and FP8 KV support. A wheel with FP8 KV/linear support does not necessarily
include FP8 QMoE: inspect `onnxruntime.get_build_info()` for `fp8-qmoe=1`.
The runtime also needs the non-shared quantized KV-cache preservation fix
in `DequantizeFlashAttentionFallback`; otherwise prefill can succeed while
the first decode step reads uninitialized cache history.

The output includes decoder, vision encoder, embedding, and one-layer MoE
MTP graphs plus an `ornith_export.json` limitations manifest. The MTP sidecar
uses the existing **external** coordination contract, not native Engine
speculative decoding. The target exposes its post-final-norm state as
`mtp_seed`. GenAI Engine still needs packed/paged target and MTP graph I/O,
hybrid-state rollback wiring, and multimodal image admission/coordinates.
Do not treat this package or the component smoke check as an Engine
benchmark, end-to-end image generation, or an accuracy result.
`--diagnostics` exposes early-layer activations during smoke validation;
`--skip-mtp` explicitly limits the probe to decoder and vision components.
A successful full smoke check also records its report in the export manifest's
`component_validation` field without marking the package Engine-ready.

#### Ornith packed/paged text Engine export

The separate text-only Engine exporter reuses the validated component export's
INT2 and FP8 initializers, recovering unfused norms, routing weights, and the
target embedding table from the pinned checkpoint without requantizing the
122B model. It leaves the component and vision export unchanged.

```sh
python examples/ornith_engine_export.py \
  --checkpoint /path/to/Ornith-1.5-122B-hf \
  --components /path/to/Ornith-1.5-122B-int2-fp8 \
  --output /path/to/Ornith-1.5-122B-int2-fp8-engine
ORT_DISABLE_MOE_GEMV=1 ORT_ENABLE_XQA=0 ORT_PAGED_ATTENTION_DECODE_NUM_SPLITS=8 \
python examples/ornith_engine_benchmark.py \
  --model /path/to/Ornith-1.5-122B-int2-fp8-engine \
  --report /path/to/engine-benchmark.json
```

The packed target and MTP use rank-two hidden states, FP8 paged attention,
native variable-length convolution/GatedDeltaNet kernels, and native
`model.mtp` configuration. The hybrid target exposes compact convolution and
recurrent-state captures for speculative rollback; prefix caching is disabled.
The default benchmark uses batch one, greedy decoding, exactly
2048/8192/16384/32768 input tokens and 512 generated tokens, one warmup and
three measured runs per context. It rejects inactive or failed MTP execution
rather than reporting target-only timings. Before timing, it compares 64 greedy
output tokens against an MTP-disabled Engine using the same 128-token prompt.
A second parity check at each requested context compares the entire requested
output sequence before any timings are collected. A mismatch is saved in the
report, leaves `engine_validated` false, and stops
the benchmark. Target-only references are collected first, then that Engine and
model are released before loading the MTP model; validation does not require
two simultaneous model copies on the GPU.
This is a correctness gate, not an accuracy evaluation or a
guarantee of equivalence for all prompts. TTFT includes Engine admission,
prefill, and sampling; it is not an isolated prefill measurement.
Measured outputs must also match the corresponding target-only reference;
a repeatability failure stops timing without marking the export validated.

Use a GenAI build with per-verification-row EOS suppression for minimum
generated token counts. Older builds disable MTP until the minimum is reached,
so setting both output limits to 512 otherwise prevents drafting entirely.
The Engine runtime must load the same FP8-QMoE-enabled ONNX Runtime used for
component validation; the benchmark records the loaded native libraries.
`ornith_engine_export.json` initially marks execution unvalidated.
The reproducibility settings above select a consistent MoE path and portable
FP32 paged decode with a fixed split count instead of shape-dependent split-KV
reductions for single-token and short multi-token verification. They
require the ORT portable quantized speculative-attention dispatch extension.
Default fast kernel selection can amplify shape-dependent rounding into different
greedy outputs; the benchmark records both environment settings in its report.
Engine validation applies only to the recorded runtime and settings.

With the extended INT2 MoE GEMV dispatch (up to 64 expanded token/expert rows),
single-token decode and top-8 speculative verification can use the same fast
expert kernels instead of disabling GEMV. The following opt-in settings were
validated with 512-token greedy MTP/target-only parity at all four contexts:

```sh
ORT_DISABLE_MOE_GEMV=0 ORT_ENABLE_XQA=0 ORT_PAGED_ATTENTION_DECODE_NUM_SPLITS=32 \
python examples/ornith_engine_benchmark.py \
  --model /path/to/Ornith-1.5-122B-int2-fp8-engine \
  --cuda-graph --report-only \
  --report /path/to/engine-optimization.json
```

`--cuda-graph` enables native Engine graph capture for both the speculative
model and the target-only reference. The report records the applied overlay
separately from the on-disk configuration hash. `--report-only` preserves the
export's existing validation record while evaluating alternative settings.
On one H200, the optimized configuration measured approximately 102/97/57/31
decode tokens/s at 2K/8K/16K/32K, respectively (batch one, one warmup and three
measured runs, unchanged INT2/FP8 precision). This does not reach 200 tokens/s.
The wider GEMV path changes floating-point rounding relative to the
GEMV-disabled baseline, so its generated sequence can differ; its MTP output
must still exactly match its own target-only reference. Re-run the parity
gate after changing kernels, graph settings, or the fixed attention split count.

For fewer decode launches, add `--pack-projections` to the Engine export command.
After binding the existing quantized weights, this opt-in transformation packs
compatible shared-input INT2 projections along their output-channel dimension
and splits the results back to their original consumers. It removes 180
projection nodes from this decoder without requantization, changing FP8
components, or changing the recurrent-state ABI. Packing is limited to raw
group-128 INT2 weights without optional inputs or extra attributes; other
projections are left unchanged.

Combined with the portable attention cache-offset reuse optimization, a packed
export measured approximately 142/128/93/57 decode tokens/s at
2K/8K/16K/32K on one H200 with the same benchmark settings and full
MTP/target-only parity. Wider prefill matrix multiplies change floating-point
rounding, so the generated sequence and MTP acceptance can differ from the
unpacked export. These are measurements on the repeated-text benchmark, not
accuracy results or guaranteed performance on other prompts.

A subsequent run with the same decoder and runtime binary, using sequential
target-only/MTP model loading, measured 132/120/89/55 decode tokens/s.
All twelve measured 512-token outputs exactly matched the earlier run.
The report identifies this validation lifecycle as
`parity_reference_lifecycle: "sequential"`; the earlier measurements are retained
as historical results, not a guaranteed current throughput. Further expert
tiling, dense tensor-core dispatch, and attention-reduction experiments did not
establish a reliable improvement and were not retained. The 200 tokens/s goal
has not been reached.

### Ornith shared-expert gate fusion

`--fuse-shared-expert-gates` additionally replaces each FP16 shared-expert
`MatMul -> Sigmoid -> Mul -> Add -> Add` chain with the CUDA contrib operator
`com.microsoft::MatMulSigmoidGatedResidual`. Its inputs are activation `[M,K]`,
scalar-gate weight `[1,K]`, and shared/routed/residual tensors `[M,N]`. The
rewrite only accepts compatible rank-two shapes and single-use intermediate
values, preserving externally visible or reused values. In this decoder it
removes 48 of the 96 standard MatMuls, 192 elementwise nodes, and 48 unused
weight transposes, inserting 48 fused nodes (240 fewer nodes overall).
The 48 remaining MatMuls are expert-router projections. It does not replace
the INT2 `MatMulNBits` or FP8 LM-head projections.

This option requires the matching ORT CUDA kernel and is off by default.
The dot product accumulates in FP32; its result, sigmoid, multiply, and both
adds each round to FP16 at the original graph boundaries. The reduction order
can differ from cuBLAS, so output IDs need not match an unfused graph. Validate
the fused variant against its own target-only reference and keep its benchmark
report separate from the original export validation.

With packed projections, low-memory settings and CUDA graphs, the fused
decoder measured **171.4/144.5/114.1/57.2 decode tokens/s** at
2K/8K/16K/32K on H200. All twelve measured 512-token outputs exactly matched
the fused decoder's target-only references; MTP reported zero failures.
The 32K parity-gated memory run retained the **43.2 GiB** sampled peak.
A separate fresh 2K run measured 161.0 tokens/s with identical IDs, while an
unfused 2K control on the same runtime measured 132.3 tokens/s and retained
the old baseline IDs. These are workload-dependent results, not guaranteed
speedups: the changed sequence increased 2K MTP acceptance from 44.9% to
73.5%, so the end-to-end gain cannot all be attributed to kernel fusion.
The 200 tokens/s goal and model accuracy evaluation remain outstanding.

A same-input GPU-only microbenchmark isolated the fusion: for FP16
`K=N=3072`, unfused/fused latencies were 8.89/4.02 us at one row,
8.09/4.06 us at two rows, 10.80/4.09 us at eight rows, and 25.20/6.93 us at
1,024 rows. These are median CUDA-event timings over five samples, each
replaying a graph of 128 consecutive operator calls twenty times, with
ORT graph optimization disabled. They exclude Python launch gaps and do
not imply the same proportional speedup for the whole decoder.

### Ornith FP16 recurrent-state storage

Add `--recurrent-state-fp16` to the Engine export to store GatedDeltaNet recurrent
state in FP16 instead of the default FP32. This requires matching ORT and GenAI
builds. Query/key/value activations remain FP16; decay, beta, compact rollback
capsules, and recurrent update/output arithmetic remain FP32. Recurrent execution
rounds the state to FP16 after computing each token's output, including during
multi-token speculative verification. GenAI rollback uses identical per-token
rounding. Chunked prefill retains its existing internal arithmetic and rounds the
stored final state. The MTP layer in this export is full attention, so it has no
GatedDeltaNet recurrent state to convert.

The 36 recurrent tensors shrink from 144 MiB to 72 MiB per batch slot per copy.
This does **not** guarantee an equivalent decrease in total reserved GPU memory:
the parity-gated 32K Engine run peaked at **43.194 GiB**, only **6 MiB** below the
previous FP32-state run. The pool owns multiple copies and allocator reservations
can mask the reduced tensor storage.

The opt-in fused-gate/FP16-state decoder measured
**163.4/133.9/111.7/66.1 decode tokens/s** at 2K/8K/16K/32K on H200.
All twelve measured 512-token outputs exactly matched its own target-only
references, with zero MTP failures. The 2K individual timings were
153.7/168.0/168.4 tokens/s. Generated IDs differ from the FP32-state variant at
every tested context; changed MTP acceptance and runtime variability prevent
attributing these throughput changes solely to state precision.
The FP32 decoder and its validation reports remain unchanged.

FP16-state operator tests cover finite outputs through 32K tokens, agreement
with FP32 within synthetic-test tolerances, exact one-token/multi-token state
and output parity, compact replay, in-place operation, and recurrent/chunked/
split-chunked engines. These checks are not a model accuracy evaluation: FP16
state can accumulate error or overflow on other inputs. Long-context accuracy
evaluation remains outstanding.

### Ornith looped MTP

These benchmarks use the native GenAI `Engine`, not the Generator API. The Engine
can repeatedly invoke the one-layer FP8 MTP head, feeding each stage's predicted
token and hidden-state output into the next stage, then verify the proposed block
with the target. Use `--draft-tokens 2` or `--draft-tokens 4` on
`ornith_engine_benchmark.py`; the exported rollback capacity of eight permits at
most seven drafts. The default remains one draft.

A 2K sweep on the fused-gate/FP16-state decoder, with CUDA graphs, low-memory
settings, 512 output tokens, one warmup and three timed runs, measured:

| Draft width | Decode tokens/s | Target forward passes | MTP forward passes |
|---|---:|---:|---:|
| 1 | 167.4 | 302 | 300 |
| 2 | 164.2 | 228 | 450 |
| 4 | 152.8 | 175 | 691 |

Forward-pass counts are from each width's first timed turn and include prefill.
All nine timed outputs matched both their own target-only references and each
other exactly, using identical runtime fingerprints, with zero MTP failures.
Width two did not establish a speedup; width four was slower despite fewer target
passes because it required substantially more MTP work. Retain width one at 2K.
This does not establish the optimal width at longer contexts, where target
attention is more expensive; those require a separate parity-qualified sweep.

### Ornith single-Engine GPU memory

The CUDA runtime supports an opt-in packed-only INT2 expert mode. It releases
the raw expert initializer after runtime packing, because both Ornith prefill
and decode already consume the packed copy. No weights are requantized, no
precision is reduced, and FP8 MTP remains enabled. Direct initializer
allocations are also required so the freed copies leave the CUDA arena.
Both settings are opt-in; the original model configuration is unchanged.

```sh
ORT_QMOE_INT2_PACKED_ONLY=1 ORT_DISABLE_MOE_GEMV=0 \
ORT_ENABLE_XQA=0 ORT_PAGED_ATTENTION_DECODE_NUM_SPLITS=32 \
python examples/ornith_engine_memory.py \
  --model /path/to/Ornith-1.5-122B-int2-fp8-engine \
  --reference-report /path/to/engine-benchmark-packed-projections.json \
  --device-allocator-for-initializers \
  --report /path/to/engine-memory-low-memory-2k.json
```

The memory tool loads one speculative Engine, enables CUDA graphs, checks every
output against a previously parity-validated report for the same decoder, and
samples device-wide used memory every 10 ms through model loading, Engine
creation, one warmup and three measured 512-token runs. Use an otherwise idle
GPU. This is a **sampled peak**, including the Torch context, not an exact
allocator peak or an accuracy test. Add `--context 32768` for the 32K case.

On H200, a fresh default-arena 2K baseline peaked at **73.7 GiB**; packed-only
experts plus direct initializer allocations peaked at **43.2 GiB** at both
2K and 32K (about **30.5 GiB / 41% less**). All outputs matched the earlier
validated references exactly. The identical peaks reflect the unchanged fixed
KV-pool capacity and bounded prefill chunk size, not zero per-token KV storage.
Changing only the arena growth strategy did not deliver a comparable saving.
The low-memory full benchmark measured 141.7/127.8/93.6/56.5 decode tokens/s
at 2K/8K/16K/32K, with all twelve timed outputs exactly matching both the
target-only references and the previous packed baseline. Memory optimization
did not require disabling MTP or changing the output length.

For a fresh full MTP/target-only correctness-gated benchmark in this mode, use
the alternative-decoder benchmark command below with `ORT_QMOE_INT2_PACKED_ONLY=1`
and add `--device-allocator-for-initializers`. Keep `--report-only` to preserve the
original export validation. Unsupported packed-only expert invocations fail
explicitly rather than attempting to use freed raw weights; reload without
the option if a raw-weight fallback is needed.

An alternative decoder graph saved alongside an existing export can be tested
without changing its default configuration:

```sh
ORT_DISABLE_MOE_GEMV=0 ORT_ENABLE_XQA=0 ORT_PAGED_ATTENTION_DECODE_NUM_SPLITS=32 \
python examples/ornith_engine_benchmark.py \
  --model /path/to/Ornith-1.5-122B-int2-fp8-engine \
  --decoder-file decoder/model-packed.onnx --cuda-graph --report-only \
  --report /path/to/engine-benchmark-packed-projections.json
```

The report fingerprints the selected decoder graph and records the filename
overlay. `--decoder-file` must refer to an existing graph inside the model
directory. `--draft-tokens` overrides the fixed proposal width for tuning,
subject to the decoder's rollback capacity; it is applied to the speculative
model and recorded in the report while the reference still disables MTP.
Two draft tokens passed 2K parity but were slower than one in this workload,
so the export retains its default one-token proposal width.

This package does **not** support Engine image admission or image generation:
the original export's vision/fusion smoke is still a component-only test.
Neither benchmark performs an accuracy evaluation.

## Architecture

```
HuggingFace Hub
       │
       ▼
 ArchitectureConfig ◄── from_transformers() / from_diffusers()
       │
       ▼
 Model Module ◄── Reusable Components (Attention, MLP, RMSNorm, RoPE, …)
       │
       ▼
 Task ◄── CausalLMTask, VisionLanguageTask, VAETask, DenoisingTask, …
       │
       ▼
 ONNX Model ◄── preprocess_weights() + apply_weights()
```

The package is organised into four layers:

- **Components** — `onnxscript.nn.Module` building blocks (Attention, MLP,
  DecoderLayer, RoPE, VisionEncoder, MoELayer, …)
- **Models** — Full architectures composed from components
- **Tasks** — Define the ONNX graph I/O contract (inputs, outputs, KV cache)
- **Registry** — Maps HuggingFace `model_type` strings to model classes

See the [design document](https://onnxruntime.github.io/mobius/design.html) for details.

## Development

```bash
# Unit tests (fast, no network needed)
pytest tests/build_graph -v

# Integration tests (downloads models)
pytest tests/integration -m integration -v

# All unit tests (components, configs, tasks, models)
pytest src tests -m "not integration" -v

# Linting
lintrunner f --all-files
```

### Adding a new model

To [request a new model](https://github.com/onnxruntime/mobius/issues/new?template=model-request.yml),
or ask an AI coding agent to implement a new model, use the resources below:

See the [AI-assisted model support strategy](https://onnxruntime.github.io/mobius/ai-model-support-strategy.html)
and the developer skills in `.agents/skills/`:

| Skill | Use when |
|-------|----------|
| `adding-a-new-model` | Adding any new HuggingFace model architecture |
| `reusable-components` | Creating or extending components |
| `moe-models` | Adding a Mixture-of-Experts model |
| `multimodal-models` | Adding a vision-language model |
| `writing-tests` | Writing unit or integration tests |
| `writing-rewrite-rules` | Adding ONNX graph rewrite rules |
