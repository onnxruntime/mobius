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

### Ornith router projection integration

`--fuse-moe-routers` integrates the remaining FP16 router
`MatMul -> Reshape -> QMoE` chains into `com.microsoft::QMoEWithRouter`.
Combined with packed projections and shared-expert gate fusion, it removes all
96 standard MatMuls from the target decoder: 48 shared gates and 48 routers.
The router rewrite removes another 144 graph nodes, including unused weight
transposes. INT2 `MatMulNBits` and the FP8 LM-head operator remain.

This is opt-in and requires the matching ORT CUDA build. The guarded rewrite
requires the same rank-two FP16 activation for projection and expert evaluation,
constant router weights, compatible expert dimensions, and single-use
intermediates that are not graph outputs. It accepts only the symmetric,
group-128 INT2 SwiGLU configuration used here, without expert biases or additional
optional QMoE inputs.

The operator projects into internal FP16 logits, then reuses QMoE routing and
expert kernels. **It does not fuse projection and TopK into one CUDA launch.**
By default, for up to eight rows, a warp per expert accumulates the dot product in FP32 with
the same reduction order for every row count. Larger prefill batches use CUDA
MatMul internally. Thus zero standard ONNX MatMul nodes does not mean zero dense
matrix multiplication or zero cuBLAS calls.

Router reduction order can change logits, selected experts, generated IDs, and
MTP acceptance. Benchmark the alternate decoder against its own target-only
reference; do not assume equality with the unfused-router model or promote it
based only on node count. The original decoder/configuration remain unchanged.
No GenAI code change is needed for this operator integration.
The parity-gated 32K sampled peak remained **43.198 GiB**, effectively unchanged
from the previous 43.194 GiB run.

The experimental fused-gate/FP16-state/router decoder measured:

| Context | Router-integrated decode tokens/s | Prior FP16-state run |
|---|---:|---:|
| 2K | 147.1 | 163.4 |
| 8K | 135.8 | 133.9 |
| 16K | 104.5 | 111.7 |
| 32K | 62.9 | 66.1 |

All twelve timed 512-token outputs matched the new variant's target-only
references exactly, with zero MTP failures. IDs differ from the prior variant at
every context, so these are not fixed-output performance comparisons.
At 2K, accepted drafts fell from 211/300 to 176/334 and target forward passes
rose from 302 to 337. Removing the remaining standard MatMul nodes therefore
**did not establish an end-to-end speedup**; do not promote this variant over
the existing decoder. The 200 tokens/s goal and accuracy evaluation remain open.
A fresh 2K unfused-router control on the exact final runtime and same H200
measured **157.8 tokens/s**, preserving its prior output IDs. Router integration
was **6.8% slower** at 147.1 tokens/s; runtime fingerprints match between these
two reports.

A same-input synthetic component benchmark measured MatMul-plus-QMoE versus
QMoEWithRouter at **63.77/61.18 us** for one row, **67.49/64.81 us** for two,
**85.87/84.88 us** for eight, and **147.89/147.82 us** for 32.
It uses FP16, 256 experts, hidden size 3072, and intermediate size 128
(smaller than Ornith's 1024). Timings are median CUDA-event samples of twenty
native CUDA-graph replays, each containing 32 operator calls; graph optimization
is disabled. These modest local gains do not imply a whole-model speedup.

#### Preserving router MatMul numerics

Add `--moe-router-use-matmul` together with `--fuse-moe-routers` to set
`QMoEWithRouter.router_use_matmul=1`. This uses the existing CUDA MatMul
implementation inside the operator at every row count rather than the custom
decode GEMV. The default remains zero, preserving existing experimental graphs.
This mode still leaves zero standard ONNX MatMul nodes in the target decoder;
it does not eliminate the projection computation or combine projection and TopK.

The separate MatMul-router/FP16-state decoder measured
**168.0/133.4/111.6/66.0 decode tokens/s** at 2K/8K/16K/32K.
All twelve timed 512-token outputs matched both their own target-only references
and the prior unfused-router FP16-state decoder exactly, with unchanged MTP
counters and zero failures. Its 32K sampled peak was **43.196 GiB**.
This restores the original variant's tested numerics, not a source-model accuracy
guarantee.

A fresh same-runtime 2K control measured 162.5 tokens/s, with individual runs
157.7/162.2/167.7 versus 168.0/168.1/168.0 for the integrated decoder.
An earlier integrated run measured 157.7 tokens/s. Given this variability and
negligible same-input component differences, a reliable end-to-end speedup is
not established; neither router variant is promoted to the default.
The same-input synthetic component benchmark described above measured
baseline/integrated-MatMul latencies of 64.28/63.79, 67.46/67.48,
85.79/85.92, and 147.98/147.81 us at 1/2/8/32 rows.

#### Current-baseline CUDA profiling

Fresh Nsight Systems node-level traces of the faster unfused-router,
fused-gate/FP16-state decoder show these steady-decode GPU kernel-time shares:

| Family | 2K | 32K |
|---|---:|---:|
| Dense INT2 projections | 22.45% | 10.54% |
| INT2 expert FC1 + SwiGLU | 12.00% | 5.63% |
| INT2 expert FC2 | 6.02% | 2.83% |
| FP8 paged attention, including reduction | 10.79% | 58.18% |
| Normalization + casts + Split | 13.58% | 6.38% |
| Expert routing / finalize | 5.16% | 2.46% |

Each trace followed a complete 512-token warmup and captured prefill through at
least 128 generated tokens. Both full 512-token turns at both contexts exactly
matched the prior parity-qualified baseline; CUDA graphs, one-draft FP8 MTP,
packed-only INT2 and direct initializer allocation remained enabled.
These percentages describe summed kernel duration, not wall time, occupancy,
or measured hardware bandwidth. Node-level tracing adds overhead.
Prioritize INT2 dense/expert work for short-context decode and paged attention
for long-context decode, rather than more router-wrapper work. The 200 tokens/s
goal and source-model accuracy evaluation remain outstanding.

### Ornith dense INT2 reduction scheduling

The CUDA FP16 symmetric group-128 INT2 small-row kernel now unrolls its
reduction loop to overlap loads and dequantization instructions. Two iterations
are unrolled normally; eight are used for long reductions (`K>=8192`) with
`N<=4096` and at most four rows. Eight-row tiles retain the smaller unroll factor
to limit register pressure. This applies only to the two-column warp tile;
other types, group sizes, explicit zero points, single-row and large-prefill
dispatches retain their existing paths.

No graph rewrite, weight conversion or precision change is involved.
FP16 dequantization, half2 fused multiply-add order, FP32 warp reduction and
output rounding are preserved. Exact tests compare small-batch outputs with the
separate single-row kernel, including all five Ornith projection shapes, odd
row counts, partial reduction iterations, dispatch boundaries, other group
sizes and explicit zero points.

Using the existing fused-gate/FP16-state decoder (without integrated routers),
the optimized runtime measured **174.1/137.5/112.9/67.1 decode tokens/s** at
2K/8K/16K/32K. All twelve timed 512-token outputs and MTP counters exactly match
the prior FP16-state baseline, with zero MTP failures. A separate optimized 2K
run measured 173.8 tokens/s, versus a fresh original-kernel control at
158.2 tokens/s on the same H200 and unchanged graph/settings (about 9.8% higher
throughput). Historical baseline runs vary; this is a measured workload-specific
gain, not a guarantee for other models or inputs.
The parity-gated 32K sampled peak remained **43.196 GiB**, effectively unchanged.

Same-input CUDA-event measurements, cycling 128 copies of the weights to exceed
L2 capacity, reduced the two-row `K=8192,N=3072` projection from **15.85 to
11.99 us**. The two large `K=3072` packed projections fell from
**19.57 to 17.59 us** (`N=20608`) and **18.24 to 16.09 us** (`N=17408`).
Across all five model shapes and 2/4/8 rows, the selected dispatch reduced
component latency by 5.0–24.3%, with bitwise-identical outputs.

A fresh correctness-qualified 2K CUDA trace confirms unchanged dense INT2 launch
count (15,360) and reduces summed dense INT2 time from **179.33 to 154.38 ms**
(13.9%) over the same captured decode workload. Summed total kernel time falls
from 798.89 to 774.16 ms; trace kernel shares and ordinary Engine throughput
are different metrics and should not be equated. The 200 tokens/s goal and
source-model accuracy validation remain outstanding.

### Ornith INT2 expert FC1/SwiGLU scheduling

The bias-free FP16 group-128 INT2 FC1/SwiGLU kernel has a targeted scheduling
path for Ornith's `K=3072`, intermediate size 1024 experts. It unrolls two
reduction iterations. With more than eight expanded token/expert rows, it also
issues the four column-weight loads before dequantization to compensate for
register pressure. Simply unrolling every grid was rejected: it helped one
token but slowed larger expanded batches in the component sweep.
Expert layouts, FP32 dot accumulation order, reduction, FP16 rounding, SwiGLU,
FC2, routing, precision and exported graph are unchanged.

Same-input CUDA-event measurements with sixteen separate 256-expert weight
banks (exceeding L2 capacity) measured baseline/optimized FC1 latencies of
**15.98/13.52 us** at eight expanded rows, **24.18/23.11 us** at sixteen,
**46.06/44.37 us** at 32 and **86.62/83.19 us** at 64.
Every tested scheduling variant's output was bitwise equal to the original
arithmetic. These are deterministic synthetic inputs with distinct selected
experts, not end-to-end model timings.

A parity-qualified 2K CUDA trace, with the previous dense INT2 optimization
already enabled in both variants, preserves all 3,840 expert FC1 launches and
reduces summed FC1 time from **96.05 to 89.08 ms** (7.3%). Total summed kernel
time changes from 774.16 to 767.87 ms. Node-level tracing adds overhead; these
durations are not normal Engine latency.

The matched Engine comparison uses the same graph, settings and runtime
libraries except for the CUDA provider, with one warmup and three measured
512-token runs per context:

| Context | Dense-optimized control (tokens/s) | Expert scheduling (tokens/s) | Change |
| --- | ---: | ---: | ---: |
| 2K | 162.700 | 163.986 | +0.79% |
| 8K | 130.217 | 130.993 | +0.60% |
| 16K | 109.310 | 109.809 | +0.46% |
| 32K | 65.019 | 65.287 | +0.41% |

All twelve timed outputs and MTP counters exactly match the control and the
prior FP16-state decoder, with zero MTP failures. These small whole-model gains
are sensitive to timing variation. An additional 2K run
measured 164.4 tokens/s versus a fresh dense-optimized control at 162.9 tokens/s
(about 0.9% higher). The earlier 174.1 tokens/s dense-only run was measured at a
different time and is not a contemporaneous control; the expert scheduling
change does not establish a new best whole-model throughput.
The parity-gated 32K sampled peak remains **43.196 GiB**.
The 200 tokens/s goal and source-model accuracy evaluation remain outstanding.

### Ornith shared-expert INT2 projection/SwiGLU fusion

`--pack-projections --fuse-shared-expert-swiglu` replaces compatible packed
`MatMulNBits -> Split -> Swish -> Mul` chains (or explicit
`QuickGelu(alpha=1)` instead of Swish) with
`com.microsoft::MatMulNBitsSwiGLU`. This is separate from the existing scalar
gate/residual fusion: it computes the shared expert's gate/up projection and
activation, leaving the down projection and residual path unchanged.

The opt-in rewrite accepts only rank-two FP16 activations, raw symmetric INT2
weights with group size 128, equal gate/up halves and Swish alpha exactly one.
It rejects optional projection inputs/attributes, dynamic weights, incompatible
shapes, and intermediate values exposed as graph outputs or used elsewhere.
For Ornith it replaces 48 four-node chains with 48 fused nodes (144 fewer
graph nodes), without changing weight bytes or the recurrent-state ABI.

For one through eight rows, the CUDA kernel pairs corresponding gate/up
columns, reuses activation loads, and writes the 1024-wide activated result
directly instead of materializing the 2048-wide projection and elementwise
intermediates. FP16 dequantization, half2 dot accumulation order, FP32 warp
reduction and projection rounding are retained. The activation matches ORT's
optimized `QuickGelu(alpha=1)` path: sigmoid/product arithmetic is FP32, the
activated gate rounds to FP16, and the final up multiply rounds to FP16.
This intentionally matches the deployed optimized decoder, not an unoptimized
Swish expansion that rounds sigmoid separately. The flag therefore requires
validation against the optimized unfused graph, not just an isolated Swish.
Larger prefill batches retain dequantize/GEMM and use a fused activation
epilogue. The matching ORT CUDA runtime is required; this option is off by
default and does not change MTP weights or GenAI Engine scheduling.

The same-runtime comparison on one H200, with native Engine CUDA graphs,
packed-only INT2 experts, FP16 recurrent state, one warmup and three measured
512-token runs per context, measured:

| Context | Unfused control (tokens/s) | Projection/SwiGLU fusion (tokens/s) | Change |
| --- | ---: | ---: | ---: |
| 2K | 164.185 | 167.915 | +2.27% |
| 8K | 131.153 | 133.532 | +1.81% |
| 16K | 109.894 | 111.513 | +1.47% |
| 32K | 65.323 | 66.008 | +1.05% |

All twelve timed output sequences and MTP counters exactly match the unfused
control and the prior FP16-state decoder, with zero MTP failures. This is a
modest matched improvement, not a new best throughput across historical runs.
The initial implementation's separately rounded sigmoid passed isolated Swish
tests but changed full-model IDs; that arithmetic was rejected. Connected
SiLU/QuickGelu optimization, bitwise FP16 results, empty inputs, small-M row
invariance, large-M GEMM prefill and CUDA-graph replay now have regression
coverage. The 200 tokens/s goal and source-model accuracy evaluation remain
outstanding.

In same-input ORT CUDA graphs with 64 independent INT2 weight banks exceeding
L2 capacity, baseline/fused median CUDA-event times were **8.77/5.22 us** at
one row, **11.76/6.82 us** at two rows, **15.94/10.34 us** at four rows and
**22.55/17.31 us** at eight rows. These use the correct QuickGelu baseline and
bitwise-equal deterministic synthetic inputs, not whole-model timings.

A same-runtime 2K Nsight Systems comparison preserves all 3,840 shared
projection launches, replacing them with fused projection/activation kernels.
It removes 3,840 Split, 3,840 FP16 QuickGelu and 3,840 multiply launches:
**155,747 -> 144,227** total kernels in the synchronized steady-decode window.
Summed kernel time falls from **768.01 to 751.55 ms** (2.14%). Both complete
512-token profiled outputs match the prior baseline. Node-level tracing adds
overhead; these are summed GPU durations, not normal Engine wall latency.
The parity-gated 32K sampled device-memory peak is **43.194 GiB**, essentially
unchanged. The exact tested runtime is packaged in the matching CUDA wheel.

### Ornith portable paged-attention reduction scheduling

A refreshed 32K Nsight Systems capture on the shared-expert projection/SwiGLU
decoder attributes **59.88%** of summed steady-decode GPU kernel time to portable
paged attention. This is a kernel-duration share, not measured memory bandwidth
or an unprofiled wall-time fraction.

The CUDA paged-decode kernel now performs each tile's softmax maximum and sum
with three block barriers instead of ten. The first warp combines shared
values using the original 128/64/32 pairing, then finishes the 16/8/4/2/1
stages with shuffle operations. Unlike a differently ordered warp reduction,
this retains the original FP32 addition tree. It removes fourteen block
barriers per nonempty KV tile without changing QK/PV arithmetic, cache format,
split count, model graph, precision, or Engine scheduling. No new flag is
required; the benchmark still uses fixed split count 32 and one-draft FP8 MTP.

Regression coverage includes partial/full tiles, long-context speculative
queries, per-tensor/per-channel FP8 scales, sliding windows, softcap, head
sinks, query-shape invariance, and existing FP16/INT8/INT4 decode behavior.
An earlier four-value FP8 cache-load scheduling experiment preserved bits but
generally regressed component timings and was discarded.

Matched H200 native Engine measurements, with one warmup and three timed
512-token turns per context, measured:

| Context | Control (tokens/s) | Ordered reduction (tokens/s) | Change |
| --- | ---: | ---: | ---: |
| 2K | 167.950 | 168.455 | +0.30% |
| 8K | 133.496 | 135.000 | +1.13% |
| 16K | 111.522 | 113.467 | +1.74% |
| 32K | 66.016 | 67.590 | +2.38% |

All twelve timed outputs and MTP counters match the control and prior fused
decoder exactly, with zero MTP failures. A candidate run before the control
also measured 67.557 tokens/s at 32K, versus 67.590 in the final candidate
run after the control. Sixteen synthetic FP8 component cases (1/2/4/8 query
rows at 2K/8K/16K/32K past lengths) are bitwise equal to the original kernel.
Component timings use warm-cache ORT CUDA graphs, not a cold-cache bandwidth
measurement.

The matched 32K profiles retain **173,121** kernel launches in the synchronized
steady-decode window. Summed attention-kernel time falls from **1188.09 to
1139.66 ms** (4.08%); total summed kernel time falls from **1984.17 to
1935.91 ms** (2.43%). Complete warmup/profiled outputs and MTP counters are
unchanged. Node-level tracing adds overhead; these sums are not ordinary
Engine wall latency. The sampled 32K memory peak is **43.196 GiB**, essentially
unchanged from 43.194 GiB. The exact tested runtime is packaged in the matching
CUDA wheel. Attention remains the largest measured long-context bottleneck;
the 200 tokens/s goal and source-model accuracy evaluation remain outstanding.

#### Rejected grouped-query KV-read reuse experiment

A later experiment shared each FP8 K/V read and conversion across two query
heads within a CTA, retaining the original per-head FP32 arithmetic and fixed
split count 32. A second version also shared the block barriers between the
two heads' softmax reductions. Neither improved native Engine throughput:

| Context | Fresh control (tokens/s) | Paired heads (tokens/s) | Paired heads, batched reductions (tokens/s) |
| --- | ---: | ---: | ---: |
| 2K | 168.196 | 168.248 | 165.879 |
| 8K | 134.736 | 133.923 | 128.735 |
| 16K | 113.221 | 111.985 | 105.178 |
| 32K | 67.548 | 66.363 | 60.631 |

These are sequential same-settings H200 runs with one warmup and three timed
512-token turns per context, one-draft MTP, FP8 KV/LM head/MTP, INT2 target,
FP16 state, CUDA graphs, packed-only experts and direct initializer allocations.
All 24 timed prototype outputs and MTP counters match the control and prior
decoder exactly, with zero MTP failures. Four-head and 128-thread variants
were evaluated only as synthetic components, not full-model benchmarks.
Fewer redundant KV loads therefore did not establish a speedup; neither
prototype is enabled or retained in the runtime. The previously validated
ordered-reduction runtime and matching wheel are restored.

The paired-head 32K profile confirms grid-X halves from 32 to 16 while the
number of attention kernel launches is unchanged. It does not establish
reduced physical HBM traffic, improved occupancy or memory-bandwidth
saturation. Original component timings included per-run provider
synchronization and possible host submission gaps; they are not evidence
of isolated GPU latency gains. Subsequent component checks disable that
synchronization and use explicit stream/event completion.

Additional regression tests remain for FP8 head-256 grouped attention,
including bitwise equality to an equivalent expanded-MHA cache with ragged
queries, fragmented/unmapped pages, per-channel scales, sliding windows,
softcap and head sinks. The restored runtime passes 100 focused CUDA tests
plus 20 INT4 subtests. Source-model accuracy remains unevaluated.

### Ornith temporal convolution prefill

The variable-length causal convolution optimization from
[ORT PR #33381](https://github.com/microsoft/onnxruntime/pull/33381) is selectively
ported into the Ornith runtime, without the PR's allocator or small-N GEMV
changes. Eligible single-request prefill uses 32-token temporal tiles instead
of looping over the entire request in each channel block. A separate final-state
kernel runs after all tiles have consumed the old state, retaining safe
initial/final-state aliasing.

The specialization requires at least 256 tokens, kernel size four, dilation
one or three, a device supporting at least 256 threads per block, and a temporal
grid within CUDA's grid-Y limit. Other geometries retain the existing dispatch.
The deployed 1024-token Engine chunks qualify; ordinary one/two-token decode
does not. No model graph, export option, quantization, or Engine scheduling
change is required.

Matched H200 Engine measurements use the existing shared-SwiGLU/FP16-state
decoder, INT2 target weights, FP8 KV/LM head/MTP, width-one MTP, CUDA graphs,
packed-only experts, and direct initializer allocations. Each context has
one warmup and three timed 512-token turns:

| Context | Control TTFT (ms) | Temporal convolution TTFT (ms) | TTFT reduction | Control / candidate decode (tokens/s) |
| --- | ---: | ---: | ---: | ---: |
| 2K | 278.6 | 223.5 | 19.77% | 168.571 / 168.635 |
| 8K | 1117.0 | 894.5 | 19.92% | 134.975 / 135.086 |
| 16K | 2298.4 | 1854.6 | 19.31% | 113.425 / 113.392 |
| 32K | 4898.5 | 4015.8 | 18.02% | 67.633 / 67.652 |

TTFT includes Engine admission, prompt processing and first-token sampling,
not isolated convolution latency. Decode changes range from -0.03% to +0.08%;
this is a prefill optimization, not an established decode throughput gain.
All twelve timed outputs and MTP counters exactly match the fresh control and
previous approved decoder, with zero MTP failures. The candidate also passes
its own target-only greedy checks at every context.

A parity-checked 32K Nsight Systems capture reduces summed prefill convolution
kernel duration from **962.25 to 77.26 ms** (91.97%), including the separate
state-update kernel. The synchronized prefill/first-token range decreases from
4906.65 to 4031.47 ms. Node tracing adds overhead, and these kernel sums are not
measurements of bandwidth or occupancy.

The added Python regression suite passes two tests with 102 subcases:
bitwise equality to the general kernel for FP32/FP16/BF16, dispatch boundaries,
capture-count clamping, optional bias/capture, aliased state and subsequent
continuation. Sixteen existing parity tests plus eight subcases also pass.
Three native regression tests are ported from the PR but were not executed
because native test targets are disabled in this runtime build.

The exact tested provider is installed, staged and packaged in the matching
CUDA wheel; binary hashes are verified against the saved runtime.
`engine-conv-temporal-summary.json` records benchmark, profile and package
provenance alongside the export artifacts. The previously approved wheel is
retained for rollback. Source-model accuracy remains unevaluated, and portable
paged attention remains the main long-context decode bottleneck.

### Ornith small-row MoE finalization decode

The staged finalization optimization from
[ORT PR #33473](https://github.com/microsoft/onnxruntime/pull/33473) is selectively
adapted to the Ornith branch's existing launch-capability and programmatic
dependent-launch handling. For one/two-row staged finalization, one warp owns
each column tile instead of assigning all output columns to one CTA per row.
For hidden size 3072 and FP16 output, the main grid changes from `[2,1,1]`
with 384 threads per block to `[2,12,1]` with 32 threads per block.

Expert selection, routing weights, per-output top-k accumulation order,
quantization, shared-expert fusion, and Engine scheduling remain unchanged.
Larger-row and other finalization modes retain existing dispatch. The routing
fusion, FP8 projection, BF16 shared-expert and NVFP4 changes from the PR are
not included.

Matched H200 native Engine measurements use the approved temporal-convolution
runtime as control, the same model graph, width-one FP8 MTP, INT2 target,
FP8 KV/LM head, FP16 state, CUDA graphs, packed-only experts and direct
initializer allocations. Each context has one warmup and three timed
512-token turns; the candidate is followed by a fresh control:

| Context | Control decode (tokens/s) | Tiled finalization decode (tokens/s) | Change |
| --- | ---: | ---: | ---: |
| 2K | 168.736 | 180.744 | +7.12% |
| 8K | 135.110 | 142.810 | +5.70% |
| 16K | 113.484 | 118.642 | +4.55% |
| 32K | 67.677 | 69.911 | +3.30% |

All twelve timed outputs and MTP counters exactly match both the fresh
control and previous approved runtime. The candidate passes its own
target-only greedy checks, with zero MTP failures. Thirty-seven focused
CUDA tests plus 36 subcases pass, including top-eight INT2 at one/two/three/eight
rows, hidden sizes 512/3072, group sizes 64/128, packed-only expert storage,
FP8 FP16/BF16 paths, biases, all-expert selection and CUDA graph replay.

The parity-checked 32K node-level CUDA profile retains 173121 steady-decode
kernel launches. Summed finalization-kernel duration falls from 20.07 to
16.09 ms, and the observed grids match the intended specialization.
The change in summed GPU kernel time is much smaller than the unprofiled
throughput gain: tracing overhead and graph scheduling prevent attributing
the end-to-end gain solely to this sum. Hardware-counter access remains
unavailable, so no bandwidth, achieved-occupancy or stall diagnosis is claimed.
Portable paged attention still accounts for about 59% of summed decode GPU
kernel time at 32K.

A preceding head-256 register-accumulator attention experiment passed all
sixteen bitwise component cases but regressed long-context asynchronous
warm-cache timings by about 0.2-0.8%. It was rejected before full-model
benchmarking; no attention-kernel changes from it remain. The accepted
runtime retains the temporal-convolution prefill optimization.

The installed provider, staged binaries, saved runtime and wheel are
hash-verified. `engine-decode-priorities-summary.json` alongside the export
artifacts records the accepted/rejected evidence and package provenance.
The previous temporal-convolution runtime remains available for rollback.
Source-model accuracy is still unevaluated, and 200 decode tokens/s has not
been reached.

### Ornith exact gated RMS normalization fusion

The Engine exporter accepts opt-in `--fuse-gated-rmsnorm` to fuse the 36
GatedDeltaNet output norms. Each compatible FP16-to-FP32 cast, FP32
RMSNormalization, FP32 gate cast/SiLU, multiply and FP16 output cast becomes
one `com.microsoft.GatedRMSNormExact` node. The existing surrounding reshapes
and scale tensor are retained, and no weights are requantized.

This CUDA-only operator deliberately has a narrow contract: FP16 input and
gate of shape `[rows,128]`, FP32 scale `[128]`, and finite nonnegative epsilon.
It reproduces the width-128 single-warp FP32 RMSNormalization reduction,
separately rounded FP32 standardization and scaling, and optimized
QuickGelu(alpha=1) arithmetic including its negative tail. There is only one
FP16 conversion at the output. The pre-existing generic `GatedRMSNorm` uses
a different reduction order and is not silently substituted.

The rewrite checks types, shapes, normalization/activation attributes and
exclusive intermediate users; incompatible or shared chains are left intact.
It supports exported Swish and already optimized QuickGelu, is idempotent,
and records the number fused as `gated_rmsnorms_fused` in export provenance.
It requires the matching ORT branch; older runtimes cannot load this operator.
FP8 KV cache, INT2 projections, FP8 LM head/MTP, recurrent state and attention
behavior are unchanged.

Fresh matched H200 Engine measurements use the same rebuilt runtime for both
the original and fused decoder graphs, width-one FP8 MTP, CUDA graphs,
packed-only INT2 experts and direct initializer allocations. Each context has
one warmup and three timed 512-token turns:

| Context | Original graph decode (tokens/s) | Fused graph decode (tokens/s) | Change |
| --- | ---: | ---: | ---: |
| 2K | 169.330 | 184.193 | +8.78% |
| 8K | 135.373 | 145.049 | +7.15% |
| 16K | 113.571 | 120.032 | +5.69% |
| 32K | 67.756 | 70.503 | +4.05% |

TTFT improves by about 2.8-3.2%. All twelve output-token sequences and MTP
counters match both the matched control and previous approved model, with zero
MTP failures. The fused graph also passes its own target-only greedy checks.
Thirty-one CUDA tests cover bitwise FP16 outputs against the actual unfused
FP32 RMSNormalization/QuickGelu graph, zero rows, extreme finite inputs,
zero epsilon, prefill up to 65536 rows, CUDA graph replay with changing inputs,
and invalid contracts. All 108 exporter/ABI tests pass, including positive,
negative and idempotent rewrite cases.

The isolated decoder variant is
`decoder/model-fused-shared-swiglu-fp16-state-gated-rms.onnx`; select it with
the benchmark's `--decoder-file` option. It removes 180 graph nodes
(2325 to 2145), preserving every remaining initializer byte and external-data
reference. The previous approved decoder remains unchanged. These gains use
fresh matched controls and must not be added cumulatively to older benchmark
tables. Source-model accuracy remains unevaluated.

Matched, parity-checked 32K CUDA node traces confirm exactly five launches
removed for every fused norm. Prefill launches fall from 68739 to 62979
(-5760); captured steady-decode launches fall from 173121 to 155841 (-17280).
Summed steady-decode GPU kernel time falls from 1932.67 to 1916.66 ms.
These sums are not wall-time or hardware-counter measurements. The installed,
staged, saved and wheel binaries are hash-verified; `engine-gated-rms-summary.json`
records the benchmark, test, trace and package provenance. The previous runtime
is retained for rollback.

### Ornith remaining exact graph fusions

The following additional Engine export flags can be combined with
`--fuse-gated-rmsnorm`. Each is opt-in and requires the matching ORT branch:

| Flag | Target chains | Implementation |
| --- | ---: | --- |
| `--fuse-block-rmsnorm` | 97 | `RMSNormFp32`: FP16 storage, FP32 RMSNorm/scale arithmetic, one final FP16 conversion |
| `--fuse-gdn-inputs` | 36 | Pass packed convolution QKV and FP16 decay/beta directly to GatedDeltaNet |
| `--fuse-qk-norm-rotary` | 24 | `RMSNormRotaryFp32`: FP32 norm, FP16 norm output rounding, then partial FP16 rotary |
| `--fuse-attention-output-gates` | 12 | `SigmoidMul`: sigmoid rounded to FP16 before the FP16 multiply |

Block and Q/K norm fusions reuse the core CUDA RMSNormalization reduction and
launch heuristic rather than substituting a different reduction. Block norm
is deliberately limited to `[tokens,3072]`; Q/K norm+rotary accepts
`[tokens,2|32,256]`, FP32 scale `[256]`, pre-gathered FP16 cos/sin
`[1,tokens,32]`, and noninterleaved rotary dimension 64.

Packed GDN input executes directly from the token-major Q/K/V layout with
explicit input row strides. It avoids the old internal unpack copies as well
as exported Split/Reshape and gate Cast nodes. Decay/beta can now be FP16 or
FP32; activation, normalization, recurrent arithmetic, FP16 state rounding,
capture factors and rollback semantics are unchanged. All recurrent,
chunked and chunked-split input paths use the same strided loads.

The attention gate fusion combines sigmoid/multiply in one epilogue kernel;
it does not fuse the gate into attention softmax or value accumulation, and
does not change any attention backend or FP8 cache setting. Residual Add
nodes are intentionally retained. No residual-add/RMSNorm fusion is enabled
by these flags.

Rewrites require compatible types, shapes, attributes and exclusive
intermediate users, preserve external initializer bytes, and are idempotent.
The export manifest records `block_rmsnorms_fused`, `gdn_inputs_fused`,
`qk_norm_rotary_fused` and `attention_output_gates_fused`. Combining all four
with the previously accepted gated-RMS graph removes another 518 graph
nodes, from 2145 to 1627, without changing the model ABI or weight precision.

The combined graph passes 116 CUDA component tests (one empty-graph capture
case is skipped), 136 exporter/ABI tests, and all four-context target-only
and MTP greedy checks. All twelve timed 512-output-token sequences and MTP
counters exactly match both the approved gated-RMS model and the fresh
same-runtime control. The component suite includes FP16 state, capture factors,
recurrent/chunked/chunked-split paths, graph replay, finite extremes and strict
rewrite rejection cases.

However, fewer nodes did **not** establish a general decode speedup. A final
reverse-order control, with no concurrent GPU tests, gives:

| Context | Approved graph on new runtime (tokens/s) | Combined graph (tokens/s) | Change | Combined sampled peak (GiB) |
| --- | ---: | ---: | ---: | ---: |
| 2K | 174.470 | 174.883 | +0.24% | 43.265 |
| 8K | 140.380 | 139.124 | -0.90% | 43.265 |
| 16K | 117.049 | 116.157 | -0.76% | 43.265 |
| 32K | 69.121 | 70.187 | +1.54% | 43.265 |

These are three timed turns after one warmup, with width-one FP8 MTP,
CUDA graphs, packed-only INT2 experts and direct initializer allocations.
TTFT improves about 1.2-1.5%. Memory is sampled every 10 ms in separate
processes through loading, warmup and generation; it includes CUDA context
allocations and is not an exact allocator peak. Shared-host timing drift is
visible relative to historical tables, so only the fresh matched controls
support the comparisons above.

Parity-checked 2K node-level CUDA traces confirm that steady-decode launches
fall from 129827 to 98947 (-30880), and summed GPU kernel time falls from
759.89 to 734.81 ms. However, summed normalization-kernel time rises from
45.39 to 57.24 ms, offsetting part of the cast/elementwise savings. These
trace sums are not wall-time or hardware-counter measurements, and fewer
launches alone do not establish end-to-end throughput gains. Separate
component microtimings were rejected because host submission gaps or stream
capture limitations made them unsuitable for performance claims.

**The combined graph/runtime is experimental and is not promoted as the
recommended deployment.** The previously accepted gated-RMS decoder/runtime
remain the recommendation. `decoder/model-fused-all-exact.onnx` and a
separately saved matching wheel preserve the experiment for further tuning;
older runtimes cannot load the new contrib operators. Default Engine config
is unchanged. `engine-all-fusions-summary.json` records exact parity,
benchmarks, memory, package identity and the non-promotion decision.

### Ornith FP8 DFlash2 experiment

[`incoai/Ornith-1.5-122B-DFlash2`](https://huggingface.co/incoai/Ornith-1.5-122B-DFlash2)
is an Ornith-trained block drafter, separate from the model's built-in MTP.
`examples/ornith_dflash2_export.py` reuses the native GenAI DFlash2 graph builder
and Engine scheduler. Add the matching GenAI checkout's `src/python/py` to
`PYTHONPATH`; no SGLang installation is needed.

```bash
export PYTHONPATH=/path/to/onnxruntime-genai/src/python/py
python examples/ornith_dflash2_export.py --model /path/to/engine-export
python examples/ornith_engine_benchmark.py --model /path/to/engine-export \
  --config-overlay /path/to/engine-export/dflash2-fp8-config.json \
  --contexts 2048 8192 16384 32768 --output-tokens 512 --warmup 1 --runs 3 \
  --cuda-graph --device-allocator-for-initializers --report-only \
  --report /path/to/engine-export/engine-benchmark-dflash2-fp8.json
```

The export pins draft revision `9baefc1bec85c08193ab442e0ca5f7057669b61d`.
Body projections use E4M3FN FP8 weights with FP32 row-block-128 scales
(`MatMulBlockQuantizedFp8Weight`), while activations and the drafter's own KV
cache stay BF16. Dynamic-convolution coefficients, norms and candidate-selector
parameters remain unquantized. The LM head uses the exact already-exported target
FP8 weights/scales, not a separately quantized approximation. The supplied
`mask_embedding.safetensors` row is restored **after** casting the target's FP16
embedding lookup to BF16, so tiny mask values are not lost to FP16 underflow.
This is weight-only FP8, not FP8 activations or an FP8 draft cache.

The checkpoint uses 16 query rows, but the current target has an eight-token
recurrent rollback capacity. This initial experiment explicitly uses eight
query rows (anchor + seven drafts); it is **not** a benchmark of the published
16-row configuration. `--num-draft-tokens` can select 1–15 for the draft graph.
Verification is separately capped at `state_update_capacity - 1`, so a
16-row draft can be tested without increasing the target's rollback capacity:
the Engine only verifies its first seven proposals. Non-default block sizes
use distinct graph/config filenames. The draft's eight
post-residual target taps are layers 1/7/14/20/26/32/39/45 (zero-based).
No target weights or arithmetic are changed by exporting these taps.

The exporter writes a separate decoder graph, drafter and config overlay.
It refuses to overwrite an existing draft; `--bf16` exports a separate dense
control using the same architecture. The default `genai_config.json`, accepted
gated-RMS graph/runtime and previous benchmark reports remain unchanged.
The overlay disables built-in MTP and enables DFlash2 instead. Memory sampling
can reuse `ornith_engine_memory.py` with the DFlash2 reference report and
`--decoder-file decoder/model-gated-rms-dflash2.onnx`.

The benchmark performs execution, zero-failure and target-only greedy-output
checks before recording timings. These are benchmark-integrity checks, not a
task-accuracy evaluation or comparison against the source BF16 target.
Source-model and task accuracy remain deferred.

The first single-H200 run (one warmup, three measured 512-output turns per
context, CUDA graphs, same accepted runtime) produced:

| Context | Fresh accepted MTP control (tokens/s) | FP8 DFlash2 (tokens/s) |
| --- | ---: | ---: |
| 2K | 184.13 | 50.01 |
| 8K | 145.05 | 37.04 |
| 16K | 120.11 | 27.44 |
| 32K | 70.51 | 19.61 |

All twelve measured outputs match the target-only references, with zero draft
failures or disables, but only 3.73%/3.33%/2.75%/9.80% of evaluated proposals
were accepted at 2K/8K/16K/32K. Evaluated proposals stop at rejection, so those
percentages are not acceptance length or accepted/proposed ratios.
**This experiment is not promoted:** drafting/verification costs outweigh the
benefit at this acceptance level. There is no BF16 draft
control yet, so the low acceptance is not attributed specifically to FP8.
`engine-dflash2-fp8-summary.json` preserves package hashes, the fresh control,
per-context sampled peak memory and the benchmark-only decision. The accepted
gated-RMS plus FP8 MTP deployment remains the recommendation.

#### DFlash2 slowdown diagnostics

No additional MTP run is needed when only the draft changes; reuse its saved
control. Short, 2K-context, 128-output diagnostic runs found:

| DFlash2-only configuration | Decode tokens/s | Accepted drafts / target passes |
| --- | ---: | ---: |
| FP8 body, eight-row draft, seven verified drafts | 48.28 | 2 / 127 |
| BF16 body, otherwise unchanged | 48.38 | 2 / 127 |
| FP8 body, trained 16-row draft, seven verified drafts | 47.49 | 1 / 128 |
| FP8 eight-row draft, only one verified draft | 95.62 | 2 / 127 |

The one-draft result averages three measured runs; the other diagnostics
use one measured run after warmup. These are not replacements for the longer
four-context benchmark. BF16 alone and restoring 16-row drafting do not
resolve the low acceptance. Native chat formatting improves short-prompt
acceptance somewhat, but the tested long chat prompt still accepts only
three drafts in 126 target passes.

A warmed 2K Nsight trace through 128 outputs attributes **2351.20 ms** of
summed steady-decode GPU kernel time to target verification, **206.20 ms**
to the drafter and **18.94 ms** to other kernels. Target work is about **91.3%**
of this sum. The phase wall time is 3057.61 ms; summed kernel durations are
not wall-time shares. Almost all proposed rows are rejected after paying for
their target execution, which explains the large cost of seven-draft
verification. The FP8 draft uses the direct small-row FP8 kernels here, not
full-weight dequantization for every projection.

`--fp32-conv` is a separate arithmetic diagnostic: the compiled public
dynamic-convolution reference accumulates in FP32 and rounds once, whereas
the decomposed BF16 graph rounds each operation. An isolated learned-weight
test exactly matches compiled output with FP32 accumulation; the old
decomposition matches only 61.2% of elements. The opt-in graph performs
coefficient construction and accumulation in FP32, then casts once, while
leaving the default path untouched. It uses distinct `-fp32conv` filenames.
However, this change **does not fix acceptance** in the 2K test (one accepted
draft in 128 target passes, 46.86 tokens/s); it is not promoted.

Follow-up replay captured 2K real target features and sixteen committed tokens,
checking the tokens against the saved target-only baseline. The public
`z-lab/dflash` reference drafter loads the checkpoint strictly and uses the same
features, positions, anchor, mask embedding and exported FP8 head as ONNX.
Across five decode positions, final-hidden cosine similarity is
**0.999888--0.999912** for the BF16 body and approximately **0.9984--0.9985**
for the FP8 body. The reference itself proposes the wrong first token at all
five positions. Original BF16 target embedding/head controls do not change
those first proposals. Incremental cache replay also remains close to the
full-context reference; exact proposal paths are not universally identical
because small numerical differences can change selector decisions.

A separate source-target numerical control loads one original layer at a time,
with a fixed teacher-forced input, rather than running a task evaluation.
The deployed target's post-layer-45 features have cosine **-0.0171** to the
original BF16 source features on the 2K prompt. An independent PyTorch control
applies the exporter's symmetric group-128 INT2 RTN code/FP16-scale rule, then
runs ordinary reference operators on dequantized weights:

| Source-only arithmetic control | Post-layer-45 cosine to original BF16 |
| --- | ---: |
| Original weights, FP16 activations | 0.98273 |
| INT2 RTN/dequantized weights, FP16 activations | 0.04362 |

These aggregate cosines use FP64 reductions. The quantized-reference control
reproduces the large feature drift without ORT's INT2 kernels, establishing
that weight quantization is a major compatibility problem, not merely drafter
FP8 precision. It is not exact end-to-end parity: the quantized PyTorch and
deployed target still differ in late-layer features, so additional target
numerical differences are not ruled out.

With the original BF16 target's features, its actual anchor and its original
embedding/head, the reference drafter produces a block whose **first six of
fifteen proposals are accepted consecutively** by an original-target numerical
verification pass. This is one block, not an acceptance average or a BF16
throughput benchmark. Substituting source features into the INT2 deployment
would not be a valid fix: the verifier and features must describe the same
target.

The evidence now points strongly to compatibility with the current
uncalibrated INT2 RTN target. Improving/calibrating that INT2 quantization, or
adapting the drafter to the exact deployed target, is more promising than
drafter-kernel tuning alone. No 350 tokens/s reproduction is established;
the colleague's hardware/runtime/precision/workload are unavailable.
The accepted target/runtime/default configuration and saved MTP results remain
unchanged. No task/source-model accuracy benchmark was run.

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
This historical 2K sweep predates dense/expert scheduling and shared-expert
projection/SwiGLU fusion; its absolute rates are not a matched comparison
against the current decoder.

A separate long-context sweep on the current shared-expert projection/SwiGLU
decoder uses the same INT2/FP8 precision and FP16 recurrent state, one H200,
batch one, CUDA graphs, packed-only experts, 512 output tokens, one warmup and
three timed runs per case. Widths run sequentially on GPU0 with identical
runtime fingerprints and fixed attention split count 32:

| Context | Width 1 (tokens/s) | Width 2 (tokens/s) | Width 4 (tokens/s) |
| --- | ---: | ---: | ---: |
| 8K | 133.689 | 131.007 | 117.542 |
| 16K | 111.585 | 111.204 | 101.365 |
| 32K | 65.664 | 56.416 | 42.269 |

A reverse-order 16K confirmation (width two, then width one; again one warmup
and three timed turns) measured 111.134 and 111.483 tokens/s respectively.
Width two was 0.31% slower, versus 0.34% in the initial sweep: there is no
repeatable benefit from widening at 16K. All repeat outputs and MTP counters
match their respective initial cases.

All 27 timed output sequences match their own target-only reference, each
other and the prior fused decoder exactly, with zero MTP failures. Wider
drafting reduces target passes but increases draft work:

| Context | Draft width | Accepted/evaluated drafts | Target passes | MTP passes |
| --- | ---: | ---: | ---: | ---: |
| 8K | 1 | 65.91% | 316 | 308 |
| 8K | 2 | 69.49% | 248 | 478 |
| 8K | 4 | 70.95% | 199 | 760 |
| 16K | 1 | 72.30% | 313 | 296 |
| 16K | 2 | 76.33% | 240 | 447 |
| 16K | 4 | 77.93% | 188 | 688 |
| 32K | 1 | 41.55% | 393 | 361 |
| 32K | 2 | 42.79% | 353 | 641 |
| 32K | 4 | 45.68% | 321 | 1154 |

Counts include prefill and are identical across the three timed turns of a
case; they measure work, not isolated draft/verification GPU latency. At 32K,
width four only reduces target passes by 18.3% while requiring 3.2 times as
many MTP passes. Wider verification also executes more target rows per pass.
Higher accepted/evaluated draft fraction therefore does not imply better
whole-model throughput. Keep the exported default of one draft; this
repeated-text workload does not establish an optimum for other prompts.
The 200 tokens/s goal and source-model accuracy evaluation remain outstanding.

### Ornith single-Engine GPU memory

`ornith_engine_memory.py` inherits any `speculative` configuration overlay from
its parity-qualified reference benchmark. In particular, a reference measured
with `--draft-tokens 2` or `--draft-tokens 4` is sampled at that same width,
instead of silently reverting to the exported one-draft default. The effective
overlay is recorded in the memory report.

On the shared-expert projection/SwiGLU decoder, fresh 32K processes using the
long-context sweep references measured:

| Draft width | Sampled peak device memory (GiB) |
| --- | ---: |
| 1 | 43.194 |
| 2 | 43.208 |
| 4 | 43.229 |

Each process includes one warmup and three measured 512-token turns. All
outputs and MTP counters match the corresponding width's reference. Wider
drafting adds little sampled peak memory here (about 14 MiB at width two and
36 MiB at width four), but does not improve throughput on these prompts.
These are 10 ms device-wide samples, not exact allocator peaks.

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
