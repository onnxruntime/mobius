# Flint routed-expert INT4 with Olive

**Experimental CUDA-only target, not a publisher-qualified release.** This
recipe quantizes the routed experts of `RecursiveSI/flint` at revision
`2a247be0c0aec895ccb0ac7ca92229ec6df37288`. Independent publisher BF16
logit/cache qualification still fails its frozen tolerances. Agreement between
ORT and GenAI, and the arithmetic checks below, do not replace that gate.

## Prerequisites

- Mobius [PR #792](https://github.com/onnxruntime/mobius/pull/792).
- A complete pinned BF16 CUDA package: `model.onnx`, all external weights,
  `weight-loading-report.json`, tokenizer/chat assets, `genai_config.json`,
  and `lib/libonnxruntime_source_compat_cuda.so`.
- The CUDA extension from ORT
  [PR #33594](https://github.com/microsoft/onnxruntime/pull/33594), and a
  compatible ORT GPU build with BF16 raw-weight `QMoE` prepacking.
- The BF16 cache fix from GenAI
  [PR #2778](https://github.com/microsoft/onnxruntime-genai/pull/2778).
- An Olive checkout containing `OnnxMoEQuantization`; released Olive 0.13.0
  does **not** contain this pass. Install that checkout in the conversion
  environment, not an unrelated agent's environment.

The measured runtime was ORT 1.31.0 and source-built GenAI 0.18.0-dev on one
H100 80GB HBM3. Other GPU architectures, padding, batching, beam search,
static/shared caches and CUDA graph capture are not qualified.

Conversion is CPU-side and took about 490 seconds. Budget approximately
100 GB of host RAM, the 80.47 GB source package, the 24.15 GB destination,
and one source-payload-sized staging file. It does not retain a second full
BF16 copy. Deployment scripts do not import PyTorch.

## Convert and execute

Run from the Mobius checkout, with the source packages/bindings above installed
in an isolated environment. Never overwrite the source directory.

```bash
python examples/olive/flint/quantize.py \
  --input-model /models/flint-bf16 \
  --output-dir /models/flint-routed-int4

CUDA_VISIBLE_DEVICES=0 python examples/olive/flint/inference.py \
  --model /models/flint-routed-int4 \
  --output /results/flint-int4-ort.json

python examples/olive/flint/benchmark.py \
  --model /models/flint-routed-int4 --gpu-index 0 \
  --output /results/flint-int4-benchmark.json

# Run separately, with the same bindings and an otherwise idle selected GPU.
python examples/olive/flint/benchmark.py \
  --model /models/flint-bf16 --gpu-index 0 \
  --output /results/flint-bf16-benchmark.json
```

`int4.json` configures symmetric round-to-nearest INT4, group size 128. This
is **not** NF4 or Q4_K_M. Attention, dense/shared experts, embeddings and the
FP32 LM head are not quantized.

The source graph has six separately weighted singleton routes per MoE layer.
The recipe batches those into one row-wise call, then splits its outputs back
into the unchanged external FP32 sigmoid-weighting chain. It does not replace
Flint's routing with a top-6 softmax. This produces 47 `QMoE` nodes and exactly
94 packed banks, avoiding six independent CUDA prepacked copies per layer.
Weight-only FC1 interleaving is folded before quantization. Scales retain the
activation dtype, and code selection uses their actual stored precision.

`inference.py` validates full-logit/cache shapes and finiteness, own-cache
greedy decoding, CUDA placement of critical arithmetic, and Torch-free
execution. It transfers outputs to the host for inspection and is **not** a
throughput benchmark. The INT4 ORT and GenAI probes agreed on all 28 selected
tokens, including EOS; GenAI independently answered `4`, `42`, and `7`.
These are functional spot checks, not a general quantization quality score.

## Measured single-GPU comparison

One H100 80GB HBM3, driver 580.159.04, identical ORT/GenAI/native-library
builds. Decimal GB is used for files, GiB for device memory.

| Metric | BF16 | Routed INT4 |
|---|---:|---:|
| Complete package | 80.47 GB | 24.15 GB |
| Model construction | 188.61 s | 7.37 s |
| Sampled generation device peak | 78.48 GiB | 43.64 GiB |
| 16-token prompt, median TTFT | 84.89 ms | 71.37 ms |
| 16-token prompt, median decode | 12.06 tok/s | 14.47 tok/s |
| 196-token prompt, median TTFT | 106.55 ms | 73.62 ms |
| 196-token prompt, median decode | 11.88 tok/s | 14.67 tok/s |

The package is 69.99% smaller. File size is not a VRAM requirement: prepacking,
allocator reservations, caches and workspace are included in device usage.
Construction time is not a cold-filesystem benchmark.

Each prompt uses two complete warmup generations and three measured runs of
50 selected tokens. EOS is suppressed only for fixed-length performance
runs; the three natural quality runs require exact answers and a selected
EOS. TTFT includes prompt append/prefill and first-token selection, excluding
tokenization and generator construction. Decode excludes the first token.
CUDA is synchronized at timing boundaries. An independent NVML process
samples total selected-device usage every 20 ms, avoiding native GIL sampling
gaps; peaks are sampled, not allocator-exact. Receipts record all measured
token IDs, native library/build hashes and the graph-header hash.

The BF16 artifact was exported from pre-rebase Mobius commit
`6752128f2e0043a6d71ce661925b8fa48cbc6dc0`; its header SHA-256 is
`fcc3ce07d2732f97712c21b664b7a682538799923d0143dbee7ef49ecb8628dd`.
The subsequent rebase removes unrelated inherited Nemotron changes and does
not change Flint model arithmetic. Do not interpret these measurements as
independent publisher parity or assume they apply to another export.
