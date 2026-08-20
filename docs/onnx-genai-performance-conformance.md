# ONNX GenAI workflow performance conformance

Functional workflow conformance is necessary but not sufficient for release. A metadata-driven
workflow passes only when it matches or improves on the equivalent native implementation under
identical conditions.

## Controlled comparison

Every workflow/native pair must record and exactly match:

- model artifact hash and weights;
- execution provider, device, precision, provider options, and runtime build;
- batch size and every input/state shape;
- prompt and generated-token count;
- sampling algorithm and request parameters, including seed and RNG offset;
- dense/shared/paged KV mode and capacity;
- graph-capture enablement and shape specialization; and
- warmup count, measured iterations, and synchronization timing points.

Do not compare different quantization, kernels, KV layouts, sampling math, or capture settings.
Report p50/p95 and raw samples; the release gate uses the median after warmup.

## Required scenarios

1. **Decoder min-p:** decoder, last-token logits, request-parameterized min-p sampler, and
   termination. The steady-state policy path must form and replay one same-device execution
   island.
2. **Speculative accept:** proposer, verifier, acceptance/prefix policy, state correction, and
   termination with a mostly accepted deterministic proposal fixture.
3. **Speculative reject:** the same artifacts and shapes with a deterministic rejection fixture.
   Rollback/truncation tensor math must remain device-resident.
4. **Grammar boundary:** repeat decoder and speculative cases with a stateful grammar adapter.
   Record the expected adapter island boundary and verify that pure ONNX work on each side is
   still fused and captured independently.

Run at batch 1 and one representative batched shape. Include fixed decode shapes and a bounded
shape transition case. Use the same accepted/evaluated token history for adaptive-K comparisons.

## Required instrumentation

The runtime benchmark record consumed by
`mobius.integrations.onnx_genai.performance.compare_performance` contains:

- throughput and unit (`tokens_per_second` or `steps_per_second`);
- TTFT for generation workflows;
- peak device memory;
- host-to-device and device-to-host copy counts and bytes;
- explicit device synchronization count;
- ORT/composite session boundary count;
- kernel launch count;
- per-island component list, device, eligibility, capture count, replay count, and fallback reason.

ORT profiling supplies node placement, kernel launches, session boundaries, and memcpy events.
Allocator/provider telemetry supplies peak device memory and stable-address failures. Island
diagnostics must be sampled after warmup and after measured replay; a merely eligible island is
not evidence of capture.

The comparison gate rejects mismatched identities before evaluating performance. By default it
allows at most a 5% throughput, TTFT, or memory regression and permits no additional host/device
copies, explicit synchronizations, or session boundaries. Projects may tighten this threshold but
must not silently loosen it for individual model families.

## Failure reporting

Never mark a workflow ready from functional E2E alone. For every failed scenario, retain the
native and workflow records and report:

1. the first divergent metric;
2. the responsible execution island and component boundary;
3. profiler evidence such as CPU placement, memcpy, synchronization, allocation, or unsupported
   capture kernel;
4. whether the cause is producer structure, planner lowering, provider support, or runtime memory
   management; and
5. the ordinary-execution fallback result.

As of ONNX GenAI `8bacf8c`, an application-overridable sampler is rejected by island formation
before override resolution. This prevents the required decoder/sampler/termination capture
demonstration even when the selected implementation is pure same-device ONNX. Performance
acceptance remains blocked until override selection precedes island partitioning and the resolved
implementation is evaluated for purity and placement.

## Fixed-capacity cache and FP8 runtime evidence

Measured on an NVIDIA H200 with `onnxruntime-gpu==1.29.0`, CUDA execution
provider, against a real `--features static-cache` Qwen2 export.

### Static cache — executes as specified

| Check | Result |
| --- | --- |
| B=2 prefill vs per-row B=1 | max abs delta 0.0 |
| Scatter confined to `[0, nonpad_kv_seqlen)` | all rows; zero energy outside |
| Decode, divergent per-row cursors (row 0 advancing, row 1 finished), 3 steps | max abs delta 3.1e-6 |
| Compaction: the finished row reclaims its own last slot each step | verified, no writes past the reclaimed slot |

Per-row cursor divergence during decode therefore works. Ragged *prefill* is
not claimed and was not measured: ONNX leaves the region between
`nonpad_kv_seqlen` and the query length undefined.

### FP8 KV cache — not executable with the shipped kernel, and not an IMA

Session creation fails during graph partitioning; no kernel is ever launched,
so this is not an illegal memory access:

```
transformer_memcpy.cc:253 IsNodeCompatibleWithProvider —
Provider type for GroupQueryAttention node 'node_GroupQueryAttention_9' is not set
```

ORT reports an unassigned node as an initialization exception rather than a type
error, so the cause was isolated by a three-way comparison of the same graph:

| Variant | Result |
| --- | --- |
| No FP8 pass | loads and runs on CUDA EP |
| 14-input GQA with `k_scale`/`v_scale`, quantization attributes, **FLOAT** KV | loads and runs on CUDA EP |
| Same node with **FLOAT8E4M3FN** KV | node unassigned, session init fails |

The rejection is the KV *type constraint* — `tensor(float8e4m3fn)` is not in the
CUDA `GroupQueryAttention` past/present type list in 1.29.0 — not the scale
input arity and not the attributes. The exported graph and its metadata are
well-formed and validate; only the local kernel is missing.

### Canonical representation — lowering verified

Against ONNX GenAI `1d8cfefe`, with no `model.io` in any package:

| Check | Result |
| --- | --- |
| `validate_metadata` over the checked-in fixtures | 11/11 valid |
| `mobius_workflow_conformance` (engine executes each package) | 11/11 passed |
| `mobius_static_cache_workflow_executes` specifically | passed |

The last row is the one that matters. The fixed-capacity decode path needs the
write cursor, the valid length and the per-layer buffer pairs; it resolved all
of them by lowering the workflow, which is what makes removing the second copy
safe rather than merely tidy.

## Current measured baseline

ONNX GenAI `8bacf8c` reports paired five-sample synthetic native/composite measurements over
100 iterations. These establish instrumentation, not Mobius producer readiness:

| Device | Path | Workflow/native throughput | Warm TTFT workflow/native | Result |
| --- | --- | ---: | ---: | --- |
| CPU | decoder policy | 1.032 | not reported | within 5% |
| CPU | min-p policy | 0.962 | not reported | within 5% |
| H200, ORT 1.28 CUDA | decoder policy | 0.903 | 3.03/17.93 ms | throughput fails |
| H200, ORT 1.28 CUDA | min-p policy | 0.957 | 4.36/3.64 ms | TTFT fails |

The CUDA islands captured once and replayed 503 times; the speculative verifier/policy fixture
also captured and replayed. Cold workflow startup remained 467/49 ms for decoder and 231/18 ms
for min-p because the first workflow run discovers output extents and constructs stable bindings.
The remaining steady-state decoder throughput gap requires ORT kernel/provider profiling; it is
not explained away by functional parity. Min-p throughput is within the default bar, but its warm
TTFT is not.

These measurements do not cover the real Mobius package, KV service mode, per-row serving, or the
application-overridable sampler path. Those cases remain blocked/not demonstrated and must produce
records accepted by the comparison gate before PR readiness.
