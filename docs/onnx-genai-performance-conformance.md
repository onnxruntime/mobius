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

Against ONNX GenAI `52339e10`, with no `model.io` in any package and no port
contracts on any component that ships an artifact:

| Check | Result |
| --- | --- |
| `validate_metadata` over the generated packages | 11/11 valid |
| `mobius_workflow_conformance` (engine executes each package) | 11/11 passed |
| `mobius_static_cache_workflow_executes` specifically | passed |

The last row is the one that matters. The fixed-capacity decode path needs the
write cursor, the valid length and the per-layer buffer pairs; it resolved all
of them by lowering the workflow, which is what makes removing the second copy
safe rather than merely tidy.

ONNX GenAI later moved the document-level invariants onto
`load_metadata_package`, so the package loader — the path the
`validate_metadata` binary and every on-disk consumer take — now enforces rules
that previously only ran for callers holding a parsed document. That is a strictness increase applied to the exact entry
point our fixtures go through, and all 11 were re-checked against it rather
than assumed to be unaffected. They pass because they carry no `model:` block
at all: a package with one serialized ABI has nothing for a coexistence rule to
find.

### Why this document names one SHA and not several

The ONNX GenAI branch we validate against is rebased as its own base moves, so
its commits are rewritten and the old hashes stop being reachable from any ref.
A citation to one of them does not merely go stale: it becomes unresolvable
once the unreferenced object is collected. Only `.github/workflows/main.yml`
pins a hash, because only CI needs to fetch an exact tree, and a bump there is
gated on re-running both suites. Everywhere else, changes are described by what
they do.

This is not hypothetical. The pin was verified against the branch head, that
head was later rebased, and the pinned commit became reachable from no ref —
`git ls-remote` showed zero matches and no remote ref had it as an ancestor.
`actions/checkout` would have kept succeeding until the object was collected
and then failed with `reference is not a tree` on a commit no one could
inspect. Bumping a pin is therefore checked with `ls-remote` plus an ancestry
test rather than with an API call that answers just as happily for an orphan.

### What the fixtures commit, and what they do not

Only the metadata is committed. The graphs and the adapter weight file are a
deterministic function of `tests/generate_onnx_genai_validation_packages.py`,
and committing them stored 146 files and roughly 14 MB — including a 5 MB
weight blob — restating in unreadable bytes what the script already says. CI
already regenerated the whole tree and compared it, so the committed copies
were never the thing under test.

Textproto was the other option and is the right one for ONNX GenAI's own
fixtures, which are 26 KB synthetic graphs. It is the wrong one here: these
carry real weight blobs, a text encoding would grow rather than shrink them,
and this repository does not use the protobuf APIs a textproto writer needs.

Regeneration takes about four seconds for all eleven packages, so tests that
need a graph build them once per session through the
`materialized_workflow_packages` fixture.

The change has a consequence worth stating plainly, because it decides where
CI must point: a package whose artifacts are absent does not validate. A
checkout of the committed tree alone fails with `component
'cache_length_update' artifact ... cannot be opened`. That is correct — a
workflow claims to be a complete description of something executable, and it
should not pass when the executable part is missing. Validation and
conformance therefore run against the regenerated tree, and the comparison
step additionally asserts that every artifact the committed metadata names was
really produced, so a generator that quietly stopped emitting one would be
caught rather than leaving an assertion with nothing to check.

### The role a fixed-capacity cache cannot do without

A component named in an `indexed_scatter` group's `write_indices_ports` or
`kv_length_ports` must declare a sequence role. Those two control ports are
consumed by exactly one thing — the driver that writes into a preallocated
buffer at an index — and that driver binds from the resolved decode ABI, every
field of which is found by role. A component handed the cursor while declaring
no role is therefore unresolvable as a decoder, and the package degrades to
inferring ports from shapes.

Nothing upstream rejects that combination, and the reason is worth recording:
identifying the decoder *requires* a sequence role, so a component that omits
one is invisible to the check that would have caught it. A validator's
sole-decoder guard is additionally scoped to workflows with a single ONNX
component, which no package with policy graphs ever is — ours ship ten. The
contradiction is only visible to the producer, so
`test_a_scatter_bound_component_is_always_resolvable_as_a_decoder` asserts it on
both the built packages and the shipped fixtures. Measured: dropping the role
from `static_cache` still validates upstream, and fails here.

### Where the transcription boundary actually falls

A component backed by a shipped `.onnx` declares `ports.roles` and nothing else:
the artifact is authoritative for its ports, and `pipeline_admission` checks
declarations against the live session, so a YAML copy can only drift.

Policy graphs are the exception, and the boundary was measured rather than
assumed. A workflow SSA value inherits its dtype, rank and request axis from the
port that produced it (`validation.rs` binds `value_contracts` from
`ports.outputs`), so a validator — which reads metadata without the artifacts —
has no other source. Dropping policy contracts made 4 of the 11 packages invalid
with `<node>.when is row-wise but <node>.value declares no request_aligned
batch_layout`. Those contracts type the workflow's own dataflow; they are not a
description of an external interface, and they stay.

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
