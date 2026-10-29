# Drama 3 standalone runtime consumer

`scripts/audit_drama3.py` consumes already-exported graphs and independent source
fixtures. It does not export a model, download licensed weights, generate an
oracle, build ORT, install SDKs, or open the public reduced-precision guards.
The existing source/export commands and sampled/teacher-forced benchmark routes
remain available. The single-pass greedy route in `benchmark_drama3.py` reuses
the same request executor.

The target interpreter needs only NumPy, ml_dtypes, and ONNX Runtime. Run from
this source checkout; `drama3_runtime.py` loads the existing pure position/cache
validator directly from its source file without importing `mobius.__init__`.
Neither the builder, onnxscript, Torch, Torch DLPack, nor protobuf APIs run in the
target process. Policies must be built and saved beforehand in the export
environment. `main_policy` is the existing `benchmark_drama3.greedy_model` saved
ONNX graph, not the stochastic `build_fish_main_sampler`.

## Two independent pinned manifests

Supply SHA256 digests out of band for both JSON manifests. Paths are explicit
local paths, not network download instructions.

The **input DATA manifest** is the coordinator's immutable schema-v1
`cuda13-input-manifest.json`. Its checkpoint/source revisions, per-dtype request,
golden hash, NPZ hashes/byte counts/field lists, initial prompt hash, frame
inventory and historical reference environment are verified read-only. Its
`consumer_selection` is historical readiness context, not a selected runtime:
that frozen file must never be edited to mark a new consumer ready.

Only state zero's `input_frames` is an execution operand. The consumer adds its
batch-one dimension; frozen production prompts have shape `[30,11]`.
Other source fields, including later input frames and positions, are expectations
only. Positions are derived from the actual request. Target embeddings feed the
decoder, target hidden/shared activations feed the tower, and target returned
caches and generated frames feed the next call. The actual generated codes feed
the saved FP32 waveform codec. No source output, hidden state, cache, or later
frame is substituted, including after divergence.

The **consumer selection manifest** is separate and must describe actual saved
artifacts, not the historical prototype paths. Its required schema is:

```json
{
  "schema_version": 1,
  "status": "ready",
  "checkpoint_revision": "b2b5ee998034b2975720906c815e7fb0e9f007bd",
  "source_revision": "541566932abbaebcb0a40e562a1f846738383f38",
  "inputs_sha256": "<input DATA manifest SHA256>",
  "dtype": "bf16",
  "expert_execution": "dynamic_fp8",
  "device": "cuda",
  "device_id": 0,
  "gpu_uuid": "GPU-<explicitly leased physical GPU>",
  "max_total_length": 256,
  "optimization": "ORT_ENABLE_ALL",
  "providers": [
    ["CUDAExecutionProvider", {
      "device_id": "0",
      "use_tf32": "0",
      "enable_cuda_graph": "0",
      "arena_extend_strategy": "kSameAsRequested",
      "cudnn_conv_algo_search": "EXHAUSTIVE"
    }],
    "CPUExecutionProvider"
  ],
  "actual_providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
  "actual_provider_options": {"<provider>": {"<option>": "<actual value>"}},
  "disabled_optimizers": [],
  "session_config": {"session.disable_prepacking": "1"},
  "policy": "saved-greedy-main-v1",
  "config": {"path": "<pinned config.json>", "sha256": "<SHA256>"},
  "runtime": {
    "ort_version": "<actual version>",
    "ort_build": "<exact ort.get_build_info()>",
    "cuda_toolkit": "13.0.48",
    "cublas": "13.0.0.19",
    "cudnn": "9.14.0.64",
    "wheel": {"path": "<shared official-build wheel>", "sha256": "<SHA256>"},
    "files": [{"path": "<loaded ORT native binding>", "sha256": "<SHA256>"}],
    "libraries": [{"path": "<actual loaded CUDA DSO>", "sha256": "<SHA256>"}]
  },
  "components": {
    "embedding": {
      "graph": {"path": "<embedding/model.onnx>", "sha256": "<SHA256>"},
      "external_data": [{"path": "<embedding/weights>", "sha256": "<SHA256>"}],
      "dtype": "bf16",
      "required_runtime_features": "<exact graph metadata value, or empty string>"
    }
  }
}
```

Expand `components` to exactly `embedding`, `decoder`, `audio_generator`,
`main_policy`, and `codec`, each with the same four fields. The three model
components use the selected dtype; codec is `f32`, and the logits-only greedy
policy can be recorded as `int64`. Codec metadata must declare sample rate
44100 and 2048 samples per frame. Each graph must have its own directory
containing only its graph and declared external-data files, with no symlinks.
The exporter/selection producer is responsible for closing the external-data
references inside those directories. The consumer hashes every file with
bounded reads and rejects unlisted files; it does not parse graph protobufs.
Artifact inventory is reverified after execution.

The native compatible expert mode requires the exact
`com.microsoft::QMoE[fp8_execution_profile=fp16_compatible]` feature plus matching
`mobius.weight_loading` provenance: Fish architecture, pinned revision,
dynamic-FP8 activation quantization, native FP8 storage and mixed block-FP8-128
format. Generic QMoE availability, an unknown profile, or weight metadata alone
cannot select this mode. Decoder dtype and exact selected runtime-feature
metadata are checked independently. Package-level context metadata may appear
on all components: decoder `inputs_embeds` and tower `hidden_states` calls
validate positions and caches; the stateless embedding has no such controls.
An explicit unsupported or conflicting QMoE profile is rejected before the
legacy dynamic-activation marker is considered; that marker cannot override an
unknown profile. Prepared native launch commands use the exact composed
interpreter with `-I` isolation and insert only the explicitly pinned harness
script path, never a builder environment through `PYTHONPATH`.

`files` must include the installed native ORT binding and all loaded ORT
provider/runtime binaries. `libraries` pins the **expected allowed** CUDA
library inventory (driver, cudart, cuBLAS/Lt, cuDNN sublibraries, NVRTC, and
nvJitLink when applicable), using resolved paths and actual file hashes. All
actual loaded CUDA libraries must belong to this pinned inventory; a permitted
DSO need not already be loaded (for example cuDNN CNN before the first codec
operation). Pre-op identity checks verify file hashes, not whether future lazy
libraries are mapped. Actual maps are captured/verified after the first tower
prefill, the complete neural audit, codec execution, and measured request.
Reports distinguish `actual_loaded_cuda_libraries` from
`allowed_cuda_libraries_not_loaded`; neither is inferred from the other. Loaded
library attestation currently requires Linux `/proc/self/maps`; unsupported
attestation is an error, not a qualified fallback. This inventory and the pinned
wheel/build identify the runtime; version labels alone are not acceptance.
The runtime owner supplies exact source/toolkit provenance for that shared build.
Private plugin registration is deliberately unsupported.

Library failures retain progressive evidence through the standard-library-only
`drama3_library_identity.py` helper: all relevant observed map rows, original
and resolved paths, basenames, independently read file hashes, expected source
keys and explicit match/rejection decisions. Collection finishes before an
identity rejection is raised. Expected hashes are never labelled actual-loaded
observations; an unapproved path fails even if its file matches an allowed hash.
Missing/deleted files and unavailable maps remain explicit errors. These reports
cannot reconstruct mappings from an already-ended process.

Native classification groups all segments of each file-backed object before
deciding, so a non-executable segment cannot hide a later executable segment.
Within the ORT/CUDA scope (expected canonical paths, installation roots and
relevant raw/resolved names), bounded observed ELF headers, shared-library names
and executable segments distinguish native candidates from readable non-ELF
data. Relevant data with neither executable segments nor a shared-library name
is retained as `EXCLUDED_NON_NATIVE_DATA`, not added to a native allowlist.
Unknown native candidates, deleted/unreadable files and ambiguous/short headers
still fail. File headers and hashes describe resolved file bytes, not resident
memory or the headers of an ended process.

This is **scoped ORT/CUDA attestation**, not a full-process native-code or general
adversarial-renaming closure guarantee. Unrelated Python, system and NumPy
objects are labelled `OUTSIDE_ATTESTED_SCOPE`, not validated or called non-native;
they do not require new native allowlist entries.

Parent activation must preserve native-component qualification scopes. QMoE,
attention and TRSM operator PASS evidence is mandatory, alongside the full-model
gates. A pinned standard-primitives report that records the historical real
expert BF16 BMM mismatch is diagnostic, not a fabricated all-PASS result and not
a blanket bit-identity prerequisite for compatible QMoE. Its discrepancy stays
visible; full-model vector/raw-bit qualification remains unchanged.

All components use the same explicit provider/options/session-config/disabled
optimizer selection. Actual ORT providers/options must match it exactly.
`drama3_session_profile.py` is an ORT-only shared constructor/readback helper.
Every actual saved embedding, decoder, policy, tower and codec session records
its full `get_provider_options()` map, registered provider order, observed
`get_session_options()` fields and explicit session-config getters. The loader
then rechecks **all five actual sessions together before the first neural call**;
any required semantic, map, option, config or declared-device mismatch aborts.
Failure reports preserve available readbacks and keep the barrier false.
Getter exceptions retain observations incrementally: a missing config entry or
failed SessionOptions getter records its explicit error without discarding the
already observed full provider map, and without inventing unavailable defaults.
CUDA profiles must explicitly preserve TF32 off, capture off, `kSameAsRequested`,
`EXHAUSTIVE`, and disabled prepacking; no missing control is silently repaired.

A tiny weightless discovery session may bootstrap a candidate expected map only
under a separately parent-approved grant-time metadata stage. It is not evidence
for any real model session and must never run inference or load checkpoint
weights. The actual all-model readbacks above remain mandatory even when tiny
discovery succeeds. Registered providers are not proof of per-node execution
placement. ORT has no disabled-optimizer getter: that list is labelled
source-bound constructor/requested evidence, not fabricated runtime readback.
The initial active CUDA 13 profile may use the explicit empty
`disabled_optimizers: []` list. No RoPE fusion exclusion is required or inserted.
Disabling `RotaryEmbeddingFusion` is an unverified historical candidate, not an
active CUDA 13 default. Only actual CUDA 13 optimized-operator and rounding
evidence can justify a recorded per-profile exclusion, applied identically
before both audit and measurement. Never fall back to BASIC/EXTENDED or insert
zero/identity barriers to avoid a fusion. Native ordered
reductions are optional: declare runtime requirements only when the actual
graph metadata requires them. This harness does not lower/rewrite graphs.

## Execution and gates

The immutable DATA manifest has **no independent waveform oracle**. This harness
executes the actual-frame FP32 codec and checks dtype, shape, finite samples,
duration and hashes. Without the optional independent reference below, reports
mark `codec_numerical_quality.status` as `UNVERIFIED` and list the independent
FP32 waveform numerical-quality gate as blocked. Even if neural and performance
checks pass (`native_cuda13_neural_and_performance_gates_passed`), root
`native_cuda13_acceptance` remains false. A report `status: passed` describes
only performed checks, not a missing codec qualification.

An optional `waveform_reference` in the **separately hash-pinned selection**
provides an independent reference provenance JSON descriptor:

```json
{
  "waveform_reference": {
    "path": "<independent-waveform-provenance.json>",
    "sha256": "<out-of-band provenance manifest SHA256>"
  }
}
```

The provenance JSON must contain:

```json
{
  "schema_version": 1,
  "checkpoint_revision": "b2b5ee998034b2975720906c815e7fb0e9f007bd",
  "source_revision": "541566932abbaebcb0a40e562a1f846738383f38",
  "codec_revision": "b2b5ee998034b2975720906c815e7fb0e9f007bd",
  "collector": {"path": "<independent CPU collector>", "sha256": "<SHA256>"},
  "source_device": "cpu",
  "independent_source_invocation": true,
  "source_command": ["<source CPU python>", "<collector>", "<recorded arguments>"],
  "frames_sha256": "<actual target-generated INT64 frame bytes SHA256>",
  "frames_shape": [100, 11],
  "frames_dtype": "int64",
  "dtype": "float32",
  "sample_rate": 44100,
  "samples_per_frame": 2048,
  "waveform": {"path": "<independent-waveform.npz>", "sha256": "<SHA256>"}
}
```

The NPZ contains only the `waveform` array. The reference producer runs an
independently invoked source CPU codec in a **separate process**, later, using
the actual captured target frames; it must not import Torch into the CUDA 13
target process. That producer owns truthful source/collector/checkpoint pins
and independence provenance. The consumer verifies the pinned declaration and
files, not the historical process itself, and never executes the collector.
No reference generation, trained weights or source arrays are added to this
repository, and no existing oracle is overwritten.

The target codec still receives only actual generated codes. Expected samples
reach only the comparator. A supplied malformed, stale, differently generated,
wrong-codec, or failed numerical reference is an explicit failure, never an
unverified success. The codec graph's actual `codec_revision` metadata must
match the independent provenance. Manifest/NPZ/collector hashes and actual
frame dtype/shape/bytes, 44100Hz and hop 2048 are checked.

Both expected and actual must be finite FP32 `[1,1,frames*2048]`, expected RMS
must exceed `1e-3`, and the existing frozen waveform gate is unchanged:
`np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=1e-3)`.
RMSE/SNR are reported only, with no added threshold (exact match reports SNR
as `"infinity"`). The independently referenced audit waveform and measured
waveform must both pass for a benchmark root pass. A verified codec comparison
cannot bypass actual CUDA 13 identity, neural/raw-bit audit, non-instrumented
performance, leased ownership, frame count or RTF <= 1.0 requirements. CPU/mock
comparison tests do not qualify real checkpoint or CUDA 13 waveform parity.

Use `--diagnostics` for a separate, explicitly **unqualified** evidence run.
ORT saves its actual optimized graph for each component under the new output
root, and emits execution profiles. The report hashes these graphs and profile
files with bounded reads, inventories emitted files, and summarizes actual
provider/opcode call counts from profile JSON without importing IR/protobuf.
This is a future CUDA 13 fusion/rounding investigation hook, not existing CUDA 13
evidence or a reason to disable a fusion. The selected ALL/provider/exclusion
controls are unchanged. Diagnostic instrumentation always makes timing
unqualified, even when full-vector checks pass and measured RTF is below 1.0.
Repeat the non-instrumented audit/benchmark with the same recorded selection
for performance admission.

```powershell
python scripts\audit_drama3.py `
  --inputs <immutable-data-manifest> --inputs-sha256 <data-SHA256> `
  --selection <actual-consumer-selection> --selection-sha256 <selection-SHA256> `
  --output <new-separate-output-root> --raw-bits --benchmark --warmup 2 `
  --lease-uuid GPU-<explicitly-leased-UUID>
```

The output root must be new and outside every input directory. Outputs include
the generated frames, actual-frame waveform, and a report. Failed execution
writes an explicit failed report when preflight has advanced to execution, then
raises. Preflight readiness/provenance failures raise before opening sessions.

Full-vector checks enforce shape, target dtype, finite values, every source
field, and the unchanged original validator tolerances: FP32 `atol=rtol=1e-3`,
FP16/BF16 `1e-2`. `--raw-bits` additionally compares every transported F32 bit
pattern, including signed zero. Counts are derived from the actual fields and
arrays, not hardcoded historical totals. Frame equality cannot replace vector
checks. The audit copies inspected activations to CPU using ORT output bindings;
benchmark calls omit these audit copies. Both use the same graph sessions and
optimization/provider selection. Audit timing is not performance evidence.

Caches remain on the execution device. BF16 inspection copies ORT-owned CPU
storage without casting or Torch; a BF16 GPU pointer is never dereferenced.
Scalar booleans and empty KV dimensions retain their ABI. Reduced-precision
decoder metadata must declare `shared-contiguous-dynamic-v1`; its existing
validator checks unpadded shared positions and dynamic KV append length equal
to position start. Initial caches reset both carries. No mixed-row/static-KV,
capture, whole-model B2, or T257 claim follows from component tests.

The benchmark runs complete warmup requests, preserves resident returned caches
within each request, and retains every measured frame duration, including the
first cached call and outliers. UUID/ownership is checked before work, after
warmup, and after measurement. A foreign process appearing at any point
invalidates timing. End-to-end RTF includes prefill, token generation, and the
actual codec call, divided by the verified waveform duration. The unchanged
gate is RTF <= 1.0, with at least 50 frames, a full raw-bit audit, and CUDA
ownership required for qualification. No percentile trimming or frame-hash-only
admission is supported. A CPU run is always diagnostic, never native acceptance.

Each `nvidia-smi` query has one absolute 20-second deadline starting before
process creation. Startup time consumes this budget; `wait` receives only its
remaining time, never a new 20 seconds. An already-expired startup is UNKNOWN
and goes straight to owned cleanup, even if it produced plausible output.
Spawn errors retain actual elapsed time with `owned_pid`/`returncode` null.
OS process creation itself cannot be interrupted before it returns an owned
handle; such startup overruns are reported, not relabelled deadline-compliant.
Timeout or wait errors trigger
one kill of that query's owned child and a bounded two-second cleanup/reap grace;
no automatic retry, other-GPU retry, process-name/global kill, or unbounded pipe
draining is used. A snapshot has two queries, so reserve up to 44 seconds for a
failed/slow snapshot in any future lease budget, in addition to model work.
Timeout, nonzero exit, spawn/kill/reap failure, malformed ownership/health CSV,
missing/ambiguous selected GPU or oversized output aborts with explicit
**UNKNOWN** health, never zero/empty healthy defaults. An unreaped owned PID is
reported as unresolved, not claimed cleaned up. Partial stdout/stderr, return
status and cleanup details are preserved in a `gpu_snapshot_error` JSON event
and in audit failure reports; successful snapshots retain raw query evidence.
An empty successful compute-app query is valid, but failed collection is not.
Device/UUID selection and contention/numeric/provider/performance gates remain
unchanged. These bounds are verified with CPU mocks, not physical NVML queries.

The current native consumer graphs/shared CUDA 13 runtime and lease are external
prerequisites. A not-ready selection fails clearly. CPU synthetic/mock checks
and historical CUDA 12 source fixtures are not CUDA 13 execution, full-model
parity, or performance qualification. No reference, licensed trained array, or
changed tolerance is included in this work.
