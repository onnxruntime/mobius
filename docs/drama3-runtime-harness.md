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

`files` must include the installed native ORT binding and all loaded ORT
provider/runtime binaries. `libraries` must equal the actual loaded CUDA
library inventory (driver, cudart, cuBLAS/Lt, cuDNN sublibraries, NVRTC, and
nvJitLink when loaded), using resolved paths and actual file hashes. Loaded
library attestation currently requires Linux `/proc/self/maps`; unsupported
attestation is an error, not a qualified fallback. This inventory and the pinned
wheel/build identify the runtime; version labels alone are not acceptance.
The runtime owner supplies exact source/toolkit provenance for that shared build.
Private plugin registration is deliberately unsupported.

All components use the same explicit provider/options/session-config/disabled
optimizer selection. Actual ORT providers/options must match it exactly.
If source RoPE requires `RotaryEmbeddingFusion` disabled, list it explicitly
before both audit and measurement, never only in one stage. Native ordered
reductions are optional: declare runtime requirements only when the actual
graph metadata requires them. This harness does not lower/rewrite graphs.

## Execution and gates

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

The current native consumer graphs/shared CUDA 13 runtime and lease are external
prerequisites. A not-ready selection fails clearly. CPU synthetic/mock checks
and historical CUDA 12 source fixtures are not CUDA 13 execution, full-model
parity, or performance qualification. No reference, licensed trained array, or
changed tolerance is included in this work.
