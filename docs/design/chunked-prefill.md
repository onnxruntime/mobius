# Explicit source-chunk prefill numerical candidate

`mobius.components.ChunkedPrefillScan` is a parameter-free `nn.Module`.
It is **not selected by any default model, task, registry, or export**.
It implements explicit prefill, including one-token prefill, not cached decode.

```python
scan = ChunkedPrefillScan(num_heads=128, num_groups=8, head_dim=64, state_dim=128)
readout, final_state = scan(op, x, dt, negative_a, b, c, initial_state)
```

| Argument/result | Dtype | Layout |
| --- | --- | --- |
| `x` | BF16 | B, T, H, head_dim |
| `dt` | FP32 | B, T, H |
| `negative_a` | FP32 | H |
| `b`, `c` | BF16 | B, T, groups, state_dim |
| optional `initial_state` | FP32 | B, H, head_dim, state_dim |
| `readout` | FP32 | B, T, H, head_dim |
| `final_state` | FP32 | B, H, head_dim, state_dim |

Batch/time may be symbolic; time must be positive. Head/group/dimension
sizes are constructor constants. Omitted state means zero, not a dirty cache.
An explicitly supplied initial state is consumed, as in the generic SSD scan.
There are no learned parameters and no changes to existing parameter paths.

## Preparation and precision boundary

The caller supplies prepared time steps: promote raw dt and bias to FP32,
add, use softplus when the sum is at most 20 (identity above 20), then clamp.
The component pads **after** that preparation with FP32 zero. It does not
apply a mask to B, C, or dt, and does not reset state at real masked tokens.
`negative_a` is already negative FP32 A, not the learned A_log.

Chunks contain exactly 128 lanes. Log decay uses the pinned four-warp
inclusive-scan association, including the padded lane 127 as chunk total.
The first scan stage and state-passing multiply-add use widened expression
evaluation followed by an FP32 store to reproduce source FMA rounding on
the qualified fixture domain. This does not widen the stored/feedback state.
All incoming, running, and final chunk states remain FP32.

Local-state operands round the complete scaled B to BF16. CB dots store FP32
without an early BF16 CB store. Carry readout rounds only a copy of the
incoming state to BF16. Intrachunk dots round the complete causal scaled CB
coefficient, including dt, to BF16. None of these rounded readout operands
is fed back into the carried state.

The returned readout is **before D*x**, with no BF16 output store. The
existing qualified FP32 D-before-final-store suffix must remain outside
this component, followed by the existing gate/norm/output projection.

## Runtime and qualification scope

The candidate uses CUDA `com.microsoft.GemmFloat8` with **BF16 inputs and
FP32 output**, explicitly supported by that operator's schema and verified
on the pinned runtime. It performs no float8 conversion. A nested Scan
implements batched mixed-precision GEMM without a new kernel or dependency.
Ordinary BF16 MatMul would round its result prematurely; widening its
inputs to FP32 MatMul changes the source tensor-core reduction association.
TF32 is disabled in qualification. Exact BF16-to-FP32 widening around Pad
avoids unsupported BF16 Pad kernels without changing operand bits.

The sealed CUDA regressions check all 30 source primitive fixtures, including
B2/H128/G8/head64/state128 multichunk input and finite dt clamp. CB and
cumulative log decay are bitwise gated. Full FP32 intermediate tensors use
an explicit candidate-only `NRMSE < 1e-5` numeric gate; the observed head-zero
scaled-B operand is limited to one BF16 ULP. CUDA Pow and source
`ex2.approx` are not bitwise interchangeable: numeric parity here is **not**
a claim of bitwise output/cache parity or unchanged full-model MoE routing.
The earlier checkpoint-backed 27-token prefix additionally checks both
observed operand stores and the D-before-final-store output.
Every executed CUDA test audits placement, rejecting CPU floating-point
work (host integer shape/count operations are allowed).

CPU tests cover API, ranks/dtypes/dimensions, parameter absence, shape
inference/checking, and the association-sensitive padded-lane counterexample.
The full mixed-precision primitive is CUDA-only; CPU execution is not promised.
This is a correctness candidate, not a performance-qualified implementation.
It does not qualify packed `seq_idx`, cached S>1, full-model generation,
quantization, other runtimes, or other execution providers.

CUDA tests require explicitly configured read-only local evidence and the
existing qualification worker on `PYTHONPATH`:

```bash
export MOBIUS_CHUNK_PREFILL_FIXTURES=/path/to/sealed/chunk-contract
export MOBIUS_CHUNK_PREFILL_EVIDENCE=/path/to/fresh/evidence
export MOBIUS_CHUNK_PREFILL_ORT_BUILD=/path/to/qualified/runtime-build
python -m pytest src/mobius/components/_chunked_prefill_test.py -q
```

The CUDA qualification worker explicitly reserves physical GPU 0. When no
fixture root is configured, only these opt-in evidence tests are skipped.

## Manual-only block/model/task adapter

`ChunkedPrefillMamba2Block(Float32SiLUMamba2Block)` replaces the SSM hook,
not the projection, ConvSiLU, gate/norm, or output-projection modules. Its public
arguments append `has_previous_state` after the existing optional padding mask.
It requires native BF16 activations and FP32 SSM state at graph construction;
dtype is not frozen at module construction. All learned parameter paths,
including native packed projection paths, are preserved.

The explicit unregistered model/task pairs are:

| Model | Task | Decoder |
| --- | --- | --- |
| `NemotronHPuzzleSourceChunkCausalLMModel` | `SourceChunkHybridCausalLMTask` | `model`, token IDs |
| `NemotronHOmniSourceChunkModel` | `SourceChunkNemotronHOmniTask` | `decoder`, embeddings |

For example, using an already obtained Omni config:

```python
import onnx_ir as ir
from mobius import build_from_module
from mobius.models import NemotronHOmniSourceChunkModel
from mobius.tasks import SourceChunkNemotronHOmniTask

config.dtype = ir.DataType.BFLOAT16
module = NemotronHOmniSourceChunkModel(config)
package = build_from_module(
    module, config, task=SourceChunkNemotronHOmniTask(), execution_provider="cuda"
)
```

This constructs unweighted IR; it does not authorize a full weighted package.
Omni's embedding and vision graphs remain unchanged. There is no registry entry,
CLI flag, default model/task replacement, or new attention implementation.

### Caller-owned phase ABI

The task creates the actual decoder graph input **`has_previous_state: BOOL []`**.
Graph introspection emits `decoder.inputs.has_previous_state` only when that
valid scalar input exists. A similarly named config property is not evidence
that a graph needs the input. Wrong dtype/rank/unknown shape is rejected.
Existing graphs and metadata without the actual input retain their old ABI.
GenAI use additionally requires the separately reviewed runtime implementation
of this optional mapping; released runtime support is not asserted here.

* `False`: prefill, even S1 or offset100. Ignore dirty convolution/SSM history,
  prepare FP32 threshold-20/clamped dt, and run the chunk scan with **no initial
  state**. The native cache orientation remains **B,H,state_dim,head_dim**.
* `True`: consume the FP32 cached state, even offset0. Keep the existing FP32
  Softplus/LinearAttention cached path, with **no prefill clamp**. An executed
  cached S>1 branch fails its explicit reshape guard rather than inventing
  unsupported source semantics.

Neither token count, position IDs, offset, mask values, nor nonzero cache
contents select phase. FP32 D*x is added before the sole BF16 output store,
followed by the existing grouped norm/gating order. Convolution keeps the
native K3 cache ABI.

The source's **mask offset is separate from phase**. The ordinary task uses a
2D binary total mask: total length minus query length is its physical cache
position. The direct model interface also appends an explicit INT64
`cache_position` vector for independent controls; its first element disables
Mamba mask updates when positive. Only normalized hidden before projection
and x after ConvSiLU are masked. B/C/dt are not zeroed and real masked tokens
do not reset state. 3D/4D masks are rejected, not silently ignored. These
Mamba checks do **not** qualify masked attention/whole-decoder parity:
attention remains the existing route, including its independent limitations.

`mobius._testing.source_chunk.ExplicitCachePhaseCaller` is a fresh experimental
validation wrapper, not a change to shared/frozen callers. It owns a stable
NumPy BOOL scalar: false initially, true only after the synchronous decoder
callback succeeds. The callback owns atomic cache-output commitment; phase
does not advance on failure, and any managed failure permanently poisons the
caller: no retry/reset. Instantiate a new caller for every fresh
generation. Scheduler/Engine, prompt chunking, rewind, CUDA Graph capture,
contradictory supplied phases, nonbinary masks, and cached S>1 are rejected.

The fresh qualification evidence also contains runnable
`bolt_source_chunk_native_run.py` and `bolt_source_chunk_export.py` entrypoints,
not replacements for the frozen helpers. The ordinary caller validates the
actual session's canonical BOOL[] input and complete 96-cache/97-output ABI,
executes the frozen lossless BOOL/BF16 transport, preserves every output and
hash, handles finite logits and all caches, and only then commits phase.
It retains the original eight-step/776-decoder-output, embedding, prompt
provenance, runtime-library, profile, provider and EOS reporting behavior.
Wrong graphs fail explicitly; no monkeypatch or hidden input fallback is used.
The ordinary entrypoint pins the sealed qualification contract and exact
`MAMBA_ONLY`/native-attention variant. At each real execution boundary it
captures actual embedding input IDs, mask shape/dtype/raw SHA, embedding raw
SHA, and all 96 actual zero/empty/carried cache SHAs. Its separate
`phase-execution.json` follows the sealed template, records success-only phase
transitions, and binds the final ordinary-receipt SHA without a circular hash.
Failure/early-EOS records remain explicitly incomplete; unexecuted template
entries are never presented as actual feeds. Candidate declarations remain
templates until a reviewed actual export/permitted run supplies genuine
graph, export, inventory, external-reference and runtime identities.
The export entrypoint defaults to printing a plan without graph/weight/package
creation. Package writing additionally requires exact independently reviewed
HEAD/resource-plan hashes and the separately reviewed bounded streaming-save
API, with one worker and explicit 1-GiB shards. Neither a full export nor
ordinary full-model run has been performed as part of this adapter task.

### Adapter validation and important numerical limits

Tests cover tiny text/three-component Omni graphs, unchanged ordinary graphs,
parameter paths including packed projections, deferred dtype, positional API,
graph-derived metadata, caller success/failure, full tiny-block dirty caches,
opposite phase/offset controls, and all **70 checkpoint mask/cache pairs**.
Those checkpoint tests exercise the saved head-zero **post-convolution SSM,
readout/D and cache-transition boundaries**, not unsaved whole-block outputs.
Cached results are also bitwise compared against the existing FP32 cached path.

The new H128 production and finite-clamp zero-initial-state controls verify
**bitwise integration with the accepted standalone primitive on identical
prepared operands**, exact source CB/padded-cumsum boundaries, FP32 final
state, and D-before-native-store. They also retain raw and input-conditioned
Triton comparisons as explicitly labelled diagnostics. They do not weaken the
sealed standalone `1e-5` gate or turn failed source comparisons into passes.

There are additional qualification limits: native/source FP32 dt preparation
can differ by about `1.2e-7`, arbitrary synthetic negative A is not recovered
bitwise through `log(-A)` followed by Exp, and Pow versus source approximate
exp2 can change BF16 readout operands. With zero incoming state, cancellation
exposes errors hidden by larger nonzero carry: these additional raw-source
readout controls exceed `1e-5` (about `4.74e-5` and `3.32e-5` NRMSE).
Even matched-preparation finite-clamp readout can exceed that gate. Final
BF16 differences can be magnified near zero by D/readout cancellation and
source FMA. **This candidate is not an all-state bitwise or full-model pass.**
The added matched-preparation zero-initial finite-clamp final-state diagnostic
also exceeds `1e-5` (about `1.88e-5` NRMSE); it remains labelled a failure,
not a qualification pass. These are additional diagnostic controls, not
replacements for the sealed primitive fixtures.
DOUBLE temporaries remain limited to the standalone association/FMA expression;
stored, incoming, and feedback states remain FP32.

Adapter CUDA tests additionally require **separate immutable inputs and fresh
outputs**. Set `MOBIUS_CHUNK_ADAPTER_FIXTURES` to the sealed
`source-chunk-adapter-v1` directory containing the original-source
zero-initial/prepared-input diagnostics. Its manifest SHA256 is pinned in the
tests; every consumed diagnostic JSON/NPZ is hash-verified against it. Set
`MOBIUS_CHUNK_ADAPTER_EVIDENCE` to a different, fresh output directory.
Never copy newly generated outputs into the source fixture root or relabel
them as independent source fixtures. The original receipt/fixtures remain
read-only. `MOBIUS_CHUNK_ADAPTER_GPU` selects the explicitly allocated physical
GPU for adapter tests (default 0); recheck availability before running.

For just the two production controls, after setting the common fixture/runtime
environment above and allocating the requested GPU:

```bash
export MOBIUS_CHUNK_ADAPTER_FIXTURES=/path/to/sealed/source-chunk-adapter-v1
export MOBIUS_CHUNK_ADAPTER_EVIDENCE=/path/to/fresh/adapter-evidence
export MOBIUS_CHUNK_ADAPTER_GPU=1  # Only when physical GPU 1 is allocated/free.
python -m pytest src/mobius/components/_chunked_mamba_test.py \
  -k production_prefill_primitive_integration_and_source_diagnostics -q
```

Source generation commands/scripts and hashes stay in the sealed input evidence;
new execution profiles/results belong only in the fresh output directory.
Do not reuse either the standalone or original adapter receipt path.
No full package, generation, performance, memory-budget, or shipping-readiness
claim follows from these component/task tests.
