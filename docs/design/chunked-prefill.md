# Standalone chunked prefill candidate

`mobius.components.ChunkedPrefillScan` is a parameter-free `nn.Module`.
It is **not selected by any model, task, registry, or default export**.
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

## Proposed next-phase adapter (not implemented)

An opt-in `Float32AccumulationMamba2Block` subclass can preserve all
projection/conv/norm modules and their parameter names, replacing only its
prefill SSM evaluation:

1. In `components/_mamba_block.py`, expose an SSM evaluation hook after
   the existing post-convolution x/B/C split. Proposed inputs are
   `(op, x_hidden, b_mat, c_mat, dt_raw, ssm_state, has_previous_state)`.
   Leave existing cached dt preparation/recurrence unchanged; the opt-in
   prefill branch prepares threshold-20/clamped FP32 dt explicitly.
   Reshape x/B/C for this component. Reshape FP32 readout back to B,T,d_inner
   before calling the existing `_add_ssm_skip`.
2. For source-model prefill, omit `initial_state`: source prefill ignores
   dirty prior SSM/conv caches. Adapt state orientation explicitly:
   this component uses head_dim,state_dim; the existing native cache uses
   state_dim,head_dim. Preserve the independently qualified conv adaptation.
3. Add explicit phase/cache-validity metadata to the task/block interface,
   derived from the actual source `cache.has_previous_state` lifecycle.
   Never infer it from T==1 or a position offset. Propagate it through the
   model's decoder-layer interface, not an exporter-only heuristic.
   Concretely, `tasks/_causal_lm.py::HybridCausalLMTask.build` currently
   provides input IDs/mask/positions and mixed caches, but no phase input.
   Its opt-in successor should pass a scalar BOOL `has_previous_state`
   through the model and recurrent decoder layers. Keep
   `tasks/_cache_utils.py::_make_hybrid_cache_inputs`'s FP32 SSM ABI
   and sparse mixed-cache slot names intact. A model-specific caller must
   supply real lifecycle metadata; no task-level substitute is equivalent.
4. Keep the current FP32 cached recurrence unchanged and reject source
   cached S>1 until separately qualified.
5. Replay the checkpoint-backed mask/cache transitions and unchanged frozen
   full-logit/state/hidden/token gates before any integration or shipping.

No part of this adapter, phase selector, or export wiring is included here.
