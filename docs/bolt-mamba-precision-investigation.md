# External Bolt Mamba: precision-boundary investigation

**Investigation only; not an exporter or CUDA-kernel fix.** A reduced replay of
an externally supplied `bolt::MambaVerify` binary confirms two differences from
the existing BF16 diagnostic reference. It does not establish that either
difference causes the full decoder's logit failure, nor that the diagnostic's
convolution backend is the intended contract for this binary.

The external package and its CUDA implementation are not part of Mobius.
No model weights, extracted kernels, or third-party binaries are included here.

**Source follow-up:** the owning public repository was subsequently located.
[nemotron_ops PR #1](https://github.com/tianleiwu/nemotron_ops/pull/1) now
contains the actual CUDA changes and weight-free executable regressions.
Both precision boundaries are independently opt-in; existing defaults remain
unchanged. All four CUDA policy regressions executed and passed, and all four
rebuilt libraries were replayed through the package's ORT runtime.
This does not establish full decoder parity or generation correctness.

## Identical-input findings

The original compiled operator was invoked using its original layer-zero
parameters and two retained, real projection blocks. Its existing five
outputs expose the convolution activation and FP32 SSM readout without
patching the implementation:

| Boundary | Existing diagnostic reference | Observed external operator |
| --- | --- | --- |
| Convolution to SiLU | BF16 convolution result, then SiLU | FP32 convolution result through SiLU, then BF16 store |
| SSM readout to grouped gated RMS norm | Readout stored as BF16 before normalization | FP32 readout consumed directly by normalization |

The first block starts from zero state. The second block reuses the first
block's returned states, selecting committed row seven. Each block contains
eight real rows; the second is **not** a single-token decode experiment.
Both were executed by `CUDAExecutionProvider` in the package's bundled ORT
1.31.0, with CPU EP fallback disabled, TF32 disabled and CUDA graph capture
disabled. This is an operator diagnostic, not a requalification of the
full captured GenAI execution path.

### Convolution boundary

The CPU control emulates the observed bias-first, four-tap FP32 FMA order.
Each product-plus-accumulator is computed in FP64 and rounded to FP32, which
is sufficient to represent these BF16-operand products exactly before the
FP32 result rounding. Both controls use that same accumulator. The only
intentional difference is a BF16 cast before versus after SiLU.

NRMSE below means `||actual - reference||_2 / ||reference||_2`.

| Block | FP32-through-SiLU NRMSE | BF16-before-SiLU NRMSE | Actual vs FP32 differing elements | Actual vs BF16 differing elements |
| --- | --- | --- | --- | --- |
| Initial eight rows | 0.00000247803 | 0.00167082591 | 1 / 81,920 | 19,882 / 81,920 |
| Reused-state eight rows | 0 | 0.00160225290 | 0 / 81,920 | 20,192 / 81,920 |

There are 19,881 and 20,192 elements at which the two controls predict
different BF16 values. The actual operator matches the FP32-through-SiLU
control at **every one** of those elements. One additional initial-block
element differs from both controls; the CPU transcendental calculation is
not claimed to be bitwise identical to the GPU implementation.

Inspection of the binary's retained PTX independently finds FP32
convolution accumulation and FP32 SiLU before `cvt.rn.bf16.f32`. There is
no pre-SiLU BF16 conversion on that inspected path.

### Readout boundary

This control holds the **actual operator's FP32 readout** and original gate
and gamma fixed. It changes only whether the readout is rounded to BF16
before the common normalization calculation. It does not compare different
SSM recurrence implementations.

The dimensions are 8,192 channels and eight groups: **group size 1,024**,
epsilon approximately `1e-5`. Gate SiLU, gated activations, variance,
normalization and gamma multiplication use FP32; output is stored in BF16.
These are the external operator's dimensions and arithmetic, not the
different production dimensions or precision policy of Lightning exports.

| Block | FP32-readout control NRMSE | BF16-readout control NRMSE | Actual vs FP32 differing elements | Actual vs BF16 differing elements |
| --- | --- | --- | --- | --- |
| Initial eight rows | 0.00000107554 | 0.00291287361 | 1 / 65,536 | 19,644 / 65,536 |
| Reused-state eight rows | 0.00000391864 | 0.00298829161 | 1 / 65,536 | 19,204 / 65,536 |

For each block, all 65,536 actual FP32 readout values change when cast to
BF16. The two normalization controls predict different outputs at 19,645
and 19,203 positions. Actual output matches the FP32 control at 19,644 and
19,203 of those positions, respectively. Small residual CPU/GPU
reduction/transcendental differences remain; the CPU control is not an
independently established bitwise GPU oracle.

The inspected normalization PTX loads readout with `ld.global.nc.f32`,
gates and normalizes it in FP32, and converts only the final result to BF16.
The SSM readout is exposed as FP32 by the original operator interface.

## Reference qualification matters

The retained diagnostic is **not an unmodified native HuggingFace run**.
It explicitly sets `causal_conv1d_fn = None` and supplies a cached
`conv_update` implemented as separate PyTorch `conv1d` and `silu`
operations. With BF16 convolution output, that creates the first boundary
in the table. A fused native causal-convolution backend can have a
different rounding boundary. A mismatch to this forced fallback alone
does not establish that an external fused kernel is wrong.

The retained Mamba Triton source allocates the prefill chunk-scan output
with `dtype=x.dtype` and the cached selective-state-update output with
`torch.empty_like(x)`. For BF16 inputs, this stores readout in BF16 before
the grouped normalization wrapper consumes it. Maintaining FP32 recurrent
state does **not** make the readout tensor FP32. The gated-normalization
kernel then upcasts its input and gate internally.

Before choosing a production correction, pin the intended source/backend
contract and reproduce it with matching packed weights. The available
diagnostic checkpoint also has packed-weight differences from the external
package, including Q8 differences beyond scale dtype. Those are independent
confounders for a full-model comparison.

## Reproduction and retained evidence

The complete local evidence directory is:

```text
/home/mobius/bolt-coworker-mamba-precision-20261022/
```

It contains `probe.py`, `runner.cpp`, the compiled runner, the reduced
`mamba.onnx`, original projection inputs, embedded original small parameters,
raw outputs for all five stages in both blocks, CPU rounding controls,
the ORT provider profile, execution/resource records and source/binary pins.
The separately retained PTX snippets are inspection evidence, not patched
kernels.

The reduced graph has four inputs:

```text
projection: BF16 [8,18560]
past_conv: BF16 [8,10240,3]
past_ssm: FP32 [8,128,128,64]
state_row: INT32 [1]
```

The projection splits into gate `[8,8192]`, convolution x/B/C
`[8,10240]`, and dt `[8,128]`. The original six small initializers are
embedded unchanged; no projection or full model weights are recomputed.
Original operator attributes are copied unchanged.

The outputs are normalized BF16 `[8,8192]`, BF16 convolution states
`[8,10240,3]`, FP32 SSM states `[8,128,128,64]`, BF16 convolution
activation `[8,10240]`, and FP32 readout `[8,8192]`.

The one-shot controller command was:

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /home/mobius/validation_lightning_h100_normfix_20261011/runtime/bin/python3.12 \
  -B /home/mobius/bolt-coworker-mamba-precision-20261022/probe.py
```

The controller intentionally refuses to overwrite a completed execution.
Its exact private-loader/library command is recorded in `execution.json`;
it must not be replaced with the incompatible system ORT. No installation
or download was needed. The external package and previous evidence were
not modified.

| Artifact | SHA-256 |
| --- | --- |
| Original external `model.onnx` | `f8f0cd64226c1c575cc0ada277cb6b9f3411708430eaa69d7a75b3a5a1158cc8` |
| Original external `libbolt_ops.so` | `da5f6a5996d2c5711ba42f7a130f7517c7f5d77045c354e66b1c4af9c3a5d575` |

`pins.json` records the reduced graph, compiled runner, runner/probe source,
projection inputs and retained reference hashes. This document is a
record of local package-specific evidence, not a self-contained public
reproducer: the external package, retained inputs and isolated runtime
must be supplied separately.

## Source-owned intervention

The source changes are based on upstream
`e4f63ffdcb830e2faf5ddb149047953d922a6afb` in
[`tianleiwu/nemotron_ops`](https://github.com/tianleiwu/nemotron_ops).
The public source includes optimizations beyond the supplied binary's
inspected kernel template signatures. Nevertheless, the rebuilt default
matches the supplied binary **bitwise for all five stage outputs in both
retained blocks**. This is fixture-level equivalence, not proof that every
input or full-model execution is unchanged.

The source PR adds two independent CMake options, both defaulting to `OFF`:

- `BOLT_MAMBA_ROUND_CONV_BEFORE_SILU`
- `BOLT_MAMBA_ROUND_READOUT_BEFORE_NORM`

They apply to both `MambaStep` and `MambaVerify`, including the specialized
eight-row convolution and group-size 1,024 normalization paths. They do
not change gamma arithmetic or the FP32 recurrent-state layout. The
readout-only variant preserves convolution activations, convolution
states, SSM states and the diagnostic readout bitwise in both blocks.

Four focused CUDA executable tests cover all option combinations on
discriminating inputs, generic and specialized convolution, raw state
history, and group-size 1,024 and 512 normalization. All four executed and
passed on H100. They require no model weights, Python ORT or protobuf
construction.

All four shared-library variants were additionally rebuilt with CUDA 12.6
and ORT 1.31/API-31 headers and exercised through the original isolated
ORT 1.31 runtime on the retained real operands. Each output is closer to
its declared cast-policy control than to the opposite policy, with
remaining CPU/GPU differences at most one adjacent BF16 value.

The combined variant's cached normalization has two CPU-control
differences, NRMSE `0.000011093073433104719`. The CPU pre-cast values are
zero and one FP32 ulp from their BF16 rounding midpoints. An initial
arbitrary `1e-5` CPU-control NRMSE proxy failed and was preserved; the
follow-up instead reports exact BF16-step distances and discrimination
between the two policies. NumPy reduction/transcendental calculations
are not a bitwise CUDA FP32 oracle. **No full-model gate was changed.**

Additional evidence, including all four library builds, executed CUDA
regressions, raw five-output ORT replays, profiles, unchanged-state
comparisons, midpoint diagnosis and preserved failures, is retained in:

```text
/home/mobius/bolt-coworker-source-rounding-20261022/
```

The initial runtime launch rejected a library built with the unrelated
checkout's newer API-32 headers. The successful libraries use public
ORT v1.31.0 headers pinned to
`402979ac1bef89dcd55add41eaa7277c407599eb`. The original package and its
bundled runtime were not modified.

## Remaining work before a full-model fix can be claimed

The binary did not bundle its CUDA source. Its owning repository has now
been located and the source-owned controls are implemented in PR #1;
a Mobius-side change is not needed to patch this external custom operator.
Neither PR claims that changing these boundaries corrects full-model output.

After agreeing on the intended reference contract:

1. Use the independent source controls in separately labeled full-model
   packages. Preserve the baseline, original weights and cache semantics.
2. Extend the same-operand comparisons with a true single-token
   reused-state case and the full captured execution path. The current
   real-operand ORT replay covers two eight-row blocks only.
3. Compare full prefill and every cached-step logit, hybrid states and exact
   ordinary greedy token sequences against a matching pinned reference.
   Retain the original gates: finite outputs, NRMSE at most 0.03,
   cosine at least 0.999, and complete token identity.
4. Only attribute the full-model failure to these boundaries if the controlled
   full-model experiment demonstrates it. Only then evaluate speculative
   integration or performance.

**Current disposition:** both boundary differences confirmed; contribution
to full decoder failure unproven; opt-in source-owned precision controls
implemented with executed CUDA regressions and ORT replays; full-model
decoder parity and generation for these variants not run.
