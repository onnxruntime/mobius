# Nemotron-H Omni: native CUDA export

The pinned Bolt checkpoint is
`nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB` at
`c6cc2e8ffa353403213d6b9b18aaf8eb684d705a`. It combines C-RADIOv4, an
image/video projector, and an 88-layer Puzzle decoder with heterogeneous
experts: 40 Mamba, 40 MoE and 8 attention blocks. It is not a Lightning
checkpoint alias and has no built-in MTP sidecar.

## Export

```bash
mobius build \
  --model nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB \
  --revision c6cc2e8ffa353403213d6b9b18aaf8eb684d705a \
  --task nemotron-h-omni --dtype bf16 --ep cuda \
  --external-data onnx \
  --output /fresh/path/bolt-native
```

Use a new output directory. The native streaming loader validates every
checkpoint source before committing any of the three package components:
`decoder`, `vision_encoder` and `embedding`. Packed projection shapes,
formats, finite scales and source/target coverage are checked explicitly.
Conditioner statistics and summary indices are validated constants, not
unexplained ignored tensors. NVFP4 E4M3 scale bytes are reinterpreted,
never numerically converted to unsigned integers.

The pinned source has 90 shards, 28,885 tensors and 24,083,745,496 checkpoint
bytes. Assess available storage and runtime memory before downloading.
Preserve checkpoint caches and previous evidence. Export size is not a
measurement of peak runtime VRAM, and the model name is not a measured
24 GB deployment guarantee.

## Required native operator contracts

The mixed-format decoder requires CUDA operators from
[ORT PR #33400](https://github.com/microsoft/onnxruntime/pull/33400).
Stock ORT builds without those operators cannot execute this package.
There is no silent affine-INT2 substitution or full BF16 reconstruction.

| Operator (`com.microsoft`, version 1) | Source weight storage | Arithmetic boundary |
|---|---|---|
| `MatMulGGMLIQ2XXS` | UINT8 `[N,K/256,66]` | FP32 decode, native reconstructed-weight rounding, FP32 accumulation, native output rounding, then optional native bias |
| `MatMulGGMLQ8_0` | UINT8 `[N,K/32,34]` | Preserve FP16 scale bits; round the complete reconstructed weight, not the scale alone; same accumulation/output/bias boundaries |
| `MatMulNVFP4A16` | E2M1 UINT8 `[N,K/2]`, E4M3 UINT8 `[N,K/16]`, FP32 global scale | Local native weight operands, FP32 accumulation, FP32 global scaling, native output; bias is a separate `Add` |

Retaining unrounded FP32 GGML weights until the output cast does not match
the pinned BF16 source. Neither does substituting BF16 scales into
`MatMulNBits`. The generic `MatMulBlockQuantizedFp4Weight` has different
global-scale rounding behavior for larger M; it is not the A16 replacement.

The decoder uses FP32 SSM state. Its grouped gated norm computes gating,
per-group variance, normalization and gamma multiplication in FP32, then
casts once to model dtype. This differs from the Lightning norm's
intermediate rounding boundary.

Puzzle Mamba convolution completes accumulation and bias addition in FP32,
then rounds once before Swish. Its state readout and `D*x` skip are also
combined in FP32 before one native output store. The reusable
`Float32AccumulationMamba2Block` supplies these boundaries only for Puzzle;
the default Mamba/Lightning path is unchanged.

RADIOv4's biased dense projections use `Float32Linear`: MatMul accumulation
and bias addition complete in FP32 before the output is rounded to model
dtype. A native-dtype MatMul followed by Add rounds too early; declaring
BF16 Gemm did not enforce the required boundary on the qualified runtime.
This dense source contract differs from the packed GGML operators'
explicit output-before-bias rounding. It does not change their arithmetic.

The upstream b12x A16 backend targets SM120/SM121; it is not executable
on the H100 validation machine. A source-derived, independently decoded
BF16 reference is a diagnostic reference, not evidence that b12x ran on
H100, that ORT used reconstructed dense weights, or that deployment fits
in 24 GB. Report reference provenance and operator placement separately.
The A16 recipe is explicit; an automatic b12x activation-quantized recipe
is a different reference and must not be silently substituted.

## Image/video and generation handoff

The vision graph accepts already normalized FLOAT32 NCHW pixels and
scalar BOOL `is_video`. Do not normalize twice or temporally pack its
input again. It returns flat model-dtype `image_features`, BOOL
`retention_mask` and INT64 `kept_indices`.

Run the pinned processor, execute vision explicitly, compact the prompt
using its EVS retention metadata, then pass retained features and compacted
IDs to the embedding/decoder pipeline. See
[the source-bound host bridge](nemotron-h-omni-processing.md) for exact
ordering, timestamp preservation, odd-frame handling, cached no-media
decoding and the official mixed-image/video forward limitation.

[GenAI PR #2741](https://github.com/microsoft/onnxruntime-genai/pull/2741)
provides the prepared-feature route. Initial scope is unpadded batch 1,
beam 1. Metadata retains all 88 layer descriptors, including trailing
stateless MoE layers, and maps sparse KV, convolution and SSM cache names
from the graph. Shared buffers and CUDA graph capture are disabled initially.
The pinned generation configuration's EOS IDs are `[2,11]`; do not invent
additional stop tokens.

No fake ORT-extensions processor pipeline is generated or referenced.
Do not call `CreateMultiModalProcessor` for this prepared-feature entry.
Its public input IDs are CPU INT32; the runtime converts them for the
embedding graph's INT64 input. The C++/C entry supports actual BF16 feature
buffers. The current Python NumPy tensor converter does not support BF16;
casting those features to FP16 is not an equivalent workaround.

## Readiness gates

Tiny graphs, component parity, source accounting and export success are
separate evidence levels. They do not establish full-model readiness.
Report source fidelity, full decoder parity, ordinary generation, vision
parity, native execution and speculative integration separately as
`PASS`, `FAIL` or `NOT_RUN`.

Full qualification requires finite outputs, normalized RMSE <= 0.03,
cosine >= 0.999, prefill and reused-state logits/hidden/KV/conv/SSM
comparisons, and exact required greedy tokens. Validate image and video
paths independently. Preserve arrays, profiles, commands, source/runtime
hashes and exact token sequences. Locate the first divergent stage before
changing arithmetic. Do not loosen gates or claim speculative integration,
performance or a 24 GB footprint before ordinary correctness is established.

### Preserved native-v5 qualification snapshot

The native-v5 package and retained runtime-v3 have the following independent
results. That runtime declared checkout
`9c9c43ff4edfe1886ac37a399a75bb1c1e6ea137`, but subsequent binary provenance
investigation found a stale activation object; it was not fully source-matched.
Both the original binary and failures remain preserved. These are not
full-model shipping claims.

| Category | Result |
|---|---|
| Source fidelity | PASS: all 28,885 sources; 28,882 bound targets and 3 validated metadata tensors |
| Native packed CUDA execution | PASS: IQ2, raw Q8 and A16 NVFP4; no dense full-checkpoint reconstruction |
| Ordinary eight-token identity | PASS in standalone ORT and the actual C++ GenAI prepared-feature pipeline |
| Full decoder numerical parity | FAIL: all logits pass, but 308 of 768 state arrays fail unchanged gates |
| Image vision parity | PASS at the pinned processor's ordinary 480x640 output |
| Five-frame video parity | PASS; odd final-frame replication and retention indices checked |
| Seven-frame feature-dependent EVS | FAIL: two different retained positions; compact-feature NRMSE 0.306331 |
| Full multimodal generation | NOT_RUN |
| Speculative integration | NOT_RUN |
| 24 GB deployment footprint | Not established; the GenAI run sampled 27,971 MiB |

The fixed 27-token counting prompt produces
`[1049,1044,1032,1050,1044,1032,1051,1044]`, decoded as `1, 2, 3,`.
The reference uses actual checkpoint values with bounded diagnostic BF16
projection reconstruction, pinned eager text attention and pinned Mamba
Triton kernels with explicitly qualified convolution and FP32-state handling.
It does not run the unsupported native b12x backend on H100.

Decoder comparison covers prefill and seven reused-state steps. All 776
logit/state arrays are finite. The first state gate failure is layer 36's
prefill key cache (NRMSE 0.034269, cosine 0.999415); a later convolution
state reaches NRMSE 0.120708. Exact generated tokens do not override these
failed internal-state gates.

A separately instrumented trace compares 712 hidden arrays while preserving
all 776 original logit/state outputs bitwise. Its first hidden gate failures
are prefill MoE34 and cached step 2 MoE3. At cached MoE3, one selected expert
changes from 163 to 221; replaying the source on identical native inputs
reduces output NRMSE from 0.06398 to 0.0000186. This identifies amplification
of upstream drift, not its primitive cause. Same-input and same-prior-state
localization in layers 0–2 remains required before changing arithmetic.

Video dense features pass before pruning. Applying EVS to identical feature
arrays reproduces the source scores, masks and indices exactly: the observed
selection discrepancy comes from upstream numerical differences, not a
demonstrated EVS implementation error. First-norm standalone BF16 differences
are preserved separately; neither an FP64 oracle comparison nor disabling
fusion at that first norm establishes a uniquely incorrect variance algorithm.
Do not change EVS ordering, precision or acceptance thresholds to conceal it.

Runtime profiles permit host integer shape/count arithmetic. Floating payload
computation remains CUDA; this is not a strict
`session.disable_cpu_ep_fallback=1` qualification. Keep this distinction
explicit when reporting provider placement or CUDA graph eligibility.

### Fresh native-v6 source qualification

Native-v6 was exported from clean Mobius source
`61d2dff1731e2c312217d891ecd11ba3c3f5a1e2`, rebased onto
`b1e9c8e9dec1dfa206183f957e77d07cfb734df2`. Independent fidelity again
validates all 28,885 sources, 28,882 exact bindings and 3 metadata tensors,
with valid external references and no unexpected FP32 downcasts.

Fresh standalone execution produces the same eight tokens. All 784 saved
outputs, including 776 decoder logit/state arrays and eight embedding
outputs, are bitwise identical to the preserved v5 run. Clearing only model
metadata yields identical decoder and vision graphs; external payload hashes
also match. Therefore the decoder and vision failures above remain failures,
not newly passing results.

The embedding graph intentionally differs: absent media now gates placeholder
replacement off, preserving ordinary token 18 embeddings on cached text-only
steps. Tested counting-prompt outputs are identical, but the embedding graphs
are not universally equivalent. Actual v6 C++ GenAI execution independently
matches all eight tokens and preserves eight finite logit arrays, bitwise
equal to v5 for this prompt. Its combined three-session residency, without
a helper vision session, sampled 27,961 MiB; this is still not 24 GB
qualification.

The coherent clean rebased non-integration regression passes 11,142 tests,
with 84 skips and 55 deselections using the standard suite selectors and
four failures reproduced on the exact original base excluded. Subsequent
test-formatting-only changes do not alter exported arithmetic. This broad
regression result does not override the model's failed numerical gates.

### Post-v6 vision correction

Identical-input experiments identify premature BF16 rounding between dense
MatMul and bias as a causal vision discrepancy. The FP32 epilogue reduces
one production projection's NRMSE from 0.00220645 to 0.00002552. A separately
labeled actual-checkpoint graph variant then passes image, five-frame and
seven-frame cases, including exact retention masks and indices. The
seven-frame compact-feature NRMSE is 0.009237, with cosine 0.999957.

The correction is integrated into source, with 170 focused tests passing
without skips and clean typing for the changed components. These variant
results do not relabel the preserved v5/v6 failures. The fresh corrected
source qualification below supersedes the variant for current status.
FP32 operands can increase dense vision weight residency; measure final
memory rather than reusing the earlier footprint observation.

### Fresh native-v7 qualification

Native-v7 was exported from clean Mobius source
`52205332bd4fe488204de0b902b126ba8c188008`, including the dense vision and
Puzzle Mamba corrections. It uses the separately built, source-matched
runtime-v4 activation overlay, not the retained runtime-v3 binary.
The CUDA provider SHA256 is
`394cc8fadb5006635368e96c8cd0fe124462d1a2a6eb67c3b7fc953f11d19c5d`;
the unchanged core library SHA256 is
`62cc0188ef19594c724b3e7f4047b8908fd5852164d56edfd3e08a24734fb4d8`.
Actual mapped provider hashes are checked in the execution receipts.

| Category | Result |
|---|---|
| Source fidelity | PASS: 28,885 sources, 28,882 exact bindings and 3 validated metadata tensors; valid external references and zero unexpected FP32 downcasts |
| Ordinary greedy generation | PASS: all eight expected tokens in standalone ORT and actual C++ GenAI |
| Decoder logits | PASS: 8/8 arrays |
| Decoder hidden states | FAIL: 429/712 arrays pass |
| Decoder KV cache | FAIL: 106/128 arrays pass |
| Decoder convolution state | FAIL: 175/320 arrays pass |
| Decoder SSM state | FAIL: 313/320 arrays pass |
| Image vision parity | PASS: NRMSE 0.007395104, cosine 0.999973093 |
| Five-frame compact video features | PASS: NRMSE 0.010337272, cosine 0.999946613 |
| Seven-frame compact video features | PASS: NRMSE 0.009236995, cosine 0.999957353 |
| Full multimodal generation | NOT_RUN: held until ordinary decoder numerical gates pass |
| Speculative integration | NOT_RUN |
| 24 GB deployment footprint | Not established: actual three-session GenAI execution sampled 30,019 MiB |

All compared arrays are finite. The first failed state is prefill layer 25
key cache (NRMSE 0.033919); decode step 7 layer 40 convolution state reaches
NRMSE 0.070001. The first hidden failure is prefill hidden49; decode step 1
hidden20 reaches NRMSE 0.102489. The separately instrumented hidden graph
preserves all 776 original logit/state outputs bitwise. No reference,
backend or numerical gate was changed to obtain these results. Full
decoder parity therefore remains **FAIL**, despite exact ordinary tokens.

All three vision cases have exact retention masks and indices. These are
actual exported-v7 executions, not mechanically patched graph variants.
Video dense diagnostics use auxiliary same-batch tubelet permutations;
they are not direct unpruned-output fetches. Compact features and retention
are compared directly. Vision's 7.407 GiB observation is a sampled NVML
device peak, not an allocator or process-attributed peak.

The ordinary GenAI diagnostic metadata bounds context to 128 tokens; it
does not establish the source's 262,144-token capability. A prepared
328-token image handoff is not a generation result and exceeds that
diagnostic context bound. Any media probe requires a separately labeled
metadata variant without mutating the qualified package.

The clean corrected-source regression passes 11,208 tests, with 84 skips
and 63 deselections under the documented standard selectors and four
exact-base exclusions described above. All 75 targeted Mamba cases pass,
including eight opt-in actual-checkpoint boundary tests. The L1 failure
from unresolved activation dtype in component-quantized graph construction
is fixed with six deferred-type regressions; the exact `52205332` head's
[remote L1 smoke test passes](https://github.com/onnxruntime/mobius/actions/runs/38819403236/job/116712574716).
These software checks do not override the failed full-model state gates.
