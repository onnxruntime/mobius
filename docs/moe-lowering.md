# Explicit native float MoE lowering (opset 29)

```python
import onnx_ir as ir
from mobius._flags import override_flags
from mobius.functions import lower_moe

model = ir.load("native.onnx")
with override_flags(experimental_opset29=True):
    lower_moe(model)  # in-place, also returns model
ir.save(model, "standard.onnx")
```

**Experimental, default off.** The same permission as `lower_qmoe`,
`experimental_opset29` (or `MOBIUS_EXPERIMENTAL_OPSET29=1` before import),
allows this explicit pass. The flag alone does not run either pass, register a
MoE function body, or change ordinary native exports, global opset 24, generic
MoELayer dispatch/fallbacks, configuration, or checkpoint loading.

This replaces `com.microsoft::MoE` with two standard `GroupedMatMul-29` calls,
targeting the pinned **CPU reference**, not all execution providers. It requires
ONNX opset-29 schemas and a runtime implementing them. ORT 1.31.1 cannot execute these
graphs directly; tests expand GroupedMatMul on a test copy at opset 24. There is
no production expansion or performance/memory-use promise.

## Supported ABI

* Input 0: FLOAT or FLOAT16 hidden states `[T,H]` or `[B,S,H]`, positive static
  H. Dynamic token axes and empty token dimensions are supported; `T=B*S`.
* Input 1: matching-dtype router **logits**, named `router_probs`, `[T,E]`.
* Input 2: matching-dtype modern row-major W1 `[E,N,H]`.
* Input 3: optional matching-dtype bias1 `[E,N]`.
* Input 4: matching-dtype modern row-major W2 `[E,H,I]`.
* Input 5: optional matching-dtype bias2 `[E,H]`.
* Positive static E, I, H; `N=2I` for SwiGLU, otherwise `N=I`; `1<=k<=E`.
  Square non-SwiGLU projections are qualified as modern row-major, consistent
  with the pinned helper's legacy-layout predicate.
* Explicit `activation_type`: `relu`, `gelu` (TANH approximation, **not erf**),
  `silu`, `identity`, or `swiglu`. SwiGLU requires `swiglu_fusion=1` and
  alternating gate/up W1 rows. Other activations require fusion absent or 0.
* SwiGLU is
  `min(gate,L)*sigmoid(alpha*min(gate,L))*(clip(up,-L,L)+beta)`.
  Defaults: alpha=1, beta=0, L=infinity. Alpha/beta must be finite; L must be
  nonnegative and not NaN (infinity allowed). As in CPU, these attributes have
  no effect on non-SwiGLU activations.
* `normalize_routing_weights` defaults to **0** and must be 0 or 1.
  `use_sparse_mixer` and `swiglu_limit_after_activation` must be absent or 0.

FC3 inputs 6/7, extended slots, overloads, unknown attributes, GEGLU,
fusion=2, sparse mixer, post-activation clipping, legacy-transposed layouts,
mixed dtypes, BFLOAT16, malformed tensor metadata, and missing input shape/type
metadata are rejected before mutation. Output metadata is optional (native
custom-op intermediates may be untyped/unshaped); missing metadata is inferred
from X. Existing output metadata must agree with hidden input rank, known
dimensions, and dtype, and is preserved.

## Routing and arithmetic

Unlike QMoE, native CPU MoE converts half router inputs to FLOAT, softmaxes
**all E logits before TopK**, and selects k probabilities. Ties in those FLOAT
probabilities (including probability-rounding ties) prefer the **higher expert
ID**. The lowering reverses the expert axis before ONNX TopK and maps back.
With normalization enabled, selected full-softmax values are renormalized;
otherwise their sum may be less than one.

CPU enqueues routes with weight **>0**, not QMoE's `>1e-8`. The lowering retains
tiny positive weights and uses `Where` to mask inactive contributions, rather
than allowing `0*NaN` to contaminate the output. Router feeds must be finite or
masked `-inf` with at least one finite logit per row. All-infinite/NaN logits
and other out-of-contract feeds need not be runtime-validated. Hidden states,
weights, and biases must contain finite values.

FLOAT16 CPU arithmetic rounds GEMM1 output to half, then bias1 addition to
half, evaluates activation in FLOAT and rounds it to half, rounds GEMM2 output
to half, and then bias2 addition to half. The lowering computes in FLOAT with
explicit half round-trips at each of these boundaries. Route weighting and
accumulation remain FLOAT; only the final output is cast to half. This differs
from QMoE's FLOAT-through-both-GEMMs behavior. CPU thread/expert accumulation
order and math-library implementations may differ; bitwise equality is not
claimed.

This defines a FLOAT-accumulation reference, not parity with every
platform-specific MLAS half-GEMM kernel. Near a top-k boundary, different Softmax
implementations may round probabilities differently and select different experts.
The higher-ID tie rule applies to equal probabilities produced by the graph.

The first GMM maps `[T,H]` to `[T,k,N]`; activation produces `[T,k,I]`.
The second GMM consumes `[T*k,I]` and expert indices `[T*k,1]`, then reshapes
to `[T,k,H]`. After optional bias2, route weighting, and reduction, the result
is restored using `Shape(X)` with `allowzero=1`, including empty rank-3 axes.
Original initializer names and input/output names, shape/type, documentation,
and output metadata are preserved.

## Native runtime qualification

The tested ORT **1.31.1** wheel reports build commit
`6af1a932e2286d5c58166085d24c68783033edee`. Its CPU MoE kernel registration
accepts **FLOAT only**, not FLOAT16. ORT automatically inserts FLOAT casts for
half X, router, and weights, executes FLOAT MoE, then casts the output to half.
Even successful normalized-half execution therefore does **not** validate the
native half-stage rounding described above.

That wheel also has an older FLOAT kernel aliasing bug: it copies hidden input
only when `normalize_routing_weights` is enabled, then clears the output.
Because its kernel permits `MayInplace(0,0)`, clearing an output aliased to an
internally inserted FLOAT input buffer destroys the hidden states when
normalization is disabled. External feed arrays can remain unchanged while
that internal buffer is corrupted. A bias-free identity/k=1 example with
X=`[1,2,3,4]`, tied logits, and expert output `[1,2,3,6]` incorrectly returns
zeros instead of `[1/3,2/3,1,2]` through the half-to-FLOAT fallback.
The [exact wheel source](https://github.com/microsoft/onnxruntime/blob/6af1a932e2286d5c58166085d24c68783033edee/onnxruntime/contrib_ops/cpu/moe/moe_cpu.cc)
contains the normalization-dependent copy. The
[pinned reference source](https://github.com/microsoft/onnxruntime/blob/bbedf7b115be0077a2f4b76fd927fa94a82d0d1a/onnxruntime/contrib_ops/cpu/moe/moe_cpu.cc)
instead correctly copies whenever input and output buffers alias.

Native parity tests inspect the CPU kernel's **actual T dtype constraint**
and explicitly skip unsupported dtypes, rather than accepting automatic
FLOAT fallback or marking discrepancies as expected failures. FLOAT native
comparisons remain active. FLOAT16 verification here is reference-based, using
the pinned CPU source and independent rounded-stage NumPy reference, with
test-copy GroupedMatMul expansion; every independent FLOAT16 test remains
active, including the proof that intermediate half rounding is observable.
This is not a claim of native-half parity against the 1.31.1 wheel.

## Atomicity and import alignment

All target calls, all existing scopes, and every detached replacement are
validated/generated before any graph replacement or import mutation.
Caller-defined MoE functions cannot be overridden. Calls nested in subgraphs
and local functions are supported. Existing function identities and custom
imports are preserved. The bounded audit is shared with `lower_qmoe`; see
[its import-alignment table](qmoe-lowering.md#import-alignment). No audited
schemas were widened for this pass; arbitrary graphs are not silently relabeled.
Default-domain aliases, explicit node versions, and function/subgraph imports
are aligned under the same rules.

Models without MoE are byte-preserving no-ops regardless of the flag, including
QMoE-only models and models with older unaudited functions. Repeated calls are
harmless. `lower_moe` and `lower_qmoe` compose in either order.

## Pinned authorities

ORT revision `bbedf7b115be0077a2f4b76fd927fa94a82d0d1a`:

* [Native CPU execution and routing](https://github.com/microsoft/onnxruntime/blob/bbedf7b115be0077a2f4b76fd927fa94a82d0d1a/onnxruntime/contrib_ops/cpu/moe/moe_cpu.cc)
* [Shape/layout checks](https://github.com/microsoft/onnxruntime/blob/bbedf7b115be0077a2f4b76fd927fa94a82d0d1a/onnxruntime/contrib_ops/cpu/moe/moe_helper.h)
* [Activation formula](https://github.com/microsoft/onnxruntime/blob/bbedf7b115be0077a2f4b76fd927fa94a82d0d1a/onnxruntime/contrib_ops/cpu/moe/moe_utils.cc)
* [Attribute defaults](https://github.com/microsoft/onnxruntime/blob/bbedf7b115be0077a2f4b76fd927fa94a82d0d1a/onnxruntime/contrib_ops/cpu/moe/moe_base_cpu.h)

ONNX revision `82721c0908fa965dd84e674c006147791313f3e1`:
[GroupedMatMul schema/function](https://github.com/onnx/onnx/blob/82721c0908fa965dd84e674c006147791313f3e1/onnx/defs/math/defs.cc).
