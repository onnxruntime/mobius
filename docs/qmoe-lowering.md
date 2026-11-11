# Explicit INT4 QMoE lowering (opset 29)

```python
import onnx_ir as ir
from mobius._flags import override_flags
from mobius.functions import lower_qmoe

model = ir.load("native.onnx")
with override_flags(experimental_opset29=True):
    lower_qmoe(model)  # in-place, also returns model
ir.save(model, "standard.onnx")
```

**Experimental, disabled by default.** Enable `experimental_opset29` explicitly,
or set `MOBIUS_EXPERIMENTAL_OPSET29=1` before importing mobius. A QMoE conversion
without this opt-in raises `ValueError` before changing the model. Models without
QMoE remain unchanged, regardless of the flag.

The flag permits experimental lowering; it does not automatically upgrade models
or imply ORT/EP kernel support. Schema/checker support alone does not establish
runtime execution support. In particular, the tested ORT 1.31.1 cannot execute
these opset-29 graphs directly.

This opt-in pass replaces `com.microsoft::QMoE` with standard ONNX blocked
`DequantizeLinear` and two `GroupedMatMul-29` calls. When at least one node is
lowered, it upgrades the model and compatible existing function/subgraph imports
to 29 after a non-mutating compatibility audit. It is not automatically run during
export. Native fused defaults, the global opset 24, float MoE, initializer names,
and checkpoint loading are unchanged. Repeated calls are harmless.

## Initial qualified ABI

* FLOAT or FLOAT16 activations, with matching routing/scales dtype; rank-2
  `[T,H]` or rank-3 `[B,S,H]` hidden input, static H and dynamic token dimensions.
  For rank-3 input, `T = B*S`: token dimensions are flattened for expert
  computation and restored after the weighted sum; router inputs remain `[T,E]`.
  The raw CPU reference uses FLOAT for dequantization, both GEMMs, activation,
  routing, and accumulation, casting to FLOAT16 only at the final output.
* `quant_type="int"`, `expert_weight_bits=4`, explicit `weights_prepacked=0`,
  uniform projection bit widths; `accuracy_level` absent, 0, or 1 (FLOAT GEMM).
* Raw UINT8 packed weights `[E,2I,H/2]`, `[E,H,I/2]`, low nibble first.
  Static expert/projection sizes, power-of-two block size >=16 dividing H and I.
* Block scales `[E,N,K/block_size]`, optional packed UINT8 zero points
  `[E,N,ceil(nblocks/2)]`. Omitted zero points mean **8**, not ONNX DQ's default 0.
  Unpacking trims the spare zero-point nibble for an odd number of blocks.
* `activation_type="swiglu"`, `swiglu_fusion=1`: alternating gate/up FC1 rows.
  Pre-activation clipping, `activation_alpha`, and `activation_beta` are supported:
  `min(gate,L) * sigmoid(alpha * min(gate,L)) * (clip(up,-L,L) + beta)`.
* `1 <= k <= E`; router inputs are `[T,E]` even for rank-3 hidden input.

Bias, separate FC3, other activations, concatenated fusion=2, sparse mixer,
post-activation clipping, row-wise quantization, prepacked/legacy-transposed
weights, mixed bits, FP4/MXFP4, and BFLOAT16 inputs are **not qualified**.
Unsupported inputs/attributes/shapes raise `ValueError` before any graph
mutation, including when another QMoE node is otherwise supported. Shape/type
metadata is required; runtime feeds must obey these shapes and have finite
hidden/scale/explicit routing-weight values.

## Exact routing reference

The reference is the **raw CPU QMoE path**, not a provider-independent
interpretation of the contrib schema:

1. Select the largest k `router_probs` values (these are selection logits);
   ties prefer the higher expert ID, matching CPU's descending pair sort.
2. With input 14 (`router_weights`), gather its selected entries. If
   `normalize_routing_weights=1`, divide by their sum, with a zero sum producing
   all-zero weights. With flag 0, use the gathered entries unchanged.
3. Without input 14, softmax the selected logits, **regardless of the normalize
   flag**. CPU does not use a full-expert softmax in this path.
4. CPU does not enqueue routes whose resulting weight is <=1e-8 (including
   negative weights); the lowering masks their contributions to zero.

CUDA ignores input 14 and has different normalization behavior. This pass must
not be used to claim CUDA-native parity for those cases. All-selected `-inf`,
NaN logits, and nonfinite explicit routing weights are outside the contract.
Standard Softmax remains well-defined for masked `-inf` logits when at least
one selected logit is finite.

The first grouped matmul produces `[T,k,2I]`. After deinterleaving and SwiGLU,
the down projection uses `[T*k,I]` and `[T*k,1]` selected expert indices.
Its `[T,k,H]` result is weighted and summed before restoring the original shape.
This keeps per-route intermediate vectors separate when k>1.

## Import alignment

The ONNX checker requires compatible standard-operator schemas across the
model and its local functions. Lowering a QMoE inside a function therefore also
upgrades the model to 29. Conversely, lowering a top-level QMoE aligns existing
registry and caller function imports rather than leaving them at 24.
Function identities, bodies, attributes, and custom-domain imports are preserved.

Before any mutation, every existing standard node in every scope must either
resolve to the same schema at 29 or have one of these narrowly audited changes:

* Type extensions only: Cast 24→28, CastLike 24→25, Constant/ConstantOfShape,
  Identity, If, Reshape, Shape, Squeeze, Transpose, Unsqueeze 24→25, and
  DequantizeLinear 24→28. Known types must obey the original schema; untyped
  function parameters retain the original valid-type contract.
* Attention 24→25 adds window attributes whose defaults are unbounded; neither
  new attribute may be supplied by an old-schema node.
* Range 11→27 adds half/bfloat inputs and their accumulation attribute; the
  original input types and behavior are unchanged.
* BitShift 11→28 adds signed/out-of-range behavior. Alignment requires an
  unsigned constant shift in `[0,7]`, where every original valid type agrees.

These are audited against the pinned
[official ONNX definitions](https://github.com/onnx/onnx/tree/82721c0908fa965dd84e674c006147791313f3e1/onnx/defs).
Other schema changes (for example Softmax-11's changed default-axis semantics),
unknown operators, future imports, or unprovable shifts raise `ValueError`
**before** lowering any node or changing any import. Caller-owned functions are
not trusted merely because of their name or registry metadata. A model without
QMoE is a byte-preserving no-op, even when it contains older functions.

## Runtime and validation

The output requires ONNX schemas and a runtime implementing **opset 29**.
Older ORT releases cannot directly load it. Tests expand the official
GroupedMatMul function algorithm on a *test copy* to run on prior kernels;
that expansion is not part of the public pass. Packed weights stay under their
original names and are dequantized in the graph; this is a portable reference
lowering, not a promise of fused-kernel memory use or speed. Native kernels may
use approximate/integer-activation GEMMs; numerical comparisons use the FLOAT
dequantized reference, not those provider-specific approximations.

Checker and shape-inference validation used a source build of official ONNX
at the pinned revision below (reporting version 1.24), with the opset-29 schema.
This does not require or imply an available ONNX 1.24 release wheel. Released ONNX
1.23.2 stops at opset 28. ORT 1.31.1 does **not** execute these opset-29 graphs:
even with its development-opset guard disabled, it rejects standard operators
at version 29 before reaching GroupedMatMul. Direct-runtime tests report that
capability limitation explicitly; numerical coverage uses the test-copy
expansion plus independent references and native CPU QMoE comparisons.
Lowering tests require schema availability, and native parity tests depend
on the presence of the CPU QMoE kernel, so older environments can still run the
existing non-opset-29 tests. The public pass reports a clear error if the schemas
needed for its compatibility audit are unavailable.

Authoritative sources (pinned ORT revision
`bbedf7b115be0077a2f4b76fd927fa94a82d0d1a`):

* [CPU routing, packing, and ABI](https://github.com/microsoft/onnxruntime/blob/bbedf7b115be0077a2f4b76fd927fa94a82d0d1a/onnxruntime/contrib_ops/cpu/moe/moe_quantization_cpu.cc)
* [CPU shape checks](https://github.com/microsoft/onnxruntime/blob/bbedf7b115be0077a2f4b76fd927fa94a82d0d1a/onnxruntime/contrib_ops/cpu/moe/moe_helper.h)
* [CPU activation formula](https://github.com/microsoft/onnxruntime/blob/bbedf7b115be0077a2f4b76fd927fa94a82d0d1a/onnxruntime/contrib_ops/cpu/moe/moe_utils.cc)
* [CPU attribute defaults](https://github.com/microsoft/onnxruntime/blob/bbedf7b115be0077a2f4b76fd927fa94a82d0d1a/onnxruntime/contrib_ops/cpu/moe/moe_base_cpu.h)
* [QMoE schema documentation](https://github.com/microsoft/onnxruntime/blob/bbedf7b115be0077a2f4b76fd927fa94a82d0d1a/docs/ContribOperators.md#com.microsoft.QMoE)
* [CUDA QMoE](https://github.com/microsoft/onnxruntime/blob/bbedf7b115be0077a2f4b76fd927fa94a82d0d1a/onnxruntime/contrib_ops/cuda/moe/moe_quantization.cc)
* [GroupedMatMul-29 specification](https://onnx.ai/onnx/operators/onnx__GroupedMatMul.html)
  and [official function algorithm at ONNX revision 82721c0908fa965dd84e674c006147791313f3e1](https://github.com/onnx/onnx/blob/82721c0908fa965dd84e674c006147791313f3e1/onnx/defs/math/defs.cc).
