# INT4 QMoE function fallback (experimental opset 29)

```python
from mobius import build_from_module
from mobius._flags import override_flags

# module/config use the normal mobius model and configuration APIs.
with override_flags(experimental_opset29=True):
    package = build_from_module(module, config, execution_provider="onnx-standard")
model = package["model"]
```

**Experimental, disabled by default.** Enable `experimental_opset29` explicitly,
or set `MOBIUS_EXPERIMENTAL_OPSET29=1` before importing mobius. With this flag,
EPs without fused MoE support build new graphs and ordinary functions at opset
29. The existing factory → `register_function_bodies` → EP-controlled `InlinePass`
route expands native QMoE calls into blocked `DequantizeLinear` and two
`GroupedMatMul-29` calls. There is no separate `lower_qmoe` API or import auditor.

Except for the QNN profile correction below, flag-off builds stay unchanged.
Native-capable EPs keep opset 24/native calls without running the restricted
fallback validator. The global default
`OPSET_VERSION` remains 24; the flag selects the per-build override described above.
Generic dense MoE implementations, gate algebra, checkpoint loading, and packed native
input lists are unchanged. The flag does not imply runtime support:
construction and registration do not require installed opset-29 schemas.
Checking and execution require schemas and a runtime implementing these operators;
ORT 1.31.1 cannot execute these opset-29 graphs directly. This is a portable CPU
semantic reference, not a promise of fused-kernel memory use or speed.

**QNN exception, with the flag off or on:** this PR corrects QNN's profile to
`supports_fused_moe=False`, because QNN has no ORT contrib MoE/QMoE kernels.
That changes existing capability-based model dispatch even with the flag off:
Qwen4 selects `_standard_dispatch`, and eligible unquantized Gemma4 configurations
that previously selected native MoE use the existing dense path instead.
Flag-off QNN still builds at opset 24 without registering these MoE/QMoE
fallback bodies. This capability correction does not qualify QNN device
execution of opset 29 or quantized Gemma4.

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

## Registration and import scopes

Advanced callers may use `register_function_bodies(model, moe_fallback=True)`
under the flag on an existing **opset-29** model. An opset-24 model fails with
**“rebuild under experimental opset29”**. No imported model or caller function
is converted or relabeled. Containing function roots must import 29 even when
lexical child graph hints say 29. IR's normal inliner rejects both import-mismatch
directions between functions and models.

All native calls in the graph, lexical subgraphs, and local functions are
validated and all specializations built before target overload/function mutation.
`qmoe(QMoEFunctionConfig(...))` creates a fresh `ir.Function` with **15 native
positional formals**; router weights remain **input 14**. Optional zero-point and
router-weight masks are part of the frozen config along with dtype, dimensions,
and attributes. Configs contain no token counts, names, or tensor data.
Native input lists, weights, and attributes are preserved.
Direct factory configs are validated before body construction, including positive
integer dimensions, top-k bounds, block divisibility, supported dtype, finite
alpha/beta, and nonnegative non-NaN limit. Alpha, beta, and limit are canonicalized
to ONNX FLOAT (binary32) for body constants and identities; registered calls remain
idempotent after saving and loading.

Deterministic full-config hash overloads, stable infinity encoding, and existing
function-body ownership metadata provide idempotence and model isolation.
Caller-owned source or destination functions cannot be overwritten or
specialized; unrelated user overloads remain unchanged. Output metadata may be
absent; known dtype/rank/static-shape contradictions fail before specialization,
while compatible symbolic/static refinements are preserved. The existing inline
boundary preserves output names, annotations, docs, and metadata. No-target
registration uses normal ordinary-function registration, not a byte-preserving
no-op. See also [the float MoE boundary](moe-lowering.md).

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
