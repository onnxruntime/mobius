# Drama 3 CUDA 13.0 qualification candidate

The private builder module `mobius.models._fish_omni_cuda13` constructs
**unqualified** native consumer graphs without changing the registered tasks,
FP32 defaults, checkpoint layout, or public reduced-precision admission.
Graph construction and operator availability are not numerical qualification.

Use an already-authorized, pinned checkpoint in a separate builder process.
The target runtime process must not import this module, Mobius, or Torch.

```python
from mobius import build_from_module
from mobius.models._fish_omni_cuda13 import (
    Cuda13GenerationQualificationTask,
    Cuda13QualificationModel,
    native_candidate_manifest,
    saved_native_candidate_manifest,
)

# config: FishOmniConfig, FLOAT16/BFLOAT16, preserved block-FP8 expert storage.
module = Cuda13QualificationModel(config)
package = build_from_module(
    module,
    config,
    task=Cuda13GenerationQualificationTask(),
    execution_provider="cuda",
)
inventory = native_candidate_manifest(package)  # readiness is always false
# Apply the existing streaming weight plan and save into a NEW owned directory,
# never into or underneath checkpoint/source directories.
# saved_native_candidate_manifest(paths) inventories reloaded artifacts and hashes.
```

`Cuda13QualificationTask` constructs the teacher-forced package instead.
Neither task is registered or selected by the CLI. Calling the ordinary public
tasks with this model still rejects reduced precision.

## Selected boundaries

| Boundary | Emitted contract |
|---|---|
| Experts | Existing `com.microsoft::QMoE` v1 slots/storage, plus `fp8_execution_profile="fp16_compatible"`. True F16/BF16 activations/router logits/output, E256/top8, normalized routing, H/I divisible by 128, typed E4M3FN banks and FP32 square-128 block scale grids. |
| Attention | `com.microsoft::CausalRightAttention` v1, typed Q/K/V `[B,H,S/T,256]`, positive finite `scale`, no mask or causal attributes. Right causality is `j <= T-S+i`. |
| Chunk solve | `com.microsoft::UnitLowerTriangularSolve` v1, FLOAT only, square A, equal-rank/exact-prefix B, no broadcasting or attributes. Unit diagonal is implicit; upper triangle ignored; empty dimensions allowed. |

QMoE means **FP8 storage and dynamic activation quantization with register-local
FP16 MMA**, not native hardware FP8 MMA. The runtime default `native` profile is
unchanged when this adapter is not selected.

Packed attention Q/new KV are reshaped and transposed. Real past KV are
concatenated before repeating KV heads in source order. Present caches remain
unexpanded rank-4 KV tensors; attended output returns to packed rank 3.
An internal construction token represents the scoped model's unpadded contract.
The adapter rejects arbitrary masks; it does not inspect node/value names or
strip a supplied additive mask.

The adapter registers these two native shape/type contracts using
`onnx_shape_inference.registry`'s documented extension API when a native handler
is absent. Attention returns Q's shape/type; the unit-lower solve returns B's
shape/type after checking exact prefixes. This supplies missing builder
inference without replacing a pass, suppressing a warning, or advertising a
runtime capability. Existing dependency handlers take precedence.

Before **every** execution the caller must validate shared contiguous nonnegative
position rows, an all-one INT64 attention mask, dynamic KV append length equal to
the position start, matching cache shapes/dtypes, and:

- Exactly CUDA **13.0**, SM90, no CUDA graph capture or static/mixed-position cache.
- F16/BF16 activations and KV caches; B1..8, H1..32, head width 256.
- Nonempty queries, `1 <= S <= T <= 256`.

The builder's `validate_cuda13_context` checks this input contract, but accepting
a caller-supplied runtime identity does not prove that identity. The independent
runtime harness must measure loaded runtime/provider/library identities.

## Inventory and remaining gates

`native_candidate_manifest` reports actual emitted operator counts and required
features, including custom operators nested in control-flow graphs. The saved
variant reloads graphs and hashes the graph and external tensor files. Inventory
must remain separate from immutable source DATA and measured runtime selection;
it never marks a model ready or qualified.

The initial intended profile is `ORT_ENABLE_ALL` with no disabled optimizers.
Historical `RotaryEmbeddingFusion` rounding risk is recorded separately, not
treated as a verified CUDA 13.0 requirement. Only actual optimized-graph and
full-vector evidence can justify a per-profile disabled optimizer, identically
selected for audit and performance while retaining ALL. Constructing a graph
does not configure an ORT session or establish the optimizer/provider profile.
Portable ordered row
and matrix reductions remain selected. Optional native ordered operators are
off. Portable Softplus is not a native Log1p/source-bit guarantee.

Admission still requires supported production kernels on exact CUDA 13.0,
saved/reloaded real-weight full-vector comparison under the **same**
ALL/provider/options/library/disabled-optimizer profile used for performance,
actual target-generated frames through the FP32 codec, and unchanged end-to-end
RTF <= 1.0. Historical CUDA 12 results, CPU adapter tests, matching frame hashes
alone, and wheel availability do not satisfy these gates. Existing independent
oracles and numerical tolerances must not be changed to obtain a pass.
