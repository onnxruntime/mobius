# Execution Provider (EP) Aware Building

Mobius can generate ONNX graphs optimized for a specific runtime **execution
provider** (EP). The default output is portable ONNX that runs on any
conformant runtime. Passing `execution_provider` to `build()` or
`build_from_module()` activates EP-specific fusions and lowering passes that
make the graph faster — or correct — for that target.

---

## Quick Start

```python
import mobius

# Default: portable ONNX — runs on any conformant runtime
pkg = mobius.build("Qwen/Qwen2.5-7B")

# CUDA-optimized: GQA fusion, SkipLayerNorm, PackedAttention
pkg = mobius.build("Qwen/Qwen2.5-7B", execution_provider="cuda")

# With trace output: see exactly what each rule changed
import logging
logging.basicConfig(level=logging.INFO)
pkg = mobius.build(
    "Qwen/Qwen2.5-7B",
    execution_provider="cuda",
    trace_optimization=True,
)
```

The same API works with `build_from_module()` for custom modules:

```python
from mobius import build_from_module, ArchitectureConfig
from mobius.models import CausalLMModel

config = ArchitectureConfig(...)
pkg = build_from_module(
    CausalLMModel(config),
    config,
    execution_provider="cuda",
    trace_optimization=True,
)
```

---

## Supported Execution Providers

| EP name | Pass to `execution_provider=` | Description |
|---|---|---|
| **default** | `"default"` (built-in default) | Portable ONNX. All custom ops with function bodies are kept as-is — function bodies are the executable fallback. No vendor-specific fusions. |
| **CPU** | `"cpu"` | ORT CPU EP. GQA fusion for FP32. INT4 accuracy level 4. |
| **CUDA** | `"cuda"` | ORT CUDA EP. GQA fusion for FP16/BF16. PackedAttention for FP32/FP16/BF16. CUDA graph support. |
| **DirectML** | `"dml"` | DirectML (Windows GPU). GQA for FP16. RoPE and packed QKV lowered to separate ops. |
| **WebGPU** | `"webgpu"` | ORT WebGPU EP. GQA for FP32/FP16. `Shape` eliminated, INT64→INT32 for gather indices. |
| **TRT-RTX** | `"trt-rtx"` | NVIDIA TensorRT-RTX. GQA for FP16/BF16. `SkipLayerNorm`/`SkipSimplifiedLayerNorm` decomposed (TRT handles these primitives natively). |

> **Note:** Passing an unknown EP name raises `ValueError` before graph
> construction begins. Use `KNOWN_EPS` from `mobius._ep_validation` for the
> canonical set.

---

## Three-Tier Support Strategy

Mobius uses three tiers to balance portability and performance:

### Tier 1 — ONNX Functions (portability by default)

Standard components like `op.Attention` are emitted as ONNX opset 24 ops.
Some custom ops (e.g. `LinearAttention`) are emitted as `ir.Function` values
with domain `com.microsoft` and a full fallback body in standard ONNX ops.

When the default EP is used, **no decomposition occurs**. A runtime that
understands the custom op executes the fused kernel. A conformant runtime that
does not understand it expands the function body. Both paths produce correct
output — portability is automatic.

### Tier 2 — EP-specific fusions (optional, performance)

For EPs that support vendor-specific fused kernels, mobius promotes standard
ops to fused equivalents:

- `Attention` → `com.microsoft::GroupQueryAttention` (CUDA, CPU, DML, WebGPU, TRT-RTX)
- `Add + LayerNorm` → `com.microsoft::SkipLayerNormalization` (all except TRT-RTX)
- `Add + RMSNorm` → `com.microsoft::SkipSimplifiedLayerNorm` (all except TRT-RTX)
- `Add + GELU` → `com.microsoft::BiasGelu` (all EPs)

These are applied during **Stage 2: Fusion** of the optimization pipeline.

### Tier 3 — EP-specific constraints (required, correctness)

Some EPs cannot execute certain ONNX ops. These are handled by **lowering
rules** in Stage 3:

| Constraint | Affected EPs | Rule |
|---|---|---|
| No fused RoPE inside GQA | DML | `SeparateRoPE`: GQA `do_rotary=1` → explicit `RotaryEmbedding` + GQA `do_rotary=0` |
| No packed QKV in GQA | DML | `UnpackQKV`: packed GQA → 3 separate `MatMul` projections |
| No `Shape` operator | WebGPU | `EliminateShape`: `Shape(attention_mask)` → `ReduceSum` + `ReduceMax` |
| INT64 Gather indices | WebGPU | `CastInt64ToInt32`: INT64 gather indices cast to INT32 |
| No `SkipLayerNorm` kernel | TRT-RTX | `DecomposeSkipLayerNorm`: fused ops → primitives |

---

## The `EpCapabilities` Dataclass

Every EP is fully described by a single `EpCapabilities` entry. To add a new
EP, you add one entry to `_EP_REGISTRY` — no other code changes.

```python
@dataclasses.dataclass
class EpCapabilities:
    name: str
    gqa_dtypes: frozenset[ir.DataType]        # dtypes where GQA fusion fires
    packed_attn_dtypes: frozenset[ir.DataType] # dtypes where PackedAttention fires
    supports_fused_rope: bool = True           # False → SeparateRoPE lowering
    supports_shape: bool = True                # False → EliminateShape lowering
    supports_skip_layer_norm: bool = True      # False → DecomposeSkipLayerNorm
    default_int4_accuracy_level: int = 0       # 0 = no INT4; 4 = INT4 w/ accuracy
```

### Current registry

```python
_EP_REGISTRY = {
    "default": EpCapabilities(name="default", gqa_dtypes=frozenset(), ...),
    "cpu":     EpCapabilities(name="cpu",     gqa_dtypes={FLOAT}, ...),
    "cuda":    EpCapabilities(name="cuda",    gqa_dtypes={FLOAT16, BFLOAT16}, ...),
    "dml":     EpCapabilities(name="dml",     gqa_dtypes={FLOAT16},
                               supports_fused_rope=False, ...),
    "webgpu":  EpCapabilities(name="webgpu",  gqa_dtypes={FLOAT, FLOAT16},
                               supports_shape=False, ...),
    "trt-rtx": EpCapabilities(name="trt-rtx", gqa_dtypes={FLOAT16, BFLOAT16},
                               supports_skip_layer_norm=False, ...),
}
```

---

## The Optimization Pipeline

`_optimize()` runs a four-stage pipeline on each model in the package:

```
Stage 1: Cleanup      EP-agnostic. Always applied.
         ↓ Identity elimination, CSE, dedup initializers,
           constant folding, symbolic shape inference, metadata cleanup.

Stage 2: Fusion       EP-gated. Promotes standard ops to EP-specific fused ops.
         ↓ GQAFusion, SkipNorm, SkipLayerNorm, BiasGelu
           (each only fires if the EP's capabilities support it)

Stage 3: Lowering     EP-gated. Decomposes ops the EP cannot handle.
         ↓ SeparateRoPE, UnpackQKV, EliminateShape,
           CastInt64ToInt32, DecomposeSkipLayerNorm
           (each only fires if the EP's capabilities require it)

Stage 4: Fold         EP-agnostic. Always applied.
           Dead-node removal + constant folding after rewrites.
```

### Multi-model packages and `model_role`

`build()` can produce multiple ONNX models (e.g. vision encoder + text
decoder for a VLM). Each model has a **role** that gates certain fusions:

| Package key | Role | GQA fusion? |
|---|---|---|
| `"model"`, `"decoder"` | `"decoder"` | ✅ Yes |
| `"vision"` | `"vision"` | ❌ No |
| `"embedding"` | `"embedding"` | ❌ No |
| `"encoder"` | `"encoder"` | ❌ No |

GQA fusion only applies to decoder-role models. Vision encoders use standard
`Attention` ops; applying GQA there would be incorrect.

---

## The Default EP: Portable ONNX

When `execution_provider` is omitted (or set to `"default"`), mobius produces
**portable ONNX**:

- Standard ops (`Attention`, `LayerNormalization`, etc.) are emitted as-is.
- Custom ops with ONNX function bodies (`LinearAttention`, etc.) are emitted
  with their function bodies intact. Any conformant runtime can expand the
  body if it does not recognise the custom op.
- No vendor-specific fused ops are emitted (`GroupQueryAttention`,
  `SkipLayerNormalization`, etc. are not present).
- Only cleanup and constant folding run (Stages 1 and 4).

This means the default output is the broadest possible ONNX — maximum
compatibility, minimum performance assumptions.

---

## Adding a New Execution Provider

Because all EP logic is encoded in `EpCapabilities`, adding a new EP is a
four-step change touching only one file:

**Step 1:** Add a `EpCapabilities` entry to `_EP_REGISTRY` in
`src/mobius/_builder.py`:

```python
_EP_REGISTRY["my-ep"] = EpCapabilities(
    name="my-ep",
    gqa_dtypes=frozenset({ir.DataType.FLOAT16}),
    packed_attn_dtypes=frozenset({ir.DataType.FLOAT16}),
)
```

**Step 2:** If the EP has hard constraints not expressible via existing
`EpCapabilities` fields, add a new boolean field and a corresponding lowering
rule in `src/mobius/rewrite_rules/`. Follow the pattern in `_separate_rope.py`
(for pattern rewrite rules) or `_eliminate_shape.py` (for shape-related passes).

**Step 3:** Add the EP to `KNOWN_EPS` in `src/mobius/_ep_validation.py` and
add any deny-list entries for incompatible model types.

**Step 4:** Write tests:
- A unit test for each new rewrite rule (graph-level, no ORT required)
- A `test_ep_produces_expected_ops_*` entry in `tests/ep_optimization_test.py`
- An ORT execution test (`test_rewritten_model_runs_with_ort`) for each rule

No changes to components, models, or tasks are needed. EP support is
automatic for any model built from standard components.

---

## Adding EP Support to a New Model

If you are adding a new model architecture to mobius, you **do not need to
think about EPs**. EP support is automatic:

- Standard components (`Attention`, `MLP`, `RMSNorm`, etc.) emit the correct
  op patterns that the EP rewrite rules know how to match.
- Your model class only needs a correct `forward()` and `preprocess_weights()`.
- The EP optimization pipeline runs after graph construction, invisibly.

The only exception is if your model uses a non-standard attention pattern. In
that case, verify that the GQA rewrite rule matches your attention subgraph —
check the trace output if in doubt.

---

## Debugging: Trace Mode

Set `trace_optimization=True` to get step-by-step diagnostic output:

```python
import logging
logging.basicConfig(level=logging.INFO)

pkg = mobius.build(
    "meta-llama/Llama-3.2-3B",
    execution_provider="cuda",
    dtype="f16",
    load_weights=False,
    trace_optimization=True,
)
```

Sample output:

```
INFO mobius._builder: [EP Trace] Target: cuda, dtype: FLOAT16, role: decoder
INFO mobius._builder: [EP Trace] Stage 1: Cleanup (9 passes)
INFO mobius._builder: [EP Trace]   Cleanup: 512 → 486 nodes (-26)
INFO mobius._builder: [EP Trace] Stage 2: Fusion (4 rule groups)
INFO mobius._builder: [EP Trace]   GQAFusion                 : +28 GroupQueryAttention, -28 Attention
INFO mobius._builder: [EP Trace]   SkipNorm                  : +28 SkipSimplifiedLayerNorm, -56 Add, -28 SimplifiedLayerNormalization
INFO mobius._builder: [EP Trace]   SkipLayerNorm             : no matches (0 nodes affected)
INFO mobius._builder: [EP Trace]   BiasGelu                  : no matches (0 nodes affected)
INFO mobius._builder: [EP Trace] Stage 3: Lowering (0 rule groups for cuda)
INFO mobius._builder: [EP Trace] Stage 4: Constant folding
INFO mobius._builder: [EP Trace]   Fold: 374 → 371 nodes (-3)
INFO mobius._builder: [EP Trace] Summary:
INFO mobius._builder: [EP Trace]   Rule                      | Matched | +Nodes | -Nodes
INFO mobius._builder: [EP Trace]   -------------------------+--------+-------+-------
INFO mobius._builder: [EP Trace]   GQAFusion                 |      28 |     28 |     28
INFO mobius._builder: [EP Trace]   SkipNorm                  |      28 |     28 |     84
INFO mobius._builder: [EP Trace]   SkipLayerNorm             |       0 |      0 |      0
INFO mobius._builder: [EP Trace]   BiasGelu                  |       0 |      0 |      0
```

### Reading the trace

- **Matched** = nodes removed/consumed by the rule (the "source" side of each
  rewrite).
- **+Nodes** = new nodes introduced.
- **-Nodes** = nodes eliminated.
- `no matches (0 nodes affected)` — the rule is registered for this EP but
  found no patterns. Common reasons: wrong model type (e.g. BiasGelu on a
  SiLU model), or the pattern was already eliminated in Stage 1.

### Fusion assertions

After Stage 2, if GQA fusion was expected for the `(ep, dtype)` combination
but produced zero `GroupQueryAttention` nodes while `Attention` nodes remain,
a `UserWarning` is emitted:

```
UserWarning: GQA fusion expected for ep='cuda'/dtype=FLOAT16 but found 0
GroupQueryAttention and 28 Attention nodes. The model may run slower than
expected on this EP. Check that the attention pattern matches the GQA rewrite rule.
```

If you see this warning, enable `trace_optimization=True` and check whether
the `GQAFusion` stage shows matches. If it shows `no matches`, the attention
subgraph in your model doesn't match the rewrite rule's pattern — compare your
`Attention` op inputs against `_group_query_attention.py`.

---

## EP Compatibility Validation

Before graph construction, `validate_ep_support(model_type, ep)` is called
automatically by `build()`. It rejects known-incompatible combinations:

```python
from mobius._ep_validation import validate_ep_support

validate_ep_support("mixtral", "dml")
# ValueError: Model type 'mixtral' is not compatible with execution provider
# 'dml': Mixtral (MoE) uses dynamic expert routing unsupported by DML

validate_ep_support("llama", "rocm")
# ValueError: Unknown execution provider 'rocm'. Supported: [cpu, cuda, dml, ...]
```

Current deny-list categories:
- **MoE models on DML/WebGPU** — dynamic Top-K routing + Scatter not supported
- **Mamba/Mamba2 on WebGPU** — Scan op not supported
- **Jamba on TRT-RTX** — hybrid MoE+Mamba not supported

---

## Design Decisions

### Why ONNX functions for fusions, not rewrite rules?

Some custom ops (`LinearAttention`, future `GroupQueryAttention`) are emitted
as `ir.Function` values with a fallback body. This gives silent portability:
runtimes that recognise the kernel use it; others expand the body. Pattern
fragility is eliminated for these ops — there is no pattern to match.

Rewrite rules are used when the transformation is structural (changing the
graph topology) rather than semantic substitution. See
[GitHub issue #100](https://github.com/onnxruntime/mobius/issues/100) for the
full design discussion.

### Why do components stay EP-agnostic?

Components (`Attention`, `MLP`, `RMSNorm`, etc.) emit generic ONNX that
matches the rewrite rules' input patterns. If components emitted EP-specific
ops directly, adding a new EP or model would require modifying components —
multiplying the maintenance surface. The rewrite-rule layer cleanly separates
"what the model computes" from "how to execute it efficiently on EP X."

### Why is `"default"` separate from `"cpu"`?

`"cpu"` activates GQA fusion for FP32 (an ORT CPU EP kernel). `"default"`
does not — it's for runtimes other than ORT that understand standard ONNX ops
but not ORT custom ops. The distinction matters for cross-framework export:
a `"default"` model can be loaded by any ONNX-conformant runtime (TFLite,
CoreML, ONNX Runtime, etc.), while a `"cpu"` model is ORT-specific.
