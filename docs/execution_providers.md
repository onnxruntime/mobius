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

# CUDA-optimized: GQA fusion, SkipLayerNorm
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
| **CUDA** | `"cuda"` | ORT CUDA EP. GQA fusion for FP16/BF16. SkipLayerNorm fusion. |
| **DirectML** | `"dml"` | DirectML (Windows GPU). GQA for FP16. RoPE and packed QKV lowered to separate ops. |
| **WebGPU** | `"webgpu"` | ORT WebGPU EP. GQA for FP32/FP16. `Shape` eliminated. INT4 accuracy level 4. |
| **MLX** | `"mlx"` | Apple-silicon MLX plugin EP. GQA for FP32/FP16/BF16 with unpacked Q/K/V and shared KV buffers. |
| **TRT-RTX** | `"trt-rtx"` | NVIDIA TensorRT-RTX. GQA for FP16/BF16. `SkipLayerNorm`/`SkipSimplifiedLayerNorm` expanded via InlinePass (TRT handles these primitives natively). GPU graph capture enabled. |
| **ONNX-standard** | `"onnx-standard"` | Strict ONNX. Zero custom-domain ops — all `com.microsoft` ops (FusedMatMul, SkipLayerNorm, PackedMHA) are expanded to standard ONNX via InlinePass. No GQA or QKV packing. Use for non-ORT runtimes that do not support ORT extensions. |

> **Note:** Passing an unknown EP name raises `ValueError` during
> optimization. Use `ep_registry` from `mobius` to query registered EPs.

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
- `Add + RMSNorm` → `com.microsoft::SkipSimplifiedLayerNormalization` (all except TRT-RTX)
- `decomposed GELU chain` → standard ONNX `Gelu` (all EPs)

These are applied during **Stage 2: Fusion** of the optimization pipeline.

### Tier 3 — EP-specific constraints (required, correctness)

Some EPs cannot execute certain ops. These are handled by **lowering rules**
(Stage 3) or **InlinePass expansion** (Stage 2b):

| Constraint | Affected EPs | Mechanism |
|---|---|---|
| No fused RoPE inside GQA | DML | `SeparateRoPE` rewrite: GQA `do_rotary=1` → explicit `RotaryEmbedding` + GQA `do_rotary=0` |
| No packed QKV in GQA | DML | `UnpackQKV` rewrite: packed GQA → 3 separate `MatMul` projections |
| No `SkipLayerNorm` kernel | TRT-RTX, onnx-standard | `InlinePass`: expands fused ops using their registered `ir.Function` bodies |
| No `FusedMatMul` kernel | onnx-standard | `InlinePass`: expands to `Transpose + MatMul` |
| No `PackedMultiHeadAttention` kernel | onnx-standard | `InlinePass`: expands to block-diagonal attention bias + standard `Attention` |

---

## The `EpCapabilities` Dataclass

Every EP is fully described by a single `EpCapabilities` entry in the
`EpRegistry`. To add a new EP, you add one entry to `_register_builtins()` in
`src/mobius/_execution_providers.py` — no other code changes.

```python
@dataclasses.dataclass(frozen=True)
class EpCapabilities:
    name: str
    gqa_dtypes: frozenset[ir.DataType]               # dtypes where GQA fusion fires
    qkv_pack_dtypes: frozenset[ir.DataType]          # dtypes where PackQKV fusion fires
    supports_fused_rope: bool = True                 # False → SeparateRoPE + UnpackQKV
    supports_skip_layer_norm: bool = True            # False → InlinePass expansion
    supports_fused_matmul: bool = True               # False → Transpose + MatMul via InlinePass
    supports_fused_moe: bool = True                  # False → decompose fused MoE ops
    supports_packed_multi_head_attention: bool = False  # True → PackedMultiHeadAttention kernel
    default_int4_accuracy_level: int = 0             # 0 = highest accuracy; 4 = fastest
    provider_options: dict[str, str]                 # Default ORT GenAI provider options
    enable_graph_capture: bool = False               # GPU graph capture default
```

### Current registry

Built-in EPs are registered by `_register_builtins()` at module import:

```python
EpCapabilities(name="default", gqa_dtypes=frozenset(), ...)
EpCapabilities(name="cpu",     gqa_dtypes={FLOAT}, default_int4_accuracy_level=4)
EpCapabilities(name="cuda",    gqa_dtypes={FLOAT16, BFLOAT16},
               enable_graph_capture=True, provider_options={...})
EpCapabilities(name="dml",     gqa_dtypes={FLOAT16},
               supports_fused_rope=False)
EpCapabilities(name="webgpu",  gqa_dtypes={FLOAT, FLOAT16},
               default_int4_accuracy_level=4,
               provider_options={"enableGraphCapture": "0", ...})
EpCapabilities(name="trt-rtx", gqa_dtypes={FLOAT16, BFLOAT16},
               supports_skip_layer_norm=False, enable_graph_capture=True,
               provider_options={"enable_cuda_graph": "1"})
EpCapabilities(name="onnx-standard",
               gqa_dtypes=frozenset(), qkv_pack_dtypes=frozenset(),
               supports_fused_rope=False,  # not used by default for this EP, since it does not enable GQA fusion
               supports_skip_layer_norm=False,
               supports_fused_matmul=False,
               supports_packed_multi_head_attention=False)
```

Out-of-tree EPs can register at runtime via `register_ep()`:

```python
import onnx_ir as ir
from mobius import EpCapabilities, register_ep

register_ep(EpCapabilities(
    name="my-ep",
    gqa_dtypes=frozenset({ir.DataType.FLOAT16}),
))
```

---

## The Optimization Pipeline

`optimize_model()` in `_optimizations.py` runs a four-stage pipeline on each
model in the package:

```
Stage 1:  Cleanup      EP-agnostic. Always applied.
          ↓ Identity elimination, CSE, dedup initializers,
            constant folding, symbolic shape inference, metadata cleanup.

Stage 2:  Fusion       EP-gated. Promotes standard ops to EP-specific fused ops.
          ↓ GQAFusion, SkipNorm, SkipLayerNorm, GeluFusion
            (each only fires if the EP's capabilities support it)

Stage 2b: InlinePass   EP-gated. Expands custom ops the EP cannot execute
          ↓ using their registered ir.Function bodies (e.g. SkipLayerNorm
            decomposition for TRT-RTX).

Stage 3:  Lowering     EP-gated. Structural rewrites for EP constraints.
          ↓ SeparateRoPE, UnpackQKV
            (each only fires if the EP's capabilities require it)

Stage 4:  Fold         EP-agnostic. Always applied.
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
- Custom ops with ONNX function bodies (`LinearAttention`, `FusedMatMul`,
  `SkipLayerNormalization`, etc.) are emitted with their function bodies intact.
  Any conformant runtime can expand the body if it does not recognise the
  custom op.
- Standard fusions (SkipNorm, FusedMatMul, Gelu) are applied — they produce
  `com.microsoft` ops that are portable thanks to their ONNX function bodies.
- No EP-specific fused ops are emitted (`GroupQueryAttention`, PackQKV are
  not applied).
- Only cleanup, standard fusion, and constant folding run (Stages 1, 2, and 4).

This means the default output is the broadest possible ONNX — maximum
compatibility, minimum performance assumptions.

---

## Adding a New Execution Provider

Because all EP logic is encoded in `EpCapabilities`, adding a new EP is a
three-step change:

**Step 1:** Add an `EpCapabilities` entry to `_register_builtins()` in
`src/mobius/_execution_providers.py`:

```python
EpCapabilities(
    name="my-ep",
    gqa_dtypes=frozenset({ir.DataType.FLOAT16}),
    qkv_pack_dtypes=frozenset({ir.DataType.FLOAT16}),
)
```

The entry is automatically registered at module import. No other registration
is needed — the `EpRegistry` validates EP names at optimization time.

**Step 2:** If the EP has hard constraints not expressible via existing
`EpCapabilities` fields, add a new boolean field and a corresponding lowering
rule in `src/mobius/rewrite_rules/`. Follow the pattern in `_separate_rope.py`
(for pattern rewrite rules) or `_eliminate_shape.py` (for op-elimination
passes). Alternatively, if the constraint can be resolved at graph-build time
(i.e. within a component's `forward()`), query `ep_capabilities()` from
`mobius` and branch on the flag there — no rewrite rule needed.

**Step 3:** Write tests:
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
INFO mobius._optimizations: [EP Trace] Target: cuda, dtype: FLOAT16, role: decoder
INFO mobius._optimizations: [EP Trace] Stage 1: Cleanup (9 passes)
INFO mobius._optimizations: [EP Trace]   Cleanup: 512 → 486 nodes (-26)
INFO mobius._optimizations: [EP Trace] Stage 2: Fusion (5 rule groups)
INFO mobius._optimizations: [EP Trace]   GQAFusion                 : +28 GroupQueryAttention, -28 Attention
INFO mobius._optimizations: [EP Trace]   PackQKV                   : +28 Concat, -56 FusedMatMul
INFO mobius._optimizations: [EP Trace]   SkipNorm                  : +28 SkipSimplifiedLayerNorm, -56 Add, -28 RMSNormalization
INFO mobius._optimizations: [EP Trace]   SkipLayerNorm             : no matches (0 nodes affected)
INFO mobius._optimizations: [EP Trace]   GeluFusion                : no matches (0 nodes affected)
INFO mobius._optimizations: [EP Trace] Stage 2b: InlinePass (expand unsupported custom ops)
INFO mobius._optimizations: [EP Trace]   InlinePass                : no matches (0 nodes affected)
INFO mobius._optimizations: [EP Trace] Stage 3: Lowering (0 rule groups for cuda)
INFO mobius._optimizations: [EP Trace] Stage 4: Constant folding
INFO mobius._optimizations: [EP Trace]   Fold: 374 → 371 nodes (-3)
INFO mobius._optimizations: [EP Trace] Summary:
INFO mobius._optimizations: [EP Trace]   Rule                      | Matched | +Nodes | -Nodes
INFO mobius._optimizations: [EP Trace]   -------------------------+--------+-------+-------
INFO mobius._optimizations: [EP Trace]   GQAFusion                 |      28 |     28 |     28
INFO mobius._optimizations: [EP Trace]   PackQKV                   |      56 |     28 |     56
INFO mobius._optimizations: [EP Trace]   SkipNorm                  |      28 |     28 |     84
INFO mobius._optimizations: [EP Trace]   SkipLayerNorm             |       0 |      0 |      0
INFO mobius._optimizations: [EP Trace]   GeluFusion                |       0 |      0 |      0
INFO mobius._optimizations: [EP Trace]   InlinePass                |       0 |      0 |      0
```

### Reading the trace

- **Matched** = nodes removed/consumed by the rule (the "source" side of each
  rewrite).
- **+Nodes** = new nodes introduced.
- **-Nodes** = nodes eliminated.
- `no matches (0 nodes affected)` — the rule is registered for this EP but
  found no patterns. Common reasons: wrong model type (e.g. GeluFusion on a
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

## EP Validation

Passing an unknown EP name to `optimize_model()` raises `ValueError`:

```python
# In optimize_model():
caps = ep_registry.get(ep)
if caps is None:
    raise ValueError(
        f"Unknown execution provider {ep!r}. Supported: {sorted(ep_registry)}"
    )
```

The `EpRegistry` also exposes `require()` for direct lookup:

```python
from mobius import get_ep

get_ep("cuda")   # → EpCapabilities(name="cuda", ...)
get_ep("rocm")   # → ValueError: Unknown execution provider 'rocm'. ...
```

---

## Design Decisions

### Direct emission vs. rewrite rules

Mobius uses two strategies for custom ops, chosen based on whether the pattern
is **self-contained** or **cross-component**:

| Strategy | When to use | Examples |
|---|---|---|
| **Direct emission** (component emits custom op with `ir.Function` body) | Op is self-contained within a single component — no cross-layer dependencies | `LinearAttention`, `CausalConvWithState` |
| **Rewrite rule** (post-export graph transformation) | Pattern spans multiple components or crosses layer boundaries | `SkipSimplifiedLayerNormalization`, `SkipLayerNormalization`, `GroupQueryAttention`, `BiasGelu` |

**Direct emission** gives silent portability: runtimes that recognise the
kernel use it; others expand the function body. Pattern fragility is
eliminated — there is no pattern to match.

**Rewrite rules** are necessary when the fusion spans graph regions that no
single component can see. The key examples:

#### SkipNorm is a cross-layer pattern

In the standard pre-norm decoder layer, the residual Add + Norm pattern occurs
twice per layer — but one of them **crosses the layer boundary**.

The concrete code is in `DecoderLayer._forward_pre_norm()` at
`src/mobius/components/_decoder.py` (lines 118–151):

```python
def _forward_pre_norm(self, op, hidden_states, ...):
    residual = hidden_states
    hidden_states = self.input_layernorm(op, hidden_states)      # L128: Norm

    attn_output, present_key_value = self.self_attn(op, ...)     # L130-137

    hidden_states = op.Add(residual, attn_output)                # L141: Add #1

    residual = hidden_states                                     # L143: consumer 1 of Add #1
    hidden_states = self.post_attention_layernorm(op, hidden_states)  # L144: consumer 2 → SkipNorm ✓ INTRA-LAYER
    hidden_states = self.mlp(op, hidden_states)                  # L145

    hidden_states = op.Add(residual, hidden_states)              # L149: Add #2 (returned to caller)

    return hidden_states, present_key_value
```

**Add #1** (line 141) has two consumers within the same `forward()` call:
the residual save (line 143) and `post_attention_layernorm` (line 144).
The rewrite rule fuses this into `SkipSimplifiedLayerNormalization` — this is
the **intra-layer** pattern.

**Add #2** (line 149) is returned as the layer output. In the model's layer
loop, the next layer receives it as `hidden_states`. When layer N+1's
`_forward_pre_norm()` runs, the same value flows to line 127 (`residual =
hidden_states`) and line 128 (`self.input_layernorm(op, hidden_states)`).
In the flattened ONNX graph, Add #2 has two consumers — the next layer's
residual path and its `input_layernorm` — exactly the same pattern the
rewrite rule matches. This is the **cross-layer** pattern.

A component emitting SkipNorm directly can only capture Add #1 → line 144
(intra-layer, ~50% of opportunities). It cannot fuse Add #2 → next layer's
`input_layernorm` because that `Norm` call is in a different `forward()`
invocation. Rewrite rules operate on the flattened ONNX graph where layer
boundaries don't exist, so they catch **both** patterns. For a 28-layer
model, this means 56 SkipNorm fusions (2 per layer) instead of 28.

#### GQA fusion requires graph-level visibility

`GroupQueryAttention` replaces the standard `Attention` op plus its
surrounding `RotaryEmbedding`, `MatMul` projections, and KV cache handling.
This spans the Attention component, the RoPE component, and the KV cache
plumbing in the task layer — no single component owns the full pattern.

#### BiasGelu has narrow applicability

`BiasGelu` fuses `Add(x, bias) + Gelu(approximate="tanh")` into a single
kernel. It only matches models that use:
- `FCMLP` (not gated MLP — most modern LLMs use SiLU-gated MLPs)
- Bias enabled (`bias=True` in linear layers)
- Tanh-approximate GELU (`gelu_new`, `gelu_fast` — not standard `gelu`)

In practice this covers ~2–5 model families (GPT-2, possibly Falcon, GPT-NeoX,
GPTJ, Phi). The rewrite rule handles these without requiring component changes.

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

### Why is `"onnx-standard"` separate from `"default"`?

Both target non-ORT runtimes, but they differ in how they handle `com.microsoft`
custom ops:

- **`"default"`** keeps all `com.microsoft` ops in the graph. Custom ops carry
  ONNX function bodies, so a conformant runtime can expand them on demand — but
  the ops are still present in the graph. Some runtimes reject models that contain
  unrecognised op domains, even with function bodies.

- **`"onnx-standard"`** proactively expands all `com.microsoft` ops via
  InlinePass at export time. The output graph contains **zero custom-domain ops**
  — only standard ONNX opset ops. This is the safest choice for runtimes that
  strictly refuse any non-standard domain (e.g. WASM runtimes, embedded
  inference engines, or validation pipelines that fail on unknown domains).

The trade-off: `"onnx-standard"` models may be larger (FusedMatMul expands to
Transpose + MatMul) and slower (no GQA, no packed KV cache). Use `"default"`
when the target runtime can silently expand function bodies; use
`"onnx-standard"` when it cannot.

---

## When to Use `build_context()` vs `optimize_model()` Directly

### The normal path: `build()` / `build_from_module()`

For most users, `build()` and `build_from_module()` are the only entry points
you need. They automatically:

1. Set the build context (`build_context()`) for the duration of graph
   construction so components can query EP capabilities.
2. Call `optimize_model()` on every model in the package after the graph
   is built.

You do not need to call `build_context()` or `optimize_model()` yourself.

### When to call `optimize_model()` directly

If you have an existing `ir.Model` (e.g. from a custom pipeline or a
previously exported graph) and want to apply EP-aware passes without going
through the full `build()` pipeline:

```python
from mobius import optimize_model
import onnx_ir as ir

# Apply CUDA optimizations to an already-constructed model
optimize_model(model, ep="cuda", dtype=ir.DataType.FLOAT16)
```

Note: `optimize_model()` runs **outside** any `build_context()`. This means
that any component that calls `ep_capabilities()` during graph construction
will not see these EP settings unless the graph was built inside a
`build_context()` first.

### When to call `build_context()` directly

If you are building a graph using task-level APIs (e.g. `task.build(module,
config)`) and want to control EP capabilities without going through
`build_from_module()`:

```python
import onnx_ir as ir
from mobius import build_context, ep_registry

caps = ep_registry.require("cuda")
with build_context(caps, ir.DataType.FLOAT16):
    pkg = task.build(module, config)  # components see CUDA capabilities here
```

This is the correct pattern for testing EP-conditional component behaviour,
building multiple graphs with different EPs in the same process, or
integrating mobius into a custom build pipeline.

### Summary

| Use case | API |
|---|---|
| Build from a HuggingFace model ID | `build()` |
| Build from a custom `nn.Module` | `build_from_module()` |
| Optimize an existing `ir.Model` | `optimize_model()` |
| Build graphs with explicit EP context | `build_context()` + `task.build()` |
| Test EP-conditional component logic | `build_context()` (see `src/mobius/_build_context_test.py`) |

---

## Build-Time EP Queries

During graph construction, components can query the active EP capabilities
using the **build context API**. This lets a component branch on the target
EP without requiring the caller to thread capability objects through every
layer.

### Reading the active context

```python
from mobius import ep_capabilities, get_build_dtype
import onnx_ir as ir

def forward(self, op, ...):
    capabilities = ep_capabilities()
    dtype = get_build_dtype()

    if dtype in capabilities.gqa_dtypes:
        # Emit a GQA-specific op for this EP
        ...
    else:
        # Fall back to portable ONNX
        ...
```

The context is automatically set by `build_from_module()` and `build()` for
the duration of graph construction. No explicit passing is required.

### Accessing from outside a build

When called outside any active `build_context`, both functions return safe
defaults: the `"default"` EP capabilities (no fusions) and `ir.DataType.FLOAT`.

### Setting the context manually

Advanced users who build graphs outside the standard pipeline can set the
context explicitly:

```python
import onnx_ir as ir
from mobius import build_context, ep_registry

capabilities = ep_registry.require("cuda")
with build_context(capabilities, ir.DataType.FLOAT16):
    pkg = task.build(module, config)
```

Contexts nest correctly — an inner `build_context` shadows the outer one
and restores it on exit, even if an exception is raised.

### Thread and async safety

Each OS thread and asyncio task has its own independent context stack. Two
concurrent builds with different EPs never interfere:

```python
import asyncio
import onnx_ir as ir
from mobius import build_context, ep_registry

async def build_cuda():
    caps = ep_registry.require("cuda")
    with build_context(caps, ir.DataType.FLOAT16):
        return task.build(module, config)

async def build_dml():
    caps = ep_registry.require("dml")
    with build_context(caps, ir.DataType.FLOAT16):
        return task.build(module, config)

# These run concurrently without interfering
cuda_pkg, dml_pkg = await asyncio.gather(build_cuda(), build_dml())
```

This is implemented using Python's `contextvars` module, which provides
copy-on-write isolation for threads and coroutines.
