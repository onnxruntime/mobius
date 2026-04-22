---
name: debugging-memcpy
description: >
  Debug and reduce CUDA Memcpy nodes in ONNX models built by mobius. Use when
  ORT warns about MemcpyFromHost/MemcpyToHost nodes added for CUDAExecutionProvider,
  when profiling shows excessive host-device transfers, or when CUDA graph capture
  fails due to Memcpy. Covers root-cause analysis, op-level attribution, and
  proven fix patterns for the most common offenders.
---

# Skill: Debugging CUDA Memcpy Nodes

## When to use

- ORT emits the warning: `N Memcpy nodes are added to the graph ... for
  CUDAExecutionProvider`
- Profiling shows `MemcpyFromHost` or `MemcpyToHost` latency
- CUDA graph capture fails because of Memcpy nodes
- You want to pre-emptively audit a model for CPU-fallback ops before
  shipping to CUDA EP

## Background: why Memcpy nodes appear

ORT assigns each ONNX node to an execution provider (EP).  When a CUDA EP
node produces a tensor consumed by a CPU-only node (or vice versa), ORT
inserts a `Memcpy` node to transfer data between host and device.  These
transfers:

1. Add latency (PCIe bandwidth is orders of magnitude slower than GPU HBM)
2. Block CUDA graph capture (which requires all ops on a single device)
3. Serialize the pipeline (GPU must wait for CPU, then CPU waits for GPU)

The fix is always one of:
- **Replace the CPU-only op** with a CUDA-supported alternative
- **Precompute the value** at graph-build time (static constant)
- **Restructure the graph** so CPU ops only feed other CPU ops (no
  GPU→CPU→GPU round-trip)

## Quick diagnostic flow

```
1. Build model with execution_provider='cuda'
2. Dump per-model op counts (see §Profiling recipe)
3. Identify CPU-likely ops (see §CPU-only op reference)
4. For each CPU op, trace its consumers
   → If consumer is a GPU op (MatMul, Attention, etc.) → Memcpy source
   → If consumer is another CPU op → no Memcpy (chain stays on CPU)
5. Apply fix pattern from §Fix patterns
6. Re-run op count to verify reduction
```

## Profiling recipe

### Step 1: Build and count ops

```python
import onnx_ir as ir
from mobius import build

pkg = build(model_id, execution_provider='cuda', dtype='float16',
            load_weights=False)

cpu_likely = {
    'Shape', 'CumSum', 'Equal', 'And', 'Not', 'Or',
    'GreaterOrEqual', 'Less', 'Greater', 'LessOrEqual',
    'OneHot', 'Where', 'NonZero', 'Range', 'ConstantOfShape',
}

for name, model in pkg.items():
    ops = {}
    for node in model.graph:
        if node.op_type in cpu_likely:
            ops[node.op_type] = ops.get(node.op_type, 0) + 1
    total = sum(ops.values())
    print(f'{name}: {total} CPU-likely ops: {ops}')
```

### Step 2: Trace consumers of CPU ops

For each CPU-likely op, check whether its output feeds a GPU op:

```python
for node in model.graph:
    if node.op_type in cpu_likely:
        for out in node.outputs:
            consumers = [
                n for n in model.graph
                if any(inp is not None and inp.name == out.name
                       for inp in n.inputs)
            ]
            consumer_types = [c.op_type for c in consumers]
            print(f'{node.op_type} -> {consumer_types}')
```

A CPU op whose consumer is `MatMul`, `Attention`, `GroupQueryAttention`,
`Mul`, `Add`, `Gather` (on large tensors), `Reshape`, etc. is a confirmed
Memcpy source.

### Step 3: Verify with ORT session (requires weights)

If you have weights, load the model in ORT with verbose logging:

```python
import onnxruntime as ort

so = ort.SessionOptions()
so.log_severity_level = 1  # INFO — shows Memcpy insertion details
sess = ort.InferenceSession(
    model_path, so,
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
)
```

ORT logs will show exactly which nodes were placed on CPU and which
triggered Memcpy insertions.

## CPU-only op reference

These ONNX ops typically run on CPU in ORT's CUDA EP, causing Memcpy when
their outputs feed GPU ops:

### Always CPU (shape/index computation)

| Op | Why CPU | Common source |
|----|---------|---------------|
| `Shape` | Extracts tensor dimensions as CPU int64 | Dynamic reshape, Concat for shape |
| `NonZero` | Returns indices of non-zero elements | Sparse masking |
| `Range` | Generates integer sequences | Position indices |
| `ConstantOfShape` | Creates tensor from shape input | Dynamic zero-fill |

### Usually CPU (boolean/comparison)

| Op | Why CPU | Common source |
|----|---------|---------------|
| `Equal` | Bool comparison | Token ID matching, padding detection |
| `Where` | Conditional selection on bool | Masking, scatter |
| `And` / `Or` / `Not` | Bool logic | Combining masks |
| `GreaterOrEqual` / `Less` | Comparison | Causal mask, window mask |

### Context-dependent

| Op | When CPU | When GPU |
|----|----------|----------|
| `CumSum` | INT64 inputs | FLOAT inputs |
| `OneHot` | Dynamic depth parameter | Static depth |
| `Gather` | Small index tensor from CPU op | Large data tensor on GPU |
| `Cast` | To/from bool | Between float types |

## Fix patterns

### Pattern 1: Where(bool, zero, tensor) → Mul

**Problem**: `Where` on a bool condition is CPU-only.  When one branch is
zero and the other is a GPU tensor, this forces a GPU→CPU→GPU round-trip.

**Fix**: Multiply by the bool mask cast to float.

```python
# BEFORE (CPU Where)
zero = op.CastLike(op.Constant(value_float=0.0), tensor)
result = op.Where(is_padding, zero, tensor)  # CPU

# AFTER (GPU Mul)
not_pad = op.CastLike(
    op.Cast(op.Not(is_padding), to=ir.DataType.FLOAT), tensor
)
not_pad = op.Unsqueeze(not_pad, [2])  # broadcast dim
result = op.Mul(tensor, not_pad)  # GPU
```

**Variant** — zero-masking with negative infinity (attention bias):

```python
# BEFORE
attn_bias = op.Where(is_padding, neg_inf, zero_bias)  # CPU

# AFTER — is_padding * neg_inf: True→neg_inf, False→0
is_pad_f = op.CastLike(
    op.Cast(is_padding, to=ir.DataType.FLOAT), hidden_states
)
attn_bias = op.Mul(is_pad_f, neg_inf)  # GPU
```

**When to use**: Any `Where(bool, constant, tensor)` or
`Where(bool, tensor, constant)` where one branch is 0 or a broadcast scalar.

**When NOT to use**: When both branches are non-trivial tensors (e.g.,
`Where(mask, gathered_features, original_embeddings)`).  In that case the
`Where` is genuinely conditional and must stay.

### Pattern 2: Shape-based Reshape → static or [0, 0, -1]

**Problem**: `Shape` is always CPU.  Using it to build a reshape target
(e.g., `Concat(Shape(x, 0), Shape(x, 1), Constant(hidden))`) adds CPU ops
that feed the GPU `Reshape`.

**Fix A** — Use ONNX reshape's dimension-preserving syntax:

```python
# BEFORE (CPU Shape → Concat → Reshape)
batch = op.Shape(x, start=0, end=1)
seq = op.Shape(x, start=1, end=2)
target = op.Concat(batch, seq, op.Constant(value_ints=[hidden]), axis=0)
result = op.Reshape(x, target)

# AFTER (no Shape needed)
result = op.Reshape(x, [0, 0, -1])  # 0 = keep dim, -1 = infer
```

**Fix B** — Use static dimension from config:

```python
# BEFORE
hidden = op.Shape(x, start=2, end=3)
result = op.Reshape(x, op.Concat([-1], hidden, axis=0))

# AFTER — hidden_size is known at build time
result = op.Reshape(x, op.Constant(value_ints=[-1, config.hidden_size]))
```

### Pattern 3: Additive float bias → bool mask with is_causal

**Problem**: `create_attention_bias()` builds a float additive mask using
`CumSum` → `GreaterOrEqual` → `Where` — all CPU ops.

**Fix**: Use `create_padding_mask()` or `create_sliding_window_mask()` to
produce a bool mask, and rely on the Attention op's `is_causal=1` for
causal masking.

```python
from mobius.components import create_padding_mask, create_sliding_window_mask

# Full attention: padding mask only (causality handled by op)
mask = create_padding_mask(op, input_ids, attention_mask)

# Sliding window: bool mask with window constraint
mask = create_sliding_window_mask(op, input_ids, attention_mask, window_size=512)
```

**Caveat**: `create_sliding_window_mask()` still uses `CumSum` internally
for distance computation.  This is unavoidable without a fundamentally
different algorithm.  But it eliminates the `Where` for float conversion
and the `GreaterOrEqual` for causality.

**When to use**: Any attention layer using the standard ONNX `Attention` op
with `is_causal=1`.  Not applicable to `GroupQueryAttention` (GQA handles
masking internally via `local_window_size`).

### Pattern 4: Use GQA's built-in local_window_size

**Problem**: Sliding-window attention requires an explicit mask (CumSum-based)
when using the standard Attention op.

**Fix**: Emit `com.microsoft.GroupQueryAttention` with `local_window_size`
attribute.  GQA handles sliding-window masking internally — no explicit mask
tensor needed.

```python
from mobius.components._attention import GQAContext

# Sliding-window layers get local_window_size
gqa_ctx = GQAContext(
    seqlens_k=seqlens_k,
    total_seq_len=total_seq_len,
    cos_cache=rotary_emb.cos_cache,
    sin_cache=rotary_emb.sin_cache,
    local_window_size=config.sliding_window,  # e.g. 512
)
```

This completely eliminates the CumSum/Less/And mask chain for sliding layers.

**When to use**: When the EP supports GQA (`caps.gqa_dtypes` includes the
build dtype and `caps.supports_fused_rope` is True).  Currently CUDA EP
supports GQA for FLOAT16 and BFLOAT16.

### Pattern 5: Precompute at build time

**Problem**: A dynamic computation produces a value that is actually
deterministic from the model config.

**Fix**: Compute the value in Python and emit it as a constant.

```python
# BEFORE — dynamic Shape extraction
hidden = op.Shape(x, start=2, end=3)
result = op.Reshape(x, op.Concat([-1], hidden, axis=0))

# AFTER — known from config
result = op.Reshape(x, op.Constant(value_ints=[-1, config.hidden_size]))
```

This applies to any value derivable from the config: hidden sizes, head
counts, vocabulary sizes, sequence lengths, etc.

## Inherently CPU ops (hard to eliminate)

Some patterns are fundamentally CPU-bound and require algorithmic redesign:

- **CumSum on INT64**: Used for computing occurrence indices (e.g., "this is
  the 3rd image token").  No GPU-friendly alternative for arbitrary masks.
- **OneHot with dynamic depth**: The depth parameter comes from data
  (e.g., max spatial coordinate).  Could use a static upper bound, but
  that increases memory.
- **Embedding scatter** (`Equal` → `CumSum` → `Gather` → `Where`): The
  CumSum-based indexing is the standard pattern for scattering features at
  placeholder positions.  Could be replaced with `ScatterND` but that has
  its own CUDA limitations.

For these, the pragmatic approach is to minimize the number of Memcpy
boundaries rather than eliminating CPU ops entirely.  For example, if
multiple CPU ops chain together without touching GPU tensors in between,
ORT batches them into a single CPU segment with only entry/exit Memcpy.

## Gemma4 case study

The Gemma4 multimodal model (E2B) was optimized to reduce Memcpy from
~30 to ~17 CPU-likely ops across all sub-models:

### Vision encoder (9 → ~5 Memcpy)

| Source | Fix | Ops removed |
|--------|-----|-------------|
| `Where(is_pad, zero, pos_emb)` | `Mul(pos_emb, CastLike(Not(is_pad)))` | 1 Where |
| `Where(is_pad, neg_inf, zero)` | `Mul(Cast(is_pad, float), neg_inf)` | 1 Where |
| `Where(is_pad, zero, hidden)` | `Mul(hidden, CastLike(Not(is_pad)))` | 1 Where |
| `Shape(features, start=2)` for Reshape | Static `config.hidden_size` | 1 Shape |
| **Remaining**: `OneHot` (dynamic depth), `Equal`/`And` (pooler padding) | Inherent | — |

### Embedding model (3 Memcpy — unchanged)

| Source | Status |
|--------|--------|
| `CumSum` for scatter indexing (×2) | Inherent (INT64 cumsum) |
| `Where` for conditional scatter (×2) | Inherent (both branches non-trivial) |

### Decoder (reduced by switching to bool masks + GQA)

| Source | Fix | Ops removed |
|--------|-----|-------------|
| `create_attention_bias()` for full-attn | `create_padding_mask()` + `is_causal=1` | CumSum, GreaterOrEqual, Where |
| `create_attention_bias()` for sliding | `create_sliding_window_mask()` (bool) | 2 Where (CumSum remains) |
| Shape-based KV reshape (×4) | `op.Reshape(x, [0, 0, -1])` | 4 Shape |
| Non-shared layers | `GroupQueryAttention` with `local_window_size` | All mask ops |
| **Remaining**: per-layer token masking (`Equal`/`Where`), sliding CumSum | Low-impact or inherent | — |

## Impact assessment

Not all CPU ops are equal.  Prioritize fixes by **tensor size**:

| Priority | Pattern | Tensor size | Impact |
|----------|---------|-------------|--------|
| **High** | `Where` on `[B, S, H]` activation tensors | Large | Copies full hidden states |
| **High** | `create_attention_bias` chain | `[B, 1, Q, T]` | Scales with sequence length² |
| **Medium** | `Shape` for Reshape | Scalar/1D | Small transfer, but blocks pipeline |
| **Low** | `Equal` on `[B, S]` input_ids | Small | Tiny tensor, one-time |
| **Low** | `Not`/`And` on bool scalars | Trivial | Negligible transfer cost |

Focus on ops that touch `[B, S, H]` or `[B, Q, T]` tensors first.  Shape
and comparison ops on small metadata tensors rarely matter for throughput.

## Reference files

> Read these files for implementation details of the fix patterns:

- `src/mobius/components/_attention.py` — `GQAContext` (with `local_window_size`),
  `_forward_gqa()`, `_apply_attention()`
- `src/mobius/components/_common.py` — `create_attention_bias()`,
  `create_padding_mask()`, `create_sliding_window_mask()`
- `src/mobius/models/base.py` — `TextModel.forward()` GQA decision logic
- `src/mobius/models/gemma4.py` — Case study: vision Mul patterns, decoder
  bool mask fallback, GQA + sliding window
- `src/mobius/_execution_providers.py` — `EpCapabilities` (gqa_dtypes,
  supports_fused_rope, supports_fused_moe)
