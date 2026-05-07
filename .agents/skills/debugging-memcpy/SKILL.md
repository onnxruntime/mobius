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
1. Build model with execution_provider='cuda' and save with external data
2. Load in ORT with log_severity_level=1 to get exact Memcpy attribution
3. Parse the logs to identify which nodes triggered Memcpy
4. Apply fix pattern from §Fix patterns
5. Re-run to verify reduction
```

## Profiling recipe

### Step 1: Build, save, and load with ORT verbose logging

The authoritative way to identify Memcpy sources is ORT's own verbose
logging.  It tells you exactly which nodes lack CUDA kernels and where
`MemcpyFromHost` / `MemcpyToHost` nodes are inserted.

```python
import os, tempfile
import onnx_ir as ir
import onnxruntime as ort
from mobius import build

pkg = build(model_id, execution_provider='cuda', dtype='float16')

tmpdir = tempfile.mkdtemp(prefix="memcpy_profile_")

for name, model in pkg.items():
    # IMPORTANT: lower opset to 23 — CUDA EP doesn't register many ops
    # (including Reshape, Cast) at opset 24.  Without this, you'll see
    # hundreds of false-positive Memcpy from ops that work fine at opset 23.
    if model.opset_imports.get("", 0) > 23:
        model.opset_imports[""] = 23

    path = os.path.join(tmpdir, f"{name}.onnx")
    ir.save(model, path, external_data=f"{name}.onnx.data")

    # Load with verbose logging
    opts = ort.SessionOptions()
    opts.log_severity_level = 1  # INFO level — shows Memcpy details
    sess = ort.InferenceSession(
        path, opts,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
```

### Step 2: Read the ORT logs

ORT emits two kinds of relevant log messages:

**1. Kernel not found (INFO)** — the op falls back to CPU:
```
CUDA kernel not found in registries for Op type: Equal node name: decoder/model/Equal_node_8
```

**2. Memcpy insertion (INFO)** — a transfer node is added at a
CPU↔GPU boundary:
```
Add MemcpyFromHost after v_decoder.model.Equal_8 for CUDAExecutionProvider
```

**3. Summary (WARNING)** — total count per sub-model:
```
4 Memcpy nodes are added to the graph ... for CUDAExecutionProvider
```

### Step 3: Categorize Memcpy sources

Parse the `AddCopyNode` lines to identify patterns:

```bash
# Extract and categorize Memcpy sources from ORT log output
grep "AddCopyNode" ort_output.log \
  | sed 's/.*AddCopyNode] //' \
  | sort | uniq -c | sort -rn
```

This gives you the exact node names causing Memcpy.  Common categories:
- **Graph inputs** (`input_ids`, `attention_mask`) — expected, unavoidable
- **Reshape with dynamic shape** — fix with static dims or `[0, 0, -1]`
- **Equal/Cast on token IDs** — usually low-impact (small tensors)
- **CumSum on INT64** — inherent, no GPU kernel

### Critical: opset 24 false positives

**Always lower opset to 23 before profiling.**  ORT CUDA EP (≤1.24.x)
does not register kernels for many standard ops at opset 24, including
`Reshape`, `Cast`, and others.  A Gemma4 decoder at opset 24 shows
**280 Memcpy** nodes; at opset 23, it shows **4**.  The
`ort_lower_opset_for_ep` flag in `_flags.py` handles this at runtime,
but you must apply it manually when profiling with raw ORT sessions.

## CPU-only op reference (ORT CUDA EP)

These are ops that ORT's CUDA EP does **not** register kernels for in
certain configurations.  The authoritative check is always
`log_severity_level=1` (see above), but this table covers common cases:

### Always CPU (shape/index computation)

| Op | Why CPU | Common source |
|----|---------|---------------|
| `Shape` | Extracts tensor dimensions as CPU int64 | Dynamic reshape, Concat for shape |
| `NonZero` | Returns indices of non-zero elements | Sparse masking |
| `Range` | Generates integer sequences | Position indices |
| `ConstantOfShape` | Creates tensor from shape input | Dynamic zero-fill |

### Context-dependent

| Op | When CPU | When GPU |
|----|----------|----------|
| `CumSum` | INT64 inputs | FLOAT inputs |
| `OneHot` | Dynamic depth parameter | Static depth |
| `Gather` | Small index tensor from CPU op | Large data tensor on GPU |
| `Cast` | Some type combinations | Most float↔float casts |
| `Equal` | Some type/opset combinations | Often CUDA-supported |
| `Reshape` | **Opset 24** (no CUDA kernel) | **Opset ≤23** (CUDA-supported) |

### CUDA-supported (do NOT try to eliminate)

| Op | Notes |
|----|-------|
| `Where` | Fully supported on CUDA EP. Do not replace with `Mul` patterns. |
| `And` / `Or` / `Not` | Bool logic — generally CUDA-supported |
| `GreaterOrEqual` / `Less` | Comparison — generally CUDA-supported |

## Fix patterns

### Pattern 1: Shape-based Reshape → static or [0, 0, -1]

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

### Pattern 2: Additive float bias → bool mask with is_causal

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

### Pattern 3: Use GQA's built-in local_window_size

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

### Pattern 4: Precompute at build time

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

The Gemma4 multimodal model (E2B) was profiled with ORT verbose logging
at opset 23.  Results after optimization:

### Final Memcpy counts (opset 23, CUDA EP)

| Sub-model | Memcpy | Sources |
|-----------|--------|---------|
| Vision | 0 | Clean — all ops on GPU |
| Audio | 0 | Clean — all ops on GPU |
| Decoder | 4 | `input_ids` (input), 2× `Equal` (token masks), `Where` (bool mask) |
| Embedding | 3 | `input_ids` (input), 2× `Equal` (token masks) |

### Opset 24 trap

Without opset lowering, the decoder showed **280 Memcpy** nodes because
CUDA EP doesn't register `Reshape`, `Cast`, and other standard ops at
opset 24.  The `ort_lower_opset_for_ep` flag (enabled by default) fixes
this at runtime.  Always lower to opset 23 before profiling.

## Impact assessment

Not all CPU ops are equal.  Prioritize fixes by **tensor size**:

| Priority | Pattern | Tensor size | Impact |
|----------|---------|-------------|--------|
| **High** | `create_attention_bias` chain (CumSum) | `[B, 1, Q, T]` | Scales with sequence length² |
| **High** | `Shape` for Reshape | Scalar/1D | Small transfer, but blocks pipeline |
| **Medium** | `CumSum` on INT64 | Varies | Inherently CPU, no workaround |
| **Low** | `Equal` on `[B, S]` input_ids | Small | Tiny tensor, one-time |
| **Low** | `Not`/`And` on bool scalars | Trivial | Negligible transfer cost |

> **Note**: `Where` IS supported by CUDA EP and does not cause Memcpy nodes.
> Do not replace `Where` with `Mul` patterns for Memcpy elimination — the
> `Where` is cleaner and runs on GPU.

Focus on ops that touch `[B, S, H]` or `[B, Q, T]` tensors first.  Shape
and comparison ops on small metadata tensors rarely matter for throughput.

## Reference files

> Read these files for implementation details of the fix patterns:

- `src/mobius/components/_attention.py` — `GQAContext` (with `local_window_size`),
  `_forward_gqa()`, `_apply_attention()`
- `src/mobius/components/_common.py` — `create_attention_bias()`,
  `create_padding_mask()`, `create_sliding_window_mask()`
- `src/mobius/models/base.py` — `TextModel.forward()` GQA decision logic
- `src/mobius/models/gemma4.py` — Case study: static reshape, decoder
  bool mask fallback, GQA + sliding window
- `src/mobius/_execution_providers.py` — `EpCapabilities` (gqa_dtypes,
  supports_fused_rope, supports_fused_moe)

## Cross-references

- **Attention optimization:** `.agents/skills/attention-optimization/SKILL.md`
  — kernel dispatch tables, mask type selection, Flash requirements
