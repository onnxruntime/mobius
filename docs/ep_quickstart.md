# EP-Aware Building: Quick Start

This guide covers the common tasks in under 3 minutes. For full reference
documentation, see [Execution Provider Aware Building](execution_providers.md).

---

## 1. Build for a specific EP

Pass `execution_provider` to `build()` or `build_from_module()`:

```python
import mobius

# Portable ONNX — runs on any conformant runtime (default)
pkg = mobius.build("meta-llama/Llama-3.2-1B")

# CUDA EP — GQA fusion, SkipLayerNorm fusion, PackQKV
pkg = mobius.build("meta-llama/Llama-3.2-1B",
                   execution_provider="cuda", dtype="f16")

# WebGPU EP — GQA fusion, Shape → ReduceSum+ReduceMax lowering
pkg = mobius.build("meta-llama/Llama-3.2-1B",
                   execution_provider="webgpu", dtype="f16")

# DML (DirectML) — GQA for FP16, fused RoPE lowered to separate ops
pkg = mobius.build("meta-llama/Llama-3.2-1B",
                   execution_provider="dml", dtype="f16")

# TRT-RTX — GQA for FP16/BF16, SkipLayerNorm expanded, GPU graph capture
pkg = mobius.build("meta-llama/Llama-3.2-1B",
                   execution_provider="trt-rtx", dtype="f16")

# ONNX-standard — zero custom-domain ops, safe for any conformant ONNX runtime
pkg = mobius.build("meta-llama/Llama-3.2-1B",
                   execution_provider="onnx-standard")  # dtype defaults to f32; you can also pass f16/bf16

pkg.save("output/llama/")
```

CLI equivalent (see [CLI Reference](cli_reference.md) for all `--ep` options):

```bash
mobius build --model meta-llama/Llama-3.2-1B output/ \
  --ep cuda --dtype f16
```

---

## 2. See what each EP does

Enable `trace_optimization=True` to get a per-stage log of what the
optimizer changed:

```python
import logging
import mobius

logging.basicConfig(level=logging.INFO)

pkg = mobius.build(
    "meta-llama/Llama-3.2-1B",
    execution_provider="cuda",
    dtype="f16",
    load_weights=False,       # skip download for diagnostics
    trace_optimization=True,
)
```

Example output:

```
INFO  [EP Trace] Target: cuda, dtype: FLOAT16, role: decoder
INFO  [EP Trace] Stage 1: Cleanup (9 passes)
INFO  [EP Trace]   Cleanup: 512 → 486 nodes (-26)
INFO  [EP Trace] Stage 2: Fusion (5 rule groups)
INFO  [EP Trace]   GQAFusion    : +16 GroupQueryAttention, -16 Attention
INFO  [EP Trace]   PackQKV      : +16 Concat, -32 FusedMatMul
INFO  [EP Trace]   SkipNorm     : +16 SkipSimplifiedLayerNorm, -48 nodes
INFO  [EP Trace]   SkipLayerNorm: no matches (0 nodes affected)
INFO  [EP Trace]   GeluFusion   : no matches (0 nodes affected)
INFO  [EP Trace] Stage 2b: InlinePass (expand unsupported custom ops)
INFO  [EP Trace]   InlinePass   : no matches (0 nodes affected)
INFO  [EP Trace] Stage 3: Lowering (0 rule groups for cuda)
INFO  [EP Trace] Stage 4: Constant folding
INFO  [EP Trace]   Fold: 374 → 355 nodes (-19)
```

---

## 3. Query which EPs are available

```python
from mobius import ep_registry, get_ep

# List all registered EP names
print(sorted(ep_registry))
# ['cpu', 'cuda', 'default', 'dml', 'onnx-standard', 'trt-rtx', 'webgpu']

# Inspect an EP's capabilities
caps = get_ep("cuda")
print(caps.gqa_dtypes)          # frozenset({FLOAT16, BFLOAT16})
print(caps.qkv_pack_dtypes)     # frozenset({FLOAT, FLOAT16, BFLOAT16})
print(caps.supports_fused_rope) # True
print(caps.provider_options)    # {'enable_cuda_graph': '0', ...}
```

---

## 4. Register a custom EP

Out-of-tree EPs register via `register_ep()` before calling `build()`:

```python
import onnx_ir as ir
from mobius import EpCapabilities, register_ep

register_ep(EpCapabilities(
    name="my-ep",
    gqa_dtypes=frozenset({ir.DataType.FLOAT16}),
    qkv_pack_dtypes=frozenset({ir.DataType.FLOAT16}),
    supports_fused_rope=True,
    provider_options={"some_option": "value"},
))

# Now build() will accept "my-ep" as an execution provider
pkg = mobius.build("meta-llama/Llama-3.2-1B",
                   execution_provider="my-ep", dtype="f16")
```

Unrecognised EP names raise `ValueError` during build-time validation, before
graph construction or optimization starts — safe-fail by default.

---

## 5. Common EP configurations at a glance

| Goal | EP | dtype | Notes |
|---|---|---|---|
| Portable ONNX (maximum compatibility) | `"default"` | any | No EP-specific vendor fusions (e.g. no GQA/PackQKV) |
| Strict standard ONNX (no custom ops at all) | `"onnx-standard"` | any | All `com.microsoft` ops expanded via InlinePass; safe for non-ORT runtimes |
| ORT CPU inference | `"cpu"` | `"f32"` | GQA fusion for FP32 |
| NVIDIA GPU | `"cuda"` | `"f16"` or `"bf16"` | GQA + SkipNorm + PackQKV |
| Windows GPU (DirectX) | `"dml"` | `"f16"` | RoPE lowered separately |
| Browser / WebAssembly | `"webgpu"` | `"f16"` or `"f32"` | Shape ops replaced |
| NVIDIA TensorRT-RTX | `"trt-rtx"` | `"f16"` or `"bf16"` | SkipLayerNorm expanded |

---

## What to read next

- [Full EP reference](execution_providers.md) — Tier 1/2/3 strategy, pipeline
  internals, design decisions, build-time EP queries
- [Adding a new EP](execution_providers.md#adding-a-new-execution-provider) — 3-step
  process with test guidance
- [Debugging with trace mode](execution_providers.md#debugging-trace-mode) — reading
  trace output, fusion assertions
