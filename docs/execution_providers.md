# Execution Provider (EP) Aware Building

Mobius accepts an execution provider when a graph-construction or packaging
contract depends on the target runtime. The default output uses the portable
construction path.

Mobius does **not** apply post-export graph fusions or EP-specific lowerings.
Apply those transformations with Olive after Mobius exports the weighted ONNX
model.

## Quick start

```python
import mobius

# Portable construction path
package = mobius.build("Qwen/Qwen2.5-7B")

# CUDA construction and runtime configuration
package = mobius.build(
    "Qwen/Qwen2.5-7B",
    execution_provider="cuda",
    dtype="f16",
)
```

The CLI equivalent is:

```bash
mobius build --model Qwen/Qwen2.5-7B --ep cuda --dtype f16 output/
```

## What the EP controls

The selected EP may affect:

- Direct operator emission inside components, such as
  `com.microsoft::GroupQueryAttention` or packed multimodal attention.
- Whether registered ONNX local functions are inlined for the target.
- Quantized operator construction defaults and buffer-size limits.
- ORT GenAI provider options, graph-capture settings, and KV-buffer sharing.

It does not trigger pattern-based fusion, QKV packing, RoPE separation,
attention decomposition, normalization fusion, or other post-export graph
surgery.

## Registered providers

| Name | Purpose |
|---|---|
| `default` | Portable construction path with local function bodies retained. |
| `cpu` | ORT CPU construction and runtime defaults. |
| `cuda` | ORT CUDA construction and runtime defaults. |
| `dml` | DirectML construction and runtime defaults. |
| `webgpu` | WebGPU buffer and graph-capture constraints. |
| `mlx` | Apple-silicon MLX plugin contracts. |
| `trt-rtx` | TensorRT-RTX construction and runtime defaults. |
| `qnn` | QNN HTP construction constraints and supported local-function inlining. |
| `openvino` | OpenVINO runtime defaults. |
| `onnx-standard` | Inline non-standard local functions into standard ONNX. |

Unknown names raise `ValueError` before graph finalization. Query the registry
with:

```python
from mobius import ep_registry, get_ep

print(sorted(ep_registry))
cuda = get_ep("cuda")
print(cuda.provider_options)
```

## Custom providers

Out-of-tree providers may register construction and packaging capabilities:

```python
import onnx_ir as ir
from mobius import EpCapabilities, register_ep

register_ep(
    EpCapabilities(
        name="my-ep",
        gqa_dtypes=frozenset({ir.DataType.FLOAT16}),
        supports_past_present_share_buffer=True,
    )
)
```

If a target needs additional post-export graph changes, add them to the Olive
optimization workflow rather than to Mobius.

## Finalization trace

`trace_optimization=True` reports the exporter-owned finalization stages:

```text
[EP Trace] Stage 1: Cleanup
[EP Trace] Stage 2: Inline unsupported local functions
[EP Trace] Stage 3: Constant folding
```

This trace does not include downstream Olive graph transformations.
