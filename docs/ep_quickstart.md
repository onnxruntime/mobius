# EP-Aware Export: Quick Start

Mobius exports a canonical graph without automatically applying EP graph
rewrites. The target execution provider and device select structural build
requirements and runtime metadata only. Olive decides whether to preserve
function-backed custom operators, expand them to strict ONNX, or apply
EP-specific rewrites.

For full reference documentation, see
[Execution Provider Graph Profiles](execution_providers.md).

---

## 1. Export a canonical graph

```python
import mobius

# Canonical graph with model-local fallback functions.
pkg = mobius.build("meta-llama/Llama-3.2-1B")

# CUDA target contract and runtime metadata; no CUDA graph rewrites.
pkg = mobius.build(
    "meta-llama/Llama-3.2-1B",
    execution_provider="cuda",
    device="gpu",
    dtype="f16",
)

# OpenVINO structural contract, including rank-4 Gemma4 per-layer inputs.
pkg = mobius.build(
    "google/gemma-4-E2B-it",
    execution_provider="openvino",
    device="npu",
)

pkg.save("output/model/")
```

`execution_provider` never enables GQA, PackQKV, SkipNorm, or EP lowering
during Mobius export.

---

## 2. Resolve the graph in Olive

Keep the canonical graph and apply a target profile:

```bash
olive capture-onnx-graph \
  --model_name_or_path meta-llama/Llama-3.2-1B \
  --use_mobius_builder \
  --execution_provider cuda \
  --device gpu \
  --output_path output/cuda
```

Expand all model-local custom functions into strict standard ONNX:

```bash
olive capture-onnx-graph \
  --model_name_or_path google/gemma-4-E2B-it \
  --use_mobius_builder \
  --execution_provider openvino \
  --device npu \
  --onnx_standard \
  --output_path output/openvino
```

`--onnx_standard` may also be used without an EP/device. When supplied with
an EP/device, Olive retains that target's structural build contract and runtime
configuration while expanding every function-backed non-standard operator.

---

## 3. Inspect available target contracts

```python
from mobius import get_ep

openvino = get_ep("openvino")
print(openvino.layered_per_layer_inputs)  # True

webgpu = get_ep("webgpu")
print(webgpu.max_buffer_size)  # 268435456

qnn = get_ep("qnn")
print(qnn.supports_range)  # False
```

These fields are projected into Mobius' `BuildContract`. Fusion and lowering
capabilities remain available as the reference used to implement equivalent
Olive surgeons, but Mobius does not apply them automatically.

---

## 4. Develop or test rewrite rules explicitly

Mobius retains `optimize_model()` as a low-level rule-development API while
rewrites migrate to Olive. Calling it is explicit and separate from export:

```python
import onnx_ir as ir
from mobius import optimize_model

optimize_model(
    model,
    ep="cuda",
    dtype=ir.DataType.FLOAT16,
    model_role="decoder",
)
```

Production export should use Olive's EP profiles rather than calling this API.
