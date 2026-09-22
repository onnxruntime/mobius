# Execution Provider Build Contracts

Mobius exports a canonical ONNX graph and does not contain or execute
execution-provider graph rewrites. Fusion, lowering, strict-ONNX expansion,
and target operator selection are owned by downstream tooling such as Olive.

The `execution_provider` and `device` arguments select only structural build
requirements and runtime metadata:

```python
import mobius

package = mobius.build(
    "google/gemma-4-E2B-it",
    execution_provider="openvino",
    device="npu",
)
```

For this example Mobius preserves OpenVINO's rank-4 Gemma4
`per_layer_inputs` interface, but does not fuse GQA, pack QKV, fuse
normalization, decompose operators, or inline custom functions.

## BuildContract

Target-independent graph construction and target structural requirements are
kept separate:

```python
from mobius import BuildContract

BuildContract(
    target_execution_provider="openvino",
    target_device="npu",
    max_buffer_size=None,
    layered_per_layer_inputs=True,
    supports_range=True,
)
```

The current structural fields are:

| Field | Purpose |
|---|---|
| `layered_per_layer_inputs` | Keep Gemma4 layer/projection dimensions separate across component boundaries. |
| `max_buffer_size` | Split build-time resources that exceed a target buffer limit. |
| `supports_range` | Materialize static ranges when a target cannot execute dynamic `Range`. |
| `target_execution_provider` | Identify the runtime target without selecting graph rewrites. |
| `target_device` | Preserve the requested CPU/GPU/NPU target for downstream packaging. |

Components read these requirements through `get_build_contract()`. They do
not query target fusion/lowering capabilities.

## Canonical graph resolution in Olive

Apply an EP profile:

```bash
olive capture-onnx-graph \
  --model_name_or_path meta-llama/Llama-3.2-1B \
  --use_mobius_builder \
  --execution_provider cuda \
  --device gpu \
  --output_path output/cuda
```

Expand every function-backed non-standard operator into strict ONNX:

```bash
olive capture-onnx-graph \
  --model_name_or_path google/gemma-4-E2B-it \
  --use_mobius_builder \
  --execution_provider openvino \
  --device npu \
  --onnx_standard \
  --output_path output/openvino
```

Mobius preserves model-local standard fallback functions in its canonical
graph. Olive's `InlineModelLocalFunctions` surgeon performs strict expansion
and rejects any remaining non-standard operator without a fallback body.

## Registering a target

Target registrations in Mobius describe structural and runtime requirements,
not graph rewrite support:

```python
from mobius import EpCapabilities, register_ep

register_ep(
    EpCapabilities(
        name="my-ep",
        provider_options={"device_type": "NPU"},
        max_buffer_size=268_435_456,
        layered_per_layer_inputs=True,
        supports_range=False,
    )
)
```

Any graph surgery required by the target belongs in Olive.
