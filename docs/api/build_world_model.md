# `build_world_model()`

Export a supported world-model checkpoint as a `PipelinePackage`.

```python
from mobius import build_world_model

package = build_world_model(
    "nvidia/Cosmos3-Nano",
    load_weights=True,
    execution_provider="cuda",
)
```

Save the returned package with `package.save(output_directory)`.

## Supported models

| `model_type` | Checkpoints |
|---|---|
| `cosmos3_omni` | Qwen3-VL-based Cosmos3-Nano, Cosmos3-Super, Policy-DROID, Text2Image, and Image2Video variants whose public component configs match the supported architecture |
| `cosmos3_edge` | `nvidia/Cosmos3-Edge` |

`nvidia/Cosmos3-Edge-Policy-DROID` is detected automatically despite its
different top-level model type.

## Options

`load_weights=False` builds and validates the complete graph topology without
downloading tensor payloads. Small configuration files, runtime assets, and
safetensors header metadata may still be downloaded.

Use `dtype="f32"` for CPU inference or the checkpoint's native BF16 dtype for
CUDA inference.
