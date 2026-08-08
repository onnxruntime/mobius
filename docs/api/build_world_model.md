# `build_world_model()`

Build every registered neural component of a complete world-model checkpoint.

```python
from mobius import build_world_model

package = build_world_model(
    "nvidia/Cosmos3-Nano",
    load_weights=True,
    execution_provider="cuda",
)
```

The function dispatches by the checkpoint's `model_type` through
`world_model_registry` and returns a `PipelinePackage`. Unlike `build()`, one
builder may combine multiple model families, task contracts, configs, and
weight layouts.

The package's `pipeline.json` uses schema 1.1 and a versioned model profile
(`cosmos3-omni` or `cosmos3-edge`). It includes generated-input programs,
state lifecycle, scheduler/sampling controls, parameterized transforms, and
component dtype/EP hints so a compatible runtime can execute it without
architecture-specific name inference.

Currently registered complete implementation:

| `model_type` | Checkpoints |
|---|---|
| `cosmos3_omni` | Qwen3-VL-based Cosmos3-Nano, Cosmos3-Super, Policy-DROID, Text2Image, and Image2Video variants whose public component configs match the supported architecture |
| `cosmos3_edge` | `nvidia/Cosmos3-Edge` |

`nvidia/Cosmos3-Edge-Policy-DROID` currently advertises top-level
`model_type="cosmos3_omni"` but uses the distinct Cosmos3-Edge backbone.
Mobius detects `text_config.model_type="cosmos3_edge_text"` and routes it to
the Edge builder automatically.

`load_weights=False` builds and validates the complete graph topology without
downloading tensor payloads. Small configuration files, runtime assets, and
safetensors header metadata may still be downloaded.
