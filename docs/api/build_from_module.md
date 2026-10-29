# `build_from_module()`

Build an ONNX `ModelPackage` from a module instance and config.

```python
from mobius import build_from_module
```

## Signature

```python
def build_from_module(
    module: nn.Module,
    config: BaseModelConfig,
    task: str | ModelTask = "text-generation",
) -> ModelPackage:
```

## Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `module` | `nn.Module` | (required) | An `onnxscript.nn.Module` instance. Its `forward()` signature must be compatible with the task. |
| `config` | `BaseModelConfig` | (required) | Architecture configuration. The `dtype` field controls precision. |
| `task` | `str \| ModelTask` | `"text-generation"` | Task name string or `ModelTask` instance. |

## Returns

`ModelPackage` — A dict-like collection of named `ir.Model` objects.

## Examples

```python
from mobius import build_from_module, ArchitectureConfig
from mobius.models import CausalLMModel

config = ArchitectureConfig(
    vocab_size=32000,
    max_position_embeddings=4096,
    hidden_size=4096,
    intermediate_size=11008,
    num_hidden_layers=32,
    num_attention_heads=32,
    num_key_value_heads=32,
    hidden_act="silu",
    head_dim=128,
    pad_token_id=0,
)

module = CausalLMModel(config)
pkg = build_from_module(module, config)
pkg["model"]  # ir.Model
```

### Fixed-capacity Gemma4 vision

Standard Gemma4 vision exports default to dynamic patch/output lengths and one
image per run. Opt into fixed capacities through `Gemma4Config` before creating
the module:

```python
from dataclasses import replace

from transformers import AutoConfig

from mobius._configs import Gemma4Config
from mobius.models import Gemma4Model
from mobius.tasks import Gemma4Task

hf_config = AutoConfig.from_pretrained("google/gemma-4-E2B-it")
config = Gemma4Config.from_transformers(hf_config.text_config, parent_config=hf_config)
config = replace(config, vision_patch_capacity=2520, vision_batch_size=None)
pkg = build_from_module(Gemma4Model(config), config, task=Gemma4Task())
```

`vision_batch_size=None` allows a symbolic image batch; use a positive integer
for a fixed batch (`1` for sequential image runs). The standard 3x3 pooler gives:

```text
pixel_values:        [B, 2520, 768]
pixel_position_ids:  [B, 2520, 2]
image_features:      [B * 280, text_hidden_size]
```

The output is flattened in image-major order. Within each 280-row block, valid
features occupy a contiguous prefix and padding occupies the tail. Position IDs
must describe a complete rectangular patch grid, with both dimensions divisible
by three; padded patches use `(-1, -1)`. Capacity must accommodate all valid
patches. The caller retains the actual pooled count for each image, typically
from preprocessing, and packs only valid prefixes for embedding. Do not add
image placeholders for padded feature rows or infer validity from feature values.

The normalization and learned projection are unchanged. This option does not
change Gemma4 Unified, enable independent prompt batching in embedding, or add
runtime compaction to ORT GenAI. Runtime integration must honor the fixed-output
contract; the existing dynamic-output per-image output bindings are insufficient.

## Behavior

1. Validates config (if `validate()` is available)
2. Casts module parameters to target dtype
3. Resolves the task and builds the ONNX graph
4. Applies optimization passes (identity elimination, CSE, etc.)
