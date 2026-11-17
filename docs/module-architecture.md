# Module Architecture Guide

This guide documents the internal module structure of `mobius`,
covering model registration, decoder layer extension, and weight loading security.

## Module Overview

The original `_exporter.py` monolith (1,188 lines) has been split into focused
modules:

| Module | Responsibility |
|---|---|
| `_registry.py` | Model registration — maps HuggingFace `model_type` strings to `nn.Module` subclasses |
| `integrations/transformers/_config_resolver.py` | Resolves HuggingFace `PretrainedConfig` objects to internal `BaseModelConfig` subclasses |
| `integrations/_weight_loading.py` | Eager and bounded-streaming checkpoint binding for ONNX IR models |
| `_builder.py` | Ecosystem-agnostic graph construction via `build_from_module()` |
| `integrations/transformers/_builder.py` | Transformers checkpoint orchestration |
| `integrations/diffusers/_builder.py` | Diffusers pipeline orchestration |

### Dependency graph

```
integrations/transformers/_builder.py  ← build_transformers_model()
  ├── _registry.py
  ├── integrations/transformers/_config_resolver.py
  ├── integrations/_weight_loading.py
  └── _builder.py    (build_from_module, resolve_dtype)
integrations/diffusers/_builder.py
  ├── _builder.py    (build_from_module, resolve_dtype)
  └── integrations/_weight_loading.py
```

---

## 1. Registering a New Model Architecture

The `ModelRegistry` in `_registry.py` maps HuggingFace `config.model_type`
strings to the `nn.Module` subclass used to build the ONNX graph. The global
singleton `registry` is pre-populated with ~190 built-in architectures.

### Quick registration

If your model follows the standard Llama-like decoder pattern (GQA + RoPE +
pre-norm), you can register it against the existing `CausalLMModel`:

```python
from mobius._registry import registry

registry.register("my_new_arch", CausalLMModel)
```

### Full registration with task and config

```python
from mobius._registry import registry, ModelRegistration

registry.register(
    "my_new_arch",
    MyCustomModule,
    task="text-generation",        # auto-detected default task
    config_class=MyCustomConfig,   # HF config parser (optional)
)
```

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `architecture` | `str` | Must match the HuggingFace `config.model_type` value |
| `module_class` | `type[nn.Module]` | Your model class — must accept an `ArchitectureConfig` and implement `forward()` |
| `task` | `str \| None` | Default task name (e.g. `"text-generation"`, `"vision-language"`). Falls back to `module_class.default_task` |
| `config_class` | `type[BaseModelConfig] \| None` | Config parser for the HF config. Falls back to `ArchitectureConfig` |

### Lookup API

```python
# Get module class (raises KeyError if not found)
cls = registry.get("llama")

# Get full registration entry
reg = registry.get_registration("llama")
print(reg.module_class, reg.task, reg.config_class)

# Check existence
if "my_arch" in registry:
    ...

# List all registered architectures
print(registry.architectures())
```

### Testing with a custom registry

The registry is a plain class instance — you can create isolated registries
for testing without affecting the global singleton:

```python
from mobius._registry import ModelRegistry

test_registry = ModelRegistry()
test_registry.register("test_arch", MyTestModule)
assert "test_arch" in test_registry
assert len(test_registry) == 1
```

### Deprecated: `MODEL_MAP`

The `MODEL_MAP` dict is a backward-compatible alias that exposes the
registry's internal `_map` dict. **Use `registry.get()` /
`registry.register()` instead.** `MODEL_MAP` will be removed in a future
version.

---

## 2. Adding a New Decoder Layer Variant

Decoder layers define the per-layer transformer computation (norm → attention
→ residual → norm → MLP → residual). The base implementation lives in
`components/_decoder.py`.

### Existing variants

| Class | Location | Key difference |
|---|---|---|
| `DecoderLayer` | `components/_decoder.py` | Standard pre-norm (Llama-style) |
| `PostNormDecoderLayer` | `components/_decoder.py` | Post-norm (OLMo-2 style) |
| `GemmaDecoderLayer` | `models/gemma.py` | Uses `OffsetRMSNorm` (+1.0 offset) |
| `Gemma2DecoderLayer` | `models/gemma.py` | Adds sliding window + post-attn norm |
| `GraniteDecoderLayer` | `models/granite.py` | Adds `residual_multiplier` scaling |
| `Qwen35DecoderLayer` | `models/qwen35.py` | Hybrid GatedDeltaNet / full-attention dispatch via `config.layer_types`; uses `OffsetRMSNorm` |
| `DeepSeekMLADecoderLayer` | `models/deepseek.py` | Multi-head Latent Attention (MLA) — structurally different |

### Step-by-step: create a new decoder layer

**Step 1.** Create your decoder layer class as an `nn.Module`. Follow the
same `forward()` signature as `DecoderLayer`:

```python
# In models/my_arch.py
from onnxscript import nn
from onnxscript import OpBuilder
from mobius._configs import ArchitectureConfig
from mobius.components import Attention, MLP, RMSNorm


class MyDecoderLayer(nn.Module):
    """Custom decoder layer with <describe your variant>."""

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.self_attn = Attention(config)
        self.mlp = MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        op: OpBuilder,
        hidden_states,
        attention_bias,
        position_embeddings: tuple,
        past_key_value: tuple | None,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(op, hidden_states)

        attn_output, present_key_value = self.self_attn(
            op,
            hidden_states=hidden_states,
            attention_bias=attention_bias,
            position_embeddings=position_embeddings,
            past_key_value=past_key_value,
        )
        # Example: add a residual multiplier
        hidden_states = op.Add(residual, op.Mul(attn_output, self.multiplier))

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(op, hidden_states)
        hidden_states = self.mlp(op, hidden_states)
        hidden_states = op.Add(residual, hidden_states)

        return hidden_states, present_key_value
```

**Step 2.** Create your model class that uses the custom decoder layer.
Typically you subclass or follow the pattern of `CausalLMModel` /
`TextModel`:

```python
from mobius.models.base import CausalLMModel, TextModel

class MyTextModel(TextModel):
    """Override to use custom decoder layers."""

    def __init__(self, config):
        super().__init__(config)
        # Replace the default DecoderLayer with yours
        self.layers = nn.ModuleList(
            [MyDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )

class MyCausalLMModel(CausalLMModel):
    def __init__(self, config):
        super().__init__(config)
        self.model = MyTextModel(config)
```

**Step 3.** Register your architecture:

```python
from mobius._registry import registry
registry.register("my_arch", MyCausalLMModel)
```

### Important: `forward()` signature contract

All decoder layers **must** use the explicit typed signature:

```python
def forward(
    self,
    op: OpBuilder,
    hidden_states,           # [batch, seq_len, hidden_size]
    attention_bias,          # attention mask/bias tensor
    position_embeddings: tuple,  # (cos, sin) from RoPE
    past_key_value: tuple | None,  # (key, value) cache or None
) -> tuple:                  # (hidden_states, present_key_value)
```

Do **not** use `*args` or `**kwargs`. The explicit signature enables
static analysis and makes the data flow traceable through the ONNX graph.

### Edge case: OffsetRMSNorm (+1.0)

Gemma-family models use `OffsetRMSNorm` which adds 1.0 to the weight before
normalization (`weight + 1.0`). If your architecture stores norm weights with
this offset convention, use `OffsetRMSNorm` instead of `RMSNorm`. Keep the
norm behavior inside the norm class — do **not** add `isinstance` checks in
the decoder layer.

---

## 3. Weight Loading Security Policy

### Policy: prefer safetensors; restrict legacy PyTorch loading

Bounded streaming requires
[safetensors](https://github.com/huggingface/safetensors), whose headers can be
inspected without materializing tensor payloads. The generic eager loader
prefers safetensors and supports legacy PyTorch state dictionaries only through
`torch.load(..., weights_only=True)`.

> Unrestricted pickle deserialization is prohibited. Legacy files must load on
> CPU with `weights_only=True`. The generic Transformers loader also validates
> the result as a mapping from string names to tensors. Architecture-specific
> streaming paths must not fall back to pickle formats.

This policy is enforced by:

1. **Format preference** — `_download_weights()` checks safetensors indexes
   and single-file checkpoints before legacy PyTorch files.
2. **Restricted fallback** — legacy loading uses `weights_only=True`; the
   generic Transformers loader also uses `map_location="cpu"` and validates
   the returned state dict.
3. **Streaming preflight** — streaming planners inspect safetensors headers,
   reject duplicate or unclassified keys, and validate source and target
   metadata before publication.

### How weight loading works

```
_download_weights(model_id)
  │
  ├── Try model.safetensors.index.json  (sharded checkpoint)
  │     └── Download all shards in parallel via _parallel_download()
  │
  ├── Fall back to model.safetensors   (single file)
  │     └── safetensors.torch.load_file()
  │
  └── Fall back to pytorch_model.bin(.index.json)
        └── torch.load(weights_only=True, map_location="cpu")
        └── Validate dict[str, torch.Tensor]
                                        ↓
                                   state_dict: dict[str, torch.Tensor]
                                        ↓
apply_weights(model, state_dict)
  │
  └── For each weight in state_dict:
        ├── Skip if not in model.graph.initializers
        ├── If dtype mismatch → wrap in ir.LazyTensor (lazy cast)
        └── Assign to initializer.const_value
```

Large or layout-changing checkpoints use bounded streaming instead:

```text
integrations/transformers/_builder.py
  -> architecture-specific planner (_<model>_weights.py)
       -> inspect safetensors headers
       -> classify every source
       -> validate source and target metadata
       -> build StreamingWeightPlan
  -> integrations/_weight_loading.py
       -> bind lazy sources to graph initializers
  -> ModelPackage.save()
       -> materialize one source/transform at a time
       -> write ONNX external data transactionally
```

The source types have separate responsibilities:

- `StreamingWeightSource` binds one source tensor directly.
- `StreamingExpertBankSource` assembles per-expert matrices.
- `StreamingTransformedWeightSource` validates and lazily transforms one
  source into a differently laid-out target.

The generic loader owns lifecycle, metadata checks, and lazy materialization.
The architecture planner owns checkpoint names, topology, and source-to-target
mapping. Model graph math and format-specific transforms remain in the model
module. The Transformers builder may select a planner, but should not contain
operator layout or repacking logic.

### For diffusers components

`integrations/diffusers/_builder.py` uses
`_download_diffusers_component_weights()` which
prefers safetensors and uses the same restricted `weights_only=True` fallback
for legacy PyTorch files. It looks for:
- `{component}/diffusion_pytorch_model.safetensors(.index.json)`
- `{component}/pytorch_model.safetensors(.index.json)`
- `{component}/model.safetensors(.index.json)`
- `{component}/diffusion_pytorch_model.bin(.index.json)`
- `{component}/pytorch_model.bin(.index.json)`
- `{component}/model.bin(.index.json)`

### Key implementation details

- **Lazy casting**: When the weight dtype doesn't match the model's declared
  type, `ir.LazyTensor` defers the cast to serialization time, avoiding
  eager memory allocation.
- **Bounded transformation**: Custom repacking runs during serialization, so
  only the current source, scratch space, and target tensor are live.
- **Parallel downloads**: Sharded checkpoints are downloaded using a
  thread pool (`max_workers=8`) for faster loading.
- **No temp files**: Weights are loaded directly from the HuggingFace
  cache — no temporary copies are created.

---

## 4. Import Conventions

All code imports directly from the specific module that owns the symbol:

```python
from mobius._builder import build_from_module
from mobius._registry import registry
from mobius.integrations._weight_loading import apply_weights
from mobius.integrations.diffusers._builder import build_diffusers_pipeline
from mobius.integrations.transformers._builder import build_transformers_model
```

### Module → symbols reference

| Source module | Symbols |
|---|---|
| `_builder` | `DTYPE_MAP`, `build_from_module`, `resolve_dtype`, `_cast_module_dtype` |
| `integrations.transformers._builder` | `build_transformers_model` |
| `integrations.transformers._config_resolver` | `_config_from_hf`, `_default_task_for_model`, `_dict_to_pretrained_config`, `_try_load_config_json` |
| `integrations.diffusers._builder` | `build_diffusers_pipeline`, `_DIFFUSERS_CLASS_MAP`, `_download_diffusers_component_weights`, `_init_diffusers_class_map`, `_load_diffusers_component_config`, `_load_diffusers_pipeline_index` |
| `_registry` | `MODEL_MAP`, `ModelRegistration`, `ModelRegistry`, `registry` |
| `integrations._weight_loading` | `apply_weights`, `_download_weights`, `_parallel_download` |

---

## Quick Reference: Common Tasks

### Build a model from HuggingFace

```python
from mobius import build

pkg = build("meta-llama/Llama-3-8B")
pkg.save("/output/llama/")
```

### Build from a custom module

```python
from mobius import build_from_module

module = MyCausalLMModel(config)
pkg = build_from_module(module, config, task="text-generation")
```

### Register and build a custom architecture

```python
from mobius import build
from mobius._registry import registry

registry.register("my_arch", MyCausalLMModel)
pkg = build("my-org/my-model")  # auto-detects "my_arch" from config.json
```

### Apply weights separately

```python
from mobius import build
from mobius.integrations._weight_loading import apply_weights, _download_weights

pkg = build("meta-llama/Llama-3-8B", load_weights=False)
state_dict = _download_weights("meta-llama/Llama-3-8B")
for model in pkg.values():
    apply_weights(model, state_dict)
```
