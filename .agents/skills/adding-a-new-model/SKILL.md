---
name: adding-a-new-model
description: >
  Use this skill when adding a new HuggingFace model architecture to
  mobius — including LLM, encoder-only, encoder-decoder, vision, audio,
  diffusion, or multimodal models. Covers the full workflow: config
  extraction, model class creation, registry registration, weight
  preprocessing, and testing. Also covers MoE and hybrid architectures.
---

# Skill: Adding a New Model

## When to use

Use this skill when adding support for a new HuggingFace model architecture
(e.g. a new LLM family, vision model, encoder-decoder, audio model, or
diffusion component) to the `mobius` package.

## Reference files

Read these when you need deeper detail on a specific topic:

- Read [`references/architecture-patterns.md`](references/architecture-patterns.md)
  when implementing a **non-LLM model** (encoder-only, encoder-decoder, vision,
  audio, diffusion, multimodal), when dealing with **KV sharing across layers**,
  or when checking **false compatibility pitfalls** for registry aliases.
- Read [`references/weight-preprocessing.md`](references/weight-preprocessing.md)
  when handling **weight name mismatches**, **fused weight splitting**,
  **precision/dtype issues**, or debugging **logit mismatches** traced to
  weight loading, identity folding, or fp32 upcast problems.

## Prerequisites

- Identify the HuggingFace `model_type` string (from the model's `config.json`)
- Find a small checkpoint on HuggingFace Hub for testing
- Have the HuggingFace `transformers` source available to reference the
  PyTorch implementation

## Step-by-step

### 1. Check if the base `CausalLMModel` already works

Many models (LLaMA, Mistral, Qwen2, DeepSeek) use the standard decoder-only
architecture with no special components.  Before writing a custom class,
check whether `CausalLMModel` from `models/base.py` produces correct results:

```python
from mobius._registry import registry
from mobius.models.base import CausalLMModel

registry.register("my_model_type", CausalLMModel)
model = build("org/my-model-id", load_weights=True)
```

If the logits match HuggingFace, you only need the registry entry.

### 2. Create the model file

Create `src/mobius/models/<model_name>.py`.  The minimal template:

```python
from __future__ import annotations

import torch
from onnxscript import nn

from mobius._configs import ArchitectureConfig
from mobius.components import (
    Attention, DecoderLayer, Embedding, Linear, MLP, RMSNorm,
    create_attention_bias, initialize_rope,
)
from mobius.models.base import CausalLMModel


class MyTextModel(nn.Module):
    """Text model for MyArchitecture."""

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.embed_tokens = Embedding(config.vocab_size, config.hidden_size)
        self.layers = [MyDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = initialize_rope(config)

    def forward(self, op, input_ids, attention_mask, position_ids, past_key_values=None):
        hidden_states = self.embed_tokens(op, input_ids)
        position_embeddings = self.rotary_emb(op, position_ids)
        attention_bias = create_attention_bias(
            op, input_ids=input_ids, attention_mask=attention_mask,
        )
        present_key_values = []
        past_kvs = past_key_values or [None] * len(self.layers)
        for layer, past_kv in zip(self.layers, past_kvs):
            hidden_states, present_kv = layer(
                op, hidden_states=hidden_states,
                attention_bias=attention_bias,
                position_embeddings=position_embeddings,
                past_key_value=past_kv,
            )
            present_key_values.append(present_kv)
        hidden_states = self.norm(op, hidden_states)
        return hidden_states, present_key_values


class MyCausalLMModel(CausalLMModel):
    """Causal LM wrapper for MyArchitecture."""

    def __init__(self, config: ArchitectureConfig):
        nn.Module.__init__(self)
        self.config = config
        self.model = MyTextModel(config)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)
```

> If your text model **subclasses `TextModel`** (rather than `nn.Module`) and
> overrides `__init__` with `nn.Module.__init__(self)` to swap in a custom
> decoder layer, you must also set `self.config = config` in that subclass —
> `TextModel.forward` reads `self.config`. See troubleshooting §7.


#### Class metadata attributes

Every registered model class should set two class-level attributes:

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `default_task` | `str` | `"text-generation"` | Task auto-selected by `build()` / CLI. |
| `category` | `str` | `"Text Generation"` | Grouping label in generated docs. |

Override these when the model isn't a standard text-generation model:

```python
class MyMultiModalModel(nn.Module):
    default_task: str = "vision-language"
    category: str = "Multimodal"
```

Standard categories: `"Text Generation"`, `"Mixture of Experts"`,
`"Multimodal"`, `"Speech-to-Text"`, `"Audio"`, `"Diffusion"`,
`"autoencoder"`, `"encoder-only"`, `"encoder"`, `"encoder-decoder"`,
`"vision"`, `"causal-lm"`.

### 3. Identify what's different

Compare the HuggingFace PyTorch source against `CausalLMModel` / `DecoderLayer`.
Common variations to look for:

| Variation | Example | Solution |
|-----------|---------|----------|
| Custom norm (weight + 1) | Gemma | Subclass `RMSNorm` |
| Embedding scaling | Gemma (`* sqrt(d)`) | Subclass `Embedding` |
| Extra norms (pre/post feedforward) | Gemma2, Gemma3 | Custom `DecoderLayer` |
| QK normalization | Gemma3, Qwen3 | Set `attn_qk_norm=True` in config |
| Sliding window attention | Gemma2, Gemma3 | Alternating layer types + `sliding_window` config |
| Different activation | Various | Set `hidden_act` in config (handled by `MLP`) |
| Biased attention projections | Phi, PhiMoE | Set `attn_qkv_bias=True`, `attn_o_bias=True` |
| LayerNorm epsilon | Whisper (`1e-5`) | Pass eps from config — default `1e-6` is wrong for many models |
| Custom attention scale | Granite | Pass `scale=config.attention_multiplier` to `Attention` |
| Embedding/logits/residual multipliers | Granite | Apply in `forward` — see troubleshooting §3 |
| MoE layers | PhiMoE, GPTOSS | See the **moe-models** skill |
| Vision encoder | Gemma3 | See the **multimodal-models** skill |
| Gated attention output | Qwen3.5 | Subclass `Attention` with doubled q_proj → Q+gate split |
| Hybrid layer types | Qwen3.5 | Use `config.layer_types` list to dispatch per-layer |
| Fused QKV / gate+up | ModernBERT | Split in `preprocess_weights` |
| Subclass-only (weight rename) | BLIP, TrOCR | Override only `preprocess_weights` |

### 4. Handle weight name mismatches (`preprocess_weights`)

If HuggingFace uses different weight names than your component tree, override
`preprocess_weights`:

```python
class MyCausalLMModel(CausalLMModel):
    def preprocess_weights(self, state_dict):
        renamed = {}
        for key, value in state_dict.items():
            new_key = key.replace("old_prefix.", "new_prefix.")
            renamed[new_key] = value
        return super().preprocess_weights(renamed)
```

> For detailed examples (fused QKV splitting, expert renames, weight-free
> norms, identity folding), read
> [`references/weight-preprocessing.md`](references/weight-preprocessing.md).

### 5. Register the model

Add to `_create_default_registry()` in `src/mobius/_registry.py`:

```python
from mobius.models import MyCausalLMModel
reg.register("my_model_type", MyCausalLMModel)
```

Also export from `src/mobius/models/__init__.py`.

### 6. Update `ArchitectureConfig.from_transformers` if needed

If the model has unusual top-level config fields (vocab size, head counts,
RoPE knobs, etc.), update `from_transformers()` in `src/mobius/_configs/_base.py`.
Use safe defaults (1.0 for multipliers, None for optional features) so
existing models are unaffected.

**For audio- or vision-capable models** (i.e. models whose HF config has an
`audio_config` or `vision_config` sub-object), prefer adding a per-model hook
under `src/mobius/_configs/per_model/` rather than editing the central file:

```python
# src/mobius/_configs/per_model/_my_model_vision.py
from mobius._configs._extractors import register_vision_hook

@register_vision_hook("my_model_type")
def _my_model_vision(config, parent_config, model_type, fields):
    fields.update(hidden_size=..., num_attention_heads=..., ...)
    return None  # contribute fields, defer VisionConfig instantiation
```

Then add the new module to `src/mobius/_configs/per_model/__init__.py` so its
side-effect registration runs at import time. The dispatcher filters hooks by
the declared model_type strings, so unrelated models never see your hook.

### 7. Write tests

See the **writing-tests** skill for full details.  At minimum:

1. **Add a config entry to `tests/_test_configs.py`** in the appropriate
   group (`CAUSAL_LM_CONFIGS`, `ENCODER_CONFIGS`, `SEQ2SEQ_CONFIGS`,
   `VISION_CONFIGS`, or `DETECTION_CONFIGS`):
   ```python
   CAUSAL_LM_CONFIGS: list[tuple[str, dict, bool]] = [
       ("my_model_type", {"hidden_act": "gelu"}, True),
   ]
   ```
   - `is_representative=True` if the model has unique behaviour
   - `is_representative=False` if it's an alias with no special config

   Then verify: `pytest tests/build_graph_test.py -k "my_model_type"`

2. **Add a small model to `tests/integration_test.py`** if a small
   checkpoint exists (< 1B parameters preferred).

3. **Testing large models with random weights:** Create a reduced HF model:
   ```python
   c = AutoConfig.from_pretrained("Qwen/Qwen3.5-27B")
   tc = c.text_config
   tc.num_hidden_layers = 4
   hf_model = Qwen3_5ForCausalLM._from_config(tc, dtype=torch.float32)
   ```
   Then use `build_from_module` with the HF state dict to compare logits.

4. **Test with the CLI:**
   ```bash
   mobius build --model org/my-small-model mymodel/output
   ```

### 8. Documentation

Model documentation is **auto-generated** from class metadata by
`docs/_generate_models.py`.  No manual doc update is needed if you set
`default_task`, `category`, and a good class docstring.

## Checklist

This is the **implementation** checklist.  For the full **definition-of-done**
quality checklist (L1–L5 tests, ORT GenAI, Foundry Local, Olive, multi-dtype),
see the [quality-checklist skill](../quality-checklist/SKILL.md).

- [ ] Model file in `src/mobius/models/` with Microsoft MIT copyright header
- [ ] Class has `default_task` and `category` attributes (if not standard text-generation)
- [ ] Class has a descriptive docstring (first paragraph used in generated docs)
- [ ] `preprocess_weights` handles any key mismatches
- [ ] Registered in `_create_default_registry()`
- [ ] Exported from `models/__init__.py`
- [ ] Config extraction works (`ArchitectureConfig.from_transformers`)
- [ ] Tiny config in `tests/_test_configs.py` (with `is_representative` flag)
- [ ] L2 YAML test case in `testdata/cases/` with `test_model_id`
- [ ] L3 synthetic parity passes (`tests/synthetic_parity_test.py -k "<model_type>"`)
- [ ] Integration test in `tests/integration_test.py` (if small checkpoint available)
- [ ] L4 golden file generated and committed (`testdata/golden/`)
- [ ] L5 generation golden file generated and committed
- [ ] ORT GenAI test added to `tests/ort_genai_test.py` (text-generation and VLM models)
- [ ] CLI build works (`mobius build --model ...`)
- [ ] Multi-dtype correctness verified (fp32, fp16, bf16)

**Note:** Default optimizer passes (CSE, deduplicate initializers, identity
elimination, remove unused nodes/opsets) are applied automatically.

## Example: minimal diff for a LLaMA-compatible model

If the new model is fully LLaMA-compatible, the entire change is:

```python
# _registry.py
reg.register("my_llama_variant", CausalLMModel)
```

No new model file needed.

## Example: adding a non-LLM model

For non-LLM architectures, use the appropriate base class and task.
See [`references/architecture-patterns.md`](references/architecture-patterns.md)
for the full model type → base class → task mapping table.

Quick reference:

| Model type | Base class | Task |
|------------|-----------|------|
| Encoder-only (BERT-like) | `BertModel` | `feature-extraction` |
| Encoder-decoder (BART/T5) | `BartForConditionalGeneration` | `seq2seq` |
| Vision (ViT-like) | `ViTModel` | `image-classification` |
| Multimodal (LLaVA-like) | `LLaVAModel` | `vision-language` |
| Diffusion denoiser | Custom | `denoising` |

Many models can be registered as aliases of existing classes if the
architecture matches.

## ModelPackage key conventions

Multi-model tasks produce a `ModelPackage` with standardised keys:

| Key | Role | On-disk path |
|-----|------|-------------|
| `"decoder"` | Text decoder | `decoder/model.onnx` |
| `"vision_encoder"` | Vision encoder | `vision_encoder/model.onnx` |
| `"audio_encoder"` | Audio encoder | `audio_encoder/model.onnx` |
| `"embedding"` | Embedding | `embedding/model.onnx` |
| `"model"` | Single-model (LLM) | `model.onnx` |

Legacy keys (`"model"` for decoder, `"vision"` for vision encoder, `"audio"`
for audio encoder, `"speech"` for audio) are mapped via `_MODEL_ROLE_MAP` in
`_builder.py` for backward compatibility.
New tasks should use the standardised keys above.

Module attribute names on the model class (declared in `ComponentSpec`)
also use `vision_encoder` and `audio_encoder`, matching the ModelPackage keys.

## Troubleshooting: common pitfalls

### 1. ORT rejects graph with `tensor(double)` / wrong dtype

NumPy creates float64 arrays by default.  Always pass `dtype=np.float32`:
```python
# BAD — creates float64
long_factor = np.array(config.rope_scaling["long_factor"])
# GOOD
long_factor = np.array(config.rope_scaling["long_factor"], dtype=np.float32)
```

### 2. Normalization type mismatch (RMSNorm vs LayerNorm)

**Symptom:** Large max abs diff (> 0.5) from HuggingFace.  Check what norm
class HF actually uses — LayerNorm and RMSNorm are NOT interchangeable.
Key cases: OLMo-1B uses weight-free LayerNorm, Whisper uses eps=1e-5,
Gemma adds 1 to RMSNorm weight.

### 3. Missing scaling multipliers

**Symptom:** Generation diverges after a few tokens.  Check for multiplier
config fields (`embedding_multiplier`, `attention_multiplier`,
`logits_scaling`, `residual_multiplier`).

**Critical:** Residual scaling direction matters:
```python
# CORRECT: residual + output * multiplier
# WRONG:   residual * multiplier + output
```

### 4. Config fields not extracted

**Symptom:** Model builds but multipliers default to 1.0. Add extraction
to `ArchitectureConfig.from_transformers()` in `src/mobius/_configs/_base.py`
with safe defaults. For audio/vision-specific fields, register a per-model
hook under `src/mobius/_configs/per_model/` instead — see step 6 above.

### 5. Debugging workflow for logit mismatches

1. Check max abs diff on prefill (> 0.01 suspicious, > 0.5 is a bug)
2. Check the HuggingFace norm class and epsilon
3. Check for model-specific config fields (multiplier/scaling/factor/epsilon)
4. Check weight dtype (numpy arrays must be float32)
5. Compare layer by layer (moderate diff → norm/residual issue; huge → wrong weights)

### 6. HF `_init_weights` corrupting checkpoint values

**Symptom:** HF reference inference is non-deterministic across model loads —
different argmax each time, despite identical inputs. Affects golden data
generation and L4 parity tests.

**Root cause:** Some HF models' `_init_weights` re-initialise parameters
with random values (e.g. `torch.rand`) AFTER `from_pretrained` loads the
checkpoint. Known cases:
- **NemotronH**: `_init_weights` clobbers Mamba2 `dt_bias` with `torch.rand()`

**Fix:** `_fix_nemotron_h_dt_bias()` in `mobius._testing.torch_reference`
reads correct values from safetensors files and patches them in-place.
Always call after `from_pretrained` for NemotronH models.

**Diagnosis pattern:** Load the model twice and compare outputs. If argmax
differs between loads, suspect `_init_weights` corruption. Set
`torch.manual_seed(42)` before `from_pretrained` — if that makes outputs
deterministic, `_init_weights` is the culprit. Then compare specific
parameters between the loaded model and the safetensors checkpoint.

### 7. `AttributeError: '<X>TextModel' object has no attribute 'config'`

**Symptom:** Building (or running build-graph / GQA rewrite-rule tests for)
a model raises `AttributeError: '…TextModel' object has no attribute
'config'` from inside `TextModel.forward` (e.g. `_gqa_local_window_size`).

**Root cause:** The base `TextModel.__init__` sets `self.config = config`,
and `TextModel.forward` relies on it (sliding-window detection, etc.). A
`TextModel` **subclass** that overrides `__init__` with
`nn.Module.__init__(self)` — instead of `super().__init__(config)` — to swap
in a custom decoder layer must re-establish the contract by setting
`self.config = config` itself. Forgetting it crashes only that model.

**Fix:** Add `self.config = config` right after `nn.Module.__init__(self)`
in the subclass `__init__`. Real examples: `Glm4TextModel` (`models/glm.py`),
`_LoRATextModel` (`models/phi.py`). Do **not** paper over it with
`getattr(self, "config", None)` in the base — that silently disables
config-driven features for the offending subclass.

> For additional troubleshooting (gated attention split ordering, DeltaNet
> scaling, identity node folding, fp32 upcast patterns, multi-token prefill,
> embedding table off-by-one), read
> [`references/weight-preprocessing.md`](references/weight-preprocessing.md).

## Reference examples

| Complexity | File | Why |
|---|---|---|
| **Minimal** | `models/phi3.py` | Only overrides `preprocess_weights()` |
| **Minimal encoder** | `models/layoutlmv3.py` | Encoder subclass, weight rename only |
| **Moderate** | `models/gemma.py` | Custom attention, MLP, normalization |
| **Complex** | `models/qwen3_tts.py` | 4-model TTS architecture |

> For the full reference implementation table (20+ models) and KV sharing
> patterns, read
> [`references/architecture-patterns.md`](references/architecture-patterns.md).

## Cross-references

- **[moe-models](../moe-models/SKILL.md)** — MoE gate variants, expert weight naming
- **[multimodal-models](../multimodal-models/SKILL.md)** — Vision encoders, projectors, VisionLanguageTask
- **[diffusion-models](../diffusion-models/SKILL.md)** — UNet, VAE, DiT, Flux, SD3
- **[weight-name-alignment](../weight-name-alignment/SKILL.md)** — Aligning ONNX parameter names with HF
- **[writing-tests](../writing-tests/SKILL.md)** — Unit, integration, and generation test patterns
- **[quality-checklist](../quality-checklist/SKILL.md)** — L1–L5 definition of done
- **[reusable-components](../reusable-components/SKILL.md)** — Component library and design principles
