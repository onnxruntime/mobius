---
name: adding-vl-models
description: >
  Step-by-step guide for adding vision-language (VL) models to mobius.
  Covers VL anatomy, the 3-model ONNX split, task selection, vision encoders
  (CLIP, SigLIP, SigLIP-2, InternViT, RADIO), projector variants (MLP,
  Linear, Gemma3), config extraction, weight mapping, common pitfalls,
  build-graph tests, and registry wiring. Use this skill when adding any
  model that processes both images and text.
---

# Skill: Adding Vision-Language Models

## When to use

Use this skill when adding a model that processes both images and text:
LLaVA, LLaVA-NeXT, LLaVA-OneVision, Phi-3-Vision, Phi-4-MM, PaliGemma,
InternVL2, Qwen2.5-VL, Qwen3-VL, Gemma3, Mllama (Llama 3.2 Vision),
BLIP-2, Pixtral, Molmo, Florence2, GLM4V, and similar models.

---

## 1. VL model anatomy

Every VL model in mobius is decomposed into **three sub-models**, each
exported as a separate ONNX file:

```
pixel_values ──► [vision] ──► image_features
                                    │
input_ids ──► [embedding] ◄─────────┘   ──► inputs_embeds
                                                  │
inputs_embeds ──► [decoder] ──────────────────────┘ ──► logits + KV cache
```

| ONNX model | I/O contract | Purpose |
|------------|-------------|---------|
| **vision** | `pixel_values [B,C,H,W] → image_features [N, text_hidden]` | Vision encoder + projector |
| **embedding** | `input_ids [B,S] + image_features [N, H] → inputs_embeds [B,S,H]` | Token embedding + image scatter |
| **decoder** | `inputs_embeds [B,S,H] → logits + KV cache` | Causal text generation |

The `VisionLanguageTask` (`tasks/_vision_language_3model.py`) wires all three
into a `ModelPackage({"decoder": …, "vision": …, "embedding": …})`.

### Why three models?

ORT GenAI runs the three models at different frequencies:
- **vision**: once per image at prefill time
- **embedding**: once at prefill (mixes text + image features)
- **decoder**: every token during generation (with KV cache)

Splitting avoids re-running the (expensive) vision encoder each step.

---

## 2. Task selection

Choose the task based on the model family:

| Task | TASK_REGISTRY key | Models |
|------|--------------------|--------|
| `VisionLanguageTask` | `"vision-language"` | LLaVA, LLaVA-NeXT, LLaVA-OneVision, PaliGemma, Pixtral, Molmo, Video-LLaVA, InternVL2, Gemma3, GLM4V, BLIP-2 |
| `QwenVLTask` | `"qwen-vl"` | Qwen2.5-VL, Qwen3-VL (MRoPE + packed vision) |
| `HybridQwenVLTask` | `"hybrid-qwen-vl"` | Qwen3.5-VL (MRoPE + DeltaNet hybrid cache) |
| `Qwen3VLVisionLanguageTask` | `"qwen3-vl-vision-language"` | Qwen3-VL single-model variant |
| `MllamaVisionLanguageTask` | `"mllama-vision-language"` | Mllama (Llama 3.2 Vision) — cross-attention |
| `MultiModalTask` | `"multimodal"` | Phi4-MM (audio + vision + text) |

**Default for new models**: use `VisionLanguageTask` unless the model has
MRoPE, hybrid cache, or cross-attention (see §8 for non-standard cases).

---

## 3. Common vision encoders

| Encoder | Used by | mobius component | Notes |
|---------|---------|-----------------|-------|
| SigLIP (ViT-SO/400M) | LLaVA, PaliGemma, InternVL2, Gemma3, BLIP-2 | `VisionModel` | Pre-norm ViT, no CLS token |
| CLIP ViT-L/14 | LLaVA-1.5, Phi-3-Vision | `VisionModel` | Post-norm ViT, CLS token |
| SigLIP-2 | Phi-4-MM, Phi-4-SigLIP (phi4-siglip) | `VisionModel` (with S2 tiling wrapper) | Multi-resolution |
| InternViT (InternVL2) | InternVL2 family | Custom `_InternViT*` classes in `internvl.py` | Fused QKV, layer scale, CLS token |
| Qwen2.5-VL ViT | Qwen2.5-VL | `Qwen25VLVisionModel` in `components/` | Conv3D patches, 2D-RoPE, window attention |
| Qwen3-VL ViT | Qwen3-VL | `Qwen3VLVisionModel` in `components/` | Similar to Qwen2.5 with full-attn blocks |
| RADIO (ViT-H/16 + CPE) | NemotronH VL | Not yet implemented | Conditional position encoding (40 LOC new component) |

### Using `VisionModel` (SigLIP/CLIP pattern)

`VisionModel` in `components/_vision.py` implements the standard SigLIP-style
ViT and automatically handles the `vision_tower.vision_model.*` HuggingFace
naming convention.

```python
from mobius.components import VisionModel

class _MyVisionEncoder(nn.Module):
    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.vision_tower = VisionModel(config)  # HF: vision_tower.vision_model.*
        self.multi_modal_projector = MLPMultiModalProjector(
            vision_hidden_size=config.vision.hidden_size,
            text_hidden_size=config.hidden_size,
        )

    def forward(self, op: builder.OpBuilder, pixel_values: ir.Value):
        vision_features = self.vision_tower(op, pixel_values)   # [B, N, vision_hidden]
        return self.multi_modal_projector(op, vision_features)  # [B, N, text_hidden]
```

`VisionModel` nests as `self.vision_tower` → HF keys like
`vision_tower.vision_model.embeddings.patch_embedding.weight` align
automatically with `self.vision_tower.vision_model.embeddings.patch_embedding.weight`.

---

## 4. Projector variants

| Projector | Architecture | Models |
|-----------|-------------|--------|
| `MLPMultiModalProjector` | Linear → GELU → Linear | LLaVA, LLaVA-NeXT, InternVL2, Molmo, Pixtral |
| `LinearMultiModalProjector` | Single Linear | PaliGemma, Idefics2/3, Florence2 |
| `Gemma3MultiModalProjector` | AvgPool2d → RMSNorm → MatMul | Gemma3 |

```python
from mobius.components import (
    MLPMultiModalProjector,
    Gemma3MultiModalProjector,
)

# LLaVA-style (most common)
projector = MLPMultiModalProjector(
    vision_hidden_size=config.vision.hidden_size,  # e.g. 1024
    text_hidden_size=config.hidden_size,           # e.g. 4096
    bias=True,
)

# Gemma3: pools from 4096 patch tokens down to 256 mm_tokens_per_image
projector = Gemma3MultiModalProjector(
    vision_hidden_size=config.vision.hidden_size,           # 1152
    text_hidden_size=config.hidden_size,                    # 2560
    patches_per_image=config.vision.image_size // config.vision.patch_size,  # 448//14=32
    tokens_per_image=config.vision.mm_tokens_per_image,     # 256
    norm=OffsetRMSNorm(config.vision.hidden_size, eps=config.vision.norm_eps),
)
```

---

## 5. Config structure

### `VisionConfig` fields (from `_configs.py`)

```python
@dataclasses.dataclass
class VisionConfig:
    hidden_size: int | None = None          # Vision transformer hidden dim
    intermediate_size: int | None = None    # MLP intermediate dim
    num_hidden_layers: int | None = None    # Number of encoder layers
    num_attention_heads: int | None = None  # Attention heads
    image_size: int | None = None           # Input image resolution (e.g. 336, 448)
    patch_size: int | None = None           # Patch size (e.g. 14, 16)
    norm_eps: float = 1e-6                  # LayerNorm/RMSNorm epsilon
    mm_tokens_per_image: int | None = None  # Projected tokens per image (for Gemma3)
    image_token_id: int | None = None       # Special token ID for image placeholders
    in_channels: int = 3                    # Input channels (usually 3)
    # Qwen VL extras:
    spatial_merge_size: int = 2             # Spatial merging ratio
    temporal_patch_size: int = 2            # Temporal patch size
    # Phi4MM extras:
    image_crop_size: int | None = None      # HD crop size
    lora: dict | None = None                # LoRA config
```

### How `ArchitectureConfig.from_transformers` extracts vision fields

`_extract_vision_config()` in `_configs.py` automatically reads:
- `hf_config.vision_config.*` → `VisionConfig` fields
- `hf_config.image_token_id` → `config.vision.image_token_id`
- `hf_config.mm_tokens_per_image` → `config.vision.mm_tokens_per_image`

For most models, no manual config extraction is needed — just pass
the top-level HF config to `ArchitectureConfig.from_transformers()`.

### Manual config for build-graph tests

Use `VisionConfig` directly in test `_MODEL_CONFIGS`:

```python
from mobius._configs import VisionConfig

_TINY_VISION = VisionConfig(
    hidden_size=32,
    intermediate_size=64,
    num_hidden_layers=1,
    num_attention_heads=2,
    image_size=16,
    patch_size=8,
    norm_eps=1e-6,
)

# In _MODEL_CONFIGS:
("llava", {"vision": _TINY_VISION, "image_token_id": 32000}, True),
```

---

## 6. Weight mapping

### Standard VLM weight layout (LLaVA family)

```
language_model.model.embed_tokens.weight    → decoder/embedding
language_model.model.layers.N.*             → decoder
language_model.lm_head.weight               → decoder
vision_tower.vision_model.embeddings.*      → vision
vision_tower.vision_model.encoder.layers.*  → vision
multi_modal_projector.linear_1.*            → vision
multi_modal_projector.linear_2.*            → vision
```

### `vlm_decoder_weights` and `vlm_embedding_weights`

Two helpers in `_weight_utils.py` handle the common prefix stripping:

```python
from mobius._weight_utils import vlm_decoder_weights, vlm_embedding_weights

class _MyDecoder(nn.Module):
    def preprocess_weights(self, state_dict):
        # Strips "language_model." prefix; handles weight tying
        return vlm_decoder_weights(state_dict, tie=self.config.tie_word_embeddings)

class _MyEmbedding(nn.Module):
    def preprocess_weights(self, state_dict):
        # Filters keys containing "embed_tokens", strips "language_model.model." prefix
        return vlm_embedding_weights(state_dict)
```

`vlm_decoder_weights` signature:
```python
def vlm_decoder_weights(
    state_dict,
    prefix="language_model.",   # stripped prefix
    tie=False,                  # copy embed → lm_head if head missing
    embed_key="model.embed_tokens.weight",
    head_key="lm_head.weight",
) -> dict
```

### Top-level `preprocess_weights` for weight tying

When `tie_word_embeddings=True`, the HF checkpoint omits `lm_head.weight`.
Inject it before dispatching to sub-modules:

```python
class MyVLModel(nn.Module):
    def preprocess_weights(self, state_dict):
        if self.config.tie_word_embeddings:
            embed_key = "language_model.model.embed_tokens.weight"
            head_key = "language_model.lm_head.weight"
            if head_key not in state_dict and embed_key in state_dict:
                state_dict[head_key] = state_dict[embed_key]
        return state_dict
```

### Non-standard prefix examples

| Model | HF prefix | `vlm_decoder_weights(prefix=...)` |
|-------|-----------|-----------------------------------|
| LLaVA, Gemma3, Mllama | `"language_model."` | default |
| InternVL2 | `"language_model."` | default |
| BLIP-2 | `"language_model."` | default |
| GLM4V | `"language_model."` | default |

---

## 7. Step-by-step: adding a standard VL model

### Step 1 — Identify sub-model boundaries

Read the HuggingFace `modeling_<name>.py` to find:
1. Vision encoder class and its config keys
2. Projector type (MLP-2x, single Linear, pooling+norm, etc.)
3. Text model class and top-level weight prefixes
4. `image_token_id` in the tokenizer/config

### Step 2 — Create `models/<name>.py`

Follow the LLaVA/Gemma3 pattern: three inner classes + one outer class.

```python
# models/mymodel.py
"""MyModel vision-language model (3-model split)."""

from __future__ import annotations
from typing import TYPE_CHECKING
import torch
from onnxscript import nn
from onnxscript._internal import builder
from mobius._configs import ArchitectureConfig
from mobius._weight_utils import vlm_decoder_weights, vlm_embedding_weights
from mobius.components import Embedding, Linear, MLPMultiModalProjector, VisionModel
from mobius.models.base import TextModel

if TYPE_CHECKING:
    import onnx_ir as ir


class _MyDecoderModel(nn.Module):
    """Text decoder: inputs_embeds → logits + KV cache."""

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.config = config
        self.model = TextModel(config)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, op, inputs_embeds, attention_mask, position_ids, past_key_values=None):
        hidden_states, present_kv = self.model(
            op,
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
        )
        return self.lm_head(op, hidden_states), present_kv

    def preprocess_weights(self, state_dict):
        return vlm_decoder_weights(state_dict, tie=self.config.tie_word_embeddings)


class _MyVisionEncoderModel(nn.Module):
    """Vision encoder: pixel_values → image_features."""

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.vision_tower = VisionModel(config)
        self.multi_modal_projector = MLPMultiModalProjector(
            vision_hidden_size=config.vision.hidden_size,
            text_hidden_size=config.hidden_size,
        )

    def forward(self, op, pixel_values):
        return self.multi_modal_projector(op, self.vision_tower(op, pixel_values))

    def preprocess_weights(self, state_dict):
        return {
            k: v
            for k, v in state_dict.items()
            if k.startswith(("vision_tower.", "multi_modal_projector."))
        }


class _MyEmbeddingModel(nn.Module):
    """Token embedding + image feature scatter."""

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.image_token_id = config.image_token_id or 0

    def forward(self, op, input_ids, image_features):
        text_embeds = self.embed_tokens(op, input_ids)
        image_mask = op.Equal(input_ids, op.Constant(value_int=self.image_token_id))
        image_mask_3d = op.Unsqueeze(image_mask, [-1])
        # Replace image-token positions with gathered vision features
        cumsum = op.CumSum(op.Cast(image_mask, to=7), op.Constant(value_int=1))
        indices = op.Clip(op.Sub(cumsum, op.Constant(value_int=1)), op.Constant(value_int=0))
        gathered = op.Gather(image_features, indices, axis=0)
        return op.Where(image_mask_3d, gathered, text_embeds)

    def preprocess_weights(self, state_dict):
        return vlm_embedding_weights(state_dict)


class MyVLModel(nn.Module):
    """MyModel vision-language model (3-model split)."""

    default_task: str = "vision-language"
    category: str = "Multimodal"

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.config = config
        self.decoder = _MyDecoderModel(config)
        self.vision_encoder = _MyVisionEncoderModel(config)
        self.embedding = _MyEmbeddingModel(config)

    def forward(self, op, **kwargs):
        raise NotImplementedError(
            "MyVLModel uses VisionLanguageTask which calls each sub-module separately."
        )

    def preprocess_weights(self, state_dict):
        # Weight tying: inject lm_head.weight if missing
        if self.config.tie_word_embeddings:
            embed_key = "language_model.model.embed_tokens.weight"
            head_key = "language_model.lm_head.weight"
            if head_key not in state_dict and embed_key in state_dict:
                state_dict[head_key] = state_dict[embed_key]
        return state_dict
```

### Step 3 — Export from `models/__init__.py`

```python
from mobius.models.mymodel import MyVLModel
```

### Step 4 — Register in `_registry.py`

```python
# In _create_default_registry():
from mobius.models.mymodel import MyVLModel

reg.register("my_model_type", MyVLModel, task="vision-language")
```

Add a canonical HF model ID to `_MODEL_CONFIGS` for integration tests:

```python
"my_model_type": "org/my-model-7b",
```

### Step 5 — Add build-graph test in `tests/_test_configs.py`

```python
("my_model_type", {"vision": _TINY_VISION, "image_token_id": 32000}, True),
```

The third element is `True` to run the test, `False` to skip (mark as `xfail`).

### Step 6 — Run the test

```bash
PYTHONPATH=$(pwd)/src python -m pytest tests/build_graph_test.py -k "my_model_type" -sv
```

### Step 7 — Full test suite before committing

```bash
PYTHONPATH=$(pwd)/src python -m pytest tests/build_graph_test.py tests/synthetic_parity_test.py \
    -q --tb=short -n auto 2>&1 | tail -20
```

---

## 8. Non-standard VL tasks

### Qwen2.5-VL / Qwen3-VL (MRoPE + packed vision)

These models use **3D multi-modal RoPE** (temporal, height, width) and pack
all image patches from a batch into a flat `[total_patches, pixel_dim]` tensor.

Use `QwenVLTask` (or `HybridQwenVLTask` for Qwen3.5-VL with DeltaNet):

```python
reg.register("qwen2_vl", Qwen25VLCausalLMModel, task="qwen-vl")
```

Decoder position_ids shape changes from `[B, S]` → `[3, B, S]`.
Vision input changes from `[B, C, H, W]` → `[total_patches, pixel_dim]` + `image_grid_thw`.

### Mllama / Llama 3.2 Vision (cross-attention)

Mllama interleaves self-attention and cross-attention layers. Cross-attention
layers attend to vision encoder output. Use `MllamaVisionLanguageTask`:

```python
reg.register("mllama", MllamaCausalLMModel, task="mllama-vision-language")
```

Decoder gets an extra `cross_attention_states [B, N, H]` input (filled on
prefill, empty tensor on decode steps). Config must use `MllamaConfig` with
`cross_attention_layers` list. See `tasks/_vision_language_3model.py` and
`models/mllama.py` for the full implementation.

### BLIP-2 (Q-Former)

BLIP-2 adds a **Q-Former cross-attention module** between the vision encoder
and the language model. The Q-Former produces a fixed number of query tokens
(`num_query_tokens`, typically 32) that compress variable-length vision
features. Uses standard `VisionLanguageTask`. See `models/blip2.py`.

### Phi4-MM (audio + vision + text)

Phi4-MM processes speech and images simultaneously using `Phi4MMMultiModalTask`
(task key: `"phi4mm-multimodal"`). See the `phi4mm-component-parity` skill.

---

## 9. Common pitfalls

### ❌ Dual-use text model (shared embedding vs separate decoder)

Many VLMs have `tie_word_embeddings=True`: the `lm_head.weight` and
`embed_tokens.weight` are the **same tensor** in the HF checkpoint. The HF
file only stores one of them. Always inject the missing key in
`preprocess_weights` before sub-modules are called.

### ❌ Wrong `image_token_id`

Different models use very different IDs for image placeholder tokens:

| Model | `image_token_id` |
|-------|-----------------|
| LLaVA-1.5 (vicuna) | 32000 |
| LLaVA-OneVision (Qwen2) | 151655 |
| Gemma3 | 262144 |
| InternVL2 | 151667 |
| Mllama | 128256 |
| BLIP-2 | 50265 |

Always read `image_token_id` from `hf_config` rather than hardcoding.

### ❌ Missing `vision` field in config during build-graph test

`VisionLanguageTask._build_vision` uses `config.vision.image_size`. If
`config.vision` is `None`, it falls back to `224`. Always set `vision` in
`_MODEL_CONFIGS` using a `VisionConfig` instance.

### ❌ `VisionModel` naming mismatch

`VisionModel` wraps `_VisionModelInner` as `self.vision_model`, so the full
path is `self.<attr_name>.vision_model.*`. If you name the attribute
`vision_tower` (as in LLaVA), ONNX parameter paths match HF keys:
`vision_tower.vision_model.embeddings.patch_embedding.weight`.

If you use a different attribute name (e.g. `encoder`), you need a rename in
`preprocess_weights`.

### ❌ `forward()` raising on the top-level module

`VisionLanguageTask` calls `module.decoder(op, ...)`, `module.vision_encoder(op, ...)`,
and `module.embedding(op, ...)` **directly** — it never calls `module(op, ...)`.
The top-level `forward` should raise `NotImplementedError` to signal this.

### ❌ Weight namespace collisions in sub-module dispatch

Each of the three sub-models (`decoder`, `vision_encoder`, `embedding`) must
have a **unique attribute name** on the top-level module class. `VisionLanguageTask`
accesses `module.decoder`, `module.vision_encoder`, `module.embedding`. Do not
rename these attributes unless you override the task's `build()` method.

### ❌ `weight_namespace` not set for shared text weights

For models where the same text model class is used standalone (causal LM) and
inside VLM, the decoder's `preprocess_weights` must correctly strip the
`"language_model."` prefix. The `vlm_decoder_weights` helper handles this.

---

## 10. Vision encoder deep-dives

### InternViT (InternVL2)

InternViT differs from standard SigLIP in three ways:
1. **Fused QKV**: single `qkv` linear (3×hidden) instead of separate Q/K/V
2. **Layer scale**: learnable vectors `ls1`/`ls2` that multiply sub-layer
   output before residual add
3. **CLS token**: prepended before patch tokens (num_patches + 1 positions)

InternVL2 also uses a **pixel shuffle downsampling** (`downsample_ratio=0.5`)
before the MLP projector, reducing spatial resolution by 2×. See `internvl.py`
for the `_InternViT*` classes and the 4-element projector (`LayerNorm → Linear
→ GELU → Linear`).

### Qwen2.5/3-VL vision encoder

Key differences from standard SigLIP:

| Feature | SigLIP | Qwen VL |
|---------|--------|---------|
| Patch embedding | Conv2d | Conv3d (temporal × spatial) |
| Position encoding | Learnable table | 2D RoPE (height, width) |
| Attention | Standard | Windowed + full-attention alternating |
| Input shape | `[B, C, H, W]` | `[total_patches, pixel_dim]` (packed) |
| Normalization | LayerNorm | RMSNorm |
| Output | `[B, N, hidden]` | `[total_merged_patches, out_hidden]` (spatially merged) |

See `components/_qwen25_vl_vision.py` and `components/_qwen3_vl_vision.py`.

### RADIO (for NemotronH VL — not yet in mobius)

RADIO is ViT-H/16 with **conditional position encoding (CPE)**: a depthwise
Conv2d that injects position information into patch features adaptively (no
fixed grid). This enables dynamic-resolution images. CPE is ~40 LOC new
component; full NemotronH VL is estimated at ~Hard complexity. See
`architect-f87e0093/microsoft_missing_models_design_briefs.md` in session
artifacts for the full design brief.

---

## 11. Registering custom configs

Some VL model types need specialized `ArchitectureConfig` subclasses:

| Model | Config class | Extra fields |
|-------|-------------|-------------|
| Mllama | `MllamaConfig` | `cross_attention_layers: list[int]` |
| Standard VL | `VisionLanguageConfig` | Standard + `VisionConfig` sub-config |
| InternVL2 | `ArchitectureConfig` | Uses `_extract_vision_config` autodetect |

To use a custom config class, pass it as `config_cls` to `reg.register`:

```python
from mobius._configs import MllamaConfig

reg.register("mllama", MllamaCausalLMModel,
             task="mllama-vision-language",
             config_cls=MllamaConfig)
```

---

## 12. Golden test YAML for VL models

VL models use `task_type: "vision-language"` in YAML test cases.  Because the
golden generation requires an actual image, most VL golden tests are currently
marked `skip_reason: "requires image input"` or not yet generated.

If you do add a VL golden YAML, pin the HF revision SHA:

```yaml
model_id: "llava-hf/llava-1.5-7b-hf"
revision: "<sha>"
task_type: "vision-language"
inputs:
  prompts: ["USER: <image>\nDescribe the image. ASSISTANT:"]
  image_url: "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg"
level: "L4"
```

---

## 13. Quick reference: file locations

| File | Purpose |
|------|---------|
| `src/mobius/models/llava.py` | Reference implementation (LLaVA 3-split) |
| `src/mobius/models/gemma3.py` | Gemma3 (AvgPool projector, OffsetRMSNorm) |
| `src/mobius/models/internvl.py` | InternVL2 (fused QKV, pixel shuffle, CLS) |
| `src/mobius/models/mllama.py` | Mllama (cross-attention, dual KV cache) |
| `src/mobius/models/blip2.py` | BLIP-2 (Q-Former) |
| `src/mobius/tasks/_vision_language_3model.py` | `VisionLanguageTask` + Qwen/Mllama variants |
| `src/mobius/tasks/_vision_language.py` | `Qwen3VLVisionLanguageTask` (single-model) |
| `src/mobius/components/_vision.py` | `VisionModel`, `VisionEncoder`, `VisionEncoderLayer` |
| `src/mobius/components/_multimodal.py` | Projectors, `InputMixer` |
| `src/mobius/_weight_utils.py` | `vlm_decoder_weights`, `vlm_embedding_weights` |
| `src/mobius/_configs.py` | `VisionConfig`, `VisionLanguageConfig`, `MllamaConfig` |
| `src/mobius/_registry.py` | `_create_default_registry()` — all VL model registrations |
| `tests/_test_configs.py` | `_TINY_VISION`, VL test config entries |
| `tests/build_graph_test.py` | `TestBuildGraphVisionLanguage` class |
