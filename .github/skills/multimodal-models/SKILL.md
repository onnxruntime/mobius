---
name: multimodal-models
description: >
  How to add multimodal (vision + language) models to mobius.  Covers
  projector variants (Gemma3, MLP, Linear), VisionModel encoder,
  InputMixer, VisionLanguageTask, the 3-model ONNX split, task variants,
  vlm weight helpers, InternViT/RADIO encoders, Mllama cross-attention,
  BLIP-2 Q-Former, image_token_id reference, and weight name mappings.
  Use this skill when adding a model that processes both images and text.
---

# Skill: Multimodal (Vision + Language) Models

## Related skills

- **`ort-genai-config`** — Complete reference for `genai_config.json` and
  `processor_config.json` formats for ORT GenAI deployment.
- **`debugging-vl-pipeline`** — Systematic debugging of VL model output parity.
- **`scan-and-multi-image`** — Multi-image support via ONNX Scan op.

## When to use

Use this skill when adding a model that processes both images and text — such
as Gemma3, LLaVA, LLaVA-NeXT, Phi-3-Vision, PaliGemma, InternVL2, Pixtral,
Idefics2/3, Molmo, Florence2, or Video-LLaVA.

## Architecture overview

```
pixel_values ──► VisionModel ──► MultiModalProjector ──► InputMixer ──┐
                                                                       ├──► TextDecoder ──► logits
input_ids ──────► Embedding ──────────────────────────────────────────┘
```

### Key components

| Component | File | Purpose |
|-----------|------|---------|
| `VisionModel` | `components/_vision.py` | SigLIP-style patch embedding + transformer encoder |
| `PatchEmbedding` | `components/_vision.py` | Conv2d → positional embedding |
| `Gemma3MultiModalProjector` | `components/_multimodal.py` | AvgPool2d → RMSNorm → MatMul |
| `MLPMultiModalProjector` | `components/_multimodal.py` | Linear → GELU → Linear |
| `LinearMultiModalProjector` | `components/_multimodal.py` | Single Linear |
| `InputMixer` | `components/_multimodal.py` | Scatter vision embeddings at image-token positions |
| `VisionLanguageTask` | `tasks/__init__.py` | ONNX I/O contract with `pixel_values` input |

## Projector variants

Different model families use different projectors to bridge vision and text
embedding spaces.  Choose the one that matches the HuggingFace implementation:

| Projector | Architecture | Models |
|-----------|-------------|--------|
| `Gemma3MultiModalProjector` | AvgPool2d → RMSNorm → MatMul | Gemma3 |
| `MLPMultiModalProjector` | Linear → GELU → Linear | LLaVA, LLaVA-NeXT, VipLLaVA, Phi-4-MM, InternVL2, Pixtral, Molmo |
| `LinearMultiModalProjector` | Single Linear | PaliGemma, Qwen2-Audio, Idefics2, Florence2 |

### Gemma3MultiModalProjector

```python
Gemma3MultiModalProjector(
    vision_hidden_size=1152,     # SigLIP hidden dim
    text_hidden_size=2560,       # Text model hidden dim
    patches_per_image=64,        # sqrt(num_patches) per side
    tokens_per_image=256,        # mm_tokens_per_image from config
    norm=Gemma3RMSNorm(1152),    # Gemma3-specific RMSNorm with +1 offset
)
```

The pooling kernel is computed as `patches_per_image / sqrt(tokens_per_image)`.
For Gemma3-4B: `64 / 16 = 4`, so `AvgPool2d(kernel_size=4, stride=4)`.

### MLPMultiModalProjector

```python
MLPMultiModalProjector(
    vision_hidden_size=1024,
    text_hidden_size=4096,
    bias=True,
)
```

Two-layer MLP with GELU activation.  The most common projector pattern.

### LinearMultiModalProjector

```python
LinearMultiModalProjector(
    vision_hidden_size=1024,
    text_hidden_size=4096,
    bias=True,
)
```

Simple single linear layer.

## Step-by-step: adding a new multimodal model

### 1. Identify the projector architecture

Look at the HuggingFace source in `modeling_<model>.py`:

```bash
grep -n "class.*Projector\|class.*projector" \
    transformers/models/<model>/modeling_<model>.py
```

Match it to one of the three projector variants, or create a new one.

### 2. Extract vision config

Multimodal HF configs have a `vision_config` sub-object.  Extract vision
fields in the test or integration code:

```python
hf_config = transformers.AutoConfig.from_pretrained(model_id)
text_config = hf_config.text_config
vision_config = hf_config.vision_config

config = ArchitectureConfig.from_transformers(text_config)
# Add vision fields
config.vision_hidden_size = vision_config.hidden_size
config.vision_intermediate_size = vision_config.intermediate_size
config.vision_num_hidden_layers = vision_config.num_hidden_layers
config.vision_num_attention_heads = vision_config.num_attention_heads
config.vision_image_size = vision_config.image_size
config.vision_patch_size = vision_config.patch_size
config.vision_norm_eps = getattr(vision_config, "layer_norm_eps", 1e-6)
config.mm_tokens_per_image = getattr(hf_config, "mm_tokens_per_image", None)
config.image_token_id = getattr(hf_config, "image_token_id", None)
```

### 3. Create the model class

**Important:** Always invoke child modules through `__call__` (not by
accessing their sub-modules directly) so that `onnxscript.nn.Module` pushes
the correct naming context.  Direct access like
`self.language_model.model.embed_tokens(op, x)` skips intermediate naming
scopes and produces wrong initializer names.

The recommended pattern is to pass vision embeddings as a kwarg through the
`__call__` chain, and have the text model perform the mixing internally:

```python
class _MyTextModelForMultimodal(MyTextModel):
    """Text model that mixes vision embeddings into the input."""

    def __init__(self, config):
        super().__init__(config)
        self.input_mixer = InputMixer(image_token_id=config.image_token_id or 0)

    def forward(self, op, input_ids, attention_mask, position_ids,
                past_key_values=None, vision_embeddings=None):
        hidden_states = self.embed_tokens(op, input_ids)
        if vision_embeddings is not None:
            hidden_states = self.input_mixer(
                op, hidden_states, vision_embeddings, input_ids
            )
        return super().forward(
            op, input_ids, attention_mask, position_ids,
            past_key_values=past_key_values, inputs_embeds=hidden_states,
        )


class _MyForMultimodalLM(MyCausalLMModel):
    """CausalLM that passes vision_embeddings to the text model."""

    def __init__(self, config):
        nn.Module.__init__(self)
        self.config = config
        self.model = _MyTextModelForMultimodal(config)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)


class MyMultiModalModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vision_tower = VisionModel(config)
        self.multi_modal_projector = MLPMultiModalProjector(
            vision_hidden_size=config.vision_hidden_size,
            text_hidden_size=config.hidden_size,
        )
        self.language_model = _MyForMultimodalLM(config)

    def forward(self, op, input_ids, attention_mask, position_ids, pixel_values,
                past_key_values=None):
        # 1. Encode vision
        vision_features = self.vision_tower(op, pixel_values)
        vision_embeddings = self.multi_modal_projector(op, vision_features)

        # 2. Pass through __call__ chain — naming is correct automatically
        return self.language_model(
            op, input_ids, attention_mask, position_ids,
            past_key_values=past_key_values,
            vision_embeddings=vision_embeddings,
        )
```

See `models/gemma3.py` for the full working example.

### 4. Handle weight name mismatches

Multimodal HF models often prefix text weights differently:

| HF key | Our key |
|--------|---------|
| `language_model.model.layers.0.…` | `layers.0.…` |
| `vision_tower.vision_model.encoder.…` | `vision_tower.encoder.…` |
| `multi_modal_projector.mm_input_projection_weight` | `multi_modal_projector.weight` |

Implement `preprocess_weights` to strip prefixes and rename keys.

### 5. Handle weight tying

If `tie_word_embeddings=True`, the HF checkpoint may not include
`lm_head.weight`.  Copy it from `embed_tokens.weight`:

```python
if self.config.tie_word_embeddings:
    if "lm_head.weight" not in renamed and "embed_tokens.weight" in renamed:
        renamed["lm_head.weight"] = renamed["embed_tokens.weight"]
```

### 6. Use VisionLanguageTask

Build with the `VisionLanguageTask` to add `pixel_values` to graph inputs:

```python
from mobius.tasks import VisionLanguageTask

onnx_model = build_from_module(module, config, task=VisionLanguageTask())
```

### 7. PatchEmbedding naming

`PatchEmbedding` has three parameters.  These use explicit `name=` because the
attribute names don't match the desired ONNX names (e.g. `patch_embedding`
needs to map to `patch_embedding.weight`):

```python
self.patch_embedding = nn.Parameter([...], name="patch_embedding.weight")
self.patch_embedding_bias = nn.Parameter([...], name="patch_embedding.bias")
self.position_embedding = nn.Parameter([...], name="position_embedding.weight")
```

In most cases, `name=` is **not needed** because `nn.Module.__setattr__`
automatically sets the parameter name from the attribute name.  Only use
`name=` when the attribute name differs from the desired ONNX initializer name.

## Qwen2.5-VL / Qwen3-VL vision encoder specifics

These models use a **custom vision encoder** (not SigLIP) with unique
architectural features. The encoder is in
`components/_qwen25_vl_vision.py` and `components/_qwen3_vl_vision.py`.

### Architecture differences from standard VisionModel

| Feature | Standard (SigLIP) | Qwen2.5-VL / Qwen3-VL |
|---------|-------------------|----------------------|
| Patch embedding | Conv2d | **Conv3d** (temporal + spatial) |
| Position encoding | Learnable embedding | **2D rotary** (height, width) |
| Attention | Standard self-attention | **Windowed + full attention** alternating |
| Normalization | LayerNorm | **RMSNorm** |
| Output merging | CLS token or mean pool | **Spatial merge** (2×2 → 1) |
| MLP | fc1/fc2 | **Gated MLP** (gate_proj/up_proj/down_proj + SiLU) |

### Critical: 2D rotary embedding dimension

The vision encoder computes separate rotary frequencies for height and
width positions. The rotary embedding dimension must be `head_dim // 2`
(not `head_dim`):

```python
# CORRECT: each spatial dimension gets head_dim//4 frequencies
self.rotary_pos_emb = Qwen25VLVisionRotaryEmbedding(head_dim // 2)

# WRONG: produces 2× too many frequencies with wrong values
self.rotary_pos_emb = Qwen25VLVisionRotaryEmbedding(head_dim)
```

The frequency table has shape `(num_patches, head_dim//2)`. Each half
(`head_dim//4` values) covers one spatial dimension. The `forward` method
concatenates `cos(h_freqs)` and `cos(w_freqs)` to produce the final
`(num_patches, head_dim)` rotary embeddings.

### Critical: fullatt_block_indexes config

Qwen2.5-VL uses a **hybrid attention pattern**: most blocks use windowed
attention (local windows for efficiency), but certain blocks use full
attention (all patches attend to all patches):

```python
# Must be extracted from HF vision_config
fullatt_block_indexes = [7, 15, 23, 31]  # For 32-block encoder
window_size = 112  # Window size in patches for windowed blocks
```

If `fullatt_block_indexes` is missing, ALL blocks use windowed attention,
causing massive feature divergence (cos ≈ 0.25). The first few blocks may
appear correct since they happen to be windowed blocks.

**Config extraction** — these must be in `_configs.py` VisionConfig:

```python
@dataclasses.dataclass
class VisionConfig:
    ...
    fullatt_block_indexes: list[int] | None = None
    window_size: int | None = None
```

### Window index and attention bias

- **Windowed blocks**: Patches are grouped into windows of `window_size`.
  Each window attends only within itself. The attention bias is block-diagonal.
- **Full attention blocks**: Use `cu_seqlens` (not `cu_window_seqlens`) to
  attend across all patches in each image.
- `window_index` permutes patches into window-ordered layout before the
  transformer blocks, then `reverse_indices = argsort(window_index)` restores
  the original order after.

### Multi-image support

Both vision encoders support multiple images via the ONNX `Scan` op.
Per-image values (position IDs, window indices, cu_seqlens) are computed
in a Scan body and concatenated. See `.github/skills/scan-and-multi-image/SKILL.md`.

### Spatial merge (post-encoder)

After the transformer blocks, a spatial merge layer combines 2×2 patches
into 1 token:
```
(num_patches, hidden_size) → reshape to (num_merged, 4*hidden_size) → MLP → (num_merged, text_hidden_size)
```
The merge reduces token count by 4× and projects to text model dimension.

## Qwen3.5-VL

Qwen3.5-VL uses the same **3-model split** as Qwen3-VL (decoder + vision +
embedding), but swaps the text decoder for the **Qwen3.5 architecture**
which uses hybrid DeltaNet + full attention instead of standard GQA.

### Architecture

The vision encoder is **identical to Qwen3-VL** — it reuses
`Qwen3VLVisionModel` (patch_size=16, hidden=1152, depth=27). Only the
text decoder changes.

| Component | Class | Notes |
|-----------|-------|-------|
| 3-model composite | `Qwen35VL3ModelCausalLMModel` | Splits into decoder + vision + embedding |
| Decoder (standalone) | `Qwen35VLDecoderModel` | Uses `Qwen35TextModel` internally |
| Text model | `Qwen35VLTextModel` | Text-only decoder; strips VL weight prefixes |

### Registration

| `model_type` | Variant | Description |
|--------------|---------|-------------|
| `qwen3_5_vl` | 3-model split | Full VLM with vision encoder |
| `qwen3_5_vl_text` | Text-only | Decoder without vision |

### Task

Reuses `Qwen3VLVisionLanguage3ModelTask` (task name: `qwen35-vl`).

### Config

The HF config is VL-style with a nested `text_config`:

```
config.json          →  model_type: "qwen3_5_vl"
config.text_config   →  model_type: "qwen3_5"   (or "qwen3_5_text")
```

### Token IDs

| Token | ID |
|-------|----|
| `image` | 248056 |
| `video` | 248057 |
| `vision_start` | 248053 |
| `vision_end` | 248054 |

### Interleaved MRoPE

Uses `InterleavedMRope` (not `ChunkedMRope`) with:

- `partial_rotary_factor=0.25`
- `mrope_section=[11, 11, 10]`

### Key insight

The vision pipeline is completely shared with Qwen3-VL — only the text
decoder differs (hybrid DeltaNet + full attention). This means vision
encoder bugs/fixes apply to both models equally.

## Testing multimodal models

### Image token count

Insert `mm_tokens_per_image` image tokens (not 1!) into the input to match
the number of vision features the projector produces:

```python
mm_tokens = config.mm_tokens_per_image or 1
img_tokens = np.full((1, mm_tokens), image_token_id, dtype=np.int64)
input_ids = np.concatenate([input_ids[:, :1], img_tokens, input_ids[:, 1:]], axis=1)
```

### Dummy pixel values

Use random pixel values for testing (we only need numerical parity, not
meaningful images):

```python
rng = np.random.default_rng(42)
pixel_values = rng.standard_normal((1, 3, image_size, image_size)).astype(np.float32)
```

### Tolerances

Use `rtol=1e-2, atol=1e-2` for multimodal tests — the vision pipeline
introduces more floating-point variance than text-only models.

### Decode step

After prefill with image, the decode step is text-only but still needs
`pixel_values` as a graph input (use zeros):

```python
decode_pixel_values = np.zeros_like(pixel_values)
```

## 3-model split for ORT GenAI

For deployment with onnxruntime-genai, multimodal models are split into
3 separate ONNX models:

```
[vision.onnx]    pixel_values, grid_thw → image_features
[embedding.onnx] input_ids, image_features → inputs_embeds
[model.onnx]     inputs_embeds, attention_mask, position_ids, past_kv → logits, present_kv
```

### I/O contract

| Model | Inputs | Outputs |
|-------|--------|---------|
| Vision | `pixel_values: float32`, `grid_thw: int64` | `image_features: float32` |
| Embedding | `input_ids: int64`, `image_features: float32` | `inputs_embeds: float32` |
| Decoder | `inputs_embeds: float32`, `attention_mask: int64`, `position_ids: int64`, `past_key_values.*` | `logits: float32`, `present.*` |

### genai_config.json required fields for VLMs

ORT GenAI needs these fields to compute 3D M-RoPE position_ids for
image tokens. **Without them, image inputs produce wrong output.**

```json
{
  "model": {
    "image_token_id": 151655,
    "video_token_id": 151656,
    "vision_start_token_id": 151652,
    "vision": {
      "filename": "vision.onnx",
      "config_filename": "processor_config.json",
      "spatial_merge_size": 2,
      "tokens_per_second": 2.0,
      "inputs": { "pixel_values": "pixel_values", "image_grid_thw": "image_grid_thw" },
      "outputs": { "image_features": "image_features" },
      "session_options": { "log_id": "onnxruntime-genai", "provider_options": [] }
    },
    "embedding": {
      "filename": "embedding.onnx",
      "inputs": { "input_ids": "input_ids", "image_features": "image_features" },
      "outputs": { "inputs_embeds": "inputs_embeds" },
      "session_options": { "log_id": "onnxruntime-genai", "provider_options": [] }
    }
  }
}
```

**Required fields** (without these, VLM output is wrong):
- `image_token_id`: Token ID for `<|image_pad|>` — needed for 3D M-RoPE
- `vision_start_token_id`: Token ID for `<|vision_start|>` — marks image boundaries
- `spatial_merge_size`: Grid merge factor (2 for Qwen2.5-VL) — used in position computation

**Recommended fields** (from olive reference):
- `video_token_id`: Token ID for video frames (151656)
- `config_filename`: Points to `processor_config.json` for image preprocessing
- `tokens_per_second`: Controls temporal position increment (2.0)
- `session_options`: ORT session configuration for each sub-model

See `.github/skills/ort-genai-config/SKILL.md` for the complete reference
and `.github/skills/debugging-vl-pipeline/SKILL.md` for troubleshooting.

### processor_config.json for image preprocessing

ORT GenAI uses ort-extensions for image preprocessing (not HuggingFace).
The `processor_config.json` must use this format:

```json
{
  "processor": {
    "name": "qwen2_5_image_processor",
    "transforms": [
      {"operation": {"name": "decode_image", "type": "DecodeImage", "attrs": {"color_space": "RGB"}}},
      {"operation": {"name": "convert_to_rgb", "type": "ConvertRGB"}},
      {"operation": {"name": "resize", "type": "Resize", "attrs": {
        "width": 960, "height": 672,
        "smart_resize": 1, "min_pixels": 3136, "max_pixels": 12845056,
        "patch_size": 14, "merge_size": 2
      }}},
      {"operation": {"name": "rescale", "type": "Rescale", "attrs": {"rescale_factor": 0.00392156862745098}}},
      {"operation": {"name": "normalize", "type": "Normalize", "attrs": {"mean": [0.4814, 0.4578, 0.4082], "std": [0.2686, 0.2613, 0.2758]}}},
      {"operation": {"name": "patch_image", "type": "PatchImage", "attrs": {"patch_size": 14, "temporal_patch_size": 2, "merge_size": 2}}}
    ]
  }
}
```

**Important:** The `width`/`height` in the Resize transform are used as
direct target dimensions, unlike HF's smart_resize which computes targets
from original image dimensions. Compute them as
`round(original_dim / (patch_size * merge_size)) * (patch_size * merge_size)`.

### Embedding model padding for text-only input

The embedding model must handle the case where `num_image_tokens=0`
(text-only input). Pad `image_features` with a zero row before Gather
so that the Gather index doesn't go out-of-bounds:

```python
# In embedding model forward:
padded = op.Concat(zero_row, image_features, axis=0)
gathered = op.Gather(padded, indices, axis=0)
# Where mask selects only real features; padding row is never used
result = op.Where(image_mask, gathered, text_embeddings)
```

## Task variants

Different VL architectures need different task wiring:

| Task | `TASK_REGISTRY` key | Models | Key difference |
|------|---------------------|--------|----------------|
| `VisionLanguageTask` | `"vision-language"` | LLaVA, LLaVA-NeXT, PaliGemma, Pixtral, Molmo, InternVL2, Gemma3, GLM4V | Standard split |
| `QwenVLTask` | `"qwen-vl"` | Qwen2.5-VL, Qwen3-VL | MRoPE + packed vision input |
| `HybridQwenVLTask` | `"hybrid-qwen-vl"` | Qwen3.5-VL | MRoPE + DeltaNet hybrid cache |
| `HybridVisionLanguageTask` | `"hybrid-vision-language"` | LFM2-VL | Hybrid (conv+KV) cache in decoder |
| `Qwen3VLVisionLanguageTask` | `"qwen3-vl-vision-language"` | Qwen3-VL single-model variant | Single model (uses `InputMixer`) |
| `MllamaVisionLanguageTask` | `"mllama-vision-language"` | Mllama (Llama 3.2 Vision) | Cross-attention layers in decoder |
| `MultiModalTask` | `"multimodal"` | Phi4-MM | Audio + vision + text (4 models) |
| *(standard + Q-Former)* | `"vision-language"` | BLIP-2 | Q-Former between vision and embedding |

**Default for new models**: `VisionLanguageTask` unless MRoPE, hybrid cache,
cross-attention, or audio are involved.

## Vision encoder summary

| Encoder | Used by | mobius component | Notes |
|---------|---------|-----------------|-------|
| SigLIP (ViT-SO/400M) | LLaVA, PaliGemma, InternVL2, Gemma3 | `VisionModel` | Standard encoder |
| CLIP ViT-L/14 | LLaVA-1.5, Phi-3-Vision | `VisionModel` | Standard encoder |
| SigLIP-2 | Phi-4-MM, Phi-4-SigLIP | `VisionModel` (+ S2 tiling) | Standard encoder |
| InternViT | InternVL2 family | `_InternViT*` in `internvl.py` | Fused QKV, layer scale, CLS |
| Qwen2.5-VL ViT | Qwen2.5-VL | `Qwen25VLVisionModel` | Conv3D, 2D-RoPE |
| Qwen3-VL ViT | Qwen3-VL | `Qwen3VLVisionModel` | Similar to Qwen2.5 |
| RADIO (ViT-H/16 + CPE) | NemotronH VL | Not yet in mobius | See RADIO section |

## `VisionConfig` fields reference

`VisionConfig` is extracted from `hf_config.vision_config` by
`ArchitectureConfig.from_transformers`. Available fields:

| Field | Type | HF key | Default |
|-------|------|--------|---------|
| `hidden_size` | `int` | `hidden_size` | 1152 |
| `image_size` | `int` | `image_size` | 224 |
| `patch_size` | `int` | `patch_size` | 14 |
| `num_hidden_layers` | `int` | `num_hidden_layers` | 27 |
| `num_attention_heads` | `int` | `num_attention_heads` | 16 |
| `intermediate_size` | `int` | `intermediate_size` | 4304 |
| `num_channels` | `int` | `num_channels` | 3 |
| `model_type` | `str` | `model_type` | `""` |

Always set `vision` in build-graph test configs:

```python
from mobius._configs import VisionConfig, VisionLanguageConfig

"my_vlm": VisionLanguageConfig(
    hidden_size=64, num_hidden_layers=2, num_attention_heads=2,
    intermediate_size=128, vocab_size=256,
    vision=VisionConfig(hidden_size=64, patch_size=14, image_size=56,
                        num_hidden_layers=2, num_attention_heads=2,
                        intermediate_size=128),
),
```

## Weight mapping with vlm helpers

### `vlm_decoder_weights` and `vlm_embedding_weights`

`_weight_utils.py` exports two helpers to strip the `"language_model."` prefix
that HuggingFace wraps the text backbone in for VL models:

```python
from mobius._weight_utils import vlm_decoder_weights, vlm_embedding_weights

class MyVLModel(nn.Module):
    def preprocess_weights(self, weights):
        # Fix lm_head weight tying BEFORE stripping "language_model." prefix
        if "language_model.lm_head.weight" not in weights:
            weights["language_model.lm_head.weight"] = weights[
                "language_model.model.embed_tokens.weight"
            ]
        return {
            "decoder": vlm_decoder_weights(weights),
            "vision_encoder": _vision_encoder_weights(weights),
            "embedding": vlm_embedding_weights(weights),
        }
```

### Weight-tying pitfall

Many VL models share `embed_tokens.weight` with `lm_head.weight`
(`tie_word_embeddings=True`). HuggingFace does not save `lm_head.weight`.
You **must** inject it at the **top-level** `preprocess_weights` before
`vlm_decoder_weights` strips the prefix:

```python
# CORRECT — inject at top level, before prefix strip
def preprocess_weights(self, weights):
    if "language_model.lm_head.weight" not in weights:
        weights["language_model.lm_head.weight"] = weights[
            "language_model.model.embed_tokens.weight"
        ]
    return {"decoder": vlm_decoder_weights(weights), ...}
```

```python
# WRONG — inject inside _DecoderModel.preprocess_weights is too late;
# by then the key is "model.embed_tokens.weight" and the decoder
# module cannot see "language_model.*" keys anymore.
```

### Non-standard prefixes

| Model | HF prefix | Helper |
|-------|-----------|--------|
| LLaVA, InternVL2, Gemma3, Mllama, PaliGemma | `language_model.` | `vlm_decoder_weights` |
| BLIP-2 | `language_model.` | `vlm_decoder_weights` |
| Phi-3-Vision | `model.language_model.` | custom strip |

## InternViT encoder (InternVL2)

InternViT differs from standard SigLIP/CLIP in three ways:

1. **Fused QKV**: single `qkv` linear (3*hidden) instead of separate Q/K/V
2. **Layer scale**: learnable scalar vectors `ls1`/`ls2` that multiply
   sub-layer output before residual add
3. **CLS token**: prepended before patch tokens (total = num_patches + 1)

InternVL2 also uses **pixel shuffle downsampling** (`downsample_ratio=0.5`)
before the MLP projector, halving spatial resolution. The projector is
4-layer: `LayerNorm -> Linear -> GELU -> Linear`.

See `src/mobius/models/internvl.py` for `_InternViT*` implementations.

**Key weight name mappings:**

| HF key | mobius path |
|--------|------------|
| `vision_model.embeddings.class_embedding` | `vision_encoder.encoder.cls_token` |
| `vision_model.encoder.layers.N.attn.qkv.weight` | `vision_encoder.encoder.layers.N.attn.qkv.weight` |
| `vision_model.encoder.layers.N.layer_scale1.weight` | `vision_encoder.encoder.layers.N.ls1` |

## RADIO encoder (NemotronH VL — not yet in mobius)

RADIO (nvidia/RADIO-L) is a ViT-H/16 with **conditional position encoding
(CPE)**: a depthwise Conv2d that injects adaptive position information into
each patch feature, enabling dynamic-resolution images without re-scaling.

Architecture: standard ViT-H encoder + CPE modules inserted per-layer.
New component needed: `ConditionalPositionEncoding` (~40 LOC depthwise Conv2d).

## Non-standard tasks

### Mllama / Llama 3.2 Vision (cross-attention)

Mllama uses **cross-attention layers** in the decoder that attend to vision
features directly — unlike standard VL where vision features are only injected
at the embedding step.

Key details:
- `MllamaVisionLanguageTask` creates a decoder that accepts both
  `inputs_embeds` and `cross_attention_states` (vision features)
- Decoder has two KV caches: one for self-attention, one for cross-attention
- Cross-attention layer indices come from `config.cross_attention_layers`

### BLIP-2 (Q-Former)

BLIP-2 inserts a **Q-Former** (Querying Transformer) between the vision
encoder and the LLM. Q-Former uses learned query vectors that attend to
vision features, producing a fixed-length sequence (32 tokens) regardless
of image resolution. Q-Former is part of the `vision_encoder` sub-model.

### Phi4-MM (audio + vision + text)

Uses `MultiModalTask` (4 ONNX models: embedding, speech_encoder,
vision_encoder, decoder). See `tasks/_multi_modal.py` and `models/phi4mm.py`.
For debugging, see the `phi4mm-component-parity` skill.

## `image_token_id` reference

| Model family | `image_token_id` | Source in HF config |
|-------------|-----------------|---------------------|
| LLaVA-1.5 (Vicuna) | 32000 | `image_token_index` |
| LLaVA-OneVision (Qwen2) | 151655 | `image_token_index` |
| Gemma3 | 262144 | `image_token_id` |
| InternVL2 | 151667 | `img_context_token_id` |
| Mllama | 128256 | `image_token_id` |
| BLIP-2 | 50265 | `image_token_id` |
| PaliGemma | 257152 | `image_token_id` |
| Qwen2.5-VL / Qwen3-VL | 151655 | `vision_start_token_id` |
| Phi-3-Vision | 32044 | `image_token_id` |

Always read `image_token_id` from `hf_config` — never hardcode.

## Common pitfalls

### Wrong sub-module attribute names

`VisionLanguageTask` accesses `module.decoder`, `module.vision_encoder`, and
`module.embedding` by name. Using different names silently produces a
`ModelPackage` missing those models.

### lm_head weight missing in decoder

If `tie_word_embeddings=True`, `lm_head.weight` is absent from the HF
checkpoint. Inject it at the top-level `preprocess_weights` **before**
calling `vlm_decoder_weights()`. See the vlm helpers section.

### `forward()` called on top-level module

`VisionLanguageTask` calls sub-modules directly. The top-level `forward()`
should raise `NotImplementedError` to catch accidental calls.

### Missing `vision` in build-graph test config

`VisionLanguageTask._build_vision` uses `config.vision.image_size`. Always
set `vision=VisionConfig(...)` in your test config entry.

### Projector dim mismatch

The projector output dim must equal the text decoder hidden size:
`MLPMultiModalProjector(vision_hidden_size=config.vision.hidden_size,
text_hidden_size=config.hidden_size)`.

## Quick reference: file locations

| File | Purpose |
|------|---------|
| `src/mobius/models/llava.py` | Reference 3-split implementation |
| `src/mobius/models/gemma3.py` | Gemma3 (AvgPool projector, OffsetRMSNorm) |
| `src/mobius/models/internvl.py` | InternVL2 (fused QKV, pixel shuffle, CLS) |
| `src/mobius/models/mllama.py` | Mllama (cross-attention, dual KV cache) |
| `src/mobius/models/blip2.py` | BLIP-2 (Q-Former) |
| `src/mobius/tasks/_vision_language_3model.py` | All task variants |
| `src/mobius/components/_vision.py` | `VisionModel`, SigLIP encoder |
| `src/mobius/components/_multimodal.py` | Projectors, `InputMixer` |
| `src/mobius/_weight_utils.py` | `vlm_decoder_weights`, `vlm_embedding_weights` |
| `src/mobius/_configs.py` | `VisionConfig`, `VisionLanguageConfig`, `MllamaConfig` |
| `src/mobius/_registry.py` | All VL model registrations |
| `tests/_test_configs.py` | VL tiny test config entries |
