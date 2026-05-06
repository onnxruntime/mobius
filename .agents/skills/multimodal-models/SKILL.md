---
name: multimodal-models
description: >
  Add or modify multimodal (vision + language + audio) models in mobius.
  Use when wiring a VisionModel, projector, InputMixer, or
  VisionLanguageTask; handling image/audio token placeholders; choosing
  a projector variant; or splitting a model into 3-or-4 ONNX sub-models
  for ORT GenAI deployment.
---

# Skill: Multimodal (Vision + Language + Audio) Models

## When to use

Use this skill when adding a model that processes both images and text — such
as Gemma3, Gemma4, LLaVA, LLaVA-NeXT, Phi-3-Vision, PaliGemma, InternVL2,
Pixtral, Idefics2/3, Molmo, Florence2, or Video-LLaVA — or a model that
also processes audio (e.g. Gemma4 with speech/audio inputs).

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
| `PixtralVisionTower` | `components/_pixtral_vision.py` | Pixtral 2D RoPE vision encoder |
| `Gemma3MultiModalProjector` | `components/_multimodal.py` | AvgPool2d → RMSNorm → MatMul |
| `MLPMultiModalProjector` | `components/_multimodal.py` | Linear → GELU → Linear |
| `Mistral3MultiModalProjector` | `components/_pixtral_vision.py` | RMSNorm → spatial merge → Linear → GELU → Linear |
| `LinearMultiModalProjector` | `components/_multimodal.py` | Single Linear |
| `InputMixer` | `components/_multimodal.py` | Scatter vision embeddings at image-token positions |
| `VisionLanguageTask` | `tasks/__init__.py` | ONNX I/O contract with `pixel_values` input |

## Projector variants

Choose the projector that matches the HuggingFace implementation:

| Projector | Architecture | Models |
|-----------|-------------|--------|
| `Gemma3MultiModalProjector` | AvgPool2d → RMSNorm → MatMul | Gemma3 |
| `MLPMultiModalProjector` | Linear → GELU → Linear | LLaVA, LLaVA-NeXT, VipLLaVA, Phi-4-MM, InternVL2, Molmo |
| `Mistral3MultiModalProjector` | RMSNorm → spatial merge → Linear → GELU → Linear | Mistral-3, Pixtral |
| `LinearMultiModalProjector` | Single Linear | PaliGemma, Qwen2-Audio, Idefics2, Florence2 |

> Read `references/projector-variants.md` when you need detailed constructor
> arguments, Qwen-VL vision encoder specifics (Conv3d, 2D rotary, windowed
> attention, spatial merge), or Qwen3.5-VL architecture details.

## InputMixer pattern

`InputMixer` replaces placeholder tokens in the text embedding sequence
with projected vision (or audio) features using `GatherElements` + `Where`:

1. Find special token positions in `input_ids`
2. Scatter vision embeddings at those positions
3. Return fused `hidden_states` for the decoder

**Always invoke child modules through `__call__`** so `onnxscript.nn.Module`
pushes the correct naming context. Direct access like
`self.language_model.model.embed_tokens(op, x)` skips intermediate naming
scopes and produces wrong initializer names.

## VisionLanguageTask I/O contract

For ORT GenAI deployment, multimodal models split into 3 (or 4) ONNX models:

| Model | ModelPackage key | Inputs | Outputs |
|-------|-----------------|--------|---------|
| Vision | `"vision_encoder"` | `pixel_values: float32`, `grid_thw: int64` | `image_features: float32` |
| Embedding | `"embedding"` | `input_ids: int64`, `image_features: float32` | `inputs_embeds: float32` |
| Decoder | `"decoder"` | `inputs_embeds: float32`, `attention_mask: int64`, `position_ids: int64`, `past_key_values.*` | `logits: float32`, `present.*` |
| Audio (optional) | `"audio_encoder"` | `input_features: float32` | `audio_features: float32` |

### Component naming conventions

**ModelPackage keys** (used in code and on-disk directory names):
`"decoder"`, `"vision_encoder"`, `"audio_encoder"`, `"embedding"`.
On-disk layout: `decoder/model.onnx`, `vision_encoder/model.onnx`, etc.

**Module attribute names** (on the model class, declared in `ComponentSpec`):
`decoder`, `vision_encoder`, `audio_encoder`, `embedding`.

**Backward compat:** `_MODEL_ROLE_MAP` in `_builder.py` maps legacy keys
(`"model"` → decoder role, `"vision"` → vision encoder role, `"audio"` →
audio encoder role, `"speech"` → encoder role) for older tasks.

**genai_config.json sections** use `"vision"` and `"speech"` (ORT GenAI
convention), not `vision_encoder`/`audio_encoder`.

The embedding model must handle `num_image_tokens=0` (text-only input) by
zero-padding `image_features` before Gather so indices stay in-bounds.

### Conditional 3-or-4-model task

Some models come in two tiers (e.g. Gemma4): small variants include an
audio encoder (4 models), large variants are vision-only (3 models). A
single task class checks `config.audio is not None` to decide whether to
include the audio encoder. Reference: `Gemma4Task` in
`src/mobius/tasks/_gemma4.py`.

### Audio mask pattern (Gemma4)

Audio encoder takes `input_features_mask: BOOL [B, T]` — a contiguous,
right-padded mask indicating which frames are real vs padding. The encoder
outputs `audio_features_mask: BOOL [B, T//4]` (downsampled through conv
stride), which the runtime uses to strip padding before token replacement.

Vision uses padded patches with `(-1, -1)` sentinel position IDs instead
of an explicit mask.

### Gemma4 embedding split

Gemma4's embedding model only maps `input_ids → inputs_embeds` (no
multimodal fusion). Image/audio token replacement happens at the runtime
level, NOT in the embedding ONNX model. The decoder takes both
`inputs_embeds` and `input_ids` — the latter is needed for per-layer
embeddings when `hidden_size_per_layer_input > 0`.

## Image and audio token handling

### Image tokens

Insert `mm_tokens_per_image` placeholder tokens (not 1!) to match the
number of vision features the projector produces:

```python
mm_tokens = config.mm_tokens_per_image or 1
img_tokens = np.full((1, mm_tokens), image_token_id, dtype=np.int64)
input_ids = np.concatenate([input_ids[:, :1], img_tokens, input_ids[:, 1:]], axis=1)
```

### Audio boundary markers (Gemma4)

HuggingFace wraps audio tokens with boundary markers:
```
<|audio> (256000) + N × <|audio|> (258881) + <audio|> (258883)
```
**Missing boundary markers** cause garbled transcription even when the
audio encoder output is numerically correct.

### Testing tolerances

Use `rtol=1e-2, atol=1e-2` for multimodal tests. After prefill with image,
the decode step still needs `pixel_values` as input (use zeros).

## ClippableLinear (critical for Gemma4)

Gemma4's vision and audio encoders use `ClippableLinear` — a `Linear` with
learned finite input/output activation clamping. Using plain `Linear`
causes max diff 52.68 (audio) / 3.92 (vision) → 0.0003 / 0.00007 after
fix. See `reusable-components` skill for full API reference.

## Weight name mapping overview

Multimodal HF models often prefix text weights differently. Implement
`preprocess_weights()` to strip prefixes and rename keys:

| HF key | Our key |
|--------|---------|
| `language_model.model.layers.0.…` | `layers.0.…` |
| `vision_tower.vision_model.encoder.…` | `vision_tower.encoder.…` |

> Read `references/weight-mappings.md` when you need full weight mapping
> tables, shape mismatch fixes, ClippableLinear weight conventions, or
> per-layer embedding splitting details.

> Read `references/vision-encoder-details.md` when you need step-by-step
> instructions for adding a new multimodal model, vision config extraction
> code, or the full model class template with InputMixer wiring.

> Read `references/scan-pattern.md` when building multi-image support using
> the ONNX Scan op, especially for variable-length per-image outputs that
> need the Scan + Padding + Compaction pattern.

## genai_config.json required fields for VLMs

**Required fields** (without these, VLM output is wrong):
- `image_token_id`: Token ID for `<|image_pad|>` — needed for 3D M-RoPE
- `vision_start_token_id`: Token ID for `<|vision_start|>` — marks boundaries
- `spatial_merge_size`: Grid merge factor (2 for Qwen2.5-VL)

See `.agents/skills/ort-genai-config/SKILL.md` for the complete reference
and `.agents/skills/debugging-multimodal/SKILL.md` for troubleshooting.

### image_processor.json for image preprocessing

ORT GenAI uses ort-extensions for image preprocessing (not HuggingFace).
All VLMs use `image_processor.json`:

| Model family | Filename | Notes |
|-------------|----------|-------|
| All VLMs | `image_processor.json` | ort-extensions transforms pipeline |

Audio models use `audio_processor.json` for audio preprocessing.

The vision processor must use `DecodeImage → ConvertRGB → Resize → Rescale
→ Normalize → PatchImage` transforms. The `width`/`height` in Resize are
direct target dimensions; compute them as
`round(original_dim / (patch_size * merge_size)) * (patch_size * merge_size)`.

**Gemma4 vision:** No mean/std normalization — just rescale to [0,1].

## Gemma4: per-layer embeddings (CUDA ORT workaround)

Gemma4 uses `embed_tokens_per_layer` with shape `[V, L*D]`. For large
models this overflows ORT's CUDA Gather kernel. **Workaround:** Split into
L separate `Embedding([V, D])` tables via `nn.ModuleList`, and use `Slice`
instead of `Gather` for per-layer projection indexing.

## Vision/audio encoder f32 input casting

> **This applies to ALL multimodal models** — Gemma3, Gemma4, LLaVA,
> Phi-3-Vision, Qwen-VL, Whisper, and any future vision or audio model.
> It is not architecture-specific.

ORT GenAI's image and audio processors always output **float32** tensors,
regardless of the model's compute dtype. This means vision and audio
encoder ONNX graphs must accept f32 inputs even when the model is built
in f16 or bf16.

### How it works

The encoder graph adds a `Cast(f32 → model_dtype)` at its entry point:

```
Input (f32 from GenAI processor)
    ↓
Cast(to=FLOAT16)    ← inserted automatically
    ↓
Vision/Audio encoder (weights in f16/bf16)
    ↓
Output (model_dtype)
```

This is handled automatically by mobius when building with
`--runtime ort-genai`. The encoder weights still use the requested
dtype (f16/bf16) for memory efficiency — only the graph inputs are f32.

### Why this is needed

- **GenAI image_processor:** Runs pixel normalization in f32 (resize,
  normalize, tile). Outputs f32 tensors.
- **GenAI audio_processor:** Runs mel spectrogram computation in f32.
  Outputs f32 features.
- **Model expects model_dtype:** The encoder's internal ops (attention,
  conv, linear) are all in f16/bf16.

Without the Cast-at-input, ORT throws a type mismatch error:
```
Type Error: Type parameter (T) bound to different types
(tensor(float) and tensor(float16))
```

### For model authors

If you're adding a new multimodal model, you don't need to handle this
manually — the task layer inserts the Cast automatically when
`--runtime ort-genai` is used. If you're building without `--runtime`,
the graph inputs match the model dtype directly.

## Cross-references

- **Multimodal debugging:** `.agents/skills/debugging-multimodal/SKILL.md`
- **ORT GenAI config:** `.agents/skills/ort-genai-config/SKILL.md`
- **Weight name alignment:** `.agents/skills/weight-name-alignment/SKILL.md`
- **Reusable components (ClippableLinear):** `.agents/skills/reusable-components/SKILL.md`
- **Profiling:** `.agents/skills/profiling-onnx-models/SKILL.md`
