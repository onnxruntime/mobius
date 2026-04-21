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

| Model | Inputs | Outputs |
|-------|--------|---------|
| Vision | `pixel_values: float32`, `grid_thw: int64` | `image_features: float32` |
| Embedding | `input_ids: int64`, `image_features: float32` | `inputs_embeds: float32` |
| Decoder | `inputs_embeds: float32`, `attention_mask: int64`, `position_ids: int64`, `past_key_values.*` | `logits: float32`, `present.*` |
| Speech (optional) | `input_features: float32` | `audio_features: float32` |

The embedding model must handle `num_image_tokens=0` (text-only input) by
zero-padding `image_features` before Gather so indices stay in-bounds.

### Conditional 3-or-4-model task

Some models come in two tiers (e.g. Gemma4): small variants include a
speech encoder (4 models), large variants are vision-only (3 models). A
single task class checks `config.audio is not None` to decide whether to
include the speech encoder. Reference: `Gemma4Task` in
`src/mobius/tasks/_gemma4.py`.

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

## genai_config.json required fields for VLMs

**Required fields** (without these, VLM output is wrong):
- `image_token_id`: Token ID for `<|image_pad|>` — needed for 3D M-RoPE
- `vision_start_token_id`: Token ID for `<|vision_start|>` — marks boundaries
- `spatial_merge_size`: Grid merge factor (2 for Qwen2.5-VL)

See `.agents/skills/ort-genai-config/SKILL.md` for the complete reference
and `.agents/skills/debugging-vl-pipeline/SKILL.md` for troubleshooting.

### processor_config.json for image preprocessing

ORT GenAI uses ort-extensions for image preprocessing (not HuggingFace).
The `processor_config.json` must use `qwen2_5_image_processor` format with
DecodeImage → ConvertRGB → Resize → Rescale → Normalize → PatchImage
transforms. The `width`/`height` in Resize are direct target dimensions;
compute them as
`round(original_dim / (patch_size * merge_size)) * (patch_size * merge_size)`.

## Gemma4: per-layer embeddings (CUDA ORT workaround)

Gemma4 uses `embed_tokens_per_layer` with shape `[V, L*D]`. For large
models this overflows ORT's CUDA Gather kernel. **Workaround:** Split into
L separate `Embedding([V, D])` tables via `nn.ModuleList`, and use `Slice`
instead of `Gather` for per-layer projection indexing.

## Cross-references

- **VL debugging:** `.agents/skills/debugging-vl-pipeline/SKILL.md`
- **ORT GenAI config:** `.agents/skills/ort-genai-config/SKILL.md`
- **Weight name alignment:** `.agents/skills/weight-name-alignment/SKILL.md`
- **Multi-image Scan pattern:** `.agents/skills/scan-and-multi-image/SKILL.md`
- **Component parity debugging:** `.agents/skills/phi4mm-component-parity/SKILL.md`
- **Reusable components (ClippableLinear):** `.agents/skills/reusable-components/SKILL.md`
