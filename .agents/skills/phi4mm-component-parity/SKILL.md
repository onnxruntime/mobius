---
name: phi4mm-component-parity
description: >
  Use this skill when building or adding a new multi-encoder multimodal
  model (vision + language + audio) and verifying component-by-component
  parity with HuggingFace. Covers pipeline isolation methodology, common
  failure modes from real debugging experience (Phi4MM, Gemma4), the
  step-by-step process for isolating which component (vision encoder,
  speech encoder, embedding, or decoder) causes numerical divergence,
  and integration test patterns. For debugging an existing deployed ORT
  GenAI pipeline, use the debugging-vl-pipeline skill instead.
---

# Skill: Multimodal Component Parity Debugging

## When to use

Use this skill when:

- A multimodal ONNX model's logits diverge systematically from HuggingFace
- You're adding a new multimodal model and need to verify each component
- Integration tests fail with large numerical differences (not just tolerance)
- Weights appear to load but the model produces wrong outputs
- You need to isolate which component (vision, speech, embedding, decoder)
  is causing divergence

For vision-language-only models (no speech), see also the
`debugging-vl-pipeline` skill which covers VLM-specific issues like 3D M-RoPE.

## Pipeline isolation methodology

Multimodal models with N encoders have N+2 stages (encoders + embedding +
decoder). Debug by comparing each stage independently against HuggingFace
at every boundary.

### 4-model multimodal pipeline (e.g., Phi4MM)

```
pixel_values ──► [1. Vision Encoder] ──► image_features ──┐
                                                           │
audio_embeds ──► [2. Speech Encoder] ──► speech_features ──┤
                                                           │
input_ids    ──► [3. Embedding/Fusion] ◄───────────────────┘
                        │
                        ▼
                   inputs_embeds
                        │
                        ▼
                 [4. Decoder + LoRA] ──► logits
```

**Golden rule:** Start from the simplest case (text-only, no encoders),
verify it matches HF, then add one modality at a time.

### Stage-by-stage comparison

**Stage 1 — Vision encoder:** Compare SigLIP/ViT output. Check output
shape `(num_image_tokens, text_hidden_size)`, projection MLP correctness,
and position embeddings (2D vs 3D shape). Target: cos_sim > 0.99.

**Stage 2 — Speech encoder:** Compare Conformer output. Check compression
rate (typically 8× time reduction), projection branch selection, and conv
subsampling output length.

**Stage 3 — Embedding/fusion:** Compare token embeddings. Text-only should
match HF `embed_tokens` exactly (< 1e-5). With features, verify token
replacement at correct positions. InputMixer must handle zero-length tensors.

**Stage 4 — Decoder:** Compare logits. Acceptable float32 metrics:
max_diff 5-10, mean_diff 0.5-1.5, cos_sim > 0.98, argmax match exact.

## Quick-start 4-step process

1. **Text-only baseline:** Build ONNX → run embedding + decoder (skip
   encoders) → compare against HF `embed_tokens` and full forward logits.
   If this diverges, fix weight loading / decoder before touching encoders.

2. **Add vision:** Run vision encoder → feed features to embedding →
   compare. If newly divergent, isolate vision encoder output vs HF.

3. **Add audio:** Same as above for speech encoder.

4. **Combined:** All modalities together. If divergent only in combined
   mode, suspect LoRA mode mismatch or embedding fusion ordering.

## Gotchas

### Module forward() bypass

The #1 source of missing weights. Directly accessing nested sub-module
parameters (`self.glu.ext_pw_conv_1d.weight`) instead of calling
`self.glu(op, x)` makes onnxscript unable to resolve the full module path.
**Detection:** Conv/MatMul nodes where weight inputs have
`is_initializer=False` and generic names.

### LoRA mode mismatch

Some models (Phi4MM) apply LoRA conditionally per modality. If ONNX applies
all adapters unconditionally, run HF reference with `input_mode=3` to match.
**Detection:** Text-only inference diverges but output is reasonable (not
garbage).

### ClippableLinear omission (Gemma4)

Using plain `Linear` instead of `ClippableLinear` in Gemma4 vision/audio
encoders causes max_diff 52.68 (audio) / 3.92 (vision). Check HF source
for `ClippableLinear` usage.

### Empty tensor handling

Text-only inference crashes when no image/audio features are present.
**Fix:** Zero-pad `image_features` before Gather, then mask with Where.

### Missing boundary tokens

Audio/image boundary markers (`<|audio>`, `<audio|>`) are required for
correct modality region identification. Missing markers → garbled output
even with correct encoder output.

> Read `references/common-failures.md` when you need detailed code examples
> for each failure mode, including weight name alignment (ModuleList subclass
> name doubling, setattr with dotted names), shape mismatches, dtype
> mismatches (float64 vs float32), and HD transform format issues.

> Read `references/debugging-cookbook.md` when you need step-by-step
> debugging procedures with code for each phase, intermediate value
> extraction methods, integration test patterns (text-only, audio, vision),
> tolerance guidelines, weight loading verification, and HD multi-crop
> verification.

## Reference files

- **Integration tests:** `tests/phi4mm_integration_test.py`,
  `tests/integration_test.py`
- **VL debugging skill:** `.agents/skills/debugging-vl-pipeline/SKILL.md`
- **Model implementation:** `src/mobius/models/phi.py`,
  `src/mobius/models/gemma4.py`
- **Audio components:** `src/mobius/components/_audio.py`,
  `src/mobius/components/_gemma4_audio.py`
- **Vision components:** `src/mobius/components/_vision.py`
- **ClippableLinear:** `src/mobius/components/_gemma4_audio.py`
- **LoRA component:** `src/mobius/components/_lora.py`
- **Weight loading:** `src/mobius/_weight_loading.py`
- **Feature flags:** `src/mobius/_flags.py`
- **ORT GenAI config skill:** `.agents/skills/ort-genai-config/SKILL.md`
- **Weight name alignment skill:**
  `.agents/skills/weight-name-alignment/SKILL.md`
- **Gemma4 example:** `examples/gemma4_multimodal.py`
