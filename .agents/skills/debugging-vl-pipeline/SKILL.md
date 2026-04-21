---
name: debugging-vl-pipeline
description: >
  Use this skill when debugging wrong, garbled, or divergent output from an
  existing ORT GenAI multimodal pipeline (vision-language or vision+audio).
  Covers the 3-stage pipeline isolation methodology, quick diagnostic flow,
  3D M-RoPE position ID issues, CUDA EP gotchas, and numerical tolerance
  expectations. For building or adding a new multi-encoder model, use the
  phi4mm-component-parity skill instead.
---

# Skill: Debugging VL Pipeline Issues

> **Scope boundary:** This skill is for debugging runtime/inference issues
> in an **existing** ORT GenAI multimodal pipeline. If you are **building
> or adding** a new multi-encoder model and need component-by-component
> parity verification, use the `phi4mm-component-parity` skill instead.

## When to use

Use this skill when:

- ORT GenAI produces wrong or irrelevant output for image inputs
- ONNX model logits diverge significantly from HuggingFace
- The model generates text-only descriptions ignoring the image
- Image features appear correct but decoder output is wrong
- Audio transcription is garbled or wrong despite correct encoder output
- CUDA EP crashes or produces different results than CPU

## Quick diagnostic flow

1. **Text-only works?** If yes → problem is in vision/audio path or
   position IDs. If no → decoder or weight loading issue.
2. **Vision cos_sim > 0.99?** If no → vision encoder issue (see Stage 1).
3. **Embedding text positions match?** If no → token replacement bug.
4. **3D M-RoPE fields set?** Missing `image_token_id`,
   `vision_start_token_id`, or `spatial_merge_size` causes 1D fallback.
5. **CUDA-only failure?** See CUDA EP section below.

## Debugging methodology: isolate each stage

VL models have 3 stages. Debug by isolating and validating each stage
independently, comparing against HuggingFace at every boundary.

```
pixel_values ──► [1. Vision] ──► image_features
                                       │
input_ids    ──► [2. Embedding] ◄──────┘
                       │
                       ▼
              inputs_embeds + position_ids + attention_mask
                       │
                       ▼
                 [3. Decoder] ──► logits
```

### Stage 1: Vision model

**What to check:**
- Output shape: `(num_patches, hidden_size)`
- Expected patches = `t * (h / merge) * (w / merge)` from `grid_thw`
- Compare features against HF vision encoder output

```python
# HF reference
with torch.no_grad():
    hf_vision_out = hf_model.model.visual(
        pixel_values, grid_thw=grid_thw
    )
# ONNX
session = OnnxModelSession(pkg["vision"])
onnx_out = session.run({"pixel_values": pv, "grid_thw": grid_thw})

# Compare
cos_sim = np.dot(hf_flat, onnx_flat) / (norm_hf * norm_onnx)
print(f"Vision cos_sim: {cos_sim:.6f}")  # Should be > 0.99
```

**Common issues:**
- Wrong pixel value normalization (mean/std mismatch)
- `grid_thw` shape or values don't match HF processor output
- Missing `temporal_patch_size` in patch embedding

### Stage 2: Embedding model

**What to check:**
- Image features injected at correct token positions
- Non-image positions have correct text embeddings
- Output shape: `(1, seq_len, hidden_size)`

```python
# Verify image token positions
image_mask = (input_ids[0] == image_token_id)  # 151655
num_image_positions = image_mask.sum()
assert num_image_positions == image_features.shape[0]

# Compare embeddings at text positions (should match HF exactly)
text_mask = ~image_mask
cos_sim_text = cosine_similarity(
    onnx_embeds[0, text_mask], hf_embeds[0, text_mask]
)
print(f"Text embedding cos_sim: {cos_sim_text:.6f}")  # Should be 1.0
```

**Common issues:**
- Image token count mismatch between processor and vision model
- Missing zero-padding row in embedding model (for text-only inputs)
- Wrong `image_token_id` used for Gather/Where mask

### Stage 3: Decoder

**What to check:**
- Logits shape matches HF: `(1, seq_len, vocab_size)`
- First token prediction matches HF (argmax of last position)
- Cosine similarity of logit vectors

```python
# With HF-computed position_ids (ground truth):
onnx_logits = decoder_session.run(feeds)["logits"]
hf_logits = hf_model(**hf_inputs).logits.numpy()

max_diff = np.abs(onnx_logits - hf_logits).max()
cos_sim = cosine_similarity(onnx_logits[0, -1], hf_logits[0, -1])
print(f"max_diff={max_diff:.2f}, cos_sim={cos_sim:.4f}")
# Typical: max_diff=5-10, cos_sim>0.98
```

**Common issues:**
- Wrong position_ids (see "3D M-RoPE" section below)
- Missing KV cache initialization
- Wrong attention_mask length

## Critical: 3D M-RoPE position IDs

Qwen2-VL / Qwen2.5-VL / Qwen3-VL use **3D Multimodal RoPE** where
`position_ids` has shape `(3, batch, seq_len)`:

```
position_ids[0] = temporal positions
position_ids[1] = height positions
position_ids[2] = width positions
```

**Text tokens:** all 3 dimensions have the same sequential value.

**Image tokens:** temporal is constant, height/width vary over the
image grid `(h/merge, w/merge)`:
```
temporal: [offset, offset, offset, ..., offset]
height:   [offset, offset+1, offset+1, ..., offset+h/merge-1]
width:    [offset, offset+1, offset, offset+1, ..., offset+w/merge-1]
```

**Text after image:** all 3 dimensions resume from
`max(temporal, height, width) + 1`.

### ORT GenAI config requirements

For ORT GenAI to compute 3D M-RoPE automatically, the following
`genai_config.json` fields are **required**:

| Field | Level | Purpose |
|-------|-------|---------|
| `model.image_token_id` | model | Token ID for `<\|image_pad\|>` (e.g. 151655) |
| `model.vision_start_token_id` | model | Token ID for `<\|vision_start\|>` (e.g. 151652) |
| `model.vision.spatial_merge_size` | vision | Grid merge factor (typically 2) |

**Without these fields**, ORT GenAI falls back to standard 1D positions,
which produces completely wrong output for image inputs (the model may
describe a "snowy landscape" instead of the actual image content).

## CUDA EP gotchas

### ORT Gather int32 overflow

CUDA EP crashes or produces incorrect results for models with large
embedding tables (> 2^31 elements). CPU EP works correctly.
**Workaround:** Split large embeddings via `nn.ModuleList`.
ORT bug: microsoft/onnxruntime#28107

### Opset 24 kernel registration

ORT ≤1.24.x CUDA/TRT EPs don't register kernels for opset 24.
**Fix:** Use the `ort_lower_opset_for_ep` feature flag (enabled by
default). See `src/mobius/_flags.py`.

## Integration test patterns

### Full VL forward test
```python
# Build model → process image with HF processor → run ONNX → compare logits
assert_logits_close(onnx_logits, hf_logits, rtol=2e-2, atol=2e-1)
```

### 3-model pipeline test
```python
# Vision → Embedding → Decoder, each stage uses OnnxModelSession
# Compare final decoder logits against HF single-model forward
```

### ORT GenAI end-to-end test
```python
# Build → save flat → write genai_config → load with ort_genai → generate
# Verify output length > input (basic sanity)
```

## Reference files

- **Detailed failure modes (10):** `references/failure-modes.md`
- **Intermediate value extraction methods:** `references/extraction-methods.md`
- **Integration tests:** `tests/integration_test.py`
  (`TestVLFullForward`, `TestQwen25VL3Model`)
- **ORT GenAI tests:** `tests/ort_genai_test.py`
  (`TestOrtGenaiQwen25VL.test_multimodal_image_generation`)
- **Example scripts:** `examples/qwen25_vl_ort_genai.py`,
  `examples/qwen3_vl_ort_genai.py`, `examples/gemma4_multimodal.py`
- **genai_config reference:** `.agents/skills/ort-genai-config/SKILL.md`
- **Component parity skill:** `.agents/skills/phi4mm-component-parity/SKILL.md`
- **ORT GenAI position_ids code (external):**
  `onnxruntime-genai/src/models/position_inputs.cpp:617-814`
- **Feature flags:** `src/mobius/_flags.py`
  (`ort_lower_opset_for_ep`, `ort_cuda_grouped_rmsnorm_workaround`)
