---
name: debugging-multimodal
description: >
  Debug wrong, garbled, or divergent output from multimodal ONNX models
  (vision-language, vision+audio, multi-encoder). Use when ORT GenAI
  multimodal output doesn't match HuggingFace, when building a new
  multimodal model and verifying component-by-component parity (vision
  encoder, speech encoder, embedding/projector, text decoder), when
  integration tests fail with large numerical differences, or when CUDA
  EP produces different results than CPU. Covers the 3-stage VL pipeline
  and 4-model multi-encoder pipeline isolation, 3D M-RoPE position IDs,
  CUDA EP gotchas, and systematic stage-by-stage comparison methodology.
---

# Skill: Debugging Multimodal Pipeline Issues

## When to use

Use this skill when:

- ORT GenAI produces wrong or irrelevant output for image/audio inputs
- ONNX model logits diverge significantly from HuggingFace
- The model generates text-only descriptions ignoring the image
- Audio transcription is garbled or wrong despite correct encoder output
- You're adding a new multimodal model and need to verify each component
- Integration tests fail with large numerical differences (not just tolerance)
- Weights appear to load but the model produces wrong outputs
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

### 3-stage VL pipeline (vision-language only)

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

### 4-model multi-encoder pipeline (e.g. Phi4MM, Gemma4)

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

### Stage 1: Vision encoder

**What to check:**
- Output shape: `(num_patches, hidden_size)` or
  `(num_image_tokens, text_hidden_size)` after projection
- Expected patches = `t * (h / merge) * (w / merge)` from `grid_thw`
- Compare features against HF vision encoder output

```python
# HF reference
with torch.no_grad():
    hf_vision_out = hf_model.model.visual(
        pixel_values, grid_thw=grid_thw
    )
# ONNX — use standardized ModelPackage key "vision_encoder"
session = OnnxModelSession(pkg["vision_encoder"])
onnx_out = session.run({"pixel_values": pv, "grid_thw": grid_thw})

# Compare
cos_sim = np.dot(hf_flat, onnx_flat) / (norm_hf * norm_onnx)
print(f"Vision cos_sim: {cos_sim:.6f}")  # Should be > 0.99
```

**Common issues:**
- Wrong pixel value normalization (mean/std mismatch)
- `grid_thw` shape or values don't match HF processor output
- Missing `temporal_patch_size` in patch embedding
- Wrong rotary embedding dimension (must be `head_dim // 2` for 2D)
- Missing `fullatt_block_indexes` (windowed vs full attention)

### Stage 2: Speech/audio encoder (multi-encoder models)

**What to check:**
- Compression rate (typically 8× time reduction)
- Projection branch selection
- Conv subsampling output length
- Audio mask: `input_features_mask: BOOL [B, T]` must be contiguous
  (right-padded). Output `audio_features_mask: BOOL [B, T//4]` after
  conv stride downsampling.

Compare Conformer output against HF. Target: cos_sim > 0.99.

**Common issues:**
- Missing `input_features_mask` input (causes wrong padding handling)
- `audio_features_mask` shape mismatch (T//4 for stride-4 conv)

### Stage 3: Embedding/fusion

**What to check:**
- Image/audio features injected at correct token positions
- Non-image positions have correct text embeddings
- Output shape: `(1, seq_len, hidden_size)`

```python
# Verify image token positions
image_mask = (input_ids[0] == image_token_id)
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
- InputMixer must handle zero-length tensors

### Stage 4: Decoder

**What to check:**
- Logits shape matches HF: `(1, seq_len, vocab_size)`
- First token prediction matches HF (argmax of last position)
- Cosine similarity of logit vectors

```python
onnx_logits = decoder_session.run(feeds)["logits"]
hf_logits = hf_model(**hf_inputs).logits.numpy()

max_diff = np.abs(onnx_logits - hf_logits).max()
cos_sim = cosine_similarity(onnx_logits[0, -1], hf_logits[0, -1])
print(f"max_diff={max_diff:.2f}, cos_sim={cos_sim:.4f}")
# Typical: max_diff=5-10, cos_sim>0.98
```

## Quick-start 4-step process (new models)

1. **Text-only baseline:** Build ONNX → run embedding + decoder (skip
   encoders) → compare against HF `embed_tokens` and full forward logits.
   If this diverges, fix weight loading / decoder before touching encoders.

2. **Add vision:** Run vision encoder → feed features to embedding →
   compare. If newly divergent, isolate vision encoder output vs HF.

3. **Add audio:** Same as above for speech encoder.

4. **Combined:** All modalities together. If divergent only in combined
   mode, suspect LoRA mode mismatch or embedding fusion ordering.

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
image grid `(h/merge, w/merge)`.

**Text after image:** all 3 dimensions resume from
`max(temporal, height, width) + 1`.

### ORT GenAI config requirements

For ORT GenAI to compute 3D M-RoPE automatically, these
`genai_config.json` fields are **required**:

| Field | Level | Purpose |
|-------|-------|---------|
| `model.image_token_id` | model | Token ID for `<\|image_pad\|>` (e.g. 151655) |
| `model.vision_start_token_id` | model | Token ID for `<\|vision_start\|>` (e.g. 151652) |
| `model.vision.spatial_merge_size` | vision | Grid merge factor (typically 2) |

**Without these fields**, ORT GenAI falls back to standard 1D positions,
which produces completely wrong output for image inputs.

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

Audio/image boundary markers (`<|audio>`, `<audio|>`, `<|image>`,
`<image|>`) are required for correct modality region identification.
Missing markers → garbled output even with correct encoder output.

## CUDA EP gotchas

### ORT Gather int32 overflow

CUDA EP crashes or produces incorrect results for models with large
embedding tables (> 2^31 elements). CPU EP works correctly.
**Workaround:** Split large embeddings via `nn.ModuleList`.
ORT bug: microsoft/onnxruntime#28107

### Opset 24 kernel registration

ORT ≤1.24.x CUDA/TRT EPs didn't register kernels for opset 24.
This has been fixed in newer ORT versions.  The `ort_lower_opset_for_ep`
feature flag is available as a workaround (disabled by default, opt-in
via `MOBIUS_ORT_LOWER_OPSET_FOR_EP=1`).  See `src/mobius/_flags.py`.

### Encoder input dtype alignment

Real vision/audio processors emit float32. Encoder graph inputs should
therefore be float32 even for fp16/bf16 exports, with one Cast to
`config.dtype` at graph entry. Verify this with an actual processor batch
and a graph I/O dtype assertion; synthetic feeds can hide the mismatch.
Some older task builders still use `config.dtype` and are not safe templates.

### GQA for KV-shared layers

Gemma4 KV-shared layers now emit `GroupQueryAttention` with empty K/V
inputs (`kv_sequence_length=0`) and borrowed source-layer KV wired via
`past_key`/`past_value`, avoiding extra Transpose/Reshape cache ops.

Runtime support depends on ORT having KV-shared GQA support (tracked in
microsoft/onnxruntime#28242; still upstreaming as of this writing). On
ORT builds without that support, this path can fail at runtime.

### NaN for large head_dim (> 256)

CUDA EP may produce NaN values when `head_dim > 256` in attention ops.
Tracked in microsoft/onnxruntime#28195 and #28196. CPU EP is unaffected.

### `nonpad_kv_seqlen` requires static cache

The `nonpad_kv_seqlen` optimization only works with static KV cache.
ORT asserts that no `past_key`/`past_value` inputs exist — if they do,
the model will fail to run on CUDA EP.

### EP-aware building

`--ep` flag drives both graph construction and optimization (e.g. GQA
fusion with `do_rotary=1`). `--optimize` is for post-hoc rewrite rules
only (separate from EP). After EP-aware optimization, unused graph inputs
(e.g. `position_ids` absorbed by GQA) are removed by
`RemoveDeadGraphInputsPass`.

## Tolerance guidelines

| Precision | atol | rtol | Notes |
|-----------|------|------|-------|
| float32 | 1e-4 | 2e-2 | Standard for single-forward-pass |
| float32 (deep model, 32+ layers) | 1e-3 | 5e-2 | Error compounds over layers |
| float16 / bfloat16 | 0.01 | 0.05 | Wider tolerance for mixed precision |
| Cosine similarity (last token) | > 0.98 | — | Primary correctness metric |
| Argmax match (first prediction) | exact | — | Should always match |

## Integration test patterns

### Full VL forward test
```python
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

> Read `references/failure-modes.md` when you need detailed code examples
> for each failure mode, including weight name alignment, shape mismatches,
> dtype mismatches, ClippableLinear, HD transform format, and CUDA EP issues.

> Read `references/extraction-methods.md` when you need to extract
> intermediate ONNX values for block-by-block comparison against HuggingFace.

> Read `references/debugging-cookbook.md` when you need step-by-step
> debugging procedures with code for each phase, intermediate value
> extraction, integration test patterns, tolerance guidelines, weight
> loading verification, and HD multi-crop verification.

- **Integration tests:** `tests/integration/vlm_test.py`
  (`TestVLTextForward`, `TestQwen25VL3Model`),
  `tests/integration/multimodal_pipeline_test.py`,
  `tests/phi4mm_integration_test.py`
- **ORT GenAI tests:** `tests/ort_genai_test.py`
- **Example scripts:** `examples/qwen25_vl_ort_genai.py`,
  `examples/qwen3_vl_ort_genai.py`, `examples/gemma4_multimodal.py`
- **genai_config reference:** `.agents/skills/ort-genai-config/SKILL.md`
- **ORT GenAI position_ids code (external):**
  `onnxruntime-genai/src/models/position_inputs.cpp:617-814`
- **Feature flags:** `src/mobius/_flags.py`
- **Model implementations:** `src/mobius/models/phi.py`,
  `src/mobius/models/gemma4.py`
- **Audio components:** `src/mobius/components/_audio.py`,
  `src/mobius/components/_gemma4_audio.py`
- **Vision components:** `src/mobius/components/_vision.py`
- **ClippableLinear:** `src/mobius/components/_gemma4_audio.py`
- **LoRA component:** `src/mobius/components/_lora.py`
- **Weight loading:** `src/mobius/_weight_loading.py`
- **Weight name alignment skill:**
  `.agents/skills/weight-name-alignment/SKILL.md`
