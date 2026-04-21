# Common Failure Modes and Fixes

Detailed failure mode reference for debugging existing ORT GenAI multimodal
pipelines. For the high-level debugging methodology, see the parent
`SKILL.md`.

---

## 1. Image not recognized (wrong output for image inputs)

**Symptoms:** Model produces generic or hallucinated descriptions that
don't match the input image. Text-only generation works correctly.

**Root causes (in order of likelihood):**

1. **Missing genai_config fields** — `image_token_id`,
   `vision_start_token_id`, or `spatial_merge_size` not set.
   Without these, position_ids are 1D instead of 3D M-RoPE.

2. **Image resize mismatch** — ORT processor resizes image to different
   dimensions than HF processor, producing different number of vision
   tokens. ORT's `width`/`height` in `processor_config.json` are used
   as direct resize targets, unlike HF's smart_resize which computes
   target from original image dimensions.

3. **Processor config format** — ORT GenAI expects ort-extensions format
   `processor_config.json`, not HuggingFace format. The file must include
   `DecodeImage`, `ConvertRGB`, `Resize`, `Rescale`, `Normalize`, and
   `PatchImage` transforms with correct attributes.

**Fix for resize mismatch:**
```python
def _update_resize_for_image(processor_config_path, image_path):
    """Recompute resize dimensions from actual image like HF does."""
    from PIL import Image
    img = Image.open(image_path)
    w, h = img.size
    factor = 14 * 2  # patch_size * merge_size
    new_w = round(w / factor) * factor
    new_h = round(h / factor) * factor
    # Update width/height in processor_config.json
```

## 2. Numerical divergence in greedy decoding

**Symptoms:** First 1-3 tokens match HF, then output diverges.

**Expected behavior:** This is inherent to ONNX vs PyTorch numerical
differences. ONNX models use different operator implementations that
accumulate small floating-point errors.

**Typical metrics for Qwen2.5-VL 3B:**
- max_diff in logits: 5-10
- mean_diff in logits: 0.5-1.5
- cosine similarity: 0.98-0.99
- First token: matches HF
- Greedy decoding: diverges at token 3-5

**This is NOT a bug** if the metrics above are within range. Both models
produce semantically similar descriptions.

## 3. Vision model output shape mismatch

**Symptoms:** Vision model produces wrong number of patches.

**Debug:** Check `grid_thw` values:
```python
# For Qwen2.5-VL with merge_size=2:
t, h, w = grid_thw[0]
expected_patches = t * (h // 2) * (w // 2)
actual_patches = vision_output.shape[0]
assert expected_patches == actual_patches
```

## 4. Embedding model text-only failure

**Symptoms:** Error when running without images (num_image_tokens=0).

**Fix:** Ensure embedding model pads `image_features` with a zero row
before Gather, then uses a Where mask to select only real features:
```python
# Pad with zero row so Gather with index 0 doesn't fail
padded = op.Concat(
    op.ConstantOfShape(...),  # (1, hidden_size) zeros
    image_features,
    axis=0,
)
```

## 5. Vision encoder internal divergence (cos < 0.5)

**Symptoms:** Vision features have very low cosine similarity (< 0.5)
against HuggingFace, even though patch embedding and weights are correct.

**Debug with block-by-block comparison** (see extraction methods in
`references/extraction-methods.md`). Common root causes:

1. **Wrong rotary embedding dimension** — Qwen2.5-VL vision uses 2D
   position encoding (height + width). The rotary dim must be
   `head_dim // 2`, not `head_dim`. Each half (head_dim // 4 frequencies)
   covers one spatial dimension. With full `head_dim`, you get 2× too
   many frequencies with wrong values. **Result: cos ≈ 0.25.**

2. **Missing `fullatt_block_indexes`** — Qwen2.5-VL alternates between
   windowed attention (local windows of `window_size` patches) and full
   attention (all patches attend to all). Blocks at indexes `[7, 15, 23, 31]`
   use full attention. If `fullatt_block_indexes` is not extracted from
   HF config, all blocks use windowed attention. **Result: blocks 0-6
   are perfect (they're windowed anyway), but block 7+ diverges.**

3. **Wrong attention bias construction** — Full-attention blocks should
   have an all-zeros bias (everything attends to everything). Windowed
   blocks have a block-diagonal bias. Check the bias by inspecting
   sparsity: `(bias == -inf).float().mean()` should be ~0% for full
   attention, ~98% for windowed.

**Config extraction checklist for vision encoders:**
```python
# These fields MUST be extracted from HF vision_config:
fullatt_block_indexes = getattr(vc, "fullatt_block_indexes", None)
window_size = getattr(vc, "window_size", None)
spatial_merge_size = getattr(vc, "spatial_merge_size", 2)
temporal_patch_size = getattr(vc, "temporal_patch_size", 2)
```

## 6. ClippableLinear divergence (Gemma4)

**Symptoms:** Vision or audio encoder output has large max diff (> 1.0)
against HuggingFace, even though weights are loaded correctly.

**Root cause:** Gemma4 uses `Gemma4ClippableLinear` with learned finite
input/output activation clamping for ALL linear layers in its vision and
audio encoders. Using plain `Linear` misses the clamping.

**Detection:** Check if the HuggingFace model uses `ClippableLinear`:
```bash
grep -n "ClippableLinear" transformers/models/<model>/modeling_<model>.py
```

**Fix:** Use `ClippableLinear` (from `mobius.components`) for all
affected linear layers. For vision attention: q/k/v/o_proj. For MLP:
pass `linear_class=ClippableLinear` to the MLP component.

**Impact:**
- Audio: max diff 52.68 → 0.0003 after fix
- Vision: max diff 3.92 → 0.00007 after fix

## 7. Missing audio boundary markers

**Symptoms:** Audio transcription is garbled or completely wrong, even
though the audio encoder output matches HuggingFace.

**Root cause:** HuggingFace wraps audio placeholder tokens with boundary
markers in `input_ids`:
```
<|audio> (256000) + N × <|audio|> (258881) + <audio|> (258883)
```
If boundary markers are missing, the model cannot distinguish audio
regions from text, producing wrong output.

**Fix:** Add boundary markers to `build_input_ids()` when constructing
audio inputs, matching HuggingFace's token wrapping.

## 8. CUDA EP: ORT Gather int32 overflow

**Symptoms:** CUDA EP crashes or produces incorrect results for models
with large embedding tables. CPU EP works correctly.

**Root cause:** ORT CUDA `gather_impl.cu` uses `int32` for element
offset computation: `input_index = idx * cols + col_offset`. For
tensors with > 2^31 elements (e.g. Gemma4 per-layer embedding
[262144, 8960] = 2.35B elements), this overflows.

**ORT bug:** microsoft/onnxruntime#28107

**Workaround:** Split large embeddings into smaller tables via
`nn.ModuleList` so each individual Gather stays under the int32 limit.
Use Slice instead of Gather for column-wise indexing on large tensors.

## 9. CUDA EP: opset 24 kernel registration

**Symptoms:** ORT CUDA EP fails to find kernels for standard ops
(Squeeze, Reshape, etc.) even though they work on CPU.

**Root cause:** ORT ≤1.24.x CUDA/TRT EPs don't register kernels for
opset 24, even though the op semantics are unchanged from opset 23.

**Fix:** Use the `ort_lower_opset_for_ep` feature flag (enabled by
default) which lowers the declared opset import from 24 to 23 for
non-CPU EPs. See `src/mobius/_flags.py` and
`src/mobius/_testing/ort_inference.py`.

## 10. Wrong audio feature extractor

**Symptoms:** Audio model produces completely wrong output. Audio
encoder features don't match HuggingFace at all.

**Root cause:** Using `WhisperFeatureExtractor` instead of the
model-specific feature extractor (e.g. `Gemma4AudioFeatureExtractor`).
Different extractors produce different mel spectrograms.

**Fix:** Always use the correct feature extractor for the model:
```python
from transformers import AutoFeatureExtractor
feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
```
