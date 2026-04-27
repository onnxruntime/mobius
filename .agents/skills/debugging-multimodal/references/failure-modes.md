# Failure Modes — Detailed Reference

Comprehensive failure mode reference for debugging multimodal ONNX pipelines.
Merges failure modes from both VL pipeline debugging and multi-encoder
component parity debugging. For the high-level methodology, see the parent
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
   tokens. ORT's `width`/`height` in `image_processor.json` are used
   as direct resize targets, unlike HF's smart_resize which computes
   target from original image dimensions.

3. **Processor config format** — ORT GenAI expects ort-extensions format
   `image_processor.json`, not HuggingFace format. The file must include
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
    # Update width/height in image_processor.json
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

**Debug with block-by-block comparison** (see `references/extraction-methods.md`).
Common root causes:

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
audio encoders (attention q/k/v/o projections AND MLP gate/up/down
projections). Using plain `Linear` misses the clamping.

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

## 7. Missing audio/image boundary markers

**Symptoms:** Audio transcription or vision description is garbled, but
encoder output matches HuggingFace.

**Root cause:** HuggingFace wraps modality placeholder tokens with
boundary markers:
- Image: `<|image>` (open) + N × `<|image|>` (pad) + `<image|>` (close)
- Audio: `<|audio>` (open) + N × `<|audio|>` (pad) + `<audio|>` (close)

Missing boundary markers prevent the model from correctly identifying
modality regions in the input sequence.

**Fix:** Ensure `build_input_ids` wraps placeholder tokens with the
correct open/close marker token IDs.

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

## 11. Weight name alignment (missing weights)

**Symptoms:** Hundreds or thousands of weights reported as "unmatched" by
`apply_weights`. Model runs but produces garbage output.

**Root causes encountered:**

### a. Module forward() bypass (240 missing weights in Phi4MM)

The most insidious bug. When a component's `forward()` method directly
accesses nested sub-module parameters (e.g., `self.glu.ext_pw_conv_1d.weight`)
instead of calling the sub-module's `forward()` method, onnxscript cannot
resolve the full module path for the parameter. The weight ends up as an
unnamed, non-initializer constant in the graph.

```python
# BAD — weights become unnamed
def forward(self, op, x):
    return op.Conv(x, self.glu.ext_pw_conv_1d.weight,
                      self.glu.ext_pw_conv_1d.bias, ...)

# GOOD — onnxscript resolves full module path
def forward(self, op, x):
    return self.glu(op, x)  # GLU.forward() calls op.Conv internally
```

**Detection:** Check the ONNX graph for Conv/MatMul nodes where weight
inputs have `is_initializer=False` and generic names like "weight"/"bias".

**Fix:** Add `forward()` methods to sub-modules and call them instead of
directly accessing their parameters.

### b. ModuleList subclass causing name doubling (8 weights)

Subclassing `nn.ModuleList` causes the module's own name to appear twice
in the parameter path: `img_projection.img_projection.0.weight` instead
of `img_projection.0.weight`.

```python
# BAD — name doubling
class ProjectionMLP(nn.ModuleList):
    def __init__(self):
        super().__init__()
        self.append(nn.Linear(1152, 3072))
        self.append(nn.Linear(3072, 3072))

# GOOD — use nn.Module with indexed children
class ProjectionMLP(nn.Module):
    def __init__(self):
        super().__init__()
        layers = [nn.Linear(1152, 3072), nn.Linear(3072, 3072)]
        for i, layer in enumerate(layers):
            setattr(self, str(i), layer)
```

### c. setattr with dotted names

Using `setattr(self, "audio_projection.speech", module)` creates a single
attribute with a dot in its name, rather than a nested module. The resulting
ONNX parameter names won't match HuggingFace's `ModuleDict`-style naming.

**Fix:** Use `nn.ModuleDict` or create proper nested attributes.

## 12. Shape mismatches (position embedding 2D vs 3D)

**Symptoms:** `RuntimeError: shape mismatch` during weight loading.

**Root cause:** The ONNX component declares a parameter with a different
number of dimensions than the HuggingFace weight. Example: PatchEmbedding
declares `position_embedding.weight` as `[num_patches, hidden_size]` (2D),
but HF stores `[1, num_patches, hidden_size]` (3D).

**Fix in `preprocess_weights()`:**
```python
if "position_embedding.weight" in key and state_dict[key].dim() == 3:
    state_dict[key] = state_dict[key].squeeze(0)  # [1,N,H] → [N,H]
```

## 13. Dtype mismatches (float64 vs float32)

**Symptoms:** ONNX Runtime error: "type mismatch in Mul/Add node" during
inference.

**Root causes:**

### a. NumPy default float64

`numpy.array(python_float)` defaults to float64. Any constant created
from a Python scalar without explicit dtype will be float64 in the graph.

```python
# BAD — float64 constant
scale = numpy.array(alpha / rank)  # defaults to float64
op.Mul(x, scale)  # Mul(float32, float64) → type error

# GOOD — explicit float32
scale = numpy.array(alpha / rank, dtype=numpy.float32)
op.Mul(x, scale)
```

### b. Python int auto-promotion

When passing a Python `int` to an op that expects a tensor, onnxscript
may auto-promote to float64 (implementation-dependent).

```python
# RISKY — Python int may become float64
op.Mul(int64_tensor, self.max_position_embeddings)

# SAFE — explicit constant
op.Mul(int64_tensor,
       op.Constant(value_int=self.max_position_embeddings))
```

## 14. LoRA application mismatch (conditional vs unconditional)

**Symptoms:** Systematic divergence (> 80% logits mismatch) across ALL
test cases, but the model structurally runs correctly.

**Root cause:** Some models apply LoRA adapters conditionally based on
input modality. For example, Phi4MM applies:
- `input_mode=0` (text): no adapters
- `input_mode=1` (vision): vision LoRA only
- `input_mode=2` (speech): speech LoRA only
- `input_mode=3` (combined): both adapters

**Quick fix for integration tests:** Run the HF reference with the mode
that matches the ONNX model's behavior (e.g., `input_mode=3`).

**Proper fix:** Add an `input_mode` input to the decoder model and use
conditional logic to selectively apply adapters.

**Detection:** If text-only inference diverges but the model generates
reasonable (not garbage) output, suspect LoRA mode mismatch. Temporarily
zero out all LoRA weights — if base model matches HF perfectly, the
LoRA application mode is the issue.

## 15. Empty tensor handling (zero-length features)

**Symptoms:** Crash during text-only inference when no image/audio
features are present.

**Root cause:** The embedding model's `InputMixer` uses `GatherElements`
to place features at special token positions. With zero features, the
gather indices are empty but the operation may still execute on the
padded dimension, causing shape errors.

**Fix pattern:** Zero-pad before Gather, then use Where to mask results:
```python
padded = op.Concat(
    op.ConstantOfShape(op.Constant(value_ints=[1, hidden_size])),
    features,  # may be [0, hidden_size]
    axis=0,
)
result = op.Where(feature_mask, gathered, text_embeddings)
```

## 16. HD transform image format (5D vs 4D)

**Symptoms:** Vision model crashes or produces wrong output with multi-crop
HD images.

**Root cause:** HD-capable vision models expect images in different formats:
- Some expect `[batch, channels, height, width]` (4D, single crop per batch)
- Others expect `[num_images, num_crops, channels, height, width]` (5D)

**Fix:** Check the HF model's preprocessing code for the expected format,
and ensure the ONNX model's input signature matches.

## 17. Causal mask construction (inputs_embeds vs input_ids)

**Symptoms:** Attention mask has wrong length, causing decoder crash or
wrong output.

**Root cause:** When the decoder receives `inputs_embeds` instead of
`input_ids`, the sequence length must be derived from the embeds tensor
shape, not from input_ids.

**Fix:** Always derive `seq_len` from `inputs_embeds.shape[1]` when the
model uses inputs_embeds as input.
