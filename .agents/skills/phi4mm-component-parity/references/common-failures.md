# Common Failure Modes — Detailed Reference

## 1. Weight name alignment (missing weights)

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

## 2. Shape mismatches (position embedding 2D vs 3D)

**Symptoms:** `RuntimeError: shape mismatch` during weight loading.

**Root cause:** The ONNX component declares a parameter with a different
number of dimensions than the HuggingFace weight. Example: PatchEmbedding
declares `position_embedding.weight` as `[num_patches, hidden_size]` (2D),
but HF stores `[1, num_patches, hidden_size]` (3D).

**Fix in `preprocess_weights()`:**
```python
# Squeeze the extra batch dimension to match ONNX declaration
if "position_embedding.weight" in key and state_dict[key].dim() == 3:
    state_dict[key] = state_dict[key].squeeze(0)  # [1,N,H] → [N,H]
```

**General rule:** Check whether the preprocess_weights transform goes
in the correct direction (squeeze vs unsqueeze). A common mistake is
writing the transform backwards.

## 3. Dtype mismatches (float64 vs float32)

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

**Detection:** Run the ONNX model and look for type mismatch errors.
The error message includes the node name — trace it back to the source.

## 4. LoRA application mismatch (conditional vs unconditional)

**Symptoms:** Systematic divergence (> 80% logits mismatch) across ALL
test cases, but the model structurally runs correctly.

**Root cause:** Some models apply LoRA adapters conditionally based on
input modality. For example, Phi4MM applies:
- `input_mode=0` (text): no adapters
- `input_mode=1` (vision): vision LoRA only
- `input_mode=2` (speech): speech LoRA only
- `input_mode=3` (combined): both adapters

If the ONNX model unconditionally applies all adapters (both vision and
speech LoRA always active), it diverges from HF when HF uses a different
input mode.

**Quick fix for integration tests:** Run the HF reference with the mode
that matches the ONNX model's behavior (e.g., `input_mode=3` to match
unconditional application of both adapters).

**Proper fix:** Add an `input_mode` input to the decoder model and use
conditional logic to selectively apply adapters.

**Detection:** If text-only inference diverges but the model generates
reasonable (not garbage) output, suspect LoRA mode mismatch. Temporarily
zero out all LoRA weights — if base model matches HF perfectly, the
LoRA application mode is the issue.

## 5. Empty tensor handling (zero-length features)

**Symptoms:** Crash during text-only inference when no image/audio
features are present.

**Root cause:** The embedding model's `InputMixer` uses `GatherElements`
to place features at special token positions. With zero features, the
gather indices are empty but the operation may still execute on the
padded dimension, causing shape errors.

**Fix pattern:** Zero-pad before Gather, then use Where to mask results:
```python
# Pad with one zero row so Gather never accesses out-of-bounds
padded = op.Concat(
    op.ConstantOfShape(op.Constant(value_ints=[1, hidden_size])),
    features,  # may be [0, hidden_size]
    axis=0,
)
# After Gather, mask out the padding positions with Where
result = op.Where(feature_mask, gathered, text_embeddings)
```

## 6. HD transform image format (5D vs 4D)

**Symptoms:** Vision model crashes or produces wrong output with multi-crop
HD images.

**Root cause:** HD-capable vision models expect images in different formats:
- Some expect `[batch, channels, height, width]` (4D, single crop per batch)
- Others expect `[num_images, num_crops, channels, height, width]` (5D)

The HF processor output format must match the ONNX model's input format.
If using the HF processor for test input preparation, verify it produces
the expected format.

**Fix:** Check the HF model's preprocessing code for the expected format,
and ensure the ONNX model's input signature matches. For tests, either:
- Use the HF processor: `processor(images=image, return_tensors="np")`
- Or manually construct the correct format for simple test cases

## 7. Causal mask construction (inputs_embeds vs input_ids)

**Symptoms:** Attention mask has wrong length, causing decoder crash or
wrong output.

**Root cause:** When the decoder receives `inputs_embeds` instead of
`input_ids`, the sequence length must be derived from the embeds tensor
shape, not from input_ids. If the mask is built from input_ids length but
the actual sequence includes fused image/audio tokens, the lengths diverge.

**Fix:** Always derive `seq_len` from `inputs_embeds.shape[1]` when the
model uses inputs_embeds as input.

## 8. ClippableLinear not used in encoder (Gemma4)

**Symptoms:** Encoder output has large numerical divergence (max diff > 1.0)
from HuggingFace, despite all weights loading correctly.

**Root cause:** Some HuggingFace models (e.g. Gemma4) use
`ClippableLinear` — a linear layer with learned finite input/output
activation clipping — for ALL linear layers in their vision and audio
encoders (attention q/k/v/o projections AND MLP gate/up/down projections).
Using plain `Linear` misses the clamping and causes divergence.

**Detection:** Check HF source for `ClippableLinear`:
```bash
grep -n "ClippableLinear" transformers/models/<model>/modeling_<model>.py
```

**Fix:** Use `ClippableLinear` from `mobius.components`:
- For attention: use `ClippableLinear` for q/k/v/o_proj
- For MLP: pass `linear_class=ClippableLinear` parameter

**Impact (Gemma4):**
- Audio: max diff 52.68 → 0.0003
- Vision: max diff 3.92 → 0.00007

## 9. Missing audio/image boundary tokens

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
