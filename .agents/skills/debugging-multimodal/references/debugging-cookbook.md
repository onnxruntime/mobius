# Debugging Cookbook — Step-by-Step Procedures

Detailed step-by-step debugging procedures for multimodal model parity
issues. For the high-level methodology, see the parent `SKILL.md`.

---

## Step-by-step debugging process

### Phase 1: Text-only baseline

1. **Build ONNX model** with tiny config (for fast iteration) or full
   weights (for accuracy).
2. **Run text-only** through embedding → decoder (skip encoders).
3. **Compare embedding output** against `hf_model.model.embed_tokens(ids)`.
   If this diverges, the issue is in weight loading or embedding model.
4. **Compare decoder logits** against HF full forward.
   If embedding matches but logits diverge, issue is in decoder.

### Phase 2: Isolate decoder issues

5. **Check weight count** — verify all expected weights are loaded:
   ```python
   pkg = build(model_id, load_weights=True)
   # apply_weights prints statistics: applied, skipped, unmatched
   ```
6. **Disable LoRA** — if the model uses LoRA, zero out adapter weights and
   compare base model output against HF with adapters disabled.
7. **Layer-by-layer** — add intermediate outputs to the ONNX graph (see
   `references/extraction-methods.md`) to find which decoder layer first
   diverges.

### Phase 3: Add modalities

8. **Vision only** — run vision encoder, feed features to embedding, compare.
9. **Audio only** — run speech encoder, feed features to embedding, compare.
10. **Combined** — all modalities together.

At each step, if a newly added component causes divergence, isolate that
component's output against HF.

### Phase 4: LoRA verification

11. **Match input modes** — ensure HF reference uses the same adapter
    activation mode as ONNX (e.g., `input_mode=3` for both adapters).
12. **Compare with LoRA** — verify LoRA scaling factor: `alpha / rank`.
13. **Check adapter routing** — for multi-adapter models, verify the correct
    adapter set is active for each modality combination.

## Integration test patterns

### Test configuration

```python
# Always use for HF reference:
hf_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    attn_implementation="eager",  # No flash_attn dependency
    torch_dtype=torch.float32,     # Match ONNX precision
)

# For models with conditional LoRA:
hf_model.input_mode = 3  # Match ONNX unconditional LoRA
# Or: pass input_mode=3 to forward() if supported
```

### Text-only test

```python
def test_text_only_prefill_logits_match(self):
    input_ids = tokenizer.encode("Hello world", return_tensors="np")
    empty_image = np.zeros((0, hidden_size), dtype=np.float32)
    empty_audio = np.zeros((0, hidden_size), dtype=np.float32)

    embeds = embedding_session.run({
        "input_ids": input_ids,
        "image_features": empty_image,
        "audio_features": empty_audio,
    })["inputs_embeds"]

    onnx_logits = decoder_session.run({
        "inputs_embeds": embeds,
        "attention_mask": np.ones((1, seq_len), dtype=np.int64),
        "position_ids": np.arange(seq_len).reshape(1, -1),
        # ... zero KV cache
    })["logits"]

    hf_logits = hf_model(input_ids=..., input_mode=3).logits.numpy()
    assert_logits_close(onnx_logits, hf_logits)
```

### Audio test

```python
def test_audio_prefill_logits_match(self):
    # Prepare mel spectrogram input
    mel = load_audio_as_mel(audio_path)  # [1, n_mel, time]

    speech_out = speech_session.run({
        "audio_embeds": mel,
        "audio_sizes": np.array([[mel.shape[-1]]], dtype=np.int64),
        "audio_projection_mode": np.array(0, dtype=np.int64),
    })
    speech_features = speech_out["audio_features"]

    # Build input_ids with audio placeholder tokens
    input_ids = build_audio_input_ids(prompt, num_audio_tokens)

    embeds = embedding_session.run({
        "input_ids": input_ids,
        "image_features": np.zeros((0, hidden_size), dtype=np.float32),
        "audio_features": speech_features,
    })["inputs_embeds"]

    onnx_logits = decoder_session.run(...)["logits"]
    hf_logits = hf_forward_with_audio(...)
    assert_logits_close(onnx_logits, hf_logits)
```

## Weight loading verification

After `apply_weights`, check the statistics:
```python
# Expected output:
# Applied: 485/485 weights
# Skipped: 0  (weights in state_dict but not in graph)
# Unmatched: 0  (weights in state_dict with no graph match)

# If unmatched > 0, dump the names to find alignment issues:
pkg = build(model_id)
state_dict = download_weights(model_id)
state_dict = module.preprocess_weights(state_dict)
# Compare state_dict.keys() vs graph initializer names
```

## Vision-specific verification (HD multi-crop)

For models with HD dynamic resolution (Phi4MM, Phi3-Vision):

### Input preparation

```python
from transformers import AutoProcessor

processor = AutoProcessor.from_pretrained(
    model_id, trust_remote_code=True
)
inputs = processor(
    images=image,
    text=prompt,
    return_tensors="pt",
)
pixel_values = inputs["pixel_values"]  # [num_crops, C, H, W] or 5D
image_sizes = inputs["image_sizes"]     # [num_images, 2]
```

### HD transform verification

The HD transform typically:
1. Splits image into base (global) + sub-image crops
2. Encodes each crop through vision encoder → `[num_patches, hidden_size]`
3. Applies spatial merge (e.g., AvgPool2d + reshape) → compressed tokens
4. Adds learned separators (glb_GN between global/sub, sub_GN between subs)
5. Projects to text dimension via MLP

```python
# Verify token count matches expected:
# global: (image_size/patch_size)^2 / merge^2 tokens
# per sub-image: same count
# separators: 1 glb_GN + (num_subs - 1) sub_GN rows
total_expected = global_tokens + num_subs * sub_tokens + separator_count
assert image_features.shape[0] == total_expected
```

### Testing without HD (simpler)

For initial validation, use base resolution (single crop, no HD):
```python
# Single image at base resolution — bypasses HD transform
pixel_values = np.random.randn(1, 3, 384, 384).astype(np.float32)
image_sizes = np.array([[384, 384]], dtype=np.int64)
```
