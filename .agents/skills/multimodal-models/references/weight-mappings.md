# Weight Name Mappings — Full Reference

## Common multimodal weight prefixes

Multimodal HF models often prefix text weights differently from the ONNX
model structure.  These must be handled in `preprocess_weights()`.

| HF key | ONNX key |
|--------|---------|
| `language_model.model.layers.0.…` | `layers.0.…` |
| `vision_tower.vision_model.encoder.…` | `vision_tower.encoder.…` |
| `multi_modal_projector.mm_input_projection_weight` | `multi_modal_projector.weight` |

## Weight tying

If `tie_word_embeddings=True`, the HF checkpoint may not include
`lm_head.weight`.  Copy it from `embed_tokens.weight`:

```python
if self.config.tie_word_embeddings:
    if "lm_head.weight" not in renamed and "embed_tokens.weight" in renamed:
        renamed["lm_head.weight"] = renamed["embed_tokens.weight"]
```

## ClippableLinear weight mapping

HuggingFace stores `<prefix>.linear.weight` for the actual weight (the
`.linear.` segment is stripped by `preprocess_weights`), and
`<prefix>.input_min`, `<prefix>.input_max`, `<prefix>.output_min`,
`<prefix>.output_max` as direct scalar buffers.

## Per-layer embedding weight splitting (Gemma4)

Gemma4 uses per-layer embedding: `embed_tokens_per_layer` with shape
`[V, L*D]` where V=vocab_size, L=num_layers, D=per_layer_dim. In
`preprocess_weights`, split the HF weight column-wise:

```python
for i in range(num_layers):
    renamed[f"embed_tokens_per_layer.{i}.weight"] = value[:, i*D:(i+1)*D]
```

## PatchEmbedding parameter names

`PatchEmbedding` uses explicit `name=` because the attribute names don't
match the desired ONNX names:

```python
self.patch_embedding = nn.Parameter([...], name="patch_embedding.weight")
self.patch_embedding_bias = nn.Parameter([...], name="patch_embedding.bias")
self.position_embedding = nn.Parameter([...], name="position_embedding.weight")
```

## Shape mismatches requiring preprocess_weights transforms

Some HF weights have different shapes from the ONNX parameter declaration:

```python
# Squeeze extra batch dimension from position embeddings
# HF: [1, num_patches, hidden_size] (3D) → ONNX: [num_patches, hidden_size] (2D)
if "position_embedding.weight" in key and state_dict[key].dim() == 3:
    state_dict[key] = state_dict[key].squeeze(0)
```

**General rule:** Check whether the preprocess_weights transform goes
in the correct direction (squeeze vs unsqueeze). A common mistake is
writing the transform backwards.

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
