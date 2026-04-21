# Projector Variants — Detailed Reference

## Gemma3MultiModalProjector

```python
Gemma3MultiModalProjector(
    vision_hidden_size=1152,     # SigLIP hidden dim
    text_hidden_size=2560,       # Text model hidden dim
    patches_per_image=64,        # sqrt(num_patches) per side
    tokens_per_image=256,        # mm_tokens_per_image from config
    norm=Gemma3RMSNorm(1152),    # Gemma3-specific RMSNorm with +1 offset
)
```

The pooling kernel is computed as `patches_per_image / sqrt(tokens_per_image)`.
For Gemma3-4B: `64 / 16 = 4`, so `AvgPool2d(kernel_size=4, stride=4)`.

## MLPMultiModalProjector

```python
MLPMultiModalProjector(
    vision_hidden_size=1024,
    text_hidden_size=4096,
    bias=True,
)
```

Two-layer MLP with GELU activation.  The most common projector pattern.

## LinearMultiModalProjector

```python
LinearMultiModalProjector(
    vision_hidden_size=1024,
    text_hidden_size=4096,
    bias=True,
)
```

Simple single linear layer.

## Mistral3MultiModalProjector

RMSNorm → spatial merge → Linear → GELU → Linear.  Used by Mistral-3 and
Pixtral.  Defined in `components/_pixtral_vision.py`.

## Qwen2.5-VL / Qwen3-VL vision encoder specifics

These models use a **custom vision encoder** (not SigLIP) with unique
architectural features. The encoder is in
`components/_qwen25_vl_vision.py` and `components/_qwen3_vl_vision.py`.

### Architecture differences from standard VisionModel

| Feature | Standard (SigLIP) | Qwen2.5-VL / Qwen3-VL |
|---------|-------------------|----------------------|
| Patch embedding | Conv2d | **Conv3d** (temporal + spatial) |
| Position encoding | Learnable embedding | **2D rotary** (height, width) |
| Attention | Standard self-attention | **Windowed + full attention** alternating |
| Normalization | LayerNorm | **RMSNorm** |
| Output merging | CLS token or mean pool | **Spatial merge** (2×2 → 1) |
| MLP | fc1/fc2 | **Gated MLP** (gate_proj/up_proj/down_proj + SiLU) |

### Critical: 2D rotary embedding dimension

The vision encoder computes separate rotary frequencies for height and
width positions. The rotary embedding dimension must be `head_dim // 2`
(not `head_dim`):

```python
# CORRECT: each spatial dimension gets head_dim//4 frequencies
self.rotary_pos_emb = Qwen25VLVisionRotaryEmbedding(head_dim // 2)

# WRONG: produces 2× too many frequencies with wrong values
self.rotary_pos_emb = Qwen25VLVisionRotaryEmbedding(head_dim)
```

The frequency table has shape `(num_patches, head_dim//2)`. Each half
(`head_dim//4` values) covers one spatial dimension. The `forward` method
concatenates `cos(h_freqs)` and `cos(w_freqs)` to produce the final
`(num_patches, head_dim)` rotary embeddings.

### Critical: fullatt_block_indexes config

Qwen2.5-VL uses a **hybrid attention pattern**: most blocks use windowed
attention (local windows for efficiency), but certain blocks use full
attention (all patches attend to all patches):

```python
# Must be extracted from HF vision_config
fullatt_block_indexes = [7, 15, 23, 31]  # For 32-block encoder
window_size = 112  # Window size in patches for windowed blocks
```

If `fullatt_block_indexes` is missing, ALL blocks use windowed attention,
causing massive feature divergence (cos ≈ 0.25). The first few blocks may
appear correct since they happen to be windowed blocks.

**Config extraction** — these must be in `_configs.py` VisionConfig:

```python
@dataclasses.dataclass
class VisionConfig:
    ...
    fullatt_block_indexes: list[int] | None = None
    window_size: int | None = None
```

### Window index and attention bias

- **Windowed blocks**: Patches are grouped into windows of `window_size`.
  Each window attends only within itself. The attention bias is block-diagonal.
- **Full attention blocks**: Use `cu_seqlens` (not `cu_window_seqlens`) to
  attend across all patches in each image.
- `window_index` permutes patches into window-ordered layout before the
  transformer blocks, then `reverse_indices = argsort(window_index)` restores
  the original order after.

### Multi-image support

Both vision encoders support multiple images via the ONNX `Scan` op.
Per-image values (position IDs, window indices, cu_seqlens) are computed
in a Scan body and concatenated. See `references/scan-pattern.md`.

### Spatial merge (post-encoder)

After the transformer blocks, a spatial merge layer combines 2×2 patches
into 1 token:
```
(num_patches, hidden_size) → reshape to (num_merged, 4*hidden_size) → MLP → (num_merged, text_hidden_size)
```
The merge reduces token count by 4× and projects to text model dimension.

## Qwen3.5-VL

Qwen3.5-VL uses the same **3-model split** as Qwen3-VL (decoder + vision +
embedding), but swaps the text decoder for the **Qwen3.5 architecture**
which uses hybrid DeltaNet + full attention instead of standard GQA.

### Architecture

The vision encoder is **identical to Qwen3-VL** — it reuses
`Qwen3VLVisionModel` (patch_size=16, hidden=1152, depth=27). Only the
text decoder changes.

| Component | Class | Notes |
|-----------|-------|-------|
| 3-model composite | `Qwen35VL3ModelCausalLMModel` | Splits into decoder + vision + embedding |
| Decoder (standalone) | `Qwen35VLDecoderModel` | Uses `Qwen35TextModel` internally |
| Text model | `Qwen35VLTextModel` | Text-only decoder; strips VL weight prefixes |

### Registration

| `model_type` | Variant | Description |
|--------------|---------|-------------|
| `qwen3_5_vl` | 3-model split | Full VLM with vision encoder |
| `qwen3_5_vl_text` | Text-only | Decoder without vision |

### Task

Reuses `Qwen3VLVisionLanguage3ModelTask` (task name: `qwen35-vl`).

### Config

The HF config is VL-style with a nested `text_config`:

```
config.json          →  model_type: "qwen3_5_vl"
config.text_config   →  model_type: "qwen3_5"   (or "qwen3_5_text")
```

### Token IDs

| Token | ID |
|-------|----|
| `image` | 248056 |
| `video` | 248057 |
| `vision_start` | 248053 |
| `vision_end` | 248054 |

### Interleaved MRoPE

Uses `InterleavedMRope` (not `ChunkedMRope`) with:

- `partial_rotary_factor=0.25`
- `mrope_section=[11, 11, 10]`

### Key insight

The vision pipeline is completely shared with Qwen3-VL — only the text
decoder differs (hybrid DeltaNet + full attention). This means vision
encoder bugs/fixes apply to both models equally.
