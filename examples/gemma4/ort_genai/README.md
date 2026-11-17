# Gemma 4 ORT GenAI Config Files

ORT GenAI configuration files for the **`google/gemma-4-E2B-it`** checkpoint
(Gemma 4 Embedding-Enhanced 2B instruction-tuned model).

> **Important:** The `gemma4`, `gemma4_text`, and `gemma4_any_to_any` model
> types are not yet in a released ORT GenAI build. These configs require ORT
> GenAI support for the Gemma 4 architecture. See the *ORT GenAI Support*
> section below.

---

## Directory layout

```
ort_genai/
├── text/
│   └── genai_config.json        # Text-only decoder (model.onnx)
└── vlm/
    ├── genai_config.json        # Full multimodal config (vision + audio + text)
    └── image_processor.json      # SigLIP image processor config
```

The VLM config is the full any-to-any config — it includes the audio section
when the model has audio support.  ORT GenAI auto-detects the pipeline variant
from which ONNX files are present in the directory.

---

## Key architecture values (`google/gemma-4-E2B-it`)

| Field | Value |
|---|---|
| `vocab_size` | 262 144 |
| `hidden_size` | 1 536 |
| `num_attention_heads` | 8 |
| `num_key_value_heads` | 1 |
| `head_dim` (local/sliding layers) | 256 |
| `global_head_dim` (full-attention layers) | 512 |
| `num_hidden_layers` (total) | 35 |
| `num_kv_shared_layers` | 20 |
| **KV cache depth** | **15** (= 35 − 20) |
| `sliding_window` | 512 |
| `max_position_embeddings` | 131 072 |
| `bos_token_id` | 2 |
| `eos_token_id` | `[1, 106]` |
| `image_token_id` | 255 999 (`boi_token_id`) |
| `audio_token_id` | 258 881 |
| Vision patch size | 16 |
| Vision tokens per image | 280 |

---

## KV cache sharing (15, not 35, layers)

Gemma 4 uses **KV projection sharing**: the last `num_kv_shared_layers = 20`
decoder layers reuse the K and V projections from the preceding layers.
Those 20 layers therefore have **no independent KV cache entries**.

Only the first `35 − 20 = 15` layers produce their own KV cache, so:

- `num_hidden_layers` in `genai_config.json` is **15**, not 35.
- `past_key_values.{0..14}.key/value` are the only KV inputs/outputs.

ORT GenAI must be aware of this sharing pattern to feed the correct cached
K/V values to the shared layers during autoregressive decoding.

---

## Sliding window attention (5:1 local:global pattern)

The 35-layer stack alternates 4 sliding-window (local) layers followed by
1 full-attention (global) layer, repeated 7 times:

```
layers 0–3   : sliding_attention  (window = 512 tokens)
layer  4     : full_attention
layers 5–8   : sliding_attention
layer  9     : full_attention
layers 10–13 : sliding_attention
layer 14     : full_attention
(layers 15–34 share KV — no independent cache)
```

The `sliding_window.layers` field in the configs lists the **sliding** layer
indices within the 15-entry KV cache: `[0, 1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13]`.

### Dual head_dim

Sliding (local) attention layers use `head_dim = 256`; full (global)
attention layers use `global_head_dim = 512`. The `head_size` field in
`genai_config.json` is set to **256** (the local value). ORT GenAI support
for the Gemma 4 model type must handle the per-layer head_dim difference
internally.

---

## Vision encoding

Gemma 4 uses a **SigLIP** vision encoder (ViT, patch_size=16, image_size=448)
with a "pan-and-scan" tiling strategy for high-resolution images.

The ONNX vision model (`vision_encoder/model.onnx`) takes **pre-patchified** inputs:
- `pixel_values [batch, num_patches, 3 * 16 * 16]` — flattened patch pixels
- `pixel_position_ids [batch, num_patches, 2]` — (row, col) patch coordinates

This differs from other VLMs that pass raw images and use `image_grid_thw`.
The HuggingFace `AutoProcessor` produces the pre-patchified format directly.

Each image produces **280 soft tokens** (`vision_soft_tokens_per_image = 280`).

---

## Audio encoding

Gemma 4 Any-to-Any uses a **Conformer** audio encoder (12 layers,
hidden_size=1024) with 4× subsampling.

The ONNX audio model (`audio_encoder/model.onnx`) takes:
- `input_features [batch, time, 128]` — 128-dim mel spectrogram

Output: `audio_features [batch, time/4, 1536]` — projected to text hidden_size.

`audio_token_id = 258881` identifies audio soft-token positions in `input_ids`.

---

## ORT GenAI support required

The decoder-only config uses the generic `decoder` type. Multimodal configs
retain `gemma4` because that value selects the vision/audio pipeline:

| Config | `model.type` | Pipeline variant |
|---|---|---|
| `text/genai_config.json` | `decoder` | Decoder-only (text) |
| `vlm/genai_config.json` | `gemma4` | Multimodal (vision + audio + text) |

The specialized multimodal runtime selects its pipeline variant from which
ONNX files are present.

---

## Usage example

```python
import onnxruntime_genai as og

# Text-only
model = og.Model("path/to/gemma4/ort_genai/text")
tokenizer = og.Tokenizer(model)
params = og.GeneratorParams(model)
params.set_search_options(max_length=512)
generator = og.Generator(model, params)

# VLM (requires vision_encoder/ + embedding/ alongside decoder/)
model = og.Model("path/to/gemma4/ort_genai/vlm")
processor = og.MultiModalProcessor(model)
# ... load image and tokenize prompt with processor ...
```

The model directory must contain the ONNX files exported from mobius:
```
text/
  model.onnx          ← exported by Gemma4TextCausalLMTask
  tokenizer.json
  genai_config.json

vlm/
  decoder/model.onnx          ← decoder (Gemma4VisionLanguageTask / Gemma4AnyToAnyTask)
  vision_encoder/model.onnx  ← vision encoder
  audio_encoder/model.onnx   ← Conformer audio encoder (when model has audio)
  embedding/model.onnx       ← embedding fusion
  tokenizer.json
  genai_config.json
  image_processor.json
```
