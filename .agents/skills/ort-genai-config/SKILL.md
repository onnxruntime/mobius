---
name: ort-genai-config
description: >
  Use this skill when generating genai_config.json or image_processor.json
  for onnxruntime-genai model exports, debugging ORT GenAI model loading
  errors, understanding the model type registry, or integrating ONNX models
  with the onnxruntime-genai runtime. Covers the full config format, the
  MultiModal pipeline architecture (vision/audio/embedding/decoder), and
  the ort-extensions image_processor.json format.
---

# Skill: ORT GenAI Config Format

## When to use

Use this skill when:

- Writing `genai_config.json` for a new model export
- Writing `image_processor.json` for image preprocessing (or `audio_processor.json` for audio)
- Debugging ORT GenAI model loading errors (protobuf parsing, missing keys)
- Understanding how the ORT GenAI pipeline feeds inputs to vision, embedding,
  and decoder models
- Adding support for a new model type in ORT GenAI

## Detailed references

Read these companion documents for exhaustive field-by-field details:

- Read **[`references/genai-config-fields.md`](references/genai-config-fields.md)**
  when you need the complete field table for any `genai_config.json` section
  (model, decoder, vision, embedding, speech, encoder, search, engine,
  session_options).
- Read **[`references/processor-config-fields.md`](references/processor-config-fields.md)**
  when you need the full `image_processor.json` transform reference, the
  HuggingFace-to-ort-extensions conversion code, or the Qwen2.5-VL example.
- Read **[`references/multimodal-pipeline.md`](references/multimodal-pipeline.md)**
  when you need the VLM 3-model prompt/generation flow, input routing
  details, QwenImageProcessor output tensors, the multimodal processor
  factory table, or the full `_write_genai_config` helper code.

---

## Overview

onnxruntime-genai loads models from a directory containing:

```
model_dir/
├── genai_config.json          # Required — model config + search params
├── model.onnx                 # Decoder model (single-model) OR:
├── decoder/model.onnx         # Decoder model (multimodal)
├── vision_encoder/model.onnx   # Vision encoder (multimodal only)
├── audio_encoder/model.onnx   # Audio encoder (multimodal only)
├── embedding/model.onnx       # Embedding model (multimodal only)
├── tokenizer.json             # Tokenizer (HuggingFace format)
├── tokenizer_config.json      # Tokenizer config
├── chat_template.jinja        # Chat template (optional)
├── image_processor.json       # Image processor (ort-extensions, all VLMs)
└── audio_processor.json       # Audio processor (multimodal only)
```

**Processor config filenames**: `image_processor.json` for all VLMs,
`audio_processor.json` for audio models.

## genai_config.json — Structure

The config has three top-level sections:

```json
{
  "model": { ... },
  "search": { ... },
  "engine": { ... }
}
```

- **`model`**: Model architecture — `type`, token IDs, and sub-sections
  `decoder`, `vision`, `embedding`, `speech`, `encoder`.
- **`search`**: Generation parameters — sampling, beam search, max length.
- **`engine`** *(optional)*: Batched serving (dynamic or static batching).

Key model-level fields: `type` (required), `vocab_size`, `context_length`
(required), `eos_token_id`, `pad_token_id`. VLMs also need `image_token_id`
and `vision_start_token_id`.

Key decoder fields: `filename`, `hidden_size`, `head_size`,
`num_attention_heads`, `num_key_value_heads`, `num_hidden_layers`, plus
`inputs`/`outputs` name mappings.

> For the complete field-by-field tables, see
> [`references/genai-config-fields.md`](references/genai-config-fields.md).

---

## Model type registry

### LLM (decoder-only → `DecoderOnly_Model`)

```
chatglm, decoder, ernie4_5, gemma, gemma2, gemma3_text, gpt2,
gptoss, granite, internlm2, llama, mistral, nemotron, olmo,
phi, phimoe, phi3, phi3small, qwen2, qwen3, smollm3
```

### VLM (vision-language → `MultiModalLanguageModel`)

```
fara, gemma3, phi3v, qwen2_5_vl
```

### MMM (vision + audio → `MultiModalLanguageModel`)

```
gemma4, phi4mm
```

### ALM (audio-language → `WhisperModel`)

```
whisper
```

### Pipeline models → `DecoderOnlyPipelineModel`

```
phi3small_pipeline, qwen2_5_vl_pipeline
```

### Special handling

- `fara` / `qwen2_5_vl` with non-empty `model.decoder.pipeline` →
  `Qwen2_5_VL_PipelineModel`
- `IsQwen25VL()` (type == `"fara"` or `"qwen2_5_vl"`) enables 3D MRoPE
  position ID handling
- `gpt2` has a special code path (`Gpt_Model`) but is also in the LLM list

---

## Minimal valid config — decoder-only LLM

```json
{
  "model": {
    "type": "llama",
    "vocab_size": 32000,
    "context_length": 4096,
    "eos_token_id": 2,
    "pad_token_id": 0,
    "decoder": {
      "filename": "model.onnx",
      "hidden_size": 4096,
      "head_size": 128,
      "num_attention_heads": 32,
      "num_key_value_heads": 8,
      "num_hidden_layers": 32,
      "inputs": {
        "input_ids": "input_ids",
        "attention_mask": "attention_mask",
        "position_ids": "position_ids",
        "past_key_names": "past_key_values.%d.key",
        "past_value_names": "past_key_values.%d.value"
      },
      "outputs": {
        "logits": "logits",
        "present_key_names": "present.%d.key",
        "present_value_names": "present.%d.value"
      }
    }
  },
  "search": {
    "do_sample": false,
    "max_length": 4096,
    "num_beams": 1,
    "past_present_share_buffer": false
  }
}
```

VLM models additionally require `model.vision`, `model.embedding`,
`image_token_id`, and `vision_start_token_id`. See
[`references/genai-config-fields.md`](references/genai-config-fields.md) for
the full vision/embedding/speech/encoder schemas.

---

## image_processor.json overview

> **Critical:** ORT GenAI expects the **ort-extensions** format — NOT the
> HuggingFace format (HF's `preprocessor_config.json` uses `"image_processor"`
> as the top key). The ort-extensions `image_processor.json` uses `"processor"`
> with an ordered transform pipeline.

Structure:

```json
{
  "processor": {
    "name": "<processor_name>",
    "transforms": [
      { "operation": { "name": "...", "type": "...", "attrs": { ... } } }
    ]
  }
}
```

Transform types: `DecodeImage`, `ConvertRGB`, `Resize`, `Rescale`,
`Normalize`, `PatchImage`.

> For the full Qwen2.5-VL example, transform field reference, and
> HuggingFace conversion code, see
> [`references/processor-config-fields.md`](references/processor-config-fields.md).

---

## MultiModal pipeline overview

VLM models use a 3-model split:

```
pixel_values + image_grid_thw  →  [vision_encoder/model.onnx]  →  image_features
                                                                   │
input_ids + image_features     →  [embedding/model.onnx]   →  inputs_embeds
                                                                   │
inputs_embeds + position_ids   →  [decoder/model.onnx]     →  logits
              + past_kv
```

During generation, the vision model runs once at prompt time. The embedding
and decoder models run each token step.

### GenAI config input introspection

Decoder, vision, embedding, and audio input mappings in `genai_config.json`
are discovered from the ONNX graph via `_graph_input_names()` — they are
NOT hardcoded. This helper filters out KV cache inputs
(`past_key_values.*`, `past_*`) and returns semantic input names.

```python
# In auto_export.py
decoder_input_names = _graph_input_names(decoder_model)
decoder_inputs = {name: name for name in decoder_input_names}
```

This means the genai config automatically adapts when `RemoveDeadGraphInputsPass`
removes unused inputs (e.g. `position_ids` absorbed by GQA fusion).

Hybrid cache metadata must preserve global layer indices across KV, convolution,
and recurrent states. Derive slot count from the maximum
`past_key_values.<index>.*` input index plus one; counting only `.key` inputs
silently drops non-KV or sparsely indexed layers.

CUDA Graph capture belongs on stable-shape autoregressive decoder sessions,
not one-shot variable-shape vision/embedding stages. Keep an explicit decoder
opt-out and validate capture together with shared KV buffers on the real model.

> For the full generation flow, input routing, QwenImageProcessor output
> tensors, and the multimodal processor factory, see
> [`references/multimodal-pipeline.md`](references/multimodal-pipeline.md).

---

## Troubleshooting

### "Protobuf parsing failed"

Missing `model.vision` and/or `model.embedding` sections in genai_config.json.
VLM models require all three model sections.

### "key 'processor' not found"

The `image_processor.json` is in HuggingFace format instead of ort-extensions
format. The HF format has `"image_processor"` as the top key; ORT extensions
needs `"processor"` with a transforms pipeline.

The filename is specified via `config_filename` in the vision section of
genai_config.json. Audio uses `audio_processor.json`.

### "Missing Input: cu_window_seqlens"

The vision ONNX model expects packed-attention inputs that the ORT GenAI
processor doesn't provide. Either:
1. Compute them externally and inject via NamedTensors, or
2. Modify the vision model to compute them from `image_grid_thw` internally

### Config writes successfully but runtime cannot execute

Config/schema success is not runtime support. Run load plus generation through
the exported contract. If the runtime cannot route required feature inputs,
position-ID rank, cache state, scheduler, or multimodal metadata, reject export
before writing artifacts and report the exact runtime version/limitation.

NemotronH is unsupported in ORT GenAI 0.15.2 because sparse global layer
indices mix attention key/value caches with Mamba `conv_state` and
`ssm_state`. Do not hard-code a model-type or runtime-version rejection.
Inspect the actual decoder graph and reject only when it requires cache inputs
the generated config has no template for (currently `ssm_state`). This keeps
the guard structural: remove it when config emission can represent that graph
contract. Record the tested-version waiver and provide a direct ONNX Runtime
loop for current users.

### "input_ids size exceeds max length"

For image prompts, the tokenized input_ids (including image_pad tokens) can
be much longer than the default `max_length` in search options. Use
`params.set_search_options(max_length=4096)` or a sufficiently large value.

### "OrtValue shape verification failed"

Mismatch between `num_image_tokens` (computed by the processor) and the
actual vision model output shape. Ensure the same image processor is used
consistently — don't mix ORT GenAI processor output with HF processor
pixel_values.

### Image not recognized despite being processed

If the model generates coherent text but fails to describe image contents:

1. **Missing `image_token_id` or `spatial_merge_size`:** Without these,
   ORT GenAI cannot compute 3D M-RoPE position IDs. Add `image_token_id`,
   `vision_start_token_id` at model level and `spatial_merge_size` under
   model.vision.

2. **image_processor.json resize mismatch:** The Resize transform uses
   `width`/`height` as direct target dimensions. If too small, image loses
   detail. Compute correct dimensions:
   ```python
   factor = patch_size * merge_size  # 28
   new_h = max(factor, round(orig_h / factor) * factor)
   new_w = max(factor, round(orig_w / factor) * factor)
   ```

3. **ONNX model numerical accuracy:** Logits may differ from HF (typical
   max_diff ~8 for VLMs), causing greedy decoding to diverge after 3-4
   tokens.

---

## Source reference files

- **ORT GenAI config structs:**
  `/home/justinchu/dev/onnxruntime-genai/src/config.h`
- **ORT GenAI config parsing:**
  `/home/justinchu/dev/onnxruntime-genai/src/config.cpp`
- **Model type registry:**
  `/home/justinchu/dev/onnxruntime-genai/src/model_type.h`
- **VLM pipeline:**
  `/home/justinchu/dev/onnxruntime-genai/src/models/multi_modal.cpp`
- **Qwen image processor:**
  `/home/justinchu/dev/onnxruntime-genai/src/models/qwen2_5_vl_image_processor.cpp`
- **Reference image_processor.json:**
  `/home/justinchu/dev/onnxruntime-genai/test/test_models/qwen-vision-preprocessing/processor_config.json`
- **Example genai_config generation:**
  `examples/qwen25_vl_ort_genai.py`
