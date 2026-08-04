# MultiModal Pipeline Architecture

This document describes the detailed ORT GenAI MultiModal pipeline
architecture, including the VLM prompt flow, generation loop, input routing,
and processor outputs. For an overview, see the parent [SKILL.md](../SKILL.md).

---

## VLM prompt flow (3-model split)

```
pixel_values + image_grid_thw  →  [vision_encoder/model.onnx]  →  image_features
                                                          │
input_ids + image_features     →  [embedding.onnx] → inputs_embeds
                                                          │
inputs_embeds + position_ids   →  [model.onnx]    →  logits
              + past_kv
```

---

## VLM generation flow

```
Prompt stage:
  1. VisionState.Run()         →  image_features
  2. EmbeddingState.ReuseFeaturesBuffer(image_features)
  3. EmbeddingState.Run()      →  inputs_embeds
  4. DecoderState.Run()        →  logits + present_kv
  5. VisionState destroyed (no longer needed)

Token generation stage (loop):
  1. EmbeddingState.Run()      →  inputs_embeds (from single token)
  2. DecoderState.Run()        →  logits + present_kv
```

---

## Input flow

When `generator.set_inputs(named_tensors)` is called:

1. Tensors matching vision model input names → fed to VisionState
2. Tensors matching embedding model input names → fed to EmbeddingState
3. `input_ids` → used for token counting and embedding lookup
4. `num_image_tokens` → used to allocate image_features buffer size

---

## QwenImageProcessor outputs

| Tensor | Shape | Description |
|---|---|---|
| `input_ids` | (1, seq_len) | Tokenized prompt with image_pad tokens |
| `pixel_values` | (total_patches, C×T×P×P) | Flattened image patches |
| `image_grid_thw` | (num_images, 3) | Grid dimensions per image |
| `num_image_tokens` | (1,) | Total merged image tokens |

> **Important:** The ORT GenAI QwenImageProcessor does NOT produce
> `cu_seqlens`, `cu_window_seqlens`, or `rotary_pos_ids`. If the vision
> ONNX model requires these, they must be computed externally and injected
> into the NamedTensors.

---

## Multimodal processor factory

When `model.create_multimodal_processor()` is called:

| model.type | Processor class |
|---|---|
| `phi3v` | PhiImageProcessor |
| `whisper` | WhisperProcessor |
| `phi4mm` | PhiMultiModalProcessor |
| `gemma3` | GemmaImageProcessor |
| `fara` | QwenImageProcessor |
| `qwen2_5_vl` | QwenImageProcessor |

> Models not in this table cannot use `create_multimodal_processor()`.

---

## Writing genai_config.json from ArchitectureConfig

```python
def _write_genai_config(config, output_dir, model_type="qwen2_5_vl"):
    genai_config = {
        "model": {
            "bos_token_id": config.bos_token_id or 151643,
            "context_length": 4096,
            "decoder": {
                "session_options": {
                    "log_id": "onnxruntime-genai",
                    "provider_options": [],
                },
                "filename": "decoder/model.onnx",
                "head_size": config.head_dim,
                "hidden_size": config.hidden_size,
                "inputs": {
                    "inputs_embeds": "inputs_embeds",
                    "attention_mask": "attention_mask",
                    "position_ids": "position_ids",
                    "past_key_names": "past_key_values.%d.key",
                    "past_value_names": "past_key_values.%d.value",
                },
                "outputs": {
                    "logits": "logits",
                    "present_key_names": "present.%d.key",
                    "present_value_names": "present.%d.value",
                },
                "num_attention_heads": config.num_attention_heads,
                "num_hidden_layers": config.num_hidden_layers,
                "num_key_value_heads": config.num_key_value_heads,
            },
            "embedding": {
                "filename": "embedding/model.onnx",
                "inputs": {
                    "input_ids": "input_ids",
                    "image_features": "image_features",
                },
                "outputs": {
                    "inputs_embeds": "inputs_embeds",
                },
            },
            "vision": {
                "filename": "vision_encoder/model.onnx",
                "spatial_merge_size": 2,
                "inputs": {
                    "pixel_values": "pixel_values",
                    "image_grid_thw": "image_grid_thw",
                },
                "outputs": {
                    "image_features": "image_features",
                },
            },
            "eos_token_id": config.eos_token_id or [151645, 151643],
            "pad_token_id": config.pad_token_id or 151643,
            "image_token_id": 151655,
            "vision_start_token_id": 151652,
            "type": model_type,
            "vocab_size": config.vocab_size,
        },
        "search": {
            "do_sample": False,
            "early_stopping": True,
            "max_length": 4096,
            "num_beams": 1,
            "num_return_sequences": 1,
            "past_present_share_buffer": False,
            "repetition_penalty": 1.0,
            "temperature": 1.0,
            "top_k": 1,
            "top_p": 1.0,
        },
    }
    with open(os.path.join(output_dir, "genai_config.json"), "w") as f:
        json.dump(genai_config, f, indent=4)
```
