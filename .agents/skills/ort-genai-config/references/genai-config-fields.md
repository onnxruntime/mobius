# genai_config.json — Complete Field Reference

This document is the exhaustive field-by-field reference for every section of
`genai_config.json`. For an overview and minimal config example, see the
parent [SKILL.md](../SKILL.md).

---

## model section — Model-level fields

| Field | Type | Required | Description |
|---|---|---|---|
| `type` | string | **yes** | Model type identifier (see registry in SKILL.md) |
| `vocab_size` | int | yes | Vocabulary size |
| `context_length` | int | **yes** | Maximum context length; must be > 0 |
| `bos_token_id` | int | no | Beginning-of-sequence token |
| `eos_token_id` | int \| int[] | no | End-of-sequence token(s); defaults to `pad_token_id` |
| `pad_token_id` | int | no | Padding token |
| `sep_token_id` | int | no | Separator token |
| `decoder_start_token_id` | int | no | Decoder start token (encoder-decoder models) |
| `image_token_id` | int | VLM | Token ID for image placeholders (e.g. 151655 for Qwen2.5-VL). **Required** for 3D M-RoPE position ID computation. |
| `video_token_id` | int | no | Token ID for video placeholders (e.g. 151656) |
| `vision_start_token_id` | int | VLM | Token ID for `<\|vision_start\|>` (e.g. 151652). Used to locate image/video regions in input_ids. |

---

## model.decoder

The decoder (text model) configuration.

### Core fields

| Field | Type | Required | Description |
|---|---|---|---|
| `filename` | string | **yes** | ONNX model filename (e.g. `"model.onnx"`) |
| `hidden_size` | int | yes | Hidden dimension |
| `head_size` | int | yes | Size per attention head |
| `num_attention_heads` | int | yes | Number of query attention heads |
| `num_key_value_heads` | int | yes | Number of KV heads (for GQA) |
| `num_hidden_layers` | int | yes | Number of transformer layers |
| `session_options` | object | no | ORT session configuration |
| `run_options` | object | no | ORT run options |

### Decoder inputs

```json
"inputs": {
  "input_ids": "input_ids",
  "inputs_embeds": "inputs_embeds",
  "attention_mask": "attention_mask",
  "position_ids": "position_ids",
  "past_key_names": "past_key_values.%d.key",
  "past_value_names": "past_key_values.%d.value"
}
```

The `%d` in `past_key_names` / `past_value_names` is replaced with the layer
index (0 to num_hidden_layers-1) at load time.

Additional optional inputs for advanced scenarios:

```json
"past_names": "",
"cross_past_key_names": "",
"cross_past_value_names": "",
"past_key_values_length": "past_key_values_length",
"past_sequence_length": "past_sequence_length",
"current_sequence_length": "current_sequence_length",
"total_sequence_length": "total_sequence_length",
"cache_indirection": "cache_indirection",
"encoder_hidden_states": "encoder_hidden_states",
"encoder_attention_mask": "encoder_attention_mask",
"cumulative_sequence_lengths": "cumulative_sequence_lengths",
"past_sequence_lengths": "past_sequence_lengths",
"block_table": "block_table"
```

### Decoder outputs

```json
"outputs": {
  "logits": "logits",
  "present_key_names": "present.%d.key",
  "present_value_names": "present.%d.value"
}
```

### Sliding window (optional)

```json
"sliding_window": {
  "window_size": 4096,
  "pad_value": -1,
  "alignment": "right",
  "slide_key_value_cache": true,
  "slide_inputs": true,
  "layers": [0, 2, 4]
}
```

---

## model.embedding

Required for VLM and MMM models. Merges text token embeddings with vision/audio
features.

```json
"embedding": {
  "filename": "embedding.onnx",
  "inputs": {
    "input_ids": "input_ids",
    "image_features": "image_features",
    "audio_features": "audio_features"
  },
  "outputs": {
    "inputs_embeds": "inputs_embeds"
  }
}
```

---

## model.vision

Required for VLM and MMM models.

| Field | Type | Default | Description |
|---|---|---|---|
| `filename` | string | — | Vision ONNX model |
| `config_filename` | string | `"image_processor.json"` | Processor config file |
| `adapter_filename` | string | — | Optional adapter model |
| `spatial_merge_size` | int | 2 | **Required for Qwen2.5-VL.** Controls how many vision patches are merged into one token. Used to compute grid dimensions for 3D M-RoPE position IDs (h/merge × w/merge). |
| `tokens_per_second` | float | 2.0 | Video tokens/second |

### Vision inputs

```json
"inputs": {
  "pixel_values": "pixel_values",
  "image_sizes": "image_sizes",
  "image_grid_thw": "image_grid_thw",
  "attention_mask": "image_attention_mask"
}
```

### Vision outputs

```json
"outputs": {
  "image_features": "image_features"
}
```

### Vision pipeline (optional)

For models that split vision into stages (e.g. patch_embed → attention →
merger):

```json
"pipeline": [
  {
    "filename": "patch_embed.onnx",
    "model_id": "patch_embed",
    "inputs": ["pixel_values"],
    "outputs": ["patch_embeddings"],
    "run_on_cpu": false,
    "session_options": {}
  }
]
```

---

## model.speech

For audio-language models (whisper, phi4mm).

```json
"speech": {
  "filename": "speech.onnx",
  "config_filename": "audio_processor.json",
  "inputs": {
    "audio_embeds": "audio_embeds",
    "attention_mask": "audio_attention_mask",
    "audio_sizes": "audio_sizes",
    "audio_projection_mode": "audio_projection_mode"
  },
  "outputs": {
    "audio_features": "audio_features"
  }
}
```

---

## model.encoder

For encoder-decoder models (whisper).

```json
"encoder": {
  "filename": "encoder.onnx",
  "hidden_size": 1280,
  "num_attention_heads": 20,
  "num_hidden_layers": 32,
  "head_size": 64,
  "inputs": {
    "input_ids": "input_ids",
    "attention_mask": "attention_mask"
  },
  "outputs": {
    "encoder_hidden_states": "encoder_hidden_states"
  }
}
```

---

## search section

Controls generation behavior.

| Field | Type | Default | Description |
|---|---|---|---|
| `do_sample` | bool | false | Sampling vs greedy |
| `min_length` | int | 0 | Minimum output length |
| `max_length` | int | context_length | Maximum total length (prompt + output) |
| `batch_size` | int | 1 | Batch size |
| `num_beams` | int | 1 | Beam width (1 = greedy) |
| `num_return_sequences` | int | 1 | Sequences to return |
| `top_k` | int | 50 | Top-K sampling |
| `top_p` | float | 0.0 | Nucleus sampling |
| `temperature` | float | 1.0 | Sampling temperature |
| `repetition_penalty` | float | 1.0 | Repetition penalty (1.0 = none) |
| `length_penalty` | float | 1.0 | Beam search length penalty |
| `early_stopping` | bool | true | Stop beam search early |
| `past_present_share_buffer` | bool | false | Share KV cache buffer (CUDA) |
| `random_seed` | int | -1 | RNG seed (-1 = random) |
| `chunk_size` | int | — | Prefill chunking size |

---

## engine section (optional)

For batched serving.

```json
"engine": {
  "dynamic_batching": {
    "block_size": 256,
    "num_blocks": 16,
    "gpu_utilization_factor": 0.9,
    "max_batch_size": 16
  }
}
```

Or static batching:

```json
"engine": {
  "static_batching": {
    "max_batch_size": 4
  }
}
```

Dynamic and static batching are mutually exclusive.

---

## session_options

Nested inside `decoder`, `encoder`, `vision`, `speech`, or `embedding`.

```json
"session_options": {
  "intra_op_num_threads": 8,
  "inter_op_num_threads": 1,
  "log_id": "onnxruntime-genai",
  "log_severity_level": 2,
  "enable_cpu_mem_arena": true,
  "enable_mem_pattern": true,
  "enable_profiling": "profile.json",
  "graph_optimization_level": "ORT_ENABLE_EXTENDED",
  "provider_options": [
    {
      "cuda": {
        "device_id": "0"
      }
    }
  ]
}
```

Graph optimization levels: `ORT_DISABLE_ALL`, `ORT_ENABLE_BASIC`,
`ORT_ENABLE_EXTENDED`, `ORT_ENABLE_ALL`.

Provider names are normalized: `"qnn"` → `"QNN"`, `"dml"` → `"DML"`,
`"webgpu"` → `"WebGPU"`, `"openvino"` → `"OpenVINO"`.
