---
name: adding-vl-models
description: >
  Step-by-step guide for adding vision-language (VL) models that use the
  3-model ONNX split (decoder / vision_encoder / embedding) to mobius.
  Covers VisionLanguageTask sub-module requirements, task variants, vlm
  weight helpers, weight-tying pitfalls, InternViT/RADIO encoders, Mllama
  cross-attention, BLIP-2 Q-Former, image_token_id table, and VisionConfig
  fields. Complements the multimodal-models skill -- read that first for
  projector variants, single-model InputMixer pattern, SigLIP VisionModel
  usage, and Qwen2.5/3-VL vision encoder deep dives.
---

# Skill: Adding VL Models (3-Model ONNX Split)

## Relationship to multimodal-models skill

**Read `multimodal-models` first** for:
- Projector variants (Gemma3, MLP, Linear) and their code
- `VisionModel`/SigLIP encoder usage and attribute naming
- Single-model `InputMixer` pattern (when decoder sees `inputs_embeds`)
- Qwen2.5/3-VL vision encoder deep dives
- ORT GenAI `genai_config.json` / `processor_config.json` formats
- Testing patterns (build-graph YAML configs, integration tests)

Use **this skill** for:
- How `VisionLanguageTask` splits into 3 ONNX models and sub-module naming
- Choosing among task variants (QwenVL, Mllama, BLIP-2, Phi4-MM)
- `vlm_decoder_weights` / `vlm_embedding_weights` weight helpers
- lm_head weight-tying across the decoder split
- InternViT encoder (fused QKV, layer scale, CLS, pixel shuffle)
- RADIO encoder (conditional position encoding -- not yet in mobius)
- `VisionConfig` fields reference
- `image_token_id` values table
- Mllama cross-attention and BLIP-2 Q-Former non-standard tasks

---

## 1. The 3-model ONNX split

Every VL model in mobius is decomposed into **three sub-models**:

```
pixel_values ---> [vision] ---> image_features
                                    |
input_ids ---> [embedding] <--------+   ---> inputs_embeds
                                                  |
inputs_embeds ---> [decoder] ---------------------> logits + KV cache
```

| ONNX model | I/O contract | Runs |
|------------|-------------|------|
| **vision** | `pixel_values [B,C,H,W] -> image_features [N, text_hidden]` | Once per image at prefill |
| **embedding** | `input_ids [B,S] + image_features [N,H] -> inputs_embeds [B,S,H]` | Once at prefill |
| **decoder** | `inputs_embeds [B,S,H] -> logits + KV cache` | Every token during generation |

### Sub-module naming requirements

`VisionLanguageTask` accesses sub-modules **by attribute name**. These names
are **mandatory** -- wrong names silently omit models from `ModelPackage`:

```python
class MyVLModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.decoder = _MyDecoderModel(config)        # MUST be "decoder"
        self.vision_encoder = _MyVisionEncoder(config) # MUST be "vision_encoder"
        self.embedding = _MyEmbeddingModel(config)     # MUST be "embedding"

    def forward(self, op, *args, **kwargs):
        raise NotImplementedError  # task calls sub-modules directly
```

---

## 2. Task variants

| Task | `TASK_REGISTRY` key | Models | Key difference |
|------|---------------------|--------|----------------|
| `VisionLanguageTask` | `"vision-language"` | LLaVA, LLaVA-NeXT, PaliGemma, Pixtral, Molmo, InternVL2, Gemma3, GLM4V | Standard split |
| `QwenVLTask` | `"qwen-vl"` | Qwen2.5-VL, Qwen3-VL | MRoPE + packed vision input |
| `HybridQwenVLTask` | `"hybrid-qwen-vl"` | Qwen3.5-VL | MRoPE + DeltaNet hybrid cache |
| `HybridVisionLanguageTask` | `"hybrid-vision-language"` | LFM2-VL | Hybrid (conv+KV) cache in decoder |
| `Qwen3VLVisionLanguageTask` | `"qwen3-vl-vision-language"` | Qwen3-VL single-model variant | Single model (uses `InputMixer`) |
| `MllamaVisionLanguageTask` | `"mllama-vision-language"` | Mllama (Llama 3.2 Vision) | Cross-attention layers in decoder |
| `MultiModalTask` | `"multimodal"` | Phi4-MM | Audio + vision + text (4 models) |
| *(standard + Q-Former)* | `"vision-language"` | BLIP-2 | Q-Former between vision and embedding |

**Default for new models**: `VisionLanguageTask` unless MRoPE, hybrid cache,
cross-attention, or audio are involved.

---

## 3. Vision encoder summary

| Encoder | Used by | mobius component | Notes |
|---------|---------|-----------------|-------|
| SigLIP (ViT-SO/400M) | LLaVA, PaliGemma, InternVL2, Gemma3 | `VisionModel` | See multimodal-models skill |
| CLIP ViT-L/14 | LLaVA-1.5, Phi-3-Vision | `VisionModel` | See multimodal-models skill |
| SigLIP-2 | Phi-4-MM, Phi-4-SigLIP | `VisionModel` (+ S2 tiling) | See multimodal-models skill |
| InternViT | InternVL2 family | `_InternViT*` in `internvl.py` | Fused QKV, layer scale, CLS -- section 7 |
| Qwen2.5-VL ViT | Qwen2.5-VL | `Qwen25VLVisionModel` | Conv3D, 2D-RoPE -- see multimodal-models |
| Qwen3-VL ViT | Qwen3-VL | `Qwen3VLVisionModel` | Similar to Qwen2.5 -- see multimodal-models |
| RADIO (ViT-H/16 + CPE) | NemotronH VL | Not yet in mobius | Section 8 |

---

## 4. `VisionConfig` fields

`VisionConfig` is extracted from `hf_config.vision_config` by
`ArchitectureConfig.from_transformers`. These are the usable fields:

| Field | Type | HF key | Default |
|-------|------|--------|---------|
| `hidden_size` | `int` | `hidden_size` | 1152 |
| `image_size` | `int` | `image_size` | 224 |
| `patch_size` | `int` | `patch_size` | 14 |
| `num_hidden_layers` | `int` | `num_hidden_layers` | 27 |
| `num_attention_heads` | `int` | `num_attention_heads` | 16 |
| `intermediate_size` | `int` | `intermediate_size` | 4304 |
| `num_channels` | `int` | `num_channels` | 3 |
| `model_type` | `str` | `model_type` | `""` |

Always set `vision` in build-graph test configs using `VisionConfig`:

```python
from mobius._configs import VisionConfig, VisionLanguageConfig

"my_vlm": VisionLanguageConfig(
    hidden_size=64, num_hidden_layers=2, num_attention_heads=2,
    intermediate_size=128, vocab_size=256,
    vision=VisionConfig(hidden_size=64, patch_size=14, image_size=56,
                        num_hidden_layers=2, num_attention_heads=2,
                        intermediate_size=128),
),
```

---

## 5. Weight mapping with vlm helpers

### `vlm_decoder_weights` and `vlm_embedding_weights`

`_weight_utils.py` exports two helpers to strip the `"language_model."` prefix
that HuggingFace wraps the text backbone in for VL models:

```python
from mobius._weight_utils import vlm_decoder_weights, vlm_embedding_weights

class MyVLModel(nn.Module):
    def preprocess_weights(self, weights):
        # Fix lm_head weight tying BEFORE stripping "language_model." prefix
        if "language_model.lm_head.weight" not in weights:
            weights["language_model.lm_head.weight"] = weights[
                "language_model.model.embed_tokens.weight"
            ]
        return {
            "decoder": vlm_decoder_weights(weights),     # strips "language_model."
            "vision_encoder": _vision_encoder_weights(weights),  # custom
            "embedding": vlm_embedding_weights(weights), # strips "language_model."
        }
```

`vlm_decoder_weights(weights)` returns all keys starting with
`"language_model."`, renamed without that prefix.

`vlm_embedding_weights(weights)` returns `{"embed_tokens.weight": ...}` mapped
from `"language_model.model.embed_tokens.weight"`.

### weight-tying pitfall

Many VL models share `embed_tokens.weight` with `lm_head.weight`
(`tie_word_embeddings=True`). HuggingFace does not save `lm_head.weight` in
this case. You **must** inject it at the **top-level** `preprocess_weights`
before `vlm_decoder_weights` strips the prefix:

```python
# CORRECT -- inject at top level, before prefix strip
def preprocess_weights(self, weights):
    if "language_model.lm_head.weight" not in weights:
        weights["language_model.lm_head.weight"] = weights[
            "language_model.model.embed_tokens.weight"
        ]
    return {"decoder": vlm_decoder_weights(weights), ...}
```

```python
# WRONG -- inject inside _DecoderModel.preprocess_weights is too late;
# by then the key is already "model.embed_tokens.weight" and the decoder
# module cannot see "language_model.*" keys anymore.
```

### Non-standard prefixes

| Model | HF prefix | Helper |
|-------|-----------|--------|
| LLaVA, InternVL2, Gemma3, Mllama, PaliGemma | `language_model.` | `vlm_decoder_weights` |
| BLIP-2 | `language_model.` | `vlm_decoder_weights` |
| Phi-3-Vision | `model.language_model.` | custom strip |

---

## 6. Step-by-step: adding a standard VL model

For projector code and VisionModel wiring, see `multimodal-models`. This is
a checklist of mobius-specific steps:

1. **Identify sub-model boundaries** from HF source:
   - `language_model.*` -> decoder + embedding
   - `vision_tower.*` / `vision_encoder.*` -> vision_encoder
   - `multi_modal_projector.*` / `connector.*` -> part of vision_encoder output

2. **Create `models/<name>.py`** with three inner classes + top-level wrapper:
   - `_<Name>DecoderModel` -- causal LM (reuse existing if possible)
   - `_<Name>VisionEncoderModel` -- vision encoder + projector
   - `_<Name>EmbeddingModel` -- embed_tokens + image scatter
   - `<Name>Model` -- composes all three, implements `preprocess_weights`

3. **Export** from `models/__init__.py`

4. **Register** in `_registry.py`:
   ```python
   reg.register("my_model_type", MyVLModel, task="vision-language")
   ```

5. **Add tiny test config** in `tests/_test_configs.py` using `VisionLanguageConfig`

6. **Run tests**:
   ```bash
   PYTHONPATH=$(pwd)/src python -m pytest tests/build_graph_test.py -k "my_model_type" -q
   PYTHONPATH=$(pwd)/src python -m pytest tests/build_graph_test.py tests/synthetic_parity_test.py -q --tb=short -n auto
   ```

---

## 7. InternViT encoder (InternVL2)

InternViT differs from standard SigLIP/CLIP in three ways:

1. **Fused QKV**: single `qkv` linear (3*hidden) instead of separate Q/K/V
2. **Layer scale**: learnable scalar vectors `ls1`/`ls2` that multiply
   sub-layer output before residual add
3. **CLS token**: prepended before patch tokens (total = num_patches + 1)

InternVL2 also uses **pixel shuffle downsampling** (`downsample_ratio=0.5`)
before the MLP projector, halving spatial resolution. The projector is
4-layer: `LayerNorm -> Linear -> GELU -> Linear`.

See `src/mobius/models/internvl.py` for `_InternViT*` implementations.

**Key weight name mappings:**

| HF key | mobius path |
|--------|------------|
| `vision_model.embeddings.class_embedding` | `vision_encoder.encoder.cls_token` |
| `vision_model.encoder.layers.N.attn.qkv.weight` | `vision_encoder.encoder.layers.N.attn.qkv.weight` |
| `vision_model.encoder.layers.N.layer_scale1.weight` | `vision_encoder.encoder.layers.N.ls1` |

---

## 8. RADIO encoder (NemotronH VL -- not yet in mobius)

RADIO (nvidia/RADIO-L) is a ViT-H/16 with **conditional position encoding (CPE)**:
a depthwise Conv2d that injects adaptive position information into each patch
feature, enabling dynamic-resolution images without re-scaling.

Architecture: standard ViT-H encoder + CPE modules inserted per-layer.
New component needed: `ConditionalPositionEncoding` (~40 LOC depthwise Conv2d).
Estimated complexity: Medium (CPE component only).

NemotronH VL uses RADIO as its vision encoder. You cannot substitute SigLIP
because the text decoder is cross-modally aligned to RADIO outputs.

---

## 9. Non-standard tasks

### Mllama / Llama 3.2 Vision (cross-attention)

Mllama uses **cross-attention layers** in the decoder that attend to vision
features directly -- unlike standard VL where vision features are only injected
at the embedding step.

Key details:
- `MllamaVisionLanguageTask` creates a decoder that accepts both
  `inputs_embeds` and `cross_attention_states` (vision features)
- Decoder has two KV caches: one for self-attention, one for cross-attention
- Cross-attention layer indices come from `config.cross_attention_layers`
- Use `MllamaConfig` from `mobius._configs`:

```python
from mobius._configs import MllamaConfig
reg.register("mllama", MllamaCausalLMModel,
             task="mllama-vision-language", config_cls=MllamaConfig)
```

### BLIP-2 (Q-Former)

BLIP-2 inserts a **Q-Former** (Querying Transformer) between the vision
encoder and the LLM. Q-Former uses learned query vectors that attend to
vision features, producing a fixed-length sequence (32 tokens) regardless
of image resolution. Q-Former is part of the `vision_encoder` sub-model.

### Phi4-MM (audio + vision + text)

Uses `MultiModalTask` (4 ONNX models: embedding, speech_encoder,
vision_encoder, decoder). See `tasks/_multi_modal.py` and `models/phi4mm.py`.
For debugging, see the `phi4mm-component-parity` skill.

---

## 10. `image_token_id` reference

| Model family | `image_token_id` | Source in HF config |
|-------------|-----------------|---------------------|
| LLaVA-1.5 (Vicuna) | 32000 | `image_token_index` |
| LLaVA-OneVision (Qwen2) | 151655 | `image_token_index` |
| Gemma3 | 262144 | `image_token_id` |
| InternVL2 | 151667 | `img_context_token_id` |
| Mllama | 128256 | `image_token_id` |
| BLIP-2 | 50265 | `image_token_id` |
| PaliGemma | 257152 | `image_token_id` |
| Qwen2.5-VL / Qwen3-VL | 151655 | `vision_start_token_id` |
| Phi-3-Vision | 32044 | `image_token_id` |

Always read `image_token_id` from `hf_config` -- never hardcode.

---

## 11. Common pitfalls

### Wrong sub-module attribute names

`VisionLanguageTask` accesses `module.decoder`, `module.vision_encoder`, and
`module.embedding` by name. Using different names silently produces a
`ModelPackage` missing those models.

### lm_head weight missing in decoder

If `tie_word_embeddings=True`, `lm_head.weight` is absent from the HF
checkpoint. Inject it at the top-level `preprocess_weights` **before**
calling `vlm_decoder_weights()`. See section 5.

### `forward()` called on top-level module

`VisionLanguageTask` calls sub-modules directly. The top-level `forward()`
should raise `NotImplementedError` to catch accidental calls.

### Missing `vision` in build-graph test config

`VisionLanguageTask._build_vision` uses `config.vision.image_size`. Always
set `vision=VisionConfig(...)` in your test config entry.

### Projector dim mismatch

The projector output dim must equal the text decoder hidden size:
`MLPMultiModalProjector(vision_hidden_size=config.vision.hidden_size,
text_hidden_size=config.hidden_size)`.

---

## 12. Quick reference: file locations

| File | Purpose |
|------|---------|
| `src/mobius/models/llava.py` | Reference 3-split implementation |
| `src/mobius/models/gemma3.py` | Gemma3 (AvgPool projector, OffsetRMSNorm) |
| `src/mobius/models/internvl.py` | InternVL2 (fused QKV, pixel shuffle, CLS) |
| `src/mobius/models/mllama.py` | Mllama (cross-attention, dual KV cache) |
| `src/mobius/models/blip2.py` | BLIP-2 (Q-Former) |
| `src/mobius/tasks/_vision_language_3model.py` | All task variants |
| `src/mobius/components/_vision.py` | `VisionModel`, SigLIP encoder |
| `src/mobius/components/_multimodal.py` | Projectors, `InputMixer` |
| `src/mobius/_weight_utils.py` | `vlm_decoder_weights`, `vlm_embedding_weights` |
| `src/mobius/_configs.py` | `VisionConfig`, `VisionLanguageConfig`, `MllamaConfig` |
| `src/mobius/_registry.py` | All VL model registrations |
| `tests/_test_configs.py` | VL tiny test config entries |
