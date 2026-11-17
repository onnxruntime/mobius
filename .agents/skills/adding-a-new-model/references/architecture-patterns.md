# Architecture Patterns Reference

Detailed code templates, compatibility rules, and advanced patterns for
non-standard model architectures. Read this when implementing a model that
is **not** a standard decoder-only causal LM.

## Non-LLM model type table

For models that aren't causal LMs, use the appropriate base class and task:

| Model type | Base class / pattern | Task | Config |
|------------|---------------------|------|--------|
| Encoder-only (BERT-like) | `BertModel` | `feature-extraction` | `ArchitectureConfig` |
| Encoder-only (ModernBERT) | `ModernBertModel` | `feature-extraction` | `ArchitectureConfig` |
| Encoder-decoder (BART/T5-like) | `BartForConditionalGeneration` or `T5ForConditionalGeneration` | `seq2seq` | `ArchitectureConfig` |
| Vision (ViT-like) | `ViTModel` or `CLIPVisionModel` | `image-classification` | `ArchitectureConfig` |
| Object detection | `YolosForObjectDetection` | `object-detection` | `ArchitectureConfig` |
| Depth estimation | `DepthAnythingForDepthEstimation` | `image-classification` | `ArchitectureConfig` |
| Segmentation | `SegformerForSemanticSegmentation` or `Sam2VisionModel` | `image-classification` | `ArchitectureConfig` |
| Audio encoder (Wav2Vec2-like) | `Wav2Vec2Model` | `audio-feature-extraction` | `ArchitectureConfig` |
| Multimodal (LLaVA-like) | `LLaVAModel` | `vision-language` | `ArchitectureConfig` |
| Document AI | `LayoutLMv3Model` | `feature-extraction` | `ArchitectureConfig` |
| OCR decoder | `TrOCRForConditionalGeneration` | `seq2seq` | `ArchitectureConfig` |
| Diffusion denoiser | Custom (`UNet2DConditionModel`, etc.) | `denoising` | Custom config (e.g. `UNet2DConfig`) |
| VAE | `AutoencoderKLModel` | `vae` | `VAEConfig` |
| Adapter | `T2IAdapterModel` / `IPAdapterModel` | `adapter` | Custom config |

Many new models can be registered as aliases of existing classes (e.g.
`reg.register("my_bert_variant", BertModel)`) if the architecture matches.

## False Compatibility Pitfalls

When registering models as aliases of existing base classes, **tests passing
does not mean the mapping is correct.** Graph-build tests only check that an
ONNX graph can be constructed — they do NOT verify that the graph matches
the model's actual computation.

### Safe approximate mappings

The project accepts "approximate" registry aliases when the model uses
similar-but-not-identical attention. These produce structurally correct ONNX
graphs; weight-loading may need minor adjustments:

| Model | Maps to | Why it works |
|-------|---------|-------------|
| DeBERTa | `BertModel` | Disentangled attention is a variant of standard attention |
| Swin | `ViTModel` | Shifted window attention is still self-attention over patches |
| SqueezeBERT | `BertModel` | Grouped convolution replaces dense attention, but same I/O shape |

### NEVER safe as registry aliases

These model families have fundamentally different computation that **cannot**
be represented by standard base classes, even though the L1 graph suite passes:

| Category | Models | Why it fails |
|----------|--------|-------------|
| Pure CNNs | ConvNeXt, ResNet, MobileNet, EfficientNet, RegNet | No attention at all — base ViT/BERT classes produce attention-based graphs |
| Spatial pooling | PoolFormer | Uses spatial average pooling instead of attention — structurally incompatible |
| SSM / state-space models | Mamba, Mamba2, FalconMamba, RWKV, RecurrentGemma | Sequential scan / linear recurrence, not attention |
| Fundamentally different attention | Longformer (sparse), BigBird (block sparse), Funnel (downsampling) | Attention pattern differs from dense self-attention at a structural level |
| Custom tokenization | CANINE (character-level) | Byte-level input, hash embeddings — not a standard vocab embedding |

**Rule of thumb:** If the HuggingFace model's `forward()` method doesn't call
`self_attn(query, key, value)` in a standard way, it is NOT a safe alias.

### Future work

CI currently only runs graph-build tests (shape inference, op validity). To
catch false compatibility in approximate mappings, we need **weight-loading
tests** that:
1. Load real HuggingFace weights into the ONNX graph
2. Run inference on a test input
3. Compare output against HuggingFace PyTorch output
4. Fail if max abs diff exceeds a threshold (e.g. 0.01)

This would catch shape mismatches, wrong norm types, and missing scaling
factors that graph-build tests cannot detect.

## KV sharing across layers (num_kv_shared_layers)

Some models (e.g. Gemma 4) reduce parameter count by having the last N
decoder layers **borrow** Key and Value states from an earlier "source" layer
of the same type instead of projecting their own K,V. This is controlled by
`num_kv_shared_layers` in the HuggingFace config.

### What it means

```
first_kv_shared_idx = num_hidden_layers - num_kv_shared_layers

Layers [0 .. first_kv_shared_idx - 1]:  normal — own k_proj, v_proj, k_norm
Layers [first_kv_shared_idx .. end]:     shared — NO k_proj/v_proj weights
```

Each shared layer reuses K,V from the **last non-shared layer of the same
attention type** (e.g. sliding vs. full attention). Only Q is computed fresh.

### Impact on the checkpoint

Shared layers have **no `k_proj`, `v_proj`, `k_norm`** keys in the
HuggingFace checkpoint. `preprocess_weights` must not assert these keys
exist for shared-layer indices — they simply won't be present.

```python
def preprocess_weights(self, state_dict):
    # shared layers have no k/v proj — remove them silently if accidentally present
    first_shared = self.config.num_hidden_layers - self.config.num_kv_shared_layers
    for i in range(first_shared, self.config.num_hidden_layers):
        for suffix in ("k_proj.weight", "v_proj.weight", "k_norm.weight"):
            state_dict.pop(f"model.layers.{i}.self_attn.{suffix}", None)
    return super().preprocess_weights(state_dict)
```

### Attention module: is_kv_shared_layer flag

The attention class detects at `__init__` time whether it is a shared layer:

```python
class Gemma4Attention(nn.Module):
    def __init__(self, config, layer_idx, layer_types, first_kv_shared_idx, ...):
        self.is_kv_shared_layer = layer_idx >= first_kv_shared_idx > 0
        prev_layers = layer_types[:first_kv_shared_idx]

        if self.is_kv_shared_layer:
            # Index of the source layer whose K,V this layer borrows
            self.kv_shared_layer_index = (
                len(prev_layers) - 1 - prev_layers[::-1].index(layer_types[layer_idx])
            )
            self.store_full_length_kv = False
        else:
            self.kv_shared_layer_index = None
            # True for the last non-shared layer of each type that has downstream
            # KV-shared layers depending on it — it stores K,V for reuse.
            self.store_full_length_kv = first_kv_shared_idx > 0 and (
                layer_idx
                == len(prev_layers) - 1 - prev_layers[::-1].index(layer_types[layer_idx])
            )

        # All layers have Q projection
        self.q_proj = Linear(config.hidden_size, num_heads * head_dim)
        self.q_norm = RMSNorm(head_dim)
        self.o_proj = Linear(num_heads * head_dim, config.hidden_size)

        # Only non-shared layers have K/V projections
        if not self.is_kv_shared_layer:
            self.k_proj = Linear(config.hidden_size, num_kv_heads * head_dim)
            self.v_proj = Linear(config.hidden_size, num_kv_heads * head_dim)
            self.k_norm = RMSNorm(head_dim)
```

### forward(): shared layers consume shared_kv_states dict

Pass a mutable `shared_kv_states` dict through the forward call. Source
layers populate it; shared layers read from it:

```python
def forward(self, op, hidden_states, ..., shared_kv_states, past_key_value):
    # Q projection (all layers)
    query_states = self.q_proj(op, hidden_states)
    ...

    if self.is_kv_shared_layer:
        # Borrow K,V from source layer (already in shared_kv_states)
        src_key, src_value = shared_kv_states[self.kv_shared_layer_index]
        # Reshape from present_kv 4D [B, kv_heads, total_seq, head_dim]
        # to Attention input 3D [B, total_seq, kv_heads * head_dim]
        src_key = op.Transpose(src_key, perm=[0, 2, 1, 3])
        key_states = op.Reshape(src_key, ...)
        value_states = ...
    else:
        # Normal K/V projection + norm
        key_states = self.k_proj(op, hidden_states)
        value_states = self.v_proj(op, hidden_states)
        ...

    hidden_out, present_kv = _apply_attention(op, query_states, key_states, ...)

    if self.store_full_length_kv:
        # Store present_kv [B, kv_heads, total_seq, head_dim] for downstream shared layers
        shared_kv_states[self.layer_idx] = (present_kv_key, present_kv_value)

    return hidden_out, present_kv
```

### Text model: KV cache has only num_kv_layers entries

KV-shared layers do **not** append to `present_key_values`. The output list
has `num_hidden_layers - num_kv_shared_layers` entries, not `num_hidden_layers`:

```python
# In Gemma4TextModel.forward():
shared_kv_states: dict = {}
present_key_values = []

# past_key_values has only num_kv_layers entries (no entry for KV-shared layers).
# Expand it to a full per-layer list so we can zip cleanly over all layers.
if past_key_values is not None:
    kv_iter = iter(past_key_values)
    past_kvs: list = [
        None if layer.self_attn.is_kv_shared_layer else next(kv_iter)
        for layer in self.layers
    ]
else:
    past_kvs = [None] * len(self.layers)

for i, (layer, layer_type, past_kv) in enumerate(
    zip(self.layers, self.layer_types, past_kvs)
):
    hidden_states, present_kv = layer(
        op,
        hidden_states=hidden_states,
        attention_bias=attention_bias_dict[layer_type],
        position_embeddings=position_embeddings_dict[layer_type],
        shared_kv_states=shared_kv_states,
        past_key_value=past_kv,
    )
    # KV-shared layers borrow K,V — exclude from present_key_values so the
    # output has exactly num_kv_layers (not num_hidden_layers) entries.
    if not layer.self_attn.is_kv_shared_layer:
        present_key_values.append(present_kv)
```

The task's KV cache inputs/outputs must use the correct count:
`num_kv_layers = config.num_hidden_layers - config.num_kv_shared_layers`.

## Reference implementations

| Model | File | Key differences from base |
|-------|------|--------------------------|
| Granite | `models/granite.py` | 4 scaling multipliers, custom attention scale |
| OLMo-1B | `models/olmo.py` | Weight-free LayerNorm (not RMSNorm), eps=1e-5 |
| OLMo-2 | `models/olmo.py` | Post-norm decoder layers, QK full norm |
| Gemma | `models/gemma.py` | RMSNorm weight+1, embedding scaling |
| Whisper | `components/_whisper.py` | Q pre-scaling, LayerNorm eps=1e-5, is_causal attr |
| Phi3.5 | `components/_rotary_embedding.py` | LongRope with float32 factors |
| Qwen3.5 | `models/qwen.py` | Hybrid DeltaNet + full attention, gated GQA, OffsetRMSNorm, interleaved MRoPE |
| Qwen3.5-MoE | `models/qwen.py` | Same hybrid attention + MoE FFN with shared expert (sigmoid gate) |
| Qwen3-TTS | `models/qwen3_tts.py` | 4-model TTS split, 2-token code predictor prefill, small_to_mtp projection, Identity-exposed weights |
| **BLIP** | `models/blip.py` | Subclass of ViTModel — only `preprocess_weights` (fused QKV split, renaming) |
| **YOLOS** | `models/yolos.py` | ViT + detection tokens + DETR-style MLP heads. New `object-detection` task |
| **Depth Anything** | `models/depth_anything.py` | ViT backbone + DPT decoder (reassemble + fusion + depth head). Uses `ConvTranspose2d` |
| **Segformer** | `models/segformer.py` | Hierarchical 4-stage encoder, efficient attention (strided Conv2d on K/V), Mix-FFN with depthwise conv |
| **SAM2** | `models/sam2.py` | Hiera backbone (per-stage dim transitions, fused QKV attention) + FPN neck with top-down fusion |
| **LayoutLMv3** | `models/layoutlmv3.py` | Subclass of BertModel — only `preprocess_weights` (spatial embedding filtering) |
| **TrOCR** | `models/trocr.py` | Subclass of BartForConditionalGeneration — only `preprocess_weights` (`output_projection` rename) |
| **ModernBERT** | `models/modernbert.py` | Pre-norm encoder with RoPE + GeGLU + bidirectional attention. Fused QKV/Wi splitting. Both encoder and decoder variants |
| **Gemma3n** | `models/gemma3n.py` | AltUp predict/correct, Laurel low-rank, per-layer input gating, hybrid local/global attention |
| **Mllama** | `models/mllama.py` | Interleaved cross-attention decoder, tanh-gated residual, manual QK-norm |

## Reference examples by complexity

When adding a new model, use these files as canonical references:

| Complexity | File | Why |
|---|---|---|
| **Minimal** — base class works, only weight mapping needed | `models/phi3.py` (38 lines) | Extends `CausalLMModel`, only overrides `preprocess_weights()` to split fused QKV and gate-up projections. Shows the simplest possible model addition. |
| **Minimal** — encoder subclass | `models/layoutlmv3.py` | Extends `BertModel`, only overrides `preprocess_weights()`. Same pattern for encoder-only models. |
| **Moderate** — custom components | `models/gemma.py` | Adds custom attention (soft-capping), custom MLP (GeGLU), and custom normalization. Good example of component subclassing. |
| **Complex** — multi-model architecture | `models/qwen3_tts.py` | 4-model TTS split with talker, code predictor, embedding, and speaker encoder sub-modules. Shows how to structure multi-model architectures. |
