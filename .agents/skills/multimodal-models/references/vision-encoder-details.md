# Vision Encoder Details

## VisionModel / VisionEncoder construction

The standard vision encoder (`components/_vision.py`) follows a SigLIP-style
architecture:

| Component | File | Purpose |
|-----------|------|---------|
| `VisionModel` | `components/_vision.py` | SigLIP-style patch embedding + transformer encoder |
| `PixtralVisionTower` | `components/_pixtral_vision.py` | Pixtral 2D RoPE vision encoder (bidirectional attention) |
| `PatchEmbedding` | `components/_vision.py` | Conv2d → positional embedding |

### PatchEmbedding naming

`PatchEmbedding` has three parameters.  These use explicit `name=` because the
attribute names don't match the desired ONNX names (e.g. `patch_embedding`
needs to map to `patch_embedding.weight`):

```python
self.patch_embedding = nn.Parameter([...], name="patch_embedding.weight")
self.patch_embedding_bias = nn.Parameter([...], name="patch_embedding.bias")
self.position_embedding = nn.Parameter([...], name="position_embedding.weight")
```

In most cases, `name=` is **not needed** because `nn.Module.__setattr__`
automatically sets the parameter name from the attribute name.  Only use
`name=` when the attribute name differs from the desired ONNX initializer name.

## Step-by-step: adding a new multimodal model

### 1. Identify the projector architecture

Look at the HuggingFace source in `modeling_<model>.py`:

```bash
grep -n "class.*Projector\|class.*projector" \
    transformers/models/<model>/modeling_<model>.py
```

Match it to one of the projector variants, or create a new one.

### 2. Extract vision config

Multimodal HF configs have a `vision_config` sub-object.  Extract vision
fields in the test or integration code:

```python
hf_config = transformers.AutoConfig.from_pretrained(model_id)
text_config = hf_config.text_config
vision_config = hf_config.vision_config

config = ArchitectureConfig.from_transformers(text_config)
# Add vision fields
config.vision_hidden_size = vision_config.hidden_size
config.vision_intermediate_size = vision_config.intermediate_size
config.vision_num_hidden_layers = vision_config.num_hidden_layers
config.vision_num_attention_heads = vision_config.num_attention_heads
config.vision_image_size = vision_config.image_size
config.vision_patch_size = vision_config.patch_size
config.vision_norm_eps = getattr(vision_config, "layer_norm_eps", 1e-6)
config.mm_tokens_per_image = getattr(hf_config, "mm_tokens_per_image", None)
config.image_token_id = getattr(hf_config, "image_token_id", None)
```

### 3. Create the model class

**Important:** Always invoke child modules through `__call__` (not by
accessing their sub-modules directly) so that `onnxscript.nn.Module` pushes
the correct naming context.  Direct access like
`self.language_model.model.embed_tokens(op, x)` skips intermediate naming
scopes and produces wrong initializer names.

The recommended pattern is to pass vision embeddings as a kwarg through the
`__call__` chain, and have the text model perform the mixing internally:

```python
class _MyTextModelForMultimodal(MyTextModel):
    """Text model that mixes vision embeddings into the input."""

    def __init__(self, config):
        super().__init__(config)
        self.input_mixer = InputMixer(image_token_id=config.image_token_id or 0)

    def forward(self, op, input_ids, attention_mask, position_ids,
                past_key_values=None, vision_embeddings=None):
        hidden_states = self.embed_tokens(op, input_ids)
        if vision_embeddings is not None:
            hidden_states = self.input_mixer(
                op, hidden_states, vision_embeddings, input_ids
            )
        return super().forward(
            op, input_ids, attention_mask, position_ids,
            past_key_values=past_key_values, inputs_embeds=hidden_states,
        )


class _MyForMultimodalLM(MyCausalLMModel):
    """CausalLM that passes vision_embeddings to the text model."""

    def __init__(self, config):
        nn.Module.__init__(self)
        self.config = config
        self.model = _MyTextModelForMultimodal(config)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)


class MyMultiModalModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vision_tower = VisionModel(config)
        self.multi_modal_projector = MLPMultiModalProjector(
            vision_hidden_size=config.vision_hidden_size,
            text_hidden_size=config.hidden_size,
        )
        self.language_model = _MyForMultimodalLM(config)

    def forward(self, op, input_ids, attention_mask, position_ids, pixel_values,
                past_key_values=None):
        # 1. Encode vision
        vision_features = self.vision_tower(op, pixel_values)
        vision_embeddings = self.multi_modal_projector(op, vision_features)

        # 2. Pass through __call__ chain — naming is correct automatically
        return self.language_model(
            op, input_ids, attention_mask, position_ids,
            past_key_values=past_key_values,
            vision_embeddings=vision_embeddings,
        )
```

See `models/gemma3.py` for the full working example.

### 4. Handle weight name mismatches

Multimodal HF models often prefix text weights differently:

| HF key | Our key |
|--------|---------|
| `language_model.model.layers.0.…` | `layers.0.…` |
| `vision_tower.vision_model.encoder.…` | `vision_tower.encoder.…` |
| `multi_modal_projector.mm_input_projection_weight` | `multi_modal_projector.weight` |

Implement `preprocess_weights` to strip prefixes and rename keys.

### 5. Handle weight tying

If `tie_word_embeddings=True`, the HF checkpoint may not include
`lm_head.weight`.  Copy it from `embed_tokens.weight`:

```python
if self.config.tie_word_embeddings:
    if "lm_head.weight" not in renamed and "embed_tokens.weight" in renamed:
        renamed["lm_head.weight"] = renamed["embed_tokens.weight"]
```

### 6. Use VisionLanguageTask

Build with the `VisionLanguageTask` to add `pixel_values` to graph inputs:

```python
from mobius.tasks import VisionLanguageTask

onnx_model = build_from_module(module, config, task=VisionLanguageTask())
```
