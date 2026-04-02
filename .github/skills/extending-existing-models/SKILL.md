---
name: extending-existing-models
description: >
  How to create VL or MoE variants of an existing text model in mobius.
  Covers the MoE variant pattern (subclassing decoder layers, adding
  num_dense_layers, weight renaming for router/expert projections) and the
  VL variant pattern (3-model split, VisionModel encoder, embedding scatter,
  HybridVisionLanguageTask). Uses LFM2 → lfm2_moe and lfm2_vl as the
  running example. Use this skill when adding a new model_type that is a
  variant of an existing architecture rather than a brand-new model family.
---

# Skill: Extending an Existing Model (MoE & VL Variants)

## When to use

Use this skill when you need to add a new `model_type` that is a **variant**
of an existing text model rather than a completely new architecture:

- A **MoE variant** replaces dense MLP layers with Mixture-of-Experts FFN
  (e.g. `lfm2_moe` extends `lfm2`)
- A **VL variant** adds a vision encoder and produces a 3-model ONNX split
  (e.g. `lfm2_vl` extends `lfm2`)

For a brand-new architecture see the `adding-a-new-model` skill instead.

## Decision: variant vs. new model from scratch

| Signal | Variant | New model |
|--------|---------|-----------|
| Same decoder layers, different FFN | ✅ MoE variant | |
| Same text backbone, adds vision | ✅ VL variant | |
| Completely new attention mechanism | | ✅ New model |
| Different layer topology | | ✅ New model |
| HF class name is `XxxForCausalLM` with `MoE`/`Vl` suffix | ✅ Variant | |

The golden rule: if you can **subclass** the decoder layer (or reuse the text
model as-is) rather than rewriting it, you are adding a variant.

---

## Part 1: MoE Variant

### Overview

A MoE variant replaces the dense `MLP` in some (or all) decoder layers with a
`MoELayer`. The first `num_dense_layers` layers keep the standard MLP; the
remaining layers switch to MoE.

**Example**: `lfm2_moe` (`LiquidAI/LFM2-8B-A1B`) extends `lfm2`.

### 1.1 Create the model file

Create `src/mobius/models/lfm2_moe.py`. Import the base decoder layers and
replace `feed_forward` with a `MoELayer`:

```python
from mobius.components._moe import MoELayer
from mobius.models.lfm2 import Lfm2ConvDecoderLayer, Lfm2AttentionDecoderLayer

class _Lfm2MoeConvDecoderLayer(nn.Module):
    """LFM2 ShortConv layer with MoE FFN."""

    def __init__(self, config: Lfm2MoeConfig):
        super().__init__()
        self.conv = ShortConv(...)
        self.operator_norm = RMSNorm(...)
        self.ffn_norm = RMSNorm(...)
        self.feed_forward = MoELayer(config, gate=_Lfm2MoeGate(config))  # MoE replaces MLP
```

The `forward()` method is **identical** to the base class except it calls
`self.feed_forward(op, hidden_states)` through `MoELayer` instead of `MLP`.

### 1.2 Custom routing gate

If the model uses a non-standard gate (e.g. optional per-expert bias), create
a custom gate class:

```python
class _Lfm2MoeGate(nn.Module):
    """TopK routing gate with optional per-expert bias."""

    def __init__(self, config: Lfm2MoeConfig):
        super().__init__()
        self.weight = nn.Parameter([config.num_local_experts, config.hidden_size])
        if config.use_expert_bias:
            self.e_score_correction_bias = nn.Parameter([config.num_local_experts])

    def forward(self, op, hidden_states):
        weight_t = op.Transpose(self.weight, perm=[1, 0])
        router_logits = op.MatMul(hidden_states, weight_t)
        if self.use_expert_bias:
            router_logits = op.Add(router_logits, self.e_score_correction_bias)
        k = op.Constant(value_ints=[self.top_k])
        routing_weights, selected_experts = op.TopK(router_logits, k, axis=-1, _outputs=2)
        routing_weights = op.Softmax(routing_weights, axis=-1)
        return routing_weights, selected_experts
```

If the model uses the standard `TopKGate` from `components/_moe.py`, use it
directly instead.

### 1.3 Build the text model with `num_dense_layers`

The key is per-layer type selection:

```python
class _Lfm2MoeTextModel(nn.Module):
    def __init__(self, config: Lfm2MoeConfig):
        super().__init__()
        layer_types = config.layer_types or []
        num_dense = config.num_dense_layers  # first N layers stay dense

        self.layers = nn.ModuleList([])
        for i in range(config.num_hidden_layers):
            ltype = layer_types[i] if i < len(layer_types) else "full_attention"
            use_moe = i >= num_dense  # switch to MoE after dense prefix
            if ltype == "conv":
                layer = _Lfm2MoeConvDecoderLayer(config) if use_moe else Lfm2ConvDecoderLayer(config)
            else:
                layer = _Lfm2MoeAttentionDecoderLayer(config) if use_moe else Lfm2AttentionDecoderLayer(config)
            self.layers.append(layer)
```

### 1.4 Top-level model class

```python
class Lfm2MoeCausalLMModel(nn.Module):
    default_task: str = "hybrid-text-generation"  # same task as base LFM2
    category: str = "Hybrid Conv+Attention"
    config_class: type = Lfm2MoeConfig

    def __init__(self, config: Lfm2MoeConfig):
        super().__init__()
        self.model = _Lfm2MoeTextModel(config)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)
```

### 1.5 Weight renaming for MoE layers

HuggingFace MoE checkpoints typically use short weight names (`w1`, `w2`, `w3`)
for expert MLPs, and `router` for the gate. Map them in `preprocess_weights`:

```python
def preprocess_weights(self, state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = _rename_lfm2_moe_weight(key)
        new_state_dict[new_key] = value
    return new_state_dict


def _rename_lfm2_moe_weight(key: str) -> str:
    m = _LAYER_RE.match(key)
    if m is None:
        return key
    idx, rest = m.group(1), m.group(2)

    # Base LFM2 renames (conv, attention, dense MLP)
    rest = rest.replace("conv.conv.weight", "conv.conv_weight")
    rest = rest.replace("self_attn.out_proj.", "self_attn.o_proj.")
    rest = rest.replace("feed_forward.w1.", "feed_forward.gate_proj.")
    rest = rest.replace("feed_forward.w3.", "feed_forward.up_proj.")
    rest = rest.replace("feed_forward.w2.", "feed_forward.down_proj.")

    # MoE gate: router → gate
    rest = rest.replace("feed_forward.router.", "feed_forward.gate.")

    # Expert MLP: w1→gate_proj, w3→up_proj, w2→down_proj
    rest = re.sub(r"feed_forward\.experts\.(\d+)\.w1\.", r"feed_forward.experts.\1.gate_proj.", rest)
    rest = re.sub(r"feed_forward\.experts\.(\d+)\.w3\.", r"feed_forward.experts.\1.up_proj.", rest)
    rest = re.sub(r"feed_forward\.experts\.(\d+)\.w2\.", r"feed_forward.experts.\1.down_proj.", rest)

    return f"model.layers.{idx}.{rest}"
```

### 1.6 Config class

Extend the base config by adding only MoE-specific fields:

```python
# In src/mobius/_configs.py

@dataclasses.dataclass
class Lfm2MoeConfig(Lfm2Config):
    """LFM2-MoE: Lfm2Config + MoE fields.

    MoE fields (from ArchitectureConfig base): num_local_experts,
    num_experts_per_tok, moe_intermediate_size, norm_topk_prob,
    routed_scaling_factor.
    """
    num_dense_layers: int = 2
    use_expert_bias: bool = True

    @classmethod
    def from_transformers(cls, config, parent_config=None) -> Lfm2MoeConfig:
        base = Lfm2Config.from_transformers(config, parent_config)
        base_fields = _shallow_fields(base)
        return cls(
            **base_fields,
            num_dense_layers=getattr(config, "num_dense_layers", 2),
            use_expert_bias=getattr(config, "use_expert_bias", True),
        )
```

The `ArchitectureConfig` base already has `num_local_experts`,
`num_experts_per_tok`, `moe_intermediate_size`, `norm_topk_prob`, and
`routed_scaling_factor` — you do not need to re-declare them.

### 1.7 Registration

```python
# In src/mobius/_registry.py  _create_default_registry()
from mobius.models.lfm2_moe import Lfm2MoeCausalLMModel
reg.register("lfm2_moe", Lfm2MoeCausalLMModel)
```

---

## Part 2: VL Variant

### Overview

A VL variant adds a vision encoder to an existing text model and produces
**three separate ONNX models** (vision, embedding, decoder). The text backbone
is reused verbatim — only the entry-point wrapper changes.

**Example**: `lfm2_vl` (`LiquidAI/LFM2-VL-450M`) extends `lfm2`.

### 2.1 Architecture layout

```
pixel_values ──► _Lfm2VlVisionModel ─────────────────────────────┐
                  (VisionModel encoder + MLP projector)            │
                                                        image_features
input_ids ──────► _Lfm2VlEmbedding ─────────────────────────────┘
                   (token lookup + image-token scatter)
                                  │
                         inputs_embeds
                                  │
                         _Lfm2VlDecoder ──► logits + hybrid KV cache
                          (LFM2 text backbone)
```

### 2.2 Vision encoder

Combine the reusable `VisionModel` component with a projector:

```python
class _Lfm2VlProjector(nn.Module):
    """Two-layer MLP: vision_hidden → projector_hidden → text_hidden."""

    def __init__(self, config: Lfm2VlConfig):
        super().__init__()
        vision_hidden = config.vision.hidden_size if config.vision else 1152
        self.linear_1 = Linear(vision_hidden, config.projector_hidden_size, bias=config.projector_bias)
        self.linear_2 = Linear(config.projector_hidden_size, config.hidden_size, bias=config.projector_bias)

    def forward(self, op, features):
        hidden = self.linear_1(op, features)
        hidden = op.Gelu(hidden)
        return self.linear_2(op, hidden)


class _Lfm2VlVisionModel(nn.Module):
    def __init__(self, config: Lfm2VlConfig):
        super().__init__()
        self.vision_tower = VisionModel(config)  # SigLIP2 encoder
        self.projector = _Lfm2VlProjector(config)

    def forward(self, op, pixel_values):
        vision_features = self.vision_tower(op, pixel_values)   # (batch, patches, vision_hidden)
        projected = self.projector(op, vision_features)          # (batch, patches, text_hidden)
        # Flatten batch*patches → num_image_tokens for the embedding model
        text_hidden_size = op.Shape(projected, start=2, end=3)
        flat_shape = op.Concat(op.Constant(value_ints=[-1]), text_hidden_size, axis=0)
        return op.Reshape(projected, flat_shape)                 # (num_image_tokens, text_hidden)

    def preprocess_weights(self, state_dict):
        # Keep only vision-side weights; decoder weights are routed separately
        return {
            key: value for key, value in state_dict.items()
            if key.startswith(("vision_tower.", "projector."))
        }
```

### 2.3 Embedding model

The embedding model handles image-token replacement using CumSum-based
indexing (no ScatterND required):

```python
class _Lfm2VlEmbedding(nn.Module):
    def __init__(self, config: Lfm2VlConfig):
        super().__init__()
        self.embed_tokens = Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.image_token_id = config.image_token_id

    def forward(self, op, input_ids, image_features):
        text_embeds = self.embed_tokens(op, input_ids)          # (batch, seq, text_hidden)

        image_mask = op.Equal(input_ids, op.Constant(value_int=self.image_token_id))
        image_mask_3d = op.Unsqueeze(image_mask, [-1])           # (batch, seq, 1)

        # CumSum gives a zero-based index into image_features for each image token
        mask_int = op.Cast(image_mask, to=7)                     # INT64
        cumsum = op.CumSum(mask_int, op.Constant(value_int=1))
        indices = op.Sub(cumsum, op.Constant(value_int=1))
        indices = op.Clip(indices, op.Constant(value_int=0))

        gathered = op.Gather(image_features, indices, axis=0)
        return op.Where(image_mask_3d, gathered, text_embeds)

    def preprocess_weights(self, state_dict):
        return {key: value for key, value in state_dict.items() if "embed_tokens" in key}
```

### 2.4 Decoder

Reuse the existing text model class (`_Lfm2TextModel`) directly — no changes
needed:

```python
class _Lfm2VlDecoder(nn.Module):
    def __init__(self, config: Lfm2VlConfig):
        super().__init__()
        self.model = _Lfm2TextModel(config)   # reuse base text model
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, op, inputs_embeds, attention_mask, position_ids, past_key_values=None):
        hidden_states, present_key_values = self.model(
            op,
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,   # VL path: skip token embedding
        )
        return self.lm_head(op, hidden_states), present_key_values

    def preprocess_weights(self, state_dict):
        return {_rename_lfm2_weight(key): value for key, value in state_dict.items()}
```

### 2.5 Top-level VL model class

The top-level class wires together the three sub-models and handles weight
routing from the HuggingFace checkpoint:

```python
class Lfm2VlModel(nn.Module):
    default_task: str = "hybrid-vision-language"
    category: str = "Multimodal"
    config_class: type = Lfm2VlConfig

    def __init__(self, config: Lfm2VlConfig):
        super().__init__()
        self.decoder = _Lfm2VlDecoder(config)
        self.vision_encoder = _Lfm2VlVisionModel(config)
        self.embedding = _Lfm2VlEmbedding(config)

    def forward(self, op, **kwargs):
        raise NotImplementedError(
            "Lfm2VlModel uses HybridVisionLanguageTask which calls each "
            "sub-module (decoder, vision_encoder, embedding) separately."
        )

    def preprocess_weights(self, state_dict):
        # Handle weight tying
        if self.config.tie_word_embeddings:
            lm_embed = "language_model.model.embed_tokens.weight"
            lm_head = "language_model.lm_head.weight"
            if lm_head not in state_dict and lm_embed in state_dict:
                state_dict[lm_head] = state_dict[lm_embed]

        renamed = {}
        for key, value in state_dict.items():
            if key.startswith("language_model."):
                inner = key[len("language_model."):]
                renamed_inner = _rename_lfm2_weight(inner)
                renamed[f"decoder.{renamed_inner}"] = value
                # Duplicate embed_tokens for the embedding sub-model
                if key == "language_model.model.embed_tokens.weight":
                    renamed["embedding.embed_tokens.weight"] = value
            elif key.startswith("vision_tower."):
                renamed[f"vision_encoder.{key}"] = value
            elif key.startswith("projector."):
                renamed[f"vision_encoder.{key}"] = value
            else:
                renamed[key] = value
        return renamed
```

**Key weight routing rules:**

| HF checkpoint prefix | ONNX initializer prefix |
|----------------------|------------------------|
| `language_model.*` | `decoder.*` |
| `language_model.model.embed_tokens.weight` | also `embedding.embed_tokens.weight` |
| `vision_tower.*` | `vision_encoder.vision_tower.*` |
| `projector.*` | `vision_encoder.projector.*` |

### 2.6 Choose the right task

| Task string | Class | When to use |
|-------------|-------|-------------|
| `"vision-language"` | `VisionLanguageTask` | Standard GQA decoder (Qwen2-VL, LLaVA, etc.) |
| `"hybrid-vision-language"` | `HybridVisionLanguageTask` | Hybrid conv+attn decoder (LFM2-VL, etc.) |
| `"mllama-vision-language"` | `MllamaVisionLanguageTask` | Mllama cross-attention decoder |

Use `"hybrid-vision-language"` when the decoder has a mixed conv+attention
cache (i.e. it builds on a base model registered under
`"hybrid-text-generation"`).

### 2.7 Config class for VL

Add only the vision-specific fields; all text fields come from the base config:

```python
@dataclasses.dataclass
class Lfm2VlConfig(Lfm2Config):
    image_token_id: int = 396
    projector_hidden_size: int = 2048
    projector_bias: bool = True
    downsample_factor: int = 2

    @classmethod
    def from_transformers(cls, config, parent_config=None) -> Lfm2VlConfig:
        base = Lfm2Config.from_transformers(config, parent_config)
        base_fields = _shallow_fields(base)
        vl_source = parent_config if parent_config is not None else config
        return cls(
            **base_fields,
            image_token_id=getattr(vl_source, "image_token_id", 396),
            projector_hidden_size=getattr(vl_source, "projector_hidden_size", 2048),
            projector_bias=getattr(vl_source, "projector_bias", True),
            downsample_factor=getattr(vl_source, "downsample_factor", 2),
        )
```

The HuggingFace VL config is typically nested: a top-level config wraps a
`text_config` for language fields and a `vision_config` for vision fields.
`ArchitectureConfig.from_transformers` auto-extracts these via
`_extract_vision_config` — pass `parent_config` to capture top-level VL fields
(like `image_token_id`) that aren't inside `text_config`.

### 2.8 Registration

```python
# In src/mobius/_registry.py  _create_default_registry()
from mobius.models.lfm2_vl import Lfm2VlModel
reg.register("lfm2_vl", Lfm2VlModel, task="hybrid-vision-language")
```

---

## Part 3: Testing

### 3.1 Build graph test config

Add a tiny config to `tests/_test_configs.py` so the graph can be built
without weights. Keep model dimensions very small (hidden=64, 2 layers):

**MoE variant** (`_lfm2_moe_config`):
```python
_lfm2_moe_config = Lfm2MoeConfig(
    hidden_size=64,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    intermediate_size=128,
    vocab_size=256,
    layer_types=["conv", "full_attention"],
    num_dense_layers=1,          # layer 0 dense, layer 1 MoE
    num_local_experts=4,
    num_experts_per_tok=2,
    moe_intermediate_size=64,
    norm_topk_prob=False,
    routed_scaling_factor=1.0,
    use_expert_bias=True,
    short_conv_kernel=4,
    short_conv_bias=False,
    rms_norm_eps=1e-5,
)
```

**VL variant** (`_lfm2_vl_config`):
```python
_lfm2_vl_config = Lfm2VlConfig(
    hidden_size=64,
    num_hidden_layers=2,
    ...
    image_token_id=3,
    projector_hidden_size=128,
    projector_bias=True,
    vision=VisionConfig(hidden_size=32, num_hidden_layers=1, ...),
)
```

Add test classes to `tests/build_graph_test.py`:

```python
class TestBuildLfm2MoeGraph:
    def test_build_lfm2_moe(self):
        _assert_build(Lfm2MoeCausalLMModel, _lfm2_moe_config)

class TestBuildLfm2VlGraph:
    def test_build_lfm2_vl_vision(self):
        _assert_build_vl(Lfm2VlModel, _lfm2_vl_config, "vision")

    def test_build_lfm2_vl_embedding(self):
        _assert_build_vl(Lfm2VlModel, _lfm2_vl_config, "embedding")

    def test_build_lfm2_vl_decoder(self):
        _assert_build_vl(Lfm2VlModel, _lfm2_vl_config, "model")
```

### 3.2 YAML golden test case

Add a YAML file in `testdata/cases/` for each new `model_type`. Set
`skip_reason` for models that are too large or require special hardware:

**MoE** (`testdata/cases/causal-lm/lfm2-moe-8b-a1b.yaml`):
```yaml
model_id: "LiquidAI/LFM2-8B-A1B"
revision: "main"
task_type: "text-generation"
dtype: "float32"

inputs:
  prompts:
    - "The quick brown fox"

level: "L4+L5"

generation:
  max_new_tokens: 20
  do_sample: false

skip_reason: "LFM2-MoE 8B — too large for CI golden data generation."
notes: "LFM2-MoE 8B×1B. Hybrid ShortConv+attention backbone with MoE FFN layers from Liquid AI."
```

**VL** (`testdata/cases/vision-language/lfm2-vl-450m.yaml`):
```yaml
model_id: "LiquidAI/LFM2-VL-450M"
revision: "main"
task_type: "image-text-to-text"
dtype: "float32"

inputs:
  prompts:
    - "Describe the image."
  images:
    - "pipeline-cat-chonk.jpeg"

level: "L4+L5"

generation:
  max_new_tokens: 30
  do_sample: false

skip_reason: "LFM2-VL 450M — vision-language model requires image input pipeline."
notes: "LFM2-VL 450M. SigLIP2 vision encoder + MLP projector + LFM2 hybrid decoder from Liquid AI."
```

### 3.3 Model coverage entries

**`src/mobius/_registry.py`** — add to `_TEST_MODEL_IDS` near other LFM2 entries:
```python
"lfm2_moe": "LiquidAI/LFM2-8B-A1B",
"lfm2_vl": "LiquidAI/LFM2-VL-450M",
```

**`tests/model_coverage_test.py`** — add to `_COVERAGE_SKIP`:
```python
# --- Very large models without small public checkpoints ---
"lfm2_moe": "MoE model (8B active 1B) — too large for CI golden data generation",

# --- Vision-language models (require image/video inputs) ---
"lfm2_vl": "VL model — requires image inputs",
```

---

## Summary checklist

For a **MoE variant**:
- [ ] `src/mobius/models/<name>_moe.py` with gate class, MoE decoder layers, text model, top-level model
- [ ] `src/mobius/_configs.py`: config class extending base config with MoE fields
- [ ] `src/mobius/models/__init__.py`: export new model class
- [ ] `src/mobius/_registry.py`: `reg.register("xxx_moe", XxxMoeCausalLMModel)` and `_TEST_MODEL_IDS` entry
- [ ] `tests/_test_configs.py`: tiny test config
- [ ] `tests/build_graph_test.py`: test class
- [ ] `testdata/cases/causal-lm/<name>-moe.yaml`: YAML golden case
- [ ] `tests/model_coverage_test.py`: `_COVERAGE_SKIP` entry (if too large for CI)

For a **VL variant**:
- [ ] `src/mobius/models/<name>_vl.py` with projector, vision model, embedding, decoder, top-level model
- [ ] `src/mobius/_configs.py`: config class extending base config with VL fields
- [ ] `src/mobius/models/__init__.py`: export new model class
- [ ] `src/mobius/_registry.py`: `reg.register("xxx_vl", XxxVlModel, task="vision-language")` and `_TEST_MODEL_IDS` entry
- [ ] `tests/_test_configs.py`: tiny VL test config (include `VisionConfig`)
- [ ] `tests/build_graph_test.py`: test class with vision/embedding/decoder sub-tests
- [ ] `testdata/cases/vision-language/<name>-vl.yaml`: YAML golden case
- [ ] `tests/model_coverage_test.py`: `_COVERAGE_SKIP` entry
