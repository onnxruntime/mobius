---
name: weight-name-alignment
description: >
  Use this skill when adding or modifying a model's preprocess_weights method
  to align ONNX parameter names with HuggingFace weight names. Covers
  nn.ModuleList for Sequential patterns, wrapper modules for nesting,
  placeholder modules, non-consecutive indices, and which rename categories
  cannot be eliminated. Reduces or eliminates weight name renames by
  structuring nn.Module attributes to match HuggingFace naming conventions.
---

# Skill: Weight Name Alignment

## When to use

Use this skill when:
- Adding a new model and designing `preprocess_weights`
- Simplifying an existing model's `preprocess_weights` method
- Debugging weight loading failures (mismatched parameter names)
- Deciding whether to restructure model construction vs. rename in
  `preprocess_weights`

## Core principle

**The best `preprocess_weights` is a no-op.** Most renames exist because the
ONNX module hierarchy doesn't match HuggingFace's. By restructuring
`nn.Module` construction to produce parameter names that match HF directly,
you can eliminate renames entirely.

## How parameter names are formed

In `onnxscript.nn`, parameter names are built from the Python attribute chain:

```python
class MyModel(nn.Module):
    def __init__(self):
        self.layers = nn.ModuleList([MyLayer()])
        # layers[0].weight → "layers.0.weight"
        
class MyLayer(nn.Module):
    def __init__(self):
        self.linear = _Linear(4, 4)
        # linear.weight → "linear.weight"
```

The full name is `"layers.0.linear.weight"`.

## Categories of renames

### ✅ Can be eliminated (restructure model construction)

#### 1. Sequential index patterns (nn.Sequential / nn.ModuleList)

**HF pattern:** `nn.Sequential(SiLU(), Linear(...))` → weights at `mod.1.weight`

**Problem:** Using a plain `_Linear(...)` produces `mod.weight` (no index).

**Preferred solution — `nn.Sequential`:**

`nn.Sequential` (from `onnxscript.nn`) registers children with numeric keys
like PyTorch's `nn.Sequential`, AND chains `forward()` calls automatically.
This gives both correct naming and clean call sites:

```python
from mobius.components import Linear, SiLU

# Produces "img_mod.1.weight" — matching HF
self.img_mod = nn.Sequential(SiLU(), Linear(dim, 6 * dim))

# Forward: output chains through each child automatically
result = self.img_mod(op, temb)
```

`nn.Sequential` subclasses `nn.ModuleList`. Key implementation detail: it
overrides `_set_name` to keep children with simple "0", "1" names (not
fully-qualified), because `__call__` already pushes the parent name onto the
scope stack. Without this override, children would be double-prefixed.

**Fallback — `nn.ModuleList` with manual indexing:**

If `nn.Sequential` is not yet available, use `nn.ModuleList` with explicit
`[i]` indexing:

```python
self.img_mod = nn.ModuleList([SiLU(), Linear(dim, 6 * dim)])

# Forward: manual chaining
result = self.img_mod[1](op, self.img_mod[0](op, temb))
```

This produces the same parameter names but requires manual forward logic.

#### 2. Non-consecutive indices with placeholder modules

**HF pattern:** `nn.Sequential(Linear, GELU, Linear)` → weights at `0.weight`
and `2.weight` (GELU at index 1 has no params).

**Problem:** `nn.ModuleList([linear1, linear2])` produces indices 0, 1.

**Solution:** Include activation modules to fill gaps:

```python
class _NoOpModule(nn.Module):
    """Placeholder for HF Dropout (no params, identity at inference)."""
    def forward(self, op, x):
        return x

class _GELUGate(nn.Module):
    """Matches HF GEGLU wrapper with .proj sub-attribute."""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.proj = _Linear(in_features, out_features)

# Matches HF: net.0.proj.weight, net.2.weight
self.net = nn.ModuleList([
    _GELUGate(dim, inner_dim * 2),  # index 0
    _NoOpModule(),                   # index 1 (Dropout placeholder)
    _Linear(inner_dim, dim),         # index 2
])
```

#### 3. Wrapper modules for extra nesting

**HF pattern:** `time_text_embed.timestep_embedder.linear_1.weight`

**Problem:** Flat structure produces `linear_1.weight` (missing prefix).

**Solution:** Create wrapper module matching HF nesting:

```python
class _TimestepMLP(nn.Module):
    def __init__(self, in_channels, time_embed_dim):
        super().__init__()
        self.linear_1 = _Linear(in_channels, time_embed_dim)
        self.linear_2 = _Linear(time_embed_dim, time_embed_dim)

class _TimestepEmbedding(nn.Module):
    def __init__(self, in_channels, time_embed_dim):
        super().__init__()
        self.timestep_embedder = _TimestepMLP(in_channels, time_embed_dim)
```

#### 4. Bare Parameter → Module wrapper

**HF pattern:** `txt_norm.weight` (from `RMSNorm` module)

**Problem:** Using `nn.Parameter` produces `txt_norm` (no `.weight` suffix).

**Solution:** Use a proper module:

```python
# BAD — produces "txt_norm" as a bare parameter name
self.txt_norm = nn.Parameter((dim,))

# GOOD — produces "txt_norm.weight"
class _RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter((dim,))
        self._eps = eps
    def forward(self, op, x):
        return op.RMSNormalization(x, self.weight, epsilon=self._eps)

self.txt_norm = _RMSNorm(dim)
```

#### 5. Inner model wrapper for prefix nesting

**HF pattern:** `model.layers.0.self_attn.q_proj.weight`

**Problem:** Without a `model` wrapper, you get `layers.0.self_attn...`.

**Solution:** Create inner model class:

```python
class _TextModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList([...])
        self.norm = _RMSNorm(config.hidden_size)

class MyCausalLMModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = _TextModel(config)  # Creates "model." prefix
        self.lm_head = _Linear(config.hidden_size, config.vocab_size)
```

### ❌ Cannot be eliminated (must stay in preprocess_weights)

#### 1. QKV splitting

HuggingFace fuses Q, K, V into a single tensor (`query_key_value`,
`c_attn`, `qkv_proj`), but ONNX uses separate `q_proj`, `k_proj`, `v_proj`.

```python
def preprocess_weights(self, state_dict):
    new_state = {}
    for key, tensor in state_dict.items():
        if "query_key_value" in key:
            q, k, v = self._split_qkv(tensor, self.config)
            new_state[key.replace("query_key_value", "q_proj")] = q
            new_state[key.replace("query_key_value", "k_proj")] = k
            new_state[key.replace("query_key_value", "v_proj")] = v
        else:
            new_state[key] = tensor
    return new_state
```

**Models affected:** GPT-2, Falcon, InternLM2, ChatGLM, Phi3/Phi3Small

#### 2. Conv1D → Linear transpose

GPT-2 uses Conv1D `[in, out]` layout; ONNX Linear needs `[out, in]`.

```python
if key.endswith(".weight") and tensor.ndim == 2:
    tensor = tensor.t()
```

**Models affected:** GPT-2

#### 3. Deep structural naming differences

BERT, T5, BART have deeply different naming conventions that would require
rewriting fundamental component classes to match.

```python
# BERT: "encoder.layer.0.attention.self.query.weight"
# Ours: "encoder.layer.0.self_attn.q_proj.weight"
```

Changing this would require BERT-specific Attention, MLP components — not
worth the complexity for a simple rename.

**Models affected:** BERT, DistilBERT, RoBERTa, ALBERT, T5, BART, mBART,
Marian, CLIP, SigLIP

#### 4. MoE expert weight remapping

MoE models have mixed naming across architectures (Mixtral: `w1/w2/w3`,
Qwen2-MoE: `gate_proj/up_proj/down_proj`). The current `_rename_moe_expert_weights`
handles both conventions optimally.

#### 5. Weight tying

Always needed when `tie_word_embeddings=True`:

```python
if self.config.tie_word_embeddings:
    if "lm_head.weight" in state_dict:
        state_dict["model.embed_tokens.weight"] = state_dict["lm_head.weight"]
    elif "model.embed_tokens.weight" in state_dict:
        state_dict["lm_head.weight"] = state_dict["model.embed_tokens.weight"]
```

#### 6. Weight deletion

Some HF weights are not needed (e.g. `rotary_emb.inv_freq` — RoPE
frequencies are computed at runtime).

## ⚠️ Scope mechanism pitfalls

Understanding how `onnxscript.nn` builds parameter names is critical.  Names
come from the **call-stack of `__call__` invocations**, not from the Python
attribute chain used to *reach* a module.

### Rule: only `__call__` pushes scope

When you call `module(op, x)`, the base `nn.Module.__call__` method:
1. Pushes `module._name` onto the scope stack
2. Calls `module.forward(op, x)`
3. Pops the scope

**Accessing a child and calling its method directly bypasses the parent scope:**

```python
# ❌ BAD — self.shared.child(op, x) only pushes "child", NOT "shared"
result = self.shared.child(op, x)
# Parameter name: "child.weight" (missing "shared." prefix!)

# ✅ GOOD — call shared's forward, which internally calls child
result = self.shared(op, x)
# Parameter name: "shared.child.weight" ✓
```

### ModuleList indexing requires `__call__`

The same rule applies to `nn.ModuleList`.  `self.items[i]` returns the
module object — its `_name` is `"items.{i}"` — but that scope is only
pushed when you call it:

```python
# ❌ BAD — sub_module's scope is pushed, but items.0 scope is NOT
result = self.items[0].sub_module(op, x)
# Parameter name: "sub_module.weight" (missing "items.0." prefix!)

# ✅ GOOD — items[0].__call__ pushes "items.0", then forward calls sub_module
result = self.items[0](op, x)
# Parameter name: "items.0.sub_module.weight" ✓
```

**Key takeaway:** If you need per-index distinct parameter names (e.g.
per-layer adapters), you **must** call the ModuleList element via
`self.items[idx](op, ...)`, not reach into its sub-attributes.

### Shared weights via single module instance

When multiple layers reuse the same weights (e.g. Zamba2's shared
transformer), register ONE module instance.  Calling it multiple times
produces the same initializer names — ONNX uses a single initializer:

```python
class _TextModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        # ONE shared module → ONE set of initializers
        self.shared_transformer = SharedLayer(config)

    def forward(self, op, x):
        for i in range(num_uses):
            # Same scope "shared_transformer.*" each time → same initializer
            x = self.shared_transformer(op, x)
```

### Circular dependency: shared weights + per-instance data

When shared weights and per-instance data (e.g. adapters) must interact in
the same computation, the scope model creates a tension:

- **Shared weights** must be inside a shared module (for correct scope)
- **Per-instance data** must be outside (different scope per use)

**Solution: split the computation.** Have the shared module return an
intermediate value.  The caller computes per-instance contributions at its
scope, then continues the computation with shared weights at its level:

```python
class _TextModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.shared_attn = SharedAttention(config)   # shared weights
        self.adapters = nn.ModuleList([...])          # per-layer
        self.gate_proj = Linear(...)                  # shared MLP at model scope

    def forward(self, op, hidden):
        for idx in range(num_layers):
            # Phase 1: shared attention (inside shared module scope)
            intermediate = self.shared_attn(op, hidden)
            # Phase 2: per-layer adapter (at "adapters.{idx}" scope)
            adapter_out = self.adapters[idx](op, intermediate)
            # Phase 3: shared MLP at model scope (gate_proj is model attr)
            hidden = self.gate_proj(op, op.Add(intermediate, adapter_out))
```

**Reference implementation:** `models/zamba2.py` — Zamba2 hybrid model with
shared transformer + per-layer Q/K/V/MLP low-rank adapters.

## How to analyze a model's preprocess_weights

1. **Compare HF names to ONNX names:**
   ```python
   # Print ONNX parameter names
   module = MyModel(config)
   for name, _ in module.named_parameters():
       print(name)
   
   # Print HF weight names (from safetensors)
   from safetensors import safe_open
   with safe_open("model.safetensors", framework="pt") as f:
       for key in f.keys():
           print(key)
   ```

2. **Categorize each rename** as one of the types above.

3. **For eliminable renames**, restructure the module constructor.

4. **For non-eliminable renames**, keep them in `preprocess_weights`.

## Non-consecutive index patterns (setattr fallback)

When HF uses `nn.Sequential` with non-consecutive parameter indices AND the
gap modules have no natural implementation:

```python
# HF: Sequential(Conv2d, SiLU, Conv2d, SiLU, Conv2d, SiLU, Conv2d)
# Weights at indices: 0, 2, 4, 6 (SiLU at 1, 3, 5 has no params)
# But we process differently in forward, so ModuleList doesn't work

from mobius.components import Conv2d

# Fallback: manual setattr
class _SequentialConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        conv = Conv2d(in_channels, out_channels, **kwargs)
        setattr(self, "1", conv)  # Matches HF Sequential index
```

Use this only when `nn.ModuleList` with activation placeholders doesn't
work (e.g., different forward logic, or HF Sequential wraps padding + conv).

## Reference implementations

| Pattern | Model | File |
|---------|-------|------|
| No-op (fully aligned) | QwenImage transformer | `models/qwen_image.py` |
| Weight tying only | CausalLMModel (base) | `models/base.py` |
| Sequential index (ModuleList) | UNet, DiT, VAE | `models/unet.py`, `models/dit.py`, `models/vae.py` |
| Wrapper + placeholder modules | QwenImage (all patterns) | `models/qwen_image.py` |
| QKV splitting | Falcon, GPT-2 | `models/falcon.py`, `models/gpt2.py` |
| Conv1D transpose | GPT-2 | `models/gpt2.py` |
| MoE expert remapping | MoE models | `models/moe.py` |
| Deep structural renames | BERT, T5 | `models/bert.py`, `models/t5.py` |
| Shared weights + per-layer adapters | Zamba2 | `models/zamba2.py` |
| Scope-aware ModuleList adapters | Zamba2 | `models/zamba2.py` |
