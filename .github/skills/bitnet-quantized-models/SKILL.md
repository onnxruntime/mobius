---
name: bitnet-quantized-models
description: >
  How to add quantized or low-bit weight models (BitNet ternary, future
  2-bit/1-bit architectures) to mobius. Covers the linear_class injection
  pattern, weight packing/unpacking conventions, sub-layer norm patterns,
  and preprocess_weights for custom quantization formats. Use this skill
  when adding a model with non-standard weight representations.
---

# Adding Quantized / Low-Bit Weight Models to Mobius

## When to use

- Adding a model with custom weight quantization (ternary, binary, 2-bit)
- Understanding how `linear_class` injection works for quantized layers
- Writing `preprocess_weights()` for packed or non-standard weight formats
- Adding models that modify standard components (sub-layer norms, custom
  activations) while keeping the Llama-like backbone

## Key Architecture Pattern: `linear_class` Injection

Mobius uses a **dependency injection** pattern for linear layers. The
`linear_class` parameter flows through the component hierarchy:

```
TextModel.__init__(config)
  → detects config.quantization
  → creates linear_class = make_quantized_linear_factory(bits=..., ...)
  → passes to DecoderLayer(config, linear_class=linear_class)
    → passes to Attention(config, linear_class=linear_class)
      → q_proj = linear_class(hidden, heads*dim, bias=...)
      → k_proj, v_proj, o_proj = ...
    → passes to MLP(config, linear_class=linear_class)
      → gate_proj, up_proj, down_proj = ...
```

**Key files:**
- `src/mobius/components/_common.py` — `Linear` (standard)
- `src/mobius/components/_quantized_linear.py` — `QuantizedLinear`
  (MatMulNBits for 4/8-bit), `make_quantized_linear_factory()`
- `src/mobius/models/base.py` — `TextModel.__init__()` does the detection
- `src/mobius/_configs.py` — `QuantizationConfig.from_transformers()`

### When to use `linear_class` vs standard Linear

| Scenario | Approach |
|----------|----------|
| GPTQ/AWQ 4-bit | `QuantizedLinear` via `linear_class` injection |
| BitNet ternary (initial) | Standard `Linear`, unpack weights in `preprocess_weights` |
| BitNet ternary (optimized) | Future: `QuantizedLinear` with `bits=2` |
| Standard fp16/bf16 | Default `Linear`, no changes needed |

## BitNet Implementation Pattern

BitNet b1.58 is the reference implementation for adding a model with
non-standard weight representations. Key decisions:

### 1. Architecture changes as subclasses

BitNet differs from Llama in three ways:
- **Sub-layer RMSNorms**: `attn_sub_norm` before o_proj, `ffn_sub_norm`
  before down_proj
- **Activation**: `relu2` (squared ReLU) instead of SiLU
- **Weight format**: ternary {-1, 0, +1} packed as uint8

Each difference maps to a clean subclass:

```python
class BitNetAttention(Attention):
    """Adds attn_sub_norm before o_proj."""
    def __init__(self, config):
        super().__init__(config)
        self.attn_sub_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, op, hidden_states, ...):
        # ... standard Q/K/V + RoPE + attention ...
        attn_output = self.attn_sub_norm(op, attn_output)  # NEW
        attn_output = self.o_proj(op, attn_output)
        return attn_output, (present_key, present_value)

class BitNetMLP(MLP):
    """Adds ffn_sub_norm before down_proj."""
    def __init__(self, config, linear_class=None):
        super().__init__(config, linear_class=linear_class)
        self.ffn_sub_norm = RMSNorm(config.intermediate_size, eps=config.rms_norm_eps)

    def forward(self, op, x):
        hidden = op.Mul(self.act_fn(op, self.gate_proj(op, x)), self.up_proj(op, x))
        hidden = self.ffn_sub_norm(op, hidden)  # NEW
        return self.down_proj(op, hidden)
```

### 2. Replace layers in `__init__`, not custom TextModel

The simplest pattern for models that only change Attention/MLP is to
build the standard `CausalLMModel`, then replace components:

```python
class BitNetCausalLMModel(CausalLMModel):
    def __init__(self, config):
        super().__init__(config)
        for layer in self.model.layers:
            layer.self_attn = BitNetAttention(config)
            layer.mlp = BitNetMLP(config)
```

This is simpler than creating a custom `TextModel` subclass and avoids
duplicating the embedding/norm/rope setup logic.

### 3. Weight unpacking in `preprocess_weights`

For models with packed weight formats, handle unpacking in
`preprocess_weights()`:

```python
def preprocess_weights(self, state_dict):
    new_state_dict = {}
    weight_scales = {}

    # First pass: collect weight_scale values
    for key, value in state_dict.items():
        if key.endswith(".weight_scale"):
            prefix = key[:-len(".weight_scale")]
            weight_scales[prefix] = value

    # Second pass: unpack and scale weights
    for key, value in state_dict.items():
        if key.endswith(".weight_scale"):
            continue  # consumed above
        if key.endswith(".weight") and key[:-len(".weight")] in weight_scales:
            scale = weight_scales[key[:-len(".weight")]]
            if value.dtype == torch.uint8:
                value = _unpack_ternary_weights(value)  # {-1, 0, +1}
            value = value.float() * scale.float()
        new_state_dict[key] = value
    return new_state_dict
```

## Weight Packing Conventions

### HuggingFace BitNet 2-bit packing

```
Ternary values {-1, 0, +1} → add 1 → {0, 1, 2} (fits in 2 bits)
Pack 4 values per uint8: byte = v0 | (v1 << 2) | (v2 << 4) | (v3 << 6)
Packing dimension: first (out_features)
Packed shape: [out_features // 4, in_features]
```

### ORT MatMulNBits packing (for future optimization)

```
MatMulNBits with bits=2:
  weight: [N, n_blocks, blob_size] where blob_size = block_size * 2 / 8
  scales: [N, n_blocks]
  zero_points: [N, ceil(n_blocks * 2 / 8)]
  Attributes: K=in_features, N=out_features, bits=2, block_size=...
```

The HF packing and ORT packing use different layouts. Converting between
them requires reshaping — see `_weight_utils.py` for the GPTQ/AWQ
analogy (`_reshape_packed_qweight`).

## Checklist for Adding a Low-Bit Model

1. **Create model file** `src/mobius/models/<name>.py`
   - Subclass the appropriate base (`CausalLMModel`, etc.)
   - Add architecture-specific components (sub-norms, activations)
   - Implement `preprocess_weights()` for weight format conversion

2. **Export** from `src/mobius/models/__init__.py`

3. **Register** in `src/mobius/_registry.py`
   - Add to architecture-specific dict
   - Add `test_model_id` in `_TEST_MODEL_IDS`

4. **Add test config** in `tests/_test_configs.py`
   - Include config overrides that match the model (e.g., `hidden_act`)

5. **Add YAML test case** in `testdata/cases/<task>/`
   - Include `skip_reason` if golden JSON not yet generated

6. **Run tests:**
   ```bash
   python -m pytest tests/build_graph_test.py -k '<name>' -sv
   python -m pytest tests/model_coverage_test.py -k 'all_models' -sv
   ```

## Future: Efficient Ternary Inference

The current approach uses standard `Linear` with float weights (ternary
values scaled by `weight_scale`). For efficient inference:

- **MatMulNBits bits=2**: ORT supports this op, but the 2-bit kernel
  path is not as optimized as 4-bit. Still provides memory savings.
- **Custom ops**: T-MAC (CPU LUT-based) and BitBLAS (GPU) provide
  specialized ternary matmul kernels that could be registered as ORT
  custom ops.
- **Activation quantization**: BitNet also quantizes activations to int8
  at inference time. This is skipped in the current ONNX graph but could
  be added via `DynamicQuantizeLinear` nodes for better parity.
