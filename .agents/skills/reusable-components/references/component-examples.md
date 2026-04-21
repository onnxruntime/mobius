# Component Examples — Detailed Reference

## Attention

```python
Attention(config)
Attention(config, scale=0.015625)  # Override default 1/sqrt(head_dim) scale
# Inputs:  hidden_states, attention_bias, position_embeddings, past_key_value
# Outputs: attn_output, (key_cache, value_cache)
```

Handles MHA, GQA, and MQA via `num_key_value_heads`.  Supports optional QK
norm (`attn_qk_norm=True`) and bias on Q/K/V/O projections.

The optional `scale` parameter overrides the default `head_dim**-0.5` attention
scale.  Use this when a model specifies a custom attention multiplier (e.g.
Granite's `attention_multiplier`).  When `None` (default), uses `1/sqrt(head_dim)`.

The ONNX `Attention` op (opset 23) has an `is_causal` attribute.  For
decoder self-attention in encoder-decoder models (e.g., Whisper), set
`is_causal=1` instead of building an explicit causal mask with
`create_attention_bias`.

Some models (Whisper) require **Q pre-scaling** for numerical parity with
HuggingFace: multiply Q by `head_dim**-0.5` before passing to `op.Attention`
and set `scale=1.0`.  This matches HF's order of operations and avoids
floating-point divergence in softmax.

**Qwen35Attention** (`_attention.py`): Gated GQA variant for Qwen3.5. Doubles
the Q projection to produce both Q and a gate signal, applies per-head
`OffsetRMSNorm` to Q and K, supports partial RoPE, and gates the output with
`attn_output * sigmoid(gate)`.

## MLP

```python
MLP(config)
# Uses gate_proj + up_proj + down_proj with configurable activation
```

The activation function comes from `config.hidden_act` and is resolved by
`get_activation()`.

## DecoderLayer

```python
DecoderLayer(config)
# Pre-norm residual: LayerNorm → Attention → Add → LayerNorm → MLP → Add
```

To customise, subclass and override the components:

```python
class MyDecoderLayer(DecoderLayer):
    def __init__(self, config):
        super().__init__(config)
        # Replace norm with custom variant
        self.input_layernorm = MyRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
```

## GatedDeltaNet (Linear Attention)

```python
GatedDeltaNet(config)
# Inputs:  hidden_states, position_embeddings (unused), past_key_value (unused)
# Outputs: output, (conv_state, recurrent_state)
```

Recurrent linear attention mechanism from the Qwen3.5 hybrid architecture
(`_gated_deltanet.py`). Key operations: fused QKV projection, causal
depthwise Conv1D, L2-normalised Q/K, exponential decay gates, delta rule
recurrence, and gated output via `GatedRMSNorm`. Supports GQA-like key
head grouping (`num_k_heads` → repeat to `num_v_heads`). State is
`conv_state` + `recurrent_state` (currently zero-initialised for stateless
export).

## RoPE variants

Created via the factory function `initialize_rope(config)`:

| `config.rope_type` | Class | Use case |
|--------------------|-------|----------|
| `"default"` | `DefaultRope` | Standard RoPE |
| `"linear"` | `LinearRope` | Linear scaling (factor in `rope_scaling`) |
| `"dynamic"` | `DynamicNTKRope` | Dynamic NTK scaling |
| `"llama3"` | `Llama3Rope` | LLaMA-3 piecewise scaling |

**MRope (Multimodal RoPE):** Two variants share a `_MRopeBase` base class
that splits frequencies into temporal (T), height (H), and width (W) sections.
`ChunkedMRope` uses a chunked layout `[TTT...HHH...WWW]` (Qwen2-VL).
`InterleavedMRope` uses an interleaved layout `[T,H,W,T,H,W,...]` and
supports `partial_rotary_factor` for partial RoPE (Qwen3-VL, Qwen3.5).

RoPE embeddings are precomputed as `cos_cache` / `sin_cache` initializers
and looked up at runtime via `Gather` on `position_ids`.

## RMSNorm

```python
RMSNorm(hidden_size, eps=1e-6)
```

Uses the ONNX `RMSNormalization` op from opset 23.  The `eps` is a float
attribute (not a Parameter).

For Gemma's `weight + 1` variant, subclass:

```python
class GemmaRMSNorm(RMSNorm):
    def forward(self, op, hidden_states):
        weight_plus_one = op.Add(self.weight, 1.0)
        return apply_rms_norm(op, hidden_states, weight_plus_one, self.variance_epsilon)
```

**OffsetRMSNorm** (`_rms_norm.py`): `output * (1 + weight)` variant where
HuggingFace stores weights initialised to 0, so the effective multiplier is
`1 + weight`. Used by Qwen3.5 for per-head Q/K normalisation.

**GatedRMSNorm** (`_rms_norm.py`): `RMSNorm(x) * SiLU(gate)` — applies
RMS normalisation then element-wise gates the result with a SiLU activation
on a separate gate input. Used by GatedDeltaNet output projection.

## LayerNorm

```python
LayerNorm(hidden_size, eps=1e-6)
```

Uses the ONNX `LayerNormalization` op.  **Always check the model's HF
config for the correct epsilon** — the default `1e-6` does not match all
models.  For example, Whisper uses `1e-5`.  A wrong epsilon causes large
numerical drift that amplifies through the network.

## LayerNormNoAffine

```python
LayerNormNoAffine(dim, eps=1e-5)
```

Layer normalization **without learnable parameters** (`elementwise_affine=False`
in PyTorch).  Used in AdaLayerNorm blocks where scale/shift come from a
separate modulation projection.  Calls `op.LayerNormalization` with no
`Scale` or `Bias` inputs.

For weight-free LayerNorm that still needs frozen ones/zeros (e.g. OLMo-1B),
create constant parameters with `data=ir.tensor(...)` instead.

**Key:** RMSNorm vs LayerNorm is NOT interchangeable.  LayerNorm subtracts
the mean; RMSNorm does not.  Using the wrong type causes max abs diff > 1.0
that grows through layers.

## GroupNorm

```python
GroupNorm(num_groups, num_channels, eps=1e-5)
```

Group normalization with learnable `weight` and `bias`.  Uses the ONNX
`GroupNormalization` op.  Commonly used in diffusion models (UNet, VAE).

## Conv2d

```python
Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=0, groups=1)
```

2D convolution with bias, matching `torch.nn.Conv2d(bias=True)`.  Used in
diffusion models (VAE, UNet, ControlNet) and vision patch embeddings.
Parameters: `weight` (`[out, in/groups, kH, kW]`) and `bias` (`[out]`).

## SiLU

```python
SiLU()
# SiLU (Swish) activation as a module: x * sigmoid(x)
```

Useful in `nn.Sequential` containers where an activation needs to be a
module with a `forward()` method.  For functional use, call
`get_activation("silu")` instead.

## Linear

```python
Linear(in_features, out_features, bias=False)
# Uses MatMul (+ optional Add for bias)
```

## ClippableLinear

```python
ClippableLinear(in_features, out_features, bias=False)
# Linear with learned input/output activation clamping
# Matches HuggingFace Gemma4ClippableLinear
```

Wraps a standard `Linear` with 4 learned scalar parameters:
`input_min`, `input_max`, `output_min`, `output_max`.  Inputs are clamped
before the linear projection and outputs are clamped after:

```python
x = Clip(x, input_min, input_max)
x = MatMul(x, weight.T) [+ bias]
x = Clip(x, output_min, output_max)
```

**Critical:** HuggingFace `Gemma4ClippableLinear` stores *finite* learned
bounds (not ±inf).  Using plain `Linear` instead of `ClippableLinear` causes
large numerical divergence in Gemma4 vision and audio encoders:

- Audio encoder: max diff 52.68 → 0.0003 after fix
- Vision encoder: max diff 3.92 → 0.00007 after fix

The component is exported from the public API: `from mobius.components
import ClippableLinear`.

**Weight mapping:** HF stores `<prefix>.linear.weight` for the actual
weight (the `.linear.` segment is stripped by `preprocess_weights`), and
`<prefix>.input_min`, `<prefix>.input_max`, `<prefix>.output_min`,
`<prefix>.output_max` as direct scalar buffers.

The MLP component accepts a `linear_class` parameter, so you can pass
`linear_class=ClippableLinear` to use it for all projections in the MLP.

## Embedding

```python
Embedding(num_embeddings, embedding_dim, padding_idx=0)
# Uses Gather on weight matrix
```

## Shared weights with per-layer adapters

Some architectures reuse the same transformer block across multiple layers,
with per-layer low-rank adapters that differentiate each usage (e.g. Zamba2,
which shares one transformer across 6 hybrid layers).

### The scope challenge

`onnxscript.nn` determines ONNX initializer names from the module call stack.
A single module instance called multiple times produces the **same** initializer
names each time — which is exactly what we want for shared weights.  But
per-layer adapters need **different** names for each layer.

### Pattern: split shared + per-instance modules

```python
class _TextModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Shared weights: ONE instance → single set of ONNX initializers
        self.shared_transformer = SharedAttentionLayer(config)

        # Per-layer adapters: ModuleList → "adapters.0.*", "adapters.1.*"
        self.adapters = nn.ModuleList([
            AdapterModule(config) for _ in range(num_layers)
        ])

        # Shared MLP at model scope if adapter output must mix with MLP
        self.gate_proj = Linear(hidden, intermediate)
```

### Critical: use `__call__` for per-index scope

When iterating over adapter ModuleList elements, you **must** call the
element (triggering `__call__`) rather than accessing its sub-attributes:

```python
# ❌ Broken: adapter_out gets "q_adapter.weight" scope (same for all idx!)
adapter_out = self.adapters[idx].q_adapter(op, x)

# ✅ Correct: adapter_out gets "adapters.{idx}.q_adapter.weight" scope
adapter_out = self.adapters[idx](op, x)
```

### Handling the MLP circular dependency

When per-layer adapter output must be combined with shared MLP weights, and
the adapter input comes from inside the shared module, split the shared
module into phases:

1. **Phase 1 — Shared attention:** `shared_transformer(op, x)` → returns
   `mlp_input` (pre-MLP hidden states) + KV cache
2. **Phase 2 — Per-layer adapter:** `self.adapters[idx](op, mlp_input)` →
   per-layer contribution (correct `adapters.{idx}` scope)
3. **Phase 3 — Shared MLP at caller scope:** Apply gate/up/down projections
   registered on the caller module, combining with adapter output

```python
# In _TextModel.forward():
mlp_input, kv = self.shared_transformer(op, x)           # shared scope
adapter_out = self.mlp_adapters[idx](op, mlp_input)       # per-layer scope
gate = op.Add(self.gate_proj(op, mlp_input), adapter_out) # model scope
```

**Reference implementation:** `models/zamba2.py` — Zamba2 hybrid Mamba2 +
shared attention with Q/K/V/MLP low-rank adapters.
