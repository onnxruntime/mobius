---
name: moe-models
description: >
  Use this skill when adding or modifying a model that uses Mixture-of-Experts
  (MoE) layers. Covers gate variants (TopKGate, SparseMixerGate), MoELayer
  composition and expert routing, expert weight naming conventions for
  HuggingFace alignment, and preprocess_weights mappings for stacked
  expert tensors. Applicable to models like Mixtral, DeepSeek, and Qwen-MoE.
---

# Skill: Mixture-of-Experts (MoE) Models

## When to use

Use this skill when adding or modifying a model that uses Mixture-of-Experts
layers — where each token is routed to a subset of expert MLPs.

## Architecture overview

```
MoEDecoderLayer
 ├── RMSNorm (input_layernorm)
 ├── Attention (self_attn)
 ├── RMSNorm (post_attention_layernorm)
 └── MoELayer
      ├── Gate (routing: input → expert selection + weights)
      └── Experts[0..N-1] (each is a standard MLP)
```

### Key components

| Component | File | Purpose |
|-----------|------|---------|
| `MoELayer` | `components/_moe.py` | Routes tokens to experts, combines outputs |
| `TopKGate` | `components/_moe.py` | Standard softmax + top-k routing |
| `SparseMixerGate` | `components/_moe.py` | Sequential selection with threshold masking |
| `MoEDecoderLayer` | `models/moe.py` | Decoder layer that uses `MoELayer` instead of `MLP` |
| `MoETextModel` | `models/moe.py` | Text model with MoE decoder layers |

## How routing gates work

### TopKGate (default)

Standard routing used by most MoE models (Mixtral, GPTOSS):

1. Compute router logits: `logits = MatMul(hidden_states, gate_weight)`
2. Top-k selection: `values, indices = TopK(logits, k=num_experts_per_tok)`
3. Softmax over selected experts: `weights = Softmax(values)`

### SparseMixerGate (PhiMoE)

Sequential expert selection with threshold-based masking:

1. Compute router logits via `MatMul`
2. For each of `top_k` rounds:
   - Find max score (`ReduceMax`)
   - Threshold mask: experts whose scores are far from the max (relative to
     `jitter_eps`) are masked with `-inf`
   - Softmax over non-masked experts
   - `TopK(k=1)` to select best expert
   - `ScatterElements` to mask out the selected expert for the next round
3. Concatenate all selected expert indices and weights

## Adding a new MoE model

### 1. Determine the gate type

Check the HuggingFace implementation for the routing logic.  Look for:

- `router_type` or `routing_type` in the config
- How `router_logits` are computed and processed
- Whether top-k is applied before or after softmax

If neither `TopKGate` nor `SparseMixerGate` fits, create a new gate class
in `components/_moe.py`.

### 2. Check expert MLP naming

HuggingFace MoE models often use different weight names for expert MLPs:

| HF name | Our name | Description |
|---------|----------|-------------|
| `w1` | `gate_proj.weight` | Gate projection |
| `w2` | `down_proj.weight` | Down projection |
| `w3` | `up_proj.weight` | Up projection |

Implement a `_rename_moe_expert_weights()` function if the naming differs:

```python
def _rename_moe_expert_weights(state_dict):
    renamed = {}
    for key, value in state_dict.items():
        new_key = key
        if ".experts." in key:
            new_key = new_key.replace(".w1.", ".gate_proj.")
            new_key = new_key.replace(".w2.", ".down_proj.")
            new_key = new_key.replace(".w3.", ".up_proj.")
        renamed[new_key] = value
    return renamed
```

### 3. Check the normalization

Some MoE models use different norms than standard models:

- **PhiMoE**: Uses `LayerNorm` (with bias), not `RMSNorm`
- **Mixtral**: Uses standard `RMSNorm`

### 4. Create the model class

Use `MoETextModel` with the correct gate factory:

```python
class MyMoECausalLMModel(CausalLMModel):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.config = config
        # Pass gate_factory for custom routing
        self.model = MoETextModel(config, gate_factory=SparseMixerGate)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=True)

    def preprocess_weights(self, state_dict):
        state_dict = _rename_moe_expert_weights(state_dict)
        return super().preprocess_weights(state_dict)
```

### 5. Inject a custom gate

`MoELayer` accepts an optional `gate` parameter.  `MoETextModel` accepts
`gate_factory` — a callable `(config) -> gate_instance` that creates a gate
per layer:

```python
# Default (TopKGate)
MoETextModel(config)

# Custom gate
MoETextModel(config, gate_factory=SparseMixerGate)
```

To create a new gate, implement a class with this interface:

```python
class MyGate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.weight = nn.Parameter([config.num_local_experts, config.hidden_size])
        # ... other params

    def forward(self, op, hidden_states):
        # Returns: (expert_weights, expert_indices)
        # expert_weights: [batch, seq, num_experts_per_tok]
        # expert_indices: [batch, seq, num_experts_per_tok] (INT64)
        ...
        return weights, indices
```

## Config fields for MoE

```python
config.num_local_experts     # Total number of experts (e.g. 8, 16)
config.num_experts_per_tok   # Experts activated per token (e.g. 2)
```

These are extracted automatically from HuggingFace configs.

## TopK ONNX op gotcha

The ONNX `TopK` op requires `K` as a **1-D int64 tensor**, not a Python int:

```python
# WRONG
op.TopK(logits, self.top_k, axis=-1)

# CORRECT
k_tensor = op.Constant(value_ints=[self.top_k])
op.TopK(logits, k_tensor, axis=-1)
```

## Qwen3.5-MoE: Hybrid Attention + MoE

Qwen3.5-MoE combines the hybrid DeltaNet/full-attention architecture of
Qwen3.5 dense with MoE FFN layers instead of a dense MLP.

### Architecture

```
Qwen35MoEDecoderLayer
 ├── OffsetRMSNorm (input_layernorm)
 ├── GatedDeltaNet / Qwen35Attention (per layer_types)
 ├── OffsetRMSNorm (post_attention_layernorm)
 └── Qwen35MoEBlock
      ├── TopKGate (router)
      ├── Experts[0..N-1] (standard MLP: gate/up/down with SiLU)
      ├── SharedExpert (MLP: gate/up/down with SiLU)
      └── shared_expert_gate → sigmoid scalar
```

### Classes

| Class | File | Purpose |
|-------|------|---------|
| `Qwen35MoEBlock` | `models/qwen.py` | MoE block with routed + shared experts |
| `Qwen35MoEDecoderLayer` | `models/qwen.py` | Hybrid attention + MoE FFN layer |
| `Qwen35MoETextModel` | `models/qwen.py` | Stacks decoder layers with RoPE |
| `Qwen35MoECausalLMModel` | `models/qwen.py` | Top-level causal LM model |

### MoE block (`Qwen35MoEBlock`)

- **TopKGate routing**: 256 experts, top-8 in the full model (configurable
  via `num_local_experts` / `num_experts_per_tok`)
- **Expert MLPs**: Standard `MLP` (gate/up/down projections, SiLU activation),
  each with `moe_intermediate_size` as the intermediate dim
- **Shared expert**: A separate `MLP` that runs on **all** tokens (not routed),
  sized by `shared_expert_intermediate_size`
- **Shared expert gating**: `sigmoid(shared_expert_gate(x)) * shared_expert(x)`,
  where `shared_expert_gate` is `Linear(hidden_size, 1, bias=False)`

The key difference from standard MoE is the shared expert: its output is
gated by a learned sigmoid scalar and added to the routed expert output.

### Config fields

```python
config.moe_intermediate_size           # Intermediate size per expert MLP
config.shared_expert_intermediate_size  # Intermediate size for the shared expert
config.num_local_experts               # Total number of routed experts (e.g. 256)
config.num_experts_per_tok             # Experts activated per token (e.g. 8)
config.layer_types                     # Per-layer attention type list
```

### Weight naming

HuggingFace weights map directly (no renames needed for expert names):

```
mlp.gate.weight                               → router logits
mlp.experts.N.{gate,up,down}_proj.weight      → per-expert MLP
mlp.shared_expert.{gate,up,down}_proj.weight  → shared expert MLP
mlp.shared_expert_gate.weight                 → sigmoid gate (Linear, no bias)
```

Note: HF checkpoints store experts as fused tensors
(`experts.gate_up_proj`, `experts.down_proj`).  `preprocess_weights()`
unpacks these into per-expert tensors and also renames
`linear_attn.conv1d.weight` → `linear_attn.conv1d_weight`.

### Testing

Integration tests use a random-weight HF model with reduced layers and
experts (e.g. 4 layers, 4 experts, top-2 routing).  See
`test_qwen35_moe_prefill_logits_match` in `tests/integration_test.py`.

## Testing MoE models

MoE integration tests require a model with MoE layers.  A good test model
should be small enough for CI (~1-4B params).  The test pattern:

1. Build ONNX model with weights
2. Run prefill + decode against HuggingFace reference
3. Optionally test greedy generation (token-ID matching)

See `tests/moe_integration_test.py` for the complete pattern.

## Direct MoE op emission (com.microsoft.MoE)

OnnxRuntime ships a fused `com.microsoft.MoE` contrib op (CUDA float32/fp16/bf16,
CPU float32/fp16). For new model architectures, **emit it directly** — like
`com.microsoft.GroupQueryAttention` — rather than relying on a rewrite rule.

### When to use

Use `com.microsoft.MoE` when:
- The model uses top-k MoE routing (standard softmax gate → TopK)
- All expert weights are the same shape (no dynamic expert counts)
- The EP's `caps.supports_fused_moe` is `True`

Fall back to the loop-over-experts path when `supports_fused_moe` is `False`
(CPU EP without contrib ops, or EPs that don't support the custom op).

### Gate output: full pre-topk router_probs

The op takes the **full** `(num_tokens, num_experts)` probability tensor and
performs top-k selection internally via the `k` attribute. The gate must
produce the full softmax distribution — not already-selected top-k indices.

```python
# In your gate forward(), return shape [num_tokens, num_experts]
router_probs = op.Softmax(op.MatMul(hidden_states, self.weight), axis=-1)
```

### Emission pattern (from Gemma 4 implementation)

```python
from mobius._build_context import ep_capabilities

caps = ep_capabilities()
if caps.supports_fused_moe:
    moe_out = op.CastLike(
        op.MoE(                        # type: ignore[attr-defined]
            normed_hidden,             # [num_tokens, hidden_size]
            router_probs,              # [num_tokens, num_experts] — full pre-topk
            self.fc1_experts_weights,  # [E, inter_size, hidden_size]
            self.fc2_experts_weights,  # [E, hidden_size, inter_size]
            activation_type="silu",
            k=self._top_k,
            normalize_routing_weights=1,
            _domain="com.microsoft",
        ),
        normed_hidden,  # CastLike: preserve bf16/fp16/fp32 — NOT hardcoded float32
    )
else:
    moe_out = self._dispatch_moe_fallback(op, normed_hidden, router_probs)
```

**Critical: use `CastLike` after the MoE op.** The `com.microsoft.MoE` custom
op has `type=None` on its output — ONNX type propagation cannot infer the
output dtype. `op.CastLike(moe_output, target=input)` restores the correct
dtype (bf16/fp16/fp32), which allows downstream ops to share scalar
initializers and avoids hard-coded `Cast` to float32.

### preprocess_weights: expert weight mapping

HuggingFace Gemma 4 stores experts as a 3D tensor per projection:
`layers.N.experts.gate_up_proj [E, 2*inter, H]`
`layers.N.experts.down_proj    [E, H, inter]`

Map these to the parameter names used by the ONNX MoE op:

```python
def preprocess_weights(self, state_dict):
    for key in list(state_dict.keys()):
        if ".experts.gate_up_proj" in key:
            new_key = key.replace(".experts.gate_up_proj", ".fc1_experts_weights")
            state_dict[new_key] = state_dict.pop(key)
        elif ".experts.down_proj" in key:
            new_key = key.replace(".experts.down_proj", ".fc2_experts_weights")
            state_dict[new_key] = state_dict.pop(key)
    return super().preprocess_weights(state_dict)
```

For models that store per-expert weights separately (one matrix per expert),
stack them into 3D tensors in `preprocess_weights`:

```python
n = config.num_local_experts
gate = torch.stack([state_dict.pop(f"experts.{i}.gate_proj.weight") for i in range(n)])
down = torch.stack([state_dict.pop(f"experts.{i}.down_proj.weight") for i in range(n)])
state_dict["fc1_experts_weights"] = gate   # [E, inter, hidden]
state_dict["fc2_experts_weights"] = down   # [E, hidden, inter]
```

### EP capability check (matches GQA pattern)

```python
# In _execution_providers.py EpCapabilities:
supports_fused_moe: bool = True  # set False for EPs without com.microsoft.MoE

# In model forward():
from mobius._execution_providers import ep_capabilities
caps = ep_capabilities()
if caps.supports_fused_moe:
    # emit com.microsoft.MoE
else:
    # fallback loop
```

### Fallback: TopKGate + loop dispatch

When `supports_fused_moe` is False, implement a static unroll:

```python
def _dispatch_moe_fallback(self, op, hidden, router_probs):
    k_tensor = op.Constant(value_ints=[self._top_k])
    top_weights, top_indices = op.TopK(router_probs, k_tensor, axis=-1)
    top_weights = op.Softmax(top_weights, axis=-1)   # renormalize
    output = op.CastLike(op.ConstantOfShape(op.Shape(hidden), value=0.0), hidden)
    for e_idx in range(self._num_experts):
        w1 = op.Squeeze(op.Gather(self.fc1_experts_weights, [e_idx], axis=0), [0])
        w2 = op.Squeeze(op.Gather(self.fc2_experts_weights, [e_idx], axis=0), [0])
        # expert output, gated by routing weight
        ...
    return output
```

## NemotronH MoE (sigmoid routing + shared experts + latent projection)

NemotronH uses a non-standard MoE architecture that differs from the
standard softmax top-k pattern in several ways.

### Architecture

```
NemotronHMoEBlock
 ├── NemotronHMoEGate (sigmoid top-k with correction bias)
 ├── [optional] fc1_latent_proj (hidden → latent_size)
 ├── Experts[0..N-1] (non-gated FCMLP: up → act → down)
 ├── [optional] fc2_latent_proj (latent_size → hidden)
 └── SharedExperts (FCMLP, all tokens, residual add)
```

### Classes

| Class | File | Purpose |
|-------|------|---------|
| `NemotronHMoEGate` | `models/nemotron_h.py` | Sigmoid routing with e_score_correction_bias |
| `NemotronHMoEBlock` | `models/nemotron_h.py` | MoE dispatch with shared expert + latent proj |
| `NemotronHMoELayer` | `models/nemotron_h.py` | Pre-norm → MoE block → residual (stateless) |

### Sigmoid gate with correction bias

NemotronH does NOT use softmax routing. Instead:

1. `router_logits = hidden_states @ gate_weight.T`
2. `probs = sigmoid(router_logits)` — these become final routing weights
3. `choice_scores = probs + e_score_correction_bias` — bias affects selection only
4. `selected = topk(choice_scores)` — select top-k using biased scores
5. `weights = gather(probs, selected)` — gather from UNBIASED sigmoid probs
6. Normalize + scale by `routed_scaling_factor`

**Key difference from standard MoE**: The correction bias shifts expert
selection but does NOT affect final routing weights. The `com.microsoft.MoE`
op's built-in softmax routing is incompatible — you must use the fallback
loop or pre-compute routing weights and pass them to a modified MoE call.

### Non-gated FCMLP experts

Unlike Mixtral/Qwen (gated: `gate_proj * up_proj → down_proj`), NemotronH
experts are simple FCMLPs: `up_proj → activation → down_proj`. This maps
to `fc1_experts_weights` / `fc2_experts_weights` without a gate projection.

### Latent projection (120B only)

The 120B model has `moe_latent_size=1024` (vs `hidden_size=4096`):

```
hidden → fc1_latent_proj(4096→1024) → experts(1024→inter→1024) → fc2_latent_proj(1024→4096)
```

Gate routes on original hidden states (NOT latent). Shared expert operates
on original hidden_size (no latent projection).

### Shared expert

A single FCMLP that runs on ALL tokens (not routed), added as residual:

```python
output = routed_expert_output + shared_experts(original_hidden)
```

### HF weight format

HF stores expert weights as 3D stacked tensors:
```
experts.up_proj: [num_experts, intermediate_size, input_size]
experts.down_proj: [num_experts, hidden_size, intermediate_size]
```

`preprocess_weights()` splits these into per-expert 2D tensors for the
loop-based dispatch, or keeps them stacked for the fused MoE op path.

### Config fields (NemotronHConfig)

```python
config.num_local_experts             # Total experts (128 for 30B, 512 for 120B)
config.num_experts_per_tok           # Top-k (6 for 30B, 22 for 120B)
config.moe_intermediate_size         # Per-expert hidden dim
config.moe_latent_size               # Optional latent projection dim (120B: 1024)
config.shared_expert_intermediate_size  # Shared expert hidden dim
config.norm_topk_prob                # Whether to normalize routing weights
config.routed_scaling_factor         # Post-normalization scale
```

### com.microsoft.MoE compatibility

**Not compatible with NemotronH.** Three blockers:

1. **Squared ReLU activation**: NemotronH experts use `relu2` (squared ReLU:
   `relu(x)^2`). The fused MoE op only supports `silu`, `gelu`, `relu`,
   `none` — no squared ReLU. Since the activation is applied between the
   two matmuls inside the fused op, there is no way to inject a custom
   activation.

2. **Sigmoid routing with correction bias**: NemotronH gate uses
   `sigmoid → add_bias → topk` for expert selection, but final routing
   weights come from the unbiased sigmoid probs. The fused op has no
   option to bypass its internal softmax/topk routing.

3. **Shared expert + latent projection**: These must run outside the fused
   op regardless, adding complexity without eliminating the main bottleneck.

NemotronH uses loop-over-experts dispatch. See `NemotronHMoEBlock` docstring.

### Graph size impact

The loop-over-experts fallback creates one subgraph per expert per MoE layer:
- 30B: 128 experts × 23 MoE layers = 2,944 expert subgraphs → ~40K nodes
- 120B: 512 experts × 40 MoE layers = 20,480 expert subgraphs → ~270K nodes

The fused MoE op replaces each per-layer loop with a single op, dramatically
reducing graph size and enabling batched GPU execution.

### HF dt_bias corruption bug

**Critical**: The NemotronH remote-code `_init_weights` re-initialises
Mamba2 `dt_bias` parameters with `torch.rand()` AFTER `from_pretrained`
loads checkpoint weights, silently corrupting the model. HF inference
becomes non-deterministic — different argmax on each model load.

Fix: Use `_fix_nemotron_h_dt_bias()` from `mobius._testing.torch_reference`
after loading the HF model. This reads correct `dt_bias` values from the
safetensors files and patches them in-place. Without the fix, golden
reference data is unreliable.

```python
from mobius._testing.torch_reference import _fix_nemotron_h_dt_bias
model = AutoModelForCausalLM.from_pretrained(model_id, ...)
_fix_nemotron_h_dt_bias(model, model_id)  # Must call before eval()
```
