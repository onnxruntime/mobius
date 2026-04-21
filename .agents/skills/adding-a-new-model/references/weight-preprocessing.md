# Weight Preprocessing Reference

Detailed `preprocess_weights` examples, troubleshooting for weight loading,
and advanced debugging patterns. Read this when you need to handle weight
name mismatches, fused weight splitting, or diagnose numerical issues traced
to weight loading or dtype problems.

## Common preprocess_weights operations

- **Strip prefixes:** `language_model.model.X` → `X` (multimodal models)
- **Rename expert weights:** `w1` → `gate_proj` (MoE models)
- **Weight tying:** copy `embed_tokens.weight` → `lm_head.weight`
- **Split fused QKV:** `Wqkv.weight` [3H, H] → `q_proj`, `k_proj`, `v_proj` (ModernBERT)
- **Split fused gate+up:** `Wi.weight` [2I, H] → `gate_proj`, `up_proj` (ModernBERT)
- **Split fused BLIP QKV:** `in_proj_weight` [3H, H] → separate Q/K/V projections

## Debugging mismatches: initializer comparison

Build the ONNX model, list its initializer names, and compare against
the HuggingFace state dict keys to find mismatches:

```python
model_names = set(onnx_model.graph.initializers.keys())
hf_names = set(state_dict.keys())
print("In HF but not model:", hf_names - model_names)
print("In model but not HF:", model_names - hf_names)
```

## Attention scale override

**Symptom:** Attention scores are wrong, causing gradual drift in generation.

**Diagnosis:** Some models override the default `1/sqrt(head_dim)` scale:

```python
# Check HuggingFace attention class
from transformers.models.granite.modeling_granite import GraniteAttention
print(GraniteAttention.__init__)  # Look for self.scaling = ...
```

**Fix:** Use the `scale` parameter on the `Attention` component:

```python
# In your custom DecoderLayer:
self.self_attn = Attention(config, scale=config.attention_multiplier)
```

The `Attention.__init__` accepts an optional `scale: float | None` parameter.
When `None`, it defaults to `head_dim**-0.5`.

## Weight-free norms (no learnable parameters)

**Symptom:** Weight keys like `model.norm.weight` appear in the model but not
in the HuggingFace state dict, and `preprocess_weights` fills them with ones.
The norm still uses the wrong algorithm (e.g. RMSNorm instead of LayerNorm).

**Fix:** Use `_WeightFreeLayerNorm` from `models/olmo.py` which creates
`nn.Parameter` with constant data (ones for scale, zeros for bias) that
the ONNX `LayerNormalization` op requires, but the HuggingFace model has no
corresponding weights for:

```python
class _WeightFreeLayerNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.scale = nn.Parameter(
            [hidden_size], data=ir.tensor(np.ones(hidden_size, dtype=np.float32))
        )
        self.bias = nn.Parameter(
            [hidden_size], data=ir.tensor(np.zeros(hidden_size, dtype=np.float32))
        )
        self.eps = eps

    def forward(self, op, hidden_states):
        return op.LayerNormalization(
            hidden_states, self.scale, self.bias, epsilon=self.eps, axis=-1
        )
```

## Gated attention Q/gate split ordering

**Symptom:** Large logit diff (~2.0) on Qwen3.5 or similar gated attention models.

**Root cause:** HF splits Q and gate *within each head*: reshapes to
`[B, S, num_heads, 2*head_dim]` then chunks on last dim.  A naive midpoint
split of the flat tensor gives wrong results.

**Fix:** Reshape to per-head layout before splitting:
```python
# WRONG: split flat tensor at midpoint
q, gate = op.Split(qg_proj, num_outputs=2, axis=-1)

# CORRECT: reshape to per-head, then split
qg = op.Reshape(qg_proj, [0, 0, num_heads, 2 * head_dim])
q, gate = op.Split(qg, [head_dim, head_dim], axis=-1)
```

## DeltaNet missing query scaling

**Symptom:** Linear attention output is orders of magnitude too large.

**Root cause:** After L2-normalizing Q and K, you still need a
`1/sqrt(key_head_dim)` scaling factor on the query, similar to standard
attention.

## Extracting `last_hidden_state` before vs after norm

**Symptom:** Downstream model (e.g. code predictor, projection layer)
receives wrong hidden states. Prefill logits match HF exactly, but
generation diverges immediately.

**Root cause:** HuggingFace's `outputs.last_hidden_state` is the
**post-norm** hidden state (after RMSNorm/LayerNorm). If you extract
the hidden state before the final norm, downstream consumers get
pre-norm values. In single-model LLMs this doesn't matter (the lm_head
is after the norm). In multi-model pipelines (TTS, VLM), the
hidden state is passed to another model, so norm ordering is critical.

**Fix:** Always extract hidden state *after* the model's final norm:
```python
# WRONG: hidden_states before norm
hidden_states = decoder_output  # pre-norm
logits = lm_head(norm(hidden_states))  # logits correct, but...
return logits, hidden_states  # hidden_states is WRONG for downstream

# CORRECT: apply norm first, then use for both logits and output
hidden_states = norm(decoder_output)  # post-norm
logits = lm_head(hidden_states)
return logits, hidden_states  # hidden_states matches HF
```

## Identity node folding renames initializers

**Symptom:** Weight loading fails — `preprocess_weights` maps to the
original parameter name (e.g. `code_predictor.stacked_codec_embedding`)
but the initializer in the ONNX graph has been renamed to something
like `v_code_predictor.Identity_174`.

**Root cause:** The IR optimizer folds `Identity(initializer)` by
removing the Identity node and renaming the initializer to the
output name. If you then set a custom name on the output (e.g.
`codec_embeddings.name = "codec_embeddings"`), the initializer gets
renamed to `codec_embeddings` — breaking weight loading.

**Fix:** Use `op.Identity()` to create a *real* Identity node between
the initializer and the graph output. Ensure the elimination pass
retains Identity nodes that feed graph outputs. This creates a separate
output value, so renaming the output doesn't affect the initializer:
```python
# In forward():
codec_embeddings = op.Identity(self.stacked_codec_embedding)
return logits, present_key_values, codec_embeddings

# In task (safe to rename — Identity separates the names):
codec_embeddings.name = "codec_embeddings"
graph.outputs.append(codec_embeddings)
```

## `np.ascontiguousarray` promotes 0-d arrays to 1-d

**Symptom:** ONNX `Gather` axis-reducing semantics break — the output
has an extra dimension (e.g. `(1, vocab)` instead of `(vocab,)`).

**Root cause:** `np.ascontiguousarray(scalar_array)` promotes shape
`()` to `(1,)`. This changes `Gather(axis=0)` from axis-reducing
(scalar index) to axis-preserving (1-d index).

**Fix:** Guard against 0-d arrays:
```python
if v.ndim > 0:
    v = np.ascontiguousarray(v)
```

## Multi-token prefill in code predictors

**Symptom:** Code predictor generates garbage. Prefill logits are
slightly off compared to HF.

**Root cause:** Some architectures (e.g. Qwen3-TTS code predictor) use
a **2-token prefill**: `concat(projected_hidden, embed(code_0))` as two
separate tokens through the transformer. Summing them into 1 token
changes attention patterns and all subsequent hidden states.

**Diagnosis:** Compare the inputs_embeds shape at step 0. If HF passes
`(batch, 2, hidden)` but your model uses `(batch, 1, hidden)`, the
attention context window is wrong.

**Fix:** Construct inputs_embeds externally to match HF's exact flow:
```python
# Step 0 (prefill): 2 tokens
inputs = np.concatenate([talker_hidden, embed(code_0)], axis=1)  # (1, 2, H)
# Steps 1+: 1 token
inputs = cp_embed[step-1, code_i, :].reshape(1, 1, -1)  # (1, 1, H)
```

## Embedding table index off-by-one in multi-step generation

**Symptom:** Codes are plausible but audio quality is wrong. Codec sum
doesn't match HF.

**Root cause:** In multi-step code prediction, HF uses
`embed[step-1](code)` at generation step `step`, not `embed[step]`.
The off-by-one means every embedding lookup uses the wrong table.

**Fix:** Carefully trace HF's generation loop to determine which
embedding table index corresponds to which generation step. Write a
comparison script that checks individual embedding lookups match.

## Codec sum uses output codes, not input codes

**Symptom:** codec_sum diverges from HF even though individual
embeddings weights are identical.

**Root cause:** The codec sum `Σ embed[i](code_{i+1})` uses the
*generated* (output) code at each step, not the input code. If the
model returns embeddings of the input codes, the sum is wrong.

**Fix:** Compute codec_sum externally using the codes actually
generated at each step:
```python
codec_sum = talker_embed(code_0)
for i in range(num_groups - 1):
    # codes[i+1] is the OUTPUT of code predictor step i
    codec_sum += cp_embed[i, codes[i + 1], :]
```

## Precision-sensitive ops need fp32 upcast

**Symptom:** Type mismatch errors (`tensor(float) vs tensor(bfloat16)`)
when loading a model built with `--dtype bf16`, or numerical drift compared
to HuggingFace when running in fp16/bf16.

**Root cause:** Operations like `exp`, `softplus`, `sigmoid` (in gated norms),
and RMSNorm variance are numerically sensitive and must run in float32 to
match HuggingFace, which explicitly upcasts with `.float()` /
`.to(torch.float32)`.

**Two distinct problems:**

1. **Naive `CastLike` everywhere** — keeps everything in the model dtype
   (e.g. bf16), but `exp` overflows and the SSM state diverges.
2. **Naive `Cast(to=ir.DataType.FLOAT)` everywhere** — computes in fp32 but forgets to cast
   back, producing type mismatches with downstream bf16 ops.

**Correct pattern — upcast → compute → cast back:**
```python
# 1. Upcast to fp32 for the sensitive region
dt_f32 = op.Cast(dt, to=ir.DataType.FLOAT)
dt_f32 = op.Softplus(dt_f32)
a_neg = op.Neg(op.Exp(op.Cast(self.A_log, to=ir.DataType.FLOAT)))
da = op.Exp(op.Mul(dt_4d, a_4d))  # all fp32 here
...
# 2. Cast back to input dtype at the boundary
y = op.CastLike(y_f32, x)
new_state = op.CastLike(new_state_f32, ssm_state)
```

**How to identify which ops need fp32:** Check the HuggingFace source for
`.float()` or `.to(torch.float32)` calls.  Each one marks an fp32 region
that the ONNX graph must replicate.

**Known fp32-required regions:**

| Region | HF evidence | ONNX pattern |
|--------|-------------|-------------|
| SSM recurrence (A, dt, exp, state) | `self.A_log.float()`, `hidden_states.float()`, `B.float()`, `C.float()` | `Cast(to=ir.DataType.FLOAT)` all inputs, `CastLike` output |
| GatedRMSNorm (SiLU + variance) | `hidden_states.to(torch.float32)`, `gate.to(torch.float32)` | Explicit fp32 for both, `CastLike` output |
| RMSNorm variance | `hidden_states.to(torch.float32)` | ONNX `RMSNormalization` handles via `stash_type=1` (default) |

**When fp32 upcast is NOT needed:**
- Linear projections (`MatMul`) — runtime handles mixed precision
- SiLU on conv output — HF keeps in model dtype
- Standard attention — ONNX `Attention` op handles precision internally

**Use `CastLike` for** parameters/constants that should match the *current*
compute dtype (which is fp32 inside an upcast region, or the model dtype
outside).  Use `Cast(to=ir.DataType.FLOAT)` to explicitly enter an fp32 region.
