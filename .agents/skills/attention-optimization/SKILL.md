---
name: attention-optimization
description: >
  Use this skill when choosing attention mask types, understanding ORT
  CUDA kernel dispatch, or optimizing attention performance. Covers
  bool mask vs float additive bias, Flash/MEA/unfused kernel selection,
  GQA dispatch rules, and nonpad_kv_seqlens for Flash eligibility.
---

# Skill: Attention Optimization

## When to use

Use this skill when:
- Choosing between bool mask, float additive bias, or no mask
- Diagnosing why ORT selected a slower attention kernel than expected
- Understanding Flash Attention requirements and limitations
- Working with GQA models and attention bias
- Optimizing attention for sliding window, KV-shared, or mixed head_dim

## Mask type decision table

| Scenario | Recommended | Why |
|----------|------------|-----|
| Causal only | `attn_mask=None` + `is_causal=1` | Enables Flash (fastest for prefill) |
| Padding (batch>1) | `nonpad_kv_seqlens` (+ static cache) or bool mask | `nonpad_kv_seqlens` enables Flash with no mask; pair with `TensorScatter` static cache for decode (can't combine with `past_key`/`past_value` inputs) |
| Sliding window (simple) | GQA `local_window_size` or bool mask | `local_window_size` keeps the fast GQA path; bool mask if you need ONNX Attention |
| Complex (sliding+KV-shared+dual head_dim) | Float additive bias | Avoids mask construction bugs in multi-constraint patterns |
| Custom pattern | Float additive bias | Arbitrary values |

## Bool mask vs float additive bias

### When to use each

| Pattern | Recommended mask type |
|---------|----------------------|
| Simple causal-only | No mask — use `is_causal=1` (enables Flash) |
| Sliding window (simple, GQA path) | GQA `local_window_size` (fast path); bool mask only if you must use ONNX Attention |
| KV-shared layers | Float additive bias |
| Mixed head_dim (e.g. Gemma4) | Float additive bias |
| Padding + causal | `nonpad_kv_seqlens` or bool mask |
| Multiple constraints combined | Float additive bias |

### Why float additive bias is safer for complex patterns

Bool mask and float additive bias are **equally precise** — both can
represent any attention pattern. ORT converts bool→float internally
via `ConvertAttnMaskToBias()`, so they have identical kernel dispatch.

The reason we use float bias for complex models is **bug avoidance**,
not a fundamental limitation of bool masks. Constructing correct bool
masks for multi-constraint patterns is error-prone:

- **Sliding window + KV-shared** — mask shape must match borrowed KV
  dimensions, not the current layer's. Off-by-one errors are silent.
- **`is_causal=1` + bool mask** — double-applies causal constraints
- **Dual head_dim** (e.g. Gemma4 local=128, global=256) — mask shapes
  differ per layer type, increasing construction complexity

For simpler models (e.g. Mistral with only sliding window), bool mask
is fine and uses less memory. Float bias is recommended when multiple
constraints interact.

### Common misconception: bool masks and Flash Attention

Bool masks do **NOT** enable Flash Attention. Flash Attention requires
`attn_mask=nullptr` (no mask at all). Both bool and float masks route
to Memory-Efficient Attention (MEA) or unfused attention — ORT
converts bool masks to float additive bias internally via
`ConvertAttnMaskToBias()`, so they have **identical kernel dispatch**.

### Flash Attention: when it actually helps

Flash Attention primarily helps during **prefill** (long prompt
processing). During single-token **decode**, attention is
memory-bandwidth bound regardless of kernel — Flash's compute
advantages don't help when `seq_len=1`.

**Gemma4 example:** Flash Attention cannot be used for *any* layer:
- **Sliding window layers:** Require an explicit mask → disqualifies Flash
- **Full attention layers:** `head_dim=512` → exceeds Flash's 256 limit

MEA is the effective best kernel for Gemma4. This is representative
of complex models — Flash is most beneficial for simple architectures.

### Recommendation

Use float additive bias via `create_attention_bias()` for all models
with complex attention patterns (Gemma4, sliding window models). Only
use `is_causal=1` (no explicit mask) for simple causal-only patterns
— this is also the only way to get Flash Attention.

```python
# BEST: no mask — enables Flash Attention for simple causal models
# (set is_causal=1 on the Attention op)

# GOOD: float additive bias — explicit, correct for complex patterns
attention_bias = create_attention_bias(
    op, input_ids=input_ids, attention_mask=attention_mask,
)

# AVOID for complex patterns: bool mask with manual sliding window logic
```

## Flash Attention requirements

Flash Attention is the fastest kernel for **prefill** (long sequences)
but has strict requirements and provides minimal benefit during
single-token decode (memory-bandwidth bound regardless of kernel):

| Requirement | Details |
|-------------|---------|
| No mask | `attn_mask == nullptr` (use `is_causal=1` instead) |
| Precision | fp16 or bf16 only (not fp32) |
| head_dim | ≤ 256 |
| Symmetric heads | `head_size == v_head_size` |
| GPU | SM≥8.0 (Ampere or newer) |

### `nonpad_kv_seqlens` — variable-length without an explicit mask

ONNX Attention opset 24 adds `nonpad_kv_seqlens` input, which tells
the kernel the actual (non-padded) KV sequence length per batch item.
This enables Flash Attention with variable-length sequences **without
providing an explicit mask** — the kernel applies causal masking
internally using the sequence length info.

> ⚠️ **Cannot be combined with the `past_key` / `past_value` inputs.** ORT
> *rejects* `nonpad_kv_seqlens` when `past_key`/`past_value` are supplied:
> *"nonpad_kv_seqlens should not be used together with past_key and
> past_value inputs."* It is therefore **not** usable with the *growing*
> (dynamic) cache mode beyond the prefill pass. It **is** used at every
> decode step in the **static-cache** mode below, where the full cache is
> passed in the `key`/`value` slots and `past_key`/`past_value` are unused.

#### In-place KV for ONNX Attention via `TensorScatter` (static cache)

The opset-24 ONNX `Attention` schema has **no `past_present_share_buffer`
attribute**, so the naive dynamic mode does `present = concat(past, new)`
every step (an O(N) copy of distinct `past`/`present` tensors). But ONNX
`Attention` **can** still update a KV cache *in place* — by pairing it with
the opset-24 **`TensorScatter`** op:

1. Pre-allocate a fixed-size KV buffer (`StaticCacheState` in mobius).
2. `TensorScatter` writes the new token(s) into the buffer at
   `write_indices` (in place when the buffer is IO-bound to the same
   device memory) — no growing concat.
3. Pass the **full** scattered cache in the `key`/`value` slots (not
   `past_key`/`past_value`), with `nonpad_kv_seqlens` giving the valid
   length and `is_causal=1` for masking.

So in-place KV is **not** GQA-exclusive. The contrib **GroupQueryAttention**
op has a built-in shared buffer (`past_present_share_buffer`); ONNX
`Attention` reaches the same effect explicitly with `TensorScatter` +
static cache. GQA's residual decode edge comes mostly from its dedicated
`seq==1` decode kernels (XQA / Flash-decode), not from buffer management
alone.

```python
# Dynamic (growing) mode — nonpad_kv_seqlens is PREFILL ONLY here,
# because it cannot be combined with past_key/past_value:
attn_out = op.Attention(
    query, key, value,
    attn_mask=None,                # nullptr → Flash eligible
    nonpad_kv_seqlens=seqlens_k,   # opset 24, prefill pass only
    q_num_heads=num_heads,
    kv_num_heads=kv_heads,
    is_causal=1,
)

# Static-cache mode — in-place KV at EVERY step (prefill + decode):
updated_k = op.TensorScatter(key_cache, key, write_indices, axis=1)
updated_v = op.TensorScatter(value_cache, value, write_indices, axis=1)
attn_out = op.Attention(
    query, updated_k, updated_v,   # full cache in key/value slots
    None, None, None,              # no attn_mask, no past_key/past_value
    nonpad_kv_seqlens,             # valid length per batch — used every step
    q_num_heads=num_heads,
    kv_num_heads=kv_heads,
    is_causal=1,
)
```

## ORT CUDA Attention Kernel Dispatch

ORT selects attention kernels via a cascade — first match wins.

### Contrib MultiHeadAttention (`com.microsoft`)

Cascade: LeanAttention → Flash → cuDNN SDPA → TRT FusedCross →
TRT FusedRunner → MEA → Unfused

| Kernel | Required conditions |
|--------|---------------------|
| LeanAttention | `USE_LEAN_ATTENTION` build flag, `seq_len==1`, `past_seq>0`, no bias, no padding mask, `head_size==v_head_size` |
| Flash | No bias, no padding mask, no `past_seq`, no `cache_indirection`, `head_size==v_head_size`, fp16/bf16, SM≥8.0 |
| cuDNN SDPA | `enable_cudnn_flash_attention_`, mask NONE or 1D_KEY_SEQ_LEN |
| TRT FusedCross | NOT unidirectional, no padding/bias/past, `hidden==v_hidden` |
| TRT FusedRunner | NOT unidirectional, no bias, mask none or 1D, `seq_len==kv_seq_len` |
| MEA (CUTLASS) | Long sequence, bias alignment OK (null or `seq % 4*sizeof(T) == 0`), no past/cache |
| Unfused | Always available (fallback) |

### Contrib GroupQueryAttention (`com.microsoft`)

Cascade: XQA → Flash → MEA → Unfused. **Rejects `attention_bias`
entirely.**

| Kernel | Required conditions |
|--------|---------------------|
| XQA | SM≥8.0, `seq==1`, `past_present_share_buffer`, `softcap==0`, `local_window==-1`, `head_size ∈ {64, 128, 256}` |
| Flash | fp16/bf16, SM≥8.0. FastDecode: `seq==1`, `past_present_share_buffer`, no KV quant |
| MEA | No bias (rejected upstream), head_size check |
| Unfused | Fallback |

### ONNX Attention — MHA (`q_num_heads == kv_num_heads`)

Cascade: Flash → MEA → Unfused

| Kernel | Required conditions |
|--------|---------------------|
| Flash | fp16/bf16, `head_size ≤ 256`, `head_size == v_head_size`, `attn_mask == nullptr`, SM≥8.0 |
| MEA | `head_size ≤ 1024` & `% 8 == 0`, if mask then `total_seq % 4 == 0`, if `past_key` then `head_size == v_head_size` |
| Unfused | Always available |

### ONNX Attention — GQA (`q_num_heads != kv_num_heads`)

Same cascade as MHA with extra MEA constraints:

| Kernel | Required conditions |
|--------|---------------------|
| Flash | Same as MHA |
| MEA | MHA conditions + `head_size == v_head_size` + not float32 |
| Unfused | Always available, handles GQA via in-kernel reshape |

### GQA + float additive bias dispatch

When using float bias with GQA, Flash is disabled (`attn_mask !=
nullptr`). The effective dispatch:

| Condition | Kernel |
|-----------|--------|
| fp16/bf16, `head_size == v_head_size`, `total_kv % 4 == 0` | MEA ✅ |
| fp16/bf16, `head_size == v_head_size`, `total_kv % 4 != 0` | Unfused (bias alignment) |
| fp16/bf16, `head_size != v_head_size` (asymmetric V) | Unfused |
| fp32 (GQA) | Unfused (explicitly excluded) |
| `qk_matmul_output_mode != kNone` | Unfused (or error) |

This explains why Gemma4's KV-shared layers fall to unfused: they
borrow K/V from a layer with different `head_size`, creating
`head_size != v_head_size` which disqualifies MEA.

## GQA vs ONNX Attention tradeoffs

| Feature | Contrib GQA | ONNX Attention |
|---------|-------------|----------------|
| Attention bias | ❌ Rejected | ✅ Supported |
| Flash Attention | ✅ (no mask) | ✅ (no mask) |
| XQA kernel | ✅ | ❌ |
| In-place KV buffer | ✅ built-in `past_present_share_buffer` | ✅ via `TensorScatter` + static cache (no growing concat) |
| Sliding window | ✅ `local_window_size` attribute | Via float/bool bias only |
| Variable-length | Via `seqlens_k` (works with past KV) | Via `nonpad_kv_seqlens` (with static cache, not the `past_key`/`past_value` inputs) |

**Guideline:** Use Contrib GQA when you don't need attention bias
(simple causal models, sliding-window via `local_window_size`). Use ONNX
Attention when you need a float bias (KV-shared, dual head_dim, or
mixed/alternating per-layer windows that one global window can't express).

### GQA sliding window via `local_window_size`

GroupQueryAttention takes a `local_window_size` attribute that masks each
query to the most recent `W` keys (positions `[i-W+1, i]`) — exactly
matching HuggingFace `sliding_window=W`. This keeps a uniform-window model
on the fast GQA path instead of forcing it onto ONNX `Attention` with a
baked float window mask. In mobius this is wired in `TextModel.forward`
from `config.sliding_window` (see `GQAContext.local_window_size`), guarded
to uniformly-sliding models — mixed `layer_types` (Gemma2/3/4, gpt-oss)
use custom per-layer `GQAContext`s instead.

> Note: `local_window_size` only *masks* attention; it does not shrink the
> physical KV buffer, so bounding memory still needs a circular/static
> cache. Also, the post-hoc GQA rewrite (`RotaryAttentionToGQA`) cannot
> recover a window from an already-baked float mask, so sliding windows
> must be set on the **direct** GQA path (GQAContext), not via the rewrite.
>
> Decode trade-off: a non-default `local_window_size != -1` disqualifies the
> dedicated `seq==1` **XQA** decode kernel (which requires `local_window==-1`,
> see the CUDA cascade table above), so windowed GQA decode falls back to
> Flash-decode/MEA. This is still on the fast GQA path and is the correct
> trade for models that genuinely need the window; do not set it on
> full-attention models.

## Key takeaways for model builders

1. **Flash requires `attn_mask == nullptr`** — any explicit mask
   disables Flash. Use `is_causal=1` instead.
2. **`nonpad_kv_seqlens`** enables Flash with variable-length sequences
   without an explicit mask. It cannot be combined with the
   `past_key`/`past_value` inputs (so it is prefill-only in the *growing*
   cache mode), but it is used at **every** step in the static-cache mode
   (full cache in the `key`/`value` slots).
3. **GQA contrib op rejects `attention_bias`** — use standard ONNX
   `Attention` if you need bias with GQA.
4. **SM≥8.0** (Ampere+) required for Flash on all paths.
5. **Float bias is safer** than bool mask for complex attention patterns.
6. **MEA requires alignment** — `total_kv % 4 == 0` for bias tensors.
7. **In-place KV is not GQA-exclusive.** GQA has a built-in shared buffer
   (`past_present_share_buffer`); ONNX `Attention` (opset 24) has no such
   attribute but reaches the same effect with **`TensorScatter` + a static
   cache** (vs. the naive growing `concat(past, new)`). GQA's residual
   decode edge is mostly its dedicated `seq==1` kernels (XQA/Flash-decode).
8. **Sliding window on the fast path:** set GQA `local_window_size` (=
   `config.sliding_window`) for uniform-window models instead of baking a
   float window mask into ONNX `Attention`.

## Cross-references

- **Debugging memcpy:** `.agents/skills/debugging-memcpy/SKILL.md`
- **Profiling:** `.agents/skills/profiling-onnx-models/SKILL.md`
- **Reusable components:** `.agents/skills/reusable-components/SKILL.md`
