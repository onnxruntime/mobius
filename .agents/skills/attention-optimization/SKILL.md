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
| Causal only | `attn_mask=None` + `is_causal=1` | Enables Flash (fastest) |
| Padding (batch>1) | Bool mask or `nonpad_kv_seqlens` | Simple, ORT optimized |
| Sliding window | Float additive bias | Precise window control |
| Complex (causal+sliding+padding) | Float additive bias | Most flexible |
| Custom pattern | Float additive bias | Arbitrary values |

## Bool mask vs float additive bias

### When to use each

| Pattern | Recommended mask type |
|---------|----------------------|
| Simple causal-only | No mask — use `is_causal=1` (enables Flash) |
| Sliding window | Float additive bias |
| KV-shared layers | Float additive bias |
| Mixed head_dim (e.g. Gemma4) | Float additive bias |
| Padding + causal | Float additive bias |

### Why float additive bias is safer for complex patterns

ONNX `Attention` supports bool mask (`True`=attend, `False`=ignore).
ORT correctly converts bool→float internally via
`ConvertAttnMaskToBias()`. However, constructing correct bool masks
for complex patterns is error-prone:

- **Sliding window boundaries** must align with KV cache positions —
  off-by-one errors silently produce wrong attention patterns
- **KV-shared layers** borrow K/V from other layers — the mask shape
  must match the borrowed KV dimensions, not the current layer's
- **`is_causal=1` + bool mask** double-applies constraints — the
  `is_causal` flag adds its own causal mask on top of the explicit one
- **Dual head_dim** (e.g. Gemma4 local=128, global=256) means mask
  shapes differ per layer type

Float additive bias gives explicit control:
- `0.0` for "attend" positions
- `-inf` (or `-10000.0`) for "ignore" positions
- No ambiguity in kernel interpretation

### Common misconception: bool masks and Flash Attention

Bool masks do **NOT** enable Flash Attention. Flash Attention requires
`attn_mask=nullptr` (no mask at all). Both bool and float masks route
to Memory-Efficient Attention (MEA) or unfused attention.

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

Flash Attention is the fastest kernel path but has strict requirements:

| Requirement | Details |
|-------------|---------|
| No mask | `attn_mask == nullptr` (use `is_causal=1` instead) |
| Precision | fp16 or bf16 only (not fp32) |
| head_dim | ≤ 256 |
| Symmetric heads | `head_size == v_head_size` |
| GPU | SM≥8.0 (Ampere or newer) |

### `nonpad_kv_seqlens` — Flash with variable lengths

ONNX Attention opset 24 adds `nonpad_kv_seqlens` input, which tells
the kernel the actual (non-padded) KV sequence length per batch item.
This enables Flash Attention with variable-length sequences **without
providing an explicit mask** — the kernel applies causal masking
internally using the sequence length info.

```python
# Enables Flash + past_present_share_buffer for efficient KV cache
attn_out = op.Attention(
    query, key, value,
    attn_mask=None,           # nullptr → Flash eligible
    past_key=past_k,
    past_value=past_v,
    nonpad_kv_seqlens=seqlens_k,  # opset 24
    q_num_heads=num_heads,
    kv_num_heads=kv_heads,
    is_causal=1,
    past_present_share_buffer=1,
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
| KV cache management | Built-in (`past_present_share_buffer`) | Manual (separate past/present) |
| Variable-length | Via `seqlens_k` | Via `nonpad_kv_seqlens` (opset 24) |

**Guideline:** Use Contrib GQA when you don't need attention bias
(simple causal models). Use ONNX Attention when you need float bias
(sliding window, KV-shared, padding).

## Key takeaways for model builders

1. **Flash requires `attn_mask == nullptr`** — any explicit mask
   disables Flash. Use `is_causal=1` instead.
2. **`nonpad_kv_seqlens`** enables Flash with variable-length sequences
   without an explicit mask.
3. **GQA contrib op rejects `attention_bias`** — use standard ONNX
   `Attention` if you need bias with GQA.
4. **SM≥8.0** (Ampere+) required for Flash on all paths.
5. **Float bias is safer** than bool mask for complex attention patterns.
6. **MEA requires alignment** — `total_kv % 4 == 0` for bias tensors.

## Cross-references

- **Debugging memcpy:** `.agents/skills/debugging-memcpy/SKILL.md`
- **Profiling:** `.agents/skills/profiling-onnx-models/SKILL.md`
- **Reusable components:** `.agents/skills/reusable-components/SKILL.md`
