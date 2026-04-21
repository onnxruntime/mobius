# Tolerance Guidelines Reference

Detailed tolerance tables, debugging strategies, and per-dtype guidance for
numerical parity testing. See the main [SKILL.md](../SKILL.md) for the
overview.

## Tolerance table by model type

| Test type | Recommended rtol/atol |
|-----------|----------------------|
| Standard text models | `1e-3` / `1e-3` |
| Encoder-only (BERT) | `1e-3` / `1e-3` |
| Encoder-decoder (Whisper, BART, T5) | `1e-3` / `1e-3` |
| Multimodal models | `1e-2` / `1e-2` |
| Diffusion models (UNet, DiT, VAE) | `1e-3` / `1e-3` |
| Audio encoder models | `1e-3` / `1e-3` |
| Generation (token IDs) | Exact match |

Multimodal models use looser tolerances because the vision pipeline
introduces additional floating-point variance.

## `assert_logits_close` behavior

`assert_logits_close` uses `np.testing.assert_allclose(..., strict=True)`,
which checks shape, dtype, and value equality within tolerance.

## Tolerance failure checklist

If tolerances fail, verify these in order:

1. **Norm epsilon** — LayerNorm/RMSNorm eps must match HF config exactly
   (e.g., Whisper uses `1e-5`, not the default `1e-6`)
2. **Norm type** — Check if the model uses RMSNorm or LayerNorm.  OLMo-1B
   uses weight-free LayerNorm (not RMSNorm).  Using the wrong type causes
   max abs diff > 1.0.
3. **Q scaling order** — some models (Whisper) pre-scale Q before attention
   and pass `scale=1.0` to the op, which is numerically different from
   passing `scale=head_dim**-0.5`
4. **Attention scale** — some models (Granite) replace `1/sqrt(head_dim)` with
   a custom `attention_multiplier` from the config
5. **Scaling multipliers** — check HF config for `embedding_multiplier`,
   `logits_scaling`, `residual_multiplier` that aren't in standard Llama
6. **Residual pattern** — verify `residual + output * scale` vs
   `residual * scale + output` by reading HF source
7. **Weight loading** — compare ONNX initializers against HF state_dict to
   rule out name mapping bugs
8. **Float64 contamination** — numpy arrays created from config values default
   to float64; always use `dtype=np.float32`

## Debugging large logit differences

When max abs diff is large (> 0.5), run this diagnostic:

```python
import numpy as np
diff = np.abs(onnx_logits[0, -1] - hf_logits)
print(f"Max abs diff: {diff.max():.4f}")
print(f"Mean abs diff: {diff.mean():.4f}")
# If max > 0.5, it's likely a norm or scaling bug, not just floating-point
# If max > 10, weights are probably loaded to wrong parameters
```

Check the HF norm class directly:
```python
import inspect
from transformers.models.olmo.modeling_olmo import OlmoLayerNorm
print(inspect.getsource(OlmoLayerNorm))
```

Check for unextracted config fields:
```python
config = AutoConfig.from_pretrained("model-id")
for k, v in config.to_dict().items():
    if any(s in k for s in ("multiplier", "scaling", "factor", "epsilon")):
        print(f"{k}: {v}")
```

## Dtype-specific tolerance guidance

### fp32

Standard tolerance: `atol=1e-3, rtol=1e-3`. Target **100% token match** in
greedy generation.

### fp16

Looser tolerance: `atol=1e-2, rtol=1e-2`. May diverge after the first few
tokens in generation due to floating-point accumulation. Acceptable if logit
parity holds.

**fp16 Exp overflow:** `exp(x)` overflows to `inf` for `x > ~11.09` in
fp16. The Softplus activation (`log(1 + exp(x))`) and decay computation
`exp(-softplus(x))` are common overflow sites. Always upcast to float32
for Exp/Softplus in fp16 models:

```python
x_f32 = op.Cast(x, to=ir.DataType.FLOAT)
result = op.Exp(x_f32)
result = op.CastLike(result, x)  # cast back to x's dtype
```

### bf16

Same tolerance as fp16: `atol=1e-2, rtol=1e-2`. bf16 has the same exponent
range as fp32 (no overflow at 11.09) but much less precision (7-bit mantissa
vs 10-bit). bf16 does NOT need the Exp upcast workaround.

If a computation works in bf16 but not fp16, check for Exp overflow first.
