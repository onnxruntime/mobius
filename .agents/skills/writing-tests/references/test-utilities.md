# Test Utilities Reference

Detailed API reference for mobius testing utilities, fixture patterns, and
test feed creation. See the main [SKILL.md](../SKILL.md) for the overview.

## Utility API reference

| Utility | Import path | Purpose |
|---------|-------------|---------|
| `OnnxModelSession(model)` | `_testing.ort_inference` | Save + load + run ONNX model via ONNX Runtime |
| `OnnxGenerator(session, config)` | `_testing.generation` | Multi-step greedy decoding loop |
| `load_torch_model(id)` | `_testing.torch_reference` | Load HuggingFace model + tokenizer |
| `torch_forward(model, ...)` | `_testing.torch_reference` | Single forward pass through HF model |
| `torch_generate_greedy(...)` | `_testing.generation` | Multi-token HF greedy generation |
| `assert_logits_close(a, b)` | `_testing.comparison` | Logit comparison with diagnostics |
| `assert_generation_match(a, b)` | `_testing.comparison` | Token-ID exact match assertion |

## `OnnxModelSession`

Wraps the build → save → load → run cycle for integration tests:

```python
from mobius._testing.ort_inference import OnnxModelSession

onnx_model = build(model_id, load_weights=True)
session = OnnxModelSession(onnx_model)
outputs = session.run(feed_dict)
```

## `OnnxGenerator`

Implements multi-step greedy decoding over an `OnnxModelSession`:

```python
from mobius._testing.generation import OnnxGenerator

generator = OnnxGenerator(session, config)
token_ids = generator.generate(input_ids, max_new_tokens=10, eos_token_id=...)
```

## Comparison functions

### `assert_logits_close(actual, expected, rtol, atol)`

Uses `np.testing.assert_allclose` with `strict=True` (checks shape + dtype).
On failure, prints diagnostic info including max/mean abs diff.

### `assert_generation_match(actual_ids, expected_ids)`

Exact match on token ID lists. Fails with a clear diff showing the first
divergent position.

## Test feed creation: symbolic dimensions

ONNX models export symbolic batch/sequence dimensions. When feeding the
model for ORT inference:

- **Recurrent state batch dim must match input batch dim** — unlike KV
  cache (which initialises to zeros and grows), recurrent state tensors
  have a fixed `(B, ...)` shape. Feeding batch=0 produces a zero-sized
  carry state that collapses the Scan output.
- **Scan carry state is not KV cache** — do not copy the KV cache
  zero-initialisation pattern for recurrent state; the batch dimension
  must be the actual inference batch size.

```python
# WRONG — batch=0 zeros out Scan carry
past_state = np.zeros((0, num_heads, d_k, d_v), dtype=np.float32)

# CORRECT — must match actual batch size
batch_size = input_ids.shape[0]
past_state = np.zeros((batch_size, num_heads, d_k, d_v), dtype=np.float32)
```

## ONNX function registration

When renaming a custom function's `op_type` (e.g. `CausalConvNdWithState`
→ `CausalConvWithState`), the function must be re-registered under the new
name in ORT's function decomposition list. ORT needs the function embedded
in `model.functions` to decompose the custom op before execution.

**Checklist when renaming a custom function:**
1. Rename the Python factory function and the `ir.Function.name`
2. Update all call sites that reference the old op_type string
3. Update any integration tests that check the op_type name
4. Verify the function appears in `onnx_model.functions` after build
