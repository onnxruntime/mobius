# Common ONNX Op Patterns — Detailed Reference

## Scalar constants

Many ONNX ops require tensor inputs, not Python scalars:

```python
# K for TopK must be a 1-D tensor
k = op.Constant(value_ints=[2])
values, indices = op.TopK(logits, k, axis=-1)

# Integer constants
one = op.Constant(value_int=1)

# Float constants
eps = op.Constant(value_float=1e-6)
```

## Dtype-agnostic casting with `CastLike`

When a parameter or constant needs to match an activation tensor's dtype
without knowing what it is at graph-build time, use `op.CastLike`:

```python
# GOOD — adapts to whatever dtype hidden_states has
scale = op.CastLike(op.Constant(value_float=1e-6), hidden_states)
```

**When `op.Cast(to=...)` IS appropriate:** converting between fundamentally
different types (e.g. int64 position_ids to float for arithmetic, or float
timesteps to the model's compute type), and for the fp32 upcast pattern
below.

## Precision-sensitive ops: fp32 upcast pattern

Some operations are numerically unstable in float16/bfloat16 and must run
in float32 to match HuggingFace's behaviour.  The pattern is:
**upcast → compute → cast back**.

```python
# Upcast inputs to fp32 for numerically sensitive exp/softplus
dt_f32 = op.Cast(dt, to=ir.DataType.FLOAT)
dt_f32 = op.Softplus(dt_f32)
a_neg = op.Neg(op.Exp(op.Cast(self.A_log, to=ir.DataType.FLOAT)))
...
# Cast output back to input dtype
y = op.CastLike(y_f32, x)
```

**Operations that need fp32 (based on HuggingFace source):**

| Op | Why | HF pattern |
|----|-----|-----------|
| `Exp` on A_log/decay | Overflow/underflow in fp16 range | `self.A_log.float()` |
| `Softplus` (dt) | Uses exp internally | `softplus(dt + dt_bias)` stays in fp32 context |
| `Exp(dt * A)` (discretisation) | Exponential of product | `A.to(dtype=torch.float32)` |
| SSM state update | Accumulates over many steps | `hidden_states.float()`, `B.float()`, `C.float()` |
| RMSNorm variance | Small values squared then averaged | `hidden_states.to(torch.float32)` |
| GatedRMSNorm (SiLU + norm) | Both gate and variance need fp32 | `gate.to(torch.float32)` |

**When fp32 upcast is NOT needed:**

- Linear projections (`MatMul`) — handled by the runtime
- SiLU activation on conv output — stays in model dtype in HF
- Standard attention — ONNX `Attention` op handles precision internally
- `RMSNormalization` op — has `stash_type=1` (default) which auto-upcasts
  the variance computation to fp32

**Rule of thumb:** Check the HuggingFace source for `.float()` or
`.to(torch.float32)` calls.  Every such call indicates an fp32 upcast
region that the ONNX component must replicate with explicit
`op.Cast(to=ir.DataType.FLOAT)` ... `op.CastLike(result, input)` bracketing.

## Shape manipulation

Use `op.Shape` with `start` and `end` attributes to extract specific
dimensions directly — do **not** use `Gather(Shape(x), index)`:

```python
# GOOD — single Shape node with start/end
batch_size = op.Shape(x, start=0, end=1)   # 1-D [1]-element tensor
seq_len    = op.Shape(x, start=1, end=2)
hidden_dim = op.Shape(x, start=2, end=3)

# BAD — unnecessary Gather
batch_size = op.Gather(op.Shape(x), [0], axis=0)
```

Building dynamic shapes for Reshape/Concat:

```python
new_shape = op.Concat(batch_size, hidden_dim, op.Constant(value_ints=[-1]), axis=0)
reshaped = op.Reshape(x, new_shape)
```

Since `Shape(start, end)` returns a 1-D tensor, it can be passed directly
to ops expecting 1-D shape inputs (e.g. `Slice` starts/ends, `Reshape`,
`Concat` for shape building) without intermediate `Reshape` calls.

## Conditional operations

```python
mask = op.Equal(input_ids, op.Constant(value_int=token_id))
result = op.Where(mask, true_value, false_value)
```

## Module lists and sequential containers

Use `nn.ModuleList` to register a list of child modules. It automatically
registers children with numeric keys (`"0"`, `"1"`, ...) and supports
iteration, indexing, and `len()`:

```python
# GOOD — nn.ModuleList
self.layers = nn.ModuleList(
    [DecoderLayer(config) for _ in range(config.num_hidden_layers)]
)

# BAD — manual setattr loop
self.layers = [DecoderLayer(config) for _ in range(config.num_hidden_layers)]
for i, layer in enumerate(self.layers):
    setattr(self, f"layers.{i}", layer)
```

For sequential containers where children should be called in order (e.g.
matching HF `nn.Sequential`), use `nn.Sequential`. It subclasses
`nn.ModuleList` and adds automatic forward chaining:

```python
from mobius.components import Linear, SiLU

# nn.Sequential chains forward calls: SiLU → Linear
self.img_mod = nn.Sequential(SiLU(), Linear(dim, 6 * dim))

# Clean call — output chains through each child
result = self.img_mod(op, temb)  # equivalent to Linear(SiLU(temb))
```

`nn.Sequential` produces the same parameter names as `nn.ModuleList`
(`img_mod.0.weight`, `img_mod.1.weight`). The key implementation detail:
it overrides `_set_name` to keep children with simple "0", "1" names
(not fully-qualified), because `__call__` already pushes the parent name
onto the scope stack.

**When to use which:**
- `nn.Sequential` — children are called in a fixed chain (e.g. `to_out`,
  modulation layers, FFN with activation gaps)
- `nn.ModuleList` — children need custom iteration logic (e.g. decoder
  layers with residual connections, down/up blocks with skip connections)

For non-consecutive indices (e.g. matching HF `nn.Sequential` with
activation/dropout layers at skipped positions), include parameter-free
placeholder modules to fill the gaps:

```python
class _NoOpModule(nn.Module):
    """Placeholder for HF Dropout (no params, identity at inference)."""
    def forward(self, op, x):
        return x

# Matches HF net.0.proj.weight, net.2.weight (Dropout at index 1)
self.net = nn.Sequential(
    _GELUGate(dim, inner_dim * 2),  # index 0
    _NoOpModule(),                   # index 1 (Dropout placeholder)
    Linear(inner_dim, dim),          # index 2
)
result = self.net(op, x)  # chains: GELUGate → NoOp → Linear
```

If `nn.Sequential` is not available, fall back to `nn.ModuleList` with
explicit indexing:

```python
self.img_mod = nn.ModuleList([SiLU(), Linear(dim, 6 * dim)])
# Manual chaining:
result = self.img_mod[1](op, self.img_mod[0](op, temb))
```

## Exposing parameters as graph outputs

Sometimes a generation loop needs access to model weights for external
computation (e.g. embedding lookups in numpy). Use `op.Identity()` to
expose a parameter as a graph output without affecting the initializer
name used for weight loading:

```python
class MyModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Stacked weight exposed for external lookup
        self.stacked_embedding = nn.Parameter([num_groups, vocab, hidden])

    def forward(self, op, ...):
        # Use Identity to create a separate output value.
        # This prevents the optimizer from renaming the initializer
        # when the task sets a custom output name.
        embeddings_out = op.Identity(self.stacked_embedding)
        return logits, present_key_values, embeddings_out
```

In the task, you can safely rename the Identity output:

```python
# Safe — Identity separates the output name from the initializer name
embeddings_out.name = "codec_embeddings"
graph.outputs.append(embeddings_out)
```

**Important:** Without the Identity node, the optimizer may fold the
reference and setting `output.name = "..."` would rename the
initializer itself, breaking `preprocess_weights` name mapping.

The generation loop extracts the weights once via a dummy inference:

```python
weights = session.run(dummy_inputs)["codec_embeddings"]  # (N, vocab, H)
# Use as numpy lookup: embed = weights[step, code_id, :]
```
