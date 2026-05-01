---
name: custom-functions
description: >
  Use this skill when calling custom ONNX functions from components or
  models — e.g. LinearAttention, CausalConvWithState,
  PackedMultiHeadAttention. Covers the op.call() API, static vs parametric
  functions, function auto-registration, and the ir.Function factories in
  src/mobius/functions/.
---

# Skill: Custom ONNX Functions (`op.call()`)

## When to use

Use this skill when:
- A component needs to invoke a custom ONNX operator defined as an
  `ir.Function` (e.g. LinearAttention, CausalConvWithState)
- You are adding a new hybrid model that mixes standard attention with
  recurrent/state-space layers
- You need to understand the difference between `op.call()` for
  ir.Functions vs `_domain="com.microsoft"` for contrib ops
- You want to create a new custom ONNX function factory

## Quick reference

```python
# 1. Import the factory from mobius.functions
from mobius.functions import linear_attention, causal_conv_nd_with_state

# 2. Create the ir.Function at __init__ time (parametric — bakes in config)
self._attn_fn = linear_attention(
    update_rule="gated_delta", scale=1.0/sqrt(head_dim),
    stash_type=ir.DataType.FLOAT,
)
self._conv_fn = causal_conv_nd_with_state(
    kernel_size=4, channels=conv_dim, ndim=1, activation="silu",
)

# 3. Call it in forward() via op.call()
output, new_state = op.call(
    self._attn_fn,
    query, key, value, recurrent_state, decay, beta,
    scale=1.0 / (head_dim ** 0.5),
    q_num_heads=num_heads,
    kv_num_heads=kv_num_heads,
    update_rule="gated_delta",
    _outputs=2,
)
```

## How `op.call()` works

`op.call(ir_function, *args, **attr_kwargs)` emits an ONNX node that
invokes the given `ir.Function`:

| Parameter | Description |
|-----------|-------------|
| `ir_function` | An `ir.Function` object (the custom op definition) |
| `*args` | Positional ONNX values — the op's inputs |
| `**attr_kwargs` | ONNX attributes passed as keyword arguments |
| `_outputs=N` | Special kwarg: number of outputs to expect (default 1) |

**Key behaviour:** `op.call()` auto-registers the function on the
`GraphBuilder`, which collects it in `builder.functions`. The task layer
then passes all collected functions to `_make_model(graph,
builder.functions.values())`, which attaches them to the `ir.Model`.
You never need to manually register functions.

## Static vs parametric functions

There are two kinds of custom functions in mobius:

### Static functions (config-independent)

These have a fixed function body that works for any model configuration.
They are registered **globally** on every model via
`register_function_bodies()`.

| Function | Domain | Description |
|----------|--------|-------------|
| `PackedMultiHeadAttention` | `com.microsoft` | Block-diagonal attention from cu_seqlens |
| `SkipLayerNormalization` | `com.microsoft` | Fused skip + LayerNorm |
| `SkipSimplifiedLayerNormalization` | `com.microsoft` | Fused skip + simplified LayerNorm |

Static factories take **no arguments**:

```python
from mobius.functions import packed_multi_head_attention
fn = packed_multi_head_attention()  # Returns ir.Function
```

You typically don't call these directly — they're attached to the model
automatically and invoked via rewrite rules.

### Parametric functions (config-dependent)

These bake model-specific parameters (kernel size, channel count, head
counts) into the function body. Each model instance gets its own
specialized `ir.Function`.

| Function | Domain | Config parameters |
|----------|--------|-------------------|
| `CausalConvWithState` | `com.microsoft` | `kernel_size`, `channels`, `ndim`, `activation` |
| `LinearAttention` | `com.microsoft` | `update_rule`, `scale`, `stash_type` |

Parametric factories require arguments:

```python
from mobius.functions import linear_attention, causal_conv_nd_with_state

attn_fn = linear_attention(
    update_rule="gated_delta",
    scale=1.0 / math.sqrt(head_dim),
    stash_type=ir.DataType.FLOAT,
)

conv_fn = causal_conv_nd_with_state(
    kernel_size=4,
    channels=128,
    ndim=1,
    activation="silu",
)
```

Components create these at `__init__` time and store them as instance
attributes (`self._attn_fn`, `self._conv_fn`).

## Pattern: Using `op.call()` in a component

Here is the complete pattern, based on real components in the codebase:

### Step 1: Import and create the function at `__init__`

```python
from mobius.functions import causal_conv_nd_with_state, linear_attention

class MyRecurrentLayer(nn.Module):
    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        # Parametric: config values baked into the function body
        self._conv_fn = causal_conv_nd_with_state(
            kernel_size=config.conv_kernel,
            channels=config.intermediate_size,
            ndim=1,
            activation="silu",
        )
        self._attn_fn = linear_attention(
            update_rule="gated_delta",
            scale=1.0 / math.sqrt(config.head_dim),
            stash_type=ir.DataType.FLOAT,
        )
        # ... other parameters ...
```

### Step 2: Call the function in `forward()`

```python
    def forward(self, op, hidden_states, conv_state, recurrent_state):
        # ... compute query, key, value, etc. ...

        # CausalConvWithState: returns (output, updated_conv_state)
        conv_out, new_conv_state = op.call(
            self._conv_fn,
            input_val,         # (B, D, T) channels-first
            self.weight,       # (D, 1, K) depthwise kernel
            conv_bias,         # (D,) bias
            conv_state,        # (B, D, K-1) carry state
            activation="silu",
            _outputs=2,
        )

        # LinearAttention: returns (output, updated_recurrent_state)
        attn_out, new_recurrent_state = op.call(
            self._attn_fn,
            query, key, value,
            recurrent_state,   # (B, num_heads, d_k, d_v)
            decay,             # (B, T, num_heads)
            beta,              # (B, T, num_heads) — for gated_delta
            scale=1.0 / math.sqrt(self.head_dim),
            q_num_heads=self.num_heads,
            kv_num_heads=self.kv_num_heads,
            update_rule="gated_delta",
            _outputs=2,
        )

        return output, new_conv_state, new_recurrent_state
```

### Step 3: No manual registration needed

The task layer handles registration automatically:

```python
# Inside CausalLMTask.build() (tasks/_causal_lm.py):
model = _make_model(graph, builder.functions.values())
# All ir.Functions used via op.call() are collected and registered here
```

## `op.call()` vs `_domain="com.microsoft"` vs `op.op()`

There are three distinct patterns for custom/contrib ops. Don't confuse them:

| Pattern | Context | Example |
|---------|---------|---------|
| `op.call(ir_function, ...)` | Component `forward()` | LinearAttention, CausalConvWithState |
| `op.MoE(..., _domain="com.microsoft")` | Component `forward()` | Fused MoE contrib op |
| `op.op("OpName", ..., domain="com.microsoft")` | Rewrite rules only | PackedMultiHeadAttention in rewrites |

### When to use each

- **`op.call(ir_function, ...)`** — Use when the op has an `ir.Function`
  body (a standard-ONNX decomposition). The function body serves as both
  a semantic spec and a fallback for runtimes without native kernels.
  This is the preferred pattern for new custom ops.

- **`op.MoE(..., _domain="com.microsoft")`** — Use for ORT contrib ops
  that have no `ir.Function` body (e.g. fused MoE, GroupQueryAttention).
  These rely on the runtime having a native kernel.

- **`op.op("OpName", ..., domain=...)`** — Use only inside rewrite rule
  `rewrite()` methods, where the `op` parameter is an IR tape builder
  (not the same `OpBuilder` used in component `forward()` methods).

## Available function factories

All factories live in `src/mobius/functions/`:

```
functions/
├── __init__.py                        # Registry + register_function_bodies()
├── causal_conv.py                     # causal_conv_nd_with_state(), causal_conv1d_with_state()
├── linear_attention.py                # linear_attention()
├── packed_multi_head_attention.py     # packed_multi_head_attention()
└── skip_layer_normalization.py        # skip_layer_normalization(), skip_simplified_layer_normalization()
```

Import from the public API:

```python
from mobius.functions import linear_attention, causal_conv_nd_with_state
```

## Creating a new custom function

To add a new ir.Function factory:

### 1. Create the function file

```python
# src/mobius/functions/my_custom_op.py
from __future__ import annotations

import onnx_ir as ir
from onnxscript._internal import builder
from mobius._constants import OPSET_VERSION

DOMAIN = "com.microsoft"


def my_custom_op(*, param1: int, param2: float) -> ir.Function:
    """Build an ir.Function for MyCustomOp.

    Args:
        param1: Description of parameter.
        param2: Description of parameter.
    """
    b = builder.GraphBuilder(OPSET_VERSION)
    # Define inputs
    x = b.input("x", ir.DataType.FLOAT, shape=["B", "T", param1])
    state = b.input("state", ir.DataType.FLOAT, shape=["B", param1])

    # Build the function body using standard ONNX ops
    op = b.op
    result = op.Add(x, state)
    new_state = op.ReduceMean(x, keepdims=False)

    b.output(result)
    b.output(new_state)

    return b.create_function(
        DOMAIN,
        "MyCustomOp",
        # Declare attributes the caller can pass
        attributes={"scale": ir.AttributeType.FLOAT},
    )
```

### 2. Export from `__init__.py`

For **static** functions (no config parameters), add to `_FUNCTION_BUILDERS`:

```python
_FUNCTION_BUILDERS[(_DOMAIN, "MyCustomOp", "")] = my_custom_op
```

For **parametric** functions (config-dependent), just export the factory:

```python
from mobius.functions.my_custom_op import my_custom_op

__all__ = [..., "my_custom_op"]
```

### 3. Write tests

Create `src/mobius/functions/my_custom_op_test.py` co-located with the source.

## Naming conventions

| Python | ONNX |
|--------|------|
| Factory function: `snake_case` (e.g. `linear_attention`) | Op type: `PascalCase` (e.g. `LinearAttention`) |
| Domain: `DOMAIN = "com.microsoft"` | All custom functions use the `com.microsoft` domain |
| Instance attribute: `self._attn_fn` | Underscore prefix for private ir.Function references |

## Models that use custom functions

These models use `op.call()` with ir.Functions. Use them as reference
implementations:

| Model | Component | Functions used |
|-------|-----------|----------------|
| Qwen3.5 (hybrid) | `GatedDeltaNet` | `LinearAttention`, `CausalConvWithState` |
| MiniMax | `LightningAttention` | `LinearAttention` |
| SmolLM3 / NemotronH | `Mamba2Block` | `LinearAttention`, `CausalConvWithState` |
| Qwen3-VL / Qwen2.5-VL | `_Qwen3VLVisionAttention` | `PackedMultiHeadAttention` |

## Cross-references

- **Component design:** `.agents/skills/reusable-components/SKILL.md`
- **Adding models:** `.agents/skills/adding-a-new-model/SKILL.md`
  (see "Hybrid models" section)
- **MoE fused ops:** `.agents/skills/moe-models/SKILL.md`
  (uses `_domain="com.microsoft"` directly, NOT `op.call()`)
- **Rewrite rules:** `.agents/skills/writing-rewrite-rules/SKILL.md`
  (uses `op.op()` in rewrite context, NOT `op.call()`)
