---
name: reusable-components
description: >
  Create or extend reusable ONNX building blocks in the mobius component
  library. Use when adding Attention, MLP, norm, RoPE, or embedding
  components; understanding parameter naming and nn.Module conventions;
  applying design principles (subclass over flags, model-agnostic);
  or wiring shared-weight / per-layer adapter patterns.
---

# Skill: Reusable Components

## When to use

Use this skill when creating or extending the building blocks that models are
composed from — attention layers, MLPs, normalisations, embeddings, RoPE
variants, and activations.

## Component library overview

All components live in `src/mobius/components/` and inherit from
`onnxscript.nn.Module`.  Each component's `forward(op, ...)` method builds
ONNX nodes via the `OpBuilder`.

```
components/
├── _activations.py       # get_activation(), SiLU module
├── _attention.py          # Multi-head / GQA attention with KV cache; Qwen35Attention (gated GQA)
├── _audio.py              # ConformerEncoder (NeMo subsampling, T5 bias, Conformer layers)
├── _common.py             # Embedding, Linear, LayerNorm, LayerNormNoAffine, GroupNorm, create_attention_bias
├── _gemma4_audio.py       # Gemma4 audio encoder: ClippableLinear, ConvSubsampling, SlidingWindowAttention
├── _conv.py               # Conv2d (2D convolution with bias and groups)
├── _decoder.py            # DecoderLayer (pre-norm residual block)
├── _encoder.py            # BertEmbeddings, EncoderAttention, EncoderLayer
├── _lora.py               # LoRALinear (base + per-adapter A/B/scale)
├── _mlp.py                # Gate-up-down MLP
├── _moe.py                # MoELayer, TopKGate, SparseMixerGate
├── _multimodal.py         # Projectors + InputMixer
├── _qwen3_vl_vision.py    # Qwen3-VL block-diagonal vision encoder
├── _gated_deltanet.py     # GatedDeltaNet (recurrent linear attention for Qwen3.5 hybrid)
├── _rms_norm.py           # RMSNorm, OffsetRMSNorm (1+weight), GatedRMSNorm (norm * SiLU gate)
├── _rotary_embedding.py   # RoPE variants (Default, Linear, Dynamic, Llama3, InterleavedMRope, ChunkedMRope)
├── _vision.py             # PatchEmbedding, VisionEncoder, VisionModel
└── _whisper.py            # Conv1d, WhisperAttention, WhisperDecoderLayer, WhisperEncoderLayer
```

Model files import shared primitives from `components/` and alias them with
an underscore prefix for local use:

```python
from mobius.components import Conv2d as _Conv2d, SiLU as _SiLU
```

Inside `src/mobius/components/_*.py`, import sibling primitives directly
(`mobius.components._common`, etc.) so a partially initialized public package
cannot create a circular import.

Model-specific compound blocks (e.g. `_TimestepEmbedding`, `_DiTBlock`,
`_ResNetBlock2D`) remain in the model files they belong to.

## How to create a new component

### 1. Define the class

```python
from onnxscript import nn
from onnxscript import OpBuilder


class MyComponent(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.weight = nn.Parameter([hidden_size])

    def forward(self, op: OpBuilder, hidden_states):
        return op.Mul(hidden_states, self.weight)
```

### 2. Parameter naming

Parameter names are **automatically set** from the attribute name by
`nn.Module.__setattr__`.  You do **not** need to pass `name=` when the
attribute name matches the desired ONNX name:

```python
# GOOD — name is automatically "weight"
self.weight = nn.Parameter([hidden_size])

# Only use name= when the attribute name differs from the desired ONNX name
self.patch_embedding = nn.Parameter(
    [out_ch, in_ch, kH, kW], name="patch_embedding.weight"
)
```

When the component is nested in a module tree, names are automatically
prefixed by parent attribute names:

```python
# In model: self.layer = MyComponent(...)
# Resulting ONNX name: "layer.weight"
```

**Critical:** Parameter names must be unique within a component.  If two
parameters share the same attribute name at different levels, one will
silently overwrite the other.

To create a parameter with precomputed data (e.g. frozen positional embeddings),
use the `data=` argument:

```python
import onnx_ir as ir
self.embed_positions = nn.Parameter(
    [max_positions, d_model],
    name="embed_positions.weight",
    data=ir.tensor(numpy_array),
)
```

Do **not** assign `_const_value` directly.

### 3. Export from `__init__.py`

Add to `src/mobius/components/__init__.py`:

```python
__all__ = [..., "MyComponent"]
from mobius.components._my_component import MyComponent
```

### 4. Write unit tests

Create `_my_component_test.py` alongside the source:

```python
from mobius._testing import create_test_builder, create_test_input

class TestMyComponent:
    def test_forward(self):
        comp = MyComponent(hidden_size=64)
        b, op, graph = create_test_builder()
        x = create_test_input(b, "x", [1, 10, 64])
        result = comp(op, x)
        b._adapt_outputs([result])
        assert graph.num_nodes() > 0

    def test_parameter_names(self):
        comp = MyComponent(hidden_size=64)
        names = [n for n, _ in comp.named_parameters()]
        assert "weight" in names
```

## Key components (concise)

| Component | Signature | Notes |
|-----------|-----------|-------|
| `Attention(config)` | MHA/GQA/MQA, optional QK norm, custom scale | Opset 23 `op.Attention` |
| `Qwen35Attention(config)` | Gated GQA with partial RoPE | `attn_output * sigmoid(gate)` |
| `MLP(config)` | gate_proj + up_proj + down_proj | Activation from `config.hidden_act` |
| `DecoderLayer(config)` | Pre-norm residual block | Subclass to customize norms |
| `GatedDeltaNet(config)` | Recurrent linear attention | Qwen3.5 hybrid; delta rule recurrence |
| `RMSNorm(h, eps)` | Opset 23 `RMSNormalization` | `OffsetRMSNorm` for `1+weight` variant |
| `LayerNorm(h, eps)` | `LayerNormalization` op | Check HF config for correct eps |
| `ClippableLinear(in, out)` | Linear + learned input/output clipping | Critical for Gemma4 vision/audio encoders |
| `Embedding(V, D)` | Gather on weight matrix | |

### Audio mask pattern

Audio encoders that handle variable-length inputs (e.g. Gemma4) use a
boolean mask pair:

- **Input:** `input_features_mask: BOOL [B, T]` — contiguous, right-padded
  mask indicating real vs padding frames.
- **Output:** `audio_features_mask: BOOL [B, T//stride]` — downsampled
  through conv stride (typically stride=4), passed to runtime for padding
  stripping before token replacement.

This is different from vision padding, which uses `(-1, -1)` sentinel
position IDs instead of an explicit mask.

> Read `references/component-examples.md` when you need detailed constructor
> arguments, Attention/MLP/norm variant code, ClippableLinear weight mapping,
> RoPE factory usage, shared-weight + per-layer adapter patterns, or the
> `op.Identity` pattern for exposing parameters as graph outputs.

## Design principles

1. **Favour subclasses over flags.** When a model family has a unique variant
   (e.g. Gemma's `weight + 1` norm), create a subclass rather than adding a
   boolean flag to the base class.

2. **Keep components model-agnostic.** A component should work for any model
   that has the right config fields.  Model-specific wiring belongs in the
   model module.

3. **One file per concern.** Attention in `_attention.py`, RoPE in
   `_rotary_embedding.py`, etc.  Tests co-located as `_*_test.py`.

4. **Reuse across model families.** The same `Attention` component is used by
   LLaMA, Mistral, Qwen, Phi, and others.  Only override when the
   architecture genuinely differs.

5. **Multiple reusable variants, not one-size-fits-all.** When models need
   different behaviour (e.g. MoE gates, projector types), create separate
   classes rather than cramming everything into one class with many branches.

6. **Comment generously with architecture context.** Annotate tensor shapes
   after ops (e.g. `# (N, num_heads, head_dim)`), explain multi-step
   computations (window reordering, RoPE, spatial merge), and document how
   the ONNX graph maps to the HuggingFace reference implementation.

7. **Match HuggingFace's precision behaviour.** Components must work with any
   compute dtype (float32, float16, bfloat16).  For numerically sensitive ops
   (`exp`, `softplus`, RMSNorm variance), upcast to float32 with
   `op.Cast(to=ir.DataType.FLOAT)`, compute, then cast back with `op.CastLike(result, input)`.
   For dtype-adaptive parameters, use `op.CastLike(param, reference)`.

8. **Prefer canonical ONNX operators.** Emit standard `BatchNormalization`,
   `RMSNormalization`, and activation ops when semantics match so ORT can fold
   and fuse them. Scale-free RMSNorm still needs a schema-valid 1-D scale and
   must preserve HF's fp32 variance semantics before casting back. Prefer
   `stash_type=FLOAT` over a decomposed graph when it matches HF, and verify
   normalize-in-fp32 → cast activation → apply gamma ordering. Keep a manual
   form when preventing an incorrect provider fusion is intentional and tested.

9. **Preserve public call compatibility.** Append new optional arguments after
   existing positional parameters and add a positional-call regression test.

## ONNX op patterns overview

Key patterns for building components:

- **Scalar constants:** Use `op.Constant(value_ints=[...])` for tensor inputs
- **CastLike:** Use `op.CastLike(param, activation)` for dtype-agnostic casting
- **fp32 upcast:** `op.Cast(to=FLOAT)` → compute → `op.CastLike(result, input)`
  for numerically sensitive ops (Exp, Softplus, RMSNorm variance)
- **Shape extraction:** `op.Shape(x, start=i, end=i+1)` — never `Gather(Shape(x), ...)`
- **ModuleList vs Sequential:** Use `nn.Sequential` for fixed chains,
  `nn.ModuleList` for custom iteration

> Read `references/onnx-op-patterns.md` when you need full code examples for
> scalar constants, CastLike, fp32 upcast tables, shape manipulation, module
> containers, conditional ops, or the `op.Identity` graph output pattern.

## Cross-references

- **Weight name alignment:** `.agents/skills/weight-name-alignment/SKILL.md`
- **Multimodal components:** `.agents/skills/multimodal-models/SKILL.md`
- **MoE components:** `.agents/skills/moe-models/SKILL.md`
- **Writing tests:** `.agents/skills/writing-tests/SKILL.md`
- **Rewrite rules:** `.agents/skills/writing-rewrite-rules/SKILL.md`
