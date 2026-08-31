# Contributing to mobius

## Development setup

```bash
pip install -e ".[transformers]"
pip install pytest
```

## Running tests

Unit tests live in `tests/build_graph` and verify graph construction
for all supported architectures using tiny synthetic configs (no network needed).

Integration tests live in `tests/integration_test.py` and verify numerical
accuracy against HuggingFace PyTorch models (requires network and memory).

```bash
# Unit tests (fast, parallel with -n auto)
pytest tests/build_graph -v -n auto

# Run a single model type
pytest tests/build_graph -k "phi4mm"

# Integration tests (slow, downloads models)
pytest tests/integration_test.py -m integration -k "qwen2.5-0.5b"
```

## Coding conventions

### Imports

**onnxscript**: Import the `nn` namespace, not individual symbols.

```python
# Good
from onnxscript import nn, OpBuilder

class MyLayer(nn.Module):
    def __init__(self):
        self.param = nn.Parameter([64], name="weight")

    def forward(self, op: OpBuilder, x):
        ...

# Bad
from onnxscript.nn import Module, Parameter
from onnxscript._internal.builder import OpBuilder
```

**Components in models**: Import from the public `components` package, not
private submodules.

```python
# Good (in models/)
from mobius.components import (
    Attention,
    DecoderLayer,
    Embedding,
    Linear,
    MLP,
    RMSNorm,
    create_attention_bias,
    initialize_rope,
)

# Bad (in models/)
from mobius.components._common import Linear
from mobius.components._rms_norm import RMSNorm
```

Within the `components/` package itself, private cross-imports
(e.g. `from mobius.components._common import Linear`) are fine.

### File organization

- **Source**: `src/mobius/`
- **Unit tests**: `tests/build_graph` (graph construction, no weights)
- **Integration tests**: `tests/integration_test.py` (numerical accuracy vs PyTorch)
- **Test helpers**: `src/mobius/_testing/`

### Style

- Use `from __future__ import annotations` in every module.
- Follow [PEP 8](https://peps.python.org/pep-0008/) conventions.
- Use [ruff](https://docs.astral.sh/ruff/) for linting and formatting.
- Prefer type annotations for function signatures.
- Only comment code that needs clarification; avoid obvious comments.
- **Move all imports to the top of the module.** Inline/local imports are only
  acceptable for: (a) avoiding circular dependencies, (b) optional dependencies
  that may not be installed (e.g. `transformers`, `safetensors`), or
  (c) docstring examples.

### Zero protobuf operations

**Never use `onnx.helper`, `onnx.TensorProto`, `onnx.ModelProto`, or any
`onnx_proto` / protobuf construction APIs anywhere in this codebase.**

Always use `onnx_ir` to construct ONNX models (`onnxscript.ir` is a
deprecated alias — do not use it):

```python
# Good — onnx_ir APIs
import onnx_ir as ir

graph = ir.Graph(...)
node = ir.Node("", "MatMul", inputs=[a, b])
func = ir.Function(domain="com.microsoft", name="MyOp")
tensor = ir.Tensor(np.array([1.0], dtype=np.float32))

# Bad — protobuf construction (forbidden)
import onnx
node = onnx.helper.make_node("MatMul", inputs=["a", "b"], outputs=["c"])
graph = onnx.helper.make_graph(nodes, "my_graph", inputs, outputs)
tensor = onnx.TensorProto()
```

This rule applies to all code: models, components, tasks, rewrite rules,
tests, fixtures, and utilities. No exceptions.

### Model architecture modules

Each model file in `models/` should:

1. Import components from the public `mobius.components` package.
2. Subclass `nn.Module` for any custom layers.
3. Subclass `CausalLMModel` (or another base) for the top-level model class.
4. Implement `preprocess_weights()` if the architecture has non-standard
   weight names or fused projections (e.g. qkv_proj → q/k/v_proj).

### Adding a new model architecture

1. Create `models/<name>.py` with the model class.
2. Export it from `models/__init__.py`.
3. Register it in `_registry.py`'s `_create_default_registry()`.
4. Add a tiny config entry to the appropriate list in `tests/_test_configs.py`.
5. Add a small HuggingFace model to `_TEXT_MODELS` in `tests/integration_test.py`.

### Adding a new task

1. Create a subclass of `ModelTask` in `tasks/__init__.py` (or a new file).
2. Register it in `TASK_REGISTRY`.
3. Add tests in `tasks/_task_test.py`.

### Adding vision/multimodal components

Vision components live in `components/_vision.py`, multimodal bridging
components in `components/_multimodal.py`.

**Vision encoder patterns**:
- Use `VisionLayerNorm` (LayerNorm, not RMSNorm) for vision encoders.
- Use `VisionAttention` for bidirectional attention (no causal mask, no KV cache).
- Use `_VisionLinear` (with bias) for vision projections.
- `LayerNormalization` takes `epsilon` as an attribute (float), not an input value.

**Multimodal model patterns**:
- Compose `VisionModel`, `MultiModalProjector`, and a text model (e.g. `Gemma3CausalLMModel`).
- Use `InputMixer` to replace placeholder tokens with vision embeddings.
- Use `VisionLanguageTask` which adds `pixel_values` to the graph inputs.
- Register the multimodal model in `_create_default_registry()` with a distinct key
  (e.g. `gemma3_multimodal` vs `gemma3` for text-only).

**MoE model patterns**:
- Use `MoELayer` from `components/_moe.py` which includes `MoEGate` (TopK routing)
  and a list of expert `MLP` instances.
- MoE decoder layers replace the standard `DecoderLayer`; create `MoEDecoderLayer`
  and `MoETextModel` that use `MoELayer` instead of `MLP`.
- MoE models require `num_local_experts` and `num_experts_per_tok` in config.

## Quality checklist (definition of done)

Before a new model PR is considered **done**, all items in the quality
checklist must be satisfied (or explicitly waived with a written reason in
the PR description).  The full checklist with explanations lives in the
[quality-checklist skill](.agents/skills/quality-checklist/SKILL.md).

### Summary checklist

**Code quality**
- [ ] Microsoft MIT copyright header in every new file
- [ ] Descriptive class docstring (`default_task`, `category` set if non-standard)
- [ ] `preprocess_weights()` verified (no dropped or mangled weight names)
- [ ] No protobuf construction APIs used anywhere in new code
- [ ] Automated code review run; all findings resolved or dismissed

**Test coverage (L1 – L5)**
- [ ] **L1** — Graph build: entry in `tests/_test_configs.py` + weight-alignment test passes
- [ ] **L2** — Config compatible: YAML test case in `testdata/cases/` with `test_model_id`
- [ ] **L3** — Synthetic parity: test in `tests/synthetic_parity_test.py` passes
- [ ] **L4** — Golden match: golden file in `testdata/golden/` and `e2e_golden_test` passes
- [ ] **L5** — Generation verified: generation golden file committed and test passes

> L4 and L5 are the most commonly missed. Graph-build tests (L1) never
> execute the graph with real data — only L4/L5 catch wrong norms, missing
> scaling multipliers, and runtime shape errors.

**Multi-dtype correctness**
- [ ] fp32 (target: exact greedy token match)
- [ ] fp16 (target: logit parity `atol=1e-2`)
- [ ] bf16 (target: logit parity `atol=1e-2`)

**Runtime and deployment**
- [ ] CLI build: `mobius build --model <hf-model-id> /tmp/out` succeeds
- [ ] ORT GenAI fast lane passes on the pinned latest stable runtime:
  `pytest tests/ort_genai_e2e_test.py -m ort_genai_fast -v`
- [ ] Runtime-supported real routes declare `ort_genai` evidence in their
  `testdata/cases/` YAML and pass:
  `pytest tests/gguf_small_model_runtime_integration_test.py -m ort_genai_real -v`
- [ ] Foundry Local: exported package loads and responds to a short prompt
- [ ] Olive quantization: INT4/INT8 quantization runs to completion; quantized model produces coherent output

**Documentation**
- [ ] Class docstring (first paragraph) describes the model family and HuggingFace class
- [ ] README model table updated (if this is a significant new model)
