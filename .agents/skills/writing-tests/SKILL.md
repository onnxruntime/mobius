---
name: writing-tests
description: >
  Use this skill when writing or modifying tests for mobius models and
  components. Covers the L1–L5 confidence system: unit tests (graph
  construction from tiny configs), integration tests (numerical parity
  with HuggingFace), golden tests (pre-computed reference comparison),
  and generation tests (multi-token output verification). Includes test
  commands, shared config infrastructure, testing utilities, tolerance
  guidelines, and common pitfalls.
---

# Skill: Writing Tests

## When to use

Use this skill whenever you need to:
- Add tests for a new model or component
- Write integration tests comparing ONNX output to HuggingFace
- Debug numerical parity failures
- Add golden test coverage (L4/L5)
- Understand the test infrastructure and conventions

## References

Detailed material is extracted into reference files:

- **Read [`references/test-examples.md`](references/test-examples.md)** when
  you need full code examples for any test type, YAML test case format,
  golden file format, or step-by-step coverage instructions.
- **Read [`references/tolerance-guidelines.md`](references/tolerance-guidelines.md)**
  when debugging numerical mismatches, choosing tolerance values, or
  investigating dtype-specific bugs.
- **Read [`references/test-utilities.md`](references/test-utilities.md)** when
  using `OnnxModelSession`, `OnnxGenerator`, comparison functions, or
  dealing with test feed creation for symbolic dimensions.

---

## Confidence levels (L1–L5)

Each level is detected and counted **independently**.  A model can pass L3
without passing L2, or have L4 golden data without passing L3.

| Level | Name | What it verifies | Data source |
|-------|------|-----------------|-------------|
| **L1** | Graph builds | ONNX graph builds from a tiny synthetic config | `tests/_test_configs.py` + `tests/build_graph_test.py` |
| **L2** | Config compatible | Full-size HF config produces a valid graph | `test_model_id` in YAML test case (`testdata/cases/`) |
| **L3** | Synthetic parity | Random-weight forward pass matches HF numerically | `tests/integration_test.py` parametrized tests |
| **L4** | Golden match | Real-weight prefill logits match pre-computed reference | `testdata/golden/<cat>/<model>.json` |
| **L5** | Generation verified | Full multi-token generation matches golden output | `testdata/golden/<cat>/<model>_generation.json` |

The dashboard shows **per-flag counts** — a model is counted at every level
it passes, not just the highest.  L1 equals the total number of registered
models.

---

## Test commands

```bash
# All non-integration tests (fast, no downloads)
python -m pytest tests/build_graph_test.py tests/cli_test.py src/ -q \
  -k "not phi4mm and not apply_weights_unknown" --tb=short

# Representative models only (~5 seconds)
python -m pytest tests/build_graph_test.py --fast

# Single model type
python -m pytest tests/build_graph_test.py -k "phi4mm"

# Integration tests (slow, downloads models)
python -m pytest tests/integration_test.py -m integration -k "qwen2.5-0.5b"

# L4/L5 golden tests
python -m pytest tests/e2e_golden_test.py -m golden --level L4 -v
python -m pytest tests/e2e_golden_test.py -m golden --level L5 -v

# Generate golden data
python scripts/generate_golden.py --level L4 --filter 'my-model*'
```

---

## Test file layout

```
tests/
├── build_graph_test.py       # L1: graph construction (no weights)
├── _test_configs.py          # shared model configs for all tests
├── integration_test.py       # L3: real-weight numerical parity
├── e2e_golden_test.py        # L4 + L5: golden file comparison
├── yaml_schema_test.py       # YAML test case schema validation
├── weight_alignment_test.py  # L1: preprocess_weights correctness
└── arch_validation_test.py   # L2: full HF config graph build

testdata/
├── cases/                    # YAML test case definitions (L2, L4, L5)
└── golden/                   # Pre-computed reference outputs
```

---

## Shared test configuration

All model configs live in `tests/_test_configs.py`, organized by category:

| List | Test class | Task type |
|------|-----------|-----------|
| `CAUSAL_LM_CONFIGS` | `TestBuildGraph` | text-generation |
| `ENCODER_CONFIGS` | `TestBuildEncoderGraph` | feature-extraction |
| `SEQ2SEQ_CONFIGS` | `TestBuildSeq2SeqGraph` | seq2seq |
| `VISION_CONFIGS` | `TestBuildVisionGraph` | image-classification |
| `DETECTION_CONFIGS` | `TestBuildDetectionGraph` | object-detection |

Each entry is a 3-tuple: `(model_type, config_overrides, is_representative)`.

- **`is_representative=True`**: Models with unique behaviour (custom class,
  softcapping, MoE, ALiBi, etc.). Always tested.
- **`is_representative=False`**: Simple aliases. Skipped with `--fast`.
- **Auto-generation**: Text-generation models with no explicit entry get
  `(model_type, {}, False)` automatically from the registry.

To add a new model:
```python
CAUSAL_LM_CONFIGS: list[tuple[str, dict, bool]] = [
    ("my_model", {"hidden_act": "gelu", "attn_qkv_bias": True}, True),
]
```

---

## L1: Graph build tests

Located in `tests/build_graph_test.py`. Uses tiny synthetic configs
(64 hidden, 2 layers, 256 vocab) — no weights, no network.

The framework checks: inputs exist (`input_ids`, `attention_mask`,
`position_ids`), outputs exist (`logits`, KV cache), and initializers
are present.

VLM/audio models use dedicated test methods tracked in
`_SPECIALIZED_TEST_MODEL_TYPES`.

### Weight alignment tests

`tests/weight_alignment_test.py` verifies `preprocess_weights()` maps
HF state dict keys to ONNX initializer names correctly. Catches bugs
like prefix replacement corrupting names or fused weight names being dropped.

### Rewrite rule unit tests

Place rewrite rule tests **next to** the source file:
- Source: `src/mobius/rewrite_rules/_packed_attention.py`
- Test: `src/mobius/rewrite_rules/_packed_attention_test.py`

---

## L2: Config compatibility

Detected from the `test_model_id` field in YAML test cases. To add L2:
create `testdata/cases/<category>/my-model.yaml` with `test_model_id`
set to a real HF model ID.

---

## L3: Integration tests

Located in `tests/integration_test.py`. Parametrized with
`(model_id, trust_remote_code)`. Prefer models ≤ 1B, publicly accessible,
one per distinct model class.

> Read [`references/test-examples.md`](references/test-examples.md) for
> full prefill/decode/generation code patterns.

---

## L4 + L5: Golden tests

Compare ONNX outputs against pre-computed golden files in `testdata/golden/`.

| Level | File pattern | Contents |
|-------|-------------|----------|
| L4 | `<model>.json` | Prefill top-1/top-2 token IDs + logit summary |
| L5 | `<model>_generation.json` | Prompt + generated token IDs + text |

> Read [`references/test-examples.md`](references/test-examples.md) for
> YAML format, golden file format, and step-by-step coverage instructions.

---

## Testing utilities

| Utility | Purpose |
|---------|---------|
| `OnnxModelSession(model)` | Save + load + run ONNX model |
| `OnnxGenerator(session, config)` | Multi-step greedy decoding |
| `load_torch_model(id)` | Load HF model + tokenizer |
| `torch_forward(model, ...)` | Single forward pass |
| `torch_generate_greedy(...)` | Multi-token HF generation |
| `assert_logits_close(a, b)` | Logit comparison with diagnostics |
| `assert_generation_match(a, b)` | Token-ID exact match |

> Read [`references/test-utilities.md`](references/test-utilities.md) for
> detailed API, feed creation patterns, and ONNX function registration.

---

## Tolerances (quick reference)

| Model type | rtol / atol |
|------------|-------------|
| Standard text, encoder, seq2seq, diffusion, audio | `1e-3` / `1e-3` |
| Multimodal (vision pipeline) | `1e-2` / `1e-2` |
| Generation (token IDs) | Exact match |
| fp16/bf16 logits | `1e-2` / `1e-2` |

Key rules:
- `assert_logits_close` checks shape + dtype match via `np.testing.assert_allclose(..., strict=True)`
- If max abs diff > 0.5 → likely a norm or scaling bug
- If max abs diff > 10 → weights loaded to wrong parameters

> Read [`references/tolerance-guidelines.md`](references/tolerance-guidelines.md)
> for the full failure checklist, debugging scripts, and dtype-specific guidance.

---

## Gotchas and common mistakes

### L1 tests are necessary but not sufficient

L1 verifies graph construction, not execution. A Scan body MatMul shape
mismatch can pass all L1 tests but crash at runtime. **Always write an
integration test alongside any new custom function or Scan op.**

### Integration tests must exercise all code paths

- **Text-only first** — verify logit parity before adding other modalities
- **Vision with real pixel values** — zeros don't exercise the encoder
- **All dtypes** (f32, f16, bf16) — each can expose different bugs
- **GPU when available** — different kernels on CUDA
- **Actual package wiring** — feed each ONNX stage from the preceding ONNX
  stage, not an HF intermediate that bypasses the integration under test
- **Defaults and masks** — assert constructor/config defaults in emitted ONNX
  attributes and test padding invariance across prefill plus cached decode
- **Real processor contract** — record input names, shapes, dtypes, media-row
  ordering, and sampled frame positions from nonzero image/video/audio data
- **Batch and decode edges** — use two rows with distinguishable media
  features, mixed modality order, and a decode step with zero new media

### Golden tests must be reproducible and exact

- Pin one revision through config, processor, weight shards, reference
  generation, and ONNX build; test that plumbing forwards it.
- Assert sequence lengths before exact token/frame comparison. For CTC, compare
  the full argmax frame sequence and collapsed transcript.
- Test image-only, video-only, and mixed media; pass processor kwargs only for
  media that are present.
- On hosted runners, isolate and eagerly delete each test's Hub, assets, and
  Xet caches. Patching `HF_HOME` after `huggingface_hub` import is insufficient;
  patch its imported cache constants too.
- Run the exact L2 discovery path: YAML schema, `test_model_id`, revision, and
  trust flags can be missed by model-local tests.
- Run affected-model detection on the full diff before GPU CI. A new model
  should select its targeted L4/L5 cases; distinguish all-model timeout or
  runner termination from a model assertion failure.
- Reference goldens must come from an independently invoked upstream pipeline,
  never from the implementation under test or ad-hoc intermediate features.
- When the checkpoint is too large, HTTP-range-read safetensors headers and the
  exact tensors for a production-dimension reduced fixture. Cover every layer
  family and cache contract, record the source layer/row derivation, and create
  L4/L5 goldens from the independently invoked HuggingFace model. Treat this as
  reduced real-weight evidence, not as a claim of full-checkpoint parity.
- Before accepting an architecture xfail, verify config vocabulary, epsilon,
  and layer-kind translation. A stale `mlp`->`moe` mapping can look like an SSM
  numerical failure while loading the wrong weights entirely.
- Compare full prefill and every token-by-token prompt/decode logit on the
  target EP. Fused cache kernels can match multi-token prefill yet diverge on
  the first reused-state step. If full-precision and quantized packages fail at
  the same reused-state step, suspect source cache semantics rather than Olive.
  Test the equivalent standard-ONNX cache graph before assigning blame; prefer
  the portable graph when it restores the numeric gate.

### Recurrent state ≠ KV cache

Recurrent state batch dim must match the actual input batch size. Do not
copy the KV cache `batch=0` initialization pattern:
```python
# WRONG — collapses Scan output
past_state = np.zeros((0, num_heads, d_k, d_v), dtype=np.float32)
# CORRECT
past_state = np.zeros((batch_size, num_heads, d_k, d_v), dtype=np.float32)
```

### fp16 Exp overflow

`exp(x)` overflows to `inf` for `x > ~11.09` in fp16. Upcast to float32
for Exp/Softplus. bf16 does NOT need this workaround (same exponent range
as fp32).

### Compare full logits, not just tokens

Generated tokens hide logit divergence — two different logit vectors can
agree on top-1. Always use `assert_logits_close` on the full tensor.

### Use `--compare-hf` in example scripts

The `--compare-hf` flag is the gold-standard correctness check. Run it
for all supported dtypes as part of every significant model change.

### Enable automated code review

Code review catches bugs that unit tests cannot (fp16 overflow risks,
missing input validation). Enable it on every PR modifying model code.
