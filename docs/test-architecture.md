# Test Architecture

This document describes the complete test infrastructure for mobius: the
five-level confidence system (L1–L5), how tests are organized, the YAML
test case and golden data pipeline, and the dashboard.

---

## Contents

1. [Confidence levels (L1–L5)](#confidence-levels-l1l5)
2. [How levels are counted](#how-levels-are-counted)
3. [Test organization](#test-organization)
4. [L1 — Graph build tests](#l1--graph-build-tests)
5. [L2 — Config compatibility](#l2--config-compatibility)
6. [L3 — Synthetic parity (integration tests)](#l3--synthetic-parity-integration-tests)
7. [L4/L5 — Golden tests](#l4l5--golden-tests)
8. [YAML test case schema](#yaml-test-case-schema)
9. [Golden data pipeline](#golden-data-pipeline)
10. [Dashboard](#dashboard)
11. [Adding coverage for a new model](#adding-coverage-for-a-new-model)

---

## Confidence levels (L1–L5)

Every registered model type has a confidence level for each of five
independent dimensions.  Levels are **not hierarchical** — a model can
pass L3 without passing L2 (e.g. it has an integration test but no YAML
test case), or have L4 golden data without passing L3.  Each level is
detected from its own data source.

| Level | Name | What it verifies | Data source |
|-------|------|-----------------|-------------|
| **L1** | Graph builds | ONNX graph builds from a tiny synthetic config (no weights, no network) | `_MODEL_CONFIGS` / `_SPECIALIZED_TEST_MODEL_TYPES` in `tests/build_graph_test.py` |
| **L2** | Config compatible | Full-size HuggingFace config loads and produces a valid graph | `test_model_id` field in YAML test case (`testdata/cases/`) |
| **L3** | Synthetic parity | Random-weight forward pass logits match HuggingFace within tolerance | `tests/integration_test.py` (and related `*_integration_test.py` files) |
| **L4** | Golden match | Real-weight prefill logits match a pre-computed golden reference | `*.json` files in `testdata/golden/` |
| **L5** | Generation verified | Full multi-token generation matches a pre-computed golden reference | `*_generation.json` files in `testdata/golden/` |

---

## How levels are counted

The dashboard counts levels **per flag**: L1 count = number of models
where `l1_graph_build=True`, L2 count = models where
`l2_arch_validation=True`, etc.  A model is counted at every level it
passes, not just the highest one.

This means:
- **L1 ≈ total registered models** — every model type in the registry has
  at least one graph build test (either in `_MODEL_CONFIGS` or
  `_SPECIALIZED_TEST_MODEL_TYPES`).
- **L2 ≤ number of YAML test cases** — only models with a `test_model_id`
  field in their YAML case are counted.
- **L3, L4, L5** are independent subsets that can overlap in any combination.

The detection logic lives in `scripts/generate_dashboard.py`:

| Scanner | What it reads |
|---------|--------------|
| `_scan_l1_configs` | `tests/_test_configs.py::ALL_CONFIGS` + `tests/build_graph_test.py::_SPECIALIZED_TEST_MODEL_TYPES` |
| `_scan_l2_arch_tests` | `test_model_id` presence in YAML files under `testdata/cases/` |
| `_scan_l3_synthetic_parity` | Model type presence in `tests/integration_test.py` and related integration test files |
| `_scan_l4_golden_files` | Existence of `testdata/golden/<category>/<model>.json` |
| `_scan_l5_generation_golden` | Existence of `testdata/golden/<category>/<model>_generation.json` |

---

## Test organization

```
tests/
├── _test_configs.py              # Shared model configs for parametrized tests
├── conftest.py                   # pytest fixtures, --fast flag
├── build_graph_test.py           # L1: graph construction (no weights)
├── weight_alignment_test.py      # L1: preprocess_weights() coverage
├── arch_validation_test.py       # L2: full HF config graph build
├── integration_test.py           # L3: causal-LM numerical parity
├── synthetic_parity_test.py      # L3: synthetic parity parametrized
├── multimodal_integration_test.py # L3: vision-language parity
├── vision_integration_test.py    # L3: vision encoder parity
├── seq2seq_integration_test.py   # L3: seq2seq / encoder-decoder parity
├── whisper_integration_test.py   # L3: Whisper speech-to-text parity
├── mamba2_integration_test.py    # L3: Mamba2 SSM parity
├── moe_integration_test.py       # L3: MoE model parity
├── phi4mm_integration_test.py    # L3: Phi4MM multimodal parity
├── e2e_golden_test.py            # L4 + L5: golden file comparison
├── yaml_schema_test.py           # Validates all YAML test cases against schema
├── cli_test.py                   # CLI smoke tests
├── gguf_test.py                  # GGUF weight loading
└── ort_genai_test.py             # ORT GenAI config generation

src/mobius/
└── **/*_test.py                  # Unit tests co-located with source

testdata/
├── cases/                        # YAML test case definitions (L2, L4, L5)
│   ├── schema.json               # JSON Schema for YAML validation
│   ├── causal-lm/
│   ├── vision-language/
│   ├── audio/
│   ├── encoder/
│   ├── seq2seq/
│   ├── vision/
│   └── diffusion/
└── golden/                       # Pre-computed reference outputs
    ├── causal-lm/
    │   ├── gpt2.json             # L4 prefill golden
    │   └── gpt2_generation.json  # L5 generation golden
    └── ...

scripts/
├── generate_dashboard.py         # Scans all sources, renders HTML dashboard
├── generate_golden.py            # Runs HF inference to produce golden files
└── templates/
    └── dashboard.html.j2         # Jinja2 template for the dashboard
```

Unit tests (files ending `_test.py` under `src/`) are co-located with the
source they test.  Integration and golden tests live in `tests/`.

---

## L1 — Graph build tests

**File:** `tests/build_graph_test.py`  
**Runs in CI:** Always (no network, no weights)

Builds each model from a tiny `ArchitectureConfig` (64 hidden, 2 layers,
256 vocab) and verifies graph structure: correct inputs, outputs, and
initializer presence.

### Model config sources

**Parametrized models** are declared in `tests/_test_configs.py`:

```python
CAUSAL_LM_CONFIGS: list[tuple[str, dict, bool]] = [
    #  model_type        config_overrides           is_representative
    ("llama",            {},                         True),
    ("qwen2",            {"attn_qkv_bias": True},    True),
    ("phi4",             {"partial_rotary_factor": 0.5}, True),
]
```

Each entry is `(model_type, config_overrides, is_representative)`.  The
`is_representative` flag controls whether the model runs with `--fast`
(skip non-representative models to cut test time to ~5 seconds).

| Config list | Test class | Task type |
|-------------|-----------|-----------|
| `CAUSAL_LM_CONFIGS` | `TestBuildGraph` | text-generation |
| `ENCODER_CONFIGS` | `TestBuildEncoderGraph` | feature-extraction |
| `SEQ2SEQ_CONFIGS` | `TestBuildSeq2SeqGraph` | seq2seq |
| `VISION_CONFIGS` | `TestBuildVisionGraph` | image-classification |
| `DETECTION_CONFIGS` | `TestBuildDetectionGraph` | object-detection |

**Specialised models** (VLM, audio, TTS) have dedicated test methods and
are tracked in `_SPECIALIZED_TEST_MODEL_TYPES`.  These build a full
multi-model package (e.g. vision encoder + embedding + decoder for VLMs).

**Auto-generated configs:** Text-generation model types registered in
`_registry.py` that have no explicit entry in `_test_configs.py` get an
auto-generated `(model_type, {}, False)` entry so they always receive
basic L1 coverage.

### Weight alignment tests

`tests/weight_alignment_test.py` verifies that `preprocess_weights()`
maps every HuggingFace state dict key to an ONNX initializer — no dropped
or mangled names.  These run in CI alongside the graph build tests.

---

## L2 — Config compatibility

**Detected from:** `test_model_id` field in YAML test case  
**File:** `tests/arch_validation_test.py`

L2 verifies that a real HuggingFace model config (not the tiny synthetic
one) produces a valid ONNX graph.  A model is counted as L2 on the
dashboard if and only if its YAML test case has a non-empty `test_model_id`
field.

To add L2 coverage, create a YAML test case (see [YAML test case schema]
(#yaml-test-case-schema)) and set `test_model_id`.

---

## L3 — Synthetic parity (integration tests)

**Files:** `tests/integration_test.py` and `tests/*_integration_test.py`  
**Runs in CI:** Opt-in (`-m integration`; requires network and model download)

L3 tests load a real HuggingFace checkpoint, build the ONNX graph, apply
weights, run a single forward pass (prefill), and compare logits against
the PyTorch reference with a numerical tolerance.

```bash
# Run all integration tests
python -m pytest tests/integration_test.py -m integration

# Run a specific model
python -m pytest tests/integration_test.py -m integration -k "qwen2.5-0.5b"

# Run multimodal integration tests
python -m pytest tests/multimodal_integration_test.py -m integration
```

**Tolerance:** `atol=rtol=1e-3` for standard text models; `atol=rtol=1e-2`
for multimodal (vision pipeline accumulates extra float variance).

The L3 scanner in `generate_dashboard.py` reads `integration_test.py` to
detect which model types are tested.  It also reads `_scan_l3_parity_status`
to mark models as `pass`, `xfail`, or `skip`.

---

## L4/L5 — Golden tests

**File:** `tests/e2e_golden_test.py`  
**Data:** `testdata/cases/` (YAML) + `testdata/golden/` (JSON)  
**Runs in CI:** Opt-in (`-m golden` or `-m generation`)

Golden tests are fully **data-driven**: adding coverage requires adding a
YAML file and a JSON golden file — no code changes needed.

| Level | pytest mark | What it checks |
|-------|-------------|---------------|
| L4 | `@pytest.mark.golden` | Last-position prefill logits match golden |
| L5 | `@pytest.mark.generation` | Generated token sequence matches golden |

```bash
# Run all L4 tests
python -m pytest tests/e2e_golden_test.py -m golden -v

# Run all L5 tests
python -m pytest tests/e2e_golden_test.py -m generation -v

# Run a specific model
python -m pytest tests/e2e_golden_test.py -k "gpt2"
```

---

## YAML test case schema

**Location:** `testdata/cases/<category>/<model>.yaml`  
**Schema:** `testdata/cases/schema.json` (JSON Schema draft 2020-12)  
**Validated by:** `tests/yaml_schema_test.py` (runs in CI)

Categories match task types: `causal-lm`, `vision-language`, `audio`,
`encoder`, `seq2seq`, `vision`, `diffusion`.

### Required fields

| Field | Type | Description |
|-------|------|-------------|
| `model_id` | string | HuggingFace model ID, e.g. `"Qwen/Qwen2.5-1.5B-Instruct"` |
| `revision` | string | Git commit SHA (preferred) or `"main"` |
| `task_type` | enum | One of: `text-generation`, `image-text-to-text`, `speech-to-text`, `audio-feature-extraction`, `feature-extraction`, `seq2seq`, `image-classification`, `object-detection` |
| `dtype` | enum | `"float32"`, `"float16"`, or `"bfloat16"` |
| `level` | enum | `"L4"`, `"L5"`, or `"L4+L5"` |

### Optional fields

| Field | Type | Description |
|-------|------|-------------|
| `inputs.prompts` | string[] | Text prompts (text-generation / VL tasks). Default: `["Here is my poem:"]` |
| `inputs.images` | string[] | Image file paths relative to `testdata/` (VL tasks) |
| `inputs.audio` | string[] | Audio file paths relative to `testdata/` (speech tasks) |
| `inputs.decoder_prompt` | string | Forced decoder prefix for seq2seq tasks |
| `generation.max_new_tokens` | int | Override token generation limit for L5 |
| `generation.do_sample` | bool | `false` = greedy decoding (required for reproducible golden) |
| `test_model_id` | string | HF model ID used for L2 config compatibility check |
| `skip_reason` | string | If set, test is skipped; dashboard shows model as "skipped" (not counted toward coverage) |
| `trust_remote_code` | bool | Pass `trust_remote_code=True` to HF loaders (default: `false`) |
| `min_token_match_ratio` | float (0–1) | Minimum fraction of generated tokens that must match the golden. Used for VL/audio pipelines where float variance causes later tokens to diverge. Dashboard color: green ≥0.9, yellow 0.5–0.9, red <0.5. |
| `notes` | string | Human-readable description; not parsed by test runner |

### Example: causal-LM test case

```yaml
model_id: "openai-community/gpt2"
revision: "607a30d783dfa663caf39e06633721c8d4cfcd7e"
task_type: "text-generation"
dtype: "float32"
level: "L4+L5"

inputs:
  prompts:
    - "Here is my poem:"

generation:
  max_new_tokens: 20
  do_sample: false

notes: "GPT-2 124M. Absolute positional embeddings, no RoPE."
```

### Example: vision-language test case with tolerance override

```yaml
model_id: "Qwen/Qwen2.5-VL-3B-Instruct"
revision: "66285546d2b821cf421d4f5eb2576359d3770cd3"
task_type: "image-text-to-text"
dtype: "float32"
level: "L4+L5"

inputs:
  prompts:
    - "Describe this image in detail."
  images:
    - "pipeline-cat-chonk.jpeg"

generation:
  max_new_tokens: 30
  do_sample: false

min_token_match_ratio: 0.25

notes: "Qwen2.5-VL 3B. Vision encoder + embedding + decoder pipeline."
```

### Skipped test case

```yaml
model_id: "mistralai/Mixtral-8x7B-v0.1"
revision: "main"
task_type: "text-generation"
dtype: "float32"
level: "L4+L5"
skip_reason: "Model too large (47B MoE) for CPU golden generation."

inputs:
  prompts:
    - "Here is my poem:"
```

---

## Golden data pipeline

### L4 golden files (`<model>.json`)

**Location:** `testdata/golden/<category>/<model>.json`

Contain the last-position prefill output from a HuggingFace forward pass
with real weights:

```json
{
  "model_id": "openai-community/gpt2",
  "top1_id": 1234,
  "top2_id": 5678,
  "top10_ids": [1234, 5678, ...],
  "top10_logits": [12.3, 11.1, ...],
  "logits_summary": {"mean": 0.01, "std": 4.2, "min": -8.1, "max": 14.5}
}
```

The L4 test (`TestL4CheckpointE2E`) runs the ONNX model, takes the
last-position logits, and asserts `top1_id` and `top2_id` match.

### L5 generation files (`<model>_generation.json`)

**Location:** `testdata/golden/<category>/<model>_generation.json`

Contain the result of a full greedy generation run:

```json
{
  "model_id": "openai-community/gpt2",
  "prompt": "Here is my poem:",
  "generated_tokens": [1234, 5678, 910, ...],
  "generated_text": "Here is my poem: Roses are red..."
}
```

The L5 test (`TestL5GenerationE2E`) reads this file, runs the ONNX
generator for the same `max_new_tokens`, and checks that at least
`min_token_match_ratio` of the generated tokens match exactly.

### Generating golden files

`scripts/generate_golden.py` runs HuggingFace inference and writes both
file types:

```bash
# All test cases
python scripts/generate_golden.py

# Specific task type
python scripts/generate_golden.py --task-type causal-lm

# Specific model (glob)
python scripts/generate_golden.py --filter 'gpt2*'

# Specific YAML file
python scripts/generate_golden.py --case testdata/cases/causal-lm/gpt2.yaml

# Overwrite existing files
python scripts/generate_golden.py --force

# Use GPU for large models
python scripts/generate_golden.py --device cuda
```

Golden files must be **committed alongside the YAML test case** — the test
runner does not generate them on demand.

---

## Dashboard

**Script:** `scripts/generate_dashboard.py`  
**Template:** `scripts/templates/dashboard.html.j2`  
**Output:** `docs/dashboard/index.html` (deployed to GitHub Pages)

The dashboard is a self-contained static HTML file.  It is generated by
scanning the registry, test files, and golden data — it does not execute
any tests.

### What it shows

- **Summary bar:** Per-flag counts for L1–L5, L3 parity status breakdown,
  and number of YAML test cases defined.
- **Model table:** One row per registered model type, with confidence badge,
  coverage dots, code-path tags, and expandable detail panel showing config
  overrides and copy-paste test commands.
- **Family rows:** Models grouped by family (e.g. "qwen", "llama") with a
  histogram showing `L1:N L2:N L3:N L4:N L5:N` per family.
- **Missing coverage list:** Model types with no test coverage at any level.

### How it is generated

```bash
# Generate the dashboard
python scripts/generate_dashboard.py --output docs/dashboard/index.html

# With git commit SHA
python scripts/generate_dashboard.py \
  --output docs/dashboard/index.html \
  --commit $(git rev-parse --short HEAD)
```

The script runs six scanners in order:

```
_scan_registry()               → populate ModelInfo for every registered type
_scan_l1_configs()             → mark L1 from ALL_CONFIGS + _SPECIALIZED_TEST_MODEL_TYPES
_scan_l2_arch_tests()          → mark L2 from test_model_id in YAML cases
_scan_l3_synthetic_parity()    → mark L3 from integration_test.py
_scan_l3_parity_status()       → mark pass/xfail/skip per model from test results
_scan_yaml_test_cases()        → load all YAML cases (must run before golden scans)
_scan_l4_golden_files()        → mark L4 from testdata/golden/
_scan_l5_generation_golden()   → mark L5 from testdata/golden/*_generation.json
_scan_integration_tests()      → fill test_model_id, has_integration_test
```

The golden file scanners use a two-strategy matching:
1. **Direct:** golden file stem matches the `model_type` string.
2. **YAML-derived:** when a model has a YAML case, derive the expected golden
   path from the YAML file location and case ID.

### Jinja2 template

The HTML is rendered from `scripts/templates/dashboard.html.j2` using
Jinja2 with `autoescape=True`.  JSON data blobs injected into `<script>`
tags are marked `|safe` — they are pre-serialized by `_to_js_json()` which
handles `<\/` escaping to prevent premature script-tag closure.

JavaScript template literals (`${...}`) conflict with Jinja2's `{{ }}`
syntax, so all JavaScript logic lives inside a `{% raw %}...{% endraw %}`
block.

### CI deployment

`.github/workflows/pages.yml` generates the dashboard and deploys it to
GitHub Pages on every push to `main` and on a weekly schedule.  `jinja2`
is installed from `docs/requirements.txt` before the generation step.

---

## Adding coverage for a new model

### L1 — Graph builds

For a **text-generation** model:

```python
# In tests/_test_configs.py:
CAUSAL_LM_CONFIGS: list[tuple[str, dict, bool]] = [
    # ...
    ("my_model", {"attn_qkv_bias": True}, True),
]
```

For a **VLM or audio** model, add a dedicated test method in
`build_graph_test.py` and add the `model_type` string to
`_SPECIALIZED_TEST_MODEL_TYPES`.

Verify:
```bash
python -m pytest tests/build_graph_test.py -k "my_model"
```

### L2 — Config compatible

Create `testdata/cases/<category>/my-model.yaml` and set `test_model_id`:

```yaml
model_id: "org/my-model"
revision: "main"
task_type: "text-generation"
dtype: "float32"
level: "L4+L5"
test_model_id: "org/my-model"
inputs:
  prompts:
    - "Here is my poem:"
```

Verify schema:
```bash
python -m pytest tests/yaml_schema_test.py
```

### L3 — Synthetic parity

Add the model to the appropriate parametrized list in `tests/integration_test.py`:

```python
_TEXT_MODELS = [
    # ...
    pytest.param("org/my-model", False, id="my-model"),
]
```

Run:
```bash
python -m pytest tests/integration_test.py -m integration -k "my-model"
```

### L4 — Golden match

1. Ensure the YAML test case exists with `level: "L4"` or `"L4+L5"`.
2. Generate the golden file:
   ```bash
   python scripts/generate_golden.py --filter 'my-model*'
   ```
3. Commit `testdata/golden/<cat>/my-model.json`.
4. Run the L4 test:
   ```bash
   python -m pytest tests/e2e_golden_test.py -m golden -k "my-model"
   ```

### L5 — Generation verified

1. Set `level: "L5"` or `"L4+L5"` in the YAML test case.
2. Add a `generation:` block:
   ```yaml
   generation:
     max_new_tokens: 20
     do_sample: false
   ```
3. Generate the generation golden file:
   ```bash
   python scripts/generate_golden.py --filter 'my-model*'
   ```
4. Commit `testdata/golden/<cat>/my-model_generation.json`.
5. Run the L5 test:
   ```bash
   python -m pytest tests/e2e_golden_test.py -m generation -k "my-model"
   ```

For VL/audio pipelines that accumulate float variance over decode steps,
set `min_token_match_ratio` to an appropriate value (e.g. `0.25`) rather
than expecting 100% token match.
