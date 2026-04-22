# Test Examples Reference

Full code examples for each test type in mobius. See the main
[SKILL.md](../SKILL.md) for the overview and decision framework.

## L1: Graph build test patterns

### Adding a new model type

Add an entry to the appropriate list in `tests/_test_configs.py` with the
model type, config overrides, and `is_representative` flag:

```python
# In tests/_test_configs.py:
CAUSAL_LM_CONFIGS: list[tuple[str, dict, bool]] = [
    ("llama", {}, True),
    ("my_model", {"attn_qkv_bias": True, "hidden_act": "gelu"}, True),
    # ...
]
```

The test framework automatically creates a tiny config, builds the graph,
and checks:
- Graph has inputs (`input_ids`, `attention_mask`, `position_ids`)
- Graph has outputs (`logits`, `present.{i}.key`, `present.{i}.value`)
- Graph has initializers (embedding, attention, MLP/expert parameters)

### Model-specific structure tests

For models with unique structure (e.g. LoRA), add a dedicated test class:

```python
class TestBuildGraphLoRA:
    def test_lora_initializers_present(self):
        config = _base_config(
            vision_lora={"r": 4, "lora_alpha": 8},
            speech_lora={"r": 8, "lora_alpha": 16},
        )
        model_cls = registry.get("phi4mm")
        module = model_cls(config)
        task = CausalLMTask()
        model = task.build_graph(module, config, opset_version=23)

        init_names = list(model.graph.initializers)
        lora_names = [n for n in init_names if "lora" in n]
        assert len(lora_names) > 0
```

## L3: Integration test patterns

### Prefill + decode numerical comparison

```python
@pytest.mark.integration
@pytest.mark.parametrize("model_id,trust_remote_code", _TEXT_MODELS)
class TestForwardNumerical:
    def test_prefill_logits_match(self, model_id, trust_remote_code):
        onnx_model = build(model_id, load_weights=True)
        torch_model, tokenizer = load_torch_model(model_id)
        config = _get_config(model_id, trust_remote_code)

        # Tokenize, run both models, compare
        feeds = _make_prefill_feeds(config, input_ids, attention_mask, position_ids)
        onnx_outputs = session.run(feeds)
        assert_logits_close(onnx_outputs["logits"], torch_logits, rtol=1e-3, atol=1e-3)

    def test_decode_step_logits_match(self, model_id, trust_remote_code):
        # Prefill first, then feed next token + KV cache
        decode_feeds = _make_decode_feeds(config, ...)
        onnx_out_2 = session.run(decode_feeds)
        assert_logits_close(onnx_out_2["logits"], torch_logits_2, rtol=1e-3, atol=1e-3)
```

### Greedy generation

```python
@pytest.mark.integration
class TestGreedyGeneration:
    def test_generate_tokens_match(self, model_id, trust_remote_code):
        session = OnnxModelSession(onnx_model)
        generator = OnnxGenerator(session, config)
        onnx_ids = generator.generate(input_ids, max_new_tokens=10, eos_token_id=...)

        torch_ids = torch_generate_greedy(torch_model, input_ids, max_new_tokens=10, eos_token_id=...)
        assert_generation_match(onnx_ids[0].tolist(), torch_ids[0].tolist())
```

### Adding a new model to integration tests

Add a `pytest.param` to `_TEXT_MODELS`:

```python
_TEXT_MODELS = [
    pytest.param("Qwen/Qwen2.5-0.5B", False, id="qwen2.5-0.5b"),
    pytest.param("my-org/my-small-model", False, id="my-model"),
    # ...
]
```

Guidelines for choosing models:
- Prefer models ≤ 1B parameters for CI speed
- Models must be publicly accessible (no gated/private repos)
- One representative model per distinct model class

## L4 + L5: Golden test patterns

### YAML test case format

**Location:** `testdata/cases/<category>/<model>.yaml`

Categories match task types: `causal-lm`, `encoder`, `seq2seq`, `audio`,
`vision`, `vision-language`, `diffusion`.

**Required fields:**

```yaml
model_id: "Qwen/Qwen2.5-1.5B-Instruct"   # HuggingFace model ID
revision: "main"                            # Git revision / commit SHA
task_type: "text-generation"               # Task type string
dtype: "float32"                           # "float32", "float16", or "bfloat16"
level: "L4+L5"                             # "L4", "L5", or "L4+L5"

inputs:
  prompts:
    - "Here is my poem:"                   # Text prompt(s); use this default
```

For image models, use `images:` instead of (or alongside) `prompts:`:

```yaml
inputs:
  images:
    - "pipeline-cat-chonk.jpeg"            # Path relative to testdata/
```

For audio models:

```yaml
inputs:
  audio:
    - "652-129742-0006.flac"
```

**Optional fields:**

```yaml
# Identifier for the test model used in L2 config compatibility check.
# If set, the dashboard counts this model as L2 (full HF config valid).
test_model_id: "Qwen/Qwen2.5-1.5B-Instruct"

# Skip this test case entirely (model too large, gated repo, etc.).
# Dashboard shows the model as 'skipped' rather than counting it toward coverage.
skip_reason: "Model too large (47B MoE) for CPU golden generation."

# Pass trust_remote_code=True when loading HuggingFace model (default: false).
trust_remote_code: true

# Minimum fraction of generated tokens that must match the golden reference.
# Use for VL/audio pipelines where floating-point variance causes later tokens
# to diverge. A value of 0.25 means at least 25% of tokens must match exactly.
# Green (≥0.9) / Yellow (0.5–0.9) / Red (<0.5) on dashboard.
min_token_match_ratio: 0.25

# Human-readable notes about this model.
notes: "GPT-2 124M. Absolute positional embeddings, no RoPE."

generation:
  max_new_tokens: 20                       # Override token generation limit
  do_sample: false
```

**`skip_reason` vs `_SKIP_REASONS` dict:** Always use the YAML `skip_reason`
field for new cases. The legacy `_SKIP_REASONS` dict in `e2e_golden_test.py`
has been removed — YAML is the canonical location.

### Golden file format

**L4 golden file** (`testdata/golden/<cat>/<model>.json`):
Generated automatically by `generate_golden.py`. Contains `top1_id`,
`top2_id`, `top10_ids`, `top10_logits`, and `logits_summary` from the last
token position of the prefill pass.

**L5 generation file** (`testdata/golden/<cat>/<model>_generation.json`):
Contains `model_id`, `prompt`, `generated_tokens` (list of token IDs), and
`generated_text`. This is the authoritative source for L5 tests — the main
golden JSON does **not** contain generation data.

### Generating golden data

```bash
# Generate for all test cases at a given level
python scripts/generate_golden.py --level L4

# Generate for a specific task type
python scripts/generate_golden.py --level L4 --task-type causal-lm

# Generate for a specific model (glob filter on model name)
python scripts/generate_golden.py --level L4 --filter 'llama*'
```

Golden files must be committed alongside new test case YAML files.

### Step-by-step: adding coverage for a new model

**L1 — Graph builds:**
1. Add `("my_model", {config_overrides}, True)` to the appropriate list in
   `tests/_test_configs.py` (or add a dedicated method if the model is a VLM/audio).
2. Run `python -m pytest tests/build_graph_test.py -k "my_model"`.

**L2 — Config compatible:**
1. Create `testdata/cases/<category>/my-model.yaml`.
2. Set `test_model_id: "org/my-model-id"`.
3. Run schema validation: `python -m pytest tests/yaml_schema_test.py`.

**L3 — Synthetic parity:**
1. Add `pytest.param("org/my-model", False, id="my-model")` to the
   appropriate parametrized list in `tests/integration_test.py`.
2. Run `python -m pytest tests/integration_test.py -m integration -k "my-model"`.

**L4 — Golden match:**
1. Create/update `testdata/cases/<category>/my-model.yaml` with `level: "L4"`.
2. Set `inputs.prompts: ["Here is my poem:"]` (standard default prompt).
3. Run `python scripts/generate_golden.py --level L4 --filter 'my-model*'`.
4. Commit the generated `testdata/golden/<cat>/my-model.json`.
5. Run `python -m pytest tests/e2e_golden_test.py -m golden --level L4 -k "my-model"`.

**L5 — Generation verified:**
1. Update YAML to `level: "L5"` or `"L4+L5"`.
2. Add a `generation:` block with `max_new_tokens` and `do_sample: false`.
3. Optionally set `min_token_match_ratio` if you expect partial divergence
   (VL pipelines, long generation sequences).
4. Run `python scripts/generate_golden.py --level L5 --filter 'my-model*'`.
5. Commit `testdata/golden/<cat>/my-model_generation.json`.
6. Run `python -m pytest tests/e2e_golden_test.py -m golden --level L5 -k "my-model"`.

## Debugging multi-model pipelines (TTS, VLM)

When a multi-model pipeline produces wrong output but individual
model prefill logits look correct, isolate each model boundary:

1. **Compare each model's output against HF at the boundary** — e.g.
   `last_hidden_state` from the talker, `codec_sum` from embeddings,
   `inputs_embeds` constructed for the code predictor.

2. **Check pre-norm vs post-norm** — `outputs.last_hidden_state` in HF
   is typically post-norm. If your ONNX model returns pre-norm hidden
   states, downstream models receive wrong values.

3. **Verify external construction matches HF** — for models where the
   generation loop constructs inputs externally (e.g. concatenating
   hidden states with embeddings), write a comparison script that
   checks the constructed input matches HF token-by-token:
   ```python
   # Compare inputs_embeds at each generation step
   for step in range(num_steps):
       onnx_input = construct_inputs_embeds(step, ...)
       hf_input = hf_model.get_inputs_embeds(step, ...)
       diff = np.abs(onnx_input - hf_input).max()
       print(f"Step {step}: max diff = {diff:.6f}")
   ```

4. **Embedding weight vs lookup mismatch** — if embedding weights are
   identical but lookups differ, the issue is usually which code index
   or embedding table is being used (off-by-one errors).

## Parity testing methodology

Compare full logit tensors, not just generated tokens. Generated tokens
hide logit divergence (two very-different logit vectors can agree on the
top-1 token):

```python
# Always compare full logits at every position
assert_logits_close(onnx_logits, hf_logits, atol=1e-3, rtol=1e-3)  # fp32
assert_logits_close(onnx_logits, hf_logits, atol=1e-2, rtol=1e-2)  # fp16/bf16

# Also check last-position argmax matches (quick sanity check)
assert onnx_logits[0, -1].argmax() == hf_logits[0, -1].argmax()
```

If argmax matches but full logit tolerance fails, the model is numerically
correct but some intermediate accumulation differs — this is usually
acceptable for fp16/bf16 and worth a brief comment in the test.

## Examples as QA tools

The `--compare-hf` flag in example scripts is the gold-standard correctness
check for a model. Run it as part of every significant change:

```bash
# Primary correctness check
python examples/qwen35_text_generation.py --compare-hf

# Test all supported dtypes
python examples/qwen35_text_generation.py --compare-hf --dtype f16
python examples/qwen35_text_generation.py --compare-hf --dtype bf16

# Test on GPU (if available)
python examples/qwen35_text_generation.py --compare-hf --device cuda
```

Target: **100% token match** in fp32 greedy generation. fp16/bf16 may
diverge after the first few tokens due to floating-point accumulation, which
is acceptable if logit parity holds at `atol=rtol=1e-2`.
