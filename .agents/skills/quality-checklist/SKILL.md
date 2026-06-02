---
name: quality-checklist
description: >
  Use this skill when verifying that a new model is truly done and ready to
  merge. Provides a Definition-of-Done checklist covering all five test
  confidence levels (L1 graph build through L5 Foundry Local smoke-test),
  ORT GenAI runtime validation, Olive quantization compatibility, multi-dtype
  (f32/f16/bf16) and multi-EP (CPU/CUDA/DML) correctness, documentation
  requirements, and code review criteria.
---

# Skill: Quality Checklist

## When to use

Use this checklist before marking a new model addition as **done**.
Every item must be checked — or explicitly waived with a written reason —
before the PR is merged.

---

## The Checklist

### 1. Code quality

- [ ] Model file is in `src/mobius/models/` with the Microsoft MIT copyright
      header (`# Copyright (c) Microsoft Corporation. / # Licensed under the
      MIT License.`)
- [ ] Class has a descriptive one-paragraph docstring (first paragraph is
      used in generated docs)
- [ ] Class has `default_task` and `category` class-level attributes if
      the model is not a standard text-generation model
- [ ] `preprocess_weights()` correctly maps every HuggingFace state-dict key
      to the ONNX initializer name (verified by the weight-alignment test)
- [ ] All new components use `from mobius.components import ...` (public API),
      not private submodule paths
- [ ] No explicit protobuf operations anywhere in new code
      (`onnx.helper`, `onnx.TensorProto`, etc. are forbidden)
- [ ] Tensor shapes are annotated in comments after non-trivial operations
- [ ] Automated code review (Copilot/PR review) has been run and all
      findings are resolved or explicitly dismissed with a reason
- [ ] `lintrunner -a` is clean — zero lint errors before merging
      (`lintrunner f --output oneline --all-files` to auto-fix, then re-run to confirm)

### 2. L1 — Graph builds

- [ ] Entry exists in `tests/_test_configs.py` (or a dedicated test method
      for VLM / audio models)
- [ ] `is_representative=True` if the model has unique behaviour (custom
      class, special attention, MoE, hybrid layers, etc.)
- [ ] `python -m pytest tests/build_graph_test.py -k "<model_type>"` passes
- [ ] Weight-alignment test passes:
      `python -m pytest tests/weight_alignment_test.py -k "<model_type>"`

### 3. L2 — Config compatible

- [ ] YAML test case created at `testdata/cases/<category>/<model>.yaml`
- [ ] `test_model_id` field set to a real HuggingFace model ID
- [ ] Schema validates: `python -m pytest tests/yaml_schema_test.py`

### 4. L3 — Synthetic parity

- [ ] Model type is covered in `tests/synthetic_parity_test.py` (driven by
      `_test_configs.py`; added automatically for text-generation models)
- [ ] `python -m pytest tests/synthetic_parity_test.py -k "<model>"` passes
      with `atol=1e-3` / `rtol=1e-3` (or `1e-2` for multimodal)
- [ ] Real-weight parity also checked via `tests/integration_test.py` (add
      model to `_TEXT_MODELS` or equivalent if a small checkpoint is available)

### 5. L4 — Golden match

- [ ] YAML test case has `level: "L4"` or `"L4+L5"`
- [ ] `inputs.prompts: ["Here is my poem:"]` (standard default prompt unless
      audio/image model)
- [ ] Golden file generated:
      `python scripts/generate_golden.py --level L4 --filter '<model>*'`
- [ ] Golden file committed to `testdata/golden/<cat>/<model>.json`
- [ ] `python -m pytest tests/e2e_golden_test.py -m golden -k "<model>"` passes

### 6. L5 — Generation verified

- [ ] YAML test case has `level: "L5"` or `"L4+L5"`
- [ ] `generation.max_new_tokens` set (≥ 20 recommended)
- [ ] `generation.do_sample: false` (deterministic greedy decode)
- [ ] Generation golden file generated:
      `python scripts/generate_golden.py --level L5 --filter '<model>*'`
- [ ] Generation golden file committed to
      `testdata/golden/<cat>/<model>_generation.json`
- [ ] `python -m pytest tests/e2e_golden_test.py -m generation -k "<model>"` passes

> **Speech-language models:** The golden generation script supports the
> `speech-language` task type for models that process audio inputs (e.g.,
> Gemma4). The `_generate_speech_language()` function in
> `scripts/generate_golden.py` handles audio feature extraction and input
> construction for these models. Ensure the correct feature extractor
> (e.g. `Gemma4AudioFeatureExtractor`, not `WhisperFeatureExtractor`) is
> auto-detected via `AutoFeatureExtractor.from_pretrained()`.

> **Why L4/L5 matter:** Graph-build tests (L1) only verify ONNX graph
> construction; they never execute the graph with real data.  A MatMul shape
> mismatch that crashes at runtime, a wrong normalisation type, or a missing
> scaling multiplier can all pass L1 while producing completely wrong output.
> L4/L5 are the only tests that catch these classes of bugs.

### 7. Multi-dtype correctness

- [ ] fp32 tests pass (target: exact token match in greedy generation)
- [ ] fp16 tests pass (target: logit parity `atol=1e-2`; token match for
      first N tokens)
- [ ] bf16 tests pass (target: logit parity `atol=1e-2`)

Use the example `--compare-hf --dtype f16/bf16` flag if a comparison script
exists:

```bash
python examples/<model>_text_generation.py --compare-hf --dtype f16
python examples/<model>_text_generation.py --compare-hf --dtype bf16
```

### 8. CLI build

- [ ] `mobius build --model <hf-model-id> /tmp/out` completes without error
- [ ] Output directory contains the expected ONNX files and `genai_config.json`
- [ ] Multi-model tasks use standardised ModelPackage keys: `"decoder"`,
      `"vision_encoder"`, `"audio_encoder"`, `"embedding"` (not legacy names
      like `"model"`, `"vision"`, `"audio"`, or `"speech"`)
- [ ] Processor config uses correct filename: `image_processor.json` for
      all VLMs, `audio_processor.json` for audio

### 8a. Multi-EP correctness (CUDA)

- [ ] Model runs correctly with `--ep cuda` (or `--device cuda`)
- [ ] CUDA results match CPU results (compare generation output)
- [ ] No crashes from large tensor operations (see ORT Gather int32
      overflow: microsoft/onnxruntime#28107)
- [ ] No NaN from large head_dim > 256 (tracked in
      microsoft/onnxruntime#28195, #28196)
- [ ] `ort_lower_opset_for_ep` flag available for opset 24→23 lowering
      on older ORT builds (disabled by default in `src/mobius/_flags.py`,
      opt-in via `MOBIUS_ORT_LOWER_OPSET_FOR_EP=1`)
- [ ] Dead graph inputs removed after EP-aware optimization
      (`RemoveDeadGraphInputsPass` in Stage 4 of `optimize_model()`)
- [ ] For models with dual head_dim (e.g. Gemma4: 256 for sliding,
      512 for full attention), verify whether ORT GenAI requires
      `search.past_present_share_buffer=false` because GenAI allocates
      uniform KV cache shapes; if so, explicitly override the generated
      `genai_config.json` before runtime validation rather than assuming
      the default generated setting is correct
- [ ] Encoder inputs (vision/audio) are declared with `dtype=config.dtype`
      at sub-model entry (no stale float32-only cast guidance)

### 9. ORT GenAI runtime

- [ ] Model can be loaded with `ort_genai.Model(output_dir)` without error
- [ ] Greedy text generation produces non-empty, coherent output
- [ ] ORT GenAI test added to `tests/ort_genai_test.py`
      (or confirmed covered by an existing parametrized test)

Run the ORT GenAI integration test:

```bash
python -m pytest tests/ort_genai_test.py -m integration_slow -k "<model>" -sv
```

### 10. Foundry Local smoke test

- [ ] Model exported package can be loaded and run in Foundry Local
- [ ] At minimum, verify that the `genai_config.json` and all ONNX files
      are present and the model responds to a short prompt

> If Foundry Local is not available in the current environment, document the
> skip with a `# TODO: verify with Foundry Local` comment in the PR.

### 11. Olive quantization compatibility

- [ ] Model can be loaded from the exported ONNX package by Olive
- [ ] INT4 / INT8 quantization runs to completion without errors
- [ ] Quantized model produces non-degenerate output (coherent text)
- [ ] If quantization changes the graph structure (e.g. MatMulNBits), verify
      the `genai_config.json` still loads correctly in ORT GenAI

Run the quantization integration test suite to confirm existing patterns
are not broken:

```bash
python -m pytest tests/quantization_integration_test.py -v
```

For new architectures, add a quantized variant test if the architecture has
novel weight layouts (e.g. fused QKV, non-standard expert routing).

### 12. Documentation

- [ ] Class docstring (first paragraph) clearly describes the model family
      and the HuggingFace class it replicates
- [ ] `default_task` and `category` are set correctly (auto-generates the
      model page and index entry)
- [ ] If the model has a notable architectural difference from the base class,
      a comment in the source file or the skill notes explains it
- [ ] README model table updated if this is a significant new addition

---

## Waiver policy

Any item that cannot be completed must be waived explicitly in the PR
description:

```
**Waivers:**
- L5 golden: Model is 70B — generating golden data exceeds CI resources.
  skip_reason added to YAML.
- Foundry Local: Not available in this environment. Tracked in issue #NNN.
```

Unchecked items without a waiver are grounds to request changes before merge.

---

## Quick reference commands

```bash
# Lint (auto-fix then verify clean)
lintrunner f --output oneline --all-files
lintrunner -a

# L1 – graph build
python -m pytest tests/build_graph_test.py -k "<model_type>"

# L1 – weight alignment
python -m pytest tests/weight_alignment_test.py -k "<model_type>"

# L2 – YAML schema
python -m pytest tests/yaml_schema_test.py

# L3 – synthetic parity
python -m pytest tests/synthetic_parity_test.py -k "<model>" -sv

# L3 – real-weight integration (if small checkpoint available)
python -m pytest tests/integration_test.py -m integration -k "<model>" -sv

# L4 – generate golden
python scripts/generate_golden.py --level L4 --filter '<model>*'

# L4 – run golden test
python -m pytest tests/e2e_golden_test.py -m golden -k "<model>" -v

# L5 – generate generation golden
python scripts/generate_golden.py --level L5 --filter '<model>*'

# L5 – run generation golden test
python -m pytest tests/e2e_golden_test.py -m generation -k "<model>" -v

# ORT GenAI runtime
python -m pytest tests/ort_genai_test.py -m integration_slow -k "<model>" -sv

# Quantization
python -m pytest tests/quantization_integration_test.py -v

# CLI build
mobius build --model <hf-model-id> /tmp/out
```
