---
name: onnx-export-quantization
description: >
  Use this skill when exporting ONNX models with mobius and quantizing
  them with Olive for deployment. Covers the mobius CLI, EP options,
  INT4 quantization (Q4_K_M and NF4), HuggingFace upload structure,
  GPU-accelerated quantization, common issues, and testing quantized
  models.
---

# Skill: ONNX Export and Quantization

## When to use

Use this skill when:
- Exporting a model from HuggingFace to ONNX format using `mobius build`
- Quantizing an ONNX model to INT4 (Q4_K_M or NF4) with Olive
- Uploading ONNX models to HuggingFace Hub in the standard directory layout
- Debugging export or quantization failures
- Choosing between execution provider (EP) variants

## Exporting models with `mobius build`

### Basic command

```bash
mobius build \
  --model <hf-model-id> \
  --dtype <f16|bf16> \
  --ep <default|cuda|onnx-standard> \
  --runtime ort-genai \
  --external-data safetensors \
  --max-shard-size 5GB \
  <output-directory>/
```

### Flag reference

| Flag | Description |
|------|-------------|
| `--model <id>` | HuggingFace model ID (e.g. `google/gemma-4-27b-it`) |
| `--dtype <f16\|bf16>` | Model precision — `f16` (float16) or `bf16` (bfloat16) |
| `--optimize [RULES]` | Apply mobius rewrite rules after building (e.g. `group_query_attention`, `packed_attention`, `skip_norm`). Use without value for all rules, or specify comma-separated names. Not needed for basic exports. |
| `--ep <variant>` | Execution provider variant (see below) |
| `--runtime ort-genai` | Generate `genai_config.json` and copy tokenizer files for ORT GenAI runtime |
| `--external-data safetensors` | Store weights externally in safetensors format |
| `--max-shard-size 5GB` | Split external data into shards ≤ 5GB |

### Execution provider (EP) variants

Build separate ONNX models per EP because each applies different graph
rewrites and fused ops:

| EP | Flag | When to use |
|----|------|-------------|
| `default` | `--ep default` | Portable ONNX — no vendor-specific fusions. Compatible with all execution providers and runtimes. This is the default if `--ep` is omitted. |
| `cuda` | `--ep cuda` | NVIDIA GPU inference. Emits `com.microsoft` fused ops (GroupQueryAttention, MoE, etc.) for maximum CUDA performance. |
| `onnx-standard` | `--ep onnx-standard` | Strict ONNX-only — inlines all custom-domain functions into standard ONNX ops. Use when targeting runtimes that don't support `com.microsoft` ops. |

Other EPs are available (`cpu`, `dml`, `webgpu`, `trt-rtx`). Run
`mobius list eps` to see all options.

**Typical export matrix:** Build each dtype × EP combination:

```bash
for dtype in f16 bf16; do
  for ep in default cuda onnx-standard; do
    mobius build --model google/gemma-4-12b-it \
      --dtype $dtype --ep $ep \
      --runtime ort-genai \
      --external-data safetensors --max-shard-size 5GB \
      output/${dtype}/${ep}/
  done
done
```

### Multi-model outputs

For multimodal models (VLMs, audio-language), `mobius build` produces
multiple sub-models:

```
output/
├── decoder/           # Text decoder
│   ├── model.onnx
│   └── model.onnx.data.safetensors
├── embedding/         # Embedding model
│   ├── model.onnx
│   └── model.onnx.data.safetensors
├── vision_encoder/    # Vision encoder (VLMs)
│   ├── model.onnx
│   └── model.onnx.data.safetensors
├── audio_encoder/     # Audio encoder (ALMs)
│   ├── model.onnx
│   └── model.onnx.data.safetensors
└── genai_config.json
```

### Direct GGUF import

Direct GGUF conversion preserves supported quantization by default because the
usual intent is a quantized ONNX model:

```bash
mobius build-gguf model.gguf --output output/
```

The equivalent API call is:

```python
from mobius import build_from_gguf

package = build_from_gguf("model.gguf")
```

Use `mobius build-gguf model.gguf --dequantize` or
`build_from_gguf("model.gguf", keep_quantized=False)` to request a fully float
model. The older `--keep-quantized` CLI flag remains a compatibility alias for
the default and must not be combined with `--dequantize`. F32-, F16-, and
BF16-only GGUFs build through the float path because they have no quantized
tensors to preserve.

Preservation does not mean every source tensor remains byte-identical. In
text-only builds, runtime-native IQ/MXFP4 projection blocks can retain their
bytes. Supported affine blocks are repacked, but the conversion can be lossy
for source types such as Q4_1 and Q4_K. Multimodal, mixed, or unsupported source
qtypes may be dequantized and requantized or rejected. Apply the classification
and acceptance checks below before making preservation claims.

### Direct GGUF import acceptance

Do not infer a GGUF's quantization from its filename. Presets such as
`Q4_K_M`, `MXFP4_MOE`, and Unsloth Dynamic variants can contain large tensors
in different per-tensor formats (for example Q5_0, Q5_1, Q8_0, and MXFP4).
Before implementing or claiming a direct conversion:

1. Pin the GGUF repository revision, filename, file size, and LFS SHA-256.
2. Inspect the GGUF metadata and complete tensor table (names, logical shapes,
   and quantization types) without downloading tensor payloads when range
   requests are available.
3. Compare the layer schedule and tensor shapes with a separately pinned
   official config and safetensors headers. `block_count` may include auxiliary
   MTP/draft blocks rather than decoder backbone layers.
4. Classify every large tensor as byte-preserved native blocks, affine
   repacking (lossless or lossy, with numerical validation), or
   dequantize/requantize. If any large tensor takes the third path, the
   conversion does **not** preserve the source quantization.
5. Reject split GGUF shards unless the importer explicitly assembles every
   shard. Reading one shard can build a plausible but incomplete model.
6. Compare embedded tokenizer special-token IDs and chat template with the
   pinned upstream tokenizer. A self-contained package is invalid when padding,
   EOS, or BOS semantics disagree.
7. Require real-weight full-logit parity and deterministic multi-token
   generation through both ONNX Runtime and the declared GenAI runtime. Graph,
   config, and session creation are not acceptance evidence.

Use Hub GGUF architecture metadata to fail before downloading multi-gigabyte
unsupported files, then repeat the guard from the local GGUF header so local
paths receive the same actionable error.

## Quantization with Olive

### Installation

Olive with ONNX quantization support (install from PR if needed for
latest features):

```bash
pip install olive-ai
# Or from a specific PR for bleeding-edge features:
pip install git+https://github.com/microsoft/Olive.git@refs/pull/2406/head
```

For GPU-accelerated quantization (highly recommended for large models):

```bash
pip install cupy-cuda12x
```

### Q4_K_M quantization (k-quant)

K-quant quantization uses mixed block sizes with importance-based bit
allocation. Q4_K_M is a good balance of quality and size.

The repo uses Olive's config-driven `olive.run()` pattern (see
`examples/olive/` for working examples). A typical Olive config for
k-quant quantization:

```json
{
  "input_model": { "type": "OnnxModel", "model_path": "decoder/model.onnx" },
  "passes": {
    "kquant": {
      "type": "OnnxKQuantQuantization",
      "bits": 4,
      "block_size": 32
    }
  },
  "output_dir": "output/Q4_K_M/default/decoder"
}
```

```bash
olive run --config kquant_config.json
```

### NF4 quantization (4-bit NormalFloat)

NF4 uses a normal-distribution-optimized 4-bit format. Fast native C++
implementation — no GPU needed.

```json
{
  "input_model": { "type": "OnnxModel", "model_path": "decoder/model.onnx" },
  "passes": {
    "nf4": {
      "type": "OnnxBnb4Quantization",
      "precision": "nf4"
    }
  },
  "output_dir": "output/NF4/default/decoder"
}
```

> See `examples/olive/ministral-3-3b-vlm/` for a complete working
> example that combines mobius export with Olive quantization.

### GPU acceleration with cupy

Installing `cupy-cuda12x` gives a **19–51x speedup** for k-quant
quantization:

| Method | CPU time per matrix | GPU time per matrix | Speedup |
|--------|--------------------|--------------------|---------|
| K-quant (Q4_K_M) | 3–27s | 0.17–0.52s | 19–51x |
| NF4 | 42ms for 67M params | N/A (C++ native) | Already fast |

```bash
# Install cupy for CUDA 12.x
pip install cupy-cuda12x

# Olive auto-detects cupy and uses GPU when available
```

### Isolate Olive from unrelated provider DLLs

Olive 0.13 may auto-register every provider DLL bundled in an ORT GPU wheel
even when a weight-only pass explicitly targets CPU. A missing TensorRT DLL can
then abort K-quant before the pass starts. Keep the accelerator CPU-only and,
for programmatic workflows, suppress `olive.systems.local` EP-library
registration around `olive.workflows.run`; restore it immediately afterward.
This is safe for `OnnxKQuantQuantization`, which does not create an inference
session. Still load and execute the resulting package with the intended EP.

### Quantizing multi-model exports

Quantize each sub-model independently. Typically only the decoder is
quantized (it has the most parameters). Copy all other files needed
for a complete ORT GenAI package:

```bash
# Quantize decoder only (largest model)
olive run --config kquant_decoder.json

# Copy other sub-models as-is (already small)
cp -r output/f16/default/embedding/ output/Q4_K_M/default/embedding/
cp -r output/f16/default/vision_encoder/ output/Q4_K_M/default/vision_encoder/

# IMPORTANT: Copy config, tokenizer, and processor files too
cp output/f16/default/genai_config.json output/Q4_K_M/default/
cp output/f16/default/tokenizer* output/Q4_K_M/default/
cp output/f16/default/image_processor.json output/Q4_K_M/default/ 2>/dev/null
cp output/f16/default/audio_processor.json output/Q4_K_M/default/ 2>/dev/null
```

Without the tokenizer and processor config files, ORT GenAI will fail
to load the model.

## HuggingFace upload structure

### Standard directory layout

```
<org>/<model>-onnx/
├── f16/
│   ├── default/          # Portable ONNX (no vendor fusions)
│   ├── cuda/             # CUDA EP (fused ops)
│   └── onnx-standard/   # Strict ONNX-only (inlined functions)
├── bf16/
│   ├── default/
│   ├── cuda/
│   └── onnx-standard/
├── Q4_K_M/
│   └── default/          # Quantized models typically CPU-only
└── NF4/
    └── default/
```

Each EP directory contains the full model structure (decoder/,
embedding/, vision_encoder/, audio_encoder/ as applicable) plus
`genai_config.json`.

### Upload with huggingface_hub

```python
from huggingface_hub import HfApi

api = HfApi()
api.upload_folder(
    folder_path="output/f16/default",
    path_in_repo="f16/default",
    repo_id="org/model-onnx",
    repo_type="model",
)
```

### Verify uploads

After uploading, verify all shards are present. Incomplete uploads are
a common issue with large models:

```python
from huggingface_hub import HfApi

api = HfApi()
files = api.list_repo_files("org/model-onnx")
# Check that all expected .safetensors shards exist
for variant in ["f16/default", "f16/cuda", "bf16/default"]:
    shards = [f for f in files if f.startswith(variant) and f.endswith(".safetensors")]
    print(f"{variant}: {len(shards)} shards")
```

## Common issues and fixes

### 1. MoE expert weight mapping

Models with Mixture-of-Experts (e.g. Gemma4 26b-a4b) may need expert
weight remapping in `preprocess_weights()`. HuggingFace stores experts
as 3D tensors (`experts.gate_up_proj [E, 2*inter, H]`) that must be
mapped to the fused MoE op's parameter names (`fc1_experts_weights`,
`fc2_experts_weights`).

**Symptom:** Weight loading errors or incorrect MoE outputs.

**Fix:** Check the model's `preprocess_weights()` maps HF expert weight
names to the ONNX parameter names. See the `moe-models` skill for
the pattern.

### 2. Hybrid attention v_proj shape mismatches

Models with hybrid attention (e.g. Gemma4 31b with different `head_dim`
for local vs global attention layers) may have shape mismatches in
value projections.

**Symptom:** Shape errors during weight loading or forward pass.

**Fix:** Ensure `v_proj` dimensions account for per-layer head
configurations. Check `num_global_key_value_heads` vs
`num_key_value_heads` in the config.

### 3. CUDA GQA head_dim limitations

Older versions of ORT had a limitation where `head_dim > 256` would fail
with the CUDA GroupQueryAttention kernel.

**Symptom:** CUDA runtime error during inference with large head
dimensions.

**Status:** This limitation has been removed in recent ORT versions.
If using an older ORT build, fall back to `--ep default` or
`--ep onnx-standard`.

### 4. Incomplete uploads

Large models with many shards can have incomplete uploads to HuggingFace
Hub, especially on unstable connections.

**Symptom:** Model fails to load with file-not-found errors for
specific shard files.

**Fix:** Verify all shards are present after upload (see the verify
script above). Re-upload missing shards with `api.upload_file()`.

### 5. BF16 type mismatches

Some components may produce FP32 outputs when the model is built in
BF16, causing type mismatch errors in ORT.

**Symptom:** `Type Error: Type parameter (T) bound to different types
(tensor(bfloat16) and tensor(float))`.

**Fix:** Check for constants, initializers, or norm layers that stay
FP32 when the model is BF16. Add `op.CastLike(result, input)` to
ensure dtype consistency. See the `reusable-components` skill's
section on precision behaviour.

### 6. Attention rewrite schema mismatch

Quantizers may rewrite Attention to a contrib op with fewer outputs or a
different cache contract. Restrict the quantized op set when schemas do not
match, then load and execute the result; successful conversion alone is not
evidence. Document the narrowed recipe and unsupported rewrite explicitly.

## Testing quantized models

### L4: Golden data generation

Generate reference outputs from the full-precision HuggingFace model
using the golden data generation script:

```bash
# Generate golden files for all test cases
python scripts/generate_golden.py

# Generate for a specific task type
python scripts/generate_golden.py --task-type causal-lm

# Generate for a single test case
python scripts/generate_golden.py --case testdata/cases/causal-lm/gpt2.yaml

# Use GPU for large models
python scripts/generate_golden.py --device cuda
```

Golden reference files are stored in `testdata/golden/` as JSON. Use
`compare_golden()` from `mobius._testing.parity` to compare model
outputs against the reference:

```python
from mobius._testing.parity import compare_golden

compare_golden(
    model_output=output_logits,
    golden_path="testdata/golden/causal-lm/my_model.json",
)
```

### Optional ORT GenAI downstream smoke test

When useful, run inference with the quantized model through ORT GenAI:

```python
import onnxruntime_genai as og

model = og.Model("output/Q4_K_M/default/")
tokenizer = og.Tokenizer(model)
params = og.GeneratorParams(model)
params.set_search_options(max_length=50, do_sample=False)
params.input_ids = tokenizer.encode("Hello, world!")

output_ids = model.generate(params)
print(tokenizer.decode(output_ids[0]))
```

ORT GenAI acceptance is not a Mobius export gate. Always validate the final
quantized ONNX package directly; treat ORT GenAI load/generation as optional
downstream evidence and record its version/outcome without blocking export.

### Numerical parity verification

Quantized models will have some numerical divergence from the
full-precision model. Expected tolerances:

| Quantization | Typical divergence | Notes |
|-------------|-------------------|-------|
| Q4_K_M | Moderate | Top-1 token agreement ~95%+ for coherent text |
| NF4 | Moderate | Similar to Q4_K_M |
| F16 (no quant) | Minimal | Should match BF16 closely |

Verify that generated text is coherent and semantically correct rather
than requiring exact numerical matches.

## Inference speed: Q4_K_M vs NF4

For the benchmark table, use the canonical reference in
`.agents/skills/profiling-onnx-models/SKILL.md` ("Quantization benchmark
reference (Gemma4 E2B-IT)").

**Q4_K_M is recommended over NF4** for both speed and quality:
- Faster on both CPU and CUDA than F16 in the referenced measurement
- NF4 is slower than F16 in the same measurement
- Quality is comparable between Q4_K_M and NF4 in spot checks
- Q4_K_M uses less memory than F16 (~4x compression)

## Cross-references

- **Adding models:** `.agents/skills/adding-a-new-model/SKILL.md`
- **MoE weights:** `.agents/skills/moe-models/SKILL.md`
- **Component precision:** `.agents/skills/reusable-components/SKILL.md`
- **ORT GenAI config:** `.agents/skills/ort-genai-config/SKILL.md`
- **Quality checklist:** `.agents/skills/quality-checklist/SKILL.md`
