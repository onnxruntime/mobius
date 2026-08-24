# CLI Reference

Complete reference for the `mobius` command-line interface.

## Usage

```bash
mobius <command> [options]
```

## `mobius build`

Build an ONNX model from a HuggingFace model ID or local config directory.

### Synopsis

```bash
mobius build --model MODEL_ID --output OUTPUT_DIR [options]
mobius build --config CONFIG_PATH --output OUTPUT_DIR [options]
```

The model task is auto-detected from the model type. For example, Whisper
models automatically use `speech-to-text`, standard LLMs use
`text-generation`, and diffusers pipelines are detected and built as
multi-component packages.

### Output Option

| Option | Description |
|--------|-------------|
| `--output OUTPUT_DIR`, `-o OUTPUT_DIR` | Required output directory for the ONNX model files. Created if it doesn't exist. |

### Source Options (mutually exclusive)

| Option | Description |
|--------|-------------|
| `--model MODEL_ID` | HuggingFace model identifier (e.g. `meta-llama/Llama-3-8B`). Downloads config and weights from the Hub. |
| `--config CONFIG_PATH` | Path to a local model directory containing `config.json`. Safetensors weights are also required unless `--no-weights` is provided; use `--no-weights` to build from a config-only directory. Alternative to `--model`. |

### Execution Provider (`--ep`)

```
--ep EP, --execution-provider EP
```

Target execution provider for EP-aware optimizations. Default: `default`
(portable ONNX with no vendor-specific fusions).

EP-aware building drives the **entire** build pipeline — graph construction,
operator fusion, dead input removal, and KV cache sizing are all tailored
for the target EP. This is the recommended way to optimize for a specific
runtime or hardware target.

#### Available Execution Providers

| EP | Typical dtype | Description |
|----|---------------|-------------|
| `default` | any | Portable ONNX — no EP-specific vendor fusions (e.g. no GQA/PackQKV). Standard fusions are emitted as model local functions. |
| `cpu` | `f32` | ORT CPU inference — GQA fusion for FP32. |
| `cuda` | `f16` or `bf16` | NVIDIA GPU — GQA fusion, SkipNorm, PackQKV. |
| `dml` | `f16` | DirectML (Windows GPU) — GQA without fused RoPE. |
| `trt-rtx` | `f16` or `bf16` | NVIDIA TensorRT-RTX — GQA, no SkipLayerNorm. |
| `webgpu` | `f16` or `f32` | Browser / WebAssembly — Shape ops replaced with portable alternatives. |
| `onnx-standard` | any | Strict ONNX standard — zero custom-domain ops; safe for any conformant ONNX runtime. |

Run `mobius list eps` to see all registered execution providers and their
capabilities.

#### Examples

```bash
# Default (portable ONNX with standard fusions as model local functions)
mobius build --model meta-llama/Llama-3.2-1B --output output/

# CPU (GQA fusion for f32)
mobius build --model meta-llama/Llama-3.2-1B --output output/ --ep cpu

# CUDA GPU (GQA, SkipNorm, PackQKV fusions for f16/bf16)
mobius build --model meta-llama/Llama-3.2-1B --output output/ --ep cuda --dtype f16

# DirectML (GQA without fused RoPE)
mobius build --model meta-llama/Llama-3.2-1B --output output/ --ep dml --dtype f16

# TensorRT-RTX (GQA, no SkipLayerNorm)
mobius build --model meta-llama/Llama-3.2-1B --output output/ --ep trt-rtx --dtype f16

# WebGPU
mobius build --model meta-llama/Llama-3.2-1B --output output/ --ep webgpu --dtype f16

# Strict ONNX standard (zero custom ops)
mobius build --model meta-llama/Llama-3.2-1B --output output/ --ep onnx-standard
```

### Optimization Rules (`--optimize`)

```
--optimize [RULES]
```

Apply rewrite rules after building. Use without a value to apply all
available rules, or specify a comma-separated list of rule names.

**Use `--optimize` only** for manual, targeted rewrite rule application.
Rules are applied post-hoc and do not affect graph construction. This is
useful for experimentation or when `--ep` doesn't cover a specific
optimization.

#### Available Rules

| Rule | Description |
|------|-------------|
| `group_query_attention` | Fuse multi-head attention into GroupQueryAttention. |
| `packed_attention` | Pack Q/K/V projections into a single MatMul. |
| `skip_norm` | Fuse skip connections with normalization. |
| `skip_layer_norm` | Fuse skip connections with LayerNorm. |
| `bias_gelu` | Fuse bias addition with GELU activation. |

#### Examples

```bash
# Apply specific rules
mobius build --model meta-llama/Llama-3.2-1B --output output/ \
    --optimize=group_query_attention,skip_norm

# Apply all available rules
mobius build --model meta-llama/Llama-3.2-1B --output output/ --optimize

# Combine EP-aware building with additional post-hoc rules
mobius build --model meta-llama/Llama-3.2-1B --output output/ \
    --ep cuda --dtype f16 --optimize=bias_gelu
```

### `--ep` vs `--optimize`: When to Use Which

**Prefer `--ep`** for production builds. It affects both graph construction
and optimization (EP-aware KV cache sizing, dead input removal, operator
fusion), while `--optimize` only applies rewrite rules after the graph is
built.

They can be combined when you need both EP-aware construction and additional
post-hoc rules:

```bash
mobius build --model meta-llama/Llama-3.2-1B --output output/ \
    --ep cuda --dtype f16 --optimize=bias_gelu
```

### ORT GenAI Runtime (`--runtime`)

```
--runtime RUNTIME
```

Generate runtime-specific configuration files after building. Currently
supports `ort-genai`.

When set to `ort-genai`, mobius writes `genai_config.json` and copies
tokenizer files to the output directory:

- With `--model`: tokenizer files are downloaded from HuggingFace.
- With `--config` (local directory): tokenizer files are copied from that
  directory.

#### Example

```bash
mobius build --model Qwen/Qwen2.5-0.5B --output output/ \
    --ep cuda --dtype f16 --runtime ort-genai
```

### Build Features (`--features`)

Build-mode toggles are collected under a single cargo-style `--features`
option. Pass a comma-separated list (and/or repeat the flag):

```
--features fp8-kv-cache,static-cache
--features prune-prefill-prefix
--features text-only
```

Available features:

| Feature | Effect |
|---------|--------|
| `static-cache` | Pre-allocate fixed-size KV cache buffers using `TensorScatter` (pair with `--max-seq-len N`). Requires `DecoderLayer` / `MoEDecoderLayer` models. Cannot combine with `--task`. |
| `fp8-kv-cache` | Store the `GroupQueryAttention` KV cache as `FLOAT8E4M3FN` (per-tensor E4M3), halving KV-cache memory. Requires a GQA build (e.g. `--ep cuda --dtype f16`) and an ORT runtime with the FP8 KV-cache kernel (SM89+). Pair with `--kv-cache-scale-file` for calibrated scales. |
| `prune-prefill-prefix` | Emit logits shaped `[B, 1, vocab]` by selecting the final token before the LM head. Gemma 4 also prunes its KV-sharing layer suffix and per-layer inputs to reduce prefill compute. |
| `text-only` | Export the text backbone of a multimodal checkpoint as a standalone decoder-only LLM (see below). |

The legacy boolean flags `--static-cache`, `--fp8-kv-cache`, and
`--text-only` have been removed in favor of `--features`.

```bash
mobius build --model meta-llama/Llama-3.2-1B --output output/ \
    --features static-cache --max-seq-len 2048

mobius build --model Qwen/Qwen2.5-0.5B --output output/ \
    --ep cuda --dtype f16 --features fp8-kv-cache

mobius build --model meta-llama/Llama-3.2-1B --output output/ \
    --features prune-prefill-prefix
```

### Static Cache (`--features static-cache`)

```
--features static-cache
--max-seq-len N
```

Pre-allocate fixed-size KV cache buffers using TensorScatter. Useful when
the maximum sequence length is known up front.

- `--features static-cache` enables static cache mode. Requires models using
  `DecoderLayer` or `MoEDecoderLayer`.
- `--max-seq-len N` sets the maximum sequence length for static cache
  buffers. Only valid with static cache. Defaults to
  `max_position_embeddings` from the model config.

Cannot be combined with `--task`.

#### Example

```bash
mobius build --model meta-llama/Llama-3.2-1B --output output/ \
    --features static-cache

# With explicit max sequence length
mobius build --model meta-llama/Llama-3.2-1B --output output/ \
    --features static-cache --max-seq-len 2048
```

### Release Builds (`--release`)

```
--release
```

Reduce saved model size by removing build-time debug and provenance
metadata immediately before serialization. This includes source module paths,
class hierarchies, name scopes, originating rewrite rules, and
symbolic-shape-inference internals. Functional metadata whose keys begin with
`mobius.` is preserved.

`--release` applies to both `mobius build` and `mobius build-gguf`. It changes
metadata only, not the graph structure, weights, or inference behavior. Leave
it off when the build-time provenance would help inspect or debug the graph.

```bash
# Release export from a HuggingFace model
mobius build --model meta-llama/Llama-3.2-1B --output output/ --release

# Release export from GGUF
mobius build-gguf model.gguf --output output/ --release
```

### Other Flags

| Option | Description |
|--------|-------------|
| `--task TASK` | Model task (auto-detected if not specified). Use `mobius list tasks` to see available tasks. |
| `--dtype DTYPE` | Target dtype for model weights: `f16`, `bf16`, `f32` (also accepts `float16`, `bfloat16`, `float32`). If omitted, the dtype is auto-detected from the HuggingFace config (`torch_dtype`); provide `--dtype` to override it. Weights are cast at save time. |
| `--revision REVISION` | Immutable HuggingFace revision used consistently for config, weights, tokenizer, processor, and runtime metadata artifacts. |
| `--no-weights` | Export graph structure only, without weight data. Useful for inspection or testing. |
| `--external-data FORMAT` | External data format: `onnx` (default) or `safetensors`. |
| `--max-shard-size SIZE` | Maximum shard size for safetensors external data (e.g. `5GB`). Only used with `--external-data safetensors`. |
| `--release` | Strip build-time debug and provenance metadata before saving while preserving functional `mobius.` metadata. |
| `--trust-remote-code` | Trust remote code when loading the HuggingFace model config. |
| `--component NAME` | Build only one component from a diffusers pipeline (e.g. `--component vae_decoder`). |
| `--kv-cache-scale-file PATH` | Optional JSON file of calibrated per-layer FP8 KV-cache scales (onnxruntime-genai format). Only used with the `fp8-kv-cache` feature; without it all layers use a unit scale of 1.0. |

#### Text-only example

```bash
# Export gemma-4-12B's text backbone as a GQA decoder-only LLM
mobius build --model google/gemma-4-12B --output output/ \
    --features text-only --ep cuda --dtype f16
```

For a full ORT-GenAI text-only package (with `genai_config.json`), use
`auto_export(..., text_only=True)` — see
`examples/gemma4_12b_text_ort_genai.py`.

### More Examples

```bash
# Build from a HuggingFace model ID
mobius build --model Qwen/Qwen2.5-0.5B --output output_dir/

# Build without weights (graph skeleton only)
mobius build --model meta-llama/Llama-3.2-1B --output output_dir/ --no-weights

# Build from a local config directory
mobius build --config /path/to/model/ --output output_dir/

# Export with safetensors external data
mobius build --model Qwen/Qwen2.5-0.5B --output output_dir/ \
    --external-data safetensors

# Build encoder-decoder model (produces encoder.onnx + decoder.onnx)
mobius build --model openai/whisper-tiny --output output_dir/

# Build a diffusers pipeline (auto-detected)
mobius build --model Qwen/Qwen-Image-2512 --output output_dir/

# Build only the VAE decoder from a diffusers pipeline
mobius build --model Qwen/Qwen-Image-2512 --output output_dir/ \
    --component vae_decoder

# Override task explicitly
mobius build --model google/gemma-3-4b-pt --output output_dir/ \
    --task vision-language

# Build for ORT GenAI runtime
mobius build --model Qwen/Qwen2.5-0.5B --output output_dir/ \
    --ep cuda --dtype f16 --runtime ort-genai
```

---

## `mobius build-gguf`

Build an ONNX model from a GGUF file (e.g. from llama.cpp).
Supported GGUF quantization is preserved by default. This can involve
byte-preserving native blocks in text-only builds, affine repacking, or
dequantize/requantize for multimodal and mixed source qtypes.

> **Note**: Requires the optional `gguf` package: `pip install mobius-onnx[gguf]`

### Synopsis

```bash
mobius build-gguf GGUF_PATH --output OUTPUT_DIR [options]
```

### Arguments

| Argument | Description |
|----------|-------------|
| `GGUF_PATH` | Path to a `.gguf` model file. |

### Options

| Option | Description |
|--------|-------------|
| `--output OUTPUT_DIR`, `-o OUTPUT_DIR` | Required output directory for the ONNX model. |
| `--max-shard-size SIZE` | Maximum external-data shard size (e.g. `5GB`). |
| `--dequantize` | Explicitly dequantize all GGUF weights to float. |
| `--dtype DTYPE` | Target dtype for model weights: `f16`, `bf16`, `f32`. |
| `--external-data FORMAT` | External data format: `onnx` (default) or `safetensors`. |
| `--ep EP` | Target execution provider for EP-aware optimization. |
| `--runtime RUNTIME` | `onnx-genai` emits supported runtime metadata. `ort-genai` is rejected until GGUF cache/tokenizer generation coverage exists. |
| `--release` | Strip build-time debug and provenance metadata before saving while preserving functional `mobius.*` metadata. |
| `--static-cache` | Build a fixed-width cache where supported. |
| `--max-seq-len N` | Set the fixed cache length; requires `--static-cache`. |

### Examples

```bash
# Basic GGUF conversion (preserves supported quantization)
mobius build-gguf model.gguf --output output/

# Explicitly dequantize all weights
mobius build-gguf model.gguf --output output-float/ --dequantize

# Convert with specific dtype
mobius build-gguf model.gguf --output output/ --dtype f16
```

F32-, F16-, and BF16-only files build normally as float models because they
contain no quantization to preserve.
Quantized files containing only qtypes with no supported preservation target
(for example, pure Q6_K or Q5_K weights) fail instead of silently becoming
float. Re-run with `--dequantize` to request explicit float conversion.

Encoder-only BERT and ModernBERT GGUF backbones auto-select
`feature-extraction` and output `last_hidden_state`; they do not produce logits
or cache tensors. Static cache, generative task overrides, pooled/reranker
metadata, classifier tensors, and unsupported ModernBERT sliding-window variants
are rejected explicitly.
Quantized encoder linear weights use `MatMulNBits`, but quantized token
embeddings dequantize because these graphs do not yet implement
`GatherBlockQuantized`. BERT GQA metadata and quantized fused QKV are rejected;
float fused QKV is split losslessly.

Sharded GGUF inputs are rejected because a single shard has an incomplete
tensor table. `nemotron_h_moe` is also rejected until its MTP block, Mamba2
parity, mixed expert quantization, tokenizer provenance, and real ORT/ORT GenAI
generation are validated. See
[`build_from_gguf()`](api/build_from_gguf.md#nvidia-nemotron-35-lightning-waiver).

---

## `mobius list`

List supported models, tasks, dtypes, or execution providers.

### Synopsis

```bash
mobius list {models,tasks,dtypes,eps}
```

### Resources

| Resource | Description |
|----------|-------------|
| `models` | All supported model architectures with their default task and category. |
| `tasks` | Available task types (e.g. `text-generation`, `vision-language`). |
| `dtypes` | Supported dtype options with aliases. |
| `eps` | Registered execution providers with capabilities. |

### Examples

```bash
# List all 130+ supported model architectures
mobius list models

# List all available tasks
mobius list tasks

# List available dtype options
mobius list dtypes

# List execution providers and their capabilities
mobius list eps
```

---

## `mobius info`

Show information about a model without building it. Displays model type,
task, module class, and key config fields.

### Synopsis

```bash
mobius info MODEL_ID [--trust-remote-code]
```

### Arguments

| Argument | Description |
|----------|-------------|
| `MODEL_ID` | HuggingFace model ID to inspect. |

### Options

| Option | Description |
|--------|-------------|
| `--trust-remote-code` | Trust remote code when loading the HuggingFace model config. |

### Examples

```bash
# Inspect a transformers model
mobius info meta-llama/Llama-3.2-1B

# Inspect a diffusers pipeline
mobius info Qwen/Qwen-Image-2512
```
