---
name: mobius-onnx-export-gotchas
description: Use when building ONNX models with the Mobius CLI, especially when selecting an execution provider or static KV cache.
---

# Mobius ONNX export gotchas

## CLI syntax

`mobius build` requires `--model <hf_id>` and accepts `--output` / `-o` for the
output directory.

```bash
mobius build --model microsoft/Phi-3.5-mini-instruct \
  --dtype f16 --execution-provider cuda \
  --external-data onnx --trust-remote-code \
  --output /path/to/output_dir
```

Valid dtype aliases include `f16`/`float16`, `bf16`/`bfloat16`, and
`f32`/`float32`. `fp16` is not accepted.

## EP selection is not post-export optimization

`--execution-provider` is an alias of `--ep`. It selects construction-time
operator contracts and runtime packaging defaults. Mobius performs exporter
cleanup and local-function inlining, but no longer applies post-export rewrite
rules.

Apply graph fusions and EP-specific lowerings with Olive after export.

## Static cache and direct GQA emission

Static-cache graphs use standard ONNX Attention with explicit TensorScatter KV
updates. The direct GroupQueryAttention construction path uses the runtime's
shared past/present KV buffer instead.

- For direct GQA-capable exports, select the appropriate EP without
  `--features static-cache`.
- For explicit fixed-size KV buffers, use
  `--features static-cache --max-seq-len N`.

## Validate the saved artifact

Always load the saved ONNX model with the intended runtime before profiling.
Check operator domains, graph inputs and outputs, initializer dtypes, and one
representative inference result. HTTP or serialization success alone does not
prove that the runtime accepts the graph.
