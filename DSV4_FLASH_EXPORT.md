# DeepSeek-V4-Flash export design

## Source configuration

The authoritative checkpoint is
[`deepseek-ai/DeepSeek-V4-Flash`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash);
the target GGUF is
[`unsloth/DeepSeek-V4-Flash-GGUF`](https://huggingface.co/unsloth/DeepSeek-V4-Flash-GGUF).

- HF: `model_type=deepseek_v4`, `DeepseekV4ForCausalLM`.
- GGUF: `general.architecture=deepseek4`.
- 43 layers, hidden size 4096, 64 attention heads, one shared KV head,
  head size 512, rotary size 64, Q LoRA rank 1024.
- Output projection: 8 groups, rank 1024.
- MoE: 256 routed experts, top 6, intermediate size 2048, one shared
  expert, scale 1.5, normalized `sqrt(softplus(logit))` routing. The first
  three layers select experts from a token-id hash table.
- Hyper-Connections: multiplicity 4, 20 Sinkhorn iterations, epsilon `1e-6`.
- Context: 1,048,576. Raw/local KV uses base theta 10,000; compressed
  history uses YaRN factor 16 from 65,536 with theta 160,000.
- Sparse attention: 128-token local window plus learned KV compression
  (`compress_ratios` alternating 4/128), a learned indexer for ratio-4
  layers, compressed theta 160,000, and attention sinks.
- One MTP layer is present but is not part of ordinary target-model logits.

## V3 reuse and V4 differences

V3's shared-expert MoE structure and low-rank Q path remain useful concepts,
but V4 is not the V3 MLA layout: it has a single normalized 512-wide KV
projection, inverse RoPE on attention outputs, grouped low-rank output
projection, sqrt-softplus/hash routing, clamped SwiGLU, and
Hyper-Connections. A separate model module avoids forcing those semantics
into `DeepSeekMLA`.

This increment exports the V4 projections, full Hyper-Connections, V4 MoE
routing/shared experts, and KV-cache-compatible **dense causal attention**.
The dense graph is a structurally valid fallback for short-context bring-up,
but it is not numerically equivalent to CSA/HCA's learned compressed-history
selection. Compression, indexer tensors, attention sinks, and MTP are
explicitly skipped during weight preprocessing, so this support remains
preview-quality until the sparse runtime path lands.

## Quantization and follow-ups

Mobius can quantize the implemented `Linear` layers to its existing 4-bit or
8-bit `MatMulNBits` path. Unsloth Dynamic files also contain mixed low-bit
expert tensors (including 2/1-bit variants, and packed FP4/MXFP4 in some
presets) that are not directly representable by the currently deployed
MatMulNBits kernels.

Follow-ups:

1. Add runtime custom ops for compressed sparse attention/cache management
   and expose them through mobius execution-provider capability config.
2. Add a packed sub-4-bit/MXFP4 expert custom op, or track the needed
   MatMulNBits functionality in ONNX Runtime and teach the GGUF repacker to
   preserve those tensors.
3. Export the optional MTP head as a separate speculative-decoding component.
