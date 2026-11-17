# GGUF import design

Mobius constructs model graphs from GGUF metadata without tracing PyTorch.
The current contract and evidence live in
[`build_from_gguf()`](../api/build_from_gguf.md) and the
[capability catalog](../gguf-capability-catalog.md).

## Packed storage

The experimental opset-29 path binds original GGUF byte ranges as external
tensors. Standard `DequantizePacked` nodes describe the stored codecs; ordinary
ONNX layout nodes express model-specific reshaping and permutation. Typed
tensors retain their original representation. Existing name/layout processors
run on metadata-only tensors, so the exporter does not expand weights.

Packed embedding lookup decodes selected rows. Ordinary bias-free gated MoE
experts use `GroupedMatMul`, preserving routing and scoped parameter names.
Decoder/matmul fusion is an execution-provider optimization, not a custom
exporter operator.

Source files are immutable dependencies, not temporary inputs. Flat packages
save beside their original GGUF shards, retain exact external bindings, and
record source hashes. Unknown formats, nonrepresentable transforms and
unsupported packaging fail explicitly.

## Legacy and explicit float routes

Exact legacy affine/native layouts remain available. Lossy zero-point rounding
and decode/requantize conversions have been removed; use the opset-29 path
instead. Explicit float-storage export is a separate user choice and does not
claim original-byte preservation.

The earlier QDQ/requantization exploration is retained in Git history rather
than presented as an available conversion option.
