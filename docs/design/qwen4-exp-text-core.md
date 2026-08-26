# Qwen4-Exp text core

Mobius implements the text decoder identified by Hugging Face
`model_type=qwen4_exp_text`, including the composite
`Qwen4ExpForConditionalGeneration` route when `text_only=True`.

The implementation is pinned to:

- `Qwen/Qwen3.8-Flash-Next@f5d08274bafd880402bd16f5e3e6c514136ec06c`
- `unsloth/Qwen3.8-Flash-Next-FP8@41cc25fe32cc20053a59c89716196897580cddf6`
- `unsloth/Qwen3.8-Flash-Next-GGUF@d3bc75ee6ccef3efc1e228ec00a6cc2cdb1e2249`
- `huggingface/transformers@598d8ba8baaec7fec5a22da0e2844c7bf4ea20e1`

Exported models record that pin as `mobius.semantic_reference_revision` and
record the caller's requested checkpoint revision separately as
`mobius.source_revision` (`unpinned` when no revision was supplied).

## Exported architecture

The ONNX graph includes the repeating three-linear/one-QSA attention
schedule, full-kernel Gated-DeltaNet convolution state, recurrent delta-rule
state, four-stream gated residual hyper-connections, exact softmax-first
top-k routed MoE plus the sigmoid-gated shared expert, QSA block pooling and
token selection, and PLE hashed n-gram embeddings with their dilated
convolution and token-context states. The pinned BF16 checkpoint keeps
DeltaNet recurrent math and recurrent cache state in float32, while convolution
state, projections, sparse-attention caches, and logits remain in model dtype.
Official safetensors are loaded through a bounded-memory transform: packed
experts are sliced lazily and the PLE table is allocated once, then populated
one checkpoint shard at a time.

The flattened cache ABI is:

```text
past_position_ids -> present_position_ids
linear layer:
  conv_state, recurrent_state
PLE linear layer:
  conv_state, recurrent_state, ple_conv_state, ple_context
QSA layer:
  key, value, index_key
```

QSA uses standard ONNX operators to reproduce the selected-token mask, then
runs ordinary dense attention under that mask. This is numerically faithful,
including contiguous left padding, but it does not provide the memory savings
of a dedicated sparse-attention runtime kernel.

## Guarded features

The pinned official Transformers implementation explicitly ignores `mtp.*`
checkpoint tensors and does not define the `fc_embedding`/`fc_hidden`
combination equation or a flattened NextN cache ABI. Mobius follows that
ordinary-decoder behavior: it preserves the source MTP metadata, warns while
dropping sidecar-only tensors, and exports the same next-token causal model.
Configurations with dedicated MTP embeddings fail closed because omitting
those embeddings could change the decoder contract. A future standalone
NextN sidecar requires an authoritative execution equation and cache ABI.

FP8/NVFP4 checkpoint lowering and the multimodal wrapper remain outside this
text-core implementation. The nested `qwen4_exp_text` configuration and
architecture registration are present so a later multimodal wrapper can reuse
the existing Qwen3/Qwen3.5 vision and embedding components without aliasing
this decoder.

## GGUF header support and payload guard

The pinned GGUF is a text-only `general.architecture=qwen4exp` split set:

| Shard | Tensors | Bytes | LFS SHA-256 |
|---|---:|---:|---|
| `UD-IQ1_S/Qwen3.8-Flash-Next-UD-IQ1_S-00001-of-00003.gguf` | 0 | 10,946,624 | `88a1420825a9304063e882ada29d438263617f51ac8923d438d927496693bafd` |
| `UD-IQ1_S/Qwen3.8-Flash-Next-UD-IQ1_S-00002-of-00003.gguf` | 595 | 49,990,818,368 | `3a62e35bbf9add4733bd1438ebd3a67649d5edd6cb0e72bb78e33c913992b2b6` |
| `UD-IQ1_S/Qwen3.8-Flash-Next-UD-IQ1_S-00003-of-00003.gguf` | 629 | 22,544,696,352 | `0e25ceaeb89b8a80aa973c6c0c7448943682f7408c2855b2ebd016b7643a861a` |

Shard 0 owns all model/tokenizer metadata and no tensors. The importer
inherits that metadata across the complete set and requires the exact
`0 + 595 + 629 = 1224` closure. Header validation covers every
hyper-connection, PLE, QSA/indexer, DeltaNet, routed/shared expert, and final
output mixer tensor. GGUF's split indexer query/key matrices are concatenated
row-wise into Hugging Face's fused `index_qk_proj`; they are not Q/K-permuted.

Payload conversion deliberately fails before Hub download. The combined PLE
table is an enormous IQ4_NL embedding for which the graph has no compatible
native gather ABI. Routed experts are rank-3 banks with IQ1_S gate/up and
IQ4_NL down tensors, while the released runtime has neither a mixed-format
sparse native-block MoE ABI nor real-weight execution evidence. Treating these
as ordinary affine `MatMulNBits` would be incorrect. Explicit float
dequantization is also rejected because the PLE table alone expands beyond the
bounded single-tensor materialization policy. The exact header/config/mapping
support is therefore a fail-closed foundation for future runtime ABI work, not
a quantized execution claim.
