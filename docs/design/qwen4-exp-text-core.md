# Qwen4-Exp multimodal pipeline

Mobius implements both the `qwen4_exp_text` decoder and the
`Qwen4ExpForConditionalGeneration` composite published as
`Qwen/Qwen3.8-Flash-Next`. The composite exports a standard three-model package:
`decoder`, `vision_encoder`, and `embedding`. `text_only=True` selects the same
decoder without the vision stages.

The implementation evidence was collected from:

- `Qwen/Qwen3.8-Flash-Next@f5d08274bafd880402bd16f5e3e6c514136ec06c`
- `unsloth/Qwen3.8-Flash-Next-FP8@41cc25fe32cc20053a59c89716196897580cddf6`
- `unsloth/Qwen3.8-Flash-Next-GGUF@d3bc75ee6ccef3efc1e228ec00a6cc2cdb1e2249`
- `huggingface/transformers@598d8ba8baaec7fec5a22da0e2844c7bf4ea20e1`

Exported models record the semantic reference as `mobius.semantic_reference_revision` and
record the caller's requested checkpoint revision separately as
`mobius.source_revision` (`unpinned` when no revision was supplied).

## Exported architecture

The ONNX graph includes the repeating three-linear/one-QSA attention
schedule, full-kernel Gated-DeltaNet convolution state, recurrent delta-rule
state, four-stream gated residual hyper-connections, exact softmax-first
top-k routed MoE plus the sigmoid-gated shared expert, QSA block pooling and
token selection, and PLE hashed n-gram embeddings with their dilated
convolution and token-context states. The evidenced BF16 checkpoint keeps
DeltaNet recurrent math and recurrent cache state in float32, while convolution
state, projections, sparse-attention caches, and logits remain in model dtype.
Official safetensors are loaded through a bounded-memory package transaction:
decoder, embedding, and vision bindings are all validated from one shard index
before any graph is mutated; parameter payloads remain lazy and no source state
dict is retained. The PLE table is allocated once when serialized, then
populated one checkpoint shard at a time.

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

The multimodal decoder takes fused `inputs_embeds` and the original lexical
`ple_input_ids` as independent inputs. Position state has shape `[4, B, S]`:
channel 0 is the text/causal sequence axis, while channels 1–3 carry temporal,
height, and width M-RoPE positions used by both sparse attention and QSA.

## Vision and embedding reuse

The checkpoint's vision config proves identity with the no-DeepStack Qwen3.5
tower: 27 blocks, hidden size 1152, intermediate size 4304, 16 heads, patch
size 16, temporal patch size 2, spatial merge size 2, and 2304 learned position
embeddings. Mobius reuses the Qwen3 vision implementation with DeepStack
disabled and the merger projected to the decoder width of 2560.

The embedding graph scatters `image_features` at token 248056 while preserving
the original token IDs for PLE. The processor contract follows the evidenced Qwen3
image processor with vision start/end tokens 248053/248054. This package is
explicitly image-only: config extraction validates the checkpoint's video token
but removes it from runtime metadata, the embedding graph exposes no video
feature input, and direct configs that request video support fail closed. The
embedding graph also publishes `mobius.unsupported_token_ids` and carries a
dynamic ONNX Reshape guard: any source video token is sanitized before
vocabulary lookup and then requests two output elements from a one-element
tensor. ONNX's element-count invariant is enforced before every ORT EP executes,
so direct graph execution cannot silently treat `<|video_pad|>` as ordinary
text. This deliberately avoids out-of-range Gather, which CUDA zero-fills.

QSA uses standard ONNX operators to reproduce the selected-token mask, then
runs ordinary dense attention under that mask. This is numerically faithful,
including contiguous left padding, but it does not provide the memory savings
of a dedicated sparse-attention runtime kernel.

## Guarded features

The evidenced ordinary Transformers forward preserves MTP metadata but does not
execute its `mtp.*` sidecar. Mobius mirrors that next-token route and does not
publish an MTP task. Dedicated MTP embeddings fail closed because no flattened
NextN cache ABI exists. Alternative vision geometries and nonempty DeepStack
configurations also fail closed.

## GGUF header support and payload guard

The GGUF evidence artifact is a text-only `general.architecture=qwen4exp` split set:

| Shard | Tensors | Bytes | LFS SHA-256 |
|---|---:|---:|---|
| `UD-IQ1_S/Qwen3.8-Flash-Next-UD-IQ1_S-00001-of-00003.gguf` | 0 | 10,946,624 | `88a1420825a9304063e882ada29d438263617f51ac8923d438d927496693bafd` |
| `UD-IQ1_S/Qwen3.8-Flash-Next-UD-IQ1_S-00002-of-00003.gguf` | 595 | 49,990,818,368 | `3a62e35bbf9add4733bd1438ebd3a67649d5edd6cb0e72bb78e33c913992b2b6` |
| `UD-IQ1_S/Qwen3.8-Flash-Next-UD-IQ1_S-00003-of-00003.gguf` | 629 | 22,544,696,352 | `0e25ceaeb89b8a80aa973c6c0c7448943682f7408c2855b2ebd016b7643a861a` |

Shard 0 owns all model/tokenizer metadata and no tensors in that evidence set.
Production routing is not bound to its repository, revision, filenames, byte
sizes, hashes, or shard distribution. Bounded header inspection identifies
`general.architecture=qwen4exp`, while validation uses the model metadata and
complete 1,224-name tensor shape/qtype contract. GGUF's split indexer query/key
matrices are concatenated row-wise into Hugging Face's fused `index_qk_proj`;
they are not Q/K-permuted.

Payload conversion deliberately fails before tensor materialization. A
successful bounded-header preflight rejects the architecture before Hub payload
download. If that range request fails or the metadata header exceeds its bounded
range, the intentional best-effort fallback downloads the immutable file or
complete shard set for full local validation before reaching the same fail-closed
guard. The combined PLE table is an enormous IQ4_NL embedding for which the graph
has no compatible native gather ABI. Routed experts are rank-3 banks with IQ1_S
gate/up and IQ4_NL down tensors, while the released runtime has neither a
mixed-format sparse native-block MoE ABI nor real-weight execution evidence.
Treating these as ordinary affine `MatMulNBits` would be incorrect. Explicit
float dequantization is also rejected because the PLE table alone expands beyond
the bounded single-tensor materialization policy. The exact
header/config/mapping support is therefore a fail-closed foundation for future
runtime ABI work, not a quantized execution claim.

Released onnxruntime-genai and the current ONNX GenAI workflow schema cannot
represent Qwen4-Exp's `ple_input_ids`, four-axis position state, and
heterogeneous per-layer PLE/QSA membership. Both metadata exporters therefore
fail closed instead of emitting missing bindings, unsupported semantic keys, or
lossy `%d` cache templates. The decoder graph carries a separate
`mobius.state_manifest` metadata document with explicit role-to-layer
membership for direct ONNX Runtime orchestration.

## FP8 checkpoint evidence

The committed evidence for `unsloth/Qwen3.8-Flash-Next-FP8` was collected at
immutable revision `41cc25fe32cc20053a59c89716196897580cddf6`. Library builds do
not lock to that revision: omitting `revision` follows the Hugging Face default,
and an explicit branch, tag, or SHA is forwarded unchanged to config and weight
loading. Its 131 safetensors headers were range-read without downloading the
185.5 GB tensor payload. The schema evidence records the config/index hashes,
complete tensor census, and canonical header-schema hash.

The checkpoint uses three text-weight paths:

- 73,728 routed-expert matrices use `F8_E4M3` values plus BF16 inverse-scale
  grids. Every grid is validated as exactly
  `[ceil(rows / 128), ceil(cols / 128)]`.
- 128 PLE embedding shards use `F8_E4M3` storage and one shared BF16
  `ngram_embedding.weight_scale` scalar. The evidenced payload is BF16 bits
  `0x3951` (`0.00019931793212890625` as float32). Mobius requires that exact
  scalar and reconstructs each shard lazily as
  `shard.astype(target_dtype) * weight_scale`. The shards remain separate in
  the ONNX graph instead of concatenating a roughly 95 GiB dense table during
  export.
- Remaining text weights are ordinary BF16 tensors. The 943-entry
  `modules_to_not_convert` list resolves completely against the evidenced header.

By default Mobius preserves every FP8 code tensor and BF16 scale tensor as an
external-data initializer. Standard ONNX QDQ reconstructs the logical weights:

- a 2-D block weight is padded and transformed
  `[R,C] -> [Br,128,Bc,128] -> [Br,Bc,128,128] -> [Br*Bc,16384]`;
- its `[Br,Bc]` scale grid becomes `[Br*Bc]` and feeds
  `DequantizeLinear(axis=0)`;
- inverse reshape/transpose plus a final slice restores `[R,C]`;
- each PLE shard is scalar-dequantized independently before its Gather (standard
  ONNX Gather does not accept FLOAT8); a dependency chain prevents the next
  shard DQ from running before the previous token-sized Gather completes.
  Masked outputs accumulate without a full-table Concat or a roughly 95 GiB
  destination, so peak runtime storage is one code shard plus one dense shard
  plus its BF16 scale and any explicit output-dtype cast.

The transform is invertible for source codes, and the external data keeps their
exact bytes. `weight-loading-report.json` records
`output_weight_format: fp8_qdq`, `storage_preserving: true`, and
`native_fp8: false`: QDQ storage is faithful, but no current ORT execution or
fusion capability is claimed. Stock ORT may reject the float8 DQ kernel while
the ONNX package remains schema-valid and round-trippable.

Because ONNX IR buffers one output shard before flushing it, streaming packages
default to 1 GiB output shards, reject shard limits above 5 GB, and force
external-data serialization to one worker. Missing scales, wrong grids,
changed deterministic PLE buffers, orphan scales, duplicate source names,
unknown tensors, and missing graph targets all fail closed.

The report separates excluded checkpoint families instead of hiding them in a
single aggregate: all 3,101 `mtp.*` tensors are marked
`mtp_exported: false` with the missing-forward/cache-ABI reason, and all 333
`model.visual.*` tensors are identified as belonging to the dependent
multimodal PR.

```bash
mobius build \
  --model unsloth/Qwen3.8-Flash-Next-FP8 \
  --revision 41cc25fe32cc20053a59c89716196897580cddf6 \
  --text-only \
  --external-data safetensors \
  --max-shard-size 5GB \
  output/
```

Pass `--dequantize` (or API `keep_quantized=False`) only when an explicitly
dense BF16 reconstruction is required.

The FP8 loader currently targets the text-only component and reports visual
tensors separately. The branch includes the multimodal graph implementation,
but FP8 visual-package loading is not claimed until every visual tensor is
classified through the same strict streaming contract. The MTP sidecar remains
excluded for the same authoritative-forward/cache-ABI reason documented above.
NVFP4 checkpoint lowering remains out of scope.
