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
the original token IDs for PLE. With an empty image-feature stream, it preserves
all lexical embeddings, including an image token emitted during cached decode,
matching the HF forward without new pixels. The processor contract follows the
evidenced Qwen3 image processor with vision start/end tokens 248053/248054. This
package is explicitly image-only: config extraction validates the checkpoint's video token
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

## Standard-ONNX-only export

Select the existing `execution_provider="onnx-standard"` to export the text
decoder or all three image-text components without custom-domain operators.
Other EPs retain the existing fused `com.microsoft::MoE` route, parameter
layouts, tasks, and cache ABI. No additional model alias or configuration flag
is needed.

BF16 image-text export with `onnx-standard` is rejected at the Qwen4-Exp
vision component, including when that component is built independently.
Official-weight production-size vision parity failed all four tested cases
at `rtol=atol=1e-2`, with maximum absolute error `0.04052734375`; the cause
remains unresolved. Use FP32/FP16 for image-text export, or `text_only=True`
for a BF16 decoder. BF16 text-only exports and other EPs are unchanged.
This guard does not establish numerical qualification for those alternatives.

The standard expert dispatch selects TopK from float32 router probabilities
with a one-dimensional INT64 K tensor. When requested, it divides the selected
probabilities by their float32 sum (not a second softmax), then casts route
weights to model dtype. A single Scan traverses ascending expert IDs and carries
only the `[T,H]` output accumulator, with no scan-output history. NonZero finds
token/route pairs for each expert; an If skips empty experts. The nonempty branch
scalar-gathers the expert's contiguous `[2I,H]` gate/up and `[H,I]` down matrices
from lexically captured parent banks, applies SwiGLU, multiplies by route weights,
and replaces the affected accumulator rows with their old values plus the
contribution. SiLU evaluates `gate / (1 + exp(-gate))` in FP32 and casts its
result back before multiplying by `up`, preserving HF's activation rounding
without prematurely materializing a low-precision sigmoid. Expert projections,
the up product, and accumulation stay in model dtype; optional routed scaling
is applied after accumulation. The shared sigmoid-gated expert is unchanged.

The compute graph is independent of expert count: E=2 and E=512 tiny fixtures
have the same recursive node count and exactly two MatMuls in the expert body.
Weight storage remains O(EHI); per-expert activation workspace is O(RI+RH),
where R<=T, plus routing `[T,K]` and accumulator `[T,H]`. There are no
`[T,K,H,I]` gathered banks or dense `[T,E,I]` activations. Scan still visits all
E experts and may incur control-flow, NonZero, and scatter overhead; this is
not a sparse-kernel performance claim. Existing FP8 QDQ bank reconstruction can
remain O(E) in runtime memory and is not optimized by this dispatch.

Standard-only validation runs after optimization, after package opset lowering,
and after Transformers dense/streaming/QDQ binding. It recursively checks main
graphs, GRAPH/GRAPHS attributes, and remaining function signatures and bodies
against standard domains and applicable ONNX schemas. Unresolved custom ops
fail with component/path/domain/op diagnostics instead of a warning or EP
fallback. This global guard may expose previously unresolved exports in other
architectures; it does not change other EPs.

Local evidence uses tiny synthetic text/image-text graphs, ONNX full checker
validation in FP32/FP16 (plus BF16 text-only), independent FP32 expert parity
(K=1/intermediate/E, ties, near ties, saturated logits, normalization, scaling,
mixed/unused experts, batches and single tokens), FP16 routing execution,
complete cache/logit prefill/decode comparisons across the QSA boundary, left padding, multimodal
decoder parity, a nonzero vision→embedding→decoder pipeline, and reduced dense
and storage-preserving FP8 QDQ streaming fixtures. The existing ORT
`1.30.0.dev20260823001` CPU runtime executes the FP32 tests and the tiny FP16
expert fixture; its BF16 expert fixture stops at the missing `Expand(13)`
kernel. Schema validity is not proof of kernel support for every dtype,
efficient sparse execution, or full-checkpoint generation parity. No full
checkpoint generation or performance benchmark is claimed.

An independent `torch.nn.functional.silu` boundary fixture checks CPU FP32 and
CUDA FP32/FP16/BF16 with graph optimization both disabled and enabled. It covers
negative tails amplified by large up values, ordinary values, and large finite
inputs through the actual Scan/If dispatch, and profiles CUDA expert MatMuls.
CPU FP16 numerical parity is not an acceptance target; its routing smoke test
does not establish HF SiLU parity.

This change does not enable video, MTP execution, GGUF payload loading, GenAI
engine integration, or a paged-cache ABI.

### Independent reduced-model parity

The focused integration tests use randomly initialized HF models, not outputs
from another Mobius graph. The reference source is Transformers commit
`598d8ba8baaec7fec5a22da0e2844c7bf4ea20e1`; it requires
`tokenizers>=0.23.1,<0.24.0`. Install or expose that reference in an isolated
environment before running:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m pytest \
  src/mobius/models/qwen4_exp_test.py -q -m integration \
  -k "reduced_random_weight_huggingface or reduced_standard_huggingface"
```

The primary reduced-model acceptance paths are CPU FP32 and CUDA FP16.
The standard-ONNX text cases also retain CUDA FP32/BF16 coverage. They compare
full prefill and per-step logits, greedy token choices, linear convolution/recurrent
state, PLE convolution/context,
QSA key/value/indexer caches, and position history. Text prefill spans the sparse
selection boundary and is followed by six cached decode steps. The image-text
cases run CPU FP32 and CUDA FP32/FP16 and compare two nonzero, differently shaped
image grids through the actual ONNX vision-to-embedding-to-decoder pipeline,
independently checking vision features and HF-fused embeddings, then three
decode steps with no new images.
BF16 image-text regression tests verify export rejection rather than running
the unqualified pipeline. Shapes and dtypes are checked alongside numerical
values; tolerances are `rtol=atol=1e-3` for FP32 and `1e-2` for FP16/BF16.
Recurrent state remains FP32
even when weights, embeddings, and convolution/KV caches are FP16/BF16. The expert
fixture separately checks FP16/BF16 CUDA dispatch with an empty expert branch
and near-tie float32 routing probabilities.

The hyper-connection mean retains the rounded model-dtype products, reduces
them with an FP32 accumulator, then casts the result back to model dtype,
matching the HF mean without requiring an ORT BF16 ReduceMean kernel. QSA
compares the mask bias to zero in FP32, avoiding a BF16 Equal kernel without
changing visibility.

This is independent reduced-model evidence, not official checkpoint parity,
semantic generation quality, full-size-model reduced-precision parity, or GenAI
driver compatibility. Tests skip when the experimental HF classes are absent;
CUDA cases additionally require an available CUDA device and ORT CUDA provider.
Regression tests simulate either backend being unavailable, including CUDA
PyTorch paired with CPU-only ORT, and verify that all three affected test
entry points skip before constructing a model or session.

On October 16, 2026, all seven standard reduced-model HF cases passed on ORT
`1.30.0` (`f1d07f0d1`) and `1.32.0.dev20261016001` (`f33c1744e`).
CUDA FP16 profiles recorded 370 decoder matrix executions for text, and 6
vision plus 226 decoder matrix executions for image-text, on each runtime.
These are actual CUDA execution events; ordinary CPU metadata/control-flow
handling is allowed. This is not full-checkpoint or performance qualification.

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

## Proposed opt-in ngram/main contract (N0; approval required)

This section proposes a new interface; it is not an implemented or approved
ABI, and it does not qualify split graphs. The source baseline is Mobius at
[#755 head](https://github.com/onnxruntime/mobius/tree/954e8f1053d64d377d8893e5c700d1b8380d584a).
That PR was unmerged when this proposal was drafted. The bounded official
evidence above—selected components, four-layer teacher-forced composition, and
selected PLE rows—does not establish all 48 main layers or all 128 full ngram
shards. A maintainer must approve the contract before N1 (#770); agent
completion or PR merge is not semantic approval.

### Opt-in and package keys

Keep the existing default task, graphs, package keys, parameters, and cache ABI
unchanged. Add a task-instance opt-in, using the existing `build(..., task=...)`
acceptance of a `ModelTask`, rather than changing the defaults:

```python
package = mobius.build(
    model_id,
    execution_provider="onnx-standard",
    task=Qwen4ExpCausalLMTask(
        split_ngram=True,
        requested_components={"ngram", "main"},
    ),
)
```

Proposed task options:

```python
Qwen4ExpCausalLMTask(
    *,
    split_ngram: bool = False,
    requested_components: frozenset[str] | set[str] | None = None,
)
Qwen4ExpVisionLanguageTask(
    *,
    split_ngram: bool = False,
    requested_components: frozenset[str] | set[str] | None = None,
)
```

With `split_ngram=False` (the default), `requested_components` must be `None`;
any non-`None` selection is rejected before graph construction. The current
text key is `model`, and the current VLM keys remain
`decoder`, `vision_encoder`, and `embedding`. With `split_ngram=True`, the new
proposed main key is `main` (role `decoder`) and the lookup key is `ngram`
(role `embedding`). Their proposed module attributes are `main` and `ngram`
for text, and `decoder` and `ngram` for VLM; existing VLM attributes
`vision_encoder` and `embedding` remain unchanged. The default selected set is
the complete set for the task: `{ngram, main}` for text, and
`{ngram, main, vision_encoder, embedding}` for VLM. Unknown keys and empty
selections fail before graph construction.
The opt-in is accepted only with `execution_provider="onnx-standard"`;
`split_ngram=True` on any other profile is rejected before graph construction,
with the build entry point validating the combination before `task.build()`.
The task instance does not override the build entry point's EP default.
The option is not ignored or silently replaced by an unsplit build. With the opt-in disabled,
other models and EP profiles retain their existing behavior. This proposal
adds a Python task-instance interface, not a CLI flag or new task-string alias.
`requested_components={"main"}` and `{"ngram"}` are valid partial builds;
neither silently rebuilds the legacy unsplit graph.

| Topology | Package / selector key | Module attribute | Role |
|---|---|---|---|
| text | `main` | `main` | `decoder` |
| VLM | `main` | `decoder` | `decoder` |
| either | `ngram` | `ngram` | `embedding` |
| VLM | `vision_encoder` | `vision_encoder` | `encoder` |
| VLM | `embedding` | `embedding` | `embedding` |

The VLM example opts into the same interface:

```python
package = mobius.build(
    model_id,
    execution_provider="onnx-standard",
    task=Qwen4ExpVisionLanguageTask(
        split_ngram=True,
        requested_components={"ngram", "main", "vision_encoder", "embedding"},
    ),
)
```

`main` can execute independently when the caller supplies its declared inputs,
including `ngram_embeddings`; it does not imply that the ngram graph or VLM
preprocessing graphs are loaded. For end-to-end image input, request all four
VLM components. Callers selecting fewer components own supplying the omitted
runtime inputs. The old `text_only=True` route continues to choose its existing
unsplit model unless the caller explicitly supplies the split task.

### Minimal seam and exact proposed tensor ABI

Only hashed lookup and token history cross the seam. The ngram component owns
IDs-to-raw-embedding lookup and its rolling token history. The main component
owns PLE key/value projections, normalization, hidden-dependent gates,
convolution, and every other decoder operation. In particular, do not move the
PLE layer's hidden-dependent work into `ngram`.

Let `B` be batch, `S` current chunk length, `P` cached token length, `L` the
number of configured PLE layers, `N` `ngram_size`, `D` `ple_embed_dim`, `H`
`hidden_size`, `C` `hc_count`, `K` =
`linear_num_key_heads * linear_key_head_dim`, and `U` =
`linear_num_value_heads * linear_value_head_dim`. `L` indexes the configured
`ple_layer_ids` in their existing order (the values are 1-based decoder layer
IDs); it is not sorted or inferred from a different layer schedule. `D` is the
raw lookup width for each PLE layer.

| Graph | Input | Output | Dtype and shape |
|---|---|---|---|
| `ngram` | `input_ids` (host substitutes EOS for masked IDs) | — | INT64 `[B,S]` |
| `ngram` | `past_ngram_context` | — | INT64 `[L,B,N-1]` |
| `ngram` | — | `ngram_embeddings` | model dtype `[L,B,S,D]` |
| `ngram` | — | `present_ngram_context` | INT64 `[L,B,N-1]` |
| text `main` | `input_ids` | — | INT64 `[B,S]` |
| VLM `main` | `inputs_embeds` | — | model dtype `[B,S,H]` |
| VLM `main` | `ple_input_ids` (original lexical IDs) | — | INT64 `[B,S]` |
| either `main` | `ngram_embeddings` | — | model dtype `[L,B,S,D]` |
| either `main` | `attention_mask` | — | INT64 `[B,P+S]` |
| text `main` | `position_ids`, `past_position_ids` | — | INT64 `[B,S]`, `[B,P]` |
| VLM `main` | `position_ids`, `past_position_ids` | — | INT64 `[4,B,S]`, `[4,B,P]` |
| either `main` | — | `logits` | model dtype `[B,S,vocab_size]` |
| either `main` | — | `present_position_ids` | INT64 `[B,P+S]` (text); `[4,B,P+S]` (VLM) |

For both `main` forms, retain the existing heterogeneous state input/output
names and types for all state other than `ple_context`:

| State owner / role | Input → output | Dtype and shape |
|---|---|---|
| `main`: linear-attention `conv_state` | `past_key_values.i.conv_state` → `present.i.conv_state` | model dtype `[B,2K+U,linear_conv_kernel_dim]` |
| `main`: linear-attention `recurrent_state` | `past_key_values.i.recurrent_state` → `present.i.recurrent_state` | `mamba_ssm_dtype` `[B,linear_num_value_heads,linear_key_head_dim,linear_value_head_dim]` |
| `main`: PLE `ple_conv_state` | `past_key_values.i.ple_conv_state` → `present.i.ple_conv_state` | model dtype `[B,C*H,(ple_conv_kernel_size-1)*N]` |
| `main`: QSA `key`, `value` | `past_key_values.i.key/value` `[B,num_key_value_heads,P,head_dim]` → `present.i.key/value` `[B,num_key_value_heads,P+S,head_dim]` | model dtype; append current tokens |
| `main`: QSA `index_key` | `past_key_values.i.index_key` `[B,P,indexer_head_dim]` → `present.i.index_key` `[B,P+S,indexer_head_dim]` | model dtype |
| `ngram`: lexical context | `past_ngram_context` → `present_ngram_context` | INT64 `[L,B,N-1]`; replace with last `N-1` history IDs |

Layer `i` retains the existing configured layer index and role membership:
linear attention has convolution and recurrent state; PLE-enabled linear layers
also have PLE convolution state; QSA layers have key, value, and index key.
`ple_context` is removed from the split `main` state and represented only by
the ngram component's `[L,B,N-1]` context. The unsplit graph retains its current
`past_key_values.i.ple_context` ABI exactly. All output dimensions above are
symbolic; concrete configured widths and layer membership come from the model
config. ONNX tensors use the configured model dtype except where explicitly
INT64 or `mamba_ssm_dtype`.

Flatten the final embedding dimension in the existing order: increasing ngram
order (2 through `N`), then head order within each ngram order, then each
head's embedding channels. Preserve the PLE-layer axis rather than concatenating
it into `D`. This `[L,B,S,D]` layout is a proposal, not an approved ABI; a list
of `[B,S,D]` outputs or a flattened `[B,S,L*D]` tensor is an alternative, but
is less explicit about layer-to-table correspondence and must not be mixed
with this proposal.

### Padding, EOS, chunking, and positions

On a cold start, initialize every configured layer's history to the first
configured EOS ID: `past_ngram_context[l,b,:] = eos_token_id`. The current
unsplit graph performs this initialization when `past_position_ids` has zero
length. Since the proposed ngram graph has no position input, the host supplies
the equivalent initialized context; do not add an implicit sentinel or infer
cold start from token values.

Preserve #767's mask-free ngram seam: the host derives current-token validity
from the final `S` entries of the full `[B,P+S]` mask and substitutes EOS for
masked current IDs before invoking ngram. The ngram graph receives only those
prepared IDs and its history, with no mask input or dependency on `P`; do not
zero the raw lookup result. Keep a separate copy of the original lexical IDs:
text main receives them for token embedding, and VLM main receives them as
`ple_input_ids` for query-length/mask bookkeeping alongside image-fused
`inputs_embeds`. Main retains the unchanged full attention mask. Prior ngram
history already reflects each chunk's host substitution; do not reapply the
past mask to it.

For ngram hashing, an EOS in a nearer history position causes farther shifted
tokens to be treated as EOS. This lookup-only EOS-crossing rule does not rewrite
the stored context: next context is the final `N-1` entries of the prior
context concatenated with current IDs after masked-current-ID substitution.
Distinguish this rule from padding substitution, which does affect stored
history. The ngram output represents each current token, including masked
positions using EOS lookup. In main, the existing mask zeros the current
gated value and current convolution input before the PLE convolution.
Previously stored convolution state can still contribute at a masked
position; preserve that behavior rather than adding a post-convolution mask.

Every chunk call consumes the last `N-1` context from the preceding chunk and
returns the next context; it must not restart EOS initialization between chunks.
The ngram graph has no position input and is independent of text/image
positions. Text `position_ids` remain `[B,S]`. VLM positions remain
`[4,B,S]`, ordered text/causal, temporal, height, width; the ngram component
uses the original lexical token IDs, not image-fused embeddings or positions.
The VLM image `embedding` component still replaces image placeholders only in
`inputs_embeds`. Zero-length chunks, EOS-list handling beyond the configured
first EOS, and non-binary masks need explicit maintainer decisions before
implementation.

### Component ownership, weights, and selected requests

Use `ComponentSpec` for component keys/module attributes, task roles for
optimization, and `ComponentManifest` plus `SharedWeightInfo` for source
ownership and declared aliases. The current VLM source roots overlap at
`model.language_model.layers`; a split manifest must assign each PLE lookup
table and its deterministic hash buffers to `ngram`, while PLE projections,
normalization and convolution stay with `main`. If existing prefix ownership
cannot express that disjoint subtree, add only a narrow source-prefix exclusion
or explicit-owner mechanism; do not route by ambiguous overlapping prefixes.
Continue using the Qwen4-Exp adapter for source renames, PLE shards, and
deterministic-buffer validation, and use the existing lazy/transactional
package binding rather than retaining a second checkpoint state dict.

| Request / package | Selected graph weights | Runtime dependencies not implicitly built |
|---|---|---|
| text `{ngram}` | All configured PLE lookup shards and deterministic hash buffers | Caller supplies host-prepared IDs and context; no main graph |
| text `{main}` | Text token table, LM head, decoder and PLE projection/norm/convolution weights; no PLE lookup table | Caller supplies raw ngram embeddings and all main inputs/states |
| text `{ngram, main}` | Union of the preceding disjoint owners | Complete text split pair |
| VLM `{ngram}` | PLE lookup shards and deterministic buffers only | Caller supplies host-prepared IDs and context |
| VLM `{main}` | Decoder, LM head, PLE projection/norm/convolution weights; no visual tower, embedding graph, or ngram table | Caller supplies fused embeddings, ngram embeddings, and main states |
| VLM `{embedding}` | Token-embedding table and image-placeholder mixer/guard | Caller supplies IDs and image features |
| VLM `{vision_encoder}` | Visual tower | Caller supplies pixels/grid and consumes image features |
| VLM all four keys | Union of the four owners | Complete image-text split pipeline |

The text `main` starts from `input_ids`, so it owns the ordinary token embedding
table as well as `lm_head`. In VLM, the existing `embedding` graph owns the
token table and image scatter; `main` consumes fused embeddings, the original
lexical `ple_input_ids` for query-length/mask bookkeeping, and owns the output
head. Ngram owns neither. Treat the source token-table parameter and
`lm_head.weight` as distinct identities unless config/source aliases explicitly
prove they are tied. When that proof exists, declare it through
`SharedWeightInfo`; do not infer tying from common dimensions, a common external
data file, or observed equality of values. Ordinary token-table/head ownership
is specified here in N0 (#768); MTP dependencies and identity declarations
require joint coordination with M0 (#769). A logical shared-weight declaration
does not guarantee a shared runtime allocation across independently loaded
ONNX sessions.

The existing manifest and adapter are building blocks, not a claim that they
already support this split. N1 must define instance-specific descriptors and
source-name mappings for the keys above, with deterministic-buffer validation
limited to the selected dependency closure. Supported checkpoint formats must
be explicit; the proposal does not qualify split FP8 binding.
The instance-specific manifest must expose those descriptors and roles before
`task.build()`, not infer them later from unknown package keys. Ngram lookup
tables must not claim the ordinary token-table alias used for tied-weight
inference; that alias belongs only to the actual token-table owner above.
Cross-package shared endpoints between the root and nested MTP sidecar remain
a joint N0/M0 approval item. Ordinary text/VLM ownership is specified above;
the endpoint representation and cross-package binding mechanism are not yet
approved and must not be inferred from same-manifest `SharedWeightInfo`.

### Build, load, save, and nested MTP behavior

`requested_components` selects graphs to construct and optimize, not merely
which graphs to save. Weight-indexing and source-to-target validation operate
on the selected components' dependency closure. A selected build must not fail
because an unselected component's weights are absent; if two selected
components share a proven logical source, bind it once and materialize only the
required component initializers. For a multi-component request, validate all
selected bindings against one source index on cloned models and publish none
until all selected bindings pass. Preserve exact rollback on a mapping failure.
This is a narrow selection extension to the existing transactional streaming
path, not a new loader or a second package format.

Propose these relevant API additions:

```python
ModelPackage.save(
    directory: str,
    *,
    components: Callable[[str], bool] | None = None,
    include_mtp_head: bool = True,
)
ModelPackage.load(
    directory: str,
    *,
    components: Collection[str] | None = None,
    include_mtp_head: bool = True,
)
```

`components=None` on load retains the existing
load-everything behavior; a set selects only those root package keys, and an
unknown requested name is an error. A selected ONNX graph remains independently
loadable and carries its ABI version and PLE-layer ordering in metadata; loading
`main` does not silently load or execute `ngram`. The caller must supply a
matching ngram ABI at invocation time. Loading an individual component
subdirectory remains possible, subject to the split-only identity recovery
rule below.

**Proposed singleton identity rule:** retain the existing disk layout: one
selected component is written as root `model.onnx`; multiple components use
named subdirectories. The pinned loader currently recovers a flat component's
key from the ONNX filename, so it would return `model` for a singleton split
graph. N1 therefore needs a narrow metadata-based identity extension, not just
the existing loader.

Every graph selected by an opt-in split task, including VLM `embedding` and
`vision_encoder`, carries these proposed ONNX model metadata properties:

- `mobius.qwen4_exp.split_abi`: `qwen4-exp-ngram-main.v1`.
- `mobius.qwen4_exp.component_key`: its canonical selector key from the table
  above, not merely its optimization role.

For marked graphs, `ModelPackage.load()` resolves the canonical key from this
metadata before applying the requested root-key selection. It does not infer
`main` versus `ngram` from a flat filename. An absent member of this metadata
pair, an unsupported ABI/key, duplicate recovered keys, or disagreement with a
component subdirectory name when loading a multi-component package is an
explicit error. Unmarked legacy graphs retain the existing filename/directory
key behavior. The graph remains independently loadable by an ONNX runtime;
the metadata governs its identity in the Mobius package wrapper.

| Saved selection / load entry | Layout | Recovered root key(s) |
|---|---|---|
| Singleton `{ngram}` build or full package filtered to `{ngram}` | `model.onnx` | `ngram` |
| Singleton `{main}` build or full package filtered to `{main}` | `model.onnx` | `main` |
| Singleton VLM `{embedding}` or `{vision_encoder}` | `model.onnx` | Selected canonical key |
| Complete text split package | `ngram/model.onnx`, `main/model.onnx` | `ngram`, `main` |
| Direct load of either marked component subdirectory | Local `model.onnx` | Its metadata key |
| Unmarked legacy flat package | `model.onnx` | `model`, unchanged |

These rules are a proposal requiring N0 approval and implementation in N1.
They do not claim that the current loader already recovers split keys.

`ModelPackage.save(components=...)` already filters root package components.
Retain that existing default and behavior. Make clear that its filter is a
serialization filter, not a build/load dependency selector: it does not change
which weights were bound, nor make an omitted runtime dependency unnecessary.
Today an attached `mtp_head` package is saved in full and restored by
`ModelPackage.load()` regardless of the root component filter. Preserve this
backward-compatible default and propose the same explicit
`include_mtp_head: bool = True` option shown above for `ModelPackage.save()`
and `load()`.
When true, persist/restore the nested sidecar in full; when false, omit it.
Do not treat it as a normal split component or apply the root predicate
recursively. MTP remains the existing nested sidecar and is not selected by
`split_ngram`; ordinary generation does not gain an MTP graph.
Do not introduce a new persistence scheme or claim that sharing ONNX external
data shares runtime memory.

In addition to the package identity metadata above, ngram/main metadata records
component role, configured `ple_layer_ids`, `N`, `D`, and model dtype.
Store the main graph's existing state manifest with
`ple_context` removed only for split mode; ngram metadata documents context
initialization, order, and state names. Preserve current keys and metadata for
unsplit graphs. Use the existing save layout and ONNX external data, then the
proposed split-aware load rule. Reject a pair with mismatched split ABI or
layer order at orchestration time. Shape/ABI agreement alone does not prove
that hash configuration and trained weights belong together; a semantic
pairing identity/check remains an approval item before qualification.

### Complete state ownership and restore/replay proposal for S1 (#773)

State for a split text or VLM decode is the tuple of ngram context, every main
layer state in the table above, `past_position_ids`, and the host's logical
past length and attention-mask history. For VLM, also preserve the active
position axes and the exact token/embedding inputs needed to replay the accepted
suffix; image features already incorporated into cached main state must not be
replaced with text-only inputs during replay. The ngram context is not derivable
from QSA KV length alone.

Propose a host transaction with `snapshot(state)`, `restore(snapshot)`, and
`commit_prefix(snapshot, accepted_ids, positions, masks)`. Snapshot must own
copies (no mutable aliasing) of ngram context, linear convolution and recurrent
state, PLE convolution state, QSA key/value/index-key state, position history,
logical length/mask state, and any later MTP states. Initially restore the
complete snapshot and replay accepted IDs through ngram and main; KV cropping
alone cannot roll back recurrent, convolution, PLE-history, or position state.
Specify atomic commit/failure behavior before S1 implementation. Zero proposal
tokens bypass speculative execution. This is a proposed state contract for
review, not an approved transaction API or a claim of speculative correctness.

M0 may later add an explicitly requested main pre-final tap; `split_ngram`
does not request it or an MTP graph. The tap selector, output metadata, and
text/VLM position mapping remain joint N0/M0 decisions before M1, not implicit
extensions approved by this document.

### Alternatives, test targets, and approval gates

Alternatives requiring explicit review include returning per-layer embedding
outputs instead of `[L,B,S,D]`, carrying only one shared `[B,N-1]` history
instead of per-`ple_layer_ids` history, and assigning the VLM split main key
`decoder` instead of `main`. Keep the proposed history axis unless tests prove
one shared history is equivalent across every configured PLE layer and the
maintainer approves the ABI change. The current `Qwen4ExpNGramEmbedding`
computes next history and each layer's raw embedding separately; the proposed
batched leading `L` axis is a new graph boundary and must be proven equivalent.

Proposed tiny local CPU targets (not run for this documentation-only task):

| Target | Required assertion |
|---|---|
| NEW `src/mobius/models/qwen4_exp_split_test.py` (#770) | Tiny random config; compare text/VLM logits and every state across prefill, chunked prefill, and multi-token cached decode; exact INT64 histories/positions and dtype-appropriate floats |
| Extend `src/mobius/integrations/transformers/_qwen4_exp_weights_test.py` (#770) | Missing/malformed selected sources fail transactionally; same-shape/different-value sentinels prove bound values and unchanged sources; instrument unrelated payloads to prove they are not accessed |
| `src/mobius/_model_package_test.py::test_split_components_load_independently` | Singleton builds, filtered singleton saves, direct component-directory loads and multi-component packages recover canonical keys; unmarked legacy flat loads remain `model`; missing/invalid/conflicting identity metadata fails explicitly |
| `src/mobius/_model_package_test.py::test_split_package_roundtrip_with_mtp_sidecar` | Full and filtered root save/load; all selected canonical keys and ABI metadata survive; nested MTP persists/restores in full by default and explicit omission is tested if the proposed option is accepted |
| NEW `src/mobius/models/qwen4_exp_split_generation_test.py` (#771) | Later, after graph parity: newly generated IDs feed back through host EOS preparation, ngram and main over multiple steps; compare independently generated reference tokens and every state, including image/no-new-image decode |

The NEW files above are required future entry points, not existing or executed
tests. Their CPU fixtures use local tiny/random tensors with
`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`; they do not require official model
downloads. Missing required HF reference capability or skipped mandatory cases
are blockers, not passes.

Add adversarial padding (left and chunk-boundary), EOS crossing, repeated IDs,
batching, and multimodal text/image-position cases. Teacher-forced parity alone
does not qualify closed-loop generation. Require `onnx_ir`/`onnxscript.nn`,
no explicit protobuf operations, default opset 24, and strict standard ONNX
validation after binding, optimization, save/reload. Retain the deferred
QSA exact-cutoff tie discrepancies and the standard BF16 vision rejection;
neither this proposal nor tiny CPU tests resolve them. No test, build,
installation, model acquisition, numerical claim, or allocation is part of
this documentation-only design assignment.

Pinned source cross-checks:

- [Text and VLM task signatures, state construction, and existing package keys](https://github.com/onnxruntime/mobius/blob/954e8f1053d64d377d8893e5c700d1b8380d584a/src/mobius/tasks/_qwen4_exp.py#L31-L143).
- [VLM fused embeddings, lexical IDs, and four-axis positions](https://github.com/onnxruntime/mobius/blob/954e8f1053d64d377d8893e5c700d1b8380d584a/src/mobius/tasks/_qwen4_exp.py#L263-L342).
- [Ngram hash, flattening, history, lookup, and PLE hidden-dependent operations](https://github.com/onnxruntime/mobius/blob/954e8f1053d64d377d8893e5c700d1b8380d584a/src/mobius/models/qwen4_exp.py#L410-L511) and [PLE projection/gate/convolution](https://github.com/onnxruntime/mobius/blob/954e8f1053d64d377d8893e5c700d1b8380d584a/src/mobius/models/qwen4_exp.py#L578-L641).
- [Main attention masking, current-token EOS substitution, and state flow](https://github.com/onnxruntime/mobius/blob/954e8f1053d64d377d8893e5c700d1b8380d584a/src/mobius/models/qwen4_exp.py#L1333-L1379).
- [VLM source component roots and embedding/head routing](https://github.com/onnxruntime/mobius/blob/954e8f1053d64d377d8893e5c700d1b8380d584a/src/mobius/models/qwen4_exp.py#L1930-L1939).
- [Component declarations and manifests](https://github.com/onnxruntime/mobius/blob/954e8f1053d64d377d8893e5c700d1b8380d584a/src/mobius/tasks/_base.py#L23-L103), [shared-weight metadata and endpoint validation](https://github.com/onnxruntime/mobius/blob/954e8f1053d64d377d8893e5c700d1b8380d584a/src/mobius/_component_manifest.py#L41-L103) and [manifest ownership validation](https://github.com/onnxruntime/mobius/blob/954e8f1053d64d377d8893e5c700d1b8380d584a/src/mobius/_component_manifest.py#L161-L205), [tied-weight inference](https://github.com/onnxruntime/mobius/blob/954e8f1053d64d377d8893e5c700d1b8380d584a/src/mobius/_component_manifest.py#L297-L323), and [weight-adapter boundary](https://github.com/onnxruntime/mobius/blob/954e8f1053d64d377d8893e5c700d1b8380d584a/src/mobius/weights/_adapters.py#L26-L102).
- [Transactional Qwen4-Exp package streaming](https://github.com/onnxruntime/mobius/blob/954e8f1053d64d377d8893e5c700d1b8380d584a/src/mobius/integrations/transformers/_qwen4_exp_weights.py#L272-L306), [MTP sidecar save recursion](https://github.com/onnxruntime/mobius/blob/954e8f1053d64d377d8893e5c700d1b8380d584a/src/mobius/_model_package.py#L963-L970) and [load/restore](https://github.com/onnxruntime/mobius/blob/954e8f1053d64d377d8893e5c700d1b8380d584a/src/mobius/_model_package.py#L1304-L1396).
- [Current selected-component layout](https://github.com/onnxruntime/mobius/blob/954e8f1053d64d377d8893e5c700d1b8380d584a/src/mobius/_model_package.py#L70-L102), [singleton save path](https://github.com/onnxruntime/mobius/blob/954e8f1053d64d377d8893e5c700d1b8380d584a/src/mobius/_model_package.py#L876-L893), and [directory/filename key recovery](https://github.com/onnxruntime/mobius/blob/954e8f1053d64d377d8893e5c700d1b8380d584a/src/mobius/_model_package.py#L1347-L1364).
- [Parent evidence bounds and deferred issues](https://github.com/onnxruntime/mobius/issues/767), [N0 ordinary ownership contract](https://github.com/onnxruntime/mobius/issues/768), [joint MTP identity coordination with M0](https://github.com/onnxruntime/mobius/issues/769), and [S1 complete-state acceptance](https://github.com/onnxruntime/mobius/issues/773).
