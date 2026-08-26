# `build_from_gguf()`

Build an ONNX `ModelPackage` from a GGUF model file.

```python
from mobius import build_from_gguf
```

> **Note**: Requires the optional `gguf` package:
> `pip install mobius-onnx[gguf]`

<!-- BEGIN GGUF CLOSURE SUMMARY -->

**Pinned source:** `ggml-org/llama.cpp@8d9af256337d1a501250f9bbf4c0859a654bddd6` (2026-08-23T16:59:42Z).

| Census | Total | Closure |
|---|---:|---|
| Architectures | 147 | graph verdicts: {'deferred': 62, 'rejected': 2, 'supported': 83}; importable: 82; quantized import: {'rejected': 27, 'supported': 120}; runtime: {'deferred': 144, 'rejected': 2, 'supported': 1} |
| Active stored qtypes | 25 | 24 have an import route; 1 are explicitly deferred with no route |
| Serialized projector strings | 60 | {'graph-importable': 5, 'runtime-supported': 0} |
| Tokenizer pre identifiers | 87 | 56 semantic groups; all default to deferred and become materializable only from a validated embedded `tokenizer.huggingface.json` or an exact pinned source in runtime evidence |

`SUPPORTED` means the named capability is implemented and mechanically tested. `DEFERRED` means it is intentionally unavailable pending the stated work. `REJECTED` means the input or route is invalid by policy. Graph support proves construction/execution only; runtime support additionally requires a pinned real artifact, independent parity, and deterministic generation or stateful semantics. Tokenizer `copy` delegates algorithm semantics to an embedded, vocabulary-identical tokenizer JSON. A `pinned-source` route additionally binds an immutable Hub revision, exact asset hashes, and all reconstructible GGUF tokenizer semantics.

<!-- END GGUF CLOSURE SUMMARY -->

## Tokenizer truthfulness

Graph import and runtime-package emission are separate capabilities. GGUF
normally stores only an opaque `tokenizer.ggml.pre` identifier, not the full
normalizer, pre-tokenizer, added-token, decoder, and post-processor pipeline.
Mobius never substitutes a generic BPE/SentencePiece tokenizer.

The `copy` route requires `tokenizer.huggingface.json` in the GGUF and an exact
ordered-vocabulary match. The `pinned-source` route is narrower: a runtime evidence
record names one immutable Hub revision and every copied asset's size and SHA-256.
Mobius additionally compares ordered token IDs, merge order, special-token IDs,
flags, and chat templates against all corresponding GGUF metadata. The remaining
normalizer, pre-tokenizer, decoder, and post-processor semantics are accepted only
through those exact asset hashes, never reconstructed from `tokenizer.ggml.pre`.
Any missing or contradictory field rejects before durable output.

The graph package records a canonical digest of every tokenizer metadata field.
Runtime packaging rechecks that digest before writing, so replacing a local
GGUF between graph construction and package emission cannot mix tokenizer
identity. The manifest records the selected route, immutable source revision,
asset hashes, and GGUF tokenizer metadata hash.

This generated policy table is pinned to llama.cpp commit
`8d9af256337d1a501250f9bbf4c0859a654bddd6`. It enumerates all 87 accepted
identifiers. Aliases share a row only where the pinned pre-type and hard-coded
overrides are identical; every policy defaults to `deferred`.

<!-- BEGIN GGUF TOKENIZER PRE SUPPORT MATRIX -->

| Exact identifier | Canonical semantic group | Pinned pre-type | Default route | Exactness/restriction |
|---|---|---|---|---|
| `a.x-4.0` | `gpt-2` | `GPT2` | `deferred` | Exact-copy only after embedded tokenizer JSON and ordered vocabulary validation; otherwise runtime packaging is deferred. |
| `afmoe` | `afmoe` | `AFMOE` | `deferred` | Exact-copy only after embedded tokenizer JSON and ordered vocabulary validation; otherwise runtime packaging is deferred. |
| `bailingmoe` | `bailingmoe` | `BAILINGMOE` | `deferred` | Exact-copy only after embedded tokenizer JSON and ordered vocabulary validation; otherwise runtime packaging is deferred. |
| `bailingmoe2` | `bailingmoe` | `BAILINGMOE` | `deferred` | Exact-copy only after embedded tokenizer JSON and ordered vocabulary validation; otherwise runtime packaging is deferred. |
| `bloom` | `bloom` | `BLOOM` | `deferred` | Exact-copy only after embedded tokenizer JSON and ordered vocabulary validation; otherwise runtime packaging is deferred. |
| `chameleon` | `chameleon` | `CHAMELEON` | `deferred` | Exact-copy only after embedded tokenizer JSON and ordered vocabulary validation; otherwise runtime packaging is deferred. |
| `chatglm-bpe` | `glm4` | `CHATGLM4` | `deferred` | Exact-copy only after embedded tokenizer JSON and ordered vocabulary validation; otherwise runtime packaging is deferred. |
| `codeshell` | `codeshell` | `CODESHELL` | `deferred` | Exact-copy only after embedded tokenizer JSON and ordered vocabulary validation; otherwise runtime packaging is deferred. |
| `cohere2moe` | `tiny_aya` | `TINY_AYA` | `deferred` | Exact-copy only after embedded tokenizer JSON and ordered vocabulary validation; otherwise runtime packaging is deferred. |
| `command-r` | `command-r` | `COMMAND_R` | `deferred` | Exact-copy only after embedded tokenizer JSON and ordered vocabulary validation; otherwise runtime packaging is deferred. |
| `dbrx` | `dbrx` | `DBRX` | `deferred` | Exact-copy only after embedded tokenizer JSON and ordered vocabulary validation; otherwise runtime packaging is deferred. |
| `deepseek-coder` | `deepseek-coder` | `DEEPSEEK_CODER` | `deferred` | Exact-copy only after embedded tokenizer JSON and ordered vocabulary validation; otherwise runtime packaging is deferred. |
| `deepseek-llm` | `deepseek-llm` | `DEEPSEEK_LLM` | `deferred` | Exact-copy only after embedded tokenizer JSON and ordered vocabulary validation; otherwise runtime packaging is deferred. |
| `deepseek-r1-qwen` | `qwen2` | `QWEN2` | `deferred` | Exact-copy only after embedded tokenizer JSON and ordered vocabulary validation; otherwise runtime packaging is deferred. |
| `deepseek-v3` | `deepseek-v3` | `DEEPSEEK3_LLM` | `deferred` | Exact-copy only after embedded tokenizer JSON and ordered vocabulary validation; otherwise runtime packaging is deferred. |
| `default` | `default` | `DEFAULT` | `deferred` | Exact-copy only after embedded tokenizer JSON and ordered vocabulary validation; otherwise runtime packaging is deferred. |
| `exaone` | `exaone` | `EXAONE` | `deferred` | Exact-copy only after embedded tokenizer JSON and ordered vocabulary validation; otherwise runtime packaging is deferred. |
| `exaone-moe` | `exaone-moe` | `EXAONE_MOE` | `deferred` | Exact-copy only after embedded tokenizer JSON and ordered vocabulary validation; otherwise runtime packaging is deferred. |
| `exaone4` | `gpt-2` | `GPT2` | `deferred` | Exact-copy only after embedded tokenizer JSON and ordered vocabulary validation; otherwise runtime packaging is deferred. |
| `f2llmv2` | `qwen2` | `QWEN2` | `deferred` | Exact-copy only after embedded tokenizer JSON and ordered vocabulary validation; otherwise runtime packaging is deferred. |
| `falcon` | `falcon` | `FALCON` | `deferred` | Exact-copy only after embedded tokenizer JSON and ordered vocabulary validation; otherwise runtime packaging is deferred. |
| `falcon-h1` | `llama3` | `LLAMA3` | `deferred` | Exact-copy only after embedded tokenizer JSON and ordered vocabulary validation; otherwise runtime packaging is deferred. |
| `falcon3` | `llama3` | `LLAMA3` | `deferred` | Exact-copy only after embedded tokenizer JSON and ordered vocabulary validation; otherwise runtime packaging is deferred. |
| `gemma4` | `gemma4` | `GEMMA4` | `deferred` | Exact-copy only after embedded tokenizer JSON and ordered vocabulary validation; otherwise runtime packaging is deferred. |
| `gigachat` | `gpt-2` | `GPT2` | `deferred` | Exact-copy only after embedded tokenizer JSON and ordered vocabulary validation; otherwise runtime packaging is deferred. |
| `glm4` | `glm4` | `CHATGLM4` | `deferred` | Exact-copy only after embedded tokenizer JSON and ordered vocabulary validation; otherwise runtime packaging is deferred. |
| `gpt-2` | `gpt-2` | `GPT2` | `deferred` | Exact-copy only after embedded tokenizer JSON and ordered vocabulary validation; otherwise runtime packaging is deferred. |
| `gpt-4o` | `gpt-4o` | `GPT4O` | `deferred` | Exact-copy only after embedded tokenizer JSON and ordered vocabulary validation; otherwise runtime packaging is deferred. |
| `gpt3-finnish` | `gpt3-finnish` | `GPT3_FINNISH` | `deferred` | Exact-copy only after embedded tokenizer JSON and ordered vocabulary validation; otherwise runtime packaging is deferred. |
| `granite-docling` | `granite-docling` | `GRANITE_DOCLING` | `deferred` | Exact-copy only after embedded tokenizer JSON and ordered vocabulary validation; otherwise runtime packaging is deferred. |
| `granite-embed-multi-311m` | `gemma4` | `GEMMA4` | `deferred` | Exact-copy only after embedded tokenizer JSON and ordered vocabulary validation; otherwise runtime packaging is deferred. |
| `granite-embed-multi-97m` | `granite-embed-multi-97m` | `GRANITE_EMB_MULTI` | `deferred` | Exact-copy only after embedded tokenizer JSON and ordered vocabulary validation; otherwise runtime packaging is deferred. |
| `grok-2` | `grok-2` | `GROK_2` | `deferred` | Exact-copy only after embedded tokenizer JSON and ordered vocabulary validation; otherwise runtime packaging is deferred. |
| `hunyuan` | `hunyuan` | `HUNYUAN` | `deferred` | Exact-copy only after embedded tokenizer JSON and ordered vocabulary validation; otherwise runtime packaging is deferred. |
| `hunyuan-dense` | `hunyuan-dense` | `HUNYUAN_DENSE` | `deferred` | Exact-copy only after embedded tokenizer JSON and ordered vocabulary validation; otherwise runtime packaging is deferred. |
| `jais` | `jais` | `JAIS` | `deferred` | Exact-copy only after embedded tokenizer JSON and ordered vocabulary validation; otherwise runtime packaging is deferred. |
| `jais-2` | `jais-2` | `JAIS2` | `deferred` | Exact-copy only after embedded tokenizer JSON and ordered vocabulary validation; otherwise runtime packaging is deferred. |
| `jina-de` | `gpt-2` | `GPT2` | `deferred` | Exact-copy only after embedded tokenizer JSON and ordered vocabulary validation; otherwise runtime packaging is deferred. |
| `jina-es` | `gpt-2` | `GPT2` | `deferred` | Exact-copy only after embedded tokenizer JSON and ordered vocabulary validation; otherwise runtime packaging is deferred. |
| `jina-v1-en` | `jina-v1-en` | `GPT2_ADD_SEP` | `deferred` | Exact-copy only after embedded tokenizer JSON and ordered vocabulary validation; otherwise runtime packaging is deferred. |
| `jina-v2-code` | `jina-v1-en` | `GPT2_ADD_SEP` | `deferred` | Exact-copy only after embedded tokenizer JSON and ordered vocabulary validation; otherwise runtime packaging is deferred. |
| `jina-v2-de` | `gpt-2` | `GPT2` | `deferred` | Exact-copy only after embedded tokenizer JSON and ordered vocabulary validation; otherwise runtime packaging is deferred. |
| `jina-v2-es` | `gpt-2` | `GPT2` | `deferred` | Exact-copy only after embedded tokenizer JSON and ordered vocabulary validation; otherwise runtime packaging is deferred. |
| `jina-v5-nano` | `llama3` | `LLAMA3` | `deferred` | Exact-copy only after embedded tokenizer JSON and ordered vocabulary validation; otherwise runtime packaging is deferred. |
| `joyai-llm` | `joyai-llm` | `JOYAI_LLM` | `deferred` | Exact-copy only after embedded tokenizer JSON and ordered vocabulary validation; otherwise runtime packaging is deferred. |
| `kanana2` | `gpt-4o` | `GPT4O` | `deferred` | Exact-copy only after embedded tokenizer JSON and ordered vocabulary validation; otherwise runtime packaging is deferred. |
| `kimi-k2` | `kimi-k2` | `KIMI_K2` | `deferred` | Exact-copy only after embedded tokenizer JSON and ordered vocabulary validation; otherwise runtime packaging is deferred. |
| `kormo` | `qwen2` | `QWEN2` | `deferred` | Exact-copy only after embedded tokenizer JSON and ordered vocabulary validation; otherwise runtime packaging is deferred. |
| `laguna` | `laguna` | `LAGUNA` | `deferred` | Exact-copy only after embedded tokenizer JSON and ordered vocabulary validation; otherwise runtime packaging is deferred. |
| `lfm2` | `llama3` | `LLAMA3` | `deferred` | Exact-copy only after embedded tokenizer JSON and ordered vocabulary validation; otherwise runtime packaging is deferred. |
| `llada-moe` | `bailingmoe` | `BAILINGMOE` | `deferred` | Exact-copy only after embedded tokenizer JSON and ordered vocabulary validation; otherwise runtime packaging is deferred. |
| `llama-bpe` | `llama3` | `LLAMA3` | `deferred` | Exact-copy only after embedded tokenizer JSON and ordered vocabulary validation; otherwise runtime packaging is deferred. |
| `llama-v3` | `llama3` | `LLAMA3` | `deferred` | Exact-copy only after embedded tokenizer JSON and ordered vocabulary validation; otherwise runtime packaging is deferred. |
| `llama3` | `llama3` | `LLAMA3` | `deferred` | Exact-copy only after embedded tokenizer JSON and ordered vocabulary validation; otherwise runtime packaging is deferred. |
| `llama4` | `gpt-4o` | `GPT4O` | `deferred` | Exact-copy only after embedded tokenizer JSON and ordered vocabulary validation; otherwise runtime packaging is deferred. |
| `megrez` | `megrez` | `QWEN2_CLEAN_SPACES` | `deferred` | Exact-copy only after embedded tokenizer JSON and ordered vocabulary validation; otherwise runtime packaging is deferred. |
| `mellum` | `gpt-2` | `GPT2` | `deferred` | Exact-copy only after embedded tokenizer JSON and ordered vocabulary validation; otherwise runtime packaging is deferred. |
| `mellum2` | `mellum2` | `MELLUM2` | `deferred` | Exact-copy only after embedded tokenizer JSON and ordered vocabulary validation; otherwise runtime packaging is deferred. |
| `midm-2.0` | `llama3` | `LLAMA3` | `deferred` | Exact-copy only after embedded tokenizer JSON and ordered vocabulary validation; otherwise runtime packaging is deferred. |
| `minerva-7b` | `minerva-7b` | `MINERVA` | `deferred` | Exact-copy only after embedded tokenizer JSON and ordered vocabulary validation; otherwise runtime packaging is deferred. |
| `minicpm5` | `minicpm5` | `MINICPM5` | `deferred` | Exact-copy only after embedded tokenizer JSON and ordered vocabulary validation; otherwise runtime packaging is deferred. |
| `minimax-m2` | `minimax-m2` | `MINIMAX_M2` | `deferred` | Exact-copy only after embedded tokenizer JSON and ordered vocabulary validation; otherwise runtime packaging is deferred. |
| `modern-bert` | `gpt-2` | `GPT2` | `deferred` | Exact-copy only after embedded tokenizer JSON and ordered vocabulary validation; otherwise runtime packaging is deferred. |
| `mpt` | `mpt` | `MPT` | `deferred` | Exact-copy only after embedded tokenizer JSON and ordered vocabulary validation; otherwise runtime packaging is deferred. |
| `olmo` | `olmo` | `OLMO` | `deferred` | Exact-copy only after embedded tokenizer JSON and ordered vocabulary validation; otherwise runtime packaging is deferred. |
| `phi-2` | `gpt-2` | `GPT2` | `deferred` | Exact-copy only after embedded tokenizer JSON and ordered vocabulary validation; otherwise runtime packaging is deferred. |
| `pixtral` | `llama3` | `LLAMA3` | `deferred` | Exact-copy only after embedded tokenizer JSON and ordered vocabulary validation; otherwise runtime packaging is deferred. |
| `poro-chat` | `poro-chat` | `PORO` | `deferred` | Exact-copy only after embedded tokenizer JSON and ordered vocabulary validation; otherwise runtime packaging is deferred. |
| `qwen2` | `qwen2` | `QWEN2` | `deferred` | Exact-copy only after embedded tokenizer JSON and ordered vocabulary validation; otherwise runtime packaging is deferred. |
| `qwen35` | `qwen35` | `QWEN35` | `deferred` | Exact-copy only after embedded tokenizer JSON and ordered vocabulary validation; otherwise runtime packaging is deferred. |
| `refact` | `refact` | `REFACT` | `deferred` | Exact-copy only after embedded tokenizer JSON and ordered vocabulary validation; otherwise runtime packaging is deferred. |
| `roberta-bpe` | `jina-v1-en` | `GPT2_ADD_SEP` | `deferred` | Exact-copy only after embedded tokenizer JSON and ordered vocabulary validation; otherwise runtime packaging is deferred. |
| `sarvam-moe` | `sarvam-moe` | `SARVAM_MOE` | `deferred` | Exact-copy only after embedded tokenizer JSON and ordered vocabulary validation; otherwise runtime packaging is deferred. |
| `seed-coder` | `seed-coder` | `SEED_CODER` | `deferred` | Exact-copy only after embedded tokenizer JSON and ordered vocabulary validation; otherwise runtime packaging is deferred. |
| `smaug-bpe` | `smaug-bpe` | `SMAUG` | `deferred` | Exact-copy only after embedded tokenizer JSON and ordered vocabulary validation; otherwise runtime packaging is deferred. |
| `smollm` | `smollm` | `SMOLLM` | `deferred` | Exact-copy only after embedded tokenizer JSON and ordered vocabulary validation; otherwise runtime packaging is deferred. |
| `solar-open` | `solar-open` | `SOLAR_OPEN` | `deferred` | Exact-copy only after embedded tokenizer JSON and ordered vocabulary validation; otherwise runtime packaging is deferred. |
| `stablelm2` | `stablelm2` | `STABLELM2` | `deferred` | Exact-copy only after embedded tokenizer JSON and ordered vocabulary validation; otherwise runtime packaging is deferred. |
| `starcoder` | `starcoder` | `STARCODER` | `deferred` | Exact-copy only after embedded tokenizer JSON and ordered vocabulary validation; otherwise runtime packaging is deferred. |
| `superbpe` | `superbpe` | `SUPERBPE` | `deferred` | Exact-copy only after embedded tokenizer JSON and ordered vocabulary validation; otherwise runtime packaging is deferred. |
| `talkie` | `gpt-4o` | `GPT4O` | `deferred` | Exact-copy only after embedded tokenizer JSON and ordered vocabulary validation; otherwise runtime packaging is deferred. |
| `tekken` | `tekken` | `TEKKEN` | `deferred` | Exact-copy only after embedded tokenizer JSON and ordered vocabulary validation; otherwise runtime packaging is deferred. |
| `tiny_aya` | `tiny_aya` | `TINY_AYA` | `deferred` | Exact-copy only after embedded tokenizer JSON and ordered vocabulary validation; otherwise runtime packaging is deferred. |
| `trillion` | `trillion` | `TRILLION` | `deferred` | Exact-copy only after embedded tokenizer JSON and ordered vocabulary validation; otherwise runtime packaging is deferred. |
| `viking` | `viking` | `VIKING` | `deferred` | Exact-copy only after embedded tokenizer JSON and ordered vocabulary validation; otherwise runtime packaging is deferred. |
| `whitespace` | `whitespace` | `WHITESPACE` | `deferred` | Exact-copy only after embedded tokenizer JSON and ordered vocabulary validation; otherwise runtime packaging is deferred. |
| `youtu` | `youtu` | `YOUTU` | `deferred` | Exact-copy only after embedded tokenizer JSON and ordered vocabulary validation; otherwise runtime packaging is deferred. |

<!-- END GGUF TOKENIZER PRE SUPPORT MATRIX -->

### Metadata audit

The pinned loader consumes tokenizer model/pre, token strings/types/count,
scores, merges, BOS/EOS/EOT/EOM/UNK/PAD/SEP/mask/FIM IDs, add-token and
whitespace/normalizer flags, suppression IDs, precompiled charsmap, and
default/named chat templates. `tokenizer.chat_templates` is the converter's
named-template inventory. `tokenizer.huggingface.json` and
`tokenizer.rwkv.world` are declared extension keys but are not consumed by the
pinned vocabulary loader. There is no pinned GGUF `byte_fallback` field;
byte-fallback behavior is implicit in the complete tokenizer pipeline.

## Signature

```python
def build_from_gguf(
    gguf_path: str | Path,
    *,
    task: str | None = None,
    dtype: str | None = None,
    keep_quantized: bool = True,
    execution_provider: str = "default",
    mmproj: str | Path | None = None,
    static_cache: bool = False,
    max_seq_len: int | None = None,
    reuse_gguf_weights: bool = False,
    target_config: str | Path | Mapping[str, object] | None = None,
) -> ModelPackage:
```

## Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `gguf_path` | `str \| Path` | (required) | Local `.gguf` path or `owner/repo:filename.gguf` Hub reference. |
| `task` | `str \| None` | `None` | Override the model task (e.g. `"text-generation"`). When `None`, the task is auto-detected from the model type. |
| `dtype` | `str \| None` | `None` | Override model dtype (e.g. `"f16"`). When `None`, defaults to float32. |
| `keep_quantized` | `bool` | `True` | Preserve quantization when present. Supported affine blocks are repacked as `MatMulNBits`; in text-only builds, supported native IQ/MXFP4 projection blocks retain their bytes. Projection tensors that would require lossy requantization are rejected. Set to `False` to dequantize all weights. |
| `execution_provider` | `str` | `"default"` | Target EP for EP-aware graph optimization. |
| `mmproj` | `str \| Path \| None` | `None` | Optional companion multimodal-projector GGUF. |
| `static_cache` | `bool` | `False` | Build a fixed-width KV cache when the architecture supports it. |
| `max_seq_len` | `int \| None` | `None` | Static-cache sequence limit. |
| `reuse_gguf_weights` | `bool` | `False` | Reuse byte-compatible F32/F16 and native IQ/MXFP4 tensor ranges from the original GGUF instead of copying them into ONNX external data. |
| `target_config` | `str \| Path \| Mapping[str, object] \| None` | `None` | Exact target directory/config plus `tokenizer.json`, or an explicit config mapping with the complete `tokenizer_json` object. Required only for `dflash` and `eagle3`. |

## Returns

`ModelPackage` — A dict-like collection of named `ir.Model` objects.

## Examples

```python
from mobius import build_from_gguf

# Basic conversion preserves supported quantization by default
pkg = build_from_gguf("llama-3.2-1b-q4_0.gguf")
pkg.save("output/llama/")
```

```python
# Explicitly dequantize every weight to float
pkg = build_from_gguf("llama-3.2-1b-q4_0.gguf", keep_quantized=False)
pkg.save("output/llama-float/")
```

```bash
# Via CLI
mobius build-gguf llama-3.2-1b-q4_0.gguf --output output/llama/
```

### Reuse the original GGUF as external data

Place the GGUF directly in the output directory, then opt in:

```bash
mobius build-gguf output/llama/model.gguf \
  --output output/llama/ \
  --reuse-gguf-weights
```

The package contains `model.onnx`, the original GGUF, `model.onnx.data` with
only converted/materialized tensors, and `gguf-reuse.json`. The manifest pins
the GGUF location, size, SHA-256, qtypes, and exact reused ranges. ONNX runtimes
do not enforce that digest; applications can call
`mobius.integrations.gguf.verify_gguf_reuse_manifest()` before session creation.

Initial support is deliberately limited to one flat text model. Multimodal,
MTP, nested, sharded, symlinked, absolute, and parent-relative GGUF layouts are
rejected rather than copied or linked. F32/F16 tensors and compatible
IQ4_NL/IQ4_XS/IQ3_S/IQ3_XXS/IQ2_XXS/IQ2_XS/IQ2_S/IQ1_S/IQ1_M/MXFP4
projection/output tensors can be reused when no logical transform is required.
Repacked, requantized, dequantized, or synthesized tensors are true storage
changes and therefore go to the sidecar.

Float transpose, norm-offset arithmetic, Llama Q/K row permutation, and Mamba
`A_log`/shape transforms are not inherent storage incompatibilities. Reuse mode
keeps their original F32/F16 bytes in the GGUF and inserts the corresponding
ONNX `Transpose`, `Sub`, `Neg`/`Log`, or `Reshape` operations before their graph
consumers. Opaque packed UINT8 blocks are never passed through a generic ONNX
`Transpose`; only a consumer with a proven native block layout can reuse them.

Create ORT sessions with graph optimization disabled
(`SessionOptions.graph_optimization_level = ORT_DISABLE_ALL`) so ORT does not
constant-fold them into another materialized weight copy. This first version
therefore rejects `--runtime ort-genai`: the current `genai_config.json` schema
has no supported session option that can require disabled constant folding.
Use direct ONNX Runtime for reuse packages.

Disabling folding preserves storage reuse but does not make transforms free at
runtime. A transpose/permutation or arithmetic node may allocate a transformed
weight buffer and consume startup or execution time. The package avoids a second
on-disk copy; peak runtime memory and performance remain transform- and
execution-provider-dependent.

```bash
# Draft GGUFs require the exact target config and tokenizer directory
mobius build-gguf draft.gguf --target-config target/ --output output/draft/
```

## Behavior

1. Reads GGUF metadata to detect architecture and config
2. Maps GGUF tensor names to HuggingFace weight names
3. Preserves supported quantized tensors by default using value-preserving
   repacking or text-only native-block retention; any projection requiring
   lossy requantization fails closed, while `keep_quantized=False` explicitly
   dequantizes every tensor
4. Applies architecture-specific tensor processors (e.g. Q/K permute)
5. Builds the ONNX graph using the same pipeline as `build()`
6. Runs `preprocess_weights()` (HF → ONNX name mapping)
7. Applies weights to the graph

F32-, F16-, and BF16-only GGUFs use the normal float import path even though
`keep_quantized=True` is the default: there is no quantization to preserve.
Quantized GGUFs containing only qtypes with no supported preservation target
(for example, pure Q5_K weights) fail with an actionable error rather
than silently becoming float. Pass `keep_quantized=False` to request that float
conversion explicitly.
Mixed presets such as Q4_K_M also fail when their projection inventory includes
Q5/Q6/Q8 tensors that cannot share one lossless MatMulNBits contract.

### Native blocks, conversion, and source-file reuse

“Exact native bytes” in the qtype matrix means eligible projection blocks are
copied byte-for-byte into ONNX external data for the matching runtime custom-op
format. It does **not** mean the saved package points back into the source GGUF.
Mobius currently emits a self-contained package and does not publish
source-range external-data references, a GGUF source manifest, or sidecar-only
converted-byte overlays. Therefore immutable source/range/SHA/qtype
revalidation, cross-process locking, recovery of a source-backed transaction,
and symlink containment for source references are not claimed.

Conversion is required for affine `MatMulNBits` routes and for tensor roles that
cannot consume native blocks. Native preservation is restricted to eligible
linear projections. Token embeddings need a compatible
`GatherBlockQuantized` representation; otherwise they dequantize to ordinary
`Gather`. Norms, biases, convolution/state tensors, calibration values, and
unsupported sidecars use their declared float route or reject. Tied
embedding/output ownership is resolved once; an absent output may reuse the
validated embedding owner, but Mobius never invents a second head or silently
drops an explicit conflicting output.

Complete runtime packages are staged in a sibling temporary directory and
published with one atomic rename. An existing destination is rejected rather
than moved aside, so readers never observe a replacement gap or mixed package.
This is not a multi-process lock; callers racing to the same absent destination
must handle one publication failure. The source tokenizer digest is rechecked
before publication. Publication also requires the exact runtime
version from the evidence record and rechecks the source filename, size, SHA-256,
canonical architecture, tensor/qtype census, and complete graph-shaping import
route captured during construction. The serialized ONNX file list and graph
SHA-256 must match evidence after graph serialization, and the complete staged
file list and package SHA-256 must match again after writing the exact tokenizer
and runtime configuration. ORT GenAI emission also requires an explicit
evidenced CPU, CUDA, or DirectML execution provider; it never silently defaults
to CPU. The tokenizer manifest records a canonical logical filename plus content
SHA-256 rather than a machine-local Hub cache path. `save_model=False` is
rejected because pre-existing graph files cannot prove that association. Runtime
loading fails closed when a required graph, tokenizer, MTP contract, or component
is absent.

## Supported GGUF Architectures

Support is declared once, in
`mobius/integrations/gguf/_arch_registry.py`, as five independent verdicts per
architecture — config extraction, tensor mapping, graph construction, runtime
packaging, and quantized import. Anything short of "supported" carries a reason,
and the table below is checked against the registry by
`_arch_registry_test.py::TestDocumentedSupportMatrix`, so it cannot drift.

Being listed is not a support claim. Graph import requires supported config,
tensor-map, and graph verdicts. Runtime-package emission separately requires a
supported runtime verdict backed by structured evidence; otherwise it rejects
before graph construction or durable output.

<!-- BEGIN GGUF SUPPORT MATRIX (generated; see _arch_registry.py) -->

| Canonical architecture | Aliases | Import route | Tensor exactness | Config/tensor/graph/runtime/quantized import | Restriction or evidence gap |
|---|---|---|---|---|---|
| `afmoe` | — | none (fails before config extraction) | audited-direct-loader-conditional-union | config=deferred; tensor_map=deferred; graph=deferred; runtime=deferred; quantized_import=supported | AFMoE combines sandwich norms, Q/K norms, sigmoid-gated attention, MuP embedding scaling, a dense prefix, correction-biased routed/shared experts, and optional interleaved sliding-window attention. Mobius has no graph or cache task owning that complete topology or its expert sidecars. |
| `apertus` | — | model=`apertus`; tensor=`llama`+`apertus_extras` | exact-direct-loader-conditional-union | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=supported | Config extraction, exact tensor-name closure, and a full synthetic GGUF graph build are covered, but no representative real-weight GGUF has yet passed ORT parity or generation validation. Runtime packaging remains deferred until that evidence exists. |
| `arcee` | — | model=`arcee`; tensor=`arcee` | not claimed | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=supported | Config extraction, exact tensor-name closure, and a full synthetic GGUF graph build are covered, but no representative real-weight GGUF has yet passed ORT parity or generation validation. Runtime packaging remains deferred until that evidence exists. |
| `arctic` | — | none (fails before config extraction) | not claimed | config=deferred; tensor_map=deferred; graph=deferred; runtime=deferred; quantized_import=supported | The pinned Arctic graph is not a standard pre-norm MoE block: every layer runs a dense parallel SwiGLU branch, then adds a separately normalized routed-expert branch computed from the pre-attention residual. Mobius's generic MoE graph replaces the dense FFN instead, so aliasing the existing Hugging Face 'arctic' registration would change residual topology and normalization. |
| `arwkv7` | — | none (fails before config extraction) | audited-direct-loader-conditional-union | config=deferred; tensor_map=deferred; graph=deferred; runtime=deferred; quantized_import=supported | ARWKV7 wraps RWKV7's delta-rule matrix recurrence in a distinct one-shift RMSNorm/Qwen residual topology with optional five-versus-six-way interpolation, optional gate/group norm, and Qwen SwiGLU. Treating it as RWKV7, Qwen, or Mamba would accept the wrong tensor closure and state ABI. |
| `baichuan` | — | model=`baichuan`; tensor=`llama` | not claimed | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=supported | Config extraction, exact tensor-name closure, and a full synthetic GGUF graph build are covered, but no representative real-weight GGUF has yet passed ORT parity or generation validation. Runtime packaging remains deferred until that evidence exists. |
| `bailingmoe` | — | model=`bailing_moe`; tensor=`llama`+`diffusion_fused_qkv`+`moe_extras` | audited-direct-loader-conditional-union | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=supported | Config extraction, exact tensor-name closure, and a full synthetic GGUF graph build are covered, but no representative real-weight GGUF has yet passed ORT parity or generation validation. Runtime packaging remains deferred until that evidence exists. |
| `bailingmoe2` | — | none (fails before config extraction) | audited-direct-loader-conditional-union | config=deferred; tensor_map=deferred; graph=deferred; runtime=deferred; quantized_import=supported | BailingMoE2 serializes complete dense-or-routed/shared expert trailing blocks plus NextN and layer-output norms, but the pinned loader marks every trailing tensor skipped and exposes no MTP graph. Mobius cannot invent executable head semantics from preserved storage. |
| `bailingmoe3` | — | none (fails before config extraction) | not claimed | config=deferred; tensor_map=deferred; graph=deferred; runtime=deferred; quantized_import=supported | BailingMoE3 alternates head-wise KDA recurrent layers with gated MLA layers, so each sequence carries three causal-convolution histories plus a matrix state alongside attention cache. Its routed sigmoid/correction-bias experts, always-on shared experts, and optional single NextN block also require tensor and task contracts Mobius does not implement. Ordinary KV or Mamba state would be wrong. |
| `bert` | — | model=`bert`; tensor=`bert` | not claimed | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=supported | Config extraction, exact pinned tensor closure, encoder-only task dispatch, and synthetic ORT execution are covered, but no pinned real GGUF artifact has passed independent embedding parity. Runtime packaging remains deferred until that evidence exists. |
| `bitnet` | — | none (fails before config extraction) | audited-direct-loader-conditional-union | config=deferred; tensor_map=deferred; graph=deferred; runtime=deferred; quantized_import=supported | BitNet requires post-attention and post-FFN sub-norms plus ternary projection weights with optional scalar scale tensors. Generic Llama topology and qtype handling do not represent that graph or quantization ABI. |
| `bloom` | — | model=`bloom`; tensor=`bloom` | not claimed | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=rejected | Config extraction, exact tensor-name closure, and a full synthetic GGUF graph build are covered, but no representative real-weight GGUF has yet passed ORT parity or generation validation. Runtime packaging remains deferred until that evidence exists. Quantization preservation is rejected because canonical Bloom GGUF stores one fused QKV projection that must be reordered and split into three graph targets. |
| `chameleon` | — | none (fails before config extraction) | exact-direct-loader-conditional-union | config=deferred; tensor_map=deferred; graph=deferred; runtime=deferred; quantized_import=supported | The pinned Chameleon converter deliberately omits the VQ image tokenizer while the text graph still requires bias-bearing Q/K norms, an additional swin_norm, and image-vocabulary logit suppression. Mobius's similarly named Hugging Face VLM graph does not prove that text-only GGUF contract, and full multimodal generation cannot be reconstructed from the serialized file. |
| `chatglm` | — | model=`chatglm`; tensor=`chatglm` | not claimed | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=rejected | Config extraction, exact tensor-name closure, and a full synthetic GGUF graph build are covered, but no representative real-weight GGUF has yet passed ORT parity or generation validation. Runtime packaging remains deferred until that evidence exists. Quantization preservation is rejected because fused QKV and gate/up tensors must be split into separate packed graph targets. |
| `clip` | — | none (fails before config extraction) | not claimed | config=rejected; tensor_map=rejected; graph=rejected; runtime=rejected; quantized_import=rejected | This is a multimodal projector sidecar, not a language model. Upstream it is a quant-only stub whose runtime lives outside libllama. Pass it to `build_from_gguf(text_gguf, mmproj=...)` alongside its text backbone rather than building it directly. |
| `codeshell` | — | model=`kclgpt`; tensor=`legacy_layernorm` | audited-direct-loader-conditional-union | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=rejected | Config extraction, exact tensor-name closure, and a full synthetic GGUF graph build are covered, but no representative real-weight GGUF has yet passed ORT parity or generation validation. Runtime packaging remains deferred until that evidence exists. Quantization preservation is rejected because the pinned loader accepts a fused QKV tensor that must be split into separate graph projections. |
| `cogvlm` | — | none (fails before config extraction) | exact-direct-loader-conditional-union | config=deferred; tensor_map=deferred; graph=deferred; runtime=deferred; quantized_import=supported | The pinned CogVLM text graph has modality-routed visual-expert Q/K/V/output and FFN banks in addition to the language projections. Its required cogvlm clip sidecar and feature-selection contract are also deferred, so aliasing a generic LLaVA or Llama graph would drop model-owned tensors and build the wrong package. |
| `cohere2` | — | model=`cohere2`; tensor=`llama` | not claimed | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=supported | Config extraction, exact tensor-name closure, and a full synthetic GGUF graph build are covered, but no representative real-weight GGUF has yet passed ORT parity or generation validation. Runtime packaging remains deferred until that evidence exists. |
| `cohere2moe` | — | none (fails before config extraction) | audited-direct-loader-conditional-union | config=deferred; tensor_map=deferred; graph=deferred; runtime=deferred; quantized_import=supported | Cohere2MoE's executable head uses sigmoid routed fused-or-split experts, optional shared experts, no FFN norm, and interleaved sliding-window KV state. Mobius's single dense Qwen3.5 MTP head cannot represent that graph or cache ABI. |
| `command-r` | — | model=`command_r`; tensor=`llama`+`command_r_extras` | audited-direct-loader-conditional-union | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=supported | Config extraction, exact tensor-name closure, and a full synthetic GGUF graph build are covered, but no representative real-weight GGUF has yet passed ORT parity or generation validation. Runtime packaging remains deferred until that evidence exists. Import requires canonical logit_scale metadata and is restricted to split Q/K/V tensors in the 40-layer Command-R profile; quantization preservation is supported only for that split route. Pinned variants with 64 or more layers require distinct per-head Q/K LayerNorm parameters that the current Attention graph cannot represent. |
| `dbrx` | — | none (fails before config extraction) | not claimed | config=deferred; tensor_map=deferred; graph=deferred; runtime=deferred; quantized_import=supported | The pinned DBRX graph requires LayerNorm, a fused QKV projection, Q/K/V projection clamping from attention.clamp_kqv, and a second LayerNorm before its routed experts. Mobius's generic MoE graph uses RMSNorm, separate Q/K/V projections, and no K/Q/V clamp; the existing Hugging Face 'dbrx' registration is therefore not GGUF-compatible. |
| `deci` | — | model=`llama`; tensor=`llama` | not claimed | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=supported | Config extraction, exact tensor-name closure, and a full synthetic GGUF graph build are covered, but no representative real-weight GGUF has yet passed ORT parity or generation validation. Runtime packaging remains deferred until that evidence exists. |
| `deepseek` | — | model=`deepseek`; tensor=`llama`+`diffusion_fused_qkv`+`deepseek_shared_moe_extras` | audited-direct-loader-conditional-union | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=supported | Config extraction, exact tensor-name closure, and a full synthetic GGUF graph build are covered, but no representative real-weight GGUF has yet passed ORT parity or generation validation. Runtime packaging remains deferred until that evidence exists. |
| `deepseek2` | — | none (fails before config extraction) | audited-direct-loader-conditional-union | config=deferred; tensor_map=deferred; graph=deferred; runtime=deferred; quantized_import=supported | DeepSeek2 MTP is a complete MLA plus routed/shared MoE block with compressed KV cache, Q/KV LoRA alternatives, target-owned embedding/head fallbacks, and architecture-specific gating. It is not Mobius's dense full-attention MTP head. |
| `deepseek2-ocr` | — | none (fails before config extraction) | exact-direct-loader-conditional-union | config=deferred; tensor_map=deferred; graph=deferred; runtime=deferred; quantized_import=supported | DeepSeek-OCR2 is a paired text-plus-vision package, not a generic DeepSeek text model. The text loader, deepseekocr/deepseekocr2 clip sidecars, SAM/projector stages, special token mixing, and cache contract have no single suffix-exact Mobius ownership map. Existing Hugging Face components therefore cannot justify partial GGUF construction. |
| `deepseek32` | — | none (fails before config extraction) | audited-direct-loader-conditional-union | config=deferred; tensor_map=deferred; graph=deferred; runtime=deferred; quantized_import=supported | DeepSeek3.2 extends the DeepSeek2 MLA/MoE head with DSA indexer projections, normalization, bias, and sparse-cache metadata. A normal KV-cache NextN task would omit required executed tensors and state. |
| `deepseek4` | — | none (fails before config extraction) | not claimed | config=deferred; tensor_map=deferred; graph=deferred; runtime=deferred; quantized_import=supported | The pinned DeepSeek-V4 GGUF runtime uses a dedicated raw sliding-window, CSA, HCA, and indexer compressed-cache ABI with persistent compressor state, rollback snapshots, four-stream hyper-connections, hash/sqrt-softplus routing, and optional MTP storage. Mobius's Hugging Face DeepSeek-V4 graph intentionally exports a dense attention fallback with ordinary KV state, so it is not an exact GGUF runtime graph. |
| `dflash` | — | model=`DFlashDraftModel`; tensor=`dflash` | not claimed | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=supported | This is a target-coupled speculative draft, never a standalone CausalLM. Config extraction, exact tensor closure, target shape/tokenizer validation, and synthetic draft execution are covered, but no pinned real GGUF pair has passed independent target+draft full-logit/proposed-token parity. Runtime packaging remains deferred until that evidence and an acceptance-loop integration exist. |
| `dots1` | — | model=`dots1`; tensor=`llama`+`diffusion_fused_qkv`+`moe_qk_norm_extras`+`deepseek_shared_moe_extras` | audited-direct-loader-conditional-union | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=supported | Config extraction, exact tensor-name closure, and a full synthetic GGUF graph build are covered, but no representative real-weight GGUF has yet passed ORT parity or generation validation. Runtime packaging remains deferred until that evidence exists. |
| `dots3note` | — | none (fails before config extraction) | audited-direct-loader-conditional-union | config=deferred; tensor_map=deferred; graph=deferred; runtime=deferred; quantized_import=supported | Dots3Note preserves an MLA/DSA trunk and a dense sliding-MLA NextN block, but the pinned loader explicitly has no MTP graph and skips the head. Mobius cannot infer runtime semantics from its serialized tensors. |
| `dream` | — | model=`dream`; tensor=`llama`+`diffusion_fused_qkv` | not claimed | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=supported | Config extraction, suffix-exact tensor closure, masked-diffusion task dispatch, and synthetic full-sequence execution are covered, but no pinned real GGUF has passed independent Hugging Face/llama.cpp masked-step logit parity and deterministic multi-step generation parity. Runtime packaging remains deferred until both exist. |
| `eagle3` | — | model=`Eagle3DraftModel`; tensor=`eagle3` | not claimed | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=supported | This is a target-coupled speculative draft, never a standalone CausalLM. Config extraction, exact tensor closure, target shape/tokenizer validation, and synthetic draft execution are covered, but no pinned real GGUF pair has passed independent target+draft full-logit/proposed-token parity. Runtime packaging remains deferred until that evidence and an acceptance-loop integration exist. |
| `ernie4_5` | — | model=`ernie4_5`; module=`gguf_legacy`; tensor=`legacy_layernorm` | audited-direct-loader-conditional-union | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=rejected | Config extraction, exact tensor-name closure, and a full synthetic GGUF graph build are covered, but no representative real-weight GGUF has yet passed ORT parity or generation validation. Runtime packaging remains deferred until that evidence exists. Import is narrowed to the dense split-Q/K/V, split-SwiGLU, full-RoPE variant and rejects all expert, fused, sectioned-position, and bias alternatives. Quantization preservation is rejected because this dedicated graph does not yet provide quantization-aware projection targets; use keep_quantized=False. |
| `ernie4_5-moe` | — | none (fails before config extraction) | audited-direct-loader-conditional-union | config=deferred; tensor_map=deferred; graph=deferred; runtime=deferred; quantized_import=supported | ERNIE 4.5 MoE selects periodic expert blocks after a dense prefix, permits an optional gate-expert matrix, and uses normalized routing with optional shared experts. Mobius has no matching per-layer schedule or converter transform. |
| `eurobert` | — | model=`eurobert`; module=`eurobert_gguf`; tensor=`eurobert` | exact-direct-loader-conditional-union | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=rejected | Config extraction, exact pinned tensor closure, encoder-only task dispatch, and synthetic ORT execution are covered, but no pinned real GGUF artifact has passed independent embedding parity. Runtime packaging remains deferred until that evidence exists. The mobius graph uses floating Linear modules for this architecture, so no MatMulNBits or BlockQuantizedMatMul target can consume preserved GGUF projection weights. Use keep_quantized=False for explicit float import. |
| `exaone` | — | model=`exaone`; tensor=`llama` | not claimed | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=supported | Config extraction, exact tensor-name closure, and a full synthetic GGUF graph build are covered, but no representative real-weight GGUF has yet passed ORT parity or generation validation. Runtime packaging remains deferred until that evidence exists. |
| `exaone-moe` | — | none (fails before config extraction) | audited-direct-loader-conditional-union | config=deferred; tensor_map=deferred; graph=deferred; runtime=deferred; quantized_import=supported | EXAONE-MoE serializes a dense trailing NextN block after an iSWA routed/shared expert trunk, but the pinned loader skips appended blocks. Mobius has no exact SWA schedule, remapped global-MTP transform, or executable head contract. |
| `exaone4` | — | none (fails before config extraction) | audited-direct-loader-conditional-union | config=deferred; tensor_map=deferred; graph=deferred; runtime=deferred; quantized_import=supported | EXAONE4 serializes attention/FFN post-norm trailing blocks and NextN tensors with optional synthetic Llama3 RoPE factors, but the pinned loader skips them. No Mobius task owns those preserved-only semantics. |
| `falcon` | — | model=`falcon`; tensor=`falcon` | not claimed | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=supported | Config extraction, exact tensor-name closure, and a full synthetic GGUF graph build are covered, but no representative real-weight GGUF has yet passed ORT parity or generation validation. Runtime packaging remains deferred until that evidence exists. |
| `falcon-h1` | — | model=`falcon_h1`; tensor=`falcon_h1` | not claimed | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=supported | The dedicated graph and GGUF importer preserve parallel Attention+Mamba2 layers and their four-state ABI, but runtime packaging remains deferred pending heterogeneous-state schema support (onnxruntime/mobius#605) and real full-logit plus deterministic stateful-generation evidence. |
| `gemma` | — | model=`gemma`; tensor=`llama` | not claimed | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=supported | Config extraction, exact tensor-name closure, and a full synthetic GGUF graph build are covered, but no representative real-weight GGUF has yet passed ORT parity or generation validation. Runtime packaging remains deferred until that evidence exists. |
| `gemma-embedding` | — | none (fails before config extraction) | audited-direct-loader-conditional-union | config=deferred; tensor_map=deferred; graph=deferred; runtime=deferred; quantized_import=supported | Gemma Embedding is a bidirectional stateless embedding graph with alternating sliding-window attention, four norm sites, sqrt(hidden) embedding scaling, and optional pooling/dense modules. Causal Gemma3 text/VLM tasks expose the wrong ABI. |
| `gemma2` | — | model=`gemma2`; tensor=`llama`+`gemma2_extras` | not claimed | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=supported | Config extraction, exact tensor-name closure, and a full synthetic GGUF graph build are covered, but no representative real-weight GGUF has yet passed ORT parity or generation validation. Runtime packaging remains deferred until that evidence exists. |
| `gemma3` | — | model=`gemma3_text`; tensor=`llama`+`gemma3_extras`; mmproj=`gemma3` | not claimed | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=supported | Config extraction, exact tensor-name closure, and a full synthetic GGUF graph build are covered, but no representative real-weight GGUF has yet passed ORT parity or generation validation. Runtime packaging remains deferred until that evidence exists. |
| `gemma3n` | — | none (fails before config extraction) | exact-direct-loader-conditional-union | config=deferred; tensor_map=deferred; graph=deferred; runtime=deferred; quantized_import=supported | Gemma3n GGUF is the text member of a vision-and-audio package whose gemma3nv and gemma3na clip companions carry distinct encoders and projectors. The text graph's per-layer embeddings, multimodal token replacement, processor assumptions, and package roles have not been validated against those pinned sidecar ABIs. |
| `gemma4` | — | model=`gemma4_text`; tensor=`llama`+`gemma4_extras`; mmproj=`gemma4` | not claimed | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=supported | Config extraction, exact tensor-name closure, and a full synthetic GGUF graph build are covered, but no representative real-weight GGUF has yet passed ORT parity or generation validation. Runtime packaging remains deferred until that evidence exists. |
| `gemma4-assistant` | — | none (fails before config extraction) | audited-direct-loader-conditional-union | config=deferred; tensor_map=deferred; graph=deferred; runtime=deferred; quantized_import=supported | Gemma4 Assistant is a standalone target-coupled model with pre/post projections, masked embeddings, scalar layer scales, its own KV cache, and a live target-model context. It is neither Gemma4 text nor the target-config-only draft/MTP ABI. |
| `glm-dsa` | `glm_dsa` | none (no tensor mapping route) | audited-direct-loader-conditional-union | config=supported; tensor_map=deferred; graph=supported; runtime=deferred; quantized_import=supported | Config extraction and the glm_moe_dsa graph are both available, but no GGUF→HuggingFace tensor-name mapping has been written for GLM-5.2's MLA + DSA-indexer tensor families yet, so weights cannot be routed into the graph. mobius has no GGUF→HuggingFace tensor-name mapping for this architecture, so its weights cannot be routed into a graph. Build from the Hugging Face checkpoint with `mobius build` instead. |
| `glm4` | — | none (fails before config extraction) | audited-direct-loader-conditional-union | config=deferred; tensor_map=deferred; graph=deferred; runtime=deferred; quantized_import=supported | GLM4 serializes complete fused-FFN trailing blocks and NextN tensors, but the pinned loader skips appended blocks; GLM-OCR converter transforms also permute Q/K for M-RoPE. It must not alias Qwen or an executable Mobius MTP head. |
| `glm4moe` | — | none (fails before config extraction) | audited-direct-loader-conditional-union | config=deferred; tensor_map=deferred; graph=deferred; runtime=deferred; quantized_import=supported | GLM4-MoE serializes biased attention and periodic dense/routed expert trailing blocks with mandatory router bias, but the pinned loader skips them. Mobius has no executable contract for the preserved head or its expert sidecars. |
| `gpt-oss` | — | none (fails before config extraction) | not claimed | config=deferred; tensor_map=deferred; graph=deferred; runtime=deferred; quantized_import=supported | The pinned GPT-OSS converter splits interleaved gate/up expert rows and repacks checkpoint block+scale tensors into expert-major MXFP4 values. Its loader additionally consumes expert biases, router bias, attention sinks, output bias, post-attention RMSNorm, and sliding-window RoPE metadata. Mobius does not yet own that complete GGUF transform and packed-expert ABI, so partial float or MXFP4 import is deferred. |
| `gpt2` | — | model=`gpt2`; tensor=`gpt2` | not claimed | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=rejected | Config extraction, exact tensor-name closure, and a full synthetic GGUF graph build are covered, but no representative real-weight GGUF has yet passed ORT parity or generation validation. Runtime packaging remains deferred until that evidence exists. Quantization preservation is rejected because canonical GPT-2 GGUF projections must be transposed into graph order, and the current packed route cannot transpose values together with their scales and zero-points. Use keep_quantized=False for explicit float import. |
| `gptj` | — | none (fails before config extraction) | no-loader | config=rejected; tensor_map=rejected; graph=rejected; runtime=rejected; quantized_import=supported | The pinned census reserves gptj but llama.cpp has no model loader for it. There is no bounded header-to-tensor contract to import, so conversion is rejected before config extraction. |
| `gptneox` | — | model=`gpt_neox`; module=`gguf_legacy`; tensor=`legacy_layernorm` | audited-direct-loader-conditional-union | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=rejected | Config extraction, exact tensor-name closure, and a full synthetic GGUF graph build are covered, but no representative real-weight GGUF has yet passed ORT parity or generation validation. Runtime packaging remains deferred until that evidence exists. The admitted subset requires parallel residual MHA. Quantization preservation is rejected because fused QKV rows must be split. |
| `granite` | — | none (fails before config extraction) | audited-direct-loader-conditional-union | config=deferred; tensor_map=deferred; graph=deferred; runtime=deferred; quantized_import=supported | The granite architecture is a conditional dense-or-MoE union with residual, embedding, attention, and inverse-logit scales, optional biases/RoPE factors, shared experts, and optional deep-stack inputs. It is not the GraniteMoE or GraniteHybrid GGUF contract. |
| `granite_swa` | — | none (fails before config extraction) | audited-direct-loader-conditional-union | config=deferred; tensor_map=deferred; graph=deferred; runtime=deferred; quantized_import=supported | Granite SWA requires attention sinks, a complete interleaved sliding-window schedule, residual/logit scaling, fused routed gate-up experts, and optional fused shared experts/deep-stack injection. Mobius owns neither that cache ABI nor its fused expert sidecars. |
| `granitehybrid` | — | model=`granitemoehybrid`; tensor=`granitehybrid` | not claimed | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=rejected | Exact mixed attention/Mamba2 scheduling, architecture-wide dense or routed MoE feed-forward selection, optional shared experts, Granite scaling, value-preserving float expert fusion, and strict pinned tensor closure are supported. Quantized sources require explicit dequantization because the current graph has no exact packed 3-D expert ABI; use keep_quantized=False. Generic ORT GenAI runtime packaging remains deferred because its released cache schema cannot represent heterogeneous KV, convolution, and recurrent state slots; tracked by onnxruntime/mobius#605. |
| `granitemoe` | — | model=`granitemoe`; tensor=`llama`+`moe_extras` | not claimed | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=supported | Config extraction, exact tensor-name closure, and a full synthetic GGUF graph build are covered, but no representative real-weight GGUF has yet passed ORT parity or generation validation. Runtime packaging remains deferred until that evidence exists. |
| `graniteswitch` | — | none (fails before config extraction) | audited-direct-loader-conditional-union | config=deferred; tensor_map=deferred; graph=deferred; runtime=deferred; quantized_import=supported | GraniteSwitch repurposes an appended synthetic layer as a token-history-driven adapter router and carries fourteen switched-LoRA tensors per block in addition to decoder KV state. It is not MTP, and Mobius has no switched-LoRA graph, MUL_MAT_ID quantization contract, or package ABI. |
| `grok` | — | none (fails before config extraction) | not claimed | config=deferred; tensor_map=deferred; graph=deferred; runtime=deferred; quantized_import=supported | The pinned Grok graph applies embedding, attention-output, and logit scales; attention, and optional final-logit softcaps; post-attention and post-FFN norms; and a dense-plus-routed expert residual scaled by sqrt(2)/2. Mobius's generic MoE graph has none of that combined topology, so the Hugging Face-style expert names are not evidence that a GGUF alias is safe. |
| `grovemoe` | — | none (fails before config extraction) | not claimed | config=deferred; tensor_map=deferred; graph=deferred; runtime=deferred; quantized_import=supported | The pinned GroveMoE graph shares router logits across two distinct expert banks but performs separate selections for normal and grouped chunk experts, then scales the adjugate contribution independently. It also applies per-head Q/K RMSNorm. Mobius has no graph or quantized ownership contract for the chunk-expert tensors, and silently treating them as ordinary experts would drop a required branch. |
| `hunyuan-dense` | `hunyuan_v1_dense` | model=`hunyuan_v1_dense`; tensor=`llama`+`hunyuan_extras` | not claimed | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=supported | Config extraction, exact tensor-name closure, and a full synthetic GGUF graph build are covered, but no representative real-weight GGUF has yet passed ORT parity or generation validation. Runtime packaging remains deferred until that evidence exists. |
| `hunyuan-moe` | — | none (fails before config extraction) | audited-direct-loader-conditional-union | config=deferred; tensor_map=deferred; graph=deferred; runtime=deferred; quantized_import=supported | Hunyuan-MoE applies Q/K normalization after RoPE and runs normalized routed experts with an always-parallel shared branch in every layer. Hunyuan dense/VL registrations are different architectures and cannot be reused. |
| `hunyuan_vl` | — | none (fails before config extraction) | exact-direct-loader-conditional-union | config=deferred; tensor_map=deferred; graph=deferred; runtime=deferred; quantized_import=supported | The pinned Hunyuan-VL decoder uses its own M-RoPE and Q/K-normalized text contract and pairs with a hunyuanvl clip sidecar. Mobius's Hunyuan-VL-MoT registration is a different dual-path architecture, so neither it nor the dense Hunyuan text model is a valid alias. |
| `hy_v3` | — | none (fails before config extraction) | audited-direct-loader-conditional-union | config=deferred; tensor_map=deferred; graph=deferred; runtime=deferred; quantized_import=supported | Hunyuan-V3 executes a full-attention NextN head with Q/K norms and optional sigmoid routed/shared experts, seeded from target hidden state. Its hyper/routing semantics are not Mobius's dense Qwen3.5 sidecar. |
| `internlm2` | — | model=`internlm2`; tensor=`llama` | not claimed | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=rejected | Config extraction, exact tensor-name closure, and a full synthetic GGUF graph build are covered, but no representative real-weight GGUF has yet passed ORT parity or generation validation. Runtime packaging remains deferred until that evidence exists. The mobius graph uses floating Linear modules for this architecture, so no MatMulNBits or BlockQuantizedMatMul target can consume preserved GGUF projection weights. Use keep_quantized=False for explicit float import. |
| `jais` | — | model=`jais`; module=`gguf_legacy`; tensor=`legacy_layernorm` | audited-direct-loader-conditional-union | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=rejected | Config extraction, exact tensor-name closure, and a full synthetic GGUF graph build are covered, but no representative real-weight GGUF has yet passed ORT parity or generation validation. Runtime packaging remains deferred until that evidence exists. Converter-baked MuP scales are retained exactly, while fused biased QKV must be split and therefore cannot preserve packed quantization. |
| `jais2` | — | model=`jais2`; tensor=`legacy_layernorm` | audited-direct-loader-conditional-union | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=supported | Config extraction, exact tensor-name closure, and a full synthetic GGUF graph build are covered, but no representative real-weight GGUF has yet passed ORT parity or generation validation. Runtime packaging remains deferred until that evidence exists. |
| `jamba` | — | model=`jamba`; tensor=`jamba` | not claimed | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=supported | Exact mixed attention/Mamba and dense/routed-MoE schedules, strict tensor closure and shapes, GGUF value transforms, compatible projection quantization, value-checked expert ordering, reduced Transformers parity, and multi-token ORT state threading, reorder, and replay are covered. Generic ORT GenAI runtime packaging remains deferred because its released cache schema cannot represent heterogeneous KV, convolution, and recurrent state slots; tracked by #605. |
| `jina-bert-v2` | — | model=`bert`; module=`jina_bert_v2_gguf`; tensor=`jina_bert_v2` | exact-direct-loader-conditional-union | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=rejected | Config extraction, exact pinned tensor closure, encoder-only task dispatch, and synthetic ORT execution are covered, but no pinned real GGUF artifact has passed independent embedding parity. Runtime packaging remains deferred until that evidence exists. Optional Q/K norms and fused GeGLU inputs have no complete packed quantization route; use keep_quantized=False. |
| `jina-bert-v3` | — | none (fails before config extraction) | not claimed | config=deferred; tensor_map=deferred; graph=deferred; runtime=deferred; quantized_import=supported | JinaBERT v3 uses RoPE and may alternate dense GELU and routed MoE layers. BertModel has absolute positions and no MoE path. |
| `kimi-k3` | — | model=`kimi_k3`; tensor=`kimi_k3` | not claimed | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=supported | The exact KDA/NoPE gated-MLA schedule, four-state recurrent ABI, attention-residual banks, Stable LatentMoE routing, SiTU activation, strict metadata/tensor closure, and compatible projection quantization are supported. Released generic ORT GenAI cache schemas cannot represent the heterogeneous KV plus convolution/matrix state ABI; tracked by onnxruntime/mobius#605. |
| `kimi-linear` | — | model=`kimi_linear`; tensor=`kimi_linear` | not claimed | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=supported | The exact KDA/NoPE-MLA schedule, four-state recurrent ABI, dense/MoE topology, correction-bias routing, pinned metadata, tensor closure, and compatible MatMul/expert quantization are supported. Released generic ORT GenAI cache schemas cannot represent heterogeneous KV plus three convolution histories and a matrix state, and representative real-weight GGUF evidence is pending; package runtime remains tracked by onnxruntime/mobius#605. |
| `laguna` | — | none (fails before config extraction) | audited-direct-loader-conditional-union | config=deferred; tensor_map=deferred; graph=deferred; runtime=deferred; quantized_import=supported | Laguna combines per-head-or-element softplus attention gates, dual-RoPE interleaved sliding-window attention, a dense prefix, and sigmoid correction-biased routed/shared experts. Mobius has no exact graph or iSWA cache contract. |
| `lfm2` | — | model=`lfm2`; tensor=`lfm2` | not claimed | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=supported | Config extraction, exact pinned tensor-name closure, GGUF value transforms, and synthetic recurrent-state execution are covered, but no representative real-weight GGUF has yet passed independent full-logit parity and deterministic multi-token stateful ORT generation. Runtime packaging remains deferred until that evidence exists. |
| `lfm2moe` | — | model=`lfm2_moe`; tensor=`lfm2`+`lfm2_moe_extras` | not claimed | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=rejected | Config extraction, exact pinned tensor-name closure, GGUF value transforms, and synthetic recurrent-state execution are covered, but no representative real-weight GGUF has yet passed independent full-logit parity and deterministic multi-token stateful ORT generation. Runtime packaging remains deferred until that evidence exists. The mobius graph uses floating Linear modules for this architecture, so no MatMulNBits or BlockQuantizedMatMul target can consume preserved GGUF projection weights. Use keep_quantized=False for explicit float import. |
| `llada` | — | model=`llada`; tensor=`llama` | not claimed | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=supported | Config extraction, suffix-exact tensor closure, masked-diffusion task dispatch, and synthetic full-sequence execution are covered, but no pinned real GGUF has passed independent Hugging Face/llama.cpp masked-step logit parity and deterministic multi-step generation parity. Runtime packaging remains deferred until both exist. |
| `llada-moe` | — | model=`llada`; module=`llada_moe`; tensor=`llama`+`diffusion_fused_qkv`+`moe_qk_norm_extras`+`moe_extras` | not claimed | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=supported | Config extraction, suffix-exact tensor closure, masked-diffusion task dispatch, and synthetic full-sequence execution are covered, but no pinned real GGUF has passed independent Hugging Face/llama.cpp masked-step logit parity and deterministic multi-step generation parity. Runtime packaging remains deferred until both exist. |
| `llama` | `mistral` | model=`llama`; tensor=`llama` | not claimed | config=supported; tensor_map=supported; graph=supported; runtime=supported; quantized_import=supported | Runtime support is restricted to exact structured evidence matches. Currently that is only neopolita/smollm-135m-gguf F16 at the pinned artifact, CPU import route, evidenced ONNX Runtime/ORT GenAI versions, and the pinned HuggingFaceTB/SmolLM-135M tokenizer revision. |
| `llama-embed` | — | none (fails before config extraction) | audited-direct-loader-conditional-union | config=deferred; tensor_map=deferred; graph=deferred; runtime=deferred; quantized_import=supported | llama-embed is a canonical embedding architecture that inherits Llama's conditional tensor loader but exposes the embedding graph rather than causal logits. Mobius has no GGUF embedding task/package contract for this ID, so it must not alias ordinary llama. |
| `llama4` | — | none (fails before config extraction) | exact-direct-loader-conditional-union | config=deferred; tensor_map=deferred; graph=deferred; runtime=deferred; quantized_import=supported | Llama4 GGUF is the text member of a paired multimodal package and may contain routed experts and architecture-specific cross-modal layer scheduling. The llama4 clip vision tower, token mixing, position IDs, and package ABI remain deferred; text-backbone similarity is not evidence that the complete GGUF tensor closure is owned. |
| `maincoder` | — | none (fails before config extraction) | audited-direct-loader-conditional-union | config=deferred; tensor_map=deferred; graph=deferred; runtime=deferred; quantized_import=supported | Maincoder applies Q/K RMSNorm after RoPE and uses an exact tied-output QK-normalized SwiGLU closure. Existing generic QK-normalized graphs use different ordering, so a family alias would change attention. |
| `mamba` | — | model=`mamba`; tensor=`mamba` | not claimed | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=rejected | Config extraction, exact pinned tensor-name closure, GGUF value transforms, and synthetic recurrent-state execution are covered, but no representative real-weight GGUF has yet passed independent full-logit parity and deterministic multi-token stateful ORT generation. Runtime packaging remains deferred until that evidence exists. The mobius graph uses floating Linear modules for this architecture, so no MatMulNBits or BlockQuantizedMatMul target can consume preserved GGUF projection weights. Use keep_quantized=False for explicit float import. |
| `mamba2` | — | model=`mamba2`; tensor=`mamba2` | not claimed | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=rejected | Config extraction, exact pinned tensor-name closure, GGUF value transforms, and synthetic recurrent-state execution are covered, but no representative real-weight GGUF has yet passed independent full-logit parity and deterministic multi-token stateful ORT generation. Runtime packaging remains deferred until that evidence exists. The mobius graph uses floating Linear modules for this architecture, so no MatMulNBits or BlockQuantizedMatMul target can consume preserved GGUF projection weights. Use keep_quantized=False for explicit float import. |
| `mellum` | — | none (fails before config extraction) | audited-direct-loader-conditional-union | config=deferred; tensor_map=deferred; graph=deferred; runtime=deferred; quantized_import=supported | Mellum requires an untied head, Q/K norms, routed experts in every layer, and a metadata-defined full/sliding attention schedule with distinct RoPE behavior. Mobius has no matching iSWA cache or expert preservation contract. |
| `mimo2` | — | none (fails before config extraction) | audited-direct-loader-conditional-union | config=deferred; tensor_map=deferred; graph=deferred; runtime=deferred; quantized_import=supported | MiMo2 requires fused-QKV dense MTP blocks, attention sinks, interleaved sliding KV cache, and three chained heads selected by offsets. Mobius permits one head and cannot preserve that state or FP8 converter transform. |
| `minicpm` | — | none (fails before config extraction) | audited-direct-loader-conditional-union | config=deferred; tensor_map=deferred; graph=deferred; runtime=deferred; quantized_import=supported | MiniCPM requires architecture-specific embedding, residual, and logit scales, Q/K permutation, optional long/short RoPE tensors, and a conditional dense-or-MoE loader. The existing MiniCPM graph does not prove this complete GGUF contract. |
| `minicpm3` | — | none (fails before config extraction) | not claimed | config=deferred; tensor_map=deferred; graph=deferred; runtime=deferred; quantized_import=supported | The pinned MiniCPM3 graph uses MLA Q/KV LoRA projections, separate NoPE/RoPE query and key channels, and embedding, residual, and LM-head scales. The current Mobius MiniCPM graph does not represent that exact topology or its scales. |
| `minimax-01` | — | model=`minimax`; tensor=`minimax` | not claimed | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=supported | Graph import is exact, but released ORT GenAI packaging cannot represent the heterogeneous KV/recurrent state slots or bounded rollback snapshots; runtime packaging remains tracked by #605. |
| `minimax-m2` | — | none (fails before config extraction) | audited-direct-loader-conditional-union | config=deferred; tensor_map=deferred; graph=deferred; runtime=deferred; quantized_import=supported | MiniMax-M2 uses full-vector Q/K norms, partial RoPE, and all-layer correction-biased routed experts under metadata-selected gating. Mobius has no exact graph or suffix-safe expert import for that topology. |
| `minimax-m3` | — | none (fails before config extraction) | audited-direct-loader-conditional-union | config=deferred; tensor_map=deferred; graph=deferred; runtime=deferred; quantized_import=supported | MiniMax-M3 adds F32 sparse-indexer tensors and a second index-key cache with position/cell maps, block masks, rollback, and reorder semantics alongside main K/V state. Mobius has no MSA cache task or sparse-index operators; dense fallback would change the model. |
| `mistral3` | — | none (fails before config extraction) | exact-direct-loader-conditional-union | config=deferred; tensor_map=deferred; graph=deferred; runtime=deferred; quantized_import=supported | The pinned Mistral3 loader selects dense or routed-expert text blocks from metadata and applies architecture-specific output temperature scaling. A VLM package additionally requires the deferred Pixtral clip sidecar and exact patch/merge/token contract. The existing Hugging Face Mistral3 graph does not cover that conditional GGUF closure. |
| `mistral4` | — | none (fails before config extraction) | audited-direct-loader-conditional-union | config=deferred; tensor_map=deferred; graph=deferred; runtime=deferred; quantized_import=supported | Mistral4 has no NextN metadata or MTP graph; it inherits Mistral3's conditional dense/MoE tensor loader and overrides graph construction. It is not llama, mistral alias text, or any DeepSeek/Qwen MTP family. |
| `modern-bert` | — | model=`modernbert`; tensor=`modern_bert` | not claimed | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=supported | Config extraction, exact pinned tensor closure, encoder-only task dispatch, and synthetic ORT execution are covered, but no pinned real GGUF artifact has passed independent embedding parity. Runtime packaging remains deferred until that evidence exists. |
| `mpt` | — | model=`mpt`; module=`gguf_legacy`; tensor=`legacy_layernorm` | audited-direct-loader-conditional-union | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=rejected | Config extraction, exact tensor-name closure, and a full synthetic GGUF graph build are covered, but no representative real-weight GGUF has yet passed ORT parity or generation validation. Runtime packaging remains deferred until that evidence exists. The admitted subset rejects learned positions, Q/K norms, KQV clipping, AWQ activation scales, and inconsistent optional bias families. Quantization preservation is rejected because fused QKV must be split. |
| `muse-glimmer` | `muse_glimmer` | model=`muse_glimmer_text`; tensor=`llama`+`muse_glimmer_extras`; mmproj=`muse_glimmer` | not claimed | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=supported | Config extraction, exact tensor-name closure, and a full synthetic GGUF graph build are covered, but no representative real-weight GGUF has yet passed ORT parity or generation validation. Runtime packaging remains deferred until that evidence exists. |
| `nanbeige` | — | none (fails before config extraction) | audited-direct-loader-conditional-union | config=deferred; tensor_map=deferred; graph=deferred; runtime=deferred; quantized_import=supported | Nanbeige reuses physical layer weights across a configurable logical loop count, optionally normalizes between loops, and allocates a distinct KV slot for every logical occurrence. A Llama alias would build the wrong layer count and cache ABI. |
| `nemotron` | — | model=`nemotron`; tensor=`llama` | not claimed | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=supported | Config extraction, exact tensor-name closure, and a full synthetic GGUF graph build are covered, but no representative real-weight GGUF has yet passed ORT parity or generation validation. Runtime packaging remains deferred until that evidence exists. |
| `nemotron_h` | — | model=`nemotron_h`; tensor=`nemotron_h` | not claimed | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=rejected | Config extraction, exact pinned tensor-name closure, GGUF value transforms, and synthetic recurrent-state execution are covered, but no representative real-weight GGUF has yet passed independent full-logit parity and deterministic multi-token stateful ORT generation. Runtime packaging remains deferred until that evidence exists. The mobius graph uses floating Linear modules for this architecture, so no MatMulNBits or BlockQuantizedMatMul target can consume preserved GGUF projection weights. Use keep_quantized=False for explicit float import. |
| `nemotron_h_moe` | — | model=`nemotron_h`; tensor=`nemotron_h_moe` | not claimed | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=rejected | Exact mixed attention/Mamba2/dense/MoE scheduling, sigmoid correction-bias routing, shared experts, optional latent projections, and strict GGUF tensor closure are supported. Generic ORT GenAI runtime packaging remains deferred because its released cache schema cannot represent heterogeneous KV, convolution, and recurrent state slots; tracked by onnxruntime/mobius#605. Quantization preservation is unsupported because mixed Mamba2 recurrent parameters must remain dequantized and correction-biased sigmoid experts cannot use the fused MoE ABI. Use keep_quantized=False for explicit float import. |
| `neo-bert` | — | model=`neobert`; module=`neo_bert_gguf`; tensor=`neo_bert` | exact-direct-loader-conditional-union | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=rejected | Config extraction, exact pinned tensor closure, encoder-only task dispatch, and synthetic ORT execution are covered, but no pinned real GGUF artifact has passed independent embedding parity. Runtime packaging remains deferred until that evidence exists. Packed QKV and fused SwiGLU have no complete quantized split route; use keep_quantized=False. |
| `nomic-bert` | — | model=`nomic_bert`; module=`nomic_bert_gguf`; tensor=`nomic_bert` | exact-direct-loader-conditional-union | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=rejected | Config extraction, exact pinned tensor closure, encoder-only task dispatch, and synthetic ORT execution are covered, but no pinned real GGUF artifact has passed independent embedding parity. Runtime packaging remains deferred until that evidence exists. The mobius graph uses floating Linear modules for this architecture, so no MatMulNBits or BlockQuantizedMatMul target can consume preserved GGUF projection weights. Use keep_quantized=False for explicit float import. |
| `nomic-bert-moe` | — | none (fails before config extraction) | not claimed | config=deferred; tensor_map=deferred; graph=deferred; runtime=deferred; quantized_import=supported | NomicBERT-MoE alternates dense and routed-expert FFNs according to moe_every_n_layers. Mobius has no encoder MoE graph with that schedule. |
| `olmo` | — | model=`olmo`; tensor=`olmo` | not claimed | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=supported | Config extraction, exact tensor-name closure, and a full synthetic GGUF graph build are covered, but no representative real-weight GGUF has yet passed ORT parity or generation validation. Runtime packaging remains deferred until that evidence exists. |
| `olmo2` | — | model=`olmo2`; tensor=`llama`+`olmo2_extras` | not claimed | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=supported | Config extraction, exact tensor-name closure, and a full synthetic GGUF graph build are covered, but no representative real-weight GGUF has yet passed ORT parity or generation validation. Runtime packaging remains deferred until that evidence exists. |
| `olmoe` | — | model=`olmoe`; tensor=`llama`+`moe_qk_norm_extras`+`moe_extras` | not claimed | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=supported | Config extraction, exact tensor-name closure, and a full synthetic GGUF graph build are covered, but no representative real-weight GGUF has yet passed ORT parity or generation validation. Runtime packaging remains deferred until that evidence exists. |
| `openelm` | — | model=`openelm`; module=`gguf_legacy`; tensor=`legacy_layernorm`+`exact_legacy_gguf_extras` | audited-direct-loader-conditional-union | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=rejected | Config extraction, exact tensor-name closure, and a full synthetic GGUF graph build are covered, but no representative real-weight GGUF has yet passed ORT parity or generation validation. Runtime packaging remains deferred until that evidence exists. Quantization preservation is rejected because every OpenELM layer stores fused QKV rows that must be split into per-layer Q/K/V graph projections. |
| `orion` | — | model=`orion`; tensor=`legacy_layernorm` | audited-direct-loader-conditional-union | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=rejected | Config extraction, exact tensor-name closure, and a full synthetic GGUF graph build are covered, but no representative real-weight GGUF has yet passed ORT parity or generation validation. Runtime packaging remains deferred until that evidence exists. Fused QKV input is rejected because its import transform is not implemented. Quantization preservation is also rejected for this architecture. |
| `paddleocr` | — | none (fails before config extraction) | strongest-converter-family-inventory-loader-inherited-from-ernie4_5-with-optional-attn-output-bias | config=deferred; tensor_map=deferred; graph=deferred; runtime=deferred; quantized_import=supported | PaddleOCR-VL uses an ERNIE-derived GGUF loader with an optional bias on attention output closure and a required paddleocr clip vision/projector sidecar. Its processor ranks, image-token counts, offsets, and package identity have no pinned Mobius GGUF parity evidence, so it cannot be accepted as an ordinary Qwen2 text file. |
| `pangu-embedded` | — | none (fails before config extraction) | audited-direct-loader-conditional-union | config=deferred; tensor_map=deferred; graph=deferred; runtime=deferred; quantized_import=supported | Pangu Embedded is a causal LM, not an embedding task, and requires a mandatory attention-output bias plus conditional LongRoPE factor tensors. Mobius has no exact graph or suffix closure for this misleadingly named architecture. |
| `phi2` | — | model=`phi`; tensor=`phi2` | not claimed | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=rejected | Config extraction, exact tensor-name closure, and a full synthetic GGUF graph build are covered, but no representative real-weight GGUF has yet passed ORT parity or generation validation. Runtime packaging remains deferred until that evidence exists. Quantization preservation is rejected because the Phi-2 attention, MLP, and output graph uses float-only linear modules. |
| `phi3` | — | model=`phi3`; tensor=`phi3` | not claimed | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=supported | Config extraction, exact tensor-name closure, and a full synthetic GGUF graph build are covered, but no representative real-weight GGUF has yet passed ORT parity or generation validation. Runtime packaging remains deferred until that evidence exists. |
| `phimoe` | — | model=`phimoe`; module=`phimoe_gguf`; tensor=`llama`+`phi3`+`moe_extras` | not claimed | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=supported | Config extraction, exact tensor-name closure, and a full synthetic GGUF graph build are covered, but no representative real-weight GGUF has yet passed ORT parity or generation validation. Runtime packaging remains deferred until that evidence exists. |
| `plamo` | — | none (fails before config extraction) | audited-direct-loader-conditional-union | config=deferred; tensor_map=deferred; graph=deferred; runtime=deferred; quantized_import=supported | PLaMo uses one RMSNorm feeding attention and FFN in parallel, fixed grouped-query geometry, and converter-specific Q/output projection shuffles. Sequential Llama topology and direct external tensor reuse would both be incorrect. |
| `plamo2` | — | model=`plamo2`; tensor=`plamo2` | not claimed | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=supported | The dedicated graph and GGUF importer preserve PLaMo2's alternating Mamba1/attention layers and mixed state ABI, but released ORT GenAI cannot represent heterogeneous per-layer state. Runtime packaging remains deferred to onnxruntime/mobius#605. |
| `plamo3` | — | none (fails before config extraction) | audited-direct-loader-conditional-union | config=deferred; tensor_map=deferred; graph=deferred; runtime=deferred; quantized_import=supported | PLaMo3 requires fused QKV and fused SwiGLU, four norm sites with architecture-specific offset transforms, Q/K norm before RoPE, and alternating full/sliding attention state. Mobius has no exact iSWA schedule or value transform. |
| `plm` | — | model=`plm`; tensor=`plm` | audited-direct-loader-conditional-union | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=supported | Config extraction, exact tensor-name closure, and a full synthetic GGUF graph build are covered, but no representative real-weight GGUF has yet passed ORT parity or generation validation. Runtime packaging remains deferred until that evidence exists. |
| `pockettts` | — | none (fails before config extraction) | not claimed | config=deferred; tensor_map=deferred; graph=deferred; runtime=deferred; quantized_import=supported | The primary GGUF is only PocketTTS's transformed causal CALM backbone: its embedding table contains folded learned conditioning rows and its duplicated embedding output is not a semantic LM head. Voice encoding, continuous 32-D flow generation, EOS scoring, and the stateful 24-kHz Mimi decoder live in a required pockettts_spkenc/pockettts_gen mmproj bundle that Mobius cannot import. Registering the backbone as text generation or TTS would expose the wrong I/O contract, so standalone conversion is refused before graph construction. |
| `qwen` | — | model=`qwen`; tensor=`llama`+`qwen1_extras` | audited-direct-loader-conditional-union | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=rejected | Config extraction, exact tensor-name closure, and a full synthetic GGUF graph build are covered, but no representative real-weight GGUF has yet passed ORT parity or generation validation. Runtime packaging remains deferred until that evidence exists. Quantization preservation is rejected because Qwen v1 stores fused QKV weights that must be split into separate graph projections. |
| `qwen2` | — | model=`qwen2`; tensor=`llama` | not claimed | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=supported | Config extraction, exact tensor-name closure, and a full synthetic GGUF graph build are covered, but no representative real-weight GGUF has yet passed ORT parity or generation validation. Runtime packaging remains deferred until that evidence exists. |
| `qwen2moe` | `qwen2_moe` | model=`qwen2_moe`; tensor=`llama`+`moe_extras` | not claimed | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=supported | Config extraction, exact tensor-name closure, and a full synthetic GGUF graph build are covered, but no representative real-weight GGUF has yet passed ORT parity or generation validation. Runtime packaging remains deferred until that evidence exists. |
| `qwen2vl` | — | model=`qwen2_vl_text`; tensor=`llama`; mmproj=`qwen_vl` | exact-direct-loader-conditional-union | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=supported | Text and paired Qwen2/Qwen2.5-VL projector graph import are supported for the exact split-QKV llama.cpp artifacts, but downstream multimodal runtime execution has not been evidenced. |
| `qwen3` | — | model=`qwen3`; tensor=`llama` | not claimed | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=supported | Config extraction, exact tensor-name closure, and a full synthetic GGUF graph build are covered, but no representative real-weight GGUF has yet passed ORT parity or generation validation. Runtime packaging remains deferred until that evidence exists. |
| `qwen35` | — | model=`qwen3_5_text`; tensor=`llama`+`qwen35_hybrid_extras` | not claimed | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=supported | Config extraction, exact pinned tensor-name closure, GGUF value transforms, and synthetic recurrent-state execution are covered, but no representative real-weight GGUF has yet passed independent full-logit parity and deterministic multi-token stateful ORT generation. Runtime packaging remains deferred until that evidence exists. |
| `qwen35moe` | — | model=`qwen3_5_moe`; tensor=`llama`+`moe_extras`+`qwen35_hybrid_extras` | not claimed | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=supported | Config extraction, exact pinned tensor-name closure, GGUF value transforms, and synthetic recurrent-state execution are covered, but no representative real-weight GGUF has yet passed independent full-logit parity and deterministic multi-token stateful ORT generation. Runtime packaging remains deferred until that evidence exists. |
| `qwen3moe` | `qwen3_moe` | model=`qwen3_moe`; tensor=`llama`+`moe_qk_norm_extras`+`moe_extras` | not claimed | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=supported | Config extraction, exact tensor-name closure, and a full synthetic GGUF graph build are covered, but no representative real-weight GGUF has yet passed ORT parity or generation validation. Runtime packaging remains deferred until that evidence exists. |
| `qwen3next` | — | model=`qwen3_next`; tensor=`llama`+`moe_extras`+`qwen3next_hybrid_extras` | not claimed | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=supported | Config extraction, exact pinned tensor-name closure, GGUF value transforms, and synthetic recurrent-state execution are covered, but no representative real-weight GGUF has yet passed independent full-logit parity and deterministic multi-token stateful ORT generation. Runtime packaging remains deferred until that evidence exists. |
| `qwen3tts` | — | none (fails before config extraction) | not claimed | config=deferred; tensor_map=deferred; graph=deferred; runtime=deferred; quantized_import=supported | The primary GGUF is only a transformed Qwen3-TTS talker backbone, not the existing Mobius Qwen3TTS conditional-generation or codec model. The converter folds the text projection into an extended text+codec embedding table and emits a 3072-row codec head whose logits are shifted into that combined vocabulary. The required speaker encoder, 15-codebook predictor, and stateful 24-kHz code-to-wave decoder live in a qwen3tts_spkenc/qwen3tts_gen mmproj bundle that Mobius cannot import. Standalone conversion is therefore refused before graph construction. |
| `qwen3vl` | — | none (fails before config extraction) | exact-direct-loader-conditional-union | config=deferred; tensor_map=deferred; graph=deferred; runtime=deferred; quantized_import=supported | Qwen3-VL text GGUF requires multimodal position IDs and an exact qwen3vl_merger clip companion, including deep-stack vision features and architecture-specific token placement. The existing Hugging Face text graph alone does not establish sidecar tensor closure, processor parity, or a safe text-only fallback. |
| `qwen3vlmoe` | — | none (fails before config extraction) | exact-direct-loader-conditional-union | config=deferred; tensor_map=deferred; graph=deferred; runtime=deferred; quantized_import=supported | Qwen3-VL-MoE combines the Qwen3-VL multimodal position/token contract and merger sidecar with routed experts in the text backbone. Generic Qwen3-MoE tensor similarity does not cover the paired vision package, expert sidecars, effective tied head ownership, or multimodal cache ABI. |
| `refact` | — | model=`refact`; module=`gguf_legacy`; tensor=`legacy_layernorm` | audited-direct-loader-conditional-union | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=rejected | Config extraction, exact tensor-name closure, and a full synthetic GGUF graph build are covered, but no representative real-weight GGUF has yet passed ORT parity or generation validation. Runtime packaging remains deferred until that evidence exists. Import is narrowed to split, bias-free dense tensors with one KV head; loaded-but-unexecuted expert, RoPE-factor, and bias families are rejected. Quantization preservation is rejected because this dedicated graph does not yet provide quantization-aware projection targets; use keep_quantized=False. |
| `rnd1` | — | model=`llada`; module=`rnd1`; tensor=`llama`+`diffusion_fused_qkv`+`moe_qk_norm_extras`+`moe_extras` | not claimed | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=supported | Config extraction, suffix-exact tensor closure, masked-diffusion task dispatch, and synthetic full-sequence execution are covered, but no pinned real GGUF has passed independent Hugging Face/llama.cpp masked-step logit parity and deterministic multi-step generation parity. Runtime packaging remains deferred until both exist. |
| `rwkv6` | — | none (fails before config extraction) | audited-direct-loader-conditional-union | config=deferred; tensor_map=deferred; graph=deferred; runtime=deferred; quantized_import=supported | RWKV6 carries two F32 states per layer (two token-shift vectors and a per-head WKV matrix) and applies token-dependent exp(-exp(decay)), a time_first read-before-update term, per-head group norm, and cumulative rescale transforms. Mobius has no RWKV state task; Mamba conv/SSM state and transformer KV cache are not equivalent. |
| `rwkv6qwen2` | — | none (fails before config extraction) | audited-direct-loader-conditional-union | config=deferred; tensor_map=deferred; graph=deferred; runtime=deferred; quantized_import=supported | RWKV6-Qwen2 is neither Qwen2 attention nor native RWKV6: it carries one F32 token-shift vector plus a per-head matrix state and uses k*(1-w) gated linear attention, optional biased/GQA projections, a sigmoid gate, and parallel Qwen SwiGLU. Mobius has no task for that recurrent ABI; Mamba conv/SSM state and transformer KV cache are not equivalent. |
| `rwkv7` | — | none (fails before config extraction) | audited-direct-loader-conditional-union | config=deferred; tensor_map=deferred; graph=deferred; runtime=deferred; quantized_import=supported | RWKV7 requires a two-shift F32 state plus a per-head matrix state, generalized delta-rule recurrence, six-way token mixing, first-layer value residuals shared across depth, ICLR/key-adaptation vectors, and an r_k residual around LayerNorm and group norm. Neither Mamba selective scan nor KV-cache plumbing represents it. |
| `seed_oss` | — | model=`seed_oss`; tensor=`llama`+`seed_oss_extras` | not claimed | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=supported | Config extraction, exact tensor-name closure, and a full synthetic GGUF graph build are covered, but no representative real-weight GGUF has yet passed ORT parity or generation validation. Runtime packaging remains deferred until that evidence exists. |
| `smallthinker` | — | none (fails before config extraction) | not claimed | config=deferred; tensor_map=deferred; graph=deferred; runtime=deferred; quantized_import=supported | The pinned SmallThinker graph computes router logits from the unnormalized layer input, uses ReLU experts with metadata-selected sigmoid or softmax gating, and can disable RoPE or select sliding-window attention per layer. Mobius's generic MoE graph routes after the FFN norm with softmax/SwiGLU experts and has no matching per-layer RoPE schedule. |
| `smollm3` | — | model=`smollm3`; tensor=`llama` | not claimed | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=supported | Config extraction, exact tensor-name closure, and a full synthetic GGUF graph build are covered, but no representative real-weight GGUF has yet passed ORT parity or generation validation. Runtime packaging remains deferred until that evidence exists. |
| `stablelm` | — | model=`stablelm`; tensor=`llama` | not claimed | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=supported | Config extraction, exact tensor-name closure, and a full synthetic GGUF graph build are covered, but no representative real-weight GGUF has yet passed ORT parity or generation validation. Runtime packaging remains deferred until that evidence exists. |
| `starcoder` | — | model=`gpt_bigcode`; tensor=`starcoder` | audited-direct-loader-conditional-union | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=rejected | Config extraction, exact tensor-name closure, and a full synthetic GGUF graph build are covered, but no representative real-weight GGUF has yet passed ORT parity or generation validation. Runtime packaging remains deferred until that evidence exists. Quantization preservation is rejected because StarCoder stores one fused biased MQA projection that must be split for the graph. |
| `starcoder2` | — | model=`starcoder2`; tensor=`llama` | not claimed | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=supported | Config extraction, exact tensor-name closure, and a full synthetic GGUF graph build are covered, but no representative real-weight GGUF has yet passed ORT parity or generation validation. Runtime packaging remains deferred until that evidence exists. |
| `step35` | — | none (fails before config extraction) | audited-direct-loader-conditional-union | config=deferred; tensor_map=deferred; graph=deferred; runtime=deferred; quantized_import=supported | Step3.5 executes one or more interleaved-SWA NextN heads with optional gates, routed/shared experts, centered-norm transforms, per-layer head geometry, and dedicated cache offsets. Mobius's one-head dense MTP ABI cannot represent it. |
| `t5` | — | model=`t5`; tensor=`t5` | not claimed | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=supported | Graph import is covered, but no independent full-logit and generation parity run has yet validated a pinned real T5 GGUF runtime package. |
| `t5encoder` | — | model=`t5encoder`; tensor=`t5` | not claimed | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=supported | Encoder hidden-state import is covered, but the pinned real artifact lacks independent provenance and full hidden-state parity evidence. |
| `talkie` | — | none (fails before config extraction) | not claimed | config=deferred; tensor_map=deferred; graph=deferred; runtime=deferred; quantized_import=supported | Talkie is a text causal LM despite its upstream survey cohort. Its graph uses weight-free RMSNorm, post-RoPE Q/K normalization, learned attention and MLP gain sidecars, a per-head query gain, and an embedding skip in every block. No Mobius graph or tensor-value transform implements that combination, and no pinned real GGUF has passed independent full-logit, KV-state, and generation parity. Aliasing it to Llama or an audio task would build the wrong model. |
| `wavtokenizer-dec` | — | none (fails before config extraction) | not claimed | config=deferred; tensor_map=deferred; graph=deferred; runtime=deferred; quantized_import=supported | wavtokenizer-dec is a stateless non-causal code-token to ISTFT-parameter network, not a waveform codec decoder. Its metadata overloads features_length as the 512-wide codebook feature input and embedding_length as the 1282-wide output, while 768-wide PosNet and ConvNeXt stacks provide the hidden width. The GGUF graph stops before the required magnitude/phase reconstruction and ISTFT processor, so mapping it to CodecTask would falsely promise waveform output. Dedicated graph, processor, quantization guards, and independent F16/Q5_1 parity remain deferred; standalone runtime packaging is refused before graph construction. |
| `xverse` | — | model=`xverse`; tensor=`llama` | audited-direct-loader-conditional-union | config=supported; tensor_map=supported; graph=supported; runtime=deferred; quantized_import=rejected | Config extraction, exact tensor-name closure, and a full synthetic GGUF graph build are covered, but no representative real-weight GGUF has yet passed ORT parity or generation validation. Runtime packaging remains deferred until that evidence exists. Fused QKV input is rejected because it cannot be combined truthfully with the required architecture-specific Q/K row permutations. Quantization preservation is also rejected for this architecture. |

<!-- END GGUF SUPPORT MATRIX -->

Canonical names are the strings llama.cpp writes into `general.architecture`,
validated against a vendored census of the 147 architectures llama.cpp defines
at commit `8d9af256337d1a501250f9bbf4c0859a654bddd6`. Aliases are spellings
llama.cpp does not emit but that mobius still accepts.

### Final census closure

The registry now resolves all 147 pinned real architecture IDs exactly once.
The final 50 IDs are explicit pre-config verdicts rather than generic-family
fallbacks:

- Dense, legacy, and embedding: `bitnet`, `codeshell`, `command-r`,
  `gemma-embedding`, `gptneox`, `jais`, `jais2`, `maincoder`, `nanbeige`,
  `orion`, `pangu-embedded`, `plamo`, `plamo3`, `plm`, `qwen`, `starcoder`,
  and `xverse`.
- Dense-attention and routed-expert: `afmoe`, `bailingmoe`, `deepseek`,
  `dots1`, `ernie4_5`, `ernie4_5-moe`, `granite`, `granite_swa`,
  `graniteswitch`, `hunyuan-moe`, `laguna`, `llama-embed`, `mellum`,
  `minicpm`, `minimax-m2`, `minimax-m3`, and `refact`.
- NextN, MTP, and target-coupled: `bailingmoe2`, `cohere2moe`, `deepseek2`,
  `deepseek32`, `dots3note`, `exaone-moe`, `exaone4`, `gemma4-assistant`,
  `glm-dsa`, `glm4`, `glm4moe`, `hy_v3`, `mimo2`, `mistral4`, and `step35`.
- `gptj` is rejected because the pinned architecture ID has no llama.cpp
  model loader.

No ID in this cohort is a fabricated alias. Their pinned loader tensor union,
converter `MODEL_TENSORS` family inventory, and required/defaulted
architecture metadata are recorded in `llamacpp_pin.json`. Every loadable ID
is deferred before config extraction because Mobius lacks at least one exact
graph, task/cache ABI, tensor transform, or quantization contract. Runtime is
also deferred: none has pinned real-artifact provenance, independent
full-logit/embedding parity, and deterministic generation or stateful-output
evidence.

The four RWKV IDs remain distinct. `rwkv6` and `rwkv7` carry two token-shift
vectors plus one per-head F32 matrix state per layer; `rwkv6qwen2` and
`arwkv7` use one shift and Qwen-style FFNs around different RWKV recurrences.
Mobius has no RWKV state task, bounded rollback/reorder ABI, WKV6 kernel, or
RWKV7 generalized delta-rule kernel. Generic CausalLM/KV-cache and Mamba
graphs are therefore never selected.

### Dense C01 cohort

The pinned llama.cpp `8d9af256337d1a501250f9bbf4c0859a654bddd6` dense cohort
adds bounded graph import for `apertus`, 32-layer `baichuan`, modern `chatglm`,
`phi2`, and 64-layer `seed_oss`. Runtime remains deferred until a pinned real
GGUF has independent full-logit and generation parity.

- Baichuan accepts only the 7B RoPE graph, reverses the converter Q/K permutation,
  and rejects the 40-layer hardcoded-ALiBi path. Phi-2 requires its complete bias
  closure, 4H tanh-approximated GELU MLP, full MHA, and an untied output.
- ChatGLM accepts only contiguous fused QKV and fused gate/up tensors. Quantized
  fused forms are rejected because splitting their packed values and sidecars is
  not yet losslessly covered. Seed-OSS maps `post_attention_norm` exactly and permits
  either an explicit output or effective ownership by the token embedding, and
  accepts Q/K/V biases only as a complete all-layer family.
- Apertus consumes serialized Llama-3 `rope_freqs` or LongRoPE short/long
  per-dimension factors exactly, applies Q/K RMSNorm before RoPE, owns xIELU
  metadata values as graph initializers, and maps optional Q/K norm and attention
  output biases. Quantized projections remain packed; quantized RoPE factors are
  rejected.
- `minicpm3`, `openelm`, and `mpt` remain explicit pre-config deferrals. Their
  MLA/scaling topology, per-layer fused/tied topology, and optional learned
  positions/QK norms/clipping/AWQ/bias closure are not represented by current
  Mobius graphs.

### Second hybrid cohort

`jamba`, `nemotron_h`, and `granitehybrid` have graph-import support for exact
pinned subsets. All three include their routed-MoE forms; GraniteHybrid selects
dense or routed feed-forward globally and may add one always-active shared
expert on every layer. Runtime packaging remains deferred pending independent
real-artifact full-logit and stateful-generation parity.

- Schedules come from suffix-exact per-layer metadata. Jamba and GraniteHybrid
  use `attention.head_count_kv` (`0` selects Mamba/Mamba2). Nemotron-H combines
  that array with per-layer `feed_forward_length` to select exactly one of
  Mamba2, attention, or dense ReLU² MLP.
- Jamba requires `ssm.inner_size == 2 * embedding_length`, SiLU experts,
  Mamba-1 with biased depthwise convolution and bias-free projections, and
  softmax-first top-k routing without post-top-k renormalization. Routed layers
  are inferred exactly from `ffn_gate_inp`; stacked expert tensors are split in
  numeric order. There are no shared experts. Nemotron-H supports its exact
  mixed dense/MoE schedule, correction-biased sigmoid routing, shared expert,
  and optional latent projections while rejecting unsupported MTP blocks.
  GraniteHybrid uses normalized softmax top-k routing on every layer, fuses
  separate GGUF expert gate/up tensors in gate-then-up order, and adds the
  optional shared SwiGLU branch.
- Every layer must provide exactly its pinned loader tensor family. Missing,
  wrong-mixer, partial, auxiliary, scale/input-scale, and out-of-range tensors
  are rejected before graph construction. Compatible attention, dense-FFN, and
  expert MatMul weights may remain quantized where the graph owns an exact
  packed ABI. GraniteHybrid and Nemotron-H quantized sources require explicit
  dequantization; neither silently drops or rewrites packed expert semantics.
  GGUF Mamba decay values are inverted
  from `-exp(A_log)`; convolution and grouped Mamba2 tensors are restored to
  graph shapes.
- State inputs and outputs are caller-owned. Mamba1 uses conv
  `[B, d_inner, conv_kernel-1]` and SSM `[B, d_inner, d_state]`; Mamba2 uses
  conv `[B, d_inner + 2*groups*d_state, conv_kernel-1]` and SSM
  `[B, heads, d_state, head_dim]`. Rollback restores a previously saved complete
  state tuple; reorder gathers every state on the batch axis. This does not
  claim llama.cpp's in-place snapshot/copy-on-write rollback manager.
- `static_cache=True` and non-hybrid task dispatch are rejected for these mixed
  state ABIs.

`minimax-01` graph import supports its exact pinned Lightning schedule,
decay/scaling semantics, recurrent state, and mixed full-attention cache.
Runtime packaging remains deferred because the released schema cannot represent
that heterogeneous state ABI or bounded rollback snapshots; this is tracked by
[`onnxruntime/mobius#605`](https://github.com/onnxruntime/mobius/issues/605).
PLaMo2 has a dedicated alternating Mamba1/attention graph and strict GGUF
tensor closure. Its mixed per-layer recurrent/KV runtime package remains
deferred to [`onnxruntime/mobius#605`](https://github.com/onnxruntime/mobius/issues/605).

Falcon-H1 graph/import support is dedicated rather than aliased to Falcon. Every
layer exposes `(key, value, conv_state, ssm_state)` in that order. Static cache is
rejected, and quantized-source imports require `--dequantize` until the mixed graph
can preserve only exact attention/FFN MatMul roles. Runtime packaging remains
deferred pending the heterogeneous-state schema tracked by
[`onnxruntime/mobius#605`](https://github.com/onnxruntime/mobius/issues/605) and
real full-logit plus deterministic stateful-generation evidence.

`nemotron_h_moe` backbone import is supported with exact mixed Mamba2,
attention, dense-MLP, and routed-MoE scheduling. Files with a folded MTP block
still fail before graph construction because no released package contract
represents that auxiliary attention+MoE head.

### Remaining hybrid cohort

`bailingmoe3` and `deepseek4` are explicitly
deferred before config extraction. Their loader inventories are
suffix-exact in the pinned census, but none has an exact Mobius graph plus state
task. No aliases are accepted.

- BailingMoE3 mixes KDA recurrence with MLA, gated attention,
  correction-biased routed/shared SwiGLU experts, and optional single-layer
  NextN storage. Its exact graph and state task remain deferred.
- Kimi-K3 now has a dedicated text graph and task. It preserves Q-LoRA gated
  NoPE MLA, lower-bounded KDA decay with full-rank output gates, cross-layer
  attention residuals, and Stable LatentMoE with SiTU. Each recurrent layer
  carries three Q/K/V convolution histories and one FP32 matrix state; static
  cache and generic OGA packaging remain deferred under #605.
- Kimi-Linear now has a dedicated graph and task. The importer reconstructs the
  exact per-layer KDA/NoPE-MLA schedule from `attention.head_count_kv`, validates
  dense-versus-MoE closure, restores `A_log`, convolution, and split MLA K/V-B
  layouts, and exposes three convolution histories plus one FP32 recurrent matrix
  for each KDA layer. Static cache and generic OGA packaging remain unsupported
  because released decoder cache schemas cannot represent that heterogeneous ABI.
- DeepSeek-V4 requires raw sliding-window K, ratio-4 CSA, ratio-128 HCA, and
  indexer compressed caches plus persistent compressor and rollback state.
  Compression ratios are per-layer and limited to `0`, `4`, or `128`; ratio 4
  alone owns indexer-compressor tensors. Hash-routed prefix layers and later
  sqrt-softplus/correction-bias layers share neither selection inputs nor cache
  behavior. The Mobius Hugging Face model intentionally exports a dense fallback
  with ordinary KV state, so it remains valid for `mobius build` but cannot be
  used for `general.architecture=deepseek4`.
- LFM2MoE serializes its arbitrary attention/short-convolution schedule as the
  complete `attention.head_count_kv` array (`0` means convolution). It switches
  from dense SwiGLU to sigmoid, correction-biased, normalized routed experts at
  `leading_dense_block_count`. Recurrent state is F32 rolling convolution
  history with copy-on-write reorder and bounded rollback snapshots; it is not
  ordinary KV or the dense LFM2 task.
- The pinned GGUF closures use separate gate/up/down expert tensors. They do not
  contain emitted `.scale` or `.input_scale` sidecars: DeepSeek-V4 and Kimi-K3
  converter input scales are consumed while forming MXFP4 expert weights.
  Recurrent coefficients, convolution kernels, biases, norms, hash tables, and
  residual/compressor sidecars are not mathematically interchangeable with
  MatMul weights and must never be silently preserved or dropped.
- Partial NextN/MTP, fused gate-up experts, mixed tied output storage, unknown
  mixer tensors, and auxiliary heads remain outside the admitted contract.
  Runtime status cannot advance without a pinned real artifact, independent
  full-logit parity, and deterministic stateful generation evidence.

### Audio/TTS/codec cohort

The four C09 architectures below were audited against llama.cpp commit
`8d9af256337d1a501250f9bbf4c0859a654bddd6`. Their converter-produced primary
tensor inventories are pinned in `llamacpp_pin.json`; being inventoried does not
mean the graph is supported.

| Architecture | Exact component and I/O | Pinned closure | Verdict |
|---|---|---:|---|
| `pockettts` | Causal CALM backbone: conditioned token/latent rows plus KV state → hidden state. It has no semantic LM head. The required mmproj performs 24-kHz voice encoding, continuous 32-D flow generation at 12.5 Hz, EOS scoring, and stateful Mimi waveform decoding. | `3 + 10L` primary tensors (`243` for the 24-layer variant); no codebook tensors. | Deferred. Current standalone use is refused before graph construction because omitting `pockettts_spkenc` and `pockettts_gen` would expose the wrong contract. |
| `qwen3tts` | Transformed causal talker: combined text+codec token IDs plus KV state → a 3072-row codec head shifted into the combined vocabulary. The required mmproj owns speaker conditioning, the remaining 15 codebooks, and stateful 24-kHz code-to-wave decoding. | `3 + 11L` primary tensors (`311` for the 28-layer 1.7B talker). | Deferred. Current standalone use is refused before graph construction; it is neither `Qwen3TTSForConditionalGeneration` nor `Qwen3TTSTokenizerV2Model`. |
| `talkie` | Text token IDs plus KV state → text logits. This is not an audio model: it uses weight-free RMSNorm, post-RoPE Q/K normalization, query/attention/MLP gains, and an embedding skip in every block. | `2 + 9L` C++ loader-created tensors (`362` for 40 layers), plus `2L` converter-only `.scale` sidecars (`80`; emitted-file union `442`). | Deferred pending a dedicated graph, Q/K sign transform, exact gain-sidecar consumption, quantization guards, and independent logit/state/generation parity. |
| `wavtokenizer-dec` | One non-causal 4096-entry code stream → `T × 1282` magnitude/phase parameters. The GGUF does **not** emit waveform audio; the missing processor must split 641-bin log-magnitude/phase and perform the exact 24-kHz, hop-320 ISTFT. No recurrent/cache state exists. | `161` tensors for the exact six-layer PosNet and 12-layer ConvNeXt graph. | Deferred. Current use is refused before graph construction until a dedicated feature-decoder task and exact ISTFT processor exist. |

The pinned loader/converter metadata contract is:

| Architecture | Required/operational metadata | Optional defaults and converter transforms |
|---|---|---|
| `pockettts` | `context_length`, `embedding_length`, `block_count`, `feed_forward_length`, `attention.head_count`, and `attention.layer_norm_epsilon`. | KV heads default to attention heads, causal attention defaults true, and RoPE base defaults to 10000. The converter derives dimensions from checkpoint tensor shapes, assumes head size 64, folds learned audio-BOS conditioning into `token_embd.weight`, and fixes the cache bound at 4096. |
| `qwen3tts` | Generic decoder geometry, `attention.layer_norm_rms_epsilon`, and the required four-entry `rope.dimension_sections`. | `n_deepstack_layers` defaults to zero. The converter folds the text projection MLP into the text embedding rows, appends codec embeddings, offsets codec token IDs by text vocabulary size, writes the shifted EOS ID, and suppresses non-semantic codec rows. |
| `talkie` | Generic decoder geometry, `attention.layer_norm_rms_epsilon`, and `logit_scale`. | KV heads default to attention heads, head lengths default to `embedding_length / head_count`, causal attention defaults true, and RoPE base defaults to 10000 (the released model writes 1000000). The converter applies Talkie's half-RoPE Q/K sign transform and emits `attn_output.scale`/`ffn_down.scale` from learned gains. Those 80 tensors are converter extras, not model-specific C++ creation sites. llama.cpp may tolerate them through generic weight-scale handling, but Mobius will not silently drop them: every Talkie import remains deferred until a graph explicitly consumes or rejects their exact semantics. |
| `wavtokenizer-dec` | `context_length`, `embedding_length`, `features_length`, `block_count`, `feed_forward_length`, both PosNet/ConvNeXt widths and counts, layer-norm epsilon, group-norm epsilon/groups, and `vocab_size`. | `attention.causal=false`; no KV or recurrent state. Training-only codebook EMA tensors are intentionally excluded, but the single 4096×512 embedding codebook is required. |

`wavtokenizer-dec` deliberately overloads standard metadata. `features_length`
is the 512-wide codebook feature input, `embedding_length` is the 1282-wide
output (not the hidden width), and both `posnet.embedding_length` and
`convnext.embedding_length` are the 768-wide hidden size. Exact files require
`posnet.block_count=6`, `convnext.block_count=12`, matching hidden widths,
non-causal centered kernel-3/kernel-7 convolutions, group-norm epsilon/groups,
and `feed_forward_length=2304`. Treating `embedding_length` as the model hidden
size is rejected as an ambiguous field interpretation.

Quantized linear matrices may only use a supported `MatMulNBits` path after
their architecture-specific transforms. Convolution kernels, codebooks,
normalization parameters, biases, gamma, and `.scale`/`.input_scale` sidecars
must be dequantized or rejected; they are never silently routed through
quantized MatMul or dropped. In particular, the pinned Qwen3-TTS and PocketTTS
converters force convolution families to F16/F32 according to the llama.cpp
consumer ABI, and WavTokenizer's available Q5_1 artifact needs float convolution
handling. No C09 runtime is marked supported because no pinned artifact has
independent reference waveform/feature or full-logit parity.

The smallest audited real files are in `ggml-org/WavTokenizer` at revision
`0c97fdc098158ec9bf4e703cd5f81a5aa20520e6`:

| File | Size | LFS SHA-256 | Census |
|---|---:|---|---|
| `WavTokenizer-Large-75-F16.gguf` | 130,186,688 bytes | `2356baa8631cc2995ea3465196a017a2733600d849a91180c0f97fa7fb375bbe` | 161 tensors; float reference candidate |
| `WavTokenizer-Large-75-Q5_1.gguf` | 73,319,616 bytes | `ad182c884841444ce6b70e8a61a7d084d9731320364dc633c9ef42632fc63d25` | Same graph; quantized matrix candidate |

That repository contains no independent processor/config bundle or reference
feature/waveform vectors. The source WavTokenizer configuration supplies the
24-kHz/75-Hz/hop-320 ISTFT semantics, but neither artifact has reproducible
PyTorch↔llama.cpp↔ORT parity evidence; both therefore remain evidence candidates,
not supported runtime fixtures. PocketTTS is gated, and the available Qwen3-TTS
and Talkie checkpoints are not reasonably small cohort fixtures.

### Speculative draft GGUF contract

`dflash` and `eagle3` are auxiliary speculative-decoding models, never
standalone causal language models. Both the API and CLI require an explicit
`target_config`; `task="text-generation"`, static cache, and standalone runtime
packaging are rejected. A successful build emits `draft_manifest.json` beside
the ONNX graph. The manifest records canonical JSON SHA-256 values for the
target config and complete tokenizer, a separate ordered-token vocabulary
SHA-256, hidden/layer/vocabulary sizes, selected target layers, draft
depth/head/intermediate/block sizes, output ownership, and any draft-to-target
(`d2t`) vocabulary map. Canonical hashes are identical for equivalent file and
mapping inputs and remain stable when a target directory is relocated; absolute
host paths are not embedded.

The target tokenizer vocabulary must match the GGUF token list in exact ID
order, and BOS/EOS/PAD IDs are checked when both sides declare them. This proves
the token-ID contract, not equivalence of tokenizer normalization, pre-tokenizer,
or merge behavior. The manifest identifies the exact config file or explicit
mapping used for validation, but Mobius cannot derive a target weight revision
from GGUF metadata. The caller must pair that manifest with the attested target
weights; runtime support therefore remains **deferred**.

`config.json` and `tokenizer.json` are size-bounded, schema-checked UTF-8 JSON
objects. Both resolved resources must remain in one target root; escaping or
mixed-directory symlinks are rejected. Split `vocab.json`/`merges.txt` or
`tokenizer.model` alternatives are not reconstructed because doing so would
lose full normalizer, pre-tokenizer, and added-token semantics.

EAGLE-3 requires exactly three valid `eagle3.target_layers` and matching
`eagle3.target_hidden_size`; target and draft hidden widths must currently match
so target-owned embeddings and an optional shared target head have the correct
width. Its graph accepts target-provided token embeddings,
the three concatenated target hidden states, recycled draft hidden state, and
draft KV cache. DFlash validates every `dflash.target_layers` index and positive
`dflash.block_size`; its graph accepts target hidden-state concatenation, noise
embeddings, separate Q/K position IDs, and draft KV cache. Pinned DFlash
layer-input indices are normalized back to zero-based target decoder-output
indices. DFlash sliding-window schedules are rejected until the graph can model
their per-layer bounded-cache semantics. DeepSeek-V4/DSpark DFlash tensor
families likewise fail the pinned Qwen-style suffix closure; they require a
different graph and are outside this cohort. Draft-owned
`token_embd.weight` is rejected rather than duplicated or ignored because these
graphs deliberately consume target-provided embeddings.

When `output.weight` is absent, the graph returns `draft_hidden` for the exact
target LM head. When present, it returns compact `draft_logits`. If `d2t` is
present, it must use the pinned I64 representation, be in bounds and unique,
and be accompanied by the
draft-owned output projection; the orchestrator takes argmax in draft-vocabulary
space and maps the proposed ID through `draft_to_target`. Unmapped target IDs
simply have no draft row. Supported quantized draft projections retain the
existing MatMulNBits/native route. Integer remaps are never dequantized, and
`.scale`/`.input_scale` sidecars are rejected when their semantics cannot be
represented instead of being dropped.

Pinned public artifacts were inspected as availability evidence, not runtime
parity:

| Architecture | Draft GGUF revision / file | Size / LFS SHA-256 | Exact target |
|---|---|---|---|
| `dflash` | `lym00/Qwen3-4B-DFlash-GGUF-Test@9d2ec464a15346d8a7d7a696c06694eb1bf690b5` / `Qwen3-4B-DFlash-q8_0.gguf` | 577,047,072 bytes / `5ecf02bb269fc42277f43961794387fea11ecc367ea2d99e86b2b71cc249aff6` | `Qwen/Qwen3-4B@1cfa9a7208912126459214e8b04321603b3df60c` |
| `eagle3` | `williamliao/Qwen3-8B-EAGLE3-Speculator-GGUF@44480ff4ea6330788818f7f5fc9a69b326dc4c06` / `Qwen3-8B-speculator.eagle3-F16.gguf` | 2,049,930,400 bytes / `d6cf1f3cf29e9cd72c02fb11f989f5192f2b24e142741fdc2de8cd590140f2f2` | `Qwen/Qwen3-8B@b968826d9c46dd6066d109eabc6255188de91218` |

The DFlash header has 58 tensors (36 Q8_0, 22 F32); it uses the post-pin
`hidden_norm.weight` spelling and is intentionally rejected by this cohort's
suffix closure. The EAGLE-3 header has 15 tensors (10 F16, 4 F32, 1 I64),
including a 32,000-entry `d2t` and a draft-owned target embedding, which this
external-embedding graph also rejects. No suitably small pinned-format pair was
found. Neither payload was executed, and independent target/draft logits plus
multi-step proposed-token parity is still required before runtime can become
supported.

### Masked language-diffusion GGUF contract

`dream`, `llada`, `llada-moe`, and `rnd1` are full-sequence, bidirectional
masked-token predictors. Their neural graph has one `input_ids [batch, sequence]`
input and returns `logits [batch, sequence, vocabulary]` plus
`proposed_tokens [batch, sequence]`. It has no causal mask, timestep/noise input,
or KV cache. The caller owns the initial mask, mask-token insertion, seed,
iteration count, confidence policy, and progressive remasking/commit schedule;
Mobius does not infer a diffusion schedule from the GGUF.

`tokenizer.ggml.mask_token_id` is therefore required and range-checked. The
llama.cpp `diffusion.shift_logits` default is preserved (`true` for Dream/RND1,
explicitly `false` for the pinned LLaDA files). A causal task override or static
cache is rejected before graph construction. Dense LLaDA alone reverses the
llama.cpp interleaved-RoPE Q/K row permutation; Dream, LLaDA-MoE, and RND1 keep
their Q/K rows in converter order. LLaDA-MoE retains raw selected softmax router
weights, while RND1 renormalizes the selected weights. Fused QKV, when admitted
by the pinned loader, is split as `[Q heads, KV heads, KV heads]` on the
float-only path.

Compatible 2-D projections and routed experts retain supported native/affine
quantization. Embeddings, norms, routers, and auxiliary tensors follow their
actual consumer ABI; unrecognized sidecars, neural timestep tensors, and noise
schedule tensors fail suffix-exact closure instead of being dropped.
A fused QKV tensor is rejected whenever any mapped tensor activates
quantization preservation, including a float fused tensor in an otherwise
quantized file. The packed diffusion graph owns separate Q/K/V targets, so
post-load splitting would leave those targets uninitialized. Use
`keep_quantized=False` or `--dequantize` for such files; split-QKV quantized
files remain supported. The pinned diffusion MoE family uses separate stacked
gate, up, and down expert tensors rather than a fused gate-up tensor, and those
tensors map directly to the graph's expert targets.

Runtime remains **deferred**. These immutable artifacts were pinned before
payload download; only the 88.8 MB LLaDA file was downloaded. Its SHA-256,
57-tensor census, mask token `126336`, `diffusion.shift_logits=false`, and
quantized CPU masked-forward smoke test were verified. A successful import is
not parity evidence.

| Architecture | GGUF revision / file | Size / LFS SHA-256 | Source config/tokenizer |
|---|---|---|---|
| `dream` | `mradermacher/Dream-v0-Base-7B-GGUF@8145ed37262d0d5769efefd33e156cd2ef98f4b2` / `Dream-v0-Base-7B.Q2_K.gguf` | 3,015,940,512 bytes / `c28476c7e7b0ea4e00e93f3b456f5e2e9b589f4200f29a975f845b8b9e5b0012` | `Dream-org/Dream-v0-Base-7B@6572adb5535263e4d1a337b56942ba48b6dee2a9` |
| `llada` | `mradermacher/LLaDA-1.5-Tiny-GGUF@752094b7115a2aa5097b6be66187b19a46ff97dc` / `LLaDA-1.5-Tiny.Q2_K.gguf` | 88,765,952 bytes / `31c5fd2c1fc6bcd4e1d8b605774759252c130977562973d721a98c1d810b50a2` | Bundled tokenizer/config metadata verified; `JakeOh/LLaDA-1.5-Tiny` source revision is access-restricted. |
| `llada-moe` | `mradermacher/LLaDA-MoE-7B-A1B-Instruct-GGUF@1080e16761f6f82a92e8bfb54a4c8998dfee0219` / `LLaDA-MoE-7B-A1B-Instruct.Q8_0.gguf` | 7,829,549,248 bytes / `7e50c4764866b64aba502d2b1e98fe20649c76f70df4b10fb9d2ece7c04fd2fd` | `inclusionAI/LLaDA-MoE-7B-A1B-Instruct@783d3467f108d28ac0a78d3e41af16ab05cabd8d` |
| `rnd1` | `vikramkr/RND1-Base-0910-Q8_0-GGUF@93d8d35e3aac7b39311e48fc778777ac21529057` / `rnd1-base-0910-q8_0.gguf` | 32,483,927,584 bytes / `689949d9661c88930045c3511f893cacb41068c5133c43dcd388570abd1af28b` | `radicalnumerics/RND1-Base-0910@f1b49afd26579c1cd4ef7e00ae88376de63f2878` |

The support matrix will remain deferred until at least one artifact for each
architecture passes independent Hugging Face or llama.cpp masked-step logits
and deterministic multi-step generation parity with identical mask, seed, and
remasking policy. The three multi-gigabyte artifacts above were not downloaded.

### Encoder-only GGUF contract

`bert` and `modern-bert` import only backbone files whose metadata requests
non-causal, unpooled token embeddings. They produce one `model` component with
`input_ids`, `attention_mask`, and `token_type_ids` inputs and one
`last_hidden_state` output shaped `[batch, sequence, hidden]`. They never emit
causal logits, KV cache, or recurrent state. `task` overrides other than
`feature-extraction`, static cache, non-`NONE` pooling, classifier labels,
`cls*` tensors, and ModernBERT sliding-window attention are rejected rather than
ignored.

Pinned BERT files may use either split Q/K/V tensors or float fused
`attn_qkv.{weight,bias}` tensors; fused rows are split losslessly before weight
application. Quantized fused QKV is rejected because splitting packed blocks is
not lossless. For both graph-importable encoders, `attention.head_count_kv` defaults to
`attention.head_count`; different values are rejected until the graphs support
GQA-sized K/V.

Compatible 2-D encoder projection matrices use `MatMulNBits`, while a quantized
`token_embd.weight` explicitly dequantizes: encoder modules do not yet expose the
`QuantizedEmbedding`/`GatherBlockQuantized` ABI. This is intentionally not a
packed-embedding preservation claim. Token-position, token-type, normalization,
and bias tensors remain float; GGUF
`.scale`/`.input_scale` sidecars are rejected before graph construction.

Runtime remains **deferred**. Header-range inspection pinned these representative
artifacts before any payload use:

| GGUF artifact | Revision / file | Size / LFS SHA-256 | Header evidence |
|---|---|---|---|
| `PierreMesure/sentence-bert-swedish-cased-gguf` | `f737e1c3fa76176845a1a9dfc3fd8aee82e8f159` / `sentence-bert-swedish-cased.F32.gguf` | 497,449,376 bytes / `9ee1505b22d4bc8d192095f924ddb62bc4783a48fbd411252310933e879930f8` | `bert`, 12x768, F32 closure; rejected because `pooling_type=MEAN`. Source config/tokenizer: `KBLab/sentence-bert-swedish-cased@6b5e83cd29c03729cfdc33d13b1423399b0efb5c`. |
| `keisuke-miyako/modernbert-ja-30m-gguf-q8_0` | `b51053917455553a61b5319560dbd00b7d53323d` / `modernbert-ja-30m-Q8_0.gguf` | 41,466,080 bytes / `b56a63b0adc5942dbc949153b071bd90557380fa32ca190cc833e85fb1529932` | `modern-bert`, 10x256, Q8_0 matrices/F32 norms; rejected because it declares symmetric sliding-window attention. Source config/tokenizer: `sbintuitions/modernbert-ja-30m@8cb03f54cb9e30e72459e5f1cedc6d89c7d8dcb5`. |

Neither artifact is runtime evidence: both are deliberately rejected before
weight loading, so no independent embedding or reranker-logit parity is claimed.
EuroBERT, JinaBERT v2/v3, NeoBERT, NomicBERT, and NomicBERT-MoE remain deferred
because their pinned normalization, ALiBi/RoPE, gated FFN, Q/K norm, or routed
expert semantics do not match the existing encoder graphs.

### T5 and T5-encoder GGUF contract

`t5` produces separate `encoder` and `decoder` components with masked encoder
states, decoder self-attention cache, and cross-attention cache. The same decoder
graph handles prefill (empty cross cache plus encoder states) and decode (populated
cross cache plus zero-length encoder states). `t5encoder` produces only a `model`
component with `input_ids` and `attention_mask` inputs and a
`last_hidden_state` output; it has no decoder, logits, or cache.

The importer preserves unequal encoder/decoder layer counts, head width,
feed-forward width, RMS-norm epsilon, relative-position buckets, decoder start,
EOS, and padding token IDs. Layer 0 must own each stack's relative-attention bias;
later per-layer overrides remain attached to their source layer. Non-gated files
use ReLU. Gated files are rejected as ambiguous: pinned llama.cpp executes any
gate tensor through tanh-approximate `LLM_FFN_GELU`/`ggml_geglu_split`, but its
converter does not serialize `feed_forward_proj` or `dense_act_fn`, so identical
GGUF metadata and tensor shapes can originate from `gated-gelu`, `gated-silu`,
or another gated activation. Mixed gated/non-gated layers, GQA, unequal key/value
widths, fused projections, projection biases, malformed suffixes, and
`.scale`/`.input_scale` sidecars are rejected before graph construction.

Compatible quantized 2-D projections use `MatMulNBits`. Shared token embeddings
explicitly dequantize to `Gather`, and relative bias, norms, and other small
tensors remain float. A `t5encoder` `output.weight` and decoder
`cross_attn_rel_b.weight` are accepted only with an explicit warning because the
pinned llama.cpp encoder output and cross-attention graph do not consume them.

Runtime remains **deferred**. The real-file audit pinned these representatives
before payload download:

| Architecture | GGUF revision and file | Size | LFS SHA-256 | Header evidence |
|---|---|---:|---|---|
| T5 | `noumenalabs/t5-small-gguf@222e7698299802b6a592054305063f22759aed0f`<br>`t5-small-f16.gguf` | 122,074,752 | `4331d3b568593e17d0de10c8755705256c2912af4338ecace92fcef0122da646` | 131-tensor full encoder/decoder census; source config/tokenizer `google-t5/t5-small@df1b051c49625cf57a3d0d8d3863ed4d13564fe4` |
| T5 encoder | `chatpig/t5-base-encoder-gguf@1d307eff4d9de02b9f74cff9a9928187606040ee`<br>`t5base-encoder-q8_0.gguf` | 117,563,552 | `36067e77cb097a99b0a1d47b4e49525c6a0c6c845abb9d22d5798b2633880b54` | 99-tensor encoder census; candidate source config/tokenizer `google-t5/t5-base@a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1`, but conversion provenance is weaker |

Synthetic float and quantized tests cover full hidden states/logits, padding
masks, save/load, prefill, cached decode, and packed-versus-explicit-dequantized
parity. They are not independent HuggingFace parity. Full-logit and generated-token
parity are still required for T5, and independently sourced full encoder-hidden
parity is still required for T5 encoder, before either runtime verdict can become
supported.

### Hybrid attention/recurrent GGUF contract

`lfm2`, `qwen35`, `qwen35moe`, and `qwen3next` use a mixed-state ABI. Full
attention layers alone expose `past_key_values.N.key/value`; LFM2 convolution
layers expose one `conv_state`, and gated-DeltaNet layers expose
`conv_state` plus `recurrent_state`. Inputs and matching `present.N.*` outputs
are ordered by decoder layer index. Generic/static KV cache tasks are rejected.

The schedule is reconstructed exactly as pinned llama.cpp does. LFM2 requires a
per-layer `attention.head_count_kv` array (`0` means short convolution).
Qwen hybrids prefer the explicit `attention.recurrent_layers` array; otherwise
they use `full_attention_interval`, whose upstream default is 4. Appended MTP
blocks are excluded from the trunk and must be full attention. Invalid lengths,
zero intervals, recurrent MTP entries, wrong-mixer tensors, extra layers, and
incomplete fused/legacy representations fail before graph construction.

DeltaNet key width is `ssm.state_size`; value-head width is
`ssm.inner_size / ssm.time_step_rank`. The importer validates positive,
divisible head/group/state/conv dimensions and preserves Qwen3.5's M-RoPE
sections. Qwen3.5 reverses llama.cpp's V-head tiling; Qwen3-Next deliberately
does not. No architecture in this cohort applies a Llama Q/K rotary
permutation. Fused experts are gate-then-up in ascending expert order.
Quantization scale/input-scale sidecars are rejected rather than dropped when
Mobius cannot express them, and recurrent convolution/state roles use the float
path.

Pinned pre-download artifact census:

| Architecture | GGUF revision and file | Size / LFS SHA-256 | Header census |
|---|---|---|---|
| LFM2 | `LiquidAI/LFM2.5-350M-GGUF@d86ad5aad24b8bd87a7c4821439e63e7ba589bc3` / `LFM2.5-350M-Q4_K_M.gguf` | 229,312,224 bytes / `7e6f72643caafc9a68256686638c4d7916f2cec76d1df478d4c3ddcd95a6aed4` | Bundled tokenizer/config metadata; 148 tensors, 16 layers, KV schedule `[0,0,8,0,0,8,0,0,8,0,8,0,8,0,8,0]`; F32/Q4_K/Q6_K |
| Qwen3.5 dense | `ggml-org/Qwen3.5-0.8B-GGUF@8fea620810c4afa23dd6443f999a48574c1611a3` / `Qwen3.5-0.8B-Q4_0.gguf` | 563,036,064 bytes / `57d1997790d1744fba5b40a7317df71ea5e2acee28c47e78f0cce39c0703f8cf` | Bundled tokenizer/config metadata; 320 tensors, 24 layers, interval 4 with no explicit recurrent array; F32/Q4_0/Q8_0 |

No reasonably sized, immutable Qwen3.5-MoE artifact is active in the pinned
llama.cpp corpus, and official Qwen3-Next GGUFs are split 80B artifacts. All
four runtime verdicts therefore remain **deferred**. Graph/state threading is
covered synthetically, but publication still requires independent real-weight
full-logit parity and deterministic stateful generation parity. Mobius exposes
only the latest recurrent state, not llama.cpp's optional bounded rollback
snapshot planes; rollback/reorder-capable runtime packaging must remain
disabled until that ABI is represented explicitly. The pinned NextN metadata key is exactly
`{general.architecture}.nextn_predict_layers` (GGUF `UINT32`). `block_count`
includes the appended heads, whose indices are
`[block_count-nextn_predict_layers, block_count)`. Mobius represents exactly
one head and rejects larger counts rather than truncating them.

The modern tensor namespace is:

- required: `blk.N.nextn.{eh_proj,enorm,hnorm}.weight`
- optional dedicated ownership:
  `blk.N.nextn.{embed_tokens,shared_head_norm,shared_head_head}.weight`
- decoder-block weights: the architecture's ordinary attention, norm, and FFN
  tensors at the same `blk.N`

The generic loader can additionally consume `.scale` and `.input_scale`
sidecars for MTP projections. Mobius rejects those sidecars before graph
construction because its MTP quantization ABI cannot represent them.
`nextn.pre_projection.weight` and `nextn.post_projection.weight` are the
distinct legacy/global namespace of `gemma4-assistant`; they are never accepted
as Qwen-style appended-head tensors. Unknown `nextn.*`/`mtp.*`, dotted
`nextn.predict_layers` metadata, mismatched architecture namespaces, missing
required tensors, non-trailing or out-of-range block indices, recurrent MTP
markers, and mixed routed-expert forms also fail closed.

The pinned loader/converter census is closed as follows:

| Mobius verdict | Pinned architectures | Reason |
|---|---|---|
| Export supported | `qwen35` | One dense, full-attention Qwen3.5 decoder block exactly matches the existing text block |
| Rejected, executable upstream sidecar | `bailingmoe3`, `cohere2moe`, `deepseek2`, `deepseek32`, `deepseek4`, `glm-dsa`, `hy_v3`, `mimo2`, `nemotron_h_moe`, `qwen35moe`, `qwen3next`, `step35` | Routed experts, MLA/DSA/KDA, hyper-connections, compressed state, or hybrid recurrent state are not represented |
| Rejected, preserved/skipped upstream | `bailingmoe2`, `dots3note`, `exaone-moe`, `exaone4`, `glm4`, `glm4moe` | The pinned loader consumes or preserves metadata/tensors but has no executable MTP graph |
| Rejected special cases | `gemma4-assistant`, `nemotron_h`, `graniteswitch` | Standalone legacy assistant; converter-conditional MoE ownership; or non-MTP router-layer metadata re-emission |

Dense Qwen3.5 uses the target's dedicated post-final-norm `mtp_seed`; ordinary
`hidden_states.N` remains the distinct pre-final-norm layer capture. The
sidecar normalizes the next-token embedding and seed independently,
concatenates them, projects `2H -> H`, runs one full-attention Qwen3.5 decoder
layer with its own dynamic KV cache, applies the dedicated or target final
norm, and scores through a dedicated head or the target's tied/untied head.
Dedicated embeddings consume `input_ids`; shared embeddings consume
`inputs_embeds`. Static-cache exports are rejected because silently dropping
the sidecar or its independent state is invalid. Packed target embedding/head
ownership remains with the target graph; dedicated sidecar tables are
dequantized to mathematically ordinary MatMul weights, so no target/MTP
initializer is duplicated. `ModelPackage.save()` persists the sidecar under
`mtp/` with weight checks, and `ModelPackage.load()` restores it as
`package.mtp_head`; ordinary API saves cannot silently lose the auxiliary head.

**MTP runtime status: DEFERRED.** The emitted graph and workflow metadata are
an export contract, not evidence that a downstream runtime executes it.
Runtime support requires a pinned real artifact, independent full-vocabulary
logit parity, and deterministic end-to-end speculative/MTP generation with
correct proposal rejection and cache rollback. None has yet been established,
so no runtime-support claim is made.

### Pure recurrent Mamba evidence

`mamba` and `mamba2` have config, suffix-exact C++ tensor-name closure,
graph-build, save/load,
quantized-source dequantization, and recurrent prompt/decode state-threading
coverage. Mamba1 ingests prompts as single-token recurrent steps; Mamba2 also
supports multi-token prefill. Their runtime verdict remains **deferred**: no pinned real GGUF has
yet passed an independent full-logit comparison plus deterministic multi-token
generation against its source implementation.

The optional Mamba `ssm.dt_b_c_rms` metadata defaults to `false`, matching the
pinned loader. Files that explicitly set it to `true` are rejected because they
require FalconMamba's additional B/C/dt norms, which the pure Mamba graph does
not implement.

The real-file audit was completed from HTTP range metadata before any payload
download:

| Architecture | GGUF revision and file | Size | LFS SHA-256 | Header evidence |
|---|---|---:|---|---|
| Mamba | `QuantFactory/mamba-130m-hf-GGUF@f781792fcca13eb3457a00dc41674715608b02da`<br>`mamba-130m-hf-Q2_K.gguf` | 69,302,464 | `922a9c947f979024fe3675b11a9257637d2226f8953f1831a351890953a5209a` | 242 tensors: Q2_K projections, Q6_K tied embedding, F16 dt projection, F32 conv/state/norm |
| Mamba2 | `rpatel622/mamba2-130m-hf-Q8_0-GGUF@0daf70963405439f2102f6fefe92d7584e2c76eb`<br>`mamba2-130m-hf-q8_0.gguf` | 180,669,152 | `c5d91a0653794a159f19d60159c362b179b506a206be120f8b63f43013155839` | 219 tensors: Q8_0 embedding/output/projections and F32 conv/state/norm; A/D use llama.cpp's `(heads, 1)` layout |

The corresponding source configuration/tokenizer revisions are
`state-spaces/mamba-130m-hf@1e76775f628fbf1350fbe4dbb3d971ba64af25a1`
and `AntonV/mamba2-130m-hf@05e8773fc4ac1cd067e8a18a5c45372ce5178405`.
The header census covers every layer and confirms the pinned llama.cpp tensor
families and qtypes without treating graph construction as runtime parity.

`arwkv7`, `rwkv6`, `rwkv6qwen2`, and `rwkv7` are explicitly deferred because
their time-mixing/channel-mixing recurrence is a distinct architecture and
Mobius has no corresponding graph. They must not be routed through either
Mamba implementation.

GraniteMoE has real-weight import and deterministic ORT execution coverage from
`bartowski/granite-3.0-1b-a400m-instruct-GGUF` revision
`0e1c3cecaa6e49ac0721be91ef441ec72eae62d4`,
`granite-3.0-1b-a400m-instruct-Q4_K_M.gguf` (821,845,024 bytes, SHA-256
`074f09e13484e54e73c93830d34e9fa9917a6319fb8bae762a22594b9b4da0dc`).
Its base config and tokenizer are pinned to
`ibm-granite/granite-3.0-1b-a400m-instruct` revision
`ffec3c35bdfd97a06f0b4cd5fcc92cd9b1584445`. The integration test checks all
24 routers and 2,304 expert projections, mixed F32/Q4_K/Q6_K routing, full
49,155-token logits, and deterministic three-token cached decoding. This is not
independent cross-runtime parity evidence, so GraniteMoE runtime support remains
deferred.

Runtime remains deferred for OLMoE, PhiMoE, Qwen2MoE, Qwen3MoE, and GraniteMoE.
The only
compatible Phi-tiny-MoE GGUF found was
`tripathyShaswata/Phi-tiny-MoE-instruct-GGUF` revision
`873ccb08cd3380ee2c08573d45267fac9a6cc81b`,
`Phi-tiny-MoE-instruct-Q8_0.gguf` (3,999,171,104 bytes, SHA-256
`297fa09e906e18aaf03850e77d6de8d9ee8e246e00916ac09787ae2cf4bb6019`);
no trustworthy reasonably sized representative was available. Its source
config and tokenizer were inspected at `microsoft/Phi-tiny-MoE-instruct`
revision `2fe50e88d0e2a5a132563815686ea0dcc8e252b5`.

An architecture outside this table is refused with a message naming its
upstream cohort, so an unsupported input is never mistaken for a broken one.
`clip` files are multimodal projector sidecars: pass them as
`build_from_gguf(text_gguf, mmproj=...)` rather than on their own.
For a Hub companion reference, Mobius range-reads only the bounded metadata
header of the exact selected filename, verifies that file is `clip`, resolves
the repository ref to an immutable commit, and downloads that same revision.
Repository-level GGUF metadata is not used because mixed text+mmproj repositories
describe the text architecture there. The downloaded local header is validated
again before multimodal builder dispatch.

### Remaining multimodal text-backbone cohort

The pinned architecture census contains eleven additional language-model
identifiers emitted by multimodal converters: `chameleon`, `cogvlm`,
`deepseek2-ocr`, `gemma3n`, `hunyuan_vl`, `llama4`, `mistral3`, `paddleocr`,
`qwen2vl`, `qwen3vl`, and `qwen3vlmoe`. These are canonical
`general.architecture` strings; similar Hugging Face class names are not GGUF
aliases.

All eleven are explicit pre-config deferrals. Registering them makes the
architecture-level verdict actionable without claiming that an existing text or
VLM graph owns their serialized contract:

- `chameleon` is the only standalone text-GGUF case in this cohort. Its converter
  drops the VQ image tokenizer, while the remaining decoder still has
  Chameleon-specific norms and image-logit suppression. Mobius does not claim
  either complete multimodal generation or an independently validated text-only
  fallback.
- The other ten are members of paired text-plus-`clip` packages. Their companions
  select distinct projector strings and may add vision or audio streams. Text
  compatibility alone cannot establish the processor, special-token, token-count,
  position-ID, cache, target-identity, or package-role contracts.
- `qwen2vl` is shared by Qwen2-VL, Qwen2.5-VL, and Qwen2.5-Omni converter paths;
  `qwen3vlmoe` combines the VLM contract with routed experts. Neither may be
  collapsed to the ordinary Qwen2 or Qwen3-MoE registry entry.
- `hunyuan_vl` is not the Mobius `hunyuan_vl_mot` graph, and `mistral3` can select
  dense or routed-expert blocks in addition to its Pixtral companion. Those
  similarly named registrations remain valid for Hugging Face builds but are not
  reused as GGUF aliases.

The registry covers all 60 serialized projector types. In particular,
`qwen2vl_merger`, `qwen2.5vl_merger`, `qwen3vl_merger`, `pixtral`, `llama4`,
`paddleocr`, `cogvlm`, `deepseekocr`, `deepseekocr2`, `hunyuanvl`, `gemma3nv`,
and `gemma3na` remain deferred. Only the exact `gemma3`+`gemma3`,
`gemma4`+`gemma4v`, and `muse-glimmer`+`muse-glimmer` pairings are
graph-importable; their runtime verdicts remain deferred.

The vendored census records suffix-exact conditional loader unions for ten
architectures, including CogVLM visual-expert banks, Gemma3n AltUp/Laurel and
per-layer tables, dense-versus-expert Llama4/Mistral3 forms, Qwen output/classifier
heads, fused-or-split QKV alternatives, optional biases, rope factors, and
loader-consumed expert `.scale`/`.input_scale` sidecars. Custom loader pointers
such as CogVLM visual experts, fused DeepSeek-OCR experts, Gemma3n projections,
and Qwen3-VL `cls_out` remain weight-only. PaddleOCR is marked separately with
only the mechanically proven 12-family converter allowlist because its
architecture file inherits the ERNIE 4.5 loader, including an optional
attention-output bias, and does not enumerate an independent tensor closure.
Converter-only extras for the paired families are explicitly unresolved rather
than represented as an empty set; inherited and conditional converter hooks must
be evaluated before any mapping claim.

No deferred family reaches config extraction, quantization probing, graph
construction, or partial package output. Therefore no MatMul/Gather ownership,
vision convolution dequantization, tied-head fallback, or package component is
claimed for this cohort. Future paired support must emit only the standard
`decoder`, `vision_encoder`, optional `audio_encoder`, and `embedding` roles after
the full text and companion closures pass identity and compatibility preflight.

### Multimodal projector sidecars

Projector support is pinned to llama.cpp
`8d9af256337d1a501250f9bbf4c0859a654bddd6`. The enum has 62 entries:
60 serialized strings below, one unserialized `MLP_NORM` compatibility entry,
and `UNKNOWN`. `clip` has vision, audio, and generated-audio presence flags;
there is no text-encoder presence key at this pin.

`gemma3`, `gemma4v`, `muse-glimmer`, `qwen2vl_merger`, and
`qwen2.5vl_merger` currently pass metadata, suffix-exact tensor closure,
target-pairing, graph, and component-parity gates. This admits graph import,
not runtime support: all five runtime verdicts remain deferred pending proven
processor asset publication and deterministic downstream orchestration.
`gemma4a` remains
deferred: the real sidecar contains `a.pre_encode.*` tensors that the partial
mapping does not consume. A sidecar may contain that audio tower alongside a
supported `gemma4v` tower, but requesting an audio graph fails before graph
construction. Packed projector tensors are rejected; supported encoder and
projector weights must be F32, F16, or BF16, and Gemma4 clipping bounds must be
F32. Unknown `.scale`/`.input_scale` tensors and out-of-closure weights are
errors rather than silently dropped.

<!-- BEGIN GGUF MMPROJ SUPPORT MATRIX (generated; see _mmproj_registry.py) -->

| Projector string | Modality | Paired text architecture | Metadata/tensor/graph/runtime | Exactness/evidence |
|---|---|---|---|---|
| `adapter` | vision | — | metadata=deferred; tensor_map=deferred; graph=deferred; runtime=deferred | GLM-Edge adapter tensor closure and graph are not implemented. |
| `cogvlm` | vision | `cogvlm` | metadata=deferred; tensor_map=deferred; graph=deferred; runtime=deferred | CogVLM feature output differs from LLaVA and is not implemented. |
| `deepseekocr` | vision | `deepseek2-ocr` | metadata=deferred; tensor_map=deferred; graph=deferred; runtime=deferred | DeepSeek-OCR SAM/projector graph is not implemented. |
| `deepseekocr2` | vision | `deepseek2-ocr` | metadata=deferred; tensor_map=deferred; graph=deferred; runtime=deferred | DeepSeek-OCR2 SAM/projector graph is not implemented. |
| `dots3note_a` | audio | — | metadata=deferred; tensor_map=deferred; graph=deferred; runtime=deferred | Dots3Note audio graph is not implemented. |
| `dots3note_v` | vision | — | metadata=deferred; tensor_map=deferred; graph=deferred; runtime=deferred | Dots3Note vision pyramid MoE is not implemented. |
| `dots_ocr` | vision | — | metadata=deferred; tensor_map=deferred; graph=deferred; runtime=deferred | DotsOCR vision merger is not implemented. |
| `exaone4_5` | vision | — | metadata=deferred; tensor_map=deferred; graph=deferred; runtime=deferred | EXAONE 4.5 vision merger is not implemented. |
| `gemma3` | vision | `gemma3` | metadata=supported; tensor_map=supported; graph=supported; runtime=deferred | artifact pins=`gemma3-4b-f16` |
| `gemma3na` | audio | `gemma3n` | metadata=deferred; tensor_map=deferred; graph=deferred; runtime=deferred | Gemma3n audio sidecar routing is not implemented. |
| `gemma3nv` | vision | `gemma3n` | metadata=deferred; tensor_map=deferred; graph=deferred; runtime=deferred | Gemma3n vision sidecar routing is not implemented. |
| `gemma4a` | audio | — | metadata=deferred; tensor_map=deferred; graph=deferred; runtime=deferred | The sidecar carries a.pre_encode tensors that the current audio map drops, and independent Conformer parity is not established. |
| `gemma4ua` | audio | — | metadata=deferred; tensor_map=deferred; graph=deferred; runtime=deferred | Encoder-free unified Gemma4 waveform embedding is not wired to GGUF. |
| `gemma4uv` | vision | — | metadata=deferred; tensor_map=deferred; graph=deferred; runtime=deferred | Encoder-free unified Gemma4 vision sidecars use a different patch embedder contract. |
| `gemma4v` | vision | `gemma4` | metadata=supported; tensor_map=supported; graph=supported; runtime=deferred | artifact pins=`gemma4-e2b-f16` |
| `glm4v` | vision | — | metadata=deferred; tensor_map=deferred; graph=deferred; runtime=deferred | GLM4V downsampler and projector are not implemented. |
| `glma` | audio | — | metadata=deferred; tensor_map=deferred; graph=deferred; runtime=deferred | GLM audio encoder/projector is not implemented. |
| `granite4_vision` | vision | — | metadata=deferred; tensor_map=deferred; graph=deferred; runtime=deferred | Granite 4 vision sidecar graph is not implemented. |
| `granite_speech` | audio | — | metadata=deferred; tensor_map=deferred; graph=deferred; runtime=deferred | Granite Speech audio encoder/projector is not implemented. |
| `hunyuanvl` | vision | `hunyuan_vl` | metadata=deferred; tensor_map=deferred; graph=deferred; runtime=deferred | HunyuanVL vision/projector graph is not implemented. |
| `idefics3` | vision | — | metadata=deferred; tensor_map=deferred; graph=deferred; runtime=deferred | Idefics3 pixel-shuffle projector GGUF routing is not implemented. |
| `internvl` | vision | — | metadata=deferred; tensor_map=deferred; graph=deferred; runtime=deferred | InternVL pixel-shuffle token ordering is not implemented for GGUF. |
| `janus_pro` | vision | — | metadata=deferred; tensor_map=deferred; graph=deferred; runtime=deferred | Janus-Pro vision/projector graph is not implemented. |
| `kimik25` | vision | — | metadata=deferred; tensor_map=deferred; graph=deferred; runtime=deferred | Kimi K2.5 vision/projector graph is not implemented. |
| `kimivl` | vision | — | metadata=deferred; tensor_map=deferred; graph=deferred; runtime=deferred | Kimi-VL vision/projector graph is not implemented. |
| `ldp` | vision | — | metadata=deferred; tensor_map=deferred; graph=deferred; runtime=deferred | MobileVLM LDP convolutional projector semantics are not implemented. |
| `ldpv2` | vision | — | metadata=deferred; tensor_map=deferred; graph=deferred; runtime=deferred | MobileVLM LDPv2 pooling/projector semantics are not implemented. |
| `lfm2` | vision | — | metadata=deferred; tensor_map=deferred; graph=deferred; runtime=deferred | The existing LFM2-VL HF graph has no pinned mmproj tensor closure or component parity. |
| `lfm2a` | audio | — | metadata=deferred; tensor_map=deferred; graph=deferred; runtime=deferred | LFM2 conformer audio graph has no GGUF tensor mapping. |
| `lightonocr` | vision | — | metadata=deferred; tensor_map=deferred; graph=deferred; runtime=deferred | LightOnOCR Pixtral variant has no exact tensor mapping. |
| `llama4` | vision | `llama4` | metadata=deferred; tensor_map=deferred; graph=deferred; runtime=deferred | Llama4 vision encoder and multimodal target package are out of scope. |
| `meralion` | audio | — | metadata=deferred; tensor_map=deferred; graph=deferred; runtime=deferred | Meralion audio projector is not implemented. |
| `mimo_audio` | audio | — | metadata=deferred; tensor_map=deferred; graph=deferred; runtime=deferred | MiMo audio RVQ/local-transformer graph is not implemented. |
| `mimovl` | vision | — | metadata=deferred; tensor_map=deferred; graph=deferred; runtime=deferred | MiMo-VL vision/projector graph is not implemented. |
| `minicpmv4_6` | vision | — | metadata=deferred; tensor_map=deferred; graph=deferred; runtime=deferred | MiniCPM-V 4.6 SAM/resampler graph is not implemented. |
| `minimax_m3` | vision | — | metadata=deferred; tensor_map=deferred; graph=deferred; runtime=deferred | MiniMax M3 vision/projector graph is not implemented. |
| `mlp` | vision | — | metadata=deferred; tensor_map=deferred; graph=deferred; runtime=deferred | LLaVA MLP topology and class-token feature selection are not implemented by the GGUF builder. |
| `muse-glimmer` | vision | `muse-glimmer` | metadata=supported; tensor_map=supported; graph=supported; runtime=deferred | artifact pins=`muse-glimmer-30b-bf16` |
| `musicflamingo` | audio | — | metadata=deferred; tensor_map=deferred; graph=deferred; runtime=deferred | Music Flamingo audio projector is not implemented. |
| `nemotron_v2_vl` | vision | — | metadata=deferred; tensor_map=deferred; graph=deferred; runtime=deferred | Nemotron V2 VL vision/projector graph is not implemented. |
| `paddleocr` | vision | `paddleocr` | metadata=deferred; tensor_map=deferred; graph=deferred; runtime=deferred | PaddleOCR vision/projector graph is not implemented. |
| `parakeet` | audio | — | metadata=deferred; tensor_map=deferred; graph=deferred; runtime=deferred | Parakeet audio encoder graph is not implemented. |
| `phi4` | vision | — | metadata=deferred; tensor_map=deferred; graph=deferred; runtime=deferred | The Phi-4 vision projector exists for HF weights but has no pinned GGUF tensor closure. |
| `pixtral` | vision | `deepseek2`, `llama`, `mistral3`, `mistral4` | metadata=deferred; tensor_map=deferred; graph=deferred; runtime=deferred | The Pixtral component has no pinned mmproj tensor mapping or positional-interpolation parity. |
| `pockettts_gen` | gen.audio | — | metadata=rejected; tensor_map=rejected; graph=rejected; runtime=rejected | Generated-audio decoder sidecars are not multimodal projectors and cannot be paired with a text target package. |
| `pockettts_spkenc` | audio | — | metadata=deferred; tensor_map=deferred; graph=deferred; runtime=deferred | PocketTTS speaker encoder graph is not implemented. |
| `qwen2.5o` | audio, vision | `qwen2vl` | metadata=deferred; tensor_map=deferred; graph=deferred; runtime=deferred | This legacy string changes meaning by modality; accepting it would create a false alias. |
| `qwen2.5vl_merger` | vision | `qwen2vl` | metadata=supported; tensor_map=supported; graph=supported; runtime=deferred | artifact pins=`qwen25-vl-3b-f16` |
| `qwen2a` | audio | — | metadata=deferred; tensor_map=deferred; graph=deferred; runtime=deferred | Qwen2 audio encoder/projector is not implemented. |
| `qwen2vl_merger` | vision | `qwen2vl` | metadata=supported; tensor_map=supported; graph=supported; runtime=deferred | artifact pins=`qwen2-vl-2b-f16` |
| `qwen3a` | audio | `qwen3vl`, `qwen3vlmoe` | metadata=deferred; tensor_map=deferred; graph=deferred; runtime=deferred | Qwen3 audio encoder/projector is not implemented. |
| `qwen3tts_gen` | gen.audio | — | metadata=rejected; tensor_map=rejected; graph=rejected; runtime=rejected | Generated-audio decoder sidecars are not multimodal projectors and cannot be paired with a text target package. |
| `qwen3tts_spkenc` | audio | — | metadata=deferred; tensor_map=deferred; graph=deferred; runtime=deferred | Qwen3-TTS speaker encoder graph is not implemented. |
| `qwen3vl_merger` | vision | `qwen35`, `qwen35moe`, `qwen3vl`, `qwen3vlmoe` | metadata=deferred; tensor_map=deferred; graph=deferred; runtime=deferred | The Qwen3-VL merger/window ordering has no GGUF tensor-closure parity test. |
| `resampler` | vision | — | metadata=deferred; tensor_map=deferred; graph=deferred; runtime=deferred | MiniCPM-V query resampler and positional interpolation are not implemented. |
| `step3vl` | vision | — | metadata=deferred; tensor_map=deferred; graph=deferred; runtime=deferred | Step3-VL vision and projector graph are not implemented. |
| `ultravox` | audio | — | metadata=deferred; tensor_map=deferred; graph=deferred; runtime=deferred | Whisper encoder plus Ultravox stack projector is not implemented. |
| `voxtral` | audio | — | metadata=deferred; tensor_map=deferred; graph=deferred; runtime=deferred | Voxtral Whisper encoder/projector is not implemented. |
| `yasa2` | vision | — | metadata=deferred; tensor_map=deferred; graph=deferred; runtime=deferred | YASA2 vision/projector graph is not implemented. |
| `youtuvl` | vision | — | metadata=deferred; tensor_map=deferred; graph=deferred; runtime=deferred | YouTu-VL vision/projector graph is not implemented. |

<!-- END GGUF MMPROJ SUPPORT MATRIX -->

Real-artifact audit pins:

| Family | Revision and file | Size | LFS SHA-256 | Metadata/tensor qtypes | Paired text target |
|---|---|---:|---|---|---|
| Gemma 3 | `ggml-org/gemma-3-4b-it-GGUF@ab31416aceb30cd095cb34cc27eea120940964e4`<br>`mmproj-model-f16.gguf` | 851,251,104 | `8c0fb064b019a6972856aaae2c7e4792858af3ca4561be2dbf649123ba6c40cb` | `gemma3`; 276 F32 + 163 F16 tensors; vision 1152→2560 | `gemma-3-4b-it-Q4_K_M.gguf` (`gemma3`) |
| Gemma4 | `unsloth/gemma-4-E2B-it-GGUF@0314792d7f1f7e229411f620751375812bb9faf2`<br>`mmproj-F16.gguf` | 985,654,080 | `337ee849e80b6169ce9d1d573d424fc1653bcafa5f0cb0cbb901beba54f4b41c` | `gemma4v` + deferred `gemma4a`; 1,163 F32 + 248 F16 tensors; vision 768→1536, audio 1024→1536 | `gemma-4-E2B-it-Q4_K_M.gguf` (`gemma4`) |
| Muse Glimmer | `unsloth/Muse-Glimmer-30B-GGUF@faa5b025c584459c13febfa5c59883516710ae39`<br>`mmproj-Muse-Glimmer-30B-BF16.gguf` | 3,849,173,728 | `7aa788cfe25ae5e4bf4837511f64df22cabe595e58223708274a67b3136f53ab` | `muse-glimmer`; 506 F32 + 303 BF16 tensors; vision 1536, merge 2, projection 6656 | `Muse-Glimmer-30B-UD-Q4_K_XL.gguf` (`muse-glimmer`) |
| Qwen2-VL | `ggml-org/Qwen2-VL-2B-Instruct-GGUF@bb307c036e8a1ed7b663bbd0c35b41c4c9294cfd`<br>`mmproj-Qwen2-VL-2B-Instruct-f16.gguf` | 1,331,656,160 | `ecb20cabcdd8dbc277de06bd6eb980aeb2adfaaba9f199a434e328d205675d03` | `qwen2vl_merger`; 324 F32 + 196 F16 tensors; vision 1280→1536 | `Qwen2-VL-2B-Instruct-Q4_K_M.gguf` (`qwen2vl`) |
| Qwen2.5-VL | `ggml-org/Qwen2.5-VL-3B-Instruct-GGUF@5037fcf163dd95d1e41d1974465f0898ed108ca2`<br>`mmproj-Qwen2.5-VL-3B-Instruct-f16.gguf` | 1,338,428,128 | `b9160fe9d814d1fadf68395677468534778b39ac33c2e7561b7b218626e60d5e` | `qwen2.5vl_merger`; 291 F32 + 228 F16 tensors; vision 1280→2048 | `Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf` (`qwen2vl`) |

The Gemma 3 processor assets are pinned to
`google/gemma-3-4b-it@093f9f388b31de276ce2de164bdc2081324b9767`.
`Gemma3Processor` emits `pixel_values` as
`float32[num_images,3,896,896]` and 256 `<image_soft_token>` placeholders
per image. The vision graph processes one image row per invocation; callers
concatenate its 256-row outputs in processor order. The gated parent assets
and ORT GenAI orchestration remain runtime waivers.

The Qwen processor assets are pinned to
`Qwen/Qwen2-VL-2B-Instruct@895c3a49bc3fa70a340399125c650a463535e71c`
and
`Qwen/Qwen2.5-VL-3B-Instruct@66285546d2b821cf421d4f5eb2576359d3770cd3`.
Their independent float32 image/video patch streams bind to separate vision
invocations, then to `image_features` and `video_features` on the embedding
graph. Qwen2.5 timing metadata and downstream M-RoPE construction are not
consumed by these graphs, so runtime support remains deferred.

The Gemma4 processor contract uses tokenizer token `<|image|>` (ID 258880 in
the audited paired GGUF) and 3×3 spatial pooling. The sidecar's image mean/std
are metadata for preprocessing; the ONNX vision graph consumes already
patchified `pixel_values` plus `pixel_position_ids`. Runtime callers must
generate those inputs in the same patch order. Mobius does not currently emit
an image-processor asset from GGUF metadata. The Gemma4 vision graph has an
explicit single-image batch contract (`pixel_values[1, N, ...]` and
`pixel_position_ids[1, N, 2]`): each image can produce a different number of
pooled tokens, and the current flattened output has no row-splits output with
which to represent those variable counts. Process multiple images separately
and concatenate their feature rows in prompt media order.

Pairing is fail-closed and independent of local filenames. Both GGUF files must
carry matching non-empty `general.name` values. If either file declares
`general.base_model.0.name` or `general.base_model.0.repo_url`, the other must
declare the same non-empty binding. Canonicalization is conservative: casing,
outer whitespace, URL host syntax, a trailing slash, and a trailing `.git` are
normalized, but meaningful name and path separators are preserved. Architecture
and matching tensor dimensions alone are not accepted as source identity. The
validator also inventories every sidecar tensor: Gemma4's exact deferred
`gemma4a` companion closure is allowed only with all pinned active-audio
metadata (including a positive integer `clip.audio.num_mel_bins`), but unknown
top-level names, near-miss prefixes/suffixes, ranks, and packed projector
tensors are rejected before graph construction. Gemma4 vision specifically
requires an RGB Conv2d patch weight shaped `[hidden, 3, patch, patch]`; rank-5
patch weights remain valid only for supported topologies whose graphs consume
temporal patches.

Sharded GGUF files are rejected. A single shard has only part of the tensor
table, and treating it as a complete checkpoint would create a corrupt model.

### Dense-transformer validation

The first dense-transformer cohort adds exact config, tensor-map, and graph
support for OLMo, OLMo2, Cohere2, Arcee, SmolLM3, and Exaone. Runtime remains
deferred for Cohere2, Arcee, and Exaone until pinned real-weight parity or
generation evidence is available; synthetic graph execution alone is not a
runtime claim.

Real-file execution evidence for the other three mappings is pinned to:

| Architecture | Revision and file | Size | SHA-256 | Stored qtypes |
|---|---|---:|---|---|
| OLMo | `QuantFactory/AMD-OLMo-1B-GGUF@5f34243a42dbae2141b8f5286320bf63d51eeefb`<br>`AMD-OLMo-1B.Q4_K_M.gguf` | 733,520,128 | `2a848051ef7a3edfd829ce915835794e789e6ed7f425066c242759b8dbc645b4` | 96 Q4_K, 17 Q6_K |
| OLMo2 | `allenai/OLMo-2-0425-1B-Instruct-GGUF@62f8c199538474c3e33ed5d7e0580abd66686a27`<br>`OLMo-2-0425-1B-Instruct-Q4_K_M.gguf` | 935,515,296 | `abd8187934a438fbf7cfff0a1de5b9d2793ce913f158794df1951dcba6c93cc6` | 97 Q4_K, 17 Q6_K, 65 F32 |
| SmolLM3 | `ggml-org/SmolLM3-3B-GGUF@4965cb60b150737b68a0408c36aeefb65078f894`<br>`SmolLM3-Q4_K_M.gguf` | 1,915,305,312 | `8334b850b7bd46238c16b0c550df2138f0889bf433809008cc17a8b05761863e` | 216 Q4_K, 37 Q6_K, 73 F32 |

For these mixed Q4_K_M artifacts, projections are not preserved natively.
Q4_K and Q6_K tensors are dequantized and affine-requantized to explicit
zero-point 4-bit/block-32 `MatMulNBits`; compatible token embeddings use
`GatherBlockQuantized`, and float normalization tensors remain float.
Pinned integration tests save the packages, open ORT CPU sessions, and verify
repeatable logits and greedy outputs. These are importer execution checks, not
independent full-logit parity or runtime-package evidence, so all architecture
runtime verdicts remain deferred.

Other C01 candidates remain excluded until their distinct semantics are
implemented and validated. These include fused/interleaved or dual-form QKV
(GPT-NeoX and Phi-2), ALiBi (Baichuan and MPT), learned position embeddings
(StarCoder), and unproven model aliases (Qwen and Command-R).

### Remaining conventional-attention MoE cohort

The next bounded C02 audit covers exactly Arctic, DBRX, GPT-OSS, Grok,
GroveMoE, and SmallThinker at the pinned llama.cpp commit. All six are
registered as explicit pre-config deferrals. Existing Hugging Face model
registrations and similar expert tensor names are deliberately not treated as
GGUF compatibility evidence.

- Arctic combines a dense parallel SwiGLU branch with a separately normalized
  routed branch sourced from a different residual.
- DBRX requires LayerNorm, fused QKV, K/Q/V clamping, and a distinct
  post-attention normalization point.
- GPT-OSS requires lossless interleaved gate/up splitting, expert-major MXFP4
  repacking, expert/router biases, attention sinks, and sliding-window
  semantics under one ownership contract.
- Grok combines embedding/attention/logit scales, attention and optional
  final-logit softcaps, extra norms, and dense-plus-routed residual scaling.
- GroveMoE shares router logits across normal and grouped chunk-expert banks,
  performs separate expert selections, and adds a scaled adjugate branch.
- SmallThinker routes from the unnormalized residual, selects sigmoid or
  softmax gating from metadata, uses ReLU experts, and has per-layer
  RoPE/sliding-window scheduling.

The vendored census records every direct loader tensor name, including the
conditional fused-or-split QKV union and optional biases from the shared loader
helper, conditional Grok norm spellings, and GPT-OSS expert biases. Per-family
suffix closure records the generic optional `.scale` and `.input_scale`
sidecars for output, attention, dense FFN, and standard routed-expert
projections, while GroveMoE chunk experts are explicitly limited to `.weight`.
No mapping is installed, so malformed, partial, fused, quantized, or auxiliary
representations cannot reach config extraction or graph construction. Runtime
remains deferred by construction.

## Supported stored quantization types

`mobius/integrations/gguf/_quant_registry.py` covers all 43 `ggml_type` slots
and all 25 active storage-quantized types at the same pinned commit. Every
stored qtype has one explicit projection/output route. The table is generated
from that registry and checked byte-for-byte by `_quant_registry_test.py`.
Those routes apply only when the architecture table above marks quantized
import supported; otherwise `keep_quantized=True` is rejected before graph
construction.

<!-- BEGIN GGUF QUANTIZATION MATRIX (generated; see _quant_registry.py) -->

| Stored qtype | ID | Projection/output route | Direct exactness | Embedding route | Expert-major route | Non-MatMul route | Runtime |
|---|---:|---|---|---|---|---|---|
| `Q4_0` | 2 | affine repack | exact | affine repack | affine repack | dequantize to float | deferred |
| `Q4_1` | 3 | affine repack | lossy | affine repack | affine repack | dequantize to float | deferred |
| `Q5_0` | 6 | dequantize/requantize | — | dequantize/requantize | dequantize/requantize | dequantize to float | deferred |
| `Q5_1` | 7 | dequantize/requantize | — | dequantize/requantize | dequantize/requantize | dequantize to float | deferred |
| `Q8_0` | 8 | affine repack | exact | affine repack | affine repack | dequantize to float | deferred |
| `Q2_K` | 10 | dequantize/requantize | — | dequantize/requantize | dequantize/requantize | dequantize to float | deferred |
| `Q3_K` | 11 | dequantize/requantize | — | dequantize/requantize | dequantize/requantize | dequantize to float | deferred |
| `Q4_K` | 12 | affine repack | lossy | affine repack | affine repack | dequantize to float | deferred |
| `Q5_K` | 13 | dequantize/requantize | — | dequantize/requantize | dequantize/requantize | dequantize to float | deferred |
| `Q6_K` | 14 | affine repack | lossy | affine repack | affine repack | dequantize to float | deferred |
| `IQ2_XXS` | 16 | native byte-preserved | — | dequantize/requantize | native byte-preserved | dequantize to float | deferred |
| `IQ2_XS` | 17 | native byte-preserved | — | dequantize/requantize | native byte-preserved | dequantize to float | deferred |
| `IQ3_XXS` | 18 | native byte-preserved | — | dequantize/requantize | native byte-preserved | dequantize to float | deferred |
| `IQ1_S` | 19 | native byte-preserved | — | dequantize/requantize | native byte-preserved | dequantize to float | deferred |
| `IQ4_NL` | 20 | native byte-preserved | — | dequantize/requantize | native byte-preserved | dequantize to float | deferred |
| `IQ3_S` | 21 | native byte-preserved | — | dequantize/requantize | native byte-preserved | dequantize to float | deferred |
| `IQ2_S` | 22 | native byte-preserved | — | dequantize/requantize | native byte-preserved | dequantize to float | deferred |
| `IQ4_XS` | 23 | native byte-preserved | — | dequantize/requantize | native byte-preserved | dequantize to float | deferred |
| `IQ1_M` | 29 | native byte-preserved | — | dequantize/requantize | native byte-preserved | dequantize to float | deferred |
| `TQ1_0` | 34 | dequantize/requantize | — | dequantize/requantize | dequantize/requantize | dequantize to float | deferred |
| `TQ2_0` | 35 | dequantize/requantize | — | dequantize/requantize | dequantize/requantize | dequantize to float | deferred |
| `MXFP4` | 39 | native byte-preserved | — | dequantize/requantize | native byte-preserved | dequantize to float | deferred |
| `NVFP4` | 40 | dequantize/requantize | — | dequantize/requantize | dequantize/requantize | dequantize to float | deferred |
| `Q1_0` | 41 | affine repack | exact | affine repack | affine repack | rejected | deferred |
| `Q2_0` | 42 | rejected | — | rejected | rejected | rejected | deferred |

<!-- END GGUF QUANTIZATION MATRIX -->

`Q4_0`, `Q8_0`, and mainline `Q1_0` have exact formulas when the graph uses
their direct affine target. Exactness is target-dependent: normalizing `Q8_0`
to a mixed graph's 4-bit/block-32 target is lossy, while `Q1_0` is rejected
when its exact 2-bit/block-128 target cannot be used because it has no decoder.
`Q4_1` rounds its floating source offset to an integer zero point; `Q4_K` and
`Q6_K` are decoded and requantized to 4-bit/block-32, so those direct affine
routes are lossy. The remaining dequantize/requantize routes use the pinned
`gguf-py` decoder and a 4-bit/block-32 target. `Q2_0` is rejected because the
pinned package has neither a decoder nor a compatible runtime kernel.

Native bytes are only valid for compatible `BlockQuantizedMatMul`
projection/output weights. They are not a `GatherBlockQuantized` embedding
ABI. Tied input/output weights may share one affine table, but native projection
support does not make the embedding native. Native expert-major tensors are
accepted only when the source is contiguous and maps to complete per-expert
projection rows. Non-native expert-major tensors are rejected in quantized mode
until a complete split-and-repack path exists; `keep_quantized=False` retains
the existing float normalization path. Norms, biases, and other non-MatMul
tensors are dequantized to float or rejected. Runtime remains **deferred** until
real-weight ONNX Runtime execution is recorded; graph construction and operator
ABI matching are not runtime evidence.

### Representative mixed-qtype evidence

The following file was downloaded and inspected; its preset name was not used
to infer tensor formats:

- Repository: `lmstudio-community/Qwen2.5-0.5B-Instruct-GGUF`
- Revision: `3b32d0bf1ac136098d417677c7d757360f1ceb6b`
- Filename: `Qwen2.5-0.5B-Instruct-Q4_K_M.gguf`
- Size: `397807936` bytes
- LFS SHA-256: `fa4d41b65761ed565cac6b5f62e35135d050408b033114a128ab308c02b2e83a`
- Tensor inventory: 290 tensors: 121 `F32`, 132 `Q5_0`, 12 `Q4_K`,
  12 `Q6_K`, and 13 `Q8_0`
- Element inventory: 71,552 `F32`; 251,854,848 `Q5_0`; 52,297,728
  `Q4_K`; 52,297,728 `Q6_K`; 137,510,912 `Q8_0`
- Build evidence: direct import produced one model with 169 `MatMulNBits`
  nodes, one `GatherBlockQuantized` node, and no `BlockQuantizedMatMul` nodes

This file therefore takes mixed routes: `Q4_K`/`Q6_K` use their declared lossy
affine conversion, `Q5_0` dequantizes/requantizes, and `Q8_0` is converted to
the graph's common 4-bit/block-32 target rather than retaining its standalone
8-bit affine layout. This is build evidence, not runtime execution evidence.

The eight retired slots (`Q4_2`, `Q4_3`, `Q4_0_4_4`, `Q4_0_4_8`,
`Q4_0_8_8`, `IQ4_NL_4_4`, `IQ4_NL_4_8`, `IQ4_NL_8_8`) have block size zero
and are unreadable. `Q8_1` and `Q8_K` are compute-only intermediates, never
valid GGUF weight storage.

## Validation scope and quality-checklist waivers

This support census is importer/registry/package-policy work, not a new model
architecture. Model-addition L1 configuration entries and L2 YAML cases are
therefore not applicable. Synthetic graph construction or ORT execution is
reported only as graph/import evidence.

- **L4/L5 and real-weight parity:** not claimed globally. An architecture stays
  runtime-deferred until a structured evidence record identifies an immutable
  GGUF and source config/tokenizer, tensor/qtype census, actual import route,
  independent full-logit or component parity, and deterministic generation or
  stateful semantics.
- **Dtypes and execution providers:** no blanket fp16, bf16, CUDA, DirectML, or
  cross-EP semantic claim is made by the census. Per-route restrictions still
  apply, and graph/session creation alone is not evidence.
- **ORT GenAI, Foundry Local, and Olive:** optional downstream probes. They do
  not gate Mobius graph export, and no support is implied when a probe is absent.
  Any recorded probe must name its exact version and result.
- **Direct source-backed GGUF reuse:** limited to compatible little-endian,
  single-file text packages and direct ONNX Runtime with optimization disabled,
  as documented above. It does not establish runtime support for an architecture.


## NVIDIA Nemotron-H MoE support boundary

`nemotron_h_moe` backbone conversion is supported against pinned llama.cpp
`8d9af256337d1a501250f9bbf4c0859a654bddd6`. Import reconstructs the exact
per-layer Mamba2/attention/dense/MoE schedule and validates every required
tensor before graph construction. Routed experts use sigmoid probabilities,
correction bias only for top-k selection, optional probability normalization,
and the serialized routed scaling factor. Experts are non-gated ReLU-squared
MLPs; one shared expert and optional latent down/up projections are represented
explicitly.

The importer accepts only the canonical stacked expert tensors and canonical
`blk.N.exp_probs_b.bias`. Missing expert tensors, separate or fused variants,
partial latent projections, unsupported scale/input-scale sidecars, and
inconsistent metadata or logical shapes fail before graph construction.
Quantized-source import is available only with `keep_quantized=False`
(`mobius build-gguf --dequantize`): mixed recurrent roles and the custom
sigmoid/ReLU-squared expert graph do not have an exact current quantized ABI.

Synthetic evidence covers:

- full-logit parity with Transformers for dense, routed/shared, and latent-MoE
  schedules, including correction-bias-driven expert selection;
- float and dequantized Q4 GGUF import, expert-order value checks, ORT execution
  with convolution/recurrent/KV state threading, and package round-trip;
- strict malformed-family, sidecar, shape, and MTP rejection.

The smallest public full-MoE source is still the 30B-A3B model, so no practical
small real-weight checkpoint exists for CI parity. The pinned public pair is
`nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16` and
`unsloth/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF`.

Those released GGUFs append a combined attention+MoE MTP block to the
52-layer backbone (23 Mamba2, 23 MoE, 6 attention). Mobius rejects that MTP
sidecar before graph construction rather than aliasing `block_count=53` to a
52-layer decoder. ORT GenAI packaging also remains deferred because released
runtime schemas do not represent the heterogeneous KV, convolution, and
recurrent state slots; see
[`onnxruntime/mobius#605`](https://github.com/onnxruntime/mobius/issues/605).
