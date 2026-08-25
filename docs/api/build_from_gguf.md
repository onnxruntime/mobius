# `build_from_gguf()`

Build an ONNX `ModelPackage` from a GGUF model file.

```python
from mobius import build_from_gguf
```

> **Note**: Requires the optional `gguf` package:
> `pip install mobius-onnx[gguf]`

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
| `keep_quantized` | `bool` | `True` | Preserve quantization when present. Supported affine blocks are repacked as `MatMulNBits`; in text-only builds, supported native IQ/MXFP4 projection blocks retain their bytes. Multimodal and mixed presets may require dequantization/requantization. Set to `False` to dequantize all weights. |
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
3. Preserves supported quantized tensors by default, using repacking,
   text-only native-block retention, or dequantize/requantize according to the
   source qtype and build path; `keep_quantized=False` dequantizes every tensor
4. Applies architecture-specific tensor processors (e.g. Q/K permute)
5. Builds the ONNX graph using the same pipeline as `build()`
6. Runs `preprocess_weights()` (HF → ONNX name mapping)
7. Applies weights to the graph

F32-, F16-, and BF16-only GGUFs use the normal float import path even though
`keep_quantized=True` is the default: there is no quantization to preserve.
Quantized GGUFs whose qtypes have no trustworthy decoder or compatible runtime
kernel (currently `Q2_0`) fail with an actionable error. Decoder-backed formats
such as `Q5_K` use the explicit dequantize/requantize route; they are not
silently treated as preserved source quantization.

## Supported GGUF Architectures

Support is declared once, in
`mobius/integrations/gguf/_arch_registry.py`, as five independent verdicts per
architecture — config extraction, tensor mapping, graph construction, runtime
packaging, and quantized import. Anything short of "supported" carries a reason,
and the table below is checked against the registry by
`_arch_registry_test.py::TestDocumentedSupportMatrix`, so it cannot drift.

Being listed is not a support claim. `build_from_gguf` refuses every
architecture that is not fully supported, naming the missing capability and the
reason.

<!-- BEGIN GGUF SUPPORT MATRIX (generated; see _arch_registry.py) -->

| GGUF architecture | Accepted aliases | mobius `model_type` | Float import | Quantized import |
|---|---|---|---|---|
| `apertus` | — | — | config deferred; tensor_map deferred; graph deferred; runtime deferred | unreachable |
| `arcee` | — | `arcee` | runtime deferred | supported |
| `arwkv7` | — | — | config deferred; tensor_map deferred; graph deferred; runtime deferred | unreachable |
| `baichuan` | — | `baichuan` | runtime deferred | supported |
| `bailingmoe3` | — | — | config deferred; tensor_map deferred; graph deferred; runtime deferred | unreachable |
| `bert` | — | `bert` | runtime deferred | supported |
| `bloom` | — | `bloom` | tensor_map deferred | unreachable |
| `chatglm` | — | `chatglm` | runtime deferred | rejected |
| `clip` | — | — | config rejected; tensor_map rejected; graph rejected; runtime rejected | unreachable |
| `cohere2` | — | `cohere2` | runtime deferred | supported |
| `deci` | — | `llama` | supported | supported |
| `deepseek4` | — | — | config deferred; tensor_map deferred; graph deferred; runtime deferred | unreachable |
| `dflash` | — | `DFlashDraftModel` | runtime deferred | supported |
| `dream` | — | `dream` | runtime deferred | supported |
| `eagle3` | — | `Eagle3DraftModel` | runtime deferred | supported |
| `eurobert` | — | — | config deferred; tensor_map deferred; graph deferred; runtime deferred | unreachable |
| `exaone` | — | `exaone` | runtime deferred | supported |
| `falcon` | — | `falcon` | supported | supported |
| `falcon-h1` | — | — | config deferred; tensor_map deferred; graph deferred; runtime deferred | unreachable |
| `gemma` | — | `gemma` | supported | supported |
| `gemma2` | — | `gemma2` | supported | supported |
| `gemma3` | — | `gemma3_text` | supported | supported |
| `gemma4` | — | `gemma4_text` | supported | supported |
| `glm-dsa` | `glm_dsa` | `glm_moe_dsa` | tensor_map deferred | unreachable |
| `gpt2` | — | `gpt2` | supported | supported |
| `granitehybrid` | — | `granitemoehybrid` | runtime deferred | rejected |
| `granitemoe` | — | `granitemoe` | runtime deferred | supported |
| `hunyuan-dense` | `hunyuan_v1_dense` | `hunyuan_v1_dense` | supported | supported |
| `internlm2` | — | `internlm2` | supported | rejected |
| `jamba` | — | `jamba` | runtime deferred | rejected |
| `jina-bert-v2` | — | — | config deferred; tensor_map deferred; graph deferred; runtime deferred | unreachable |
| `jina-bert-v3` | — | — | config deferred; tensor_map deferred; graph deferred; runtime deferred | unreachable |
| `kimi-k3` | — | — | config deferred; tensor_map deferred; graph deferred; runtime deferred | unreachable |
| `kimi-linear` | — | — | config deferred; tensor_map deferred; graph deferred; runtime deferred | unreachable |
| `lfm2` | — | `lfm2` | runtime deferred | supported |
| `lfm2moe` | — | — | config deferred; tensor_map deferred; graph deferred; runtime deferred | unreachable |
| `llada` | — | `llada` | runtime deferred | supported |
| `llada-moe` | — | `llada` | runtime deferred | supported |
| `llama` | `mistral` | `llama` | supported | supported |
| `mamba` | — | `mamba` | runtime deferred | rejected |
| `mamba2` | — | `mamba2` | runtime deferred | rejected |
| `minicpm3` | — | — | config deferred; tensor_map deferred; graph deferred; runtime deferred | unreachable |
| `minimax-01` | — | — | config deferred; tensor_map deferred; graph deferred; runtime deferred | unreachable |
| `modern-bert` | — | `modernbert` | runtime deferred | supported |
| `mpt` | — | — | config deferred; tensor_map deferred; graph deferred; runtime deferred | unreachable |
| `muse-glimmer` | `muse_glimmer` | `muse_glimmer_text` | supported | supported |
| `nemotron` | — | `nemotron` | supported | supported |
| `nemotron_h` | — | `nemotron_h` | runtime deferred | rejected |
| `nemotron_h_moe` | — | — | config rejected; tensor_map rejected; graph rejected; runtime rejected | unreachable |
| `neo-bert` | — | — | config deferred; tensor_map deferred; graph deferred; runtime deferred | unreachable |
| `nomic-bert` | — | — | config deferred; tensor_map deferred; graph deferred; runtime deferred | unreachable |
| `nomic-bert-moe` | — | — | config deferred; tensor_map deferred; graph deferred; runtime deferred | unreachable |
| `olmo` | — | `olmo` | supported | supported |
| `olmo2` | — | `olmo2` | supported | supported |
| `olmoe` | — | `olmoe` | runtime deferred | supported |
| `openelm` | — | — | config deferred; tensor_map deferred; graph deferred; runtime deferred | unreachable |
| `phi2` | — | `phi` | runtime deferred | rejected |
| `phi3` | — | `phi3` | supported | supported |
| `phimoe` | — | `phimoe` | runtime deferred | supported |
| `plamo2` | — | — | config deferred; tensor_map deferred; graph deferred; runtime deferred | unreachable |
| `pockettts` | — | — | config deferred; tensor_map deferred; graph deferred; runtime deferred | unreachable |
| `qwen2` | — | `qwen2` | supported | supported |
| `qwen2moe` | `qwen2_moe` | `qwen2_moe` | runtime deferred | supported |
| `qwen3` | — | `qwen3` | supported | supported |
| `qwen35` | — | `qwen3_5_text` | runtime deferred | supported |
| `qwen35moe` | — | `qwen3_5_moe` | runtime deferred | supported |
| `qwen3moe` | `qwen3_moe` | `qwen3_moe` | runtime deferred | supported |
| `qwen3next` | — | `qwen3_next` | runtime deferred | supported |
| `qwen3tts` | — | — | config deferred; tensor_map deferred; graph deferred; runtime deferred | unreachable |
| `rnd1` | — | `llada` | runtime deferred | supported |
| `rwkv6` | — | — | config deferred; tensor_map deferred; graph deferred; runtime deferred | unreachable |
| `rwkv6qwen2` | — | — | config deferred; tensor_map deferred; graph deferred; runtime deferred | unreachable |
| `rwkv7` | — | — | config deferred; tensor_map deferred; graph deferred; runtime deferred | unreachable |
| `seed_oss` | — | `seed_oss` | runtime deferred | supported |
| `smollm3` | — | `smollm3` | supported | supported |
| `stablelm` | — | `stablelm` | supported | supported |
| `starcoder2` | — | `starcoder2` | supported | supported |
| `t5` | — | `t5` | runtime deferred | supported |
| `t5encoder` | — | `t5encoder` | runtime deferred | supported |
| `talkie` | — | — | config deferred; tensor_map deferred; graph deferred; runtime deferred | unreachable |
| `wavtokenizer-dec` | — | — | config deferred; tensor_map deferred; graph deferred; runtime deferred | unreachable |

<!-- END GGUF SUPPORT MATRIX -->

Canonical names are the strings llama.cpp writes into `general.architecture`,
validated against a vendored census of the 147 architectures llama.cpp defines
at commit `8d9af256337d1a501250f9bbf4c0859a654bddd6`. Aliases are spellings
llama.cpp does not emit but that mobius still accepts.

### Dense C01 cohort

The pinned llama.cpp `8d9af256337d1a501250f9bbf4c0859a654bddd6` dense cohort
adds bounded graph import for 32-layer `baichuan`, modern `chatglm`, `phi2`, and
64-layer `seed_oss`. Runtime remains deferred until a pinned real GGUF has
independent full-logit and generation parity.

- Baichuan accepts only the 7B RoPE graph, reverses the converter Q/K permutation,
  and rejects the 40-layer hardcoded-ALiBi path. Phi-2 requires its complete bias
  closure, 4H tanh-approximated GELU MLP, full MHA, and an untied output.
- ChatGLM accepts only contiguous fused QKV and fused gate/up tensors. Quantized
  fused forms are rejected because splitting their packed values and sidecars is
  not yet losslessly covered. Seed-OSS maps `post_attention_norm` exactly and permits
  either an explicit output or effective ownership by the token embedding, and
  accepts Q/K/V biases only as a complete all-layer family.
- `apertus`, `minicpm3`, `openelm`, and `mpt` are explicit pre-config deferrals.
  Apertus's serialized Llama-3 `rope_freqs` tensor supplies per-dimension factors
  that the current scalar-config Mobius RoPE graph cannot consume. The others'
  MLA/scaling topology, per-layer fused/tied topology, and optional learned
  positions/QK norms/clipping/AWQ/bias closure are likewise not represented by
  current Mobius graphs.

### Second hybrid cohort

`jamba`, `nemotron_h`, and `granitehybrid` have graph-import support for exact
dense subsets only; runtime packaging remains deferred pending independent
real-artifact full-logit and stateful-generation parity.

- Schedules come from suffix-exact per-layer metadata. Jamba and GraniteHybrid
  use `attention.head_count_kv` (`0` selects Mamba/Mamba2). Nemotron-H combines
  that array with per-layer `feed_forward_length` to select exactly one of
  Mamba2, attention, or dense ReLU² MLP.
- Jamba requires `ssm.inner_size == 2 * embedding_length`. Nemotron-H rejects
  MTP and all MoE files. GraniteHybrid rejects routed-MoE files until 3-D expert
  fusion, ordering, and quantized preservation have independent value tests.
- Every layer must provide exactly its pinned loader tensor family. Missing,
  wrong-mixer, partial, auxiliary, scale/input-scale, and out-of-range tensors
  are rejected before graph construction. GGUF Mamba decay values are inverted
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

`minimax-01`, `plamo2`, and `falcon-h1` are deferred before config extraction.
MiniMax-01's pinned Lightning schedule and decay/scaling semantics do not match
the current graph; PLaMo2 needs a dedicated fused-QKV Mamba1/attention graph;
Falcon-H1 executes attention and Mamba2 in parallel in every block and therefore
needs KV plus conv/SSM states simultaneously. It is not an alias of ordinary
Falcon. `nemotron_h_moe` remains rejected because its folded MTP attention+MoE
head has no equivalent package contract.

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
not lossless. For both supported encoders, `attention.head_count_kv` defaults to
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
disabled until that ABI is represented explicitly. Dense Qwen3.5 MTP sidecars
use the post-final-norm `mtp_seed` output while preserving pre-norm
`hidden_states.N` captures. Optional dedicated `nextn.embed_tokens`,
`nextn.shared_head_norm`, and `nextn.shared_head_head` tensors are consumed
when present; otherwise the sidecar explicitly falls back to the backbone
embedding, final norm, and tied or untied output head. Qwen3.5-MoE and
Qwen3-Next MTP blocks are rejected because the current sidecar builder cannot
represent routed experts without dropping tensors. Separate MTP-only files
are outside this cohort.

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

### Multimodal projector sidecars

Projector support is pinned to llama.cpp
`8d9af256337d1a501250f9bbf4c0859a654bddd6`. The enum has 62 entries:
60 serialized strings below, one unserialized `MLP_NORM` compatibility entry,
and `UNKNOWN`. `clip` has vision, audio, and generated-audio presence flags;
there is no text-encoder presence key at this pin.

Only `gemma4v` and `muse-glimmer` currently pass metadata, suffix-exact tensor
closure, target-pairing, graph, and component-parity gates. `gemma4a` remains
deferred: the real sidecar contains `a.pre_encode.*` tensors that the partial
mapping does not consume. A sidecar may contain that audio tower alongside a
supported `gemma4v` tower, but requesting an audio graph fails before graph
construction. Packed projector tensors are rejected; supported encoder and
projector weights must be F32, F16, or BF16, and Gemma4 clipping bounds must be
F32. Unknown `.scale`/`.input_scale` tensors and out-of-closure weights are
errors rather than silently dropped.

<!-- BEGIN GGUF MMPROJ SUPPORT MATRIX (generated; see _mmproj_registry.py) -->

| Projector string | Modality | Paired text architecture | Status | Limitation |
|---|---|---|---|---|
| `mlp` | vision | — | metadata deferred; tensor_map deferred; graph deferred; runtime deferred | LLaVA MLP topology and class-token feature selection are not implemented by the GGUF builder. |
| `ldp` | vision | — | metadata deferred; tensor_map deferred; graph deferred; runtime deferred | MobileVLM LDP convolutional projector semantics are not implemented. |
| `ldpv2` | vision | — | metadata deferred; tensor_map deferred; graph deferred; runtime deferred | MobileVLM LDPv2 pooling/projector semantics are not implemented. |
| `resampler` | vision | — | metadata deferred; tensor_map deferred; graph deferred; runtime deferred | MiniCPM-V query resampler and positional interpolation are not implemented. |
| `adapter` | vision | — | metadata deferred; tensor_map deferred; graph deferred; runtime deferred | GLM-Edge adapter tensor closure and graph are not implemented. |
| `qwen2vl_merger` | vision | — | metadata deferred; tensor_map deferred; graph deferred; runtime deferred | The existing HF Qwen2-VL graph is not wired to the pinned GGUF merger ABI. |
| `qwen2.5vl_merger` | vision | — | metadata deferred; tensor_map deferred; graph deferred; runtime deferred | The Qwen2.5-VL merger/window ordering has no GGUF tensor-closure parity test. |
| `qwen3vl_merger` | vision | — | metadata deferred; tensor_map deferred; graph deferred; runtime deferred | The Qwen3-VL merger/window ordering has no GGUF tensor-closure parity test. |
| `step3vl` | vision | — | metadata deferred; tensor_map deferred; graph deferred; runtime deferred | Step3-VL vision and projector graph are not implemented. |
| `gemma3` | vision | — | metadata deferred; tensor_map deferred; graph deferred; runtime deferred | Gemma3 mmproj feature selection and projector tensor map are not implemented. |
| `gemma3nv` | vision | — | metadata deferred; tensor_map deferred; graph deferred; runtime deferred | Gemma3n vision sidecar routing is not implemented. |
| `gemma3na` | audio | — | metadata deferred; tensor_map deferred; graph deferred; runtime deferred | Gemma3n audio sidecar routing is not implemented. |
| `gemma4v` | vision | `gemma4` | supported | Exact registry-backed graph, tensor closure, target pairing, and component parity. |
| `gemma4a` | audio | — | metadata deferred; tensor_map deferred; graph deferred; runtime deferred | The sidecar carries a.pre_encode tensors that the current audio map drops, and independent Conformer parity is not established. |
| `gemma4uv` | vision | — | metadata deferred; tensor_map deferred; graph deferred; runtime deferred | Encoder-free unified Gemma4 vision sidecars use a different patch embedder contract. |
| `gemma4ua` | audio | — | metadata deferred; tensor_map deferred; graph deferred; runtime deferred | Encoder-free unified Gemma4 waveform embedding is not wired to GGUF. |
| `phi4` | vision | — | metadata deferred; tensor_map deferred; graph deferred; runtime deferred | The Phi-4 vision projector exists for HF weights but has no pinned GGUF tensor closure. |
| `idefics3` | vision | — | metadata deferred; tensor_map deferred; graph deferred; runtime deferred | Idefics3 pixel-shuffle projector GGUF routing is not implemented. |
| `pixtral` | vision | — | metadata deferred; tensor_map deferred; graph deferred; runtime deferred | The Pixtral component has no pinned mmproj tensor mapping or positional-interpolation parity. |
| `ultravox` | audio | — | metadata deferred; tensor_map deferred; graph deferred; runtime deferred | Whisper encoder plus Ultravox stack projector is not implemented. |
| `internvl` | vision | — | metadata deferred; tensor_map deferred; graph deferred; runtime deferred | InternVL pixel-shuffle token ordering is not implemented for GGUF. |
| `llama4` | vision | — | metadata deferred; tensor_map deferred; graph deferred; runtime deferred | Llama4 vision encoder and multimodal target package are out of scope. |
| `qwen2a` | audio | — | metadata deferred; tensor_map deferred; graph deferred; runtime deferred | Qwen2 audio encoder/projector is not implemented. |
| `qwen3a` | audio | — | metadata deferred; tensor_map deferred; graph deferred; runtime deferred | Qwen3 audio encoder/projector is not implemented. |
| `glma` | audio | — | metadata deferred; tensor_map deferred; graph deferred; runtime deferred | GLM audio encoder/projector is not implemented. |
| `qwen2.5o` | audio, vision | — | metadata deferred; tensor_map deferred; graph deferred; runtime deferred | This legacy string changes meaning by modality; accepting it would create a false alias. |
| `voxtral` | audio | — | metadata deferred; tensor_map deferred; graph deferred; runtime deferred | Voxtral Whisper encoder/projector is not implemented. |
| `meralion` | audio | — | metadata deferred; tensor_map deferred; graph deferred; runtime deferred | Meralion audio projector is not implemented. |
| `musicflamingo` | audio | — | metadata deferred; tensor_map deferred; graph deferred; runtime deferred | Music Flamingo audio projector is not implemented. |
| `lfm2` | vision | — | metadata deferred; tensor_map deferred; graph deferred; runtime deferred | The existing LFM2-VL HF graph has no pinned mmproj tensor closure or component parity. |
| `kimivl` | vision | — | metadata deferred; tensor_map deferred; graph deferred; runtime deferred | Kimi-VL vision/projector graph is not implemented. |
| `paddleocr` | vision | — | metadata deferred; tensor_map deferred; graph deferred; runtime deferred | PaddleOCR vision/projector graph is not implemented. |
| `lightonocr` | vision | — | metadata deferred; tensor_map deferred; graph deferred; runtime deferred | LightOnOCR Pixtral variant has no exact tensor mapping. |
| `cogvlm` | vision | — | metadata deferred; tensor_map deferred; graph deferred; runtime deferred | CogVLM feature output differs from LLaVA and is not implemented. |
| `janus_pro` | vision | — | metadata deferred; tensor_map deferred; graph deferred; runtime deferred | Janus-Pro vision/projector graph is not implemented. |
| `dots_ocr` | vision | — | metadata deferred; tensor_map deferred; graph deferred; runtime deferred | DotsOCR vision merger is not implemented. |
| `dots3note_v` | vision | — | metadata deferred; tensor_map deferred; graph deferred; runtime deferred | Dots3Note vision pyramid MoE is not implemented. |
| `dots3note_a` | audio | — | metadata deferred; tensor_map deferred; graph deferred; runtime deferred | Dots3Note audio graph is not implemented. |
| `deepseekocr` | vision | — | metadata deferred; tensor_map deferred; graph deferred; runtime deferred | DeepSeek-OCR SAM/projector graph is not implemented. |
| `deepseekocr2` | vision | — | metadata deferred; tensor_map deferred; graph deferred; runtime deferred | DeepSeek-OCR2 SAM/projector graph is not implemented. |
| `lfm2a` | audio | — | metadata deferred; tensor_map deferred; graph deferred; runtime deferred | LFM2 conformer audio graph has no GGUF tensor mapping. |
| `glm4v` | vision | — | metadata deferred; tensor_map deferred; graph deferred; runtime deferred | GLM4V downsampler and projector are not implemented. |
| `youtuvl` | vision | — | metadata deferred; tensor_map deferred; graph deferred; runtime deferred | YouTu-VL vision/projector graph is not implemented. |
| `yasa2` | vision | — | metadata deferred; tensor_map deferred; graph deferred; runtime deferred | YASA2 vision/projector graph is not implemented. |
| `kimik25` | vision | — | metadata deferred; tensor_map deferred; graph deferred; runtime deferred | Kimi K2.5 vision/projector graph is not implemented. |
| `nemotron_v2_vl` | vision | — | metadata deferred; tensor_map deferred; graph deferred; runtime deferred | Nemotron V2 VL vision/projector graph is not implemented. |
| `exaone4_5` | vision | — | metadata deferred; tensor_map deferred; graph deferred; runtime deferred | EXAONE 4.5 vision merger is not implemented. |
| `hunyuanvl` | vision | — | metadata deferred; tensor_map deferred; graph deferred; runtime deferred | HunyuanVL vision/projector graph is not implemented. |
| `minicpmv4_6` | vision | — | metadata deferred; tensor_map deferred; graph deferred; runtime deferred | MiniCPM-V 4.6 SAM/resampler graph is not implemented. |
| `granite_speech` | audio | — | metadata deferred; tensor_map deferred; graph deferred; runtime deferred | Granite Speech audio encoder/projector is not implemented. |
| `mimovl` | vision | — | metadata deferred; tensor_map deferred; graph deferred; runtime deferred | MiMo-VL vision/projector graph is not implemented. |
| `minimax_m3` | vision | — | metadata deferred; tensor_map deferred; graph deferred; runtime deferred | MiniMax M3 vision/projector graph is not implemented. |
| `granite4_vision` | vision | — | metadata deferred; tensor_map deferred; graph deferred; runtime deferred | Granite 4 vision sidecar graph is not implemented. |
| `mimo_audio` | audio | — | metadata deferred; tensor_map deferred; graph deferred; runtime deferred | MiMo audio RVQ/local-transformer graph is not implemented. |
| `parakeet` | audio | — | metadata deferred; tensor_map deferred; graph deferred; runtime deferred | Parakeet audio encoder graph is not implemented. |
| `qwen3tts_spkenc` | audio | — | metadata deferred; tensor_map deferred; graph deferred; runtime deferred | Qwen3-TTS speaker encoder graph is not implemented. |
| `qwen3tts_gen` | gen.audio | — | metadata rejected; tensor_map rejected; graph rejected; runtime rejected | Generated-audio decoder sidecars are not multimodal projectors and cannot be paired with a text target package. |
| `pockettts_spkenc` | audio | — | metadata deferred; tensor_map deferred; graph deferred; runtime deferred | PocketTTS speaker encoder graph is not implemented. |
| `pockettts_gen` | gen.audio | — | metadata rejected; tensor_map rejected; graph rejected; runtime rejected | Generated-audio decoder sidecars are not multimodal projectors and cannot be paired with a text target package. |
| `muse-glimmer` | vision | `muse-glimmer` | supported | Exact registry-backed graph, tensor closure, target pairing, and component parity. |

<!-- END GGUF MMPROJ SUPPORT MATRIX -->

Real-artifact audit pins:

| Family | Revision and file | Size | LFS SHA-256 | Metadata/tensor qtypes | Paired text target |
|---|---|---:|---|---|---|
| Gemma4 | `unsloth/gemma-4-E2B-it-GGUF@0314792d7f1f7e229411f620751375812bb9faf2`<br>`mmproj-F16.gguf` | 985,654,080 | `337ee849e80b6169ce9d1d573d424fc1653bcafa5f0cb0cbb901beba54f4b41c` | `gemma4v` + deferred `gemma4a`; 1,163 F32 + 248 F16 tensors; vision 768→1536, audio 1024→1536 | `gemma-4-E2B-it-Q4_K_M.gguf` (`gemma4`) |
| Muse Glimmer | `unsloth/Muse-Glimmer-30B-GGUF@faa5b025c584459c13febfa5c59883516710ae39`<br>`mmproj-Muse-Glimmer-30B-BF16.gguf` | 3,849,173,728 | `7aa788cfe25ae5e4bf4837511f64df22cabe595e58223708274a67b3136f53ab` | `muse-glimmer`; 506 F32 + 303 BF16 tensors; vision 1536, merge 2, projection 6656 | `Muse-Glimmer-30B-UD-Q4_K_XL.gguf` (`muse-glimmer`) |

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

Runtime coverage for the other three mappings is pinned to:

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
repeatable logits and greedy outputs.

Other C01 candidates remain excluded until their distinct semantics are
implemented and validated. These include fused/interleaved or dual-form QKV
(GPT-NeoX and Phi-2), ALiBi (Baichuan and MPT), learned position embeddings
(StarCoder), and unproven model aliases (Qwen and Command-R).

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


## NVIDIA Nemotron 3.5 Lightning waiver

Direct conversion of GGUF architecture `nemotron_h_moe` is intentionally
disabled. The following evidence is pinned:

- GGUF repository:
  `unsloth/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF` at
  `f2d3fe3694501008786e81e5f20360cbf715496a`.
- Official BF16 comparison:
  `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16` at
  `d468880b6ad3c6e0d21377ce7242adaea4cc884d`.
- The official backbone has exactly 52 layers: 23 Mamba, 23 MoE, and
  6 attention layers. GGUF block 52 is a separate combined attention+MoE MTP
  auxiliary block, so `block_count=53` cannot be aliased to the backbone.

### Quantization findings

| GGUF file | Relevant tensor inventory | Direct preservation |
|---|---|---|
| `...-Q8_0.gguf` | 32.904B parameters in `Q8_0` | Qtype-compatible, but blocked by architecture and semantic validation |
| `...-MXFP4_MOE.gguf` | 14.687B `MXFP4`, 12.772B `Q5_1`, 5.445B `Q8_0` | No; the 5-bit expert weights require a quantization-changing float round-trip |
| `...-UD-Q4_K_M.gguf` | 15.326B `Q5_0`, 12.772B `Q5_1`, 4.806B `Q8_0` | No; the preset name does not describe its actual per-tensor types |
| `BF16/...-0000*-of-00002.gguf` | 329 tensors in shard 1 and 88 in shard 2 | No; Mobius does not assemble GGUF shards |

The GGUF embeds GPT-2/Pixtral BPE metadata with BOS 1 and EOS 11, but declares
padding ID 999 (`<SPECIAL_999>`). The pinned official tokenizer declares
`<|im_end|>` (ID 11) as padding. The GGUF also names the BF16 base repository
without recording its immutable source commit. Both discrepancies must be
resolved before a self-contained runtime package can be accepted.

The guard also reflects missing semantic evidence: Nemotron-H Mamba2 synthetic
full-logit parity is not passing, and no real-weight ORT or ORT GenAI generation
has passed. Graph creation, config emission, or session creation is not a
substitute for generation.

### Reproduce the guard with a pinned download

The `Q8_0` file is the only practical candidate whose large quantized tensors
all use a currently repackable type. Download it explicitly so the source does
not move:

```powershell
$repo = "unsloth/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF"
$revision = "f2d3fe3694501008786e81e5f20360cbf715496a"
$file = "NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q8_0.gguf"

python -m pip install `
  --index-url https://packagefeedproxy.microsoft.io/pypi/simple `
  huggingface_hub
hf download $repo $file --revision $revision --local-dir .\nemotron-gguf

# Expected: fail-fast NotImplementedError; no ONNX package is emitted.
python -m mobius build-gguf ".\nemotron-gguf\$file" `
  --ep cpu `
  --external-data safetensors --output .\nemotron-gguf-onnx
```

`mobius build-gguf --runtime ort-genai` is rejected separately. The GGUF CLI
does not emit `genai_config.json` until a selected architecture's cache and
tokenizer contracts have passed real ORT GenAI generation.

To execute the pinned GGUF without changing its quantization, use current
llama.cpp instead:

```powershell
.\llama-cli.exe `
  --model ".\nemotron-gguf\$file" `
  --temp 0.6 --top-p 0.95 --min-p 0.01
```

### Option A: official BF16, then Olive

Option A is the ONNX route because it preserves authoritative config,
tokenizer, and weight provenance. It is still a candidate until Nemotron-H
semantic tests pass, and currently targets direct ONNX Runtime rather than
ORT GenAI:

```powershell
$repo = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"
$revision = "d468880b6ad3c6e0d21377ce7242adaea4cc884d"

hf download $repo --revision $revision --local-dir .\nemotron-bf16
python -m mobius build `
  --config .\nemotron-bf16 `
  --dtype bf16 --ep cuda `
  --external-data safetensors --max-shard-size 5GB `
  --output .\nemotron-bf16-onnx
```

After the BF16 package passes full-logit and generation parity, quantize its
decoder with an initialized Olive environment:

```json
{
  "input_model": {
    "type": "OnnxModel",
    "model_path": "nemotron-bf16-onnx/model.onnx"
  },
  "passes": {
    "int4": {
      "type": "OnnxKQuantQuantization",
      "bits": 4,
      "block_size": 32
    }
  },
  "output_dir": "nemotron-int4-onnx"
}
```

```powershell
python -m pip install `
  --index-url https://packagefeedproxy.microsoft.io/pypi/simple `
  olive-ai onnxruntime
olive run --config .\olive-int4.json
Copy-Item .\nemotron-bf16\tokenizer* .\nemotron-int4-onnx\
Copy-Item .\nemotron-bf16\special_tokens_map.json .\nemotron-int4-onnx\
Copy-Item .\nemotron-bf16\chat_template.jinja .\nemotron-int4-onnx\
```

The candidate can be checked for direct ORT session loading:

```python
import onnxruntime as ort

session = ort.InferenceSession(
    r".\nemotron-int4-onnx\model.onnx",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)
print(session.get_providers())
print([(value.name, value.shape, value.type) for value in session.get_inputs()])
```

Session loading is not generation evidence. There is intentionally no ORT
GenAI generation command for this model at the pinned revisions:

- ORT GenAI 0.15.2 does not register model type `nemotron_h`.
- The generated generic decoder config does not bind the graph's Mamba
  `conv_state` and `recurrent_state` cache inputs.
- The official `generation_config.json` uses EOS IDs `[2, 11]`, while the
  architecture config alone supplies EOS 2.

Do not publish the package unless BF16 full logits match the pinned reference,
direct-ORT greedy generation is coherent and deterministic through an
independently validated hybrid-cache loop, the quantized package remains
non-degenerate, and ORT GenAI model/cache/token support is implemented before
claiming ORT GenAI compatibility.

### Prerequisites for revisiting direct GGUF conversion

1. Map the 52-layer schedule exactly and model block 52 as MTP, or explicitly
   exclude it with generation evidence.
2. Fix Nemotron-H Mamba2 full-logit parity before testing quantized output.
3. Preserve every large source qtype. For Q5 variants this requires a validated
   5-bit runtime kernel and repacker; dequantize/requantize is not direct
   preservation.
4. Resolve the GGUF padding-token mismatch and record an immutable upstream
   BF16 source revision.
5. Pass real-weight prefill, cached decode, deterministic multi-token
   generation, ORT load/inference, and ORT GenAI package generation on each
   claimed EP.
