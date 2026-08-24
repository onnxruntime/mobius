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
| `arcee` | — | `arcee` | runtime deferred | supported |
| `arwkv7` | — | — | config deferred; tensor_map deferred; graph deferred; runtime deferred | unreachable |
| `bloom` | — | `bloom` | tensor_map deferred | unreachable |
| `clip` | — | — | config rejected; tensor_map rejected; graph rejected; runtime rejected | unreachable |
| `cohere2` | — | `cohere2` | runtime deferred | supported |
| `deci` | — | `llama` | supported | supported |
| `deepseek4` | — | `deepseek_v4` | supported | supported |
| `exaone` | — | `exaone` | runtime deferred | supported |
| `falcon` | — | `falcon` | supported | supported |
| `gemma` | — | `gemma` | supported | supported |
| `gemma2` | — | `gemma2` | supported | supported |
| `gemma3` | — | `gemma3_text` | supported | supported |
| `gemma4` | — | `gemma4_text` | supported | supported |
| `glm-dsa` | `glm_dsa` | `glm_moe_dsa` | tensor_map deferred | unreachable |
| `gpt2` | — | `gpt2` | supported | supported |
| `granitemoe` | — | `granitemoe` | runtime deferred | supported |
| `hunyuan-dense` | `hunyuan_v1_dense` | `hunyuan_v1_dense` | supported | supported |
| `internlm2` | — | `internlm2` | supported | rejected |
| `llama` | `mistral` | `llama` | supported | supported |
| `mamba` | — | `mamba` | runtime deferred | rejected |
| `mamba2` | — | `mamba2` | runtime deferred | rejected |
| `muse-glimmer` | `muse_glimmer` | `muse_glimmer_text` | supported | supported |
| `nemotron` | — | `nemotron` | supported | supported |
| `nemotron_h_moe` | — | — | config rejected; tensor_map rejected; graph rejected; runtime rejected | unreachable |
| `olmo` | — | `olmo` | supported | supported |
| `olmo2` | — | `olmo2` | supported | supported |
| `olmoe` | — | `olmoe` | runtime deferred | supported |
| `phi3` | — | `phi3` | supported | supported |
| `phimoe` | — | `phimoe` | runtime deferred | supported |
| `qwen2` | — | `qwen2` | supported | supported |
| `qwen2moe` | `qwen2_moe` | `qwen2_moe` | runtime deferred | supported |
| `qwen3` | — | `qwen3` | supported | supported |
| `qwen35` | — | `qwen3_5_text` | supported | supported |
| `qwen35moe` | — | `qwen3_5_moe` | supported | supported |
| `qwen3moe` | `qwen3_moe` | `qwen3_moe` | runtime deferred | supported |
| `rwkv6` | — | — | config deferred; tensor_map deferred; graph deferred; runtime deferred | unreachable |
| `rwkv6qwen2` | — | — | config deferred; tensor_map deferred; graph deferred; runtime deferred | unreachable |
| `rwkv7` | — | — | config deferred; tensor_map deferred; graph deferred; runtime deferred | unreachable |
| `smollm3` | — | `smollm3` | supported | supported |
| `stablelm` | — | `stablelm` | supported | supported |
| `starcoder2` | — | `starcoder2` | supported | supported |
| `t5` | — | `t5` | tensor_map deferred | unreachable |

<!-- END GGUF SUPPORT MATRIX -->

Canonical names are the strings llama.cpp writes into `general.architecture`,
validated against a vendored census of the 147 architectures llama.cpp defines
at commit `8d9af256337d1a501250f9bbf4c0859a654bddd6`. Aliases are spellings
llama.cpp does not emit but that mobius still accepts.

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
