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
Quantized GGUFs containing only qtypes with no supported preservation target
(for example, pure Q6_K or Q5_K weights) fail with an actionable error rather
than silently becoming float. Pass `keep_quantized=False` to request that float
conversion explicitly.

## Supported GGUF Architectures

Support is declared once, in
`mobius/integrations/gguf/_arch_registry.py`, as four independent verdicts per
architecture — config extraction, tensor mapping, graph construction, and
runtime packaging. Anything short of "supported" carries a reason, and the
table below is checked against the registry by
`_arch_registry_test.py::TestDocumentedSupportMatrix`, so it cannot drift.

Being listed is not a support claim. `build_from_gguf` refuses every
architecture that is not fully supported, naming the missing capability and the
reason.

<!-- BEGIN GGUF SUPPORT MATRIX (generated; see _arch_registry.py) -->

| GGUF architecture | Accepted aliases | mobius `model_type` | Status |
|---|---|---|---|
| `arcee` | — | `arcee` | runtime deferred |
| `bloom` | — | `bloom` | tensor_map deferred |
| `clip` | — | — | config rejected; tensor_map rejected; graph rejected; runtime rejected |
| `cohere2` | — | `cohere2` | runtime deferred |
| `deci` | — | `llama` | supported |
| `deepseek4` | — | `deepseek_v4` | supported |
| `exaone` | — | `exaone` | runtime deferred |
| `falcon` | — | `falcon` | supported |
| `gemma` | — | `gemma` | supported |
| `gemma2` | — | `gemma2` | supported |
| `gemma3` | — | `gemma3_text` | supported |
| `gemma4` | — | `gemma4_text` | supported |
| `glm-dsa` | `glm_dsa` | `glm_moe_dsa` | tensor_map deferred |
| `gpt2` | — | `gpt2` | supported |
| `hunyuan-dense` | `hunyuan_v1_dense` | `hunyuan_v1_dense` | supported |
| `internlm2` | — | `internlm2` | supported |
| `llama` | `mistral` | `llama` | supported |
| `mamba` | — | `mamba` | supported |
| `muse-glimmer` | `muse_glimmer` | `muse_glimmer_text` | supported |
| `nemotron` | — | `nemotron` | supported |
| `nemotron_h_moe` | — | — | config rejected; tensor_map rejected; graph rejected; runtime rejected |
| `olmo` | — | `olmo` | supported |
| `olmo2` | — | `olmo2` | supported |
| `phi3` | — | `phi3` | supported |
| `qwen2` | — | `qwen2` | supported |
| `qwen2moe` | `qwen2_moe` | `qwen2_moe` | supported |
| `qwen3` | — | `qwen3` | supported |
| `qwen35` | — | `qwen3_5_text` | supported |
| `qwen35moe` | — | `qwen3_5_moe` | supported |
| `qwen3moe` | `qwen3_moe` | `qwen3_moe` | supported |
| `smollm3` | — | `smollm3` | supported |
| `stablelm` | — | `stablelm` | supported |
| `starcoder2` | — | `starcoder2` | supported |
| `t5` | — | `t5` | tensor_map deferred |

<!-- END GGUF SUPPORT MATRIX -->

Canonical names are the strings llama.cpp writes into `general.architecture`,
validated against a vendored census of the 147 architectures llama.cpp defines
at commit `8d9af256337d1a501250f9bbf4c0859a654bddd6`. Aliases are spellings
llama.cpp does not emit but that mobius still accepts.

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
at the same pinned commit and answers four separate questions per slot:
whether the GGUF parse layer can read it, whether it can be dequantized to
float, whether its blocks are handed to the runtime unchanged, and whether it
can be repacked into a `MatMulNBits` affine layout.

- **Repacked to `MatMulNBits`**: `Q4_0`, `Q4_1`, `Q8_0`, `Q4_K`, `Q6_K`, `Q1_0`.
- **Preserved byte-for-byte**: `MXFP4`, `IQ4_NL`, `IQ4_XS`, `IQ3_S`, `IQ3_XXS`,
  `IQ2_XXS`, `IQ2_XS`, `IQ2_S`, `IQ1_S`, `IQ1_M`.
- **Rejected**: the eight retired slots (`Q4_2`, `Q4_3`, `Q4_0_4_4`,
  `Q4_0_4_8`, `Q4_0_8_8`, `IQ4_NL_4_4`, `IQ4_NL_4_8`, `IQ4_NL_8_8`), which have
  a block size of 0 and are unreadable at the GGUF parse layer, and `Q8_1` /
  `Q8_K`, which are compute-only intermediates that are never valid weight
  storage.


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
