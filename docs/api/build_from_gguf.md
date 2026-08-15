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

The GGUF builder maps GGUF architecture names (e.g. `llama`, `qwen2`,
`gemma`) to the same model classes used by `build()`. Most decoder-only
LLM architectures are supported.

Sharded GGUF files are rejected. A single shard has only part of the tensor
table, and treating it as a complete checkpoint would create a corrupt model.

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
  .\nemotron-bf16-onnx
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
