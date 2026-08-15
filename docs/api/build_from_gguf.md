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

## NVIDIA Nemotron 3.5 Lightning Q8_0

Mobius supports the pinned single-file `Q8_0` GGUF production slice:

- Repository: `unsloth/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF`
- Revision: `f2d3fe3694501008786e81e5f20360cbf715496a`
- File: `NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q8_0.gguf`
- Size: `35,004,643,392` bytes
- SHA-256: `dc5276dd0619c04e277504d2358a793e31ccbe39e894d767d0d14f2a221e2ca4`

The architecture adapter validates the complete 417-tensor header before graph
construction. It maps exactly 401 backbone sources, explicitly excludes the
16 tensors in auxiliary MTP block 52, and produces 6,243 logical decoder
weights. The backbone schedule must be exactly 23 Mamba + 23 MoE + 6 attention
layers. Block 52 remains a separate combined attention+MoE MTP block rather
than becoming a false 53rd decoder layer.

### Pinned build

```powershell
$repo = "unsloth/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF"
$revision = "f2d3fe3694501008786e81e5f20360cbf715496a"
$file = "NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q8_0.gguf"
$official = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"
$officialRevision = "d468880b6ad3c6e0d21377ce7242adaea4cc884d"

python -m pip install `
  --index-url https://packagefeedproxy.microsoft.io/pypi/simple `
  huggingface_hub
hf download $repo $file --revision $revision --local-dir .\nemotron-gguf
hf download $official tokenizer.json tokenizer_config.json `
  special_tokens_map.json chat_template.jinja `
  --revision $officialRevision --local-dir .\nemotron-tokenizer

python -m mobius build-gguf ".\nemotron-gguf\$file" `
  --ep cpu `
  --external-data onnx --output .\nemotron-gguf-onnx
Copy-Item .\nemotron-tokenizer\tokenizer_config.json .\nemotron-gguf-onnx\
Copy-Item .\nemotron-tokenizer\special_tokens_map.json .\nemotron-gguf-onnx\
Copy-Item .\nemotron-tokenizer\chat_template.jinja .\nemotron-gguf-onnx\
```

The Q8 blocks are affine-repacked exactly into
`MatMulNBits(bits=8, block_size=32)`. Routed expert tensors are expanded along
their leading expert axis without dequantization. The resulting graph has
6,005 `MatMulNBits` nodes and one `GatherBlockQuantized` embedding node, with
no `QuantizeLinear`/`DequantizeLinear` round trip.

On the 63.3 GiB Windows acceptance host, the pinned build completed in
227.001 seconds plus 145.792 seconds to save. The package is
36,920,438,736 bytes and the build process peaked at 54,818,070,528 bytes of
working set. The weighted graph contains all 18,255 mapped weight
initializers, including 6,006 Q8 weights.

### Tokenizer and runtime contract

The embedded GPT-2/Pixtral vocabulary and merges are reconstructed as a
ByteLevel tokenizer. The source metadata's padding ID 999 is rejected because
it names `<SPECIAL_999>`, not the model's runtime padding convention. The
package keeps two explicit contracts:

- Pinned tokenizer asset: BOS 1 and `<|im_end|>` as EOS/padding ID 11.
- Direct-ORT model contract: padding ID 0 and EOS IDs `[2, 11]`.

The reconstructed tokenizer has the exact official vocabulary, pre-tokenizer,
decoder, post-processor, special-token flags, and encode/decode behavior. The
GGUF's embedded chat template is not the template at the pinned official
revision, so it is not silently emitted as authoritative. The recipe copies
the official `tokenizer_config.json`, `special_tokens_map.json`, and
`chat_template.jinja` sidecars by immutable revision instead.

The direct-ORT runner uses unpadded prompts. Its validation also compares every
real-token logit from an unpadded prefill with the same prompt followed by
right padding and an explicit attention mask. It never continues cached
generation from recurrent states advanced through padding.

Generic ORT GenAI configuration cannot currently bind arbitrary non-KV
recurrent cache inputs such as the graph's convolution and SSM states. Use
direct ONNX Runtime generation rather than treating session creation as
generation evidence.

### Reproduce semantic acceptance

The validator builds and saves in one process, then loads and generates in a
fresh process. It records package/operator/initializer counts, mapping
completeness, runtime versions, timings, and peak working sets:

```powershell
python examples\olive\nemotron-3_5-lightning-30b\validate_gguf_q8.py `
  --phase all `
  --gguf ".\nemotron-gguf\$file" `
  --official-tokenizer-dir .\nemotron-tokenizer `
  --output .\nemotron-gguf-onnx `
  --device cpu
```

The independent greedy reference was produced by llama.cpp commit
`9d57ce456c94d241dde672b2db9cf18879766568` from prompt
`The capital of France is`:

```text
llama-server -m NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q8_0.gguf \
  -c 128 -t 12 -tb 12 -b 64 -ub 64 -ngl 0 \
  --host 127.0.0.1 --port 18081 --no-warmup
POST /completion
{"prompt":"The capital of France is","n_predict":8,"temperature":0,
 "seed":1,"cache_prompt":false,"n_probs":1}
```

- Prompt IDs: `[1784, 8961, 1307, 5498, 1395]`
- Generated IDs: `[6993, 1046, 1256, 1010, 1784, 8961, 1307, 10787]`
- Text: ` Paris.  \nThe capital of Germany`
- Throughput: 5.58 prompt tok/s and 5.95 generated tok/s
- Peak working set from a separate CLI run: 16.81 GiB

The fresh-process ONNX Runtime 1.28.0 CPU run at commit `45de2a8b06`
loaded the package in 38.329 seconds and completed the five-token prefill in
99.447 seconds. Its seven cached decode calls ran at 0.01617 steps/s and the
process peaked at 38,148,943,872 bytes of working set. The right-padded
explicit-mask check had zero maximum absolute difference across all real-token
logits, and cached greedy generation matched every llama.cpp token and the
decoded text above.

### Explicitly unsupported variants

| GGUF file | Relevant tensor inventory | Status |
|---|---|---|
| `...-Q8_0.gguf` | 32.904B parameters in `Q8_0` | Supported through exact affine repacking |
| `...-MXFP4_MOE.gguf` | 14.687B `MXFP4`, 12.772B `Q5_1`, 5.445B `Q8_0` | Rejected; the large `Q5_1` source has no validated preserved runtime mapping |
| `...-UD-Q4_K_M.gguf` | 15.326B `Q5_0`, 12.772B `Q5_1`, 4.806B `Q8_0` | Rejected; the preset name does not describe its large source qtypes |
| `BF16/...-0000*-of-00002.gguf` | 329 tensors in shard 1 and 88 tensors in shard 2 | Rejected by the generic shard-assembly guard |

Dequantizing and requantizing Q5-family tensors would change their
quantization and is not described as preservation.
