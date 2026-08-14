# Nemotron 3.5 Lightning: BF16 checkpoint + Olive

This is **Option A** for
[`nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16):
export the official BF16 checkpoint to supported FP16 ONNX, quantize the model
with Olive, assemble a direct ONNX Runtime package, then run cached generation.

Every Hub access is pinned to revision
`d468880b6ad3c6e0d21377ce7242adaea4cc884d`.

## Architecture and runtime contract

The checkpoint is a real `nemotron_h` model, not an alias:

- 52 base-decoder layers mixing Mamba2, sigmoid-routed MoE, and full GQA.
- 128 routed experts with top-6 selection and one shared expert.
- `mtp.*` contains 270 auxiliary multi-token-prediction tensors. This export
  intentionally targets the base `NemotronHForCausalLM` decoder; its forward
  graph does not instantiate MTP, and upstream marks those keys unexpected.
  No base-decoder generation input, cache, logit, or weight depends on them.

ONNX Runtime GenAI 0.15.2 cannot bind this model's mixed cache: Mamba layers
need `conv_state` plus `ssm_state`, while sparse full-attention layers need
key/value caches at global layer indices. Mobius rejects this graph contract
structurally because `genai_config.json` has no `ssm_state` input/output
template; it does not hard-code a NemotronH or runtime-version check. The guard
can be removed when config emission represents that state. This validated
package uses direct ONNX Runtime generation through `inference.py`.

## Install

From the repository root:

```powershell
python -m pip install -e ".[transformers,testing]" `
  --index-url https://packagefeedproxy.microsoft.io/pypi/simple
python -m pip install -r examples\olive\nemotron-3_5-lightning-30b\requirements.txt `
  --index-url https://packagefeedproxy.microsoft.io/pypi/simple
```

Use an ONNX Runtime GPU build with CUDA 12 and cuDNN 9 for CUDA inference.

## Full export, quantization, and smoke test

```powershell
cd examples\olive\nemotron-3_5-lightning-30b
python optimize.py `
  --source-dir output\f16\cuda `
  --output-dir output\Q4_K_M\cuda `
  --ep cuda `
  --precision q4_k_m
```

The script performs four gated steps:

1. Downloads the exact 14-shard BF16 checkpoint revision and exports FP16 ONNX.
   BF16 execution is rejected explicitly because corrected reduced-real parity
   reaches `0.8594` max logit error, above the `1e-2` reduced-precision gate.
2. Applies CUDA GQA/LinearAttention fusion and the grouped-RMSNorm workaround.
3. Runs Olive Q4 K-quant with an explicitly CPU-only target. It also suppresses
   Olive 0.13's unrelated GPU-EP DLL auto-registration, so a missing TensorRT
   installation cannot abort CPU weight-only quantization.
4. Reloads the assembled quantized package and generates four cached tokens.

To reuse an existing source package:

```powershell
python optimize.py --skip-export `
  --source-dir output\f16\cuda `
  --output-dir output\Q4_K_M\cuda `
  --ep cuda
```

`olive_q4.json` records the equivalent pass and provider-isolated target. Use
`optimize.py` rather than invoking the JSON directly when the installed ORT
wheel bundles unconfigured providers; the script contains the verified Olive
0.13 registration isolation.

## Package layout

```text
output/
├── f16/cuda/
│   ├── model.onnx
│   ├── model.onnx.data
│   ├── config.json
│   ├── generation_config.json
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   └── source_manifest.json
└── Q4_K_M/cuda/
    ├── model.onnx
    ├── model.onnx.data
    ├── config.json
    ├── generation_config.json
    ├── tokenizer.json
    ├── tokenizer_config.json
    └── source_manifest.json
```

This recipe intentionally omits `genai_config.json`: direct ONNX Runtime is
the validated runtime, and core Mobius rejects the currently unrepresentable
SSM cache contract before creating runtime artifacts.

## Direct generation and profiling

```powershell
python inference.py `
  --model-dir output\Q4_K_M\cuda `
  --device cuda `
  --prompt "What is 84 * 3 / 2?" `
  --max-new-tokens 20 `
  --profile
```

The script initializes every cache from the saved graph, processes the prompt
token by token, carries Mamba and KV state independently, and fails if CUDA was
requested but not registered.

## Reduced real-checkpoint validation

The full checkpoint is 65.8 GB and cannot execute on the validation host's
8 GB RTX A1000. The reproducible reduced check range-downloads 236 MiB of real
weights while retaining production dimensions:

- checkpoint layer 0: complete Mamba2 block;
- layer 1: router, shared expert, and four complete routed experts;
- layer 5: complete full-attention block;
- sliced real embedding and LM-head rows plus final norm.

```powershell
python validate_reduced_checkpoint.py
```

The fixture is stored persistently under `~/.cache/mobius/` by default. Each
range request validates status, `Content-Range`, declared length, and payload
length, with three bounded attempts (1s then 2s backoff). The cache metadata
must match the pinned model, revision, and fixture schema; writes are atomic.
GPU CI restores the same revision/schema-keyed cache for L4 and L5.

The supported matrix is intentionally limited to FP32/CPU and FP16/CUDA.
Reproduce the BF16 rejection evidence separately without creating a supported
package or weakening the production guard:

```powershell
python validate_reduced_checkpoint.py --bf16-rejection-evidence
```

Validated results on ORT 1.28.0 / Olive 0.13.0:

| Variant | Full-logit max abs | Generated IDs | Placement |
|---|---:|---|---|
| FP32 CPU | `9.54e-6` | `12, 13, 12, 12` | CPU |
| FP16 CUDA | `0.00977` | `12, 13, 12, 12` | 833 CUDA / 14 CPU events |
| BF16 CUDA | rejected (`0.8594`) | N/A | fails numerical gate |
| Olive Q4 | quantized | `12, 13, 12, 12` | 833 CUDA / 14 CPU events |

The reduced FP16 package is 247,380,256 bytes; Q4 is 73,190,965 bytes
(`0.296x`). Its weighted graph contains 15 `com.microsoft::MatMulNBits`
nodes and reloads successfully for multi-token generation.

## Evidence-based waivers

- Full-checkpoint L4/L5 coherent-text generation: requires roughly 66 GB just
  for checkpoint storage and substantially more than 8 GB accelerator memory.
- Full 30B Olive run: the recipe and reduced production-dimension pass are
  validated; completing all 2,944 expert subgraphs requires a large-memory
  host.
- Foundry Local: not available on this host, and its ORT GenAI-based model
  contract cannot represent NemotronH hybrid state today.
