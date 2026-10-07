# Nemotron 3.5 Lightning: pinned export preflight

**Metadata/source preflight only (2026-10-07); both full exports `NOT_RUN`.**
The [machine-readable recipe](../examples/nemotron_lightning/recipe.json)
records immutable inputs and compatibility limits, not a tested export or
validated toolchain. It adds no model implementation or NVFP4 support.

The original preflight host was Windows ARM64 with a Qualcomm Adreno X1-85,
no NVIDIA/CUDA GPU or `nvidia-smi`, Python 3.12.10, 68,171,038,720 bytes RAM
and approximately 818 GB free disk. Mobius, PyTorch, Transformers, ORT and
ORT-GenAI were absent. Only public metadata and exporter sources were
retrieved: no weight shards, exporter installation, remote model code
execution, GPU work, ONNX export or generation.

## Immutable inputs

| Input | BF16 | NVFP4 |
| --- | --- | --- |
| NVIDIA model | `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16` | `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4` |
| HF revision | `a9904d24bcc1d289a1950fa9d2b978c47cf903b9` | `bee7596271d1495f6992ae224aefde4410e816b8` |
| Index tensor bytes | 65,842,365,568 | 21,559,589,596 |
| Unique index shards | 14 | 52 |
| Config context limit | 262,144 | 1,048,576 |
| Source quantization | Unquantized BF16 | ModelOpt mixed FP8/NVFP4 |
| Classification | Architecture source present; unvalidated | Explicitly unsupported by pinned exporter |

Both configs declare `NemotronHForCausalLM` / `nemotron_h`, 52 layers
(23 Mamba, 23 MoE, 6 attention), 128 routed experts, top-6 routing, and one
NextN prediction layer. The 3B active-parameter label is **not** a
weight-residency requirement. Index sizes are metadata declarations, not
measured download sizes, output sizes, peak RAM or VRAM. Host RAM barely
exceeds BF16 tensor bytes before loader/graph/OS overhead; this does not
establish that a build fits.

The BF16 model card advertises up to 1M context, but its pinned config says
262,144. Do not silently change it. The recipe pins raw-byte SHA-256 hashes
for both configs/indexes and six inspected Mobius files at
`dc1798465f77cb2b7070e4d422479c2f890ee24b`. This is a source-evidence pin,
**not** an approved installation/dependency lock. Compatibility claims were
rechecked at `26a712b210a0a7cd895347f601680e82bbf5e452`: five files are
unchanged; `auto_export.py` differs only in unrelated Gemma4 audio settings.
The recorded source hashes remain those of the original inspected revision.

## Reproduce metadata/source verification

Run from the Mobius repository root in PowerShell. Set `$Evidence` to a
new private directory outside the checkout. This downloads only four JSON
metadata files and six exporter source files, never checkpoint shards.
It does not import/install Mobius or execute the downloaded source.
HTTP errors, hash mismatches and changed metadata stop the preflight.
Slashes in URL components and recipe source paths are not local paths.

```powershell
$ErrorActionPreference = 'Stop'
$Evidence = 'C:\private\nemotron-lightning-preflight'
if (Test-Path $Evidence) { throw 'Use a new evidence directory' }
New-Item -ItemType Directory -Path $Evidence | Out-Null
$Recipe = Get-Content .\examples\nemotron_lightning\recipe.json -Raw | ConvertFrom-Json
foreach ($Model in $Recipe.models) {
    $Variant = ($Model.model_id -split '-')[-1]
    foreach ($File in $Model.metadata) {
        $Dest = Join-Path $Evidence "$Variant-$($File.path)"
        $Url = "https://huggingface.co/$($Model.model_id)/resolve/$($Model.revision)/$($File.path)"
        Invoke-WebRequest -Uri $Url -OutFile $Dest
        if ((Get-FileHash $Dest -Algorithm SHA256).Hash.ToLowerInvariant() -ne $File.sha256) {
            throw "Metadata hash mismatch: $Variant $($File.path)"
        }
    }
    $Config = Get-Content (Join-Path $Evidence "$Variant-config.json") -Raw | ConvertFrom-Json
    $Index = Get-Content (Join-Path $Evidence "$Variant-model.safetensors.index.json") -Raw | ConvertFrom-Json
    $Shards = @($Index.weight_map.PSObject.Properties.Value | Sort-Object -Unique).Count
    if ($Index.metadata.total_size -ne $Model.checkpoint_tensor_bytes -or $Shards -ne $Model.checkpoint_shards) {
        throw "Checkpoint inventory mismatch: $Variant"
    }
    if ($Config.model_type -ne $Model.model_type -or
        $Config.architectures.Count -ne 1 -or $Config.architectures[0] -ne $Model.architecture -or
        $Config.num_hidden_layers -ne $Model.num_hidden_layers -or
        $Config.max_position_embeddings -ne $Model.max_position_embeddings -or
        $Config.num_nextn_predict_layers -ne $Model.num_nextn_predict_layers -or
        $Config.n_routed_experts -ne $Model.num_routed_experts -or
        $Config.num_experts_per_tok -ne $Model.num_experts_per_tok) {
        throw "Architecture/context mismatch: $Variant"
    }
    foreach ($Layer in $Model.layer_counts.PSObject.Properties) {
        $Count = @($Config.layers_block_type | Where-Object { $_ -eq $Layer.Name }).Count
        if ($Count -ne $Layer.Value) { throw "Layer inventory mismatch: $Variant $($Layer.Name)" }
    }
    "$Variant : metadata verified; export NOT_RUN; $($Model.compatibility)"
}
foreach ($File in $Recipe.exporter.source_evidence) {
    $Dest = Join-Path $Evidence ($File.path.Replace('/', '_'))
    $Url = "https://raw.githubusercontent.com/onnxruntime/mobius/$($Recipe.exporter.revision)/$($File.path)"
    Invoke-WebRequest -Uri $Url -OutFile $Dest
    if ((Get-FileHash $Dest -Algorithm SHA256).Hash.ToLowerInvariant() -ne $File.sha256) {
        throw "Exporter source hash mismatch: $($File.path)"
    }
}
'All ten pinned files verified; exports NOT_RUN'
```

This verifies inspected bytes and metadata, **not** executable compatibility.
Config/index hashes do not authenticate every shard or tokenizer asset.

## Source compatibility boundaries

These links identify the original inspected revision; the recheck above
does not change the scope of evidence.

| Source | Finding and limit |
| --- | --- |
| [`_registry.py`](https://github.com/onnxruntime/mobius/blob/dc1798465f77cb2b7070e4d422479c2f890ee24b/src/mobius/_registry.py) and [`_configs/_base.py`](https://github.com/onnxruntime/mobius/blob/dc1798465f77cb2b7070e4d422479c2f890ee24b/src/mobius/_configs/_base.py) | Nemotron-H registration and `NemotronHConfig.from_transformers` translate Mamba/attention/MoE layer names. This is not Lightning checkpoint parity. |
| [`models/nemotron_h.py`](https://github.com/onnxruntime/mobius/blob/dc1798465f77cb2b7070e4d422479c2f890ee24b/src/mobius/models/nemotron_h.py) | Hybrid layers, weight renaming and stacked-expert splitting exist, but no MTP implementation in this module. MoE evaluates all experts and masks contributions, not demonstrated sparse top-6 efficiency. Its source documents ReLU2 as incompatible with the described ORT 1.29 MoE/QMoE activation set; substituting ReLU is not equivalent. |
| [`_configs/_quantization.py`](https://github.com/onnxruntime/mobius/blob/dc1798465f77cb2b7070e4d422479c2f890ee24b/src/mobius/_configs/_quantization.py) | `QuantizationConfig.from_value` explicitly raises `NotImplementedError` for ModelOpt NVFP4/FP8. Reconstruction helpers do not wire checkpoint loading or native routed-expert NVFP4 emission. |
| [`integrations/ort_genai/auto_export.py`](https://github.com/onnxruntime/mobius/blob/dc1798465f77cb2b7070e4d422479c2f890ee24b/src/mobius/integrations/ort_genai/auto_export.py) | Generic recurrent packaging expects paired `conv_state` / `recurrent_state`; Nemotron-H describes `conv_state` / `ssm_state`. Actual graph names/shapes and runtime allocation are unverified, not evidence of a loadable package. |
| [`__main__.py`](https://github.com/onnxruntime/mobius/blob/dc1798465f77cb2b7070e4d422479c2f890ee24b/src/mobius/__main__.py) | The candidate below uses source-backed CLI syntax. `--revision` is only valid with `--model`, not local `--config`. |

## BF16 candidate procedure: NOT_RUN

Before any full export, obtain a suitable development host, explicit resource
budget, immutable dependency lock and CUDA/ORT version pairing. No package
installation, model download or GPU allocation is part of this preflight.
The publisher's 80 GB-class GPU deployment target is not an export-memory
measurement. Stage a complete read-only snapshot at the pinned HF revision:
config, tokenizer and every indexed weight shard. Verify shard identities,
tokenizer assets, disk capacity and loader peak memory separately. Do not
enable `trust_remote_code` or fetch mutable `main`.

On that separately prepared host, the candidate **graph-only** command is:

```powershell
mobius build --config C:\private\models\nemotron-lightning-bf16 `
    --output C:\private\exports\nemotron-lightning-bf16 `
    --ep cuda --dtype bf16 --external-data onnx
```

Use a fresh output directory; retain stdout, stderr, exit status, exact tool
versions and resource measurements. This command has **not** been executed
and may fail on Lightning-specific weights/semantics. It intentionally
omits `--runtime ort-genai`: an exported graph is not a loadable runtime
bundle. No FP16 conversion, MTP removal or generic decoder fallback is
implied by this recipe.

Before claiming success, account for every checkpoint tensor, including
MTP; validate graph/external-data references; compare source/export
full logits and deterministic multi-step cached decoding. A target-only
variant that omits MTP needs explicit scope, a separate name and validation,
not a full-checkpoint/MTP support claim. Verify Mamba convolution/SSM
state and attention cache initialization, update and reset, plus actual
CUDA node placement. Only afterward evaluate downstream ORT-GenAI
packaging, tokenizer/chat-template fidelity and generation. Record memory,
quality, generation settings and tested context limits, not model-card
throughput claims.

## NVFP4: stop before weights

The pinned ModelOpt config describes mixed FP8 Mamba projections and FP4
expert weights with group size 16. The exporter rejects this method:
**there is no supported NVFP4 build command in this recipe**. Do not
download 52 shards merely to reproduce that known source-level blocker.

Supporting it requires mixed-format/scales-aware loading, faithful ReLU2
expert execution, runtime kernel/build capability checks and real-weight
parity. Blackwell native FP4 and W4A16 reconstruction are distinct paths;
NVIDIA's vLLM support does not establish ORT support. BF16 reconstruction
or ORT INT4 / `MatMulNBits` quantization creates a different artifact and
must not be labeled a preserved NVFP4 export.
