# Nemotron 3.5 Lightning: reconstruction and compatibility gates

**CPU-tested exporter improvements; both full Lightning exports `NOT_RUN`.**
The [machine-readable recipe](../examples/nemotron_lightning/recipe.json)
records immutable inputs and compatibility limits. Native NVFP4 remains
unsupported. Explicit dense BF16 reconstruction is implemented and tested
with miniature mixed-format checkpoints, **not** the complete 30B models.
The pinned checkpoints' NextN/MTP contract remains a blocking gate.

The original preflight host was Windows ARM64 with a Qualcomm Adreno X1-85,
no NVIDIA/CUDA GPU or `nvidia-smi`, Python 3.12.10, 68,171,038,720 bytes RAM
and approximately 818 GB free disk. Mobius, PyTorch, Transformers, ORT and
ORT-GenAI were absent. Only public metadata and exporter sources were
retrieved: no weight shards, exporter installation, remote model code
execution, GPU work, ONNX export or generation during that initial preflight.
Subsequent code validation used an isolated x64 CPU Python environment on
the same ARM64 machine. Only metadata and bounded safetensors headers were
retrieved from the real checkpoints; no tensor payloads were downloaded.

## Immutable inputs

| Input | BF16 | NVFP4 |
| --- | --- | --- |
| NVIDIA model | `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16` | `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4` |
| HF revision | `a9904d24bcc1d289a1950fa9d2b978c47cf903b9` | `bee7596271d1495f6992ae224aefde4410e816b8` |
| Index tensor bytes | 65,842,365,568 | 21,559,589,596 |
| Unique index shards | 14 | 52 |
| Config context limit | 262,144 | 1,048,576 |
| Source quantization | Unquantized BF16 | ModelOpt mixed FP8/NVFP4 |
| Current full-checkpoint classification | Blocked on NextN/MTP | Native unsupported; dense reconstruction blocked on NextN/MTP |

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
The historical source table below does not describe the new reconstruction
implementation.

## Implemented exporter behavior

The current code adds an explicit Nemotron-H dense reconstruction route:
`build(..., keep_quantized=False, dtype="bf16")`, corresponding to
`--dequantize --dtype bf16`. Both options are required; implicit dtype,
FP16, native requests and unsupported layouts fail before weight loading.
This is **not a working full-Lightning command while the MTP gate remains**.

- `integrations/modelopt/_config.py` validates exact FP8/NVFP4 module groups,
  static block-16 format, redundant `quantized_layers` declarations and FP8
  activation/KV-cache schemes. Unknown formats and inconsistent inventories
  fail rather than falling back to affine integer INT4.
- `integrations/modelopt/_weights.py` reuses the numeric helpers to reconstruct
  one projection at a time, with 256-row temporary buffers and one final BF16
  rounding. It validates physical/logical shapes, scale types/layouts, positive
  finite scalar scales, nonnegative finite block scales and nonfinite results.
  Zero block scales reconstruct zero blocks. Lazy loading rechecks
  shard headers before materialization.
- `integrations/transformers/_nemotron_h_weights.py` maps per-expert sources
  exactly, rejecting collisions, extra/unclassified tensors, missing weights
  and missing scales. FP8 activation and KV-cache scales are explicitly
  validated/accounted for, **not simulated**. Export metadata and the loading
  report identify the reconstructed representation and lost quantization.
- `models/nemotron_h.py` casts router weights/bias to FP32 and accumulates routed
  outputs in FP32 before the shared-expert path, retaining sigmoid selection,
  unbiased routing weights and squared ReLU. It also retains FP32 router
  parameters during dtype conversion: the pinned NVFP4 router weight is FP32,
  unlike BF16's BF16 router weight; correction biases are FP32 in both.
  Widening BF16 sources is exact, while rounding NVFP4's FP32 router to BF16
  would lose unquantized source precision. All-expert graph unrolling remains;
  this is not sparse top-6 performance or native QMoE support.
- `NemotronHConfig` accepts Transformers' `linear_attention` vocabulary and
  preserves the checkpoint's FP32 SSM-cache dtype. The graph still uses
  `conv_state`/`ssm_state`; no ORT-GenAI recurrent ABI conversion is added.

The complete pinned ModelOpt config declares **46 FP8 projections, 5,935
NVFP4 projections, 46 FP8 input scales and 12 FP8 KV-cache scales**. A bounded
header probe verifies KV-cache scales are FP32 `[1]`; NVFP4 projection global
scales are FP32 scalars and block scales are E4M3 `[N,K/16]`. BF16's index has
6,243 non-MTP tensor names mapping uniquely and **270 `mtp.*` tensors**;
the NVFP4 index has the same MTP inventory. Name accounting and selected
headers do not validate all tensor values or demonstrate full-model parity.

**Dense BF16 reconstruction removes native FP8/NVFP4 storage, FP8 activation
quantization and FP8 KV-cache quantization.** It is neither native NVFP4
preservation nor re-quantized ORT INT4, and does not promise equivalence to
the original quantized execution.

### NextN/MTP: fail closed

Model construction rejects nonzero `num_nextn_predict_layers`; eager and
streaming weight paths also reject `mtp.*` tensors even if a config omits that
field. No automatic target-only fallback or discarded auxiliary tensors are
allowed. The [inspected HuggingFace reference](https://github.com/huggingface/transformers/blob/a96730c8c97b8efbf35bbaf7f5da33ec99231a49/src/transformers/models/nemotron_h/modeling_nemotron_h.py)
ignores MTP; the [inspected vLLM MTP implementation](https://github.com/vllm-project/vllm/blob/ef63c23d35acccbba8e014fc88da19d541da38a9/vllm/model_executor/models/nemotron_h_mtp.py)
appears to construct shared experts absent from the pinned MTP indexes.
That discrepancy is not resolved by inventing missing weights.
A separately authorized, explicitly named target-only variant or a validated
faithful MTP graph is required before these complete checkpoints can build.

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

## Full BF16 procedure: blocked, NOT_RUN

Before any full export, obtain a suitable development host, explicit resource
budget, immutable dependency lock and CUDA/ORT version pairing. No package
installation, model download or GPU allocation is part of this preflight.
The publisher's 80 GB-class GPU deployment target is not an export-memory
measurement. Only after the MTP gate is resolved, stage a complete read-only
snapshot at the pinned HF revision:
config, tokenizer and every indexed weight shard. Verify shard identities,
tokenizer assets, disk capacity and loader peak memory separately. Do not
enable `trust_remote_code` or fetch mutable `main`.

The earlier graph-only candidate is no longer presented as runnable:
the pinned config declares one NextN layer, and current construction explicitly
rejects that unsupported contract before loading shards. The recipe therefore
contains no approved full-checkpoint build command. Do not edit the config to
zero NextN or remove MTP tensors to bypass this gate. Any future command must
use a fresh output directory and retain logs, exit status, exact versions and
resource measurements; it must omit `--runtime ort-genai` until that separate
runtime contract is validated.

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

## NVFP4: native blocked; dense reconstruction is distinct

The pinned ModelOpt config describes mixed FP8 Mamba projections and FP4
expert weights with group size 16, plus static FP8 activations and KV cache.
The generic/native exporter still rejects this method. The opt-in dense
reconstruction route above handles these physical formats but remains blocked
on full Lightning MTP accounting. **There is no supported native NVFP4 or
full-Lightning build command in this recipe.** Do not download 52 shards merely
to reproduce that known contract blocker.

Native preservation additionally requires faithful ReLU2/routing kernel
semantics, runtime capability checks and real-weight parity.
Blackwell native FP4 and dense BF16 reconstruction are distinct paths;
NVIDIA's vLLM support does not establish ORT support. BF16 reconstruction
or ORT INT4 / `MatMulNBits` quantization creates a different artifact and
must not be labeled a preserved NVFP4 export.

## Offline regression validation

The added suites use reduced configurations, independently specified routing
and E2M1/E4M3 values, and local synthetic safetensors files. They exercise public
builder/CLI reconstruction, save/reload, every layer/projection family, source
accounting, malformed/changing headers, dtype/cache contracts and MTP refusal.
Existing CPU random-weight HuggingFace Nemotron parity tests also run without
model downloads. These are not full-checkpoint L4/L5 evidence.

```powershell
python -m pytest src\mobius\integrations\modelopt `
    src\mobius\models\_nemotron_h_test.py `
    src\mobius\integrations\transformers\_nemotron_h_weights_test.py -q --tb=short
python -m pytest tests\synthetic_parity_test.py -k nemotron_h -q -ra --tb=short
```

The validated isolated CPU versions were PyTorch `2.14.1+cpu` (no CUDA build),
Transformers `5.18.0`, ONNXScript `0.7.2`, ONNX IR `1.0.0` and ORT `1.30.0`.
These are execution evidence, not a recommended GPU dependency lock.
Full 30B export/parity/generation, BF16 CUDA kernels, native NVFP4,
MTP execution, resource fit and ORT-GenAI packaging remain unrun/unvalidated.
