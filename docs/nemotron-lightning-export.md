# Nemotron 3.5 Lightning: reconstruction and compatibility gates

**CPU-tested exporter improvements; full-model generation remains blocked.**
The [machine-readable recipe](../examples/nemotron_lightning/recipe.json)
records immutable inputs and compatibility limits. Native NVFP4 remains
unsupported. Explicit dense BF16 reconstruction is implemented and tested
with miniature mixed-format checkpoints, **not** the complete 30B models.
An explicitly selected **target-decoder-only** variant can omit NextN/MTP
with complete named reporting. A separate explicit `nemotron-h-mtp` task now
exports target and draft graphs with strict MTP coverage and tiny independent
CPU parity. This is **graph support**, not a speculative-generation runtime.

### Grouped Mamba normalization correction and H100 gate

User-supplied H100 evidence for export commit `897a0e05` reports exact source
weight fidelity passing, but full decoder numerical parity and eight-token
generation failing. One isolated failure is ORT
`1.28.0.dev20260722004` CUDA `RMSNormalization` with a two-dimensional grouped
scale. Scratch decomposition improves the decoder substantially but does not
clear its numerical or generation gates. Conditional MTP parity under those
decoder seeds is not end-to-end parity.

Nemotron's Mamba mixer now uses `GroupedGatedRMSNorm`, with explicit FP32
gating, per-group variance and normalization, followed by a cast to the input
dtype **before** native-dtype gamma multiplication. This matches the pinned
HF [`Zamba2RMSNormGated`](https://github.com/huggingface/transformers/blob/a96730c8c97b8efbf35bbaf7f5da33ec99231a49/src/transformers/models/zamba2/modeling_zamba2.py)
used by `NemotronHMamba2Mixer`. It avoids the faulty two-dimensional-scale
kernel without changing shared Mamba defaults, weight names, cache contracts
or task selection. The existing `MOBIUS_ORT_CUDA_GROUPED_RMSNORM_WORKAROUND`
flag is unchanged for other users; its FP32 gamma-before-cast ordering is
not the same as this source-faithful variant.

Focused regressions execute FP32/FP16 grouped math on CPU, including
production normalization dimensions (4096 channels, groups of 512), and
assert FP32/FP16/BF16 graph cast ordering. The component test also includes a
BF16 CUDA case, skipped when CUDA is unavailable. Run on the H100:

```bash
python -m pytest src/mobius/components/_rms_norm_test.py \
    -k grouped_gated_norm_source_precision_parity -q
```

The H100 gate is to re-export to a **fresh** directory with this correction,
verify the optimized graph does not recreate a two-dimensional-scale norm,
rerun the isolated CUDA norm check and full decoder prefill/cached-state
comparisons, then require complete generation-token agreement. Original
exports and checkpoint files must remain unchanged. The remaining decoder
discrepancy needs a first-divergent-stage comparison and an inspected source
reference; this correction alone is **not** a claim of resolved full parity.

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
| Current classification | Explicit target-only or target+MTP graphs; full weighted MTP untested | Explicit reconstructed-BF16 target-only or target+MTP graphs; native NVFP4 unsupported |

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
For the pinned Lightning sources, also select `target_decoder_only=True`
/ `--target-decoder-only`, or the separate `task="nemotron-h-mtp"` /
`--task nemotron-h-mtp` route. Neither MTP selection nor target-only selection
implies reconstruction; both remain explicit.

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
  The package registers lazy shard/config/index sources and both snapshot and
  resolved blob directories. Saving into source checkpoint directories or
  onto source-file aliases is rejected before serialization; use a fresh,
  separate output directory.
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

### NextN/MTP: default fail closed, two explicit export contracts

Model construction rejects nonzero `num_nextn_predict_layers`; eager and
streaming weight paths also reject `mtp.*` tensors even if a config omits that
field, **unless the public builder's separate target-only option or MTP task is explicit**.
The approved `target_decoder_only=True` / `--target-decoder-only` variant
preserves source `num_nextn_predict_layers`, uses strict streaming for both
floating per-expert and reconstructed weights, and excludes only `mtp.*` keys.
Every omitted name, source shape/dtype and reason is persisted in
`weight-loading-report.json` and ONNX metadata. Unexpected non-MTP tensors,
missing target weights and scale errors still fail. A source declaring NextN
but containing no MTP inventory fails when loading. The eager preprocessor
still rejects MTP; it is not an unreported omission path.

Graph names include `/target-decoder-only/`, metadata records
`mobius.export_variant=target-decoder-only` and `mobius.mtp_preserved=false`,
and logs warn about omissions. With `--no-weights`, inventory status is
`not_inspected`: no omitted-tensor coverage is claimed from config alone.
The [inspected HuggingFace reference](https://github.com/huggingface/transformers/blob/a96730c8c97b8efbf35bbaf7f5da33ec99231a49/src/transformers/models/nemotron_h/modeling_nemotron_h.py)
ignores MTP; the [inspected vLLM MTP implementation](https://github.com/vllm-project/vllm/blob/ef63c23d35acccbba8e014fc88da19d541da38a9/vllm/model_executor/models/nemotron_h_mtp.py)
and its [parent layer definitions](https://github.com/vllm-project/vllm/blob/ef63c23d35acccbba8e014fc88da19d541da38a9/vllm/model_executor/models/nemotron_h.py)
provide the MTP reference. **The earlier shared-expert-absence statement was
incorrect:** both pinned indexes contain
`mtp.layers.1.mixer.shared_experts.up_proj.weight` and
`mtp.layers.1.mixer.shared_experts.down_proj.weight`.
Independent bounded reads of BF16 shard 14 and NVFP4 shard 52 headers confirm
their shapes are `[3712,2688]` and `[2688,3712]`. No missing weights or
architectural substitutions are needed.

### Explicit target + Lightning MTP graph package

`build(..., task="nemotron-h-mtp")` produces `decoder/model.onnx` and
`mtp/model.onnx`. The default constructor guard and existing target-only
contract are unchanged. In PowerShell, for the pinned BF16 checkpoint:

```powershell
mobius build --model nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 `
    --revision a9904d24bcc1d289a1950fa9d2b978c47cf903b9 `
    --task nemotron-h-mtp --dtype bf16 --ep cuda `
    --external-data onnx --output C:\private\lightning-target-and-mtp
```

For the NVFP4 model, use its pinned revision above and **also** add
`--dequantize --dtype bf16`; native requests still fail. These full-size
commands are **NOT_RUN here**. They are not instructions to overwrite the
checkpoint directory, nor evidence of H100 parity.

The source architecture is one NextN prediction step containing two physical
blocks, `attention` then `moe`. The first block independently RMS-normalizes
the next-token embedding and target hidden state, concatenates them in that
order, projects `2H -> H`, then applies pre-norm GQA and residual addition.
The second applies pre-norm sigmoid/correction-bias top-6 ReLU2 MoE, including
its own always-active shared expert, residual addition, and final RMSNorm.
There is **no RoPE, position-ID input, Mamba block, or SSM state in MTP**.
Both graphs borrow the same source embedding and LM-head tensors; serialized
graphs contain separate copies rather than an undocumented runtime-sharing ABI.

| Pinned MTP source | Shape / inventory |
| --- | --- |
| `mtp.layers.0.{enorm,hnorm,norm}.weight` | `[2688]` each |
| `mtp.layers.0.eh_proj.weight` | `[2688,5376]` |
| `mtp.layers.0.mixer.{q,k,v,o}_proj.weight` | `[4096,2688]`, `[256,2688]`, `[256,2688]`, `[2688,4096]` |
| `mtp.layers.1.{norm,final_layernorm}.weight` | `[2688]` each |
| `mtp.layers.1.mixer.gate.{weight,e_score_correction_bias}` | `[128,2688]`, `[128]` |
| `mtp.layers.1.mixer.experts.0..127.{up,down}_proj.weight` | 256 tensors: `[1856,2688]`, `[2688,1856]` |
| `mtp.layers.1.mixer.shared_experts.{up,down}_proj.weight` | `[3712,2688]`, `[2688,3712]` |

Both independent headers have exactly **270 MTP tensors**, all BF16 except
the FP32 correction bias. The pinned NVFP4 config has **zero MTP quantized
module targets**, and its MTP index/header has **zero MTP scale tensors**.
The target's reconstructed LM head is reused for the MTP graph; this does not
simulate the original quantized activations or KV cache.

**Decoder-to-MTP contract:** the explicit decoder emits `mtp_seed`,
`[batch,sequence,2688]`, **after the target's final `norm_f`**, exactly the
hidden tensor feeding its LM head. Pair `mtp_seed[:,i]` (target `h_i`) with
`input_ids[:,i+1]` (`t_(i+1)`) in the draft graph. Draft logits predict
`t_(i+2)`; the graph does not shift tokens itself. For prompt initialization,
feed known shifted pairs `ids[:,1:]` and `seed[:,:-1]`. For the next draft
step, pair the last target seed with the sampled next target token. Further
recursive drafting may pair `mtp_hidden` with a sampled draft token.

The MTP graph consumes `input_ids` INT64 `[batch,sequence]`, `hidden_states`
in model dtype `[batch,sequence,2688]`, `attention_mask` INT64
`[batch,past+sequence]`, and `past_key_values.0.{key,value}` in model dtype
`[batch,2,past,128]`. It emits `logits` `[batch,sequence,131072]`,
`mtp_hidden` `[batch,sequence,2688]`, and `present.0.{key,value}` with
`past+sequence` length. Initialize draft KV with zero sequence length;
**never borrow target-layer KV**. Target hybrid caches remain in their
existing sparse layer-index namespaces.

Strict streaming validates the union of both graph source inventories before
returning the package. Every `mtp.*` source must bind to the draft graph;
missing, extra, duplicate-mapped and malformed tensors fail. The package
report records MTP names, shared table copies and per-component partition
counts, with **zero package-level MTP omissions**. Metadata with `--no-weights`
marks inventory `not_inspected`; graph construction alone is not tensor coverage.
Lazy source-file/directory protection applies to both graphs and package roots.

**Validated scope:** tiny FP32 CPU independent-equation parity for prefill,
cached one-token and multi-token blocks, two rows with different padding,
FP32 and BF16 source weights, and the actual exported target-to-MTP bridge.
Tiny FP16 CPU draft prefill and cached decode also pass independent
FP32-reference comparison at `rtol=atol=1e-2`.
Tiny tests also cover mixed FP8/NVFP4 reconstruction with floating MTP and a
shared reconstructed head, exact graph/cache I/O, f32/f16/bf16 graph dtypes,
CLI serialization, strict source accounting and checkpoint-save protection.
BF16 execution/parity, full weighted 30B MTP, CUDA/H100,
performance, Olive requantization and L4/L5 real-weight goldens are **NOT_RUN**;
no full-model numerical or speedup claim is made.

ORT-GenAI metadata can describe the external coordination contract and marks
it `runtime_unvalidated`. The exporter does **not** implement speculative
acceptance/verification. A generation runtime still needs shifted token/hidden
pairing, independent draft-KV management, target block verification/sampling,
accepted-prefix commit, and rejected-state restore/replay for **both** draft
KV and target KV/Mamba conv/SSM states. Existing generic acceptance/rollback
graphs do not establish Nemotron-H hybrid-state orchestration.

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

## Target-decoder-only procedures: implemented, full-size NOT_RUN

Before any full export, obtain a suitable development host, explicit resource
budget, immutable dependency lock and CUDA/ORT version pairing. No package
installation, model download or GPU allocation is part of this preflight.
The publisher's 80 GB-class GPU deployment target is not an export-memory
measurement. With explicit target-decoder-only scope, stage a complete read-only
snapshot at the pinned HF revision:
config, tokenizer and every indexed weight shard. Verify shard identities,
tokenizer assets, disk capacity and loader peak memory separately. Do not
enable `trust_remote_code` or fetch mutable `main`.

The following commands are **unexecuted full-size target-only candidates**,
not approved GPU work or successful exports. The local snapshot paths must
contain the verified pinned variants respectively. Do not edit config to zero
NextN or remove tensors before loading: the opt-in loader reports all omissions.

```powershell
mobius build --config C:\private\lightning-bf16-pinned `
    --output C:\private\lightning-target-decoder-only-bf16 `
    --ep cuda --dtype bf16 --target-decoder-only --external-data onnx
mobius build --config C:\private\lightning-nvfp4-pinned `
    --output C:\private\lightning-target-decoder-only-reconstructed-bf16 `
    --ep cuda --dtype bf16 --target-decoder-only --dequantize --external-data onnx
```

Use fresh output directories and retain logs, exit status, exact versions and
resource measurements. Omit `--runtime ort-genai` until that separate runtime
contract is validated. `--revision` is only valid with `--model`, not `--config`.

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
reconstruction route handles these physical formats, and the independent
target-only option accounts for omitted MTP tensors. **There is no native NVFP4
or faithful full-MTP build command in this recipe.** The candidate above
produces a differently represented, target-only artifact; do not download 52
shards merely to reproduce the known native/full-MTP blockers.

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
accounting, malformed/changing headers, dtype/cache contracts, default MTP
refusal and explicit omission/variant provenance.
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
