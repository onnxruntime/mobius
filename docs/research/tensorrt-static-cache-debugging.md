# Debugging TensorRT Static-Cache Decode: From Symptom to Fix

This is a worked debugging case, recorded on 2026-09-15, for a Mobius-exported
Qwen3-0.6B model. It follows the evidence in the order we collected it: an engine
that built successfully, incorrect generated text, competing hypotheses,
progressively smaller comparisons, and a verified exporter correction.

The central lesson is **find the first observable divergence, then design a
check that can distinguish its possible causes**. Do not change the exporter
just because a symptom sounds like a familiar bug.

## Reading the Runner

Start with `generate()` in
[the generation script](../../examples/tensorrt_static_cache_generation.py).
Its loop keeps the order visible: prepare inputs, snapshot the cache, execute
and wait, snapshot again, run the HF reference, compare, then select a token.
`parse_args()` validates options; `run()` prepares the tokenizer and inspectors.

Read only the helper for the question you are investigating:

| Module | Responsibility |
|---|---|
| [full_prefix.py](../../examples/tensorrt_debug/full_prefix.py) | Build prompt + generated token IDs and check the input-length budget. The loop visibly resets both runtimes' caches and position to zero. |
| [comparison.py](../../examples/tensorrt_debug/comparison.py) | Run the independent HF reference cache and report whole-vector logits or tensor errors. |
| [inspect_cache.py](../../examples/tensorrt_debug/inspect_cache.py) | Validate paired cache layouts, take read-only snapshots, report changed slots, and compare valid K/V against HF. |
| [inspect_attention.py](../../examples/tensorrt_debug/inspect_attention.py) | Keep TRT Q/K/V fixed, reconstruct attention with each mask hypothesis, and compare with the actual output. |
| [_attention_support.py](../../examples/tensorrt_debug/_attention_support.py) | Layer-0 HF hooks, RoPE alignment, and reference comparisons. Skip this setup while reading the mask experiment. |
| [runtime.py](../../examples/tensorrt_debug/runtime.py) | Follow just four per-step operations: prepare inputs, execute, read tensors, and reset caches. |
| [_runtime_support.py](../../examples/tensorrt_debug/_runtime_support.py) | Engine loading, allocation, aliased cache addresses, shape/layout validation, binding, and cleanup. Skip this plumbing while following the exporter investigation. |

The command-line options and commands below are unchanged. Inspectors do not
select tokens or modify the GPU caches. `read_tensor()` waits for the runner's
stream and returns an independent CPU array with the original dtype, so BF16
byte comparisons remain meaningful. The separate
[attention probe builder](../../examples/tensorrt_attention_probe.py) still builds
the diagnostic engine; `--inspect-attention` only reads its exposed outputs.

## 1. The Result in One Paragraph

The original engine generated `TheQuestionQuestion...`. Prefill was close to
HuggingFace, but cached decode was wrong. Cache snapshots showed correct write
locations and preserved history. A separate diagnostic engine revealed that
layer-0 decode attention exactly matched a top-left causal mask: the single
query attended only to cache slot 0. The diagnostic build also reported
`nonpad_kv_seqlen` as unused. Explicit position-aware causal/valid-length masking
corrected the computation. The integrated exporter then generated
`The capital of France is Paris.`, matching HuggingFace's top token on every
step through EOS in this test.

This identifies the observed execution behavior and a working correction. It
does not determine whether the original behavior is an unsupported TensorRT
parser feature or an implementation defect, nor prove behavior on all versions.

## 2. Environment and Controlled Variables

| Item | Tested configuration |
|---|---|
| Model | `Qwen/Qwen3-0.6B` |
| HF reference revision | `c1899de289a04d12100db370d81485cdf75e47ca` |
| Device | NVIDIA GeForce RTX 4060 Laptop GPU, SM 8.9, 8 GiB |
| TensorRT | 11.3.0.99, standalone runtime, not ORT TRT-RTX |
| Export/runtime dtype | BF16 |
| Reference | HuggingFace Transformers, FP32, eager attention, CPU PyTorch |
| Static cache | 28 layers, 8 KV heads, head dimension 128, capacity 4096 |
| Batch/profile | Batch 1; token input min/opt/max lengths 1/32/128 |
| Engine build | `--decomposableAttentions=*` |
| Prompt | `What is the capital of France? Answer briefly.` |
| Tokenization | Qwen chat template, generation prompt, `enable_thinking=False` |

The rendered prompt has 22 tokens in this configuration. Reuse the actual token
IDs rather than assuming a string always tokenizes to the same sequence.

The runner's `--revision` option can pin tokenizer/reference downloads. The
original engine does not independently prove which checkpoint revision produced
its weights. Record export provenance for future investigations.

Preserve separate artifacts for the failing baseline, instrumented baseline,
experimental correction, and fresh exporter-generated correction. Do not
overwrite the only reproducible failing engine.

## 3. Understand What Crosses the Prefill/Decode Boundary

Each layer owns key and value tensors shaped `[B, Hkv, capacity, D]`. The cache
contains projected values and normalized, RoPE-transformed keys. It does not
contain token IDs. The query is computed for the current input chunk and is not
stored in this cache.

For the first two calls in this case:

| Input/state | Prefill | First cached decode |
|---|---|---|
| `input_ids` | 22 prompt tokens | `[[785]]`, the token `The` |
| `position_ids` | `[[0, ..., 21]]` | `[[22]]` |
| `write_indices` | `[0]` | `[22]` |
| `nonpad_kv_seqlen` | `[22]` | `[23]` |
| Slots written | 0 through 21 | 22 |
| Last query should attend | Slots 0 through 21 | Slots 0 through 22 |

Prefill predicts `The`; the decode call consumes `The` and predicts the token
after it. This distinction prevents a common off-by-one mistake in comparisons.

`TensorScatter` writes the new K/V into the cache. `Attention` reads that cache.
Correct writes do not guarantee correct reads or masking.

For an unpadded chunk of width S starting at slot W, valid length is W + S.
Capacity is fixed at 4096; valid length is not capacity. With padding, the
relationship involves the valid token count, not necessarily the padded width.
This walkthrough's real-model run is unpadded, batch one.

## 4. The Investigation Roadmap

| Stage | Question | Evidence | Next decision |
|---|---|---|---|
| Build/load | Can TensorRT accept the artifact? | Engine built and deserialized | Run meaningful inputs |
| Generation | Does it produce sensible output? | `TheQuestionQuestion...` | Compare numerical outputs |
| Identical-token reference | When does it first diverge? | Prefill close, first decode wrong | Isolate incremental execution |
| Full-prefix control | Does the same token history work without reuse? | Full prefix predicts ` capital` | Inspect persisted state |
| Cache snapshots | Are slots missing or overwritten? | Correct slots, history preserved | Inspect computation using the cache |
| Neighboring layer | Where does the error become large? | Layer-0 K/V close; layer-1 new K/V diverge | Probe layer-0 attention |
| Intermediate outputs | Are query or attention outputs wrong? | Query close; attention wrong | Test explicit mask hypotheses |
| Own-Q/K/V reconstruction | Which mask explains the result? | Top-left mask matches exactly | Try position-aware masking |
| Corrective experiment | Does changing masking remove the symptom? | Correct attention and generation | Integrate and regression-test |

### Stage A: Build Success Is Not Correctness

Deserialization establishes that the runtime can load the engine. Finite logits
establish that the checked numbers are not NaN or infinity. Neither establishes
that the graph computes the intended function.

The Python TensorRT binding executes the native engine through
`context.execute_async_v3(...)`. Python handles tokenization, buffer allocation,
input preparation, next-token selection, and decoding text. It is not running
the model through PyTorch or ONNX Runtime.

The first attempt used separate input/output cache buffers. TensorRT rejected
execution because this engine required those pairs to alias. We then bound each
cache output to its input address. That was an explicit runtime requirement,
not evidence that aliasing was numerically correct; the snapshot tests checked
the resulting writes later.

**Earlier clue we should have prioritized:** an input that controls correctness
being reported as unused deserves immediate investigation, even if build passes.

### Stage B: Compare Identical Tokens, Not Independent Stories

Let both implementations consume the same prompt. Save their last-token logits.
Choose one next token, here TensorRT's `785`, and feed that exact token to both.
Each runtime maintains its own cache. Compare the next logit vectors.

If the implementations independently select different tokens, later differences
can be explained by different histories. Teacher forcing removes that ambiguity.

| Baseline measurement | Prefill | First decode |
|---|---:|---:|
| Mean absolute logit error | 0.051921 | 4.260486 |
| Maximum absolute error | 0.268508 | 28.258783 |
| Correlation | 0.999899 | 0.073147 |
| TRT top token | `The` | `Question` |
| HF top token | `The` | ` capital` |

This says the first tested cached call is wrong. It does **not** prove that the
cache is corrupted: write logic, read logic, positions, masking, shape-dependent
kernels, or other decode computation could explain it.

### Stage C: Recompute the Same History Without Cache Reuse

Compare two paths:

1. Process the prompt, preserve caches, then process only `[[785]]`.
2. Clear caches and process the prompt concatenated with `[[785]]` in one call.

The second path computes the same next-token prediction, but does not depend on
state persisted across calls. Both baseline runs selected `785` first, so their
second-call token histories were identical.

| Second-call result | Cached | Full prefix |
|---|---:|---:|
| TRT top token | `Question` | ` capital` |
| Correlation with HF | 0.073147 | 0.999931 |
| Mean absolute logit error | 4.260486 | 0.052296 |

This narrows the failure to differences in incremental execution. It still
does not distinguish storage from attention computation. Full-prefix mode also
changes query length and may select different runtime implementations.

For longer cross-mode comparisons, explicitly share continuation tokens. Two
independent runs are comparable only while their token histories remain equal.

### Stage D: Read the Cache, Do Not Guess About It

Before interpreting memory, check both cache bindings' location, dtype, format,
shape, and runtime strides. The probe verified device-resident LINEAR BF16 data
with shape `(1, 8, 4096, 128)` and element strides
`(4194304, 524288, 128, 1)`.

Synchronize GPU execution before copying. Keep independent CPU snapshots so
in-place updates cannot overwrite the evidence. Preserve raw BF16 bytes for
unchanged-region tests, then convert to FP32 for numerical comparisons.

For prompt length P = 22:

| Region | Expected after prefill | Expected after decode |
|---|---|---|
| `[:, :, :P, :]` | Prompt K/V | Bit-for-bit unchanged |
| `[:, :, P:P+1, :]` | Zero | New token K/V |
| `[:, :, P+1:, :]` | Zero | Still zero and unchanged |

The HF Qwen3 cache stores `[B, Hkv, S, D]`, with keys after Q/K normalization and
RoPE, matching the intended Mobius representation. Compare equivalent stages;
comparing pre-RoPE keys with post-RoPE keys would manufacture a false mismatch.

Observed for both inspected layers: prefill changed only slots 0-21, decode
changed only slot 22, old slots were unchanged, and unused slots stayed zero.

| New decode slot vs HF | Layer 0 mean abs | Layer 1 mean abs |
|---|---:|---:|
| Keys | 0.007317 | 1.325746 |
| Values | 0.000327 | 0.122456 |

Layer-0 prefill key maximum error was 2.673859, but keys reached magnitude 520
and mean error was 0.009226. Do not interpret a maximum absolute error without
the tensor's scale and dtype. Small averages are not a formal parity guarantee,
either: a few important entries can affect attention disproportionately.

The strong conclusion here is about storage: the inspected slots were written
and preserved correctly. The next large observed numerical mismatch was in
layer-1 K/V, which depend on computations after layer-0 K/V formation.

### Stage E: Expose the First Suspect Computation

A serialized engine does not automatically expose every intermediate tensor.
We built a separate diagnostic engine by loading the ONNX graph with `onnx_ir`
and adding graph outputs for:

- Layer-0 post-RoPE query: `[B, Hq, S, D]`.
- Layer-0 attention output immediately before `o_proj`: `[B, S, Hq * D]`.

The HF probe captures `o_proj` input with a forward pre-hook. It also captures
normalized Q and applies the same HF RoPE function/positions to compare queries.

Exposing outputs can change optimization and fusion. We therefore verified that
the instrumented engine still reproduced `TheQuestion`. Its logits differed
slightly from the original engine, so keep instrumented measurements separate.

| Instrumented layer-0 comparison vs HF | Prefill mean abs | Decode mean abs |
|---|---:|---:|
| Post-RoPE query | 0.004103 | 0.004588 |
| Pre-projection attention output | 0.000375 | 0.124763 |

The first directly observed large mismatch is now inside attention, before its
output projection. If attention had matched, the next probes would have followed
the output projection, residual addition, normalization, and MLP instead.

### Stage F: Test Two Masks Using the Engine's Own Q/K/V

An HF output comparison still mixes input differences with computation
differences. To isolate the mask, reconstruct attention on CPU from the
instrumented engine's actual Q and cache K/V, converted to FP32.

For grouped-query attention, repeat each KV head to match its associated query
heads. This model has 16 query heads and 8 KV heads, so each KV head serves two
query heads. The reconstruction computes:

$$
A = \operatorname{softmax}\left(\frac{QK^T}{\sqrt{D}} + M\right)V
$$

Then transpose/reshape A to the same pre-projection layout. Test two hypotheses:

| Hypothesis | Allowed keys for local query offset t |
|---|---|
| Correct cached causality | `key_slot <= write_indices[b] + t`, with valid-prefix bound |
| Top-left causality | `key_slot <= t` |

At prefill, write index is zero, so these hypotheses make the same prediction.
At single-token decode, t = 0: top-left masking allows only slot 0, while the
correct mask allows slots 0 through 22. This is why decode is discriminating.

| Original instrumented decode attention vs own-Q/K/V reconstruction | Mean abs |
|---|---:|
| Correct cached mask | 0.124732 |
| Top-left mask | **0.000000** |

The top-left reconstruction matched exactly, not merely in its top token. With
only one key allowed, softmax has a single probability of one, so the output is
the corresponding slot-0 value. This is particularly strong evidence.

The build log independently reported `Unused Input: nonpad_kv_seqlen`.
Together, these observations explain the failure: the tested path does not use
the valid-length information to obtain the intended cached causal behavior.

The reconstruction in the current probe assumes unpadded inputs, so the correct
causal frontier also bounds valid length. Extend it with an explicit validity
condition before using it for padded experiments.

## 5. The Correct Mask, With Shapes

Let B be batch size, S the current query width, and C the cache capacity.

```text
write_indices:       [B]
query_offsets:       [S]       = arange(S)
query_slots:         [B, S]    = write_indices[:, None] + query_offsets[None, :]
key_slots:           [C]       = arange(C)
nonpad_kv_seqlen:    [B]

causal:             [B, S, C] = key_slots <= query_slots[:, :, None]
valid:              [B, 1, C] = key_slots < nonpad_kv_seqlen[:, None, None]
allowed:            [B, S, C] = causal AND valid
additive bias:      [B, 1, S, C]
```

Use zero bias for allowed positions and a sufficiently negative value for
blocked positions. Broadcasting the head dimension shares the mask across heads.

For a three-token chunk starting at 22, with valid length 25:

```text
query slot 22 -> keys 0..22
query slot 23 -> keys 0..23
query slot 24 -> keys 0..24
all queries  -> block slots 25..4095
```

Setting `is_causal=0` alone is not a fix: without a full explicit mask, queries
could attend future tokens and unused capacity. Likewise, retaining the faulty
implicit causal mask on top of a correct bias would reapply the wrong restriction.

## 6. Corrective Experiment, Then Integration

### Separate graph experiment

We applied an explicit BF16 bias to all 28 attention layers, set `is_causal=0`,
and removed native Attention input #6. The graph-level `nonpad_kv_seqlen` input
remained because the explicit mask consumes it. Cache scatter behavior was not
changed.

The prototype used RoPE `position_ids` as query positions and `-inf` as blocked
bias. That is sufficient for this specific prompt, where RoPE positions equal
cache slots. It is not a general reason to equate those concepts.

After the experiment, first-decode attention mean error vs HF fell from
0.124763 to 0.000312. Logit correlation became 0.999909 and the next token became
` capital`. Generation matched all eight HF top-token predictions through EOS.

### Exporter implementation

The permanent code path reuses existing abstractions rather than copying the
prototype graph rewrite:

| Location | Responsibility |
|---|---|
| [EP capabilities](../../src/mobius/_execution_providers.py) | `supports_attention_nonpad_kv_seqlen` defaults true; false for standalone `tensorrt` |
| [TextModel](../../src/mobius/models/base.py) | `_maybe_static_cache_bias()` enables explicit bias when the EP needs it, even without a sliding window or extra feature flag |
| [Mask helper](../../src/mobius/components/_common.py) | `create_static_cache_attention_bias()` uses write indices, query offsets, and valid length |
| [Attention emission](../../src/mobius/components/_attention.py) | Requires a supplied bias on unsupported EPs, selects `is_causal=0`, omits native nonpad input |
| [Regression tests](../../tests/static_cache_metadata_test.py) | Provider/dtype contracts, mask geometry, and missing-bias guard |

Important boundaries:

- Cache layout and masking support are separate capabilities. A rank-4 cache is
  not itself a reason to disable native masking.
- Default/ORT providers keep their existing native nonpad behavior.
- The dynamic-cache branch remains unchanged.
- The reusable bias uses cache-slot positions, not an assumption about RoPE IDs.
- Other backbones that fail to provide a required bias are rejected explicitly;
  this change does not claim universal model support.
- The existing helper uses `dtype.min`, not the prototype's `-inf`. Both worked
  for the tested unpadded inputs. Fully masked rows require separate analysis;
  the two conventions can behave differently there.

### Validation of the integrated exporter

We exported the real model using the updated Mobius CLI, checked all 28 Attention
nodes, and built an uninstrumented engine directly from that export. No
experimental ONNX mask rewrite was applied to this final artifact.

```text
Response: The capital of France is Paris.
Top-1 agreement: all 8 executed steps, including EOS
Logit correlation: approximately 0.999799 to 0.999931
```

The neighboring test file passed 40 tests, including 17 new cases. New coverage
includes default/TensorRT, FLOAT/FLOAT16/BFLOAT16 graph contracts, static/dynamic
paths, prefill/decode/chunked mask geometry, valid-prefix clamping, and a
missing-bias guard. Numerical mask tests run on CPU ORT; they do not substitute
for TensorRT engine execution. Ruff and editor checks also passed.

## 7. Reproduce the Investigation

Run commands from the repository root in PowerShell. These examples refer to
local artifacts created during this case; engines and weights are not portable
source files or guaranteed to exist in a fresh checkout.

Prerequisites include the TensorRT SDK/runtime DLLs, the matching TensorRT Python
wheel, CUDA runtime bindings, NumPy with BF16 support via `ml_dtypes`, Transformers,
and PyTorch for the CPU reference. The runner's default SDK path is
`C:/TensorRT-11.3.0.99`; use `--sdk` when it differs. The build helper accepts
`--trtexec` for an alternative executable path.

The [generation runner](../../examples/tensorrt_static_cache_generation.py)
contains the comparisons. The [diagnostic builder](../../examples/tensorrt_attention_probe.py)
adds intermediate outputs and optionally applies the experimental mask.

### Baseline, full-prefix, and cache probes

```powershell
$python = '.\.venv\Scripts\python.exe'
$runner = 'examples\tensorrt_static_cache_generation.py'
$baseline = 'qwen3-06B\models\tensorrt-4k\model.engine'

& $python -X utf8 $runner --engine $baseline --compare-hf --max-new-tokens 2
& $python -X utf8 $runner --engine $baseline --compare-hf --max-new-tokens 2 --full-prefix
& $python -X utf8 $runner --engine $baseline --compare-hf --max-new-tokens 2 --inspect-cache
& $python -X utf8 $runner --engine $baseline --compare-hf --max-new-tokens 2 --inspect-cache --cache-layer 1
```

Confirm both two-step paths select the same first token. Cache inspection is
restricted to cached mode, requires `--compare-hf`, and snapshots the first two
steps. Full-prefix mode resets both caches each step and must fit the engine's
maximum query length, not merely its larger cache capacity.

### Attention probe and experimental engine

```powershell
& $python -X utf8 $runner --engine qwen3-06B\models\tensorrt-4k-attention-probe\model.engine --compare-hf --max-new-tokens 2 --inspect-attention
& $python -X utf8 $runner --engine qwen3-06B\models\tensorrt-4k-explicit-mask\model.engine --compare-hf --max-new-tokens 2 --inspect-attention
```

To rebuild diagnostic copies from the retained failing ONNX graph, choose new
output directories. The builder deliberately refuses an existing output directory:

```powershell
& $python -X utf8 examples\tensorrt_attention_probe.py --onnx qwen3-06B\models\tensorrt-4k\model.onnx --output qwen3-06B\models\attention-probe-new
& $python -X utf8 examples\tensorrt_attention_probe.py --onnx qwen3-06B\models\tensorrt-4k\model.onnx --output qwen3-06B\models\explicit-mask-new --explicit-mask
```

Do not apply `--explicit-mask` to an already corrected export: the prototype
expects the original maskless BF16 graph. Without that option, the builder only
adds diagnostic outputs; the corrected exporter does not require this tool.

### Integrated export and final validation

```powershell
.\.venv\Scripts\mobius.exe build --model Qwen/Qwen3-0.6B --dtype bf16 --ep tensorrt --feature static-cache --max-seq-len 4096 qwen3-06B/models/tensorrt-4k-exporter-new

& $python -X utf8 $runner --engine qwen3-06B\models\tensorrt-4k-exporter-fixed\model.engine --compare-hf --compare-steps 16 --max-new-tokens 16
& $python -m pytest tests/static_cache_metadata_test.py -q --tb=short
```

The first command creates ONNX, not a TensorRT engine. The second runs the
already-built final engine from this session. To compile the new ONNX yourself,
use `trtexec` with `--decomposableAttentions=*` and min/opt/max input profiles
1/32/128 for tokens/positions, batch-one fixed cache shapes, and `[1]` length/index
inputs, as done by the diagnostic builder. Use a new engine filename. Successful
compilation must still be followed by the runtime comparison.

`--compare-steps` defaults to two. Increasing only `--max-new-tokens` does not
increase numerical comparison coverage. `Compared 8/16` in this example means
EOS arrived at step 8; it is normal completion, not a runtime failure. It also
means the test did not exercise 16 execution steps.

## 8. How to Read the Metrics Without Overclaiming

- **Mean absolute error** measures average numerical distance, but can hide a
  few important wrong entries.
- **Maximum absolute error** exposes outliers, but must be interpreted relative
  to tensor magnitude and precision.
- **Relative RMS error** compares error magnitude with reference magnitude; it
  is helpful for intermediate tensors with different scales.
- **Correlation** measures similarity of variation, not equality. A constant
  offset or scaling can retain high correlation. Inspect other metrics too.
- **Top-1 agreement** checks the selected next token. It is not sufficient to
  prove that the rest of the distribution or intermediate computation is right.
- **Bitwise equality** is appropriate for cache regions that must not change.
  It is not appropriate for BF16-versus-FP32 computed tensor comparisons.

BF16 and FP32 outputs need not match exactly. There is no universal tolerance
that makes every model/runtime pair correct. The large decode error, exact
wrong-mask reconstruction, and targeted correction together are much stronger
evidence than any individual threshold.

## 9. What Is Proven, and What Remains Open

**Supported by this case:** the failing single-token attention matched top-left
causality; inspected cache writes preserved state; explicit masking corrected the
tested intermediate and final outputs; the integrated exporter reproduced that
correction; default/dynamic graph contracts passed focused regressions.

**Not established:** universal TensorRT behavior, the upstream parser/kernel
classification, all architectures, long-context generation, real multi-batch or
padded execution, all precision modes on GPU, fully masked-row semantics,
performance impact, or the entire repository's test suite.

Before broadly deploying the fix, expand tests according to those risks. Keep
correctness and performance investigations separate: an explicit mask can alter
kernel selection even when it corrects the results.

## 10. A Reusable Debugging Checklist

1. Preserve the failing artifact and record model revision, runtime, dtype, and profile.
2. Read warnings about ignored inputs or unsupported operators before blaming model quality.
3. Run meaningful inputs; build/load success is only the first gate.
4. Compare identical token IDs against a trusted reference.
5. Find the earliest failing call: prefill, decode, or a particular shape transition.
6. Remove state reuse with a full-prefix control, keeping the token history fixed.
7. Validate memory layout before interpreting raw buffers.
8. Snapshot state and verify write regions independently of numerical computation.
9. Compare equivalent intermediates and move to the first large mismatch.
10. Test competing explanations using the same actual inputs when possible.
11. Change one semantic factor in a separate experiment; verify the baseline failure still reproduces under instrumentation.
12. Integrate through the owning abstraction, preserve unrelated runtimes, and add regression tests.
13. Rebuild from the real exporter and verify again; a patched diagnostic graph is not the final product.
14. Report the exact evidence and remaining limits, not just "it works."

### Check Your Understanding

**Why did prefill work with the wrong mask?** Query offsets start at zero in
prefill, so top-left and intended cache-slot causal frontiers coincide.

**Why did correct cache snapshots not rule out an attention bug?** Storage can
hold the right values while the attention mask selects the wrong subset.

**Why was the own-Q/K/V test decisive?** It removed most projection/reference
differences and asked which masking rule reproduces the observed output.

**Why not change every provider to explicit masking?** The observed limitation
is provider-specific; other providers depend on working native behavior and may
lose performance or change edge-case semantics under a global rewrite.

**Why rebuild from the exporter after the experiment?** The experiment proves
the proposed correction can work. A fresh export proves the implementation
actually emits that correction through the normal user workflow.