# Qwen3.8-27B standard-only multimodal export qualification

## Source and scope

`Qwen/Qwen3.8-27B` uses the existing `qwen3_5` multimodal architecture and
`qwen3_5_text` decoder, not Flash-Next's `qwen4_exp` architecture. Its text
decoder has 64 layers: 48 Gated DeltaNet layers and 16 full-attention layers,
with a dense MLP in every layer.

The offline fixture `testdata/configs/qwen3_8-27b.json` preserves the official
config values from revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.
The upstream raw `config.json` has SHA256
`191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab`
and size 4312 bytes. The checked-in fixture is reformatted, so that raw-byte
hash is provenance, not the fixture's byte hash.

The target is the **complete, unquantized standard ONNX vision-language
package**, not just its text decoder. Text-only export remains an optional
subset: removing vision does not remove any hybrid decoder layers.
The checked-in tests cover full-size graph construction and reduced
random-weight text/image/video parity. They do not download model weights
or establish official-checkpoint accuracy. ModelBuilder equivalence and
GenAI Engine integration are deferred, not acceptance gates for this export.

## Operators and state contract

`execution_provider="onnx-standard"` disables custom-op fusion and inlines
the `CausalConvWithState` and `LinearAttention` function bodies. The resulting
graph uses standard convolution, per-head normalization, gate arithmetic,
`Scan` recurrence, standard attention, and dense MLP operations. Tests require
standard domains and schemas recursively, including the Scan bodies, and no
remaining local functions.

The public Transformers builder supports `text_only=True` for the official
`qwen3_5` composite and its `qwen3_5_text` sibling. The resulting package has
one `model` entry rather than a vision/embedding/decoder pipeline.

With `text_only=False`, the package contains `vision_encoder`, `embedding`,
and `decoder`. The full decoder still contains all 48 DeltaNet and 16
full-attention layers; the vision encoder contains all 27 layers.

### Visual feature contract

The vision graph accepts processor-native FLOAT32 `pixel_values` and casts
once to model dtype internally. Run image and video packed patches through
the same vision graph, supplying each stream's grid as `image_grid_thw`.

The embedding graph keeps its existing `input_ids` and `image_features`
input names. Despite the latter's name, it accepts **both image and video
features**. Pack feature rows in the order their placeholder tokens occur
in flattened, row-major `input_ids`, including across batch rows. Do not
concatenate all image rows before all video rows unless the placeholders
actually follow that order. Feature indices do not restart at each batch
row. For text-only input or decode without new media, supply an empty
`[0, hidden_size]` feature tensor of model dtype.

The same global indices apply to shared Qwen3-VL DeepStack feature maps.
Qwen3.8-27B has no DeepStack maps. A separate embedding-only CPU sentinel
test covers two maps, two-row mixed image/video prompts with reversed order,
and zero-media input; full DeepStack decoder parity remains unqualified.
This fixes shared Qwen3/Qwen3.5 split components;
existing saved graphs must be re-exported to receive the fixes. GenAI's
ability to assemble the packed image/video stream is not established here.

For each DeltaNet layer:

- Convolution state is model dtype, `[B, conv_dim, K-1]`.
- Recurrent state is `[B, value_heads, key_dim, value_dim]`.
- The official `mamba_ssm_dtype="float32"` is preserved through config
  extraction, task inputs, recurrence function registration, and outputs,
  even when weights/activations are FP16.
- Configs without an explicit state-dtype policy (or with `auto`) retain the
  prior model-dtype behavior. Invalid state dtypes are rejected.
- Qwen3.5 per-head Q/K normalization uses the reference epsilon `1e-6`,
  rather than the generic component's zero-epsilon default.

The epsilon is fixed by the Qwen3.5 reference architecture, not a configurable
Qwen3.5 export option. The shared decoder change also applies to DeltaNet
layers constructed by Qwen3.5 MoE/VL variants. Their graph/weight regressions
are included, but this qualification does not establish official-checkpoint
parity for those other variants. The existing MTP decoder is full-attention
and does not construct DeltaNet. FP16/BF16/FP64 state policies can be
represented by the config;
only the official FP32 policy is numerically qualified here.

Learned offset RMS normalization now computes input normalization, the
`1 + weight` offset and scale multiplication in FP32 before casting the
output back to model dtype. The prior implementation rounded the offset
in FP16/BF16. A nonzero learned-weight unit test reproduces that difference;
zero-initialized reference norms do not exercise it. This corrects the
shared component contract but does not resolve full-checkpoint parity alone.

The HF reference caches K convolution positions; Mobius needs the K-1
preceding positions. Comparisons explicitly select the final K-1 HF entries.
The recurrent-state comparison uses nonsquare key/value dimensions in the
reduced fixture to detect an accidental transpose. This is a dense
invocation ABI, not the native packed/paged contract in PR #738.

**Compatibility:** re-exporting a hybrid checkpoint that requests FP32 state
changes its FP16/BF16 recurrent slots to FP32. Existing saved models do not
change. Callers must allocate/rebind cache using the graph's per-slot types,
not assume that every state has model dtype. The raw ORT parity tests exercise
that contract; compatibility of a particular GenAI cache allocator is not
established by the standard export alone.

## Native ORT-GenAI bounded qualification

The extension preserves the exact three standard graphs and native Qwen3.5
multimodal dispatch; it does not substitute a generic decoder, ModelBuilder,
manual Python cache loop, PagedAttention, or fused MoE. The ordinary CLI route
and existing `--runtime ort-genai` route are covered by tiny offline VLM tests
in `integrations/ort_genai/qwen38_standard_test.py`. No new Mobius CLI flag was
added. `--ep onnx-standard` remains the build policy; execution selects CUDA
with `og.Config.clear_providers()`, `append_provider("CUDAExecutionProvider")`,
and `og.Model(config)`.

Two reproduced metadata gaps were corrected:

- Local `--config` assets reached tokenizer/generation metadata but not the
  vision processor writer. The writer now uses the explicit HF source or,
  if absent, `local_config_dir`, retaining checkpoint normalization and
  smart-resize bounds.
- The VLM's unwrapped `qwen3_5_text` configuration was missing from the Qwen
  vision-family sets, dropping `PatchImage` and Qwen vision metadata.

The VLM writer already bypasses the text-only `_inspect_decoder_abi`; its
`inputs_embeds` mapping and graph-derived highest global cache slot did not
require an inspector rewrite. Standard hybrid packaging keeps sharing false
and capture off. Generic fast CPU e2e tests remain pinned to GenAI 0.15.2.

### Native runtime evidence and provider-routing diagnosis (2026-10-08)

The isolated checkout at `477380e595bc75fcdc6a87939d5c2246741bedc1` imports
Mobius from its own `src`. Installed ORT is 1.30.0 and GenAI reports
`0.16.0-dev`; its build identity has not been established.

`tests/ort_genai_hybrid_native_test.py` is opt-in through
`MOBIUS_QWEN_NATIVE_PROBE=1`, independent of the generic fast CPU lane.
Its standard-only synthetic three-graph pipeline has eight global slots,
sparse KV at 3 and 7, FP16 convolution state, FP32 recurrent state, and 3D
positions. The final probe uses the real per-slot geometry: hidden width 5120,
KV `[B,4,pastS,256]`, conv `[B,10240,3]`, and recurrent
`[B,48,128,128]`, but only eight slots and a 32-token synthetic vocabulary,
without checkpoint weights. Expected tokens `[0, 4, 8]` depend on *both* state families and
past KV length. Every state output's dtype, shape, and progression is checked;
a fresh generator must reproduce the same sequence. Model construction alone
is not a pass.

On GPU `GPU-630e9db2-c115-0a2d-789f-4a76ccb92ecc`:

- The generic native CUDA control passed, generating `[0, 1, 2]`.
- The hybrid native model and bundled multimodal processor were constructed
  successfully, and native text inputs were produced. Thus absence of the
  Python `onnxruntime_extensions` package is not evidence of missing bundled
  processor support.
- The initial mixed-provider probe segfaulted **during
  `generator.set_inputs(inputs)` / native prefill**, not generator construction.
  The earlier attribution was corrected after explicit stage markers and GDB
  backtraces. CPU `ConstantOfShape::Compute`, and then CPU
  `GatherCopyData<long>` after replacing the synthetic embedding with Gather,
  were called through `MultiModalPipelineState::Run` and
  `OgaGenerator_SetInputs`. Recurrent-cache incompatibility was not established.
- With empty auxiliary provider lists, the execution override did not route
  vision/embedding to CUDA. `Config.overlay()` patches specifying auxiliary
  or all three providers did not resolve the fault in either tested call order.
- Rendering **CUDA runtime session metadata for all three standard graphs**
  through the existing writer, keeping capture off, resolved native prefill.
  The required `clear_providers()` / `append_provider("CUDAExecutionProvider")`
  / `Model(config)` calls remain. This does not select CUDA graph construction
  or alter/export any ONNX graph.
- The synthetic native probe then passed all state-dependent token, dtype,
  shape, multi-step state progression, and fresh-generator reset assertions:
  `[[0,4,8], [0,4,8]]`. The installed dev runtime therefore demonstrated sparse
  global indexing, mixed cache types, and non-shared swaps; published source
  capability is no longer being used as a substitute for execution evidence.

Run the passing opt-in probe with one bounded command (fresh absolute basetemp):

```bash
env PYTHONPATH=/home/titaiwang/workspace/mobius/.worktrees/qwen27b-genai-9a3ab8ed/src \
  TMPDIR=/datadisks/disk5/titaiwang/mobius-qwen38-27b-standard-9a3ab8ed/genai-integration/tmp \
  HF_HOME=/datadisks/disk5/titaiwang/mobius-qwen38-27b-standard-9a3ab8ed/genai-integration/hf \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 \
  CUDA_VISIBLE_DEVICES=GPU-630e9db2-c115-0a2d-789f-4a76ccb92ecc \
  MOBIUS_QWEN_NATIVE_PROBE=1 timeout 120 \
  /home/titaiwang/miniconda3/envs/dev/bin/python -m pytest \
  /home/titaiwang/workspace/mobius/.worktrees/qwen27b-genai-9a3ab8ed/tests/ort_genai_hybrid_native_test.py::test_native_sparse_mixed_state_progression_and_reset \
  -q -s --tb=long \
  -o cache_dir=/datadisks/disk5/titaiwang/mobius-qwen38-27b-standard-9a3ab8ed/genai-integration/native-control-cache \
  --basetemp=/datadisks/disk5/titaiwang/mobius-qwen38-27b-standard-9a3ab8ed/genai-integration/native-probe-next
```

Replacing/installing a runtime or rebuilding native C++ requires
separate approval. Do not relax parity tolerances or substitute raw ORT to
declare this gate passed.

### Metadata-only real-artifact overlay

`examples/qwen38_standard_genai.py` accepts absolute source, HF-assets, and
fresh output paths. It loads graph metadata lazily with `onnx_ir`, reconstructs
`ArchitectureConfig` from the pinned HF text config and parent, and uses the
repository's config writer. It never calls `ModelPackage.load/save`, loads an
HF model, or serializes/copies external weights. Component directories are
symlinked, preserving their sidecars; small tokenizer/metadata files are copied
before potentially mutable asset patches. Existing destinations are refused.

CUDA preparation and bounded generation passed for the immutable `onnx-rmsfix`
artifact and produced:

`/datadisks/disk5/titaiwang/mobius-qwen38-27b-standard-9a3ab8ed/genai-integration/runtime-overlay-cuda-02/overlay-receipt.json`

and `native-results.json` in that same overlay.

The audit confirmed zero functions/custom-domain nodes in all three graphs,
global KV indices `3,7,...,63`, FP16 convolution and FP32 recurrent ports, and
unchanged original HF-asset hashes. The decoder, vision, and embedding graph
metadata sizes are respectively 3,421,999, 1,065,742, and 6,663 bytes. No
54GB weight duplication occurred. Transformers emitted documentation errors
for `min_frames`/`max_frames` during processor import, but preparation exited
successfully. These warnings were not generation failures.

Use `--prepare-only` for metadata preparation. Real generation requires
`--native-proof /absolute/.../native-proof.json` from a successful synthetic
test with the same reported installed runtime version. The example selects
CUDA through `og.Config`, delegates preprocessing/positions/caches/tokens to
GenAI, and bounds generation to 1–16 new tokens. It checks text, a fresh-
generator text reset, and an optional single image. A programmatic
`second_image` argument additionally checks two different images without a new
CLI flag. It releases native objects
after use and has no success-shaped fallback.

The proof is a **local operator attestation**, not a runtime binary identity
check: development builds can share the same version string. Rerun the native
proof after **any runtime binary change**. Matching installed binary SHA256
fingerprints for this qualification are stored in
`genai-integration/qualification-receipt.json` on the qualification artifact
disk; the example does not automatically verify them.

### Bounded real native results

These results used historical `runtime-overlay-cuda-02` metadata **before the
RGB normalization correction below**. They establish execution only; fresh
metadata and a real-image rerun are required to validate the corrected inputs.

The 420-second-bounded command completed successfully with four new greedy
tokens per invocation, thinking disabled, and no HF model loaded alongside
GenAI:

| Invocation | Native-expanded prefill | Tokens | Decoded |
|---|---:|---|---|
| Text | 17 | `[760,6511,314,9338]` | `The capital of France` |
| Fresh-generator text reset | 17 | Same four tokens | Same text |
| Single `pipeline-cat-chonk.jpeg` image | 649 | `[1919,2099,4774,264]` | `This image shows a` |
| Cat + `mage-vl-dog.jpg` images | 2700 | `[760,1118,2099,4774]` | `The first image shows` |

GenAI owned image preprocessing, 3D positions, all cache slots, and token
generation. This establishes bounded native execution and text reset, not
independent numerical correctness of native preprocessing/positions or
full-checkpoint accuracy. The native model was released on completion.
The immutable source remains 54,779,987,399 bytes; only small metadata/tokenizer
files and symlinks were added. No weights were duplicated or reserialized.

The first real command covered text/reset/single-image. A second 420-second-
bounded invocation reused the same CUDA metadata overlay and immutable weights
to cover the two-image case as well; it again reproduced the same text/reset
and single-image tokens. It did not create a second model export or load two
native models simultaneously.

**Bounded phase status: executed.** Synthetic progression/reset and real text/
single-/two-image few-token generation passed. Full AI2D200 evaluation was not
run in this phase.
The prior FP16 logit-parity failures and unchanged `1e-2` tolerances remain
unresolved. No performance, CPU/DML, video, serving, or capture promise is made.

### RGB metadata producer correction

The native `DecodeImage(color_space="RGB")` operation already emits RGB.
In the upstream ORT-Extensions producer contract
(`operators/vision/decode_image.hpp` and `shared/api/image_transforms.hpp`),
nonzero `Normalize.qwen2_5_vl` or `Normalize.qwen3_vl` enables a BGR-to-RGB swap,
not patch interleaving. Injecting that flag after RGB decode therefore reversed
red and blue. The writer now leaves ordinary Normalize mean/std attributes
unchanged; `PatchImage` separately owns temporal/spatial patch layout.
Resize, rescale, all graphs, cache policy, and model tolerances are unchanged.
Regenerate processor metadata; historical overlays are not patched in place.

The lead's before-prefill diagnostic found matching image/grid/IDs but direct
pixel max-absolute difference 2.0. Red/blue reversal alone, without row reorder,
reduced it to approximately 0.0156862. That residual resize/rounding difference
remains unresolved and is **not** bitwise pixel parity.

An opt-in toy native regression uses a lossless uniform RGB `[255,128,0]`
image, only three added image-marker tokens, and no checkpoint weights.
It checks the native processor's pixel values against analytical per-channel
normalization **before any generator/prefill call**. No manual flips, row
reorders, graph changes, or tensor substitutions are used. This independently
reproduces the old producer's swapped channels and guards the corrected RGB
contract; it does not qualify real-image resize behavior or model parity.

## Verification

### Corrected-RGB native AI2D first-200 evidence

The corrected metadata-only `runtime-overlay-cuda-03` completed all 200 pinned
AI2D examples through installed ORT GenAI `0.16.0-dev` on CUDA, using native
image processing, positions, caches, and greedy generation (16 new tokens,
thinking disabled). HF processing supplied only the identical chat template
and independent diagnostics, never tensors to the native pipeline.

| Metric | Corrected native GenAI | Prior HF Torch | Prior raw ORT CUDA |
|---|---:|---:|---:|
| Correct answers | 174/200 | 175/200 | 174/200 |
| Invalid native answers | 3 | — | — |
| Exact native generated tokens | — | 199/200 | 198/200 |

This is completed execution/task evidence, **not preprocessing or full-logit
parity**. Native IDs, image grids, and pixel shapes matched HF for 172/200
examples. The other 28 have a diagnosed native smart-resize rounding
difference: ORT-Extensions uses `std::round`, whereas the installed HF
processor uses Python's ties-to-even `round`. Both rounding models predict
their respective observed grids for all 200 examples. For example, the
400-pixel width at zero-based index 130 becomes native 416 versus HF 384 at
factor 32; its native answer is B versus the HF/raw-ORT answer D. This
coincidence does not establish that resizing alone caused the answer change.
Index 124 instead matches HF's C, while prior raw ORT answered A.

For the 172 shape-comparable examples, maximum pixel absolute difference was
0.0156863928 and minimum correlation was 0.9999995187. Pixel comparisons for
the 28 different-shape examples were not counted as passes. The inspected
upstream Resize attribute contract exposes no rounding-mode override.
Runtime-side alignment remains outside this Mobius change; no tensor flip,
pre-resize workaround, new CLI flag, or ONNX weight rebuild was used.

The results and rounding diagnosis are retained in
`genai-integration/ai2d-native-rgb-fixed-first200-01/` and
`genai-integration/native-smart-resize-diagnosis.json` on the qualification
artifact disk. Prior Torch/raw-ORT results are frozen references, not fresh
reruns. The existing official FP16 full-logit parity blocker remains open.

### Input-size and repeated-inference memory findings

The native AI2D200 runner used one Python process and one Model for the entire
ordered subset, creating and deleting a Generator per question. Its preserved
runner SHA256 is
`e7f0306d524f04d1a527d9470929a447a54edd626e08c8e8afd56012fe0b5541`.
Observed native prompt lengths (including image tokens) have median 314.5
and maximum 1,705; image patch rows have median 1,064 and maximum 6,580.
Generation was capped at 16 tokens. These are observed subset bounds, not
dataset-wide maxima or a deployment-safe input limit.

A separate full-resolution large-image diagnostic used 4,875 prompt tokens,
19,200 patch rows, and a 128-token generation cap. With the same Model kept
alive and a fresh Generator per call, the first call completed and exactly
matched the standalone tokens, text, processor tensors, and prompt. The
second call failed at decoder layer 3's `Attention_node_335`, requesting
4,563,000,320 bytes. GPU usage rose from 57,947 MiB after Model loading to
81,133 MiB after the first call, leaving 20 MiB free; deleting the Generator
and collecting garbage did not reduce that usage while the Model remained
alive. Process exit restored 81,150 MiB free and 4 MiB used.

This diagnostic used an isolated GenAI build at source commit
`265917c09ffe68198a026e086d21e694cc938298` with a one-line last-token logits
allocation correction and package-local `$ORIGIN` RPATH. It retained the
installed ORT library (SHA256
`5a2e81ad05259bce596e29d6361fe76750b6c2c9569b8cdddf709ab0d944b0ee`),
unchanged ONNX graphs, full-resolution images, and generation policy.
It is not the installed GenAI binary used for AI2D200; that binary's
originating source commit is unknown. The external GenAI correction is
not included in this Mobius PR and did not resolve repeated-inference OOM.

The exported decoder contains 16 standard Attention nodes with explicit
FP16 masks and dynamic past/present KV ports, and no TensorScatter nodes.
For the failing 4,875-token case, the local ORT unfused workspace formula
matches the requested bytes exactly:
`2 * align256(24 * 4875 * 4875 * sizeof(float)) = 4,563,000,320`.
The inspected CUDA dispatch rejects Flash with an explicit mask and rejects
MEA when a masked KV row stride is not a multiple of four; `4875 % 4 = 3`.
This is strong source-and-size evidence for unfused Attention workspace,
not a captured installed-binary dispatch trace. Exact correspondence of
the inspected ORT source to the installed binary remains unverified.
Other large-image attempts also failed in the vision encoder, so decoder
KV-cache changes alone are not established as a complete remedy.

Repeated-inference failure is confirmed, but live tensor ownership, arena
retention, and fragmentation have not been distinguished. It is not a proven
KV-cache leak, graph incompatibility, or numerical-parity fix. TensorScatter
static-cache A/B testing has not been performed. Deployment needs validated
post-processor input bounds, generation/concurrency budgets, and repeated
mixed-size request testing; AI2D200 success alone does not qualify those.
No MMMU accuracy result is claimed here.

Evidence remains under `genai-runtime-fix-01/probes/candidate-case56-repeat01/`
(`receipt.json`, `execution.json`) on the qualification artifact disk,
alongside the immutable AI2D records. No additional benchmark run or graph
change is part of this documentation update.

### Native numerical diagnostics and reset

Actual native ORT GenAI was compared with the pinned HF FP16 reference using
native processor snapshots and a native-selected token prefix. HF maintained
its own cache; no HF tensors or states were injected into GenAI. This isolates
model numerics from independent preprocessing and generation, and is not a
replacement for the independently generated AI2D results above.

The completed cases cover text and AI2D indices 0, 124, and 130, plus repeated
text/image and a fresh-generator text reset. Each completed case compares
prefill and four cached decode forwards: all 248,320 last-position logits
and all 128 convolution, recurrent, and KV outputs. Raw graph logits are
FP16; the GenAI logits API's FP32 promotion was verified separately.

At unchanged `rtol=atol=1e-2`, all 35 paired forward comparisons failed
full-vocabulary logit tolerance, while all 35 argmax predictions matched.
These include repeated cases, not 35 independent examples. Of 4,480 state
comparisons, 2,907 passed and 1,573 failed. Maximum absolute errors were
0.236328125 for logits and 85.25 for states. There were no shape, dtype, or
nonfinite failures in those numerical comparisons. The first failing cache
layer for text was layer 31; image cases could fail at earlier layers.
Neither this observation nor output-token agreement identifies the root
cause or clears the numerical gate.

In the repeat run, all 30 directly captured native `position_ids` and
`attention_mask` comparisons exactly matched HF-derived shapes, int64
dtypes, and values. Recreating the generator after an intervening image
case reproduced all five text logits bitwise, all 640 state hashes, and
the selected tokens. This establishes reset repeatability and input
agreement only for those tested cases.

The two-image native processor emits two maximum-length patch slots:
1,064 and 864 valid patches become 2,128 stored rows, including 200 zero
padding rows, while the grids describe 1,928 valid patches. Native prefill
executed with finite logits and all expected state schemas, but HF could
not consume that unchanged padded snapshot. Its paired numerical case
remains blocked; this is not proof that the native pipeline is incompatible.
No unpadding, repacking, resizing, or tolerance change was used.

Receipts, full-vocabulary vectors, input captures, and state hashes are in
`genai-integration/native-numerical-validation-01/attempt-03/`, including
`final-consolidated-receipt.json` and `reset-receipt.json`. The preceding
attempt-02 evidence is preserved unchanged. Official FP16 numerical
qualification remains failed; task-level validation does not waive it.

### CI disposition for the implementation head

At implementation head `9e8d6aa88ee16a8db77df1dbe0b1a1e585d99d61`,
CI run `37867805012` did not complete its L4/L5 golden suites. Both
check-run annotations explicitly state that the jobs exceeded the
one-hour execution limit. Cancellation is not a successful numerical
or generation test, and the partial logs also contain failed test markers.

The exact base is `26a712b210a0a7cd895347f601680e82bbf5e452`.
Its main CI run `37681522335` and scheduled L4/L5 runs `37882646916` and
`37886682810` also ended without completing those suites. All 24 named
L4 failures observed on the PR head have failed markers on that exact
base. For L5, 19 of 20 named head failures have corresponding base
failed markers. This is overlap of observed failures, not proof that
their root causes or complete test outcomes are identical.

The remaining L5 case is
`test_generation_matches_golden[text-generation/granitemoe-1b]`.
It failed on the PR head, but neither inspected base L5 job reached it
before cancellation. Its baseline disposition is therefore unresolved;
absence from the base failure list is not a base pass.

The pinned SmolLM FP16 CPU ORT GenAI job was skipped by the PR workflow's
existing configuration: the caller passes `run_real: false`, and the
real lane requires that input or a scheduled/manual event. This is not
evidence that Qwen or the pinned released runtime passed that job.

The filtered logs, annotations, exact-base comparisons, and unresolved
case coverage are retained in `genai-integration/` as
`ci-disposition-before-mmmu200.json`,
`ci-golden-direct-api-log-proof.json`,
`ci-golden-node-comparison.json`, and
`ci-granitemoe-l5-disposition.json`. This historical disposition must
not be presented as the status of a later documentation/publication head.
Any merge acceptance of incomplete qualification requires an explicit
maintainer decision; this evidence does not declare the PR merge-ready.

### Reproducing graph checks

Run from the repository root with Transformers and CUDA ORT available:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 NVIDIA_TF32_OVERRIDE=0 \
  python -m pytest src/mobius/models/qwen38_standard_test.py -q -rs
```

The tests separate these confidence levels:

1. **Config compatibility:** assert the pinned full text config's layer
   schedule, dimensions, interleaved mRoPE sections, and FP32 state policy.
2. **Full-size graph construction:** build the 64-layer decoder and the
   complete 64+27-layer VLM in FP32 and FP16, without allocating/loading
   checkpoint weights; recursively verify all components' standard-only
   nodes and the 48 recurrent/16 KV cache groups.
3. **Reduced random-weight numerical parity:** compare an independently
   constructed HF `Qwen3_5ForCausalLM` with the actual exported/saved/reloaded
   ONNX package on CUDA FP32 and FP16. Cover single-token and six-token
   prefill, batch sizes one and two, then four greedy decode steps. Compare
   full logits, exact next-token IDs, and every convolution/recurrent/KV
   output, checking shapes and dtypes as well as values.
4. **Reduced VLM numerical parity:** run actual saved/reloaded ONNX vision
   outputs through the embedding graph and hybrid decoder. Cover text,
   multiple images, video, and mixed image/video prompts in FP32 and FP16,
   including two-row media batches with reversed placeholder ordering.
   Compare vision features, fused embeddings, full logits, exact greedy
   IDs, and all carried states over prefill and four decode steps.

The focused suite passed 34 tests with no skips in the qualification
environment. The related offline regression selection passed 102 tests
with 16 existing skips. This is a different selection from the later
102-test affected-component run below; their counts are not additive.
The VLM patches are synthetic, nonzero packed
processor-shaped inputs; HF supplies explicit MRoPE positions. This is not
independent qualification of processor preprocessing or host position-ID
generation.

The reduced model retains one complete 3-DeltaNet/1-attention cycle, the
original full-attention Q/KV head ratio (6:1), the DeltaNet value/key head
ratio (3:1), convolution kernel 4, partial rotary factor 0.25, and interleaved
mRoPE. Widths, vocabulary, context length, and mRoPE sections are reduced;
weights are synthetic. This is L3 evidence, not L4/L5 checkpoint evidence.

Numerical tolerances are `rtol=atol=1e-3` for FP32 and `1e-2` for FP16.
CUDA profiling must contain actual CUDA matrix execution; ordinary CPU
metadata/control-flow handling remains allowed. Provider registration alone
is not considered proof of GPU computation. CUDA tests explicitly skip when
CUDA ORT/PyTorch is unavailable; such skips are not successful CUDA evidence.

The test that inserts a custom op inside a Scan body ensures the recursive
standard-only check is exercised, rather than merely checking top-level
nodes. This is a verification assertion, not a claim that main has a generic
fail-closed standard-only production validator.

## Remaining qualification

The official 18-shard checkpoint has been downloaded at the pinned revision
(55,562,855,904 tensor bytes). The complete FP16 VLM package, including the
learned-norm correction, was exported and reloaded successfully:
**54,779,987,399 bytes**, with all three components recursively standard-only
and no remaining local functions. CUDA sessions for all three components
loaded successfully. This establishes a real-weight artifact, not numerical
qualification.

The first export timed out while writing external tensor data. A
production-matrix probe traced slow writes to non-C-contiguous folded
weights and verified byte-identical output after storage conversion.
Operational export preparation copies only strided storage and verifies
every bit, shape and dtype; it does not change graph operators or add a
generic serializer change to Mobius.

**Official FP16 runtime parity remains failed** at the unchanged
`rtol=atol=1e-2`, using Transformers 5.15.0 and ORT 1.30.0:

- The image vision comparison exceeded tolerance at 90 of 327,680 values
  (0.0275%). The norm correction does not change the vision graph.
- Text embeddings matched exactly. Corrected-source text prefill logits
  exceeded tolerance at 79,551 of 5,711,360 values (1.39%), with maximum
  absolute error 0.091796875. Validation stopped before greedy-token,
  state and reused-cache comparisons.
- Visual-only diagnosis verified all 333 visual tensors and isolated
  rounding differences in FP16 projection-plus-bias and rotary arithmetic.
  Standard `Gemm` matched the isolated HF QKV projection exactly, unlike
  separate `MatMul`/`Add`; FP32 rotary arithmetic also matched locally.
  These are candidate remedies, not a completed full-vision fix.

After the norm correction, the combined affected component and reduced
qualification selection passed 102 tests with no skips. Neither those
passes nor the serialized standard-only audit clear the official failures.
The two additional DeepStack sentinel cases also passed.

Before declaring the official model production-qualified, compare real-weight
vision/fusion, prefill and multiple reused-state decode steps, full logits
and state, and multi-token generation. Real processor video/mixed-media and
AI2D accuracy remain unqualified. Full-size runtime memory and throughput
must be measured independently of graph build and reduced-model correctness.
ModelBuilder/native-route comparison and the intended GenAI driver are
separate, deferred work.

In particular, mixed FP16 convolution/FP32 recurrent state must be qualified
against the intended GenAI runtime separately. Quantized text-only component
plans and their module exclusions are outside this unquantized qualification.
BF16, MTP, and performance qualification are also outside the current evidence.

## Input-size and repeated-inference findings (October 11, 2026)

The completed native AI2D200 result remains **174/200 (87%)**, compared with frozen HF **175/200 (87.5%)** and raw ORT **174/200 (87%)**. Native/HF tokens matched 199/200 and native/raw tokens 198/200. AI2D used **one process and one Model for all 200 questions**, with a fresh Generator per question; it was not a per-question process-isolation run.

| Observed native AI2D200 input | Median | Maximum |
|---|---:|---:|
| Prompt tokens, including image tokens | 314.5 | 1,705 |
| Image patch rows | 1,064 | 6,580 |

Generation was capped at 16 tokens. These are bounds of the tested subset, **not dataset-wide maxima or a validated deployment-safe limit**.

A separate large-image memory diagnostic used 4,875 prompt tokens, 19,200 patch rows, and a 128-token cap. The first call completed; a second identical call on the same Model with a new Generator failed at decoder layer 3 `Attention_node_335`, requesting **4,563,000,320 bytes**. After the first call, GPU usage was 81,133 MiB with only 20 MiB free; deleting the Generator and GC did not reduce usage while the Model stayed alive. Process exit restored 81,150 MiB free / 4 MiB used. The first call exactly matched the standalone tokens, text, processor tensors, and prompt.

This diagnostic used an **isolated GenAI build**, source `265917c09ffe68198a026e086d21e694cc938298`, with a one-line last-token logits allocation correction and package-local `$ORIGIN` RPATH; actual loaded installed ORT SHA256 remained `5a2e81ad05259bce596e29d6361fe76750b6c2c9569b8cdddf709ab0d944b0ee`. Graphs, weights, images, and generation policy were unchanged. It is not the installed GenAI binary used for AI2D200; the installed GenAI source origin is unknown. **The external GenAI correction is not included in this Mobius PR and does not resolve repeated-inference OOM.**

The decoder export has 16 standard Attention nodes, explicit FP16 masks, dynamic KV ports, and no TensorScatter. The inspected ORT unfused workspace formula exactly matches the allocation: `2 * align256(24 * 4875 * 4875 * 4) = 4,563,000,320`. The local dispatch excludes Flash with an explicit mask and excludes MEA for a masked KV stride not divisible by four (`4875 % 4 = 3`). This strongly supports an unfused-workspace attribution, **not a directly captured installed-binary dispatch trace**; the inspected ORT source-to-binary correspondence remains unverified. Other large-image attempts also failed in the vision encoder.

The repeated-inference memory blocker is reproducible, but live tensor ownership, arena retention, and fragmentation are not yet distinguished. **A KV-cache leak is not proven, and static TensorScatter cache has not been A/B tested.** Deployment requires validated processor-output limits, generation/concurrency budgets, and repeated mixed-size request testing, rather than assuming a successful isolated inference or AI2D subset qualifies serving stability.

Per the narrowed publication scope, **no MMMU accuracy result is claimed**. Existing official FP16 numerical parity failures, native/HF resize differences, and incomplete historical L4/L5 CI remain open; this update does not declare merge/deployment readiness. Detailed evidence and historical CI disposition are recorded in `docs/design/qwen38-27b-standard-onnx.md`. Qualification-disk receipts include `genai-runtime-fix-01/probes/candidate-case56-repeat01/{receipt,execution}.json` and the unchanged AI2D200 artifacts.
