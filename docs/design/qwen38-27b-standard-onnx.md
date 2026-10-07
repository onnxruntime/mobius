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
established by this PR.

## Verification

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
