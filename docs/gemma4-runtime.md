# Gemma4 text and multimodal runtime qualification

This runtime work is draft, not production or equivalence approved. It does not
qualify optimized text-only performance, ASR, or experimental fixed/paged-cache
execution.

## HuggingFace alignment and input contract

- Global decoder layers use causal attention only; image-block IDs must not
  unmask future keys in global layers.
- Local decoder layers use
  `valid_key AND (causal OR same_image) AND (key > query - window)`.
  Same-image future keys can be visible, but an image block cannot bypass the
  strict left window boundary. Distances use physical query/KV slots, including
  past offsets; padding holes do not compress the window. Only current-chunk
  image-block IDs are available;
  these IDs are not persisted alongside the KV cache.
- The separate vision encoder's `pixel_values` input deliberately uses ONNX
  `FLOAT`, including with BF16/FP16 projection weights. Supply FP32 processor
  patches in `[0, 1]`, not BF16 patches or pre-normalized `[-1, 1]` values.
  Normalization computes `2 * (pixels - 0.5)` in FP32 before casting to the
  projection's working dtype. Existing consumers must update their input dtype;
  `examples/gemma4_multimodal.py` demonstrates the new ABI. This does not change
  the encoder-free `gemma4_unified` vision ABI.
- Checkpoint `router.scale` is preserved. The router first multiplies the
  normalized working-dtype activation by the learned scale, then applies the
  separate `hidden_size ** -0.5` scalar with FP32 opmath and rounds back to the
  working dtype. Folding that scalar into a low-precision learned scale changes
  rounding and is not equivalent.
- Checkpoint-enabled vision standardization retains `std_bias`/`std_scale`
  directly on the vision submodel. Pooling coefficients and feature accumulation
  are FP32, including the `1/9` coefficients for a 3x3 kernel. The pooled mean
  then rounds to working dtype before FP32 scaling/standardization and the
  final cast back for projection.

## Native runtime requirements and limits

Ordinary `execution_provider="cuda"` does **not** assert custom GEGLU support.
The default-off `gemma4_custom_geglu` flag explicitly opts into the experimental
custom CUDA ABI: set `MOBIUS_GEMMA4_CUSTOM_GEGLU=1` before importing Mobius, or
use `override_flags(gemma4_custom_geglu=True)` around construction, export and
weight preprocessing. Environment values are captured when the module-level
flags instance is created on first import; changing the environment afterward
does not update that instance. Without opt-in, float tanh-GELU experts use the portable
exact-activation fallback; quantized GEGLU experts fail at construction with an
explicit requirement message. Opt-in does not enable native GEGLU on non-CUDA
providers. Tanh-GELU must not be replaced by a sigmoid-based SwiGLU approximation.

The portable fallback evaluates experts densely, but selects out unchosen expert
outputs before weighting. This prevents an overflowing unselected expert from
contaminating the result through `0 * Inf` or `0 * NaN`; nonfinite selected
experts still propagate. This safeguard is not full-checkpoint numerical or
performance qualification.

Each emitted native `MoE`/`QMoE` operator persists
`mobius.gemma4.required_runtime_feature="custom-cuda-geglu-v1"` in its ONNX
metadata. This declares the experimental CUDA `activation_type="geglu"`,
`swiglu_fusion=1` requirement; it is not a claim of released ORT support or a
claim that the exporting machine used a particular runtime binary.

The frozen pre-port experiments used an **experimental modified ORT 1.32 build**:
source `942356fce6bdc78c016e4b902e49492fdf695453` with seven dirty paths; binary
SHA-256 `3f6b7ff482f8fbd4d3c639e807ed2ec35fa240c88c205dbee7822d7ec497e988`.
This is not an unmodified upstream release. Its local patchset and full
experiment records are not included in this repository; these identity fields
do not provide a reproducible runtime build. Released ORT support is unverified.

Dynamic vision-overlay decoders declare the arbitrary-attention-mask contract.
Both local and global layers retain ONNX `Attention`: reducing a mask to a valid
token count would discard internal padding holes even in causal global layers.
This preserves the full additive bias through CUDA optimization. The default
VLM qualification path uses dynamic KV cache, graph capture off, and past/present
buffer sharing off.

Text-only export uses the existing `text_only=True` builder option (CLI
`--features text-only`) to remove the vision overlay. Its causal/windowed
attention can use the existing GQA path when supported by the execution provider.
GQA sequence lengths describe prefix validity, not arbitrary interior mask holes;
use a standard-ONNX attention path when those holes must be preserved.
That eligibility is separate from projection packing, custom GEGLU opt-in, and
runtime KV-sharing/graph-capture settings. No experimental static-cache or
paged-attention changes are part of this modeling work.

## Recorded pre-port experiments, not validation of this branch

The corrected real-weight native BF16 and INT4 VLM runs each scored
**166/200**; HF BF16 scored **167/200** and HF INT4 **162/200**. These results use
a fixed quantization-selection AI2D subset, not an independent full-AI2D
evaluation, and must not be described as such.

Those runs used frozen corrected exports built from the original dirty source
worktree, with source hashes recorded in the originating experiment records.
They were **not** exports of this isolated origin/main-based branch. Newer main
has different leaf-layout/configuration features; these recorded scores (and
any separately recorded pre-port performance results) do not validate this
branch. The frozen exports also predate this branch's corrected FP32 pooling
and physical-slot window fixes. Publication requires the original source hashes,
checkpoint/quantization configuration, export identity, and runtime version to
remain attached to the experiment provenance.

Strict full-vocabulary logit parity **failed 9/9**, and KV parity **failed
540/540**, at `atol=0.01, rtol=0.01`. Task-level scores do not establish numerical
equivalence or production readiness. The isolated
patch's tiny-graph and weight-layout tests are separate evidence. Full-checkpoint
re-export and revalidation on the ported branch remain **unverified**. No model
rebuild, download, GPU run, or performance measurement is implied by this note.
Lead review is required before publication.

## Prior combined-branch evidence

Offline, CPU-only validation against base
`d69efe03a9f4c1ada744bc31dd01077a710b3c39`:

- Related runtime, package, expert-layout, adapter, and packing tests:
  **656 passed, 2 skipped**. These include `models/gemma4_test.py`,
  `components/_common_test.py`, `_model_package_test.py`,
  `functions/matmul_nbits_test.py`, and
  `rewrite_rules/_qkv_projection_packing_test.py` under `src/mobius/`.
- GQA rewrite tests: **32 passed**. Four remote-config cases were excluded from
  that offline invocation after reproducing their failures on the exact clean
  base; their tests were not changed.
- `tests/build_graph/core_test.py -k gemma4`: **10 passed**.
  `tests/integration/gemma4_test.py::test_gemma4_bidirectional_mask_parity`:
  **11 passed** against the actual pinned HF mask entry points.

Changed-file lint, formatting, compilation, and whitespace checks passed.
Strict mypy with resolved editable ONNX Script imports has zero new normalized
diagnostic instances relative to base, but neither revision typechecks cleanly.
These checks do not replace the unresolved full-checkpoint numerical gates.
The modeling/pass split and subsequent review fixes require their own validation;
the historical counts above are not results for the reorganized revision.

## Packed-checkpoint contract

Native Gemma4 experts require an explicit decoder quantization config:
`quant_method="olive"`, integer-affine `bits=4`, a power-of-two
`group_size >= 16`, `sym`, and canonical `quantize_moe=true`.
Metadata also accepts the legacy `moe` alias; `quantize_moe` takes precedence
when both are supplied. Expert quantization opt-in and the custom-runtime flag
are separate requirements.
Compute dtype must be FP16/BF16 and expert activation must be tanh-GELU.
Both expert input dimensions must be divisible by the group size. Preserve the
authoritative HF source-module names and any component/per-projection override
or exclusion plan; unsupported expert layouts fail closed, while ordinary
projection plans remain handled by the shared component loader.

For each layer with `E` experts, hidden size `H`, intermediate size `I`, and
group size `G`, Olive sidecars under `...layers.L.experts` are:

| Stem | `_qweight` (UINT8) | `_scales` (floating point) | `_qzeros` (UINT8, asymmetric only) |
|---|---|---|---|
| `gate_up_proj` | `[E, 2*I, H/2]` | `[E, 2*I, H/G]` | `[E, 2*I, ceil((H/G)/2)]` |
| `down_proj` | `[E, H, I/2]` | `[E, H, I/G]` | `[E, H, ceil((I/G)/2)]` |

Weights use contiguous gate-then-up rows before graph-level interleaving.
The adapter preserves packed codes and folds `router.per_expert_scale` into
down-projection block scales; low-precision folding changes rounding order
and is not claimed bitwise exact. The learned `router.scale` remains unchanged.
Incomplete, stale, or shape-incompatible expert sidecars are rejected.

Float projection packing retains version-1
`mobius.qkv_projection_sources` metadata on the `Split`: original initializer
names and output widths. Keep it through serialization so PEFT attachment can
restore independent projections after weight folding. INT4 projection packing
requires homogeneous bits/block size/accuracy level, scale dtype, zero-point
presence/layout, and a known shape-generic fallback body when one is registered.
Float groups whose source initializer or logical weight view has outside
consumers are conservatively left unpacked, preserving existing tied-weight
adapter attachment rather than inventing multiple-group restoration semantics.

Restored projections remain lazy slices, not eager copies of a large packed
checkpoint. Their backing-file ownership is registered with the package save
alias guard. Saving over a backing file (same path, hard link or symlink) is
rejected before any output is written, with the source unchanged. Use a distinct,
non-aliasing output location; ordinary packages without such derived lazy
dependencies retain their existing save behavior.

Backing-file ownership is also retained on the in-memory model, so initializer
folding cannot hide it by replacing the restored tensor wrappers. A regression
refolds the lazy projections before saving, checks refusal before any external
read or source mutation, then verifies an exact round-trip at a safe destination.
This metadata is transient, not serialized provenance or a guarantee for arbitrary
model-replacement transformations.
