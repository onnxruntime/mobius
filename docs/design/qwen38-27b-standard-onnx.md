# Qwen3.8-27B standard-only text export qualification

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

This qualification covers **text-only, unquantized standard ONNX export**.
It does not qualify the official 27B weights, vision/video, MTP, quantization,
ModelBuilder numerical equivalence, GenAI Engine integration, or performance.
No model weights are downloaded by these tests.

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
are included, but this qualification does not establish their CUDA numerical
parity. The existing MTP decoder is full-attention and does not construct
DeltaNet. FP16/BF16/FP64 state policies can be represented by the config;
only the official FP32 policy is numerically qualified here.

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
2. **Full-size graph construction:** build all 64 layers in FP32 and FP16,
   without allocating/loading checkpoint weights; recursively verify
   standard-only nodes and the 48 recurrent/16 KV cache groups.
3. **Reduced random-weight numerical parity:** compare an independently
   constructed HF `Qwen3_5ForCausalLM` with the actual exported/saved/reloaded
   ONNX package on CUDA FP32 and FP16. Cover single-token and six-token
   prefill, batch sizes one and two, then four greedy decode steps. Compare
   full logits, exact next-token IDs, and every convolution/recurrent/KV
   output, checking shapes and dtypes as well as values.

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

Before declaring the official model production-qualified, compare real-weight
prefill and multiple reused-state decode steps, full logits and state, and
multi-token generation. Separately compare the ModelBuilder/native route
after adapting layouts and verify the intended GenAI driver. Full-size
runtime memory and throughput must be measured independently of graph build
and reduced-model numerical correctness.

In particular, mixed FP16 convolution/FP32 recurrent state must be qualified
against the intended GenAI runtime separately. Quantized text-only component
plans and their module exclusions are outside this unquantized qualification.
