# Qwen3.8-27B reduced-real Olive validation

This recipe range-fetches only deterministic slices from the pinned BF16
checkpoint (`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`; 18 shards/1199 tensors).
The fixture has 4 decoder layers (three DeltaNet and one full attention),
one vision block, remapped image/video IDs, and no MTP tensors because the
standard target forward does not consume the optional self-speculative drafter.
Mobius exposes that drafter through the separate `qwen35-mtp` package contract.

```powershell
python examples/olive/qwen3_8-27b/validate_reduced_checkpoint.py --matrix f32-cpu f16-cuda
```

The command validates strict reduced HF loading, all three ONNX components,
full logits, 20-token cached generation, graph/provider placement, save/load,
and then assembles and directly runs the Olive Q4_K_M package. It is purposely
not an ORT GenAI capability-gated test.

The Q4 recipe keeps DeltaNet's narrow `in_proj_a` decay and `in_proj_b`
time-step gates in FP16. Quantizing those recurrent controls destabilizes
cached generation; all larger decoder matrices remain eligible for
`MatMulNBits`. All three Q4 package components are reloaded with CUDA enabled;
the 20-token semantic run uses CPU because ORT 1.26's CUDA `MatMulNBits`
execution is itself nondeterministic for this reduced hybrid fixture.

BF16 export and package reload are valid Mobius outputs. The CUDA-12-compatible
ORT 1.26 wheel used for this reduced-real run cannot initialize the BF16 hybrid
graph (`CausalConvWithState`/`Softplus` provider placement), while newer
ORT-GPU wheels available in the test environment require CUDA 13. This is a
downstream runtime waiver, not an export or support gate.

The same old ORT wheel is nondeterministic for dynamic Qwen vision batches
through its `PackedMultiHeadAttention` CUDA kernel. The validator therefore
uses the portable standard-attention vision graph for stable real CUDA
image/video/mixed execution while retaining the CUDA-optimized decoder parity
and provider profile. Mobius still emits both graph variants without deciding
which downstream runtime version can load them.
