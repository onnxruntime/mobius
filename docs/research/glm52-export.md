# GLM-5.2 export design

## Verified architecture

Sources:

- [`zai-org/GLM-5.2` config](https://huggingface.co/zai-org/GLM-5.2/blob/main/config.json)
- Transformers `GlmMoeDsaForCausalLM` implementation
- llama.cpp `glm-dsa` GGUF metadata and tensor names
- [`unsloth/GLM-5.2-GGUF`](https://huggingface.co/unsloth/GLM-5.2-GGUF)
- [IndexShare paper, arXiv:2603.12201](https://arxiv.org/abs/2603.12201)
- [GLM-5 report, arXiv:2602.15763](https://arxiv.org/abs/2602.15763)

The Hugging Face model type is `glm_moe_dsa`, with
`GlmMoeDsaForCausalLM` as its architecture. llama.cpp writes the exact
`general.architecture` value `glm-dsa`.

| Field | GLM-5.2 value |
| --- | --- |
| Backbone layers | 78 |
| MTP layers | 1 additional NextN layer |
| Hidden / dense FFN size | 6144 / 12288 |
| Attention heads / KV heads | 64 / 64 |
| QK / NoPE / RoPE / V head dimensions | 256 / 192 / 64 / 256 |
| Q-LoRA / KV-LoRA rank | 2048 / 512 |
| Routed experts / experts per token | 256 / 8 |
| Routed / shared-expert FFN size | 2048 / 2048 |
| Shared experts | 1 |
| Dense prefix | first 3 layers |
| Router | float32 sigmoid, `noaux_tc`, normalized top-k, scale 2.5 |
| Norm / activation | pre-RMSNorm, epsilon 1e-5 / SiLU |
| RoPE | interleaved, theta 8,000,000 |
| Maximum context | 1,048,576 |
| Indexer | 32 heads, head dimension 128, top-k 2048 |

GGUF reports `block_count=79` and `nextn_predict_layers=1`; therefore the
causal-LM backbone has 78 layers. GGUF also splits the MLA KV decompression
weight into `attn_k_b` and `attn_v_b`, while Transformers exposes one
`kv_b_proj`.

## Implemented scope

The exporter now preserves the 78-layer backbone, IndexShare DSA, and the
additional improved-MTP layer. Full indexers are emitted for layers 0, 1, 2,
6, 10, ... 74; each full layer's INT32 top-k result is reused by the following
three shared layers. The indexer implements the checkpoint equations in fp32:
Q-LoRA residual projection, LayerNorm+interleaved-RoPE key projection, ReLU
scores, learned signed head weights, causal bias, and dynamic
`min(2048, key_length)` TopK.

Each full-indexer layer packs its 128-value index key beside the decompressed
MLA key in the standard `past_key_values.N.key` tensor. Shared layers retain
the normal MLA cache. This keeps the ORT GenAI key/value naming contract
without duplicating the index key across 64 attention heads.

The MTP component exports:

1. `enorm(next-token embedding)` and `hnorm(target hidden state)`;
2. concatenation and the `2H -> H` `eh_proj`;
3. the complete layer-78 DSA+MoE decoder block; and
4. `shared_head.norm`, producing `mtp_hidden` for the shared target LM head.

Multi-component packages contain `model/model.onnx`, `mtp/model.onnx`, and an
`mtp_config.json` sidecar. Both runtime aliases (`onnx-genai` and `ort-genai`)
emit these artifacts. ORT GenAI does not yet orchestrate the sidecar itself.

The importer:

- derives all MLA, MoE, IndexShare, RoPE, and NextN fields from GGUF metadata;
- separates the GGUF MTP block from the 78-layer backbone and exports it as `mtp`;
- fuses split `attn_k_b` and `attn_v_b` tensors into `kv_b_proj`;
- maps `blk.N.indexer.*` and supported `blk.78.nextn.*` tensors;
- streams stacked routed-expert tensors one expert at a time instead of
  materializing all 256 experts as one float tensor; and
- preserves supported 4-bit and 8-bit tensors as MatMulNBits, requantizing
  mixed source types to the selected graph-wide layout when necessary.

The generic portable MoE implementation evaluates every expert and masks the
outputs. It is correct for routing, but a fused sparse-MoE runtime kernel will
be required for practical GLM-5.2 execution.

## Runtime follow-ups

Portable ONNX preserves DSA selection numerics by converting the selected
indices to an additive mask for the standard `Attention` operator. No custom op
is required for correctness. Practical million-token performance still needs
a selected-token sparse-attention kernel so the runtime avoids evaluating
masked K/V positions.

`index_share_for_mtp_iteration` also needs runtime orchestration. MTP step 0 can
return `topk_indices`, but accepted speculative positions make the MTP MLA and
indexer caches advance at different logical lengths. ORT GenAI currently has
no separate indexer-cache/control-state contract, so steps 1+ recompute the
index selection rather than silently applying an incorrect packed-cache reuse.

Use `--glm-full-attention` to retain the prior dense MLA fallback. The fallback
disables the DSA-dependent MTP artifact.

## Separate workstream: Unsloth dynamic quantization

The six `UD-IQ1_M` GGUF shard headers were inspected. The dominant routed
expert tensors are below the currently targeted 4-bit floor:

- routed gate/up projections: 76 `IQ1_M` and 74 `IQ2_XXS` tensors;
- routed down projections: 71 `IQ3_XXS` and 4 `IQ4_XS` tensors;
- shared experts: mainly `Q5_K` / `Q6_K`;
- MLA projections: mainly `Q8_0` / `Q5_K`;
- norms: F32; embedding and output: Q4_K.

This increment can dequantize IQ sources and requantize them to a supported
Q4 MatMulNBits layout. That enables export but loses the intended sub-4-bit
size and may alter accuracy. Native preservation should use one or both of the
following paths.

### Runtime custom-op requirement

Add an ONNX Runtime GenAI execution-provider capability for a fused sparse-MoE
operator that:

1. consumes top-k expert indices and routing weights without evaluating all
   256 experts;
2. supports the llama.cpp IQ1_M, IQ2_XXS, and IQ3_XXS block/codebook layouts
   (plus IQ4_XS), including their scales/minimums;
3. fuses gate/up activation and down projection where profitable;
4. loads only selected expert blocks and supports CPU and target accelerator
   implementations; and
5. advertises supported bit layouts through Mobius EP capabilities so export
   selects native packed initializers, otherwise falling back to Q4
   MatMulNBits requantization.

### Draft ONNX Runtime issue

**Title:** Support sub-4-bit GGUF/IQ weights for MatMulNBits and sparse MoE

**Body:**

> GLM-5.2 dynamic GGUF distributions use IQ1_M and IQ2_XXS for most routed
> gate/up expert matrices and IQ3_XXS for most down matrices. Exporting these
> models to ONNX currently requires expansion or requantization to 4-bit,
> losing the distribution's size advantage and making a 256-expert model
> impractical. Please define supported 1/2/3-bit MatMulNBits representations,
> or provide an extensible packed/codebook-weight contract for the llama.cpp
> IQ formats, with CPU and execution-provider kernels. For MoE, an operator
> that accepts expert weights plus per-token top-k indices/weights should avoid
> evaluating or unpacking every expert. Required formats for the first target
> are IQ1_M, IQ2_XXS, IQ3_XXS, and IQ4_XS. Please also document zero-point,
> scale/codebook packing, block-size, alignment, and external-data constraints.
