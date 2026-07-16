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

This increment registers `glm_moe_dsa` and maps `glm-dsa` GGUF files to it.
It exports the 78-layer, two-norm MLA backbone, the three dense FFN layers,
the 75 routed-MoE layers, the shared expert, sigmoid/no-aux router, interleaved
RoPE, KV cache, and untied output head. It uses the existing portable ONNX
Attention and MoE components, so the graph can be packaged for both
ONNX Runtime GenAI runtime spellings (`onnx-genai` and `ort-genai`).

The importer:

- derives all MLA, MoE, IndexShare, RoPE, and NextN fields from GGUF metadata;
- removes the GGUF-only MTP block from the backbone layer count;
- fuses split `attn_k_b` and `attn_v_b` tensors into `kv_b_proj`;
- streams stacked routed-expert tensors one expert at a time instead of
  materializing all 256 experts as one float tensor; and
- preserves supported 4-bit and 8-bit tensors as MatMulNBits, requantizing
  mixed source types to the selected graph-wide layout when necessary.

The generic portable MoE implementation evaluates every expert and masks the
outputs. It is correct for routing, but a fused sparse-MoE runtime kernel will
be required for practical GLM-5.2 execution.

## Deliberately deferred

### IndexShare sparse attention

Layers 0-2 own full indexers. Starting at layer 6, one full indexer is used
every four layers and the following three layers reuse its selected token
indices. The current export omits indexer tensors and evaluates full causal
MLA instead. This is a coherent backbone/exporter increment, but it does **not**
preserve DSA numerics or its long-context cost. A follow-up should add an
indexer component, cross-layer top-k state, sparse prefill/decode attention,
and cache-aware position handling as one tested feature rather than partially
wiring index weights into dense Attention.

### Improved MTP

The final NextN block and `index_share_for_mtp_iteration` behavior are omitted.
A follow-up should model the GLM-specific MTP inputs, shared index selection,
cache contract, outputs, and ORT GenAI speculative-decoding configuration.

## Unsloth dynamic quantization

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
