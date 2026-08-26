# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Single source of truth for GGUF architecture support.

Every architecture mobius can say anything about gets exactly one
:class:`~mobius.integrations.gguf._spec.GGUFArchitectureSpec` here. Before this
module existed the same question was answered by nine hand-maintained
containers spread across five modules, keyed inconsistently on either the GGUF
``general.architecture`` string or the mobius ``model_type``. They disagreed:

* ``bloom`` and ``t5`` resolved to real model types but had no tensor mapping,
  so config extraction succeeded and the build then died contradicting itself.
* ``gemma``, ``internlm2``, ``qwen2moe`` and ``qwen3moe`` had tensor mappings
  but no config entry, so they fell through an unmapped ``.get(arch, arch)``.
* ``qwen2_moe``/``qwen3_moe``/``mistral``/``muse_glimmer``/``hunyuan_v1_dense``
  sat in architecture-keyed containers even though llama.cpp never emits them —
  they are mobius ``model_type`` strings that leaked into the architecture
  namespace, or defensive spellings.
* ``gemma3`` mapped to ``model_type`` ``gemma3_text`` while its weight processor
  was registered under ``gemma3``, so the Gemma norm un-offset silently never
  ran for Gemma 3 imports.

Two rules keep that from recurring:

1. **Capabilities are separate verdicts.** ``config``, ``tensor_map``, ``graph``
   and ``runtime`` are answered independently, and anything short of
   ``SUPPORTED`` must carry a reason. Being listed here is not a support claim.
2. **Behavior is referenced by name, never by callable.** Each spec names the
   config postprocessor, tensor-mapping recipe, and weight processor it wants;
   the module that owns each implementation resolves the name. That keeps this
   module an import leaf (it imports only ``_spec``, ``_errors`` and
   ``_upstream``) and makes both an unknown name and an unreferenced
   implementation a test failure.

Canonical ``gguf_arch`` values are validated against the pinned llama.cpp
census, so a mobius ``model_type`` can no longer masquerade as an architecture.
Defensive spellings belong in ``aliases``.
"""

from __future__ import annotations

__all__ = [
    "MMPROJ_ARCHITECTURE",
    "arch_names_with",
    "get_arch_spec",
    "iter_arch_specs",
    "model_type_for",
    "supported_architectures",
    "try_get_arch_spec",
]

import functools
from collections import Counter
from collections.abc import Callable
from types import MappingProxyType

from mobius.integrations.gguf._errors import (
    DisabledGGUFArchitectureError,
    UnsupportedGGUFArchitectureError,
)
from mobius.integrations.gguf._spec import GGUFArchitectureSpec, Support
from mobius.integrations.gguf._upstream import upstream_architecture, upstream_architectures

#: ``general.architecture`` of a multimodal projector sidecar. Upstream this is
#: a quant-only stub, not a loadable model, so it never goes down the text path.
MMPROJ_ARCHITECTURE = "clip"

# Reasons are written once and shared so the same rejection reads identically
# wherever it surfaces.
_NO_TENSOR_MAP = (
    "mobius has no GGUF→HuggingFace tensor-name mapping for this architecture, so "
    "its weights cannot be routed into a graph. Build from the Hugging Face "
    "checkpoint with `mobius build` instead."
)

_MMPROJ_REASON = (
    "This is a multimodal projector sidecar, not a language model. Upstream it is a "
    "quant-only stub whose runtime lives outside libllama. Pass it to "
    "`build_from_gguf(text_gguf, mmproj=...)` alongside its text backbone rather "
    "than building it directly."
)

_RUNTIME_VALIDATION_PENDING = (
    "Config extraction, exact tensor-name closure, and a full synthetic GGUF graph build "
    "are covered, but no representative real-weight GGUF has yet passed ORT parity or "
    "generation validation. Runtime packaging remains deferred until that evidence exists."
)

_NO_QUANTIZED_PROJECTION_REASON = (
    "The mobius graph uses floating Linear modules for this architecture, so no "
    "MatMulNBits or BlockQuantizedMatMul target can consume preserved GGUF projection "
    "weights. Use keep_quantized=False for explicit float import."
)

_ARCTIC_GGUF_GRAPH_REASON = (
    "The pinned Arctic graph is not a standard pre-norm MoE block: every layer runs a "
    "dense parallel SwiGLU branch, then adds a separately normalized routed-expert branch "
    "computed from the pre-attention residual. Mobius's generic MoE graph replaces the "
    "dense FFN instead, so aliasing the existing Hugging Face 'arctic' registration would "
    "change residual topology and normalization."
)

_DBRX_GGUF_GRAPH_REASON = (
    "The pinned DBRX graph requires LayerNorm, a fused QKV projection, Q/K/V projection "
    "clamping from attention.clamp_kqv, and a second LayerNorm before its routed experts. "
    "Mobius's generic MoE graph uses RMSNorm, separate Q/K/V projections, and no K/Q/V "
    "clamp; the existing Hugging Face 'dbrx' registration is therefore not GGUF-compatible."
)

_GPT_OSS_GGUF_GRAPH_REASON = (
    "The pinned GPT-OSS converter splits interleaved gate/up expert rows and repacks "
    "checkpoint block+scale tensors into expert-major MXFP4 values. Its loader additionally "
    "consumes expert biases, router bias, attention sinks, output bias, post-attention RMSNorm, "
    "and sliding-window RoPE metadata. Mobius does not yet own that complete GGUF transform "
    "and packed-expert ABI, so partial float or MXFP4 import is deferred."
)

_GROK_GGUF_GRAPH_REASON = (
    "The pinned Grok graph applies embedding, attention-output, and logit scales; attention, "
    "and optional final-logit softcaps; post-attention and post-FFN norms; and a "
    "dense-plus-routed expert residual scaled by sqrt(2)/2. Mobius's generic MoE graph has "
    "none of that combined topology, so the Hugging Face-style expert names are not evidence "
    "that a GGUF alias is safe."
)

_GROVEMOE_GGUF_GRAPH_REASON = (
    "The pinned GroveMoE graph shares router logits across two distinct expert banks but "
    "performs separate selections for normal and grouped chunk experts, then scales the "
    "adjugate contribution independently. It also applies per-head Q/K RMSNorm. Mobius has "
    "no graph or quantized ownership contract for the chunk-expert tensors, and silently "
    "treating them as ordinary experts would drop a required branch."
)

_SMALLTHINKER_GGUF_GRAPH_REASON = (
    "The pinned SmallThinker graph computes router logits from the unnormalized layer input, "
    "uses ReLU experts with metadata-selected sigmoid or softmax gating, and can disable RoPE "
    "or select sliding-window attention per layer. Mobius's generic MoE graph routes after "
    "the FFN norm with softmax/SwiGLU experts and has no matching per-layer RoPE schedule."
)

_CHAMELEON_GGUF_GRAPH_REASON = (
    "The pinned Chameleon converter deliberately omits the VQ image tokenizer while the "
    "text graph still requires bias-bearing Q/K norms, an additional swin_norm, and "
    "image-vocabulary logit suppression. Mobius's similarly named Hugging Face VLM graph "
    "does not prove that text-only GGUF contract, and full multimodal generation cannot be "
    "reconstructed from the serialized file."
)

_COGVLM_GGUF_GRAPH_REASON = (
    "The pinned CogVLM text graph has modality-routed visual-expert Q/K/V/output and FFN "
    "banks in addition to the language projections. Its required cogvlm clip sidecar and "
    "feature-selection contract are also deferred, so aliasing a generic LLaVA or Llama "
    "graph would drop model-owned tensors and build the wrong package."
)

_DEEPSEEK2_OCR_GGUF_GRAPH_REASON = (
    "DeepSeek-OCR2 is a paired text-plus-vision package, not a generic DeepSeek text model. "
    "The text loader, deepseekocr/deepseekocr2 clip sidecars, SAM/projector stages, special "
    "token mixing, and cache contract have no single suffix-exact Mobius ownership map. "
    "Existing Hugging Face components therefore cannot justify partial GGUF construction."
)

_GEMMA3N_GGUF_GRAPH_REASON = (
    "Gemma3n GGUF is the text member of a vision-and-audio package whose gemma3nv and "
    "gemma3na clip companions carry distinct encoders and projectors. The text graph's "
    "per-layer embeddings, multimodal token replacement, processor assumptions, and package "
    "roles have not been validated against those pinned sidecar ABIs."
)

_HUNYUAN_VL_GGUF_GRAPH_REASON = (
    "The pinned Hunyuan-VL decoder uses its own M-RoPE and Q/K-normalized text contract and "
    "pairs with a hunyuanvl clip sidecar. Mobius's Hunyuan-VL-MoT registration is a different "
    "dual-path architecture, so neither it nor the dense Hunyuan text model is a valid alias."
)

_LLAMA4_GGUF_GRAPH_REASON = (
    "Llama4 GGUF is the text member of a paired multimodal package and may contain routed "
    "experts and architecture-specific cross-modal layer scheduling. The llama4 clip vision "
    "tower, token mixing, position IDs, and package ABI remain deferred; text-backbone "
    "similarity is not evidence that the complete GGUF tensor closure is owned."
)

_MISTRAL3_GGUF_GRAPH_REASON = (
    "The pinned Mistral3 loader selects dense or routed-expert text blocks from metadata and "
    "applies architecture-specific output temperature scaling. A VLM package additionally "
    "requires the deferred Pixtral clip sidecar and exact patch/merge/token contract. The "
    "existing Hugging Face Mistral3 graph does not cover that conditional GGUF closure."
)

_PADDLEOCR_GGUF_GRAPH_REASON = (
    "PaddleOCR-VL uses an ERNIE-derived GGUF loader with an optional bias on attention output "
    "closure and a required paddleocr clip vision/projector sidecar. Its processor ranks, "
    "image-token counts, offsets, and package identity have no pinned Mobius GGUF parity "
    "evidence, so it cannot be accepted as an ordinary Qwen2 text file."
)

_QWEN2VL_GGUF_GRAPH_REASON = (
    "The qwen2vl architecture is shared by Qwen2-VL, Qwen2.5-VL, and Qwen2.5-Omni converter "
    "paths whose clip companions use different projector strings and modalities. Exact "
    "M-RoPE sections, special tokens, merger dimensions, processor ordering, and target "
    "identity must select one complete package; a generic Qwen2 alias would erase those "
    "distinctions."
)

_QWEN3VL_GGUF_GRAPH_REASON = (
    "Qwen3-VL text GGUF requires multimodal position IDs and an exact qwen3vl_merger clip "
    "companion, including deep-stack vision features and architecture-specific token "
    "placement. The existing Hugging Face text graph alone does not establish sidecar tensor "
    "closure, processor parity, or a safe text-only fallback."
)

_QWEN3VLMOE_GGUF_GRAPH_REASON = (
    "Qwen3-VL-MoE combines the Qwen3-VL multimodal position/token contract and merger "
    "sidecar with routed experts in the text backbone. Generic Qwen3-MoE tensor similarity "
    "does not cover the paired vision package, expert sidecars, effective tied head "
    "ownership, or multimodal cache ABI."
)

_MINICPM3_GRAPH_REASON = (
    "The pinned MiniCPM3 graph uses MLA Q/KV LoRA projections, separate NoPE/RoPE "
    "query and key channels, and embedding, residual, and LM-head scales. The current "
    "Mobius MiniCPM graph does not represent that exact topology or its scales."
)

_OPENELM_GRAPH_REASON = (
    "The pinned OpenELM graph requires per-layer Q/KV-head and feed-forward-width "
    "arrays, fused QKV with Q/K normalization, and mandatory tied embeddings. Those "
    "per-layer contracts cannot be scalarized into the current generic graph."
)

_MPT_GRAPH_REASON = (
    "The pinned MPT graph permits learned positions, Q/K LayerNorm, KQV clipping, AWQ "
    "FFN scales, and several optional bias families. Mobius does not represent that "
    "whole closure safely, and the current MPT preprocessing overwrites norm biases."
)

_APERTUS_GRAPH_REASON = (
    "The pinned Apertus converter emits a serialized Llama-3 rope_freqs tensor that "
    "the pinned loader consumes as per-dimension RoPE factors. The current Mobius "
    "Apertus graph computes RoPE frequencies from scalar config and cannot represent "
    "that tensor without changing attention semantics."
)

_RECURRENT_RUNTIME_VALIDATION_PENDING = (
    "Config extraction, exact pinned tensor-name closure, GGUF value transforms, and "
    "synthetic recurrent-state execution are covered, but no representative real-weight "
    "GGUF has yet passed independent full-logit parity and deterministic multi-token "
    "stateful ORT generation. Runtime packaging remains deferred until that evidence exists."
)

_JAMBA_RUNTIME_VALIDATION_PENDING = (
    "Exact mixed attention/Mamba and dense/routed-MoE schedules, strict tensor closure "
    "and shapes, GGUF value transforms, compatible projection quantization, value-checked "
    "expert ordering, reduced Transformers parity, and multi-token ORT state threading, "
    "reorder, and replay are covered. Generic ORT GenAI runtime packaging remains deferred "
    "because its released cache schema cannot represent heterogeneous KV, convolution, and "
    "recurrent state slots; tracked by #605."
)

_RWKV_GRAPH_REASONS = {
    "rwkv6": (
        "RWKV6 carries two F32 states per layer (two token-shift vectors and a per-head "
        "WKV matrix) and applies token-dependent exp(-exp(decay)), a time_first "
        "read-before-update term, per-head group norm, and cumulative rescale transforms. "
        "Mobius has no RWKV state task; Mamba conv/SSM state and transformer KV cache are "
        "not equivalent."
    ),
    "rwkv6qwen2": (
        "RWKV6-Qwen2 is neither Qwen2 attention nor native RWKV6: it carries one F32 "
        "token-shift vector plus a per-head matrix state and uses k*(1-w) gated linear "
        "attention, optional biased/GQA projections, a sigmoid gate, and parallel Qwen "
        "SwiGLU. Mobius has no task for that recurrent ABI; Mamba conv/SSM state and "
        "transformer KV cache are not equivalent."
    ),
    "rwkv7": (
        "RWKV7 requires a two-shift F32 state plus a per-head matrix state, generalized "
        "delta-rule recurrence, six-way token mixing, first-layer value residuals shared "
        "across depth, ICLR/key-adaptation vectors, and an r_k residual around LayerNorm "
        "and group norm. Neither Mamba selective scan nor KV-cache plumbing represents it."
    ),
    "arwkv7": (
        "ARWKV7 wraps RWKV7's delta-rule matrix recurrence in a distinct one-shift "
        "RMSNorm/Qwen residual topology with optional five-versus-six-way interpolation, "
        "optional gate/group norm, and Qwen SwiGLU. Treating it as RWKV7, Qwen, or Mamba "
        "would accept the wrong tensor closure and state ABI."
    ),
}

_BAILINGMOE3_GRAPH_REASON = (
    "BailingMoE3 alternates head-wise KDA recurrent layers with gated MLA layers, "
    "so each sequence carries three causal-convolution histories plus a matrix state "
    "alongside attention cache. Its routed sigmoid/correction-bias experts, always-on "
    "shared experts, and optional single NextN block also require tensor and task "
    "contracts Mobius does not implement. Ordinary KV or Mamba state would be wrong."
)

_DEEPSEEK4_GGUF_GRAPH_REASON = (
    "The pinned DeepSeek-V4 GGUF runtime uses a dedicated raw sliding-window, CSA, "
    "HCA, and indexer compressed-cache ABI with persistent compressor state, rollback "
    "snapshots, four-stream hyper-connections, hash/sqrt-softplus routing, and optional "
    "MTP storage. Mobius's Hugging Face DeepSeek-V4 graph intentionally exports a dense "
    "attention fallback with ordinary KV state, so it is not an exact GGUF runtime graph."
)

_KIMI_LINEAR_GRAPH_REASON = (
    "Kimi-Linear alternates KDA recurrent and NoPE MLA layers, carrying three rolling "
    "convolution histories and a per-head matrix state in addition to attention cache. "
    "Its two-stage decay/output gates and sigmoid correction-bias MoE routing are not "
    "represented by any Mobius graph or state task; aliasing it to Kimi-K3, Mamba, or "
    "ordinary attention would change the model."
)

_ENCODER_RUNTIME_VALIDATION_PENDING = (
    "Config extraction, exact pinned tensor closure, encoder-only task dispatch, and "
    "synthetic ORT execution are covered, but no pinned real GGUF artifact has passed "
    "independent embedding parity. Runtime packaging remains deferred until that evidence "
    "exists."
)

_DIFFUSION_RUNTIME_VALIDATION_PENDING = (
    "Config extraction, suffix-exact tensor closure, masked-diffusion task dispatch, "
    "and synthetic full-sequence execution are covered, but no pinned real GGUF has "
    "passed independent Hugging Face/llama.cpp masked-step logit parity and deterministic "
    "multi-step generation parity. Runtime packaging remains deferred until both exist."
)

_DRAFT_RUNTIME_VALIDATION_PENDING = (
    "This is a target-coupled speculative draft, never a standalone CausalLM. "
    "Config extraction, exact tensor closure, target shape/tokenizer validation, and "
    "synthetic draft execution are covered, but no pinned real GGUF pair has passed "
    "independent target+draft full-logit/proposed-token parity. Runtime packaging remains "
    "deferred until that evidence and an acceptance-loop integration exist."
)

_POCKETTTS_BUNDLE_REASON = (
    "The primary GGUF is only PocketTTS's transformed causal CALM backbone: its "
    "embedding table contains folded learned conditioning rows and its duplicated "
    "embedding output is not a semantic LM head. Voice encoding, continuous 32-D flow "
    "generation, EOS scoring, and the stateful 24-kHz Mimi decoder live in a required "
    "pockettts_spkenc/pockettts_gen mmproj bundle that Mobius cannot import. Registering "
    "the backbone as text generation or TTS would expose the wrong I/O contract, so "
    "standalone conversion is refused before graph construction."
)

_QWEN3TTS_BUNDLE_REASON = (
    "The primary GGUF is only a transformed Qwen3-TTS talker backbone, not the existing "
    "Mobius Qwen3TTS conditional-generation or codec model. The converter folds the text "
    "projection into an extended text+codec embedding table and emits a 3072-row codec "
    "head whose logits are shifted into that combined vocabulary. The required speaker "
    "encoder, 15-codebook predictor, and stateful 24-kHz code-to-wave decoder live in a "
    "qwen3tts_spkenc/qwen3tts_gen mmproj bundle that Mobius cannot import. Standalone "
    "conversion is therefore refused before graph construction."
)

_TALKIE_GRAPH_REASON = (
    "Talkie is a text causal LM despite its upstream survey cohort. Its graph uses "
    "weight-free RMSNorm, post-RoPE Q/K normalization, learned attention and MLP gain "
    "sidecars, a per-head query gain, and an embedding skip in every block. No Mobius "
    "graph or tensor-value transform implements that combination, and no pinned real "
    "GGUF has passed independent full-logit, KV-state, and generation parity. Aliasing "
    "it to Llama or an audio task would build the wrong model."
)

_WAVTOKENIZER_DEC_REASON = (
    "wavtokenizer-dec is a stateless non-causal code-token to ISTFT-parameter network, "
    "not a waveform codec decoder. Its metadata overloads features_length as the "
    "512-wide codebook feature input and embedding_length as the 1282-wide output, while "
    "768-wide PosNet and ConvNeXt stacks provide the hidden width. The GGUF graph stops "
    "before the required magnitude/phase reconstruction and ISTFT processor, so mapping "
    "it to CodecTask would falsely promise waveform output. Dedicated graph, processor, "
    "quantization guards, and independent F16/Q5_1 parity remain deferred; standalone "
    "runtime packaging is refused before graph construction."
)

_ENCODER_GRAPH_MISMATCH = {
    "jina-bert-v3": (
        "JinaBERT v3 uses RoPE and may alternate dense GELU and routed MoE layers. "
        "BertModel has absolute positions and no MoE path."
    ),
    "nomic-bert-moe": (
        "NomicBERT-MoE alternates dense and routed-expert FFNs according to "
        "moe_every_n_layers. Mobius has no encoder MoE graph with that schedule."
    ),
}

_FINAL_CENSUS_DEFERRED_REASONS = {
    # Dense / legacy / embedding.
    "bitnet": (
        "BitNet requires post-attention and post-FFN sub-norms plus ternary projection "
        "weights with optional scalar scale tensors. Generic Llama topology and qtype "
        "handling do not represent that graph or quantization ABI."
    ),
    "codeshell": (
        "CodeShell uses bias-bearing LayerNorm blocks, grouped fused/split QKV semantics, "
        "a sequential GELU FFN, and an architecture-specific embedding/output tie "
        "direction. GPT-NeoX similarity is not an exact tensor or graph contract."
    ),
    "command-r": (
        "Command-R uses one weight-only norm feeding attention and SwiGLU in parallel, "
        "conditional Q/K norms for large variants, tied output, and logit scaling. The "
        "canonical command-r and cohere2 GGUF contracts are distinct and cannot alias."
    ),
    "gemma-embedding": (
        "Gemma Embedding is a bidirectional stateless embedding graph with alternating "
        "sliding-window attention, four norm sites, sqrt(hidden) embedding scaling, and "
        "optional pooling/dense modules. Causal Gemma3 text/VLM tasks expose the wrong ABI."
    ),
    "gptneox": (
        "GPT-NeoX GGUF requires fused biased QKV, bias-bearing LayerNorm/GELU blocks, "
        "partial RoPE, and metadata-selected parallel versus sequential residual topology. "
        "Mobius implements only the parallel MHA subset and cannot admit the full loader "
        "union."
    ),
    "jais": (
        "JAIS combines fused biased QKV, causal ALiBi, 1/head_dim attention scaling, "
        "parallel SwiGLU, and converter-baked MuP embedding/output scales. Reusing Falcon "
        "or Llama would lose required value transforms."
    ),
    "jais2": (
        "JAIS2 is a distinct RoPE, bias-bearing LayerNorm decoder with split Q/K/V and a "
        "non-gated ReLU-squared FFN. It is not the ALiBi/SwiGLU JAIS graph and has no exact "
        "Mobius tensor recipe."
    ),
    "maincoder": (
        "Maincoder applies Q/K RMSNorm after RoPE and uses an exact tied-output "
        "QK-normalized SwiGLU closure. Existing generic QK-normalized graphs use different "
        "ordering, so a family alias would change attention."
    ),
    "nanbeige": (
        "Nanbeige reuses physical layer weights across a configurable logical loop count, "
        "optionally normalizes between loops, and allocates a distinct KV slot for every "
        "logical occurrence. A Llama alias would build the wrong layer count and cache ABI."
    ),
    "orion": (
        "Orion uses bias-bearing LayerNorm at attention, FFN, and output sites despite its "
        "source config naming rms_norm_eps. Mobius Llama uses RMSNorm; interpreting that "
        "metadata generically would build the wrong normalization."
    ),
    "pangu-embedded": (
        "Pangu Embedded is a causal LM, not an embedding task, and requires a mandatory "
        "attention-output bias plus conditional LongRoPE factor tensors. Mobius has no "
        "exact graph or suffix closure for this misleadingly named architecture."
    ),
    "plamo": (
        "PLaMo uses one RMSNorm feeding attention and FFN in parallel, fixed grouped-query "
        "geometry, and converter-specific Q/output projection shuffles. Sequential Llama "
        "topology and direct external tensor reuse would both be incorrect."
    ),
    "plamo3": (
        "PLaMo3 requires fused QKV and fused SwiGLU, four norm sites with architecture-"
        "specific offset transforms, Q/K norm before RoPE, and alternating full/sliding "
        "attention state. Mobius has no exact iSWA schedule or value transform."
    ),
    "plm": (
        "PLM uses latent KV projections and normalization with shared RoPE keys, expanded "
        "K/V state, tied output, and a non-gated ReLU-squared FFN. Generic MLA cache "
        "dimensions do not supply the missing graph or tensor contract."
    ),
    "qwen": (
        "Qwen-v1 requires NeoX-style RoPE, fused biased QKV, and a fused-width SwiGLU "
        "conversion contract. Mobius's HF qwen family registration is not evidence that "
        "the canonical GGUF qwen tensor layout is importable."
    ),
    "starcoder": (
        "StarCoder requires learned absolute positions, fused biased QKV, causal "
        "multi-query attention, bias-bearing LayerNorm/GELU blocks, and a one-head KV "
        "cache. StarCoder2 uses RoPE and is a different canonical GGUF architecture."
    ),
    "xverse": (
        "Xverse is Llama-shaped but requires converter-defined Q/K reverse permutations, "
        "including a GQA-specific K transform, before quantized packing. No exact Mobius "
        "metadata/tensor/value-transform recipe currently owns that contract."
    ),
    # Conventional attention / MoE.
    "afmoe": (
        "AFMoE combines sandwich norms, Q/K norms, sigmoid-gated attention, MuP embedding "
        "scaling, a dense prefix, correction-biased routed/shared experts, and optional "
        "interleaved sliding-window attention. Mobius has no graph or cache task owning "
        "that complete topology or its expert sidecars."
    ),
    "ernie4_5": (
        "ERNIE 4.5 requires exact fused-QKV and fused-gate/up converter splits plus an "
        "optional attention-output bias and ERNIE-specific position metadata. Similarity "
        "to Qwen/Llama is not a suffix-exact tensor or graph contract."
    ),
    "ernie4_5-moe": (
        "ERNIE 4.5 MoE selects periodic expert blocks after a dense prefix, permits an "
        "optional gate-expert matrix, and uses normalized routing with optional shared "
        "experts. Mobius has no matching per-layer schedule or converter transform."
    ),
    "granite": (
        "The granite architecture is a conditional dense-or-MoE union with residual, "
        "embedding, attention, and inverse-logit scales, optional biases/RoPE factors, "
        "shared experts, and optional deep-stack inputs. It is not the GraniteMoE or "
        "GraniteHybrid GGUF contract."
    ),
    "granite_swa": (
        "Granite SWA requires attention sinks, a complete interleaved sliding-window "
        "schedule, residual/logit scaling, fused routed gate-up experts, and optional "
        "fused shared experts/deep-stack injection. Mobius owns neither that cache ABI "
        "nor its fused expert sidecars."
    ),
    "graniteswitch": (
        "GraniteSwitch repurposes an appended synthetic layer as a token-history-driven "
        "adapter router and carries fourteen switched-LoRA tensors per block in addition "
        "to decoder KV state. It is not MTP, and Mobius has no switched-LoRA graph, "
        "MUL_MAT_ID quantization contract, or package ABI."
    ),
    "hunyuan-moe": (
        "Hunyuan-MoE applies Q/K normalization after RoPE and runs normalized routed "
        "experts with an always-parallel shared branch in every layer. Hunyuan dense/VL "
        "registrations are different architectures and cannot be reused."
    ),
    "laguna": (
        "Laguna combines per-head-or-element softplus attention gates, dual-RoPE "
        "interleaved sliding-window attention, a dense prefix, and sigmoid "
        "correction-biased routed/shared experts. Mobius has no exact graph or iSWA cache "
        "contract."
    ),
    "llama-embed": (
        "llama-embed is a canonical embedding architecture that inherits Llama's "
        "conditional tensor loader but exposes the embedding graph rather than causal "
        "logits. Mobius has no GGUF embedding task/package contract for this ID, so it "
        "must not alias ordinary llama."
    ),
    "mellum": (
        "Mellum requires an untied head, Q/K norms, routed experts in every layer, and a "
        "metadata-defined full/sliding attention schedule with distinct RoPE behavior. "
        "Mobius has no matching iSWA cache or expert preservation contract."
    ),
    "minicpm": (
        "MiniCPM requires architecture-specific embedding, residual, and logit scales, "
        "Q/K permutation, optional long/short RoPE tensors, and a conditional dense-or-MoE "
        "loader. The existing MiniCPM graph does not prove this complete GGUF contract."
    ),
    "minimax-m2": (
        "MiniMax-M2 uses full-vector Q/K norms, partial RoPE, and all-layer "
        "correction-biased routed experts under metadata-selected gating. Mobius has no "
        "exact graph or suffix-safe expert import for that topology."
    ),
    "minimax-m3": (
        "MiniMax-M3 adds F32 sparse-indexer tensors and a second index-key cache with "
        "position/cell maps, block masks, rollback, and reorder semantics alongside main "
        "K/V state. Mobius has no MSA cache task or sparse-index operators; dense fallback "
        "would change the model."
    ),
    "refact": (
        "Refact uses hard-coded ALiBi without RoPE, packed-KV and gate/up converter "
        "splits, optional projection/MLP biases, and a single KV head. Llama-like tensor "
        "names are not evidence for a compatible graph."
    ),
    # Appended NextN/MTP and target-coupled assistants.
    "bailingmoe2": (
        "BailingMoE2 serializes complete dense-or-routed/shared expert trailing blocks "
        "plus NextN and layer-output norms, but the pinned loader marks every trailing "
        "tensor skipped and exposes no MTP graph. Mobius cannot invent executable head "
        "semantics from preserved storage."
    ),
    "cohere2moe": (
        "Cohere2MoE's executable head uses sigmoid routed fused-or-split experts, optional "
        "shared experts, no FFN norm, and interleaved sliding-window KV state. Mobius's "
        "single dense Qwen3.5 MTP head cannot represent that graph or cache ABI."
    ),
    "deepseek2": (
        "DeepSeek2 MTP is a complete MLA plus routed/shared MoE block with compressed KV "
        "cache, Q/KV LoRA alternatives, target-owned embedding/head fallbacks, and "
        "architecture-specific gating. It is not Mobius's dense full-attention MTP head."
    ),
    "deepseek32": (
        "DeepSeek3.2 extends the DeepSeek2 MLA/MoE head with DSA indexer projections, "
        "normalization, bias, and sparse-cache metadata. A normal KV-cache NextN task "
        "would omit required executed tensors and state."
    ),
    "dots3note": (
        "Dots3Note preserves an MLA/DSA trunk and a dense sliding-MLA NextN block, but the "
        "pinned loader explicitly has no MTP graph and skips the head. Mobius cannot infer "
        "runtime semantics from its serialized tensors."
    ),
    "exaone-moe": (
        "EXAONE-MoE serializes a dense trailing NextN block after an iSWA routed/shared "
        "expert trunk, but the pinned loader skips appended blocks. Mobius has no exact "
        "SWA schedule, remapped global-MTP transform, or executable head contract."
    ),
    "exaone4": (
        "EXAONE4 serializes attention/FFN post-norm trailing blocks and NextN tensors with "
        "optional synthetic Llama3 RoPE factors, but the pinned loader skips them. No "
        "Mobius task owns those preserved-only semantics."
    ),
    "gemma4-assistant": (
        "Gemma4 Assistant is a standalone target-coupled model with pre/post projections, "
        "masked embeddings, scalar layer scales, its own KV cache, and a live target-model "
        "context. It is neither Gemma4 text nor the target-config-only draft/MTP ABI."
    ),
    "glm4": (
        "GLM4 serializes complete fused-FFN trailing blocks and NextN tensors, but the "
        "pinned loader skips appended blocks; GLM-OCR converter transforms also permute "
        "Q/K for M-RoPE. It must not alias Qwen or an executable Mobius MTP head."
    ),
    "glm4moe": (
        "GLM4-MoE serializes biased attention and periodic dense/routed expert trailing "
        "blocks with mandatory router bias, but the pinned loader skips them. Mobius has "
        "no executable contract for the preserved head or its expert sidecars."
    ),
    "hy_v3": (
        "Hunyuan-V3 executes a full-attention NextN head with Q/K norms and optional "
        "sigmoid routed/shared experts, seeded from target hidden state. Its hyper/routing "
        "semantics are not Mobius's dense Qwen3.5 sidecar."
    ),
    "mimo2": (
        "MiMo2 requires fused-QKV dense MTP blocks, attention sinks, interleaved sliding "
        "KV cache, and three chained heads selected by offsets. Mobius permits one head "
        "and cannot preserve that state or FP8 converter transform."
    ),
    "mistral4": (
        "Mistral4 has no NextN metadata or MTP graph; it inherits Mistral3's conditional "
        "dense/MoE tensor loader and overrides graph construction. It is not llama, "
        "mistral alias text, or any DeepSeek/Qwen MTP family."
    ),
    "step35": (
        "Step3.5 executes one or more interleaved-SWA NextN heads with optional gates, "
        "routed/shared experts, centered-norm transforms, per-layer head geometry, and "
        "dedicated cache offsets. Mobius's one-head dense MTP ABI cannot represent it."
    ),
}


_SPECS: tuple[GGUFArchitectureSpec, ...] = (
    # ---------------------------------------------------------------- Llama
    GGUFArchitectureSpec(
        gguf_arch="llama",
        model_type="llama",
        # llama.cpp writes Mistral checkpoints with architecture "llama"; the
        # "mistral" string is not an upstream architecture, so it is accepted
        # defensively rather than treated as canonical.
        aliases=frozenset({"mistral"}),
        tensor_map_recipe=("llama",),
        tensor_processor="llama",
        llama_qk_permute=True,
        runtime=Support.SUPPORTED,
        runtime_evidence_ids=(
            "smollm-135m-f16-onnxruntime-1.29.0",
            "smollm-135m-f16-ort-genai-0.15.2",
        ),
        reason=(
            "Runtime support is restricted to exact structured evidence matches. Currently "
            "that is only neopolita/smollm-135m-gguf F16 at the pinned artifact, CPU import "
            "route, evidenced ONNX Runtime/ORT GenAI versions, and the pinned "
            "HuggingFaceTB/SmolLM-135M tokenizer revision."
        ),
    ),
    GGUFArchitectureSpec(
        gguf_arch="deci",
        model_type="llama",
        tensor_map_recipe=("llama",),
        tensor_processor="llama",
        llama_qk_permute=True,
        runtime=Support.DEFERRED,
        reason=_RUNTIME_VALIDATION_PENDING,
    ),
    # ----------------------------------------------------------------- Qwen
    GGUFArchitectureSpec(
        gguf_arch="qwen2",
        model_type="qwen2",
        tensor_map_recipe=("llama",),
        runtime=Support.DEFERRED,
        reason=_RUNTIME_VALIDATION_PENDING,
    ),
    GGUFArchitectureSpec(
        gguf_arch="qwen3",
        model_type="qwen3",
        tensor_map_recipe=("llama",),
        runtime=Support.DEFERRED,
        reason=_RUNTIME_VALIDATION_PENDING,
    ),
    GGUFArchitectureSpec(
        gguf_arch="dflash",
        model_type="DFlashDraftModel",
        tensor_map_recipe=("dflash",),
        config_key_map="draft",
        config_postprocessor="dflash",
        required_metadata=(
            "attention.layer_norm_rms_epsilon",
            "block_size",
            "target_layers",
        ),
        runtime=Support.DEFERRED,
        reason=_DRAFT_RUNTIME_VALIDATION_PENDING,
    ),
    GGUFArchitectureSpec(
        gguf_arch="eagle3",
        model_type="Eagle3DraftModel",
        tensor_map_recipe=("eagle3",),
        tensor_processor="llama",
        llama_qk_permute=True,
        config_key_map="draft",
        config_postprocessor="eagle3",
        required_metadata=(
            "attention.layer_norm_rms_epsilon",
            "target_hidden_size",
            "target_layers",
        ),
        runtime=Support.DEFERRED,
        reason=_DRAFT_RUNTIME_VALIDATION_PENDING,
    ),
    GGUFArchitectureSpec(
        gguf_arch="qwen2moe",
        model_type="qwen2_moe",
        # "qwen2_moe" is the mobius model_type, which previously sat in the
        # architecture map by mistake. Keep accepting it, but canonically this
        # architecture is spelled the way llama.cpp writes it.
        aliases=frozenset({"qwen2_moe"}),
        tensor_map_recipe=("llama", "moe_extras"),
        config_postprocessor="moe",
        required_metadata=(
            "attention.layer_norm_rms_epsilon",
            "expert_count",
            "expert_used_count",
        ),
        runtime=Support.DEFERRED,
        reason=_RUNTIME_VALIDATION_PENDING,
    ),
    GGUFArchitectureSpec(
        gguf_arch="qwen3moe",
        model_type="qwen3_moe",
        aliases=frozenset({"qwen3_moe"}),
        tensor_map_recipe=("llama", "moe_qk_norm_extras", "moe_extras"),
        config_postprocessor="moe",
        required_metadata=(
            "attention.layer_norm_rms_epsilon",
            "expert_count",
            "expert_used_count",
        ),
        runtime=Support.DEFERRED,
        reason=_RUNTIME_VALIDATION_PENDING,
    ),
    # ----------------------------------------- Masked/diffusion language models
    GGUFArchitectureSpec(
        gguf_arch="dream",
        model_type="dream",
        tensor_map_recipe=("llama", "diffusion_fused_qkv"),
        config_postprocessor="dream",
        required_metadata=("attention.layer_norm_rms_epsilon",),
        runtime=Support.DEFERRED,
        reason=_DIFFUSION_RUNTIME_VALIDATION_PENDING,
    ),
    GGUFArchitectureSpec(
        gguf_arch="llada",
        model_type="llada",
        tensor_map_recipe=("llama",),
        tensor_processor="llama",
        llama_qk_permute=True,
        config_postprocessor="llada",
        required_metadata=("attention.layer_norm_rms_epsilon",),
        runtime=Support.DEFERRED,
        reason=_DIFFUSION_RUNTIME_VALIDATION_PENDING,
    ),
    GGUFArchitectureSpec(
        gguf_arch="llada-moe",
        model_type="llada",
        module_type="llada_moe",
        tensor_map_recipe=(
            "llama",
            "diffusion_fused_qkv",
            "moe_qk_norm_extras",
            "moe_extras",
        ),
        config_postprocessor="llada_moe",
        required_metadata=(
            "attention.layer_norm_rms_epsilon",
            "expert_count",
            "expert_used_count",
        ),
        runtime=Support.DEFERRED,
        reason=_DIFFUSION_RUNTIME_VALIDATION_PENDING,
    ),
    GGUFArchitectureSpec(
        gguf_arch="rnd1",
        model_type="llada",
        module_type="rnd1",
        tensor_map_recipe=(
            "llama",
            "diffusion_fused_qkv",
            "moe_qk_norm_extras",
            "moe_extras",
        ),
        config_postprocessor="rnd1",
        required_metadata=(
            "attention.layer_norm_rms_epsilon",
            "expert_count",
            "expert_used_count",
        ),
        runtime=Support.DEFERRED,
        reason=_DIFFUSION_RUNTIME_VALIDATION_PENDING,
    ),
    GGUFArchitectureSpec(
        gguf_arch="lfm2",
        model_type="lfm2",
        tensor_map_recipe=("lfm2",),
        required_metadata=(
            "attention.head_count_kv",
            "attention.layer_norm_rms_epsilon",
            "shortconv.l_cache",
        ),
        runtime=Support.DEFERRED,
        reason=_RECURRENT_RUNTIME_VALIDATION_PENDING,
    ),
    GGUFArchitectureSpec(
        gguf_arch="qwen35",
        model_type="qwen3_5_text",
        tensor_map_recipe=("llama", "qwen35_hybrid_extras"),
        offset_norm=True,
        v_head_reorder=True,
        required_metadata=(
            "attention.layer_norm_rms_epsilon",
            "rope.dimension_sections",
            "ssm.conv_kernel",
            "ssm.group_count",
            "ssm.inner_size",
            "ssm.state_size",
            "ssm.time_step_rank",
        ),
        runtime=Support.DEFERRED,
        reason=_RECURRENT_RUNTIME_VALIDATION_PENDING,
    ),
    GGUFArchitectureSpec(
        gguf_arch="qwen35moe",
        model_type="qwen3_5_moe",
        tensor_map_recipe=("llama", "moe_extras", "qwen35_hybrid_extras"),
        offset_norm=True,
        v_head_reorder=True,
        required_metadata=(
            "attention.layer_norm_rms_epsilon",
            "expert_count",
            "expert_used_count",
            "rope.dimension_sections",
            "ssm.conv_kernel",
            "ssm.group_count",
            "ssm.inner_size",
            "ssm.state_size",
            "ssm.time_step_rank",
        ),
        runtime=Support.DEFERRED,
        reason=_RECURRENT_RUNTIME_VALIDATION_PENDING,
    ),
    GGUFArchitectureSpec(
        gguf_arch="qwen3next",
        model_type="qwen3_next",
        tensor_map_recipe=("llama", "moe_extras", "qwen3next_hybrid_extras"),
        offset_norm=True,
        required_metadata=(
            "attention.layer_norm_rms_epsilon",
            "expert_count",
            "expert_used_count",
            "ssm.conv_kernel",
            "ssm.group_count",
            "ssm.inner_size",
            "ssm.state_size",
            "ssm.time_step_rank",
        ),
        runtime=Support.DEFERRED,
        reason=_RECURRENT_RUNTIME_VALIDATION_PENDING,
    ),
    # ---------------------------------------------------------------- Gemma
    GGUFArchitectureSpec(
        gguf_arch="gemma",
        model_type="gemma",
        tensor_map_recipe=("llama",),
        tensor_processor="unoffset_norm",
        runtime=Support.DEFERRED,
        reason=_RUNTIME_VALIDATION_PENDING,
    ),
    GGUFArchitectureSpec(
        gguf_arch="gemma2",
        model_type="gemma2",
        tensor_map_recipe=("llama", "gemma2_extras"),
        tensor_processor="unoffset_norm",
        config_postprocessor="gemma2",
        runtime=Support.DEFERRED,
        reason=_RUNTIME_VALIDATION_PENDING,
    ),
    GGUFArchitectureSpec(
        gguf_arch="gemma3",
        model_type="gemma3_text",
        tensor_map_recipe=("llama", "gemma3_extras"),
        # models/gemma3_text.py normalizes with OffsetRMSNorm, so the llama.cpp
        # `+1` baked into every *norm.weight must be removed on import.
        tensor_processor="unoffset_norm",
        config_postprocessor="gemma3",
        vlm_builder="gemma3",
        runtime=Support.DEFERRED,
        reason=_RUNTIME_VALIDATION_PENDING,
    ),
    GGUFArchitectureSpec(
        gguf_arch="gemma4",
        model_type="gemma4_text",
        tensor_map_recipe=("llama", "gemma4_extras"),
        # Deliberately no tensor_processor: models/gemma4.py normalizes with
        # plain RMSNorm, not OffsetRMSNorm, so its GGUF weights are already in
        # the form the graph consumes. Applying the Gemma un-offset here would
        # corrupt every norm.
        config_postprocessor="gemma4",
        vlm_builder="gemma4",
        runtime=Support.DEFERRED,
        reason=_RUNTIME_VALIDATION_PENDING,
    ),
    # -------------------------------------------------------------- Various
    GGUFArchitectureSpec(
        gguf_arch="phi3",
        model_type="phi3",
        tensor_map_recipe=("phi3",),
        runtime=Support.DEFERRED,
        reason=_RUNTIME_VALIDATION_PENDING,
    ),
    GGUFArchitectureSpec(
        gguf_arch="baichuan",
        model_type="baichuan",
        tensor_map_recipe=("llama",),
        config_postprocessor="baichuan",
        tensor_processor="llama",
        llama_qk_permute=True,
        required_metadata=("attention.layer_norm_rms_epsilon",),
        runtime=Support.DEFERRED,
        reason=_RUNTIME_VALIDATION_PENDING,
    ),
    GGUFArchitectureSpec(
        gguf_arch="chatglm",
        model_type="chatglm",
        tensor_map_recipe=("chatglm",),
        config_postprocessor="chatglm",
        required_metadata=("attention.layer_norm_rms_epsilon",),
        rope_interleave=True,
        runtime=Support.DEFERRED,
        quantized_import=Support.REJECTED,
        reason=(
            _RUNTIME_VALIDATION_PENDING
            + " Quantization preservation is rejected because fused QKV and gate/up "
            "tensors must be split into separate packed graph targets."
        ),
    ),
    GGUFArchitectureSpec(
        gguf_arch="phi2",
        model_type="phi",
        tensor_map_recipe=("phi2",),
        config_postprocessor="phi2",
        required_metadata=("attention.layer_norm_epsilon",),
        runtime=Support.DEFERRED,
        quantized_import=Support.REJECTED,
        reason=(
            _RUNTIME_VALIDATION_PENDING
            + " Quantization preservation is rejected because the Phi-2 attention, "
            "MLP, and output graph uses float-only linear modules."
        ),
    ),
    GGUFArchitectureSpec(
        gguf_arch="seed_oss",
        model_type="seed_oss",
        tensor_map_recipe=("llama", "seed_oss_extras"),
        config_postprocessor="seed_oss",
        required_metadata=("attention.layer_norm_rms_epsilon",),
        runtime=Support.DEFERRED,
        reason=_RUNTIME_VALIDATION_PENDING,
    ),
    GGUFArchitectureSpec(
        gguf_arch="falcon",
        model_type="falcon",
        tensor_map_recipe=("falcon",),
        runtime=Support.DEFERRED,
        reason=_RUNTIME_VALIDATION_PENDING,
    ),
    GGUFArchitectureSpec(
        gguf_arch="gpt2",
        model_type="gpt2",
        tensor_map_recipe=("gpt2",),
        tensor_processor="gpt2",
        runtime=Support.DEFERRED,
        quantized_import=Support.REJECTED,
        reason=(
            _RUNTIME_VALIDATION_PENDING
            + " Quantization preservation is rejected because canonical GPT-2 GGUF "
            "projections must be transposed into graph order, and the current packed "
            "route cannot transpose values together with their scales and zero-points. "
            "Use keep_quantized=False for explicit float import."
        ),
    ),
    GGUFArchitectureSpec(
        gguf_arch="mamba",
        model_type="mamba",
        tensor_map_recipe=("mamba",),
        config_key_map="mamba",
        config_postprocessor="mamba",
        tensor_processor="mamba",
        required_metadata=(
            "attention.layer_norm_rms_epsilon",
            "ssm.conv_kernel",
            "ssm.inner_size",
            "ssm.state_size",
            "ssm.time_step_rank",
        ),
        runtime=Support.DEFERRED,
        quantized_import=Support.REJECTED,
        reason=_RECURRENT_RUNTIME_VALIDATION_PENDING + " " + _NO_QUANTIZED_PROJECTION_REASON,
    ),
    GGUFArchitectureSpec(
        gguf_arch="mamba2",
        model_type="mamba2",
        tensor_map_recipe=("mamba2",),
        config_key_map="mamba",
        config_postprocessor="mamba2",
        tensor_processor="mamba",
        required_metadata=(
            "attention.layer_norm_rms_epsilon",
            "ssm.conv_kernel",
            "ssm.group_count",
            "ssm.inner_size",
            "ssm.state_size",
            "ssm.time_step_rank",
        ),
        runtime=Support.DEFERRED,
        quantized_import=Support.REJECTED,
        reason=_RECURRENT_RUNTIME_VALIDATION_PENDING + " " + _NO_QUANTIZED_PROJECTION_REASON,
    ),
    GGUFArchitectureSpec(
        gguf_arch="jamba",
        model_type="jamba",
        tensor_map_recipe=("jamba",),
        config_key_map="jamba",
        config_postprocessor="jamba",
        tensor_processor="mamba",
        required_metadata=(
            "attention.layer_norm_rms_epsilon",
            "ssm.conv_kernel",
            "ssm.inner_size",
            "ssm.state_size",
            "ssm.time_step_rank",
        ),
        runtime=Support.DEFERRED,
        reason=_JAMBA_RUNTIME_VALIDATION_PENDING,
    ),
    GGUFArchitectureSpec(
        gguf_arch="nemotron_h",
        model_type="nemotron_h",
        tensor_map_recipe=("nemotron_h",),
        config_key_map="nemotron_h",
        config_postprocessor="nemotron_h",
        tensor_processor="nemotron_h",
        llama_qk_permute=True,
        required_metadata=(
            "attention.layer_norm_rms_epsilon",
            "ssm.conv_kernel",
            "ssm.group_count",
            "ssm.inner_size",
            "ssm.state_size",
            "ssm.time_step_rank",
        ),
        runtime=Support.DEFERRED,
        quantized_import=Support.REJECTED,
        reason=_RECURRENT_RUNTIME_VALIDATION_PENDING + " " + _NO_QUANTIZED_PROJECTION_REASON,
    ),
    GGUFArchitectureSpec(
        gguf_arch="nemotron_h_moe",
        model_type="nemotron_h",
        tensor_map_recipe=("nemotron_h_moe",),
        config_key_map="nemotron_h",
        config_postprocessor="nemotron_h_moe",
        tensor_processor="nemotron_h",
        llama_qk_permute=True,
        required_metadata=(
            "attention.layer_norm_rms_epsilon",
            "ssm.conv_kernel",
            "ssm.group_count",
            "ssm.inner_size",
            "ssm.state_size",
            "ssm.time_step_rank",
            "expert_count",
            "expert_used_count",
            "expert_feed_forward_length",
            "expert_shared_feed_forward_length",
        ),
        runtime=Support.DEFERRED,
        quantized_import=Support.REJECTED,
        reason=(
            "Exact mixed attention/Mamba2/dense/MoE scheduling, sigmoid correction-bias "
            "routing, shared experts, optional latent projections, and strict GGUF tensor "
            "closure are supported. Generic ORT GenAI runtime packaging remains deferred "
            "because its released cache schema cannot represent heterogeneous KV, "
            "convolution, and recurrent state slots; tracked by onnxruntime/mobius#605. "
            "Quantization preservation is unsupported because mixed Mamba2 recurrent "
            "parameters must remain dequantized and correction-biased sigmoid experts "
            "cannot use the fused MoE ABI. Use keep_quantized=False for explicit float import."
        ),
    ),
    GGUFArchitectureSpec(
        gguf_arch="granitehybrid",
        model_type="granitemoehybrid",
        tensor_map_recipe=("granitehybrid",),
        config_key_map="granitehybrid",
        config_postprocessor="granitehybrid",
        tensor_processor="granitehybrid",
        llama_qk_permute=True,
        required_metadata=(
            "attention.layer_norm_rms_epsilon",
            "ssm.conv_kernel",
            "ssm.group_count",
            "ssm.inner_size",
            "ssm.state_size",
            "ssm.time_step_rank",
        ),
        runtime=Support.DEFERRED,
        quantized_import=Support.REJECTED,
        reason=(
            "Exact mixed attention/Mamba2 scheduling, architecture-wide dense or routed "
            "MoE feed-forward selection, optional shared experts, Granite scaling, "
            "value-preserving float expert fusion, and strict pinned tensor closure are "
            "supported. Quantized sources require explicit dequantization because the "
            "current graph has no exact packed 3-D expert ABI; use keep_quantized=False. "
            "Generic ORT GenAI runtime packaging remains deferred because its released cache "
            "schema cannot represent heterogeneous KV, convolution, and recurrent state "
            "slots; tracked by onnxruntime/mobius#605."
        ),
    ),
    GGUFArchitectureSpec(
        gguf_arch="bailingmoe3",
        config=Support.DEFERRED,
        tensor_map=Support.DEFERRED,
        graph=Support.DEFERRED,
        runtime=Support.DEFERRED,
        reason=_BAILINGMOE3_GRAPH_REASON,
    ),
    GGUFArchitectureSpec(
        gguf_arch="kimi-k3",
        model_type="kimi_k3",
        config_key_map="kimi_k3",
        config_postprocessor="kimi_k3",
        tensor_map_recipe=("kimi_k3",),
        tensor_processor="kimi_k3",
        required_metadata=(
            "context_length",
            "embedding_length",
            "feed_forward_length",
            "block_count",
            "attention.head_count",
            "attention.head_count_kv",
            "attention.layer_norm_rms_epsilon",
            "attention.q_lora_rank",
            "attention.key_length_mla",
            "attention.value_length_mla",
            "attention.kv_lora_rank",
            "rope.dimension_count",
            "ssm.conv_kernel",
            "kda.head_dim",
            "expert_count",
            "expert_used_count",
            "expert_feed_forward_length",
            "expert_shared_count",
            "expert_gating_func",
            "attn_res.block_size",
            "activation.situ_beta",
            "activation.situ_linear_beta",
        ),
        runtime=Support.DEFERRED,
        reason=(
            "The exact KDA/NoPE gated-MLA schedule, four-state recurrent ABI, "
            "attention-residual banks, Stable LatentMoE routing, SiTU activation, "
            "strict metadata/tensor closure, and compatible projection quantization are "
            "supported. Released generic ORT GenAI cache schemas cannot represent the "
            "heterogeneous KV plus convolution/matrix state ABI; tracked by "
            "onnxruntime/mobius#605."
        ),
    ),
    GGUFArchitectureSpec(
        gguf_arch="kimi-linear",
        model_type="kimi_linear",
        config_key_map="kimi_linear",
        config_postprocessor="kimi_linear",
        tensor_map_recipe=("kimi_linear",),
        tensor_processor="kimi_linear",
        required_metadata=(
            "context_length",
            "embedding_length",
            "feed_forward_length",
            "block_count",
            "attention.head_count",
            "attention.head_count_kv",
            "attention.layer_norm_rms_epsilon",
            "attention.key_length_mla",
            "attention.value_length_mla",
            "attention.kv_lora_rank",
            "rope.dimension_count",
            "ssm.conv_kernel",
            "kda.head_dim",
            "expert_count",
            "expert_used_count",
            "expert_feed_forward_length",
            "expert_shared_count",
            "leading_dense_block_count",
            "expert_weights_scale",
            "expert_gating_func",
        ),
        runtime=Support.DEFERRED,
        reason=(
            "The exact KDA/NoPE-MLA schedule, four-state recurrent ABI, dense/MoE "
            "topology, correction-bias routing, pinned metadata, tensor closure, and "
            "compatible MatMul/expert quantization are supported. Released generic ORT "
            "GenAI cache schemas cannot "
            "represent heterogeneous KV plus three convolution histories and a matrix "
            "state, and representative real-weight GGUF evidence is pending; package "
            "runtime remains tracked by onnxruntime/mobius#605."
        ),
    ),
    GGUFArchitectureSpec(
        gguf_arch="lfm2moe",
        model_type="lfm2_moe",
        tensor_map_recipe=("lfm2", "lfm2_moe_extras"),
        config_postprocessor="lfm2moe",
        required_metadata=(
            "attention.head_count_kv",
            "attention.layer_norm_rms_epsilon",
            "rope.freq_base",
            "shortconv.l_cache",
            "expert_count",
            "expert_used_count",
            "expert_feed_forward_length",
            "expert_gating_func",
        ),
        runtime=Support.DEFERRED,
        quantized_import=Support.REJECTED,
        reason=_RECURRENT_RUNTIME_VALIDATION_PENDING + " " + _NO_QUANTIZED_PROJECTION_REASON,
    ),
    GGUFArchitectureSpec(
        gguf_arch="minimax-01",
        model_type="minimax",
        config_key_map="minimax",
        config_postprocessor="minimax",
        tensor_map_recipe=("minimax",),
        required_metadata=(
            "context_length",
            "embedding_length",
            "block_count",
            "feed_forward_length",
            "attention.head_count",
            "attention.head_count_kv",
            "attention.key_length",
            "attention.value_length",
            "attention.layer_norm_rms_epsilon",
            "rope.freq_base",
            "rope.dimension_count",
            "expert_count",
            "expert_used_count",
            "residual_scale",
        ),
        runtime=Support.DEFERRED,
        reason=(
            "Graph import is exact, but released ORT GenAI packaging cannot represent "
            "the heterogeneous KV/recurrent state slots or bounded rollback snapshots; "
            "runtime packaging remains tracked by #605."
        ),
    ),
    GGUFArchitectureSpec(
        gguf_arch="plamo2",
        model_type="plamo2",
        config_key_map="plamo2",
        config_postprocessor="plamo2",
        tensor_map_recipe=("plamo2",),
        tensor_processor="plamo2",
        required_metadata=(
            "context_length",
            "embedding_length",
            "block_count",
            "feed_forward_length",
            "attention.head_count",
            "attention.head_count_kv",
            "attention.layer_norm_rms_epsilon",
            "rope.freq_base",
            "ssm.conv_kernel",
            "ssm.inner_size",
            "ssm.state_size",
            "ssm.time_step_rank",
            "ssm.group_count",
        ),
        runtime=Support.DEFERRED,
        reason=(
            "The dedicated graph and GGUF importer preserve PLaMo2's alternating "
            "Mamba1/attention layers and mixed state ABI, but released ORT GenAI "
            "cannot represent heterogeneous per-layer state. Runtime packaging "
            "remains deferred to onnxruntime/mobius#605."
        ),
    ),
    GGUFArchitectureSpec(
        gguf_arch="falcon-h1",
        model_type="falcon_h1",
        config_key_map="falcon_h1",
        config_postprocessor="falcon_h1",
        tensor_map_recipe=("falcon_h1",),
        tensor_processor="mamba",
        required_metadata=(
            "context_length",
            "embedding_length",
            "block_count",
            "feed_forward_length",
            "attention.head_count",
            "attention.head_count_kv",
            "attention.key_length",
            "attention.value_length",
            "attention.layer_norm_rms_epsilon",
            "rope.freq_base",
            "ssm.conv_kernel",
            "ssm.inner_size",
            "ssm.state_size",
            "ssm.time_step_rank",
            "ssm.group_count",
        ),
        runtime=Support.DEFERRED,
        reason=(
            "The dedicated graph and GGUF importer preserve parallel Attention+Mamba2 "
            "layers and their four-state ABI, but runtime packaging remains deferred "
            "pending heterogeneous-state schema support (onnxruntime/mobius#605) and "
            "real full-logit plus deterministic stateful-generation evidence."
        ),
    ),
    # --------------------------------------------------------- Encoder-only
    GGUFArchitectureSpec(
        gguf_arch="bert",
        model_type="bert",
        tensor_map_recipe=("bert",),
        config_postprocessor="bert_encoder",
        required_metadata=(
            "attention.causal",
            "attention.layer_norm_epsilon",
        ),
        runtime=Support.DEFERRED,
        reason=_ENCODER_RUNTIME_VALIDATION_PENDING,
    ),
    GGUFArchitectureSpec(
        gguf_arch="modern-bert",
        model_type="modernbert",
        tensor_map_recipe=("modern_bert",),
        config_postprocessor="modern_bert_encoder",
        required_metadata=(
            "attention.causal",
            "attention.layer_norm_epsilon",
            "rope.freq_base",
        ),
        runtime=Support.DEFERRED,
        reason=_ENCODER_RUNTIME_VALIDATION_PENDING,
    ),
    GGUFArchitectureSpec(
        gguf_arch="eurobert",
        model_type="eurobert",
        module_type="eurobert_gguf",
        tensor_map_recipe=("eurobert",),
        config_postprocessor="specialized_encoder",
        required_metadata=(
            "attention.causal",
            "attention.layer_norm_rms_epsilon",
            "rope.freq_base",
            "rope.dimension_count",
        ),
        runtime=Support.DEFERRED,
        quantized_import=Support.REJECTED,
        reason=_ENCODER_RUNTIME_VALIDATION_PENDING + " " + _NO_QUANTIZED_PROJECTION_REASON,
    ),
    GGUFArchitectureSpec(
        gguf_arch="neo-bert",
        model_type="neobert",
        module_type="neo_bert_gguf",
        tensor_map_recipe=("neo_bert",),
        config_postprocessor="specialized_encoder",
        required_metadata=(
            "attention.causal",
            "attention.layer_norm_rms_epsilon",
            "rope.freq_base",
            "rope.dimension_count",
        ),
        runtime=Support.DEFERRED,
        quantized_import=Support.REJECTED,
        reason=(
            _ENCODER_RUNTIME_VALIDATION_PENDING
            + " Packed QKV and fused SwiGLU have no complete quantized split route; "
            "use keep_quantized=False."
        ),
    ),
    GGUFArchitectureSpec(
        gguf_arch="nomic-bert",
        model_type="nomic_bert",
        module_type="nomic_bert_gguf",
        tensor_map_recipe=("nomic_bert",),
        config_postprocessor="specialized_encoder",
        required_metadata=(
            "attention.causal",
            "attention.layer_norm_epsilon",
            "rope.freq_base",
            "rope.dimension_count",
        ),
        runtime=Support.DEFERRED,
        quantized_import=Support.REJECTED,
        reason=_ENCODER_RUNTIME_VALIDATION_PENDING + " " + _NO_QUANTIZED_PROJECTION_REASON,
    ),
    GGUFArchitectureSpec(
        gguf_arch="jina-bert-v2",
        model_type="bert",
        module_type="jina_bert_v2_gguf",
        tensor_map_recipe=("jina_bert_v2",),
        config_postprocessor="specialized_encoder",
        required_metadata=(
            "attention.causal",
            "attention.layer_norm_epsilon",
        ),
        runtime=Support.DEFERRED,
        quantized_import=Support.REJECTED,
        reason=(
            _ENCODER_RUNTIME_VALIDATION_PENDING
            + " Optional Q/K norms and fused GeGLU inputs have no complete packed "
            "quantization route; use keep_quantized=False."
        ),
    ),
    *(
        GGUFArchitectureSpec(
            gguf_arch=architecture,
            config=Support.DEFERRED,
            tensor_map=Support.DEFERRED,
            graph=Support.DEFERRED,
            runtime=Support.DEFERRED,
            reason=reason,
        )
        for architecture, reason in _ENCODER_GRAPH_MISMATCH.items()
    ),
    GGUFArchitectureSpec(
        gguf_arch="starcoder2",
        model_type="starcoder2",
        tensor_map_recipe=("llama",),
        runtime=Support.DEFERRED,
        reason=_RUNTIME_VALIDATION_PENDING,
    ),
    GGUFArchitectureSpec(
        gguf_arch="stablelm",
        model_type="stablelm",
        tensor_map_recipe=("llama",),
        runtime=Support.DEFERRED,
        reason=_RUNTIME_VALIDATION_PENDING,
    ),
    GGUFArchitectureSpec(
        gguf_arch="internlm2",
        model_type="internlm2",
        tensor_map_recipe=("llama",),
        tensor_processor="llama",
        llama_qk_permute=True,
        runtime=Support.DEFERRED,
        quantized_import=Support.REJECTED,
        reason=_RUNTIME_VALIDATION_PENDING + " " + _NO_QUANTIZED_PROJECTION_REASON,
    ),
    GGUFArchitectureSpec(
        gguf_arch="olmo",
        model_type="olmo",
        tensor_map_recipe=("olmo",),
        config_postprocessor="olmo",
        required_metadata=("attention.layer_norm_epsilon",),
        tensor_processor="llama",
        llama_qk_permute=True,
        runtime=Support.DEFERRED,
        reason=_RUNTIME_VALIDATION_PENDING,
    ),
    GGUFArchitectureSpec(
        gguf_arch="olmo2",
        model_type="olmo2",
        tensor_map_recipe=("llama", "olmo2_extras"),
        config_postprocessor="dense_sliding",
        required_metadata=("attention.layer_norm_rms_epsilon",),
        runtime=Support.DEFERRED,
        reason=_RUNTIME_VALIDATION_PENDING,
    ),
    GGUFArchitectureSpec(
        gguf_arch="olmoe",
        model_type="olmoe",
        tensor_map_recipe=("llama", "moe_qk_norm_extras", "moe_extras"),
        config_postprocessor="moe",
        required_metadata=(
            "attention.layer_norm_rms_epsilon",
            "expert_count",
            "expert_used_count",
        ),
        runtime=Support.DEFERRED,
        reason=_RUNTIME_VALIDATION_PENDING,
    ),
    GGUFArchitectureSpec(
        gguf_arch="phimoe",
        model_type="phimoe",
        module_type="phimoe_gguf",
        tensor_map_recipe=("llama", "phi3", "moe_extras"),
        config_postprocessor="phimoe",
        required_metadata=(
            "attention.layer_norm_rms_epsilon",
            "expert_count",
            "expert_used_count",
        ),
        runtime=Support.DEFERRED,
        reason=_RUNTIME_VALIDATION_PENDING,
    ),
    GGUFArchitectureSpec(
        gguf_arch="granitemoe",
        model_type="granitemoe",
        tensor_map_recipe=("llama", "moe_extras"),
        config_postprocessor="granitemoe",
        required_metadata=(
            "attention.layer_norm_rms_epsilon",
            "logit_scale",
        ),
        tensor_processor="llama",
        llama_qk_permute=True,
        runtime=Support.DEFERRED,
        reason=_RUNTIME_VALIDATION_PENDING,
    ),
    GGUFArchitectureSpec(
        gguf_arch="bailingmoe",
        model_type="bailing_moe",
        tensor_map_recipe=("llama", "diffusion_fused_qkv", "moe_extras"),
        config_key_map="conventional_shared_moe",
        config_postprocessor="conventional_shared_moe",
        required_metadata=(
            "attention.layer_norm_rms_epsilon",
            "expert_feed_forward_length",
            "expert_shared_count",
        ),
        tensor_processor="llama",
        llama_qk_permute=True,
        runtime=Support.DEFERRED,
        reason=_RUNTIME_VALIDATION_PENDING,
    ),
    GGUFArchitectureSpec(
        gguf_arch="deepseek",
        model_type="deepseek",
        tensor_map_recipe=("llama", "diffusion_fused_qkv", "deepseek_shared_moe_extras"),
        config_key_map="conventional_shared_moe",
        config_postprocessor="conventional_shared_moe",
        required_metadata=("attention.layer_norm_rms_epsilon",),
        tensor_processor="llama",
        llama_qk_permute=True,
        runtime=Support.DEFERRED,
        reason=_RUNTIME_VALIDATION_PENDING,
    ),
    GGUFArchitectureSpec(
        gguf_arch="dots1",
        model_type="dots1",
        tensor_map_recipe=(
            "llama",
            "diffusion_fused_qkv",
            "moe_qk_norm_extras",
            "deepseek_shared_moe_extras",
        ),
        config_key_map="conventional_shared_moe",
        config_postprocessor="conventional_shared_moe",
        required_metadata=("attention.layer_norm_rms_epsilon",),
        tensor_processor="llama",
        runtime=Support.DEFERRED,
        reason=_RUNTIME_VALIDATION_PENDING,
    ),
    # ------------------------- Remaining conventional-attention MoE (audited/deferred)
    GGUFArchitectureSpec(
        gguf_arch="arctic",
        config=Support.DEFERRED,
        tensor_map=Support.DEFERRED,
        graph=Support.DEFERRED,
        runtime=Support.DEFERRED,
        reason=_ARCTIC_GGUF_GRAPH_REASON,
    ),
    GGUFArchitectureSpec(
        gguf_arch="dbrx",
        config=Support.DEFERRED,
        tensor_map=Support.DEFERRED,
        graph=Support.DEFERRED,
        runtime=Support.DEFERRED,
        reason=_DBRX_GGUF_GRAPH_REASON,
    ),
    GGUFArchitectureSpec(
        gguf_arch="gpt-oss",
        config=Support.DEFERRED,
        tensor_map=Support.DEFERRED,
        graph=Support.DEFERRED,
        runtime=Support.DEFERRED,
        reason=_GPT_OSS_GGUF_GRAPH_REASON,
    ),
    GGUFArchitectureSpec(
        gguf_arch="grok",
        config=Support.DEFERRED,
        tensor_map=Support.DEFERRED,
        graph=Support.DEFERRED,
        runtime=Support.DEFERRED,
        reason=_GROK_GGUF_GRAPH_REASON,
    ),
    GGUFArchitectureSpec(
        gguf_arch="grovemoe",
        config=Support.DEFERRED,
        tensor_map=Support.DEFERRED,
        graph=Support.DEFERRED,
        runtime=Support.DEFERRED,
        reason=_GROVEMOE_GGUF_GRAPH_REASON,
    ),
    GGUFArchitectureSpec(
        gguf_arch="smallthinker",
        config=Support.DEFERRED,
        tensor_map=Support.DEFERRED,
        graph=Support.DEFERRED,
        runtime=Support.DEFERRED,
        reason=_SMALLTHINKER_GGUF_GRAPH_REASON,
    ),
    # ------------------------------ Remaining multimodal text backbones (audited/deferred)
    GGUFArchitectureSpec(
        gguf_arch="chameleon",
        config=Support.DEFERRED,
        tensor_map=Support.DEFERRED,
        graph=Support.DEFERRED,
        runtime=Support.DEFERRED,
        reason=_CHAMELEON_GGUF_GRAPH_REASON,
    ),
    GGUFArchitectureSpec(
        gguf_arch="cogvlm",
        config=Support.DEFERRED,
        tensor_map=Support.DEFERRED,
        graph=Support.DEFERRED,
        runtime=Support.DEFERRED,
        reason=_COGVLM_GGUF_GRAPH_REASON,
    ),
    GGUFArchitectureSpec(
        gguf_arch="deepseek2-ocr",
        config=Support.DEFERRED,
        tensor_map=Support.DEFERRED,
        graph=Support.DEFERRED,
        runtime=Support.DEFERRED,
        reason=_DEEPSEEK2_OCR_GGUF_GRAPH_REASON,
    ),
    GGUFArchitectureSpec(
        gguf_arch="gemma3n",
        config=Support.DEFERRED,
        tensor_map=Support.DEFERRED,
        graph=Support.DEFERRED,
        runtime=Support.DEFERRED,
        reason=_GEMMA3N_GGUF_GRAPH_REASON,
    ),
    GGUFArchitectureSpec(
        gguf_arch="hunyuan_vl",
        config=Support.DEFERRED,
        tensor_map=Support.DEFERRED,
        graph=Support.DEFERRED,
        runtime=Support.DEFERRED,
        reason=_HUNYUAN_VL_GGUF_GRAPH_REASON,
    ),
    GGUFArchitectureSpec(
        gguf_arch="llama4",
        config=Support.DEFERRED,
        tensor_map=Support.DEFERRED,
        graph=Support.DEFERRED,
        runtime=Support.DEFERRED,
        reason=_LLAMA4_GGUF_GRAPH_REASON,
    ),
    GGUFArchitectureSpec(
        gguf_arch="mistral3",
        config=Support.DEFERRED,
        tensor_map=Support.DEFERRED,
        graph=Support.DEFERRED,
        runtime=Support.DEFERRED,
        reason=_MISTRAL3_GGUF_GRAPH_REASON,
    ),
    GGUFArchitectureSpec(
        gguf_arch="paddleocr",
        config=Support.DEFERRED,
        tensor_map=Support.DEFERRED,
        graph=Support.DEFERRED,
        runtime=Support.DEFERRED,
        reason=_PADDLEOCR_GGUF_GRAPH_REASON,
    ),
    GGUFArchitectureSpec(
        gguf_arch="qwen2vl",
        model_type="qwen2_vl_text",
        tensor_map_recipe=("llama",),
        required_metadata=(
            "attention.layer_norm_rms_epsilon",
            "rope.dimension_sections",
        ),
        vlm_builder="qwen_vl",
        runtime=Support.DEFERRED,
        reason=(
            "Text and paired Qwen2/Qwen2.5-VL projector graph import are supported "
            "for the exact split-QKV llama.cpp artifacts, but downstream multimodal "
            "runtime execution has not been evidenced."
        ),
    ),
    GGUFArchitectureSpec(
        gguf_arch="qwen3vl",
        config=Support.DEFERRED,
        tensor_map=Support.DEFERRED,
        graph=Support.DEFERRED,
        runtime=Support.DEFERRED,
        reason=_QWEN3VL_GGUF_GRAPH_REASON,
    ),
    GGUFArchitectureSpec(
        gguf_arch="qwen3vlmoe",
        config=Support.DEFERRED,
        tensor_map=Support.DEFERRED,
        graph=Support.DEFERRED,
        runtime=Support.DEFERRED,
        reason=_QWEN3VLMOE_GGUF_GRAPH_REASON,
    ),
    GGUFArchitectureSpec(
        gguf_arch="cohere2",
        model_type="cohere2",
        tensor_map_recipe=("llama",),
        config_postprocessor="dense_sliding",
        required_metadata=(
            "attention.layer_norm_epsilon",
            "attention.sliding_window",
            "logit_scale",
        ),
        runtime=Support.DEFERRED,
        reason=_RUNTIME_VALIDATION_PENDING,
    ),
    GGUFArchitectureSpec(
        gguf_arch="arcee",
        model_type="arcee",
        tensor_map_recipe=("arcee",),
        required_metadata=("attention.layer_norm_rms_epsilon",),
        tensor_processor="llama",
        llama_qk_permute=True,
        runtime=Support.DEFERRED,
        reason=_RUNTIME_VALIDATION_PENDING,
    ),
    GGUFArchitectureSpec(
        gguf_arch="smollm3",
        model_type="smollm3",
        tensor_map_recipe=("llama",),
        config_postprocessor="dense_sliding",
        required_metadata=("attention.layer_norm_rms_epsilon",),
        tensor_processor="llama",
        llama_qk_permute=True,
        runtime=Support.DEFERRED,
        reason=_RUNTIME_VALIDATION_PENDING,
    ),
    GGUFArchitectureSpec(
        gguf_arch="exaone",
        model_type="exaone",
        tensor_map_recipe=("llama",),
        required_metadata=("attention.layer_norm_rms_epsilon",),
        runtime=Support.DEFERRED,
        reason=_RUNTIME_VALIDATION_PENDING,
    ),
    GGUFArchitectureSpec(
        gguf_arch="nemotron",
        model_type="nemotron",
        tensor_map_recipe=("llama",),
        tensor_processor="unoffset_norm",
        runtime=Support.DEFERRED,
        reason=_RUNTIME_VALIDATION_PENDING,
    ),
    GGUFArchitectureSpec(
        gguf_arch="hunyuan-dense",
        model_type="hunyuan_v1_dense",
        aliases=frozenset({"hunyuan_v1_dense"}),
        tensor_map_recipe=("llama", "hunyuan_extras"),
        runtime=Support.DEFERRED,
        reason=_RUNTIME_VALIDATION_PENDING,
    ),
    GGUFArchitectureSpec(
        gguf_arch="muse-glimmer",
        model_type="muse_glimmer_text",
        aliases=frozenset({"muse_glimmer"}),
        tensor_map_recipe=("llama", "muse_glimmer_extras"),
        tensor_processor="muse_glimmer",
        config_key_map="muse_glimmer",
        config_postprocessor="muse_glimmer",
        vlm_builder="muse_glimmer",
        llama_qk_permute=True,
        runtime=Support.DEFERRED,
        reason=_RUNTIME_VALIDATION_PENDING,
    ),
    GGUFArchitectureSpec(
        gguf_arch="deepseek4",
        config=Support.DEFERRED,
        tensor_map=Support.DEFERRED,
        graph=Support.DEFERRED,
        runtime=Support.DEFERRED,
        reason=_DEEPSEEK4_GGUF_GRAPH_REASON,
    ),
    # GLM-5.2 GGUFs (e.g. unsloth/GLM-5.2-GGUF) tag the architecture 'glm-dsa'
    # (MLA + DeepSeek Sparse Attention + MoE) and mobius's registry key is
    # 'glm_moe_dsa'. The format bridge is keyed on the authoritative
    # general.architecture value, never on a filename, and
    # ``assert_glm_moe_dsa_resolvable`` verifies the head/layer/expert/DSA
    # properties before the builder selects GlmMoeDsaCausalLMModel.
    GGUFArchitectureSpec(
        gguf_arch="glm-dsa",
        model_type="glm_moe_dsa",
        aliases=frozenset({"glm_dsa"}),
        config_key_map="glm_dsa",
        tensor_map=Support.DEFERRED,
        reason=(
            "Config extraction and the glm_moe_dsa graph are both available, but "
            "no GGUF→HuggingFace tensor-name mapping has been written for GLM-5.2's "
            "MLA + DSA-indexer tensor families yet, so weights cannot be routed "
            "into the graph. " + _NO_TENSOR_MAP
        ),
    ),
    GGUFArchitectureSpec(
        gguf_arch="apertus",
        config=Support.DEFERRED,
        tensor_map=Support.DEFERRED,
        graph=Support.DEFERRED,
        runtime=Support.DEFERRED,
        reason=_APERTUS_GRAPH_REASON,
    ),
    GGUFArchitectureSpec(
        gguf_arch="minicpm3",
        config=Support.DEFERRED,
        tensor_map=Support.DEFERRED,
        graph=Support.DEFERRED,
        runtime=Support.DEFERRED,
        reason=_MINICPM3_GRAPH_REASON,
    ),
    GGUFArchitectureSpec(
        gguf_arch="openelm",
        model_type="openelm",
        module_type="gguf_legacy",
        tensor_map_recipe=("legacy_layernorm", "exact_legacy_gguf_extras"),
        config_postprocessor="exact_legacy_gguf",
        required_metadata=(
            "attention.layer_norm_rms_epsilon",
            "attention.head_count",
            "attention.head_count_kv",
            "feed_forward_length",
        ),
        runtime=Support.DEFERRED,
        quantized_import=Support.REJECTED,
        reason=(
            _RUNTIME_VALIDATION_PENDING
            + " Quantization preservation is rejected because every OpenELM layer stores "
            "fused QKV rows that must be split into per-layer Q/K/V graph projections."
        ),
    ),
    GGUFArchitectureSpec(
        gguf_arch="mpt",
        model_type="mpt",
        module_type="gguf_legacy",
        tensor_map_recipe=("legacy_layernorm",),
        config_postprocessor="exact_legacy_gguf",
        required_metadata=("attention.layer_norm_epsilon", "attention.max_alibi_bias"),
        runtime=Support.DEFERRED,
        quantized_import=Support.REJECTED,
        reason=(
            _RUNTIME_VALIDATION_PENDING
            + " The admitted subset rejects learned positions, Q/K norms, KQV clipping, "
            "AWQ activation scales, and inconsistent optional bias families. Quantization "
            "preservation is rejected because fused QKV must be split."
        ),
    ),
    GGUFArchitectureSpec(
        gguf_arch="gptneox",
        model_type="gpt_neox",
        module_type="gguf_legacy",
        tensor_map_recipe=("legacy_layernorm",),
        config_postprocessor="exact_legacy_gguf",
        required_metadata=("attention.layer_norm_epsilon", "use_parallel_residual"),
        runtime=Support.DEFERRED,
        quantized_import=Support.REJECTED,
        reason=(
            _RUNTIME_VALIDATION_PENDING
            + " The admitted subset requires parallel residual MHA. Quantization "
            "preservation is rejected because fused QKV rows must be split."
        ),
    ),
    GGUFArchitectureSpec(
        gguf_arch="jais",
        model_type="jais",
        module_type="gguf_legacy",
        tensor_map_recipe=("legacy_layernorm",),
        config_postprocessor="exact_legacy_gguf",
        required_metadata=("attention.layer_norm_epsilon", "attention.max_alibi_bias"),
        runtime=Support.DEFERRED,
        quantized_import=Support.REJECTED,
        reason=(
            _RUNTIME_VALIDATION_PENDING
            + " Converter-baked MuP scales are retained exactly, while fused biased QKV "
            "must be split and therefore cannot preserve packed quantization."
        ),
    ),
    GGUFArchitectureSpec(
        gguf_arch="refact",
        model_type="refact",
        module_type="gguf_legacy",
        tensor_map_recipe=("legacy_layernorm",),
        config_postprocessor="exact_legacy_gguf",
        required_metadata=("attention.layer_norm_rms_epsilon",),
        runtime=Support.DEFERRED,
        quantized_import=Support.REJECTED,
        reason=(
            _RUNTIME_VALIDATION_PENDING
            + " Import is narrowed to split, bias-free dense tensors with one KV head; "
            "loaded-but-unexecuted expert, RoPE-factor, and bias families are rejected. "
            "Quantization preservation is rejected because this dedicated graph does not "
            "yet provide quantization-aware projection targets; use keep_quantized=False."
        ),
    ),
    GGUFArchitectureSpec(
        gguf_arch="ernie4_5",
        model_type="ernie4_5",
        module_type="gguf_legacy",
        tensor_map_recipe=("legacy_layernorm",),
        config_postprocessor="exact_legacy_gguf",
        required_metadata=("attention.layer_norm_rms_epsilon",),
        runtime=Support.DEFERRED,
        quantized_import=Support.REJECTED,
        reason=(
            _RUNTIME_VALIDATION_PENDING
            + " Import is narrowed to the dense split-Q/K/V, split-SwiGLU, full-RoPE "
            "variant and rejects all expert, fused, sectioned-position, and bias alternatives. "
            "Quantization preservation is rejected because this dedicated graph does not "
            "yet provide quantization-aware projection targets; use keep_quantized=False."
        ),
    ),
    # ------------------------------------------------ known but not importable
    GGUFArchitectureSpec(
        gguf_arch="bloom",
        model_type="bloom",
        tensor_map_recipe=("bloom",),
        tensor_processor="bloom",
        config_postprocessor="conventional_legacy",
        required_metadata=("attention.layer_norm_epsilon",),
        runtime=Support.DEFERRED,
        quantized_import=Support.REJECTED,
        reason=(
            _RUNTIME_VALIDATION_PENDING
            + " Quantization preservation is rejected because canonical Bloom GGUF stores "
            "one fused QKV projection that must be reordered and split into three graph targets."
        ),
    ),
    GGUFArchitectureSpec(
        gguf_arch="codeshell",
        model_type="kclgpt",
        tensor_map_recipe=("legacy_layernorm",),
        config_postprocessor="conventional_legacy",
        required_metadata=("attention.layer_norm_epsilon",),
        runtime=Support.DEFERRED,
        quantized_import=Support.REJECTED,
        reason=(
            _RUNTIME_VALIDATION_PENDING
            + " Quantization preservation is rejected because the pinned loader accepts a "
            "fused QKV tensor that must be split into separate graph projections."
        ),
    ),
    GGUFArchitectureSpec(
        gguf_arch="command-r",
        model_type="command_r",
        tensor_map_recipe=("llama", "command_r_extras"),
        config_postprocessor="conventional_legacy",
        required_metadata=("attention.layer_norm_epsilon", "logit_scale"),
        runtime=Support.DEFERRED,
        reason=(
            _RUNTIME_VALIDATION_PENDING
            + " Import requires canonical logit_scale metadata and is restricted to split "
            "Q/K/V tensors in the 40-layer Command-R profile; quantization preservation is "
            "supported only for that split route. Pinned variants with "
            "64 or more layers require distinct per-head Q/K LayerNorm parameters that the "
            "current Attention graph cannot represent."
        ),
    ),
    GGUFArchitectureSpec(
        gguf_arch="jais2",
        model_type="jais2",
        tensor_map_recipe=("legacy_layernorm",),
        config_postprocessor="conventional_legacy",
        required_metadata=("attention.layer_norm_epsilon",),
        runtime=Support.DEFERRED,
        reason=_RUNTIME_VALIDATION_PENDING,
    ),
    GGUFArchitectureSpec(
        gguf_arch="orion",
        model_type="orion",
        tensor_map_recipe=("legacy_layernorm",),
        config_postprocessor="conventional_legacy",
        required_metadata=("attention.layer_norm_epsilon",),
        runtime=Support.DEFERRED,
        quantized_import=Support.REJECTED,
        reason=(
            _RUNTIME_VALIDATION_PENDING
            + " Fused QKV input is rejected because its import transform is not implemented. "
            "Quantization preservation is also rejected for this architecture."
        ),
    ),
    GGUFArchitectureSpec(
        gguf_arch="qwen",
        model_type="qwen",
        tensor_map_recipe=("llama", "qwen1_extras"),
        config_postprocessor="conventional_legacy",
        required_metadata=("attention.layer_norm_rms_epsilon",),
        runtime=Support.DEFERRED,
        quantized_import=Support.REJECTED,
        reason=(
            _RUNTIME_VALIDATION_PENDING
            + " Quantization preservation is rejected because Qwen v1 stores fused QKV "
            "weights that must be split into separate graph projections."
        ),
    ),
    GGUFArchitectureSpec(
        gguf_arch="starcoder",
        model_type="gpt_bigcode",
        tensor_map_recipe=("starcoder",),
        config_postprocessor="conventional_legacy",
        required_metadata=("attention.layer_norm_epsilon",),
        runtime=Support.DEFERRED,
        quantized_import=Support.REJECTED,
        reason=(
            _RUNTIME_VALIDATION_PENDING
            + " Quantization preservation is rejected because StarCoder stores one fused "
            "biased MQA projection that must be split for the graph."
        ),
    ),
    GGUFArchitectureSpec(
        gguf_arch="xverse",
        model_type="xverse",
        tensor_map_recipe=("llama",),
        tensor_processor="xverse",
        config_postprocessor="conventional_legacy",
        required_metadata=("attention.layer_norm_rms_epsilon",),
        runtime=Support.DEFERRED,
        quantized_import=Support.REJECTED,
        reason=(
            _RUNTIME_VALIDATION_PENDING
            + " Fused QKV input is rejected because it cannot be combined truthfully with the "
            "required architecture-specific Q/K row permutations. Quantization preservation "
            "is also rejected for this architecture."
        ),
    ),
    GGUFArchitectureSpec(
        gguf_arch="t5",
        model_type="t5",
        tensor_map_recipe=("t5",),
        config_key_map="t5",
        config_postprocessor="t5",
        required_metadata=(
            "context_length",
            "embedding_length",
            "block_count",
            "feed_forward_length",
            "attention.head_count",
            "attention.layer_norm_rms_epsilon",
            "attention.relative_buckets_count",
        ),
        runtime=Support.DEFERRED,
        reason=(
            "Graph import is covered, but no independent full-logit and generation "
            "parity run has yet validated a pinned real T5 GGUF runtime package."
        ),
    ),
    GGUFArchitectureSpec(
        gguf_arch="t5encoder",
        model_type="t5encoder",
        tensor_map_recipe=("t5",),
        config_key_map="t5",
        config_postprocessor="t5",
        required_metadata=(
            "context_length",
            "embedding_length",
            "block_count",
            "feed_forward_length",
            "attention.head_count",
            "attention.layer_norm_rms_epsilon",
            "attention.relative_buckets_count",
        ),
        runtime=Support.DEFERRED,
        reason=(
            "Encoder hidden-state import is covered, but the pinned real artifact lacks "
            "independent provenance and full hidden-state parity evidence."
        ),
    ),
    # ------------------------------------------------------ Audio / TTS / codec
    GGUFArchitectureSpec(
        gguf_arch="pockettts",
        config=Support.DEFERRED,
        tensor_map=Support.DEFERRED,
        graph=Support.DEFERRED,
        runtime=Support.DEFERRED,
        reason=_POCKETTTS_BUNDLE_REASON,
    ),
    GGUFArchitectureSpec(
        gguf_arch="qwen3tts",
        config=Support.DEFERRED,
        tensor_map=Support.DEFERRED,
        graph=Support.DEFERRED,
        runtime=Support.DEFERRED,
        reason=_QWEN3TTS_BUNDLE_REASON,
    ),
    GGUFArchitectureSpec(
        gguf_arch="talkie",
        config=Support.DEFERRED,
        tensor_map=Support.DEFERRED,
        graph=Support.DEFERRED,
        runtime=Support.DEFERRED,
        reason=_TALKIE_GRAPH_REASON,
    ),
    GGUFArchitectureSpec(
        gguf_arch="wavtokenizer-dec",
        config=Support.DEFERRED,
        tensor_map=Support.DEFERRED,
        graph=Support.DEFERRED,
        runtime=Support.DEFERRED,
        reason=_WAVTOKENIZER_DEC_REASON,
    ),
    *(
        GGUFArchitectureSpec(
            gguf_arch=architecture,
            config=Support.DEFERRED,
            tensor_map=Support.DEFERRED,
            graph=Support.DEFERRED,
            runtime=Support.DEFERRED,
            reason=reason,
        )
        for architecture, reason in _FINAL_CENSUS_DEFERRED_REASONS.items()
        if architecture
        not in {
            "codeshell",
            "command-r",
            "ernie4_5",
            "gptneox",
            "jais",
            "jais2",
            "orion",
            "qwen",
            "refact",
            "starcoder",
            "xverse",
        }
    ),
    GGUFArchitectureSpec(
        gguf_arch="arwkv7",
        config=Support.DEFERRED,
        tensor_map=Support.DEFERRED,
        graph=Support.DEFERRED,
        runtime=Support.DEFERRED,
        reason=_RWKV_GRAPH_REASONS["arwkv7"],
    ),
    GGUFArchitectureSpec(
        gguf_arch="rwkv6",
        config=Support.DEFERRED,
        tensor_map=Support.DEFERRED,
        graph=Support.DEFERRED,
        runtime=Support.DEFERRED,
        reason=_RWKV_GRAPH_REASONS["rwkv6"],
    ),
    GGUFArchitectureSpec(
        gguf_arch="rwkv6qwen2",
        config=Support.DEFERRED,
        tensor_map=Support.DEFERRED,
        graph=Support.DEFERRED,
        runtime=Support.DEFERRED,
        reason=_RWKV_GRAPH_REASONS["rwkv6qwen2"],
    ),
    GGUFArchitectureSpec(
        gguf_arch="rwkv7",
        config=Support.DEFERRED,
        tensor_map=Support.DEFERRED,
        graph=Support.DEFERRED,
        runtime=Support.DEFERRED,
        reason=_RWKV_GRAPH_REASONS["rwkv7"],
    ),
    GGUFArchitectureSpec(
        gguf_arch=MMPROJ_ARCHITECTURE,
        config=Support.REJECTED,
        tensor_map=Support.REJECTED,
        graph=Support.REJECTED,
        runtime=Support.REJECTED,
        quantized_import=Support.REJECTED,
        reason=_MMPROJ_REASON,
    ),
    GGUFArchitectureSpec(
        gguf_arch="gptj",
        config=Support.REJECTED,
        tensor_map=Support.REJECTED,
        graph=Support.REJECTED,
        runtime=Support.REJECTED,
        reason=(
            "The pinned census reserves gptj but llama.cpp has no model loader for it. "
            "There is no bounded header-to-tensor contract to import, so conversion is "
            "rejected before config extraction."
        ),
    ),
)


@functools.lru_cache(maxsize=1)
def _index() -> MappingProxyType[str, GGUFArchitectureSpec]:
    """Index every canonical name and alias, rejecting collisions."""
    index: dict[str, GGUFArchitectureSpec] = {}
    for spec in _SPECS:
        for name in spec.names:
            existing = index.get(name)
            if existing is not None:
                raise ValueError(
                    f"GGUF architecture {name!r} is claimed by both "
                    f"{existing.gguf_arch!r} and {spec.gguf_arch!r}"
                )
            index[name] = spec
    return MappingProxyType(index)


def _validate_census_closure(
    specs: tuple[GGUFArchitectureSpec, ...],
    upstream_names: frozenset[str],
) -> None:
    """Reject registry/pin drift before architecture dispatch can run."""
    counts = Counter(spec.gguf_arch for spec in specs)
    duplicates = sorted(name for name, count in counts.items() if count != 1)
    canonical = set(counts)
    missing = sorted(upstream_names - canonical)
    extra = sorted(canonical - upstream_names)
    aliases = {alias: spec.gguf_arch for spec in specs for alias in spec.aliases}
    alias_drift = sorted(set(aliases) & upstream_names)
    if duplicates or missing or extra or alias_drift:
        raise ValueError(
            "GGUF architecture registry does not exactly close the pinned llama.cpp "
            f"census: duplicates={duplicates}, missing={missing}, extra={extra}, "
            f"aliases_that_became_canonical={alias_drift}"
        )


_validate_census_closure(_SPECS, frozenset(upstream_architectures()))


def iter_arch_specs() -> tuple[GGUFArchitectureSpec, ...]:
    """Return every registered architecture spec, in declaration order."""
    return _SPECS


def try_get_arch_spec(architecture: str) -> GGUFArchitectureSpec | None:
    """Return the spec for *architecture*, or ``None`` when it is unregistered.

    Accepts canonical names and declared aliases, case-insensitively.
    """
    return _index().get(architecture.lower())


def _unknown_architecture_message(architecture: str) -> str:
    """Build an actionable message for an architecture with no spec."""
    upstream = upstream_architecture(architecture)
    prefix = f"Unsupported GGUF architecture: {architecture!r}."
    if upstream is None:
        return (
            f"{prefix} It is not among the {len(upstream_architectures())} "
            "architectures llama.cpp defines at the pinned commit, so the file is "
            "either newer than this build of mobius or not a llama.cpp GGUF. "
            f"Supported: {', '.join(supported_architectures())}."
        )
    if not upstream.cpp_loader:
        return (
            f"{prefix} It is registered upstream but has no llama.cpp model loader, "
            "so no tool can load it. The file cannot be converted."
        )
    return (
        f"{prefix} It is a real llama.cpp architecture (upstream cohort "
        f"{upstream.cohort}) that mobius does not import yet. Build from the "
        "Hugging Face checkpoint with `mobius build` instead. Supported: "
        f"{', '.join(supported_architectures())}."
    )


def get_arch_spec(architecture: str) -> GGUFArchitectureSpec:
    """Return the importable spec for *architecture*.

    Args:
        architecture: A GGUF ``general.architecture`` value.

    Returns:
        The matching spec, which is guaranteed importable.

    Raises:
        DisabledGGUFArchitectureError: The architecture is known but deliberately
            turned off, because converting it would build a wrong graph.
        UnsupportedGGUFArchitectureError: The architecture is unregistered, or one of
            its capabilities is not ``SUPPORTED``.
    """
    spec = try_get_arch_spec(architecture)
    if spec is None:
        raise UnsupportedGGUFArchitectureError(_unknown_architecture_message(architecture))
    if spec.is_importable:
        return spec

    blocked = sorted(
        name for name, verdict in spec.verdicts.items() if verdict is not Support.SUPPORTED
    )
    detail = (
        f"Unsupported GGUF architecture: {architecture!r}. "
        f"{', '.join(blocked)} unavailable. {spec.reason}"
    )
    if any(verdict is Support.REJECTED for verdict in spec.verdicts.values()):
        raise DisabledGGUFArchitectureError(detail)
    raise UnsupportedGGUFArchitectureError(detail)


@functools.lru_cache(maxsize=1)
def supported_architectures() -> tuple[str, ...]:
    """Return the sorted canonical names of every importable architecture."""
    return tuple(sorted(spec.gguf_arch for spec in _SPECS if spec.is_importable))


def model_type_for(architecture: str) -> str | None:
    """Return the mobius ``model_type`` for *architecture*, if it has one."""
    spec = try_get_arch_spec(architecture)
    return None if spec is None else spec.model_type


def arch_names_with(predicate: Callable[[GGUFArchitectureSpec], bool]) -> frozenset[str]:
    """Return every canonical name and alias whose spec satisfies *predicate*.

    Used to derive the architecture-keyed frozensets that the builder and the
    tensor-mapping module used to declare by hand.
    """
    return frozenset(name for spec in _SPECS if predicate(spec) for name in spec.names)
