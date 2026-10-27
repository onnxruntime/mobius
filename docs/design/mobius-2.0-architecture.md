# Mobius 2.0 & Olive 2.0 Architecture Design: Decoupling Canonical Model Representation from Hardware Lowering

## 1. Executive Summary & Problem Statement

### 1.1 The $O(M \times H)$ Combinatorial Explosion

Modern local AI inference spans **$M$ rapidly evolving model architectures** (dense LLMs, MoEs, hybrid SSM/DeltaNet-Attention models, Vision-Language-Audio models, Diffusion DiTs, and stateful World Models) across **$H$ heterogeneous hardware backends** (`CUDA`, `DirectML`, `Qualcomm QNN HTP`, `Intel OpenVINO NPU`, `WebGPU`, `TensorRT-RTX`, `Apple MLX`, and `CPU`).

Because ONNX was originally designed in the ResNet/BERT era as a **single-level, static dataflow exchange format** rather than a multi-level compiler IR, it suffers from a fundamental **Granularity Paradox**:

1. **If graphs are exported as flat, fine-grained standard ops** (`MatMul`, `Add`, `Softmax`, `Reshape`, `Slice`, `Scan`), GPU/CPU runtimes suffer severe memory-bandwidth bottlenecks without fused kernels (FlashAttention, GroupQueryAttention, fused RoPE, fused MoE). Reconstructing high-level semantics downstream via C++ or Python subgraph pattern matching is brittle: minor architectural variations (e.g., QK-Norm in Qwen3/Gemma4, logit softcapping, gated residual scaling, or RoPE layout permutations) silently break pattern matchers and trigger catastrophic CPU fallbacks (`MemcpyToHost`).
2. **If graphs are exported with coarse-grained custom/contrib ops** (`com.microsoft::GroupQueryAttention`, `SkipSimplifiedLayerNormalization`, `MatMulNBits`, `QMoE`), NPU compilers (QNN HTP, OpenVINO) and shader backends (WebGPU, DirectML) either reject the graph or require incompatible operator signatures and tensor layouts.

In **Mobius 1.x**, this tension forced the model exporter to absorb hardware-specific lowering logic (`src/mobius/_execution_providers.py`, `src/mobius/_optimizations.py`, `src/mobius/rewrite_rules/`), while model authors still had to hand-write a new Python class under `src/mobius/models/` (now 220+ files) for every new HuggingFace release.

### 1.2 Why `torch.export` and `GGUF` Do Not Solve This

- **Why `torch.export` was rejected**: Upstream HuggingFace `transformers` / `diffusers` implementations are written as imperative PyTorch Eager code entangled with training paths, Python state objects (`DynamicCache`), dynamic control flow, eager mask slicing tricks, and custom CUDA/Triton kernels. Tracing at the low-level ATen operator boundary produces fragile "spaghetti graphs" polluted with eager implementation artifacts that cannot be cleanly lowered to NPUs or fused by downstream passes.
- **Why `GGUF / ggml` is not sufficient**: GGUF is a weight + metadata container, not a computation graph IR. Its computation graphs are hardcoded in C++ inside `llama.cpp` (`llm_build_*`). Supporting a new architecture still requires hand-writing C++ graph builders and recompiling the runtime; moreover, `ggml`'s host-driven operator dispatch lacks the whole-graph AOT compilation, SRAM tiling, and static memory planning required by NPUs, as well as the general graph expressiveness needed for hybrid Diffusion and World Model pipelines.

### 1.3 The Mobius 2.0 + Olive 2.0 Vision

**Mobius 2.0** and **Olive 2.0** reduce the $O(M \times H)$ matrix into **$O(M) + O(H)$** through a strict architectural boundary:

```
HuggingFace Checkpoint + Architecture Spec
                   │
                   ▼
┌──────────────────────────────────────────────────────────────────────┐
│ MOBIUS 2.0 — O(M) Canonical Semantic Graph Builder                   │
│  • Zero EP branching: always emits Hierarchical Standard ONNX        │
│  • Preserves high-level Component boundaries via ONNX FunctionProto  │
│  • Declarative Hybrid Backbone + Weight Mapping DSL (80% config-only)│
│  • AST + Safetensors Agent Synthesis with L1/L2 numerical oracle     │
│  • Unified Multi-Component + Stateful ModelPackage contract          │
└──────────────────────────────────────────────────────────────────────┘
                   │
                   │  Single Canonical Standard ONNX ModelPackage
                   ▼
┌──────────────────────────────────────────────────────────────────────┐
│ OLIVE 2.0 — O(H) Hardware Compiler, Quantizer & EP Lowering Engine   │
│  • Accelerator-agnostic MobiusBuilder caching 1 canonical package    │
│  • Function-Boundary Graph Surgery (O(1) replacement vs regex match) │
│  • EP-specific Lowering & Legalization (CUDA, DML, QNN, OV, WebGPU)  │
│  • Orthogonal Block-Quantization & QDQ Lowering (MatMulNBits / QDQ)  │
│  • Target ABI Adaptation (KV buffer sharing, 4D PLI, 256MB chunking) │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 2. Part I: Mobius 2.0 Improvements ($O(M)$ Canonical Model Layer)

### 2.1 Always-Standard ONNX with Hierarchical `FunctionProto` Boundaries

Mobius 2.0 eliminates the `--execution-provider` axis from graph construction. Every `mobius build` produces **one canonical Standard ONNX (`ai.onnx` opset 24) graph**.

Crucially, Mobius 2.0 **must not flatten components into an unstructured sea of primitive ops**. Doing so would force Olive to rediscover high-level blocks via fragile subgraph pattern matching. Instead, Mobius 2.0 introduces **Hierarchical Standard ONNX**:

1. **Dual-Level Representation (`ir.Function` + Standard Body)**:
   - Every reusable building block in `mobius.components` (`Attention`, `RotaryEmbedding`, `MRoPE`, `RMSNorm`, `LayerNorm`, `FusedGateUpMLP`, `MoELayer`, `GatedDeltaNet`, `CausalConvWithState`, `AdaLayerNormZero`) is emitted as a call to a local **`ir.Function` (`FunctionProto`)** in the `mobius.components` domain.
   - The body of each `ir.Function` is composed **100% of conformant Standard ONNX ops (`ai.onnx` opset 24)**.
2. **Semantic Attributes on Function Calls**:
   - Each function call node carries explicit, machine-readable attributes and metadata (`q_num_heads`, `kv_num_heads`, `head_dim`, `scale`, `softcap`, `sliding_window`, `rope_theta`, `interleaved`, `moe_topk`, `norm_eps`) so downstream Olive passes never have to reverse-engineer hyperparameters from constant tensors inside the subgraph.
3. **Universal Portability + $O(1)$ Optimization**:
   - Any standard ONNX runtime can execute the model directly or inline all `ir.Function` bodies in a single pass (`onnx_ir.passes.common.InlinePass`).
   - Olive 2.0 can inspect the top-level `ir.Function` node and either replace it in $O(1)$ with an EP-specific contrib op (`com.microsoft::GroupQueryAttention`, `SkipSimplifiedLayerNormalization`, `QMoE`) or inline its standard body for NPUs.

### 2.2 Complete Removal of EP Conditional Logic from Mobius

All EP-specific capability flags currently in `src/mobius/_execution_providers.py` and consumed across `models/`, `components/`, and `tasks/` are removed from Mobius graph construction:

| Current Mobius 1.x EP Coupling | Mobius 2.0 Canonical Output | Olive 2.0 Target Pass |
| :--- | :--- | :--- |
| `ep.gqa_dtypes` / `pack_qkv_for_gqa_rules` | Canonical `mobius.components::Attention` function | `AttentionToGroupQueryAttention` + `PackQKVForGroupQueryAttention` |
| `ep.supports_fused_rope=False` (DML/QNN) | Canonical `mobius.components::RotaryEmbedding` function | `SeparateGroupQueryAttentionRoPE` / `InlinePass` |
| `ep.supports_skip_layer_norm=False` (OV/TRT/QNN) | Separate `Add` + `mobius.components::RMSNorm` | `FuseSkipRMSNormalization` (only on EPs that support it) |
| `ep.supports_rank4_rmsnorm=False` (QNN HTP) | Standard rank-4 `RMSNormalization` | `Rank4RMSNormToRank3` in Olive QNN pipeline |
| `ep.supports_tensor_scatter=False` (QNN HTP) | Standard `TensorScatter` | `TensorScatterToScatterND` in Olive QNN pipeline |
| `ep.layered_per_layer_inputs=True` (OpenVINO) | Canonical `[B, S, L, D]` Per-Layer Input tensor | Olive ABI Lowering flattens to `[B, S, L*D]` for non-OV EPs |
| `ep.max_buffer_size=256MB` (WebGPU) | Unsplit canonical `Embedding` table | Olive `SplitLargeInitializerBuffers` pass for WebGPU |
| `ep.supports_fp8_kv_cache=True` (CUDA) | Model-dtype KV cache I/O | Olive `Fp8KvCachePass` for CUDA SM89+ |

### 2.3 Declarative Hybrid Backbone & Weight-Mapping DSL (Eliminating 80% of Hand-Written Model Files)

An audit of the 220+ files in `src/mobius/models/` shows that **over 80% of new model architectures do not introduce any new mathematical operators or components**. They only vary along three declarative axes:
1. **Block Composition & Layer Schedule**: Interleaving different block types across layers (e.g., `[GatedDeltaNet, GatedDeltaNet, GatedDeltaNet, FullAttention] * 16` in Qwen3.5/3.8, or alternating Sliding-Window and Global Attention in Gemma3/4).
2. **Block Micro-Topology Hooks**: Pre-Norm vs. Post-Norm, QK-Norm (per-head vs. full-dim), logit softcapping, residual multipliers, and KV-cache sharing across layers (`shared_kv_source_layer`).
3. **Checkpoint Weight Naming & Layout**: Renaming prefixes, splitting packed `wqkv` or `gate_up_proj` tensors, stacking expert weights, or transposing Conv1d weights.

Mobius 2.0 replaces repetitive `nn.Module` subclasses in `src/mobius/models/` with a two-part declarative engine:

#### A. `DeclarativeHybridModel` Blueprint Schema
Instead of writing a new Python `forward()` method for every model family, architectures are registered via a declarative `ArchitectureBlueprint`:

```python
gemma4_text_blueprint = ArchitectureBlueprint(
    model_type="gemma4_text",
    embedding=EmbeddingSpec(scale_by_sqrt_hidden=True, per_layer_inputs=True),
    layer_schedule="config.layer_types",  # e.g. ["sliding_attention", ..., "full_attention"]
    block_variants={
        "sliding_attention": BlockSpec(
            attn=AttentionSpec(
                qk_norm="rms_head_dim",
                rope="proportional_rope",
                sliding_window="config.sliding_window",
                softcap="config.attn_logit_softcapping",
                kv_sharing="config.kv_sharing_map",
            ),
            norms=NormTopology(pre_attn=True, post_attn=True, pre_mlp=True, post_mlp=True),
            mlp=MLPSpec(kind="gated_mlp", activation="gelu_pytorch_tanh"),
        ),
        "full_attention": BlockSpec(...),
    },
    final_norm=RMSNormSpec(eps="config.rms_norm_eps"),
    lm_head=LMHeadSpec(tie_word_embeddings="config.tie_word_embeddings", softcap="config.final_logit_softcapping"),
)
```

#### B. Declarative Weight-Mapping DSL
Instead of imperative `preprocess_weights(state_dict)` loops in every model file, Mobius 2.0 provides composable, lazy-streaming-compatible weight rules:

```python
weight_rules = WeightMappingSpec([
    StripPrefix("language_model."),
    SplitInitializer(
        source="model.layers.{layer}.self_attn.qkv_proj.weight",
        targets=[
            ("model.layers.{layer}.self_attn.q_proj.weight", "q_dim"),
            ("model.layers.{layer}.self_attn.k_proj.weight", "kv_dim"),
            ("model.layers.{layer}.self_attn.v_proj.weight", "kv_dim"),
        ],
        dim=0,
    ),
    StackExperts(
        source_pattern="model.layers.{layer}.mlp.experts.{expert}.{proj}.weight",
        target="model.layers.{layer}.mlp.experts.{proj}.weight",
    ),
])
```
Because these rules are declarative and invertible, **lazy safetensors streaming (`StreamingTransformedWeightSource`) and Olive's pre-quantized sidecar binding work automatically for every model without custom per-model glue code**.

### 2.4 AST + Safetensors Guided Agent Synthesis for Novel Architectures

For the remaining ~20% of models that introduce genuinely new mathematical blocks (e.g., DeepSeek MLA/CSA, Qwen GatedDeltaNet, or a novel Diffusion/World-Model conditioning block):
1. Engineers or the automated synthesis agent only implement the **isolated mathematical leaf component** in `src/mobius/components/` using Standard ONNX ops.
2. Because Mobius 2.0 no longer requires EP-specific branching or manual quantized-sidecar plumbing inside model classes, an automated **AST + Safetensors Transpiler Workflow** can inspect the upstream HuggingFace `modeling_<arch>.py` AST alongside `model.safetensors.index.json`, synthesize the `ArchitectureBlueprint` + leaf `Component`, and validate it automatically against the **L1 (graph build + shape inference)** and **L2 (layer-by-layer numerical parity vs. PyTorch Eager at `atol=1e-4`)** gates.

### 2.5 First-Class Stateful & Multi-Component `ModelPackage` Contract

To natively support **Multimodal VLMs, Audio/Speech pipelines, Diffusion DiTs, and Stateful World Models** without ad-hoc external scripts:
- `ModelPackage` and `ComponentManifest` elevate **State Descriptors** (`StateSpec`) alongside `SharedWeightInfo`:
  - `KVCacheStateSpec(kind="paged" | "dynamic" | "static", layers=..., kv_heads=..., head_dim=...)`
  - `RecurrentStateSpec(kind="conv_state" | "ssm_state", shape=..., dtype=...)`
  - `LatentBufferStateSpec(kind="ring_buffer" | "ode_latent", shape=..., dtype=...)`
- Multi-stage pipelines (e.g., `@vision_encoder`, `@audio_encoder`, `@embedding`, `@decoder`, `@denoise_step`, `@world_state_transition`) are exported as named components in a single self-contained `ModelPackage` with verified cross-component tensor contracts and shared weight ownership.

---

## 3. Part II: Olive 2.0 Improvements ($O(H)$ Hardware Lowering & Optimization Layer)

With Mobius 2.0 guaranteeing a clean, hierarchical Standard ONNX `ModelPackage`, **Olive 2.0** becomes the single source of truth for all hardware-specific graph transformations, quantization, and runtime packaging.

### 3.1 Accelerator-Agnostic `MobiusBuilder` Pass

In `olive/passes/onnx/mobius_model_builder.py`:
1. `MobiusBuilder` defaults to `execution_provider="onnx-standard"` and sets `is_accelerator_agnostic = True`.
2. In a multi-target Olive workflow (e.g., targeting `CUDA`, `QNN HTP`, `OpenVINO NPU`, and `WebGPU` from the same checkpoint), **Olive runs `MobiusBuilder` exactly once**, caching the canonical Standard ONNX `ModelPackage` and fanning out to target-specific lowering pipelines.

### 3.2 Function-Boundary-Driven Graph Surgery (`FunctionTargetSurgeon`)

Currently, `olive/passes/onnx/graph_surgery/` relies on `RewriteRuleSurgeon` to pattern-match primitive ops after functions have been inlined. In Olive 2.0, `GraphSurgeries` operates directly on **Mobius 2.0 `ir.Function` boundaries** before inlining:

```
Canonical Mobius 2.0 Graph (with mobius.components::* Function nodes)
         │
         ├─► [Olive Target EP Capability Lookup]
         │
         ├── If EP has native fused kernel (e.g., CUDA / DML / WebGPU GQA):
         │     Replace `mobius.components::Attention` + `RotaryEmbedding` call
         │     directly with `com.microsoft::GroupQueryAttention` in O(1) using
         │     the function node's semantic attributes.
         │
         └── If EP requires primitive ops (e.g., QNN HTP / OpenVINO NPU):
               1. Inline `mobius.components::*` function body to Standard ONNX
               2. Run EP-specific primitive legalizations:
                  - QNN: `DecomposeAttention`, `DecomposeOnnxRotaryEmbedding`,
                         `Rank4RMSNormToRank3`, `TensorScatterToScatterND`
                  - OpenVINO: Preserve `RMSNormalization` (opset 24) and
                              `Attention` (opset 24) without `SkipLayerNorm`
```

**Why this is transformative**:
- **Zero false negatives**: Architectural variations (such as QK-Norm between QKV projection and RoPE, or attention logit softcapping) are explicit inputs/attributes on the `mobius.components::Attention` function boundary rather than unexpected nodes that break a 25-node `onnxscript` pattern matcher.
- **Clean separation of concerns**: Adding a new fused kernel for a new hardware backend (e.g., a fused `GatedDeltaNet` or `MLA` kernel on CUDA or NPU) only requires adding a `FunctionTargetSurgeon` in Olive—zero changes to Mobius model definitions.

### 3.3 Target ABI, Layout, and Buffer Legalization Passes in Olive

Olive 2.0 consolidates all hardware-imposed I/O and memory constraints into composable passes under `olive/passes/onnx/graph_surgery/lowering.py`:

1. **`LowerPerLayerInputsPass`**:
   - Keeps 4D `[B, S, L, D]` Per-Layer Inputs for OpenVINO NPU (`layered_per_layer_inputs=True`), and inserts the `[B, S, L*D]` flatten/reshape boundary for CUDA, CPU, DML, and WebGPU.
2. **`SplitLargeInitializersPass` (WebGPU 256 MiB Limit)**:
   - Inspects target `max_buffer_size` (e.g., `268_435_456` bytes for WebGPU) and automatically splits oversized `Gather` / `MatMul` weight tables into chunked sub-tensors with `Concat`/`Select` stitching.
3. **`GraphCaptureLegalizationPass` (WebGPU / Static Graph Capture)**:
   - Rewrites dynamic `Shape` / `ConstantOfShape` subgraphs on shared-KV and attention-bias paths into static-shape-compatible operations (`StaticEmptyKV`) when `requires_graph_capture_rewrite=True`.
4. **`StateAbiLoweringPass` (KV Cache & Recurrent State Packaging)**:
   - Adapts canonical `StateSpec` descriptors to the target runtime contract:
     - Converts dynamic KV concat to in-place `GroupQueryAttention` shared buffers (`past_present_share_buffer=True`) for CUDA/DML/WebGPU/TRT-RTX.
     - Retypes KV cache I/O to `FLOAT8E4M3FN` (`Fp8KvCachePass`) on CUDA SM89+.
     - Normalizes hybrid recurrent state names (`conv_state` / `recurrent_state`) for ORT GenAI compatibility.

### 3.4 Orthogonal Quantization & Shared-Weight Lowering

In Olive 2.0, weight quantization operates cleanly on the canonical Standard ONNX graph prior to EP fusion:
1. **Unified Weight Quantization (`OnnxMatMulNBits` / `NVModelOpt` / `INC` / `AIMET`)**:
   - Quantizes `Linear`, `Embedding`, and `MoELayer` weights on the canonical graph.
   - Respects `ComponentManifest.shared_weights` (`SharedWeightInfo`): when `embedding` and `decoder` are quantized in the same Olive workflow, tied tables share quantized initializers; when exported standalone, Olive materializes independent quantized tensors.
2. **EP-Specific Quantization Format Lowering**:
   - **CUDA / DML / WebGPU / CPU**: Retains `com.microsoft::MatMulNBits` and applies `PackQKVForGroupQueryAttention` (folding packed `q_weight`, `k_weight`, `v_weight`, scales, and zero-points via `FoldConcatInitializersPass`) and `FuseBlockQuantizedMoE` (`QMoE`).
   - **QNN HTP / TRT-RTX / OpenVINO**: Converts `MatMulNBits` to standard `DequantizeLinear + MatMul` (QDQ) via `MatMulNBitsToQDQ` (`olive/passes/onnx/mnb_to_qdq.py`).

### 3.5 Unified `EpLoweringPipeline` Profiles in Olive

Instead of requiring users to manually chain 10 individual `GraphSurgeries` and quantization passes in JSON configs, Olive 2.0 provides declarative **EP Target Profiles** (`olive/passes/onnx/genai_runtime_profiles.py`) that expand into the exact lowering sequence for each `(AcceleratorSpec, Precision)`:

| Target Profile | Olive 2.0 Automated Lowering Sequence |
| :--- | :--- |
| **`cuda`** | `CanonicalMobius` $\rightarrow$ `Quantize(MatMulNBits)` $\rightarrow$ `FunctionToGQA(pack_qkv=True, fused_rope=True)` $\rightarrow$ `FuseSkipNorm` $\rightarrow$ `FuseQMoE` $\rightarrow$ `FoldInitializers` $\rightarrow$ `GenAIConfig(cuda_graph=True, share_buffer=True)` |
| **`dml`** | `CanonicalMobius` $\rightarrow$ `Quantize(MatMulNBits)` $\rightarrow$ `FunctionToGQA(pack_qkv=False, fused_rope=False)` $\rightarrow$ `FuseSkipNorm` $\rightarrow$ `FoldInitializers` $\rightarrow$ `GenAIConfig(share_buffer=True)` |
| **`webgpu`** | `CanonicalMobius` $\rightarrow$ `Quantize(MatMulNBits)` $\rightarrow$ `FunctionToGQA(pack_qkv=True)` $\rightarrow$ `SplitLargeInitializers(256MB)` $\rightarrow$ `GraphCaptureLegalization` $\rightarrow$ `GenAIConfig(graph_capture=True, cap_kv=True)` |
| **`openvino-npu`** | `CanonicalMobius` $\rightarrow$ `InlineFunctions(keep={Attention, RMSNorm, RotaryEmbedding})` $\rightarrow$ `Keep4DPerLayerInputs` $\rightarrow$ `OpenVINOQuantize/Encapsulate` $\rightarrow$ `GenAIConfig(device_type=NPU)` |
| **`qnn-htp`** | `CanonicalMobius` $\rightarrow$ `InlineAllFunctions` $\rightarrow$ `DecomposeAttention & RoPE` $\rightarrow$ `Rank4RMSNormToRank3` $\rightarrow$ `TensorScatterToScatterND` $\rightarrow$ `MatMulNBitsToQDQ` $\rightarrow$ `QnnContextBinary` |

---

## 4. Part III: Long-Term Horizon — Next-Gen Cross-Platform IR for LLM, Diffusion & World Models

As AI models evolve from text-only Transformers to **Unified Multimodal World Models** (combining autoregressive token generation, continuous Diffusion/Flow-Matching denoisers, 3D spatio-temporal memory banks, and recurrent state-space layers), the `Mobius 2.0 -> Olive 2.0` architecture provides a bridge toward a true **Multi-Level Component & Tile IR**:

1. **Level 1 — Semantic Component & Stateful Module IR (Mobius 2.0 Output)**:
   - A single self-contained `ModelPackage` containing multiple callable entry points (`@encode_vision`, `@update_world_memory`, `@denoise_step`, `@decode_token`), shared block-quantized weights, and explicit `StateSpec` buffers.
2. **Level 2 — Known-Hotspot Kernel Dispatch + Tile-Level Auto-Fusion (Olive 2.0 + Backend Compilers)**:
   - **Known hotspots** (`Attention`, `GEMM`, `MoE`, `Conv`) map directly from `mobius.components::*` `FunctionProto` boundaries to hand-tuned vendor microkernels (FlashAttention, cuBLAS, QNN HTP, Intel XMX).
   - **Novel mathematical blocks** (new linear attention recurrences, custom world-model state transitions, 3D spatio-temporal conditioning) inline their Standard ONNX function bodies and lower to **Loop/Tile-level compilers** (e.g., MLIR `linalg` / StableHLO / IREE / WebGPU shader codegen / NPU DMA-tile schedulers), automatically generating fused hardware kernels without waiting for custom C++ operator registrations in ONNX Runtime.

---

## 5. Phased Implementation Roadmap

### Phase 1: Establish the Hierarchical Standard ONNX Contract
- [ ] Update `mobius.components` (`Attention`, `RotaryEmbedding`, `RMSNorm`, `MoELayer`, `GatedDeltaNet`, `CausalConvWithState`) to emit structured `ir.Function` nodes with semantic attributes and 100% Standard ONNX (opset 24) bodies.
- [ ] Make `execution_provider="onnx-standard"` the canonical default in Mobius 2.0 while preserving an opt-in `--inline-functions` flag for flat standard-ONNX consumers.

### Phase 2: Migrate All EP Lowering & Legalization from Mobius to Olive
- [ ] Update Olive's `MobiusBuilder` (`olive/passes/onnx/mobius_model_builder.py`) to build once with `onnx-standard` (`is_accelerator_agnostic = True`).
- [ ] Port remaining EP-specific passes from `src/mobius/_optimizations.py` and `src/mobius/_execution_providers.py` into Olive's `olive/passes/onnx/graph_surgery/` and `genai_runtime_profiles.py` (including 4D PLI handling, WebGPU 256 MiB buffer splitting, and `FunctionTargetSurgeon` rules).
- [ ] Deprecate `ep_capabilities()` branching inside `src/mobius/models/` and `src/mobius/tasks/`.

### Phase 3: Declarative Hybrid Backbone & Weight-Mapping DSL in Mobius
- [ ] Introduce `ArchitectureBlueprint` and `DeclarativeHybridModel` in `src/mobius/models/` to unify standard, sliding-window, MoE, and hybrid SSM/DeltaNet architectures under config-driven layer schedules.
- [ ] Introduce `WeightMappingSpec` to replace imperative `preprocess_weights` methods with declarative, streaming-safe weight transforms.

### Phase 4: Automated Model Synthesis & World-Model State Contracts
- [ ] Build the AST + Safetensors-driven synthesis tool that generates `ArchitectureBlueprint` definitions from HuggingFace repositories and validates them automatically with L1/L2 parity tests.
- [ ] Extend `ComponentManifest` with first-class `StateSpec` descriptors for unified KV-cache, recurrent SSM state, and Diffusion/World-Model latent memory orchestration across Mobius and Olive.
