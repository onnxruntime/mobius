# Mobius 2.0 & Olive 2.0 Architecture Design: Decoupling Canonical Model Representation from Hardware Lowering

## 1. Executive Summary & Problem Statement

### 1.1 The $O(M \times H)$ Combinatorial Explosion

Modern local AI inference spans **$M$ rapidly evolving model architectures** (dense LLMs, MoEs, hybrid SSM/DeltaNet-Attention models, Vision-Language-Audio models, Diffusion DiTs, and stateful World Models) across **$H$ heterogeneous hardware backends** (`CUDA`, `DirectML`, `Qualcomm QNN HTP`, `Intel OpenVINO NPU`, `WebGPU`, `TensorRT-RTX`, `Apple MLX`, and `CPU`).

Because ONNX was originally designed in the ResNet/BERT era as a **single-level, static dataflow exchange format** rather than a multi-level compiler IR, it suffers from a fundamental **Granularity Paradox**:

1. **If graphs are exported as flat, fine-grained standard ops** (`MatMul`, `Add`, `Softmax`, `Reshape`, `Slice`, `Scan`), GPU/CPU runtimes suffer severe memory-bandwidth bottlenecks without fused kernels (FlashAttention, GroupQueryAttention, fused RoPE, fused MoE). Reconstructing high-level semantics downstream via C++ or Python subgraph pattern matching is brittle: minor architectural variations (e.g., QK-Norm in Qwen3/Gemma4, logit softcapping, gated residual scaling, or RoPE layout permutations) silently break pattern matchers and trigger catastrophic CPU fallbacks (`MemcpyToHost`).
2. **If graphs are exported with coarse-grained custom/contrib ops** (`com.microsoft::GroupQueryAttention`, `SkipSimplifiedLayerNormalization`, `MatMulNBits`, `QMoE`), NPU compilers (QNN HTP, OpenVINO) and shader backends (WebGPU, DirectML) either reject the graph or require incompatible operator signatures and tensor layouts.

In **Mobius 1.x**, this tension forced the model exporter to absorb hardware-specific lowering logic (`src/mobius/_execution_providers.py`, `src/mobius/_optimizations.py`, `src/mobius/rewrite_rules/`). Architecture-specific composition and weight adaptation still require substantial work under `src/mobius/models/`, even where models already share a backbone.

### 1.2 Why `torch.export` and `GGUF` Do Not Solve This

- **Why `torch.export` was rejected**: Upstream HuggingFace `transformers` / `diffusers` implementations are written as imperative PyTorch Eager code entangled with training paths, Python state objects (`DynamicCache`), dynamic control flow, eager mask slicing tricks, and custom CUDA/Triton kernels. Tracing at the low-level ATen operator boundary produces fragile "spaghetti graphs" polluted with eager implementation artifacts that cannot be cleanly lowered to NPUs or fused by downstream passes.
- **Why `GGUF / ggml` is not sufficient**: GGUF is a weight + metadata container, not a computation graph IR. Its computation graphs are hardcoded in C++ inside `llama.cpp` (`llm_build_*`). Supporting a new architecture still requires hand-writing C++ graph builders and recompiling the runtime; moreover, `ggml`'s host-driven operator dispatch lacks the whole-graph AOT compilation, SRAM tiling, and static memory planning required by NPUs, as well as the general graph expressiveness needed for hybrid Diffusion and World Model pipelines.

### 1.3 The Mobius 2.0 + Olive 2.0 Vision

**Mobius 2.0** and **Olive 2.0** aim to move implementation maintenance toward **$O(M) + O(H)$** through a strict architectural boundary. This does not eliminate the model-by-target validation matrix: a backend still has to support a model's actual semantics.

```
HuggingFace Checkpoint + Architecture Spec
                   │
                   ▼
┌──────────────────────────────────────────────────────────────────────┐
│ MOBIUS 2.0 — O(M) Canonical Semantic Graph Builder                   │
│  • Zero EP branching: always emits Hierarchical Standard ONNX        │
│  • Preserves high-level Component boundaries via ONNX FunctionProto  │
│  • Incremental component composition and source weight mapping     │
│  • Existing Python model classes remain supported                  │
│  • Unified Multi-Component + Stateful ModelPackage contract          │
└──────────────────────────────────────────────────────────────────────┘
                   │
                   │  Canonical Standard ONNX ModelPackage per build specification
                   ▼
┌──────────────────────────────────────────────────────────────────────┐
│ OLIVE 2.0 — O(H) Hardware Compiler, Quantizer & EP Lowering Engine   │
│  • Accelerator-agnostic caching for each canonical build            │
│  • Function-boundary dispatch with semantic eligibility checks     │
│  • EP-specific Lowering & Legalization (CUDA, DML, QNN, OV, WebGPU)  │
│  • Orthogonal Block-Quantization & QDQ Lowering (MatMulNBits / QDQ)  │
│  • Target ABI Adaptation (KV buffer sharing, 4D PLI, 256MB chunking) │
└──────────────────────────────────────────────────────────────────────┘
```

### 1.4 Implementation Invariants

1. **One mathematical implementation**: Components remain the source of truth. Function bodies are built from the same implementation or preserve an existing standard-ONNX computation. A blueprint only composes components; it must not implement a second copy of their mathematics. Olive performs verified equivalent target transformations.
2. **Canonical per build specification, not one artifact for every use**: Checkpoint revision, model/config revision, dtype, selected components, semantic I/O and cache mode, and contract versions identify a build. Different EPs reuse that build; different dtypes or state semantics may require different packages. Cache keys must include the complete build specification.
3. **Source quantization is not target quantization**: Mobius remains responsible for importing and interpreting already-quantized checkpoints, including packed storage, scales, zero points, and shared weights. Olive owns target quantization and kernel-format lowering. Canonicalization must not require eagerly materializing every quantized weight as a floating-point matrix.
4. **Preserve support during migration**: Existing model classes and source formats remain supported. Legacy EP builds remain available until the corresponding Olive path has passed correctness and performance checks. Switching defaults and removing old paths are separate, gated changes.
5. **Explicit acceptance, not graph-build-only completion**: Validate prefill, multiple decode steps, cache/recurrent-state continuity, quantized and shared weights, package save/load, and actual target performance. L1/L2 checks are useful development gates, not sufficient evidence for end-to-end model support.

---

## 2. Part I: Mobius 2.0 Improvements ($O(M)$ Canonical Model Layer)

### 2.1 Always-Standard ONNX with Hierarchical `FunctionProto` Boundaries

The canonical build path eliminates the `--execution-provider` axis from graph construction and preserves **standard ONNX opset-24 semantics**. During Phase 1, select this path explicitly with `execution_provider="onnx-standard"`; legacy defaults and EP paths remain unchanged. The default switches only after the Phase 2 migration gate.

Crucially, Mobius 2.0 **must not flatten components into an unstructured sea of primitive ops**. Doing so would force Olive to rediscover high-level blocks via fragile subgraph pattern matching. Instead, Mobius 2.0 introduces **Hierarchical Standard ONNX**:

1. **Dual-Level Representation (`ir.Function` + Standard Body)**:
   - Preserve meaningful mathematical boundaries using local **`ir.Function`** definitions in the `mobius.components` domain. Start with attention, normalization, rotary embedding, and existing stateful function bodies; extend to compound MLP/MoE and other blocks incrementally. Wrapping every small primitive is not a requirement.
   - A self-contained function's body must resolve entirely to standard ONNX operations. Validate its nested subgraphs and transitive local-function dependencies, not just the top-level node domains.
   - Function construction uses `onnx_ir` APIs, not explicit protobuf construction.
2. **Semantic Attributes on Function Calls**:
   - Each function call node carries explicit, machine-readable attributes and metadata (`q_num_heads`, `kv_num_heads`, `head_dim`, `scale`, `softcap`, `sliding_window`, `rope_theta`, `interleaved`, `moe_topk`, `norm_eps`) so downstream Olive passes never have to reverse-engineer hyperparameters from constant tensors inside the subgraph.
   - Define versioned input/output ordering, optional-input presence, attribute defaults, parameter roles, and state updates. Semantic attributes must describe the actual body, not an independently maintained approximation.
   - Keep checkpoint-bound initializers addressable in their owning graph and pass weights explicitly across function boundaries. Preserve shared initializer identity and lazy weight binding.
3. **Portable Representation and Target-Checked Dispatch**:
   - Consumers can preserve supported local functions or expand their bodies with `onnx_ir.passes.common.InlinePass`. Provide `--inline-functions` for flat consumers.
   - A standard body does not guarantee that every EP implements every op, dtype, or dynamic shape it contains. Target legalization and runtime verification remain necessary.
   - Function identifiers make candidate dispatch direct, but kernel replacement still checks masks, cache layout, head dimensions, attribute combinations, dtype, and outputs. Neither constant-time optimization nor zero missed fusions is a correctness guarantee.

### 2.2 Complete Removal of EP Conditional Logic from Mobius

The destination architecture removes EP-specific capability flags from canonical graph construction. Existing flags in `src/mobius/_execution_providers.py` and their consumers migrate one verified target path at a time; they are not all deleted in Phase 1:

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

### 2.3 Incremental Declarative Composition and Weight Mapping

Many architectures reuse existing mathematics and differ along three composition axes. The fraction that can be configured without custom model code must be measured through representative migrations; **80% is not an established coverage result or a release requirement**.
1. **Block Composition & Layer Schedule**: Interleaving different block types across layers (e.g., `[GatedDeltaNet, GatedDeltaNet, GatedDeltaNet, FullAttention] * 16` in Qwen3.5/3.8, or alternating Sliding-Window and Global Attention in Gemma3/4).
2. **Block Micro-Topology Hooks**: Pre-Norm vs. Post-Norm, QK-Norm (per-head vs. full-dim), logit softcapping, residual multipliers, and KV-cache sharing across layers (`shared_kv_source_layer`).
3. **Checkpoint Weight Naming & Layout**: Renaming prefixes, splitting packed `wqkv` or `gate_up_proj` tensors, stacking expert weights, or transposing Conv1d weights.

Extract repeated composition patterns into a declarative layer after the canonical path works. Existing `nn.Module` model classes remain a first-class option, especially for novel topology. Avoid a universal model class controlled by an expanding collection of flags.

#### A. `DeclarativeHybridModel` Blueprint Schema
A prospective `ArchitectureBlueprint` composes existing component variants without duplicating their `forward()` mathematics. The following illustrates the intended shape, not an implemented public API:

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
Gradually replace repeated `preprocess_weights(state_dict)` operations with composable, lazy-streaming-compatible rules. Retain custom transforms for cases the common rules cannot express:

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
Rules must declare and validate source/target names, shape, dtype, packing, grouping axes, and quantized-sidecar ownership before compiling into existing streaming plans (`StreamingTransformedWeightSource`, expert-bank sources, and direct bindings). Declarative rules are not inherently invertible and do not automatically make arbitrary packed weights safe to split or transpose. Preserve fused layouts when required by the source format; shared-table and sidecar behavior require their own checks.

### 2.4 Deferred, Assisted Model Synthesis

Model synthesis is an optional productivity tool, not a dependency of the architecture:
1. Implement genuinely new mathematics as isolated components, and keep model-specific composition in a model class or an applicable blueprint.
2. A later AST + Safetensors-assisted workflow may generate draft components, blueprints, and weight plans from upstream implementations and checkpoint metadata.
3. Generated code passes the same review and numerical gates as handwritten code. L1 graph construction, L2 full-config compatibility, and L3 synthetic parity do not replace multi-token, stateful, multimodal, and target-runtime validation. Tolerances follow the dtype and existing test conventions rather than one universal threshold.

### 2.5 First-Class Stateful & Multi-Component `ModelPackage` Contract

To natively support **Multimodal VLMs, Audio/Speech pipelines, Diffusion DiTs, and Stateful World Models** without ad-hoc external scripts:
- `ModelPackage` and `ComponentManifest` elevate **State Descriptors** (`StateSpec`) alongside `SharedWeightInfo`:
  - `KVCacheStateSpec(kind="paged" | "dynamic" | "static", layers=..., kv_heads=..., head_dim=...)`
  - `RecurrentStateSpec(kind="conv_state" | "ssm_state", shape=..., dtype=...)`
  - `LatentBufferStateSpec(kind="ring_buffer" | "ode_latent", shape=..., dtype=...)`
- Multi-stage pipelines (e.g., `@vision_encoder`, `@audio_encoder`, `@embedding`, `@decoder`, `@denoise_step`, `@world_state_transition`) are exported as named components in a single self-contained `ModelPackage` with verified cross-component tensor contracts and shared weight ownership.
- State contracts describe initialization/reset semantics, input-to-output updates, shape/dtype, and logical ownership/aliasing. Hardware-specific buffer allocation and scheduling remain downstream responsibilities.
- Start with existing KV, convolution, and recurrent state conventions. Paged-cache algorithms, world-model orchestration, and latent-ring-buffer execution are not implied by adding a descriptor and are outside the first release.

---

## 3. Part II: Olive 2.0 Improvements ($O(H)$ Hardware Lowering & Optimization Layer)

With Mobius providing a canonical semantic `ModelPackage`, **Olive 2.0** owns target-specific graph transformations, target quantization, and runtime packaging. Source checkpoint import remains in Mobius.

### 3.1 Accelerator-Agnostic `MobiusBuilder` Pass

In `olive/passes/onnx/mobius_model_builder.py`:
1. In canonical mode, `MobiusBuilder` selects `execution_provider="onnx-standard"` and reports accelerator independence. Legacy target-specific builds must not claim accelerator independence.
2. Cache one canonical package **per complete build specification**, then fan out to target-specific pipelines. Different precision, source revision, selected components, or semantic state contracts do not share the same build result.
3. Generate target-specific runtime metadata after lowering. Merely changing the cache flag is insufficient while the builder still emits EP-dependent graphs or runtime configuration.

### 3.2 Function-Boundary-Driven Graph Surgery (`FunctionTargetSurgeon`)

Currently, `olive/passes/onnx/graph_surgery/` relies on `RewriteRuleSurgeon` to pattern-match primitive ops after functions have been inlined. In Olive 2.0, `GraphSurgeries` operates directly on **Mobius 2.0 `ir.Function` boundaries** before inlining:

```
Canonical Mobius 2.0 Graph (with mobius.components::* Function nodes)
         │
         ├─► [Olive Target EP Capability Lookup]
         │
         ├── If EP has native fused kernel (e.g., CUDA / DML / WebGPU GQA):
         │     Replace `mobius.components::Attention` + `RotaryEmbedding` call
         │     with `com.microsoft::GroupQueryAttention` only after checking
         │     the function contract and kernel semantic eligibility.
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
- **Less structural guesswork**: Explicit component identity and attributes reduce dependence on fragile primitive patterns. Unsupported masks, cache modes, head layouts, or attribute combinations must still reject a fusion; preserve or inline the standard body only when the target can execute it.
- **Clean separation of concerns**: A new backend kernel can be added in Olive without modifying model mathematics when the existing semantic contract fully describes it. New semantics require a compatible, versioned contract change rather than an unchecked replacement.

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

In Olive 2.0, target weight quantization operates on the canonical graph before compatible EP fusion. Already-quantized checkpoint inputs retain their source quantization semantics; this is not a requirement to quantize them again:
1. **Unified Weight Quantization (`OnnxMatMulNBits` / `NVModelOpt` / `INC` / `AIMET`)**:
   - Quantizes `Linear`, `Embedding`, and `MoELayer` weights on the canonical graph.
   - Respects `ComponentManifest.shared_weights` (`SharedWeightInfo`): when `embedding` and `decoder` are quantized in the same Olive workflow, tied tables share quantized initializers; when exported standalone, Olive materializes independent quantized tensors.
2. **EP-Specific Quantization Format Lowering**:
   - **CUDA / DML / WebGPU / CPU**: Retains `com.microsoft::MatMulNBits` and applies `PackQKVForGroupQueryAttention` (folding packed `q_weight`, `k_weight`, `v_weight`, scales, and zero-points via `FoldConcatInitializersPass`) and `FuseBlockQuantizedMoE` (`QMoE`).
   - **QNN HTP / TRT-RTX / OpenVINO**: Converts `MatMulNBits` to standard `DequantizeLinear + MatMul` (QDQ) via `MatMulNBitsToQDQ` (`olive/passes/onnx/mnb_to_qdq.py`).
3. **Function-Aware Weight Binding**:
   - Quantization passes must follow explicit function parameters to owning graph initializers, inspect function bodies where needed, and preserve shared bindings.
   - A pass that cannot process a preserved function must report that limitation or use a defined inlining stage. It must not silently skip quantization or break later semantic fusion.

### 3.5 Unified `EpLoweringPipeline` Profiles in Olive

Add declarative **EP Target Profiles** that expand into a validated lowering sequence for each `(AcceleratorSpec, Precision)`, rather than requiring users to assemble it manually. Keep this compiler-profile responsibility distinct from the existing `GenAIModelRuntimeProfiles` pass, which manages runtime overlays and decoder variants. The following table specifies intended pipelines, not an already-implemented profile API:

| Target Profile | Olive 2.0 Automated Lowering Sequence |
| :--- | :--- |
| **`cuda`** | `CanonicalMobius` $\rightarrow$ `Quantize(MatMulNBits)` $\rightarrow$ `FunctionToGQA(pack_qkv=True, fused_rope=True)` $\rightarrow$ `FuseSkipNorm` $\rightarrow$ `FuseQMoE` $\rightarrow$ `FoldInitializers` $\rightarrow$ `GenAIConfig(cuda_graph=True, share_buffer=True)` |
| **`dml`** | `CanonicalMobius` $\rightarrow$ `Quantize(MatMulNBits)` $\rightarrow$ `FunctionToGQA(pack_qkv=False, fused_rope=False)` $\rightarrow$ `FuseSkipNorm` $\rightarrow$ `FoldInitializers` $\rightarrow$ `GenAIConfig(share_buffer=True)` |
| **`webgpu`** | `CanonicalMobius` $\rightarrow$ `Quantize(MatMulNBits)` $\rightarrow$ `FunctionToGQA(pack_qkv=True)` $\rightarrow$ `SplitLargeInitializers(256MB)` $\rightarrow$ `GraphCaptureLegalization` $\rightarrow$ `GenAIConfig(graph_capture=True, cap_kv=True)` |
| **`openvino-npu`** | `CanonicalMobius` $\rightarrow$ `InlineFunctions(keep={Attention, RMSNorm, RotaryEmbedding})` $\rightarrow$ `Keep4DPerLayerInputs` $\rightarrow$ `OpenVINOQuantize/Encapsulate` $\rightarrow$ `GenAIConfig(device_type=NPU)` |
| **`qnn-htp`** | `CanonicalMobius` $\rightarrow$ `InlineAllFunctions` $\rightarrow$ `DecomposeAttention & RoPE` $\rightarrow$ `Rank4RMSNormToRank3` $\rightarrow$ `TensorScatterToScatterND` $\rightarrow$ `MatMulNBitsToQDQ` $\rightarrow$ `QnnContextBinary` |

---

## 4. Part III: Long-Term Horizon — Next-Gen Cross-Platform IR for LLM, Diffusion & World Models

This section is a research direction, not a Phase 1-4 acceptance requirement or a commitment to implement a new compiler/runtime. As models combine autoregressive generation, diffusion, memory, and recurrence, the semantic boundaries may provide a bridge toward **Multi-Level Component & Tile IR**:

1. **Level 1 — Semantic Component & Stateful Module IR (Mobius 2.0 Output)**:
   - A single self-contained `ModelPackage` containing multiple callable entry points (`@encode_vision`, `@update_world_memory`, `@denoise_step`, `@decode_token`), shared block-quantized weights, and explicit `StateSpec` buffers.
2. **Level 2 — Known-Hotspot Kernel Dispatch + Tile-Level Auto-Fusion (Olive 2.0 + Backend Compilers)**:
   - **Known hotspots** (`Attention`, `GEMM`, `MoE`, `Conv`) map directly from `mobius.components::*` `FunctionProto` boundaries to hand-tuned vendor microkernels (FlashAttention, cuBLAS, QNN HTP, Intel XMX).
   - **Novel mathematical blocks** (new linear attention recurrences, custom world-model state transitions, 3D spatio-temporal conditioning) inline their Standard ONNX function bodies and lower to **Loop/Tile-level compilers** (e.g., MLIR `linalg` / StableHLO / IREE / WebGPU shader codegen / NPU DMA-tile schedulers), automatically generating fused hardware kernels without waiting for custom C++ operator registrations in ONNX Runtime.

---

## 5. Phased Implementation Roadmap

### Phase 1: Establish the Hierarchical Standard ONNX Contract
- [x] Preserve key mathematical boundaries and existing standard stateful function bodies on the explicit `onnx-standard` build path, without duplicating component mathematics. The initial slice lifts existing `Attention`, `RMSNormalization`, `LayerNormalization`, and `RotaryEmbedding` operations and reuses stateful `LinearAttention` and `CausalConvWithState` bodies; compound component extraction remains incremental.
- [x] Validate the complete local-function closure and keep canonical core opset 24; do not apply EP kernel-version workarounds to this path.
- [x] Add opt-in `--inline-functions` and `ModelPackage.inline_functions()` for flat consumers, retaining graph I/O, initializer names, shared weights, and numerical behavior.
- [ ] Keep existing model classes, defaults, and EP paths working. Compound-block coverage can expand incrementally; do not claim that every component is already represented by a function.

### Phase 2: Verify Canonical-to-Target Pipelines, Then Migrate Defaults
- [ ] Add function-contract-aware dispatch and function-aware quantization in Olive, reusing existing graph surgeries instead of reimplementing them.
- [ ] Make canonical `MobiusBuilder` caching accelerator-independent per complete build specification and move target runtime packaging downstream.
- [ ] Prove the same canonical package works through at least two target pipelines, using a standard dense LLM and a complex hybrid or multimodal model.
- [ ] Verify prefill and multi-step decode, recurrent/cache updates, package save/load, source quantization and shared weights, and actual target performance against existing behavior. Record target/runtime versions and numerical tolerances; disclose unavailable hardware instead of claiming validation.
- [ ] Migrate EP-specific graph construction and legalization target by target. Only switch the default and remove old paths after their replacement has passed these gates.

### Phase 3: Extract Repeated Composition, Weight, and State Contracts
- [ ] Measure repeated model composition and introduce narrowly scoped blueprints using existing component variants. Preserve custom Python composition and the existing registry's model coverage.
- [ ] Compile validated weight rules into existing streaming plans, checking packed layouts, dtype/shape, source quantization sidecars, and shared ownership. Migrate representative models before making coverage claims.
- [ ] Extend `ComponentManifest` with versioned state descriptors for existing KV, convolution, and recurrent I/O. Preserve state contracts through serialization and target lowering without adding a Mobius runtime scheduler.

### Phase 4: Optional Assisted Synthesis
- [ ] Prototype AST + Safetensors-assisted generation only after component and weight contracts have stabilized.
- [ ] Treat generated artifacts as reviewable drafts subject to the same numerical, generation, packaging, and target-runtime gates as manual implementations.
- [ ] Evaluate adaptation effort and maintainability on real architectures before expanding automation. General-purpose AST translation and world-model execution remain separate research work.
