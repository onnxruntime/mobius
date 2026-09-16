# Per-Component Quantized Checkpoint Loading

**Status:** Proposed
**Date:** 2026-08-26

### Implemented precedents

The complete component manifest, codec registry, and `ModelWeightAdapter`
design below remain proposed. Native GPT-OSS MXFP4 support has since landed
several narrower building blocks that should be reused by the migration:

- `QuantizedWeightFormat` distinguishes integer-affine storage from MXFP4
  instead of inferring semantics from bit width and group size.
- `StreamingWeightPlan` supports direct, expert-bank, and transformed lazy
  sources with fail-closed source/target metadata validation.
- Architecture-specific planners can map checkpoint headers to graph
  initializers without materializing a complete state dict.
- The native GPT-OSS loader registers its lazy source artifacts and marks its
  report for transactional `ModelPackage.save()` publication. These lifecycle
  protections are opt-in requirements, not yet automatic for every streaming
  loader.

These are precedents, not an implementation of the proposed
`QuantizationCodec` or `ModelWeightAdapter` registries. The current
Transformers builder still dispatches exceptional loaders such as GPT-OSS and
Qwen4 explicitly.

---

## Executive summary

Mobius must load pre-quantized multi-component checkpoints in which each
exported component can use a different weight layout. For example:

```text
decoder        Olive INT4, group size 32
vision_encoder Olive INT8, group size 64
audio_encoder  floating point
embedding      floating point
```

Mobius does **not** quantize floating-point weights. Quantization remains the
responsibility of Olive, GPTQ, AWQ, or another upstream producer. Mobius reads
the producer's metadata and packed tensors, builds matching ONNX parameter
scaffolding, converts supported packed layouts into the canonical ORT layout,
and binds the existing quantized values.

The current architecture was designed around one model-wide
`quantization_config`. Multi-component tasks later split the ONNX graphs, but
configuration and weight preprocessing remained model-wide. As a result,
model-specific `preprocess_weights()` methods now mix two different concerns:

1. architecture-specific semantic transforms such as rename, transpose,
   split, fuse, and tied-weight synthesis; and
2. generic checkpoint mechanics such as component routing, quantization format
   dispatch, packed-sidecar grouping, validation, and binding.

This proposal separates those concerns through four reusable abstractions:

1. `ComponentManifest`
2. `WeightRecord` and `WeightBundle`
3. `QuantizationCodec`
4. `ModelWeightAdapter`

## Goals

- Represent one quantization configuration per `ModelPackage` component.
- Allow an authoritative component configuration to keep selected modules
  floating point through exact or regex exclusions.
- Determine component configuration before component module construction.
- Route checkpoint weights with one canonical component manifest.
- Preserve packed `qweight`, scale, and zero-point tensors as one logical
  record throughout name mapping.
- Keep float-to-integer quantization outside Mobius.
- Keep architecture-specific transforms available without making them own the
  generic loading pipeline.
- Fail before publication when metadata, graph parameter shapes, and packed
  checkpoint values disagree.
- Preserve current single-component and global-configuration behavior during
  migration.

## Non-goals

- Calibration, range collection, or quantization parameter estimation.
- Floating-point to INT2/INT4/INT8 conversion.
- A universal representation for native non-affine GGUF codebooks. Those
  formats retain their existing typed import path.
- Eliminating every model-specific weight transform. Some architectures
  inherently require semantic transforms that cannot be inferred safely.

## Current architecture and failure mode

The current Transformers path is effectively:

```text
HF config
  -> one ArchitectureConfig.quantization
  -> instantiate one top-level module
  -> task splits module into component graphs
  -> download one flat state_dict
  -> model.preprocess_weights(state_dict)
  -> ModelPackage.apply_weights(state_dict)
```

This is coherent when all graph components share one layout. It becomes
ambiguous when the decoder is INT4 and the vision encoder is INT8:

- component identity exists in the task;
- HuggingFace source ownership exists on the model class;
- optimization roles exist in `model_roles`;
- but no single object combines those facts;
- and `preprocess_weights()` receives the entire checkpoint plus one global
  quantization configuration.

The result is either applying decoder packing parameters to every component or
adding model-specific conditionals that reimplement component routing.

## Proposed architecture

```text
Checkpoint metadata
        |
        v
ComponentManifest + ComponentQuantizationManifest
        |
        v
Component-specific module construction
        |
        v
CheckpointReader -> WeightRecord stream
        |
        v
ComponentRouter
        |
        v
QuantizationCodec
        |
        v
ModelWeightAdapter
        |
        v
WeightBinder + publication validation
```

## 1. ComponentManifest

`ComponentManifest` is the authoritative description of package components.
It initially resolves existing declarations without requiring every model to
migrate in one change.

```python
@dataclasses.dataclass(frozen=True)
class ComponentDescriptor:
    name: str
    # Dotted Python attribute path from the root nn.Module passed to task.build().
    # This is neither a ModelPackage key nor a checkpoint key prefix.
    module_attribute_path: str
    role: str
    source_paths: tuple[str, ...]

    def source_module_names(self, local_module_path: str) -> tuple[str, ...]: ...


@dataclasses.dataclass(frozen=True)
class ComponentManifest:
    components: tuple[ComponentDescriptor, ...]

    def by_name(self, name: str) -> ComponentDescriptor: ...
```

The resolver combines:

- `ModelTask.components`
- `ModelTask.model_roles`
- model `HF_COMPONENT_SOURCES`

Every consumer uses the same manifest:

- component inspection;
- module construction;
- optimization role selection;
- weight routing;
- package persistence;
- component quantization selection.

An unresolved component is a typed error. Consumers must not independently
guess aliases or state-dict prefixes.

### Mapping local module paths to HuggingFace names

Quantizer metadata names runtime HuggingFace modules, while graph construction
walks paths inside a Mobius component. For example:

```text
HF:     model.language_model.layers.0.per_layer_input_gate
Mobius: model.layers.0.per_layer_input_gate
```

`ComponentDescriptor.source_module_names()` derives candidate HuggingFace names
by anchoring the local path at each declared source root. The shared matcher
then applies the producer's exact/regex semantics to those candidates.
Architectures whose names cannot be aligned structurally declare explicit path
aliases on their component descriptor; individual loaders must not invent
their own mapping.

## 2. Component quantization manifest

The parsed architecture config keeps legacy compatibility while exposing an
authoritative component mapping:

```python
@dataclasses.dataclass(frozen=True)
class ComponentQuantizationManifest:
    components: Mapping[str, QuantizationConfig | None]

    def for_component(self, name: str) -> QuantizationConfig | None: ...
```

Accepted metadata forms can include:

- explicit `component_quantization`;
- `quantization_config.components`;
- nested `vision_config.quantization_config`;
- nested `audio_config.quantization_config`;
- a legacy model-wide configuration converted at the parser boundary.

Missing entries mean floating-point storage. A component entry must describe
one affine layout for its quantized projections, but may include an
authoritative selection policy:

```python
@dataclasses.dataclass(frozen=True)
class ComponentQuantizationPlan:
    layout: QuantizationConfig
    modules_to_not_convert: tuple[str, ...]
    overrides: Mapping[str, QuantizationOverride]
```

`modules_to_not_convert` is evaluated for each candidate HuggingFace module
name while the component is constructed. A matching module remains a normal
`Linear`; other eligible projections use the component's packed layout.
Different packed layouts inside one component require per-module overrides and
must be represented explicitly rather than inferred from a root plan.

## 3. Component-specific construction

The component configuration is selected before creating its parameters:

```python
decoder = DecoderModel(config.for_component("decoder"))
vision = VisionModel(config.for_component("vision_encoder"))
```

Each projection additionally asks the component plan whether its source module
is quantized:

```python
linear_class = component_plan.linear_class_for(
    local_module_path="model.layers.0.per_layer_input_gate",
    descriptor=decoder_descriptor,
)
```

This keeps `per_layer_input_gate`, `per_layer_projection`, and other explicit
float exceptions as ordinary `Linear` modules without weakening the rest of a
quantized decoder.

This replaces post-construction scanning that tries to swap `Linear` instances
after a model has already encoded architecture-specific choices.

During migration, top-level model classes can expose component factories:

```python
class ModelComponents:
    def build_component(self, descriptor, config):
        ...
```

The task calls those factories using the resolved manifest. Existing
top-level modules remain supported behind a compatibility adapter until their
components migrate.

## 4. WeightRecord

Checkpoint sidecars are grouped into one typed logical weight immediately
after reading metadata:

```python
@dataclasses.dataclass(frozen=True)
class FloatWeight:
    value: torch.Tensor


@dataclasses.dataclass(frozen=True)
class AffinePackedWeight:
    qweight: torch.Tensor
    scales: torch.Tensor
    zero_points: torch.Tensor | None
    bits: int
    group_size: int
    symmetric: bool
    method: str


@dataclasses.dataclass(frozen=True)
class WeightRecord:
    source_name: str
    component: str
    storage: FloatWeight | AffinePackedWeight
```

Grouping prevents several raw sidecar keys from being renamed, split, or
overwritten independently. A missing scale or zero point is detected before
any architecture transform executes.

`WeightBundle` owns records for one component:

```python
@dataclasses.dataclass
class WeightBundle:
    component: ComponentDescriptor
    records: dict[str, WeightRecord]
```

## 5. QuantizationCodec

Format-specific layout handling lives in a registry:

```python
class QuantizationCodec(Protocol):
    method: str

    def decode_metadata(self, value: object) -> QuantizationConfig: ...

    def normalize(
        self,
        record: WeightRecord,
        target: QuantizationConfig,
    ) -> WeightRecord: ...
```

Initial codecs:

- Olive affine packed weights
- GPTQ
- AWQ

The codec:

- validates the packed payload and quantization parameters;
- checks logical and physical shapes;
- repacks supported source layouts into canonical ORT layout;
- never computes quantization parameters from float values.

Model names and model-specific parameter paths are prohibited inside codecs.

## 6. ModelWeightAdapter

The model adapter receives one already-routed component bundle:

```python
class ModelWeightAdapter(Protocol):
    def adapt(
        self,
        component: ComponentDescriptor,
        weights: WeightBundle,
        config: BaseModelConfig,
    ) -> WeightBundle: ...
```

Appropriate model-specific operations include:

- HuggingFace-to-ONNX name alignment;
- transpose;
- fused QKV split or merge;
- expert-bank stacking;
- model-specific tied-weight synthesis;
- special projector or wrapper removal.

The adapter must not:

- choose a component quantization configuration;
- read unrelated component weights;
- dispatch Olive/GPTQ/AWQ formats;
- bind graph initializers;
- silently drop unrecognized packed tensors.

Most adapters should become declarative rename/transform rules. Code hooks
remain for QMoE and other transforms whose semantics are genuinely
architecture-specific.

## 7. Generic binder and publication gate

The binder consumes normalized, adapted records and graph initializers.

Required invariants:

- every required graph initializer is bound;
- every packed checkpoint record is consumed;
- no `qweight` sidecar remains after normalization;
- bits and group size match the component graph;
- scale and zero-point shapes match the canonical parameter schema;
- a component declared quantized cannot bind float weights to packed
  parameters;
- a component declared float cannot contain packed checkpoint weights;
- unknown checkpoint records are rejected unless explicitly allowlisted.

These checks happen before save or runtime metadata generation.

## Ordering of normalization and model transforms

Some transforms operate on logical tensors while others depend on a source
packing layout. The pipeline therefore distinguishes:

1. **sidecar grouping** — always first;
2. **semantic routing/name mapping** — typed records remain packed;
3. **architecture transform planning** — split/fuse intent is declared;
4. **codec normalization** — source layout becomes canonical target layout;
5. **planned transform execution**;
6. **binding**.

A transform unsupported for packed storage fails explicitly. It must not
silently dequantize and requantize.

## Compatibility plan

The migration keeps current APIs operational:

- a global `quantization_config` maps to the legacy component;
- existing `preprocess_weights()` methods run through a legacy adapter;
- existing `ComponentSpec`, `model_roles`, and `HF_COMPONENT_SOURCES` feed the
  manifest resolver;
- loaders without a component manifest retain current single-model behavior.

Compatibility paths emit no behavior changes in the first two implementation
PRs. Removal occurs only after model migrations and parity coverage.

## Migration stack

### PR 1: Component manifest

- Add `ComponentDescriptor`, `ComponentManifest`, and the resolver.
- Convert inspection and validation to consume the manifest.
- Keep graph output and weight behavior unchanged.

### PR 2: Typed weight pipeline

- Add `WeightRecord`, `WeightBundle`, and codec interfaces.
- Group packed sidecars and validate metadata.
- Wrap existing preprocess functions behind compatibility codecs.
- Keep existing loaders as the active path.

### PR 3: Per-component loader

- Parse the component quantization manifest.
- Instantiate components with their effective config.
- Route records through manifest, codec, adapter, and binder.
- Add broad multi-component graph and synthetic packed-weight coverage.

### PR 4: Model adapter migration

- Migrate T5 and conventional VLMs first.
- Migrate Gemma4 clipped vision/audio projections.
- Migrate Qwen3.5/QMoE specialized transforms.
- Remove post-construction module scanning and migrated legacy branches.

## Testing strategy

### Unit tests

- manifest resolution from every declaration source;
- sidecar grouping for Olive/GPTQ/AWQ;
- partial sidecar rejection;
- component routing with module path differing from package name;
- exact and regex float exclusions mapped from HuggingFace names to local
  component module paths;
- codec shape and packing validation;
- binder completeness and leftover-record rejection.

### Broad graph tests

For every registered multi-component configuration:

- build with distinct component layouts;
- assert eligible component projections use the expected bits/group size;
- assert omitted components remain float;
- assert single-component and text-only aliases preserve legacy behavior.

### Weight-level tests

- T5 encoder INT8 plus decoder INT4;
- VLM decoder INT4 plus vision INT8 plus float embedding;
- decoder INT4 with float `per_layer_input_gate` and
  `per_layer_projection`;
- speech decoder INT4 plus audio INT8;
- component path differing from package name;
- tied embedding/head behavior;
- QMoE component specialization.

### Runtime evidence

At least one real-weight package must:

- load every ONNX component directly with ONNX Runtime;
- consume all packed checkpoint records;
- run deterministic logits or component parity;
- run multi-token generation through the declared runtime when applicable.

## Acceptance criteria

The migration is complete when:

- one manifest is the source of truth for every component consumer;
- no generic loader uses model-name conditionals;
- codecs contain no model-specific paths;
- migrated adapters receive only one component bundle;
- every packed record is either bound or rejected;
- Mobius never converts float weights into quantized weights;
- per-component real-weight parity is demonstrated;
- legacy global-config tests remain unchanged.
