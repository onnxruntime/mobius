# ONNX GenAI workflow metadata

Mobius emits the concise public workflow source form. The only control primitives are
`sequence`/`steps`, `invoke`, `loop`, `branch`, and `emit`.

## Structural execution frequency

- Root `steps` run once per invocation.
- `loop.setup` runs once whenever the loop is entered.
- `loop.steps` run once per iteration.
- Root suffix steps run once after the loop.
- A session-state initializer runs once when that session cell is created.
- Artifact loading and session restoration are runtime lifecycle operations.

There are no phases, strategies, `run_once`, or execution-frequency flags.

## Surface and lowered forms

Serialized metadata uses logical names and concise carries:

```yaml
state:
  cache:
    contract: { dtype: float16, rank: 4, shape: [batch, heads, cache_sequence, width] }
    scope: invocation
    initializer: cache.initial
    recurrence: { kind: bounded, axis: 2, max: max_context }
steps:
  - kind: loop
    setup: []
    steps:
      - kind: invoke
        component: decoder
        inputs: { past_key: cache }
        outputs: { present_key: cache.next }
    condition: continue
    max_iterations: max_output_tokens
    carried: [{ cell: cache, next: cache.next }]
```

The loader compiles this source form to lexical SSA, branch phi values, loop read/write
versions, and linear effect tokens. Compiler-generated names are not serialized.

Pure ONNX components have no effect declaration. RNG, caches, counters, and policy state
are ordinary explicit tensors. Effects are reserved for external mutation such as streams,
session mutation, telemetry, and stateful adapter ABIs. The loader threads and joins those
effects through structured control flow.

ONNX artifacts are authoritative for component input/output names, dtypes, ranks, and
shapes. Mobius declares ports only for adapters or constraints not represented by ONNX.
Resource placement normally determines transfers; workflows do not author transfer steps.

Versioned component contracts add semantic role mappings without changing execution:

```yaml
contract:
  id: onnx-genai.token-sampler
  version: "1"
  bindings: { logits: logits, token: token_ids }
  parameters: { mode: greedy }
```

### Decoder KV boundary

Normal decode carries each decoder `present` tensor directly to the corresponding
next-iteration `past` input. Mobius does not emit a generic `kv_update.onnx`. Physical
shared/paged allocation, slots, append, compaction, and in-place mutation belong to the
runtime's model-agnostic KV service. An ONNX state-update component is used only when
semantic tensor math is required, such as accepted-prefix truncation, dense gather,
rollback, or format conversion.

### Request-parameterized sampling

The stochastic sampler ABI accepts `temperature`, `top_k`, `top_p`, `seed`, RNG offset,
and a grammar-allow mask as typed inputs. They are request/workflow values rather than
artifact constants, so ordinary option changes do not rebuild the sampler. The generated
component is marked `application_overridable`; an application may replace it with another
implementation of the same versioned port ABI for fundamentally custom sampling.

## Compact examples

### Decoder

```yaml
steps:
  - kind: invoke
    component: initialize_decoder
    inputs: { tokens: prompt_tokens }
    outputs: { cache: cache.initial, mask: mask.initial }
  - kind: loop
    setup: []
    steps:
      - kind: invoke
        component: decoder
        inputs: { tokens: token, cache: cache, attention_mask: mask }
        outputs: { logits: logits, present: cache.next }
      - kind: invoke
        component: sampler
        inputs: { logits: logits }
        outputs: { token: token.next }
      - kind: invoke
        component: update_mask
        inputs: { current: mask }
        outputs: { next: mask.next }
      - kind: emit
        value: token.next
        output: tokens
        mode: append
    condition: continue
    max_iterations: max_output_tokens
    carried:
      - { cell: cache, next: cache.next }
      - { cell: mask, next: mask.next }
      - { cell: token, next: token.next }
```

### Vision-language

```yaml
steps:
  - kind: invoke
    component: image_preprocess
    inputs: { encoded: image }
    outputs: { pixel_values: pixels, grid: grid }
  - kind: invoke
    component: vision_encoder
    inputs: { pixel_values: pixels, grid: grid }
    outputs: { features: image_features }
  - kind: invoke
    component: embedding
    inputs: { tokens: prompt_tokens, image_features: image_features }
    outputs: { embeddings: prompt_embeddings }
  - kind: loop
    setup: []
    steps:
      - kind: invoke
        component: decoder
        inputs: { embeddings: prompt_embeddings, cache: cache }
        outputs: { logits: logits, present: cache.next }
    condition: continue
    max_iterations: max_output_tokens
    carried: [{ cell: cache, next: cache.next }]
```

Preprocessing, vision, and initial embedding are root-prefix steps and therefore run once.
Only the decoder body runs per generated token.

### Diffusion

```yaml
steps:
  - kind: invoke
    component: initialize_latent
    inputs: { noise: noise }
    outputs: { latent: latent.initial }
  - kind: loop
    setup: []
    steps:
      - kind: invoke
        component: denoiser
        inputs: { sample: latent, step: diffusion_step }
        outputs: { estimate: estimate }
      - kind: invoke
        component: solver
        inputs: { state: latent, estimate: estimate, step: diffusion_step }
        outputs: { next_state: latent.next }
    condition: continue
    max_iterations: num_steps
    iteration:
      value: diffusion_step
      contract: { dtype: int64, rank: 0, shape: [] }
    carried: [{ cell: latent, initial: latent.initial, next: latent.next }]
  - kind: invoke
    component: vae_decoder
    inputs: { latent: latent }
    outputs: { image: image }
  - kind: emit
    value: image
    output: image
    mode: replace
```

Latent initialization and VAE decoding run once; denoiser and solver run per iteration.
