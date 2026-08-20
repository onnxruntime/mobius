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

## Execution islands and graph capture

Serialized component boundaries preserve policy modularity; they do not require separate ORT
sessions, kernel launches, host round trips, or CUDA Graphs. After artifact loading, override
selection, validation, and SSA lowering, the generic planner may link adjacent pure ONNX invokes
on the same device into an execution island (or equivalent linked composite session). Intermediate
SSA values stay device-resident and optimizer-visible. Decoder logits processors, sampler,
state-update math, and termination predicates can therefore execute as one island.

Island formation is derived rather than serialized. It ends at structured host control, device
changes, explicit external effects, or stateful host adapters such as grammar clone/commit.
Application overrides are resolved before planning; a selected pure same-device ONNX replacement
remains eligible under the same rules.

CUDA Graph capture is evaluated per island and concrete shape signature. Eligibility requires:

- static or bounded runtime-specialized shapes;
- stable device-resident input, output, and state addresses;
- no host data-dependent allocation or control;
- execution-provider support for every selected kernel; and
- explicit tensor-threaded counter RNG seed/offset state.

The runtime should warm up bindings, capture an eligible equal-shape execution, and replay later
matches. Shape changes, unsupported kernels, allocator instability, or capture errors fall back
to ordinary island execution without changing workflow semantics. Diagnostics should identify
the island's component list and device, eligibility decision, capture/replay counters, and exact
fallback reason.

Release benchmarking and the native-equivalence acceptance gate are defined in
[`onnx-genai-performance-conformance.md`](onnx-genai-performance-conformance.md).

### Decoder KV boundary

Normal decode carries each decoder `present` tensor directly to the corresponding
next-iteration `past` input. Mobius does not emit a generic `kv_update.onnx`. Physical
shared/paged allocation, slots, append, compaction, and in-place mutation belong to the
runtime's model-agnostic KV service. An ONNX state-update component is used only when
semantic tensor math is required, such as accepted-prefix truncation, dense gather,
rollback, or format conversion.

### Request-parameterized sampling

The stochastic sampler ABI accepts `temperature`, `top_k`, `top_p`, `min_p`, `seed`, RNG
offset, and a grammar-allow mask as typed inputs. They are request/workflow values rather
than artifact constants, so ordinary option changes do not rebuild the sampler. After grammar
masking and temperature scaling, positive min-p is evaluated as
`scaled_logit >= max_scaled_logit + log(min_p)`; a non-positive value disables the filter.
This preserves fixed `[B,V]` shapes. The generated component is marked
`application_overridable`; an application may
replace it with another implementation of the same versioned port ABI for fundamentally
custom sampling.

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
  - kind: loop
    setup:
      # Schedule, timesteps, and scalar scales are ONNX constant components, so
      # the solver reads its sigmas from the graph instead of runtime config.
      - { kind: invoke, component: diffusion_schedule, inputs: {},
          outputs: { schedule: diffusion.schedule } }
      - { kind: invoke, component: diffusion_timesteps, inputs: {},
          outputs: { schedule: diffusion.timesteps } }
      - { kind: invoke, component: latent_row_shape, inputs: {},
          outputs: { shape: diffusion.latent_row_shape } }
      # Counter-based RNG: one private stream per row, seeded by the request.
      - { kind: invoke, component: latent_noise,
          inputs: { seed: request.seed, offset: package.rng_offset,
                    row_shape: diffusion.latent_row_shape },
          outputs: { noise: diffusion.noise, next_offset: diffusion.rng_offset } }
      - { kind: invoke, component: text_encoder,
          inputs: { input_ids: request.input_ids },
          outputs: { encoder_hidden_states: conditioning.hidden_states } }
      - { kind: invoke, component: text_encoder,
          inputs: { input_ids: request.negative_input_ids },
          outputs: { encoder_hidden_states: conditioning.unconditional } }
      - { kind: invoke, component: history_initializer,
          inputs: { reference: diffusion.noise },
          outputs: { zeros: diffusion.initial_history } }
    steps:
      - { kind: invoke, component: schedule_lookup,
          inputs: { schedule: diffusion.timesteps, step: loop.iteration },
          outputs: { timestep: diffusion.timestep } }
      # Classifier-free guidance is two denoiser invocations, not a doubled batch.
      - { kind: invoke, component: denoiser,
          inputs: { sample: latent_state, timestep: diffusion.timestep,
                    encoder_hidden_states: conditioning.unconditional },
          outputs: { noise_pred: denoiser.unconditional } }
      - { kind: invoke, component: denoiser,
          inputs: { sample: latent_state, timestep: diffusion.timestep,
                    encoder_hidden_states: conditioning.hidden_states },
          outputs: { noise_pred: denoiser.conditional } }
      - { kind: invoke, component: guidance_combine,
          inputs: { unconditional: denoiser.unconditional,
                    conditional: denoiser.conditional,
                    scale: request.guidance_scale },
          outputs: { estimate: denoiser.estimate } }
      - { kind: invoke, component: solver_step,
          inputs: { sample: latent_state, step: loop.iteration,
                    schedule: diffusion.schedule, estimate: denoiser.estimate,
                    history: history },
          outputs: { next_state: latent.body, next_history: history.body } }
      - { kind: emit, value: denoiser.estimate, output: noise_estimate, mode: append }
      - { kind: emit, value: latent.body, output: latent_trajectory, mode: append }
    max_iterations: request.max_iterations
    iteration:
      value: loop.iteration
      contract: { dtype: int64, rank: 1, shape: [batch] }
    carried:
      - { cell: latent_state, next: latent.body }
      - { cell: history, next: history.body }
  - { kind: invoke, component: tensor_scale,
      inputs: { tensor: latent_state, scale: diffusion.decoder_scale },
      outputs: { scaled: diffusion.decoder_input } }
  - { kind: invoke, component: vae_decoder,
      inputs: { latent: diffusion.decoder_input },
      outputs: { image: vae.image } }
  - { kind: emit, value: latent_state, output: latent, mode: replace }
  - { kind: emit, value: vae.image, output: image, mode: replace }
  - { kind: emit, value: diffusion.rng_offset, output: rng_offset, mode: replace }
```

Conditioning, latent sampling, and VAE decoding are structural: they sit outside the
loop body and therefore run once. Only the denoiser, guidance, and solver run per step.

Carried state is declared under `state`, where each cell names the value that
initializes it (`latent_state` ← `diffusion.noise`, `history` ←
`diffusion.initial_history`). Nothing about the loop is expressed as a phase or a
strategy; frequency falls out of where a step is nested.

#### Policy components

The producer emits the non-network parts of the pipeline as real ONNX graphs so the
runtime never has to reimplement scheduler math:

| Component | Ports | Purpose |
|---|---|---|
| `diffusion_schedule` / `diffusion_timesteps` | → `schedule` | Sigma schedule and timestep table baked in as initializers, so the solver reads its constants from the graph rather than from runtime config. |
| `decoder_input_scale` | → `value` | Build-time scalar such as `1 / vae_scaling_factor`. |
| `latent_row_shape` | → `shape` | Per-row latent shape consumed by the noise sampler. |
| `schedule_lookup` | `schedule`, `step` → `timestep` | Gathers this step's timestep from the table. |
| `latent_noise` (`onnx-genai.counter-rng@1`) | `seed`, `offset`, `row_shape` → `noise`, `next_offset` | Counter-based Box–Muller normals. Each row draws only from its own seed, so a row's noise does not depend on its batch position, and the advanced counter is returned so the caller can persist it. |
| `history_initializer` | `reference` → `zeros` | Shape-following zero initializer for solver history. |
| `guidance_combine` (`onnx-genai.guidance-combine@1`) | `unconditional`, `conditional`, `scale` → `estimate` | `uncond + scale * (cond - uncond)` with a per-row scale. |
| `tensor_scale` | `tensor`, `scale` → `scaled` | Applies `init_noise_sigma` or the VAE scale. Emitted only when the factor is not 1.0, so no identity multiplies appear in the graph. |
| `solver_step` (`onnx-genai.solver-step@1`) | `sample`, `estimate`, `history`, `step`, `schedule` → `next_state`, `next_history` | The scheduler update. Euler binds `estimate` to its `derivative` port and leaves the history ports unbound; DPM++ 2M carries the previous `x0` estimate as ordinary loop state and masks itself down to first order on the first and last steps. |

Only components with a stable cross-runtime meaning carry a `contract` id; the constant
and reshaping helpers are plain ONNX graphs identified structurally by their ports.

#### Why guidance is two invocations

Classifier-free guidance is conventionally implemented by concatenating the
conditional and unconditional batches and running the denoiser once. The workflow IR
declares batch semantics through `batch_layout`, which has no kind meaning "k rows per
request", so a doubled batch cannot be described truthfully — the runtime would be
unable to attribute a row to a request. Two invocations of the same component plus an
explicit `guidance_combine` keeps every tensor `request_aligned` and leaves batching
decisions to the runtime.

#### Append-mode outputs

`mode: append` concatenates chunks along the last axis, so an accumulating output must
name that axis with a symbol of its own (`noise_estimate_width`, not `width`).
Reusing the per-step symbol would bind the same symbol to both the chunk width and the
accumulated width and fail validation.
