# World-model pipelines

World models are not one fixed graph contract. Depending on the architecture,
they may combine an autoregressive reasoner, latent dynamics, a diffusion
generator, observation codecs, reward/value heads, audio codecs, and action
policies. Mobius therefore represents a complete world model as a validated
pipeline of independently built ONNX graphs.

## Export Cosmos3-Omni

```bash
mobius build --model nvidia/Cosmos3-Nano output/cosmos3 \
    --features world-model
```

Or from Python:

```python
import mobius

package = mobius.build_world_model("nvidia/Cosmos3-Nano")
package.save("output/cosmos3")
```

The Cosmos3 package contains:

| Component | Function |
|---|---|
| `reasoner_decoder` | Qwen3-VL autoregressive Reasoner |
| `reasoner_vision_encoder` | Reasoner vision tower with DeepStack |
| `reasoner_embedding` | Text/image feature fusion |
| `generator` | Unified MoT rectified-flow Generator, including configured Sound and Action heads |
| `video_encoder`, `video_decoder` | Wan video VAE |
| `audio_encoder`, `audio_decoder` | Cosmos3 AVAE, when shipped by the checkpoint |

`pipeline.json` is an executable, runtime-neutral contract. It records:

- schema and model profile versions;
- typed ports plus required/optional/default input semantics;
- registered programs for generated masks, positions, indexes, and timesteps;
- KV/latent/action state initialization, update, lifetime, and release;
- autoregressive sampling/stopping and iterative scheduler controls;
- parameterized patchify/reshape/cast transforms;
- component parameter dtype and ordered EP preferences;
- required runtime capabilities, public outputs, and runtime assets.

Tokenizers, processor configs, generation defaults, and scheduler config are
copied with the package. A runtime executes these declared programs instead of
inferring behavior from port names.

Cosmos3's upstream Generator accepts ragged Python lists. ONNX cannot expose
Python containers, so Mobius uses an explicit packed-token boundary. The
Generator graph consumes packed vision/sound/action tokens plus tensorized
indexes and emits packed predictions. `pipeline.json` declares scheduler,
patchify, reshape, and dtype-conversion transforms that the runtime must provide.
Initial packed latent state is an external pipeline input: a runtime may fill
it with noise for text-to-video, or use the manifest's optional
`conditioning_handoffs` to encode and patchify image/video/audio conditions.
After the first step, recurrent scheduler edges carry that state.
The Reasoner and Generator consume the same tokenized prompt under different
tensor layouts; Generator conditioning does not sample the Reasoner's logits.

The native checkpoint uses mixed precision: the Reasoner and Generator are
BF16, while the Wan VAE and AVAE are FP32. `--dtype` overrides the
Reasoner/Generator compute dtype; codecs retain their checkpoint-native
precision.

Some distilled checkpoints omit the standalone Reasoner vision tower; their
package contains the text Reasoner, Generator, and available codecs only.
Decoder-only AVAE checkpoints similarly omit `audio_encoder`. Cosmos3
Edge and Edge-Policy-DROID use the distinct Nemotron/SigLIP Edge Reasoner,
the shared MoT Generator, Wan VAE, and domain-aware Action head. They contain
no Sound tokenizer. Edge-Policy-DROID is mislabeled `cosmos3_omni` at the
checkpoint root; Mobius detects its nested `cosmos3_edge_text` config and
dispatches it to the Edge builder automatically.

The Reasoner decoder and per-diffusion-step Generator need different execution
contracts (KV-cached autoregression versus joint packed attention), so the
shared understanding-expert weights are materialized in both ONNX graphs.
`pipeline.json` records this under `shared_parameters`; deployments should
budget for the duplicate storage.

The published Cosmos3-Edge repository does not include authoritative
Transformers modeling code for its Reasoner. Its graph remains L1
architecture-validated; the Generator, VAE, and Action paths reuse the
numerically tested shared implementations.

## Compose another world model

`PipelineBuilder` composes any already-built ONNX graphs:

```python
from mobius import PipelineBuilder

builder = PipelineBuilder()
builder.add_model(
    "encoder", encoder, role="encoder",
    preferred_execution_providers=["cuda", "cpu"],
    parameter_dtype="FLOAT",
)
builder.add_model(
    "dynamics", dynamics, role="dynamics",
    preferred_execution_providers=["cuda", "cpu"],
    parameter_dtype="FLOAT",
)
builder.add_model(
    "decoder", decoder, role="decoder",
    preferred_execution_providers=["cuda", "cpu"],
    parameter_dtype="FLOAT",
)

builder.connect("encoder.state", "dynamics.state")
builder.connect(
    "dynamics.next_state",
    "dynamics.state",
    recurrent=True,
)
builder.connect("dynamics.prediction", "decoder.latent")

builder.declare_external(
    "encoder.observation", alias="observation",
    semantic="world.observation",
)
builder.declare_external(
    "dynamics.action", alias="action",
    semantic="world.action",
)
builder.add_stage("encode", "single_pass", ["encoder"])
builder.add_stage("imagine", "state_transition", ["dynamics"])
builder.add_stage("decode", "single_pass", ["decoder"])
builder.add_state(
    "latent_state",
    kind="recurrent",
    input="dynamics.state",
    output="dynamics.next_state",
    lifetime="request",
    release_after="imagine",
)
builder.add_public_output("decoder.observation", alias="predicted_observation")
builder.set_profile("custom-world-model", "1.0")

package = builder.build()
package.save("output/custom-world-model", check_weights=False)
```

Every graph input must have exactly one source: a dataflow edge, external
input, generated input, stateful input, or explicit default. Direct edges are
checked for dtype, rank, and compatible static dimensions. Registered
transforms may change tensor layout and declare the capabilities required from
the runtime. Manifests reject unknown roles, strategies, transforms, unsafe
paths, duplicate producers, and illegal cycles.

The framework is architecture-neutral, but an architecture still needs Mobius
ONNX module definitions and weight preprocessing. It does not trace or
automatically convert arbitrary PyTorch code.

## Latent dynamics compatibility

The original `WorldModelTask` represented only:

```text
observation + action + state
  -> next_state + observation_prediction + reward + continuation
```

That contract remains available for compatibility, but its accurate name is
`LatentDynamicsTask` (`"latent-dynamics"`). `WorldModelTask`,
`WorldModelConfig`, and `MLPWorldModel` are aliases of the corresponding
latent-dynamics types.
