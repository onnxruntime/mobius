# OpenVLA: experimental conversion

Mobius can construct, load and save the complete
`openvla/openvla-7b@47a0ec7fc4ec123775a391911046cf33cf9ed83f` policy as three
portable ONNX components. This is **experimental conversion support**, not a
qualified robot policy. Full-weight CPU execution reproduces one frozen source
sample's preprocessing and seven action tokens/values exactly, but the stricter
all-element prefill-logit gate still fails. Action agreement must not substitute
for that numerical gate or representative robot-task evaluation.

## Build

Review the pinned model implementation and its model/Llama license obligations
before acquiring weights. Download the three safetensors shards and configuration
assets from the exact revision into a local directory; verify their published
identities independently. The public API does not need remote-code execution:

```python
import torch
from mobius import build

torch.set_num_threads(4)
package = build(
    "/absolute/pinned/openvla-checkpoint",
    dtype="f32",
    execution_provider="default",
    trust_remote_code=False,
)
package.save("/absolute/fresh/openvla-onnx", max_workers=4, progress_bar=False)
```

Allow approximately 15 GB for source tensors and 30 GB for the FP32 package,
plus temporary files and substantial CPU RAM. The portable graphs use opset 24;
the demonstrated runtime is ORT 1.31 CPU, not older opset-21 runtimes.
Other precisions, execution-provider rewrites and quantization are unqualified.
Non-default/scaled RoPE and quantization requests fail explicitly.

## Components and checkpoint accounting

| Component | Inputs | Outputs |
|---|---|---|
| `vision_encoder` | FP32 pixels `[B,6,224,224]` | projected features `[B,256,4096]` |
| `embedding` | INT64 text tokens; FP32 image features | embeddings with image patches after BOS |
| `decoder` | embeddings, INT64 mask/positions, 32 layers of past key/value state | logits and fresh per-layer key/value state |

Each component saves `model.onnx` with its own `model.onnx.data`. The default
portable package has 714 vision, 4 embedding and 460 decoder nodes. The caller
owns preprocessing, prompt construction, generation, state and action decoding;
this package does not claim ORT-GenAI architecture integration.

The fused backbone selects the second-to-last unnormalized patch features from
DINOv2-L/14-reg4 and SigLIP-so400m/14. DINO position embeddings apply to patches,
not class/register tokens. Exact GELUs, LayerScale and the biased three-layer
projector follow the original implementation. The decoder materializes omitted
Llama-2 defaults and uses Torch FP32 default RoPE construction; other models keep
their existing RoPE policy.

All 982 source tensor names/shapes were audited: 939 map into parameters and 43
have exact-name, exact-shape omissions. Those omissions are the final vision
blocks, final normalizations and SigLIP pooling head, which cannot affect the
selected intermediate patch features. Unknown or wrong-shaped tensors reject.
The original timm forward still executes the final blocks; omitting their
unused outputs does not alter the selected features.

## Numerical evidence and limitations

An independently executed original-class tiny fixture passes actual ORT CPU
comparisons for both towers, projection, BOS insertion, prefill, decode and all
caches at `atol=1e-6, rtol=1e-5`. This randomized fixture is not checkpoint parity.
Replay it with:

```bash
MOBIUS_OPENVLA_REFERENCE=/absolute/independent/tiny-fixture \
python -m pytest tests/integration/openvla_test.py -q
```

The pinned `testdata/cases/vision-language/openvla-7b.yaml` records the blocked
multimodal-decoder golden scope with an explicit `skip_reason`. It does not
contain a fabricated passing golden or claim that VLA outputs are ordinary text.
The independent image/source fixtures and action-generation application still
need integration into the generic golden runner, after the strict baseline gate
is resolved. Registry/config/CLI coverage passing is not that qualification.

The full checkpoint reference uses original Torch 2.6 / Transformers 4.40.1 /
timm 0.9.10, BF16 source tensors upcast to FP32 and eager CPU execution.
The export environment uses Torch 2.14.1. Original and exported RoPE tables were
independently checked bit-for-bit across all 2048 positions.

At frozen `atol=rtol=1e-4`, all 128 prefill/decode cache outputs pass and all seven
greedy action tokens agree. Full prefill logits still have 59 failing elements
of 8,849,664; feeding original embeddings isolates 31 failures. Decode logits
pass the combined absolute-plus-relative criterion. Higher-precision MLP
accumulation was evaluated as a scratch control and **did not fix the gate**;
no such workaround is included.

This evidence covers one fixed image/instruction and semantic boundaries, not
every primitive graph output, representative action quality, latency targets,
GPU providers or the Windows RTX 4090 pilot. Keep numerical qualification and
release gates closed. Do not apply Olive/quantization rewrites until a baseline
passes, or infer a runtime-kernel defect from model-level drift alone.

The clean upstream-based contribution was independently re-exported and the same
full-checkpoint results reproduced. Atlas also compared all four embedding
nodes (`Gather`, two `Slice`, `Concat`) against a separate original-class
embedding-subpath producer: 1,294,336 elements match at zero tolerances, with
every embedding weight byte and instrumentation-control output byte verified.
Sequential and four-worker reports are identical. That is four of 1,178 nodes
on one sample, not a claim that the other 1,174 nodes passed qualification.
