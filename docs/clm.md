# CLM projection-head-only ranking

`Contrastive-LM/CLM-v0.1-8B` is **not a text-generation model**. Mobius exports
its independent state/action projection heads and default-temperature ranking
computation as a plain ONNX `ModelPackage["model"]`. It does not export or
download the frozen Qwen3-8B encoder, tokenizer, or text/question formatting.

```python
from mobius import build

package = build(
    "Contrastive-LM/CLM-v0.1-8B",
    revision="e939398d4556fcd9400c76fa8c5a513202f42b0a",
    task="contrastive-ranking-heads",
    dtype="f32",
    execution_provider="cpu",
)
package.save("clm-heads")
```

The official model ID defaults to that immutable revision. Local directories
containing the same `config.json` contract and a `CLM_v0.1-8B.pt` checkpoint
are also supported. Only the projection checkpoint is loaded, always with
`torch.load(..., map_location="cpu", weights_only=True)`; its topology, exact
tensor names/shapes, float32 dtype and finiteness are validated before binding.
Topology is limited to depth 2 through 64 and at most 256 MiB of float32
parameters; all tensor contracts are checked before constructing the module tree.
The official pinned download is SHA256-verified. Even `load_weights=False`
reads the checkpoint to determine topology and temperature, but does not bind
the head parameters. Only float32/unquantized export is currently supported.
All CLM sources reject `trust_remote_code=True` before custom `AutoConfig`
execution, including non-official IDs and local directories. Trusted non-CLM
configs are preflighted as plain JSON and pinned to that resolved commit before
custom configuration code can run. The size limit is checked before hashing
as well as before checkpoint deserialization. A caller-supplied
`ContrastiveRankingHeadsTask` instance is preserved, including subclass build
overrides and instance configuration.

Test coverage uses a dedicated contrastive-ranking tiny-config inventory and
the pinned official model ID/revision for config-only L2 graph validation.
Synthetic CPU parity and opt-in real-head checkpoint parity are in
`src/mobius/models/clm_test.py`. Generic token-prefill/generation goldens cannot
drive this embedding-in/ranking-out contract and are explicitly exempted;
this is not a waiver of the dedicated graph or numerical parity tests.

| ONNX name | Shape | Meaning |
|---|---|---|
| `state_embeddings` | `[states, 4096]` | External normalized state embeddings |
| `action_embeddings` | `[candidates, 4096]` | External normalized candidate embeddings |
| `state_projections` | `[states, 512]` | State-head output, L2-normalized |
| `action_projections` | `[candidates, 512]` | Action-head output, L2-normalized |
| `logits` | `[states, candidates]` | Scaled cosine similarity |
| `probabilities` | `[states, candidates]` | Row-wise softmax over the supplied candidates |

All inputs/outputs are float32. Provide at least one state and one candidate;
every state in a batch is scored against the **same candidate set**. Preserve
candidate order; sort each probability row descending outside ONNX to obtain
ranks (stable sorting preserves upstream tie order).

**Encoder boundary:** upstream `Embedder` expects Qwen3-8B last-token pooling
and normalizes each incoming encoder vector with `x / (norm(x) + 1e-12)`
before either head. Supply those already-normalized vectors, not raw hidden
states or vectors from another encoder. Mobius does not repeat input
normalization. Head output normalization instead uses
`x / max(norm(x), 1e-12)`, matching Torch `functional.normalize`.
An end-to-end deployment must separately pin and validate its encoder weights,
tokenizer, pooling, truncation and text construction; no backbone revision or
encoder parity is claimed here.

Reference semantics are pinned to
[`Contrastive-LM/CLM@bb42c6c5bf914fd449bed2f6ca65be80602cb1f7`](https://github.com/Contrastive-LM/CLM/tree/bb42c6c5bf914fd449bed2f6ca65be80602cb1f7):
`src/clm/heads.py` (`make_head`, `HeadPair`), `embedder.py` (`l2`), and
`engine.py` (`answer`, `rank`). Each head uses biased input/output linear
layers, exact GELU, and optional hidden linear/LayerNorm/activation blocks;
LayerNorm epsilon is `1e-5`. The official checkpoint has width 1536, depth 3,
one LayerNorm block, no residual connection, and projection dimension 512.
Scores are `min(exp(float32(logit_scale)), 100) * state_projection @ action_projection.T`.
Finite negative log scales may underflow to zero in float32; this matches
upstream zero logits and uniform candidate-relative probabilities.
Export temperature is **1**. For upstream's other validated temperatures
`0 < temperature <= 100`, use `softmax(logits / temperature)` outside this
graph; the exported probabilities remain at temperature 1.

Probabilities are relative to the candidate set, not calibrated correctness
scores. This is head-only numeric support, not full-text model parity or
benchmark verification. ORT GenAI, generation goldens and causal-LM optimizer
routes are inapplicable; use ordinary ONNX Runtime CPU inference on `model.onnx`.
