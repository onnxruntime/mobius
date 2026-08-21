# SenseNova-U1.5 (NEO-unify) architecture

`sensenova/SenseNova-U1.5-8B-MoT` (`model_type: neo_chat`, architecture
`NEOChatModel`) is a **native unified any-to-any** model: one backbone
performs multimodal understanding, text-to-image generation, and image
editing. This document records what was verified against the released
checkpoint and the upstream reference implementation, so the Mobius
export can be reviewed against facts rather than names.

## Source-of-truth and a Hub packaging gap

The HF repository declares remote code it does not ship:

```json
"auto_map": {
  "AutoConfig": "configuration_neo_chat.NEOChatConfig",
  "AutoModel": "modeling_neo_chat.NEOChatModel",
  "AutoModelForCausalLM": "modeling_neo_chat.NEOChatModel"
}
```

At revision `1f6ec60423d29939dde4202fd82ae340b144e280` (and at `main`),
`configuration_neo_chat.py` and `modeling_neo_chat.py` both return **HTTP
404**. `trust_remote_code=True` therefore cannot work against the Hub
repo alone. The model card resolves this: the reference implementation
lives in the **`sensenova_u1` Python package** at
<https://github.com/OpenSenseNova/SenseNova-U1> (branch `feat/u1.5`),
under `src/sensenova_u1/models/neo_unify/`. That package is the
authoritative source used here.

Two further packaging notes, both verified and both benign:

* The shard sequence has gaps — `model-000{02,03,04}-of-00016.safetensors`
  are absent. `model.safetensors.index.json` references **no** tensor in
  those shards, so all 1116 tensors resolve and the checkpoint is
  complete. This is numbering slack, not missing weights.
* There is no `preprocessor_config.json` and no `tokenizer.json`. Image
  preprocessing is defined only in code, and the tokenizer must be built
  from `vocab.json` + `merges.txt`.

## Parameter budget

17,532,854,464 parameters (≈50.2 GB on disk) despite the `8B` in the
name — the second transformer branch is not counted by the marketing
name.

| Group | Params | Bytes | Stored dtype |
|---|---|---|---|
| Understanding branch (+ embed / lm_head) | 9.348 B | 18.70 GB | BF16 |
| Generation branch (`_mot_gen`) | 8.121 B | 31.33 GB | mostly F32 |
| `fm_modules` (embedders + pixel head) | 0.046 B | 0.16 GB | mixed |
| Vision tower | 0.018 B | 0.04 GB | BF16 |

## Mixture of Transformers

Every one of the 42 Qwen3 decoder layers carries **two complete,
disjoint weight sets**:

```
self_attn.{q,k,v,o}_proj          self_attn.{q,k,v,o}_proj_mot_gen
self_attn.{q,k}_norm[_hw]         self_attn.{q,k}_norm[_hw]_mot_gen
mlp.{gate,up,down}_proj           mlp_mot_gen.{gate,up,down}_proj
input_layernorm                   input_layernorm_mot_gen
post_attention_layernorm          post_attention_layernorm_mot_gen
model.norm                        model.norm_mot_gen
```

There is no `lm_head_mot_gen`: the generation branch's output goes to
the flow-matching head instead of the vocabulary.

Routing is **per forward call**, not per token. `Qwen3Model.forward`
reduces `image_gen_indicators` once and dispatches the whole call to
either `forward_und` or `forward_gen`; the mixed path raises
`NotImplementedError` at both the attention and decoder-layer level
(upstream issue #207). Production therefore always splits the sequence
at token-type boundaries:

1. **Understanding pass** — text (and reference images) run through the
   understanding weights and write a KV cache.
2. **Generation pass** — repeated once per flow-matching step, the noisy
   image runs through the `_mot_gen` weights with `update_cache=False`,
   attending over the frozen understanding prefix.

Attention is shared in the sense that the generation pass *reads* the
keys/values the understanding pass wrote. That is the entire conditioning
mechanism. Because the split is intrinsic to the upstream design, the
Mobius export mirrors it with two separate decoder graphs rather than
one graph with runtime branching.

## Three rotary axes

`head_dim` is 128 and is partitioned per head:

| Slice | Axis | Rotary base | Position source |
|---|---|---|---|
| `[0:64)` | temporal / text | `rope_theta` = 5e6 | `indexes[0]` |
| `[64:96)` | image height | `rope_theta_hw` = 1e4 | `indexes[1]` |
| `[96:128)` | image width | `rope_theta_hw` = 1e4 | `indexes[2]` |

QK-norm is applied to *halves*, before the height/width split:
`q_norm` normalises the 64 temporal dims and `q_norm_hw` the 64 spatial
dims. This is why the checkpoint's `q_norm` / `k_norm` tensors have
shape `[64]` rather than `[128]`.

All image tokens of a single image share **one** temporal index (the
text length), while their height/width indices tile the token grid.

## Block-causal attention

```python
mask = (idx_j == idx_i) | (arange[None, :] <= arange[:, None])
```

Tokens sharing a temporal index attend to each other **bidirectionally**;
ordinary causality holds across differing temporal indices. For a
pure-text prompt every index is distinct, so this degenerates to a plain
causal mask — which is why text-only parity passes even with a naive
causal implementation, and why image parity does not.

## Vision tower — no transformer blocks

`NEOVisionModel` contains only:

```
Conv2d(3, 1024, kernel=16, stride=16)      # patchify
GELU
interleaved 2-D RoPE (theta 1e4)           # first 512 ch <- x, last 512 <- y
Conv2d(1024, 4096, kernel=2, stride=2)     # 2x2 merge into the LLM width
```

One LLM token therefore covers a **32x32 pixel** tile. The rotation is
*interleaved* — pairs `(0,1), (2,3), …` share an angle — unlike the
half-split convention used by the language backbone.

The module is instantiated twice with independent weights:
`vision_model` embeds reference images into the understanding branch, and
`fm_modules.vision_model_mot_gen` embeds the noisy latent into the
generation branch.

## Flow matching without a VAE

`use_pixel_head: true`, so `fm_modules.fm_head` is a `ConvDecoder`
operating directly in pixel space:

```
(B, 4096, H/32, W/32)
  PixelShuffle(2) -> Conv2d(1024, 1024, k3) -> GELU
  PixelShuffle(2) -> Conv2d(256, 192, k3)
  PixelShuffle(8)
(B, 3, H, W)
```

`PixelShuffle` corresponds exactly to ONNX `DepthToSpace` with
`mode="CRD"`. The total 32x upsample equals `patch_size * merge_size`;
the 192 output channels of `conv2` are `3 * 8 * 8`.

The head predicts **x0** (the clean image), not the velocity. The
sampler converts it:

```python
v = (x0 - z) / max(1 - t, t_eps)
z = z + (t_next - t) * v
```

Because `patchify` is a pure permutation, this arithmetic is identical
whether performed on patches (as upstream does) or on the pixel grid, so
the exported denoiser returns pixels directly.

### Timestep and noise-scale conditioning

Both embedders are `TimestepEmbedder(4096)` with a 256-wide sinusoidal
basis, concatenated as `[cos, sin]` (note: diffusers uses `[sin, cos]`).
The noise scale is resolution-dependent:

```python
noise_scale = min(sqrt(grid_h * grid_w / merge**2 / 64) * noise_scale, 16.0)
```

and is fed to its embedder normalised by `noise_scale_max_value`. The
two embeddings are summed and added to every image-token embedding.

### Timestep schedule

Only the `"standard"` branch of `_apply_time_schedule` is reachable —
upstream unconditionally assigns `self.time_schedule = "standard"` on
entry, making `time_shift_type`, `base_shift`, `max_shift`,
`base_image_seq_len` and `max_image_seq_len` dead for inference:

```python
sigma = 1 - t
sigma = shift * sigma / (1 + (shift - 1) * sigma)
t = 1 - sigma
```

## Exported package

| Component | Role | Contents |
|---|---|---|
| `embedding` | token lookup | `embed_tokens` + reference-image scatter |
| `vision_encoder` | reference images | understanding-branch patchify tower |
| `decoder` | understanding decoder | 42 MoT layers + `norm` + `lm_head` |
| `image_gen_embedding` | generation input | `vision_model_mot_gen` + timestep / noise-scale embedders |
| `image_gen_denoiser` | generation decoder | 42 `_mot_gen` layers + `norm_mot_gen` + pixel head |

`decoder` and `image_gen_denoiser` share the KV-cache layout, which is
what lets the denoiser consume the prefill cache.

### Numeric precision

The two branches are stored in different dtypes upstream and behave
differently under conversion:

* The understanding branch is BF16 in the checkpoint and exports cleanly
  to **float16**.
* The generation branch is stored **F32** and its activations exceed the
  fp16 range. Exported at float16 the flow-matching loop accumulates
  overflow and produces `NaN` part-way through sampling (observed at step
  15 of 20 at 512x512). It must be exported at **float32** (or a
  wide-exponent format such as bf16 where the runtime supports it).

Mixing the two is supported end to end: the sampler casts the fp16 KV
cache to the generation branch's dtype once, after prefill.

### GQA fusion does not apply

Both decoder graphs keep plain opset-24 `Attention` nodes after
CUDA/fp16 optimization, and Mobius emits its "GQA fusion expected"
warning for each. That is expected here, not a defect — three
independent guards each decline for a correct reason:

* `decoder`'s attention bias is block-causal, so it contains an `Or`.
  `local_window_from_attention_bias` treats any `Or` in the bias walk as
  unrecognized and `AttentionToGQA` bails.
* `image_gen_denoiser` has no `attention_mask` graph input at all (the
  generation pass is unmasked), and `AttentionToGQA` needs one to
  synthesize `seqlens_k`.
* RoPE here is three independent axes with different bases, hand-built
  from Slice/Mul/Add rather than `op.RotaryEmbedding`, so
  `RotaryAttentionToGQA` cannot match either.

Closing this would need `local_window_from_attention_bias` to learn the
block-causal `Or` pattern and the denoiser to gain an explicit mask
input; neither is required for correctness.

## Runtime / metadata status

Export and execution are complete — all five graphs build, load, and run
on CPU and CUDA. Mobius now emits canonical hashless metadata for all five
components plus the exact synthesized processor asset: RGB decode, pixel-area
resize aligned to 32 in the 512²–2048² range, `1/255` rescaling, and ImageNet
normalization.

The generic workflow writes conditional and unconditional understanding KV
once, casts the fp16 prefixes to fp32, and lets both CFG generation branches
read the frozen state. The loop describes per-step image embedding, x0
prediction, `v = (x0 - z) / max(1 - t, t_eps)`, and Euler integration. This is
an architecture-neutral shared-state pattern; no model-family dispatch exists
in the runtime.

PR CI uses deterministic tiny graphs to execute text-only, text-to-image, and
reference-image-edit paths, including mixed precision, shared KV, CFG, and a
two-step loop. Production evidence remains the pinned H200 run from PR #533:
revision `1f6ec60423d29939dde4202fd82ae340b144e280`, stage-by-stage L4 parity,
and L5 text/image/edit outputs. It is intentionally not repeated in ordinary CI
because the checkpoint is about 50 GB. Before a release, rerun the same pinned
export on an H200 with `mobius.build(..., revision=<revision>,
load_weights=True)`, fp16 understanding and fp32 generation components, then
repeat the shared-latent stage comparison and the three L5 output checks
recorded in PR #533.


## Dead or misleading upstream details

Recorded so future readers do not port them by mistake:

* `fm_head_dim` (1536), `fm_head_layers` (2) and `fm_head_mlp_ratio`
  describe the `SimpleMLPAdaLN` head, which is only constructed when
  `fm_head_layers > 2`. The released model uses the pixel head, and no
  `SimpleMLPAdaLN` tensors exist in the checkpoint.
* `concat_time_token_num` is only ever read in an `== 0` assertion.
* `_euler_step` is defined but never called; the generate methods inline
  the same expression.
* `t_eps` is `0.05` in `config.json` but the shipped example CLIs never
  override the `0.02` function default, so real inference uses `0.02`.
* `NEOVisionConfig` assigns `llm_hidden_size` and `downsample_ratio` with
  a trailing comma, making them one-element tuples; consuming code
  indexes `[0]`. The Mobius extractor unwraps this explicitly.
* `PatchDecoder_postps`, `PatchDecoder_preps`, `ProgressiveConvDecoder`,
  `PostConvSmoother`, `NerfEmbedder` and `PositionEmbedding` are all
  unused by the released configuration.
