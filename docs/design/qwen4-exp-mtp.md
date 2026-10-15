# Qwen4-Exp MTP forward contract

**Status: design proposal, not approved for implementation.** This document
records pinned-source observations and a conservative candidate contract. It
does not establish an authoritative MTP numerical reference. Unresolved
alignment, cache, and precision semantics below block M1 (#772) until human
maintainer approval.

This is the M0 companion to the [ngram/main API and state proposal](qwen4-exp-text-core.md)
and must use its selected-component and shared-weight decisions rather than
introducing another package format. The current Mobius text-core design omits
MTP; ordinary generation remains unchanged and does not require MTP.

## Pinned evidence and limits

The source pins are:

- [Mobius #755 head `954e8f1`](https://github.com/onnxruntime/mobius/tree/954e8f1053d64d377d8893e5c700d1b8380d584a),
  especially `src/mobius/models/qwen4_exp.py`.
- [GenAI Qwen builder `45f9adf`](https://github.com/microsoft/onnxruntime-genai/blob/45f9adfea9b499aef16672493260a8a50706f0bc/src/python/py/models/builders/qwen.py)
  and [Qwen loader at the same pin](https://github.com/microsoft/onnxruntime-genai/blob/45f9adfea9b499aef16672493260a8a50706f0bc/src/python/py/models/loaders/qwen.py).
- [Transformers Qwen4-Exp `598d8ba`](https://github.com/huggingface/transformers/blob/598d8ba8baaec7fec5a22da0e2844c7bf4ea20e1/src/transformers/models/qwen4_exp/modeling_qwen4_exp.py).

| Source-grounded observation | What it does not establish |
|---|---|
| The GenAI builder enables MTP only when `mtp_num_hidden_layers > 0`, rejects values other than exactly one, and emits a pre-final hyper-stream tap when requested (`qwen.py:2870-2919`). | That the single trained layer can be reused for arbitrary proposal depth. |
| Its main tap is the last decoder layer's hyper-stream after its residual injections and before the final hyper-connection mixer (`qwen.py:1885-1939`). | A Mobius public output or a reference-validated tensor contract. Mobius #755 returns only final mixed states and logits (`qwen4_exp.py:1375-1379,1404-1413`). |
| The MTP graph accepts `input_ids`, `hidden_states`, attention/position inputs and QSA past state; it emits logits, hidden states, and QSA present state (`qwen.py:3043-3073,3155-3178`). | Which `input_ids` row pairs with which main hidden-state row, nor the host's position advancement rule. |
| The loader separately constructs token embedding, LM head, two projections, two pre-projection norms, one `mtp.layers.0` decoder layer, and an MTP final mixer (`loaders/qwen.py:154-225`). The builder configures that layer as QSA and disables PLE (`qwen.py:3155-3165`). | Independent numerical parity, cache behavior, or a supported multi-step chain. |
| HF RMSNorm uses FP32 normalization and `(1 + weight)` scaling before casting back to the input dtype; its hyper-connection mixer normalizes streams, computes a sigmoid mix gate, divides the low-rank input by `hc_count`, and averages across streams (`modeling_qwen4_exp.py:158-178,941-969`). | That the GenAI MTP custom normalization ops round identically, or the exact linear-projection accumulation dtype. |
| HF's ordinary `Qwen4ExpForCausalLM` defines `model` and `lm_head`, ignores unexpected `mtp.*` keys, and its ordinary forward does not execute an MTP module (`modeling_qwen4_exp.py:1528-1542,1551-1563`). | Any MTP-forward oracle. Calling ordinary HF CausalLM is not an MTP reference. |
| Mobius #755 deliberately skips `mtp.*` because no authoritative forward/cache ABI exists (`qwen4_exp.py:1443-1449,1603-1607`). | An OOM or other observed runtime failure. Missing MTP is a functional gap. |

The builder and loader describe graph structure and weight ownership, not
independent numerical proof. The HF model's ordinary CausalLM path is likewise
not an MTP reference. Any unsupported detail below is an explicit blocker, not
an assumed fact.

## Candidate equations

Let `B` be batch, `S` current MTP input tokens, `P` cached MTP tokens, `H` the
per-stream hidden size, `C` the hyper-stream count, `V` vocabulary size, and
`T` the selected model I/O dtype. Let `X[b,t,:]` be the last-main-layer
pre-final hyper-stream tap, and `z[b,t]` the MTP input token ID. These equations
are the most direct standard-op translation of the builder's wiring; they are
**proposals pending an independent reference and approval**.

For the normalization weights `w_e` and `w_h`, define the candidate offset
RMS normalization:

```text
RMSδ(x; w, ε) = cast_T(
    (cast_f32(x) / sqrt(mean_last_axis(cast_f32(x)²) + ε))
    * (1 + cast_f32(w))
)
```

Apply the embedding norm per token. Apply the hidden norm independently to
each of the `C` streams, with group size `H`. The HF norm implementation
supports FP32 normalization and scaling followed by a cast to the input dtype;
the GenAI builder adds `1` to each norm initializer and uses
`SimplifiedLayerNormalization` for the embedding and a custom
`BranchwiseRMSNorm` for the hidden streams (`qwen.py:1214-1229,3183-3197,
3240-3263`; HF `modeling_qwen4_exp.py:158-178`).

The builder wires the initial projection as:

```text
e       = Gather(token_embedding, z)                       # [B,S,H]
e_norm  = RMSδ(e; w_e, ε)                                  # [B,S,H]
h_norm  = branchwise_RMSδ(X; w_h, ε)                       # [B,S,C*H]
h_group = reshape(h_norm, [B,S,C,H])
p_h     = fc_hidden(h_group)                               # [B,S,C,H]
p_e     = fc_embedding(e_norm)                             # [B,S,H]
u       = flatten(p_h + expand(p_e, stream_axis))           # [B,S,C*H]
```

`fc_hidden` is applied to the grouped hidden value, while the projected token
embedding is expanded and added to every stream. No extra `hc_count` scale is
shown on either projection in the builder (`qwen.py:3240-3307`). A standard
translation must match actual checkpoint weight shapes and MatMul orientation;
those details must be verified against the independent eager reference, not
inferred from node names.

The proposed one-layer continuation is:

```text
(S_out, state_out) = Qwen4ExpTextDecoderLayer(
    u, attention_mask, position_ids, state_in
)
y     = MTPFinalMixer(S_out)
logits = MatMul(y, lm_head.weightᵀ)
```

The loader configures `Qwen4ExpTextDecoderLayer` as one
`qwen_sparse_attention` layer with no PLE, and loads a separate MTP
`hyper_connection_mixer` (`loaders/qwen.py:171-204`). The final mixer has
`use_combine=False`; its HF equations are the per-stream offset RMSNorm,
`sigmoid(W_up(silu(W_down(normed)/C)))` gate, and mean of gated normalized
streams (`modeling_qwen4_exp.py:941-969`). It therefore has no block-injection
term. The layer itself is the pinned HF Qwen4-Exp decoder layer; use its
attention, residual-stream, and MoE operations rather than substituting
generic Qwen3.5 behavior.

### Rounding and precision blocker

The HF normalization source establishes FP32 input/statistics/scale and a
cast-back to `T` for that HF module. The GenAI builder's MTP pre-norm helper
adds the offset into a `T` initializer and requests `stash_type=1`; its
branchwise norm is a `com.microsoft` custom op. Its projection builder exposes
`T` graph values but does not specify a portable accumulation/rounding contract.
Accordingly:

- Proposed standard-only graphs use explicit standard ONNX arithmetic, with
  FP32 normalization/statistics and explicit casts to reproduce the selected
  HF behavior.
- Whether the MTP custom op's scaling, intermediate casts, or projections
  exactly match HF remains unresolved until eager intermediate comparisons
  establish the rounding points.
- Do not claim bitwise equality for floating outputs. Choose and record
  dtype-specific tolerances only after the reference is approved. Integer
  token, position, and state-length checks remain exact.

## Proposed graph seam and I/O

Expose the main tap only for an explicitly requested MTP build. Its proposed
layout is `pre_final_hyper_states[B,S,C*H]` in `T`; each final-layer row is
after the layer's attention and MLP residual injections and before the final
hyper-connection mixer. Do not expose the post-mixer `[B,S,H]` as a substitute.
The ordinary main output remains its current mixed hidden state/logits when
MTP is not requested. When requested, add the tap without replacing or
altering those outputs.
The exact opt-in selector and additive-output metadata must be settled jointly
with N0 before M1; neither `split_ngram` nor sidecar persistence requests this
tap implicitly.

The following is a **proposed logical signature**, not an approved ONNX/cache
ABI. It is non-paged and follows the existing main component's layout where
possible:

| Direction | Proposed value | Type and logical shape | Status |
|---|---|---|---|
| Input | `input_ids` | INT64 `[B,S]` | GenAI MTP builder's named input. |
| Input | `pre_final_hyper_states` | `T` `[B,S,C*H]` | Main-layer tap described above. |
| Input | `attention_mask` | Existing causal/valid-token mask for `P+S` context | Preserve the ordinary Qwen4-Exp mask semantics; exact ONNX rank/layout follows the approved main ABI. |
| Input | `position_ids` | Candidate text INT64 `[B,S]`; candidate VLM INT64 `[4,B,S]`, with channels `[text, temporal, height, width]` | These follow N0's respective main layouts, not a universal four-axis text ABI. The MTP rank, axes and mapping to GenAI remain approval items. |
| Input | `past_position_ids` | Candidate text INT64 `[B,P]`; candidate VLM INT64 `[4,B,P]` | Proposed only if standard QSA/indexer requires full cached positions; otherwise keep minimum equivalent state. Exact MTP representation remains blocked. |
| Input | MTP-only QSA `past_key`, `past_value`, `past_indexer_key` | Layer-0 state for this MTP head only | Names/roles are visible in GenAI config; exact layout, dtype, initialization and cache-length representation must be approved. |
| Output | `logits` | `T` `[B,S,V]` | Builder output. |
| Output | `hidden_states_out` | `T` `[B,S,C*H]`, pre-final MTP hyper-streams | Builder output; not an approved repeated-step input yet. |
| Output | MTP-only QSA `present_key`, `present_value`, `present_indexer_key` | Same state ABI as corresponding past values, advanced by the MTP tokens actually processed | Builder output roles; exact ABI unresolved. |
| Output | `present_position_ids` | Candidate text INT64 `[B,P+S]`; candidate VLM INT64 `[4,B,P+S]`, if position history is an explicit graph input | Proposed pairing following the selected topology; not present in GenAI's MTP config or an approved MTP ABI. |

The GenAI config names `input_ids`, `hidden_states`, `attention_mask`,
`position_ids`, past/present key/value/indexer state, and
`past_sequence_length` (`qwen.py:3043-3073`). It does not define a Mobius
position-history output or prove whether positions are current-only or
full-context at each call. The table makes the proposed seam explicit while
leaving those incompatible ABI details blocked for joint review with
[N0 (#768)](https://github.com/onnxruntime/mobius/issues/768). Reuse the
approved component/state manifest and selected-loading/shared-identity
mechanisms; do not introduce a second package format or duplicate N0's
decisions.

### Cache and position transition proposal

Treat MTP QSA cache as **dedicated state**, distinct from main decoder caches.
The MTP layer is not a main QSA layer: its own projections and weights process
the draft path. For a call consuming `S` MTP input IDs, the candidate transition
appends their own K/V and indexer entries to the MTP cache, advances the MTP
logical length by `S`, and advances positions only for those consumed IDs.
It must not mutate main decoder, PLE, or vision state. GenAI names a
`past_sequence_length` and per-layer past/present key/value/indexer values;
the non-paged indexer shapes use `[B,past_sequence_length,indexer_head_dim]`
and `[B,total_sequence_length,indexer_head_dim]` (`qwen.py:980-1032`).
Neither the loader nor the HF ordinary forward defines MTP cold-start,
chunking, cache commit/rollback, or how the MTP indexer position history is
reconstructed. The proposed append transition therefore requires independent
full-vs-chunked reference checks and maintainer approval before implementation.

Do not reuse or crop main caches as a proxy for MTP caches. Any speculative
accept/reject transaction and rollback/replay design belongs to the parent
qualification track; this M0 proposal defines no serving algorithm.

## Shifted-token alignment: explicit blocker

The builder's MTP projection gathers an embedding from its `input_ids` and adds
it to the supplied hidden streams at the same tensor row
(`qwen.py:3240-3307`). It performs no visible shift. The builder's config keeps
`input_ids` and `hidden_states` as separate named inputs
(`qwen.py:3043-3065`). These facts do **not** determine the host's intended
row pairing, which row's position applies, or the label used to train that
pair.

The following is a candidate convention to make the off-by-one decision
reviewable, **not a source-grounded semantic claim**. If main token `x_t`
produces logits predicting `x_(t+1)` and tap `h_t` after consuming `x_t`, pair
`h_t` with the selected successor token `x_(t+1)` as MTP input; that MTP row
then predicts `x_(t+2)`.

| Main input row | Main tap row | Main logits row predicts | Candidate MTP input ID paired with tap | Candidate MTP logits predict |
|---|---|---|---|---|
| `t=0: x0` | `h0` | `x1` | `x1` with `h0` | `x2` |
| `t=1: x1` | `h1` | `x2` | `x2` with `h1` | `x3` |
| `t=2: x2` | `h2` | `x3` | `x3` with `h2` | `x4` |

For a prompt `[x0,x1,x2]`, the candidate first MTP call after main selects
`x3` would consume `(x3,h2)` and assign `x3`'s next causal position to its
MTP query/key. Pairing `(x2,h2)` instead, pairing `(x3,h3)` when `h3` has not
been computed, or assigning `x3` position `p2` are distinct off-by-one errors.
This example makes the proposed convention testable; it does not resolve
whether this is Qwen4-Exp's trained/serving convention.

Before M1, an authoritative source or explicitly approved reference must settle:

1. Whether MTP input token `z_t` is aligned with `h_t`, `h_(t-1)`, or another
   shifted tap, and which next-token target its logits represent.
2. How the first draft token is selected, whether the main tap row is reused,
   and which position/mask are assigned to each MTP call.
3. How full prompt prefill, chunked prefill, and cached decode map input IDs,
   hidden rows, attention masks, and positions when `S > 1`.
4. Whether MTP's returned `hidden_states_out` feeds another MTP invocation,
   and which tokens are then considered consumed by its cache.

Until resolved, do not implement the candidate alignment as if it were
approved.

## Supported proposal depth

The pinned GenAI Qwen4-Exp builder rejects any configured MTP layer count
other than exactly one and constructs one sparse-attention layer
(`qwen.py:2904-2919,3155-3165`). The loader likewise instantiates only
`mtp.layers.0` (`loaders/qwen.py:171-188`). This supports describing the
checkpoint structure as one configured trained MTP layer; it does not prove
that the same layer can be iterated to draft arbitrary depth.

Conservative proposal: qualify **one MTP forward step only**. Do not claim
multi-token repeated drafting, repeated cache reuse, or depth `K > 1` until
training/runtime semantics and an independent reference demonstrate it.
Whether even a second sequential invocation is supported is an explicit
blocker, not an implied consequence of `hidden_states_out`.

## Dedicated weight mapping and shared identities

The Qwen4-specific loader, not the generic Qwen MTP path, is the mapping
authority. Its safetensors path selects only `mtp.*`, separately locates
`model.embed_tokens.weight` or `model.language_model.embed_tokens.weight`,
and locates `lm_head.weight`; it falls back to the embedding tensor for the
LM head only when `tie_word_embeddings` is true
(`loaders/qwen.py:80-110`). Its MTP state map is:

| Checkpoint source | MTP graph role |
|---|---|
| `mtp.fc_embedding.weight` | Token-embedding projection |
| `mtp.fc_hidden.weight` | Hyper-stream projection |
| `mtp.pre_fc_norm_embedding.weight` | Offset RMSNorm scale for token embedding |
| `mtp.pre_fc_norm_hidden.weight` | Branchwise offset RMSNorm scale for hyper-streams |
| `mtp.layers.0.*` | Qwen4-Exp sparse-attention decoder layer 0; strip `mtp.layers.0.` when loading that layer |
| `mtp.hyper_connection_mixer.*` | MTP final hyper-connection mixer; strip that prefix when loading the mixer |
| selected embedding source above | MTP token lookup table |
| `lm_head.weight`, or configured tied embedding fallback | MTP output head |

The loader rejects missing/unexpected layer or mixer tensors and requires all
four direct projection/norm weights (`loaders/qwen.py:176-214`). ModelOpt
loading is explicitly rejected for Qwen4-Exp MTP at this pin
(`loaders/qwen.py:154-157`); do not imply quantized MTP support.

Represent token-table and output-head dependencies as **two distinct logical
identities** (for example, `qwen4_exp.token_embedding.weight` and
`qwen4_exp.lm_head.weight`). Reuse each identity across components only where
the selected source/config proves sharing. HF declares the normal CausalLM
embedding/head tie (`modeling_qwen4_exp.py:1528-1542`); the loader's fallback
also checks `tie_word_embeddings`. GenAI's Qwen4 MTP builder explicitly lists
the LM-head MatMul prefix for sharing, while its token lookup is materialized
under `model.embed_tokens.weight` (`qwen.py:2904-2919,3240-3251`).
These are separate identities and sharing declarations, not permission to
collapse the token table and vocabulary head into one logical parameter.
Their physical aliasing and canonical ownership must follow N0 (#768).
N0 specifies ordinary text/VLM owners, but cross-package endpoints and binding
between the root and nested MTP sidecar are a joint approval item, not an
existing capability implied by `SharedWeightInfo`. These example identity
names are illustrative, not approved serialized keys. Do not interpret the
cross-reference as resolving the nested sharing mechanism.

## Independent reference and fixtures

Build the reference independently of the ONNX graph and GenAI builder:

1. For tiny forward checks, use a local tiny config and fixed-seed synthetic
   main, `mtp.*`, embedding and head tensors organized by source names.
   Separately gated bounded-official checks load the pinned weights/config
   and selected tensors directly from their source names. The fixture
   reference must not call `Qwen4ExpForCausalLM.forward` for MTP; that forward
   does not execute the sidecar. Official assets are not a prerequisite for
   the tiny reference.
2. In eager PyTorch, independently implement the approved offset/branchwise
   norms, projections, stream reshape/add, one HF
   `Qwen4ExpTextDecoderLayer` configured as QSA/no-PLE, the MTP-specific
   `Qwen4ExpTextGatedResidual(use_combine=False)`, and output head. Compare
   intermediates and state to standard ONNX outputs. Do not import/run the
   generated ONNX graph, GenAI custom-op graph, or GenAI loader's assembled
   MTP object as the expected-value oracle.
3. Obtain the main tap from an independent HF main-layer run immediately after
   its final layer and before `hyper_connection_mixer`. Verify the tap separately
   from post-mixer logits; the usual HF `last_hidden_state` is post-mixer.
4. First settle alignment, positions, cache initialization, precision, and
   cache advancement with an explicitly approved PyTorch reference. If no
   independent authoritative behavior can be established, report that blocker
   and do not claim MTP parity.

Minimum adversarial fixtures for that reference:

- **Shift:** distinct IDs at every index, nonuniform hidden rows, length at
  least four; assert the selected `(token ID, tap row, target ID, position)`
  tuples from the approved rule. Include first row, last prefill row, a
  one-token continuation, and a two-token chunk.
- **Norm/scale:** zero and nonzero weights (distinguish `1+w` from `w`),
  unequal stream magnitudes, and values near epsilon-sensitive boundaries;
  compare norm outputs before projections, including FP32 accumulation and
  cast-back points.
- **Projection:** identity, diagonal, one-hot, and asymmetric matrices to catch
  transpose, wrong stream-axis, broadcast, flatten-order, and omitted-scale
  errors. Compare `p_h`, `p_e`, and fused `u`.
- **Position/mask:** nonzero position offsets, distinct rotary positions,
  left padding, causal boundaries, and prefill/chunk/decode equivalence.
  Check exact integer positions/masks and QSA indexer rows.
- **Cache:** cold start; full prefill versus two chunks; cached one-token and
  multi-token continuation; exact MTP K/V/indexer lengths and values against
  eager output; verify main cache remains byte-for-byte unchanged.
- **Output:** compare MTP pre-final streams, final mixer result, logits, and
  every present-state component, not just top-1 tokens.

Qualification levels are separate:

| Qualification | What it can establish | What it cannot establish |
|---|---|---|
| Tiny synthetic forward | Standard-op graph construction, every equation/mapping branch, adversarial alignment/state behavior, and parity to the approved independent tiny reference. | Official weight coverage or useful generation quality. |
| Bounded official forward | Mapping and numerical parity for explicitly listed official tensors/layers/tokens and approved dtypes; record exact scope and tolerances. | Full-checkpoint MTP parity, arbitrary draft depth, or speculative generation. |
| Speculative generation | Only after forward and transaction semantics are approved: end-to-end output/state behavior against non-speculative generation for explicit acceptance/rejection, EOS, chunking and batching cases. | A consequence of either forward-level qualification; it is a separate gate. |

The required future M1 entry point from #772 is NEW
`src/mobius/models/qwen4_exp_mtp_test.py`. Once implemented and the reference
approved, run its tiny CPU cases with:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m pytest \
  src/mobius/models/qwen4_exp_mtp_test.py -q
```

This file and command are proposed coverage, not an executed result. Fixtures
must use local synthetic tensors, not cached official weights or a native
custom-op build. Missing required pinned HF reference capability or skipped
mandatory cases remain blockers, not passing skips. Bounded official receipts
belong to the separately approved #774 campaign.

## Implementation and approval gates

Any later implementation must remain `onnx_ir`/`onnxscript.nn`, use standard
ONNX operators, preserve the default opset 24, and pass standard-only
validation after bind, optimize, serialize/reload. No explicit protobuf APIs.
Custom-op structure in the GenAI builder is evidence of intended wiring, not
an acceptable substitute for standard-op or numerical validation.

Before M1 (#772), human maintainers must explicitly approve:

- The equations, exact normalization offsets, projection shapes/orientation,
  and all intermediate precision/rounding points.
- Main tap availability/layout and the exact shifted-token/target pairing,
  including first draft, prefill, chunking, position, and mask rules.
- MTP cache names, tensor layouts/dtypes, initialization, position history,
  append/advance semantics, and independence from main state.
- Supported proposal depth and whether/how hidden-state output is reused.
- Dedicated source mapping, token-table/head logical identities, and N0
  selected loading/shared-identity integration.
- The independent PyTorch reference and acceptance tolerances/fixtures.

Agent-authored documentation, PR completion, or merge is not semantic approval.
M1 also requires approved/merged N0 (#768), merged packaging prerequisites,
and #755 merged or an explicitly human-approved alternative base. If any
semantic item remains unsupported by pinned source and independent reference,
keep it listed as a blocker; do not invent equations, alignment, cache behavior,
or official-weight claims.
