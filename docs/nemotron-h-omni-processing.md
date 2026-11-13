# Nemotron-H Omni vision host bridge

This bridge targets the official processor/model source at HuggingFace revision
**`c6cc2e8ffa353403213d6b9b18aaf8eb684d705a`**, specifically
`processing_nemotron_h_omni.py`, `image_processing_nemotron_h_omni.py`, and
`modeling_nemotron_h_omni.py`. It does not establish full-checkpoint numerical,
native quantization, generation, or downstream runtime readiness.

## Graph inputs and caller-owned provenance

```python
import numpy as np

from mobius.integrations.transformers._nemotron_h_omni_processing import (
    OmniVisionCall,
    compact_omni_prompt,
    run_omni_vision_calls,
)
```

Construct one `OmniVisionCall` per same-canvas image batch or **single video
clip**:

```python
call = OmniVisionCall(
    pixel_values=normalized_frames,       # np.float32 [real_frames, 3, H, W]
    placeholder_positions=video_slots,   # np.int64 [dense_projected_tokens]
    is_video=True,
    patch_size=16,
    spatial_merge_size=2,
    temporal_patch_size=2,
)
```

For images, use `is_video=False` and `[images, 3, H, W]`. H/W must be positive
multiples of 32 for the pinned configuration. Differently sized images require
separate calls. Image data from the official processor may be a ragged list;
convert each image to its own NCHW call rather than inventing padding.

**Pixels are already normalized.** The bridge does not resize, rescale,
normalize, duplicate the final frame, or pack temporal channels. The vision
graph owns odd-frame repetition and temporal packing. Do not pass rank-five
tubelets or six-channel packed pixels. A five-frame clip remains five NCHW
frames in the feed, even though the graph emits three temporal groups.
The bridge cannot infer whether a caller has already duplicated an identical
last frame; submitting real frames is the caller's contract.

`call.feeds()` returns:

- `pixel_values`: unchanged normalized float32 NCHW pixels;
- `is_video`: a **rank-zero NumPy bool**, not a one-element vector.

Both input names can be overridden on `feeds()` or `run_omni_vision_calls()`.
This bridge is for the scalar-dispatch **NCHW** graph contract, not a rank-five
graph variant.

### Binding media to the expanded prompt

Both image and video feature placeholders have token ID **18** in the pinned
checkpoint. `<video>` is a processor marker, not a separate feature token ID.
The processor expands each temporal group to timestamp text followed by
`<img><image>...<image></img>`. For an odd final group it labels only real
frames, even though the graph repeats the last frame internally.

`placeholder_positions` must contain the original prompt's flattened,
row-major token positions for this call's **dense, unpruned** projected
features, in ascending order. The positions can be separated by timestamps,
newlines, and delimiters. Their order must match the graph's image-major or
tubelet-major feature order.

Keep this association while rendering/tokenizing media markers. **Do not infer
image versus video membership from token 18**, and do not assume all image
features precede video features in prompt order. For mixed media the official
`BatchFeature` does not provide sufficient provenance by itself; shared
`num_tokens`/`num_patches` fields are overwritten by the video dictionary.
Callers enabling mixed mode must supply their own unambiguous bindings.

Dense token counts are checked against pixels:

- images: `N * (H/32) * (W/32)`;
- video: `ceil(real_frames/2) * (H/32) * (W/32)`.

## Run and compact

The graph must return:

- `image_features`: floating `[K, hidden_size]`;
- `retention_mask`: bool `[dense_tokens]`;
- `kept_indices`: int64 `[K]`, equal to `flatnonzero(retention_mask)`.

```python
names = ["image_features", "retention_mask", "kept_indices"]

def run_vision(feeds):
    return dict(zip(names, vision_session.run(names, dict(feeds))))

outputs = run_omni_vision_calls(calls, run_vision)
prepared = compact_omni_prompt(
    input_ids,                       # np.int64 [batch, sequence]
    attention_mask,                  # np.int64/bool, same shape, or None
    calls,
    outputs,
    hidden_size=4096,
    image_token_id=18,
    feature_dtype=np.float32,         # match the actual graph output dtype
)
```

For bf16 graphs use the corresponding registered NumPy bfloat16 dtype. No
feature output is silently cast. Adapt the runner to the runtime's bf16 output
conversion if its ordinary `session.run()` does not expose bf16 NumPy arrays.

The result contains:

- `input_ids`, `attention_mask`: compacted consistently;
- `image_features`: compacted features sorted by their **surviving prompt
  positions**, regardless of invocation order;
- `token_retention_mask`: bool mask in the original prompt shape;
- `kept_token_indices`: original flattened prompt positions for all surviving
  tokens, including text.

Only video placeholder tokens may be removed. All timestamp labels, newlines,
`<img>`/`</img>` delimiters, text, and padding positions remain, even if every
feature in a later temporal group was pruned. This matches the source's
`retention_mask[input_ids == img_context_token_id] = evs_mask`, rather than
deleting a whole timestamped group.

Pass the returned prompt and features together to embedding/input mixing.
Never scatter compacted features into the original unpruned prompt. Do not
reuse stale position IDs; regenerate them under the decoder's contract, or
use `kept_token_indices` for explicitly aligned caller-owned fields.

The bridge fails closed on missing/overlapping/out-of-range bindings, unmatched
placeholder counts, non-finite pixels/features, dtype mismatches, inconsistent
masks/indices/features, image pruning, or dropped first-video-group tokens.
It validates graph metadata; it does not recompute EVS or prove that arbitrary
returned features numerically match their claimed indices.

## Source limitations and explicit mixed-media extension

The pinned processor supports video only for prompt batch size one and expands
only the first `<video>` marker. The host bridge preserves the batch-one video
restriction. Image-only prompt batches retain their original shape.

The official mixed image+video **model forward is not a valid numerical
reference**: both image and video branches scatter over every token-18
position. Image feature counts then mismatch the combined placeholders, and
the video EVS mask has the same ambiguity. Multiple independent clips are not
represented by that forward either.

By default the bridge therefore rejects mixed image/video or multiple-video
calls. To deliberately use the exported graph with caller-owned provenance:

```python
prepared = compact_omni_prompt(
    input_ids, attention_mask, calls, outputs,
    hidden_size=4096,
    allow_provenance_mixed=True,
)
```

This **explicit extension is not a claim of parity with the broken mixed HF
forward**. Each clip runs independently, retains its own first group, and
maps its mask only to its bound positions. The bridge then orders features by
the surviving prompt, including video-before-image or interleaved media.

The graph's existing EVS limitations still apply: unsupported non-square
projected patch counts or nonfinite cosine scores must fail at the graph.
Dynamic compaction does not promise CUDA graph capture.

## Cached decode and validation

Use `calls=()` on a cached step with no new media. `run_omni_vision_calls()`
invokes nothing. With `cached_decode=True`, compaction returns the prompt and
attention unchanged and empty `[0, hidden_size]` features. Even token 18 is an
ordinary embedding token on that absent-media source path. New media calls
are forbidden in this cached-decode mode.

Tests need no weights, downloads, or optional image dependency installation:

```bash
PYTHONPATH=src \
MOBIUS_OMNI_SOURCE_DIR=/path/to/the/pinned/source \
CUDA_VISIBLE_DEVICES=3 \
python -m pytest \
  src/mobius/integrations/transformers/_nemotron_h_omni_processing_test.py -q
```

The source-backed tests execute the official processor expansion and
normalization/target-grid methods using nonzero NumPy image fixtures, plus the
exact EVS function on CPU/CUDA. Only irrelevant processor/backend setup is
replaced by lightweight fixtures; the tested source method bodies are
unmodified. They also demonstrate the mixed-forward scatter-count failure.
