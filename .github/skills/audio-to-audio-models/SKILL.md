---
name: audio-to-audio-models
description: >
  How to add audio-to-audio (speech-to-speech) models to mobius. Covers the
  multi-model ONNX split (audio_encoder, embedding, decoder, audio_decoder),
  LFM2-Audio hybrid conv+attention cache, Moshi/PersonaPlex depformer with
  stacked per-codebook Gather weights, MoshiTask vs AudioToAudioTask,
  ShortConv component, config patterns for minimal HF configs, audio codec
  integration (EnCodec/Mimi), and real-time streaming. Use this skill when
  adding any model that consumes and/or produces audio via codec tokens.
---

# Skill: Audio-to-Audio Models

## When to use

Use this skill when adding a model that takes audio in and produces audio
out as an end-to-end speech pipeline: LFM2-Audio, Moshi, PersonaPlex, or
similar hybrid/codec-based speech-to-speech architectures.

---

## 1. Audio-to-audio model anatomy

Audio-to-audio models are exported as **3 or 4 ONNX models**:

```
waveform → mel → [audio_encoder] → audio_features          ← LFM2-Audio only
                                       │
              text_ids + audio_tokens → [embedding] → inputs_embeds
                                                          │
                         inputs_embeds → [decoder] → logits + KV/conv cache
                                                          │
                          backbone_hidden → [audio_decoder] → codebook_logits
                                                          │
                                              codec decode → waveform output
```

| ONNX model | Optional | LFM2-Audio | Moshi |
|------------|----------|-----------|-------|
| `audio_encoder` | Yes (omit if no mel encoder) | ✅ ConformerEncoder + adapter | ❌ (codec tokens as input) |
| `embedding` | No | Text token embed + audio codebook embed | Text token embed + summed per-codebook audio embeds |
| `decoder` | No | LFM2 hybrid backbone (ShortConv + Attention) | Pure transformer |
| `audio_decoder` | Optional | Depthformer (per-codebook autoregressive transformer) | Depformer (stacked per-codebook Gather weights) |

### Why split into multiple models?

- `audio_encoder` runs once per audio frame (low frequency, before embedding)
- `embedding` runs once per generation step
- `decoder` runs once per step (hot path, must be fast)
- `audio_decoder` runs once **per codebook** per step (K calls, where K=8 or 16)

---

## 2. Task selection

| Task | TASK_REGISTRY key | Use when |
|------|-------------------|----------|
| `AudioToAudioTask` | `"audio-to-audio"` | Model has mel spectrogram audio encoder (LFM2-Audio) |
| `MoshiTask` | `"moshi"` | Model consumes codec token IDs directly, no mel encoder (Moshi, PersonaPlex) |

```python
# In _registry.py:
reg.register("lfm2_audio", Lfm2AudioModel, task="audio-to-audio")
reg.register("moshi",      MoshiModel,     task="moshi")
```

`MoshiTask` is a subclass of `AudioToAudioTask` that:
1. Skips `audio_encoder` (Moshi input is already codec token IDs)
2. Overrides `_build_embedding` to accept both `input_ids` and `audio_codes [B, S, K]`
3. Overrides `_build_audio_decoder` with `head_dim = depformer_dim` (one full-dim head per codebook)

---

## 3. Hybrid cache: ShortConv + Attention layers

LFM2-Audio interleaves two layer types controlled by `config.layer_types`:

```
"conv"           → Lfm2ConvDecoderLayer  (ShortConv + MLP)
"full_attention" → Lfm2AttentionDecoderLayer  (Attention + MLP)
```

The decoder KV cache is **hybrid**: conv layers carry `conv_state`, attention
layers carry standard `(key, value)` pairs.

### Hybrid cache I/O contract

```
# Conv layer cache entry: tuple with one element
past_key_value = (conv_state,)  # (B, hidden_size, kernel_size-1)
present_key_value = (new_conv_state,)

# Attention layer cache entry: tuple with two elements
past_key_value = (past_key, past_value)  # (B, num_kv_heads, S, head_dim) each
present_key_value = (new_key, new_value)
```

### How `_make_hybrid_cache_inputs` works

`_make_hybrid_cache_inputs(config, dtype, batch, past_seq_len)` in
`tasks/_base.py` iterates `config.layer_types` and creates:

- For `"conv"` layers: `conv_state.N [batch, hidden_size, kernel_size-1]`
- For `"full_attention"` layers: `past_key_values.N.key / .value [batch, kv_heads, S, head_dim]`

`_register_hybrid_cache_outputs` mirrors this for outputs.

### Selecting hybrid vs standard cache in `AudioToAudioTask`

`AudioToAudioTask._build_decoder` auto-selects:

```python
use_hybrid = config.layer_types is not None and any(
    lt != "full_attention" for lt in config.layer_types
)
```

No code change needed — just set `layer_types` in the config.

---

## 4. `ShortConv` component

`ShortConv` in `components/_short_conv.py` implements the gated causal
depthwise Conv1d block from LFM2:

```
x → in_proj → split [B, C, x]
                  ↓
              B * x = Bx
                  ↓
    causal depthwise Conv1d(Bx, groups=hidden_size)
                  ↓
              C * conv_out
                  ↓
            out_proj → y
```

### Forward signature

```python
def forward(
    self,
    op: builder.OpBuilder,
    hidden_states: ir.Value,      # (B, S, hidden_size)
    conv_state: ir.Value | None,  # (B, hidden_size, kernel_size-1), or None for prefill
) -> tuple[ir.Value, ir.Value]:   # (output, new_conv_state)
```

During **prefill** (`conv_state=None`): pads left by `kernel_size-1` zeros.
During **generation** (`conv_state` provided): prepends cached state (no extra padding).

### New conv state extraction

`new_conv_state` = last `kernel_size - 1` timesteps of `bx_padded`:
```python
new_conv_state = op.Slice(bx_padded,
    op.Constant(value_ints=[-(kernel_size - 1)]),
    op.Constant(value_ints=[INT64_MAX]),
    op.Constant(value_ints=[2]),  # axis=2
)
```

### Weight names (HF → mobius)

| HuggingFace | mobius |
|------------|--------|
| `conv.conv.weight` | `conv.conv_weight` |
| `conv.in_proj.weight` | `conv.in_proj.weight` |
| `conv.out_proj.weight` | `conv.out_proj.weight` |

Note: `conv.conv.weight` → `conv.conv_weight` (not `.weight.weight`) to avoid
the auto-suffixing from `nn.Parameter`.

### Using `ShortConv` in a decoder layer

```python
from mobius.components import ShortConv

class MyConvLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.operator_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.conv = ShortConv(
            config.hidden_size,
            kernel_size=config.short_conv_kernel or 3,
            bias=config.short_conv_bias or False,
        )
        self.ffn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.feed_forward = MLP(config)

    def forward(self, op, hidden_states, past_key_value=None, **kwargs):
        # Conv state from hybrid cache
        conv_state = past_key_value[0] if past_key_value is not None else None
        residual = hidden_states
        hidden_states = self.operator_norm(op, hidden_states)
        conv_out, new_conv_state = self.conv(op, hidden_states, conv_state)
        hidden_states = op.Add(residual, conv_out)
        # MLP
        residual = hidden_states
        hidden_states = self.ffn_norm(op, hidden_states)
        hidden_states = op.Add(residual, self.feed_forward(op, hidden_states))
        return hidden_states, (new_conv_state,)  # always a 1-tuple for conv layers
```

---

## 5. Depformer / depthformer pattern (Moshi)

The **depformer** (depth transformer) generates audio codec tokens one
codebook at a time, given the backbone's last hidden state. It is a separate
causal transformer with a KV cache that accumulates **codebook** positions
(not time positions).

### Key design: stacked per-codebook weights + `Gather`

Different codebooks use different projection matrices. Rather than storing
them as separate sub-modules (which would create N × M separate parameters),
they are **stacked** into a single `[num_codebooks, ...]` parameter and
selected at runtime with `op.Gather(stacked_weight, codebook_idx, axis=0)`.

```python
# In _DepformerLayer.__init__:
self.stacked_gate_proj = nn.Parameter([num_codebooks, interm, depformer_dim])
self.stacked_up_proj   = nn.Parameter([num_codebooks, interm, depformer_dim])
self.stacked_down_proj = nn.Parameter([num_codebooks, depformer_dim, interm])

# In forward:
gate_w = op.Gather(self.stacked_gate_proj, codebook_idx, axis=0)  # (interm, dim)
up_w   = op.Gather(self.stacked_up_proj,   codebook_idx, axis=0)
down_w = op.Gather(self.stacked_down_proj, codebook_idx, axis=0)
# Then MatMul manually (not Linear) since weight is dynamic
gate = op.MatMul(hidden_states, op.Transpose(gate_w))
```

### Why not use `Linear`?

`Linear` wraps `op.MatMul` + `op.Transpose` but expects a **static** weight
parameter. Since `gate_w` is a dynamic output of `op.Gather`, it cannot be
passed to `Linear`. Use `op.MatMul(x, op.Transpose(w))` directly.

### Stacking per-codebook weights in `preprocess_weights`

```python
# Stack N separate linears.N.weight → stacked_output_heads [N, out, dim]
out_heads = []
for i in range(num_codebooks):
    key = f"linears.{i}.weight"
    if key in state_dict:
        out_heads.append(state_dict.pop(key))
if out_heads:
    new_sd["audio_decoder.stacked_output_heads"] = torch.stack(out_heads, dim=0)
```

### Depformer KV cache sizing

Moshi depformer uses `num_heads = num_codebooks` with `head_dim = depformer_dim`
(one full-dimensioned "head" per codebook). This is different from the main
transformer where `head_dim = hidden_size // num_heads`.

```python
# MoshiTask._build_audio_decoder:
depformer_head_dim = depformer_dim  # full head_dim per codebook

kv_inputs, past_key_values = _make_kv_cache_inputs(
    depformer_layers,
    depformer_heads,    # = num_codebooks
    depformer_head_dim, # = depformer_dim (NOT depformer_dim // num_heads)
    ...
)
```

---

## 6. Config patterns

### `Lfm2AudioConfig` (inherits `Lfm2Config`)

```python
@dataclasses.dataclass
class Lfm2AudioConfig(Lfm2Config):
    depthformer_layers: int = 6
    depthformer_dim: int = 512
    depthformer_heads: int = 8
    num_codebooks: int = 8
    audio_vocab_size: int = 2048
    # audio: AudioConfig  (inherited from base — Conformer params)
```

Key fields for `AudioConfig`:
```
attention_dim / d_model   → Conformer hidden size
attention_heads           → Conformer attention heads
num_blocks / encoder_layers → Number of Conformer blocks
linear_units / encoder_ffn_dim → FFN width
kernel_size               → Depthwise conv kernel
conv_channels             → Pointwise conv channels
t5_bias_max_distance      → T5 relative bias max distance
num_mel_bins              → Input mel bins (e.g. 128)
```

### `MoshiConfig`

```python
@dataclasses.dataclass
class MoshiConfig(ArchitectureConfig):
    depformer_dim: int = 1024
    depformer_layers: int = 6
    depformer_num_heads: int = 16     # must equal num_codebooks
    depformer_intermediate_size: int = 2816
    num_codebooks: int = 16
    audio_vocab_size: int = 2048
```

### Minimal HF config.json (Moshi pattern)

Moshi's `config.json` contains only `model_type` and `version` — **all
architecture parameters are hardcoded in the model class itself**, not loaded
from config. This is a deliberate HF design choice for the base Moshi release.

**Solution**: Override `from_transformers` to return hardcoded defaults:

```python
@classmethod
def from_transformers(cls, config, parent_config=None) -> MoshiConfig:
    # config has only model_type and version — hardcode everything
    return cls(
        hidden_size=4096,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=128,
        depformer_dim=1024,
        depformer_layers=6,
        depformer_num_heads=16,
        num_codebooks=16,
        audio_vocab_size=2049,
        vocab_size=32000,
        ...
    )
```

**Never raise an exception for missing config fields when the model class
itself provides architecture constants.** Use `getattr(config, "field", default)`.

---

## 7. Audio codec integration (EnCodec / Mimi)

Both LFM2-Audio and Moshi use **RVQ (Residual Vector Quantization)** audio
codecs at the boundary:

| Codec | Model | Codebooks | Audio vocab | Sample rate |
|-------|-------|-----------|-------------|-------------|
| EnCodec | Moshi / PersonaPlex | 16 | 2048 | 24 kHz |
| EnCodec | LFM2-Audio | 8 | 2048 | 24 kHz |
| Mimi | Kyutai's alternative | variable | variable | 24 kHz |

### Codec is NOT part of mobius ONNX export

The codec (encoder/decoder waveform ↔ tokens) is a separate model handled at
the application level (e.g., `encodec` or `moshi` Python packages). mobius
exports models that operate on codec **token IDs**, not raw audio.

**Exception**: LFM2-Audio includes a Conformer mel encoder (`audio_encoder`)
that converts mel spectrograms to audio features. This IS part of the mobius
export because it is part of the LFM2 model architecture.

### Frame rate

Both models operate at **12.5 Hz** (one step = one EnCodec frame = 80ms at 24kHz):

```python
SAMPLE_RATE   = 24_000   # Hz
FRAME_SAMPLES = 1_920    # samples per frame (80ms)
STEPS_PER_SECOND = SAMPLE_RATE // FRAME_SAMPLES  # 12.5
```

---

## 8. Step-by-step: adding a new audio-to-audio model

### Step 1 — Identify audio input type

Does the model take **mel spectrograms** or **codec token IDs** as audio input?

- Mel spectrograms → use `AudioToAudioTask` (includes `audio_encoder`)
- Codec token IDs → use `MoshiTask` (no `audio_encoder`)

### Step 2 — Create `models/<name>.py`

Follow the four-class pattern (or three for Moshi-style):

```python
class _MyAudioEncoder(nn.Module):   # ConformerEncoder + adapter
class _MyEmbedding(nn.Module):      # text + audio token embeddings
class _MyDecoder(nn.Module):        # LM backbone (hybrid or standard)
class _MyAudioDecoder(nn.Module):   # depformer (optional)

class MyAudioModel(nn.Module):
    default_task: str = "audio-to-audio"   # or "moshi"
    category: str = "Audio"

    def __init__(self, config):
        super().__init__()
        self.audio_encoder = _MyAudioEncoder(config)  # AudioToAudioTask reads this
        self.embedding     = _MyEmbedding(config)     # required
        self.decoder       = _MyDecoder(config)       # required
        self.audio_decoder = _MyAudioDecoder(config)  # optional
```

### Step 3 — Create a config class

If the HF config is minimal, create a dataclass with hardcoded defaults:

```python
@dataclasses.dataclass
class MyAudioConfig(ArchitectureConfig):
    num_codebooks: int = 8
    audio_vocab_size: int = 2048
    depthformer_dim: int = 512
    depthformer_layers: int = 4

    @classmethod
    def from_transformers(cls, config, parent_config=None):
        return cls(
            hidden_size=getattr(config, "hidden_size", 2048),
            num_codebooks=getattr(config, "codebooks", 8),
            ...
        )
```

### Step 4 — Implement `preprocess_weights`

Audio-to-audio models typically have complex HF → ONNX weight renames.
Implement a `_rename_weight(key)` helper that returns the new key or `None`
(to skip), then call it from `preprocess_weights`:

```python
def preprocess_weights(self, state_dict):
    new_sd = {}
    for key, value in state_dict.items():
        new_key = _rename_weight(key)
        if new_key is not None:
            new_sd[new_key] = value
    return new_sd
```

Common renames for LFM2-Audio (see `models/lfm2_audio.py` for complete list):
- `lfm.embed_tokens.*` → `embedding.text_embed.*`
- `lfm.layers.N.*` → `decoder.layers.N.*`
- `conformer.*` → `audio_encoder.encoder.*`
- `audio_adapter.model.0.*` → `audio_encoder.adapter.up_proj.*`
- `audio_adapter.model.3.*` → `audio_encoder.adapter.down_proj.*`
- `audio_adapter.model.1.*` → skip (BatchNorm, not used in ONNX)

### Step 5 — Register and add test config

```python
# _registry.py:
reg.register("my_audio_model", MyAudioModel, task="audio-to-audio",
             config_cls=MyAudioConfig)
```

Build-graph test config:
```python
def _my_audio_config(self):
    from mobius._configs import AudioConfig, MyAudioConfig
    return MyAudioConfig(
        vocab_size=TINY_VOCAB,
        hidden_size=TINY_HIDDEN,
        num_hidden_layers=4,
        ...
        layer_types=["conv", "conv", "full_attention", "conv"],  # if hybrid
        num_codebooks=2,
        audio_vocab_size=32,
        audio=AudioConfig(num_mel_bins=16, attention_dim=TINY_HIDDEN, ...),
    )

def test_my_audio_4_model_package_structure(self):
    from mobius.models.myaudio import MyAudioModel
    from mobius.tasks._audio_to_audio import AudioToAudioTask
    config = self._my_audio_config()
    pkg = AudioToAudioTask().build(MyAudioModel(config), config)
    assert set(pkg.keys()) == {"audio_encoder", "embedding", "decoder", "audio_decoder"}
```

---

## 9. Real-time streaming inference pattern

The `examples/moshi_realtime.py` and `examples/lfm2_audio_realtime.py` files
demonstrate the full streaming loop. Key patterns:

### Inference loop structure

```python
# Frame loop (one iteration = 80ms of audio)
for frame_idx in range(num_frames):
    # 1. Encode input audio to tokens (outside mobius)
    audio_tokens = codec_encoder(waveform_frame)   # [1, num_codebooks]

    # 2. Embed: text + audio tokens → inputs_embeds
    inputs_embeds = ort_session_embedding.run(["inputs_embeds"], {
        "input_ids": text_ids,
        "audio_codes": audio_tokens,
    })[0]

    # 3. Decode: backbone LM step
    logits, *present_kv = ort_session_decoder.run(None, {
        "inputs_embeds": inputs_embeds,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        **past_kv_dict,
    })
    past_kv_dict = update_kv_cache(present_kv)

    # 4. Audio decoder: per-codebook autoregressive loop
    backbone_hidden = decoder_hidden_state  # last hidden, not logits
    prev_emb = initial_codebook_embedding
    generated_codes = []
    for codebook_idx in range(num_codebooks):
        codebook_logits, *depth_kv = ort_session_audio_decoder.run(None, {
            "backbone_hidden": backbone_hidden,
            "prev_embedding": prev_emb,
            "codebook_idx": np.int64(codebook_idx),
            **depth_kv_dict,
        })
        token = np.argmax(codebook_logits[0, 0])
        generated_codes.append(token)
        prev_emb = audio_codebook_embedding(token, codebook_idx)
        depth_kv_dict = update_depth_kv(depth_kv)

    # 5. Decode audio tokens to waveform (outside mobius)
    output_waveform = codec_decoder(generated_codes)
```

### Hybrid cache management

For LFM2-style hybrid cache, maintain separate state arrays per layer:

```python
# Initialize hybrid cache
conv_states = {
    f"conv_state.{i}": np.zeros([batch, hidden_size, kernel_size - 1], dtype=np.float32)
    for i, lt in enumerate(layer_types) if lt == "conv"
}
kv_states = {
    f"past_key_values.{i}.key": np.zeros([batch, kv_heads, 0, head_dim], ...)
    for i, lt in enumerate(layer_types) if lt == "full_attention"
}
```

---

## 10. Common pitfalls

### ❌ `create_attention_bias` argument order

`create_attention_bias(op, input_ids, attention_mask, ...)` — **not**
`(op, hidden_states, attention_mask)`. The second argument is `input_ids`
(used only for shape: `op.Shape(input_ids, start=1, end=2)` to get
`query_length`). Passing `hidden_states` works accidentally for shape, but
passing the wrong type causes dtype/shape bugs downstream.

```python
# ✅ Correct
attention_bias = create_attention_bias(op, attention_mask, hidden_states, position_ids)
# See models/lfm2_audio.py for the exact call signature used in practice
```

### ❌ Conv weight naming: `.conv.weight` vs `.conv_weight`

HF stores the depthwise conv weight as `conv.conv.weight` (two levels of
`.conv.`). In mobius, the `ShortConv` parameter is `self.conv_weight` (not
`self.conv.weight`) to avoid the nested path. The rename in `preprocess_weights`:

```python
key = key.replace("conv.conv.weight", "conv.conv_weight")
key = key.replace("conv.conv.bias",   "conv.conv_bias")
```

### ❌ Dead inputs in audio encoder

Some `audio_adapter` layers include batch normalization (`model.1.*`,
`model.2.*`) that has no equivalent in ONNX. These keys must be **skipped**
(return `None` from the rename function), not passed through unchanged, or
`apply_weights` will fail with "unexpected key" errors.

### ❌ Depformer KV cache shape: `head_dim = depformer_dim`

In Moshi's depformer, `head_dim = depformer_dim` (full size), NOT
`depformer_dim // num_codebooks`. Always use the full `depformer_dim` for
`_make_kv_cache_inputs` in `MoshiTask._build_audio_decoder`. If you
accidentally use `depformer_dim // num_heads`, the KV cache will be
undersized and attention will produce incorrect results.

### ❌ Stacked weight shape conventions

When stacking per-codebook weights, follow the convention:
- Gate/up projections: `[num_codebooks, out_features, in_features]`
- Down projections: `[num_codebooks, in_features, out_features]`

This is consistent with how `Gather` selects `[out, in]` for manual
`op.MatMul(x, op.Transpose(w))`. Inverting the shape requires a double
transpose and confuses weight loading.

### ❌ Moshi `out_proj.weight` is stored transposed

HF Moshi stores the depformer attention `out_proj.weight` as `[K*D, D]`
(output rows × input cols), but mobius `Linear(K*D, D)` expects `[D, K*D]`
(out × in). Always `.T` when loading:

```python
new_sd[f"{dst}.self_attn.o_proj.weight"] = state_dict.pop(out_proj_key).T
```

### ❌ Missing `audio_decoder` attribute causes silent skip

`AudioToAudioTask.build` calls `hasattr(module, "audio_decoder")` to
decide whether to build the audio decoder model. If you forget to add
`self.audio_decoder = _MyAudioDecoder(config)` in `__init__`, the
`audio_decoder` ONNX model is silently omitted from the `ModelPackage`.

---

## 11. File reference

| File | Purpose |
|------|---------|
| `src/mobius/tasks/_audio_to_audio.py` | `AudioToAudioTask` and `MoshiTask` |
| `src/mobius/models/lfm2_audio.py` | LFM2-Audio (4-model split, hybrid cache) |
| `src/mobius/models/moshi.py` | Moshi/PersonaPlex (3-model, stacked Gather weights) |
| `src/mobius/models/lfm2.py` | `Lfm2ConvDecoderLayer`, `Lfm2AttentionDecoderLayer` |
| `src/mobius/components/_short_conv.py` | `ShortConv` gated causal depthwise conv |
| `src/mobius/components/_conformer.py` | `ConformerEncoder` (mel → audio features) |
| `src/mobius/components/_common.py` | `create_attention_bias` |
| `src/mobius/_configs.py` | `Lfm2AudioConfig`, `MoshiConfig`, `AudioConfig` |
| `src/mobius/tasks/_base.py` | `_make_hybrid_cache_inputs`, `_register_hybrid_cache_outputs` |
| `examples/moshi_realtime.py` | Moshi real-time streaming inference example |
| `examples/lfm2_audio_realtime.py` | LFM2-Audio real-time streaming inference example |
| `tests/build_graph_test.py` | `TestBuildGraphAudioToAudio` class (lines ~3849+) |
